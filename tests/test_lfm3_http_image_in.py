"""Host tests for image input on POST /speech (#4; native/include/lfm/lfm3_http.hpp) behind the harness's fake engine.

The Service decodes and preprocesses each image part as the engine will (lfm/image_decode.hpp, image_preprocess.hpp:
HF's front end, bit for bit, validation/images-v1 rule 1) to expand its <|image|> to the image's token count, and
refuses a bad image with its own error code before the engine sees the request. The fake answers op "speech" as
WorkerCore does: it decodes each image again and checks the prompt's <|image|> spans against them
(lfm3_speech::check_image_spans), and reports the prompt length as "frames", so these tests see the prompt built.
Token counts follow HF's smart_resize: both sides to multiples of 32, the pixel count kept within 65,536..2,097,152,
one token per 32 x 32 block. Every build variant of the harness runs (plain, ASan+UBSan, TSan).
"""
import base64
import io
import random
import threading

import pytest

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from test_lfm3_http_audio_in import wav  # noqa: E402
from test_lfm3_http_native import harness_binary, running  # noqa: E402,F401  (fixture)


def image(width, height, mode="RGB", fmt="PNG", noise=False, **save):
    if noise:
        rng = random.Random(width * 7919 + height)
        im = Image.frombytes("RGB", (width, height), rng.randbytes(width * height * 3))
    else:
        im = Image.new(mode, (width, height), (90, 140, 200) if mode == "RGB" else 120)
    buf = io.BytesIO()
    im.save(buf, fmt, **save)
    return base64.b64encode(buf.getvalue()).decode()


def ask(images, text="What is in the image?", mode="text", system="Describe the provided image.", max_new_tokens=64,
        audio=(), **extra):
    content = [{"type": "audio", "audio": a} for a in audio] + [{"type": "image", "image": i} for i in images]
    if text:
        content.append({"type": "text", "text": text})
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": content}]
    return dict(messages=messages, mode=mode, max_new_tokens=max_new_tokens, **extra)


def prompt_length(h, body):
    status, reply = h.call("POST", "/speech", body)
    assert status == 200 and reply["cancelled"] is False, (status, reply)
    assert len(reply["chunks"]) == 1
    return reply, reply["chunks"][0]["frames"]


def test_each_image_expands_to_its_token_count(harness_binary):
    with running(harness_binary) as h:
        _, small = prompt_length(h, ask([image(256, 256)]))     # 65,536 pixels: 8 x 8 = 64 tokens
        _, mid = prompt_length(h, ask([image(512, 512)]))       # 16 x 16 = 256 tokens
        _, wide = prompt_length(h, ask([image(1024, 512)]))     # 32 x 16 = 512 tokens
        _, full = prompt_length(h, ask([image(2048, 1024)]))    # the most pixels: 64 x 32 = 2,048 tokens
        assert (mid - small, wide - small, full - small) == (192, 448, 1984)
        _, tiny = prompt_length(h, ask([image(16, 16)]))        # scaled up to the minimum: 64 tokens
        assert tiny == small
        # a second image adds its tokens and its own <|vision_start|>, <|vision_end|>
        _, two = prompt_length(h, ask([image(256, 256), image(512, 512)]))
        assert two - small == 256 + 2


def test_formats_and_modes_give_the_same_tokens(harness_binary):
    with running(harness_binary) as h:
        _, png = prompt_length(h, ask([image(640, 480)]))
        for other in (image(640, 480, fmt="JPEG", quality=90), image(640, 480, mode="L"),
                      image(640, 480, mode="RGBA"), image(640, 480, mode="L", fmt="JPEG")):
            _, n = prompt_length(h, ask([other]))
            assert n == png


def test_images_and_audio_in_one_prompt_in_either_order(harness_binary):
    with running(harness_binary) as h:
        _, image_only = prompt_length(h, ask([image(512, 512)]))
        _, both = prompt_length(h, ask([image(512, 512)], audio=[wav(1.0)]))
        assert both > image_only
        # audio after the image: the same parts, the other order
        body = ask([image(512, 512)])
        body["messages"][-1]["content"].insert(1, {"type": "audio", "audio": wav(1.0)})
        _, reordered = prompt_length(h, body)
        assert reordered == both


def test_spoken_and_interleaved_replies(harness_binary):
    with running(harness_binary) as h:
        text, _ = prompt_length(h, ask([image(512, 512)]))
        reply, _ = prompt_length(h, ask([image(512, 512)], mode="interleaved"))
        assert reply["mode"] == "interleaved" and reply["split"] is False and reply["text"] == text["text"]
        # audio mode with an image is not split by default (split is for text input)
        reply, _ = prompt_length(h, ask([image(512, 512)], mode="audio", system="Respond with audio."))
        assert reply["mode"] == "audio" and reply["split"] is False


def test_a_full_resolution_photo_fits_the_speech_body(harness_binary):
    # Tripp, 2026-11-15: /speech takes up to 16 MiB so a phone photo can be sent as it is; ~6.3 MB of noise PNG is
    # ~8.4 MB of base64, over the 4 MiB every other request keeps
    big = image(1448, 1448, noise=True)
    assert len(big) > 8_000_000
    with running(harness_binary) as h:
        reply, n = prompt_length(h, ask([big]))
        assert n > 2000
        # a /speech body over 4 MiB without an image is still refused, as before images
        status, body = h.call("POST", "/speech", ask([], text="x" * 10, audio=[wav(1.0)],
                                                     system="Respond." + " " * 4_300_000))
        assert status == 413 and body == {"error": "body_limit"}, (status, body)
        # and the server keeps serving
        prompt_length(h, ask([image(256, 256)]))


@pytest.mark.parametrize("make, status, code", [
    (lambda: ask(["iVBORw0K!!!!"]), 400, "invalid_image_base64"),                   # not base64
    (lambda: ask(["QUJD"]), 400, "image_unsupported_format"),                       # base64 of "ABC"
    (lambda: ask([image(640, 480, fmt="GIF")]), 400, "image_unsupported_format"),   # GIF: not JPEG or PNG
    (lambda: ask([image(640, 480, fmt="JPEG")[:400]]), 400, "image_truncated"),     # a JPEG cut short
    (lambda: ask([image(2010, 10)]), 400, "image_aspect_ratio"),                     # aspect ratio over 200
    (lambda: ask([image(8193, 8193, mode="L")]), 413, "image_too_large"),           # over 64 megapixels decoded
    (lambda: ask([image(64, 64)] * 17), 400, "too_many_images"),                    # 16 images at most
])
def test_image_refusals_have_their_own_code_and_leave_the_server_serving(harness_binary, make, status, code):
    bad = make()
    with running(harness_binary) as h:
        got, body = h.call("POST", "/speech", bad)
        assert (got, body) == (status, {"error": code}), (got, body)
        prompt_length(h, ask([image(256, 256)]))


@pytest.mark.parametrize("make", [
    lambda: ask([""]),                                                     # empty image string
    lambda: ask([image(512, 512)], split=True, mode="audio"),              # image input is not split
    lambda: dict(messages=[{"role": "user", "content": "Hi."},
                           {"role": "assistant", "content": [{"type": "image", "image": image(64, 64)}]},
                           {"role": "user", "content": "Again."}], mode="text", max_new_tokens=16),  # assistant image
    lambda: dict(messages=[{"role": "system", "content": [{"type": "image", "image": image(64, 64)}]},
                           {"role": "user", "content": "Hi."}], mode="text", max_new_tokens=16),  # system image
    lambda: dict(messages=[{"role": "user", "content": [{"type": "image", "image": image(64, 64), "x": 1}]}],
                 mode="text", max_new_tokens=16),                          # extra key in the part
    lambda: dict(messages=[{"role": "user", "content": [{"type": "image", "image": 7}]}], mode="text",
                 max_new_tokens=16),                                       # image not a string
    lambda: dict(messages=[{"role": "user", "content": "look <|image|> here"}], mode="text",
                 max_new_tokens=16),                                       # a placeholder typed as text
    lambda: ask([image(2048, 1024), image(2048, 1024)], max_new_tokens=64),  # 4,100+ prompt tokens: over capacity
])
def test_request_refusals_leave_the_server_serving(harness_binary, make):
    bad = make()
    with running(harness_binary) as h:
        status, body = h.call("POST", "/speech", bad)
        assert status == 400 and body == {"error": "invalid_request"}, (status, body)
        prompt_length(h, ask([image(256, 256)]))


def test_no_vision_encoder_is_501_and_speech_still_serves(harness_binary):
    with running(harness_binary, "--image-input", "off") as h:
        status, reply = h.call("POST", "/speech", ask([image(256, 256)]))
        assert status == 501 and reply == {"error": "image_input_unsupported"}, (status, reply)
        status, reply = h.call("POST", "/speech", dict(messages=[{"role": "user", "content": "Read this aloud."}],
                                                        mode="audio", max_new_tokens=64))
        assert status == 200 and reply["cancelled"] is False


def _send_all(h, bodies):
    results = [None] * len(bodies)
    def run(i):
        results[i] = h.call("POST", "/speech", bodies[i])
    threads = [threading.Thread(target=run, args=(i,)) for i in range(len(bodies))]
    for t in threads:
        t.start()
    for t in threads:
        t.join(120)
    return results


def test_simultaneous_image_requests_wait_their_turn_instead_of_429(harness_binary):
    # validation/image-queue-v1 rule 1: images decode on the handlers, two at a time, and a request waits for its turn
    # (images-v1 found 8 of 11 simultaneous image requests refused 429 frontend_busy)
    pictures = [image(1024 + 32 * i, 1024, noise=True) for i in range(4)]
    with running(harness_binary, "--queue-size", "32", "--handlers", "16") as h:  # as the server with 8 rows
        alone = [prompt_length(h, ask([p]))[1] for p in pictures]
        results = _send_all(h, [ask([pictures[i % 4]]) for i in range(11)])
        for i, (status, reply) in enumerate(results):
            assert status == 200 and reply["cancelled"] is False, (i, status, reply)
            assert reply["chunks"][0]["frames"] == alone[i % 4]
        # a request of two images, simultaneous with others, keeps its images in order
        both = prompt_length(h, ask(pictures[:2]))[1]
        results = _send_all(h, [ask(pictures[:2])] + [ask([pictures[3]])] * 4)
        assert results[0][0] == 200 and results[0][1]["chunks"][0]["frames"] == both


def test_image_requests_past_their_deadline_while_waiting_are_504_never_429(harness_binary):
    big = image(1448, 1448, noise=True)
    with running(harness_binary, "--request-timeout", "1", "--queue-size", "32", "--handlers", "16") as h:
        results = _send_all(h, [ask([big])] * 8)
        for status, reply in results:
            # served, or past the deadline waiting to decode (frontend_deadline) or queued (queue_deadline): never 429
            assert (status == 200 and reply["cancelled"] is False) or \
                (status, reply) in ((504, {"error": "frontend_deadline"}), (504, {"error": "queue_deadline"})), (status, reply)
        prompt_length(h, ask([image(256, 256)]))  # and the server serves on


def test_text_requests_keep_their_429_while_images_decode(harness_binary):
    # text's admission is unchanged (rule 4): a text request meeting a busy front end is refused 429 frontend_busy or
    # served, never made to wait
    pictures = [image(1448, 1448, noise=True)] * 4
    tts = dict(messages=[{"role": "user", "content": "Read this aloud."}], mode="audio", max_new_tokens=64)
    with running(harness_binary, "--queue-size", "32", "--handlers", "16") as h:
        results = _send_all(h, [ask([p]) for p in pictures] + [tts] * 6)
        for status, reply in results[:4]:
            assert status == 200, (status, reply)
        for status, reply in results[4:]:
            assert status == 200 or (status, reply) == (429, {"error": "frontend_busy"}), (status, reply)


def _log_lines(path):
    return path.read_text().split() if path.exists() else []


def test_handed_over_pixels_are_exactly_the_engines_own(harness_binary, tmp_path):
    # image-queue-v1 part 2: the handler thread decodes and preprocesses; the engine finds the pixel values instead
    # of decoding again. The fake engine decodes anyway and refuses the request unless both are identical.
    log = tmp_path / "images.log"
    pictures = [image(640, 480, noise=True), image(300, 900, mode="RGB", fmt="JPEG", quality=90),
                image(512, 512, mode="RGBA"), image(1024, 64, noise=True)]
    with running(harness_binary, "--image-handoff", "on", "--image-log", str(log), "--image-cache-mb", "0") as h:
        for p in pictures:
            prompt_length(h, ask([p]))
        prompt_length(h, ask(pictures[:2]))  # two images in one request, in order
        results = _send_all(h, [ask([p]) for p in pictures])  # at once
        assert all(st == 200 for st, _ in results), results
    assert _log_lines(log) == ["handoff"] * 10


def test_a_resent_image_skips_the_front_ends_decode_and_changes_nothing(harness_binary, tmp_path):
    # image-cache-v1: the front end keeps each decoded image's token count under its full bytes. The same bytes again:
    # no decode on the handler (so nothing handed over: the engine, here a fake without the GPU cache, decodes);
    # one byte different: a different image, decoded again.
    log = tmp_path / "images.log"
    a = image(640, 480, noise=True)
    raw = bytearray(base64.b64decode(a))
    raw[-20] ^= 1  # inside the IEND chunk's CRC: the same picture's bytes but one, refused as corrupt
    corrupt = base64.b64encode(bytes(raw)).decode()
    b = image(641, 480, noise=True)
    with running(harness_binary, "--image-handoff", "on", "--image-log", str(log)) as h:
        first, n1 = prompt_length(h, ask([a]))
        again, n2 = prompt_length(h, ask([a]))
        assert n1 == n2 and first["text"] == again["text"]
        prompt_length(h, ask([b]))
        for _ in range(2):  # a refused image is never kept: refused again, the same way
            assert h.call("POST", "/speech", ask([corrupt]))[0] == 400
    assert _log_lines(log) == ["handoff", "decode", "handoff"]


def test_image_cache_zero_keeps_nothing(harness_binary, tmp_path):
    log = tmp_path / "images.log"
    a = image(640, 480, noise=True)
    with running(harness_binary, "--image-handoff", "on", "--image-log", str(log), "--image-cache-mb", "0") as h:
        prompt_length(h, ask([a]))
        prompt_length(h, ask([a]))
    assert _log_lines(log) == ["handoff", "handoff"]
