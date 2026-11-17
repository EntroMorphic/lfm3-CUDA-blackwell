"""Host tests for --warmup (validation/always-on-v1; native/include/lfm/lfm3_warmup.hpp) behind the harness's fake engine.

The harness runs liquid-lfm3-server's own warm-up code (lfm::http::warmup::serve_with_warmup) on the real HTTP layer
and Service, with the fake engine standing in for the GPU: the four requests (text /generate in its own session, TTS
/speech streamed, the generated 3840 x 2160 PNG, the generated clip) go through validation, the front end (the image
decoded, preprocessed and handed over), the queue and the fake's engine ops. Covered here: /health 503 while warming
and 200 after, POSTs refused 503 warming_up meanwhile, the ready line after the warm-up, nothing left behind (sessions,
queue, both image caches, the pixel handoff), a failure's exit status and message, a stop during the warm-up, sd_notify's
READY=1, and that without --warmup nothing changes. Not covered (Thor, validation/always-on-v1 rules 0-2 and 5): whether
the real engine is warm afterwards, the engine's own image cache (WorkerCore's; the fake has a stand-in with the same
observer), timings and RSS. Every build variant of the harness runs (plain, ASan+UBSan, TSan).
"""
import base64
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path

import pytest

from test_lfm3_http_native import CHECKPOINT, TOKEN, harness_binary, parse_response  # noqa: F401  (fixture)

KINDS = ["text", "tts", "image", "audio"]


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Process:
    """The harness started without waiting for its ready line: stdout and stderr read by threads, lines timestamped."""

    def __init__(self, binary, *options, env=None):
        prefix = ["setarch", os.uname().machine, "-R"] if binary.name.endswith("-tsan") else []
        self.process = subprocess.Popen(
            [*prefix, str(binary), str(CHECKPOINT), *map(str, options)], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=dict(os.environ, LFM3_SERVER_TOKEN=TOKEN, TSAN_OPTIONS="halt_on_error=1", **(env or {})))
        self.out, self.err = [], []
        self.threads = [threading.Thread(target=self._pump, args=(f, into), daemon=True)
                        for f, into in ((self.process.stdout, self.out), (self.process.stderr, self.err))]
        for t in self.threads:
            t.start()

    @staticmethod
    def _pump(f, into):
        for line in f:
            into.append((time.monotonic(), line.decode()))

    def wait_ready(self, timeout=120):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if self.out:
                return json.loads(self.out[0][1])
            if self.process.poll() is not None:
                break
            time.sleep(0.01)
        raise AssertionError(("no ready line", self.process.poll(), self.stderr()))

    def stderr(self):
        return "".join(line for _, line in self.err)

    def finish(self, timeout=60, signal_first=True):
        """(exit status, stdout lines, stderr text) after SIGTERM (never SIGKILL)."""
        if signal_first and self.process.poll() is None:
            self.process.send_signal(signal.SIGTERM)
        code = self.process.wait(timeout=timeout)
        for t in self.threads:
            t.join(timeout=10)
        return code, [line for _, line in self.out], self.stderr()


def call(port, method, path, body=None, timeout=60):
    payload = b"" if body is None else json.dumps(body).encode()
    head = [f"{method} {path} HTTP/1.1", "Host: 127.0.0.1"]
    if method == "POST":
        head += ["Content-Type: application/json", f"Content-Length: {len(payload)}", "Authorization: Bearer " + TOKEN]
    with socket.create_connection(("127.0.0.1", port), timeout=timeout) as s:
        s.sendall(("\r\n".join(head) + "\r\n\r\n").encode() + payload)
        chunks = []
        while True:
            block = s.recv(65536)
            if not block:
                break
            chunks.append(block)
    return parse_response(b"".join(chunks))


def report_of(stderr):
    lines = [json.loads(line) for line in stderr.splitlines() if line.startswith('{"warmup"')]
    assert len(lines) == 1, stderr
    return lines[0]["warmup"]


def wav_b64(path):
    return base64.b64encode(path.read_bytes()).decode()


def bodies(inputs):
    """One request of each kind, as a client sends them, with the warm-up's own image and clip."""
    image = wav_b64(inputs / "warmup.png")
    clip = wav_b64(inputs / "warmup.wav")
    return dict(
        text=dict(messages=[{"role": "user", "content": "hello there"}], max_new_tokens=32),
        tts=dict(messages=[{"role": "user", "content": "Read this. Then this."}], mode="audio", max_new_tokens=64),
        image=dict(messages=[{"role": "user", "content": [{"type": "image", "image": image},
                                                          {"type": "text", "text": "What is it?"}]}],
                   mode="text", max_new_tokens=16),
        audio=dict(messages=[{"role": "user", "content": [{"type": "audio", "audio": clip}]}], mode="interleaved",
                   max_new_tokens=32))


def send(port, kind, body):
    if kind == "text":
        status, created = call(port, "POST", "/sessions", {"capacity": 1024})
        assert status == 200, created
        sid = created["session_id"]
        try:
            return call(port, "POST", "/generate", dict(body, session_id=sid))
        finally:
            assert call(port, "POST", f"/sessions/{sid}/close", {}) == (200, {"ok": True})
    return call(port, "POST", "/speech", body)


def test_generated_inputs_decode_as_intended(harness_binary, tmp_path):
    pil = pytest.importorskip("PIL.Image")
    p = Process(harness_binary, "--warmup-inputs", tmp_path)
    p.wait_ready()
    assert p.finish()[0] == 0
    with pil.open(tmp_path / "warmup.png") as im:
        im.load()
        assert im.size == (3840, 2160) and im.mode == "RGB"
        assert im.getpixel((0, 0)) != im.getpixel((120, 0))  # tiles of 120 pixels
        assert im.getpixel((0, 0)) == im.getpixel((119, 119))
    with wave.open(str(tmp_path / "warmup.wav")) as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()) == (16000, 1, 2, 32000)
    # deterministic: the same bytes from a second process
    again = tmp_path / "again"
    again.mkdir()
    p = Process(harness_binary, "--warmup-inputs", again)
    p.wait_ready()
    assert p.finish()[0] == 0
    for name in ("warmup.png", "warmup.wav"):
        assert (again / name).read_bytes() == (tmp_path / name).read_bytes()


def test_the_measuring_tools_replica_is_the_same_image_and_clip(harness_binary, tmp_path):
    # tools/measure_lfm3_first_request.py rebuilds the warm-up's requests for rule 3's RSS comparison: the same pixels
    # (its own PNG encoding) and the same clip, byte for byte
    pil = pytest.importorskip("PIL.Image")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    import measure_lfm3_first_request as tool
    p = Process(harness_binary, "--warmup-inputs", tmp_path)
    p.wait_ready()
    assert p.finish()[0] == 0
    assert tool.warmup_wav() == (tmp_path / "warmup.wav").read_bytes()
    (tmp_path / "replica.png").write_bytes(tool.warmup_png())
    with pil.open(tmp_path / "warmup.png") as a, pil.open(tmp_path / "replica.png") as b:
        assert a.size == b.size and a.mode == b.mode and a.tobytes() == b.tobytes()


def test_health_is_503_while_warming_then_200_and_the_ready_line_comes_after(harness_binary, tmp_path):
    port = free_port()
    ops = tmp_path / "ops"
    # slow enough to watch: 16 text tokens and each speech step at 60 ms
    p = Process(harness_binary, "--warmup", "--port", port, "--token-delay-ms", 60, "--trace-ops", ops)
    seen = []  # (time, status, body)
    refused = None
    while not p.out and p.process.poll() is None:
        try:
            status, body = call(port, "GET", "/health", timeout=5)
        except OSError:  # not listening yet
            time.sleep(0.01)
            continue
        seen.append((time.monotonic(), status, body))
        if status == 503 and refused is None:
            refused = call(port, "POST", "/sessions", {"capacity": 1024})
        time.sleep(0.02)
    ready = p.wait_ready()
    ready_at = p.out[0][0]
    assert list(ready) == ["ready", "port", "warmup_seconds"] and ready["ready"] is True and ready["port"] == port
    assert isinstance(ready["warmup_seconds"], float) and ready["warmup_seconds"] > 0.5
    before = [(s, b) for t, s, b in seen if t < ready_at]
    assert before and all(x == (503, {"ready": False, "gpu_batching": False}) for x in before), before
    assert refused == (503, {"error": "warming_up"})
    assert call(port, "GET", "/health") == (200, {"ready": True, "gpu_batching": False})
    # the report came before the ready line, on stderr: startup and each kind's time
    report = report_of(p.stderr())
    assert [r["kind"] for r in report["requests"]] == KINDS
    assert all(r["seconds"] > 0 for r in report["requests"])
    assert report["warmup_seconds"] == ready["warmup_seconds"] and report["startup_seconds"] >= report["warmup_seconds"]
    for r in report["requests"][1:]:  # the engine's own timings, from the /speech replies
        assert r["prefill_seconds"] == 0.25 and r["generate_seconds"] == 0.5
    code, out, err = p.finish()
    assert code == 0 and len(out) == 1 and err.count("\n") == 1, (code, out, err)
    # every kind went through the engine: a session's create, sliced text and close; three speech rows
    trace = ops.read_text().split()
    assert trace[:3] == ["create", "text_begin", "text_step"] and "close" in trace
    assert trace.count("speech_admit") == 3


def test_warmup_leaves_no_session_queue_or_cache_entry(harness_binary, tmp_path):
    log, inputs = tmp_path / "images.log", tmp_path / "inputs"
    inputs.mkdir()
    p = Process(harness_binary, "--warmup", "--image-handoff", "on", "--image-log", log, "--engine-cache-mb", 64,
                "--warmup-inputs", inputs, "--queue-size", 1, "--max-sessions", 2, "--total-capacity", 32768)
    port = p.wait_ready()["port"]
    # the warm-up's image met the engine once, handed over (decoded once, by the front end)
    assert log.read_text().split() == ["handoff"]
    # no session left: both slots and the whole capacity are free
    sids = []
    for _ in range(2):
        status, created = call(port, "POST", "/sessions", {"capacity": 16384})
        assert status == 200, created
        sids.append(created["session_id"])
    for sid in sids:
        assert call(port, "POST", f"/sessions/{sid}/close", {}) == (200, {"ok": True})
    # nothing in either image cache: the same image from a client is decoded by the front end again (its token count
    # was never kept) and the engine has no rows for it (not "cached"); sent again, both caches now work as configured
    body = bodies(inputs)["image"]
    assert send(port, "image", body)[0] == 200
    assert log.read_text().split() == ["handoff", "handoff"]
    assert send(port, "image", body)[0] == 200
    assert log.read_text().split() == ["handoff", "handoff", "cached"]
    # the queue is empty: with --queue-size 1 a request is admitted at once
    assert send(port, "tts", bodies(inputs)["tts"])[0] == 200
    code, _, err = p.finish()
    assert code == 0, err


def test_replies_after_warmup_equal_a_cold_servers(harness_binary, tmp_path):
    # rule 2's shape on the fake: each kind's first request after the ready line, with and without --warmup
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    replies = {}
    for warm in (False, True):
        p = Process(harness_binary, *(["--warmup"] if warm else []), "--image-handoff", "on", "--engine-cache-mb", 64,
                    "--warmup-inputs", inputs)
        port = p.wait_ready()["port"]
        got = {}
        for kind, body in bodies(inputs).items():
            status, reply = send(port, kind, body)
            assert status == 200, (warm, kind, reply)
            got[kind] = {k: v for k, v in reply.items() if k not in ("prefill_seconds", "generate_seconds", "session_id")}
        replies[warm] = got
        code, _, err = p.finish()
        assert code == 0, err
    assert replies[True] == replies[False]


@pytest.mark.parametrize("op,options,kind", [
    ("text_begin", [], "text"),
    ("speech_admit", [], "tts"),
    ("speech_step", [], "tts"),
    ("speech", ["--speech-rows", 0], "tts"),
])
def test_a_warmup_failure_exits_1_with_its_reason(harness_binary, op, options, kind):
    p = Process(harness_binary, "--warmup", "--fail-op", op, *options)
    code, out, err = p.finish(signal_first=False)
    assert code == 1 and out == [], (code, out, err)
    assert err.strip().endswith(f"lfm3-http-harness: warm-up failed: warm-up {kind} request: 503 worker_unavailable"), err


def test_a_later_kinds_failure_also_fails(harness_binary):
    # the image: refused by the engine at its speech admit, the second (text and TTS warmed)
    p = Process(harness_binary, "--warmup", "--fail-op", "speech_admit", "--fail-after", 2)
    code, out, err = p.finish(signal_first=False)
    assert code == 1 and out == [] and "warm-up image request: 503 worker_unavailable" in err, (code, out, err)


def test_a_stop_while_warming_is_a_stop(harness_binary):
    port = free_port()
    p = Process(harness_binary, "--warmup", "--port", port, "--token-delay-ms", 500)
    for _ in range(1000):
        try:
            if call(port, "GET", "/health", timeout=5)[0] == 503:
                break
        except OSError:
            time.sleep(0.01)
    code, out, err = p.finish()
    assert code == 0 and out == [] and err == "lfm3-http-harness: stopped during warm-up\n", (code, out, err)


def test_kinds_the_engine_cannot_run_are_skipped(harness_binary):
    p = Process(harness_binary, "--warmup", "--speech", "off")
    p.wait_ready()
    code, _, err = p.finish()
    report = report_of(err)
    assert code == 0
    assert report["requests"][0]["kind"] == "text" and "seconds" in report["requests"][0]
    assert [(r["kind"], r.get("skipped")) for r in report["requests"][1:]] == [
        ("tts", "speech_unsupported"), ("image", "image_input_unsupported"), ("audio", "audio_input_unsupported")]
    p = Process(harness_binary, "--warmup", "--image-input", "off")
    p.wait_ready()
    code, _, err = p.finish()
    assert code == 0 and [r.get("skipped") for r in report_of(err)["requests"]] == [None, None, "image_input_unsupported", None]


@pytest.mark.parametrize("warm", [False, True])
@pytest.mark.parametrize("abstract", [False, True])
def test_ready_is_notified_to_systemd(harness_binary, tmp_path, warm, abstract):
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    address = f"\0lfm3-notify-{os.getpid()}-{warm}" if abstract else str(tmp_path / "notify")
    sock.bind(address)
    sock.settimeout(120)
    try:
        p = Process(harness_binary, *(["--warmup"] if warm else []),
                    env=dict(NOTIFY_SOCKET="@" + address[1:] if abstract else address))
        assert sock.recv(64) == b"READY=1"
        p.wait_ready()
        code, out, err = p.finish()
        assert code == 0 and len(out) == 1, (code, out, err)
    finally:
        sock.close()


def test_without_warmup_nothing_changes(harness_binary):
    p = Process(harness_binary)
    ready = p.wait_ready()
    assert list(ready) == ["ready", "port"] and ready["ready"] is True
    assert p.out[0][1] == '{"ready":true,"port":%d}\n' % ready["port"]
    assert call(ready["port"], "GET", "/health") == (200, {"ready": True, "gpu_batching": False})
    status, created = call(ready["port"], "POST", "/sessions", {"capacity": 1024})
    assert status == 200
    code, out, err = p.finish()
    assert code == 0 and len(out) == 1 and err == "", (code, out, err)


def test_warmup_option_parsing(harness_binary):
    p = Process(harness_binary, "--warmup", "--port", 0, "--warmup")
    code, out, err = p.finish(signal_first=False)
    assert code == 1 and out == [] and "duplicate option --warmup" in err
    # an option's value is never taken for the flag
    p = Process(harness_binary, "--fail-op", "--warmup")
    p.wait_ready()
    code, out, err = p.finish()
    assert code == 0 and err == "" and list(json.loads(out[0])) == ["ready", "port"]
