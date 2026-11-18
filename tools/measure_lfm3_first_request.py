#!/usr/bin/env python3
"""validation/always-on-v1 rules 0, 1 and 3 (RSS), and rule 4's startup time: how much slower each kind's first
request is on a fresh liquid-lfm3-server, with and without --warmup. Black box (Thor, lfm3-profile-v1, a quiet
machine): every server is started fresh like a deployment (token file, --port 0) and driven over HTTP only.

Kinds, in this order, each request timed to its first token at the client (loopback, request's first byte on):
  text   /generate in a fresh session (check_lfm3_audio_soak.TEXTS[0]) with max_new_tokens 1: /generate does not stream,
         so its time to first token is the whole reply of a one-token generation (the session's create and close are
         outside the timing);
  tts    /speech streamed, audio mode (check_lfm3_speech_stream's numbers-normalized-s1 case): the first audio event;
  image  /speech streamed, text mode, the stage-0 4K photograph (BUILD/vimg-e2e-a1/req/cherry.json, as
         check_lfm3_image_server.as_client sends it; 32 new tokens, as measure_lfm3_image_stall): the first text event;
  audio  /speech streamed, interleaved, the spoken question q0 (BUILD/audio-in-listen/q0-16k.wav, as
         check_lfm3_speech_stream's spoken-q0; 256 new tokens): the first text or audio event.
Each kind's reply also gives the server's own prefill_seconds and generate_seconds (the done event); /generate has none.

Parts (--parts, default a,b,c,d):
  a  one fresh server without --warmup: each kind's 1st, 2nd and 3rd request, the kinds in order (rule 0);
  b  each kind first on its own fresh server without --warmup: its 1st, 2nd and 3rd request (rule 0);
  c  --servers (3) fresh servers with --warmup, each as in a (rule 1: per kind, the median over servers of the 1st
     request within max(10%, 50 ms) of the median 3rd; rule 4: startup, from the server's own warm-up line);
     with --image-third PNG (rule 1 as amended after window 15), each warm server's 3rd image request carries that
     image instead: one of the same size the server has never seen, so the 3rd is an image-cache miss, as the 1st is
     (the same photograph three times makes the 2nd and 3rd hits);
  d  idle RSS (rule 3): a fresh server without --warmup after the warm-up's own four requests (the same bodies, the same
     generated image's pixels and clip, rebuilt here), against a fresh --warmup server, each idle 10 s first.
Rule 0's reading: proceed if any kind's 1st request (a or b) is slower than its 3rd by more than 1 s.
Writes OUT_JSON; exits 1 on a failed request or a server that does not exit 0 on SIGTERM (never SIGKILL).
Usage: measure_lfm3_first_request.py SERVER MODEL_DIR BUILD_DIR OUT_JSON [--parts a,b,c,d] [--servers N]
                                     [--image-request PATH] [--image-third PNG] [--audio-wav PATH]
                                     [--server-arg ARG]...
"""
import argparse
import base64
import json
import os
import statistics
import struct
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_lfm3_image_server import as_client  # noqa: E402
from check_lfm3_speech_stream import NUMBERS, Server  # noqa: E402

KINDS = ("text", "tts", "image", "audio")
TEXT_BODY = dict(max_new_tokens=1, messages=[{"role": "user", "content":
                                              "Explain in one paragraph how a lighthouse warns ships at night."}])
SETTLE = 10  # seconds idle before an RSS reading


# ------------------------------------------------------------------ the warm-up's own requests (lfm/lfm3_warmup.hpp)

def warmup_png():
    """The warm-up image's pixels (3840 x 2160 RGB, 120-pixel tiles, lfm::http::warmup::png_4k's colour hash), as a
    PNG of our own encoding: the same pixels, so the same decoded image, token count and encoder input."""
    w, h, tile = 3840, 2160, 120

    def colour(tx, ty, c):
        x = ((tx * 73856093) ^ (ty * 19349663) ^ (c * 83492791)) & 0xffffffff
        x ^= x >> 13
        x = (x * 0x5bd1e995) & 0xffffffff
        x ^= x >> 15
        return x & 0xff
    bands = []
    for ty in range(h // tile):
        row = bytes(colour(x // tile, ty, c) for x in range(w) for c in range(3))
        bands.append((b"\0" + row) * tile)
    raw = b"".join(bands)

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def warmup_wav():
    """lfm::http::warmup::wav_clip, byte for byte (integer arithmetic, C++'s truncating division)."""
    rate, n = 16000, 32000
    phase, pcm = 0, bytearray()
    for i in range(n):
        hz = 220 + 660 * i // n
        phase = (phase + hz * 65536 // rate) & 0xffffffff
        p = phase & 0xffff
        tri = p - 16384 if p < 32768 else 49152 - p
        env = (i if i < n // 2 else n - i) * 2
        q = abs(tri * env) // n // 2
        pcm += struct.pack("<h", q if tri * env >= 0 else -q)
    head = b"RIFF" + struct.pack("<I", 36 + n * 2) + b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16)
    return head + b"data" + struct.pack("<I", n * 2) + bytes(pcm)


def warmup_bodies():
    """The warm-up's four requests as a client sends them (text: the /generate body, without its session)."""
    b64 = lambda data: base64.b64encode(data).decode()  # noqa: E731
    return dict(
        text=dict(messages=[{"role": "user", "content": "Warm-up check: reply with one short greeting."}],
                  max_new_tokens=16),
        tts=dict(messages=[{"role": "system", "content": "Perform TTS in American English using a female voice."},
                           {"role": "user", "content": "The service is warming up."}],
                 mode="audio", max_new_tokens=48, seed=0, stream=True),
        image=dict(messages=[{"role": "system", "content": "Describe the provided image."},
                             {"role": "user", "content": [{"type": "image", "image": b64(warmup_png())},
                                                          {"type": "text", "text": "What is in the image?"}]}],
                   mode="text", max_new_tokens=8, seed=0),
        audio=dict(messages=[{"role": "system", "content": "Respond with interleaved text and audio."},
                             {"role": "user", "content": [{"type": "audio", "audio": b64(warmup_wav())}]}],
                   mode="interleaved", max_new_tokens=32, seed=0))


# ------------------------------------------------------------------ requests

def measured_bodies(build, image_request=None, audio_wav=None):
    b = Path(build)
    image = as_client(json.loads(Path(image_request or b / "vimg-e2e-a1/req/cherry.json").read_text()))
    image["max_new_tokens"] = 32
    clip = base64.b64encode(Path(audio_wav or b / "audio-in-listen/q0-16k.wav").read_bytes()).decode()
    return dict(
        text=TEXT_BODY,
        tts=dict(messages=[{"role": "system", "content": "Perform TTS in American English using a female voice."},
                           {"role": "user", "content": NUMBERS}], mode="audio", max_new_tokens=512, seed=1),
        image=image,
        audio=dict(messages=[{"role": "system", "content": "Respond with interleaved text and audio."},
                             {"role": "user", "content": [{"type": "audio", "audio": clip}]}],
                   mode="interleaved", max_new_tokens=256, seed=1))


def post(server, path, body, timeout=600):
    req = urllib.request.Request(f"http://127.0.0.1:{server.port}{path}", data=json.dumps(body).encode(), method="POST",
                                 headers={"Authorization": "Bearer " + server.token, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def text_request(server, body):
    """(status, seconds of the /generate call alone, reply) in a fresh session, closed after."""
    status, created = post(server, "/sessions", {"capacity": 1024})
    if status != 200:
        return status, None, created
    try:
        t0 = time.monotonic()
        status, reply = post(server, "/generate", dict(body, session_id=created["session_id"]))
        return status, time.monotonic() - t0, reply
    finally:
        post(server, f"/sessions/{created['session_id']}/close", {})


def one(server, kind, body):
    """One request: {status, ttft, total, prefill_seconds, generate_seconds}."""
    if kind == "text":
        status, seconds, reply = text_request(server, body)
        return dict(status=status, ttft=seconds, total=seconds)
    t0 = time.monotonic()
    status, events, times, _ = server.stream(body)
    total = time.monotonic() - t0
    want = {"tts": ("audio",), "image": ("text",), "audio": ("text", "audio")}[kind]
    first = next((t for e, t in zip(events, times) if e.get("event") in want), None)
    done = next((e for e in events if e.get("event") == "done"), {})
    out = dict(status=status, ttft=first, total=total, done=bool(done))
    for k in ("prefill_seconds", "generate_seconds"):
        if k in done:
            out[k] = done[k]
    if status != 200 or first is None or not done:
        out["error"] = events[-1] if events else None
    return out


def rss(pid):
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    return None


class Fresh:
    """A fresh server: startup (Popen to the ready line), and on stop its exit status and warm-up report."""

    def __init__(self, binary, model, scratch, extra):
        t0 = time.monotonic()
        self.s = Server(binary, model, scratch, extra=tuple(extra))
        self.startup = time.monotonic() - t0
        self.record = dict(startup_seconds_client=self.startup, args=list(extra))

    def stop(self):
        code, err = self.s.stop()
        self.record["exit"] = code
        for line in err.splitlines():
            if line.startswith('{"warmup"'):
                self.record["warmup"] = json.loads(line)["warmup"]
        return code == 0


def with_image(body, png):
    """BODY with each of its images replaced by PNG's bytes (base64), as a client sends an image."""
    body = json.loads(json.dumps(body))
    for m in body["messages"]:
        if isinstance(m["content"], list):
            for part in m["content"]:
                if part.get("type") == "image":
                    part["image"] = base64.b64encode(png).decode()
    return body


def sequence(server, bodies, kinds, repeats=3, third=None):
    """Each kind's 1st..repeats-th request, the kinds in order: {kind: [result, ...]}. THIRD: {kind: body} for the
    3rd request of that kind instead of its usual body."""
    out = {}
    for kind in kinds:
        out[kind] = []
        for i in range(repeats):
            r = one(server, kind, (third or {}).get(kind, bodies[kind]) if i == 2 else bodies[kind])
            out[kind].append(r)
            print(kind, i + 1, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    return out


def failed(results):
    return any(r["status"] != 200 or r["ttft"] is None for rs in results.values() for r in rs)


def rule0(parts):
    """Per kind and part: 1st minus 3rd; proceed if any is over 1 s."""
    rows = {}
    for part in ("a", "b"):
        for kind, rs in (parts.get(part) or {}).get("requests", {}).items():
            rows.setdefault(kind, {})[part] = dict(first=rs[0]["ttft"], third=rs[2]["ttft"],
                                                   first_minus_third=rs[0]["ttft"] - rs[2]["ttft"])
    proceed = any(v["first_minus_third"] > 1.0 for k in rows.values() for v in k.values())
    return dict(per_kind=rows, proceed=proceed)


def rule1(servers):
    """Per kind: the median over servers of the 1st and of the 3rd; within max(10% of the 3rd, 50 ms)."""
    rows, ok = {}, True
    for kind in KINDS:
        firsts = [s["requests"][kind][0]["ttft"] for s in servers if kind in s.get("requests", {})]
        thirds = [s["requests"][kind][2]["ttft"] for s in servers if kind in s.get("requests", {})]
        if not firsts:
            continue
        first, third = statistics.median(firsts), statistics.median(thirds)
        bound = max(0.10 * third, 0.050)
        rows[kind] = dict(first_median=first, third_median=third, bound=bound, passed=abs(first - third) <= bound,
                          firsts=firsts, thirds=thirds)
        ok &= rows[kind]["passed"]
    return dict(per_kind=rows, passed=ok and bool(rows))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("server")
    ap.add_argument("model")
    ap.add_argument("build")
    ap.add_argument("out_json")
    ap.add_argument("--parts", default="a,b,c,d")
    ap.add_argument("--servers", type=int, default=3)
    ap.add_argument("--image-request")
    ap.add_argument("--image-third", help="part c: a never-sent PNG of the same size for each warm server's 3rd image")
    ap.add_argument("--audio-wav")
    ap.add_argument("--server-arg", action="append", default=[], help="passed to every server (repeat it)")
    ap.add_argument("--settle", type=float, default=SETTLE, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    parts = set(a.parts.split(","))
    bodies = measured_bodies(a.build, a.image_request, a.audio_wav)
    report = dict(kinds=list(KINDS), parts=sorted(parts), server=a.server, server_args=a.server_arg,
                  text_ttft="the /generate reply at max_new_tokens 1 (no streaming on /generate)")
    ok = True
    out = Path(a.out_json)

    def save():
        out.write_text(json.dumps(report, indent=1))

    with tempfile.TemporaryDirectory() as scratch:
        def run(extra, kinds, label, third=None):
            nonlocal ok
            print(f"== {label}", flush=True)
            f = Fresh(a.server, a.model, scratch, [*a.server_arg, *extra])
            try:
                f.record["requests"] = sequence(f.s, bodies, kinds, third=third)
            finally:
                ok &= f.stop()
            ok &= not failed(f.record["requests"])
            return f.record

        if "a" in parts:
            report["a"] = run([], KINDS, "a: one cold server, the kinds in order")
            save()
        if "b" in parts:
            report["b"] = dict(requests={}, servers={})
            for kind in KINDS:
                rec = run([], (kind,), f"b: {kind} first on a cold server")
                report["b"]["requests"][kind] = rec["requests"][kind]
                report["b"]["servers"][kind] = {k: v for k, v in rec.items() if k != "requests"}
            save()
        if "c" in parts:
            third = None
            if a.image_third:
                third = dict(image=with_image(bodies["image"], Path(a.image_third).read_bytes()))
                report["image_third"] = a.image_third
            report["c"] = [run(["--warmup"], KINDS, f"c: warm server {i + 1}", third) for i in range(a.servers)]
            report["rule1"] = rule1(report["c"])
            startups = [s["warmup"]["startup_seconds"] for s in report["c"] if "startup_seconds" in s.get("warmup", {})]
            report["rule4_startup_seconds"] = dict(each=startups, median=statistics.median(startups) if startups else None)
            save()
        if "d" in parts:
            print("== d: idle RSS", flush=True)
            warm_bodies = warmup_bodies()
            readings = {}
            for label, extra in (("cold_after_four", []), ("warm", ["--warmup"])):
                f = Fresh(a.server, a.model, scratch, [*a.server_arg, *extra])
                try:
                    if not extra:
                        for kind in KINDS:
                            r = one(f.s, kind, warm_bodies[kind])
                            ok &= r["status"] == 200 and r["ttft"] is not None
                    time.sleep(a.settle)
                    readings[label] = rss(f.s.proc.pid)
                finally:
                    ok &= f.stop()
            bound = readings["cold_after_four"] + (64 << 20)
            report["rule3_rss"] = dict(readings, bound=bound, passed=readings["warm"] <= bound,
                                       settle_seconds=a.settle)
            save()
    if "a" in report or "b" in report:
        report["rule0"] = rule0(report)
    report["passed_requests"] = ok
    save()
    print("rule0", json.dumps(report.get("rule0", {}).get("proceed")), "rule1", json.dumps(report.get("rule1", {}).get("passed")),
          "rule3", json.dumps(report.get("rule3_rss", {}).get("passed")), "requests", "OK" if ok else "FAILED", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
