#!/usr/bin/env python3
"""A stand-in for liquid-lfm3-server, for the host test of tools/measure_lfm3_first_request.py (validation/always-on-v1).
Not the server: no model, no GPU. It keeps what the tool observes: the command line (with --warmup), the ready line
(with "warmup_seconds" after a warm-up) and the warm-up's stderr line, POST /sessions, /generate, /sessions/ID/close,
POST /speech streamed (NDJSON) and unstreamed, GET /health. Each kind (text, tts, image, audio) pays COLD seconds once,
on its first request, unless --warmup paid them all before the ready line. FAKE_COLD (default 0.3) sets COLD.
FAKE_IMAGE_MISS (default 0): an image whose bytes this server has not seen costs that many seconds more (an image cache,
as the real server's: a repeat is a hit). The warm-up's image is never cached, as the server's is not.
Usage: fake_lfm3_first_request_server.py --experimental MODEL_DIR --token-file FILE --port 0 [--warmup] [...]
"""
import base64
import json
import os
import secrets
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

COLD = float(os.environ.get("FAKE_COLD", "0.3"))
STEP = 0.01
IMAGE_MISS = float(os.environ.get("FAKE_IMAGE_MISS", "0"))


def kind_of(path, body):
    if path == "/generate":
        return "text"
    parts = [p for m in body.get("messages", []) if isinstance(m.get("content"), list) for p in m["content"]]
    if any(p.get("type") == "image" for p in parts):
        return "image"
    if any(p.get("type") == "audio" for p in parts):
        return "audio"
    return "tts"


def main(argv):
    started = time.monotonic()
    assert argv[1] == "--experimental", argv
    rest = [a for a in argv[3:] if a != "--warmup"]
    warmup = len(rest) != len(argv) - 3
    args = dict(zip(rest[0::2], rest[1::2]))
    token = Path(args["--token-file"]).read_text().strip()
    cold, lock, seen = {"text", "tts", "image", "audio"}, threading.Lock(), set()

    def image_cost(body):
        keys = [p["image"] for m in body.get("messages", []) if isinstance(m.get("content"), list) for p in m["content"]
                if p.get("type") == "image"]
        with lock:
            miss = any(k not in seen for k in keys)
            seen.update(keys)
        if miss:
            time.sleep(IMAGE_MISS)

    def pay(kind):
        with lock:
            first = kind in cold
            cold.discard(kind)
        if first:
            time.sleep(COLD)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send_json(self, status, obj):
            data = json.dumps(obj).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self.send_json(200 if self.path == "/health" else 404, {"ready": True, "gpu_batching": False})

        def do_POST(self):
            if self.headers.get("Authorization") != "Bearer " + token:
                return self.send_json(401, {"error": "unauthorized"})
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/sessions":
                return self.send_json(200, {"session_id": secrets.token_urlsafe(24), "capacity": body["capacity"]})
            if self.path.endswith("/close"):
                return self.send_json(200, {"ok": True})
            kind = kind_of(self.path, body)
            pay(kind)
            if kind == "image":
                image_cost(body)
            time.sleep(STEP)
            if self.path == "/generate":
                return self.send_json(200, {"text": "hi", "generated_ids": [1], "finish_reason": "length",
                                           "session_id": body.get("session_id")})
            timings = dict(prefill_seconds=COLD if kind in cold else STEP, generate_seconds=STEP)
            if not body.get("stream"):
                return self.send_json(200, dict(cancelled=False, text="x", **timings))
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.end_headers()
            pcm = base64.b64encode(b"\0\0" * 240).decode()
            events = {"tts": [dict(event="audio", chunk=0, pcm_base64=pcm)] * 3,
                      "image": [dict(event="text", chunk=0, text="a")] * 3,
                      "audio": [dict(event="text", chunk=0, text="a"), dict(event="audio", chunk=0, pcm_base64=pcm)]}[kind]
            for e in events:
                self.wfile.write((json.dumps(e) + "\n").encode())
                self.wfile.flush()
                time.sleep(STEP)
            self.wfile.write((json.dumps(dict(event="done", cancelled=False, **timings)) + "\n").encode())

    server = ThreadingHTTPServer(("127.0.0.1", int(args.get("--port", "0"))), Handler)
    signal.signal(signal.SIGTERM, lambda *_: threading.Thread(target=server.shutdown).start())
    ready = {"ready": True, "port": server.server_address[1]}
    if warmup:
        t0 = time.monotonic()
        steps = []
        for kind in ("text", "tts", "image", "audio"):
            t = time.monotonic()
            pay(kind)
            if kind == "image":
                time.sleep(IMAGE_MISS)  # its image is a miss, and is not kept
            steps.append(dict(kind=kind, seconds=round(time.monotonic() - t, 3)))
        seconds = round(time.monotonic() - t0, 3)
        print(json.dumps({"warmup": dict(startup_seconds=round(time.monotonic() - started, 3), warmup_seconds=seconds,
                                         requests=steps)}, separators=(",", ":")), file=sys.stderr, flush=True)
        ready["warmup_seconds"] = seconds
    print(json.dumps(ready, separators=(",", ":")), flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
