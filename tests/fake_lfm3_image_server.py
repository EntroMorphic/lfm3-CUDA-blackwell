#!/usr/bin/env python3
"""A stand-in for liquid-lfm3-server's image path, for host tests of the Thor tools of validation/image-cache-v1 C3-C5
(tools/check_lfm3_image_cache_fixes.py, tools/measure_lfm3_image_stall.py). Not the server: no model, no GPU. It keeps
what those tools observe: the same command line and ready line, POST /speech (unstreamed and streamed NDJSON), GET
/health; a front end on the handler threads (token counts in an LRU of --image-cache-mb, keys only) and one engine
thread (rows in an LRU of the same budget, cost key + tokens x 4096) as in image-cache-v1 with its C3 fix: a token-count
hit skips the handler's decode only while the engine holds the image. Decoding is CPU burnt on the thread that decodes
(DECODE seconds of its own CPU time), encoding on the engine thread, so /proc shows where each happened.
Environment: FAKE_DEFAULT_CACHE_MB (the default budget, 512), FAKE_C3_UNFIXED=1 (the front end skips on any count hit:
the engine thread decodes), FAKE_FAULT_KEEP=1 (the engine never keeps rows), FAKE_LEAK_MB=N (each image request
leaves N MiB allocated), FAKE_STREAM_SECONDS (an audio stream's length, 1.2).
Usage: fake_lfm3_image_server.py --experimental MODEL_DIR --token-file FILE --port 0 [--image-cache-mb N] [...]
"""
import base64
import collections
import concurrent.futures
import hashlib
import json
import os
import queue
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from make_lfm3_c3_images import chunks, tokens  # noqa: E402

DECODE, ENCODE, GENERATE, PREFILL = 0.3, 0.2, 0.1, 0.05
LEAK = []


def burn(seconds):
    end = time.thread_time() + seconds
    x = 0
    while time.thread_time() < end:
        x += 1
    return x


def reply_text(body):
    """Deterministic: the images' bytes and the text parts; never max_new_tokens."""
    h = hashlib.sha256()
    for m in body["messages"]:
        parts = m["content"] if isinstance(m["content"], list) else [{"type": "text", "text": m["content"]}]
        for p in parts:
            h.update(p.get("image", p.get("text", "")).encode())
    return "reply " + h.hexdigest()[:16]


def image_tokens(key):
    ihdr = chunks(base64.b64decode(key))[0][1]
    return tokens(int.from_bytes(ihdr[0:4], "big"), int.from_bytes(ihdr[4:8], "big"))


class Lru:
    def __init__(self, budget):
        self.budget, self.items, self.lock = budget, collections.OrderedDict(), threading.Lock()

    def find(self, key):
        with self.lock:
            if key in self.items:
                self.items.move_to_end(key)
                return self.items[key][0]
        return None

    def holds(self, key):
        with self.lock:
            return key in self.items

    def put(self, key, value, cost):
        with self.lock:
            if cost > self.budget:
                return
            self.items[key] = (value, cost)
            while sum(c for _, c in self.items.values()) > self.budget:
                self.items.popitem(last=False)


class State:
    def __init__(self, budget):
        self.front, self.engine = Lru(budget), Lru(budget)
        self.handoff, self.jobs = {}, queue.Queue()
        self.unfixed = os.environ.get("FAKE_C3_UNFIXED") == "1"
        self.fault = os.environ.get("FAKE_FAULT_KEEP") == "1"
        threading.Thread(target=self.engine_loop, daemon=True).start()

    def engine_loop(self):
        while True:
            keys, done, out = self.jobs.get()
            t0 = time.monotonic()
            for key in keys:
                if self.engine.find(key.encode()) is not None:
                    continue
                if key not in self.handoff:
                    burn(DECODE)  # the engine thread decodes (the C3 defect)
                burn(ENCODE)
                if not self.fault:
                    self.engine.put(key.encode(), True, len(key) + image_tokens(key) * 4096)  # its own copy of the key
            time.sleep(PREFILL)
            out["prefill_seconds"] = time.monotonic() - t0
            burn(GENERATE)
            done.set()

    def speech(self, body):
        keys = [p["image"] for m in body["messages"] if isinstance(m["content"], list) for p in m["content"]
                if p.get("type") == "image"]
        for key in keys:  # the handler thread: the front end
            known = self.front.find(key)
            if known is not None and (self.unfixed or self.engine.holds(key.encode())):
                continue
            burn(DECODE)
            self.handoff[key] = True
            self.front.put(key, image_tokens(key), len(key) + 4)
        if keys and os.environ.get("FAKE_LEAK_MB"):
            LEAK.append(bytearray(os.urandom(int(os.environ["FAKE_LEAK_MB"]) << 20)))
        done, out = threading.Event(), {}
        self.jobs.put(([k for k in keys], done, out))
        done.wait()
        for key in keys:
            self.handoff.pop(key, None)
        return dict(cancelled=False, text=reply_text(body), prefill_seconds=out["prefill_seconds"], generate_seconds=GENERATE)


def main(argv):
    assert argv[1] == "--experimental", argv
    args = dict(zip(argv[3::2], argv[4::2]))
    token = Path(args["--token-file"]).read_text().strip()
    budget = int(args.get("--image-cache-mb", os.environ.get("FAKE_DEFAULT_CACHE_MB", "512"))) << 20
    state = State(budget)
    stream_seconds = float(os.environ.get("FAKE_STREAM_SECONDS", "1.2"))

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
            self.send_json(200 if self.path == "/health" else 404, {"ok": True})

        def do_POST(self):
            if self.headers.get("Authorization") != "Bearer " + token:
                return self.send_json(401, {"error": "unauthorized"})
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if not body.get("stream"):
                return self.send_json(200, state.speech(body))
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.end_headers()
            if body.get("mode") == "audio":
                end = time.monotonic() + stream_seconds
                while time.monotonic() < end:
                    pcm = base64.b64encode(b"\0\0" * 480).decode()
                    self.wfile.write((json.dumps(dict(event="audio", pcm_base64=pcm)) + "\n").encode())
                    self.wfile.flush()
                    time.sleep(0.02)
                self.wfile.write((json.dumps(dict(event="done", text="")) + "\n").encode())
            else:
                r = state.speech(body)
                self.wfile.write((json.dumps(dict(event="text", text=r["text"])) + "\n").encode())
                self.wfile.write((json.dumps(dict(event="done", **r)) + "\n").encode())

    class PooledServer(HTTPServer):
        """A fixed pool of handler threads, as the server's (--handlers): a thread's CPU time outlives its request."""
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=16)

        def process_request(self, request, client_address):
            self.pool.submit(self.serve_one, request, client_address)

        def serve_one(self, request, client_address):
            try:
                self.finish_request(request, client_address)
            finally:
                self.shutdown_request(request)

    server = PooledServer(("127.0.0.1", int(args.get("--port", "0"))), Handler)
    signal.signal(signal.SIGTERM, lambda *_: threading.Thread(target=server.shutdown).start())
    print(json.dumps({"ready": True, "port": server.server_address[1]}), flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
