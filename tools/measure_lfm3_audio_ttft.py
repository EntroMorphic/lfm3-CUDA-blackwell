#!/usr/bin/env python3
"""validation/audio-ttft-v1 stage 0: the server's time to first token on audio-in ASR requests, one request at a time.
Starts liquid-lfm3-server (no extra options), then per pass and clip sends a streamed /speech request in text mode
("Perform ASR in English.", the clip, 32 new tokens) and records:
- from_file_seconds: from reading the WAV (base64 encoding and building the JSON included) to the first "text" event;
  HF's counterpart is time_lfm3_audio_hf.py's time to first token, from reading the file;
- from_send_seconds: from the request's first byte to the first "text" event;
- the done event's prefill_seconds and generate_seconds, and the reply's text.
One untimed pass over all clips, then PASSES timed passes. Writes OUT_JSON {"runs": [...], "medians": {clip: {...}},
"server_exit": code}; exits 1 if any request fails or the server exits nonzero.
Usage: measure_lfm3_audio_ttft.py SERVER MODEL_DIR CLIPS_JSON OUT_JSON PASSES
"""
import base64
import json
import socket
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_lfm3_speech_stream import Server  # noqa: E402


def one(server, wav_path):
    t_file = time.monotonic()
    clip = base64.b64encode(Path(wav_path).read_bytes()).decode()
    payload = json.dumps(dict(messages=[{"role": "system", "content": "Perform ASR in English."},
                                        {"role": "user", "content": [{"type": "audio", "audio": clip}]}],
                              mode="text", max_new_tokens=32, seed=0, stream=True)).encode()
    head = (f"POST /speech HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nConnection: close\r\n"
            f"Content-Length: {len(payload)}\r\nAuthorization: Bearer {server.token}\r\n\r\n").encode()
    s = socket.create_connection(("127.0.0.1", server.port), timeout=600)
    try:
        t_send = time.monotonic()
        s.sendall(head + payload)
        data, first = b"", None
        while True:
            block = s.recv(1 << 16)
            if not block:
                break
            data += block
            if first is None and b'"event":"text"' in data:
                first = time.monotonic()
            if b'"event":"done"' in data or b'"event":"error"' in data:
                break
    finally:
        s.close()
    status = int(data.split(b" ", 2)[1]) if data.startswith(b"HTTP/") else None
    events = []
    for line in data.partition(b"\r\n\r\n")[2].split(b"\n"):
        line = line.strip()
        if line.startswith(b"{"):
            try:
                events.append(json.loads(line))
            except ValueError:
                pass
    done = next((e for e in events if e.get("event") == "done"), None)
    text = "".join(e.get("text", "") for e in events if e.get("event") == "text")
    ok = status == 200 and done is not None and first is not None
    return dict(ok=ok, status=status, from_file_seconds=(first - t_file) if first else None,
                from_send_seconds=(first - t_send) if first else None,
                prefill_seconds=done.get("prefill_seconds") if done else None,
                generate_seconds=done.get("generate_seconds") if done else None, text=text,
                error=next((e.get("error") for e in events if e.get("event") == "error"), None))


def main(server_bin, model, clips_json, out_json, passes):
    clips = json.loads(Path(clips_json).read_text())
    runs, failed = [], False
    with tempfile.TemporaryDirectory() as scratch:
        s = Server(server_bin, model, scratch)
        try:
            for p in range(int(passes) + 1):
                for c in clips:
                    r = one(s, c["wav"])
                    r.update(clip=c["name"], timed=p > 0, pass_=p)
                    runs.append(r)
                    failed |= not r["ok"]
                    print(json.dumps({k: r[k] for k in ("clip", "pass_", "ok", "from_file_seconds", "from_send_seconds",
                                                        "prefill_seconds")}), flush=True)
        finally:
            code, err = s.stop()
    med = {}
    for c in clips:
        t = [r for r in runs if r["clip"] == c["name"] and r["timed"] and r["ok"]]
        if t:
            med[c["name"]] = {k: statistics.median(r[k] for r in t)
                              for k in ("from_file_seconds", "from_send_seconds", "prefill_seconds")}
            med[c["name"]]["texts_identical"] = len({r["text"] for r in t}) == 1
            med[c["name"]]["text"] = t[0]["text"]
    out = dict(runs=runs, medians=med, server_exit=code, server_stderr=err[-2000:])
    Path(out_json).write_text(json.dumps(out, indent=1) + "\n")
    for n, m in med.items():
        print(f"{n}: time to first token {m['from_file_seconds']:.4f} s from the file, {m['from_send_seconds']:.4f} s "
              f"from the first byte; prefill {m['prefill_seconds']:.4f} s; texts identical {m['texts_identical']}")
    sys.exit(1 if failed or code != 0 else 0)


if __name__ == "__main__":
    if len(sys.argv) != 6:
        sys.exit(__doc__)
    main(*sys.argv[1:])
