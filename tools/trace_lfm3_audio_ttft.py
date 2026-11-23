#!/usr/bin/env python3
"""validation/ttft-rest-v2 stage 0: audio time to first token, split point by point.

Usage: trace_lfm3_audio_ttft.py SERVER MODEL CLIPS_JSON OUT_JSON PASSES

The request is measure_lfm3_audio_ttft.py's (one streamed /speech request per clip, the clips in order, 1 untimed pass
and then PASSES timed passes). The client takes time.monotonic_ns() (CLOCK_MONOTONIC, the trace's clock) before and
after sendall and at the first recv that holds a text event. With a trace build of the server (-DLFM3_TTFT_TRACE),
the server's marks go to a fresh file named by LFM3_TTFT_TRACE, one request at a time, and each request's intervals
are reported; with any other server only the client's.
"""
import base64
import json
import os
import socket
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_lfm3_speech_stream import Server  # noqa: E402

# (name, from-mark, to-mark): consecutive, so they sum to t_first - t_send
INTERVALS = [
    ("send_to_head", "t_send", "h_head"),
    ("body", "h_head", "h_body"),
    ("strict_json", "h_body", "h_parsed"),
    ("validate", "h_parsed", "s_validated"),
    ("encode_speech", "s_validated", "s_encoded"),
    ("log_mel", "s_encoded", "s_features"),
    ("enqueue", "s_features", "s_enqueued"),
    ("queue_wait", "s_enqueued", "o_popped"),
    ("popped_to_t0", "o_popped", "c_t0"),
    ("encoder", "c_t0", "c_encoded"),
    ("lm_forward_call", "c_encoded", "c_forward"),
    ("lm_forward_sync", "c_forward", "c_t1"),
    ("admit_return", "c_t1", "o_admitted"),
    ("admitted_to_text", "o_admitted", "o_text"),
    ("text_to_sent", "o_text", "h_sent"),
    ("sent_to_client", "h_sent", "t_first"),
]


def one(server, wav_path):
    t_file = time.monotonic_ns()
    clip = base64.b64encode(Path(wav_path).read_bytes()).decode()
    payload = json.dumps(dict(messages=[{"role": "system", "content": "Perform ASR in English."},
                                        {"role": "user", "content": [{"type": "audio", "audio": clip}]}],
                              mode="text", max_new_tokens=32, seed=0, stream=True)).encode()
    head = (f"POST /speech HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nConnection: close\r\n"
            f"Content-Length: {len(payload)}\r\nAuthorization: Bearer {server.token}\r\n\r\n").encode()
    s = socket.create_connection(("127.0.0.1", server.port), timeout=600)
    try:
        t_send = time.monotonic_ns()
        s.sendall(head + payload)
        t_sent = time.monotonic_ns()
        data, first = b"", None
        while True:
            block = s.recv(1 << 16)
            if not block:
                break
            data += block
            if first is None and b'"event":"text"' in data:
                first = time.monotonic_ns()
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
    return dict(ok=ok, status=status, t_file=t_file, t_send=t_send, t_sent=t_sent, t_first=first,
                prefill_seconds=done.get("prefill_seconds") if done else None, text=text)


def read_marks(path):
    marks = []
    if path and Path(path).exists():
        for line in Path(path).read_text().splitlines():
            name, ns = line.split()
            marks.append((name, int(ns)))
    return marks


def split(run, marks):
    """The request's marks: those between its t_send and the next request's (the runs are one at a time)."""
    m = [(n, t) for n, t in marks if run["t_send"] <= t <= run["t_end"]]
    first = {}
    for n, t in m:
        first.setdefault(n, t)
    first.update(t_send=run["t_send"], t_first=run["t_first"])
    # the first text event, the first send after it, and the steps before it
    if "o_text" in first:
        sent = [t for n, t in m if n == "h_sent" and t >= first["o_text"]]
        if sent:
            first["h_sent"] = sent[0]
    out = {}
    for name, a, b in INTERVALS:
        if a in first and b in first:
            out[name] = (first[b] - first[a]) / 1e9
    if "o_admitted" in first and "o_text" in first:
        out["steps_before_text"] = sum(1 for n, t in m if n == "c_step" and first["o_admitted"] <= t <= first["o_text"])
    out["send_to_sent"] = (run["t_sent"] - run["t_send"]) / 1e9
    out["total"] = (run["t_first"] - run["t_send"]) / 1e9
    return out


def main(server_bin, model, clips_json, out_json, passes):
    clips = json.loads(Path(clips_json).read_text())
    runs, failed = [], False
    with tempfile.TemporaryDirectory() as scratch:
        trace = os.environ.get("LFM3_TTFT_TRACE")
        if trace:
            Path(trace).unlink(missing_ok=True)
        s = Server(server_bin, model, scratch)
        try:
            for p in range(int(passes) + 1):
                for c in clips:
                    r = one(s, c["wav"])
                    r["t_end"] = time.monotonic_ns()
                    r.update(clip=c["name"], timed=p > 0, pass_=p)
                    runs.append(r)
                    failed |= not r["ok"]
        finally:
            code, err = s.stop()
        marks = read_marks(trace)
    for r in runs:
        if r["ok"]:
            r["split"] = split(r, marks)
    med = {}
    for c in clips:
        t = [r for r in runs if r["clip"] == c["name"] and r["timed"] and r["ok"]]
        if not t:
            continue
        keys = [k for k in t[0]["split"] if all(k in r["split"] for r in t)]
        med[c["name"]] = {k: statistics.median(r["split"][k] for r in t) for k in keys}
        med[c["name"]]["prefill_seconds"] = statistics.median(r["prefill_seconds"] for r in t)
        med[c["name"]]["texts_identical"] = len({r["text"] for r in t}) == 1
        med[c["name"]]["text"] = t[0]["text"]
        parts = [statistics.median(r["split"][n] for r in t) for n, _, _ in INTERVALS if n in keys]
        med[c["name"]]["sum_of_interval_medians"] = sum(parts)
    out = dict(runs=runs, medians=med, marks=len(marks), server_exit=code, server_stderr=err[-2000:])
    Path(out_json).write_text(json.dumps(out, indent=1) + "\n")
    for n, m in med.items():
        print(n, json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in m.items() if k != "text"}))
    sys.exit(1 if failed or code != 0 else 0)


if __name__ == "__main__":
    if len(sys.argv) != 6:
        sys.exit(__doc__)
    main(*sys.argv[1:])
