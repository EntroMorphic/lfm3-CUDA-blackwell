#!/usr/bin/env python3
"""validation/audio-frontend-v1 rule 0 (second part) and rule 3, as amendment 3 sets them: how long an audio-in
request stalls the other rows' speech. Black box on liquid-lfm3-server (Thor, lfm3-profile-v1, a quiet machine), started
with `--speech-rows ROWS`: ROWS - 1 rows stream interleaved replies; 3 s after they start, one ASR request brings a
clip (text mode, 32 tokens). Per streaming row, the longest gap between consecutive audio events from the clip request's
start to 4 s after it (the gap spanning the start included), against the same window of a run without the clip. For
each clip, runs alternate (without, with) REPEATS times on one server; the stall is the difference of the medians of
each run's worst row. Clips: CLIPS_JSON ([{"name", "wav"}], rule 0's five; paths as this process sees them).
Every request retries a 429 frontend_busy (up to 3,000 tries, 10 ms apart, as tools/check_lfm3_speech_batch_server.py),
timed from its first attempt: a streaming row's first attempt counts as its first event, so a row the front end holds
back shows the wait as a gap (quiet window 8: a row's 429 ended the run, as a failure, before any measurement).
Usage: measure_lfm3_audio_stall.py SERVER MODEL_DIR CLIPS_JSON ROWS OUT_JSON [REPEATS]
"""
import base64
import json
import os
import secrets
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_lfm3_speech_stream import Server  # noqa: E402

QUESTION = ("Tell me, at some length and in plain words, how a river shapes the valley it runs through over thousands "
            "of years, from the first trickle to the wide plain near the sea.")
START, WINDOW = 3.0, 4.0


class RowsServer(Server):
    """check_lfm3_speech_stream's client, the server started with extra options (here --speech-rows)."""

    def __init__(self, binary, model, scratch, args):
        self.token = secrets.token_urlsafe(32)
        token_path = Path(scratch) / "stall.token"
        fd = os.open(token_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(self.token + "\n")
        self.proc = subprocess.Popen([binary, "--experimental", model, "--token-file", str(token_path), "--port", "0",
                                      *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        line = self.proc.stdout.readline()
        if not line:
            raise SystemExit(f"server did not start: {self.proc.stderr.read()[-500:]}")
        self.port = json.loads(line)["port"]
        token_path.unlink()


TRIES, PAUSE = 3000, 0.01  # a 429 (the front end prepares one request at a time) is retried, as a client does


def retried(call):
    """call() until it is not a 429 (as tools/check_lfm3_speech_batch_server.py's retried): (result, first try's time,
    tries)."""
    first = time.monotonic()
    for tries in range(1, TRIES + 1):
        r = call()
        if r[0] != 429:
            return r, first, tries
        time.sleep(PAUSE)
    return r, first, TRIES


def one_run(s, rows, clip_body):
    body = dict(messages=[{"role": "system", "content": "Respond with interleaved text and audio."},
                          {"role": "user", "content": QUESTION}], mode="interleaved", max_new_tokens=600)
    results = [None] * (rows - 1)

    def stream(i):
        results[i] = retried(lambda: s.stream(dict(body, seed=i + 1)))
    threads = [threading.Thread(target=stream, args=(i,)) for i in range(rows - 1)]
    for t in threads:
        t.start()
    time.sleep(START)
    t_clip = time.monotonic()
    clip_result, _, clip_tries = retried(lambda: s.post(clip_body)) if clip_body else (None, None, 0)
    clip_seconds = time.monotonic() - t_clip  # from the first attempt: time spent refused counts
    for t in threads:
        t.join()
    gaps, row_tries, rows_seen = [], [], []
    for (status, events, times, t0), first, tries in results:
        if status != 200:
            raise SystemExit(f"a streaming row failed after {tries} tries: {status} {events[:1]}")
        row_tries.append(tries)
        # the row's first attempt counts as its first event: a row held back by the front end (retrying a 429) shows
        # the wait as a gap, from its first attempt to its first audio, instead of leaving the measurement
        audio = [first] + [t0 + tm for e, tm in zip(events, times) if e["event"] == "audio"]
        inside = [a for a in audio if t_clip <= a <= t_clip + WINDOW]
        before = [a for a in audio if a < t_clip][-1:]
        seq = before + inside
        gaps.append(max((b - a for a, b in zip(seq, seq[1:])), default=None))
        # diagnostics only (seconds from the clip request's start), for a run that cannot be measured
        rel = lambda x: None if x is None else round(x - t_clip, 3)  # noqa: E731
        rows_seen.append(dict(tries=tries, first_attempt=rel(first), stream_start=rel(t0), audio_events=len(audio) - 1,
                              first_audio=rel(audio[1] if len(audio) > 1 else None), last_audio=rel(audio[-1] if len(audio) > 1 else None),
                              stream_end=rel(t0 + times[-1] if times else None), last_event=events[-1]["event"] if events else None))
    if clip_body and clip_result[0] != 200:
        raise SystemExit(f"the clip request failed: {clip_result}")
    real = [g for g in gaps if g is not None]
    if not real:
        raise SystemExit("no row streamed audio across the window ("
                         + (f"with the clip: {clip_seconds:.3f} s, {clip_tries} tries" if clip_body else "without the clip")
                         + "); rows, seconds from the clip's start: " + json.dumps(rows_seen))
    return dict(worst_gap=max(real), row_gaps=gaps, clip_seconds=clip_seconds if clip_body else None,
                clip_tries=clip_tries, row_tries=row_tries)


def main(server_bin, model, clips_json, rows, out_json, repeats=3):
    rows, repeats = int(rows), int(repeats)
    clips = json.loads(Path(clips_json).read_text())
    out = dict(rows=rows, repeats=repeats, window=WINDOW, clips={})
    with tempfile.TemporaryDirectory() as scratch:
        s = RowsServer(server_bin, model, scratch, ["--speech-rows", str(rows), "--queue-size", "40", "--handlers", "48"])
        try:
            one_run(s, rows, None)  # warm
            for c in clips:
                clip = base64.b64encode(Path(c["wav"]).read_bytes()).decode()
                clip_body = dict(messages=[{"role": "system", "content": "Perform ASR in English."},
                                           {"role": "user", "content": [{"type": "audio", "audio": clip}]}],
                                 mode="text", max_new_tokens=32)
                runs = []
                for r in range(repeats):
                    for kind in ("without", "with"):
                        res = one_run(s, rows, clip_body if kind == "with" else None)
                        res["kind"] = kind
                        runs.append(res)
                med = {k: statistics.median(x["worst_gap"] for x in runs if x["kind"] == k) for k in ("without", "with")}
                out["clips"][c["name"]] = dict(runs=runs, median_worst_gap=med, stall_seconds=med["with"] - med["without"])
                print(f"{rows} rows, {c['name']}: worst gap without {med['without']:.4f} s, with {med['with']:.4f} s, "
                      f"stall {med['with'] - med['without']:.4f} s", flush=True)
        finally:
            code, err = s.stop()
            out["server_exit"], out["server_stderr"] = code, err[-2000:]
    Path(out_json).write_text(json.dumps(out, indent=1) + "\n")
    sys.exit(0 if out["server_exit"] == 0 else 1)


if __name__ == "__main__":
    if len(sys.argv) not in (6, 7):
        sys.exit(__doc__)
    main(*sys.argv[1:])
