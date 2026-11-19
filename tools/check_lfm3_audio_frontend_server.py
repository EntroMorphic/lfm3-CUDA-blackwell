#!/usr/bin/env python3
"""validation/audio-frontend-v1 rule 4: N speech-recognition requests sent to liquid-lfm3-server at once (Thor,
lfm3-profile-v1; black box over HTTP, the server started like a deployment). The clips are N LibriSpeech test-clean
utterances spread over the corpus's lengths (sorted by length, evenly spaced, the longest included); each request is
audio-in-v1's (system "Perform ASR in English.", mode text, greedy, 512 tokens). Pass: every request 200 (none
refused: audio requests wait for the front end), and every reply's text byte-identical to lfm3-speech's for the same
conversation: lfm3-speech SPEECH runs the same messages (each clip as its WAV path) in one --batch, its outputs in
WORK_DIR/CASE.json (the reply "text" as the server decodes it); then a text /generate control before and after gives
the same ids, and the server exits 0. Both engines inherit the environment (LFM3_AUDIO_FEATURES).
(A first version compared with rule 4's corpus outputs, which were made from prompt_ids and carry no "text": every
comparison failed against None. validation/audio-frontend-v1 records it.)
Usage: check_lfm3_audio_frontend_server.py SERVER SPEECH MODEL CASES.json WORK_DIR N OUT_JSON [SERVER_ARG...]
"""
import base64
import hashlib
import json
import os
import secrets
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path


def conversation(clip):
    return [{"role": "system", "content": "Perform ASR in English."},
            {"role": "user", "content": [{"type": "audio", "audio": clip}]}]


def main(server, speech, model, cases_json, work, n, out_json, *server_args):
    n = int(n)
    cases = sorted(json.load(open(cases_json))["cases"], key=lambda c: (c["seconds"], c["case"]))
    pick = sorted({round(i * (len(cases) - 1) / (n - 1)) for i in range(n)}) if n > 1 else [len(cases) - 1]
    chosen = [cases[i] for i in pick]
    assert len(chosen) == n
    # the references: lfm3-speech on the same conversations, the clips as paths
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    with open(work / "list.txt", "w") as f:
        for c in chosen:
            req = work / f"{c['case']}.req.json"
            req.write_text(json.dumps(dict(messages=conversation(c["audio"][0]), mode="text", max_new_tokens=512, seed=0)))
            f.write(f"{req} {work / c['case']}\n")
    r = subprocess.run([speech, model, "--batch", str(work / "list.txt")], capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"lfm3-speech failed: {r.stderr[-500:]}")
    token = secrets.token_urlsafe(32)
    token_path = Path(out_json).with_suffix(".token")
    fd = os.open(token_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(token + "\n")
    proc = subprocess.Popen([server, "--experimental", model, "--token-file", str(token_path), "--port", "0",
                             *server_args], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    port = json.loads(proc.stdout.readline())["port"]
    token_path.unlink()

    def post(path, body, timeout=900):
        req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(), method="POST",
                                     headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    sid = post("/sessions", {"capacity": 1024})[1]["session_id"]
    control = dict(session_id=sid, max_new_tokens=128,
                   messages=[{"role": "user", "content": "Explain in one paragraph how a lighthouse warns ships at night."}])
    s0, before = post("/generate", control)
    post(f"/sessions/{sid}/reset", {})
    bodies = []
    for c in chosen:
        wav = c["audio"][0]
        clip = base64.b64encode(Path(wav).read_bytes()).decode()
        bodies.append(dict(messages=conversation(clip), mode="text", max_new_tokens=512, seed=0))
    results = [None] * n
    go = threading.Event()

    def run(i):
        go.wait()
        t0 = time.time()
        results[i] = (*post("/speech", bodies[i]), time.time() - t0)
    threads = [threading.Thread(target=run, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    t0 = time.time()
    go.set()
    for t in threads:
        t.join()
    wall = time.time() - t0
    report = dict(server_sha256=hashlib.sha256(Path(server).read_bytes()).hexdigest(), n=n, server_args=server_args,
                  wall_seconds=round(wall, 2), cases={})
    ok = True
    for c, (status, reply, seconds) in zip(chosen, results):
        ref = json.load(open(work / f"{c['case']}.json"))
        same = status == 200 and ref.get("text") is not None and reply.get("text") == ref.get("text")
        ok &= same
        report["cases"][c["case"]] = dict(seconds=c["seconds"], status=status, text_identical=same,
                                          error=None if status == 200 else reply, wall_seconds=round(seconds, 2),
                                          text=reply.get("text"), reference_text=None if same else ref.get("text"))
    s1, after = post("/generate", control)
    report["text_control_identical"] = s0 == s1 == 200 and before["generated_ids"] == after["generated_ids"]
    ok &= report["text_control_identical"]
    proc.terminate()
    report["exit_code"] = proc.wait(timeout=120)
    report["stderr"] = proc.stderr.read()[-2000:]
    ok &= report["exit_code"] == 0
    report["served"] = sum(1 for r in report["cases"].values() if r["status"] == 200)
    report["identical"] = sum(1 for r in report["cases"].values() if r["text_identical"])
    report["pass"] = bool(ok)
    Path(out_json).write_text(json.dumps(report, indent=1) + "\n")
    print(f"{n} at once: {report['served']} served, {report['identical']} identical to lfm3-speech, "
          f"text control {report['text_control_identical']}, exit {report['exit_code']}, {wall:.1f} s",
          "| PASS" if ok else "| FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    if len(sys.argv) < 7:
        sys.exit(__doc__)
    main(*sys.argv[1:])
