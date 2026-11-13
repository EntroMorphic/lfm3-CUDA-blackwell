#!/usr/bin/env python3
"""Long mixed audio run of liquid-lfm3-server (validation/audio-assurance-v1 rule 4; Thor, lfm3-profile-v1). Black box.

1. References: the 42 cases of check_lfm3_speech_stream.py (TTS both voices, interleaved, split long-form, the budget
   cut, normalized numbers, spoken questions, LibriSpeech transcriptions), each alone, unstreamed and streamed, and
   three text /generate bodies, on a fresh server.
2. The soak: a second fresh server (8 speech rows, the default) under CLIENTS client threads for SECONDS. Each request
   is one of: a case unstreamed, a case streamed, a case streamed and dropped after its first audio event (about 1 in
   10 streamed; not compared), or a text /generate (about 1 in 10). 429s are retried and counted. Every completed
   reply is compared with its reference (timings aside; streamed: every event).
3. Every 30 s: VmRSS of the server (the worker runs inside it), and the host's memory in use (MemTotal - MemAvailable; on Thor the
   GPU allocates from system memory and reports no per-process figure).
4. After the soak: the 42 cases again, all at once, unstreamed and streamed (a crowd), identical to the references;
   then SIGTERM, exit 0.
Pass (rule 4): every completed reply identical, no 5xx and no transport error, both processes alive throughout, each
memory figure at the end at most its maximum over the first 15 minutes plus 64 MiB, the crowd identical, exit 0.
Writes OUT_JSON (and OUT_JSON's .samples.jsonl) and exits 1 on any failure.
Usage: check_lfm3_audio_soak.py SERVER MODEL_DIR BUILD_DIR LIBRI_WAV_DIR OUT_JSON [SECONDS [CLIENTS]]
"""
import concurrent.futures
import hashlib
import importlib.util
import json
import random
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("check", HERE / "check_lfm3_speech_stream.py")
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)
spec = importlib.util.spec_from_file_location("batch", HERE / "check_lfm3_speech_batch_server.py")
batch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(batch)
strip, start = batch.strip, batch.start
TEXTS = [dict(max_new_tokens=96, messages=[{"role": "user", "content": c}]) for c in (
    "Explain in one paragraph how a lighthouse warns ships at night.",
    "List three ways to keep bread fresh longer.",
    "What is the difference between weather and climate? Answer briefly.")]
MIB = 1 << 20


def rss(pid):
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except FileNotFoundError:
        return None


def children(pid):
    try:
        return [int(c) for c in Path(f"/proc/{pid}/task/{pid}/children").read_text().split()]
    except FileNotFoundError:
        return []


def worker_of(server_pid):
    """The worker process (the server's child; it starts after the server reports its port), or None."""
    kids = children(server_pid)
    return kids[0] if kids else None


def host_used():
    m = {l.split(":")[0]: int(l.split()[1]) * 1024 for l in Path("/proc/meminfo").read_text().splitlines()}
    return m["MemTotal"] - m["MemAvailable"]


def generate(server, body):
    """(status, reply) of /generate in a fresh session; 429 retried by the caller."""
    def post(path, b):
        req = urllib.request.Request(f"http://127.0.0.1:{server.port}{path}", data=json.dumps(b).encode(), method="POST",
                                     headers={"Authorization": "Bearer " + server.token, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())
    status, created = post("/sessions", {"capacity": 1024})
    if status != 200:
        return status, created
    try:
        return post("/generate", dict(body, session_id=created["session_id"]))
    finally:
        post(f"/sessions/{created['session_id']}/close", {})


def text_key(reply):
    return reply.get("generated_ids"), reply.get("text")


def main(binary, model, build, libri, out_json, seconds=7200, clients=12):
    seconds, clients = int(seconds), int(clients)
    out = Path(out_json)
    samples_path = out.with_suffix(".samples.jsonl")
    report = dict(server_sha256=hashlib.sha256(Path(binary).read_bytes()).hexdigest(), seconds=seconds, clients=clients)
    cases = check.cases(build, libri)
    ok = True

    # 1. references, alone, on a fresh server
    ref = start(binary, model, out.parent)
    alone, alone_stream, text_ref = {}, {}, []
    try:
        ref.post(cases[0][2])  # warm
        for name, _, body in cases:
            status, reply = ref.post(body)
            assert status == 200, (name, status, reply)
            alone[name] = strip(reply)
            status, events, _, _ = ref.stream(body)
            assert status == 200, (name, status)
            alone_stream[name] = events
        for body in TEXTS:
            status, reply = generate(ref, body)
            assert status == 200, (status, reply)
            text_ref.append(text_key(reply))
    finally:
        code, _ = ref.stop()
        ok &= code == 0
        report["reference_server_exit"] = code
    print("references done", flush=True)

    def same_stream(name, status, events):
        want = alone_stream[name]
        return status == 200 and bool(events) and events[-1].get("event") == "done" and events[:-1] == want[:-1] \
            and strip(events[-1]) == strip(want[-1])

    # 2. the soak
    server = start(binary, model, out.parent, "--handlers", "96", "--queue-size", "64")
    server.post(cases[0][2])  # warm: the worker is up after a reply
    worker = worker_of(server.proc.pid)
    report["worker_found"] = worker is not None
    counts = dict(unstreamed=0, streamed=0, dropped=0, text=0, retries_429=0)
    differ, errors, statuses = [], [], {}
    lock = threading.Lock()
    stop = threading.Event()
    t_start = time.monotonic()
    samples = []

    def sampler():
        with samples_path.open("w") as f:
            while True:
                s = dict(t=round(time.monotonic() - t_start, 1), server_rss=rss(server.proc.pid),
                         worker_rss=rss(worker) if worker else None, host_used=host_used(),
                         server_alive=server.proc.poll() is None)
                samples.append(s)
                f.write(json.dumps(s) + "\n")
                f.flush()
                if stop.wait(30):
                    break

    def retry(call):
        for _ in range(30000):
            r = call()
            if r[0] != 429:
                return r
            with lock:
                counts["retries_429"] += 1
            time.sleep(0.02)
        return r

    def client(seed):
        rng = random.Random(seed)
        while not stop.is_set():
            roll = rng.random()
            name, _, body = rng.choice(cases)
            try:
                if roll < 0.1:
                    i = rng.randrange(len(TEXTS))
                    status, reply = retry(lambda: generate(server, TEXTS[i]))
                    kind, good = "text", status == 200 and text_key(reply) == text_ref[i]
                    label = f"text-{i}"
                elif roll < 0.5:
                    status, reply = retry(lambda: server.post(body))
                    kind, good, label = "unstreamed", status == 200 and strip(reply) == alone[name], name
                elif roll < 0.55:
                    status, events, _, _ = retry(lambda: server.stream(body, close_after_audio=True))
                    kind, good, label = "dropped", status == 200, name
                else:
                    status, events, _, _ = retry(lambda: server.stream(body))
                    kind, good, label = "streamed", same_stream(name, status, events), name
                with lock:
                    counts[kind] += 1
                    statuses[str(status)] = statuses.get(str(status), 0) + 1
                    if not good:
                        differ.append(dict(kind=kind, case=label, status=status, t=round(time.monotonic() - t_start, 1)))
            except Exception as e:  # noqa: BLE001  (a transport error is a failure, recorded)
                with lock:
                    errors.append(dict(case=name, error=repr(e)[:300], t=round(time.monotonic() - t_start, 1)))

    sampler_thread = threading.Thread(target=sampler, daemon=True)
    sampler_thread.start()
    with concurrent.futures.ThreadPoolExecutor(clients) as pool:
        futures = [pool.submit(client, 1000 + i) for i in range(clients)]
        while time.monotonic() - t_start < seconds and server.proc.poll() is None:
            time.sleep(5)
            done = sum(counts[k] for k in ("unstreamed", "streamed", "dropped", "text"))
            if int(time.monotonic() - t_start) % 600 < 5:
                print(f"t={int(time.monotonic() - t_start)} s: {done} requests, {len(differ)} differ, {len(errors)} errors",
                      flush=True)
        stop.set()
        for f in futures:
            f.result()
    sampler_thread.join()
    report.update(counts=counts, statuses=statuses, differ=differ[:200], differ_count=len(differ), errors=errors[:50],
                  error_count=len(errors), soak_seconds=round(time.monotonic() - t_start, 1))
    bad_status = {k: v for k, v in statuses.items() if k != "200"}

    # memory: the end against the first 15 minutes' maximum plus 64 MiB
    memory = {}
    early = [s for s in samples if s["t"] <= 900]
    for key in ("server_rss", "host_used"):  # the worker runs inside the server process (one RSS)
        vals = [s[key] for s in early if s[key] is not None]
        end = samples[-1][key] if samples else None
        bound = max(vals) + 64 * MIB if vals else None
        memory[key] = dict(early_max_mib=round(max(vals) / MIB, 1) if vals else None,
                           end_mib=round(end / MIB, 1) if end else None,
                           overall_max_mib=round(max(s[key] for s in samples if s[key]) / MIB, 1) if samples else None,
                           pass_=end is not None and bound is not None and end <= bound)
    report["memory"] = memory
    alive = all(s["server_alive"] for s in samples)
    report["alive_throughout"] = alive

    # 4. the crowd after the soak
    crowd = {}
    for streamed in (False, True):
        with concurrent.futures.ThreadPoolExecutor(len(cases)) as pool:
            if streamed:
                got = list(pool.map(lambda c: batch.stream(server, c[2]), cases))
                bad = [c[0] for c, r in zip(cases, got) if not same_stream(c[0], r[0], r[1])]
            else:
                got = list(pool.map(lambda c: batch.post(server, c[2]), cases))
                bad = [c[0] for c, r in zip(cases, got) if not (r[0] == 200 and strip(r[1]) == alone[c[0]])]
        crowd["streamed" if streamed else "unstreamed"] = dict(identical=len(cases) - len(bad), differ=bad)
    report["crowd_after"] = crowd
    code, stderr = server.stop()
    report.update(server_exit=code, server_stderr=stderr[-2000:])
    report["pass_"] = pass_ = bool(
        ok and not differ and not errors and not bad_status and alive
        and all(m["pass_"] for m in memory.values()) and all(not c["differ"] for c in crowd.values()) and code == 0)
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps(dict(pass_=pass_, counts=counts, statuses=statuses, differ=len(differ), errors=len(errors),
                          memory=memory, crowd={k: v["identical"] for k, v in crowd.items()}, exit=code)))
    print("AUDIO SOAK PASS" if pass_ else "AUDIO SOAK FAIL")
    return 0 if pass_ else 1


if __name__ == "__main__":
    if len(sys.argv) not in (6, 7, 8):
        sys.exit(__doc__)
    sys.exit(main(*sys.argv[1:]))
