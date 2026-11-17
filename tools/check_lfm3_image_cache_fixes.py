#!/usr/bin/env python3
"""validation/image-cache-v1, red-team fixes C3-C5: the Thor checks (lfm3-profile-v1; black box on liquid-lfm3-server
over HTTP, except what /proc shows about the server process). Figures and readings are set in the README's "C3-C5 Thor
procedure (set before the run)"; each subcommand writes OUT_JSON with "passed" and prints one verdict line.

  c3 SERVER MODEL_DIR ROUND_DIR PRIME_DIR FRESH_PNG OUT_JSON
     Where a re-sent image is decoded once the engine has evicted its rows but the front end still knows its token
     count. The server has no log of it, so this reads each thread's CPU time (/proc/PID/task/TID/stat) across each
     request: a 4K decode is ~0.3-0.4 s of one thread's CPU. The engine-owner thread is the one that spends the most
     on a text-only request. Order: text and "small" (warm-up); text (finds the engine thread); cherry (the seed, a
     miss); cherry (a hit: the control in which no thread decodes); PRIME_DIR/*.png, each once (1 new token); cherry
     (the probe); cherry (a hit again); FRESH_PNG (a never-seen image: the control in which a handler decodes).
  c4 SERVER MODEL_DIR ROUND_DIR PRIME_DIR OUT_JSON [LABEL]
     Memory after 1 and after 16 distinct 4K requests (PRIME_DIR's first 16; the first alone, the other 15 at once,
     so that they spread over the speech rows), the caches' own growth subtracted (computed from the images: the
     server reports none).
  c5 SERVER FAULT_SERVER MODEL_DIR ROUND_DIR OUT_JSON
     The fault-injection build (every keep of an image's rows fails) against SERVER with --image-cache-mb 0 and
     SERVER with the cache on (the hit-timing calibration).
  memcheck SANITIZER|- SERVER MODEL_DIR ROUND_DIR OUT_DIR
     A sequence of small stage-0 images at --image-cache-mb 1 (hits, misses, evictions and images over the budget;
     the expected cache states are simulated from the images' sizes and recorded), under compute-sanitizer memcheck
     (SANITIZER: its path) or plain ("-"). Writes OUT_DIR/{plain,memcheck}.json; the memcheck run is compared with
     plain.json when present. No leak check: the server ends with std::_Exit, which frees nothing by design, so every
     live allocation would be reported (leaks are lfm3-speech's memcheck's part, see the README).
  speech-list ROUND_DIR OUT_DIR NAME...
     lfm3-speech requests for its memcheck: ROUND_DIR/req/NAME.json at 16 new tokens without record_image, written
     to OUT_DIR/req, with OUT_DIR/list-plain.txt and list-memcheck.txt (outputs in OUT_DIR/plain, OUT_DIR/memcheck).
ROUND_DIR: vision_e2e_round.sh's output (build/vimg-e2e-a4: req/*.json and lfm3-speech's out/*.json).
"""
import base64
import ctypes
import gc
import json
import os
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_lfm3_image_server import as_client, streamed_text  # noqa: E402
from check_lfm3_speech_stream import Server  # noqa: E402
from make_lfm3_c3_images import chunks, costs  # noqa: E402

MIB = 1 << 20
DEFAULT_BUDGET = 512 * MIB  # --image-cache-mb's default, each side's
TEXT = dict(messages=[{"role": "user", "content": "Count from one to twenty in words."}], mode="text", max_new_tokens=64)
DECODED, NOT_DECODED, ENGINE_MARGIN = 0.20, 0.10, 0.20  # seconds of one thread's CPU (README: C3-C5 Thor procedure)
MEMORY_BAR = 64 * MIB
SETTLE, LATER = 3.0, 10.0  # seconds of nothing before a memory reading, and before the later one (c4)
TICK = os.sysconf("SC_CLK_TCK")


# ---------------------------------------------------------------------------------------------------- requests
def case(round_dir, name):
    """(the stage-0 request as a client sends it, lfm3-speech's reply text for it)."""
    rd = Path(round_dir)
    return (as_client(json.loads((rd / "req" / f"{name}.json").read_text())),
            json.loads((rd / "out" / f"{name}.json").read_text())["text"])


def with_image(body, png, max_new_tokens=None):
    """BODY with every image part's data replaced by PNG's bytes (base64)."""
    out = json.loads(json.dumps(body))
    data = base64.b64encode(Path(png).read_bytes()).decode()
    for m in out["messages"]:
        if isinstance(m["content"], list):
            for part in m["content"]:
                if part.get("type") == "image":
                    part["image"] = data
    if max_new_tokens is not None:
        out["max_new_tokens"] = max_new_tokens
    return out


def image_keys(body):
    return [p["image"] for m in body["messages"] if isinstance(m["content"], list) for p in m["content"]
            if p.get("type") == "image"]


def key_costs(key):
    """The cache costs of an image part's base64 text (make_lfm3_c3_images.costs on its decoded bytes)."""
    png = base64.b64decode(key)
    w, h = int.from_bytes(chunks(png)[0][1][0:4], "big"), int.from_bytes(chunks(png)[0][1][4:8], "big")
    c = costs(png, w, h)
    assert c["key_bytes"] == len(key), "the key is not the canonical base64 of its bytes"
    return c


def manifest_costs(png):
    m = json.loads((Path(png).parent / "manifest.json").read_text())
    return m["images"][Path(png).name]


def post_retry(s, body, timeout=600):
    """A client's post: a 429 (queue_full, frontend_busy) is retried."""
    for attempt in range(400):
        status, reply = s.post(body, timeout=timeout)
        if status != 429:
            return status, reply
        time.sleep(0.05 * (1 + attempt % 10))
    return status, reply


def health(s):
    req = urllib.request.Request(f"http://127.0.0.1:{s.port}/health", headers={"Authorization": "Bearer " + s.token})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code


# ---------------------------------------------------------------------------------------------------- /proc
def thread_cpu(pid):
    """{tid: user + system CPU seconds} of every thread of PID."""
    out = {}
    for task in Path(f"/proc/{pid}/task").iterdir():
        try:
            stat = (task / "stat").read_text()
        except (FileNotFoundError, ProcessLookupError):
            continue  # a thread that ended
        fields = stat[stat.rindex(")") + 2:].split()
        out[int(task.name)] = (int(fields[11]) + int(fields[12])) / TICK  # utime, stime (stat fields 14, 15)
    return out


def cpu_delta(before, after):
    return {tid: after[tid] - before.get(tid, 0.0) for tid in after}


def split_cpu(delta, engine_tid):
    """(the engine thread's CPU, the largest of any other thread's, that thread)."""
    others = {t: v for t, v in delta.items() if t != engine_tid}
    top = max(others, key=others.get) if others else None
    return delta.get(engine_tid, 0.0), (others[top] if top is not None else 0.0), top


def timed(s, body, engine_tid=None, timeout=600):
    pid = s.proc.pid
    before = thread_cpu(pid)
    t0 = time.monotonic()
    status, reply = s.post(body, timeout=timeout)
    seconds = time.monotonic() - t0
    delta = cpu_delta(before, thread_cpu(pid))
    row = dict(status=status, seconds=seconds, text=reply.get("text") if status == 200 else None,
               prefill_seconds=reply.get("prefill_seconds") if status == 200 else None,
               error=None if status == 200 else reply, top_threads=sorted(delta.items(), key=lambda kv: -kv[1])[:4])
    if engine_tid is not None:
        row["engine_cpu"], row["other_cpu"], row["other_tid"] = split_cpu(delta, engine_tid)
    return row, delta


def engine_thread(s):
    """The engine-owner thread: the one with the most CPU across a text-only request (64 tokens of decode steps; the
    handler only parses and tokenizes a short text)."""
    row, delta = timed(s, TEXT)
    assert row["status"] == 200, row
    tid = max(delta, key=delta.get)
    return tid, dict(tid=tid, text_request=row)


def proc_status(pid):
    f = {}
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        k, _, v = line.partition(":")
        if k in ("VmRSS", "RssAnon", "RssFile", "RssShmem", "Threads"):
            f[k] = int(v.split()[0]) * (1024 if k != "Threads" else 1)
    return f


def meminfo():
    return {l.split(":")[0]: int(l.split()[1]) * 1024 for l in Path("/proc/meminfo").read_text().splitlines()}


def trim_self():
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


def memory(pid):
    """One sample: the server's RSS fields, Thor's in-use memory, this client's anonymous RSS, and the stand-in for
    the server's GPU memory: in use (MemTotal - MemAvailable) less the server's and this client's anonymous and shmem
    RSS. On Thor the GPU allocates from system memory and no per-process field shows it (host_memory_sampler.py)."""
    server, client, m = proc_status(pid), proc_status(os.getpid()), meminfo()
    in_use = m["MemTotal"] - m["MemAvailable"]
    other = in_use - server["RssAnon"] - server["RssShmem"] - client["RssAnon"] - client["RssShmem"]
    return dict(vmrss=server["VmRSS"], rss_anon=server["RssAnon"], rss_file=server["RssFile"],
                rss_shmem=server["RssShmem"], threads=server["Threads"], in_use=in_use,
                client_anon=client["RssAnon"] + client["RssShmem"], device_standin=other)


def settled(pid, wait=None, samples=5, every=0.5):
    """The median of SAMPLES memory samples after WAIT (default SETTLE) seconds of nothing."""
    trim_self()
    time.sleep(SETTLE if wait is None else wait)
    got = []
    for _ in range(samples):
        got.append(memory(pid))
        time.sleep(every)
    return {k: statistics.median(g[k] for g in got) for k in got[0]}


# ---------------------------------------------------------------------------------------------------- c3
def c3(server, model, round_dir, prime_dir, fresh_png, out_json):
    cherry, want = case(round_dir, "cherry")
    small, _ = case(round_dir, "small")
    primes = sorted(Path(prime_dir).glob("*.png"))
    # the state the check needs, from the images' sizes (--image-cache-mb's default on both sides): after the seed and
    # the hit, cherry is the engine's most recent entry; the primes must push it out of the engine's budget while the
    # front end keeps every key
    ck = key_costs(image_keys(cherry)[0])
    sk = key_costs(image_keys(small)[0])
    pc = [manifest_costs(p) for p in primes]
    engine_after = ck["engine_cost"] + sum(c["engine_cost"] for c in pc)
    front_total = sk["front_end_cost"] + ck["front_end_cost"] + sum(c["front_end_cost"] for c in pc)
    pre = dict(primes=len(primes), cherry=ck, engine_bytes_from_cherry_on=engine_after, front_end_bytes=front_total,
               engine_evicts_cherry=engine_after > DEFAULT_BUDGET, front_end_keeps_all=front_total <= DEFAULT_BUDGET)
    print("precondition", pre, flush=True)
    if not (pre["engine_evicts_cherry"] and pre["front_end_keeps_all"]):
        Path(out_json).write_text(json.dumps(dict(passed=False, precondition=pre), indent=1))
        print("C3 PRECONDITION NOT MET (image sizes)")
        return 1
    fresh = with_image(cherry, fresh_png)
    rows = {}
    with tempfile.TemporaryDirectory() as scratch:
        s = Server(server, model, scratch)
        try:
            for body in (TEXT, small):
                st, r = s.post(body)
                assert st == 200, (st, r)
            tid, rows["engine_thread"] = engine_thread(s)
            for name in ("seed", "hit"):
                rows[name], _ = timed(s, cherry, tid)
                print(name, {k: rows[name][k] for k in ("status", "seconds", "prefill_seconds", "engine_cpu", "other_cpu")}, flush=True)
            t0, statuses = time.monotonic(), []
            for p in primes:
                st, r = s.post(with_image(cherry, p, 1))
                statuses.append(st)
            rows["primes"] = dict(seconds=time.monotonic() - t0, statuses=statuses)
            for name, body in (("probe", cherry), ("rehit", cherry), ("miss", fresh)):
                rows[name], _ = timed(s, body, tid)
                print(name, {k: rows[name][k] for k in ("status", "seconds", "prefill_seconds", "engine_cpu", "other_cpu")}, flush=True)
        finally:
            code, err = s.stop()
    hit, miss, probe = rows["hit"], rows["miss"], rows["probe"]
    ok_status = all(rows[n]["status"] == 200 for n in ("seed", "hit", "probe", "rehit", "miss")) and \
        all(st == 200 for st in rows["primes"]["statuses"])
    texts = all(rows[n]["text"] == want for n in ("seed", "hit", "probe", "rehit"))
    controls = dict(hit_not_decoded=hit["other_cpu"] <= NOT_DECODED, miss_decoded_on_handler=miss["other_cpu"] >= DECODED,
                    probe_engine_encoded=ok_status and probe["prefill_seconds"] >= (hit["prefill_seconds"] + miss["prefill_seconds"]) / 2)
    on_handler = probe["other_cpu"] >= DECODED and probe["engine_cpu"] <= miss["engine_cpu"] + ENGINE_MARGIN
    on_engine = probe["other_cpu"] <= NOT_DECODED and probe["engine_cpu"] >= miss["engine_cpu"] + ENGINE_MARGIN
    where = "void" if not all(controls.values()) else "handler" if on_handler else "engine" if on_engine else "unclear"
    passed = ok_status and texts and where == "handler"
    Path(out_json).write_text(json.dumps(dict(passed=passed, decoded_on=where, statuses_200=ok_status,
                                              cherry_texts_equal_lfm3_speech=texts, controls=controls,
                                              precondition=pre, rows=rows, server_exit=code,
                                              server_stderr_tail=err[-2000:]), indent=1))
    print(f"probe: other thread {probe['other_cpu']:.2f} s (hit {hit['other_cpu']:.2f}, miss {miss['other_cpu']:.2f}), "
          f"engine {probe['engine_cpu']:.2f} s (miss {miss['engine_cpu']:.2f}); prefill {probe['prefill_seconds']} "
          f"(hit {hit['prefill_seconds']}, miss {miss['prefill_seconds']}); texts {'equal' if texts else 'DIFFER'}; "
          f"decoded on: {where}; C3 LOCATION {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


# ---------------------------------------------------------------------------------------------------- c4
def c4(server, model, round_dir, prime_dir, out_json, label="new"):
    cherry, _ = case(round_dir, "cherry")
    small, _ = case(round_dir, "small")
    pngs = sorted(Path(prime_dir).glob("*.png"))[:16]
    assert len(pngs) == 16, f"need 16 images in {prime_dir}"
    pc = [manifest_costs(p) for p in pngs]
    sk = key_costs(image_keys(small)[0])
    # nothing is evicted (else the caches' growth below is not what they hold)
    held_engine = sk["engine_cost"] + sum(c["engine_cost"] for c in pc)
    held_front = sk["front_end_cost"] + sum(c["front_end_cost"] for c in pc)
    assert held_engine <= DEFAULT_BUDGET and held_front <= DEFAULT_BUDGET, (held_engine, held_front)
    host_cache = sum(2 * c["key_bytes"] for c in pc[1:])  # the front end's key and the engine's key, per image
    device_cache = sum(c["row_bytes"] for c in pc[1:])     # the engine's encoder rows
    bodies = [with_image(cherry, p, 32) for p in pngs]
    out = dict(label=label, server=server, images=[p.name for p in pngs], host_cache_growth=host_cache,
               device_cache_growth=device_cache)
    with tempfile.TemporaryDirectory() as scratch:
        s = Server(server, model, scratch, extra=["--queue-size", "16"])
        try:
            pid = s.proc.pid
            for body in (TEXT, small):
                st, r = s.post(body)
                assert st == 200, (st, r)
            out["idle"] = settled(pid)
            st, r = s.post(bodies[0])
            out["first_status"] = st
            out["after_1"] = settled(pid)
            statuses = [None] * 15
            def one(i):
                statuses[i] = post_retry(s, bodies[i + 1])[0]
            threads = [threading.Thread(target=one, args=(i,)) for i in range(15)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            bodies.clear()
            out["statuses"] = [st, *statuses]
            out["after_16"] = settled(pid)
            out["after_16_later"] = settled(pid, wait=LATER)
        finally:
            out["server_exit"], err = s.stop()
            out["server_stderr_tail"] = err[-2000:]
    a, b = out["after_1"], out["after_16"]
    out["rss_growth"] = b["vmrss"] - a["vmrss"]
    out["rss_excess"] = out["rss_growth"] - host_cache
    out["device_standin_growth"] = b["device_standin"] - a["device_standin"]
    out["device_excess"] = out["device_standin_growth"] - device_cache
    ok_status = all(x == 200 for x in out["statuses"])
    out["passed"] = ok_status and out["rss_excess"] <= MEMORY_BAR and out["device_excess"] <= MEMORY_BAR
    Path(out_json).write_text(json.dumps(out, indent=1))
    mb = lambda v: f"{v / MIB:+.1f} MiB"  # noqa: E731
    print(f"{label}: RSS after 1 {a['vmrss'] / MIB:.0f} MiB, after 16 {b['vmrss'] / MIB:.0f} MiB: growth "
          f"{mb(out['rss_growth'])}, caches' keys {mb(host_cache)}, excess {mb(out['rss_excess'])}; device stand-in "
          f"growth {mb(out['device_standin_growth'])}, caches' rows {mb(device_cache)}, excess {mb(out['device_excess'])}; "
          f"statuses {sorted(set(out['statuses']), key=str)}; C4 MEMORY {'PASS' if out['passed'] else 'FAIL'} (bar +64 MiB)")
    return 0 if out["passed"] else 1


# ---------------------------------------------------------------------------------------------------- c5
def c5_server(binary, model, extra, cherry, two, small):
    rec = {}
    with tempfile.TemporaryDirectory() as scratch:
        s = Server(binary, model, scratch, extra=extra)
        try:
            for body in (TEXT, small):
                st, r = s.post(body)
                assert st == 200, (st, r)
            tid, rec["engine_thread"] = engine_thread(s)
            rec["cherry"] = [timed(s, cherry, tid)[0] for _ in range(3)]
            st, events, _, _ = s.stream(cherry)
            rec["cherry_streamed"] = dict(status=st, text=streamed_text(events)[0] if st == 200 else None)
            rec["two"] = [timed(s, two, tid)[0] for _ in range(2)]
            rec["health"] = health(s)
            st, r = s.post(TEXT)
            rec["text_after"] = dict(status=st, text=r.get("text") if st == 200 else r)
            rec["alive"] = s.proc.poll() is None
        finally:
            rec["exit"], err = s.stop()
            rec["stderr_tail"] = err[-2000:]
    return rec


def texts_of(rec):
    return dict(cherry=[r["text"] for r in rec["cherry"]] + [rec["cherry_streamed"]["text"]],
                two=[r["text"] for r in rec["two"]])


def c5(server, fault, model, round_dir, out_json):
    cherry, want_cherry = case(round_dir, "cherry")
    two, want_two = case(round_dir, "two")
    small, _ = case(round_dir, "small")
    runs = {}
    for label, binary, extra in (("off", server, ["--image-cache-mb", "0"]), ("on", server, []), ("fault", fault, [])):
        runs[label] = c5_server(binary, model, extra, cherry, two, small)
        print(label, {k: [round(r["prefill_seconds"] or -1, 3) for r in runs[label][k]] for k in ("cherry", "two")},
              "other-thread CPU", [round(r["other_cpu"], 2) for r in runs[label]["cherry"]], flush=True)
    t = {k: texts_of(v) for k, v in runs.items()}
    on, fa = runs["on"]["cherry"], runs["fault"]["cherry"]
    if not all(r["status"] == 200 for v in runs.values() for r in v["cherry"] + v["two"]):
        Path(out_json).write_text(json.dumps(dict(passed=False, runs=runs), indent=1))
        print("a request failed:", {k: [r["status"] for r in v["cherry"] + v["two"]] for k, v in runs.items()}, "C5 FAIL")
        return 1
    miss_on, hit_on = on[0]["prefill_seconds"], statistics.median(r["prefill_seconds"] for r in on[1:])
    mid = (miss_on + hit_on) / 2
    checks = dict(
        fault_equals_off=t["fault"] == t["off"],
        fault_equals_lfm3_speech=all(x == want_cherry for x in t["fault"]["cherry"]) and all(x == want_two for x in t["fault"]["two"]),
        on_equals_off=t["on"] == t["off"],
        fault_stays_up=runs["fault"]["alive"] and runs["fault"]["health"] == 200 and runs["fault"]["text_after"]["status"] == 200
        and runs["fault"]["exit"] == 0,
        calibration_on_hits=hit_on <= miss_on - 0.1 and all(r["other_cpu"] <= NOT_DECODED for r in on[1:]),
        fault_caches_nothing=all(r["prefill_seconds"] >= mid and r["other_cpu"] >= DECODED for r in fa[1:]),
    )
    passed = all(checks.values())
    Path(out_json).write_text(json.dumps(dict(passed=passed, checks=checks, hit_miss_midpoint=mid, runs=runs), indent=1))
    for k, v in checks.items():
        print(("ok    " if v else "FAIL  ") + k)
    print(f"fault re-sends: prefill {[round(r['prefill_seconds'], 3) for r in fa[1:]]} (on: miss {miss_on:.3f}, hit "
          f"{hit_on:.3f}), other-thread CPU {[round(r['other_cpu'], 2) for r in fa[1:]]}; C5 {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


# ---------------------------------------------------------------------------------------------------- memcheck
MEMCHECK_BUDGET_MB = 1
MEMCHECK_SEQUENCE = ["small", "small", "wide", "tall", "small", "tall", "china", "two", "wide"]
MEMCHECK_AT_ONCE = ["small", "tall", "wide"]
MEMCHECK_LAST = ["small"]


def simulate(sequence, keys_of, budget):
    """The engine cache's states (lfm::LruBytes as Batch::admit uses it): per request, per image, "hit", "miss" (kept),
    or "over" (a miss too large to keep: key + rows > budget), and the evictions each caused."""
    order, held, out = [], {}, []  # order: most recent first
    for name in sequence:
        events = []
        for key, c in keys_of[name]:
            if key in held:
                order.remove(key)
                order.insert(0, key)
                events.append("hit")
                continue
            if c["engine_cost"] > budget:
                events.append("over")
                continue
            held[key] = c["engine_cost"]
            order.insert(0, key)
            evicted = 0
            while sum(held.values()) > budget:
                del held[order.pop()]
                evicted += 1
            events.append("miss" + (f", evicts {evicted}" if evicted else ""))
        out.append((name, events))
    return out


class SanitizedServer(Server):
    """Server under compute-sanitizer: the sanitizer's own output to its --log-file; stopped by SIGTERM to the server
    process itself (the sanitizer's child), then waiting for the sanitizer."""

    def __init__(self, prefix, binary, model, scratch, extra=()):  # noqa: D107 (same token handling as Server)
        import secrets
        self.token = secrets.token_urlsafe(32)
        token_path = Path(scratch) / "memcheck.token"
        fd = os.open(token_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(self.token + "\n")
        self.binary = binary
        self.proc = subprocess.Popen([*prefix, binary, "--experimental", model, "--token-file", str(token_path),
                                      "--port", "0", *extra], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        threading.Thread(target=self._drain_stderr, daemon=True).start()
        self.port = None
        for line in self.proc.stdout:
            try:
                ready = json.loads(line)
            except ValueError:
                continue
            if isinstance(ready, dict) and ready.get("ready"):
                self.port = ready["port"]
                break
        if self.port is None:
            raise RuntimeError(f"the server never became ready (exit {self.proc.wait()})")
        token_path.unlink()
        self.server_pid = self._server_pid()

    def _drain_stderr(self):
        self.stderr_lines = []
        for line in self.proc.stderr:
            self.stderr_lines.append(line)

    def _server_pid(self):
        """The server process: PROC itself, or its descendant running the server binary."""
        want = os.path.realpath(self.binary)
        frontier, seen = [self.proc.pid], set()
        while frontier:
            pid = frontier.pop()
            seen.add(pid)
            try:
                argv = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")[:2]
                # the binary itself, or (a script) its interpreter with the script as its first argument
                if os.path.realpath(f"/proc/{pid}/exe") == want or want in (os.path.realpath(a.decode()) for a in argv if a):
                    return pid
            except OSError:
                pass
            for p in Path("/proc").iterdir():
                if p.name.isdigit() and int(p.name) not in seen:
                    try:
                        stat = (p / "stat").read_text()
                    except OSError:
                        continue
                    if int(stat[stat.rindex(")") + 2:].split()[1]) == pid:
                        frontier.append(int(p.name))
        raise RuntimeError("cannot find the server process under the sanitizer")

    def stop(self, timeout=7200):
        os.kill(self.server_pid, signal.SIGTERM)  # SIGTERM only: the server's orderly shutdown
        self.proc.wait(timeout=timeout)
        return self.proc.returncode, "".join(getattr(self, "stderr_lines", []))


def memcheck(sanitizer, server, model, round_dir, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    label = "plain" if sanitizer == "-" else "memcheck"
    names = sorted(set(MEMCHECK_SEQUENCE + MEMCHECK_AT_ONCE + MEMCHECK_LAST))
    bodies = {n: dict(case(round_dir, n)[0], max_new_tokens=8) for n in names}
    budget = MEMCHECK_BUDGET_MB * MIB
    keys_of = {n: [(k, key_costs(k)) for k in image_keys(b)] for n, b in bodies.items()}
    expected = simulate(MEMCHECK_SEQUENCE, keys_of, budget)
    flat = [e for _, ev in expected for e in ev]
    pre = dict(hits=sum(e == "hit" for e in flat), misses=sum(e.startswith("miss") for e in flat),
               evictions=sum(e.startswith("miss, evicts") for e in flat), over_budget=sum(e == "over" for e in flat))
    assert all(pre.values()), f"the sequence does not cover hits, misses, evictions and over-budget images: {pre}"
    extra = ["--image-cache-mb", str(MEMCHECK_BUDGET_MB), "--queue-size", "16", "--request-timeout", "7200",
             "--engine-timeout", "7200", "--socket-timeout", "600"]
    prefix = [] if sanitizer == "-" else [sanitizer, "--tool", "memcheck", "--error-exitcode", "99", "--print-limit",
                                          "50", "--log-file", str(out / "memcheck.log")]
    rec = dict(label=label, expected_engine_cache=expected, coverage=pre, sequence=[], at_once={}, last=[])
    t0 = time.monotonic()
    with tempfile.TemporaryDirectory() as scratch:
        s = SanitizedServer(prefix, server, model, scratch, extra=extra)
        rec["startup_seconds"] = time.monotonic() - t0
        try:
            for n in MEMCHECK_SEQUENCE:
                st, r = s.post(bodies[n], timeout=7200)
                rec["sequence"].append(dict(name=n, status=st, text=r.get("text"), prefill_seconds=r.get("prefill_seconds")))
                print(label, n, st, r.get("prefill_seconds"), flush=True)
            got = {}
            ts = [threading.Thread(target=lambda n=n: got.__setitem__(n, post_retry(s, bodies[n], timeout=7200)))
                  for n in MEMCHECK_AT_ONCE]
            [t.start() for t in ts]
            [t.join() for t in ts]
            rec["at_once"] = {n: dict(status=st, text=r.get("text")) for n, (st, r) in got.items()}
            for n in MEMCHECK_LAST:
                st, r = s.post(bodies[n], timeout=7200)
                rec["last"].append(dict(name=n, status=st, text=r.get("text"), prefill_seconds=r.get("prefill_seconds")))
        finally:
            rec["exit"], err = s.stop()
            rec["stderr_tail"] = err[-4000:]
    rec["seconds"] = time.monotonic() - t0
    statuses = [x["status"] for x in rec["sequence"] + rec["last"]] + [x["status"] for x in rec["at_once"].values()]
    rec["statuses_200"] = all(x == 200 for x in statuses)
    if label == "memcheck":
        log = (out / "memcheck.log").read_text(errors="replace") if (out / "memcheck.log").exists() else ""
        summary = [l for l in log.splitlines() if "ERROR SUMMARY" in l]
        rec["error_summary"] = summary
        rec["zero_errors"] = rec["exit"] == 0 and len(summary) == 1 and "ERROR SUMMARY: 0 errors" in summary[0]
        plain = out / "plain.json"
        if plain.exists():
            p = json.loads(plain.read_text())
            strip = lambda r: ([x["text"] for x in r["sequence"]], {k: v["text"] for k, v in r["at_once"].items()},  # noqa: E731
                               [x["text"] for x in r["last"]])
            rec["texts_equal_plain"] = strip(p) == strip(rec)
        else:
            rec["texts_equal_plain"] = None
        rec["passed"] = rec["zero_errors"] and rec["statuses_200"] and rec["texts_equal_plain"] is True
        verdict = f"{summary}; texts {'equal to plain' if rec['texts_equal_plain'] else 'NOT EQUAL to plain'}"
    else:
        rec["passed"] = rec["statuses_200"] and rec["exit"] == 0
        verdict = f"exit {rec['exit']}"
    (out / f"{label}.json").write_text(json.dumps(rec, indent=1))
    print(f"{label}: {rec['seconds']:.0f} s, statuses {'all 200' if rec['statuses_200'] else statuses}, {verdict}; "
          f"expected engine cache {pre}; SERVER {label.upper()} {'PASS' if rec['passed'] else 'FAIL'}")
    return 0 if rec["passed"] else 1


# ---------------------------------------------------------------------------------------------------- speech-list
def speech_list(round_dir, out_dir, *names):
    out = Path(out_dir)
    (out / "req").mkdir(parents=True, exist_ok=True)
    for label in ("plain", "memcheck"):
        (out / label).mkdir(exist_ok=True)
    lines = {"plain": [], "memcheck": []}
    for i, name in enumerate(names):
        req = json.loads((Path(round_dir) / "req" / f"{name}.json").read_text())
        req["max_new_tokens"] = 16
        req.pop("record_image", None)
        stem = f"{i:02d}-{name}"
        (out / "req" / f"{stem}.json").write_text(json.dumps(req))
        for label in lines:
            lines[label].append(f"{out / 'req' / stem}.json {out / label / stem}\n")
    for label, ls in lines.items():
        (out / f"list-{label}.txt").write_text("".join(ls))
    print(f"{len(names)} requests: {', '.join(names)}")
    return 0


def main(argv):
    if not argv:
        sys.exit(__doc__)
    cmd, args = argv[0], argv[1:]
    if cmd == "c3" and len(args) == 6:
        return c3(*args)
    if cmd == "c4" and len(args) in (5, 6):
        return c4(*args)
    if cmd == "c5" and len(args) == 5:
        return c5(*args)
    if cmd == "memcheck" and len(args) == 5:
        return memcheck(*args)
    if cmd == "speech-list" and len(args) >= 3:
        return speech_list(*args)
    sys.exit(__doc__)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
