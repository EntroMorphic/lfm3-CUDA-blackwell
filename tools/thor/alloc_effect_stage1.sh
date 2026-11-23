#!/usr/bin/env bash
# validation/alloc-effect-v1 stage 1 (a diagnostic, rules in the README): alloc-placement-probe P1, P2 (after 120 s idle,
# then back to back), stage 0's lfm3-speech text run T after 120 s idle, P3 right after T, P4 after 120 s idle. Quiet
# Thor, the service stopped for the window. The readings as the README sets them.
# Usage (Thor host, copied to build/alloc-s1/run.sh): bash run.sh > run.log 2>&1
set -uo pipefail
W=$HOME/Projects/000-research/lfm3-omni-2B; R=$W/liquid-CUDA; B=$R/build; C=/workspace/liquid-CUDA/build; O=alloc-s1; S=alloc1-src
M=/workspace/models/LFM3-Omni-2B
IMG=sha256:3c2b5b8e0857ada02326ee34175d52c35f6c909d89f49df6ae12f5ac5109052f
X="docker exec -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia lfm3-profile-v1"
export XDG_RUNTIME_DIR=/run/user/$(id -u)
PAT="[l]fm3-speech|[l]iquid-lfm3|[c]ompute-sanitizer|[n]sys|[n]vcc|[c]make --build|[n]inja|[k]in_|[v]envs/hf-|[c]heck_lfm3|[a]udio-encoder-bench|[m]easure_lfm3|[a]udio-mel-check|[a]udio-handler-bench|[a]lloc-placement"
busy() { { pgrep -af "$PAT" | grep -v "$O" | grep -v "/.local/lib/liquid-lfm3/"; docker ps --format "{{.Names}}" | grep -v -x -E "lfm3-profile-v1|lfm3-omni-dev"; } | head -3; }
wait_quiet() { local quiet=0 i; for i in $(seq 1 $1); do if [ -z "$(busy)" ]; then quiet=$((quiet+1)); [ $quiet -ge 30 ] && return 0; else quiet=0; fi; [ $i = 1 ] && echo "waiting for quiet $(date +%T): $(busy | cut -c1-150)"; sleep 30; done; [ -z "$(busy)" ]; }
cd $B/$O || exit 1
docker run --rm --network none -v $W:/workspace $IMG bash -c "cd $C/$O && nvcc -O3 --fmad=false -std=c++20 -arch=sm_110 -I $C/$S/native/include \
  -DLFM3_NATIVE_GEMM -o alloc-placement-probe $C/$S/native/tests/alloc_placement_probe.cu $C/$S/native/src/safetensors.cpp > build.log 2>&1; rc=\$?; chown -R 1000:1000 .; exit \$rc"
echo "build exit $?"; sha256sum alloc-placement-probe $B/alloc-diag-speech/lfm3-speech | tee binaries.sha256
mkdir -p out-T
for k in 1 2 3 4; do for n in 128 512 1149 4096; do echo "$C/alloc-text/req/L$n.req.json $C/$O/out-T/L$n-r$k"; done; done > list-T.txt
(
wait_quiet 2880 || { echo "NOT QUIET after 24 h"; exit 1; }
SVC=0; systemctl --user is-active --quiet liquid-lfm3 && systemctl --user stop liquid-lfm3 && SVC=1 && echo "stopped liquid-lfm3 $(date +%T)"
SAMPLER=""
trap '[ -n "$SAMPLER" ] && kill -TERM $SAMPLER 2>/dev/null; [ $SVC = 1 ] && systemctl --user start liquid-lfm3 && echo "started liquid-lfm3 again $(date +%T)"' EXIT
sleep 30; [ -z "$(busy)" ] || { echo "NOT QUIET after stopping the service: $(busy)"; exit 1; }
bash $R/tools/thor/power_state.sh power-start.json > /dev/null || { echo "NOT MAXN"; exit 1; }
( while true; do t=$(date +%s.%N); pgrep -af "$PAT" | sed "s|^|$t P |"; docker ps --format "{{.Names}}" | sed "s|^|$t D |"; echo "$t T"; sleep 1; done ) > samples.log 2>&1 &
SAMPLER=$!
probe() { echo "== $1 $(date +%T)"; $X $C/$O/alloc-placement-probe 96 64 > $1.jsonl 2> $1.err; echo "$1 exit $?"; }
sleep 120; probe P1
probe P2
sleep 120; echo "== T $(date +%T)"; $X $C/alloc-diag-speech/lfm3-speech $M --batch $C/$O/list-T.txt > T.log 2>&1; echo "T exit $?"
probe P3
sleep 120; probe P4
bash $R/tools/thor/power_state.sh power-end.json power-start.json > /dev/null && echo "power: unchanged" || echo "POWER DIFFERS"
kill -TERM $SAMPLER 2>/dev/null; SAMPLER=""
)
python3 - <<'PY'
import collections, json, statistics
rows = collections.defaultdict(list)
for line in open('samples.log'):
    p = line.rstrip('\n').split(' ', 2)
    if len(p) >= 2: rows[p[0]].append(p[1:])
n = foreign = 0
for t, r in rows.items():
    if not any(x[0] == 'T' for x in r): continue
    n += 1
    procs = [x[1] for x in r if x[0] == 'P' and len(x) > 1]; dock = [x[1] for x in r if x[0] == 'D' and len(x) > 1]
    foreign += any(d not in ('lfm3-profile-v1', 'lfm3-omni-dev') for d in dock) or any(
        ('kin_' in q or 'nsys' in q or 'compute-sanitizer' in q or '/.local/lib/liquid-lfm3/' in q) for q in procs)
print(f"sampler: {n} samples; foreign in {foreign}"); print("MONITOR:", "CLEAN" if n and not foreign else "NOT CLEAN")
refs, placement, state = {}, False, False
for P in ('P1', 'P2', 'P3', 'P4'):
    L = [json.loads(l) for l in open(f'{P}.jsonl') if l.startswith('{')]
    bd = [x for x in L if 'buddyinfo_normal_before' in x or 'buddyinfo_normal_after' in x]
    B = [x for x in L if x.get('exp') == 'B']; A = [x for x in L if x.get('exp') == 'A']; ctl = [x for x in L if 'control' in x]
    ref = statistics.median(x['pass1_ms'] for x in B); refs[P] = ref; lim = 1.25 * ref
    def persist(rows, passes):
        slow = [x for x in rows if x['pass1_ms'] > lim]; fast = [x for x in rows if x['pass1_ms'] <= lim]
        kept = [x for x in slow if all(x[p] > lim for p in passes)]; turned = [x for x in fast if any(x[p] > lim for p in passes)]
        return slow, kept, fast, turned
    bs, bk, bf, bt = persist(B, ('pass2_ms', 'pass3_ms')); as_, ak, af, at = persist(A, ('pass2_ms',))
    c0 = ctl[0]['ms']; cmax = max(x['ms'] for x in ctl)
    ctl_state = cmax > 1.25 * c0; state |= ctl_state
    slow = len(bs) + len(as_); kept = len(bk) + len(ak); fast = len(bf) + len(af); turned = len(bt) + len(at)
    pl = slow >= 2 and kept >= 0.8 * slow and turned <= 0.05 * fast and not ctl_state; placement |= pl
    def q(v): v = sorted(v); return f"{v[0]:.3f} / {statistics.median(v):.3f} / {v[-1]:.3f}"
    print(f"== {P}: reference {ref:.4f} ms; B pass1/2/3 min/med/max {q([x['pass1_ms'] for x in B])} | {q([x['pass2_ms'] for x in B])} | {q([x['pass3_ms'] for x in B])}")
    print(f"   A pass1/2 {q([x['pass1_ms'] for x in A])} | {q([x['pass2_ms'] for x in A])}; chase ns/hop {q([x['chase_ns'] for x in B])}")
    print(f"   controls {len(ctl)}: first {c0:.4f}, min {min(x['ms'] for x in ctl):.4f}, max {cmax:.4f} ms{'  STATE' if ctl_state else ''}")
    print(f"   slow in pass 1: B {len(bs)}, A {len(as_)}; slow again in every later pass {kept}; fast turned slow {turned} of {fast}{'  PLACEMENT' if pl else ''}")
    if bs:
        sc = [x['chase_ns'] for x in bs]; fc = [x['chase_ns'] for x in bf]
        print(f"   chase: slow B {q(sc)}, fast B {q(fc) if fc else '-'}; translation {'supported' if fc and min(sc) > max(fc) else 'not supported'}")
    for x in bd: print("  ", x)
if max(refs.values()) > 1.25 * min(refs.values()): state = True; print("process references differ > 1.25x: STATE")
print("references:", {k: round(v, 4) for k, v in refs.items()})
print("READING:", " and ".join(r for r, f in (("PLACEMENT", placement), ("STATE", state)) if f) or "NEITHER")
print("T (prefill s: request 1 | median of 2-4; stage 0 off-2 in brackets):")
s0 = {512: 0.0673, 1149: 0.1388}
for L in (128, 512, 1149, 4096):
    try: t = [json.load(open(f'out-T/L{L}-r{k}.json'))['prefill_seconds'] for k in (1, 2, 3, 4)]
    except FileNotFoundError: print(f"  {L}: missing"); continue
    m = statistics.median(t[1:]); flag = '  SLOW' if L in s0 and m > 1.10 * s0[L] else ''
    print(f"  {L:5d}: {t[0]:.4f} | {m:.4f}" + (f"  [{s0[L]:.4f}]" if L in s0 else '') + flag)
PY
echo "ALLOC-S1-FINISHED $(date +%T)"
