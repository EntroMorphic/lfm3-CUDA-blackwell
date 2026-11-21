#!/usr/bin/env bash
# validation/audio-ttft-v1, a diagnostic (not a rule): where the server's 90 s encode differs from the bench's. On a
# quiet Thor: nsys (CUDA trace) on build/ttft-srv with four 90 s requests (tools/profile_lfm3_audio_server.py), then
# on the bench (build/ttft-bench-a3, the 90 s clip, --repeat 3). Per encode (conv_in's launch to to_bf16's end): the
# span, the kernel time by name, and the CUDA calls; and for the server, the span from to_bf16 to the first token's
# kernels is not split out (the request's prefill_seconds is printed instead).
# Usage (Thor host, copied to build/ttft-prof/run.sh): bash run.sh > run.log 2>&1
set -uo pipefail
W=$HOME/Projects/000-research/lfm3-omni-2B; R=$W/liquid-CUDA; B=$R/build; O=ttft-prof; S=ttft-src-a4
export XDG_RUNTIME_DIR=/run/user/$(id -u)
PAT="[l]fm3-speech|[l]iquid-lfm3|[c]ompute-sanitizer|[n]sys|[n]vcc|[c]make --build|[n]inja|[k]in_|[v]envs/hf-|[c]heck_lfm3|[a]udio-encoder-bench|[m]easure_lfm3"
busy() { { pgrep -af "$PAT" | grep -v "$O" | grep -v "/.local/lib/liquid-lfm3/"; docker ps --format "{{.Names}}" | grep -v -x -E "lfm3-profile-v1|lfm3-omni-dev"; } | head -3; }
wait_quiet() { local quiet=0 i; for i in $(seq 1 $1); do if [ -z "$(busy)" ]; then quiet=$((quiet+1)); [ $quiet -ge 30 ] && return 0; else quiet=0; fi; [ $i = 1 ] && echo "waiting for quiet $(date +%T): $(busy | cut -c1-150)"; sleep 30; done; [ -z "$(busy)" ]; }
fin() { echo "TTFT-PROF-FINISHED $(date +%T)"; }
cd $B/$O || exit 1
LONG=$B/server-stream-v3-check/work/joined-90s.wav
wait_quiet 2880 || { echo "NOT QUIET after 24 h"; fin; exit 1; }
SVC=0; systemctl --user is-active --quiet liquid-lfm3 && systemctl --user stop liquid-lfm3 && SVC=1 && echo "stopped liquid-lfm3 $(date +%T)"
trap '[ $SVC = 1 ] && systemctl --user start liquid-lfm3 && echo "started liquid-lfm3 again $(date +%T)"' EXIT
sleep 30; [ -z "$(busy)" ] || { echo "NOT QUIET after stopping the service: $(busy)"; fin; exit 1; }
echo "== the server $(date +%T)"
python3 $B/$S/tools/profile_lfm3_audio_server.py /usr/local/bin/nsys $B/ttft-srv/liquid-lfm3-server $W/models/LFM3-Omni-2B $LONG 4 60 srv > srv.log 2>&1
echo "server exit $?"; grep '"request"' srv.log | cut -c1-200
echo "== the bench $(date +%T)"
nsys profile --trace=cuda --sample=none --cpuctxsw=none --force-overwrite=true -o bench $B/ttft-bench-a3/audio-encoder-bench \
  $W/models/LFM3-Omni-2B bench.json --repeat 3 joined90=$LONG > bench.log 2>&1; echo "bench exit $?"
for r in srv bench; do nsys export --type sqlite --force-overwrite true -o $r.sqlite $r.nsys-rep > /dev/null 2>&1; done
python3 - <<'PY'
import sqlite3, collections
for rep in ("srv", "bench"):
    c = sqlite3.connect(f"{rep}.sqlite")
    names = dict(c.execute("select id, value from StringIds"))
    k = [(s, e, names[n], corr) for s, e, n, corr in c.execute("select start, end, shortName, correlationId from CUPTI_ACTIVITY_KIND_KERNEL order by start")]
    rt = [(s, e, names[n], corr) for s, e, n, corr in c.execute("select start, end, nameId, correlationId from CUPTI_ACTIVITY_KIND_RUNTIME order by start")]
    launch = {corr: s for s, e, n, corr in rt}
    for j, i in enumerate(x for x, y in enumerate(k) if y[2] == "conv_in"):
        end = next(m for m in range(i, len(k)) if k[m][2] == "to_bf16")
        t0 = launch.get(k[i][3], k[i][0]); t1 = k[end][1]
        kern = collections.defaultdict(float)
        for s, e, n, corr in k[i:end + 1]: kern[n] += (e - s) / 1e6
        calls = collections.Counter(n.split("_v")[0] for s, e, n, corr in rt if t0 <= s <= t1)
        busy_gpu = sum(kern.values())
        print(f"{rep} encode {j}: span {(t1 - t0) / 1e6:.1f} ms, kernels {busy_gpu:.1f} ms, idle {(t1 - t0) / 1e6 - busy_gpu:.1f} ms, "
              f"launches {end - i + 1}; calls {dict(calls.most_common(6))}")
        print("    ", ", ".join(f"{n} {v:.2f}" for v, n in sorted(((v, n) for n, v in kern.items()), reverse=True)[:10]))
PY
fin
