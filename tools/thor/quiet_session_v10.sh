#!/usr/bin/env bash
# Quiet window 10 (2026-11-17), after window 9: validation/image-cache-v1's red-team fixes C3-C5 (commit 0d743af), the
# rules set before the code and the procedure set before the run (the README's "C3-C5 Thor procedure"). Builds and
# exactness first, the stall (timing) last, alone, after a second quiet check.
#   0. builds from the tree as synced (vimg-srv-c3: build_native.sh as is; vimg-srv-c3f: the same with
#      -DLFM3_FAULT_IMAGE_KEEP; vimg-c3: build_speech.sh), in parallel, ~25 min; images (make_lfm3_c3_images.py), ~3 min
#   1. exact: the 11 stage-0 requests through lfm3-speech against vimg-e2e-a4, ~3 min; check_lfm3_image_cache.py
#      32 of 32, ~25 min
#   2. C3, where the evicted image is decoded (thread CPU), ~4 min
#   3. C4, memory after 1 and 16 distinct 4K requests, ~4 min, and the same on vimg-srv-c2 (a control, not gated)
#   4. C5, the fault build against cache off and on, ~6 min
#   5. C4, compute-sanitizer memcheck: lfm3-speech (leak check) and the server (hits, misses, evictions), ~1-3 h
#   6. C3, the stall (timing): cache on unprimed, primed (the C3 state), cache off; ~20 min
# Usage (Thor host): nohup bash tools/thor/quiet_session_v10.sh > build/quiet-w10.log 2>&1 &
#   (results: build/quiet-w10-c3c5/; REUSE_BUILDS=1 keeps existing vimg-srv-c3, vimg-srv-c3f, vimg-c3 builds;
#    LISTEN_DIR=host path of the listening set's req/ if the search below does not find it)
set -uo pipefail
W=${LFM3_WORKSPACE:-$HOME/Projects/000-research/lfm3-omni-2B}; R=$W/liquid-CUDA; B=$R/build; C=/workspace/liquid-CUDA/build
O=quiet-w10-c3c5; M=/workspace/models/LFM3-Omni-2B; T=/workspace/liquid-CUDA/tools
X="docker exec -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia lfm3-profile-v1"
IMG=sha256:3c2b5b8e0857ada02326ee34175d52c35f6c909d89f49df6ae12f5ac5109052f
SAN=$C/sanitizer-13.5.25/cuda_sanitizer_api-linux-sbsa-13.5.25-archive/compute-sanitizer/compute-sanitizer
SRV=vimg-srv-c3; FLT=vimg-srv-c3f; SPE=vimg-c3; E2E=vimg-e2e-a4; OLD=vimg-srv-c2
CHERRY=$C/vision-ref/hf-5.20.2/images/cherry.png; STALL_REQ=$C/vimg-e2e-a1/req/cherry.json
step() { echo "== $1 $(date +%T)"; }

# inputs, checked before waiting (a missing one would waste the window)
missing=0
for f in $R/native/src/lfm3_speech.cuh $B/$E2E/req/cherry.json $B/$E2E/out/cherry.json $B/vision-ref/hf-5.20.2/images/cherry.png \
         $B/vimg-e2e-a1/req/cherry.json $B/images-v1/decode-aarch64/ljt/lib/libjpeg.a ${SAN/#$C/$B} \
         $R/tools/check_lfm3_image_cache_fixes.py $R/tools/make_lfm3_c3_images.py $R/tools/compare_speech_outputs.py; do
  [ -e $f ] || { echo "MISSING $f"; missing=1; }
done
grep -q LFM3_FAULT_IMAGE_KEEP $R/native/src/lfm3_speech.cuh && grep -q shared_held_images $R/native/include/lfm/lfm3_http.hpp \
  && grep -q -- --prime-images $R/tools/measure_lfm3_image_stall.py || { echo "MISSING the C3-C5 sources (0d743af) or tools"; missing=1; }
LISTEN=${LISTEN_DIR:-}
if [ -z "$LISTEN" ]; then
  for d in $B/*/req $B/*/*/req; do [ -f $d/01-china.json ] && [ -f $d/08-mm05.json ] && { LISTEN=$d; break; }; done
fi
[ -n "$LISTEN" ] && [ -f $LISTEN/08-mm05.json ] || { echo "MISSING the listening set's req/ (set LISTEN_DIR)"; missing=1; }
[ $missing = 0 ] || { echo "NOT STARTED: inputs missing"; echo "QUIET-V10-FINISHED $(date +%T)"; exit 1; }
LISTEN_C=${LISTEN/#$B/$C}
echo "listening set: $LISTEN"

until grep -q "^QUIET-V9-FINISHED\|^NOT QUIET" $B/quiet-w9.log 2>/dev/null; do sleep 60; done
busy() { pgrep -af "[l]fm3-speech|[l]iquid-lfm3|[c]ompute-sanitizer|[l]fm3-vision-check|[a]udio-encoder-bench|[v]envs/hf-|[n]sys|[t]cgen05_gemm|[t]cgen05-soak|[t]gemm_host_race|[f]rontend-|[n]vcc|[c]make --build|[n]inja" | grep -v "quiet_" | head -3; }
wait_quiet() {  # 3 quiet checks in a row, 30 s apart, within $1 checks
  local quiet=0 i
  for i in $(seq 1 $1); do
    if [ -z "$(busy)" ]; then quiet=$((quiet + 1)); [ $quiet -ge 3 ] && return 0; else quiet=0; fi
    [ $i = 1 ] && echo "waiting for quiet $(date +%T): $(busy | cut -c1-150)"; sleep 30
  done
  [ -z "$(busy)" ]
}
wait_quiet 1440 || { echo "NOT QUIET after 12 h"; exit 1; }
echo "== quiet $(date +%T)"
mkdir -p $B/$O && cd $B/$O || exit 1
bash $R/tools/thor/power_state.sh power-start.json > /dev/null || { echo "FAIL power"; exit 1; }
(cd $R && git rev-parse HEAD 2>/dev/null; sha256sum native/src/lfm3_speech.cuh native/src/lfm3_worker.cu native/src/lfm3_server.cu \
  native/apps/lfm3_speech.cu native/include/lfm/image_handoff.hpp native/include/lfm/lfm3_http.hpp native/include/lfm/lru_bytes.hpp \
  native/include/lfm/speech_run.hpp tools/check_lfm3_image_cache_fixes.py tools/check_lfm3_image_cache.py \
  tools/measure_lfm3_image_stall.py tools/make_lfm3_c3_images.py tools/thor/quiet_session_v10.sh) > sources.txt

step "0. builds"
build_srv() { docker run --rm --network none -v $R/native:/src/native:ro -v $W:/workspace $IMG bash -c "bash /workspace/liquid-CUDA/tools/thor/build_native.sh $C/$1 ${2:-}; rc=\$?; chown -R 1000:1000 $C/$1 $C/$1.cmake.log $C/$1.build.log 2>/dev/null; exit \$rc"; }
built() { [ -n "${REUSE_BUILDS:-}" ] && [ -x $B/$1/$2 ]; }
for t in $SRV $FLT $SPE; do [ -e $B/$t ] && [ -z "${REUSE_BUILDS:-}" ] && { echo "FAIL: build/$t exists (REUSE_BUILDS=1 to keep it)"; exit 1; }; done
built $SRV liquid-lfm3-server || (build_srv $SRV > build-$SRV.log 2>&1; echo $? > build-$SRV.rc) &
built $FLT liquid-lfm3-server || (build_srv $FLT -DCMAKE_CUDA_FLAGS=-DLFM3_FAULT_IMAGE_KEEP > build-$FLT.log 2>&1; echo $? > build-$FLT.rc) &
built $SPE lfm3-speech || (LFM3_WORKSPACE=$W bash $R/tools/thor/build_speech.sh $SPE > build-$SPE.log 2>&1; echo $? > build-$SPE.rc) &
wait
for t in $SRV $FLT $SPE; do [ -f build-$t.rc ] && echo "$t: exit $(cat build-$t.rc)"; done
[ -x $B/$SRV/liquid-lfm3-server ] && [ -x $B/$FLT/liquid-lfm3-server ] && [ -x $B/$SPE/lfm3-speech ] || { echo "FAIL builds"; tail -5 build-*.log; exit 1; }
(cd $B && sha256sum $SRV/liquid-lfm3-server $SRV/liquid-lfm3-worker $FLT/liquid-lfm3-server $SPE/lfm3-speech) | tee binaries.sha256
grep -E "CMAKE_CUDA_FLAGS:|LIQUID_LFM3_NATIVE_GEMM:" $B/$SRV/CMakeCache.txt $B/$FLT/CMakeCache.txt
inj() { grep -a -c "injected: image rows not kept" $1; }
echo "fault string: $SRV $(inj $B/$SRV/liquid-lfm3-server), $FLT $(inj $B/$FLT/liquid-lfm3-server) (want 0 and >= 1)"
[ "$(inj $B/$SRV/liquid-lfm3-server)" = 0 ] && [ "$(inj $B/$FLT/liquid-lfm3-server)" -ge 1 ] || { echo "FAIL fault build"; exit 1; }
for f in $SRV/liquid-lfm3-server $FLT/liquid-lfm3-server; do readelf -d $B/$f | grep -q cublas && { echo "FAIL $f links cuBLAS"; exit 1; }; done

step "0. images"
$X python3 $T/make_lfm3_c3_images.py $CHERRY $C/$O/images/prime 1 30 > images.log 2>&1 \
  && $X python3 $T/make_lfm3_c3_images.py $CHERRY $C/$O/images/fresh 31 31 >> images.log 2>&1 || { echo "FAIL images"; tail -3 images.log; exit 1; }
(cd images && sha256sum prime/*.png fresh/*.png) > images.sha256; echo "$(wc -l < images.sha256) images"

step "1. exact: lfm3-speech, the 11 stage-0 requests against $E2E"
mkdir -p e2e/out
for q in $B/$E2E/req/*.json; do n=$(basename $q .json); echo "$C/$E2E/req/$n.json $C/$O/e2e/out/$n"; done > e2e/list.txt
$X $C/$SPE/lfm3-speech $M --batch $C/$O/e2e/list.txt > e2e/run.log 2>&1; echo "exit $?, $(wc -l < e2e/list.txt) requests"
python3 $R/tools/compare_speech_outputs.py $B/$E2E/out $B/$O/e2e/out $B/$O/e2e/compare.json && echo "E2E IDENTICAL" || echo "E2E DIFFER"

step "1. exact: check_lfm3_image_cache.py on $SRV"
$X python3 $T/check_lfm3_image_cache.py $C/$SRV/liquid-lfm3-server $M $C/$E2E $C/$O/image-cache.json $LISTEN_C > image-cache.log 2>&1
grep DIFFER image-cache.log | head; tail -1 image-cache.log

step "2. C3: where the evicted image is decoded"
$X python3 $T/check_lfm3_image_cache_fixes.py c3 $C/$SRV/liquid-lfm3-server $M $C/$E2E $C/$O/images/prime \
  $C/$O/images/fresh/c3-31.png $C/$O/c3.json > c3.log 2>&1; tail -1 c3.log

step "3. C4: memory after 1 and 16 distinct 4K requests"
$X python3 $T/check_lfm3_image_cache_fixes.py c4 $C/$SRV/liquid-lfm3-server $M $C/$E2E $C/$O/images/prime $C/$O/c4.json $SRV > c4.log 2>&1; tail -1 c4.log
if [ -x $B/$OLD/liquid-lfm3-server ]; then  # control: before C1 and C4 (not gated)
  $X python3 $T/check_lfm3_image_cache_fixes.py c4 $C/$OLD/liquid-lfm3-server $M $C/$E2E $C/$O/images/prime $C/$O/c4-control.json $OLD > c4-control.log 2>&1; tail -1 c4-control.log
fi

step "4. C5: the fault build"
$X python3 $T/check_lfm3_image_cache_fixes.py c5 $C/$SRV/liquid-lfm3-server $C/$FLT/liquid-lfm3-server $M $C/$E2E $C/$O/c5.json > c5.log 2>&1; tail -8 c5.log

step "5. C4: memcheck, lfm3-speech (leak check)"
$X python3 $T/check_lfm3_image_cache_fixes.py speech-list $C/$E2E $C/$O/memcheck-speech cherry two small small flower > memcheck-speech.log 2>&1
$X $C/$SPE/lfm3-speech $M --concurrent 4 $C/$O/memcheck-speech/list-plain.txt 3 > memcheck-speech-plain.log 2>&1; echo "plain: exit $?"
t0=$(date +%s)
$X $SAN --tool memcheck --leak-check full --error-exitcode 99 --print-limit 50 $C/$SPE/lfm3-speech $M --concurrent 4 \
  $C/$O/memcheck-speech/list-memcheck.txt 3 > memcheck-speech-run.log 2>&1; s=$?
echo "memcheck: exit $s, $(( $(date +%s) - t0 )) s; $(grep -E 'ERROR SUMMARY|LEAK SUMMARY' memcheck-speech-run.log | tr '\n' ' ')"
python3 $R/tools/compare_speech_outputs.py $B/$O/memcheck-speech/plain $B/$O/memcheck-speech/memcheck $B/$O/memcheck-speech/compare.json > /dev/null \
  && echo "  outputs identical to plain" || echo "  OUTPUTS DIFFER from plain"

step "5. C4: memcheck, the server at --image-cache-mb 1 (hits, misses, evictions)"
$X python3 $T/check_lfm3_image_cache_fixes.py memcheck - $C/$SRV/liquid-lfm3-server $M $C/$E2E $C/$O/memcheck-server > memcheck-server-plain.log 2>&1; tail -1 memcheck-server-plain.log
$X python3 $T/check_lfm3_image_cache_fixes.py memcheck $SAN $C/$SRV/liquid-lfm3-server $M $C/$E2E $C/$O/memcheck-server > memcheck-server.log 2>&1; tail -1 memcheck-server.log

step "6. C3: the stall (timing), after a second quiet check"
if wait_quiet 240; then
  bash $R/tools/thor/power_state.sh power-stall-start.json > /dev/null || echo "VOID power not MAXN"
  $X python3 $T/measure_lfm3_image_stall.py $C/$SRV/liquid-lfm3-server $M $STALL_REQ $C/$O/stall-hit.json 3 > stall-hit.log 2>&1; echo "hit (unprimed): $(tail -1 stall-hit.log)"
  $X python3 $T/measure_lfm3_image_stall.py $C/$SRV/liquid-lfm3-server $M $STALL_REQ $C/$O/stall-c3.json 3 --prime-images $C/$O/images/prime > stall-c3.log 2>&1; echo "C3 state (primed): $(tail -1 stall-c3.log)"
  $X python3 $T/measure_lfm3_image_stall.py $C/$SRV/liquid-lfm3-server $M $STALL_REQ $C/$O/stall-off.json 3 --image-cache-mb 0 > stall-off.log 2>&1; echo "cache off: $(tail -1 stall-off.log)"
  bash $R/tools/thor/power_state.sh power-stall-end.json power-stall-start.json > /dev/null || echo "VOID power changed during the stall"
  python3 - stall-hit.json stall-c3.json stall-off.json <<'EOF'
import json, sys
try:
    hit, c3, off = (json.load(open(p))["stall_seconds"] for p in sys.argv[1:4])
except (OSError, KeyError, ValueError) as e:
    sys.exit(f"STALL READING INCOMPLETE: {e}")
print(f"stall: hit {hit:.3f} s, C3 state {c3:.3f} s, cache off {off:.3f} s (window 4: hit 0.28; window 3: off 0.66; "
      f"before part 2: 1.06)")
print(f"C3 STALL AS WRITTEN (within 0.10 s of 0.28): {'PASS' if abs(c3 - 0.28) <= 0.10 else 'FAIL'}")
print(f"C3 STALL, engine thread did not decode (C3 state <= this window's cache off + 0.10): {'yes' if c3 <= off + 0.10 else 'NO'}")
EOF
else
  echo "STALL SKIPPED: not quiet within 2 h ($(busy | cut -c1-150))"
fi

bash $R/tools/thor/power_state.sh power-end.json power-start.json > /dev/null || echo "VOID power changed"
step summary
python3 -c "import json; print('e2e', json.load(open('e2e/compare.json'))['pass_'])" 2>/dev/null || echo "e2e: no record"
for f in image-cache.log c3.log c4.log c4-control.log c5.log memcheck-server.log stall-hit.log stall-c3.log stall-off.log; do
  [ -f $f ] && echo "$f: $(tail -1 $f | cut -c1-220)"
done
grep -E "ERROR SUMMARY|LEAK SUMMARY" memcheck-speech-run.log 2>/dev/null | sed 's/^/memcheck-speech: /'
echo "QUIET-V10-FINISHED $(date +%T)"
