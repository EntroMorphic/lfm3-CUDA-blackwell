#!/usr/bin/env bash
# Sanitizer rule of validation/tcgen05-probe-v1 for one variant, on the Thor host (probe in lfm3-profile-v1):
# memcheck (leak check), racecheck, synccheck and initcheck on the sanitizer case set, each output compared byte for
# byte with a plain run of the same binary.
#   tcgen05_probe_sanitize.sh BUILD_TAG VARIANT
# Writes build/tcgen05-BUILD_TAG/sanitize-vVARIANT/{plain,memcheck,racecheck,synccheck,initcheck}/ and summary.txt.
set -uo pipefail
tag=$1; variant=$2
W=~/Projects/000-research/lfm3-omni-2B
B=/workspace/liquid-CUDA/build
CS=$B/sanitizer-13.5.25/cuda_sanitizer_api-linux-sbsa-13.5.25-archive/compute-sanitizer/compute-sanitizer
host_out=$W/liquid-CUDA/build/tcgen05-$tag/sanitize-v$variant
out=$B/tcgen05-$tag/sanitize-v$variant
probe=$B/tcgen05-$tag/tcgen05-gemm-probe
cases=(--m 100,257,1000 --data int,normal,split)
run() { docker exec -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia lfm3-profile-v1 "$@"; }
mkdir -p "$host_out"/{plain,memcheck,racecheck,synccheck,initcheck}
run "$probe" sanitize "$variant" "$out/plain" "${cases[@]}" > "$host_out/plain/stdout.jsonl" 2> "$host_out/plain/stderr.log"
echo "plain rc=$?" > "$host_out/summary.txt"
run "$CS" --version > "$host_out/sanitizer-version.txt" 2>&1
for tool in memcheck racecheck synccheck initcheck; do
  extra=()
  [[ $tool == memcheck ]] && extra=(--leak-check full)
  run "$CS" --tool "$tool" "${extra[@]}" --error-exitcode 86 --log-file "$out/$tool/sanitizer.log" \
    "$probe" sanitize "$variant" "$out/$tool" "${cases[@]}" --timeout-ms 600000 \
    > "$host_out/$tool/stdout.jsonl" 2> "$host_out/$tool/stderr.log"
  rc=$?
  summary=$(grep -E "ERROR SUMMARY|LEAK SUMMARY|RACECHECK SUMMARY" "$host_out/$tool/sanitizer.log" | tr '\n' ' ')
  differ=0; compared=0
  for f in "$host_out"/plain/*.f32 "$host_out"/plain/*.bf16; do
    [[ -e $f ]] || continue
    compared=$((compared + 1))
    cmp -s "$f" "$host_out/$tool/$(basename "$f")" || differ=$((differ + 1))
  done
  echo "$tool rc=$rc outputs_compared=$compared outputs_differing=$differ :: $summary" >> "$host_out/summary.txt"
done
# Positive controls (the same binary and kernels; must be REPORTED): initcheck on a C row the probe never writes,
# memcheck on the probe writing one row past C's allocation. Without them a clean report could mean the tool saw nothing.
for ctl in "initcheck unwritten" "memcheck oob"; do
  set -- $ctl; tool=$1; kind=$2
  mkdir -p "$host_out/control-$kind"
  run "$CS" --tool "$tool" --error-exitcode 86 --log-file "$out/control-$kind/sanitizer.log" \
    "$probe" sanitize "$variant" "$out/control-$kind" --m 100 --data int --control "$kind" --timeout-ms 600000 \
    > "$host_out/control-$kind/stdout.jsonl" 2> "$host_out/control-$kind/stderr.log"
  rc=$?
  summary=$(grep -E "ERROR SUMMARY" "$host_out/control-$kind/sanitizer.log" | tr '\n' ' ')
  verdict=$([[ $rc == 86 ]] && echo "control DETECTED" || echo "control NOT DETECTED")
  echo "control $kind ($tool) rc=$rc $verdict :: $summary" >> "$host_out/summary.txt"
done
echo done >> "$host_out/summary.txt"
