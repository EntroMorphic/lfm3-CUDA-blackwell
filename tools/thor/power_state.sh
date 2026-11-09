#!/usr/bin/env bash
# Thor's power state, for the record of every speed measurement (no sudo needed). Prints one JSON line; exits 1 if
# the power mode is not MAXN (the pinned mode for measurements since 2026-11-09). "clocks_locked" is true when the
# GPU's minimum frequency equals its maximum (jetson_clocks); it is recorded, not required.
# Usage: power_state.sh [OUT.json]
set -uo pipefail
mode=$(nvpmodel -q 2>/dev/null | sed -n 's/^NV Power Mode: //p' | head -1)
f() { cat "$1" 2>/dev/null || echo 0; }
g=/sys/class/devfreq/gpu-gpc-0; c=/sys/devices/system/cpu/cpu0/cpufreq
json=$(printf '{"time":"%s","mode":"%s","gpu_governor":"%s","gpu_min_hz":%s,"gpu_max_hz":%s,"gpu_cur_hz":%s,"cpu_governor":"%s","cpu_min_khz":%s,"cpu_max_khz":%s,"clocks_locked":%s}' \
  "$(date -Is)" "$mode" "$(f $g/governor)" "$(f $g/min_freq)" "$(f $g/max_freq)" "$(f $g/cur_freq)" \
  "$(f $c/scaling_governor)" "$(f $c/scaling_min_freq)" "$(f $c/scaling_max_freq)" \
  "$([ "$(f $g/min_freq)" = "$(f $g/max_freq)" ] && echo true || echo false)")
echo "$json"
[ $# -ge 1 ] && echo "$json" > "$1"
[ "$mode" = "MAXN" ] || { echo "power mode is '$mode', not MAXN: not measuring speed" >&2; exit 1; }
