#!/usr/bin/env bash
# Tripp chose (b), 2026-11-12: the quiet server checks, fuzz and soak on assure-s3 first, then racecheck one at a time.
cd ~/Projects/000-research/lfm3-omni-2B/liquid-CUDA
echo "start $(date +%T)"
bash tools/thor/assurance_server.sh assure-a2 assure-s3 "server fuzz soak" > build/assure-server3.log 2>&1
cat build/assure-server3.log
echo "== racecheck one at a time $(date +%T)"
T=assure-a2 MODES=batch FP32=0 bash tools/thor/assurance_sanitizers.sh racecheck > build/assure-san-a2-race-batch2.log 2>&1
cat build/assure-san-a2-race-batch2.log
echo "DAY2 DONE $(date +%T)"
