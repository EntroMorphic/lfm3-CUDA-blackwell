#!/usr/bin/env bash
# After the s3 soak failed rule 4 on memory (glibc mmap threshold; fixed in assure-s4): once racecheck one at a time
# (day2.sh) and the memory diagnosis are done, and Thor is quiet, the server checks, fuzz and soak on assure-s4.
cd ~/Projects/000-research/lfm3-omni-2B/liquid-CUDA
echo "waiting $(date +%T)"
until grep -q "DAY2 DONE" build/assure/day2.log; do sleep 120; done
while pgrep -f "[d]iag_lfm3_server_memory|[c]ompute-sanitizer|[l]fm3-speech |[l]iquid-lfm3-server" > /dev/null; do sleep 60; done
echo "quiet $(date +%T)"; ps -eo pid,etime,args | grep -E "[c]ompute-sanitizer|[l]fm3-speech|[l]iquid-lfm3" || echo "no GPU jobs"
bash tools/thor/assurance_server.sh assure-a2 assure-s4 "server fuzz soak" > build/assure-server4.log 2>&1
cat build/assure-server4.log
echo "DAY3 DONE $(date +%T)"
