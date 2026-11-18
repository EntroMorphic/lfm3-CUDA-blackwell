#!/usr/bin/env bash
# always-on-v1 rule 5 on Thor (2026-11-17; linger is on, Tripp approved the service on the host and installing the
# unit). All of rule 5 but "starts at boot" (a reboot: Tripp's to approve separately). After the UVM test window.
#   1. installs a stable copy of the server (build/sig-srv, main 064c1bb: --warmup and the stop-signal forwarding) in
#      ~/.local/lib/liquid-lfm3/<sha256 prefix>/ and the unit (deploy/install-user-service.sh, the pinned CUDA libraries
#      on --library-path); systemd-analyze --user verify;
#   2. systemctl --user start (Type=notify: returns once warm), timed; /health; listening on 127.0.0.1 only;
#   3. three crash stand-ins: SIGTERM to the main process (never SIGKILL), each timed until /health answers 200 from a
#      new process; bound: rule 4's startup (4.18 s, window 15) + 10 s;
#   4. systemctl --user stop with a streamed TTS request in flight: stop exits 0 within 30 s, the unit's result success,
#      the request ended (a cancelled done event, or the connection closed);
#   5. starts it again and leaves it running (the always-on service).
# Run 1 (22:10) found the server could not rebind its port after a crash (TIME_WAIT; fixed: SO_REUSEADDR), and this
# script's pid read of 0 sent kill -TERM 0 (its own process group); a pid that is not a live server now stops the run.
# Usage (Thor host): [RULE5_SRC=BINARY] [RULE5_OUT=NAME] always_on_rule5.sh   (from the synced tree it belongs to)
set -uo pipefail
W=${LFM3_WORKSPACE:-$HOME/Projects/000-research/lfm3-omni-2B}; R=$W/liquid-CUDA; B=$R/build; O=${RULE5_OUT:-always-on-rule5}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SRC=${RULE5_SRC:-$B/sig-srv/liquid-lfm3-server}; LIBS=$B/racecheck-pinned-libs-v1:/usr/lib/aarch64-linux-gnu/nvidia
BOUND=14.18
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}
[ "$(loginctl show-user $USER -p Linger --value)" = yes ] || { echo "FAIL: linger is off"; echo "RULE5-FINISHED $(date +%T)"; exit 1; }
busy() { pgrep -af "[l]fm3-speech|[l]iquid-lfm3|[c]ompute-sanitizer|[n]sys|[n]vcc|[c]make --build" | grep -v "quiet_" | head -3; }
[ -z "$(busy)" ] || { echo "FAIL: not quiet: $(busy)"; echo "RULE5-FINISHED $(date +%T)"; exit 1; }
mkdir -p $B/$O && cd $B/$O || exit 1

echo "== 1. install $(date +%T)"
sha=$(sha256sum $SRC | cut -c1-12); D=$HOME/.local/lib/liquid-lfm3/$sha
mkdir -p $D && cp -p $SRC $D/liquid-lfm3-server && sha256sum $D/liquid-lfm3-server | tee binary.sha256
bash $HERE/../../deploy/install-user-service.sh --server $D/liquid-lfm3-server --model $W/models/LFM3-Omni-2B --library-path $LIBS > install.log 2>&1
echo "install exit $?"; head -3 install.log
cp ~/.config/systemd/user/liquid-lfm3.service unit.service
systemd-analyze --user verify ~/.config/systemd/user/liquid-lfm3.service > verify.log 2>&1; echo "verify exit $?; $(grep -c . verify.log) lines"

health() { curl -s -o /dev/null -w "%{http_code}" --max-time 2 http://127.0.0.1:8080/health; }
mainpid() { systemctl --user show -p MainPID --value liquid-lfm3; }

echo "== 2. start $(date +%T)"
systemctl --user reset-failed liquid-lfm3 2>/dev/null
t0=$(date +%s.%N); systemctl --user start liquid-lfm3; rc=$?; t1=$(date +%s.%N)
echo "start exit $rc, $(echo "$t1 - $t0" | bc) s to active; /health $(health); state $(systemctl --user is-active liquid-lfm3)"
journalctl --user -u liquid-lfm3 --since "-2 min" --no-pager -o cat | grep '"warmup"' | tail -1 | tee warmup-1.json
ss -ltnp 2>/dev/null | grep "pid=$(mainpid)," | tee listen.txt
grep -q "127.0.0.1:8080" listen.txt && [ "$(grep -c . listen.txt)" = 1 ] && echo "listening: 127.0.0.1:8080 only" || echo "LISTEN CHECK FAIL"

echo "== 3. crash stand-ins: SIGTERM to the main process (bound $BOUND s) $(date +%T)"
for i in 1 2 3; do
  p=$(mainpid)
  # never kill a pid that is not this service's live server (run 1: a pid of 0 signalled this script's own group)
  if ! [[ $p =~ ^[0-9]+$ ]] || [ "$p" -le 1 ] || ! grep -q liquid-lfm3 /proc/$p/comm 2>/dev/null; then
    echo "crash $i: no live server (MainPID '$p'): stopping the crash checks"; break
  fi
  t0=$(date +%s.%N); kill -TERM $p
  while true; do
    q=$(mainpid)
    [ "$q" != "$p" ] && [ "$q" != 0 ] && [ "$(health)" = 200 ] && break
    [ $(echo "$(date +%s.%N) - $t0 > 120" | bc) = 1 ] && { echo "crash $i: not back in 120 s"; break; }
    sleep 0.2
  done
  s=$(echo "$(date +%s.%N) - $t0" | bc)
  echo "crash $i: pid $p -> $(mainpid), warm after $s s: $([ $(echo "$s <= $BOUND" | bc) = 1 ] && echo PASS || echo FAIL)"
done

echo "== 4. stop with a request in flight $(date +%T)"
python3 - > stop.json <<'PY' &
import json, socket, time, pathlib
token = pathlib.Path.home().joinpath(".config/liquid-lfm3/token").read_text().strip()
body = json.dumps(dict(messages=[{"role": "system", "content": "Perform TTS in American English using a female voice."},
                                 {"role": "user", "content": "This is a long passage read aloud so that the request is still "
                                  "running when the service is stopped. " * 6}],
                       mode="audio", max_new_tokens=2048, seed=0, stream=True)).encode()
s = socket.create_connection(("127.0.0.1", 8080), timeout=60)
s.sendall(b"POST /speech HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nAuthorization: Bearer "
          + token.encode() + b"\r\nContent-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body)
t0, data, first_audio = time.monotonic(), b"", None
while True:
    try:
        b = s.recv(65536)
    except OSError as e:
        data += f"\n{{\"client_error\": \"{e}\"}}".encode(); break
    if not b: break
    data += b
    if first_audio is None and b'"event":"audio"' in data:
        first_audio = time.monotonic() - t0
        pathlib.Path("in-flight.flag").write_text("1")
head, _, rest = data.partition(b"\r\n\r\n")
lines = [l for l in rest.decode(errors="replace").splitlines() if l.startswith("{")]
last = lines[-1] if lines else ""
print(json.dumps(dict(status_line=head.split(b"\r\n")[0].decode(), first_audio_seconds=first_audio,
                      events=len(lines), last_event=last[:300], seconds=time.monotonic() - t0)))
PY
CL=$!
for i in $(seq 1 300); do [ -f in-flight.flag ] && break; sleep 0.1; done
[ -f in-flight.flag ] || echo "no audio event before the stop (the request may not be in flight)"
t0=$(date +%s.%N); timeout 60 systemctl --user stop liquid-lfm3; rc=$?; t1=$(date +%s.%N)
wait $CL
echo "stop exit $rc after $(echo "$t1 - $t0" | bc) s; $(systemctl --user show -p Result -p ExecMainStatus -p ExecMainCode liquid-lfm3 | tr '\n' ' ')"
echo "client: $(cat stop.json)"
pgrep -af "[l]iquid-lfm3-server" && echo "A SERVER PROCESS IS LEFT" || echo "no server process left"

echo "== 5. start again, left running $(date +%T)"
systemctl --user reset-failed liquid-lfm3 2>/dev/null  # 5 starts in this script: below the unit's StartLimitBurst, reset anyway
systemctl --user start liquid-lfm3; echo "start exit $?; /health $(health); $(systemctl --user is-enabled liquid-lfm3) $(systemctl --user is-active liquid-lfm3)"
echo "RULE5-FINISHED $(date +%T)"
