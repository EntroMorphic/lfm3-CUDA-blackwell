#!/usr/bin/env bash
# Installs liquid-lfm3-server as a systemd user service (validation/always-on-v1; docs/service.md): fills in
# deploy/liquid-lfm3.service, writes it to ~/.config/systemd/user/liquid-lfm3.service ($XDG_CONFIG_HOME/systemd/user),
# then `systemctl --user daemon-reload` and `systemctl --user enable`. It does not start the service and never uses
# sudo: starting at boot without a login needs `sudo loginctl enable-linger $USER`, which it prints for you to run.
#
# Usage: deploy/install-user-service.sh --server PATH --model DIR [--port N] [--library-path DIRS] [--speech-rows N]
#                                       [--image-cache-mb N] [--dry-run]
#   --server        the liquid-lfm3-server binary (absolute path, executable)
#   --model         the checkpoint directory (absolute path, holding config.json)
#   --port          the loopback port (default 8080)
#   --library-path  LD_LIBRARY_PATH for the server (e.g. the pinned CUDA libraries); default: none
#   --speech-rows, --image-cache-mb   passed to the server as they are
#   --dry-run       print the unit and do nothing else
# SYSTEMCTL (default systemctl) names the systemctl to run (the host tests use a stand-in).
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd "$here/.." && pwd)
server="" model="" port=8080 library_path="" extra="" dry_run=0
die() { echo "install-user-service: $*" >&2; exit 2; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --server) server=${2:?--server needs a path}; shift 2 ;;
    --model) model=${2:?--model needs a directory}; shift 2 ;;
    --port) port=${2:?--port needs a number}; shift 2 ;;
    --library-path) library_path=${2:?--library-path needs directories}; shift 2 ;;
    --speech-rows|--image-cache-mb)
      [[ ${2:-} =~ ^[0-9]+$ ]] || die "$1 needs a whole number"
      extra+=" $1 $2"; shift 2 ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help) sed -n '2,15p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) die "unknown option $1" ;;
  esac
done
[[ -n $server && -n $model ]] || die "--server and --model are required (--help)"
[[ $port =~ ^[0-9]+$ ]] && (( port >= 1 && port <= 65535 )) || die "--port must be 1..65535"
# Paths go into the unit inside double quotes: refuse what systemd would read differently there.
for p in "$server" "$model" "$library_path" "$repo"; do
  [[ $p != *[\"\\%\$\`\|\&]* && $p != *$'\n'* ]] || die "a path holds a character the unit cannot carry: $p"
done
[[ $server == /* && $model == /* ]] || die "--server and --model must be absolute paths"
[[ -x $server && -f $server ]] || die "not an executable file: $server"
[[ -f $model/config.json ]] || die "no config.json in $model"

unit=$(sed -e "s|@SERVER@|$server|g" -e "s|@MODEL_DIR@|$model|g" -e "s|@PORT@|$port|g" -e "s|@EXTRA_ARGS@|$extra|g" \
           -e "s|@REPO@|$repo|g" -e "s|@LIBRARY_PATH@|$library_path|g" "$here/liquid-lfm3.service")
if [[ -z $library_path ]]; then  # no Environment line (an empty LD_LIBRARY_PATH entry would mean the working directory)
  unit=$(printf '%s\n' "$unit" | grep -v '^Environment="LD_LIBRARY_PATH=')
fi
unit=$(printf '%s\n' "$unit" | grep -v -e '^# A template: ' -e 'Not usable as it is\.$')
unit="# Installed by deploy/install-user-service.sh from $repo/deploy/liquid-lfm3.service on $(date -u +%Y-%m-%dT%H:%MZ).
$unit"
if grep -q '@[A-Z_]*@' <<< "$unit"; then die "a template value was left unfilled"; fi

if (( dry_run )); then
  printf '%s\n' "$unit"
  exit 0
fi
dir=${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user
mkdir -p "$dir"
printf '%s\n' "$unit" > "$dir/liquid-lfm3.service.tmp"
mv "$dir/liquid-lfm3.service.tmp" "$dir/liquid-lfm3.service"
echo "installed $dir/liquid-lfm3.service"
systemctl=${SYSTEMCTL:-systemctl}
"$systemctl" --user daemon-reload
"$systemctl" --user enable liquid-lfm3.service
cat <<EOF
enabled liquid-lfm3.service (systemd user service; it starts with your user manager).
  start it now:   systemctl --user start liquid-lfm3
  watch it:       systemctl --user status liquid-lfm3; journalctl --user -u liquid-lfm3 -f
  is it warm:     curl -s http://127.0.0.1:$port/health   ({"ready":true,...} once warm)
For it to start at boot, without anyone logged in, lingering must be on for $USER. This script does not use sudo;
run this yourself, once:
  sudo loginctl enable-linger $USER
EOF
