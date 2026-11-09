#!/usr/bin/env bash
# Pin Thor's power state for speed measurements (2026-11-09, Tripp's decision): MAXN, clocks locked at the mode's
# maximum (jetson_clocks), both persistent across reboots. Run once on Thor as root: sudo bash pin_power.sh
# Undo: set "< PM_CONFIG DEFAULT=1 >" back in /etc/nvpmodel.conf (backup kept beside it), then
#       systemctl disable --now jetson-clocks.service; rm /etc/systemd/system/jetson-clocks.service; reboot.
set -euo pipefail
[ "$(id -u)" = 0 ] || { echo "run as root (sudo)"; exit 1; }

# 1. MAXN as the boot default (the file ships with DEFAULT=1, 120W)
conf=/etc/nvpmodel.conf
grep -q "< POWER_MODEL ID=0 NAME=MAXN >" $conf || { echo "no MAXN mode with ID 0 in $conf"; exit 1; }
[ -e $conf.before-pin ] || cp -p $conf $conf.before-pin
sed -i 's/^< PM_CONFIG DEFAULT=[0-9]* >$/< PM_CONFIG DEFAULT=0 >/' $conf
grep -q "^< PM_CONFIG DEFAULT=0 >$" $conf || { echo "could not set the default mode"; exit 1; }

# 2. MAXN now (nvpmodel may say a reboot is needed for some settings; it asks before rebooting)
nvpmodel -m 0

# 3. jetson_clocks at every boot, after nvpmodel has set the mode
cat > /etc/systemd/system/jetson-clocks.service <<'EOF'
[Unit]
Description=Lock CPU/GPU/EMC clocks at the power mode's maximum (jetson_clocks), for reproducible speed measurements
After=nvpmodel.service
Wants=nvpmodel.service

[Service]
Type=oneshot
ExecStart=/usr/bin/jetson_clocks
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable --now jetson-clocks.service

# 4. Show the result
nvpmodel -q
jetson_clocks --show
