#!/usr/bin/env bash
set -euo pipefail

CONF_FILE="/etc/sysctl.d/99-high-throughput-network.conf"

echo "[+] Writing sysctl configuration to $CONF_FILE"

sudo tee "$CONF_FILE" >/dev/null <<'EOF'
# =====================================================
# High Throughput / Low Latency Network Tuning
# Applied at boot + runtime
# =====================================================

# === SOCKET BUFFERS ===
net.core.rmem_max = 134217728
net.core.wmem_max = 134217728
net.core.rmem_default = 16777216
net.core.wmem_default = 16777216

# === TCP AUTOTUNING ===
net.ipv4.tcp_rmem = 4096 16777216 134217728
net.ipv4.tcp_wmem = 4096 16777216 134217728
net.ipv4.tcp_window_scaling = 1

# === NIC QUEUE / BUDGET ===
net.core.netdev_max_backlog = 32768
net.core.netdev_budget = 600
net.core.netdev_budget_usecs = 8000

# === CONNECTION BEHAVIOUR ===
net.ipv4.tcp_slow_start_after_idle = 0
net.ipv4.tcp_mtu_probing = 1
net.ipv4.tcp_congestion_control = bbr
net.core.default_qdisc = fq

# === CONNECTION LIFETIME / REUSE ===
net.ipv4.ip_local_port_range = 32768 65535
net.ipv4.tcp_fin_timeout = 15
net.ipv4.tcp_tw_reuse = 1
net.ipv4.tcp_fastopen = 3
net.core.somaxconn = 8192
net.ipv4.tcp_max_tw_buckets = 262144

# === KEEPALIVE SETTINGS ===
net.ipv4.tcp_keepalive_time = 60
net.ipv4.tcp_keepalive_intvl = 15
net.ipv4.tcp_keepalive_probes = 5
EOF

echo "[+] Applying sysctl settings immediately"
sudo sysctl --system >/dev/null

echo "[✓] Network tuning applied and will persist across reboot"
