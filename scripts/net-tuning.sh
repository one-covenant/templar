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
net.core.default
