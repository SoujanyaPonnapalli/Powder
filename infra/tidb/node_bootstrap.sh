#!/usr/bin/env bash
# Powder node bootstrap: EC2 user-data, runs once as root on every node of a run (exec #7, plan §5).
#
#  * mounts the instance-store NVMe at /data (ext4, nodelalloc,noatime as TiDB recommends); nodes without
#    instance store (control plane) just get a /data directory on the root volume
#  * OS prerequisites TiUP checks for: swap off, transparent hugepages off, sysctl, file limits, chrony
#  * tools the failure emulator and collectors use: iptables, tc (iproute2), jq, rsync
#  * writes /var/lib/powder/node.json and, last of all, /var/lib/powder/bootstrap.done (deploy waits for it)
#
# Instance-store caveat: stop/start or terminate loses /data; a plain `reboot` keeps it.
set -euo pipefail
exec > >(tee -a /var/log/powder-bootstrap.log) 2>&1
echo "[bootstrap] start $(date -u +%FT%TZ)"
export DEBIAN_FRONTEND=noninteractive
mkdir -p /var/lib/powder

retry() { local n=0; until "$@"; do n=$((n + 1)); [[ $n -ge 5 ]] && return 1; sleep $((n * 3)); done; }

retry apt-get update -y
retry apt-get install -y chrony iptables iproute2 jq rsync curl numactl sysstat

# ---- swap, THP, sysctl, limits ---------------------------------------------------------------
swapoff -a || true
sed -i '/\sswap\s/ s/^/#/' /etc/fstab || true

cat > /etc/systemd/system/powder-thp.service <<'UNIT'
[Unit]
Description=Disable transparent hugepages (TiDB prerequisite)
After=sysinit.target
[Service]
Type=oneshot
ExecStart=/bin/sh -c 'echo never > /sys/kernel/mm/transparent_hugepage/enabled; echo never > /sys/kernel/mm/transparent_hugepage/defrag'
[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
systemctl enable --now powder-thp.service

cat > /etc/sysctl.d/99-powder.conf <<'SYSCTL'
vm.swappiness = 0
fs.file-max = 1000000
net.core.somaxconn = 32768
net.ipv4.tcp_syncookies = 0
net.ipv4.ip_local_port_range = 10000 65000
SYSCTL
sysctl --system >/dev/null

cat > /etc/security/limits.d/99-powder.conf <<'LIMITS'
*    soft nofile 1000000
*    hard nofile 1000000
*    soft stack  10240
root soft nofile 1000000
root hard nofile 1000000
LIMITS

systemctl enable --now chrony || true

# ---- /data on instance-store NVMe ----------------------------------------------------------
dev=""
for d in /dev/disk/by-id/nvme-Amazon_EC2_NVMe_Instance_Storage_*; do
  [[ -e "$d" && "$d" != *-part* ]] && { dev=$d; break; }
done
if [[ -n "$dev" ]]; then
  echo "[bootstrap] instance store: $dev"
  mkfs.ext4 -F -q "$dev"
  mkdir -p /data
  mount -o defaults,nodelalloc,noatime "$dev" /data
  uuid=$(blkid -s UUID -o value "$dev")
  echo "UUID=$uuid /data ext4 defaults,nodelalloc,noatime,nofail 0 2" >> /etc/fstab
else
  echo "[bootstrap] no instance store; /data is a directory on the root volume"
  mkdir -p /data
fi
mkdir -p /data/tikv /data/powder

data_bytes=$(df -B1 --output=size /data | tail -1 | tr -d ' ')
cat > /var/lib/powder/node.json <<JSON
{"instance_store_device": "${dev}", "data_bytes": ${data_bytes}, "bootstrapped_at": "$(date -u +%FT%TZ)"}
JSON

touch /var/lib/powder/bootstrap.done
echo "[bootstrap] done $(date -u +%FT%TZ)"
