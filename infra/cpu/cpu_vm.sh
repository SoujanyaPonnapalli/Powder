#!/usr/bin/env bash
# One CPU VM for Monte Carlo / ILP experiments (pattern of infra/gpu/gpu_vm.sh, without the GPU).
#
#   infra/cpu/cpu_vm.sh up        # tagged VPC/subnet/IGW/SG/key + one c7i.8xlarge (32 vCPU, 64 GB)
#   infra/cpu/cpu_vm.sh setup     # rsync code + fault library, rustup + powder-mc build, venv, Gurobi licence
#   infra/cpu/cpu_vm.sh run CMD   # run CMD in the repo on the VM (venv active)
#   infra/cpu/cpu_vm.sh fetch SRC # copy powder-repo/SRC back to the same path locally
#   infra/cpu/cpu_vm.sh down      # terminate + delete every resource, then verify nothing is left
#   infra/cpu/cpu_vm.sh status
#
# Billable (c7i.8xlarge ~$1.4/h on demand). Everything is tagged Project=powder and Handle=$HANDLE. The
# instance shuts itself down (and so terminates) after $POWDER_CPU_MAX_MIN minutes as a backstop. Never leave
# it running unattended; always finish with `down`. ~/gurobi.lic (WLS) is copied to the VM for gurobipy and
# goes away with the instance.
set -euo pipefail

REGION=${POWDER_CPU_REGION:-us-west-2}
TYPE=${POWDER_CPU_TYPE:-c7i.8xlarge}
HANDLE=${POWDER_CPU_HANDLE:-cpu-mc}
MAX_MIN=${POWDER_CPU_MAX_MIN:-120}
STATE=${POWDER_CPU_STATE:-$HOME/.cache/powder/$HANDLE.env}
KEY=$HOME/.ssh/powder-$HANDLE
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
AMI_NAME="ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-amd64-server-*"
CANONICAL=099720109477

ec2() { aws ec2 --region "$REGION" "$@"; }
tags() { echo "ResourceType=$1,Tags=[{Key=Project,Value=powder},{Key=Handle,Value=$HANDLE},{Key=Name,Value=powder-$HANDLE}]"; }
save() { mkdir -p "$(dirname "$STATE")"; echo "$1=$2" >> "$STATE"; }
load() { [[ -f $STATE ]] && source "$STATE" || true; }
ip() { load; ec2 describe-instances --instance-ids "$INSTANCE" --query 'Reservations[0].Instances[0].PublicIpAddress' --output text; }
sshv() { ssh -i "$KEY" -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 "ubuntu@$(ip)" "$@"; }

up() {
  load
  if [[ -n ${INSTANCE:-} ]]; then echo "already up: $INSTANCE ($(ip))"; return; fi
  if [[ -s $STATE ]]; then echo "partial resources in $STATE: run '$0 down' first" >&2; exit 1; fi
  : > "$STATE"
  trap 'echo "up failed; clean up with: $0 down" >&2' ERR
  local az my_ip ami
  az=${POWDER_CPU_AZ:-$(ec2 describe-instance-type-offerings --location-type availability-zone \
        --filters Name=instance-type,Values="$TYPE" --query 'InstanceTypeOfferings[0].Location' --output text)}
  my_ip=$(curl -s https://checkip.amazonaws.com)
  ami=$(ec2 describe-images --owners "$CANONICAL" --filters "Name=name,Values=$AMI_NAME" \
        --query 'sort_by(Images,&CreationDate)[-1].ImageId' --output text)
  echo "region $REGION az $az type $TYPE ami $ami ssh-from $my_ip"

  VPC=$(ec2 create-vpc --cidr-block 10.43.0.0/16 --tag-specifications "$(tags vpc)" --query Vpc.VpcId --output text)
  save VPC "$VPC"
  ec2 modify-vpc-attribute --vpc-id "$VPC" --enable-dns-hostnames
  SUBNET=$(ec2 create-subnet --vpc-id "$VPC" --cidr-block 10.43.1.0/24 --availability-zone "$az" \
           --tag-specifications "$(tags subnet)" --query Subnet.SubnetId --output text)
  save SUBNET "$SUBNET"
  ec2 modify-subnet-attribute --subnet-id "$SUBNET" --map-public-ip-on-launch
  IGW=$(ec2 create-internet-gateway --tag-specifications "$(tags internet-gateway)" \
        --query InternetGateway.InternetGatewayId --output text)
  save IGW "$IGW"
  ec2 attach-internet-gateway --internet-gateway-id "$IGW" --vpc-id "$VPC"
  RTB=$(ec2 create-route-table --vpc-id "$VPC" --tag-specifications "$(tags route-table)" \
        --query RouteTable.RouteTableId --output text)
  save RTB "$RTB"
  ec2 create-route --route-table-id "$RTB" --destination-cidr-block 0.0.0.0/0 --gateway-id "$IGW" >/dev/null
  RTB_ASSOC=$(ec2 associate-route-table --route-table-id "$RTB" --subnet-id "$SUBNET" --query AssociationId --output text)
  save RTB_ASSOC "$RTB_ASSOC"
  SG=$(ec2 create-security-group --group-name "powder-$HANDLE" --description "powder $HANDLE ssh" --vpc-id "$VPC" \
       --tag-specifications "$(tags security-group)" --query GroupId --output text)
  save SG "$SG"
  ec2 authorize-security-group-ingress --group-id "$SG" --protocol tcp --port 22 --cidr "$my_ip/32" >/dev/null
  [[ -f $KEY ]] || ssh-keygen -q -t ed25519 -N "" -f "$KEY"
  ec2 import-key-pair --key-name "powder-$HANDLE" --public-key-material "fileb://$KEY.pub" \
      --tag-specifications "$(tags key-pair)" >/dev/null
  save KEYPAIR "powder-$HANDLE"

  local ud
  ud=$(printf '#!/bin/bash\nshutdown -h +%s "powder backstop"\n' "$MAX_MIN")
  INSTANCE=$(ec2 run-instances --image-id "$ami" --instance-type "$TYPE" --key-name "powder-$HANDLE" \
      --subnet-id "$SUBNET" --security-group-ids "$SG" --user-data "$ud" \
      --instance-initiated-shutdown-behavior terminate \
      --metadata-options HttpTokens=required,HttpEndpoint=enabled \
      --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=40,VolumeType=gp3,DeleteOnTermination=true}' \
      --tag-specifications "$(tags instance)" "$(tags volume)" "$(tags network-interface)" --query 'Instances[0].InstanceId' --output text)
  save INSTANCE "$INSTANCE"
  ec2 wait instance-running --instance-ids "$INSTANCE"
  echo "instance $INSTANCE at $(ip); self-terminates after $MAX_MIN min"
  for _ in $(seq 60); do sshv true 2>/dev/null && break; sleep 5; done
  sshv nproc
}

setup() {
  load
  local host; host=$(ip)
  rsync -az -e "ssh -i $KEY -o StrictHostKeyChecking=accept-new" --relative \
    "$ROOT/./powder" "$ROOT/./scripts" "$ROOT/./rust" "$ROOT/./data/fault_library" "$ROOT/./pyproject.toml" \
    "ubuntu@$host:powder-repo/" --exclude __pycache__ --exclude target --exclude '*.so'
  scp -i "$KEY" -q "$HOME/gurobi.lic" "ubuntu@$host:gurobi.lic"
  sshv 'set -e; sudo apt-get -qq update; sudo DEBIAN_FRONTEND=noninteractive apt-get -qq install -y build-essential python3-venv pkg-config >/dev/null
        curl -sSf https://sh.rustup.rs | sh -s -- -y -q --profile minimal >/dev/null; . "$HOME/.cargo/env"
        cd powder-repo; cargo build -q --release --manifest-path rust/Cargo.toml
        python3 -m venv .venv; . .venv/bin/activate; pip -q install -U pip; pip -q install numpy scipy gurobipy==13.0.3
        python -c "import gurobipy as g; m=g.Model(); print(\"gurobi\", g.gurobi.version())"; ls -la rust/target/release/powder-mc; nproc'
}

run() { sshv "cd powder-repo && . .venv/bin/activate && $*"; }

fetch() {
  load
  local src=$1
  mkdir -p "$ROOT/$(dirname "$src")"
  rsync -az -e "ssh -i $KEY" "ubuntu@$(ip):powder-repo/$src" "$ROOT/$(dirname "$src")/"
}

down() {
  load
  if [[ -n ${INSTANCE:-} ]]; then
    ec2 terminate-instances --instance-ids "$INSTANCE" >/dev/null || true
    ec2 wait instance-terminated --instance-ids "$INSTANCE" || true
  fi
  [[ -n ${KEYPAIR:-} ]] && ec2 delete-key-pair --key-name "$KEYPAIR" || true
  [[ -n ${SG:-} ]] && ec2 delete-security-group --group-id "$SG" || true
  [[ -n ${RTB_ASSOC:-} ]] && ec2 disassociate-route-table --association-id "$RTB_ASSOC" || true
  [[ -n ${RTB:-} ]] && ec2 delete-route-table --route-table-id "$RTB" || true
  if [[ -n ${IGW:-} ]]; then
    ec2 detach-internet-gateway --internet-gateway-id "$IGW" --vpc-id "$VPC" || true
    ec2 delete-internet-gateway --internet-gateway-id "$IGW" || true
  fi
  [[ -n ${SUBNET:-} ]] && ec2 delete-subnet --subnet-id "$SUBNET" || true
  [[ -n ${VPC:-} ]] && ec2 delete-vpc --vpc-id "$VPC" || true
  rm -f "$STATE"
  status
}

status() {
  echo "live resources tagged Handle=$HANDLE in $REGION:"
  ec2 describe-instances --filters "Name=tag:Handle,Values=$HANDLE" \
      "Name=instance-state-name,Values=pending,running,stopping,stopped,shutting-down" \
      --query 'Reservations[].Instances[].[InstanceId,State.Name,PublicIpAddress]' --output text
  ec2 describe-vpcs --filters "Name=tag:Handle,Values=$HANDLE" --query 'Vpcs[].VpcId' --output text
  ec2 describe-key-pairs --filters "Name=tag:Handle,Values=$HANDLE" --query 'KeyPairs[].KeyName' --output text
}

cmd=${1:-status}; shift || true
case $cmd in
  up|setup|down|status) "$cmd" ;;
  run) run "$@" ;;
  fetch) fetch "$@" ;;
  ssh) sshv "$@" ;;
  *) echo "usage: $0 up|setup|run CMD|ssh|fetch SRC|down|status" >&2; exit 2 ;;
esac
