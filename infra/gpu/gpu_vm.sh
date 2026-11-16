#!/usr/bin/env bash
# One GPU VM for the large Markov solves (docs/plans/figs-01-whatif-modeling-scale-sensitivity.md, Figure 2).
#
#   infra/gpu/gpu_vm.sh up        # tagged VPC/subnet/IGW/SG/key + one g5.4xlarge (A10G 24 GB, 64 GB RAM)
#   infra/gpu/gpu_vm.sh setup     # rsync the code + RSM configs, venv with numpy/scipy/cupy
#   infra/gpu/gpu_vm.sh run CMD   # run CMD in the repo on the VM (venv active)
#   infra/gpu/gpu_vm.sh fetch     # copy outputs/figs/modeling/markov back
#   infra/gpu/gpu_vm.sh down      # terminate + delete every resource, then verify nothing is left
#   infra/gpu/gpu_vm.sh status
#
# POWDER_GPU_AZ picks the zone (on InsufficientInstanceCapacity: down, then retry another zone).
# Billable (~$1.6/h). Everything is tagged Project=powder (required by the work VM's powder-deploy policy) and
# Handle=$HANDLE. The instance shuts itself down (and so terminates) after $POWDER_GPU_MAX_MIN minutes as a
# backstop. Never leave it running unattended; always finish with `down`.
set -euo pipefail

REGION=${POWDER_GPU_REGION:-us-west-2}
TYPE=${POWDER_GPU_TYPE:-g5.4xlarge}
HANDLE=${POWDER_GPU_HANDLE:-gpu-markov}
MAX_MIN=${POWDER_GPU_MAX_MIN:-240}
STATE=${POWDER_GPU_STATE:-$HOME/.cache/powder/$HANDLE.env}
KEY=$HOME/.ssh/powder-$HANDLE
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
AMI_NAME="Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04)*"

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
  az=${POWDER_GPU_AZ:-$(ec2 describe-instance-type-offerings --location-type availability-zone \
        --filters Name=instance-type,Values="$TYPE" --query 'InstanceTypeOfferings[0].Location' --output text)}
  my_ip=$(curl -s https://checkip.amazonaws.com)
  ami=${POWDER_GPU_AMI:-$(ec2 describe-images --owners amazon --filters "Name=name,Values=$AMI_NAME" \
        --query 'sort_by(Images,&CreationDate)[-1].ImageId' --output text)}
  echo "region $REGION az $az type $TYPE ami $ami ssh-from $my_ip"

  VPC=$(ec2 create-vpc --cidr-block 10.42.0.0/16 --tag-specifications "$(tags vpc)" --query Vpc.VpcId --output text)
  save VPC "$VPC"
  ec2 modify-vpc-attribute --vpc-id "$VPC" --enable-dns-hostnames
  SUBNET=$(ec2 create-subnet --vpc-id "$VPC" --cidr-block 10.42.1.0/24 --availability-zone "$az" \
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
      --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=150,VolumeType=gp3,DeleteOnTermination=true}' \
      --tag-specifications "$(tags instance)" "$(tags volume)" "$(tags network-interface)" --query 'Instances[0].InstanceId' --output text)
  save INSTANCE "$INSTANCE"
  ec2 wait instance-running --instance-ids "$INSTANCE"
  echo "instance $INSTANCE at $(ip); self-terminates after $MAX_MIN min"
  for _ in $(seq 60); do sshv true 2>/dev/null && break; sleep 5; done
  sshv 'nvidia-smi -L 2>/dev/null || nproc'
}

setup_cpu() {
  # CPU benchmark VM (e.g. POWDER_GPU_TYPE=c7i.8xlarge with an Ubuntu 24.04 AMI): the committed tree via
  # git archive (uncommitted local edits never reach the VM), Rust engine build, Python deps, Gurobi WLS license.
  load
  local host; host=$(ip)
  (cd "$ROOT" && git archive --format=tar HEAD) | sshv 'rm -rf powder-repo && mkdir -p powder-repo && tar -x -C powder-repo'
  sshv "cd powder-repo && echo $(cd "$ROOT" && git rev-parse --short HEAD) > COMMIT"
  scp -i "$KEY" -o StrictHostKeyChecking=accept-new "$HOME/gurobi.lic" "ubuntu@$host:gurobi.lic"
  sshv 'set -e; sudo apt-get -qq update; sudo DEBIAN_FRONTEND=noninteractive apt-get -qq install -y build-essential python3-venv pkg-config >/dev/null
        command -v cargo >/dev/null || (curl -sSf https://sh.rustup.rs | sh -s -- -y -q --profile minimal)
        . ~/.cargo/env; cd powder-repo; cargo build --release -q --manifest-path rust/Cargo.toml
        python3 -m venv .venv; . .venv/bin/activate; pip -q install -U pip
        pip -q install numpy scipy pandas pyyaml plotly gurobipy==13.0.3
        python -c "import gurobipy as g; m=g.Model(); print(\"gurobi ok\", g.gurobi.version())"
        ls -la rust/target/release/powder-mc; nproc'
}

fetch_bench() {
  load
  mkdir -p "$ROOT/outputs/bench"
  rsync -az -e "ssh -i $KEY" --exclude cache "ubuntu@$(ip):powder-repo/${POWDER_FETCH_DIR:-outputs/bench}/" "$ROOT/${POWDER_FETCH_DIR:-outputs/bench}/"
}

setup() {
  load
  local host; host=$(ip)
  rsync -az -e "ssh -i $KEY -o StrictHostKeyChecking=accept-new" --relative \
    "$ROOT/./powder" "$ROOT/./scripts/figs" "$ROOT/./outputs/figs/modeling/rsms.json" \
    "$ROOT/./outputs/figs/modeling/rsms.pkl" "$ROOT/./pyproject.toml" "ubuntu@$host:powder-repo/" \
    --exclude __pycache__ --exclude '*.so'
  sshv 'set -e; cd powder-repo; python3 -m venv .venv; . .venv/bin/activate; pip -q install -U pip;
        pip -q install numpy scipy pandas pyyaml plotly cupy-cuda12x;
        python -c "import cupy, sys; print(\"cupy\", cupy.__version__, cupy.cuda.runtime.getDeviceCount(), \"GPU\"); print(sys.version)"'
}

run() { sshv "cd powder-repo && . .venv/bin/activate && $*"; }

fetch() {
  load
  mkdir -p "$ROOT/outputs/figs/modeling"
  rsync -az -e "ssh -i $KEY" "ubuntu@$(ip):powder-repo/outputs/figs/modeling/markov" "$ROOT/outputs/figs/modeling/"
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
  up|setup|fetch|down|status) "$cmd" ;;
  setup-cpu) setup_cpu ;;
  fetch-bench) fetch_bench ;;
  run) run "$@" ;;
  ssh) sshv "$@" ;;
  *) echo "usage: $0 up|setup|run CMD|ssh|fetch|down|status" >&2; exit 2 ;;
esac
