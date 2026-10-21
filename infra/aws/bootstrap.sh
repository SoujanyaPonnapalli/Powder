#!/usr/bin/env bash
# Launch (or reuse) the Powder work VM in us-west-1 and set it up.
#
#   infra/aws/bootstrap.sh            # step 1: create role, SG, key, instance; wait; print deploy key
#   (add the printed key to GitHub as a deploy key WITH write access)
#   infra/aws/bootstrap.sh --finish   # step 2: clone, build, install cron + runner on the VM
#
# Run from your Mac with working AWS credentials (aws sts get-caller-identity).
# Idempotent: re-running reuses everything tagged Project=powder.
# Tear down with infra/aws/teardown.sh.
set -euo pipefail

REGION=${POWDER_REGION:-us-west-1}
INSTANCE_TYPE=${POWDER_INSTANCE_TYPE:-t3.large}
DISK_GB=${POWDER_DISK_GB:-100}
NAME=powder-work-vm
ROLE=powder-work-vm-role
PROFILE=powder-work-vm-profile
SG_NAME=powder-work-vm-ssh
KEY_NAME=powder-work-vm
KEY_FILE=${POWDER_KEY_FILE:-$HOME/.ssh/powder-work-vm.pem}
REPO_SSH=${POWDER_REPO:-git@github.com:SoujanyaPonnapalli/Powder.git}
BRANCH=${POWDER_BRANCH:-exec01-fault-library}
HERE=$(cd "$(dirname "$0")" && pwd)
AWS="aws --region $REGION"

log() { printf '\033[1m[bootstrap]\033[0m %s\n' "$*"; }

instance_id() {
  $AWS ec2 describe-instances --filters "Name=tag:Name,Values=$NAME" \
    "Name=instance-state-name,Values=pending,running,stopping,stopped" \
    --query 'Reservations[].Instances[].InstanceId' --output text | awk '{print $1}'
}

public_ip() {
  $AWS ec2 describe-instances --instance-ids "$1" --query 'Reservations[0].Instances[0].PublicIpAddress' --output text
}

ssh_vm() { ssh -o StrictHostKeyChecking=accept-new -i "$KEY_FILE" "ubuntu@$IP" "$@"; }

if [[ "${1:-}" == "--finish" ]]; then
  ID=$(instance_id); [[ -n "$ID" ]] || { echo "no instance; run without --finish first"; exit 1; }
  IP=$(public_ip "$ID")
  log "finishing setup on $IP"
  scp -o StrictHostKeyChecking=accept-new -i "$KEY_FILE" "$HERE/vm_setup.sh" "ubuntu@$IP:/tmp/vm_setup.sh"
  ssh_vm "bash /tmp/vm_setup.sh '$REPO_SSH' '$BRANCH'"
  log "done. SSH: ssh -i $KEY_FILE ubuntu@$IP"
  log "next: on the VM, run 'az login --use-device-code' once (Azure reference data), then"
  log "      configure Claude Code auth in ~/.powder-runner.env (see infra/aws/README.md) and"
  log "      'sudo systemctl start powder-runner' (or let the timer start it)."
  exit 0
fi

aws sts get-caller-identity > /dev/null || { echo "AWS credentials not working"; exit 1; }

# --- IAM role for the instance (no long-lived keys on the VM) -------------
if ! aws iam get-role --role-name "$ROLE" > /dev/null 2>&1; then
  log "creating IAM role $ROLE"
  aws iam create-role --role-name "$ROLE" --assume-role-policy-document "file://$HERE/trust-ec2.json" \
    --tags Key=Project,Value=powder > /dev/null
fi
aws iam put-role-policy --role-name "$ROLE" --policy-name powder-work-vm \
  --policy-document "file://$HERE/iam-policy.json"
if ! aws iam get-instance-profile --instance-profile-name "$PROFILE" > /dev/null 2>&1; then
  aws iam create-instance-profile --instance-profile-name "$PROFILE" > /dev/null
  aws iam add-role-to-instance-profile --instance-profile-name "$PROFILE" --role-name "$ROLE"
  sleep 10  # IAM propagation
fi

# --- key pair ---------------------------------------------------------------
if [[ ! -f "$KEY_FILE" ]]; then
  log "creating key pair $KEY_NAME -> $KEY_FILE"
  $AWS ec2 delete-key-pair --key-name "$KEY_NAME" > /dev/null 2>&1 || true
  mkdir -p "$(dirname "$KEY_FILE")"
  $AWS ec2 create-key-pair --key-name "$KEY_NAME" --query KeyMaterial --output text > "$KEY_FILE"
  chmod 600 "$KEY_FILE"
fi

# --- security group: SSH from this machine's public IP only -----------------
VPC=$($AWS ec2 describe-vpcs --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text)
SG=$($AWS ec2 describe-security-groups --filters "Name=group-name,Values=$SG_NAME" "Name=vpc-id,Values=$VPC" \
     --query 'SecurityGroups[0].GroupId' --output text)
if [[ "$SG" == "None" ]]; then
  SG=$($AWS ec2 create-security-group --group-name "$SG_NAME" --description "SSH to Powder work VM" \
       --vpc-id "$VPC" --query GroupId --output text)
fi
MYIP=$(curl -fsS https://checkip.amazonaws.com | tr -d '\n')
$AWS ec2 authorize-security-group-ingress --group-id "$SG" --protocol tcp --port 22 --cidr "$MYIP/32" \
  > /dev/null 2>&1 || true

# --- instance ---------------------------------------------------------------
ID=$(instance_id)
if [[ -z "$ID" ]]; then
  AMI=$($AWS ssm get-parameter --name /aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id \
        --query Parameter.Value --output text)
  log "launching $INSTANCE_TYPE ($AMI) in $REGION"
  ID=$($AWS ec2 run-instances --image-id "$AMI" --instance-type "$INSTANCE_TYPE" --key-name "$KEY_NAME" \
       --security-group-ids "$SG" --iam-instance-profile "Name=$PROFILE" \
       --block-device-mappings "DeviceName=/dev/sda1,Ebs={VolumeSize=$DISK_GB,VolumeType=gp3}" \
       --metadata-options HttpTokens=required \
       --user-data "file://$HERE/user-data.sh" \
       --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$NAME},{Key=Project,Value=powder}]" \
       --query 'Instances[0].InstanceId' --output text)
else
  log "reusing $ID"
  $AWS ec2 start-instances --instance-ids "$ID" > /dev/null || true
fi
$AWS ec2 wait instance-running --instance-ids "$ID"
IP=$(public_ip "$ID")
log "instance $ID at $IP; waiting for cloud-init (installs toolchains, ~5-10 min)"
until ssh_vm "test -f /var/lib/cloud/instance/boot-finished" 2> /dev/null; do sleep 15; done
ssh_vm "cloud-init status --wait > /dev/null; cat /var/log/powder-user-data.done 2>/dev/null || true"
log "deploy key (add to GitHub repo -> Settings -> Deploy keys, tick 'Allow write access'):"
ssh_vm "cat ~/.ssh/powder_deploy.pub"
log "then run: infra/aws/bootstrap.sh --finish"
