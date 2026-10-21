#!/usr/bin/env bash
# Terminate the Powder work VM and (optionally) delete its role, key and SG.
#   infra/aws/teardown.sh            # terminate the instance only
#   infra/aws/teardown.sh --all      # also delete security group, key pair, IAM role/profile
set -euo pipefail
REGION=${POWDER_REGION:-us-west-1}
AWS="aws --region $REGION"
ID=$($AWS ec2 describe-instances --filters Name=tag:Name,Values=powder-work-vm \
     "Name=instance-state-name,Values=pending,running,stopping,stopped" \
     --query 'Reservations[].Instances[].InstanceId' --output text)
if [[ -n "$ID" ]]; then
  $AWS ec2 terminate-instances --instance-ids $ID > /dev/null
  $AWS ec2 wait instance-terminated --instance-ids $ID
  echo "terminated $ID"
fi
if [[ "${1:-}" == "--all" ]]; then
  SG=$($AWS ec2 describe-security-groups --filters Name=group-name,Values=powder-work-vm-ssh \
       --query 'SecurityGroups[0].GroupId' --output text)
  [[ "$SG" != "None" ]] && $AWS ec2 delete-security-group --group-id "$SG" && echo "deleted SG $SG"
  $AWS ec2 delete-key-pair --key-name powder-work-vm && echo "deleted key pair (local ~/.ssh/powder-work-vm.pem kept)"
  aws iam remove-role-from-instance-profile --instance-profile-name powder-work-vm-profile \
    --role-name powder-work-vm-role 2>/dev/null || true
  aws iam delete-instance-profile --instance-profile-name powder-work-vm-profile 2>/dev/null || true
  aws iam delete-role-policy --role-name powder-work-vm-role --policy-name powder-work-vm 2>/dev/null || true
  aws iam delete-role --role-name powder-work-vm-role 2>/dev/null && echo "deleted IAM role" || true
fi
