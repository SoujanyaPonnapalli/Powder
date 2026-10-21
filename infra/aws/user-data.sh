#!/bin/bash
# cloud-init user data for the Powder work VM (Ubuntu 24.04). Runs once as root.
set -euxo pipefail
export DEBIAN_FRONTEND=noninteractive

apt-get update
apt-get install -y --no-install-recommends \
  build-essential git jq unzip curl ca-certificates tmux pkg-config libssl-dev \
  python3 python3-venv python3-pip python3-dev poppler-utils libgl1

# AWS CLI v2
curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip
unzip -q /tmp/awscliv2.zip -d /tmp && /tmp/aws/install --update

# Azure CLI (for the optional Azure reference fetches)
curl -fsSL https://aka.ms/InstallAzureCLIDeb | bash

# Node.js LTS + Claude Code CLI
curl -fsSL https://deb.nodesource.com/setup_lts.x | bash -
apt-get install -y nodejs
npm install -g @anthropic-ai/claude-code

# Rust toolchain for the ubuntu user (powder-mc)
sudo -u ubuntu bash -c 'curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal'

# Deploy key for GitHub (printed by bootstrap.sh; add it to the repo with write access)
sudo -u ubuntu bash -c '
  mkdir -p ~/.ssh && chmod 700 ~/.ssh
  [ -f ~/.ssh/powder_deploy ] || ssh-keygen -t ed25519 -N "" -C "powder-work-vm" -f ~/.ssh/powder_deploy
  cat > ~/.ssh/config <<EOF
Host github.com
  IdentityFile ~/.ssh/powder_deploy
  IdentitiesOnly yes
EOF
  ssh-keyscan github.com >> ~/.ssh/known_hosts 2>/dev/null
'
date -u > /var/log/powder-user-data.done
