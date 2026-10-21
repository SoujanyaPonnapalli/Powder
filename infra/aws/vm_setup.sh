#!/usr/bin/env bash
# Second-stage setup, run on the VM by `bootstrap.sh --finish` (as ubuntu).
#   bash vm_setup.sh <repo-ssh-url> <branch>
set -euo pipefail
REPO=$1
BRANCH=$2
DIR=$HOME/Powder

if [[ ! -d "$DIR/.git" ]]; then
  git clone "$REPO" "$DIR"
fi
cd "$DIR"
git fetch origin
git checkout "$BRANCH" 2>/dev/null || git checkout -b "$BRANCH" "origin/$BRANCH"
git config user.name "Powder work VM"
git config user.email "powder-work-vm@users.noreply.github.com"

# Python environment (nbstripout filter needs it)
python3 -m venv "$DIR/.venv-vm"
"$DIR/.venv-vm/bin/pip" install -q --upgrade pip
"$DIR/.venv-vm/bin/pip" install -q -e ".[dev]" pdfplumber opencv-python-headless
git config filter.nbstripout.clean "$DIR/.venv-vm/bin/python -m nbstripout"
git config filter.nbstripout.smudge cat
git config filter.nbstripout.required true

# Rust engine
source "$HOME/.cargo/env"
cargo build --release --manifest-path rust/Cargo.toml

# Data refresh cron + Claude runner (systemd)
crontab infra/aws/crontab
sudo cp infra/aws/powder-runner.service infra/aws/powder-runner.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable powder-runner.timer
[[ -f "$HOME/.powder-runner.env" ]] || cp infra/aws/powder-runner.env.example "$HOME/.powder-runner.env"

# First data pull and build right away (prices need no credentials)
PYTHON="$DIR/.venv-vm/bin/python" bash infra/aws/refresh_data.sh || echo "first refresh failed; see logs"
echo "vm_setup complete"
