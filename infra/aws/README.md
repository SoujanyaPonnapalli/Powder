# Powder work VM (AWS us-west-1)

A small always-on VM that keeps the fault library's data fresh and lets an
unattended Claude Code session continue the exec #1 checklist, so work does
not depend on a laptop.

## What it does

* **Data refresh (cron, UTC):** daily Spot Advisor snapshot (its history
  cannot be fetched later), weekly on-demand + transfer prices (public AWS
  Price List Bulk API), weekly 90-day spot price history (instance role). Each
  refresh rebuilds the library (`scripts/fault_library/build_all.sh`, tests
  included), then commits and pushes.
* **Claude Code runner (systemd timer, every 6 h):** checks out
  `exec01-fault-library-vm` (created from `exec01-fault-library`), does the
  first unchecked item of §8 in `docs/plans/exec-01-canonical-failure-data.md`,
  runs the build/tests, ticks the item, commits, and pushes. You review and
  merge.

## One-time setup (from your Mac, ~15 min)

```bash
aws sts get-caller-identity            # credentials must work
infra/aws/bootstrap.sh                 # role, SG (SSH from your IP), key, t3.large Ubuntu 24.04, cloud-init
# add the printed public key to GitHub: repo -> Settings -> Deploy keys -> "Allow write access"
infra/aws/bootstrap.sh --finish        # clone, venv, cargo build, cron, runner timer, first data refresh
ssh -i ~/.ssh/powder-work-vm.pem ubuntu@<ip>
nano ~/.powder-runner.env              # pick Claude auth: ANTHROPIC_API_KEY, or Bedrock via the instance role
az login --use-device-code             # optional: Azure reference data (coordinates, ARG spot)
sudo systemctl start powder-runner     # first session now (the timer repeats every 6 h)
```

The repo (`SoujanyaPonnapalli/Powder`) is not under your account, so the owner
may need to add the deploy key.

## Cost and security

* `t3.large` with 100 GB gp3, on 24/7 in us-west-1: roughly $70-80/month for
  the instance plus about $8/month for the disk (check current prices). Use
  `POWDER_INSTANCE_TYPE` to change it.
* No long-lived AWS keys on the VM: the instance role allows read-only EC2
  pricing/spot/health calls and (optionally) Bedrock model invocation. See
  `iam-policy.json`.
* SSH only from the public IP you bootstrapped from; IMDSv2 required.
* The deploy key can push to the repo. The runner never pushes to `main`.

## Operations

* Logs: `~/powder-logs/` (`refresh-*.log`, `runner-*.log`).
* Status: `systemctl list-timers powder-runner.timer`, `crontab -l`.
* Pause the runner: `sudo systemctl disable --now powder-runner.timer`.
* Tear down: `infra/aws/teardown.sh` (instance) or `infra/aws/teardown.sh --all`
  (also SG, key pair, IAM role).

## Files

| File | Role |
|---|---|
| `bootstrap.sh` | create/reuse the VM (run on your Mac) |
| `user-data.sh` | cloud-init: toolchains, AWS/Azure CLIs, Node + Claude Code, Rust, deploy key |
| `vm_setup.sh` | second stage on the VM: clone, venv, cargo build, cron, systemd |
| `refresh_data.sh`, `crontab` | scheduled data refresh + rebuild + push |
| `run_claude.sh`, `runner_prompt.md`, `powder-runner.{service,timer}` | unattended work sessions |
| `powder-runner.env.example` | Claude auth template (copied to `~/.powder-runner.env`) |
| `iam-policy.json`, `trust-ec2.json` | instance role |
| `teardown.sh` | clean up |
