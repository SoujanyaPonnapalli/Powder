# TiDB deployment and experiments on AWS (exec #7)

Deploy a multi-region TiDB cluster on AWS, run an availability/repartitioning experiment against it, collect the
results, and tear it all down. Design and decisions: `docs/plans/exec-07-tidb-deployment.md`. Library: `powder/deploy/`.

> **These scripts create billable AWS resources.** Every creating command refuses without `--yes` (or
> `POWDER_ALLOW_SPEND=1`) and prints the estimated $/h first. The unattended Claude runner must never set it.
> **Live AWS validation: gates G1-G5 all passed on real AWS (2026-11-04/05): infra deploy + teardown, TiDB install, metric
> self-test (after the client fixes in the plan's §12), full `run_all` smoke pipeline, abort safety**
>
> Client policy: every request has a 5 s SLO; past it the request is lost (never retried) and `powder-load` ends its
> query with `KILL QUERY`, keeping the connection; per-partition connection budgets isolate dead partitions. TiDB runs
> with its default configuration.: the rest is tested against moto, fakes,
> and a real TiDB on one machine, but not yet against real AWS. Start with gate G1 (two nodes, ~$1).

## Prerequisites

* AWS credentials for an account that can launch `i4i.2xlarge` in us-east-1/us-east-2/us-west-1/us-west-2, with enough
  **On-Demand Standard vCPU quota** per region (the smoke run needs ~32 per region; 30 TiKV nodes need ~60+). The
  deploy preflight checks quotas and stops before creating anything. A least-privilege draft policy is in
  `iam-deploy-policy.json` (unvalidated: run G1 under it and fix any `AccessDenied`).
* Python deps: `pip install -e '.[deploy]'` (boto3). `ssh` and `rsync` on this machine. No Go needed locally.
* Submodule for PD source (only needed for exec #9): `git submodule update --init --depth 1 third_party/pd`.

## The five scripts

All take `--out-root` (default `outputs/tidb/`); a deployment's files live in `outputs/tidb/<handle>/`.

| Script | Does | Key flags |
|---|---|---|
| `deploy_infra.py` | VPC per region, full peering mesh, SG (SSH/Grafana only from your IP), key pair, one tagged instance per node, waits for SSH + node bootstrap | `--spec`, `--dry-run`, `--yes`, `--handle`, `--resume`, `--operator-cidr`, `--skip-quota-check` |
| `install_tidb.py` | On the controller: re-verify pinned artifact hashes, install pinned TiUP, `tiup cluster check/deploy/start`, health gate, build and distribute `powder-load` | `--handle` |
| `run_experiment.py` | On an installed deployment: settle gate, prepare table, snapshot PD, probers + open-loop load, collect, `summary.json` | `--handle` (required), `--experiment FILE` or `--self-test`, `--settle-timeout` |
| `teardown_infra.py` | List / delete deployments by tag, in every region, even if local state is lost | `--list`, `<handle>...`, `--all`, `--older-than 6h`, `--verify`, `--dry-run`, `--yes` |
| `run_all.py` | deploy -> install -> experiment(s) -> collect -> teardown (teardown always runs) | `--spec`, `--experiment` (repeatable), `--keep-infra`, `--self-test`, `--yes`, `--dry-run` |

Each deployment gets a random handle (`OrangePanda`) unless `--handle` is given; it is printed first and last, tags every
AWS resource, and names `outputs/tidb/<handle>/`. Experiments **must** name the handle they run on.

```bash
# 1. preview: machines, network, $/h, vCPU per region, modeled cost. Creates nothing.
python3 infra/tidb/deploy_infra.py --spec infra/tidb/specs/smoke.yaml --dry-run

# 2. G1: a tiny deployment, then confirm teardown leaves nothing behind
python3 infra/tidb/deploy_infra.py --spec my-two-node.yaml --yes          # prints the handle
python3 infra/tidb/teardown_infra.py <Handle> --yes && python3 infra/tidb/teardown_infra.py --verify

# 3. everything at once (smoke: 6 TiKV over 3 regions, 10 min, about $6/h while up)
python3 infra/tidb/run_all.py --spec infra/tidb/specs/smoke.yaml --experiment infra/tidb/experiments/smoke.yaml --yes

# 4. metric self-test (gate G3): stops a quorum of known RSMs for 60 s, checks both metrics agree
python3 infra/tidb/run_all.py --spec infra/tidb/specs/smoke.yaml --self-test --yes

# debugging loop: keep the cluster, iterate on experiments, tear down yourself
python3 infra/tidb/run_all.py --spec ... --experiment ... --keep-infra --yes
python3 infra/tidb/run_experiment.py --handle <Handle> --experiment infra/tidb/experiments/smoke.yaml
python3 infra/tidb/teardown_infra.py --list
python3 infra/tidb/teardown_infra.py <Handle> --yes
```

If a run is interrupted or fails, `run_all.py` tears down in `finally` and prints the exact command if anything is left.
`teardown_infra.py --list` shows anything still billing, by name. Optional safety net on the work VM:
`teardown_infra.py --older-than 48h --yes` from cron (not installed by default).

## Specs and experiments

* Deployment spec (`specs/*.yaml`): slots (`NAME: {machine: MACHINE@REGION}`, same grammar as
  `scripts/fault_library/make_trace.py --slot`), control plane, `pd_config` / `tikv_config` / `tidb_config` passthrough,
  `limits`. Unknown keys, duplicate keys, machines the library does not offer in a region are all errors.
* Experiment config (`experiments/*.yaml`): workload (rate, duration, key distribution, arrival profile),
  `request_timeout` (what counts as "timed out"), prober interval, `region_map_interval`, `pd_runtime_config`
  (applied then restored), `injector` (`none`; `trace:<file>` arrives with exec #8).
* **Node-replacement timeout what-if:** re-replication of a dead store is gated by *both* PD
  `schedule.max-store-down-time` (`pd_runtime_config` or `pd_config`) **and** TiKV `raftstore.max-peer-down-duration`
  (`tikv_config`, deploy time only, **not** changeable online). Measured on v8.5.8: with PD at 1 min, replacement still
  started 9 minutes after the kill. The spec warns when only the PD half is set.

## Results (`outputs/tidb/<handle>/experiments/<name>-<utc>/`)

| Path | Content |
|---|---|
| `summary.json` | **the numbers**: prober unavailability and workload unavailability (timeouts / (ok + timeouts)) per partition, per RSM, per client region and overall, with request weights; cause tables (every non-ok request has a cause); repartition metrics; collection problems |
| `workload/<region>/{requests,probes}.jsonl.gz` | raw per-request / per-probe records (`ts, row, outcome, cause, code, layer, latency_ms`) |
| `pd/snapshots/` | PD Region maps + stores every `region_map_interval` (attribution of requests to RSMs; split/merge/move detection) |
| `logs/<node>/` | PD/TiKV/TiDB logs (PD logs give the lossless operator lifecycle) |
| `metrics/` | Prometheus exports (names verified against v8.5.8) |
| `faults/{planned,applied}.jsonl` | scheduled actions (fault injection hook) and what ran |
| `config.json`, `../manifest.json` | handle, spec hash, TiDB/TiUP versions and pins, git sha + dirty flag, stage timings |

`outputs/` is git-ignored; force-add summaries you cite (`git add -f`), never `ssh/` or raw logs.

Two availability numbers, deliberately different: **probes** (every Raft region once a second from every client region;
time-based, comparable to the simulator's per-RSM unavailability) and **workload** (what real requests saw, weighted by
where traffic goes; conflicts excluded from numerator and denominator). `summary.json` has both, plus flags
(`low_coverage`, `late_records`, `other_rate_above_threshold`, `low_sample_rsms`) that say when a number should not be
trusted.

## Powder PD repair oracle (off-cloud prototype)

The opt-in PD patch and loopback-only Python oracle implement **missing/down/offline peer destination
ranking**. PD still filters eligible TiKV stores, builds and throttles the operators, and falls back to
its stock ranking on a timeout, invalid response, model mismatch, or missing score. Raft leader election,
physical node provisioning, post-split moves, and merge decisions are **not** changed by this patch.
The score file is *per deployment*: PD store IDs are not interchangeable with sampled catalogue
machine IDs. `POWDER_ORACLE_URL` and `POWDER_ORACLE_MODEL_ID` must both be set on the **PD process** to enable
advice; without them, the patched binary acts like stock PD. The URL must be an HTTP loopback endpoint
ending in `/v1/recommend`; the client has a 200 ms deadline.

To prepare a score file, supply a JSON `{ "candidates": [{ "candidate_id": "c1", "machine_ids":
["m1", "m2", "m3"], "availability": 0.999, "cost_per_hour": 1.2 }, ...] }` of **already
scored** `PlacementCandidate`s, a JSON `{ "m1": 123, ... }` mapping the exact optimizer physical
machine IDs to this deployment's PD store IDs, and a captured PD `/pd/api/v1/stores` JSON document.
`--model-id` must identify the immutable model/scoring input for that run. Missing candidates are
not scored online: PD falls back to native placement. Reject a store map that pairs a sampled
machine with an unrelated TiKV store; the label consistency check only detects changes to the PD
stores *after* the score file is built. The example numeric values above describe the input shape,
**not** measured results.

```bash
python3 scripts/export_pd_oracle_scores.py --candidates scored.json --machine-map machine-to-store.json \
  --stores pd-stores.json --model-id <immutable-model-id> --output scores.json
python3 infra/tidb/oracle_server.py --scores scores.json  # binds only 127.0.0.1:8650
bash infra/tidb/apply_pd_oracle_patch.sh --check  # pinned clean v8.5.8 submodule
bash infra/tidb/apply_pd_oracle_patch.sh          # opt-in source patch; no deployment
(cd third_party/pd && go test ./pkg/schedule/checker)
# Local build without Dashboard UI assets (for build validation only):
(cd third_party/pd && go build -tags without_dashboard -o /tmp/pd-powder-check ./cmd/pd-server)
```

A plain `go build ./cmd/pd-server` fails on this checkout until the upstream Dashboard UI
assets are generated (`make dashboard-ui`); `without_dashboard` is only the local build
check, not a claim of parity with the pinned production binary.

The patch is stored in the **parent repository** so the upstream submodule commit remains pinned
and reproducible; the apply script refuses a dirty or wrong-revision checkout. Building/patching a
running PD or installing this service is deliberately not automatic. First validate a local TiUP
cluster with at least four eligible TiKV stores for RF=3, then wire a systemd oracle service and
PD environment into the install path before any authorized live experiment. See
`docs/plans/tidb-pd-ilp-integration.md` for the remaining split/merge and benchmarking gates.

## Testing without AWS

```bash
python3 -m venv --system-site-packages ~/.cache/powder/venv && ~/.cache/powder/venv/bin/pip install 'moto[ec2,ssm]' boto3
~/.cache/powder/venv/bin/python -m pytest tests/test_deploy_*.py     # moto tests skip if moto is missing
(cd infra/tidb/loadgen && go vet ./... && go test ./...)             # Go toolchain: see GO_PIN
python3 infra/tidb/local_pipeline_check.py                           # REAL TiDB on this machine, no AWS (~5 min)
```

`local_pipeline_check.py` starts `tiup playground` (3 TiKV), builds `powder-load`, pauses two TiKV processes for 20 s
through the same scheduled-action hook the fault emulator will use, and prints the resulting metrics. It validates the Go
tool and the metric chain on real TiDB; it cannot validate SSH, TiUP deployment or cross-region networking (gates G1-G5).
Real PD/TiKV output captured for the parser tests lives in `tests/fixtures/tidb/` (see its README).

## Known gaps / things to verify at the live gates

* TiUP topology/flags are from TiUP's docs and unit-tested only; `tiup cluster check` advisory failures are saved to
  `install/check.txt` but not fatal. Tune at **G2**.
* PD `/members` JSON shape is assumed from PD's API; the stores/regions shapes are verified on real output.
* `iam-deploy-policy.json` is a draft; multi-resource EC2 actions may need extra statements.
* Exporter versions (`node_exporter`, `blackbox_exporter`, `alertmanager`) are chosen by TiUP; `install/components.txt`
  records them so they can be pinned in `third_party/tidb/VERSIONS`.
* `powder-load` caps in-flight requests (`--max-inflight`); hitting it is recorded as `other/loadgen_saturated`, never as
  unavailability.
