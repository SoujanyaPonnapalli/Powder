# Plan: Execution Order #7 — TiDB in the repo + deployable/experiment scripts on AWS

Scope: October Goals, execution order #7 ("Add TiDB to the repo and run a test deployment on AWS"),
Goal 6.2 ("Make TiDB deployment script"), and the plumbing that #8 (failure emulator), #9 (PD queries
the ILP) and #10 (timing benchmarks) will sit on.

Status: **v3 (2026-11-03): steps 1-7 implemented and committed on `exec07-tidb-deployment`; live AWS gates G1-G5 NOT yet
run** (the machine this was built on has no AWS credentials). Everything is tested against moto (emulated EC2), fakes, the
real TiDB output captured in `tests/fixtures/tidb/`, and a real TiDB on one machine (`infra/tidb/local_pipeline_check.py`).
Section 14 lists what changed from the earlier drafts and what the first live runs must verify. Earlier revisions: v2.2
repartitioning metrics, experiments name their handle, simplified teardown; v2.1 random handles, per-RSM attribution; v2
live-AWS-first testing, workload metric.

## 0. Decisions proposed (change any before implementation starts)

| Topic | Proposal | Why |
|---|---|---|
| TiDB version | One pinned **LTS release** (`v8.5.8` at time of writing; confirm at step 1) for TiDB, TiKV, PD, TiUP. Recorded in `third_party/tidb/VERSIONS` and in every run's `manifest.json`. | Results must be reproducible; PD patches in #9 must rebase onto a known tag. |
| "Copying TiDB" | **Binaries from the official release tarballs (via the TiUP mirror, hash-pinned) now; PD as a pinned fork submodule `third_party/pd`** (not built or modified until #9). TiKV and TiDB stay unmodified binaries; TiKV source only if #10 needs extra instrumentation. | PD is the only component we will change (#9). Vendoring all three Go/Rust trees costs GBs and build time with no benefit today. |
| Cluster manager | **TiUP cluster** (`tiup cluster deploy/start/scale-out/scale-in/patch`). | Native labels, scale-out/in (= replacement), and `tiup cluster patch` swaps in our PD binary for #9 without a new deploy path. |
| Language | **Python 3 (boto3 + PyYAML) for orchestration**, bash only on-node. Library in `powder/deploy/`, thin CLIs in `infra/tidb/`. | VPC-peering meshes, tagging, retries and teardown ordering are unmaintainable in bash; Python can import `powder.faults` (spec, trace, prices) directly. `boto3` becomes an optional dependency (`pyproject` extra `deploy`). |
| Regions | The 4 US commercial regions of exec #1 (`powder.faults.regions.AWS_US_REGIONS`); any subset per run. | Matches the library and ILP. |
| Instance types | TiKV nodes: **`i4i.2xlarge`**, local NVMe (exec #1 §3.5). Control plane (controller, PD, TiDB, monitoring, load generators): **separate smaller type, not fault-injected** (default `m6i.xlarge`). | Faults are injected into the nodes the ILP chooses; the control plane must not die from our own injection. Open question Q1: PD is itself a Raft group — do we ever fault it? |
| Ingress | Public IPs on all nodes for egress (no NAT gateway cost); ingress only: SSH from the operator's IP + all ports between nodes by CIDR. TiUP runs **from a controller VM inside the mesh**. | One SSH entry point; no per-region NAT cost; TiUP needs SSH to every node. |
| Run isolation | Every deployment ("run") gets a **random handle** such as `OrangePanda` (or an explicit `--handle`). A handle names a **deployment**; experiments run *on* a handle. Every resource is tagged `Project=powder`, `Run=<handle>`, `Owner=<user>`, `CreatedAt=<utc>`. Listing and teardown work from tags alone. | Teardown must survive a lost state file and a half-failed deploy, and you must be able to see and kill leftover deployments by name. |

## 1. Goals / non-goals

Goals:
1. Pinned TiDB in the repo with a reproducible fetch.
2. Four composable scripts (deploy infra, teardown infra, install TiDB, run experiment + collect) and one
   orchestrator that runs all of them with guaranteed teardown.
3. A first real smoke test on AWS (exec #7 done-criterion).
4. Interfaces that do not change when fault injection (#8) and PD↔ILP (#9) land (§8).

Non-goals (this step): fault injection, a modified PD, ILP-driven placement, calibrated timing (#10),
the full diurnal/hot-partition workload set (Goal 6.3), analysis/graph scripts. Each has a reserved hook.

## 2. Repository layout

```
third_party/
  tidb/VERSIONS                 # tidb/tikv/pd/tiup versions + sha256 of every tarball
  pd/                           # git submodule (fork of tikv/pd) pinned to the VERSIONS tag; unused until #9
powder/deploy/                  # library (importable, unit-tested, no side effects at import)
  spec.py                       # ClusterSpec: slots -> machine class @ region; roles; validation; JSON <-> YAML
  state.py                      # RunState (state.json): instances, IPs, VPC ids, key path, stage status
  names.py                      # AdjectiveNoun handle generator + validation + uniqueness check
  pricing.py                    # bill estimate vs modeled cost; vCPU-per-region quota arithmetic
  netplan.py                    # pure plan: per-region CIDRs, subnets, peering mesh, routes, SG rules
  aws_net.py                    # (step 3) executes the netplan: VPCs, subnets, IGW, routes, peering, SGs
  plan.py                       # `python -m powder.deploy.plan SPEC`: dry-run preview (no AWS)
  outcomes.py, error_classes.json   # ok/timeout/conflict/other + cause; code table shared with powder-load
  regionmap.py                  # normalized PD Region map (+ PD JSON loader), row keys, partition buckets
  availability.py               # workload + probe aggregators: per partition / RSM / client region / overall
  repartition.py                # snapshot-diff events, episodes, under-replication, GB moved, flag index
  pdlog.py                      # PD operator-lifecycle log parser (formats captured from a real v8.5.8 cluster)
  aws_nodes.py                  # AMI lookup, key pairs, launch/terminate, user-data, quotas preflight
  teardown.py                   # tag-based discovery + dependency-ordered deletion + verify
  topology.py                   # ClusterSpec + RunState -> TiUP topology.yaml (labels incl. machine_class)
  remote.py                     # ssh/scp/rsync helpers (via controller), retries, timeouts
  workloads/                    # base.py (interface), smoke.py ; later diurnal/hot/shifting are configs
  prober.py                     # per-Raft-region availability prober (runs on load-gen nodes)
  loadgen/                      # powder-load: open-loop Go load generator with timeout/conflict classification
  collect.py                    # logs, Prometheus export, prober output -> outputs/tidb/<handle>/
infra/tidb/
  deploy_infra.py               # script 1
  teardown_infra.py             # script 2
  install_tidb.py               # script 3
  run_experiment.py             # script 4
  run_all.py                    # script 5 (orchestrator)
  fetch_tidb.sh                 # download + checksum-verify tarballs, build PD from submodule if asked
  node_bootstrap.sh             # on-node: NVMe mount, OS tuning (pushed by user-data / install)
  iam-deploy-policy.json        # least-privilege policy for the deploy role (§9)
  README.md
tests/test_deploy_*.py          # NumPy/pytest-free where cheap, matching the library test style
outputs/tidb/<handle>/          # results (§7); raw logs gitignored, summaries tracked
docs/plans/exec-07-tidb-deployment.md   # this file
```

Existing `infra/aws/` (work VM) is untouched; `infra/tidb/` reuses its conventions (idempotent, tag-based,
`log()` style, `POWDER_*` env overrides).

## 3. TiDB in the repo (implemented, step 1)

1. **Pinned artifacts.** `third_party/tidb/VERSIONS` lists, per component, version, sha256, size and mirror URL for
   `tidb`, `tikv`, `pd`, `ctl` (pd-ctl), `prometheus`, `grafana` at **v8.5.8** and `tiup`, `cluster` at v1.17.1
   (linux/amd64, ~1.4 GB total). It is generated by `infra/tidb/pin_versions.py` from PingCAP's TiUP mirror
   manifests (`--check` reports drift) and is the only place versions live. The exporters
   (`node_exporter`, `blackbox_exporter`, `alertmanager`) are selected by tiup-cluster and get pinned in step 4,
   when we see which versions it picks.
2. **Fetch.** `infra/tidb/fetch_tidb.sh [--check] [--extract] [component ...]` downloads from the mirror into
   `~/.cache/powder/tidb/<version>/` (never into the repo) and **deletes and fails on any hash that differs from
   the pin**. We use the TiUP mirror (`tiup-mirrors.pingcap.com`), not `download.pingcap.org` (not resolvable from
   the sandbox, and not what TiUP itself uses). Trust model: HTTPS + hash pinning against drift; TiUP additionally
   verifies the mirror's signatures when it installs on the controller (step 4). The controller will run
   `fetch_tidb.sh --check` before `tiup cluster deploy` so a changed artifact stops a run.
3. **PD source.** `third_party/pd` is a shallow submodule of upstream `tikv/pd` at tag `v8.5.8`, commit
   `4ecadf2f59e4fa04fe0e611c6d7808e8b1806ce7`, which is **exactly the commit the stock `pd-server v8.5.8` binary
   reports**, so the source we will patch and the binary we run as baseline are the same code. On a fresh clone:
   `git submodule update --init --depth 1 third_party/pd`. We do not build PD in step 1: it needs **Go 1.25.12**
   (`go.mod`) and nothing uses it until #9, so the build script is deferred to #9's first step.
4. `docs/egress_allowlist.md` gains `tiup-mirrors.pingcap.com`.

## 4. Core contract: ClusterSpec in, RunState out

All five scripts communicate only through files in `outputs/tidb/<handle>/`, so any stage can be re-run
alone and `run_all` is just sequencing.

**`spec.yaml` (input, hand-written or generated by the ILP later):**

```yaml
handle: auto                    # auto = random AdjectiveNoun (e.g. OrangePanda); or an explicit name
label: smoke                    # free-text description, informational only
tidb_version: v8.5.8            # default from VERSIONS
scenario: scenario/canonical    # fault-library scenario (recorded, used by #8)
seed: 1
control_plane: {region: us-east-1, instance_type: m6i.xlarge, pd: 3, tidb_per_region: 1, loadgen_per_region: 1}
slots:                           # same grammar as scripts/fault_library/make_trace.py --slot
  tikv-1: {machine: machine/ford10-google-storage@us-east-1}
  tikv-2: {machine: machine/ford10-google-storage@us-east-2}
  ...
limits: {max_hourly_usd: 25, max_hours: 3}
```

* `slot` is the stable identity (survives instance replacement); `machine` is the library class whose faults
  #8 will inject; the region is the placement. This is the exact contract `trace.generate` already uses, so
  a trace for the run is `make_trace.py` over `spec.slots` with no translation.
* The spec is **declarative and idempotent**: running `deploy_infra` and `install_tidb` against a spec with
  an extra/removed slot adds/removes only that node. That *is* the replacement primitive #8/#9 need.

**`state.json` (output):** per slot `{instance_id, region, az, private_ip, public_ip, machine, state}`,
VPC/subnet/peering/SG ids per region, controller address, key path, stage status
(`infra: done|partial`, `install`, `experiments[...]`), creation time, estimated $/h. Written atomically
after every AWS call that creates something, so a crash never loses an id.

### 4.1 Run handles

A **run** is one deployment (infra + TiDB cluster) and can host several experiments over its life (experiments
keep descriptive names like `smoke`, `hotspot-60m`; the *handle* names the deployment).

* **Generation:** two CamelCase words from committed lists (`powder/deploy/names.py`, 101 adjectives x 92
  nouns (~9,300 handles), easy to read aloud, no look-alike or offensive words), e.g. `OrangePanda`, `QuietFalcon`. The
  generator rejects any handle that collides with a live `Run` tag in **any region** or an existing
  `outputs/tidb/<handle>/` directory, so a handle is never reused, even after teardown, and a results directory is
  never overwritten.
* **Explicit handles:** `--handle MyName` / `handle: MyName` in the spec. Validated (`[A-Za-z][A-Za-z0-9-]{2,30}`),
  must not already exist unless you are resuming that same run (`--resume`).
* **Where it appears:** AWS tags (`Run`, and `Name=<handle>-<slot>` so the console groups by run), the results
  directory, `state.json`/`manifest.json`, the key-pair name, VPC names, log lines, and the first and last line
  `run_all.py` prints (it also prints the exact teardown command on any failure).

### 4.2 Experiments run on a handle

The deployment spec (above) says what to *build*. An **experiment config** says what to *do* on a built
deployment, and is a separate file so one deployment can host several experiments:

```yaml
# infra/tidb/experiments/smoke.yaml
name: smoke
workload: {type: smoke, key_dist: uniform, rate_per_region: 500, duration: 10m}
request_timeout: 5s              # defines "timed out" for the workload metric (§7)
prober: {interval: 1s}
region_map_interval: 10s         # PD Region-map snapshots for RSM attribution + repartition metrics (§7)
pd_runtime_config: {}            # optional pd-ctl overrides applied at start and restored at end,
                                 #   e.g. {max-store-down-time: 10m} for the node-timeout what-if; recorded
injector: none                   # `trace:<file>` reserved for #8
```

* **`run_experiment.py` requires `--handle <Handle>`** (no default, no "latest run" guessing). It fails fast if
  that handle has no `state.json`, install is not finished, or the controller is unreachable, and it never
  creates infrastructure.
* Results go to `outputs/tidb/<handle>/experiments/<name>-<utc>/` (the timestamp keeps repeated runs of the same
  experiment from overwriting each other); the experiment's `config.json` and the run `manifest.json` both
  record the handle, git sha, and the deployment spec hash it ran against.
* `run_all.py --spec S --experiment E1 [--experiment E2 ...]` generates the handle (or uses the spec's), then
  deploys, installs, runs each experiment **on that handle** in order, collects, and tears down.

## 5. Network design (script 1)

Per region R in the run (all four at most): one VPC `10.<i>.0.0/16` (non-overlapping by region index), one
public subnet per used AZ (AZ is recorded in state because exec #2-az results are AZ-aware), IGW, route
table; **full-mesh inter-region VPC peering** (6 pairs for 4 regions: request, accept in the peer region,
routes in both route tables to the peer CIDR). Security group per region: all TCP/UDP from the four VPC
CIDRs (cluster traffic, TiUP SSH from controller), SSH (22) from the operator's /32, Grafana/Prometheus only
from the operator's /32. No cross-region SG references (unsupported over inter-region peering) — CIDRs only.
Intra-mesh traffic uses private IPs, so inter-region $/GB billing matches the `transfer` prices in the library.

Node launch: Ubuntu 24.04 AMI resolved per region from the public SSM parameter; one key pair per run
(created in each region, private key stored `0600` under `outputs/tidb/<handle>/` and gitignored); IMDSv2
required; user-data runs `node_bootstrap.sh`: format the instance-store NVMe (`ext4 -O ... nodelalloc,noatime`
as TiDB recommends), mount at `/data`, swap off, THP off, ulimits, chrony. **Instance-store caveat:** stop/start
or terminate loses NVMe data (matches "permanent" faults); plain `reboot` keeps it. #8 must therefore use
process kill / `reboot` for transients, never EC2 stop/start.

Preflight (fails before creating anything): credentials, the Service Quotas for
"Running On-Demand Standard (A,C,D,H,I,M,R,T,Z) instances" vCPUs in every used region
(30 TiKV x 8 vCPU = 240 vCPU total), `i4i.2xlarge` offered in the chosen AZs, `max_hourly_usd` /
`max_hours` limits vs. `pricing.py` estimate, and an explicit spend acknowledgement (`--yes` or
`POWDER_ALLOW_SPEND=1`).

## 6. The scripts

All scripts take the run **handle** (`--handle OrangePanda`; its directory is `outputs/tidb/<handle>`, override with `--run-dir`), `--dry-run` (print the plan and estimated cost,
touch nothing), `--log-level`, non-zero exit on failure, and are resumable (re-running skips finished work).

1. **`deploy_infra.py --spec spec.yaml`** — preflight (§5) → network → controller + node launch → wait for
   SSH + bootstrap done → write `state.json`. On partial failure it records what exists and exits
   non-zero; it never tears down by itself (the orchestrator decides).
2. **`teardown_infra.py`** — `--list` shows every deployment still holding AWS resources;
   `teardown_infra.py OrangePanda [BlueFalcon ...]` tears down the named handles; `--all` tears down every
   `Project=powder` deployment; `--older-than 6h` tears down those older than that (trivial: it filters on the
   `CreatedAt` tag); `--verify` fails if anything tagged `Project=powder` remains (details in §6.0). Discovery is
   **by tag** in every region (not only from `state.json`); deletion is in dependency order (instances → wait
   terminated → peering → routes → IGW → subnets → SGs → VPCs → key pairs) and tolerant of already-deleted
   resources.
3. **`install_tidb.py --handle H`** — on the controller: fetch/verify tarballs, `tiup cluster check
   --apply`, render `topology.yaml` from spec+state (`topology.py`): PD x3 spread over regions, TiKV
   per slot with labels `region`, `zone`, `host`, **`machine_class`**, TiDB + monitoring + load generators
   per §0; PD `location-labels = ["region","zone","host"]`, `isolation-level = "region"`, replication
   factor 3; `tiup cluster deploy/start`; health gate: all PD members up, every store `Up` with expected
   labels, `SELECT 1` through each TiDB, Grafana reachable. Also builds/copies `powder-load` and the prober to
   the load generators. Output `install/` (topology, `tiup cluster display`, versions).
4. **`run_experiment.py --handle H --experiment smoke [--workload smoke] [--duration 10m]`** — creates
   the schema (a table pre-split into one key range per target Raft region, e.g. 50 to mirror the 50-RSM
   study), starts the prober on each load generator and the workload, waits, stops them, then runs
   collection (§7). `--injector none` is the only value today; `--injector trace:<trace.jsonl>` is reserved
   for #8 (the flag, the `faults/` output directory and the manifest field exist, the implementation errors
   with a clear message).
5. **`run_all.py --spec spec.yaml [--experiment ...] [--keep-infra] [--yes]`** — `deploy → install →
   experiment(s) → collect → teardown`, using the same library functions (not subprocesses where avoidable).
   Teardown runs in `try/finally` and on SIGINT/SIGTERM; teardown failure is loud, prints the exact
   `teardown_infra.py OrangePanda` command, and exits non-zero even if the experiment succeeded.
   `--keep-infra` leaves everything up for debugging and prints the hourly burn. `--resume` continues a
   run from `state.json`.

   *Future:* `infra/tidb/campaigns/<name>.py` composes `run_all` for a fixed experiment set and then calls an
   analysis/plot script over `outputs/tidb/<handle>/`. That works because experiments are keyed by name
   inside a run directory and every artifact is machine-readable (§7).

## 6.0 Listing and tearing down deployments

`teardown_infra.py --list` scans all regions for `Project=powder` resources and prints one row per handle:

```
HANDLE        AGE    REGIONS (nodes)                $/h    LOCAL DIR
OrangePanda   3h10m  use1(4) use2(4) usw2(4) =12    9.4    yes
QuietFalcon   26h    use1(2) usw2(2)                3.1    yes
BoldHeron     2d     use2(1)                        0.7    no     <- deployed from another machine, or dir deleted
SwiftOtter    1h     VPCs only (0 instances)        0.0    yes    <- partial deploy
```

Teardown selection is deliberately small:

| Command | Effect |
|---|---|
| `teardown_infra.py OrangePanda [BlueFalcon ...]` | the named handles (a handle that matches nothing is an error, not a no-op) |
| `teardown_infra.py --all` | every `Project=powder` deployment |
| `teardown_infra.py --older-than 6h` | deployments whose `CreatedAt` tag is older than that |
| `teardown_infra.py --list` / `--verify` / `--dry-run` | inspect; fail if anything tagged remains; show the plan only |

Anything selected is printed with node counts and $/h and needs a typed `yes` (or `--yes`). Only resources with
`Project=powder` are ever touched. Teardown never deletes the local `outputs/tidb/<handle>/` results and writes
`teardown.json` there (what was deleted, when, residual check). No owner filtering, globbing, groups or expiry
tags: add them only if a real need appears.

## 6.1 How the scripts reserve VMs

Only **`deploy_infra.py`** creates VMs (it is also the only path that adds nodes later, by re-running it on an
edited spec). `teardown_infra.py` deletes them. `install_tidb.py` and `run_experiment.py` only SSH into
machines that already exist, and `run_all.py` just calls the other four. "Reserve" means plain **on-demand
`RunInstances`**, no Spot and no Capacity Reservations by default.

Sequence inside `deploy_infra.py` (every call is tagged `Project=powder, Run=<handle>, Owner, CreatedAt`; `Name` tags read `<handle>-<slot>`, e.g. `OrangePanda-tikv-3`):

1. **Preflight, no side effects:** `sts:GetCallerIdentity`; per region `service-quotas` on the "Running
   On-Demand Standard (A, C, D, H, I, M, R, T, Z)" vCPU quota vs. the vCPUs this spec needs;
   `DescribeInstanceTypeOfferings` per AZ for `i4i.2xlarge` and the control-plane type; price/limit check
   against `limits`. Anything short of the quota is reported as "request N vCPUs in region R" before we start.
2. **Network per region:** `CreateVpc` (`10.<i>.0.0/16`) -> `CreateSubnet` per AZ used -> `CreateInternetGateway`
   + attach -> route table with `0.0.0.0/0` -> IGW -> `CreateSecurityGroup` + ingress rules.
3. **Peering mesh:** for each region pair, `CreateVpcPeeringConnection` from the lower-indexed region, then
   `AcceptVpcPeeringConnection` in the peer region (waiting for `active`), then `CreateRoute` to the peer CIDR in
   both route tables.
4. **Key pair:** one SSH key is generated locally for the run and `ImportKeyPair`'d into every region (so the same
   key opens every node).
5. **Instances:** `RunInstances` once per (region, AZ, instance type) group with `MinCount=MaxCount`, Ubuntu 24.04
   AMI from the SSM public parameter, root EBS 50 GB gp3 (the i4i NVMe volume attaches automatically), IMDSv2
   required, `InstanceInitiatedShutdownBehavior=terminate`, user-data = `node_bootstrap.sh`,
   **`TagSpecifications` on the instance, volume and ENI at launch** (so there is never an untagged window in
   which a crash could leak a machine), and a **`ClientToken` derived from `handle + slot`** so a retried call
   cannot double-launch. Each instance carries tag `Slot=tikv-3` / `Role=controller|pd|tidb|loadgen`, which is
   the slot -> instance mapping; re-running with the same spec looks nodes up by `Run + Slot` and launches only
   the missing ones (this is also how a permanently lost node is replaced).
6. **Capacity failures:** `InsufficientInstanceCapacity` in an AZ retries the other AZs of that region if the slot
   did not pin one (spec `az: auto|us-east-1c`); pinned AZs fail loudly. `VcpuLimitExceeded` can only appear if
   preflight was bypassed. If i4i capacity is a recurring problem for the 30-node run we add optional On-Demand
   Capacity Reservations (`CreateCapacityReservation`) in step 3 of the work breakdown; not needed for the
   smoke test.
7. **Wait + readiness:** `instance_running` waiter -> `instance_status_ok` -> SSH reachable through the controller ->
   `/var/lib/powder/bootstrap.done` present on every node (NVMe mounted at `/data`, OS tuning applied) ->
   `state.json` complete. Failures leave `state.json` describing what exists; nothing is torn down implicitly.

`teardown_infra.py` reverses it from tags alone, across **all** regions (so a lost `state.json` or a half-failed
deploy still gets cleaned up): terminate instances -> wait for `terminated` -> delete peering -> routes -> detach
and delete IGW -> subnets -> SGs -> VPCs -> key pairs, then `--verify` rescans every region for anything still
tagged `Project=powder` and fails if it finds any.

| Script | Creates | Deletes | Reads |
|---|---|---|---|
| `deploy_infra.py` | VPCs, subnets, IGW, routes, peering, SGs, key pairs, instances | nothing | quotas, offerings, AMIs, prices |
| `teardown_infra.py` | nothing | everything tagged for the named handle(s) / all / older than X | tags, resource graph |
| `install_tidb.py` | files/services on instances (no AWS resources) | nothing | `state.json` |
| `run_experiment.py` | files on instances; local result files | nothing | `state.json`, TiDB/Prometheus APIs |
| `run_all.py` | (via the above) | (via teardown) | |

## 6.2 PD and the cluster manager (TiUP) in detail

**What PD is.** The Placement Driver is TiDB's control plane: a small Raft group (3 members here) that (a) stores
the metadata of every Raft group ("Region": key range, peers, leader, epoch), (b) hands out transaction
timestamps (TSO), and (c) **schedules**: each TiKV store sends heartbeats and per-Region reports; PD decides
changes and returns them as *operators* (add peer, remove peer, transfer leader, split, merge) in heartbeat
responses. TiKV executes them. TiDB servers are stateless SQL layers that ask PD where a key lives.

**Why it matters for Powder.** Everything the paper compares against is a PD behavior, and several of our knobs are
PD configuration:

| Behavior | PD mechanism | Relevance |
|---|---|---|
| Replace a dead node | A store silent for `max-store-down-time` (default 30 min) is marked down and the replica/rule checker re-creates its replicas elsewhere | This **is** the PDF's "node replica timeout" what-if (10/15/20 min); we set it per run via spec `pd_config`. |
| Where replicas go | `location-labels`, `isolation-level`, and *placement rules* (counts, roles incl. `learner`, label constraints) | Baseline "TiDB PD" placement; learner rules = the "active + passive learner" what-if. |
| Merge small Regions | merge checker, bounded by `max-merge-region-size/keys` and schedule limits | Goal 1.4: ILP/simulator picks which nodes stay in a merge. |
| Split / new Region placement | TiKV splits a Region in place (`region-split-size`), PD then balances; `scatter-region` for pre-split tables | Goal 1.4: ILP picks which nodes take a split partition. |
| Rate of data movement | `region/replica/leader/merge-schedule-limit`, per-store `store-limit` | Recovery speed; feeds Goal 2.1 calibration and the GB-movement constraint. |
| Rolling software updates | `tiup cluster upgrade` evicts leaders (`evict-leader` scheduler) per node and restarts | Goal 5: the weekly-update downtime we compare against. |

The #9 hook is therefore **inside PD's store-selection path for add-peer, remove-peer and merge.** Read from the
pinned `third_party/pd` (v8.5.8) in step 1; to be re-read when #9 starts:

| Decision | Where (under `pkg/schedule/`) | Notes |
|---|---|---|
| Which store gets a new replica | `checker/replica_strategy.go` `ReplicaStrategy.SelectStoreToAdd` (filters, then isolation score, then lowest region score), `SelectStoreToFix` (replace a down/offline peer) | Called from `checker/rule_checker.go` `addRulePeer`, `replaceUnexpectedRulePeer`, `fixLooseMatchPeer`. With placement rules enabled (default) the rule checker is the live path; `replica_checker.go` is the legacy one. |
| Which replica is removed | `ReplicaStrategy.SelectStoreToRemove`; rule checker `fixOrphanPeers` | |
| Rebalancing | `schedulers/balance_region.go`, `balance_leader.go`, `hot_region.go` | Candidate-store selection in each. |
| Merges | `checker/merge_checker.go` `MergeChecker.Check` picks the adjacent target, checks size/keys limits, and `operator.CreateMergeRegionOperator` aligns the source's peers to the target's stores | The merged Region's store set is a PD decision: the ILP/simulator can choose which stores survive. |
| Splits | TiKV splits a Region in place (children start on the parent's stores); PD then moves replicas via the paths above. `splitter/` and `checker/split_checker.go` handle PD-initiated splits (placement-rule key ranges, pre-split tables), `scatter/region_scatterer.go` spreads new Regions | "Which nodes take a split partition" = the follow-up placement of the children. |
| Dead-node timeout | `config/config.go` `MaxStoreDownTime` (default 30 min), checked by the rule checker (`isStoreDownTimeHitMaxDownTime`) | The what-if knob. |

There is also `plugin_interface.go` (Go `plugin` shared objects), but Go plugins must be built with the exact
toolchain and dependency versions of the host binary, so the primary plan is a patched store-selection function
that calls the oracle over HTTP, shipped with `tiup cluster patch -R pd`. Today the plan only guarantees what that
work needs: the PD source pinned to the baseline binary's commit, store labels including `machine_class`, and a
reserved oracle endpoint.

**What TiUP is and what it does for us.** TiUP is PingCAP's package/cluster manager. `tiup cluster` takes a
topology YAML (`global`, `server_configs` for pd/tikv/tidb, `pd_servers`, `tikv_servers`, `tidb_servers`,
`monitoring_servers`, `grafana_servers`) and, over SSH from a control machine, creates a service user, pushes
binaries, writes configs, installs **systemd units**, and starts everything in dependency order. We use:
`check --apply` (OS prerequisites), `deploy`, `start`, `display`, `scale-out`/`scale-in` (node replacement, `--force`
when the instance is already gone), `edit-config`/`reload` (change PD/TiKV settings for what-if runs),
`exec` (run a command on all nodes: log collection), `patch` (PD swap for #9), `upgrade` (rolling update).
TiKV store labels come from `config.server.labels` per node (`region`, `zone`, `host`, `machine_class`).

**Constraints we design around.** (1) TiUP uses the topology `host` both to SSH and as the advertised address, so
the control machine must reach **private** IPs: hence the controller VM inside the peering mesh. (2) TiUP wants a
passwordless-sudo SSH user on every node (the `ubuntu` user + the run's key). (3) All TiUP components download to
the controller once; nodes never hit the internet for TiDB bits.

**Alternatives considered.** *tidb-operator on Kubernetes (EKS):* production-grade but adds a k8s layer, pod
networking and node pools between us and the processes/NVMe/iptables the failure emulator must manipulate.
*Hand-rolled systemd + tarballs:* maximal control but we would re-implement labels, scale-in/out, rolling restarts and
config reload. TiUP is the standard route for VM deployments and keeps #8/#9 small. Revisit only if TiUP's
scale-in of a permanently lost store proves too slow or fragile in step 4's testing.

## 7. Experiments and result collection

**Three measurements, collected independently: two availability metrics and repartitioning.**

1. **Partition (RSM) availability: the prober.** `prober.py` runs on each load generator: every 1 s
   (configurable) it does a short write+read transaction on one key per target Raft region (via the local TiDB)
   with a hard timeout, and logs `{ts, region_key, ok, latency_ms, error}`. Per-RSM unavailability =
   failed-probe seconds / run seconds; aggregate = mean over RSMs and the worst RSM. This is the quantity the
   simulator/ILP predict (unit: time, per partition). Probing from every region means a region-isolating fault
   is visible as a client-side effect rather than assumed. It is independent of the load workload so it is not
   starved by it.
2. **Workload availability: fraction of requests that timed out.** What a user of the database sees, weighted by
   where the traffic actually goes (a hot partition counts more). Defined precisely:

   | Outcome class | Examples | Counts as |
   |---|---|---|
   | `ok` | committed / read returned | available |
   | `timeout` (unavailable) | client deadline exceeded, connect refused/reset, TiDB `9001` PD timeout, `9002` TiKV timeout, `9003` TiKV busy, `9005` Region unavailable | **unavailable** |
   | `conflict` | `9007` write conflict, `1213` deadlock, `1062` duplicate key, `8022`/retryable txn errors | **excluded from numerator and denominator** (application-level contention, not availability) |
   | `other` | anything unclassified | reported separately; the run is flagged if it exceeds 0.1% so nothing is silently dropped |

   `workload_unavailability = timeout / (ok + timeout)`; conflicts and `other` are reported as their own rates.
   The error-code list is checked against the pinned TiDB version's `errno` table in step 5, not assumed.

   **Every non-`ok` request records its cause, whatever its class** (not only timeouts): `outcome` class,
   `cause` (a stable slug such as `client_deadline`, `conn_refused`, `conn_reset`, `tidb_9005_region_unavailable`,
   `tidb_9007_write_conflict`, `mysql_1213_deadlock`, `unclassified`), the numeric `error_code`, the `layer` it
   surfaced at when known (client / TiDB / TiKV / PD), and the error text (truncated to 200 characters and
   stored through a dictionary so the log stays compact). `summary.json` carries a **cause breakdown table**
   (count and share) overall, per RSM, per client region and per 10 s interval, so a plot can show *why* requests
   failed during a window (e.g. election gap vs. region unavailable vs. connection loss) and #8 can later
   line causes up against injected fault events. Prober failures record the same fields.

   Two design requirements make this number meaningful: the generator is **open-loop** (requests are issued on a
   fixed or time-varying arrival schedule, each with its own client-side deadline), because a closed-loop client
   stops sending while the database is down and so *under-counts* the outage; and every request is tagged with its
   target Region/key range so the metric can be broken down per RSM and per client region. The deadline is a
   spec parameter (`request_timeout`, default 5 s) and is part of the metric's definition, so it is recorded in
   `config.json` and `summary.json`; the analysis can also recompute at other thresholds from the per-request log
   (`latency_ms` kept for every request in a compact binary/JSONL log).

   **Attribution to partitions and RSMs.** Requests are single-key point operations (range scans are tagged by
   their first key), so each maps to exactly one Region at issue time. Region boundaries and peer sets change during
   a run (splits, merges, rebalancing, replacement), so `run_experiment` snapshots the Region map from PD every 10 s
   (`pd/regions-<ts>.json`: key range, peer stores, leader) and attribution is done offline with the map valid at
   each request's timestamp. Three groupings are reported, for **both** metrics:

   | Grouping | Meaning | Used for |
   |---|---|---|
   | **per partition** | one TiKV Region (key range) | raw numbers; per-RSM roll-ups |
   | **per RSM** | the set of stores holding the Region's replicas (the Raft group's membership); Regions with the same store set are merged | comparison with the ILP/Markov per-RSM prediction and the placement it chose |
   | **per client region** | where the request originated | region-outage visibility |

   For each: prober unavailability (failed-probe time / run time), workload unavailability and its cause breakdown,
   request count, and p50/p99 latency. A Region with too few requests to be meaningful (e.g. < 100) is flagged
   rather than reported as 0% unavailable, and the weight each RSM carries in the workload metric (its share of
   requests) is always shown next to its number.

   The two metrics can legitimately disagree (uniform traffic vs. a failed cold partition; a leader-election gap
   shorter than one probe period), and reporting both is the point; `summary.json` has both.

3. **Repartitioning: how often and how long.** "Repartitioning" here means any change to how data is divided
   or placed: **splits**, **merges** (key ranges change), **replica moves** (add / remove / move peer: the RSM's
   membership changes and data is copied), and **leader transfers** (no data moves, but the serving node changes;
   tracked separately). Three sources are collected and cross-checked, because no single one is complete:

   | Source | Gives | Limits |
   |---|---|---|
   | **PD operator lifecycle**, from PD logs and the Prometheus operator counters/duration histograms | each operator's kind, Region, steps, creating scheduler/checker, create -> start -> finish/timeout/cancel times | metric and log-line formats vary by version: verified against the pinned tag in step 5 |
   | **Region-map snapshots** (every `region_map_interval`, default 10 s) diffed over time | version-independent ground truth for splits, merges, peer-set changes, leader changes, Region sizes, under-replicated / pending peers | 10 s resolution; events shorter than that are only seen via the operator log |
   | **TiKV counters** (admin commands: split / merge; snapshot bytes and durations sent/received) | actual bytes transferred and snapshot times | per store, not per Region |

   Reported in `summary.json` -> `repartition`, and as raw events in `repartition/events.jsonl`:

   | Metric | Definition |
   |---|---|
   | **How often** | event counts and per-hour rates by kind (split / merge / add-, remove-, move-peer / leader transfer) and by trigger (balance-region, balance-leader, hot-region, merge checker, replica/rule checker after a node loss, scatter, manual); Region count over time |
   | **How long** | per-operator duration distribution (p50 / p90 / p99 / max) by kind, with timeouts and cancels counted; **episodes** (operators separated by < 60 s gaps are one episode): episode count, duration distribution, and the **share of run time with at least one operator in flight** |
   | **How much** | GB moved (snapshot bytes), split intra- vs inter-region from the store -> region map, and the implied inter-region $ (exec #6 GB-movement constraint; paper Fig 8) |
   | **Per RSM** | times each RSM's store set changed, and time spent under-replicated (fewer healthy voters than the replication factor, or a learner still catching up): the restore-replication time #10 needs |
   | **Tie to availability** | every failed request/probe is flagged `during_repartition` if an operator touching its Region was in flight or a leader transfer happened within a few seconds, so the cause table shows how much unavailability coincides with repartitioning (paper Fig 7) |

   Initial smoke runs will mostly show splits and balancing at load start; the interesting numbers appear with
   hotspots (Goal 6.3) and node loss, which is why they are recorded from the first experiment.

**Load generator: `powder-load`** (small Go program, built on the controller, `database/sql` + MySQL driver),
not `go-ycsb`, because go-ycsb is closed-loop and does not classify errors this way. It takes a key distribution
(uniform / zipf / hot-range / shifting-hot-range) and an arrival-rate profile (constant / diurnal), which
are exactly Goal 6.3's four workloads, so only the smoke workload is built now and the others are config.
Throughput/latency comparisons against vanilla TiDB (paper Fig 5/6) come from the same tool. If it
proves too heavy, the fallback is `go-ycsb` for performance plus an asyncio Python client for the availability
metric.

Workload interface (`workloads/base.py`): `prepare(cluster)`, `run(cluster, duration, seed) -> artifacts`. The smoke
workload is uniform keys at a fixed modest rate (e.g. 500 req/s per region) for the duration.

**Collected per experiment** (`outputs/tidb/<handle>/experiments/<name>-<utc>/`):

| Path | Content |
|---|---|
| `config.json` | experiment config, handle, seed, start/end UTC |
| `prober/<region>.jsonl` | raw probe results (`ts, region_key, ok, latency_ms, cause, error_code`) |
| `workload/<client-region>/requests.*` | per-request log from `powder-load` (`ts, key range, outcome class, cause, error_code, layer, latency_ms`; error text via `errors.dict.json`) + per-interval throughput/latency |
| `metrics/*.json` | Prometheus `query_range` export of a fixed metric list (QPS, p99, region/leader counts per store, PD scheduler counters, store state, raft apply/commit, snapshot send/recv bytes) |
| `logs/<slot>/{pd,tikv,tidb}.log.gz` | service logs from every node |
| `pd/` | `pd-ctl` dumps: `store`, `config show all`, placement rules, `operator` history at start/end, and the periodic Region-map snapshots `regions-<ts>.json` used for attribution |
| `repartition/events.jsonl` | one line per repartitioning event: `{kind, region_id, key_range, trigger, from_store, to_store, t_create, t_start, t_finish, duration_s, outcome, bytes_moved, source}` |
| `faults/{planned,applied}.jsonl` | empty today; the #8 trace and what was actually injected |
| `summary.json` | prober unavailability and workload unavailability (timeouts / (ok + timeouts)) **per partition, per RSM, per client region and aggregate**, with request weights; conflict and other rates; **cause breakdown tables** (overall, per RSM, per interval); the `repartition` block (below); throughput/latency percentiles; node-count, hours, **modeled $ cost** (library prices + transfer, not the bill) and the **actual** estimated AWS cost |

Run-level `manifest.json`: git sha + dirty flag, library content hashes (from `trace` header logic), TiDB
versions, spec hash, instance ids, stage timings (deploy/install/experiment/teardown wall times — also
seed data for #10's setup-timing distributions). Large raw logs are gitignored; `manifest.json`,
`summary.json` and `config.json` are committed so a result can be cited the way library entries are.

## 8. Interfaces reserved for #8, #9, #10 and Goal 1

* **Fault injection (#8):** `trace.generate` already emits `{slot, region, machine, kind, cause, start,
  duration}`. #8 adds an `injector` that connects through the controller and implements the §3.5 action
  table: kill/restart TiKV (transient), kill + `rm -rf /data/tikv` + replace the instance (permanent),
  graceful stop (planned), `iptables` partition of a region (region outage). This plan guarantees what it
  needs: stable slot→instance mapping in `state.json`, systemd units with predictable names
  (`tikv-20160.service`), data dir on the NVMe at `/data`, `iptables`/`tc` present on nodes,
  idempotent spec-driven node replacement (§4), and `faults/` output paths.
* **PD ↔ ILP (#9, Goal 1.4):** TiKV stores carry the label **`machine_class`** (and region/zone/host), so a
  patched PD (or the oracle process it calls) can see each store's library class directly from store labels
  without a side channel. The controller reserves a `powder-oracle` slot (port 8650, SG rule, systemd
  unit placeholder, spec field `oracle: {enabled: false}`), and `tiup cluster patch` is the path to ship the
  forked PD from `third_party/pd`. ILP-driven *initial* deployment (Goal 1.2.1) = generate `spec.yaml` from
  an ILP placement; nothing else changes.
* **Recovery timing (#10, Goal 2.1):** stage wall times and `pd-ctl` operator history are already collected;
  #10 adds dedicated benchmarks (inter-region download speed, leader election) as new experiments under
  the same `run_experiment` skeleton.
* **Storage/size constraints (Goal 1.1):** spec records per-slot disk (`/data` size from the instance
  metadata) so the ILP's storage model can be checked against real stores.

## 9. Safety, cost, permissions

* **Spend guard (budget is not the concern; unattended and forgotten resources are).** `deploy_infra.py` and
  `run_all.py` refuse to create anything unless given `--yes` or `POWDER_ALLOW_SPEND=1`, and always print the
  estimated $/h and total first. Interactive use: you pass `--yes` and it is a one-word confirmation. The point
  is the *unattended* sessions: `infra/aws/run_claude.sh` starts a Claude Code session every 6 h on the work VM
  with Bash access and nobody watching. Its environment (`~/.powder-runner.env`) will not export
  `POWDER_ALLOW_SPEND`, and its instance role (`infra/aws/iam-policy.json`) is read-only for EC2, so a runner that
  decides to "just test the deploy script" gets a refusal, not a 30-node bill. If you later want unattended
  deploy runs, make it a deliberate step: a separate role with the tag-scoped deploy policy, the env var set in a
  *different* runner profile, and `limits.max_hourly_usd` / `max_hours` caps enforced by the script.
* Cost estimate printed before launch: 1 h smoke test of 6 TiKV (`i4i.2xlarge` ≈ $0.69–0.76/h) plus control
  plane is roughly **$5–6**; the 30-node 24 h study is roughly **$600+** plus inter-region transfer
  ($0.02/GB class). `pricing.py` computes the real figure from the price library.
* `iam-deploy-policy.json`: EC2 create/delete restricted by tag condition where AWS supports it
  (`aws:RequestTag/Project=powder`), VPC peering, SSM `GetParameter` for AMIs, `servicequotas:Get*`,
  `pricing:Get*`. Attached to a separate role/user from the work VM role (`infra/aws/iam-policy.json`).
* Teardown safety: the orchestrator's `finally`, tag-based `teardown_infra.py --verify`, and `--list` to see
  forgotten deployments by name (§6.0). Optional backstop (step 7, only if wanted): a daily cron on the work VM
  running `teardown_infra.py --older-than 48h --yes` and logging `--list` to `~/powder-logs/`. `limits.max_hours` is
  used for the cost preview and a warning, not as an expiry tag.
* Never print or log the private key; `*.pem` and `outputs/tidb/*/ssh/` in `.gitignore`.

## 10. Testing strategy (live AWS first)

Live AWS is the real test, but the code was built where no AWS credentials exist, so v3 reinstates stand-ins: **moto**
(emulated EC2, a dev dependency; tests skip without it) for deploy/teardown/run_all, scripted/fake runners for the SSH stages,
the real PD/TiKV output captured in `tests/fixtures/tidb/` for the parsers, and `infra/tidb/local_pipeline_check.py` (real TiDB via
`tiup playground`) for the Go tool and the metric chain. These caught real bugs (below) but cannot prove that SSH, TiUP-over-peering,
IAM, quotas or cross-region behaviour work: only the live gates can. What we keep from v2 is the cheap logic tests and the fast live dev loop.

**1. Pure-logic unit tests (seconds, no AWS, no mocks).** Only for logic where a bug would be silent or expensive
to find live:

* the **availability metrics**: outcome classification (every error code in §7's table, plus unknown -> `other`),
  open-loop accounting, and `summary.json` math, against canned request/probe logs with hand-computed answers
  (e.g. 600 s run, one RSM down 60 s -> 0.1 unavailability; conflicts excluded from numerator and denominator);
* network planning: non-overlapping CIDRs, exactly C(n,2) peerings, route entries on both sides;
* `topology.py` golden file (labels, `location-labels`, PD/TiKV counts per region);
* handle generation (format, uniqueness against fake tag/dir lists, collision retry, explicit-name validation) and
  teardown selection over a fake deployment list (handle list, `--all`, `--older-than`, unknown handle is an error);
* repartition detection: operator-log parser on canned PD log lines; Region-map snapshot diffs that must report the
  right splits, merges, peer moves and leader changes, durations, episodes and bytes moved;
* RSM attribution: Region maps that split/merge/rebalance mid-run give the right per-RSM numbers on a canned log;
* teardown ordering over a resource graph (dependencies before dependents; tolerates already-deleted nodes);
* spec validation and the cost/quota arithmetic behind `--dry-run`.

**2. `--dry-run` everywhere** (needed anyway for the cost/quota preview).

**3. A fast live dev loop.** A *dev run* (e.g. 3 regions x 2 TiKV, ~$5/h) is created once with `--keep-infra`.
`install_tidb.py` and `run_experiment.py` are re-runnable against it (`--resume`, re-install = `tiup cluster
destroy` + deploy, experiments are keyed by name), so we iterate on topology, prober, load generator and
collection without paying the ~5-10 min VM+peering setup each time. Teardown at the end of the session;
`teardown_infra.py --list` shows anything left over.

**4. Live gates (each must pass before the next step starts):**

| Gate | What runs | What it proves |
|---|---|---|
| G1 infra | `deploy_infra` for 2 nodes in 2 regions, ping + SSH over private IPs, NVMe mounted, then teardown + `--verify` | peering/routes/SGs/quotas; no leaks |
| G2 install | 6 TiKV across 3 regions + control plane: health gate green, labels present, `SELECT 1` per TiDB | TiUP over the mesh, PD placement by labels |
| G3 metric self-test | `run_experiment --self-test`: (a) pick a Region via `pd-ctl`, stop a majority of its peers for a scripted 60 s, restart, repeat for 3 Regions; (b) scripted repartitioning: `SPLIT TABLE` into a known number of Regions, a `pd-ctl operator add` peer move and leader transfer | (a) prober and `powder-load` report ~60 s unavailability for exactly those RSMs, and ~0 for the others; (b) the repartition log reports exactly the known splits, the peer move and the leader transfer, with plausible durations; no fault emulator needed |
| G4 smoke | full `run_all --spec smoke.yaml`: deploy -> install -> 10 min experiment -> collect -> teardown, `--verify` clean | the whole pipeline; **exec #7 done** |
| G5 abort safety | run G4 again, SIGINT during the experiment; then another run with `state.json` and its directory deleted: `--list` must still show its handle (LOCAL DIR `no`) and `teardown_infra.py <handle>` + `--verify` must clean it; also start two deployments and tear one down by handle, leaving the other untouched | teardown guarantees under failure; handle selection never touches the other deployment |

G3 is the one that matters most for the paper: it calibrates the measurement itself before any emulated-fault
result is trusted, and the known outage durations give an exact expected answer.

## 11. Work breakdown (each step its own commit; tick in §12)

1. **Pin + fetch (done):** `third_party/tidb/VERSIONS`, `pin_versions.py`, `fetch_tidb.sh` with checksums,
   `third_party/pd` submodule at the matching tag, PD hook files recorded in §6.2, allowlist addition.
2. **Library + unit tests + `--dry-run` (done, §12):** spec/state/pricing/topology, network planning, metric math (§10.1). No AWS calls.
3. **`deploy_infra.py` + `teardown_infra.py`** (handles, `--list`, selectors); gate G1.
4. **`install_tidb.py` + `node_bootstrap.sh`**; gate G2. Decide here whether TiUP scale-in of a lost store is good
   enough (§6.2 alternatives).
5. **`powder-load`, `prober.py`, workload interface + smoke workload, `collect.py`, `run_experiment.py`
   (including `--self-test`)**; gate G3.
6. **`run_all.py`** with signal-safe teardown, `--resume`, `--keep-infra`; gates G4 and G5 = **exec #7 done**.
7. **Docs:** `infra/tidb/README.md`, update `CLAUDE.md` "Where things are", `iam-deploy-policy.json`; optional `--older-than` cron on the work VM.

Steps 1-2 can be done anywhere; steps 3-6 need AWS credentials on the machine running them.

## 12. Progress checklist

- [x] Step 1 (2026-11-02): pin + fetch + `third_party/pd`. Pinned v8.5.8 (+ tiup/cluster v1.17.1) from the TiUP mirror; fetch/verify/tamper paths tested; stock `pd-server`, `tikv-server`, `tidb-server` report 8.5.8; PD submodule commit equals the stock PD binary's commit. Decisions: mirror not `download.pingcap.org`; PD build deferred to #9 (needs Go 1.25.12).
- [x] Step 2 (2026-11-02/03): `powder/deploy` library (spec, state, handles, netplan, pricing incl. fetched control-plane prices,
  topology), metric math (outcomes, availability, repartition), real-format Region-map loader and PD log parser written against
  captured v8.5.8 output, `python -m powder.deploy.plan` dry-run. Deviation: handle word lists are 101 x 92 (~9,300 names), not 200 x 200.
- [x] Step 3 (code, 2026-11-03): `deploy_infra.py` / `teardown_infra.py`, moto-tested. **G1 passed on real AWS 2026-11-04** (handle
  `AmberPheasant`: 8 nodes us-east-1 + us-west-2, 2x i4i.2xlarge; all private IPs ping from the controller, ~61 ms
  cross-region; SSH over private IPs via the controller; 1.7 TB instance NVMe mounted at `/data` on both TiKV; teardown +
  `--verify` clean). Runs 1-3 found and fixed: peer-region `InvalidVpcPeeringConnectionID.NotFound` lag (code), VPC-scoped
  creates and untagged accepter side of cross-region peerings (`iam-deploy-policy.json`, now validated for G1's calls).
- [x] Step 4 (code, 2026-11-03): `install_tidb.py` + `node_bootstrap.sh`, tested with a scripted runner. **G2 passed on real AWS 2026-11-04**
  (handle `HappyMongoose`, `specs/smoke.yaml`: 16 nodes, 3 regions): all 14 TiUP instances `Up`; every TiKV store has
  `region/zone/host/machine_class`; `SELECT 1` and v8.5.8 through each of the 3 regional TiDBs; all 65 Regions have 3 peers
  in 3 distinct AWS regions; teardown + `--verify` clean. Fixes found on the way: fetch TiUP's `root.json` trust anchor;
  `machine_class` appended to `location-labels` (tiup rejects store labels not listed there; below `host` it cannot change
  placement); chown `/data/tikv` to `tidb` after deploy (bootstrap made it root-owned); skip `deploy` when tiup already has
  the cluster (re-runs continue); `start --wait-timeout 1200` (first-TiDB schema bootstrap took ~2.5 min cross-region).
- [x] Step 5 (code, 2026-11-03): `powder-load` (Go) + probes, collection, `run_experiment.py`, `--self-test` planner/evaluator;
  validated on a real local TiDB. **G3 run on AWS 2026-11-04 (handle `FrostyDragonfly`, 4 attempts): NOT passed.**
  The measurement itself is validated: with crash faults, partitions whose Region lost quorum read 65-73 s for a 60 s
  hold (outage + TiKV restart + election) and partitions that only lost a leader read 0-14 s; round 2 of the last
  attempt passes outright. Fixes made on the way (all in `exec07-g1-live`): probers wrote the same 50 rows from 3
  regions and server-side lock waits (50 s) outlived the 5 s client deadline -> self-sustaining lock pile-up (each
  prober now has its own row, `innodb_lock_wait_timeout`/`max_execution_time` = client deadline); outages were planned
  on the pre-prepare Region map (system tables) -> planned after prepare on the probed Regions; per-RSM sums of
  partition-seconds -> per-round, per-partition evaluation from raw probes, leaving out Regions that moved around the
  kill; `systemctl stop` is a graceful shutdown that tries leader eviction for up to 20 s -> SIGKILL; simultaneous
  actions ran ~7 s apart -> run in parallel; log collection never created its directory on PD/TiKV/TiDB nodes;
  TiDB's default `token-limit` 1000 filled with statements retrying on dead Regions and stalled *all* queries when
  ~half the keys were down -> topology sets 20000. **Open (decide before the paper's baseline runs):**
  (1) round 3 (stores 10+11) still made 38 healthy partitions unavailable for ~40 s (e.g. a Region on stores 2,5,12
  with its leader on 12), starting ~30 s after the kill while TiDB logs `regionMiss backoffer.maxSleep 40000ms` for the
  dead Regions: cross-RSM interference under an open-loop client that abandons requests at 5 s. Real TiDB behaviour or
  harness artefact is not settled; it directly affects any per-RSM availability claim. (2) Whether baseline runs keep
  `token-limit` 20000 or TiDB's default. (3) One leader-only partition read 25 s (allowance 20 s). Evidence:
  `~/powder-runs/tidb/FrostyDragonfly/experiments/` on the work VM (runs 1-4, incl. TiDB/TiKV/PD logs for 2-4).
- [x] Step 6 (code, 2026-11-03): `run_all.py` with guaranteed teardown (success, failure, half-failed deploy, SIGINT/SIGTERM,
  repeated signals, failing teardown). **G4 and G5 not run: exec #7 is NOT done until they pass.**
- [x] Step 7 (2026-11-03): `infra/tidb/README.md`, `iam-deploy-policy.json` (draft, unvalidated), `CLAUDE.md`, unattended-runner
  safety line, `deploy`/`dev` extras. The `--older-than` cron is documented, not installed.
- [ ] **Owner: run gates G1 -> G5 (§10) on real AWS.** First `deploy_infra.py --dry-run`, then G1 (~$1).

## 13. Open questions (owner)

1. **Is PD ever fault-injected?** Proposal: no (stable control plane, 3 PD across 3 regions) for the paper's
   first results; note as a limitation. If yes, PD nodes become `slots` with a role.
2. **Fork or vanilla PD submodule now?** Proposal: submodule of upstream `tikv/pd` at the pinned tag today; add
   your own fork remote when #9 starts.
3. **Instance type for the control plane / load generators** (`m6i.xlarge` proposed) — load generators in the
   same region as TiDB servers, or one region only?
4. **Regions for the smoke test:** 3 of the 4 regions (cheaper) vs all 4. Proposal: 3 for the smoke test, 4 for
   the first real study.
5. **Result storage:** local `outputs/` only, or also an S3 bucket for raw logs (needs a bucket + policy)?
6. **Where the scripts run:** your Mac, or a work VM with a deploy role (separate from the unattended Claude
   role)? The plan supports both; it only changes where the deploy key and role live.
7. **Request timeout** that defines "timed out" for the workload metric (default 5 s): what does the paper's
   availability SLA call a failed request? The per-request log lets us recompute at other thresholds.
8. **Should unattended runner sessions ever be allowed to deploy?** Default: no (§9).

## 14. Implementation notes (v3)

**Deviations from the earlier drafts**

* `RunInstances` is **one call per node**, not per (region, AZ, type) group: each node needs its own `Slot` tag and idempotency
  token. AZs are chosen per region as those offering every instance type used there (up to 3); capacity errors fall back to the
  other AZs unless the slot pins one.
* `powder-load` is built **on the controller** (hash-pinned Go in `infra/tidb/loadgen/GO_PIN`), so no Go is needed on the operator's
  machine; the binary reports a hash of its embedded error table, which must equal the Python table's.
* Operator history is lossy, so lifecycles come from **PD logs** (`pdlog.py`); `pd-ctl operator history` is saved only as a cross-check.
* Real-data findings that changed the design: `down_peers` never appears (down = voter on a non-Up store), `pending_peers` is
  omitted when empty, Region lists are unordered, keys are memcomparable-encoded, membership changes shorter than one snapshot
  interval are only visible through `epoch.conf_ver` (**peer_churn** events), TiKV `raftstore.max-peer-down-duration` (10 m, not
  online-modifiable) gates dead-store re-replication (spec warns), and PD logs some lines twice (deduplicated).
* A **settle gate** (no Regions missing a replica, no running operators) precedes load: a fresh cluster is still up-replicating, which
  would otherwise be measured as repartitioning. Experiments abort loudly if the cluster never settles (`--settle-timeout`).
* Two aggregator bugs found only on real output and fixed with regression tests: probe records are written at completion (late by up
  to the request timeout) so bins must stay open for a lag allowance (`late_records` is counted and flagged, never double-counted),
  and probe timestamps on bin boundaries need a 1 ms tolerance.
* `local_pipeline_check.py` (real TiDB on one machine) is reinstated as a developer tool even though v2 dropped local backends: it
  was the only way to validate the Go tool and the metrics without AWS.

**First real-TiDB result (local, 3 TiKV, 70 s, 2 of 3 TiKV paused for 21 s):** probes show 35 s of unavailability per partition
(0.50): the 21 s pause plus ~14 s of Raft re-election and recovery; every failure carries cause `client_deadline`; leader transfers
afterwards were the only repartitioning. Timings are loopback numbers, not AWS numbers.

**What the live gates must verify (known unknowns)**

* G1: IAM policy sufficiency, quotas, cross-region peering acceptance latency, NVMe mount on `i4i.2xlarge`, teardown leaves nothing.
* G2: TiUP flags and topology schema, `tiup cluster check` advisories to tolerate, PD `/members` shape, exporter versions to pin,
  time to deploy 6-30 nodes.
* G3: stopping `tikv-20160` via systemd matches `plan_rounds`' expectations; allowed gaps (`ELECTION_GAP_S`, `RECOVERY_GAP_S`) are right.
* G4/G5: whole pipeline, SIGINT mid-experiment, teardown by tag with `state.json` deleted, two deployments independent.
