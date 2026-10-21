# Plan: Execution Order #1 — Fault + price library (HW, SW, spot, planned, region)

Scope: October Goals, execution order #1, subtasks 1.1 (planned weekly downtime)
and 1.2 (region outage frequency/duration).

Status: v2.1, implementation in progress (see §8). Created 2026-10-19, rewritten 2026-10-20
after the decisions in §0. v2.1 changes: region outages from the nearest Azure
region only; planned-downtime and recovery durations from Ford et al. OSDI'10;
8 vCPU; `i4i.2xlarge`; AWS credentials available; σ_W = 12 h.

## 0. Decisions (2026-10-20)

| Topic | Decision |
|---|---|
| Target cloud | **AWS.** Experiments run on AWS: AWS regions, AWS prices, AWS cross-region transfer costs. Azure data is still recorded as reference sources. |
| Regions | **All US AWS regions**, enumerated programmatically from the AWS price list / `describe-regions`. Today that is the 4 commercial regions us-east-1, us-east-2, us-west-1, us-west-2. GovCloud (us-gov-east-1, us-gov-west-1) is recorded but excluded: it is a separate partition you cannot deploy to from a standard account. Local/Wavelength Zones are excluded. |
| Fault sources | Record **every** fault-rate source, then build a **library of machine classes** from them. |
| Region outages | **Azure only.** Each AWS US region uses the outage rate and duration of its **closest Azure region** (great-circle distance). AWS region outage data is **not** used (§7.8). |
| Storage | **Local NVMe (instance store) everywhere.** Disk failure, host loss, and spot interruption all mean permanent data loss for that node. Managed/EBS-backed variants are not built (the schema still allows them). |
| Planned downtime | Gap **W ~ Normal(7 d, σ_W = 12 h)**; random per-node phase. Duration **D = the "planned reboot" unavailability-duration distribution from Ford et al., OSDI'10**, reverse-engineered from the paper's CDF figure (§7.7, §7.14). Ford data is used wherever it fits (recovery durations, reference rates). |
| Deployment hardware | Every experiment runs on **one identical 8 vCPU x86 AWS instance type: `i4i.2xlarge`** (fallback `m6id.2xlarge`), see §3.5. Heterogeneity is *emulated*: each node is assigned a library machine class, and the emulator injects that class's faults. |
| "8-core" | **8 vCPU** (AWS's unit, i.e. `*.2xlarge`). |
| On-demand prices | Built from the **distribution of current on-demand $/h of all 8 vCPU x86 AWS instance types in US regions**, assigned to machine classes two ways: **independent** of reliability, and **reliability-ranked** (more reliable → more expensive). Both are kept; the real-world trade-off is assumed to lie between them (§3.4). |
| AWS credentials | Available on your machine: enables `describe-spot-price-history` and the AWS Health API for our own instances. |
| Use | The library is shared by **all** experiments, in **simulation** (Markov/ILP, Rust MC) and **emulation** (fault injection on real TiDB nodes). |
| GPU | Out of scope (no GPU machines in deployments). |
| Private data | None. Azure subscription available via `az` (logged in on your machine). |

Practical constraint: the Cowork sandbox has no outbound network, so every
fetch script is committed and run on your machine (`az`, `aws`, curl).

## 1. What exists today (survey)

- Every study uses hand-picked `VM_PROFILES` in
  `notebooks/raft_markov_quality_benchmark.py` (Standard / Spot / Unreliable:
  MTBF 30d, recovery 20 min, MTTDL 3y / 1d / 1y, $0.10 / $0.03 / $0.05).
  None of these numbers are sourced. All nodes sit in `region="us-east"`.
- `NodeConfig` (`powder/simulation/node.py`) carries `region`, `cost_per_hour`,
  `failure_dist`, `recovery_dist`, `data_loss_dist`, `log_replay_rate_dist`,
  `snapshot_download_time_dist`, `spawn_dist`.
- Region outages exist in both engines (`NetworkConfig` in Python,
  `NetworkConfigSpec` in `rust/powder-mc/src/config.rs`), with one outage /
  duration distribution shared by all regions.
- A weekly periodic-hazard model exists only as a notebook subclass
  (`WeeklyWindowSimulator`); the Rust engine is the production MC path.

## 2. Data sources (AWS target; Azure for region outages)

No provider publishes an absolute on-demand VM fault rate (§7.13), so fault
numbers are assembled from many sources with recorded provenance. Prices and
spot data come directly from AWS.

| Parameter | Primary (AWS) | Reference / fallback |
|---|---|---|
| On-demand $/h, all 8 vCPU x86 types × US regions | AWS Price List **Bulk API** (public JSON, no credentials): `pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonEC2/current/<region>/index.json` | — |
| Cross-region / cross-AZ / egress $/GB | Same per-region EC2 price files (data-transfer SKUs; attribute names verified at fetch time) | AWS pricing page (authoritative if the two disagree) |
| Spot interruption band per type × region | **AWS Spot Advisor** JSON (`spot-bid-advisor.s3.amazonaws.com/spot-advisor-data.json`, public): "rate at which Spot has reclaimed capacity during the trailing month", bands <5%, 5–10%, 10–15%, 15–20%, >20% | SpotLake 2025 history; Azure ARG eviction buckets (reference) |
| Spot price history | SpotLake AWS open data (all of 2025, 10-min, public S3); `aws ec2 describe-spot-price-history` (90 d, needs AWS credentials) | Azure ARG spot price history (reference) |
| HW transient / permanent rates | Literature + datasets (Vishwanath, Birke, LANL, Google 2019, Backblaze), capped by the AWS EC2 instance-level SLA (99.5%) | Azure docs/SLA, Narya/AFPD (reference) |
| SW failure rate | LANL root-cause mix (SW:HW ratio) | Blue Waters cause mix |
| Planned downtime | Gap: decision §0 (weekly, σ 12 h). Duration: Ford et al. OSDI'10 "planned reboot" CDF, digitized (§7.14) | Ford planned-reboot *rate* (alternative gap); Google 2019 REMOVE→ADD (reference) |
| Recovery durations (`hw_reboot`, `sw`) | Ford et al. "unplanned reboot" and "node restart" duration CDFs, digitized (§7.14) | exec #10 measurements later (`@v2`) |
| Region outages | **Azure Post Incident Reviews (≈5 yrs)**, mapped to each AWS region's closest Azure region (§7.8) | Gunawi SoCC'16 (cross-check only). AWS outage data deliberately not used |

## 3. The library (main deliverable)

### 3.0 Concept

1. Every fault source becomes **source records** and one or more **fault
   profiles**.
2. Fault profiles plus prices become **machine classes**: the "machines" the
   ILP chooses between and the emulator imitates.
3. A **scenario** picks machine classes, a region-outage profile, a
   planned-downtime profile, a fault level, a price assignment, and a
   deployment binding.
4. **Simulation** turns a scenario into `NodeConfig`s / powder-mc jobs.
   **Emulation** turns the same scenario into a seeded **fault trace** that is
   injected into identical AWS instances (§3.5).

So the canonical set (`canonical-{low,central,high}`) is just three
scenarios, not a separate artefact.

### 3.1 Layers

1. **Sources** (`data/fault_library/sources/<id>.json`): one per paper,
   dataset, measurement, or explicit assumption. Holds bibliographic info,
   population/exposure, and each extracted measurement: verbatim metric,
   value, CI, event count, page/table, verification status (V/S/U, §7.11).
   Numbers only; never loaded by the simulator.
2. **Fault profiles** (`data/fault_library/profiles/<kind>/<id>.json`),
   each with `derived_from` (source IDs) and `built_by` (script):
   - `node`: per-cause rates (`hw_reboot`, `hw_host_loss`, `disk`, `sw`,
     `spot_interrupt`, `pause`) as `{low, central, high}` triples, per-cause
     duration dists, hazard shape (exponential or Weibull k), optional frailty
     `a`, and `spawn` (replacement launch time). Tier tag `cloud` or
     `reference` (§3.2).
   - `component`: one part (drive). Used only as an ingredient of node
     profiles; can never be a machine class.
   - `planned_downtime`: period dist, duration dist, phase rule, `graceful`
     flag (§7.7).
   - `region_outage`: per-AWS-region rate posterior and duration dist, each
     with the Azure region it was mapped from and the distance (§7.8).
   - `correlation`: reserved for exec #3.
3. **Prices** (`data/fault_library/prices/aws/…`; bulky series in `raw/`),
   keyed by `pool_key = aws:<instance_type>:<region>[:<az>]:<tier>:linux`:
   - `compute_ondemand`: $/h, `fetched_at`, exact query/file version.
   - `compute_spot_series`: $/h time series (pointer + content hash) plus
     summary stats, time-aligned with the Spot Advisor band series for the
     same pool.
   - `transfer`: $/GB for intra-AZ, inter-AZ, each directed US region pair
     (12 pairs for 4 regions), and internet egress.
   - `price_distribution`: the empirical distribution of on-demand $/h for
     8 vCPU x86 instance types in US regions, with variants (§3.4).
   - `price_assignment`: a frozen mapping machine class → base $/h produced
     by one assignment method + seed (§3.4).
4. **Machine classes** (`data/fault_library/machines/<id>.json`): what the
   ILP chooses between. Fields: `fault_profile`, `tier`
   (ondemand/spot), `storage_mode: local`, `regions[]`, and a price
   reference: either `native` (a real AWS `pool_key`, e.g. spot classes) or
   `assigned` (looked up from a `price_assignment`, which is how reference
   and on-demand cloud profiles get prices). Hardware specs (`vcpu`,
   `mem_gb`, `storage_gb`) come from the deployment instance type, since all
   experiments run on it.
5. **Scenarios** (`data/fault_library/scenarios/<id>.json`): machine
   classes, one `region_outage` profile, one `planned_downtime` profile, fault
   `level` (low/central/high), `price_assignment` ID (or `none` for
   availability-only), `price_policy` for spot series (`latest`,
   `mean_over_window`, `at_timestamp`), and a `deployment` block
   (instance type, regions, AZ layout).

### 3.2 Initial contents

Fault profiles:

| Profile ID | Kind / tier | Price route | From |
|---|---|---|---|
| `legacy/vm-standard`, `legacy/vm-spot`, `legacy/vm-unreliable` | node / cloud | legacy constants | current `VM_PROFILES`, unchanged so existing studies still reproduce |
| `cloud/aws-ondemand-nvme` | node / cloud | assigned (§3.4); actual price of deployment type also recorded | §7.4 bottom-up from the reference/component profiles, capped by AWS SLA |
| `spot/aws-<type>-<region>` | node / cloud | **native** AWS spot pool | Spot Advisor band + SpotLake (§7.4) |
| `spot/azure-<sku>-<region>` | node / reference | assigned (if used) | ARG eviction buckets (cross-cloud reference) |
| `vm/birke14-vm`, `vm/birke14-pm` | node / reference | assigned | Birke DSN'14 (once the PDF is read) |
| `host/vishwanath10` | node / reference | assigned | SoCC'10 |
| `cluster/google2019-<cell>` | node / reference | assigned | ClusterData2019 (§7.12) |
| `hpc/lanl05-<system>`, `hpc/lanl05-pooled` | node / reference | assigned | LANL (§7.12) |
| `disk/backblaze-hdd-2025`, `disk/backblaze-ssd-2023`, `disk/pooled` | component | never priced | Backblaze (+ Schroeder & Gibson, Vishwanath); feeds the `disk` cause |
| `cluster/ford10-google-storage` | node / reference | assigned | Ford et al. OSDI'10: per-cause event rates (digitized) + per-cause duration CDFs (§7.14) |
| `planned_downtime/weekly-ford10` (**default**) | planned_downtime | n/a | gap N(7 d, 12 h); duration = Ford planned-reboot CDF |
| `planned_downtime/ford10-rate` | planned_downtime | n/a | gap mean from Ford's planned-reboot rate, σ 12 h; Ford duration (sensitivity) |
| `planned_downtime/google2019` | planned_downtime | n/a | Google REMOVE→ADD (reference/sensitivity) |
| `region_outage/aws-us-from-azure` (+ `-incl-control-plane`, `-pooled-us`) | region_outage | n/a | Azure PIRs mapped to nearest Azure region (§7.8) |
| `synthetic/*` | node / reference | assigned | stress profiles for tests/sensitivity |

Every node profile, reference ones included, can now become a priced machine
class through a price assignment. Only `component` profiles are never priced.

Price data ingested:

| Data | Source | Kind |
|---|---|---|
| On-demand $/h, all 8 vCPU x86 types, US regions | Bulk API per-region EC2 files | `compute_ondemand` + `price_distribution` |
| Transfer $/GB (intra-AZ, inter-AZ, inter-region pairs, egress) | same files | `transfer` |
| Spot band per type × region (current) | Spot Advisor JSON; **daily snapshot cron** to accumulate history | band series |
| Spot price + band + placement score, 2025 | SpotLake open data (S3, no account) | `compute_spot_series` |
| Spot price, last 90 d | `describe-spot-price-history` (if AWS credentials available) | `compute_spot_series` |
| Azure spot price/eviction (reference) | ARG `SpotResources` via your `az` login | reference series |
| Legacy $0.10 / $0.03 / $0.05 | `VM_PROFILES` | `compute_ondemand` (`provider: legacy`) |

### 3.3 Which machine classes have which prices

| Machine class built from | Price | Why |
|---|---|---|
| `spot/aws-*` | real AWS spot price of that pool (series → scalar via `price_policy`) | spot price and interruption are measured together on AWS |
| `cloud/aws-ondemand-nvme` and every `reference` node profile | **assigned** from the 8 vCPU x86 on-demand distribution (§3.4) | no source pairs these fault rates with a price |
| `component` (Backblaze) | none, ever | a drive is not a machine; it only feeds the `disk` term |

The generated catalogue (`docs/fault_library.md`) lists every machine class
with its fault profile, price route (native / assigned + method), and
single-node MTBF, MTTR, MTTDL, and availability.

### 3.4 On-demand price distribution and the two assignment strategies

**Step 1: price distribution.**

- Input: Bulk API per-region EC2 files for every US region in §0.
- Filter: `productFamily = Compute Instance`, `vcpu = 8`, x86-64 processors
  (Intel/AMD; Graviton excluded), Linux, shared tenancy, no pre-installed
  software, `capacitystatus = Used`, OnDemand term. GPU / accelerator / FPGA /
  Mac families are excluded.
- Each point is one (instance type, region) with $/h and tags: family,
  current-generation flag, burstable flag (t3/t3a), instance-store size/NVMe.
- Stored as `price_distribution/aws-us-8vcpu-x86-ondemand@<fetch-date>`:
  the points, empirical CDF, summary (min, p5, p25, p50, p75, p95, max,
  mean, CV), and a fitted lognormal for smooth sampling. Variants:
  `current-gen`, `non-burstable`, and `instance-store` (the subset that
  matches our local-NVMe assumption). Default = `current-gen, non-burstable`.
- **Region factor**: `f_r = price(deployment type, r) / price(deployment type,
  us-east-1)`. Assigned prices are base (us-east-1-equivalent) prices × `f_r`.

**Step 2: reliability score per machine class** (lower = more reliable),
using central values:

`score_m = H · Σ_{i ∈ transient causes} λ_i · E[D_i]  +  H · λ_P · R`

- `H = 8766` h/yr, so the score is expected unavailable hours per node-year.
- `R` is a reference rebuild time for a permanent loss (default 1 h;
  replaced by exec #10 measurements).
- Planned downtime and region outages are excluded because they are the same
  for every machine class in a scenario.
- Rank robustness: report Kendall τ between rankings with `R = 0.25 h` and
  `R = 4 h`. If τ < 0.9, the ranked assignment is reported for both.

**Step 3: draw prices.** For n machine classes, take stratified quantiles
of the chosen distribution, `p_k = F^{-1}((k − 0.5)/n)` for k = 1…n. This
is deterministic and spans the distribution. iid draws are a sensitivity
variant.

**Step 4: assign** with a Gaussian-copula rule, which covers both strategies
and everything between:

- `s_m = Φ^{-1}((rank_m − 0.5)/n)`, where rank 1 = *least* reliable
  (highest score).
- `z_m = ρ·s_m + √(1−ρ²)·ε_m`, with `ε_m ~ N(0,1)` from a recorded seed.
- Machine class with the k-th smallest `z` gets `p_k`.
- **ρ = 0 → independent**: a seeded random permutation, unrelated to
  reliability.
- **ρ = 1 → reliability-ranked**: the most reliable class gets the highest
  price (comonotone).
- Optional ρ = 0.5 midpoint for a sweep.

Both endpoint assignments (ρ = 0 and ρ = 1) are always produced. Because
ρ = 0 is random, it is generated for several seeds (e.g. 20), and results
report the spread across seeds. Each assignment is frozen as a
`price_assignment` entry recording method, ρ, seed, distribution ID, the
score inputs, and the realised Spearman correlation between price and
reliability. Paper framing: the true price/reliability relation lies between
ρ = 0 and ρ = 1, so claims must hold at both ends.

### 3.5 Simulation and emulation contract

All experiments run on one identical 8 vCPU x86 instance type with local
NVMe: **`i4i.2xlarge`** (Intel, storage-optimized; expected 64 GiB RAM and
one ~1.9 TB NVMe instance-store volume, to be confirmed from the price list in
Step B). Why: TiKV is storage-bound and RF-replicated, so the large NVMe per
vCPU leaves headroom for exec #6's per-machine storage constraint. A
storage-optimized family also keeps disk behaviour consistent across runs.
**Fallback `m6id.2xlarge`** (general purpose, smaller ~474 GB NVMe) if
`i4i.2xlarge` is missing from any US region or capacity is a problem. Each
node is assigned a machine class; its faults are injected.

- **Trace generator** (`powder/faults/trace.py`):
  `trace(scenario, node→machine-class map, horizon, seed) → JSONL` of events
  `{node, region, kind, cause, start, duration}`. It uses exactly the
  distributions the simulator uses, so simulation and emulation see
  statistically identical fault processes. With a shared seed, emulation runs
  are reproducible.
- **Emulator actions** (implemented in exec #8; the library only defines
  them):

  | Event kind | Causes | Emulator action | Data |
  |---|---|---|---|
  | `transient` | `hw_reboot`, `sw`, `pause` > election timeout | stop TiKV/TiDB process (or reboot instance) for `duration`, then restart | kept |
  | `permanent` | `disk`, `hw_host_loss`, `spot_interrupt` | kill process and wipe the NVMe data dir (or terminate instance); the replacement path launches a new node after `spawn` | lost |
  | `planned` | weekly maintenance | stop for `duration`; graceful leader eviction first if `graceful = true` | kept |
  | `region` | region outage | network-isolate every node in the region (security group / iptables) for `duration` | kept |

- **Simulator side**: today the Rust MC samples internally. A trace-replay mode
  (MC consumes the same JSONL) is a follow-up under Goal 2.2 ("simulator and
  deployment agree on node selection"). The schema is fixed now so it does not
  have to change later.
- **Prices in emulation**: the AWS bill reflects the single deployment type.
  Experiment cost is computed from the scenario's *assigned/native* prices per
  machine class, plus transfer $/GB from `transfer` entries.

### 3.6 Code and files

```
powder/faults/
  schema.py      # source/profile/price/assignment/machine/scenario schemas + validation
  library.py     # FaultLibrary.load(); get(); query(kind=, tags=); lineage(); content hashes
  prices.py      # price entries, spot series -> scalar (price_policy), transfer lookup,
                 #   price distribution, region factors
  assign.py      # reliability score, stratified quantiles, Gaussian-copula assignment
  convert.py     # machine class + level + region -> NodeConfig; scenario -> ILP catalog,
                 #   NetworkConfig, PlannedDowntimeSpec, powder-mc job fragment
  trace.py       # seeded fault trace for emulation (and future MC replay)
  planned_downtime.py  # schedule generator (shared by MC, trace, emulator)
  fit.py         # unit conversion, random-effects pooling, NB frailty, Gamma-Poisson
                 #   region fit, duration fits, CDF-curve fits for digitized data
  regions.py     # AWS region -> nearest Azure region mapping (great-circle)
scripts/fault_library/
  fetch_aws_prices.py        # Bulk API: on-demand + transfer for US regions
  fetch_spot_advisor.py      # Spot Advisor JSON (cron daily to build history)
  fetch_spotlake.py          # SpotLake AWS 2025 (S3, --no-sign-request)
  fetch_spot_history.py      # describe-spot-price-history (needs AWS creds)
  fetch_azure_reference.py   # ARG spot buckets/prices via az (reference only)
  build_region_incidents.py  # helpers for curating the Azure PIR incident CSV
  azure_locations.py         # `az account list-locations` -> Azure region coordinates
  digitize_figure.py         # PDF figure -> (x, F) points: vector extraction first,
                             #   raster fallback (§7.14)
  extract_lanl.py, extract_google2019.py, ...
  build_profiles.py          # sources -> profiles (deterministic)
  build_machines.py          # profiles + prices/assignments -> machine classes
  render_catalog.py          # -> docs/fault_library.md
data/fault_library/
  raw/  sources/  profiles/  prices/  machines/  scenarios/
  region_incidents_azure.csv
  region_map.json            # AWS region -> Azure region, distance, coordinates used
  digitized/ford10/          # per-figure calibration JSON + extracted CSV + overlay PNG
docs/reading/Ford_OSDI10.pdf  # present (added 2026-10-20)
docs/fault_library.md     # generated catalogue
docs/calibration.md       # methodology + extraction rows
tests/test_fault_library.py
```

### 3.7 Rules

- **Immutable IDs**: an entry used in a committed result is never edited;
  changes create `<id>@v2`. Results record entry IDs plus content hashes.
- **Provenance enforced**: tests fail if `derived_from` does not resolve or
  cites a measurement marked **U**.
- **Numeric triples** (`low ≤ central ≤ high`); each parameter declares which
  end is pessimistic (high for rates/durations, low for Weibull k).
- **Every entry loads**: every profile, machine class and scenario converts to
  `NodeConfig`, powder-mc JSON and a trace, and runs a tiny MC job.
- **Prices dated and explicit**: every price has `fetched_at` and the exact
  file/query. A scenario with a cost objective fails validation if any machine
  class lacks a price in one of its regions. No silent defaults, no $0.
- **Units**: USD; $/h compute; $/GB transfer; seconds for all durations.
- **Distributions** use powder-mc's `DistributionSpec` JSON (plus the
  optional `lognormal`, `mixture` and `empirical` additions, §7.6, §7.14).

## 4. Work breakdown (each step is its own commit)

**Step A — Library skeleton.** `powder/faults/` schema/library/convert, the
layer layout, `legacy/*` profiles + legacy prices + legacy machine classes +
`legacy` scenario, `render_catalog.py`. Tests: schema, provenance, every entry
loads, and the `legacy` scenario reproduces today's `VM_PROFILES`
`NodeConfig`s exactly.

**Step B — AWS prices.** `fetch_aws_prices.py` (on-demand + transfer, all US
regions), `price_distribution` with variants, region factors, `prices.py`.
Confirm `i4i.2xlarge` specs and availability in all US regions from the
fetched files (fallback `m6id.2xlarge`). Commit fetched files with
`fetched_at`.

**Step C — Spot.** Spot Advisor snapshot + daily cron (history cannot be
fetched retroactively), SpotLake 2025 ingest, `describe-spot-price-history`
(90 d, using your AWS credentials), `spot/aws-*` profiles with native prices.
Azure ARG reference fetch via `az` (optional, cheap).

**Step C2 — Ford digitization (§7.14).** PDF present; all plots are vector,
so exact coordinates are extracted. Digitize Fig 1, Fig 2 (per-cause duration
CDFs) and Fig 3 (per-cause event rates), and record Tables 1–2. Produces the `ford10` source record, the `empirical` DistributionSpec
(Rust + Python), and `cluster/ford10-google-storage`. Steps D (recovery
durations) and E (planned downtime) depend on this.

**Step D — HW/SW fault profiles from literature.** Source records,
`host/vishwanath10`, `vm/birke14-*`, `disk/*`, and composed
`cloud/aws-ondemand-nvme` (§7.4, §7.9).

**Step D2 — Raw-dataset profiles** (downloads on your machine):
`hpc/lanl05-*`, `cluster/google2019-*`, `planned_downtime/google2019`.
Independent of other steps.

**Step E — Subtask 1.1: planned downtime.** `planned_downtime.py`,
`planned_downtime/weekly-ford10` (default) and `planned_downtime/ford10-rate`;
Rust: optional `PlannedDowntimeSpec` and
`PLANNED_DOWNTIME_START/END` events. Planned downtime never triggers node
replacement (it is announced), whatever its duration. Markov/ILP folds it into
an effective transient rate `1/E[W]` with mean duration `E[D]` (documented
approximation; weekly-windows study is the error template). Tests in §7.10.

**Step F — Subtask 1.2: region outages.** Azure PIR incident CSV, Azure
region coordinates via `az`, AWS→nearest-Azure mapping (`region_map.json`),
empirical-Bayes fit over all Azure regions, `region_outage/aws-us-from-azure`
(+ variants). Engine change: per-region outage distributions in
`NetworkConfigSpec` (backward compatible) and a `lognormal` `DistributionSpec`
if needed.

**Step G — Machine classes, price assignments, scenarios, trace.**
`assign.py` (ρ = 0 over 20 seeds, ρ = 1, optional ρ = 0.5), `build_machines.py`,
`canonical-{low,central,high}` scenarios with deployment blocks, `trace.py`.

**Step H — Verification.** pytest + `cargo test`; library-wide MC rate checks;
trace-vs-MC statistical agreement; independent re-derivation of profiles from
source records (subagent).

## 5. Why this does not block later tasks

- #2 (experiment size/length): sizing runs over library scenarios and both
  price assignments. Planned downtime dominates in a 24 h run (§7.7), which is
  useful to know up front.
- #3 (correlated failures): per-region outage spec + `multi_region` tags +
  reserved `correlation` kind. Ford et al. also quantify correlated failure
  bursts and rack-level domains (Table 2 lists a rack MTTF), so the same
  digitization pipeline (§7.14) can feed exec #3.
- #4 (region-dependent download speed): `transfer` is keyed by directed
  region pair; a measured bandwidth matrix (exec #10) sits next to it.
- #5 (reject single-region placements): region set and outage model come from
  the scenario.
- #6 (storage + GB movement): `storage_gb` from the deployment type;
  intra/inter-region $/GB from `transfer`.
- #7/#8 (TiDB on AWS, failure emulator): scenario `deployment` block +
  `trace.py` + emulator action table (§3.5).
- #9 (PD queries ILP): ILP catalog comes from `convert.py`.
- #10 (timing benchmarks): replaces placeholder recovery, spawn, and rebuild
  (`R`) values via new `@v2` profiles.
- Goal 5 (SW update scheduling): planned downtime is its own event type, so a
  chosen schedule can replace the random phase.

## 6. Open questions

Resolved 2026-10-20:

- no private dataset; Azure subscription available;
- local NVMe, with disk failure = permanent loss;
- all US regions; AWS target;
- library used for all experiments in both simulation and emulation;
- 8 vCPU; deployment type `i4i.2xlarge` (my pick, fallback `m6id.2xlarge`);
- AWS credentials available;
- σ_W = 12 h;
- region outages from the closest Azure region only;
- planned-downtime duration and recovery durations from Ford et al. OSDI'10.

Still open (small):

1. ~~us-east-1 tie~~ Resolved: us-east-1 → eastus2, no pooling.
2. ~~Ford PDF~~ Resolved: available at `docs/reading/Ford_OSDI10.pdf` and
   inspected (§7.14).
3. ~~Spot probe budget~~ Resolved: spot probe removed from the plan.

No open questions remain.

## 7. Methodology: from sources to fault curves

For every machine class and region, produce what the simulator/ILP consume:

| Output | Simulator field | Form |
|---|---|---|
| Transient hazard `λ_T` | `failure_dist` | exponential (default) or Weibull with equal mean |
| Recovery time `D_T` | `recovery_dist` | cause-weighted mixture (§7.6) |
| Permanent-loss hazard `λ_P` | `data_loss_dist` | exponential |
| Frailty `Z ~ Gamma(a, 1/a)` | multiplier on λ_T, λ_P | optional |
| Spawn time | `spawn_dist` | placeholder until exec #10 |
| Planned downtime | `PlannedDowntimeSpec` | §7.7 |
| Region outages `λ_R(r)`, `D_R` | `NetworkConfigSpec` per region | §7.8 |

### 7.1 Event taxonomy (AWS, local NVMe)

| Cause | Data kept? | Counts toward |
|---|---|---|
| `hw_reboot`: host/instance reboot in place (instance store survives a reboot) | yes | λ_T |
| `hw_host_loss`: host failure/retirement forcing the instance onto new hardware (stop/start or terminate) | **no** (instance store lost) | λ_P |
| `disk`: NVMe drive failure | **no** | λ_P |
| `spot_interrupt`: spot reclaim (terminate, stop, or hibernate all lose instance store) | **no** | λ_P |
| `sw`: OS / TiKV / TiDB crash fixed by restart | yes | λ_T |
| `pause`: platform freeze; ignored if shorter than the Raft election timeout (TiKV default 10 s) | yes | λ_T if longer |
| `planned`: weekly maintenance (§7.7) | yes | separate process |
| `region`: region outage (§7.8) | yes | separate correlated process |

Causes are competing risks, so `λ_T` and `λ_P` are exact sums and
P(cause i | failure) = `λ_i/Σλ`. Exact AWS recovery behaviour for
instance-store instances (e.g. whether automatic recovery applies) is checked
in exec #10 and recorded as a source.

### 7.2 Extraction protocol

Each source gets a `sources/*.json` record (human-readable row in
`docs/calibration.md`): population and exposure, metric *verbatim* with
page/table, value and CI, event count `n`, which §7.1 causes it covers, and the
conversion applied. Numbers are taken from the primary document, not
summaries. No number enters a profile without such a record.

### 7.3 Unit normalisation (→ events per node-hour, `H = 8766` h/yr)

- **Fraction with ≥1 event per year (p)**: `λ = −ln(1−p)/H`.
- **Events per unit-year (m)**, e.g. Backblaze AFR (failures / drive-days ×
  365): `λ = m/H`. Each record states whether it is p or m; when both are
  given, the gap is the repeat-failure signal (§7.5).
- **Host → instance**: `λ_inst = λ_host·(1 − f_LM)`, where `f_LM` is the share
  of host faults hidden by live migration/live update (bracket 0.5 / 0.25 /
  0); the remainder splits into `hw_reboot` vs `hw_host_loss` by `f_loss`
  (bracket 0.2 / 0.5 / 0.8).
- **AWS spot band `[q_lo, q_hi]`**: defined as the rate at which Spot reclaimed
  capacity over the trailing month. Default reading: monthly fraction
  interrupted, `λ = −ln(1−q)/730` per hour (5% → mean ≈ 1.6 years of runtime
  per instance; 20% → ≈ 4.5 months). AWS does not publish the denominator. This
  reading is taken as given, and each band's low/mid/high is stored for
  sensitivity sweeps (no probe, §7.4).
- **Azure ARG bucket** (reference only): the ambiguous per-hour vs
  per-28-day readings (≈670× apart) are both stored.

### 7.4 Rate estimation per cause

**Disk**: DerSimonian–Laird random-effects pooling of ≥3 comparable
estimates (Schroeder & Gibson, Vishwanath, Backblaze SSD/HDD) on the log
scale. `y_s = ln λ_s`, `v_s = 1/n_s` (or 0.25 if `n_s` is unknown, noted);
`w_s = 1/v_s`; `Q = Σw_s(y_s − ȳ_w)²`;
`τ² = max(0, (Q−(k−1))/(Σw_s − Σw_s²/Σw_s))`; `w*_s = 1/(v_s+τ²)`;
`μ̂ = Σw*_s y_s/Σw*_s`; `SE = (Σw*_s)^{-1/2}`. Central `exp(μ̂)`, low/high
`exp(μ̂ ∓ 1.645√(SE²+τ²))`. Per-source profiles are kept; the pool is an
extra `disk/pooled` profile. Datacenter NVMe vs Backblaze boot SSDs is a
known proxy gap, noted on the profile.

**Host faults (`hw_reboot`, `hw_host_loss`)**:
1. Bottom-up: non-disk host repair rate (Vishwanath; Google 2019 failure
   removals; Birke VM-vs-PM ratio once read), via §7.3.
2. Top-down cap: AWS EC2 **instance-level SLA 99.5%** monthly → at most
   ≈43.8 h unplanned downtime per instance-year. Check
   `H·(λ_reboot·E[D_reboot] + λ_host_loss·E[rebuild]) ≤ 43.8 h`. This bound
   is loose (the SLA is a floor), so `high` is clipped to it only if
   exceeded, and that is logged.
3. Azure facts (99.9% single-VM SLA, ~30 s pause ≤ once per 18 months,
   Narya/AFPD relative reductions) are recorded as reference sources and used
   only as cross-checks.

**SW (`sw`)**: `λ_sw = ρ_SW · λ_HW`, with `ρ_SW` the pooled LANL
software:hardware failure ratio (random effects across systems). Ford's
node-restart vs unplanned-reboot rates (§7.14) give a second, more
cloud-like estimate of the ratio, entered into the same pool. Durations come
from Ford's node-restart CDF (§7.6).

**Spot (`spot_interrupt`)**: per (type, region) from the Spot Advisor band
(§7.3), with band history from daily snapshots and SpotLake 2025.
The band's low/high ends are the uncertainty range. No spot probe is run
(removed 2026-10-20). At realistic bands a week-long probe of 6,720
instance-hours expects only ≈0.5–2 interruptions, and about 33k–142k
instance-hours would be needed for ~10 events. Spot behaviour is emulated on
on-demand machines anyway, so only the rates are needed. Optional free
cross-check: SpotLake's follow-up "Making Cloud Spot Instance Interruption
Events Visible" (WWW'24), if it compares observed interruptions with the
bands (to be confirmed). Co-interruption statistics for exec #3 are deferred
to exec #3.

**Pause**: dropped if shorter than the election timeout; otherwise folded
into `hw_reboot` with its own short duration.

### 7.5 Shape

- **Default exponential** (Markov/ILP needs it; host age unknown to the
  instance → close to stationary).
- **Weibull variant**: `k ∈ [0.7, 0.8]` (Schroeder & Gibson FAST'07, LANL
  DSN'06), `scale = (1/λ)/Γ(1+1/k)` so the mean is unchanged; renewal at
  recovery.
- **Frailty**: from (p, P(≥2 | ≥1)) or per-node counts (LANL): negative
  binomial with `P0 = (1+μ/a)^{−a}`, `P1 = μ(1+μ/a)^{−a−1}`; solve
  `1−P0 = p`, `(1−P0−P1)/(1−P0) = r` for `(μ, a)`. Off by default; used to test
  lemon-machine robustness.

### 7.6 Recovery-time curves

- `hw_reboot`: Ford's **unplanned reboot** duration CDF (digitized, §7.14),
  used as an `empirical` distribution.
- `sw`: Ford's **node restart** duration CDF (digitized, §7.14), `empirical`.
- Both are replaced by `@v2` profiles once exec #10 measures our own timings
  (≥50 timed reboots / process restarts to TiKV-ready; MLE + KS). The Ford
  versions are kept as reference profiles.
- Permanent causes have no "recovery": the node is replaced after `spawn`,
  then re-replicates (rebuild time from the simulator's snapshot/log model).
- `recovery_dist` = mixture `Σ_{i∈T}(λ_i/λ_T)·D_i` (exact for competing
  risks); needs optional `lognormal`, `mixture` and `empirical`
  `DistributionSpec`s (Rust + Python). Markov/ILP uses `E[D_T]`.

### 7.7 Planned downtime (subtask 1.1)

- Gap `W ~ Normal(7 d, σ_W = 12 h, min_val = 1 d)` (σ_W sensitivity 0–48 h).
- Duration `D ~ F_planned`: the **planned reboot** unavailability-duration CDF
  from Ford et al. OSDI'10, digitized into an `empirical` distribution
  (§7.14). A parametric fit (lognormal or 2-component lognormal mixture) is
  stored alongside for the Markov/ILP mean and for smooth sampling.
- Alternative profile `planned_downtime/ford10-rate`: gap mean taken from
  Ford's planned-reboot *rate* (events per node per day, digitized), σ_W =
  12 h, same Ford duration. Used as a sensitivity check on the weekly
  assumption.
- Phase: first event drawn from the equilibrium residual of the renewal process
  (length-biased `W̃`, then `U·W̃`), so the planned-downtime rate is constant
  from t = 0. This refines "uniform within the first week"; for σ_W = 12 h they
  are nearly identical.
- Consequences for exec #2 (filled in once digitized): long-run planned
  downtime per node = `E[D]/E[W]`. In a 24 h run with N nodes, expect about
  **N/7 planned events** (≈1.3 for N = 9), each with duration drawn from Ford's
  planned-reboot CDF. With durations in the minutes-to-hours range, planned
  downtime will very likely dominate unplanned HW downtime.
- Planned events never trigger replacement. `graceful` defaults to false
  (abrupt stop) to match the simulator; true is a variant.

### 7.8 Region outages (subtask 1.2): Azure only, mapped to nearest region

AWS region outage data is deliberately **not** used.

1. **Azure incident list** (`region_incidents_azure.csv`): Azure Post Incident
   Reviews from the public Azure status history (≈5-year retention), e.g.
   2021-01-01 to 2026-09-30. Columns: region(s), start/end UTC, duration,
   scope (region / AZ / service-only), `compute_affected`, cause, URL.
   **Counting rule**: an incident counts toward Azure region r if VM
   connectivity in r was lost region-wide or in ≥2 AZs. Single-AZ incidents go
   to `az_incidents` (exec #3). Control-plane / single-service incidents are
   logged with `compute_affected = false` and excluded by default, with a
   variant that includes them. Multi-region incidents count once per region
   and are tagged `multi_region = true`.
2. **Exposure** `T_r`: region-years in the window (from region GA dates).
3. **Rates** via empirical-Bayes Gamma–Poisson over **all Azure regions
   worldwide** (gives a stable hyperprior). Fit `(α, β)` by maximising
   `Π_r Γ(α+n_r)/(Γ(α) n_r!) · (β/(β+T_r))^α · (T_r/(β+T_r))^{n_r}`;
   per-region posterior `Gamma(α+n_r, β+T_r)`: central = mean, low/high =
   5th/95th percentiles. If `α → ∞`, use the pooled rate.
4. **Durations `D_R`**, pooled over Azure regions (per-region counts are too
   small): MLE exponential / Weibull / lognormal, AIC choice, KS with
   parametric bootstrap (1000 reps), bootstrap 90% CIs on median and p90;
   store parametric dist + empirical quantiles.
5. **AWS → nearest Azure mapping** (`regions.py`, `region_map.json`):
   great-circle distance from each AWS region's metro (AWS does not publish
   coordinates: Ashburn VA, Columbus OH, San Jose CA, Boardman OR) to Azure
   region coordinates from `az account list-locations`. Preliminary
   distances (Azure's published coordinates):

   | AWS region | Nearest Azure | Distance | Runner-up |
   |---|---|---|---|
   | us-east-1 (N. Virginia) | **eastus2** (fixed by decision; eastus is equally close) | ≈275 km | eastus ≈275 km |
   | us-east-2 (Ohio) | eastus | ≈399 km | northcentralus ≈444 km |
   | us-west-1 (N. California) | westus | ≈68 km | westus3 ≈989 km |
   | us-west-2 (Oregon) | westus2 | ≈155 km | westus ≈924 km |

   Each AWS region maps to exactly **one** Azure region; regions are never
   pooled. us-east-1 → **eastus2** (decision 2026-10-20; eastus is the same
   distance). us-east-2 → eastus, with a `northcentralus` variant because the
   runner-up is close. `region_map.json` records these choices explicitly
   instead of recomputing them, so the mapping cannot drift if coordinates
   change.
   The AWS region inherits the mapped posterior rate and the pooled duration
   distribution. `multi_region` tags carry over: two AWS regions mapped to
   Azure regions hit by the same incident share that event (input for exec #3).
6. **Variants**: `-incl-control-plane`; `-pooled-us` (all four AWS regions
   get the pooled Azure-US rate, a check on whether the mapping matters).
7. **Bias and assumption**: public PIRs over-represent big incidents (rate
   biased low, durations high), so `high` = 95th percentile × 2 as an explicit
   under-reporting allowance. "An AWS region fails like its nearest Azure
   region" is a modelling assumption, stated as such in the paper.

### 7.9 Assembly (AWS, local NVMe)

```
λ_T(m,r) = λ_reboot(m) + λ_sw(m) + λ_pause>thr(m)
λ_P(m,r) = λ_disk(m) + λ_host_loss(m) + [spot]·λ_spot(m,r)
recovery = mixture over the λ_T causes (§7.6)
```

Implemented in `convert.py` from the per-cause rates in each node profile.
Scenarios: `canonical-{low,central,high}` (all optimistic / central /
pessimistic) × price assignment {ρ = 0 (20 seeds), ρ = 1}, plus contrasting
reference-environment scenarios (LANL-like, Google-like, spot-heavy). A
one-at-a-time tornado sweep covers `f_LM`, `f_loss`, Weibull k, spot band
band position, σ_W, `R`, region under-reporting factor, region mapping variant,
Ford-curve digitization band (§7.14), and ρ.

### 7.10 Validation

1. **Rate round-trip**: simulate 10⁴ node-years per machine class; per-cause
   counts inside the Poisson 99% CI of `λ·T`.
2. **SLA consistency**: unplanned downtime of `cloud/aws-ondemand-nvme` ≤ AWS
   instance-level allowance (43.8 h/yr).
3. **Spot bands**: each `spot/aws-*` rate lies inside its Spot Advisor band
   after the §7.3 conversion (catches conversion bugs).
4. **Planned downtime**: long-run fraction ≈ `E[D]/E[W]`; gaps pass KS
   against N(7 d, 12 h); sampled durations reproduce the digitized Ford CDF
   (KS distance below the digitization band); phases match the equilibrium
   residual.
4b. **Digitization**: checks in §7.14 (overlay, double digitization, values
   stated in the text).
5. **Trace vs MC**: per-cause event counts and durations from `trace.py`
   match the MC's sampled events statistically (same scenario, many seeds).
6. **Price assignment**: ρ = 1 gives Spearman(price, −score) = 1; ρ = 0
   seeds give a mean near 0; the price multiset always equals the stratified
   quantiles.
7. **Independent re-derivation** of every profile from its source records
   (subagent), diffed.
8. Tests 1, 4, 5 run for every library entry automatically.

### 7.11 Source inventory (2026-10-20, web search; PDFs not yet read)

Status: **V** = stated in the paper/official page per search; **S** =
third-party summary, check against the primary document; **U** = unconfirmed,
do not use.

| Source | What it gives | Absolute rate? | Role |
|---|---|---|---|
| AWS EC2 SLA | Instance-level 99.5% monthly; region-level 99.99% for multi-AZ (V) | cap only | §7.4 top-down cap |
| AWS Spot Advisor | Band definition and per-type/region bands (V) | yes (bands) | §7.3 / §7.4 spot |
| Ford et al., "Availability in Globally Distributed Storage Systems", OSDI'10 (Google) — **PDF read 2026-10-20** (`docs/reading/Ford_OSDI10.pdf`) | Data from tens of Google storage cells (1000–7000 nodes each), one year (V). Fig 1: CDF of all node-unavailability durations; "less than 10% of events last longer than 15 minutes" (V). Fig 2: duration CDF by cause (node restarts, planned reboots, unplanned reboots), one cell, **not** excluding <15 min events (V). Node restarts vary most (<1 min to >1 h) but are usually shortest; unplanned reboots have the longest mean (V). Fig 3: events per 1000 nodes per day by cause, one cell, 3 months (V). Table 1 (all cells), unavailability % avg/min/max: node restarts 0.0139/0.0004/0.1295; planned reboots 0.0154/0.0050/0.0563; unplanned reboots 0.0025/0.0000/0.0122; unknown 0.0142/0.0013/0.0454 (V). Table 2 MTTF: disk 10–50 yr, node 4.3 months, rack 10.2 yr, "inclusive of software errors and hardware failures" (V). The rest of the paper analyses events ≥15 min (V). | yes (Google storage nodes, ~2010) | planned-downtime duration, `hw_reboot`/`sw` durations, per-cause rates, `disk/pooled` point, exec #3 bursts/racks (Figs 5–8, Table 2 rack) |
| AWS Post-Event Summaries, AWS Health Dashboard | Recorded but **not used** for region outages (decision §0) | — | — |
| SpotLake (IISWC'22) | AWS 2025 spot price + band + placement score at 10 min, public S3; Azure/GCP via web (V) | yes | spot series |
| Narya OSDI'20; Pandey et al. arXiv 1910.12200; AFPD 2025 | AIR definition (interruptions per 100 VM-years), interruption taxonomy, relative reductions (−26%, −36%) (V) | no | taxonomy, Azure reference |
| Azure VM reboot doc, Azure SLA | ~30 s pause ≤ once per 18 months; single-VM 99.9% (V) | bound | Azure reference |
| Vishwanath & Nagappan SoCC'10 | >100k servers, 14 months; disks #1; 11–25% with ≥1 disk event in big clusters (V); "AFR 8%", ~2 repairs per repaired machine (S) | yes (host, 2010) | host + frailty inputs |
| Birke et al. DSN'14 | 10k VMs + PMs, 1 yr; VMs fail less and recur less (V); numbers need PDF | yes (once read) | VM-level reference |
| Backblaze | HDD 2025 AFR 1.36%, lifetime 1.30%; SSD boot drives mid-2023 Q1 0.96%, Q2 1.05% (V); AFR is a rate (failures/drive-days × 365) | yes (drive) | `disk/*` |
| Schroeder & Gibson FAST'07 | Time between disk replacements Weibull k 0.7–0.8 (V) | shape | §7.5 |

### 7.12 HPC and cluster datasets

Absolute HPC rates do not transfer to cloud instances; dimensionless
quantities (cause mix, hazard shape, repair-time spread, repeat-failure
heterogeneity, co-failure ratios) transfer better. Every dataset is still
recorded as a reference profile, and any of them can become a priced machine
class via §3.4.

| Dataset | Raw data? | What it gives (V unless noted) | Used for |
|---|---|---|---|
| LANL (Schroeder & Gibson DSN'06/TDSC'09), via CFDR / FTA `lanl05` | yes | 22 systems, 1996–2005, ~23k failures, 4,750 machines; root cause HW/SW/network/environment/human; Weibull k 0.7–0.8; lognormal repair times, mean <1 h to >1 day | `ρ_SW`, per-cause repair σ, frailty `a`, co-failure inflation (exec #3) |
| Failure Trace Archive (Javadi et al. JPDC'13) | yes | many traces in one schema | cross-system consistency of shapes |
| Google ClusterData2019 | yes (BigQuery/GCS) | 8 cells × ~12k machines × 31 days ≈ 8,000 machine-years; REMOVE→ADD ≈ maintenance, REMOVE without ADD ≈ failure | reference node profile; `planned_downtime/google2019` (sensitivity vs the Ford-based default); rack co-removal via `switch_id` |
| Blue Waters (Di Martino et al. DSN'14) | not public | HW = 42% of failures, 23% of repair time | cause-mix cross-check |

GPU clusters (Llama 3, Meta RSC) were reviewed and excluded (§0).

Extraction: LANL per-system cause counts → `ρ_SW` (pooled); Weibull fits to
per-node gaps (last gap censored); lognormal repair fits per cause; NB frailty
on per-node counts; co-failure within 10 min vs independence baseline
`1 − exp(−(N−1)·λ·10 min)`. Google: pair REMOVE→next ADD per machine
(durations; unpaired = censored at 31 days); per-machine maintenance frequency;
group by `switch_id`.

### 7.13 VM-level fault rates (any cloud)

No public dataset gives a modern on-demand VM interruption rate (searched
AWS, Azure, GCP, Alibaba, literature). The on-demand profile is therefore
derived (§7.4) and carries wide bands. Ways to tighten it:

- **Instrument every instance we run** (exec #7 deployment, exec #10
  benchmarks): heartbeat every 5 s to a collector in another
  region, boot-ID reboot detection, AWS scheduled-event / Health labels.
  Event = gap > 10 s.
- **Exposure maths** (90% Garwood, multiple of the estimate): n=3 →
  [0.27×, 2.6×]; n=5 → [0.39×, 2.1×]; n=10 → [0.54×, 1.7×]; n=20 →
  [0.66×, 1.45×]; zero events in T instance-years → 95% upper bound 3/T. The
  realistic outcome is an upper bound plus a few labelled events with
  durations.
- **Free history**: AWS Health API keeps account events about 90 days; export
  from any long-running accounts you or colleagues have.

Implication: unplanned on-demand faults are rare and planned downtime
dominates in a 24 h run, and the emulator injects faults at library rates
either way. So results must hold across the `{low, central, high}` band and
across both price assignments, not at a single point estimate.

### 7.14 Reverse-engineering the Ford et al. CDFs

Goal: turn Ford et al.'s published figures into distributions the library
can sample, with a stated digitization error.

**PDF inspection (2026-10-20).** `docs/reading/Ford_OSDI10.pdf` is a
pdfTeX file with **no embedded raster images**: every plot is vector paths. So
digitization reads exact path coordinates, and the only error is axis
calibration (§ "Error" below). What was found:

| Item | Page | What is in the PDF | Colour / series mapping |
|---|---|---|---|
| **Fig 1**: CDF of all node-unavailability durations (tens of cells, 1 year) | 2 | one path, ~1,000 points | red = all causes |
| **Fig 2**: duration CDF **by cause** (one cell; includes <15 min events; a period ends when the node responds again or its data has been rebuilt) | 4 | three paths of ~750–1,100 points | from the legend swatches: **red = node restarts**, **green = planned reboots**, **blue = unplanned reboots** |
| **Fig 3**: events per 1000 nodes per day by cause, one cell, 3 months | 4 | stacked bars (filled rectangles: green, yellow, blue, red) | mapping read from the Fig 3 legend at extraction time (causes: unknown, node restarts, planned reboots, unplanned reboots) |
| **Fig 4**: unavailability (%), one-week rolling window, by cause | 4 | stacked paths | cross-check only |
| **Table 1** (all cells): unavailability % by cause, avg/min/max | 4 | text | numbers, no digitizing |
| **Table 2**: MTTF disk 10–50 yr, node 4.3 months, rack 10.2 yr | 5 | text | numbers |
| Figs 5–8: failure bursts, window size, rack/node burst frequency | 5–6 | vector | exec #3 |

Axes for Figs 1–2: y = "Events (%)", 0–100. x = unavailability event
duration on a **log scale** with labelled ticks at 1 s, 10 s, 1 min, 15 min,
1 h, 6 h, 1 d, 7 d, 1 mon. Label spacing is about 23–25 pt per decade,
consistent with log10.

**Targets → library:**

| Source | Library use |
|---|---|
| Fig 2 green (planned reboots) | planned-downtime duration `D` (default `planned_downtime/weekly-ford10`) |
| Fig 2 red (node restarts) | `sw` recovery duration |
| Fig 2 blue (unplanned reboots) | `hw_reboot` recovery duration |
| Fig 1 (all causes) | reference mixture; cross-check that Fig 2's causes plausibly mix into it |
| Fig 3 bar heights (time-averaged per cause) | one-cell per-cause rates → `cluster/ford10-google-storage`; planned-reboot rate → `planned_downtime/ford10-rate`; node-restart : unplanned-reboot ratio → a `ρ_SW` point |
| Table 1 averages ÷ E[D] from Fig 2 | **all-cell** per-cause rates, `λ_c = U_c / E[D_c]` (U as a fraction of node time). Stored next to the Fig 3 rates; their disagreement is part of the band |
| Table 2 | node MTTF 4.3 months → total-rate anchor for the reference profile (≥15-min convention of the paper, flagged); disk 10–50 yr → `disk/pooled` point (`λ = 1/MTTF`, a rate); rack 10.2 yr → exec #3 |

**Pipeline** (`scripts/fault_library/digitize_figure.py`; runs in the sandbox
with `pdfplumber` + numpy):

1. **Vector extraction (confirmed applicable).** Use `pdfplumber` to read each
   figure's curve objects. Select series by stroke colour (Fig 2) or by fill
   colour (Fig 3 rectangles), and map colours to causes from the legend
   swatches automatically: each swatch line sits on the same baseline as its
   label.
2. **Raster fallback** (kept for other papers): `pdftoppm -r 600` + OpenCV
   colour separation. Not needed for Ford.
3. **Axis calibration.** Use the tick-mark line objects (not label text
   positions) for each labelled tick and record (PDF coordinate, value) pairs
   in `digitized/ford10/<fig>.calib.json`. The x map is fitted in log10
   (seconds) space with all 9 ticks (least squares, residual reported), and
   the y map linearly from 0–100. The calibration file is committed, so the extraction is
   reproducible and reviewable.
4. **Clean-up.** Sample each curve at ~200 points evenly spaced in log-time.
   Enforce monotonicity with isotonic regression and clip F to [0, 1]. Record
   the visible x-range `[x_min, x_max]` and F at the ends. Duration CDFs often
   do not reach 0 or 1 inside the plotted range.
5. **Tails.** Fit lognormal, Weibull, and a 2-component lognormal mixture by
   minimising squared CDF error on the points. The raw event data is not
   available, so MLE is not possible. Choose the fit by the KS distance to the
   digitized curve. Mass outside the plotted range is assigned by the chosen
   fit's tail, and this is flagged on the profile.
6. **Output.**
   - `empirical` distribution: a monotone piecewise-linear inverse CDF in
     log-time over the digitized range, with fitted tails. This is a new
     optional `DistributionSpec` type, `{"type": "empirical", "x": […],
     "F": […], "log_x": true, "tails": {…}}`, in Rust and Python, sampled by
     inverse transform.
   - Parametric fit: used where a closed form is needed (Markov/ILP means,
     sensitivity).
   - The source record stores the figure ID, calibration, and the
     digitization error.
   - An overlay PNG (digitized points redrawn on the original figure) is
     committed for review.

**Error and validation:**

- *Resolution*: for vector paths the error is the axis-fit residual plus line
  width. Both are recorded in log-seconds and in F. The `low`/`high` duration
  variants shift the curve by ± max(fit residual, half the line width)
  horizontally, which gives the "digitization band" in the tornado sweep.
- *Double digitization*: two independent passes (second pass by a separate
  agent with its own calibration). Report the maximum vertical gap between the
  two CDFs. If it exceeds 0.02, re-examine the figure.
- *Text cross-checks*:
  - Fig 1 must give F(15 min) > 0.90 ("less than 10% of events last longer
    than 15 minutes").
  - Fig 2 must show node restarts spanning <1 min to >1 h, and unplanned
    reboots with the largest mean.
  - Per cause, Table 1's average unavailability should be consistent with
    (Fig 3 rate × Fig 2 mean duration) within the cell-to-cell min/max range
    Table 1 reports.
- *Censoring / truncation*: Fig 2 includes events under 15 min, so it is
  **not** conditional on D ≥ 15 min. A period ends when the node responds
  **or** GFS has rebuilt its data, so the upper tail is truncated by Google's
  recovery process. The profile records this, and the tail fit beyond the last
  plotted point is flagged.
- *Planned-downtime consistency check*: Table 1 puts average planned-reboot
  unavailability at 0.0154% of node time (≈1.35 h per node-year). Our default
  (one planned event per ~7 days with Ford's planned-reboot durations) gives
  `E[D]/7 d`. The ratio of the two is reported in the catalogue, so it is
  explicit whether our weekly cadence is harsher than Google's 2010 fleet.
  `planned_downtime/ford10-rate` is the Google-cadence variant.

**Caveat in the paper**: Ford's data comes from Google storage nodes around
2010 (bare metal, Google's own maintenance procedures). We use it because it
is the best public per-cause duration data. Our own exec #10 timings will
later replace the `hw_reboot`/`sw` durations as `@v2` profiles, while planned
downtime stays Ford-based unless you decide otherwise.

## 8. Progress checklist (implementation)

Unattended VM sessions (`infra/aws/run_claude.sh`) take the first unchecked
item, finish it, tick it with a one-line note, and commit. Run
`bash scripts/fault_library/build_all.sh` before ticking anything.

Done (2026-10-21, branch `exec01-fault-library`):

- [x] Step A: `powder/faults` (schema, library, convert); LogNormal/Mixture/Empirical distributions; legacy scenario reproduces `VM_PROFILES` exactly.
- [x] Step C2: Ford et al. digitized from vector paths (Figs 1-3, Tables 1-2); raster cross-check 100% within 0.02; Table 1/Fig 3 read as >=15-min events (summed rate matches Table 2 node MTTF within 5%); `cluster/ford10-*`, `disk/ford10`, `planned_downtime/{weekly-ford10,ford10-rate}`.
- [x] Step E (Python): planned-downtime renewal schedule with equilibrium phase; folded into Markov/ILP via `as_transient`.
- [x] Step F: Azure PIR incident list (2021-2026, by web search), AWS->nearest-Azure map (us-east-1 -> eastus2 fixed), EB rates, lognormal durations; `region_outage/aws-us-from-azure` (+ `-incl-single-az`).
- [x] Step D: literature sources, `disk/pooled` (random effects), `host/vishwanath10` (frailty), `cloud/aws-ondemand-nvme` (within AWS SLA cap).
- [x] Steps B/C (code): price/spot fetchers, parsers, `build_prices.py`; tested on format fixtures (no network in the sandbox).
- [x] Step G: reliability-score copula price assignment (rho 0 x20 seeds, 0.5, 1; extra R = 0.25 h / 4 h ranked variants because Kendall tau = 0.73), machine classes incl. synthetic x10/x0.1, canonical/reference scenarios, `trace.py` + `make_trace.py`, `ilp.py` adapter, catalogue.
- [x] AWS work VM: `infra/aws/` bootstrap, cron data refresh, unattended runner.

Remaining:

- [ ] Run the real price fetch on the VM (`fetch_aws_prices.py`, `build_prices.py`, `build_machines.py`); fix any parser failure against the real offer files (attribute names are checked and fail loudly). Confirm `i4i.2xlarge` exists in all four US regions (else switch scenarios to `m6id.2xlarge`) and record its vCPU/memory/NVMe size in the scenario `deployment` block (exec #6 needs `storage_gb`).
- [ ] Spot: Spot Advisor snapshot + `describe-spot-price-history`; build `spot/aws-*` profiles and machines; confirm the Spot Advisor JSON layout matches `parse_spot_advisor`.
- [ ] SpotLake 2025: sync the public bucket for the deployment types, inspect the file format, write `parse_spotlake` + tests, and add band/price history statistics to spot profiles (input for exec #4.2).
- [ ] Rust engine: add `lognormal`, `empirical` and `mixture` `DistributionSpec`s to powder-mc (and `mc_jobs.distribution_spec`), so `engine=True` no longer approximates; `cargo test`.
- [ ] Rust engine: `PlannedDowntimeSpec` + `PLANNED_DOWNTIME_START/END` events; replacement must ignore planned downtime; cross-check against `planned_downtime.schedule` statistics.
- [ ] Rust engine + Python `NetworkConfig`: per-region outage rates (`per_region` map); make `RegionOutageModel.as_network_config` exact instead of max/mean.
- [ ] Refine `region_incidents_azure.csv` from azure.status.microsoft directly (the sandbox could only read search snippets): verify times/scopes of S rows, add missed US incidents, rebuild region outages.
- [ ] Azure reference: `az login --use-device-code`; run `azure_locations.py` to confirm coordinates (mapping decisions stay fixed).
- [ ] LANL failure data (CFDR/FTA `lanl05`): download, extract per-system cause counts, Weibull/lognormal fits, frailty; add `hpc/lanl05-*` profiles; pool `rho_SW` with Ford's ratio.
- [ ] Google ClusterData2019 `machine_events` (BigQuery; likely needs a Google login -> may be blocked): `cluster/google2019-*` + `planned_downtime/google2019`.
- [ ] Birke et al. DSN'14 numbers (needs the PDF; drop it in `docs/reading/`): `vm/birke14-*` profiles.
- [ ] Trace replay mode in powder-mc (Goal 2.2: simulator and deployment see the same faults). Optional for exec #1.
- [ ] (exec #10) Replace placeholder durations (`hw_reboot`, `sw`, `spawn`, rebuild `R`) with measured AWS timings as `@v2` profiles.

- [x] Independent review (2026-10-21): every number re-derived from the source records matches; fixes applied:
  Spot Advisor parser now matches the real file layout (`{index, label, dots, max}`); the trace uses one transient
  clock at lambda_T plus a separate exponential data-loss clock (same process as the simulator); frailty is opt-in on
  both sides; scenario planned downtime is folded into NodeConfigs (`include_planned`); per-slot/per-layer RNG
  streams; AWS host faults consistent per level (f_loss split into -loss0.2/-loss0.8 variants); inter-AZ transfer
  counted on both sides; time-weighted spot price summaries; t-based pooling interval; validation gaps closed.
- [x] Decisions 2026-10-21: global incidents are a separate all-regions process (per-region rates now from
  region-specific incidents only); planned downtime keeps independent phases by default, with a
  `planned_downtime/weekly-ford10-rolling` variant (AZ/region waves).
- [x] Ford uncertainty: bootstrap over each digitized curve with N >= 1146 events (smallest jump = one PDF quantum);
  rate low/high = Table 1 cell range x bootstrap duration uncertainty; duration low/high = curve rescaled to the
  bootstrap 5th/95th percentile mean. Table 2 check ratio 1.05 (bootstrap 0.75-2.0): weak evidence only.

Recorded deviations from §7 (and why):

- Global outages: the one global incident in the window (2023-01-25 WAN) gives a global rate of 0.39/yr (Jeffreys,
  band 0.05-2.1); per-region rates fall to 0.13/yr. With n = 1 the global rate is the least certain number in the
  region model.
- Region outage EB uses the nine Azure US regions, not all regions worldwide: the incident search was thorough only for US regions. Window 2023-01-01 to 2026-10-20: PIRs older than ~5 years are gone, and since mid-2022 Microsoft publishes only broad incidents.
- The software cause in `cloud/aws-ondemand-nvme` uses Ford's >=15-min node-restart : unplanned-reboot ratio (1 / 2.35 / 4.49) with >=15-min durations. Short automatic restarts are left to planned downtime and exec #10, because Ford's all-events node-restart rate (~57 per node-year) reflects Google's rollout practice.
- Rust-side changes (new distributions, planned-downtime events, per-region outages) are not done: the sandbox has no Rust toolchain. Until then, `engine=True` maps new distributions to a Weibull with the same mean and median, and region outages collapse to the max regional rate.

Known caveats to state in the paper (from the independent review, 2026-10-21):

- `cloud/aws-ondemand-nvme` pairs a host-repair *rate* (Vishwanath, 2010, includes disk repairs) with Ford's
  unplanned-reboot *durations* (Google, 2010): different fleets and event definitions. f_LM and the sw-ratio low
  end (1.0) are assumptions, not measurements; the tornado sweep should show their effect.
- The disk pool mixes application-visible disk errors (Ford), replacement rates (Schroeder & Gibson) and AFRs
  (Backblaze HDD and boot SSD); the wide t-based interval is the honest summary.
- `engine=True` matches the mean and median of non-native distributions but not the tails (e.g. p99 of the
  Ford recovery mixture is about 1.7k s vs 1.2k s). Results that depend on tails should use the Python
  simulator or wait for native engine support (§8).
- `planned_downtime/ford10-rate` treats Google's planned reboots as quasi-periodic (sigma 12 h on a ~72-day gap);
  Ford's data may be closer to random arrivals. It is a sensitivity variant only.
- Independent planned-downtime phases ignore coordinated maintenance; the rolling variant brackets it.
