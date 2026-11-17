# Figures from the 1x / 30x fleet-ILP results: what-if, modeling stability, scalability, sensitivity

Status: in progress (2026-11-05). Branch `figs-whatif-modeling` (off `fault-std-ford-planned` at b698a12).
Targets the mock figures in `OSDI experimenet MVP.pdf`: Fig 4 (what-if), Fig 13 (modeling stability),
Fig 10 (scalability), Fig 11 (sensitivity), Fig 8 (reconfiguration cadence: code only, not run).
Fig 12 (adaptive SLA budget) is out of scope here.

## Decisions (user, 2026-11-05)

- What-if: timeouts {5, 10, 15, 20, 30, 60} min, **500k MC runs** per placement and timeout, **24 h horizon**.
- Markov ladder: 24 h horizon (time-averaged from all-healthy). Markov solves run on a **GPU VM**
  (`infra/gpu/gpu_vm.sh`, g5.4xlarge in us-west-2), not on the work VM.
- Sensitivity: SLA target **5 nines** at 1x rho1. Frontier scoring stays at 720 h, as the optimizer uses.
- Scalability: the table from past runs is enough; rerun only if needed.
- Reconfiguration cadence (Fig 8): implement `scripts/reconfig_cadence.py`, smoke-test, commit; **do not run**
  the 6-month study (6h/12h/1d/2d/1w/2w/1mo cadences) without explicit go-ahead.

## Inputs we already have

- Fleet-ILP v2 results, 1x and 30x, rho0 (independent prices) / rho1 (reliability-ranked prices):
  `outputs/pilot-ilp-rate-scale/lattice/rows/x{1,30}-rho{0,1}.json` (merged in `lattice-merged/`).
  Exact-MC caches shared across prices: `outputs/pilot-ilp-rate-scale/cache/exact-x{1,30}.json`.
- Setup for every result: 2000-machine catalogue (`machines-od1000-spot1000-seed0-az`, 720 h horizon),
  scenario `canonical-central-az-aware@v3`, 50 RSMs, capacity 5, Raft with election Exp(0.2/s), node
  replacement after 600 s, Rust MC with region/AZ outages.
- **Gap:** the rows store summaries (nines, cost, RF counts, timings) but **not the chosen placements**.
  The 30x rho1 50% row is "carried from the 40% budget" ($12.27/h, 4.50 nines, RF 5x20 + 7x30).
- Markov builders: `powder/markov_builders/{leaderless,raft}.py`, five quality levels each
  (SIMPLIFIED, COLLAPSED_PIPELINE, NO_ORPHANS, MERGED_PIPELINE, FULL), heterogeneous via rate classes.
  `outputs/availability-heterogeneous-markov/REPORT.md` shows the state limits: raft FULL is feasible only at
  N=3; NO_ORPHANS at N=7 exceeds 1.5 GiB with 7 distinct classes.
- Past timing records: pilot rows (`seconds` per configuration and per-budget `times`: main ILP / main MC /
  ascent), exec-02 jsonl (`seconds`, `num_rsms` 50 to 10,000, `pool`, `master_size` 1000 to 4000).

## Step 0: persist deployments (needed by figures 1 and 4)

- `scripts/pilot_fleet_frontier.py`: add `placements` (list of sorted machine-id tuples, one per RSM) to every
  `new_ilp` budget row and to `main_ilp`; `carry_forward` carries them along. The format is additive, so
  older rows still load.
- Re-run **x30 rho1 at budgets 0.4 and 0.5** (warm exact cache, ~15-20 min on 8 cores) into
  `outputs/figs/deployments/`. Same command for x1 rho1 (and both rho0 runs, which are cheap with warm caches) so
  that every figure works from saved placements.
- Gurobi's 120 s limit makes reruns slightly non-deterministic, so we record the rerun's nines next to the
  committed value. We accept the rerun if it is within 0.05 nines; otherwise we use whichever is higher and note it.

## Figure 1: what-if, node replacement timeout (Fig 4, expected bars only)

Deployment: 30x, rho1, 50% budget (from Step 0).

- New `scripts/figs/whatif_timeout.py`: for each timeout in **{5, 10, 15, 20, 30, 60} min**, run exact MC on
  every unique placement of the deployment with `NodeReplacementStrategy(failure_timeout=t)`. Everything else
  stays fixed: same machines, protocol and network.
- Per placement we record mean availability, sample std and mean cost (the MC bills replacement nodes:
  `cost_samples`). Deployment availability = mean over the 50 RSMs. Its s.e. is sqrt(sum s.e._i^2)/50, shown as
  the error bar. Cost = fleet $/h plus mean replacement overhead.
- Runs: 500k per placement (fixed seeds per placement, **common random numbers across timeouts**, so the
  differences between bars are low-noise).
- Horizon: **24 h** from a healthy start (what an exec #7 deployment observes).
- Plot: hatched "expected" bars in nines with CI, annotated with $/h. It leaves an empty "actual (24 h)" slot that
  `outputs/tidb/<handle>/` fills later. A second panel shows cost vs timeout. Also as a variant: 1x rho1 50%,
  and 30x rho0 (cheap once the deployments exist).

## Figure 2: modeling stability, Markov ladder vs MC (Fig 13 without "observed")

There is no observed availability, so the reference is Rust MC. The y-axis is **estimate - MC (nines)**, with
MC's CI as a band around 0. We state clearly that MC is the reference, not ground truth.

- RSMs: one 3-, 5- and 7-node RSM from the **1x** deployments and one of each from the **30x** deployments,
  6 RSMs in total. The 30x 50% deployment has no RF 3, so we take each RF from the saved deployments across budgets.
  For each RF we take the placement nearest the median availability; if an RF is missing entirely, we fall back to
  the fixed-RF baseline placements, which are already exact-scored.
- Ladder (each step adds one modeling feature; same scaled constant-rate machine configs as the MC):
  1. **Homogeneous erasure-coded storage:** independent nodes, all at the placement's mean rates; A = P(majority up).
  2. **+ Heterogeneous:** same model with per-node availability (Poisson-binomial).
  3. **+ Node up / out-of-date:** leaderless CTMC at NO_ORPHANS (replacement, catch-up, learner states).
  4. **+ Leader election:** raft CTMC at NO_ORPHANS (FULL where it fits: N=3).
  5. **Monte Carlo:** Rust MC, 1M runs. Two bars: **without** region/AZ outages (same failure model as the
     Markov ladder, which isolates modeling error) and **with** outages (what we optimize). The second gap shows
     what Markov cannot model at all.
- Tractability: one rate class per machine, no binning. Measured: N=5 NO_ORPHANS has 7.8k (leaderless) / 14k
  (Raft) states and builds in 0.4 s; N=7 is about 0.3M / 0.5M states. Raft FULL is attempted wherever the
  estimate is <= 6M states (N=3, N=5). Larger cells are recorded as "not tractable" instead of being dropped.
- Markov values: time-averaged over 24 h from all-healthy by **uniformization**
  (`markov_solver.time_averaged_availability_uniformization`, NumPy or CuPy). It matches the augmented
  `expm_multiply` to 6e-9 relative error at 24 h, and is 7x faster even on CPU (Lambda T ~ 4-9e4 sparse
  matvecs). Steady state is recorded as a check.
- Plot: a 2x3 grid (1x / 30x rows; N = 3/5/7 columns) of Fig-13-style waterfalls, plus one summary panel of
  |error| per ladder step. Extra output: a runtime table per step (it feeds the "++ Markov pruning" idea later).

## Figure 3: scalability (Fig 10)

Phase A uses past runs only, so it is cheap and done first. Phase B is a controlled sweep, which the scaling
claim needs.

- **A, harvest:** `scripts/figs/harvest_timings.py` collects every record with wall time and size:
  - pilot rows: 4 configurations x 7-10 budgets, with per-budget ILP / MC / ascent seconds and threads
  - exec-02 jsonl: `num_rsms`, `pool`, `master_size`, `seconds`

  It writes a CSV and a table (`outputs/figs/scalability/past_runs.md`). Caveat: past runs are N=50 except
  exec-02 rsm10000 (type-count pipeline, not the fleet ILP), and they logged wall time, not CPU time. They can
  appear as labelled overlay points, but they cannot give the slope.
- **B, sweep:** `scripts/figs/scalability_sweep.py` runs the current fleet pipeline (1x, rho1, 50% budget) at
  **N in {10, 30, 100, 300, 1000} RSMs** (cluster size = machines used) x **regions in {1, 2, 4}** (catalogue
  filtered by region).
  - CPU time is measured with `resource.getrusage(SELF) + getrusage(CHILDREN)`, since every MC call is a Rust
    child process. ILP, MC and candidate-generation time are recorded separately.
  - Optimizer $ = CPU-hours x i4i.2xlarge on-demand $/h / 8, times 30 for daily reconfiguration.
  - Monthly savings = (cost of the cheapest baseline reaching the ILP's availability - ILP cost) x 730 h. If no
    baseline reaches it, we use the price-blind baseline at matched availability by interpolating the baseline
    frontier.
  - Budget: N=1000 is the risk. Estimate N=300 first; if the projected time exceeds 3 h, cap at N=300 and
    extrapolate, and mark the result as extrapolated.
- Plot: left, CPU-hours per run vs nodes (log-log, one line per region count, past runs as hollow markers);
  right, optimizer $/month vs monthly savings, with the net-benefit band.

## Figure 4: sensitivity, 1x (Fig 11)

The question: what happens if the operator's assumed fault rates are off by a multiplier m, when the truth is 1x?

- m in **{0.25, 0.35, 0.5, 0.7, 1, 1.4, 2, 2.8, 4}**, rho1 prices. For each m, run the fleet pipeline with
  `rate_scale = m` (the existing scaling path) and save the deployments for a budget grid (densified around the
  SLA's cost).
- **SLA-driven choice:** for each m, pick the cheapest saved deployment whose *believed* (scale-m) availability
  is at least the SLA target. Then re-score it with exact MC at the **true 1x rates**.
- Plot: top, true availability vs m with the SLA line and the region that violates it shaded; bottom, $/h vs m.
- Cost: 8 new pipeline runs (m=1 exists), each ~10-60 min with warm caches.

## Code layout and outputs

- `scripts/figs/` (new): `whatif_timeout.py`, `markov_ladder.py`, `harvest_timings.py`, `scalability_sweep.py`,
  `sensitivity.py`, plus `render.py` (one shared matplotlib style, PNG + SVG). Each script writes JSON first;
  rendering only reads the JSON.
- `powder/faults/mc_scoring.py`: a small helper returning (mean, std, cost mean) per placement for a given
  strategy, so Figure 1 and Figure 4 reuse it.
- Outputs: `outputs/figs/{deployments,whatif,modeling,scalability,sensitivity}/`.
- Tests (NumPy-only where possible): the placements round-trip through the row JSON; ladder step 1 equals step 2
  when all nodes are identical; ladder step 4 at N=3 matches `build_raft_model` directly; CPU-time accounting
  includes child processes.

## Order and compute

| # | Step | Compute (8 cores) | Blocks |
|---|---|---|---|
| 0 | persist placements + rerun 30x/1x rho1 | ~40 min | 1, 2, 4 |
| 1 | what-if timeouts | ~15 min | none |
| 2 | Markov ladder | ~30 min (MC 1M runs x 6, Markov solves) | none |
| 3A | harvest past timings | minutes | none |
| 4 | sensitivity sweep | ~4-6 h, can run unattended (no cloud) | none |
| 3B | scalability sweep | ~3-6 h, can run unattended | none |

Steps 1, 2 and 3A each produce a figure the same day. Steps 4 and 3B are long, so they run on the AWS work VM
(`infra/aws/README.md`) from the checklist below.

## Decisions to confirm

- Timeouts {5, 10, 15, 20, 30, 60} min (the PDF says 10/15/20 in the text and 5 min/30 min/1 h in the figure; this
  set covers both).
- Primary MC horizon 720 h, with 24 h as a secondary.
- Sensitivity SLA target: proposed **5.0 nines** at 1x rho1 (inside the 1x frontier's 4.6-6.0 range, so both
  optimistic and pessimistic m change the choice).
- Scalability: run the new sweep (B) as well as harvesting past runs (A).

## Figure 5: reconfiguration cadence (Fig 8). Code only; do not run without a go-ahead

`scripts/reconfig_cadence.py` with `powder/faults/reconfig.py` (tests: `tests/test_reconfig.py`).

- Why cadence matters here: catalogue hazards are time-varying. Calendar curves drive hardware reboots,
  software and spot reclaim, and age curves drive disks. Weekly transient rates vary 76x (median machine) over
  6 months.
- For each cadence c in {6 h, 12 h, 1 d, 2 d, 1 w, 2 w, 1 mo}, over 6 months (4380 h): re-run the fleet ILP at
  t = 0, c, 2c, ... on hazards averaged over the coming window (`catalogue.node_config(t0_h=...)`), at a fixed
  $/h budget.
  - The previous deployment is offered as columns.
  - Hysteresis `--min-gain` (default 5%): switch only if the believed downtime drops by at least that much. In
    the smoke test, without hysteresis, it reshuffled for negligible gains.
- Ground truth: the same 6 h blocks for every cadence, using each placement's long-run availability under the
  block-average hazards (quasi-stationary; Rust MC with outages). Scores are cached per block and shared across
  cadences.
- Outputs per cadence:
  - SLA violation (min/month above the 5-nines budget, mean over RSMs) and expected downtime
  - GB moved/day (RSMs matched by the Hungarian algorithm; 100 GB/replica; copied from the cheapest member at
    library transfer prices) and transfer $/day
  - optimizer CPU-h/day (self + child CPU)
- Cost if run at full size: about 1,600 solves in total (730 of them at 6 h), at about 1-10 min each, plus about
  730 blocks x 50 placements of MC. That is days on 8 cores. Before running, consider 30x (1x sits near 6 nines,
  where violations are rare) and a coarser cadence subset.
- Smoke: `python scripts/reconfig_cadence.py --smoke` (2 cadences, 48 h, 10 RSMs, 300 machines; under 1 min).

## Checklist

- [x] 0 placements persisted; 30x rho1 (10/40/50%) and 1x rho1 (sensitivity grid) saved; rho0 reruns not needed yet
- [x] 1 what-if timeout JSON + figure (30x rho1 50%); variants (1x rho1, 30x rho0) not run
- [x] 2 Markov ladder JSON + figure (1x and 30x, N = 3/5/7) + runtime table (GPU VM, torn down)
- [x] 3A past-run timing table + figure
- [ ] 3B scalability sweep + figure
- [ ] 4 sensitivity sweep + figure
- [x] 5 reconfiguration cadence simulator + smoke test (not run)
- [x] GPU VM tooling (`infra/gpu/gpu_vm.sh`) + uniformization solver (+ test)

## Results (2026-11-06)

**Step 0.** The 30x rho1 rerun (with consolidation columns, b698a12) gives 4.51 nines at $12.29/h (40%) and 4.52 at
$15.29/h (50%), vs committed 4.50 / 4.50 (50% was carried from 40%). Accepted (within 0.05).

**Figure 1** (`outputs/figs/whatif/x30-rho1-b0.5-h24.*`). 24 h, 500k runs per placement (10 unique placements,
66 machines). Availability is flat: 4.51 nines at 5 min to 4.50 at 60 min, with overlapping 95% CIs. Replacements
fall from 145 to 10 per day (14.5 to 1.0 TB/day at 100 GB per replica). The MC's billed overhead is about $0:
a replacement overlaps the failed node only briefly, so the cost of short timeouts is data movement.

**Figure 2** (`outputs/figs/modeling/ladder.*`). Error vs MC with outages, 24 h:
- Independent-node storage models: off by +0.6 to +6.6 nines (worse with N).
- Leaderless NO_ORPHANS: worst of all (+1.6 to +8.4).
- Raft NO_ORPHANS: within 0.16 nines on 5 of 6 RSMs. The exception is 30x 3-node (+0.67), whose recovery times
  are heavy-tailed empirical distributions.
- Region/AZ outages matter little over 24 h (MC with vs without: <= 0.15 nines).

Solver notes:
- 24 h averages use uniformization: the 7-node Raft chain (0.61M states) takes 31 s on the A10G, plus a 27 s
  Python build.
- Steady state (cross-check only) needs a sparse direct solve. The old dense normalization row ruined fill-in;
  pinning the initial state (`steady_state(normalization="pin")`) fixes it. cuDSS then solves 14k states in 1.1 s
  (SciPy SuperLU: 198 s).
- The 7-node chains exceed cuDSS's 24 GB (ALLOC_FAILED) and Pardiso's in-core memory on 64 GB (error -2): LU
  fill-in on a 7-dimensional lattice. Those steady states are recorded as failed.

**Figure 3A** (`outputs/figs/scalability/past_runs.*`):
- 43 fleet-ILP solves (50 RSMs, 30-86 machines; median 89 s); 332 exec-02 type-count solves.
- Optimizer cost at daily reconfiguration: <= $0.51/month (median fleet-ILP solve). Savings: median $9.4k/month.
- Past runs span only 30-100 machines, so they cannot show a scaling slope; that needs a controlled sweep (3B).

**Figure 4.** The sweep over m = 0.25..4 is running (`outputs/figs/sensitivity/rows`); then
`scripts/figs/sensitivity.py` + `render.py sensitivity`.

### Why the 1x deployment misses the 5-nine SLA (2026-11-12, `scripts/figs/diagnostics/`)

Believed 5.35 nines, true 4.68 (independent seeds, 1M runs per placement). The model is the same on both sides:
- Engine: the committed and the rebuilt (uncommitted) engines are bit-identical on these placements, and both
  reproduce the cached scores exactly.

Cause: optimizer scores are noisy on a rare catastrophe, and the ILP selects the lucky ones.
- The 10% budget buys 10 all-spot 3-replica placements. 88% of their downtime comes from runs down > 24 h
  (140 of 3M runs, 4.7e-5 per RSM-month).
- Every one of those runs is a potential-data-loss event: two of three replicas lost permanently (spot
  interruption, disk) before a replacement catches up. Without quorum, safe-mode replacement cannot proceed, so
  the RSM stays down for the rest of the 720 h horizon.
- A 100k-run score sees about 5 such events (Poisson), so single scores vary 3-10x.
- The median single 100k score is unbiased (4.67 nines), but the ILP picks, out of thousands of candidates, the
  placements whose scores missed the catastrophes. Each chosen score is below 9-10 of 10 fresh estimates.

Fix options:
1. Re-score the chosen deployment on independent seeds (validation; needed for honest numbers).
2. Score candidates conservatively (upper confidence bound on unavailability) or with a rare-event estimator
   for quorum loss (Markov or stratified MC for P(quorum loss) x expected stuck time).
3. Model recovery from quorum loss (operator unsafe-recover or restore within hours, not the rest of the
   horizon). This changes the truth too.

### Sensitivity with the new scoring (2026-11-17, `outputs/figs/sensitivity-new/`)

All 9 assumed multipliers rerun with re-selection batches (3 x 100k) and 1M-run validation. 1x ran locally; the
other 8 ran on a 32-vCPU VM in 75 min (torn down). Each point is the cheapest deployment whose validated
availability (at the assumed rates) is >= 5 nines; truth is 1M runs per placement on separate seeds.
- SLA met from 1x upward (old scoring: from 1.4x). Truth 5.39 / 5.65 / 5.45 / 5.88 / 5.75 nines at
  1x / 1.4x / 2x / 2.8x / 4x.
- Optimistic inputs (<= 0.7x) still miss (4.10-4.77 nines): the input rates are wrong, so better scoring
  cannot help. Validation at the assumed rates reports 5.0-5.2 there.
- Cost of correctness at 1x: $3.84/h vs $2.99/h with old scoring (which truly gave 4.70 nines).
