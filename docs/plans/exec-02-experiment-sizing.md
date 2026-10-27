# Execution order #2 — sizing the placement experiment

Status: local simulation study (2026-10-27). No AWS machines are deployed by this step.

## Decision and estimands

Distinguish **M** catalogue machines available for selection from **N** distinct machines
actually activated for the placement. The 1000-machine catalogue is not a 1000-node
TiDB deployment. A menu whose on-demand recipes repeat 8 times is not eight times as
many purchasable 8-vCPU instance offers.

The independent (`rho0`) and reliability-ranked (`rho1`) on-demand price assignments
are frozen **before** optimizing. The ILP sees each individual assigned price. Spot
pools keep native SpotLake prices; the primary price-assignment experiment uses only
on-demand candidates, and mixed-tier results are a sensitivity check. For each M and
price seed, compare the same inventory under a region-balanced cheapest-first placement,
a region-balanced price-blind placement, and an ILP using each candidate's exact
machine rates in the simplified Raft Markov model. Every baseline placement is
in the ILP candidate menu: otherwise a claimed optimization gap may just be
a missing candidate. All three
use the same machine capacity and the same 3-region replica spread.

The primary comparisons are (a) ILP availability at no more than the cheapest
baseline's **unique physical machine** cost and deployed-machine count, and (b)
minimum ILP unique-machine cost with no RSM less reliable than the baseline's
worst RSM and mean availability no worse than that baseline, evaluated separately
against the cheapest and price-blind baselines. These are
modeled prices, not the AWS bill: all live instances will be identical i4i.2xlarge
hosts and model classes are emulated on top. The ILP assigns 50 three-replica RSMs,
capacity 5 RSM replicas per machine, a maximum of 64 activated machines, and
N is recorded rather than assumed to equal 64. The cheapest-first baseline
packs region-balanced placements into 30 machines rather than leaving two
partly empty. This is *not* an operational recommendation until storage and
headroom constraints exist (execution order #6).

The machine-price MILP uses `scipy.optimize.milp` (HiGHS), **not Gurobi**. Binary
machine-activation variables enforce node capacity, a unique-machine budget, and
an activated-machine cap. A separate per-RSM availability floor plus a scaled
mean-unavailability constraint supports cost minimization at matched quality.
Every candidate is scored on its exact machine rates (no rate binning), while
the Markov state model is still the simplified quality level.

## Reproduction

```bash
python3 scripts/experiment_sizing.py --price-seeds 20 --output outputs/exec-02/ondemand-packed.jsonl
python3 scripts/experiment_sizing.py --pools 250 1000 2000 --price-seeds 20 \
  --output outputs/exec-02/blind-quality-20.jsonl
python3 scripts/experiment_sizing.py --pools 1000 2000 4000 --price-seeds 3 \
  --output outputs/exec-02/large-pools-packed.jsonl
for s in 0 1 2 3 4; do python3 scripts/experiment_sizing.py --pools 250 \
  --catalogue-seed "$s" --price-seeds 3 \
  --output "outputs/exec-02/catalogue-seed-${s}.jsonl"; done
python3 scripts/experiment_sizing.py --pools 250 1000 2000 --spot-fraction 0.3 \
  --price-seeds 1 --output outputs/exec-02/mixed-packed.jsonl
python3 scripts/experiment_horizon.py --sweep outputs/exec-02/ondemand-packed.jsonl \
  --pool 1000 --output outputs/exec-02/horizons-packed.jsonl
python3 scripts/experiment_horizon.py --sweep outputs/exec-02/blind-quality-20.jsonl \
  --pool 1000 --days 1 7 --policies blind iso_blind_quality \
  --output outputs/exec-02/horizons-blind.jsonl
python3 scripts/experiment_horizon.py --sweep outputs/exec-02/ondemand-packed.jsonl \
  --pool 1000 --days 1 7 --no-network \
  --output outputs/exec-02/horizons-packed-no-network.jsonl
python3 -m pytest -q tests/test_experiment_sizing.py
cargo test --manifest-path rust/Cargo.toml
```

`experiment_sizing.py` generates one seeded master catalogue and takes nested
prefixes at M = 250, 500, 1000, 2000 (so the price/fault mapping of an existing
machine does not change when M grows), with 20 independent-price permutations and
one ranked assignment. It reports recipes, distinct price values, candidate
coverage and count, solver status/gap, selected machines, cost, exact scores,
and total wall time per arm.
Budget-limited ILP and cost-minimizing ILP are distinct evaluations. Increasing
candidate-menu size is a required search-quality check before calling a pool-size
plateau real. A 30% Spot run is a separate sensitivity: cheap native-price Spot
pools can drown out the two *on-demand* price-assignment mechanisms.

The horizon check runs the selected representative RSM, not an entire 50-RSM
cluster, through the Rust MC at 1, 3, 7 and 30 days. The Rust fixed-horizon
path now accounts for all time after actual data loss; previously it ended early
on precisely the runs that mattered. MC includes full catalogue machine hazard
curves and regional/global outages, but the weekly maintenance phase is still
folded into an exponential transient rate. It measures fault and application
unavailability-event probabilities separately. A single RSM's event probability
is **not** a 50-RSM deployment probability: RSMs share physical nodes and region
outages, and the engine has no joint-cluster deployment mode yet.

> **Stale (2026-10-27):** the catalogue lost its second transient base
> (`cloud/aws-ondemand-nvme`, see exec #1 §8), so it now has 244 distinct
> on-demand recipes, not 488, and the numbers below were computed from the old
> catalogue. Re-run the sweeps before citing any of them.

## Interpretation / limits

- Catalogue on-demand diversity is bounded by 488 distinct recipes. Beyond M=488,
  new on-demand rows are additional draws of rates and assigned prices from
  repeated recipes, not new empirical market offerings. Record the M at which a
  claimed win appears, its M/N ratio, and whether it requires these repeats.
- The primary sweep fixes fault catalogue seed 0 and varies 20 *price*
  permutations. A limited M=250 cross-check uses five catalogue seeds and
  three price seeds each. Neither is a confidence interval over measured
  hardware offers, workloads, or failure-model uncertainty.
- Independent-price assignment is a *modeling counterfactual*, not a measurement
  of paired EC2 failure rates and prices. Do not claim real procurement savings
  from the identical-host emulator's AWS bill.
- The current optimizer does not enforce storage, movement, real cluster recovery
  time, correlated failure bursts, or region-specific rebuild bandwidth. Later
  steps #3–#6 and #10 require re-running the selected cases.
- A 24-hour emulator could check predicted fault-event frequency and recovery
  behavior. It cannot validate four-/five-nines availability merely by observing
  zero quorum downtime. Set the number of repeated deployments k by power or a
  predeclared predictive interval **after** a joint-cluster model and calibrated
  TiDB recovery exist; do not mistake many RSMs on shared machines for independent
  repetitions.

## Results (local, 2026-10-27)

Under the exact-machine simplified Markov oracle, **the price-blind baseline is
substantially more expensive at matched reliability under both pricing models**.
The comparison matches its mean RSM availability *and* its worst RSM's
availability, and both policies activate 30 machines. `rho0` is the median and
full range across 20 independent price permutations; `rho1` is one deterministic
ranked assignment. All MILPs in this table finished successfully.

| Candidate pool M | Distinct on-demand recipes | `rho0` modeled cost saving vs blind | `rho1` modeled cost saving vs blind |
|---:|---:|---:|---:|
| 250 | 250 | 39.56% (32.02–52.55%) | 28.31% |
| 1000 | 488 | 44.80% (38.11–51.91%) | 25.15% |
| 2000 | 488 | 47.69% (40.02–58.47%) | 20.13% |

**M=250 already clears a 10% modeled-cost advantage at matched reliability in
both price models.** That is 8.33 candidate machines per activated machine;
there is no evidence that >1000 candidates are *needed* for this particular
price-blind comparison. The source recipes permit only 488 distinct on-demand
compositions. At 1000 and 2000 the remaining catalogue rows are fresh random
rate/price draws of repeated recipes; at 4000 there are still 488 recipes.
These are synthetic options, not 4000 independently procurable EC2 types.
As a limited inventory-draw check at M=250, five catalogue seeds crossed with
three independent-price seeds each retained matched-reliability savings
of at least 32.28% (`rho0`); the five ranked-price cases saved at least 16.21%.
This is stronger evidence that the first seed was not an outlier, not a claim
about a real purchasable catalogue. The primary 20 independent-price
assignments had realized price–reliability Spearman correlations from −0.043
to +0.031.

The stronger, **price-aware cheapest-first baseline**, now packed to the same
30 machines, changes the conclusion: median matched-reliability cost saving is
**0% at every M=250–2000** across the 20 independent-price seeds; no seed
exceeds 0.74% in the 1000-machine pool. The ranked-price run saves 0% up to
M=1000 and 0.27% at M=2000. A separate M=4000 sensitivity finds no substantial
additional saving, but the fixed-size candidate search covers only ~460 of
4000 machines, so this is *not* an optimality claim over all possible
placements. Increasing the random candidate budget from 120 to 350 per pass
at M=1000/2000 also does not uncover a large saving against cheapest-first.
With 30% Spot candidates, native Spot prices dominate the cheapest baseline:
matched-quality savings are 0% in both price models at M=1000 and 2000.
**The large price win is a win over a price-blind baseline, not over a
well-packed cheapest-first heuristic.**

The selected 30-node, 50-RSM, three-replica on-demand example (rho0 seed 0,
M=1000) expects ~4.29 weekly planned node interruptions, 0.066 hardware
reboots, 0.020 software restarts and 0.002 disk losses in 24 hours. Node
interruptions are not the same as quorum outages. Across 20,000 Rust MC runs
of one representative RSM, ~35% of 24-hour runs see any node fault and 13.5%
see any application unavailability. At that estimated rate for a *single RSM*,
21 independent 24-hour runs give a 95% chance of observing **one** unavailability
event (`ceil(log(0.05) / log(1 - 0.135))`); at seven days the chance is ~63.5%,
so three independent seven-day runs have
~95% chance of at least one event. Neither is sufficient to distinguish small
policy differences, establish availability nines, or infer the joint 50-RSM
probability: groups share machines and region outages.

With per-region/global outages included, the representative RSM's 24-hour
sample-mean availability is about **0.999619**; without those outages it is
about **0.999989**. At 20,000 runs, the with-outage mean has a simple 95%
standard-error half-width of ~0.000167, far wider than the placement effects
under investigation. The rare shared global-outage input is uncertain, and
our simplified Markov ILP does not score its effect. The Rust results are a
warning about event sparsity and model disagreement, **not** observed TiDB
availability or a statistically significant policy win.

### Decision from step #2

**Do not commit to a 24-hour, k-run emulator as a demonstration of improved
availability or actual AWS spending yet.** The pool-size question has a
positive *simulation-only* answer against price-blind placement (M=250 suffices
here), but a negative answer against cheapest-first under current assumptions.
Thirty active machines are the arithmetic minimum for 150 replicas at capacity
five, with no storage/headroom safety margin; 64 was only a search cap. The
physical N must be revisited with the storage/workload model in exec #6. A
seven-day pilot could validate that injected faults and Raft recovery are
observable, but the quoted three repetitions are **only** for seeing one event
on one RSM, not a recommendation for a superiority experiment. Choose a power-
justified k only after a joint-cluster fault/recovery simulator, measured TiDB
recovery timings (exec #10), and the later correlated-fault/network models.
Actual AWS cost remains identical when both emulated policies run the same
number of identical i4i.2xlarge hosts; all percentage cost savings above are
assigned-model prices, not bill reductions.

## Verification

On this machine: `python3 -m pytest -q tests` finished with 385 passed and 3
skipped; `cargo test --manifest-path rust/Cargo.toml` passed all 364 tests;
`python3 scripts/fault_library/run_tests.py` passed; and the fault-library
`build_all.sh` passed its 64 tests when run from a temporary Python environment
with the PDF/raster build dependencies declared in `pyproject.toml`. The latter
regenerated a few pre-existing source/catalogue bytes differently at floating
point and PNG rendering precision under this host's package versions; those
unrelated generated files were restored rather than mutating immutable library
inputs. No VM was deployed. Raw rows, catalogue hashes, placements and MC
summaries are committed in `outputs/exec-02/`.

## Follow-up: fixed-budget frontier (2026-10-27)

The first sweep's seeded candidate local search iterated a Python set, so the
candidate menu could change with `PYTHONHASHSEED`. The search now sorts the
initial candidates, and baseline placements are added in sorted order. The
original output rows are retained as a historical run; the reproducible rerun
is `outputs/exec-02/blind-quality-deterministic.jsonl`. The new
`experiment_frontier.py` reconstructs the same candidate menu and solves a
second question: **maximum mean RSM availability under the blind baseline's
cost budget, with no RSM below the baseline's worst availability**. The
nine-budget frontier for catalogue seed 0 / price seed 0 is in
`outputs/exec-02/frontier-deterministic.jsonl`.

| M=1000 comparison | Independent prices (20 assignments) | Reliability-ranked prices (one assignment) |
|---|---:|---:|
| Minimum modeled cost at blind baseline's mean and worst-RSM availability | 44.862% median saving (37.795–51.910%) | 25.135% saving |
| Incidental mean-unavailability reduction at that minimum cost | 0.226% median | 0.003% |
| Maximum mean-unavailability reduction under blind baseline's budget | 1.329% median (0.664–5.774%) | 1.251% |

The last row compares *unavailable time*, not availability percentage points.
In the plotted seed-0 example, the baseline's simplified-Markov unavailability
is ~0.878 seconds per RSM-day and the budget-constrained availability gain is
only ~0.013 seconds/day (`rho0`) or ~0.011 seconds/day (`rho1`). A one-day TiDB
run cannot resolve that difference. The figure in
`outputs/exec-02/placement-frontier.{html,png,svg}` follows the OSDI draft's
two-panel layout, but replaces its mock points with these modeled estimates;
no independent oracle or observed TiDB point is plotted. The figure's circle
at the baseline target is **cost-minimizing**, so it is expected to have almost
no vertical availability margin. The savings rely on assigned shadow prices;
identical physical AWS instances have the same actual bill at equal fleet size.
