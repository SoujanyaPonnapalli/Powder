# Hybrid Markov–Monte Carlo availability gate

Status: implementation on `exec02-experiment-sizing`, 2026-10-28. This is the
second, independent change after the RF 3/5/7 figure commit `4eb26f0`.

## Decision

`solve_catalogue_placement` defaults to `scoring="hybrid"`:

1. Generate candidate placements and score them with the existing rate-binned,
   simplified Markov model. A candidate below **4.5 nines**
   (`A < 1 - 10**(-4.5)`) is discarded before any Monte Carlo work. The binned
   approximation is a screening estimate, not a proof of a lower bound.
2. Retain at most 100 Markov survivors by alternating cheapest and most
   available candidates. This is a **tractability cap**, not an optimality
   guarantee; diagnostics record how many candidates the cap omitted.
3. For every shortlisted candidate, run Rust powder-mc on the catalogue's
   full time-varying per-machine hazard curves for the catalogue horizon
   (normally 30 days). The current scenario's weekly planned downtime remains
   an independent exponential approximation, and the per-region and global
   outage processes are enabled by default. Each candidate has a stable seed
   stream, and successive batches use disjoint seed ranges.
4. Take **at least 100,000 runs** in batches of 10,000. Compute a two-sided
   Student-t interval on *full-horizon per-run availability*. The default
   target is configurable **k = 5 nines**, with 99% nominal familywise
   confidence. The t critical value receives a Bonferroni adjustment for the
   maximum number of inspection points and shortlisted candidates. After the
   minimum runs, the candidate **passes** only if the interval is wholly above
   `1 - 10**(-k)`, **fails** if wholly below, and otherwise continues to a cap of
   10 million runs. A candidate still straddling the target is **undecided**;
   it is excluded from the ILP. Zero-variance perfect samples cannot pass.
5. The ILP sees only passing candidates and uses their **Monte Carlo mean
   availability**, not their Markov score. Machine prices remain exact and
   physical machines are billed once. No passing candidate means an explicit
   no-qualified-candidates result, not a silent relaxation of the SLA.

The legacy `scoring="markov"` path remains available for inexpensive screening
and regression tests. `MCScoringConfig` controls k, confidence, seeds, number
of candidates, batch size and run cap; the production minimum cannot be set
below 100,000. To request a different SLA target while keeping the minimum:

```python
from powder.faults.mc_scoring import MCScoringConfig
from powder.faults.ilp import solve_catalogue_placement

result = solve_catalogue_placement(lib, catalogue, protocol, strategy,
                                   mc_config=MCScoringConfig(target_nines=k))
```

## What this does not establish

A nominal t interval is not automatically well calibrated for the library's
rare, heavy-tailed outages. Bonferroni adjustment addresses repeated looks
and multiple candidates *conditional on the t approximation*, not a
mis-specified sampling distribution. The repository's coverage study already
found that early-stopped naive availability intervals undercovered. Run
independent coverage checks before treating a passing interval as a paper
claim. The 100-candidate menu is also not a global placement optimum.

The Markov prefilter omits the regional/global process, while the default MC
includes it. A local 100,000-run end-to-end smoke check on one three-replica
candidate passed the Markov screen at **0.9999894**, but its MC availability
was **0.9995195**, with the adjusted interval **[0.9994979, 0.9995411]**.
It was rejected at five nines. That one candidate does not prove all fail. The
hybrid solver may legitimately return no placements until the region model or
availability requirement is revised. On the same candidate with regional
outages explicitly disabled and `k=4.5`, the 100,000-run test passed at
**0.9999875** (adjusted interval **[0.9999872, 0.9999878]**) and its MC score
entered a feasible one-RSM ILP. This is a sensitivity check, not permission
to omit outages from a real SLA claim. A 30-day mean-RSM score also does not
measure the chance that *any* of 10,000 RSMs sharing machines is down.

## Reproduction / checks

```bash
python3 -m pytest -q tests/test_mc_candidate_scoring.py tests/test_hybrid_catalogue_ilp.py
python3 -m pytest -q tests
```

No additional VM is required. The scorer launches only the existing local
Rust Monte Carlo binary. Long, multi-candidate production runs must be given
a deliberate budget; the 100,000-run minimum applies to **each** candidate.
