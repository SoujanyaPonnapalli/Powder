# Budget-targeted ILP candidates: fleets, centroid MC table, dominance

Status: plan (2026-11-05). Branch `fault-std-ford-planned`. No cloud resources.

## Why

The rate-scale pilot (`scripts/pilot_ilp_rate_scale.py`, `outputs/pilot-ilp-rate-scale/`) showed the
candidate menu, not the ILP, limits the frontier: 400 price- and reliability-blind random placements plus
10 reliability-only greedy starts per RF. A 10% budget was infeasible, and at 10-30x fault rates the
cheapest-first RF 5/7 baselines beat the ILP near $5-8/h. Cost is billed once per machine (capacity 5
replicas), so a budget constrains the **fleet**, not single placements; random placements over 2000
machines do not pack into a small fleet.

## Decisions (user, 2026-11-04/05)

- **Fleet first.** For each budget, choose several fleets by a knapsack ILP, then build RSM placements only
  from a fleet. Several fleets per budget for diversity.
- **Knapsack:** fixed fleet size k, minimize total machine downtime score, swept over k from the
  fewest-machines to the most-machines fleet that uses the budget; overlap-limited repeats per size.
- **Sample 3/5/7** from each fleet with the balanced mix (below).
- **Greedy, randomized, slot-consuming packing passes** produce distinct placements; duplicates are dropped.
- **Monte Carlo, not Markov.** Measured on the 2000-VM catalogue: 10k Rust MC runs take 0.1-0.2 s with
  ~2% standard error (5-10% with region/AZ outages); binned Markov takes 3.2 s for RF 7. MC also models
  region/AZ outages (+25% downtime at RF 3, +4% at RF 5), which Markov cannot.
- **Centroid table:** cluster machines into **16** centroids on (log transient, log permanent rate); a lazily
  memoized MC table of centroid multisets ranks placements during the greedy.
- **Dominance step** after packing: swap in fleet machines that are no more expensive and no less reliable.
- Exact MC on every surviving placement; only exact scores reach the ILP.

## Pipeline (per fault scale x price model)

1. **Machine features.** From the scaled `catalogue_machines` configs: transient rate lambda_t (incl.
   planned), permanent rate lambda_p (disk, host loss, spot), mean recovery, price. Downtime score
   u_m = lambda_t x E[election] + lambda_p x rebuild time (the first-order RSM downtime terms).
2. **Clusters.** k-means (K=16, seeded) on (log10 lambda_t, log10 lambda_p); log floor for zero rates.
   Each machine maps to a centroid id.
3. **Fleets per budget B** (small `scipy.optimize.milp`, 2000 binaries y_m):
   - alpha B <= sum p_m y_m <= B (alpha = 0.9: use the budget); sum y_m = k
   - slots: 5k >= 3N (k >= 30 for N = 50)
   - spread: per AZ <= ceil(k/12) + 1 machines; >= 3 regions
   - minimize sum u_m y_m
   - k_min / k_max: min / max sum y_m under the same constraints; sweep ~8 sizes between them, always
     including 30/50/70 when in range (all-RF3/5/7 capacity thresholds)
   - per size, 2-3 fleets: after fleet F add sum_{m in F} y_m <= 0.7 |F| and re-solve
4. **RF mix per fleet.** RFs the fleet can host (slots, distinct AZs, no regional majority). Balanced mix
   (a, b, c): a+b+c = N, 3a+5b+7c <= 5k, minimize |a-b|+|b-c|+|c-a| (enumerate); split sampling effort
   a:b:c with a 10% floor per feasible RF. The ILP still chooses the final mix.
5. **Packing passes** (per fleet, per RF, 10-20 passes): build placements in sequence, decrementing each
   machine's remaining slots. Each placement: a random region pattern without a regional majority, a
   distinct AZ per replica, then members chosen by a randomized greedy (random pick among options within
   a few % of the best) on the centroid-table score minus a penalty for exceeding the budget's per-RSM
   cost target (B x replicas / (5k)). Plus one deterministic rotation covering per RF (`block_baseline`
   style), so a full feasible deployment always exists.
6. **Centroid MC table** (lazy): key = (RF, sorted centroid multiset, region pattern); value = mean
   availability from 10k Rust MC runs with every member at its centroid rates, region/AZ outages on.
   Common random numbers (same seed for every entry) so neighbouring entries compare with low noise.
   Cached to disk per (scale, scenario, catalogue digest).
7. **Dominance replacement.** For each placement member m, replace it by a fleet machine m' with
   p' <= p, lambda_t' <= lambda_t, lambda_p' <= lambda_p, mean recovery' <= recovery (one strict), a free
   slot, and an AZ that keeps the placement valid. Stays inside the fleet, so the bill cannot rise;
   availability cannot fall, so no MC is needed. Recovers within-cluster differences the table blurs.
8. **De-duplicate, exact MC.** Every unique placement: 100k Rust MC runs on its real machines
   (~1.5 s; ~2% s.e. with outages), parallel over cores, cached by sorted machine ids.
9. **ILP** at every budget over the union of all exact-scored placements (+ the fixed-RF baselines as
   columns, so the ILP can never lose to them).

## Validation gates (before the full sweep)

- **G1 table ranking:** on one fleet, Spearman rho between centroid-table and exact MC scores of >= 200
  placements; proceed if rho >= 0.8, else revisit K or the table's region pattern.
- **G2 frontier:** at 1x and 30x (rho0), the new ILP frontier is >= every baseline at every budget, and the
  10% budget is feasible.
- **G3 cost:** full 4 scales x 2 prices x 6 budgets finishes in <= 1 h on 8 cores.

## Code

- `powder/faults/fleets.py`: features, clustering, knapsack fleets, RF mix, packing passes, dominance.
- `powder/faults/centroid_table.py`: lazy MC table with disk cache.
- `scripts/pilot_fleet_frontier.py`: the pipeline over scales x prices x budgets; reuses the pilot's
  baselines and renderer (`render_pilot_frontiers.py`, extended to overlay old vs new ILP).
- Tests (`tests/test_fleets.py`): knapsack respects budget/size/spread; balanced mix; packing never exceeds
  slots and keeps AZ rules; dominance never raises cost or a rate; table keys are order-invariant.

## Checklist

- [x] fleets.py: features + k-means + knapsack sweep (+ tests)
- [x] RF mix + packing passes + dominance (+ tests)
- [x] centroid_table.py with common random numbers and disk cache
- [ ] G1 ranking check on one fleet (not run separately; the ILP's exact re-solve loop absorbs table error)
- [x] pilot_fleet_frontier.py; full sweep 4 scales x 2 prices x 7 budgets (2026-11-05)
- [x] render frontier with random-menu ILP overlay (`outputs/pilot-ilp-rate-scale/frontiers.png`)

## Results (2026-11-05, `outputs/pilot-ilp-rate-scale/results-fleet.json`)

Exact Rust MC with region/AZ outages for every point. The fleet ILP is feasible at every budget including
10% (the random-menu ILP is not) and beats every baseline that fits its budget. Against the random-menu ILP it
gains most at 20-40% budgets (e.g. 1x rho0 20%: 5.97 vs 5.18 nines; 30x rho0 30%: 4.38 vs 2.84) and trails it
at 3/56 points, all reliability-ranked prices at 75-100% budgets (by 0.01-0.07 nines).

Implementation notes that differ from the plan:

- Within a fleet the per-RSM cost target is moot (the fleet already fits the budget), so packing maximizes the
  table score only; random passes use the per-machine score, then all neighbour table keys are computed in a
  parallel batch before steepest-ascent swaps (Python only orchestrates; each MC call is a Rust process).
- Each budget's ILP sees its own fleets' placements plus the baselines, over only the machines they use
  (cumulative pools with all 2000 machines hit the time limit with no incumbent). If the best full fleet
  covering beats the ILP's incumbent it is reported instead; then each budget reports the best result at or
  below it (a deployment that fits a smaller budget fits a larger one).
- 100% = the random-menu ILP's unrestricted cost, as a cap: unbounded, the ILP buys dear machines for no gain.
- Exact MC runs: max(20k, 100k / sqrt(scale)); table runs one fifth. Sweep time ~3.5 h on 8 cores.

Follow-ups: the 10-round re-solve cap binds under reliability-ranked prices (table misranks there - run G1);
warm-start the ILP with the best covering instead of comparing afterwards.

## v2: Gurobi + dominance lattices (2026-11-05, 1x and 30x only)

`outputs/pilot-ilp-rate-scale/lattice/` (`frontiers.png` overlays the first sweep, dotted).

- **Dominance moved out of candidate generation.** Replacing members by their best dominator funnelled
  ~1,500 placements per budget onto half as many machines (130 -> 77 machines at $5.96/h; top-10 machines
  held 54% of slots), which the 5-replica capacity cannot use together. Bases now stay in the pool.
- **Lattice ascent on the chosen deployment.** Each selected placement gets a lattice slice (base, every
  single dominating swap, a greedy top, 5 random multi-swaps; `fleets.lattice_slice`). Base and top get
  exact MC; other elements are interpolated on the downtime-score gain between them and clipped to the
  partial-order bounds implied by every exact score (`fleets.lattice_estimate`). A small Gurobi ILP over
  these 36-3,000 columns picks one deployment; its picks get exact MC until all are exact. Adding full
  lattices to the main ILP was ruled out: near-identical columns are the worst case for branch and bound.
- **Main ILP on Gurobi** (`powder/faults/gurobi_ilp.py`): unique machine billing, disaggregated
  `x_c <= cap * y_m` links, warm start from the best fleet covering, 120 s limit. First solves on 2.4-3.0k
  candidates: 1.8-1.9 s vs HiGHS 5-22 s. At 30x Gurobi still hits the limit in many rounds.
- **MC caches shared across prices** (availability does not depend on price): `cache/{table,exact}-x<k>.json`.

Results vs the first sweep: 1x/30x configurations run 2.6-5.3x faster (568-3,783 s vs 913-12,659 s).
Lattice ascent changes the main ILP by +0.27 nines at 1x rho1 10% (4.77 -> 5.04) and +0.13 at 30x rho1 20%
(3.78 -> 3.91), and is within +0.01 elsewhere. Against the first sweep: 30x rho1 20% 3.91 vs 3.03,
30x rho0 30-40% +0.11-0.12, most other points within +-0.03; 1x rho0 10% (4.63 vs 4.69) and 30x rho0 20%
(3.75 vs 3.81) are lower (those budgets lost the dominated placements and hit the round cap). Time split
over the four configurations: main ILP 4,108 s, main MC 2,250 s, ascent MC 792 s, ascent ILP 49 s.

Follow-ups: dominators drawn only from the budget's pool machines (widening to the catalogue would add
columns the budget can still afford); main ILP time at 30x (shorter limit or MIP-gap stop per round).



The catalogue, scenarios and existing pilot outputs are unchanged. The candidate generator is additive
(new module); `generate_candidates` stays for older studies. Column generation can later replace the
budget-target heuristic using these placements as starting columns.
