# AZ-aware hybrid placement study

Status: local MC study complete (2026-10-28). No cloud resources.

## Four figures

All figures use 10,000 RSMs, one frozen on-demand catalogue, four US regions
with three logical AZs each, capacity 500 distinct RSM replicas per machine,
and the versioned `scenario/canonical-central-az-aware@v2`. Each has
independent-price and reliability-ranked-price panels:

1. RF 3 fixed: price-blind and cheapest-first baseline marks at RF 3, plus an
   ILP cost–availability frontier choosing machines for three-replica RSMs.
2. RF 5 fixed: the same at RF 5.
3. RF 7 fixed: the same at RF 7.
4. RF free choice: one ILP frontier that may select RF 3, 5 or 7 independently
   per RSM, with **all six fixed-RF baseline marks** in each price panel.
   The solution's actual RF counts accompany every budget point; there is no
   imposed mixture.

No candidate has a regional quorum or repeated AZ. The independent AZ and
per-region processes are sampled by Rust MC; the former all-region WAN process
is retained separately as stress metadata and is **not** injected in this
primary scenario. This is a partial *modeling sensitivity*, not proof that
cross-region WAN degradation is harmless. The old regional process is still
conservative: incidents affecting two AZs may count as all three.

## Availability and cost contract

Markov binning screens out candidates below 4.5 nines without MC. Each
survivor in the bounded candidate menu gets at least **100,000 full-horizon
Rust MC runs**, with a 99% Bonferroni-adjusted Student-t interval for repeated
looks and the number of candidates. The user selected **k=4.5 nines**.
The run cap for this local study is 1 million per candidate; candidates still
straddling the threshold are undecided and cannot enter a target-qualified
ILP. The full cost frontier deliberately includes MC-scored failures and
undecided candidates down to the cheapest baseline budget, with such segments
dashed; a target-qualified ILP point is drawn only when its candidate set is
certified. Points show mean *per-RSM* availability, not the probability that
none of 10,000 groups is down. Prices are assigned shadow prices, not identical
physical AWS hosts' bill.

The menu uses 20 disjoint, capacity-500 RSM blocks for each baseline and RF;
this covers the arithmetic minimum of 60/100/140 distinct machines at RF 3/5/7.
This run uses `--extras 0`: only the fixed-RF baseline block placements are
available as ILP columns, and the free-choice graph uses their union. Thus
these are **candidate-menu frontiers**, not an oracle or proof of global
optimality; adding price-aware columns may improve them. MC results are
cached by sorted machine IDs and reused across price assignments. The study
streams completed candidate decisions to `outputs/exec-02/az-mc-scores.jsonl`
for safe resumption; a fingerprint prevents reuse under a different catalogue,
configuration or candidate menu.

```bash
python3 scripts/experiment_az_hybrid_frontiers.py --extras 0 --max-runs 1000000 \
  --frontier-points 5 --workers 4
python3 scripts/render_az_frontiers.py
rsvg-convert --zoom 1.5 outputs/exec-02/az-frontier-choice.svg \
  -o outputs/exec-02/az-frontier-choice.png
```

Three initial full-rate pilots under the AZ-aware model should be interpreted separately
from the old all-region stress: initial 100,000-run cheapest-first RSM probes
were around 4.35, 4.55 and 4.82 nines at RF 3/5/7. They do not bound the ILP's
best candidates. The low-count two-event AZ proxy and uncertain incident scope
also require sensitivity analysis before submission-ready SLA claims.

## Measured local results

The checkpoint contains **180 unique placements / 32.1 million MC runs**. All
60 RF 3 candidates failed to certify 4.5 nines; of 60 RF 5 candidates, 57
passed and 3 were undecided at the one-million-run cap; all 60 RF 7 candidates
passed. None had a lower confidence bound above five nines. Every graph uses
these same cached MC estimates for both price assignments. All baseline and
frontier points shown are simulation results, not observed TiDB behavior.

| Qualified minimum modeled cost at 4.5 nines | Independent price | Reliability-ranked price |
|---|---:|---:|
| Uniform RF 3 | no certified solution | no certified solution |
| Uniform RF 5 | $35.572/h, all RF 5 | $37.129/h, all RF 5 |
| Uniform RF 7 | $50.553/h, all RF 7 | $50.630/h, all RF 7 |
| **ILP chooses 3/5/7** | **$35.572/h; 10,000 RF 5** | **$37.117/h; 9,500 RF 5 + 500 RF 7** |

The free-choice optimizer was **not** constrained to mix. At lower cost
budgets its MC-scored curve includes RF 3 placements below the target; at
higher budgets it moves through mixed RF 5/7 placements toward all RF 7.
The six fixed-RF baseline marks per price panel remain visible on the
free-choice graph. The cheapest-first RF 5 baseline has 19/20 or 18/20
certified placement blocks (independent/ranked), so its lower sticker price
is **not** an all-RSM 4.5-nines certificate.

Figures: `outputs/exec-02/az-frontier-{rf3,rf5,rf7,choice}.{svg,png}`.
Their blue full-frontier segments are dashed when any selected RSM lacks
certification; the outlined target circle appears only for a feasible
all-certified placement. MC error bars derive from the candidate intervals,
not from independent physical deployments. The RF-count table on the
free-choice figure shows actual selections at every budget, including a
9,000 RF 5 / 1,000 RF 7 point in both price panels.

## 24-hour one-shot overlay

`outputs/exec-02/az-frontier-choice-24h.png` overlays the free-choice figure
with 1000 simulated 24 h days of every ILP deployment (frontier points and the
target-qualified point), plotted at the deployment's price (±0.6 $/h jitter);
a black bar is the mean of the days. A day's value is the RSM-weighted mean of
each placement group's 24 h Rust MC availability. Groups use disjoint machines
and are simulated separately; AZ/region outages are drawn once per day and
injected into every group (4 of 1000 days had one). A machine set uses the same
seed on a given day in every deployment, so neighbouring deployments share most
draws. Injected outages use constant delays and repeat if 2t + d < 24 h
(flagged `may_repeat`; none in this run).

Why the days sit above the 30-day line (2026-10-28 diagnosis):

1. **Skew.** Downtime is concentrated in rare bad days, so the median day is
   well above the mean: for all-RF 3, the median day is 5.2 nines, the mean of
   days 4.7, and the worst 1 % of days hold ~40 % of downtime (4 % for RF 7).
2. **The engine is not stationary when AZ/region outages are on.** Without
   network outages, one RF 3 group's mean unavailability is flat with horizon
   (1.4e-5 at 1, 7 and 30 days). With them it rises: 1.5e-5 (1 d), 2.1e-5
   (7 d), 4.8e-5 (30 d). With a single AZ process at 1/(10 d) it rises
   100-fold (1.5e-4 → 1.6e-2); with replacement disabled only 3-fold. An
   isolated outage at day 15 costs a bounded ~0.76 h for RF 3, so it is
   damage from *repeated* outages that compounds through the replacement path
   (replacements copy the failed node's region/AZ config). It is unresolved
   whether this is a bug or intended; until it is, the 30-day frontier and its
   certification are partly an artifact of horizon length, and 24 h days from a
   clean start are optimistic about network events.

   **Resolved (branch `exec02-stuck-runs`): three engine bugs, both engines.**
   Event logs of the worst runs showed two failure paths after outages:
   - *Removed leader.* An outage's replacements are promoted after it ends;
     the trim back to target broke ties by removing the newest node, which
     was often the Raft leader. Elections start only when the leader fails,
     so `leader_id` dangled and the group stayed down to the horizon (one RF 7
     run: 324 h). Trims now keep the leader, and removing it starts an election.
   - *Placement drift.* The trim ran before the same pass's promotions took
     effect, so the cluster stayed over target until the next event, often a
     failure, whose trim then removed the failed sole member of a domain. The
     group kept two replicas in one AZ and the next outage there took it down
     for the full outage (one RF 3 run: 17.4 h). Trims now count same-pass
     promotions and prefer to remove a node that shares its AZ/region.
   After the fix, unavailability no longer depends on horizon (RF 3: 4.82 /
   4.76 / 4.81 nines at 1 / 7 / 30 d; RF 7: 5.08 at all three) and the worst
   200 of 20,000 30-day runs hold 27 % (RF 3) and 3 % (RF 7) of downtime,
   down from 75 % and 78 %. The two probed frontier groups move from 4.38 to
   4.81 (RF 3, previously failing 4.5) and from 4.85 to 5.08 (RF 7). **Every
   MC score, certification and frontier above predates the fix and must be
   re-run.**

```bash
python3 scripts/experiment_az_24h_oneshot.py --trials 1000
python3 scripts/render_az_frontiers.py --oneshot outputs/exec-02/az-24h-oneshot.json
rsvg-convert --zoom 1.5 outputs/exec-02/az-frontier-choice-24h.svg \
  -o outputs/exec-02/az-frontier-choice-24h.png
```
