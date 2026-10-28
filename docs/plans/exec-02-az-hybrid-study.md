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

The low-count two-event AZ proxy and uncertain incident scope also require
sensitivity analysis before submission-ready SLA claims.

## Measured local results

Re-run 2026-10-28 with the replacement-trim engine fix (`5affbb1`, checkpoint
engine tag `trim-v2`); the first run's numbers were an artifact of runs stuck
after AZ/region outages (see the 24-hour section). The checkpoint contains
**180 unique placements / 18.9 million MC runs**. 59 of 60 RF 3 candidates
certify 4.5 nines and 1 is undecided at the one-million-run cap; all 60 RF 5
and all 60 RF 7 candidates pass. 98 candidates (RF 5/7) have a lower
confidence bound above five nines. Every graph uses these same cached MC
estimates for both price assignments. All baseline and frontier points shown
are simulation results, not observed TiDB behavior.

| Qualified minimum modeled cost at 4.5 nines | Independent price | Reliability-ranked price |
|---|---:|---:|
| Uniform RF 3 | $20.359/h, 4.82 nines | $20.586/h, 4.74 nines |
| Uniform RF 5 | $34.938/h, 5.05 nines | $35.107/h, 5.02 nines |
| Uniform RF 7 | $50.553/h, 5.07 nines | $50.630/h, 5.05 nines |
| **ILP chooses 3/5/7** | **$20.359/h; 10,000 RF 3** | **$20.586/h; 10,000 RF 3** |

**4.5 nines no longer separates the policies.** Every fixed-RF baseline now
clears it; the target-qualified ILP is the cheapest all-RF 3 placement, and
the free-choice frontier is nearly flat above RF 5 (5.03–5.08 nines from
$37/h to $85/h). The free-choice optimizer was **not** constrained to mix: it
picks 9,000 RF 5 + 1,000 RF 7 at $37/h and mostly RF 7 by $52–54/h. The only
uncertified baseline block is one of the 20 in the ranked-price
cheapest-first RF 3 baseline (19/20). A target nearer five nines would make
the comparison informative again; that is an open choice for the study.

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

With the fixed engine the mean of the 1000 days matches the 30-day line
(e.g. 5.03 vs 5.05, 5.09 vs 5.07 nines; all-RF 3 4.74 vs 4.82). Individual
days still spread widely because of **skew**: downtime is concentrated in
rare bad days, so the median day is well above the mean (all-RF 3: median
day 5.2 nines, mean 4.7; the worst 1 % of days hold ~40 % of downtime, ~4–7 %
for RF 5/7). A single real 24 h run will usually look better than the line.

History (2026-10-28): before the fix the days' mean sat 0.25–0.4 nines above
the 30-day line, because the engine was **not stationary when AZ/region
outages were on**. Without network outages, one RF 3 group's mean
unavailability was flat with horizon (1.4e-5 at 1, 7 and 30 days); with
them it rose 1.5e-5 (1 d), 2.1e-5 (7 d), 4.8e-5 (30 d), and 100-fold with a
single AZ process at 1/(10 d).

   **Resolved (`5affbb1`): two engine bugs, fixed in both engines.**
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
   4.81 (RF 3, previously failing 4.5) and from 4.85 to 5.08 (RF 7). The
   study, figures and overlay above were re-run with the fix.

```bash
python3 scripts/experiment_az_24h_oneshot.py --trials 1000
python3 scripts/render_az_frontiers.py --oneshot outputs/exec-02/az-24h-oneshot.json
rsvg-convert --zoom 1.5 outputs/exec-02/az-frontier-choice-24h.svg \
  -o outputs/exec-02/az-frontier-choice-24h.png
```
