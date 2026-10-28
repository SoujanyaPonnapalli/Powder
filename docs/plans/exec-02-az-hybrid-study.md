# AZ-aware hybrid placement study

Status: local MC experiment in progress (2026-10-28). No cloud resources.

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
The alternative candidate sample is bounded, so the figures are **candidate-
menu frontiers**, not an oracle or proof of global optimality. MC results are
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
