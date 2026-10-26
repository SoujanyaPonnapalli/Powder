# Plan: Exec #1b — A catalogue of 1000+ machines drawn from the fault library

Follows exec #1 (`exec-01-canonical-failure-data.md`). Status: plan + first steps (2026-10-26).

## 0. Decisions (user, 2026-10-26)

- **Catalogue, not instances.** Build a catalogue of >= 1000 machines, each uniquely composed from the
  data sources. The Markov model (exponential approximations) and the Monte Carlo simulator score
  placements drawn from it, the ILP chooses a placement, TiDB runs on AWS in the chosen regions, and the
  emulated machines fail following the *same distributions* (not the same timings) as in the simulator.
- **Recoverability must be known.** A source whose records do not say whether data survived cannot set
  a machine's rates. Excluded: Birke DSN'14 (crash tickets), LANL 1996-2005 (its FAQ: "the data does not
  contain this information"), Vishwanath's non-disk repairs. Their profiles stay in the library, tagged
  `recoverability-unknown`, and are never instantiated. Consequence: the AWS profile's transient causes
  now come from Ford only, and non-disk host loss has no source (rate 0, a documented optimistic gap).
- **Richer rate models.** Replace constant rates by distributions of constant rates where a source has
  between-unit spread, and use time-varying curves wherever a source has them.
- **Frailty stays off.** Differences between machines come from composition and from the rate models
  below, not from per-node multipliers.
- **Markov model**: horizon-average exponential approximation of each machine (the ILP itself only
  consumes candidate scores, so it needs no change in rate semantics).
- **Backblaze raw drive stats 2023-2025**: approved. Blocked by the egress allowlist (`www.backblaze.com`,
  `f001.backblazeb2.com`); the user can add the hosts or drop the zips into the repo root, as for SpotLake.

## 1. Rate models

Every cause of a machine is one of four rate models (new `rate_model` field next to `rate_per_hour`;
existing entries stay valid as `constant`):

| Code | Model | Meaning when a machine is instantiated |
|---|---|---|
| C | `constant` | the machine gets this rate (low / central / high as before) |
| D | `population` | the machine draws one constant rate from a distribution over units (models, systems, cells, pools); stored as weighted members (`{rate, weight, member_id}`) and, where useful, a fitted lognormal |
| T | `curve` | the rate is a function of time: `age` (hazard by unit age; e.g. drive age) or `calendar` (a rate time series; e.g. the spot band series) — stored as a piecewise-constant hazard |
| DT | `curve_population` | the machine draws one curve from a set of curves (one per member) |

Age curves need an age at t = 0: each machine draws it from the source's age distribution (Backblaze
drive-days by age), and a replacement draws a new one (a replaced instance lands on another host).
Calendar curves need an offset: each machine (or each pool, see §4) draws a start point in the series.

## 2. What each usable source gives (recoverability known)

| Cause | Source | Model available | Notes |
|---|---|---|---|
| `hw_reboot` (transient) | Ford OSDI'10 unplanned reboots | D (Table 1: avg / min / max over tens of cells, coarse) and T (Fig 3: daily rate, 92 days, one cell) | x (1 - f_LM) for live migration |
| `sw` (transient) | Ford node restarts (>= 15 min) | D (Table 1) and T (Fig 3) | |
| `pause` (transient) | Azure published bound | C | |
| `disk` (permanent) | Backblaze drive stats 2023-2025 | D (per model) and DT (per-model hazard by age) | blocked on allowlist; HDD + boot SSD fleets, a proxy for NVMe |
| | Schroeder & Gibson FAST'07 | D (per-system replacement rates) and T (replacement rate by age) | PDF from www.usenix.org (allowed); digitize like Ford |
| | Vishwanath disk AFR 2.7%, Ford Table 2 disk MTTF | C | |
| `spot_interrupt` (permanent) | SpotLake 2025-26 | T per pool (daily band series) and DT across 289 US 8 vCPU x86 pools | data already extracted |
| `hw_host_loss` (permanent) | none | — | gap |
| Recovery durations | Ford Fig 2 (per cause) | duration distributions | unchanged |
| Region outages (scenario) | Azure PIRs | D (EB posterior per region) | unchanged |

The transient side is thin (Ford only); most machine-to-machine diversity will come from disks and spot.

## 3. Composition and uniqueness

A **recipe** chooses, per cause slot, a source and (for D / DT) a member; a **machine** is a recipe plus a
draw (rate from a population, age, calendar offset), all from a seeded stream. The catalogue
(`data/fault_library/catalogue/machines-<seed>.json.gz` + a generator script) records, per machine:
recipe, drawn parameters, member ids, and the source measurement for every number. Rules:

1. only `recoverability-known` sources; all-cause sources fill all their slots together;
2. no slot double counts another (e.g. a disk component is never added to a source that already counts disks);
3. uniqueness on (recipe, members); continuous draws make machines distinct beyond that;
4. capacity check before generation: with ~N_disk Backblaze models x 289 + 1 spot options x Ford draws the
   number of distinct recipes is far above 1000 (without Backblaze: 6 disk components x 290 x Ford draws).

## 4. Engines

- **Markov model**: `machine.markov_rates(horizon, age)` = horizon average of each cause's hazard
  (∫h / horizon) → the existing `NodeConfig` with exponentials. Durations unchanged.
- **Python simulator + traces** (`powder.faults.trace`): non-homogeneous Poisson sampling by inverting the
  cumulative hazard from the current age / time (age continues through transient recoveries, resets on
  replacement). Calendar curves for spot pools are **shared by all machines in the same pool** (that is
  what the data shows; it also induces correlated reclaims, an input for exec #3).
- **Rust powder-mc**: a time-dependent hazard `DistributionSpec` (`piecewise_hazard` with `clock: age |
  calendar`), sampled given the node's age or the simulation time; the node keeps an age clock.
- **Emulation**: traces from the same machine records, so emulated machines follow the same
  distributions as the simulator (different timings per seed).

## 5. ILP at catalogue scale

1000 distinct types make type-count candidate enumeration explode. Options to evaluate with the existing
`test_rsm_ilp_scaling_benchmark`: cluster machines into rate bins for candidate generation and rescore the
solution with the exact machines; column generation; Markov-score caching keyed on sorted rate tuples.

## 6. Steps (each its own commit)

1. [x] Apply the recoverability decision (Birke, LANL, Vishwanath non-disk excluded; AWS profile on Ford;
   SLA clip of the pessimistic end; reference scenarios updated).
2. [x] Schema + library support for `population`, `curve`, `curve_population` (`powder/faults/ratemodels.py`: piecewise hazards, NHPP sampling, horizon average, validation).
3. [x] Populate from data in hand (`build_rate_models.py`): Ford hw_reboot/sw = lognormal population of cell levels (mean = Table 1 average; sigma from the Table 1 max as the top quantile of n = 30 cells, capped at sigma = z = 1.85 because max/mean exceeds what 30 lognormal draws reach: a lower bound on spread) x Fig 3 relative daily shape (calendar, 91-day period); SpotLake: one calendar curve per pool (289 US pools, 396 days, offset shared by the pool).
4. [ ] Schroeder & Gibson FAST'07: fetch PDF, digitize per-system rates and the age curve.
5. [x] Backblaze 2023-2025 (12 quarters, 94 models, 314M drive-days; `extract_backblaze.py`): `disk/backblaze-models` (40 models with >= 500 drive-years; AFR 0.2%-13%) and `disk/backblaze-models-age` (39 models with SMART 9 ages; yearly hazards merged to >= 200 drive-years; machine age drawn from the model's drive-day age mix). Fleet AFR: HDD 1.56%, boot SSD 0.51%.
6. [x] Catalogue generator (`powder/faults/catalogue.py`, `build_catalogue.py` -> `catalogue/machines-n1000-seed0.json.gz`): recipe = transient base (aws-ondemand-nvme or ford10-ge15min) x disk option (44: Backblaze models + literature components) x spot pool (289) or on-demand x region. Capacity: 352 on-demand recipes, 25,432 spot recipes; on-demand recipes repeat only after all are used, with independent draws (validated: no two machines share all rates). Default 1000 machines, 30% spot; on-demand prices assigned (rho 0 and 1), spot prices = pool's SpotLake median. `node_config()` gives the 30-day horizon-average exponential machine for the Markov model.
7. [x] Python simulator / trace NHPP support; Markov horizon average. `TimeVaryingHazard` / `PiecewiseRate` in `powder.simulation.distributions` (inverse cumulative hazard; the simulator samples failure and data-loss clocks from the current time via `sample_delay`, and spawned nodes redraw age clocks); `catalogue.node_config(time_varying=True)`; `catalogue.trace()` for emulation (per-cause NHPP clocks, scenario planned/region events shared with `trace.generate`).
8. [x] Rust: `Distribution::TimeVarying` (spec `{"type": "time_varying", "parts": [...]}`) sampled with `sample_at(now)` for failure/data-loss clocks; spawned nodes redraw age clocks; `mc_jobs.distribution_spec` emits it. Cross-check: 3 nodes with a 10-days-on/10-off calendar hazard give availability 0.9210 in powder-mc vs 0.9219 analytic.
9. [ ] ILP scaling for 1000 types.

## 7. Open questions

- Host loss has no data. Keep it at 0 (optimistic, stated) or ask exec #10 / AWS retirement notices for a number later?
- The AWS on-demand profile and `cluster/ford10-google-storage-ge15min` are now nearly the same machine; fine for now.
