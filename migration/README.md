# Migration parity harness (temporary)

**This directory is temporary.** It exists to show that the Rust port in
`rust/` behaves like the Python engine in `powder/`. Once you are satisfied
with that, delete it:

```sh
rm -rf migration/
```

Nothing else depends on it. `pytest.ini` sets `testpaths = tests`, so a bare
`pytest` never picks this up — you have to ask for it by name.

## Running it

```sh
.venv/bin/python -m pytest migration/ -q
```

The first run builds `rust/target/release/powder-mc` (about a minute from
cold), then keeps one `--stream` process alive for the whole session. The
suite takes roughly 30 seconds after that.

```sh
.venv/bin/python migration/bench.py          # Python vs Rust wall clock
.venv/bin/python migration/bench.py --quick  # smaller smoke-test workload
```

## How the comparison works

`scenarios.py` is the single source of truth. Each scenario is declared once
and knows how to emit **both** the Python engine objects and the JSON job the
Rust binary reads, so the two sides cannot quietly drift apart.

The port does **not** reproduce NumPy's RNG stream — that was a deliberate
decision, since matching it would have forced CPython's float quirks and
NumPy's ziggurat tables into the Rust. So parity is established in two
layers:

### Deterministic layer

Scenarios built entirely from `Constant` distributions. With no randomness
both engines walk the same event sequence, so they are compared at
`rtol = 1e-9` on floats and **exactly** on integer counters, end reasons and
data-loss milestones. This is a tolerance check, not a bitwise one, and it is
the layer that would catch an outright logic divergence.

Coverage: staggered transient failures, cascading data loss, Raft leader
election, forced-snapshot sync under log GC, node replacement, region
outages, adaptive scaling, standby and provisioning nodes, a cluster that
starts inside an outage, a run that drains its event queue, and the rare
path where a run ends on data loss with a *lagging* survivor rather than a
third loss event.

### Statistical layer

Randomised scenarios spanning both protocols, all three strategies, all five
distribution families, region outages, and an unbounded run-to-data-loss
workload for MTTDL. Each metric is compared with:

* Welch's t-test on the means,
* a two-sample Kolmogorov-Smirnov test on the distributions, and
* a magnitude guard on the difference in means.

The magnitude guard scales with the metric's own standard error
(`max(rtol * mean, 4 * SE)`). Several of these counters are badly
over-dispersed — most runs record zero unavailability incidents and a few
record dozens — so their sampling noise alone exceeds any fixed relative
tolerance. A flat tolerance there would reject samples that both hypothesis
tests accept.

The Python reference always runs with `parallel_workers=1`. Its parallel path
collects results in completion order, which makes the aggregate
floating-point sums vary run to run; the sequential path does not.

### Engine invariants

A handful of checks that do not involve Python: the Rust summary block agrees
with the per-run samples beside it, the same seed reproduces, and a different
seed actually changes the draw.

## If something fails

Failures print both engines' means, standard deviations, the difference, the
allowance, and both p-values. A statistical failure is worth re-running with
a different `BASE_SEED` before treating it as a bug — but a *deterministic*
failure is always a real divergence and should be chased down.
