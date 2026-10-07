# powder-mc

A Rust port of the Powder Monte Carlo simulator — `powder/simulation/` and
`powder/monte_carlo.py`. Same model, same metrics, driven by JSON instead of
Python objects.

The Markov/CTMC backend, the placement optimizer, and the `notebooks/` study
drivers are **not** ported; they remain Python.

## Building

```sh
cd rust
cargo build --release          # -> target/release/powder-mc
cargo test --release           # 335 tests
```

## Running

Jobs are JSON. Changing an input never requires a rebuild.

```sh
powder-mc --config scenario.json              # one job from a file
powder-mc --config -                          # one job from stdin
POWDER_MC_CONFIG=scenario.json powder-mc      # path, or inline JSON
powder-mc --stream -j 8 --batch-size 16       # NDJSON in, NDJSON out
```

| Flag | Meaning |
| --- | --- |
| `-c`, `--config <PATH\|->` | Job file, or `-` for stdin |
| `--stream` | One JSON job per stdin line, one JSON result per stdout line |
| `-j`, `--jobs <N>` | Worker threads in stream mode (default: available parallelism) |
| `--batch-size <K>` | Jobs a worker claims at a time (default: 16) |
| `--unordered` | Emit as results complete rather than in input order |
| `--summary-only` | Drop per-run detail, keep aggregates |

### Threading

A single Monte Carlo experiment — however many simulations it contains —
runs entirely on **one thread**. Parallelism is applied across *jobs*:
`--stream` spawns `-j` workers once at startup and reuses them for the whole
run, and each worker claims a **batch** of jobs at a time so it synchronises
with the queue once per batch rather than once per job.

```
reader thread      bounded channel             persistent workers
stdin NDJSON  -->  Vec<(idx, job)> batches -->  loop {
                       (backpressure)               claim one batch
                                                    run every job in it
                                                    return a batch of results
                                                }
                                                       |
collector  <-- reorder by idx --> stdout NDJSON <------+
```

Jobs are independent and individually seeded, so output is byte-identical
for any `-j` and any `--batch-size`; only emission ordering is affected, and
only under `--unordered`.

## Job schema

```jsonc
{
  "job_id": "optional, echoed back in the result",
  "mode": "single" | "monte_carlo" | "converged",

  "node_configs": {
    "standard": {
      "region": "us-east",
      "cost_per_hour": 0.192,
      "failure_dist":                {"type": "exponential", "rate": 1.16e-5},
      "recovery_dist":               {"type": "constant", "value": 600},
      "data_loss_dist":              {"type": "weibull", "shape": 1.2, "scale": 3.15e7},
      "log_replay_rate_dist":        {"type": "normal", "mean": 100, "std": 10, "min_val": 0},
      "snapshot_download_time_dist": {"type": "uniform", "low": 30, "high": 90},
      "spawn_dist":                  {"type": "constant", "value": 120}
    }
  },

  "cluster": {
    "target_cluster_size": 5,
    "nodes": [{"node_id": "node0", "config": "standard",
               "is_available": true, "has_data": true,
               "last_applied_index": 0.0, "last_snapshot_index": 0.0}],
    "standby_nodes": [],
    "provisioning_nodes": [],
    "active_outages": [],
    "current_time": 0.0,
    "commit_index": 0.0
  },

  "protocol": {"type": "leaderless", "commit_rate": 1.0,
               "snapshot_interval": 0.0, "log_retention_ops": 0.0,
               "up_to_date_quorum": true},
  // or: {"type": "raft", "election_time_dist": {...}, "commit_rate": 1.0, ...}

  "strategy": {"type": "noop"},
  // or: {"type": "node_replacement", "failure_timeout": 900,
  //      "default_node_config": "standard", "safe_mode": true}
  // or: {"type": "adaptive_replacement", "failure_timeout": 900,
  //      "reconfiguration_dist": 30.0,   // a number OR a distribution object
  //      "scale_down_threshold": 2, "external_consensus": false,
  //      "default_node_config": "standard", "safe_mode": true}

  "network_config": null,
  // or: {"outage_dist": {...}, "outage_duration_dist": {...},
  //      "regions": ["us-east"]}

  "run": {"max_time": 2592000.0, "stop_on_data_loss": true,
          "num_simulations": 200, "base_seed": 42, "log_events": false},

  "convergence": {"confidence_level": 0.95, "relative_error": 0.05,
                  "absolute_error": null, "metrics": ["availability"],
                  "min_runs": 30, "max_runs": 10000, "batch_size": 10}
}
```

`max_time: null` with `stop_on_data_loss: true` runs each simulation until
data loss, which is what MTTDL estimation needs. `run.parallel_workers` is
accepted for schema compatibility with the Python config and ignored —
parallelism here is per job.

The result carries every `MetricsSnapshot` field per run, the aggregate
summary, the convergence state for `converged` jobs, an optional event log,
and `elapsed_seconds`.

## Performance

Measured on a 10-core machine with `examples/profile`, which times the
engine directly (no process startup, no JSON):

| workload | sims/s | allocations per simulation |
| --- | ---: | ---: |
| leaderless, 3 nodes, 7 days | 41,400 | 2 |
| leaderless, 5 nodes, 30 days | 9,300 | 3 |
| raft, 5 nodes, 30 days | 7,900 | 3 |
| replacement strategy, 5 nodes, 30 days | 1,700 | 10 |

Roughly 3.6 M events/second on the 5-node leaderless workload.

End to end against the Python engine, on identical scenarios with full
per-run output (`migration/bench.py --sims 2000`, 16,000 simulations):

| engine | wall | sims/s | events/s | vs Python, 1 process |
| --- | ---: | ---: | ---: | ---: |
| Python, 1 process | 78.8 s | 203 | 26,229 | 1.0x |
| Python, 10 processes | 20.0 s | 801 | 103,456 | 3.9x |
| Rust, `-j 1` | 1.36 s | 11,771 | 1,524,581 | **57.9x** |
| Rust, `-j 10` | 0.22 s | 72,530 | 9,394,147 | **357x** |

So about **58x single-threaded**, and **90x** comparing each engine at its
best on this 10-core machine. Python's multiprocessing only reaches 3.9x
on 10 cores here -- short simulations make its per-result IPC expensive,
which is the gap the `-j 10` column closes.

Three things got it there, in order of effect:

* **No allocation on the event path.** Returning a fresh `Vec` from every
  strategy and protocol callback, and building a fresh index vector for
  every per-event node walk, cost 3–16 allocations *per event*. Callbacks
  now append into a caller-owned buffer and the walks reuse scratch
  vectors, which is why `ClusterStrategy::on_event` and `Protocol::on_event`
  take an `out` parameter instead of returning. Allocations fell by
  175–274×, to about 0.03 per event.
* **A shared symbol table.** Each simulation clones a template cluster; a
  private symbol table meant re-interning every node and replacement name,
  thousands of times per experiment. The table is now shared through an
  `Rc`, and the interner keeps one `Rc<str>` per name rather than a
  separate `String` for the key and the reverse lookup.
* **A compact event payload.** `EventMeta` was a struct of six options at
  64 bytes, of which at most one was ever set. As an enum it is 16 bytes,
  taking `Event` from 80 to 32 and halving what the binary heap moves on
  every push and pop.
* **One simulator per experiment, not per run.** `MonteCarloRunner` builds
  a single `Simulator` per batch and calls
  [`Simulator::reset`](src/sim/simulator.rs) between runs, so the event
  heap, cancellation tables, node vector and six scratch buffers are
  allocated once instead of once per simulation. This is what takes the
  count from 13 to 3 and the bytes from 3.2 KB to 363 B per simulation.
  Two tests pin it: `reused_simulator_matches_independent_runs` checks a
  reused simulator against freshly built ones run by run, and
  `resetting_with_the_same_seed_repeats_the_run` checks that a reset
  leaves nothing behind.

Together these are about 1.7× on single-thread throughput. They matter
more than that across threads: the allocator was the contention point, so
scaling from 1 to 10 workers went from 5.0× to 9.5×.

`-C target-cpu=native` makes no measurable difference — the hot path is
branches and pointer-chasing, not vectorizable arithmetic — so the
portable binary is the one to ship.

### What is left

A profile now shows the time going to the per-event passes over the node
vector: advancing the commit frontier (13%), `can_commit` (13%), and
scanning for nodes that need a sync (8%). Those passes are inherited from
the Python structure, which recomputes availability from scratch on every
query. Replacing them with incrementally maintained counters is the main
remaining lever, worth perhaps 10–15%, but it would mean intercepting
every mutation that can change a node's availability — and a missed
invalidation is a silent wrong answer rather than a crash. It has not been
done.

## Threading at scale

The engine allocates about **three times per simulation** — roughly 0.008
allocations per event — so the allocator is not on the critical path. At
the measured 9,300 sims/s that is about 28,000 allocations per second per
thread, which any thread-caching allocator serves from its local free list
without touching a shared lock. Modern allocators are already per-thread
in the way that matters: glibc keeps a per-thread `tcache` plus multiple
arenas, macOS uses per-CPU magazines, and mimalloc, jemalloc and snmalloc
all have thread-local heaps with lock-free fast paths.

A `mimalloc` feature is available anyway, for the case where a profile
says otherwise:

```sh
cargo build --release --features mimalloc   # needs a C toolchain
```

On the allocation-heaviest workload here it is worth 1–3%, which is the
expected result once the hot path has stopped allocating. It is off by
default so the normal build stays pure Rust.

Measured scaling on 10 cores (a 400-job stream, 200 simulations each):

| `-j` | sims/s | vs `-j 1` |
| ---: | ---: | ---: |
| 1 | 8,979 | 1.00x |
| 2 | 16,842 | 1.84x |
| 4 | 30,534 | 3.34x |
| 8 | 53,333 | 5.83x |
| 10 | 61,069 | 6.68x |

Past the core count it flattens, as expected. The remaining gap from
linear is this machine's mix of performance and efficiency cores, not
contention.

### What actually limits a very wide run

Two shared points exist in `--stream`, and both are measurable:

* **The work queue.** Workers claim batches under one mutex. With
  `--batch-size 1` that caps the pipeline at about 290,000 jobs/s no
  matter how many threads are running; at 64 or more it reaches about
  690,000 jobs/s. The default of 16 is already in the flat part of that
  curve. If your jobs are very short, raise it.
* **The collector thread.** Results are encoded to NDJSON *in the
  workers*, so the collector only concatenates bytes and writes them.
  Encoding there instead would put all of a wide run's serialisation on
  one thread: on a stream of near-empty jobs, moving it out took the
  pipeline from 667,000 to 1,250,000 jobs/s at `-j 10`. It also means the
  in-order reorder buffer holds encoded lines rather than whole
  `JobResult` structs.

Two related details:

* **Batches ramp at start-up.** The first `-j` batches hold one job each
  so every worker starts immediately, then the size doubles per sweep up
  to `--batch-size`. A flat size starves a wide run: with 200 workers and
  batches of 16, the reader must get 3,200 jobs in before the last worker
  sees anything.
* **In-order output buffers.** The default holds completed results until
  their turn comes, so a slow job keeps the ones behind it in memory --
  bounded by the job channel, but still proportional to `-j`.
  `--unordered` emits as results land; every result carries its `job_id`,
  so ordering is recoverable downstream.

A memo over `availability_counts` was prototyped and rejected. Making
`can_commit` entirely free -- the absolute ceiling for any caching of it
-- is worth about 6%, so a real memo catching some fraction of calls
lands at 2-3%. That did not justify making `commit_index` and `network`
private across roughly 50 call sites and taking on an invalidation bug
whose failure mode is a silently wrong answer.

## Deliberate deviations from the Python engine

The port targets **statistical**, not bitwise, agreement. These differences
are intentional; `migration/` demonstrates they do not change behaviour.

* **RNG.** PCG64 via `rand_pcg`, with `rand_distr`'s samplers. Same
  distribution families, different stream from NumPy's `default_rng`.
* **Snapshot boundaries** use `(index / interval).floor() * interval` rather
  than CPython's `fmod`-based float floor division. The two disagree in edge
  cases — `100.0 // 0.0001` is `999999.0` in Python but `1000000.0` here —
  and the plain division is both faster and the intended answer.
* **Node ordering.** `find_sync_donor`, `most_lagging_node` and the
  cluster-size trim use an explicit total order with the node ID as the
  final tiebreaker, instead of relying on dict insertion order and Python's
  stable sort. Raft's leader election still orders candidates by node ID
  *string*, matching Python.
* **Seeding.** `powder/monte_carlo.py:573` uses `if self.config.base_seed`,
  so `base_seed=0` silently means "no seed" there while `_run_batch` treats
  it as a real seed. The port uses the `is not None` reading throughout.
* **`Weibull.mean`** uses `statrs`' gamma rather than CPython's Lanczos
  `math.gamma`. They agree to about 1e-15 relative.
* **Aggregates** (`mean`, `std`, `percentile`) are plain left-to-right
  reductions, not NumPy's pairwise summation.

## Structure

| Path | Python counterpart |
| --- | --- |
| `src/sim/distributions.rs` | `powder/simulation/distributions.py` |
| `src/sim/node.rs` | `powder/simulation/node.py` |
| `src/sim/network.rs` | `powder/simulation/network.py` |
| `src/sim/events.rs` | `powder/simulation/events.py` |
| `src/sim/cluster.rs` | `powder/simulation/cluster.py` |
| `src/sim/protocol.rs` | `powder/simulation/protocol.py` |
| `src/sim/strategy.rs` | `powder/simulation/strategy.py` |
| `src/sim/metrics.rs` | `powder/simulation/metrics.py` |
| `src/sim/simulator.rs` | `powder/simulation/simulator.py` |
| `src/monte_carlo.rs` | `powder/monte_carlo.py` |
| `src/sim/ids.rs` | (new) identifier interning |
| `src/stats.rs` | the NumPy/SciPy calls the runner makes |
| `src/config.rs`, `src/job.rs`, `src/pool.rs` | (new) JSON I/O and the worker pool |

Three representation changes carry through the port. Node IDs and region
names are **interned** to `u32`, so event cancellation tables are flat
vectors indexed by symbol rather than string-keyed hash maps. The cluster
keeps **one** `Vec<NodeState>` tagged by group instead of three dicts plus a
cache, so promotion is a field write. And event metadata is a struct of
options rather than a `dict`.

## Test parity

`tests/` mirrors the Python MC suite file for file, keeping the original
test names:

| Rust | Python |
| --- | --- |
| `tests/simulation.rs` | `tests/test_simulation.py` (engine sections) |
| `tests/monte_carlo_statistics.rs` | `tests/test_monte_carlo_statistics.py` + the convergence sections of `test_simulation.py` |
| `tests/raft_protocol.rs` | `tests/test_raft_protocol.py` + `TestRaftLikeProtocol` |
| `tests/event_timing.rs` | `tests/test_event_timing.py` |
| `tests/metrics_counters.rs` | `tests/test_metrics_counters.py` |
| `tests/availability.rs` | `tests/test_availability.py` |
| `tests/pricing.rs` | `tests/test_pricing.py` |
| `tests/closed_form_verification.rs` | `tests/test_closed_form_verification.py` |
| `tests/strategy_scaling.rs` | `tests/test_strategy_refactor.py` + `test_simple_strategy_scaling.py` |

Two Python tests were adapted rather than copied, both because they depend
on NumPy's particular stream rather than on behaviour:

* `test_max_runs_respected` pairs a 0.001 relative error with a
  low-variance scenario, and only fails to converge because of the spread
  NumPy happens to produce in the first few batches. The port uses a
  higher-variance scenario and a threshold no finite sample can meet, which
  tests the same thing — that the cap holds — without depending on the draw.
* `test_is_alias_of_up_to_date_quorum_protocol` asserts Python class
  identity (`LeaderlessUpToDateQuorumProtocol is LeaderlessProtocol`). The
  port asserts the equivalent: the named constructor builds the same type
  with the flag set, rather than a separate protocol.

`tests/test_weekly_window_study.py` is out of scope: it tests
`notebooks/availability_weekly_windows.py`, not the engine.
