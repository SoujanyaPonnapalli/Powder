//! Profiling harness: fixed workloads, with allocation counting.
//!
//! Not part of the test suite -- run with
//! `cargo run --release --example profile`.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use powder_mc::monte_carlo::{MonteCarloConfig, MonteCarloRunner, ScenarioFactory};
use powder_mc::sim::cluster::ClusterState;
use powder_mc::sim::distributions::{days, hours, minutes, Distribution};
use powder_mc::sim::node::{NodeConfig, NodeConfigRef};
use powder_mc::sim::protocol::{LeaderlessProtocol, Protocol, RaftLikeProtocol};
use powder_mc::sim::strategy::{ClusterStrategy, NoOpStrategy, NodeReplacementStrategy};
use std::rc::Rc;

static ALLOCS: AtomicU64 = AtomicU64::new(0);
static BYTES: AtomicU64 = AtomicU64::new(0);

struct Counting;

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(new_size as u64, Ordering::Relaxed);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

fn config(failure: Distribution, recovery: Distribution, data_loss: Distribution) -> NodeConfigRef {
    Rc::new(NodeConfig {
        region: "us-east".to_string(),
        cost_per_hour: 0.192,
        failure_dist: failure,
        recovery_dist: recovery,
        data_loss_dist: data_loss,
        log_replay_rate_dist: Distribution::constant(100.0),
        snapshot_download_time_dist: Distribution::constant(0.0),
        spawn_dist: Distribution::constant(minutes(2.0)),
    })
}

fn cluster(n: usize, cfg: NodeConfigRef) -> ClusterState {
    let mut c = ClusterState::new(n);
    for i in 0..n {
        c.add_named_node(&format!("node{i}"), cfg.clone());
    }
    c
}

struct Workload {
    name: &'static str,
    nodes: usize,
    cfg: NodeConfigRef,
    max_time: f64,
    raft: bool,
    replacement: bool,
}

fn run(w: &Workload, sims: usize) -> (f64, u64, u64, u64) {
    let before_allocs = ALLOCS.load(Ordering::Relaxed);
    let before_bytes = BYTES.load(Ordering::Relaxed);

    // Go through `MonteCarloRunner`, which is what the binary uses.  It
    // reuses one simulator across the runs of an experiment, so timing a
    // hand-rolled loop here would miss that entirely.
    let (raft, replacement, nodes) = (w.raft, w.replacement, w.nodes);
    let cfg_cluster = w.cfg.clone();
    let cfg_strategy = w.cfg.clone();

    let make_cluster = move || cluster(nodes, cfg_cluster.clone());
    let make_strategy = move || -> Box<dyn ClusterStrategy> {
        if replacement {
            Box::new(NodeReplacementStrategy::new(
                minutes(15.0),
                Some(cfg_strategy.clone()),
                true,
            ))
        } else {
            Box::new(NoOpStrategy)
        }
    };
    let make_protocol = move || -> Box<dyn Protocol> {
        if raft {
            Box::new(RaftLikeProtocol::new(Distribution::constant(minutes(1.0))))
        } else {
            Box::new(LeaderlessProtocol::majority_available(1.0))
        }
    };

    let scenario = ScenarioFactory {
        cluster: &make_cluster,
        strategy: &make_strategy,
        protocol: &make_protocol,
        network_config: None,
    };

    let config = MonteCarloConfig {
        num_simulations: sims,
        max_time: Some(w.max_time),
        stop_on_data_loss: false,
        base_seed: Some(0),
        log_events: false,
    };

    let start = Instant::now();
    let results = MonteCarloRunner::new(config).run(&scenario, None);
    let elapsed = start.elapsed().as_secs_f64();

    let events: u64 = results
        .transient_failure_samples
        .iter()
        .chain(results.dataloss_failure_samples.iter())
        .chain(results.nodes_spawned_samples.iter())
        .sum();

    (
        elapsed,
        ALLOCS.load(Ordering::Relaxed) - before_allocs,
        BYTES.load(Ordering::Relaxed) - before_bytes,
        events,
    )
}

fn report_sizes() {
    use powder_mc::sim::events::Event;
    use powder_mc::sim::node::NodeState;
    use powder_mc::sim::strategy::Action;
    println!("size_of Event      = {}", std::mem::size_of::<Event>());
    println!("size_of EventMeta  = {}", std::mem::size_of::<powder_mc::sim::events::EventMeta>());
    println!("size_of NodeState  = {}", std::mem::size_of::<NodeState>());
    println!("size_of Action     = {}", std::mem::size_of::<Action>());
    println!();
}

fn main() {
    if std::env::args().any(|a| a == "--sizes") {
        report_sizes();
        return;
    }
    if std::env::args().any(|a| a == "--long") {
        // A long single workload, for attaching an external sampler.
        let cfg = config(
            Distribution::exponential(1.0 / hours(8.0)).unwrap(),
            Distribution::exponential(1.0 / minutes(15.0)).unwrap(),
            Distribution::exponential(1.0 / days(120.0)).unwrap(),
        );
        let w = Workload { name: "long", nodes: 5, cfg, max_time: days(30.0), raft: false, replacement: false };
        let (elapsed, _, _, _) = run(&w, 400_000);
        println!("long run: {elapsed:.1}s");
        return;
    }

    let flaky = config(
        Distribution::exponential(1.0 / hours(8.0)).unwrap(),
        Distribution::exponential(1.0 / minutes(15.0)).unwrap(),
        Distribution::exponential(1.0 / days(120.0)).unwrap(),
    );

    let inert = config(
        Distribution::constant(days(9999.0)),
        Distribution::constant(0.0),
        Distribution::constant(days(9999.0)),
    );

    let workloads = [
        // Zero events fire: isolates per-simulation setup cost.
        Workload { name: "setup-only-5node",     nodes: 5, cfg: inert.clone(),  max_time: 1.0,       raft: false, replacement: false },
        Workload { name: "leaderless-3node-7d",  nodes: 3, cfg: flaky.clone(), max_time: days(7.0),  raft: false, replacement: false },
        Workload { name: "leaderless-5node-30d", nodes: 5, cfg: flaky.clone(), max_time: days(30.0), raft: false, replacement: false },
        Workload { name: "raft-5node-30d",       nodes: 5, cfg: flaky.clone(), max_time: days(30.0), raft: true,  replacement: false },
        Workload { name: "replacement-5node-30d",nodes: 5, cfg: flaky.clone(), max_time: days(30.0), raft: false, replacement: true },
    ];

    let sims = 2000;
    println!(
        "{:<26}{:>10}{:>12}{:>12}{:>14}{:>12}",
        "workload", "wall (s)", "sims/s", "allocs/sim", "bytes/sim", "events/sim"
    );
    println!("{}", "-".repeat(86));

    for w in &workloads {
        // Warm up so the first workload is not charged for lazy init.
        run(w, 50);
        let (elapsed, allocs, bytes, events) = run(w, sims);
        println!(
            "{:<26}{:>10.3}{:>12.0}{:>12.0}{:>14.0}{:>12.1}",
            w.name,
            elapsed,
            sims as f64 / elapsed,
            allocs as f64 / sims as f64,
            bytes as f64 / sims as f64,
            events as f64 / sims as f64,
        );
    }
}
