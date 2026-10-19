//! Stratified sampling: the forced schedule against native clocks, and the
//! stratified estimators against plain Monte Carlo.

use rand::SeedableRng;
use rand_distr::{Distribution as _, Poisson};

use powder_mc::config::Job;
use powder_mc::job::run_job_json;
use powder_mc::sim::distributions::Rng;
use powder_mc::sim::schedule::{random_slot, Candidate, ForcedSchedule};
use powder_mc::sim::simulator::Simulator;
use powder_mc::stats::welch_t_test;
use powder_mc::stratified::{poisson_axes, Allocation, StratumAxis};

const HORIZON: f64 = 2.0 * 86400.0;

/// Three Raft nodes with replacement, failing often enough that every
/// mechanism -- elections, replacement, promotion, data loss -- shows up
/// in a few thousand two-day runs.
fn job(mode: &str, extra: &str) -> String {
    format!(
        r#"{{
        "mode": "{mode}",
        "node_configs": {{
            "flaky": {{
                "region": "us-east", "cost_per_hour": 0.1,
                "failure_dist": {{"type": "exponential", "rate": {fail}}},
                "recovery_dist": {{"type": "exponential", "rate": {recover}}},
                "data_loss_dist": {{"type": "exponential", "rate": {loss}}},
                "log_replay_rate_dist": {{"type": "constant", "value": 1e6}},
                "snapshot_download_time_dist": {{"type": "constant", "value": 1.0}},
                "spawn_dist": {{"type": "exponential", "rate": {spawn}}}
            }},
            "sturdy": {{
                "region": "us-east", "cost_per_hour": 0.1,
                "failure_dist": {{"type": "exponential", "rate": {fail}}},
                "recovery_dist": {{"type": "exponential", "rate": {recover}}},
                "data_loss_dist": {{"type": "exponential", "rate": {loss_sturdy}}},
                "log_replay_rate_dist": {{"type": "constant", "value": 1e6}},
                "snapshot_download_time_dist": {{"type": "constant", "value": 1.0}},
                "spawn_dist": {{"type": "exponential", "rate": {spawn}}}
            }}
        }},
        "cluster": {{
            "target_cluster_size": 3,
            "nodes": [
                {{"node_id": "n0", "config": "sturdy"}},
                {{"node_id": "n1", "config": "flaky"}},
                {{"node_id": "n2", "config": "flaky"}}
            ]
        }},
        "protocol": {{"type": "raft", "election_time_dist": {{"type": "exponential", "rate": 0.2}}}},
        "strategy": {{"type": "node_replacement", "failure_timeout": 3600,
                      "default_node_config": "sturdy"}},
        "run": {{"max_time": {HORIZON}, "stop_on_data_loss": false, "base_seed": 7}}
        {extra}
    }}"#,
        fail = 1.0 / 43200.0,
        recover = 1.0 / 3600.0,
        loss = 1.0 / (6.0 * 3600.0),
        loss_sturdy = 1.0 / (2.0 * 86400.0),
        spawn = 1.0 / 1800.0,
    )
}

struct Sample {
    availability: Vec<f64>,
    lost: Vec<f64>,
    transient: Vec<f64>,
    permanent: Vec<f64>,
}

fn collect(
    runs: usize,
    mut run: impl FnMut(u64) -> powder_mc::sim::metrics::MetricsSnapshot,
) -> Sample {
    let mut s = Sample {
        availability: Vec::new(),
        lost: Vec::new(),
        transient: Vec::new(),
        permanent: Vec::new(),
    };
    for i in 0..runs {
        let m = run(i as u64);
        s.availability.push(m.availability_fraction());
        s.lost.push(if m.time_to_actual_data_loss.is_some() {
            1.0
        } else {
            0.0
        });
        s.transient.push(m.total_transient_failures as f64);
        s.permanent.push(m.total_dataloss_failures as f64);
    }
    s
}

/// Native runs against forced-schedule runs whose counts are drawn from
/// their Poisson laws -- the mixture over all strata, which must be the
/// native process.
fn compare_with_native(headroom: usize) {
    let job: Job = serde_json::from_str(&job("monte_carlo", "")).unwrap();
    let r = job.resolve().unwrap();
    let axes = poisson_axes(
        &r.cluster_template,
        r.strategy_default_config.as_ref(),
        true,
        headroom,
        HORIZON,
        None,
    )
    .unwrap();
    const RUNS: usize = 6000;

    let native = collect(RUNS, |seed| {
        let mut sim = Simulator::new(
            r.build_cluster(),
            r.build_strategy(),
            r.build_protocol(),
            None,
            Some(seed),
            false,
        );
        sim.run_through(HORIZON).metrics
    });

    let mut native_nodes = 0;
    let forced = collect(RUNS, |seed| {
        let mut sim = Simulator::new(
            r.build_cluster(),
            r.build_strategy(),
            r.build_protocol(),
            None,
            Some(1_000_000 + seed),
            false,
        );
        let mut rng = Rng::seed_from_u64(2_000_000 + seed);
        let mut candidates = Vec::new();
        let mut times = Vec::new();
        for (index, axis) in axes.iter().enumerate() {
            let k = Poisson::new(axis.mean).unwrap().sample(&mut rng) as u32;
            times.clear();
            axis.sample_times(k, HORIZON, &mut rng, &mut times);
            for &time in &times {
                let slot = random_slot(&axis.axis, &mut rng);
                candidates.push(Candidate {
                    time,
                    axis: index,
                    slot,
                });
            }
        }
        let mut schedule = ForcedSchedule::new(axes.iter().map(|a| a.axis.clone()).collect());
        schedule.reset(candidates);
        sim.forced = Some(schedule);
        let metrics = sim.run_through(HORIZON).metrics;
        native_nodes += sim
            .forced
            .as_ref()
            .unwrap()
            .native_count()
            .iter()
            .sum::<u64>();
        metrics
    });
    if headroom == 0 {
        // Replacements had no slots: the native fallback carried them.
        assert!(native_nodes > 0);
    }

    for (name, a, b) in [
        ("availability", &native.availability, &forced.availability),
        ("data loss", &native.lost, &forced.lost),
        ("transient failures", &native.transient, &forced.transient),
        ("permanent failures", &native.permanent, &forced.permanent),
    ] {
        let (_, p) = welch_t_test(a, b);
        let mean = |xs: &[f64]| xs.iter().sum::<f64>() / xs.len() as f64;
        assert!(
            p > 1e-3,
            "headroom {headroom}, {name}: native {} vs forced {} (p = {p})",
            mean(a),
            mean(b)
        );
    }
    // The scenario must actually exercise the rare paths.
    let losses = native.lost.iter().sum::<f64>();
    assert!(losses > 50.0, "only {losses} data-loss runs");
    assert!(native.availability.iter().any(|&a| a < 0.999));
}

#[test]
fn forced_schedule_reproduces_native_clocks() {
    compare_with_native(4);
}

#[test]
fn full_slots_fall_back_to_native_clocks_without_bias() {
    compare_with_native(0);
}

#[test]
fn axes_group_nodes_by_kind_and_rate() {
    let job: Job = serde_json::from_str(&job("monte_carlo", "")).unwrap();
    let r = job.resolve().unwrap();
    let axes = poisson_axes(
        &r.cluster_template,
        r.strategy_default_config.as_ref(),
        true,
        4,
        HORIZON,
        None,
    )
    .unwrap();
    let summary: Vec<_> = axes
        .iter()
        .map(|a| (a.axis.kind.as_str(), a.axis.slots))
        .collect();
    // One shared transient rate; two permanent rates, with the headroom on
    // the replacement config's ("sturdy").
    assert_eq!(
        summary,
        vec![("transient", 7), ("permanent", 5), ("permanent", 2)]
    );
}

fn stratified(allocation: Allocation, metric: &str, tolerance: f64) -> serde_json::Value {
    let allocation = serde_json::to_string(&allocation).unwrap();
    let text = job(
        "stratified",
        &format!(
            r#", "stratified": {{"allocation": {allocation}, "metric": "{metric}",
                 "absolute_error": {tolerance}, "max_runs": 2000000}}"#
        ),
    );
    let result = run_job_json(&text);
    assert!(result.error.is_none(), "{:?}", result.error);
    serde_json::to_value(result.stratified.unwrap()).unwrap()
}

#[test]
fn stratified_and_naive_estimates_agree() {
    for (metric, tolerance) in [("availability", 0.004), ("data_loss_probability", 0.02)] {
        let naive = stratified(Allocation::Naive, metric, tolerance);
        assert_eq!(naive["converged"], true);
        for allocation in [
            Allocation::Proportional,
            Allocation::Neyman,
            Allocation::Bernstein,
        ] {
            let other = stratified(allocation, metric, tolerance);
            assert_eq!(other["converged"], true, "{allocation:?} {metric}");
            let gap = (naive["target"]["mean"].as_f64().unwrap()
                - other["target"]["mean"].as_f64().unwrap())
            .abs();
            let allowed = naive["target"]["half_width"].as_f64().unwrap()
                + other["target"]["half_width"].as_f64().unwrap();
            assert!(
                gap <= allowed,
                "{allocation:?} {metric}: gap {gap} > {allowed}"
            );
            let mass =
                other["kept_mass"].as_f64().unwrap() + other["pruned_mass"].as_f64().unwrap();
            assert!((mass - 1.0).abs() < 1e-12);
            assert!(other["pruned_mass"].as_f64().unwrap() <= 0.1 * tolerance);
        }
    }
}

#[test]
fn the_failure_free_stratum_is_deterministic() {
    let job: Job = serde_json::from_str(&job("monte_carlo", "")).unwrap();
    let r = job.resolve().unwrap();
    let mut results = Vec::new();
    for seed in [1, 2] {
        let mut sim = Simulator::new(
            r.build_cluster(),
            r.build_strategy(),
            r.build_protocol(),
            None,
            Some(seed),
            false,
        );
        let mut schedule = ForcedSchedule::new(
            poisson_axes(
                &r.cluster_template,
                r.strategy_default_config.as_ref(),
                true,
                4,
                HORIZON,
                None,
            )
            .unwrap()
            .into_iter()
            .map(|a| a.axis)
            .collect(),
        );
        schedule.reset([]);
        sim.forced = Some(schedule);
        let m = sim.run_through(HORIZON).metrics;
        results.push((m.availability_fraction(), m.time_to_actual_data_loss));
    }
    assert_eq!(results[0], (1.0, None));
    assert_eq!(results[0], results[1]);
}

#[test]
fn non_exponential_failures_are_rejected() {
    let text = job(
        "stratified",
        r#", "stratified": {"allocation": "proportional", "metric": "availability",
             "absolute_error": 0.01}"#,
    )
    .replacen(
        r#""failure_dist": {"type": "exponential", "rate": 0.000023148148148148147}"#,
        r#""failure_dist": {"type": "weibull", "shape": 1.5, "scale": 43200}"#,
        1,
    );
    let error = run_job_json(&text).error.expect("weibull must be rejected");
    assert!(error.contains("exponential"), "{error}");
}

#[test]
fn emitted_strata_carry_all_kept_mass() {
    let text = job(
        "stratified",
        r#", "stratified": {"allocation": "proportional", "metric": "availability",
             "absolute_error": 0.01, "emit_strata": true}"#,
    );
    let record = run_job_json(&text).stratified.unwrap();
    let strata = record.strata.unwrap();
    assert_eq!(strata.len(), record.num_strata);
    assert!(strata.iter().filter(|s| s.exact).count() <= 1);
    let mass: f64 = strata.iter().map(|s| s.weight).sum();
    assert!((mass - record.kept_mass).abs() < 1e-12);
}

#[test]
fn leader_strata_agree_with_naive_and_hold_their_exact_cells() {
    let run = |allocation: &str, leader: bool| {
        let text = job(
            "stratified",
            &format!(
                r#", "stratified": {{"allocation": "{allocation}", "metric": "availability",
                     "absolute_error": 0.004, "leader_strata": {leader}, "exact_checks": 20}}"#
            ),
        );
        let result = run_job_json(&text);
        assert!(result.error.is_none(), "{:?}", result.error);
        result.stratified.unwrap()
    };
    let naive = run("naive", false);
    let plain = run("proportional", false);
    let leader = run("proportional", true);
    assert!(leader.converged);
    assert_eq!(leader.axes.iter().filter(|a| a.leader).count(), 2);
    // Singling out the leader can only enlarge the known part.  In this
    // harsh scenario both are near zero; the 1-day study is where it bites.
    assert!(leader.exact_mass >= plain.exact_mass, "{} < {}", leader.exact_mass, plain.exact_mass);
    let gap = (naive.target.mean - leader.target.mean).abs();
    assert!(gap <= naive.target.half_width + leader.target.half_width, "gap {gap}");
}

#[test]
fn every_sampled_stratum_gets_the_minimum_runs() {
    let text = job(
        "stratified",
        r#", "stratified": {"allocation": "proportional", "metric": "availability",
             "absolute_error": 0.05, "min_stratum_runs": 30, "emit_strata": true}"#,
    );
    let record = run_job_json(&text).stratified.unwrap();
    let strata = record.strata.unwrap();
    assert!(strata.iter().filter(|s| !s.exact).all(|s| s.runs >= 30));
}
