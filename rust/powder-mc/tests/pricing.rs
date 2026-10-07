//! Port of `tests/test_pricing.py`.
//!
//! Verifies billing under failure and replacement: transient failures keep
//! billing, data loss stops it, and a strategy that waits before replacing
//! leaves a measurable gap in the bill.

mod common;

use common::{days, ConfigBuilder};

use powder_mc::sim::cluster::ClusterState;
use powder_mc::sim::distributions::{Distribution, Rng};
use powder_mc::sim::events::{Event, EventType};
use powder_mc::sim::node::NodeConfigRef;
use powder_mc::sim::protocol::{LeaderlessProtocol, Protocol};
use powder_mc::sim::simulator::Simulator;
use powder_mc::sim::strategy::{Action, ClusterStrategy, NoOpStrategy};

/// Deterministic config at $1/hour, with every event effectively disabled
/// unless a test overrides it.
fn make_pricing_config(
    failure: Option<Distribution>,
    recovery: Option<Distribution>,
    data_loss: Option<Distribution>,
    spawn: Option<Distribution>,
) -> NodeConfigRef {
    ConfigBuilder::new()
        .region("us-east")
        .cost(1.0)
        .failure(failure.unwrap_or(Distribution::constant(days(3650.0))))
        .recovery(recovery.unwrap_or(Distribution::constant(0.0)))
        .data_loss(data_loss.unwrap_or(Distribution::constant(days(3650.0))))
        .log_replay_rate(Distribution::constant(100.0))
        .snapshot_download(Distribution::constant(0.0))
        .spawn(spawn.unwrap_or(Distribution::constant(0.0)))
        .build()
}

fn stable_config() -> NodeConfigRef {
    make_pricing_config(None, None, None, None)
}

/// Replaces a node only after a fixed gap, to save cost.
///
/// On data loss it leaves the node in place -- billing stops on its own --
/// and arms a timer.  When the timer fires it spawns a replacement, which
/// starts billing again.
struct GapReplacementStrategy {
    downtime: f64,
    node_config: NodeConfigRef,
    spawn_counter: usize,
}

impl GapReplacementStrategy {
    fn new(downtime: f64, node_config: NodeConfigRef) -> Self {
        GapReplacementStrategy {
            downtime,
            node_config,
            spawn_counter: 0,
        }
    }
}

impl ClusterStrategy for GapReplacementStrategy {
    fn on_event(
        &mut self,
        event: &Event,
        cluster: &ClusterState,
        _rng: &mut Rng,
        _protocol: &dyn Protocol,
        out: &mut Vec<Action>,
    ) {
        match event.event_type {
            EventType::NodeDataLoss => {
                // Billing stops by itself for a node that lost its data, so
                // the node stays put and we just start the clock.  The timer
                // rides on a synthetic target so it cannot collide with a
                // real node's replacement check.
                let name = format!("timer_{}", cluster.name_of(event.target_id));
                out.push(Action::ScheduleReplacementCheck {
                    node_id: cluster.intern(&name),
                    timeout: self.downtime,
                });
            }
            EventType::NodeReplacementTimeout => {
                if cluster.name_of(event.target_id).starts_with("timer_") {
                    self.spawn_counter += 1;
                    let new_node_id =
                        cluster.intern(&format!("node_repl_{}", self.spawn_counter));
                    out.push(Action::SpawnNode {
                        node_config: self.node_config.clone(),
                        node_id: new_node_id,
                        standby: false,
                    });
                }
            }
            _ => {}
        }
    }
}

#[test]
fn test_transient_failure_billing() {
    // Three machines; two never fail and the third cycles six days up, one
    // day down, for a year.  Transient failure does not stop the meter, so
    // the bill is three machines for the whole year.
    let transient = make_pricing_config(
        Some(Distribution::constant(days(6.0))),
        Some(Distribution::constant(days(1.0))),
        None,
        None,
    );

    let mut cluster = ClusterState::new(3);
    cluster.add_named_node("node1", stable_config());
    cluster.add_named_node("node2", stable_config());
    cluster.add_named_node("node3", transient);

    let mut sim = Simulator::new(
        cluster,
        Box::new(NoOpStrategy),
        Box::new(LeaderlessProtocol::default()),
        None,
        Some(42),
        false,
    );

    let duration_days = 364.0;
    let result = sim.run_for(days(duration_days));

    let expected_cost = 3.0 * 24.0 * duration_days;
    let relative = (result.metrics.total_cost - expected_cost).abs() / expected_cost;
    assert!(
        relative < 1e-9,
        "cost {} vs expected {expected_cost}",
        result.metrics.total_cost
    );
}

#[test]
fn test_data_loss_billing() {
    // The third machine loses its data after a week, which stops its
    // billing: two machines for a year plus one for a week.
    let data_loss = make_pricing_config(None, None, Some(Distribution::constant(days(7.0))), None);

    let mut cluster = ClusterState::new(3);
    cluster.add_named_node("node1", stable_config());
    cluster.add_named_node("node2", stable_config());
    cluster.add_named_node("node3", data_loss);

    let mut sim = Simulator::new(
        cluster,
        Box::new(NoOpStrategy),
        Box::new(LeaderlessProtocol::default()),
        None,
        Some(42),
        false,
    );

    let result = sim.run_for(days(365.0));

    let expected_cost = (2.0 * 365.0 * 24.0) + (7.0 * 24.0);
    let relative = (result.metrics.total_cost - expected_cost).abs() / expected_cost;
    assert!(
        relative < 1e-9,
        "cost {} vs expected {expected_cost}",
        result.metrics.total_cost
    );
}

#[test]
fn test_replacement_strategy_billing_gap() {
    // The third slot cycles: six days running, data loss, a one-day gap
    // with nothing billed, then a replacement spawns and the cycle repeats.
    //
    // Over 52 weeks that is 52 x 6 = 312 billed days for the slot, against
    // 364 for each of the two stable machines.
    let cycle_config = make_pricing_config(None, None, Some(Distribution::constant(days(6.0))), None);

    let mut cluster = ClusterState::new(3);
    cluster.add_named_node("node1", stable_config());
    cluster.add_named_node("node2", stable_config());
    cluster.add_named_node("node3", cycle_config.clone());

    let mut sim = Simulator::new(
        cluster,
        Box::new(GapReplacementStrategy::new(days(1.0), cycle_config)),
        Box::new(LeaderlessProtocol::default()),
        None,
        Some(42),
        false,
    );

    let result = sim.run_for(days(364.0));

    let billed_days_dynamic = 312.0;
    let total_days = (2.0 * 364.0) + billed_days_dynamic;
    let expected_cost = total_days * 24.0;
    let relative = (result.metrics.total_cost - expected_cost).abs() / expected_cost;
    assert!(
        relative < 1e-9,
        "cost {} vs expected {expected_cost}",
        result.metrics.total_cost
    );
}

/// Not in the Python suite: a node still being provisioned is *not*
/// billed, because `execute_action` creates it with `has_data` clear and
/// billing skips nodes without data.  The Python docstring claims
/// provisioning bills "from launch"; the flag it sets says otherwise, and
/// the port follows the code.
#[test]
fn provisioning_nodes_are_not_billed_until_they_have_data() {
    // node3 loses data on day 1, a one-day gap follows, and the
    // replacement takes another day to spawn -- landing exactly at the end
    // of the run.  So the slot bills for day 0-1 and nothing after.
    let slow_spawn = make_pricing_config(
        None,
        None,
        Some(Distribution::constant(days(1.0))),
        Some(Distribution::constant(days(1.0))),
    );

    let mut cluster = ClusterState::new(3);
    cluster.add_named_node("node1", stable_config());
    cluster.add_named_node("node2", stable_config());
    cluster.add_named_node("node3", slow_spawn.clone());

    let mut sim = Simulator::new(
        cluster,
        Box::new(GapReplacementStrategy::new(days(1.0), slow_spawn)),
        Box::new(LeaderlessProtocol::default()),
        None,
        Some(42),
        false,
    );

    let result = sim.run_for(days(3.0));

    // Two stable machines for three days, plus one machine-day for node3.
    let expected_cost = (2.0 * 3.0 + 1.0) * 24.0;
    let relative = (result.metrics.total_cost - expected_cost).abs() / expected_cost;
    assert!(
        relative < 1e-9,
        "cost {} vs expected {expected_cost}",
        result.metrics.total_cost
    );
}

/// A zero timeout schedules no replacement check at all, matching the
/// `timeout > 0` guard the Python simulator applies.
#[test]
fn a_zero_timeout_schedules_no_replacement() {
    let cycle = make_pricing_config(None, None, Some(Distribution::constant(days(1.0))), None);

    let mut cluster = ClusterState::new(3);
    cluster.add_named_node("node1", stable_config());
    cluster.add_named_node("node2", stable_config());
    cluster.add_named_node("node3", cycle.clone());

    let mut sim = Simulator::new(
        cluster,
        Box::new(GapReplacementStrategy::new(0.0, cycle)),
        Box::new(LeaderlessProtocol::default()),
        None,
        Some(42),
        false,
    );

    let result = sim.run_for(days(3.0));

    // No replacement ever spawns, so the bill is two machines for three
    // days plus node3's single day.
    let expected_cost = (2.0 * 3.0 + 1.0) * 24.0;
    let relative = (result.metrics.total_cost - expected_cost).abs() / expected_cost;
    assert!(relative < 1e-9, "cost {}", result.metrics.total_cost);
    assert_eq!(result.metrics.total_nodes_spawned, 0);
}
