//! Availability, cost and data-loss metrics.
//!
//! Port of `powder/simulation/metrics.py`.
//!
//! The split between [`MetricsCollector::record_elapsed`] (called *before* an
//! event is applied) and [`MetricsCollector::update`] (called *after*) is
//! load-bearing: the interval since the previous event must be attributed to
//! the cluster as it stood during that interval, not as it stands once the
//! event has landed.  Costs are accumulated in `record_elapsed` for the same
//! reason -- nodes added or removed by an event must not be billed for the
//! interval before it.

use serde::{Deserialize, Serialize};

use super::cluster::ClusterState;
use super::distributions::Seconds;
use super::events::EventType;
use super::protocol::Protocol;

/// Timing of maximal can-commit=false episodes within the observed run.
/// The open episode, if any, is truncated at the run's end, not extrapolated.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct EpisodeMetrics {
    /// Caller-selected duration threshold, in seconds (strictly exceeded).
    pub threshold_seconds: Seconds,
    /// Number of episodes with duration strictly greater than the threshold.
    pub count_above_threshold: u64,
    /// Sum of all observed episode durations (including the censored tail).
    pub total_unavailable_seconds: Seconds,
    /// Longest observed episode, including a censored final episode.
    pub max_duration_seconds: Seconds,
    /// Sum over episodes of max(duration - threshold, 0), in seconds.
    pub post_threshold_seconds: Seconds,
    /// Sum of *whole* durations of episodes above the threshold, in seconds.
    pub long_episode_seconds: Seconds,
    /// Duration of the open episode truncated at the end, if any.
    pub end_censored_duration_seconds: Option<Seconds>,
    /// Whether the cluster was already unavailable at the observation start.
    pub start_censored: bool,
}

impl EpisodeMetrics {
    fn new(threshold_seconds: Seconds) -> Self {
        Self {
            threshold_seconds,
            count_above_threshold: 0,
            total_unavailable_seconds: 0.0,
            max_duration_seconds: 0.0,
            post_threshold_seconds: 0.0,
            long_episode_seconds: 0.0,
            end_censored_duration_seconds: None,
            start_censored: false,
        }
    }

    fn add_episode(&mut self, duration: Seconds) {
        self.total_unavailable_seconds += duration;
        self.max_duration_seconds = self.max_duration_seconds.max(duration);
        if duration > self.threshold_seconds {
            self.count_above_threshold += 1;
            self.post_threshold_seconds += duration - self.threshold_seconds;
            self.long_episode_seconds += duration;
        }
    }
}

/// Accumulates metrics over a simulation run.
#[derive(Debug, Clone)]
pub struct MetricsCollector {
    /// Total time the system could commit.
    pub time_available: Seconds,
    /// Total time the system could not commit.
    pub time_unavailable: Seconds,
    /// Accumulated node costs, in dollars.
    pub total_cost: f64,
    /// When quorum was first lost.
    pub time_to_potential_data_loss: Option<Seconds>,
    /// When data was definitely lost.
    pub time_to_actual_data_loss: Option<Seconds>,
    /// Count of `NODE_FAILURE` events.
    pub total_transient_failures: u64,
    /// Count of `NODE_DATA_LOSS` events.
    pub total_dataloss_failures: u64,
    /// Count of `NODE_SPAWN_COMPLETE` events.
    pub total_nodes_spawned: u64,
    /// Count of can-commit to cannot-commit transitions.
    pub total_unavailability_incidents: u64,
    /// Count of successful leader elections.
    pub total_leader_elections: u64,
    /// When the system first became unavailable.
    pub time_to_first_unavailability: Option<Seconds>,
    episode_metrics: Option<EpisodeMetrics>,
    open_episode_since: Option<Seconds>,
    // Wait for positive available time before closing: simultaneous events
    // with zero available time cannot split a maximal outage.
    pending_episode_end: Option<Seconds>,

    last_update_time: Seconds,
    had_potential_loss: bool,
    had_actual_loss: bool,
    was_available: bool,
}

impl Default for MetricsCollector {
    fn default() -> Self {
        MetricsCollector::new()
    }
}

impl MetricsCollector {
    /// A collector with everything zeroed and the system presumed available.
    pub fn new() -> Self {
        Self::with_episode_threshold(None)
    }

    /// Enable episode timing only when a threshold was explicitly supplied.
    pub fn with_episode_threshold(threshold: Option<Seconds>) -> Self {
        MetricsCollector {
            time_available: 0.0,
            time_unavailable: 0.0,
            total_cost: 0.0,
            time_to_potential_data_loss: None,
            time_to_actual_data_loss: None,
            total_transient_failures: 0,
            total_dataloss_failures: 0,
            total_nodes_spawned: 0,
            total_unavailability_incidents: 0,
            total_leader_elections: 0,
            time_to_first_unavailability: None,
            episode_metrics: threshold.map(EpisodeMetrics::new),
            open_episode_since: None,
            pending_episode_end: None,
            last_update_time: 0.0,
            had_potential_loss: false,
            had_actual_loss: false,
            was_available: true,
        }
    }

    /// Return every field to its starting value, for reuse across runs.
    pub fn reset(&mut self) {
        *self = MetricsCollector::with_episode_threshold(
            self.episode_metrics.map(|m| m.threshold_seconds),
        );
    }

    /// Attribute the interval since the last call to availability and cost.
    ///
    /// Must be called *before* the event at `current_time` is applied, so the
    /// cluster still reflects the interval being recorded.
    ///
    /// # Panics
    ///
    /// Panics if time moves backwards, matching the Python `ValueError`.
    pub fn record_elapsed(
        &mut self,
        current_time: Seconds,
        cluster: &ClusterState,
        protocol: &dyn Protocol,
        can_commit: Option<bool>,
    ) {
        let time_delta = current_time - self.last_update_time;
        assert!(
            time_delta >= 0.0,
            "Time went backwards: {} -> {current_time}",
            self.last_update_time
        );

        if time_delta > 0.0 {
            let available = can_commit.unwrap_or_else(|| protocol.can_commit(cluster));
            if available {
                if let (Some(start), Some(end), Some(episodes)) = (
                    self.open_episode_since.take(),
                    self.pending_episode_end.take(),
                    self.episode_metrics.as_mut(),
                ) {
                    episodes.add_episode(end - start);
                }
                self.time_available += time_delta;
            } else {
                self.time_unavailable += time_delta;
            }

            // Cloud VMs bill from launch through failures until termination,
            // so every node that still has data is charged regardless of
            // whether it is currently serving.
            let hours_elapsed = time_delta / 3600.0;
            for node in cluster.all_nodes_for_billing() {
                self.total_cost += node.config.cost_per_hour * hours_elapsed;
            }
        }

        self.last_update_time = current_time;
    }

    /// Increment the counter for an event type, if it has one.
    pub fn record_event(&mut self, event_type: EventType) {
        match event_type {
            EventType::NodeFailure => self.total_transient_failures += 1,
            EventType::NodeDataLoss => self.total_dataloss_failures += 1,
            EventType::NodeSpawnComplete => self.total_nodes_spawned += 1,
            _ => {}
        }
    }

    /// Record the initial state before the first event, including an already
    /// unavailable cluster. The initial unavailable episode is left-censored.
    pub fn record_initial_availability(&mut self, can_commit: bool) {
        if self.episode_metrics.is_some() && !can_commit {
            self.record_unavailability_transition(false, self.last_update_time);
            self.episode_metrics.as_mut().unwrap().start_censored = true;
        }
    }

    /// Record a transition from available to unavailable (or its reverse).
    pub fn record_unavailability_transition(&mut self, can_commit: bool, current_time: Seconds) {
        if self.was_available && !can_commit {
            self.total_unavailability_incidents += 1;
            if self.time_to_first_unavailability.is_none() {
                self.time_to_first_unavailability = Some(current_time);
            }
            if let (Some(start), Some(end), Some(episodes)) = (
                self.open_episode_since,
                self.pending_episode_end.take(),
                self.episode_metrics.as_mut(),
            ) {
                if current_time > end {
                    episodes.add_episode(end - start);
                    self.open_episode_since = None;
                }
                // At exactly the same instant there was no available time;
                // keep the original start and merge the two outages.
            }
            if self.episode_metrics.is_some() && self.open_episode_since.is_none() {
                self.open_episode_since = Some(current_time);
            }
        } else if !self.was_available && can_commit {
            if self.open_episode_since.is_some() {
                self.pending_episode_end = Some(current_time);
            }
        }
        self.was_available = can_commit;
    }

    /// Record a completed leader election.
    pub fn record_leader_election(&mut self) {
        self.total_leader_elections += 1;
    }

    /// Check for data-loss milestones.
    ///
    /// Must be called *after* the event has been fully applied.
    ///
    /// `can_commit` is an optimisation: when the system can commit, quorum is
    /// met by definition and potential data loss is impossible.
    pub fn update(
        &mut self,
        cluster: &ClusterState,
        current_time: Seconds,
        protocol: &dyn Protocol,
        can_commit: Option<bool>,
    ) {
        if !self.had_potential_loss {
            let has_potential = match can_commit {
                Some(true) => false,
                _ => protocol.has_potential_data_loss(cluster),
            };
            if has_potential {
                self.time_to_potential_data_loss = Some(current_time);
                self.had_potential_loss = true;
            }
        }

        if !self.had_actual_loss && protocol.has_actual_data_loss(cluster) {
            self.time_to_actual_data_loss = Some(current_time);
            self.had_actual_loss = true;
        }
    }

    /// Fraction of simulated time the system was available.
    ///
    /// Returns 1.0 when no time has passed.
    pub fn availability_fraction(&self) -> f64 {
        let total = self.total_time();
        if total <= 0.0 {
            return 1.0;
        }
        self.time_available / total
    }

    /// Total simulated time.
    pub fn total_time(&self) -> Seconds {
        self.time_available + self.time_unavailable
    }

    /// Availability as a percentage.
    pub fn availability_percent(&self) -> f64 {
        self.availability_fraction() * 100.0
    }

    /// Availability in "nines" notation, or `None` at perfect availability.
    pub fn nines_of_availability(&self) -> Option<f64> {
        let fraction = self.availability_fraction();
        if fraction >= 1.0 {
            return None;
        }
        if fraction <= 0.0 {
            return Some(0.0);
        }
        Some(-(1.0 - fraction).log10())
    }

    /// Freeze the current values.
    pub fn snapshot(&self) -> MetricsSnapshot {
        // Include an open or pending interval without changing the collector:
        // run_until may resume, so the final episode must not be counted twice.
        let mut episode_metrics = self.episode_metrics;
        if let (Some(start), Some(episodes)) = (self.open_episode_since, episode_metrics.as_mut()) {
            if let Some(end) = self.pending_episode_end {
                episodes.add_episode(end - start);
            } else {
                let duration = self.last_update_time - start;
                episodes.add_episode(duration);
                episodes.end_censored_duration_seconds = Some(duration);
            }
        }
        MetricsSnapshot {
            episode_metrics,
            time_available: self.time_available,
            time_unavailable: self.time_unavailable,
            total_cost: self.total_cost,
            time_to_potential_data_loss: self.time_to_potential_data_loss,
            time_to_actual_data_loss: self.time_to_actual_data_loss,
            total_transient_failures: self.total_transient_failures,
            total_dataloss_failures: self.total_dataloss_failures,
            total_nodes_spawned: self.total_nodes_spawned,
            total_unavailability_incidents: self.total_unavailability_incidents,
            total_leader_elections: self.total_leader_elections,
            time_to_first_unavailability: self.time_to_first_unavailability,
        }
    }
}

/// Immutable metrics taken at the end of a run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MetricsSnapshot {
    /// Episode timing, absent unless a threshold was configured.
    pub episode_metrics: Option<EpisodeMetrics>,
    /// Total time the system could commit.
    pub time_available: Seconds,
    /// Total time the system could not commit.
    pub time_unavailable: Seconds,
    /// Accumulated node costs, in dollars.
    pub total_cost: f64,
    /// When quorum was first lost.
    pub time_to_potential_data_loss: Option<Seconds>,
    /// When data was definitely lost.
    pub time_to_actual_data_loss: Option<Seconds>,
    /// Count of `NODE_FAILURE` events.
    pub total_transient_failures: u64,
    /// Count of `NODE_DATA_LOSS` events.
    pub total_dataloss_failures: u64,
    /// Count of `NODE_SPAWN_COMPLETE` events.
    pub total_nodes_spawned: u64,
    /// Count of can-commit to cannot-commit transitions.
    pub total_unavailability_incidents: u64,
    /// Count of successful leader elections.
    pub total_leader_elections: u64,
    /// When the system first became unavailable.
    pub time_to_first_unavailability: Option<Seconds>,
}

impl MetricsSnapshot {
    /// Fraction of simulated time the system was available.
    pub fn availability_fraction(&self) -> f64 {
        let total = self.time_available + self.time_unavailable;
        if total <= 0.0 {
            return 1.0;
        }
        self.time_available / total
    }

    /// Total simulated time.
    pub fn total_time(&self) -> Seconds {
        self.time_available + self.time_unavailable
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::distributions::Distribution;
    use crate::sim::node::{NodeConfig, NodeConfigRef};
    use crate::sim::protocol::LeaderlessProtocol;
    use std::rc::Rc;

    fn config_at(cost_per_hour: f64) -> NodeConfigRef {
        Rc::new(NodeConfig {
            region: "us-east".to_string(),
            az: None,
            cost_per_hour,
            failure_dist: Distribution::constant(1000.0),
            recovery_dist: Distribution::constant(10.0),
            data_loss_dist: Distribution::constant(10_000.0),
            log_replay_rate_dist: Distribution::constant(100.0),
            snapshot_download_time_dist: Distribution::constant(5.0),
            spawn_dist: Distribution::constant(30.0),
        })
    }

    fn cluster_of(n: usize, cost_per_hour: f64) -> ClusterState {
        let mut c = ClusterState::new(n);
        for i in 0..n {
            c.add_named_node(&format!("node{i}"), config_at(cost_per_hour));
        }
        c
    }

    #[test]
    fn fresh_collector_reports_perfect_availability() {
        let m = MetricsCollector::new();
        assert_eq!(m.availability_fraction(), 1.0);
        assert_eq!(m.total_time(), 0.0);
        assert_eq!(m.nines_of_availability(), None);
    }

    #[test]
    fn elapsed_time_is_split_by_commit_ability() {
        let mut m = MetricsCollector::new();
        let c = cluster_of(3, 0.0);
        let p = LeaderlessProtocol::default();

        m.record_elapsed(100.0, &c, &p, Some(true));
        m.record_elapsed(150.0, &c, &p, Some(false));

        assert_eq!(m.time_available, 100.0);
        assert_eq!(m.time_unavailable, 50.0);
        assert_eq!(m.total_time(), 150.0);
        assert!((m.availability_fraction() - 100.0 / 150.0).abs() < 1e-12);
    }

    #[test]
    fn cost_accrues_per_billable_node_hour() {
        let mut m = MetricsCollector::new();
        let c = cluster_of(3, 2.0);
        let p = LeaderlessProtocol::default();

        // One hour at $2/hour across three nodes.
        m.record_elapsed(3600.0, &c, &p, Some(true));
        assert!((m.total_cost - 6.0).abs() < 1e-12);
    }

    #[test]
    fn nodes_that_lost_data_stop_being_billed() {
        let mut m = MetricsCollector::new();
        let mut c = cluster_of(3, 2.0);
        let p = LeaderlessProtocol::default();

        let n0 = c.sym_of("node0").unwrap();
        c.get_node_mut(n0).unwrap().has_data = false;
        m.record_elapsed(3600.0, &c, &p, Some(true));
        assert!((m.total_cost - 4.0).abs() < 1e-12);
    }

    #[test]
    fn unavailable_nodes_are_still_billed() {
        let mut m = MetricsCollector::new();
        let mut c = cluster_of(3, 2.0);
        let p = LeaderlessProtocol::default();

        let n0 = c.sym_of("node0").unwrap();
        c.get_node_mut(n0).unwrap().is_available = false;
        m.record_elapsed(3600.0, &c, &p, Some(true));
        assert!((m.total_cost - 6.0).abs() < 1e-12);
    }

    #[test]
    fn zero_length_intervals_cost_nothing() {
        let mut m = MetricsCollector::new();
        let c = cluster_of(3, 2.0);
        let p = LeaderlessProtocol::default();
        m.record_elapsed(0.0, &c, &p, Some(true));
        assert_eq!(m.total_cost, 0.0);
        assert_eq!(m.total_time(), 0.0);
    }

    #[test]
    #[should_panic(expected = "Time went backwards")]
    fn time_must_not_move_backwards() {
        let mut m = MetricsCollector::new();
        let c = cluster_of(1, 0.0);
        let p = LeaderlessProtocol::default();
        m.record_elapsed(100.0, &c, &p, Some(true));
        m.record_elapsed(50.0, &c, &p, Some(true));
    }

    #[test]
    fn event_counters_track_their_event_types() {
        let mut m = MetricsCollector::new();
        m.record_event(EventType::NodeFailure);
        m.record_event(EventType::NodeFailure);
        m.record_event(EventType::NodeDataLoss);
        m.record_event(EventType::NodeSpawnComplete);
        // Not counted.
        m.record_event(EventType::NodeRecovery);
        m.record_event(EventType::NodeSyncComplete);

        assert_eq!(m.total_transient_failures, 2);
        assert_eq!(m.total_dataloss_failures, 1);
        assert_eq!(m.total_nodes_spawned, 1);
    }

    #[test]
    fn unavailability_incidents_count_edges_not_states() {
        let mut m = MetricsCollector::new();
        m.record_unavailability_transition(true, 10.0);
        assert_eq!(m.total_unavailability_incidents, 0);

        m.record_unavailability_transition(false, 20.0);
        assert_eq!(m.total_unavailability_incidents, 1);
        assert_eq!(m.time_to_first_unavailability, Some(20.0));

        // Staying unavailable is not a second incident.
        m.record_unavailability_transition(false, 30.0);
        assert_eq!(m.total_unavailability_incidents, 1);

        m.record_unavailability_transition(true, 40.0);
        m.record_unavailability_transition(false, 50.0);
        assert_eq!(m.total_unavailability_incidents, 2);
        // Only the first unavailability time is kept.
        assert_eq!(m.time_to_first_unavailability, Some(20.0));
    }

    #[test]
    fn episode_timing_handles_strict_boundary_multiple_outages_and_reset() {
        let mut m = MetricsCollector::with_episode_threshold(Some(5.0));
        for (available, time) in [
            (false, 1.0),
            (true, 4.0), // below: 3s
            (false, 6.0),
            (false, 8.0),
            (true, 11.0), // exactly 5s
            (false, 13.0),
            (true, 20.0), // above: 7s
        ] {
            m.record_unavailability_transition(available, time);
        }
        let episodes = m.snapshot().episode_metrics.unwrap();
        assert_eq!(episodes.count_above_threshold, 1);
        assert_eq!(episodes.total_unavailable_seconds, 15.0);
        assert_eq!(episodes.max_duration_seconds, 7.0);
        assert_eq!(episodes.post_threshold_seconds, 2.0);
        assert_eq!(episodes.long_episode_seconds, 7.0);
        assert_eq!(episodes.end_censored_duration_seconds, None);
        assert_eq!(m.total_unavailability_incidents, 3);
        m.reset();
        assert_eq!(m.snapshot().episode_metrics.unwrap().threshold_seconds, 5.0);
        assert_eq!(
            m.snapshot().episode_metrics.unwrap().count_above_threshold,
            0
        );
    }

    #[test]
    fn simultaneous_available_state_does_not_split_maximal_episode() {
        let c = cluster_of(1, 0.0);
        let p = LeaderlessProtocol::default();
        let mut m = MetricsCollector::with_episode_threshold(Some(5.0));
        m.record_unavailability_transition(false, 0.0);
        m.record_elapsed(4.0, &c, &p, Some(false));
        m.record_unavailability_transition(true, 4.0);
        m.record_unavailability_transition(false, 4.0);
        m.record_elapsed(10.0, &c, &p, Some(false));
        m.record_unavailability_transition(true, 10.0);
        let e = m.snapshot().episode_metrics.unwrap();
        assert_eq!(e.total_unavailable_seconds, 10.0);
        assert_eq!(e.max_duration_seconds, 10.0);
        assert_eq!(e.count_above_threshold, 1);
        assert_eq!(e.post_threshold_seconds, 5.0);
        assert_eq!(e.end_censored_duration_seconds, None);
    }

    #[test]
    fn zero_threshold_and_censored_snapshot_do_not_double_count_on_resume() {
        let c = cluster_of(1, 0.0);
        let p = LeaderlessProtocol::default();
        let mut m = MetricsCollector::with_episode_threshold(Some(0.0));
        m.record_initial_availability(false); // starts unavailable at t=0
        m.record_elapsed(4.0, &c, &p, Some(false));
        let first = m.snapshot().episode_metrics.unwrap();
        assert!(first.start_censored);
        assert_eq!(first.end_censored_duration_seconds, Some(4.0));
        assert_eq!(first.count_above_threshold, 1);
        assert_eq!(first.post_threshold_seconds, 4.0);
        m.record_elapsed(8.0, &c, &p, Some(false));
        m.record_unavailability_transition(true, 8.0);
        let final_result = m.snapshot().episode_metrics.unwrap();
        assert_eq!(final_result.count_above_threshold, 1);
        assert_eq!(final_result.long_episode_seconds, 8.0);
        assert_eq!(final_result.post_threshold_seconds, 8.0);
        assert_eq!(final_result.end_censored_duration_seconds, None);
        assert_eq!(m.time_unavailable, 8.0);
        assert_eq!(m.time_to_first_unavailability, Some(0.0));
    }

    #[test]
    fn data_loss_milestones_record_only_the_first_occurrence() {
        let mut m = MetricsCollector::new();
        let mut c = cluster_of(3, 0.0);
        let p = LeaderlessProtocol::default();

        m.update(&c, 10.0, &p, Some(true));
        assert_eq!(m.time_to_potential_data_loss, None);

        // Two of three down: quorum lost.
        for i in c.active_indices().into_iter().take(2) {
            c.node_at_mut(i).is_available = false;
        }
        m.update(&c, 20.0, &p, Some(false));
        assert_eq!(m.time_to_potential_data_loss, Some(20.0));

        m.update(&c, 30.0, &p, Some(false));
        assert_eq!(m.time_to_potential_data_loss, Some(20.0));
    }

    #[test]
    fn actual_data_loss_is_recorded_when_every_node_is_gone() {
        let mut m = MetricsCollector::new();
        let mut c = cluster_of(3, 0.0);
        let p = LeaderlessProtocol::default();

        for i in c.active_indices() {
            c.node_at_mut(i).has_data = false;
        }
        m.update(&c, 42.0, &p, Some(false));
        assert_eq!(m.time_to_actual_data_loss, Some(42.0));
    }

    #[test]
    fn can_commit_short_circuits_the_potential_loss_check() {
        let mut m = MetricsCollector::new();
        let mut c = cluster_of(3, 0.0);
        let p = LeaderlessProtocol::default();

        // Quorum is genuinely lost, but passing can_commit=true asserts it
        // is not, and the check is skipped.
        for i in c.active_indices().into_iter().take(2) {
            c.node_at_mut(i).is_available = false;
        }
        m.update(&c, 20.0, &p, Some(true));
        assert_eq!(m.time_to_potential_data_loss, None);
    }

    #[test]
    fn nines_of_availability() {
        let mut m = MetricsCollector::new();
        m.time_available = 999.0;
        m.time_unavailable = 1.0;
        let nines = m.nines_of_availability().unwrap();
        assert!((nines - 3.0).abs() < 1e-12, "got {nines}");

        m.time_available = 0.0;
        m.time_unavailable = 100.0;
        assert_eq!(m.nines_of_availability(), Some(0.0));
    }

    #[test]
    fn snapshot_copies_every_field() {
        let mut m = MetricsCollector::new();
        m.time_available = 90.0;
        m.time_unavailable = 10.0;
        m.total_cost = 1.5;
        m.time_to_potential_data_loss = Some(5.0);
        m.time_to_actual_data_loss = Some(7.0);
        m.total_transient_failures = 3;
        m.total_dataloss_failures = 2;
        m.total_nodes_spawned = 1;
        m.total_unavailability_incidents = 4;
        m.total_leader_elections = 6;
        m.time_to_first_unavailability = Some(5.0);

        let s = m.snapshot();
        assert_eq!(s.time_available, 90.0);
        assert_eq!(s.time_unavailable, 10.0);
        assert_eq!(s.total_cost, 1.5);
        assert_eq!(s.time_to_potential_data_loss, Some(5.0));
        assert_eq!(s.time_to_actual_data_loss, Some(7.0));
        assert_eq!(s.total_transient_failures, 3);
        assert_eq!(s.total_dataloss_failures, 2);
        assert_eq!(s.total_nodes_spawned, 1);
        assert_eq!(s.total_unavailability_incidents, 4);
        assert_eq!(s.total_leader_elections, 6);
        assert_eq!(s.time_to_first_unavailability, Some(5.0));
        assert!((s.availability_fraction() - 0.9).abs() < 1e-12);
        assert_eq!(s.total_time(), 100.0);
    }

    #[test]
    fn empty_snapshot_reports_perfect_availability() {
        let s = MetricsCollector::new().snapshot();
        assert_eq!(s.availability_fraction(), 1.0);
    }
}
