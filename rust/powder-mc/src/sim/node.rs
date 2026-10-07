//! Node model.
//!
//! Port of `powder/simulation/node.py`.  `NodeConfig` holds the static
//! properties and distributions; `NodeState` holds everything that changes
//! during a run.

use std::rc::Rc;

use super::distributions::Distribution;
use super::ids::Sym;

/// Static configuration for a node.
///
/// Python treats this as frozen and has `__deepcopy__` return `self` to avoid
/// re-copying distributions on every simulation.  Here it is shared through
/// an [`Rc`] for the same reason.
#[derive(Debug, Clone, PartialEq)]
pub struct NodeConfig {
    /// Geographic region where the node is located.
    pub region: String,
    /// Dollar cost of running this node per hour.
    pub cost_per_hour: f64,
    /// Time (seconds) until transient unavailability.
    pub failure_dist: Distribution,
    /// Time (seconds) to recover from a transient failure.
    pub recovery_dist: Distribution,
    /// Time (seconds) until permanent data loss.
    pub data_loss_dist: Distribution,
    /// Log replay rate (committed-data units replayed per second of wall time).
    pub log_replay_rate_dist: Distribution,
    /// Wall-clock time (seconds) to download a full snapshot from a donor.
    pub snapshot_download_time_dist: Distribution,
    /// Time (seconds) to spawn a fresh replacement node.
    pub spawn_dist: Distribution,
}

/// Shared handle to a node configuration.
pub type NodeConfigRef = Rc<NodeConfig>;

/// Which phase of catch-up a syncing node is in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncPhase {
    /// Downloading a full snapshot from the donor.
    SnapshotDownload,
    /// Replaying the donor's log.
    LogReplay,
}

impl SyncPhase {
    /// The string form Python uses, for diagnostics and JSON.
    pub fn as_str(self) -> &'static str {
        match self {
            SyncPhase::SnapshotDownload => "snapshot_download",
            SyncPhase::LogReplay => "log_replay",
        }
    }
}

/// Active sync state for a node catching up to its donor.
#[derive(Debug, Clone, PartialEq)]
pub struct SyncState {
    /// Node the data is being pulled from.
    pub donor_id: Sym,
    /// Whether the node is downloading a snapshot or replaying the log.
    pub phase: SyncPhase,
    /// Sampled replay rate (committed-data units per second of wall time)
    /// for this sync session.
    pub log_replay_rate: f64,
    /// Wall-clock seconds remaining on the snapshot download.  Only
    /// meaningful while `phase` is `SnapshotDownload`.
    pub snapshot_remaining: f64,
    /// Commit-index position the node jumps to once the snapshot download
    /// completes.  Only meaningful while `phase` is `SnapshotDownload`.
    pub target_snapshot_index: f64,
}

impl SyncState {
    /// A log-replay-only sync (no snapshot download phase).
    pub fn log_replay(donor_id: Sym, log_replay_rate: f64) -> Self {
        SyncState {
            donor_id,
            phase: SyncPhase::LogReplay,
            log_replay_rate,
            snapshot_remaining: 0.0,
            target_snapshot_index: 0.0,
        }
    }

    /// A sync that starts by downloading a snapshot, then replays the suffix.
    pub fn snapshot_download(
        donor_id: Sym,
        log_replay_rate: f64,
        snapshot_remaining: f64,
        target_snapshot_index: f64,
    ) -> Self {
        SyncState {
            donor_id,
            phase: SyncPhase::SnapshotDownload,
            log_replay_rate,
            snapshot_remaining,
            target_snapshot_index,
        }
    }
}

/// Which membership set a node currently belongs to.
///
/// Python keeps three separate dicts (`nodes`, `standby_nodes`,
/// `provisioning_nodes`).  The port keeps one vector and tags each entry, so
/// promotion is a field write rather than a remove-and-reinsert.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Group {
    /// Participates in quorum and availability checks.
    Active,
    /// Running and billable, but not counted toward quorum until promoted.
    Standby,
    /// Requested but not yet spawned.  Billed, but otherwise inert.
    Provisioning,
}

/// Dynamic state of a node during simulation.
#[derive(Debug, Clone, PartialEq)]
pub struct NodeState {
    /// Interned unique identifier.
    pub node_id: Sym,
    /// Interned region name, cached from `config.region` so availability
    /// checks never touch a string.
    pub region: Sym,
    /// Shared static configuration.
    pub config: NodeConfigRef,
    /// Membership set this node belongs to.
    pub group: Group,
    /// Whether the node is currently up (not transiently failed).
    pub is_available: bool,
    /// Whether the node still has its data (false after permanent loss).
    pub has_data: bool,
    /// Position in the committed data stream this node has applied up to.
    pub last_applied_index: f64,
    /// Position at which this node last took a snapshot.
    pub last_snapshot_index: f64,
    /// Active catch-up state, if the node is syncing.
    pub sync: Option<SyncState>,
}

impl NodeState {
    /// Create a healthy active node at commit index zero.
    pub fn new(node_id: Sym, region: Sym, config: NodeConfigRef) -> Self {
        NodeState {
            node_id,
            region,
            config,
            group: Group::Active,
            is_available: true,
            has_data: true,
            last_applied_index: 0.0,
            last_snapshot_index: 0.0,
            sync: None,
        }
    }

    /// Whether the node has applied everything up to `commit_index`.
    #[inline]
    pub fn is_up_to_date(&self, commit_index: f64) -> bool {
        self.last_applied_index >= commit_index
    }

    /// How much committed data the node is missing, floored at zero.
    #[inline]
    pub fn lag(&self, commit_index: f64) -> f64 {
        let d = commit_index - self.last_applied_index;
        if d > 0.0 {
            d
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::distributions::Distribution;

    fn config() -> NodeConfigRef {
        Rc::new(NodeConfig {
            region: "us-east".to_string(),
            cost_per_hour: 1.0,
            failure_dist: Distribution::constant(100.0),
            recovery_dist: Distribution::constant(10.0),
            data_loss_dist: Distribution::constant(1000.0),
            log_replay_rate_dist: Distribution::constant(10.0),
            snapshot_download_time_dist: Distribution::constant(5.0),
            spawn_dist: Distribution::constant(30.0),
        })
    }

    #[test]
    fn new_node_is_healthy_and_current() {
        let node = NodeState::new(2, 3, config());
        assert!(node.is_available);
        assert!(node.has_data);
        assert_eq!(node.group, Group::Active);
        assert!(node.sync.is_none());
        assert!(node.is_up_to_date(0.0));
    }

    #[test]
    fn up_to_date_and_lag_track_commit_index() {
        let mut node = NodeState::new(2, 3, config());
        node.last_applied_index = 50.0;
        assert!(node.is_up_to_date(50.0));
        assert!(node.is_up_to_date(49.0));
        assert!(!node.is_up_to_date(51.0));
        assert_eq!(node.lag(80.0), 30.0);
        // A node ahead of the frontier has zero lag, not negative.
        assert_eq!(node.lag(10.0), 0.0);
    }

    #[test]
    fn sync_state_constructors_set_the_phase() {
        let s = SyncState::log_replay(7, 2.0);
        assert_eq!(s.phase, SyncPhase::LogReplay);
        assert_eq!(s.snapshot_remaining, 0.0);

        let s = SyncState::snapshot_download(7, 2.0, 15.0, 400.0);
        assert_eq!(s.phase, SyncPhase::SnapshotDownload);
        assert_eq!(s.snapshot_remaining, 15.0);
        assert_eq!(s.target_snapshot_index, 400.0);
    }

    #[test]
    fn sync_phase_strings_match_python() {
        assert_eq!(SyncPhase::SnapshotDownload.as_str(), "snapshot_download");
        assert_eq!(SyncPhase::LogReplay.as_str(), "log_replay");
    }
}
