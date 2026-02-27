# Powder Test Suite Summary

A comprehensive breakdown of every test in the repo, organized by file, with a concrete **what** and **how** for each.

---

## `tests/test_simulation.py` (~2400 lines)

### `TestTimeUnitHelpers`
- **What**: That `hours()`, `days()`, `minutes()` return the correct number of seconds.
- **How**: Directly assert `hours(1) == 3600`, `days(1) == 86400`, etc.

---

### `TestDistributions`
- **What**: That `Constant`, `Exponential`, and `Bounded` distributions produce correct samples and means.
- **How**: Sample from each distribution with a fixed seed; assert the sampled value matches the expected output. For `Bounded`, verify that samples never exceed the configured max.

---

### `TestClusterState`
- **What**: That `ClusterState` correctly counts up-to-date nodes, available nodes, and finds sync donors.
- **How**: Create small clusters with hard-coded `last_applied_index` values. Set some nodes as failed or lagging, then call `num_up_to_date()`, `num_available()`, and `find_sync_donor()` and assert the exact expected counts and node IDs.

---

### `TestMetricsCollector`
- **What**: That availability fractions are computed exactly.
- **How**: Manually feed time intervals and cluster states into a `MetricsCollector`. For example, 50s of availability followed by 50s of unavailability asserts exactly 0.5 availability.

---

### `TestSimulatorBasics`
- **What**: That the event loop advances time, fires events in order, and terminates correctly.
- **How**: Create a 3-node cluster with `Constant` failure/recovery distributions. Run for a fixed number of seconds, assert `result.end_time == expected`, and that `event_log` contains events in chronological order.

---

### `TestLeaderlessProtocol`
- **What**: That the leaderless protocol's `can_commit()` only returns `True` when a quorum of up-to-date nodes exists.
- **How**: Directly call `can_commit(cluster)` with specific cluster states (e.g., 2 out of 3 nodes lagging) and assert `True` or `False`.

---

### `TestDeterministicAvailabilityMetrics`
- **What**: That availability fractions match exact arithmetic predictions for known failure timelines.
- **How**: Construct scenarios where node A fails at `t=10` and recovers at `t=110` in a 3-node cluster. Calculate the expected availability fraction analytically (e.g., the single-node failure window as a fraction of total time), then run the simulator and assert the metric matches within a tolerance of `1e-9`.

---

### `TestRaftLikeAvailability`
- **What**: That a Raft-like cluster's availability hit is proportional to **election time**, not node recovery time.
- **How**: Configure a cluster where a node fails and recovery takes 100s but election takes only 5s. Assert that the unavailable window is ~5s (election), not 100s (recovery).

---

### `TestSnapshotRecovery`

A set of tests verifying the path-selection logic for log-only vs. snapshot-based recovery:

- **`test_recovery_without_snapshot`** – **What**: A slightly-lagging node should use log-only replay. **How**: Set the lagging node to be behind the donor but still within the donor's log window (above the last snapshot boundary). Call `compute_sync_time()` and assert the result is much less than the snapshot download time.
- **`test_recovery_with_snapshot`** – **What**: A severely-lagging node (outside the GC window) must download a snapshot. **How**: Place the lagging node below the donor's `log_retention_ops` cutoff. Assert `compute_sync_time()` >= the snapshot download time.
- **`test_no_snapshot_interval_means_log_only`** – **What**: With `snapshot_interval=0`, recovery is always log-only. **How**: Set `snapshot_interval=0` and make a node very far behind. Assert sync time does not include any snapshot overhead.
- **`test_quorum_loss_freezes_commit_index`** – **What**: When quorum is lost, `commit_index` stops advancing. **How**: Fail enough nodes to break quorum at `t=10`, run to `t=200`, and assert `commit_index < 200`.
- **`test_node_snapshot_state_advances`** – **What**: Available nodes take snapshots at the right index boundaries. **How**: Set `snapshot_interval=50`, run for 200s, assert `last_snapshot_index % 50 == 0` for all available nodes at end.

---

### `TestDeterministicSyncModel`

This is the core sync-model validation suite. All distributions use `Constant` so outcomes are fully deterministic:

- **`test_basic_sync_timing`** – **What**: A node 10 units behind, with `log_replay_rate=2.0` and `commit_rate=1.0` (net rate = 1.0), syncs in exactly 10s. **How**: Set `node0.last_applied_index=0`, donors at 10. Run the simulator and check the `NODE_SYNC_COMPLETE` event fires at `t ≈ 10`.

- **`test_sync_pauses_when_donors_unavailable`** – **What**: That a node pauses when donors fail and resumes from the most up-to-date node when they return. **How**: Set `node0` 10 units behind. Fail both donors at `t=3`, recover at `t=7` (4s outage). Before the outage: 3s × net_rate 2.0 = 6 units closed; gap remaining = 4. After donors recover the sync resumes. Assert `NODE_SYNC_COMPLETE` fires at `t ≈ 9` (5s of productive time + 4s frozen).

- **`test_multinode_sync_with_donor_outage`** – **What**: Multiple lagging nodes all eventually complete sync despite repeated donor outages. **How**: Set 5 nodes, 2 lagging by different amounts, 3 donors cycling through outages at known intervals. Run for 40s and assert both lagging nodes have `sync is None` at the end.

- **`test_gc_log_only_within_window`** – **What**: When the node's position falls within the donor's log retention window, log-only replay is selected. **How**: Set `log_retention_ops=300`, donor at 250, node at 150 (50 units into the 300-unit window). Assert `SYNC_COMPLETE` fires in ~11s (not 60s+ which would indicate a snapshot was downloaded).

- **`test_gc_forced_snapshot_outside_window`** – **What**: When the node is behind the GC cutoff, a snapshot is forced, and the total sync time equals snapshot download + remaining log replay. **How**: Set `log_retention_ops=100`, donor at 250 (keeps log 150–250), node at 50 (outside window). Assert `SYNC_COMPLETE` fires at exactly `≈72.2s` (60s snapshot + 12.2s log suffix).

---

### `TestConvergenceCriteria`
- **What**: That `ConvergenceCriteria` validates its own configuration.
- **How**: Assert that specifying both `relative_error` and `absolute_error` raises `ValueError`. Assert that `confidence_level=1.0` or `min_runs=1` each raise `ValueError`. Assert computed `error_threshold` returns the correct value based on which mode is active.

---

### `TestAdaptiveMonteCarlo`
- **`test_run_until_converged_basic`** – **What**: The adaptive runner terminates and returns a valid `ConvergenceResult`. **How**: Configure a 3-node cluster with realistic failure rates. Set loose convergence criteria (10% relative error). Assert the runner completes within `max_runs=500` and the result contains availability samples.
- **`test_convergence_reduces_ci`** – **What**: CI half-width shrinks as more samples are added. **How**: Run 30 simulations, record the CI; then run 300, record the CI again. Assert `ci_30 > ci_300`.
- **`test_estimate_required_runs`** – **What**: Pilot data can predict roughly how many runs are needed. **How**: Run 30 pilot sims; call `estimate_required_runs()`. Assert the returned estimate is a positive integer and is larger than the current sample count when the CI is not yet met.

---

## `tests/test_raft_protocol.py` (~1200 lines)

### `TestLeaderElection`
- **`test_basic_election`** – **What**: A leader is elected at simulation start. **How**: Run a Raft cluster briefly, assert `protocol.leader_id is not None`.
- **`test_leader_failure_triggers_election`** – **What**: When the leader fails, an election begins. **How**: Configure the leader to fail at `t=5`. Run to `t=6`. Assert `protocol.election_in_progress == True`.
- **`test_candidate_failure_during_election`** – **What**: If a candidate fails during an election, the epoch counter prevents stale events. **How**: Force a leader failure, then immediately fail another eligible node before the `LEADER_ELECTION_COMPLETE` event fires. Assert that the stale election event is discarded (epoch mismatch) and a new election restarts.
- **`test_quorum_loss_mid_election`** – **What**: If quorum is lost during an active election, the election stalls instead of completing. **How**: Start an election by failing the leader, then fail enough other nodes to drop below quorum before the election event fires. Assert `protocol.election_stalled == True`. Then recover a node and assert a new `LEADER_ELECTION_COMPLETE` is scheduled.
- **`test_election_epoch_safety`** – **What**: Stale `LEADER_ELECTION_COMPLETE` events from a cancelled election are ignored. **How**: Inspect `protocol._election_epoch`. Induce two overlapping elections and verify only the latest epoch's completion event updates `leader_id`.

### `TestSyncDuringElection`
- **`test_sync_continues_during_election`** – **What**: A lagging node continues syncing even while the cluster is leaderless and can't commit. **How**: Start a sync, then trigger an election. Assert the node's `last_applied_index` advances at the full `log_replay_rate` (not the net rate), since `commit_rate_eff=0` when the cluster can't commit.
- **`test_leader_failure_during_sync`** – **What**: A node mid-sync handles a leader failure without losing progress. **How**: Begin a sync, then fire a `NODE_FAILURE` event on the leader. Assert the sync eventually completes after an election resolves the new leader.

---

## `tests/test_availability.py`

- **`test_single_node_cycling_leaderless`** – **What/How**: One node cycles (fail → recover → fail…) in a 3-node leaderless cluster. Because 2 of 3 nodes remain up, the cluster never loses quorum. Assert availability ≈ **100%**.
- **`test_single_node_cycling_raft`** – **What/How**: Same cycling scenario but Raft. Every time the cycling node is the leader, an election takes 5s. Assert unavailability ≈ `num_leader_tenure_windows × 5s / total_time`.
- **`test_overlapping_failures`** – **What/How**: Two nodes fail simultaneously in a 3-node cluster, dropping below quorum. Assert availability ≈ `(total_time - overlap_window) / total_time`.

---

## `tests/test_pricing.py`

- **`test_billing_transient_failure`** – **What**: A node that fails transiently (but doesn't lose data) continues to be billed the whole time. **How**: Run a cluster where one node cycles. Assert total cost = `cost_per_hour × hours × num_nodes` (the failed node still bills).
- **`test_billing_stops_on_data_loss`** – **What**: A node that experiences permanent data loss stops billing. **How**: Configure a node to lose data at `t=50`. Assert `cost_samples` reflects billing for only the first 50s for that node.
- **`test_replacement_reduces_cost`** – **What**: Replacing a failed node with a cheaper one reduces overall cost. **How**: Use a strategy that spawns a lower-cost replacement after a failure. Assert the final total cost is less than the baseline with no replacement.

---

## `tests/test_markov_utils.py`

- **What**: That Markov chain transition matrices are valid (rows sum to 1.0, no negative probabilities).
- **How**: Build small test matrices, call the utility function, assert `np.allclose(row.sum(), 1.0)` for all rows.

---

## `tests/test_rsm_model_utils.py`

- **What**: That RSM-specific model invariants hold (e.g., state space is correctly constructed).
- **How**: Build a small RSM model, call utility functions, assert expected structure (number of states, valid transitions, etc.).
