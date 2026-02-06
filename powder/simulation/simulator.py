"""
Discrete-event simulator engine for the Monte Carlo RSM simulator.

The simulator processes events in time order, updating cluster state
and invoking the strategy to react to changes.
"""

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .cluster import ClusterState
from .distributions import Seconds
from .events import Event, EventQueue, EventType
from .metrics import MetricsCollector, MetricsSnapshot
from .network import NetworkConfig
from .node import NodeConfig, NodeState
from .strategy import Action, ActionType, ClusterStrategy


@dataclass
class SimulationResult:
    """Result of a simulation run.

    Attributes:
        end_time: Final simulation time in seconds.
        end_reason: Why the simulation ended ("time_limit", "data_loss", "condition_met").
        metrics: Final metrics snapshot.
        event_log: List of all events processed (if logging enabled).
        final_cluster: Final cluster state.
    """

    end_time: Seconds
    end_reason: str
    metrics: MetricsSnapshot
    event_log: list[Event] = field(default_factory=list)
    final_cluster: ClusterState | None = None


class Simulator:
    """Discrete-event simulation engine for RSM clusters.

    The simulator maintains:
    - A cluster state that evolves over time
    - An event queue ordered by time
    - A strategy that reacts to events
    - A metrics collector tracking availability and cost

    Events are processed in time order. Each event may:
    1. Update cluster state (e.g., node fails)
    2. Trigger strategy reactions (e.g., spawn replacement)
    3. Schedule new events (e.g., recovery time)
    """

    def __init__(
        self,
        initial_cluster: ClusterState,
        strategy: ClusterStrategy,
        network_config: NetworkConfig | None = None,
        seed: int | None = None,
        log_events: bool = False,
    ):
        """Initialize the simulator.

        Args:
            initial_cluster: Starting cluster state.
            strategy: Strategy for reacting to events.
            network_config: Optional network partition configuration.
            seed: Random seed for reproducibility.
            log_events: Whether to keep a log of all events.
        """
        self.cluster = initial_cluster
        self.strategy = strategy
        self.network_config = network_config
        self.log_events = log_events

        self.rng = np.random.default_rng(seed)
        self.event_queue = EventQueue()
        self.metrics = MetricsCollector()
        self.event_log: list[Event] = []

        self._initialized = False

    def _initialize(self) -> None:
        """Set up initial events and state."""
        if self._initialized:
            return

        # Schedule initial events for each node
        for node in self.cluster.nodes.values():
            self._schedule_node_events(node)

        # Schedule network events if configured
        if self.network_config and self.network_config.affected_regions:
            self._schedule_network_outage()

        # Let strategy take initial actions
        actions = self.strategy.on_simulation_start(
            self.cluster, self.event_queue, self.rng
        )
        for action in actions:
            self._execute_action(action)

        self._initialized = True

    def _schedule_node_events(self, node: NodeState) -> None:
        """Schedule failure and data loss events for a node."""
        current_time = self.cluster.current_time

        # Schedule transient failure
        failure_time = node.config.failure_dist.sample(self.rng)
        self.event_queue.push(
            Event(
                time=Seconds(current_time + failure_time),
                event_type=EventType.NODE_FAILURE,
                target_id=node.node_id,
            )
        )

        # Schedule data loss
        data_loss_time = node.config.data_loss_dist.sample(self.rng)
        self.event_queue.push(
            Event(
                time=Seconds(current_time + data_loss_time),
                event_type=EventType.NODE_DATA_LOSS,
                target_id=node.node_id,
            )
        )

    def _schedule_network_outage(self) -> None:
        """Schedule the next network outage event."""
        if not self.network_config or not self.network_config.affected_regions:
            return

        current_time = self.cluster.current_time
        outage_delay = self.network_config.outage_dist.sample(self.rng)

        # Pick a random region that can experience an outage
        region_idx = self.rng.integers(len(self.network_config.affected_regions))
        region = self.network_config.affected_regions[region_idx]

        self.event_queue.push(
            Event(
                time=Seconds(current_time + outage_delay),
                event_type=EventType.NETWORK_OUTAGE_START,
                target_id=region,
                metadata={"region": region},
            )
        )

    def _schedule_sync_complete(self, node: NodeState) -> None:
        """Schedule sync completion for a lagging node."""
        current_time = self.cluster.current_time

        if node.is_up_to_date(current_time):
            return  # Already up to date

        sync_rate = node.config.sync_rate_dist.sample(self.rng)
        time_to_sync = node.time_to_sync(current_time, sync_rate)

        if time_to_sync is not None and time_to_sync > 0:
            sync_complete_time = Seconds(current_time + time_to_sync)
            self.event_queue.push(
                Event(
                    time=sync_complete_time,
                    event_type=EventType.NODE_SYNC_COMPLETE,
                    target_id=node.node_id,
                    metadata={"sync_to_time": sync_complete_time},
                )
            )

    def _process_event(self, event: Event) -> None:
        """Process a single event and update cluster state."""
        # Update metrics for time elapsed
        self.metrics.update(self.cluster, event.time)
        self.cluster.current_time = event.time

        if self.log_events:
            self.event_log.append(event)

        # Apply event to cluster state
        if event.event_type == EventType.NODE_FAILURE:
            self._apply_node_failure(event)

        elif event.event_type == EventType.NODE_RECOVERY:
            self._apply_node_recovery(event)

        elif event.event_type == EventType.NODE_DATA_LOSS:
            self._apply_node_data_loss(event)

        elif event.event_type == EventType.NODE_SYNC_COMPLETE:
            self._apply_node_sync_complete(event)

        elif event.event_type == EventType.NODE_SPAWN_COMPLETE:
            self._apply_node_spawn_complete(event)

        elif event.event_type == EventType.NETWORK_OUTAGE_START:
            self._apply_network_outage_start(event)

        elif event.event_type == EventType.NETWORK_OUTAGE_END:
            self._apply_network_outage_end(event)

        # Let strategy react
        actions = self.strategy.on_event(
            event, self.cluster, self.event_queue, self.rng
        )
        for action in actions:
            self._execute_action(action)

    def _apply_node_failure(self, event: Event) -> None:
        """Apply a transient node failure."""
        node = self.cluster.get_node(event.target_id)
        if node and node.has_data:
            node.is_available = False

            # Cancel any pending sync for this node
            self.event_queue.cancel_events_for(
                event.target_id, EventType.NODE_SYNC_COMPLETE
            )

            # Schedule recovery
            recovery_time = node.config.recovery_dist.sample(self.rng)
            self.event_queue.push(
                Event(
                    time=Seconds(self.cluster.current_time + recovery_time),
                    event_type=EventType.NODE_RECOVERY,
                    target_id=node.node_id,
                )
            )

            # Schedule next failure (after recovery)
            next_failure_time = node.config.failure_dist.sample(self.rng)
            self.event_queue.push(
                Event(
                    time=Seconds(
                        self.cluster.current_time + recovery_time + next_failure_time
                    ),
                    event_type=EventType.NODE_FAILURE,
                    target_id=node.node_id,
                )
            )

    def _apply_node_recovery(self, event: Event) -> None:
        """Apply node recovery from transient failure."""
        node = self.cluster.get_node(event.target_id)
        if node and node.has_data:
            node.is_available = True

            # Node may need to sync if it fell behind
            if not node.is_up_to_date(self.cluster.current_time):
                self._schedule_sync_complete(node)

    def _apply_node_data_loss(self, event: Event) -> None:
        """Apply permanent data loss to a node."""
        node = self.cluster.get_node(event.target_id)
        if node:
            node.has_data = False
            node.is_available = False

            # Cancel all pending events for this node
            self.event_queue.cancel_events_for(event.target_id)

    def _apply_node_sync_complete(self, event: Event) -> None:
        """Apply sync completion for a node."""
        node = self.cluster.get_node(event.target_id)
        if node and node.is_available and node.has_data:
            sync_to_time = event.metadata.get("sync_to_time", self.cluster.current_time)
            node.last_up_to_date_time = sync_to_time

    def _apply_node_spawn_complete(self, event: Event) -> None:
        """Apply completion of node spawning."""
        node_config = event.metadata.get("node_config")
        node_id = event.metadata.get("node_id", event.target_id)

        if node_config:
            # Create new node (starts with no data, needs to sync)
            new_node = NodeState(
                node_id=node_id,
                config=node_config,
                is_available=True,
                has_data=True,
                last_up_to_date_time=Seconds(0),  # Needs full sync
            )
            self.cluster.add_node(new_node)

            # Schedule events for new node
            self._schedule_node_events(new_node)

            # Schedule sync
            self._schedule_sync_complete(new_node)

    def _apply_network_outage_start(self, event: Event) -> None:
        """Apply start of a region network outage (all nodes in region become unavailable)."""
        region = event.metadata.get("region", event.target_id)
        if region:
            self.cluster.network.add_outage(region)

            # Schedule outage end
            duration = self.network_config.outage_duration_dist.sample(self.rng)
            self.event_queue.push(
                Event(
                    time=Seconds(self.cluster.current_time + duration),
                    event_type=EventType.NETWORK_OUTAGE_END,
                    target_id=region,
                    metadata={"region": region},
                )
            )

    def _apply_network_outage_end(self, event: Event) -> None:
        """Apply end of a region network outage."""
        region = event.metadata.get("region", event.target_id)
        if region:
            self.cluster.network.remove_outage(region)

            # Schedule next outage
            self._schedule_network_outage()

    def _execute_action(self, action: Action) -> None:
        """Execute a strategy action."""
        if action.action_type == ActionType.SPAWN_NODE:
            self._action_spawn_node(action)

        elif action.action_type == ActionType.REMOVE_NODE:
            node_id = action.params.get("node_id")
            if node_id:
                self.cluster.remove_node(node_id)
                self.event_queue.cancel_events_for(node_id)

        elif action.action_type == ActionType.SCALE_DOWN:
            new_size = action.params.get("new_size", self.cluster.target_cluster_size)
            self.cluster.target_cluster_size = new_size

        elif action.action_type == ActionType.SCALE_UP:
            new_size = action.params.get("new_size", self.cluster.target_cluster_size)
            self.cluster.target_cluster_size = new_size

        elif action.action_type == ActionType.START_SYNC:
            node_id = action.params.get("node_id")
            node = self.cluster.get_node(node_id) if node_id else None
            if node:
                self._schedule_sync_complete(node)

    def _action_spawn_node(self, action: Action) -> None:
        """Execute a spawn node action."""
        node_config: NodeConfig = action.params.get("node_config")
        node_id: str = action.params.get("node_id", f"spawned_{self.rng.integers(10000)}")

        if node_config:
            spawn_time = node_config.spawn_dist.sample(self.rng)
            self.event_queue.push(
                Event(
                    time=Seconds(self.cluster.current_time + spawn_time),
                    event_type=EventType.NODE_SPAWN_COMPLETE,
                    target_id=node_id,
                    metadata={"node_config": node_config, "node_id": node_id},
                )
            )

    def run_until(
        self,
        end_time: Seconds | None = None,
        stop_condition: Callable[[ClusterState], bool] | None = None,
    ) -> SimulationResult:
        """Run the simulation until a stopping condition is met.

        Args:
            end_time: Maximum simulation time (or None for no limit).
            stop_condition: Callable that returns True when simulation should stop.

        Returns:
            SimulationResult with final metrics and state.
        """
        self._initialize()

        end_reason = "unknown"

        while True:
            # Check for empty queue
            if self.event_queue.is_empty():
                end_reason = "no_events"
                break

            # Peek at next event
            next_event = self.event_queue.peek()
            if next_event is None:
                end_reason = "no_events"
                break

            # Check time limit
            if end_time is not None and next_event.time > end_time:
                # Update metrics up to end_time
                self.metrics.update(self.cluster, end_time)
                self.cluster.current_time = end_time
                end_reason = "time_limit"
                break

            # Process event
            event = self.event_queue.pop()
            if event:
                self._process_event(event)

            # Check for data loss first (more specific)
            if self.cluster.has_actual_data_loss():
                end_reason = "data_loss"
                break

            # Check stop condition
            if stop_condition and stop_condition(self.cluster):
                end_reason = "condition_met"
                break

        return SimulationResult(
            end_time=self.cluster.current_time,
            end_reason=end_reason,
            metrics=self.metrics.snapshot(),
            event_log=self.event_log if self.log_events else [],
            final_cluster=self.cluster,
        )

    def run_for(self, duration: Seconds) -> SimulationResult:
        """Run the simulation for a specified duration.

        Args:
            duration: How long to run in seconds.

        Returns:
            SimulationResult with final metrics and state.
        """
        end_time = Seconds(self.cluster.current_time + duration)
        return self.run_until(end_time=end_time)

    def run_until_data_loss(
        self, max_time: Seconds | None = None
    ) -> SimulationResult:
        """Run until data loss occurs or time limit reached.

        Args:
            max_time: Maximum simulation time.

        Returns:
            SimulationResult with final metrics and state.
        """
        return self.run_until(
            end_time=max_time,
            stop_condition=lambda c: c.has_actual_data_loss(),
        )
