"""
Monte Carlo simulation runner for RSM deployments.

Runs multiple simulations in parallel and aggregates results to compute
statistics on availability, data loss timing, and cost.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .simulation.cluster import ClusterState
from .simulation.distributions import Seconds
from .simulation.metrics import MetricsSnapshot
from .simulation.network import NetworkConfig, NetworkState
from .simulation.simulator import SimulationResult, Simulator
from .simulation.strategy import ClusterStrategy


@dataclass
class MonteCarloConfig:
    """Configuration for Monte Carlo simulation.

    Attributes:
        num_simulations: Number of simulation runs to execute.
        max_time: Maximum time per simulation (seconds), or None for no limit.
        stop_on_data_loss: Whether to stop each run when data loss occurs.
        parallel_workers: Number of parallel worker processes (1 = sequential).
        base_seed: Base seed for reproducibility (each run gets base_seed + run_index).
    """

    num_simulations: int
    max_time: Seconds | None = None
    stop_on_data_loss: bool = True
    parallel_workers: int = 1
    base_seed: int | None = None


@dataclass
class MonteCarloResults:
    """Aggregated results from Monte Carlo simulation.

    Attributes:
        availability_samples: Availability fraction for each run.
        time_to_potential_loss_samples: Time to potential data loss for each run.
        time_to_actual_loss_samples: Time to actual data loss for each run.
        cost_samples: Total cost for each run.
        end_reasons: Reason each simulation ended.
        simulation_results: Full results for each run (if requested).
    """

    availability_samples: list[float] = field(default_factory=list)
    time_to_potential_loss_samples: list[Seconds | None] = field(default_factory=list)
    time_to_actual_loss_samples: list[Seconds | None] = field(default_factory=list)
    cost_samples: list[float] = field(default_factory=list)
    end_reasons: list[str] = field(default_factory=list)
    simulation_results: list[SimulationResult] = field(default_factory=list)

    def availability_mean(self) -> float:
        """Calculate mean availability across all runs."""
        if not self.availability_samples:
            return 0.0
        return float(np.mean(self.availability_samples))

    def availability_std(self) -> float:
        """Calculate standard deviation of availability."""
        if len(self.availability_samples) < 2:
            return 0.0
        return float(np.std(self.availability_samples, ddof=1))

    def availability_percentile(self, p: float) -> float:
        """Calculate a percentile of availability.

        Args:
            p: Percentile (0-100).

        Returns:
            Availability at the given percentile.
        """
        if not self.availability_samples:
            return 0.0
        return float(np.percentile(self.availability_samples, p))

    def cost_mean(self) -> float:
        """Calculate mean cost across all runs."""
        if not self.cost_samples:
            return 0.0
        return float(np.mean(self.cost_samples))

    def time_to_actual_loss_samples_filtered(self) -> list[float]:
        """Get time to actual loss samples, excluding None values."""
        return [t for t in self.time_to_actual_loss_samples if t is not None]

    def time_to_potential_loss_samples_filtered(self) -> list[float]:
        """Get time to potential loss samples, excluding None values."""
        return [t for t in self.time_to_potential_loss_samples if t is not None]

    def data_loss_probability(self) -> float:
        """Calculate probability of data loss occurring.

        Returns:
            Fraction of runs that experienced actual data loss.
        """
        if not self.time_to_actual_loss_samples:
            return 0.0
        loss_count = sum(1 for t in self.time_to_actual_loss_samples if t is not None)
        return loss_count / len(self.time_to_actual_loss_samples)

    def mean_time_to_actual_loss(self) -> float | None:
        """Calculate mean time to actual data loss.

        Returns:
            Mean time in seconds, or None if no runs had data loss.
        """
        filtered = self.time_to_actual_loss_samples_filtered()
        if not filtered:
            return None
        return float(np.mean(filtered))

    def mean_time_to_potential_loss(self) -> float | None:
        """Calculate mean time to potential data loss.

        Returns:
            Mean time in seconds, or None if no runs had potential loss.
        """
        filtered = self.time_to_potential_loss_samples_filtered()
        if not filtered:
            return None
        return float(np.mean(filtered))

    def time_to_loss_pdf(
        self, bins: int = 50, actual: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute PDF histogram for time to data loss.

        Args:
            bins: Number of histogram bins.
            actual: If True, use actual loss times; if False, use potential loss times.

        Returns:
            Tuple of (bin_centers, densities).
        """
        if actual:
            samples = self.time_to_actual_loss_samples_filtered()
        else:
            samples = self.time_to_potential_loss_samples_filtered()

        if not samples:
            return np.array([]), np.array([])

        counts, bin_edges = np.histogram(samples, bins=bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        return bin_centers, counts

    def time_to_loss_cdf(self, actual: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """Compute empirical CDF for time to data loss.

        Args:
            actual: If True, use actual loss times; if False, use potential loss times.

        Returns:
            Tuple of (sorted_times, cumulative_probabilities).
        """
        if actual:
            samples = self.time_to_actual_loss_samples_filtered()
        else:
            samples = self.time_to_potential_loss_samples_filtered()

        if not samples:
            return np.array([]), np.array([])

        sorted_samples = np.sort(samples)
        cdf = np.arange(1, len(sorted_samples) + 1) / len(sorted_samples)
        return sorted_samples, cdf

    def summary(self) -> str:
        """Generate a text summary of results."""
        lines = [
            f"Monte Carlo Results ({len(self.availability_samples)} runs)",
            f"  Availability: {self.availability_mean()*100:.2f}% "
            f"(std: {self.availability_std()*100:.2f}%)",
            f"  Data loss probability: {self.data_loss_probability()*100:.1f}%",
        ]

        mttl = self.mean_time_to_actual_loss()
        if mttl is not None:
            lines.append(f"  Mean time to data loss: {mttl/86400:.1f} days")

        lines.append(f"  Mean cost: ${self.cost_mean():.2f}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"MonteCarloResults(n={len(self.availability_samples)}, "
            f"availability={self.availability_mean()*100:.2f}%)"
        )


def _run_single_simulation(
    cluster_factory: Callable[[], ClusterState],
    strategy_factory: Callable[[], ClusterStrategy],
    network_config: NetworkConfig | None,
    max_time: Seconds | None,
    stop_on_data_loss: bool,
    seed: int,
) -> SimulationResult:
    """Run a single simulation (used for parallel execution).

    This is a module-level function to support multiprocessing.
    """
    cluster = cluster_factory()
    strategy = strategy_factory()

    simulator = Simulator(
        initial_cluster=cluster,
        strategy=strategy,
        network_config=network_config,
        seed=seed,
        log_events=False,
    )

    if stop_on_data_loss:
        return simulator.run_until_data_loss(max_time=max_time)
    else:
        return simulator.run_until(end_time=max_time)


class MonteCarloRunner:
    """Runs multiple simulations and aggregates results.

    Supports parallel execution for faster results on multi-core systems.
    """

    def __init__(self, config: MonteCarloConfig):
        """Initialize the runner.

        Args:
            config: Monte Carlo configuration.
        """
        self.config = config

    def run(
        self,
        cluster_factory: Callable[[], ClusterState],
        strategy_factory: Callable[[], ClusterStrategy],
        network_config: NetworkConfig | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> MonteCarloResults:
        """Run Monte Carlo simulations.

        Args:
            cluster_factory: Callable that creates a fresh ClusterState for each run.
            strategy_factory: Callable that creates a fresh ClusterStrategy for each run.
            network_config: Optional network configuration (shared across runs).
            progress_callback: Optional callback(completed, total) for progress updates.

        Returns:
            Aggregated MonteCarloResults.
        """
        results = MonteCarloResults()

        if self.config.parallel_workers > 1:
            self._run_parallel(
                cluster_factory,
                strategy_factory,
                network_config,
                results,
                progress_callback,
            )
        else:
            self._run_sequential(
                cluster_factory,
                strategy_factory,
                network_config,
                results,
                progress_callback,
            )

        return results

    def _run_sequential(
        self,
        cluster_factory: Callable[[], ClusterState],
        strategy_factory: Callable[[], ClusterStrategy],
        network_config: NetworkConfig | None,
        results: MonteCarloResults,
        progress_callback: Callable[[int, int], None] | None,
    ) -> None:
        """Run simulations sequentially."""
        for i in range(self.config.num_simulations):
            seed = (self.config.base_seed + i) if self.config.base_seed else None

            sim_result = _run_single_simulation(
                cluster_factory=cluster_factory,
                strategy_factory=strategy_factory,
                network_config=network_config,
                max_time=self.config.max_time,
                stop_on_data_loss=self.config.stop_on_data_loss,
                seed=seed,
            )

            self._collect_result(sim_result, results)

            if progress_callback:
                progress_callback(i + 1, self.config.num_simulations)

    def _run_parallel(
        self,
        cluster_factory: Callable[[], ClusterState],
        strategy_factory: Callable[[], ClusterStrategy],
        network_config: NetworkConfig | None,
        results: MonteCarloResults,
        progress_callback: Callable[[int, int], None] | None,
    ) -> None:
        """Run simulations in parallel using ProcessPoolExecutor."""
        completed = 0

        with ProcessPoolExecutor(max_workers=self.config.parallel_workers) as executor:
            futures = []

            for i in range(self.config.num_simulations):
                seed = (self.config.base_seed + i) if self.config.base_seed else None

                future = executor.submit(
                    _run_single_simulation,
                    cluster_factory=cluster_factory,
                    strategy_factory=strategy_factory,
                    network_config=network_config,
                    max_time=self.config.max_time,
                    stop_on_data_loss=self.config.stop_on_data_loss,
                    seed=seed,
                )
                futures.append(future)

            for future in as_completed(futures):
                sim_result = future.result()
                self._collect_result(sim_result, results)

                completed += 1
                if progress_callback:
                    progress_callback(completed, self.config.num_simulations)

    def _collect_result(
        self, sim_result: SimulationResult, results: MonteCarloResults
    ) -> None:
        """Extract metrics from a simulation result and add to aggregated results."""
        metrics = sim_result.metrics

        results.availability_samples.append(metrics.availability_fraction())
        results.time_to_potential_loss_samples.append(metrics.time_to_potential_data_loss)
        results.time_to_actual_loss_samples.append(metrics.time_to_actual_data_loss)
        results.cost_samples.append(metrics.total_cost)
        results.end_reasons.append(sim_result.end_reason)


def run_monte_carlo(
    cluster_factory: Callable[[], ClusterState],
    strategy_factory: Callable[[], ClusterStrategy],
    num_simulations: int,
    max_time: Seconds | None = None,
    network_config: NetworkConfig | None = None,
    stop_on_data_loss: bool = True,
    parallel_workers: int = 1,
    seed: int | None = None,
) -> MonteCarloResults:
    """Convenience function to run Monte Carlo simulations.

    Args:
        cluster_factory: Callable that creates a fresh ClusterState for each run.
        strategy_factory: Callable that creates a fresh ClusterStrategy for each run.
        num_simulations: Number of simulation runs.
        max_time: Maximum time per simulation.
        network_config: Optional network configuration.
        stop_on_data_loss: Whether to stop on data loss.
        parallel_workers: Number of parallel workers.
        seed: Base random seed.

    Returns:
        MonteCarloResults with aggregated statistics.
    """
    config = MonteCarloConfig(
        num_simulations=num_simulations,
        max_time=max_time,
        stop_on_data_loss=stop_on_data_loss,
        parallel_workers=parallel_workers,
        base_seed=seed,
    )

    runner = MonteCarloRunner(config)
    return runner.run(
        cluster_factory=cluster_factory,
        strategy_factory=strategy_factory,
        network_config=network_config,
    )
