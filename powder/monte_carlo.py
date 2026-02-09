"""
Monte Carlo simulation runner for RSM deployments.

Runs multiple simulations in parallel and aggregates results to compute
statistics on availability, data loss timing, and cost.

Supports both fixed-count runs and adaptive convergence-based runs that
automatically determine the required sample size for high-confidence results.
"""

import copy
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

import numpy as np
from scipy import stats as scipy_stats

from .simulation.cluster import ClusterState
from .simulation.distributions import Seconds
from .simulation.metrics import MetricsSnapshot
from .simulation.network import NetworkConfig, NetworkState
from .simulation.protocol import Protocol
from .simulation.simulator import SimulationResult, Simulator
from .simulation.strategy import ClusterStrategy


class ConvergenceMetric(Enum):
    """Metrics that can be targeted for convergence."""

    AVAILABILITY = "availability"
    DATA_LOSS_PROBABILITY = "data_loss_probability"
    COST = "cost"
    MEAN_TIME_TO_DATA_LOSS = "mean_time_to_data_loss"


@dataclass
class ConvergenceCriteria:
    """Criteria for adaptive Monte Carlo convergence.

    The runner will keep adding simulation runs until the confidence
    interval for all target metrics is within the desired error tolerance,
    or until max_runs is reached.

    Supports two error modes (specify exactly one):
      - **relative_error**: CI half-width as a fraction of the mean.
            n ≈ (z * σ / (ε * μ))²
        Example: relative_error=0.05 means "within 5% of the mean".
      - **absolute_error**: CI half-width in the same units as the metric.
            n ≈ (z * σ / E)²
        Example: absolute_error=0.01 on availability means "mean is
        accurate to ±0.01" (±1 percentage point), i.e. 2 decimal places
        on the availability fraction.

    To get availability accurate to a specific number of decimal places,
    use absolute_error:
        1 decimal place (±0.1):   absolute_error=0.1
        2 decimal places (±0.01): absolute_error=0.01
        3 decimal places (±0.001): absolute_error=0.001

    For proportions (data loss probability), a Wald interval is used.

    Attributes:
        confidence_level: Desired confidence level (e.g., 0.95 for 95% CI).
        relative_error: Maximum relative half-width of CI as a fraction
            of the mean. Mutually exclusive with absolute_error.
        absolute_error: Maximum absolute half-width of CI in metric units.
            Mutually exclusive with relative_error.
        metrics: Which metrics to target for convergence. All must converge
            before stopping. Defaults to [AVAILABILITY].
        min_runs: Minimum number of runs before checking convergence.
            Must be >= 2 for variance estimation. Default 30.
        max_runs: Maximum number of runs (safety cap). Default 10000.
        batch_size: Number of runs per batch between convergence checks.
            Default 10.
    """

    confidence_level: float = 0.95
    relative_error: float | None = None
    absolute_error: float | None = None
    metrics: list[ConvergenceMetric] = field(
        default_factory=lambda: [ConvergenceMetric.AVAILABILITY]
    )
    min_runs: int = 30
    max_runs: int = 10_000
    batch_size: int = 10

    def __post_init__(self) -> None:
        if not 0 < self.confidence_level < 1:
            raise ValueError(
                f"confidence_level must be in (0, 1), got {self.confidence_level}"
            )

        # Default to relative_error=0.05 if neither is specified
        if self.relative_error is None and self.absolute_error is None:
            self.relative_error = 0.05

        if self.relative_error is not None and self.absolute_error is not None:
            raise ValueError(
                "Specify exactly one of relative_error or absolute_error, not both"
            )

        if self.relative_error is not None and self.relative_error <= 0:
            raise ValueError(
                f"relative_error must be > 0, got {self.relative_error}"
            )
        if self.absolute_error is not None and self.absolute_error <= 0:
            raise ValueError(
                f"absolute_error must be > 0, got {self.absolute_error}"
            )

        if self.min_runs < 2:
            raise ValueError(
                f"min_runs must be >= 2 for variance estimation, got {self.min_runs}"
            )
        if self.max_runs < self.min_runs:
            raise ValueError(
                f"max_runs ({self.max_runs}) must be >= min_runs ({self.min_runs})"
            )

    @property
    def uses_absolute_error(self) -> bool:
        """Whether this criteria uses absolute error mode."""
        return self.absolute_error is not None

    @property
    def error_threshold(self) -> float:
        """The configured error threshold (whichever mode is active)."""
        if self.absolute_error is not None:
            return self.absolute_error
        assert self.relative_error is not None
        return self.relative_error


@dataclass
class MetricConvergenceStatus:
    """Convergence status for a single metric.

    Attributes:
        metric: Which metric this status is for.
        converged: Whether the metric has converged.
        current_mean: Current sample mean.
        current_std: Current sample standard deviation.
        ci_half_width: Current confidence interval half-width (absolute).
        relative_error: Current relative error (ci_half_width / mean).
        estimated_runs_needed: Estimated total runs needed for convergence.
        num_samples: Number of samples collected so far.
    """

    metric: ConvergenceMetric
    converged: bool = False
    current_mean: float = 0.0
    current_std: float = 0.0
    ci_half_width: float = float("inf")
    relative_error: float = float("inf")
    estimated_runs_needed: int = 0
    num_samples: int = 0


@dataclass
class ConvergenceResult:
    """Result of an adaptive convergence run.

    Attributes:
        results: The aggregated Monte Carlo results.
        converged: Whether all target metrics converged.
        total_runs: Total number of simulation runs executed.
        metric_statuses: Convergence status for each target metric.
    """

    results: "MonteCarloResults"
    converged: bool
    total_runs: int
    metric_statuses: list[MetricConvergenceStatus] = field(default_factory=list)

    def summary(self) -> str:
        """Generate a text summary of convergence results."""
        lines = [
            self.results.summary(),
            "",
            f"Convergence: {'yes' if self.converged else 'NO'} "
            f"({self.total_runs} runs)",
        ]
        for status in self.metric_statuses:
            symbol = "+" if status.converged else "-"
            ci_lo = status.current_mean - status.ci_half_width
            ci_hi = status.current_mean + status.ci_half_width
            if status.metric == ConvergenceMetric.MEAN_TIME_TO_DATA_LOSS:
                # Display MTTDL in days for readability
                mean_d = status.current_mean / 86400
                lo_d = ci_lo / 86400
                hi_d = ci_hi / 86400
                hw_d = status.ci_half_width / 86400
                lines.append(
                    f"  [{symbol}] {status.metric.value}: "
                    f"mean={mean_d:.1f} days, "
                    f"CI=[{lo_d:.1f}, {hi_d:.1f}] days, "
                    f"±{hw_d:.1f} days (rel={status.relative_error:.4f}), "
                    f"est_n={status.estimated_runs_needed}"
                )
            else:
                lines.append(
                    f"  [{symbol}] {status.metric.value}: "
                    f"mean={status.current_mean:.4f}, "
                    f"CI=[{ci_lo:.4f}, {ci_hi:.4f}], "
                    f"±{status.ci_half_width:.4f} (rel={status.relative_error:.4f}), "
                    f"est_n={status.estimated_runs_needed}"
                )
        return "\n".join(lines)


@dataclass
class MonteCarloConfig:
    """Configuration for Monte Carlo simulation.

    Attributes:
        num_simulations: Number of simulation runs to execute.
        max_time: Maximum time per simulation in seconds. If None and
            stop_on_data_loss is True, each run continues indefinitely
            until data loss occurs (useful for MTTDL estimation).
        stop_on_data_loss: Whether to stop each run when data loss occurs.
        parallel_workers: Number of parallel worker processes (1 = sequential).
        base_seed: Base seed for reproducibility (each run gets base_seed + run_index).
    """

    num_simulations: int
    max_time: Seconds | None = None
    stop_on_data_loss: bool = True
    parallel_workers: int = 1
    base_seed: int | None = None

    def __post_init__(self) -> None:
        if self.max_time is None and not self.stop_on_data_loss:
            raise ValueError(
                "max_time is required when stop_on_data_loss is False, "
                "otherwise simulations would run indefinitely"
            )


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

    def std_time_to_actual_loss(self) -> float | None:
        """Calculate standard deviation of time to actual data loss.

        Returns:
            Standard deviation in seconds, or None if fewer than 2 runs had data loss.
        """
        filtered = self.time_to_actual_loss_samples_filtered()
        if len(filtered) < 2:
            return None
        return float(np.std(filtered, ddof=1))

    def ci_time_to_actual_loss(
        self, confidence_level: float = 0.95
    ) -> tuple[float, float] | None:
        """Calculate confidence interval for mean time to actual data loss.

        Uses the t-distribution for the CI.

        Args:
            confidence_level: Desired confidence level (e.g., 0.95 for 95% CI).

        Returns:
            Tuple of (lower_bound, upper_bound) in seconds, or None if
            fewer than 2 runs had data loss.
        """
        filtered = self.time_to_actual_loss_samples_filtered()
        n = len(filtered)
        if n < 2:
            return None
        sample_mean = float(np.mean(filtered))
        sample_std = float(np.std(filtered, ddof=1))
        alpha = 1.0 - confidence_level
        t_crit = scipy_stats.t.ppf(1 - alpha / 2, df=n - 1)
        margin = t_crit * sample_std / math.sqrt(n)
        return (sample_mean - margin, sample_mean + margin)

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
            ci = self.ci_time_to_actual_loss()
            if ci is not None:
                lines.append(
                    f"  Mean time to data loss: {mttl/86400:.1f} days "
                    f"(95% CI: [{ci[0]/86400:.1f}, {ci[1]/86400:.1f}] days)"
                )
            else:
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
    protocol: Protocol,
    network_config: NetworkConfig | None,
    max_time: Seconds | None,
    stop_on_data_loss: bool,
    seed: int,
) -> SimulationResult:
    """Run a single simulation (used for parallel execution).

    This is a module-level function to support multiprocessing.
    The protocol is deep-copied so each run has independent state
    (important for stateful protocols like RaftLikeProtocol).
    """
    cluster = cluster_factory()
    strategy = strategy_factory()
    run_protocol = copy.deepcopy(protocol)

    simulator = Simulator(
        initial_cluster=cluster,
        strategy=strategy,
        protocol=run_protocol,
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
        protocol: Protocol,
        network_config: NetworkConfig | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> MonteCarloResults:
        """Run Monte Carlo simulations.

        Args:
            cluster_factory: Callable that creates a fresh ClusterState for each run.
            strategy_factory: Callable that creates a fresh ClusterStrategy for each run.
            protocol: Protocol instance (deep-copied for each run).
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
                protocol,
                network_config,
                results,
                progress_callback,
            )
        else:
            self._run_sequential(
                cluster_factory,
                strategy_factory,
                protocol,
                network_config,
                results,
                progress_callback,
            )

        return results

    def _run_sequential(
        self,
        cluster_factory: Callable[[], ClusterState],
        strategy_factory: Callable[[], ClusterStrategy],
        protocol: Protocol,
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
                protocol=protocol,
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
        protocol: Protocol,
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
                    protocol=protocol,
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

    def run_until_converged(
        self,
        cluster_factory: Callable[[], ClusterState],
        strategy_factory: Callable[[], ClusterStrategy],
        protocol: Protocol,
        convergence: ConvergenceCriteria,
        network_config: NetworkConfig | None = None,
        progress_callback: Callable[[int, int, bool], None] | None = None,
    ) -> ConvergenceResult:
        """Run simulations adaptively until convergence criteria are met.

        Runs simulations in batches, checking after each batch whether
        the confidence intervals for all target metrics are within the
        desired relative error. Stops when all metrics converge or
        max_runs is reached.

        Args:
            cluster_factory: Callable that creates a fresh ClusterState for each run.
            strategy_factory: Callable that creates a fresh ClusterStrategy for each run.
            protocol: Protocol instance (deep-copied for each run).
            convergence: Convergence criteria specifying confidence level,
                relative error, and target metrics.
            network_config: Optional network configuration.
            progress_callback: Optional callback(completed, estimated_total, converged)
                for progress updates. estimated_total is the current best estimate
                of total runs needed (may change between batches).

        Returns:
            ConvergenceResult with the aggregated results and convergence status.
        """
        results = MonteCarloResults()
        run_count = 0

        # Phase 1: Run minimum batch
        initial_batch = convergence.min_runs
        self._run_batch(
            cluster_factory=cluster_factory,
            strategy_factory=strategy_factory,
            protocol=protocol,
            network_config=network_config,
            results=results,
            num_runs=initial_batch,
            start_index=run_count,
        )
        run_count += initial_batch

        # Check convergence after initial batch
        statuses = _check_convergence(results, convergence)
        all_converged = all(s.converged for s in statuses)

        if progress_callback:
            estimated_total = max(
                (s.estimated_runs_needed for s in statuses), default=run_count
            )
            progress_callback(run_count, estimated_total, all_converged)

        # Phase 2: Run additional batches until converged or max reached
        while not all_converged and run_count < convergence.max_runs:
            batch = min(convergence.batch_size, convergence.max_runs - run_count)
            if batch <= 0:
                break

            self._run_batch(
                cluster_factory=cluster_factory,
                strategy_factory=strategy_factory,
                protocol=protocol,
                network_config=network_config,
                results=results,
                num_runs=batch,
                start_index=run_count,
            )
            run_count += batch

            statuses = _check_convergence(results, convergence)
            all_converged = all(s.converged for s in statuses)

            if progress_callback:
                estimated_total = max(
                    (s.estimated_runs_needed for s in statuses), default=run_count
                )
                progress_callback(run_count, estimated_total, all_converged)

        return ConvergenceResult(
            results=results,
            converged=all_converged,
            total_runs=run_count,
            metric_statuses=statuses,
        )

    def _run_batch(
        self,
        cluster_factory: Callable[[], ClusterState],
        strategy_factory: Callable[[], ClusterStrategy],
        protocol: Protocol,
        network_config: NetworkConfig | None,
        results: MonteCarloResults,
        num_runs: int,
        start_index: int,
    ) -> None:
        """Run a batch of simulations and collect results.

        Args:
            cluster_factory: Callable that creates a fresh ClusterState.
            strategy_factory: Callable that creates a fresh ClusterStrategy.
            protocol: Protocol instance.
            network_config: Optional network configuration.
            results: Results object to append to.
            num_runs: Number of runs in this batch.
            start_index: Starting index for seed computation.
        """
        if self.config.parallel_workers > 1:
            with ProcessPoolExecutor(
                max_workers=self.config.parallel_workers
            ) as executor:
                futures = []
                for i in range(num_runs):
                    seed = (
                        (self.config.base_seed + start_index + i)
                        if self.config.base_seed is not None
                        else None
                    )
                    future = executor.submit(
                        _run_single_simulation,
                        cluster_factory=cluster_factory,
                        strategy_factory=strategy_factory,
                        protocol=protocol,
                        network_config=network_config,
                        max_time=self.config.max_time,
                        stop_on_data_loss=self.config.stop_on_data_loss,
                        seed=seed,
                    )
                    futures.append(future)

                for future in as_completed(futures):
                    sim_result = future.result()
                    self._collect_result(sim_result, results)
        else:
            for i in range(num_runs):
                seed = (
                    (self.config.base_seed + start_index + i)
                    if self.config.base_seed is not None
                    else None
                )
                sim_result = _run_single_simulation(
                    cluster_factory=cluster_factory,
                    strategy_factory=strategy_factory,
                    protocol=protocol,
                    network_config=network_config,
                    max_time=self.config.max_time,
                    stop_on_data_loss=self.config.stop_on_data_loss,
                    seed=seed,
                )
                self._collect_result(sim_result, results)

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


def _get_metric_samples(
    results: MonteCarloResults, metric: ConvergenceMetric
) -> np.ndarray:
    """Extract the sample array for a given convergence metric.

    Args:
        results: Monte Carlo results containing all samples.
        metric: Which metric to extract.

    Returns:
        NumPy array of sample values.
    """
    if metric == ConvergenceMetric.AVAILABILITY:
        return np.array(results.availability_samples)
    elif metric == ConvergenceMetric.COST:
        return np.array(results.cost_samples)
    elif metric == ConvergenceMetric.DATA_LOSS_PROBABILITY:
        # Convert to binary: 1 if data loss occurred, 0 otherwise
        return np.array(
            [1.0 if t is not None else 0.0 for t in results.time_to_actual_loss_samples]
        )
    elif metric == ConvergenceMetric.MEAN_TIME_TO_DATA_LOSS:
        # Only include runs where data loss actually occurred
        filtered = results.time_to_actual_loss_samples_filtered()
        return np.array(filtered) if filtered else np.array([])
    else:
        raise ValueError(f"Unknown convergence metric: {metric}")


def _check_convergence(
    results: MonteCarloResults, criteria: ConvergenceCriteria
) -> list[MetricConvergenceStatus]:
    """Check convergence for all target metrics.

    Supports two modes:
      - **Relative error**: converged when ci_half_width / |mean| <= threshold.
      - **Absolute error**: converged when ci_half_width <= threshold.

    For continuous metrics (availability, cost), uses the t-distribution
    to construct a confidence interval.

    For proportions (data_loss_probability), uses a Wald interval with
    the normal approximation.

    Args:
        results: Current Monte Carlo results.
        criteria: Convergence criteria.

    Returns:
        List of MetricConvergenceStatus for each target metric.
    """
    statuses = []
    alpha = 1.0 - criteria.confidence_level
    use_absolute = criteria.uses_absolute_error
    threshold = criteria.error_threshold

    for metric in criteria.metrics:
        samples = _get_metric_samples(results, metric)
        n = len(samples)

        if n < 2:
            statuses.append(
                MetricConvergenceStatus(
                    metric=metric,
                    converged=False,
                    num_samples=n,
                )
            )
            continue

        sample_mean = float(np.mean(samples))
        sample_std = float(np.std(samples, ddof=1))

        if metric == ConvergenceMetric.DATA_LOSS_PROBABILITY:
            p = sample_mean
            z = scipy_stats.norm.ppf(1 - alpha / 2)

            if p == 0.0 or p == 1.0:
                # With no variance, use the rule of three.
                # Upper bound on p is ~3/n at 95% confidence.
                if use_absolute:
                    rule_of_three_n = int(math.ceil(3.0 / threshold))
                else:
                    # For relative error on a zero proportion, we need
                    # enough runs to be confident p is truly near 0/1.
                    rule_of_three_n = int(math.ceil(3.0 / threshold))
                statuses.append(
                    MetricConvergenceStatus(
                        metric=metric,
                        converged=n >= rule_of_three_n,
                        current_mean=p,
                        current_std=0.0,
                        ci_half_width=3.0 / n if n > 0 else float("inf"),
                        relative_error=0.0 if p == 0.0 else float("inf"),
                        estimated_runs_needed=max(rule_of_three_n, n),
                        num_samples=n,
                    )
                )
                continue

            se = math.sqrt(p * (1 - p) / n)
            ci_half_width = z * se
            rel_err = ci_half_width / p if p > 0 else float("inf")

            if use_absolute:
                converged = ci_half_width <= threshold
                target_E = threshold
            else:
                converged = rel_err <= threshold
                target_E = threshold * p

            estimated_n = int(math.ceil(z**2 * p * (1 - p) / target_E**2))

            statuses.append(
                MetricConvergenceStatus(
                    metric=metric,
                    converged=converged,
                    current_mean=p,
                    current_std=se * math.sqrt(n),
                    ci_half_width=ci_half_width,
                    relative_error=rel_err,
                    estimated_runs_needed=max(estimated_n, n),
                    num_samples=n,
                )
            )
        else:
            # For continuous metrics, use the t-distribution CI.
            t_crit = scipy_stats.t.ppf(1 - alpha / 2, df=n - 1)
            se = sample_std / math.sqrt(n)
            ci_half_width = t_crit * se

            if sample_mean == 0.0:
                rel_err = float("inf") if sample_std > 0 else 0.0
            else:
                rel_err = ci_half_width / abs(sample_mean)

            if use_absolute:
                converged = ci_half_width <= threshold
                target_margin = threshold
            else:
                converged = rel_err <= threshold
                target_margin = (
                    threshold * abs(sample_mean) if abs(sample_mean) > 0 else 0.0
                )

            # Estimated n: n = (z * σ / target_margin)²
            z = scipy_stats.norm.ppf(1 - alpha / 2)
            if target_margin > 0:
                estimated_n = int(math.ceil((z * sample_std / target_margin) ** 2))
            else:
                estimated_n = criteria.max_runs

            # For MTTDL, estimated_n is the number of loss-event *samples*
            # needed.  Scale up by the inverse of the observed loss rate to
            # estimate total simulation runs required.
            if metric == ConvergenceMetric.MEAN_TIME_TO_DATA_LOSS:
                total_runs = len(results.availability_samples)
                if 0 < n < total_runs:
                    loss_rate = n / total_runs
                    estimated_n = int(math.ceil(estimated_n / loss_rate))

            statuses.append(
                MetricConvergenceStatus(
                    metric=metric,
                    converged=converged,
                    current_mean=sample_mean,
                    current_std=sample_std,
                    ci_half_width=ci_half_width,
                    relative_error=rel_err,
                    estimated_runs_needed=max(estimated_n, n),
                    num_samples=n,
                )
            )

    return statuses


def estimate_required_runs(
    pilot_results: MonteCarloResults,
    convergence: ConvergenceCriteria,
) -> dict[ConvergenceMetric, int]:
    """Estimate the number of runs needed for convergence from pilot results.

    Useful for planning: run a small pilot (e.g., 30 runs) and then call
    this function to see how many total runs are likely needed.

    Args:
        pilot_results: Results from a pilot run.
        convergence: Desired convergence criteria.

    Returns:
        Dictionary mapping each target metric to estimated required runs.
    """
    statuses = _check_convergence(pilot_results, convergence)
    return {s.metric: s.estimated_runs_needed for s in statuses}


def run_monte_carlo(
    cluster_factory: Callable[[], ClusterState],
    strategy_factory: Callable[[], ClusterStrategy],
    protocol: Protocol,
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
        protocol: Protocol instance (deep-copied for each run).
        num_simulations: Number of simulation runs.
        max_time: Maximum time per simulation in seconds. If None and
            stop_on_data_loss is True, runs continue until data loss
            (useful for MTTDL estimation).
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
        protocol=protocol,
        network_config=network_config,
    )


def run_monte_carlo_converged(
    cluster_factory: Callable[[], ClusterState],
    strategy_factory: Callable[[], ClusterStrategy],
    protocol: Protocol,
    max_time: Seconds | None = None,
    confidence_level: float = 0.95,
    relative_error: float | None = None,
    absolute_error: float | None = None,
    metrics: list[ConvergenceMetric] | None = None,
    network_config: NetworkConfig | None = None,
    stop_on_data_loss: bool = True,
    parallel_workers: int = 1,
    seed: int | None = None,
    min_runs: int = 30,
    max_runs: int = 10_000,
    batch_size: int = 10,
    progress_callback: Callable[[int, int, bool], None] | None = None,
) -> ConvergenceResult:
    """Convenience function to run adaptive Monte Carlo simulations.

    Automatically determines the number of runs needed to achieve the
    desired confidence level and error tolerance on the target metrics.

    Specify either relative_error or absolute_error (not both).
    If neither is given, defaults to relative_error=0.05.

    To get availability accurate to N decimal places, use absolute_error:
        2 decimal places on the fraction: absolute_error=0.01 (±0.01)
        3 decimal places: absolute_error=0.001 (±0.001)

    For MTTDL convergence, set stop_on_data_loss=True and max_time=None
    so that every run produces a data loss event and contributes to the
    MTTDL estimate.

    Args:
        cluster_factory: Callable that creates a fresh ClusterState for each run.
        strategy_factory: Callable that creates a fresh ClusterStrategy for each run.
        protocol: Protocol instance (deep-copied for each run).
        max_time: Maximum time per simulation in seconds. If None and
            stop_on_data_loss is True, runs continue until data loss.
        confidence_level: Desired confidence level (default 0.95 for 95% CI).
        relative_error: Maximum relative error as fraction of mean.
        absolute_error: Maximum absolute CI half-width in metric units.
        metrics: Which metrics to converge on (default: [AVAILABILITY]).
        network_config: Optional network configuration.
        stop_on_data_loss: Whether to stop on data loss.
        parallel_workers: Number of parallel workers.
        seed: Base random seed.
        min_runs: Minimum runs before checking convergence (default 30).
        max_runs: Maximum runs safety cap (default 10000).
        batch_size: Runs per batch between convergence checks (default 10).
        progress_callback: Optional callback(completed, estimated_total, converged).

    Returns:
        ConvergenceResult with aggregated statistics and convergence info.
    """
    if metrics is None:
        metrics = [ConvergenceMetric.AVAILABILITY]

    convergence = ConvergenceCriteria(
        confidence_level=confidence_level,
        relative_error=relative_error,
        absolute_error=absolute_error,
        metrics=metrics,
        min_runs=min_runs,
        max_runs=max_runs,
        batch_size=batch_size,
    )

    config = MonteCarloConfig(
        num_simulations=max_runs,  # Upper bound; adaptive will likely stop earlier
        max_time=max_time,
        stop_on_data_loss=stop_on_data_loss,
        parallel_workers=parallel_workers,
        base_seed=seed,
    )

    runner = MonteCarloRunner(config)
    return runner.run_until_converged(
        cluster_factory=cluster_factory,
        strategy_factory=strategy_factory,
        protocol=protocol,
        convergence=convergence,
        network_config=network_config,
        progress_callback=progress_callback,
    )
