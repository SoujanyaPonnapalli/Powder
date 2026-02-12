"""
Time units and probability distributions for the Monte Carlo simulator.

All time values use seconds as the canonical unit. Helper functions provide
readable constructors for common time units.
"""

import math
from abc import ABC, abstractmethod
from typing import NewType

import numpy as np

# Explicit time unit - all times are in seconds
Seconds = NewType("Seconds", float)


def hours(h: float) -> Seconds:
    """Convert hours to seconds."""
    return Seconds(h * 3600)


def days(d: float) -> Seconds:
    """Convert days to seconds."""
    return Seconds(d * 86400)


def minutes(m: float) -> Seconds:
    """Convert minutes to seconds."""
    return Seconds(m * 60)


class Distribution(ABC):
    """Abstract base class for probability distributions."""

    @abstractmethod
    def sample(self, rng: np.random.Generator) -> float:
        """Sample a value from the distribution.

        Args:
            rng: NumPy random number generator for reproducibility.

        Returns:
            A sampled value from the distribution.
        """
        pass

    @property
    @abstractmethod
    def mean(self) -> float:
        """Theoretical mean (expected value) of the distribution.

        Used for upfront estimation when planning sync paths
        (log-only vs snapshot), where sampling is not appropriate
        because the decision must be made before the sync starts.
        """
        pass


class Exponential(Distribution):
    """Exponential distribution parameterized by rate.

    The exponential distribution is memoryless, making it suitable for
    modeling time until random events like failures.

    Args:
        rate: Events per unit time (1/mean). Must be positive.
    """

    def __init__(self, rate: float):
        if rate <= 0:
            raise ValueError(f"Rate must be positive, got {rate}")
        self.rate = rate

    @property
    def mean(self) -> float:
        return 1.0 / self.rate

    def sample(self, rng: np.random.Generator) -> float:
        """Sample time until next event."""
        # NumPy's exponential takes scale = 1/rate
        return rng.exponential(1.0 / self.rate)

    def __repr__(self) -> str:
        return f"Exponential(rate={self.rate})"


class Weibull(Distribution):
    """Weibull distribution for modeling failure times.

    The Weibull distribution generalizes the exponential distribution and can
    model increasing (shape > 1) or decreasing (shape < 1) failure rates.

    Args:
        shape: Shape parameter (k). Must be positive.
               k < 1: decreasing failure rate (infant mortality)
               k = 1: constant failure rate (exponential)
               k > 1: increasing failure rate (wear-out)
        scale: Scale parameter (λ). Must be positive.
    """

    def __init__(self, shape: float, scale: float):
        if shape <= 0:
            raise ValueError(f"Shape must be positive, got {shape}")
        if scale <= 0:
            raise ValueError(f"Scale must be positive, got {scale}")
        self.shape = shape
        self.scale = scale

    @property
    def mean(self) -> float:
        return self.scale * math.gamma(1.0 + 1.0 / self.shape)

    def sample(self, rng: np.random.Generator) -> float:
        """Sample a value from the Weibull distribution."""
        return self.scale * rng.weibull(self.shape)

    def __repr__(self) -> str:
        return f"Weibull(shape={self.shape}, scale={self.scale})"


class Normal(Distribution):
    """Normal (Gaussian) distribution with optional lower bound.

    Args:
        mean: Mean of the distribution.
        std: Standard deviation. Must be positive.
        min_val: Minimum value (samples below this are clamped). Defaults to 0.
    """

    def __init__(self, mean: float, std: float, min_val: float = 0.0):
        if std <= 0:
            raise ValueError(f"Standard deviation must be positive, got {std}")
        self._mean = mean
        self.std = std
        self.min_val = min_val

    @property
    def mean(self) -> float:
        return self._mean

    def sample(self, rng: np.random.Generator) -> float:
        """Sample a value, clamped to min_val."""
        value = rng.normal(self._mean, self.std)
        return max(self.min_val, value)

    def __repr__(self) -> str:
        return f"Normal(mean={self._mean}, std={self.std}, min_val={self.min_val})"


class Uniform(Distribution):
    """Uniform distribution over [low, high].

    Args:
        low: Lower bound (inclusive).
        high: Upper bound (exclusive).
    """

    def __init__(self, low: float, high: float):
        if low >= high:
            raise ValueError(f"Low must be less than high, got low={low}, high={high}")
        self.low = low
        self.high = high

    @property
    def mean(self) -> float:
        return (self.low + self.high) / 2.0

    def sample(self, rng: np.random.Generator) -> float:
        """Sample a value uniformly from [low, high)."""
        return rng.uniform(self.low, self.high)

    def __repr__(self) -> str:
        return f"Uniform(low={self.low}, high={self.high})"


class Constant(Distribution):
    """Constant (deterministic) distribution.

    Always returns the same value. Useful for testing or when a parameter
    is known exactly.

    Args:
        value: The constant value to return.
    """

    def __init__(self, value: float):
        self.value = value

    @property
    def mean(self) -> float:
        return self.value

    def sample(self, rng: np.random.Generator) -> float:
        """Return the constant value."""
        return self.value

    def __repr__(self) -> str:
        return f"Constant(value={self.value})"
