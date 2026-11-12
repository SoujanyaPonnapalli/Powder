"""Compose request-success masks on one *shared* fault trajectory.

A block represents possible request start times of the given duration, after
accounting for the request deadline. ``global_ok`` means PD/TSO, schema and
frontend satisfy *every required request class*. ``local_ok`` means the named
shard's candidate satisfies all required request classes on the SAME trace.
It may already include global gates (as Rust joint replay does); in that case
candidate success must imply ``global_ok``. The masks must not come from
independently sampled per-RSM Monte Carlo runs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

Placement = tuple[str, ...]


@dataclass(frozen=True)
class ComposedResult:
    per_shard: dict[str, float]
    single_access: float
    full_access: float
    global_access: float
    horizon_seconds: float


@dataclass(frozen=True)
class CandidateTimeline:
    """Request-start intervals and candidate outcomes from a shared event trace.

    ``local_ok[shard_id][placement]`` has one bool per interval. It can be
    different for two shards on the same placement if shard-specific events
    affect them. The global mask is counted only once for full access.
    """

    seconds: tuple[float, ...]
    global_ok: tuple[bool, ...]
    local_ok: Mapping[str, Mapping[Placement, tuple[bool, ...]]]

    def __post_init__(self) -> None:
        if not self.seconds or len(self.seconds) != len(self.global_ok):
            raise ValueError("nonempty seconds and global_ok must have equal lengths")
        if (any(not math.isfinite(s) or s <= 0 for s in self.seconds)
                or not math.isfinite(sum(self.seconds))):
            raise ValueError("interval seconds and their sum must be positive and finite")
        if any(type(v) is not bool for v in self.global_ok):
            raise ValueError("global_ok values must be booleans")
        for shard_id, candidates in self.local_ok.items():
            if not isinstance(shard_id, str) or not shard_id or not candidates:
                raise ValueError("shards need a nonempty ID and candidate menu")
            for placement, mask in candidates.items():
                if not placement or len(set(placement)) != len(placement):
                    raise ValueError("candidate placement needs distinct machine IDs")
                if len(mask) != len(self.seconds) or any(type(v) is not bool for v in mask):
                    raise ValueError("local outcome mask must have one boolean per interval")

    @property
    def horizon_seconds(self) -> float:
        return sum(self.seconds)

    def candidate_access(self, shard_id: str, placement: Placement) -> float:
        """End-to-end success fraction for this one shard and placement."""
        local = self.local_ok[shard_id][placement]
        return sum(dt for dt, g, ok in zip(self.seconds, self.global_ok, local) if g and ok) / self.horizon_seconds

    def compose(self, placements: Mapping[str, Placement],
                shard_weights: Mapping[str, float] | None = None) -> ComposedResult:
        """Actual single and full access for one named deployment on this trace."""
        if not placements or set(placements) != set(self.local_ok):
            raise ValueError("placements must name every and only modeled shard")
        if shard_weights is None:
            weights = {shard: 1.0 / len(placements) for shard in placements}
        else:
            if set(shard_weights) != set(placements) or any(not math.isfinite(v) or v < 0
                                                            for v in shard_weights.values()):
                raise ValueError("shard weights must be finite, nonnegative and cover every shard")
            total = sum(shard_weights.values())
            if not math.isclose(total, 1.0, abs_tol=1e-9, rel_tol=0):
                raise ValueError("shard weights must sum to one")
            weights = shard_weights
        per_shard = {shard: self.candidate_access(shard, placement)
                     for shard, placement in placements.items()}
        global_seconds = sum(dt for dt, g in zip(self.seconds, self.global_ok) if g)
        full_seconds = sum(
            dt for k, (dt, g) in enumerate(zip(self.seconds, self.global_ok))
            if g and all(self.local_ok[shard][placement][k] for shard, placement in placements.items())
        )
        horizon = self.horizon_seconds
        return ComposedResult(per_shard=per_shard,
                              single_access=sum(weights[i] * a for i, a in per_shard.items()),
                              full_access=full_seconds / horizon,
                              global_access=global_seconds / horizon,
                              horizon_seconds=horizon)


def merge_timelines(traces: Sequence[CandidateTimeline]) -> CandidateTimeline:
    """Concatenate independent *whole-fleet runs*, never independently drawn shard runs."""
    if not traces:
        raise ValueError("at least one whole-fleet trace is required")
    shards = set(traces[0].local_ok)
    menus = {i: set(candidates) for i, candidates in traces[0].local_ok.items()}
    if any(set(t.local_ok) != shards or any(set(t.local_ok[i]) != menus[i] for i in shards) for t in traces):
        raise ValueError("all traces must use the same shard IDs and candidate menus")
    return CandidateTimeline(
        seconds=tuple(s for t in traces for s in t.seconds),
        global_ok=tuple(ok for t in traces for ok in t.global_ok),
        local_ok={i: {p: tuple(ok for t in traces for ok in t.local_ok[i][p]) for p in menus[i]}
                  for i in shards},
    )
