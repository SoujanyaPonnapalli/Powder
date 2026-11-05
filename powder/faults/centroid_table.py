"""Lazily memoized MC availability of centroid placements (docs/plans/ilp-fleet-sampling.md step 6).

Key = (sorted cluster ids, replicas-per-region pattern). Each member is its cluster's medoid machine,
moved to a canonical region/AZ slot of the pattern, so region/AZ outages act as on a real placement of
that shape. Every entry uses the same seed (common random numbers): neighbouring entries see the same
fault luck and compare with much less noise than their absolute error. Only ranks placements; the ILP
sees exact MC of the real machines.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from .mc_scoring import node_config_batch_sampler


class CentroidTable:
    def __init__(self, medoid_configs: Sequence, labels: dict[str, int], regions: Sequence[str],
                 region_of: dict[str, str], protocol, strategy, *, horizon_s: float, network=None,
                 runs: int = 10_000, seed: int = 12345, cache_path: Path | None = None):
        self.medoids = list(medoid_configs)
        self.labels = labels
        self.regions = list(regions)
        self.region_of = region_of
        self.protocol, self.strategy = protocol, strategy
        self.horizon_s, self.network = horizon_s, network
        self.runs, self.seed = runs, seed
        self.cache_path = cache_path
        self.memo: dict[str, float] = {}
        if cache_path and cache_path.exists():
            self.memo = json.loads(cache_path.read_text())
        self.computed = 0

    @staticmethod
    def key(cluster_ids: Sequence[int], pattern: Sequence[int]) -> str:
        return f"{','.join(map(str, sorted(cluster_ids)))}|{','.join(map(str, pattern))}"

    def pattern_of(self, ids: Sequence[str]) -> tuple[int, ...]:
        return tuple(sorted(Counter(self.region_of[m] for m in ids).values(), reverse=True))

    def availability(self, cluster_ids: Sequence[int], pattern: Sequence[int]) -> float:
        k = self.key(cluster_ids, pattern)
        if k not in self.memo:
            cids = sorted(cluster_ids)
            cfgs, i = [], 0
            for g, n in zip(self.regions, pattern):
                for z in range(n):
                    cfgs.append(replace(self.medoids[cids[i]], region=g, az=f"{g}:az{z + 1}"))
                    i += 1
            sampler = node_config_batch_sampler(cfgs, self.protocol, self.strategy,
                                                horizon_s=self.horizon_s, network=self.network)
            self.memo[k] = sampler(self.runs, self.seed).mean
            self.computed += 1
        return self.memo[k]

    def score(self, ids: Sequence[str]) -> float:
        """Estimated availability of a placement of real machines."""
        return self.availability([self.labels[m] for m in ids], self.pattern_of(ids))

    def save(self) -> None:
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self.memo, sort_keys=True))
