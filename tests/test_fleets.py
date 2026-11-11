"""Budget-targeted fleet candidates (powder/faults/fleets.py). NumPy + SciPy only, no Rust MC."""

from __future__ import annotations

import random
from collections import Counter

import numpy as np
import pytest

from powder.faults import fleets as F
from powder.faults.centroid_table import CentroidTable

REGIONS = ("us-east-1", "us-east-2", "us-west-1", "us-west-2")


def synthetic(n=240, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        g = REGIONS[i % 4]
        out.append(F.MachineFeatures(f"m{i:04d}", g, f"{g}:az{(i // 4) % 3 + 1}", float(rng.uniform(0.05, 1.0)),
                                     float(rng.lognormal(-7, 0.5)), float(rng.lognormal(-10, 1.5)),
                                     float(rng.uniform(600, 1800))))
    return out


def test_region_patterns_have_no_majority():
    assert F.region_patterns(3) == [(1, 1, 1)]
    assert set(F.region_patterns(5)) == {(2, 2, 1), (2, 1, 1, 1)}
    assert set(F.region_patterns(7)) == {(3, 3, 1), (3, 2, 2), (3, 2, 1, 1), (2, 2, 2, 1)}


def test_balanced_mix_fits_slots_and_is_balanced():
    mix = F.balanced_mix(50, 5 * 70, (3, 5, 7))
    assert sum(mix.values()) == 50 and sum(r * c for r, c in mix.items()) <= 350
    assert max(mix.values()) - min(mix.values()) <= 1        # 17/17/16 fits 5*70 slots
    small = F.balanced_mix(50, 5 * 30, (3, 5, 7))
    assert small == {3: 50, 5: 0, 7: 0}                       # 150 slots only fit all-RF3
    with pytest.raises(ValueError):
        F.balanced_mix(50, 100, (3, 5, 7))
    assert sum(F.split_passes({3: 50, 5: 0, 7: 0}, 6).values()) >= 6


def test_knapsack_fleets_respect_budget_size_and_spread():
    feats = synthetic()
    u = np.array([f.lam_t + f.lam_p for f in feats])
    fleets = F.fleets_for_budget(feats, u, 30.0, num_rsms=50, capacity=5, sizes=3, repeats=2)
    assert fleets
    price = {f.machine_id: f.price for f in feats}
    az = {f.machine_id: f.az for f in feats}
    for fl in fleets:
        assert 27.0 - 1e-6 <= sum(price[m] for m in fl) <= 30.0 + 1e-6
        assert len(fl) >= 30
        assert max(Counter(az[m] for m in fl).values()) <= len(fl) / 12 + 2 + 1e-9
    same_size = [fl for fl in fleets if len(fl) == len(fleets[0])]
    if len(same_size) > 1:
        assert len(same_size[0] & same_size[1]) <= 0.7 * len(same_size[0])
    assert not F.fleets_for_budget(feats, u, 1.0, num_rsms=50, capacity=5)   # too small for 30 machines


def test_pack_pass_keeps_slots_and_placement_rules():
    feats = synthetic()
    by = {f.machine_id: f for f in feats}
    fleet = feats[:72]
    u = {f.machine_id: f.lam_t for f in feats}
    for r in (3, 5, 7):
        for noise, near in ((0, 0), (0.5, 0.2)):
            pl = F.pack_pass(fleet, r, 50, 5, u, random.Random(1), noise=noise, near=near)
            assert pl and all(len(p) == r for p in pl)
            use = Counter(m for p in pl for m in p)
            assert max(use.values()) <= 5
            for p in pl:
                assert len({by[m].az for m in p}) == r
                assert max(Counter(by[m].region for m in p).values()) <= r // 2
    full = F.pack_pass(fleet, 3, 50, 5, u, random.Random(0))
    assert len(full) == 50                                   # deterministic covering fills N


def test_dominate_never_worsens_price_or_rates():
    feats = synthetic()
    by = {f.machine_id: f for f in feats}
    u = {f.machine_id: f.lam_t + f.lam_p for f in feats}
    fleet = feats[:72]
    for p in F.pack_pass(fleet, 5, 20, 5, u, random.Random(3), noise=0.5, near=0.5):
        q = F.dominate(p, fleet, u)
        assert len(q) == len(p) and len({by[m].az for m in q}) == len(q)
        assert max(Counter(by[m].region for m in q).values()) <= len(q) // 2
        assert sum(by[m].price for m in q) <= sum(by[m].price for m in p) + 1e-12
        for old in set(p) - set(q):
            assert any(by[new].dominates(by[old]) for new in set(q) - set(p))


def test_kmeans_and_table_keys():
    feats = synthetic()
    labels, cent = F.kmeans_log(feats, k=16, seed=0)
    assert len(cent) == 16 and set(labels) <= set(range(16))
    med = F.medoids(feats, labels, cent)
    assert all(labels[i] == j for j, i in enumerate(med))
    assert CentroidTable.key([3, 1, 2], (1, 1, 1)) == CentroidTable.key([1, 2, 3], (1, 1, 1))


def test_swap_neighbors_keep_azs_and_improve_never_worsens():
    feats = synthetic()
    by = {f.machine_id: f for f in feats}
    fleet = feats[:72]
    u = {f.machine_id: f.lam_t for f in feats}
    p = F.pack_pass(fleet, 5, 1, 5, u, random.Random(2), noise=0.5, near=0.5)[0]
    nbrs = F.swap_neighbors(p, fleet)
    assert nbrs and all(sorted(by[m].az for m in n) == sorted(by[m].az for m in p) for n in nbrs)
    score = lambda ids: -sum(by[m].lam_t for m in ids)  # noqa: E731
    q = F.improve(p, fleet, score)
    assert score(q) >= score(p)


def test_lattice_slice_elements_dominate_the_base_and_stay_valid():
    feats = synthetic()
    by = {f.machine_id: f for f in feats}
    fleet = feats[:120]
    u = {f.machine_id: f.lam_t + f.lam_p for f in feats}
    total = 0
    for p in F.pack_pass(fleet, 5, 10, 5, u, random.Random(4), noise=0.5, near=0.5):
        elems, top = F.lattice_slice(p, fleet, u, random.Random(0))
        assert elems[tuple(sorted(p))] == p and tuple(sorted(top)) in elems
        for pl, slots in elems.items():
            assert tuple(sorted(slots)) == pl and F.slot_leq(p, slots, by)
            assert len({by[m].az for m in pl}) == 5 and max(Counter(by[m].region for m in pl).values()) <= 2
            assert sum(by[m].price for m in pl) <= sum(by[m].price for m in p) + 1e-12
        total += len(elems)
    assert total > 10


def test_lattice_estimate_is_bounded_monotone_and_respects_known_scores():
    feats = synthetic()
    by = {f.machine_id: f for f in feats}
    fleet = feats[:120]
    u = {f.machine_id: f.lam_t + f.lam_p for f in feats}
    for p in F.pack_pass(fleet, 3, 20, 5, u, random.Random(5), noise=0.5, near=0.5):
        elems, top = F.lattice_slice(p, fleet, u, random.Random(1))
        if top == p:
            continue
        est = {pl: F.lattice_estimate(s, p, top, 0.9, 0.99, u, by) for pl, s in elems.items()}
        assert all(0.9 <= v <= 0.99 for v in est.values())
        assert est[tuple(sorted(p))] == 0.9 and est[tuple(sorted(top))] == 0.99
        for s in elems.values():                    # below the top: no higher than the top's estimate
            if F.slot_leq(s, top, by):
                assert F.lattice_estimate(s, p, top, 0.9, 0.99, u, by) <= 0.99
        single = next((s for pl, s in elems.items()
                       if sum(a != b for a, b in zip(s, p)) == 1 and F.slot_leq(s, top, by)), None)
        if single is not None:                      # an exact score below raises the floor of what is above
            assert F.lattice_estimate(top, p, top, 0.9, 0.99, u, by, known=[(single, 0.995)]) >= 0.995
        return


def test_spare_placements_use_only_spare_machines_and_stay_valid():
    feats = synthetic()
    by = {f.machine_id: f for f in feats}
    spare = {f.machine_id: 1 for f in feats[:24]}
    spare[feats[0].machine_id] = 0
    pls = F.spare_placements(spare, by, random.Random(0))
    assert pls and all(len(p) in (3, 5, 7) for p in pls)
    for p in pls:
        assert all(spare[m] > 0 for m in p)
        assert len({by[m].az for m in p}) == len(p)
        assert max(Counter(by[m].region for m in p).values()) <= len(p) // 2
    assert F.spare_placements({feats[0].machine_id: 2, feats[1].machine_id: 2}, by, random.Random(0)) == []


def test_improve_many_matches_improve_and_prefetches_each_step():
    feats = synthetic()
    by = {f.machine_id: f for f in feats}
    fleet = feats[:72]
    u = {f.machine_id: f.lam_t for f in feats}
    score = lambda ids: -sum(by[m].lam_t for m in ids)  # noqa: E731
    pls = F.pack_pass(fleet, 5, 8, 5, u, random.Random(2), noise=0.5, near=0.5)
    batches = []
    got = F.improve_many([(p, fleet) for p in pls], score, prefetch=lambda s: batches.append(len(s)))
    assert got == [F.improve(p, fleet, score) for p in pls]
    assert 1 <= len(batches) <= 2 and batches[0] > len(pls)
