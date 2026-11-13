"""Pin TiDB's own data to dedicated, never-faulted "system" TiKV stores (G3 follow-up).

TiDB keeps its schema metadata (``m`` keys) and its ``mysql.*`` system tables in ordinary TiKV Regions, which PD
places like user data. Every TiDB server must reload the schema at least once per schema lease (45 s) or it refuses
all commits (``[domain:8027] Information schema is out of date``). At G3, rounds whose dead voter set held a system
Region (``tidb_mdl_info``, ``tidb_background_subtask``) stopped writes on every table, cluster-wide. Experiments must
measure the availability of the experiment's own RSMs, so TiDB's data goes on system stores that faults never touch:

* the default rule (group ``pd``) puts **everything** on stores labelled ``machine_class=system``, so system tables
  created at any time, the meta keys and the reserved-ID tables near 2^48 all land there;
* an overriding rule (group ``powder``) puts the user-table key range ``[t{first user table id}, t{2^47})`` on the
  other (experiment) stores.

User table ids are allocated after the system ones (the global id allocator only grows), so every table an
experiment creates after install falls in the user range. Keys are memcomparable-encoded, as in PD's Region
boundaries and placement rules.
"""

from __future__ import annotations

from .regionmap import RegionMap, _enc_int, mc_encode

SYSTEM_CLASS = "system"
RESERVED_ID_FLOOR = 1 << 47          # TiDB's reserved system-table ids sit just below 2^48; user ids never get here
# all tables (and partitions) that exist at install time are TiDB's: the first user id is the next one
MAX_SYSTEM_ID_SQL = ("SELECT GREATEST((SELECT MAX(TIDB_TABLE_ID) FROM information_schema.tables "
                     f"WHERE TIDB_TABLE_ID < {RESERVED_ID_FLOOR}), IFNULL((SELECT MAX(TIDB_PARTITION_ID) FROM "
                     f"information_schema.partitions WHERE TIDB_PARTITION_ID < {RESERVED_ID_FLOOR}), 0))")


def table_prefix(table_id: int) -> bytes:
    """Encoded key where table ``table_id``'s data starts (``t{id}``)."""
    return mc_encode(b"t" + _enc_int(table_id))


def user_range(first_user_id: int) -> tuple[bytes, bytes]:
    return table_prefix(first_user_id), table_prefix(RESERVED_ID_FLOOR)


def rule_bundles(first_user_id: int, replicas: int, location_labels: list[str],
                 data_isolation: str = "region") -> list[dict]:
    """The complete placement-rule set (``POST /pd/api/v1/config/placement-rule`` replaces all groups with it)."""
    start, end = user_range(first_user_id)
    # No isolation level on the system rule. When the rules are applied, every system Region still has all its
    # replicas on experiment stores in all regions, and PD's isolation filter counts those orphan peers: no system store
    # qualified, PD added nothing, and the orphans are only removed once the rule is met (first live run: 67 Regions
    # stuck for 15 min; dropping the isolation level moved all 66 in 10 s). With one system store per region the
    # replicas still end up one per region (location_labels spread them, and count == number of system stores).
    return [
        {"group_id": "pd", "group_index": 0, "group_override": False, "rules": [{
            "group_id": "pd", "id": "default", "index": 0, "start_key": "", "end_key": "", "role": "voter",
            "count": replicas, "location_labels": location_labels, "isolation_level": "",
            "label_constraints": [{"key": "machine_class", "op": "in", "values": [SYSTEM_CLASS]}]}]},
        {"group_id": "powder", "group_index": 10, "group_override": True, "rules": [{
            "group_id": "powder", "id": "experiment-data", "index": 0, "start_key": start.hex(), "end_key": end.hex(),
            "role": "voter", "count": replicas, "location_labels": location_labels, "isolation_level": data_isolation,
            "label_constraints": [{"key": "machine_class", "op": "notIn", "values": [SYSTEM_CLASS]}]}]},
    ]


def placement_problems(m: RegionMap, first_user_id: int, replicas: int) -> list[str]:
    """Regions not (yet) where the rules want them: user-range Regions must have all voters on experiment stores,
    every other Region all voters on system stores; a Region straddling the boundary is not split yet."""
    start, end = user_range(first_user_id)
    system = {sid for sid, s in m.stores.items() if s.machine_class == SYSTEM_CLASS}
    if not system:
        return ["no store is labelled machine_class=system"]
    p = []
    for r in m.regions:
        r_end = r.end or b"\xff" * 64
        inside = start <= r.start and r_end <= end
        outside = r_end <= start or r.start >= end
        if not (inside or outside):
            p.append(f"Region {r.id} straddles the user-range boundary")
            continue
        on_sys = [s for s in r.voters if s in system]
        if inside and on_sys:
            p.append(f"user Region {r.id} has voters on system stores {on_sys}")
        if outside and len(on_sys) != len(r.voters):
            p.append(f"system Region {r.id} has voters off the system stores: {sorted(set(r.voters) - system)}")
        if len(r.voters) != replicas:
            p.append(f"Region {r.id} has {len(r.voters)} voters, want {replicas}")
    return p
