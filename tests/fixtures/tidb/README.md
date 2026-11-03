# TiDB v8.5.8 fixtures (real output, not invented)

Captured 2026-11-03 (UTC, 00:46-01:15) from a throwaway local cluster so the parsers in
`powder/deploy/` (Region-map diffing, PD operator-lifecycle log parsing, metric names) are written
against real formats. **Single host, loopback, 4 TiKV stores**: timings are not representative of AWS,
formats are.

* TiDB / TiKV / PD `v8.5.8` (PD commit `4ecadf2f`, the commit pinned in `third_party/pd`), TiUP `v1.17.1`.
* Cluster: `tiup playground v8.5.8 --db 1 --pd 1 --kv 4 --tiflash 0 --tag powderfix --without-monitor --host 127.0.0.1`
  (`TIUP_HOME=~/.cache/powder/tiup-home`, binaries from `infra/tidb/fetch_tidb.sh`). PD defaults except
  `max-store-down-time=1m` (set at 00:59 for the store-down case). Replicas = 3, so with 4 stores a store
  failure leaves a place to re-replicate.
* Raw captures were taken with `curl 127.0.0.1:2379/pd/api/v1/regions` and `/stores` (the "snapshot"),
  `pd-ctl` from the pinned `ctl` tarball (`operator add|show|history`), and `curl` on `/metrics`.
  `make_fixtures.py` trims/compresses them into this directory (it needs the raw dir, which was deleted;
  you do not need to run it). Everything here is committed as captured except gzip + compact JSON.

## Files

| Path | What |
|---|---|
| `timeline.json` | Index of every snapshot (`name`, UTC time, Region count) and every command issued (`label`, UTC time, pd-ctl output). Use it to line snapshots up with events. |
| `regions/<name>.json.gz` | Full `GET /pd/api/v1/regions` body (`{"count":N,"regions":[...]}`), 75 snapshots, see names below. |
| `stores/{00-baseline,10-down-t15,10-down-t70}.json` | `GET /pd/api/v1/stores`: all Up; one `Disconnected`; one `Down`. |
| `pd_log_samples.log` | Raw PD log lines (unmodified), three `##` sections: startup splits + first operators; every lifecycle line for the manually driven Regions; the store-down window. |
| `operator_history.cli.json`, `operator_history.rest.json` | `pd-ctl operator history <from>` and `GET /pd/api/v1/operators/records?from=<unix>` at 01:14 (63 `replace-rule-down-peer` + 1 `remove-orphan-peer`; older records had already expired, see notes). |
| `operator_show_nonempty.json` | `pd-ctl operator show` taken immediately after an `operator add`, only the non-empty ones. |
| `pd_metrics_samples.txt`, `tikv_metrics_samples.txt` | `# TYPE` + sample lines of the relevant metrics (final state, 01:14). |
| `make_fixtures.py` | Assembler (provenance only). |

### Snapshot names (`regions/`)

`00-baseline` (65 Regions, all system) · `01-after-load` (table `fx.t`, 100k rows, 1 Region) ·
`02-split-t0|t6|t31` (`SPLIT TABLE t BETWEEN (0) AND (100000) REGIONS 8`, issued 00:49:26.2) ·
for each manual operator `<NN>-<op>-pre|t1|t8|t20` (before, +1 s, +8 s, +20 s):
`03-transfer-leader` (`transfer-leader 316 6`), `04-remove-peer` (`remove-peer 320 1`), `05-add-peer` (`add-peer 324 4`),
`06-transfer-peer` (`transfer-peer 328 4 6`), `07-merge` (`merge-region 332 336`), `08-split-region` (`split-region 27 --policy=approximate`),
`09-add-learner` (`add-learner 312 4`) · store-down: `10-down-pre|t5|...|t240` (`kill -9` of the TiKV for store 6 at
00:59:28.06; t = seconds after) and `12-late-00..NN` (every 20 s, 01:03 -> 01:14, covers the re-replication).

## Known ground truth in these fixtures

| Event | What the snapshots/logs show |
|---|---|
| Split (`02`) | Count 66 -> 73. New Region IDs `312 316 320 324 328 332 336`; **no ID disappeared**: the parent (27) keeps its ID and becomes the *rightmost* piece (its `start_key` moves right). The 7 children get new IDs for the left pieces. Log: one `region batch split, generate new regions` (`total=7`) + one `region Version changed` for the parent. |
| Leader transfer (`03`) | Region 316 leader store 5 -> 6; peers unchanged; `operator finish` `admin-transfer-leader {transfer leader: store 5 to 6}` takes 3.7 ms; log `leader changed [from=5] [to=6]`. No ConfVer change. |
| Remove peer (`04`) | 320 loses its store-1 peer (`admin-remove-peer`, 5.8 ms, ConfVer 29->30), then PD itself repairs: `add-rule-peer {add peer: store [1]}` created 55 ms later, 82 ms to finish (ConfVer ->32). Net peer set unchanged. |
| Add peer (`05`) | 324 gets a 4th voter on store 4 (`admin-add-peer`, 80 ms) and its `remove-orphan-peer` (PD cleaning up) finishes 26 ms later. The 4-peer state is visible in `05-add-peer-t1`. |
| Move peer (`06`) | 328: store 4 -> 6 (`admin-move-peer`, 86 ms) in 4 steps (see step grammar below). |
| Merge (`07`) | Count 73 -> 72. Source 332 **disappears**; target 336 **keeps its ID**, `start_key` extends left to 332's start, version 73 -> 75, and its peers are the **target's store set** (`[1,4,6]`, not the source's `[4,5,6]`). PD first moves the source's peers onto the target's stores. Two operators with the same description (`admin-merge-region {merge: region 332 to 336}`), one per Region; the source one ends as `operator canceled ... cancel-reason: related merge region`, the target one `operator finish`. ~113 ms. |
| Approximate split (`08`) | `admin-split-region` on Region 27 **timed out after 1 m** (`operator timeout`, `cancel-reason: timeout`): no split happened. A real `operator timeout` sample. |
| Add learner (`09`) | `admin-add-learner` finished, then its `remove-orphan-peer` finished ~100 ms later (no rule wants the learner). The `Learner` role was never visible in any snapshot (gone before the next 1 s sample). |
| Store down (`10`, `12`) | Store 6 held 63 of 72 Regions. `Disconnected` by +15 s, `Down` by +70 s (max-store-down-time 1m), `pending_peers` on 6 Regions at +5 s, 14 at +15 s, 16 from +30 s. **Re-replication did not start until 01:08:29.8 (kill + 9 m 02 s)**; `replace-rule-down-peer {mv peer: store [6] to [N]}` operators then moved all 63 peers by 01:10:34, each in ~50-125 ms, 4 steps. `down_peers` never appeared (see notes). |

## FORMAT NOTES

### Region JSON (`GET /pd/api/v1/regions`)

```json
{"count": 72, "regions": [ {
  "id": 316,
  "start_key": "7480000000000000FF725F728000000000FF0030D40000000000FA",   // uppercase hex, ''=-inf
  "end_key":   "7480000000000000FF725F728000000000FF0061A80000000000FA",   // ''=+inf
  "epoch": {"conf_ver": 17, "version": 73},
  "peers":  [{"role_name": "Voter", "id": 318, "store_id": 4}, ...],
  "leader": {"role_name": "Voter", "id": 319, "store_id": 6},
  "pending_peers": [{"role_name": "Voter", "id": 195, "store_id": 6}],      // OPTIONAL, see below
  "cpu_usage": 0, "written_bytes": 113, "read_bytes": 0, "written_keys": 2, "read_keys": 0,
  "approximate_size": 1, "approximate_keys": 0 } ]}
```

* Always present (3535 of 3535 sampled Regions): `id, start_key, end_key, epoch{conf_ver,version}, peers[], leader, cpu_usage, written_bytes, read_bytes, written_keys, read_keys, approximate_size, approximate_keys`. Peers and leader have exactly `role_name` (`"Voter"` in every sample), `id` (peer id), `store_id`.
* **`pending_peers` is absent when empty** (present on 467 samples, all during the store-down case). **`down_peers` was never present in any snapshot**, even while PD ran `replace-rule-down-peer`; `pd_regions_status{type="down-peer-region-count"}` also stayed 0. Do not use `down_peers`; use store state + `pending_peers` + the operator events.
* `approximate_size` is in MiB (integer, 1 for tiny Regions); `approximate_keys` is a count, often 0 for small Regions.
* **The list is not sorted** (hash order) and the Regions only tile the key space **after sorting by `start_key`** (then contiguous from `''` to `''`). Sorting hex strings lexicographically equals byte order. Sort before diffing.
* **Keys are the hex of the memcomparable-encoded key.** TiDB row key `t{table_id}_r{handle}` = `b't' + be64(table_id ^ 1<<63) + b'_r' + be64(handle ^ 1<<63)`, then memcomparable-encoded (8-byte groups, each followed by `0xFF - pad`). Verified exactly: row 12500 of table 114 encodes to Region 316's `start_key` above. The table prefix `7480000000000000FF72...` identifies table 114. Keys starting `72...`/`78...` are the keyspace-API-v2 boundary Regions created at bootstrap (`labeler-split-region`); keys `74...FF{N}00...F8` are per-table-id ranges.
* Splits keep the parent ID on the right piece; merges delete the source ID (see ground truth). Region IDs are the only stable identity across snapshots; a RSM identity is the **set of `peers[].store_id`** (voters), which changes via ConfVer.
* `epoch.version` increases on split/merge (merge: +2 on the target here), `epoch.conf_ver` on every peer change (add learner, promote, remove each count: a move-peer is +3..+4).

### Stores JSON (`/pd/api/v1/stores`)

`{"count":4,"stores":[{"store":{"id","address","version","peer_address","status_address","git_hash","start_timestamp","deploy_path","last_heartbeat"(ns),"node_state","state_name"},"status":{"capacity","available","used_size","leader_count","leader_weight","leader_score","leader_size","region_count","region_weight","region_score","region_size","slow_score","slow_trend",{...},"start_ts","last_heartbeat_ts","uptime"}}]}`. `state_name` observed: `Up`, `Disconnected`, `Down`. Sizes like `capacity` are human strings ("247.2GiB"), counts are ints.

### PD log grammar

`[<YYYY/MM/DD HH:MM:SS.mmm> +00:00] [LEVEL] [<file.go>:<line>] ["<message>"] [key=value]...`; string values are quoted with `\"` escapes, and the `operator=` value is quoted **twice** (`operator="\"...\""`). Lifecycle messages (message text is the stable key; file:line may move):

| Message | Fields | Meaning |
|---|---|---|
| `add operator` (operator_controller.go) | `region-id`, `operator`, `additional-info` (empty or JSON) | operator created and started (`createAt` ~ `startAt`; no separate "start" line). |
| `send schedule command` | `region-id`, `step="..."`, `source=create\|heartbeat` | a step was sent to the leader store; one per step, repeated. |
| `operator finish` | `region-id`, `takes=<Go duration>`, `operator="\"... finished\""`, `additional-info="{\"cancel-reason\":\"\",...}"` | success; `takes` is wall time. |
| `operator timeout` | same, operator string ends ` timeout`, `cancel-reason":"timeout"` | exceeded its `timeout:[..]`. |
| `operator canceled` | same, `takes`, `cancel-reason` in additional-info | seen reasons: `epoch not match`, `already exist`, `stale status`, `related merge region`, `timeout`. Often `takes=0s` and `startAt:0001-01-01` (cancelled before start). |
| `operator canceled related merge region` | `region-id`, `additional-info="{\"related merge region\":\"336\"}"`, `takes` | the source side of a merge. |
| `operator is stale` / `operator removed` | `error="[target store is down]"`; `takes` (46 s) | an operator invalidated by a store going down. |
| `leader changed` (region.go) | `region-id`, `from`, `to` (store ids) | |
| `region ConfVer changed` | `region-id`, `detail="Add peer:{id:71 store_id:4 role:Learner },Remove peer:{...}"`, `old-confver`, `new-confver` | detail is a comma-joined list of `Add peer:{...}` / `Remove peer:{...}`; a promotion appears as Remove-learner + Add-voter; joint-consensus roles `IncomingVoter`/`DemotingVoter` appear. |
| `region Version changed` | `region-id`, `detail="StartKey Changed:{<hex>} -> {<hex>}, EndKey:{<hex>}"`, `old-version`, `new-version` | split/merge boundary change. |
| `alloc ids for region split`, `region batch split, generate new regions` | `region-id` (parent), `origin="<protobuf text with all pieces>"`, `total` | split bookkeeping. |
| `send keepalive message fail, store maybe disconnected` (WARN) | `target-store-id`, `error=EOF` | first sign of a dead store (+25 s). |
| `audit log` | HTTP request info | noise: one per API call. |

**Operator string** (inside `operator=`, also each entry of `operator history`):
`<name> {<brief>} (kind:<k1[,k2]>, region:<id>(<version>, <conf_ver>), createAt:<ts>, startAt:<ts>, currentStep:<n>, size:<MiB>, steps:[<i>:{<step text>}, ...], timeout:[<dur>])` then a status suffix ` finished (finishAt:<ts>, duration:<dur>)` (history) or ` finished` / ` timeout` (log). `region:` is **(version, conf_ver)** in that order. Names seen: `balance-region-scheduler`, `balance-leader-scheduler`, `add-rule-peer`, `replace-rule-down-peer`, `remove-orphan-peer`, `fix-peer-role`, `scatter-region`, `labeler-split-region`, `admin-transfer-leader|remove-peer|add-peer|move-peer|add-learner|merge-region|split-region`. Kinds seen: `region`, `leader`, `replica,region`, `admin,region`, `admin,leader`, `admin,merge,region`, `admin,split`, `split`. Brief examples: `{transfer leader: store 5 to 6}`, `{mv peer: store [4] to [6]}`, `{add peer: store [1]}`, `{rm peer: store [1]}`, `{merge: region 332 to 336}`. Step texts: `add learner peer N on store N` · `promote learner peer N on store N to voter` · `use joint consensus, promote learner peer N on store N to voter, demote voter peer N on store N to learner` · `leave joint state, promote ...` · `remove peer on store N` · `transfer leader from store N to store N` · `merge region N into region N` · `split region with policy USEKEY|APPROXIMATE`. A **move-peer is 4 steps** (add learner, enter joint, leave joint, remove). Timestamps inside: `2026-11-03 00:47:22.88431359 +0000 UTC m=+30.3...` (Go time with monotonic suffix; variable fractional digits).

### Operator history / records

* `pd-ctl operator history <unix>` -> JSON array of strings (operator string + ` finished (finishAt:..., duration:...)`); `GET /pd/api/v1/operators/records?from=<unix>` -> JSON array of the same strings. `from` filters on **finish time**.
* **Lossy:** records live **10 minutes** (`operatorStatusRemainTime`) and are keyed **per Region ID**, so a newer operator on a Region overwrites its previous record. At 01:14 only records from 01:08 on were left; the 00:55-00:58 manual operators were gone. Poll at <= 1 min intervals or parse the PD log (lossless); do not rely on one read at the end of a run.
* `operator show` lists only running operators (`[]` almost always: operators finish in milliseconds).

### Metrics (exact names, v8.5.8; see the two `*_metrics_samples.txt`)

PD (`:2379/metrics`): `pd_schedule_operators_count{event="create|start|finish|cancel|timeout|check|promote-*|already-have|...",type="<operator name>"}` (counter), `pd_schedule_finish_operators_duration_seconds{type}` (histogram, `_sum/_count/_bucket`), `pd_schedule_finish_operator_steps_duration_seconds{type="AddLearner|ChangePeerV2Enter|...|RemovePeer|TransferLeader|MergeRegion|SplitRegion"}`, `pd_schedule_operator_region_size{type}` (histogram, MiB; `_sum` = MiB moved by operator type), `pd_cluster_status{engine,store,type="leader_count|learner_count|region_count|storage_size|storage_capacity|..."}`, `pd_scheduler_store_status{address,store,type=...}`, `pd_regions_status{type="miss-peer|extra-peer|down-peer|pending-peer|offline-peer|learner-peer|empty-region-count"}` (gauges: `<kind>-region-count`), `pd_regions_abnormal_peer_duration_seconds{type="down-peer|miss-voter-peer|..."}` (histogram; note it counted 126 `down-peer` episodes although `down-peer-region-count` stayed 0), `pd_cluster_region_event{event}`.
TiKV (`<status-port>/metrics`, ports 20180-20183): `tikv_raftstore_admin_cmd_total{status="all|success|...",type="add_peer|add_learner|batch-split|commit_merge|prepare_merge|rollback_merge|remove_peer|transfer_leader|conf_change|compact"}` (hyphen in `batch-split`, underscore elsewhere), `tikv_raftstore_region_count{type="region|leader|buckets"}`, `tikv_raftstore_snapshot_total{status,type="generate|apply"}`, `tikv_raftstore_snapshot_duration_seconds{type}`, `tikv_snapshot_size` (histogram: `_sum` = **bytes** of snapshots sent by that store, `_count` = number), `tikv_snapshot_limit_transport_bytes{type="send|recv|recv_v2"}` (counter bytes), `tikv_server_snapshot_task_total{type}`, `tikv_server_send_snapshot_duration_seconds`, `tikv_raftstore_snapshot_traffic_total{type="sending|receiving"}` (gauge of in-flight). `tikv_raftstore_snapshot_size_*` does **not** exist (use `tikv_snapshot_size`).

## Surprises worth coding around

1. `/regions` is unordered and `down_peers` never shows up; `pending_peers` is omitted when empty.
2. Detection of a dead store's replicas is slow by default: `Down` at +70 s (our 1 m setting) but replicas moved only at **+9 m 02 s**. TiKV's `raftstore.max-peer-down-duration` (default 10 m) gates it and is **not online-modifiable** (`POST /config` and `SET CONFIG` returned 200 and left it at 10m). The "node replica timeout" what-if therefore needs both PD `max-store-down-time` and that TiKV setting (restart/config at deploy time).
3. `operator history` is a 10-minute, one-record-per-Region ring. PD log lines are the lossless source.
4. Merge/split ID semantics (above); merges use the target's store set.
5. `pd-ctl` blocks forever if stdin is not closed in a non-tty (`< /dev/null`).
6. Operator "start" is not logged separately; "add operator" carries both.
7. PD does log some lines twice with identical text and timestamp: **`send schedule command`** (153 duplicated lines in the raw log, e.g. the `leave joint state` step is sent twice in the same millisecond). Dedupe those on (timestamp, region-id, step) before counting steps. (`pd_log_samples.log` keeps these raw duplicates; its sections do not overlap.)
8. `scatter-region` and `fix-peer-role` operators appear without being requested (SPLIT TABLE scatters; PD fixes roles).
