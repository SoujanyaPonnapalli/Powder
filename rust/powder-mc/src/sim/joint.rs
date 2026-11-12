//! Opt-in deterministic whole-fleet request-start replay. This is a dependency
//! model, not a TiDB protocol simulation: no Raft logs, network partitions,
//! repair, retries, latency distributions or stochastic fault generation.
//! A request completes when all of its required gates are simultaneously up
//! at any instant within its deadline; zero service time is assumed.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Host {
    pub region: String,
    pub az: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FaultEvent {
    pub time: f64,
    /// One of host, region, az, global. A global event has no target.
    pub scope: String,
    #[serde(default)]
    pub target: Option<String>,
    pub down: bool,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JointJob {
    pub mode: String,
    #[serde(default)]
    pub job_id: Option<String>,
    pub horizon_seconds: f64,
    pub deadline_seconds: f64,
    pub schema_lease_seconds: f64,
    pub pd_election_seconds: f64,
    pub hosts: BTreeMap<String, Host>,
    pub pd_members: Vec<String>,
    pub pd_leader: String,
    /// Frontend ID -> physical host; a named frontend is selected for requests.
    pub frontends: BTreeMap<String, String>,
    pub frontend: String,
    /// Each inner list is a system-metadata Region's physical replica hosts.
    pub system_regions: Vec<Vec<String>>,
    /// Shard ID -> menu of candidate placements (each a list of replica hosts).
    pub shards: BTreeMap<String, Vec<Vec<String>>>,
    /// Shard ID -> index of selected candidate, used for the reported fleet SLA.
    pub selected: BTreeMap<String, usize>,
    /// A single, deterministic whole-fleet fault trace. Events at the same
    /// time must affect different scope/target pairs.
    #[serde(default)]
    pub events: Vec<FaultEvent>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CandidateMask {
    pub machine_ids: Vec<String>,
    /// End-to-end all-class success on this candidate, including global gates.
    pub local_ok: Vec<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct JointTimeline {
    /// Durations over eligible request-start times, NOT outage durations.
    pub seconds: Vec<f64>,
    /// All-class global-only request success, without user-Region gates.
    pub global_ok: Vec<bool>,
    /// `local_ok` is end-to-end (not independent local-only success), because
    /// global and user gates may recover at disjoint times inside a deadline.
    pub candidates: BTreeMap<String, Vec<CandidateMask>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct JointRecord {
    pub model: String,
    pub horizon_seconds: f64,
    pub eligible_start_seconds: f64,
    pub censored_seconds: f64,
    pub per_shard: BTreeMap<String, f64>,
    pub class_success: BTreeMap<String, BTreeMap<String, f64>>,
    pub single_access: f64,
    pub full_access: f64,
    pub global_access: f64,
    pub timeline: JointTimeline,
}

impl JointJob {
    pub fn validate(&self) -> Result<(), String> {
        if self.mode != "joint" {
            return Err("joint job requires mode=joint".into());
        }
        let finite_pos = |n: f64| n.is_finite() && n > 0.0;
        if !finite_pos(self.horizon_seconds)
            || !finite_pos(self.deadline_seconds)
            || self.deadline_seconds >= self.horizon_seconds
            || !finite_pos(self.schema_lease_seconds)
            || !self.pd_election_seconds.is_finite()
            || self.pd_election_seconds < 0.0
        {
            return Err("horizon, deadline and schema lease must be finite positive; deadline < horizon; PD election finite nonnegative".into());
        }
        if self.hosts.is_empty()
            || self
                .hosts
                .iter()
                .any(|(id, h)| id.is_empty() || h.region.is_empty() || h.az.is_empty())
        {
            return Err("hosts require nonempty IDs, regions and AZs".into());
        }
        let check = |ids: &[String]| -> Result<(), String> {
            if ids.is_empty()
                || ids.iter().any(|id| !self.hosts.contains_key(id))
                || ids.iter().collect::<BTreeSet<_>>().len() != ids.len()
            {
                return Err("replica sets must be nonempty, distinct known physical hosts".into());
            }
            Ok(())
        };
        check(&self.pd_members)?;
        if !self.pd_members.contains(&self.pd_leader) {
            return Err("PD leader must be a PD member".into());
        }
        if self.frontends.is_empty()
            || !self.frontends.contains_key(&self.frontend)
            || self
                .frontends
                .iter()
                .any(|(id, host)| id.is_empty() || !self.hosts.contains_key(host))
        {
            return Err(
                "frontends must map nonempty IDs to known hosts and select one frontend".into(),
            );
        }
        if self.system_regions.is_empty() {
            return Err("at least one metadata Region is required".into());
        }
        for r in &self.system_regions {
            check(r)?;
        }
        if self.shards.is_empty()
            || self.shards.keys().collect::<BTreeSet<_>>()
                != self.selected.keys().collect::<BTreeSet<_>>()
        {
            return Err("selected must name every and only modeled shard".into());
        }
        for (id, menu) in &self.shards {
            if id.is_empty() || menu.is_empty() || self.selected[id] >= menu.len() {
                return Err(
                    "shards require nonempty IDs, candidate menus and valid selected indices"
                        .into(),
                );
            }
            for candidate in menu {
                check(candidate)?;
            }
            if menu.iter().collect::<BTreeSet<_>>().len() != menu.len() {
                return Err("duplicate candidate placement".into());
            }
        }
        let mut seen = BTreeSet::new();
        for e in &self.events {
            if !e.time.is_finite() || e.time < 0.0 || e.time > self.horizon_seconds {
                return Err("fault event time outside horizon".into());
            }
            let valid = match (e.scope.as_str(), e.target.as_deref()) {
                ("host", Some(id)) => self.hosts.contains_key(id),
                ("region", Some(id)) => self.hosts.values().any(|h| h.region == id),
                ("az", Some(id)) => self.hosts.values().any(|h| h.az == id),
                ("global", None) => true,
                _ => false,
            };
            if !valid {
                return Err("invalid fault scope or target".into());
            }
            if !seen.insert((e.time.to_bits(), e.scope.clone(), e.target.clone())) {
                return Err("duplicate event at same time and scope/target".into());
            }
        }
        Ok(())
    }
}

#[derive(Default, Clone)]
struct Faults {
    host: BTreeSet<String>,
    region: BTreeSet<String>,
    az: BTreeSet<String>,
    global: bool,
}
impl Faults {
    fn apply(&mut self, e: &FaultEvent) {
        let set = match e.scope.as_str() {
            "host" => &mut self.host,
            "region" => &mut self.region,
            "az" => &mut self.az,
            _ => {
                self.global = e.down;
                return;
            }
        };
        let id = e.target.as_ref().expect("validated target");
        if e.down {
            set.insert(id.clone());
        } else {
            set.remove(id);
        }
    }
    fn up(&self, job: &JointJob, id: &str) -> bool {
        let h = &job.hosts[id];
        !self.global
            && !self.host.contains(id)
            && !self.region.contains(&h.region)
            && !self.az.contains(&h.az)
    }
}
fn quorum(ids: &[String], faults: &Faults, job: &JointJob) -> bool {
    ids.iter().filter(|id| faults.up(job, id)).count() > ids.len() / 2
}

#[derive(Clone)]
struct State {
    faults: Faults,
    pd: bool,
    schema: bool,
    frontend: bool,
}

/// Build the state-change timeline. Refresh is instantaneous while metadata
/// quorums and the frontend are healthy. Loss of the current PD leader starts
/// one deterministic election; quorum loss never elects a leader by itself.
fn states(job: &JointJob) -> Vec<(f64, State)> {
    let mut events: Vec<&FaultEvent> = job.events.iter().collect();
    events.sort_by(|a, b| a.time.total_cmp(&b.time));
    let mut faults = Faults::default();
    let mut leader = Some(job.pd_leader.clone());
    let mut election_at: Option<f64> = None;
    let mut lease_until: BTreeMap<String, f64> = job
        .frontends
        .keys()
        .map(|id| (id.clone(), job.schema_lease_seconds))
        .collect();
    let mut refreshable: BTreeMap<String, bool> =
        job.frontends.keys().map(|id| (id.clone(), true)).collect();
    let mut out = Vec::new();
    let mut ix = 0;
    let mut time = 0.0;
    loop {
        // The last refresh before an outage is at its boundary, per frontend.
        for (id, enabled) in &refreshable {
            if *enabled {
                lease_until.insert(id.clone(), time + job.schema_lease_seconds);
            }
        }
        while ix < events.len() && events[ix].time == time {
            faults.apply(events[ix]);
            ix += 1;
        }
        let pd_quorum = quorum(&job.pd_members, &faults, job);
        if leader.as_ref().is_some_and(|id| !faults.up(job, id)) {
            leader = None;
            election_at = Some(time + job.pd_election_seconds);
        }
        if leader.is_none() && pd_quorum && election_at.is_some_and(|at| at <= time) {
            leader = job.pd_members.iter().find(|id| faults.up(job, id)).cloned();
            election_at = None;
        }
        let metadata = job.system_regions.iter().all(|r| quorum(r, &faults, job));
        for (id, host) in &job.frontends {
            let enabled = faults.up(job, host) && metadata;
            refreshable.insert(id.clone(), enabled);
            if enabled {
                lease_until.insert(id.clone(), time + job.schema_lease_seconds);
            }
        }
        let frontend = faults.up(job, &job.frontends[&job.frontend]);
        out.push((
            time,
            State {
                faults: faults.clone(),
                pd: pd_quorum && leader.is_some() && frontend,
                schema: frontend && (metadata || time < lease_until[&job.frontend]),
                frontend,
            },
        ));
        let next_event = events.get(ix).map_or(job.horizon_seconds, |e| e.time);
        let mut next = next_event.min(job.horizon_seconds);
        if let Some(at) = election_at {
            if at > time {
                next = next.min(at);
            }
        }
        if !metadata {
            for (id, host) in &job.frontends {
                let expiry = lease_until[id];
                if faults.up(job, host) && expiry > time {
                    next = next.min(expiry);
                }
            }
        }
        if next <= time || time >= job.horizon_seconds {
            break;
        }
        time = next;
    }
    out
}

const CLASSES: [&str; 4] = ["normal_write", "fresh_scan", "point_read", "stale_read"];
fn gates(job: &JointJob, state: &State, placement: Option<&[String]>, class: usize) -> bool {
    let local = match placement {
        Some(ids) if class == 0 => quorum(ids, &state.faults, job),
        Some(ids) => ids.iter().any(|id| state.faults.up(job, id)),
        None => true,
    };
    // Explicit old-snapshot reads use a cached snapshot and do not need a
    // fresh schema lease. Point reads need a valid schema but not a new TSO.
    state.frontend && (class == 3 || state.schema) && local && (class > 1 || state.pd)
}

/// Evaluate any instant within [start, start + deadline]. State changes at
/// event/election/lease boundaries, so a probe at each such boundary is exact
/// under the zero-service-time model. Horizon-crossing starts are excluded.
fn succeeds(
    job: &JointJob,
    states: &[(f64, State)],
    start: f64,
    placement: Option<&[String]>,
    class: usize,
) -> bool {
    let end = start + job.deadline_seconds;
    let index = states
        .partition_point(|(at, _)| *at <= start)
        .saturating_sub(1);
    states[index..]
        .iter()
        .take_while(|(at, _)| *at <= end)
        .any(|(_, state)| gates(job, state, placement, class))
}

pub fn replay(job: &JointJob) -> Result<JointRecord, String> {
    job.validate()?;
    let states = states(job);
    let eligible = job.horizon_seconds - job.deadline_seconds;
    let mut boundaries = vec![0.0, eligible];
    for (at, _) in &states {
        for t in [*at, *at - job.deadline_seconds] {
            if t > 0.0 && t < eligible {
                boundaries.push(t);
            }
        }
    }
    boundaries.sort_by(f64::total_cmp);
    boundaries.dedup();
    let mut timeline = JointTimeline {
        seconds: Vec::new(),
        global_ok: Vec::new(),
        candidates: job
            .shards
            .iter()
            .map(|(id, menu)| {
                (
                    id.clone(),
                    menu.iter()
                        .map(|ids| CandidateMask {
                            machine_ids: ids.clone(),
                            local_ok: Vec::new(),
                        })
                        .collect(),
                )
            })
            .collect(),
    };
    let mut class_seconds: BTreeMap<String, BTreeMap<String, f64>> = job
        .shards
        .keys()
        .map(|id| {
            (
                id.clone(),
                CLASSES.iter().map(|c| (c.to_string(), 0.0)).collect(),
            )
        })
        .collect();
    let mut full_seconds = 0.0;
    let mut single_seconds: BTreeMap<String, f64> =
        job.shards.keys().map(|id| (id.clone(), 0.0)).collect();
    let mut global_seconds = 0.0;
    for pair in boundaries.windows(2) {
        let (start, dt) = (pair[0], pair[1] - pair[0]);
        // Open interval interior: at exact boundaries success may differ at a
        // point of zero measure. Avoid rounding into a preceding segment.
        let probe = start + dt / 2.0;
        let global = (0..4).all(|class| succeeds(job, &states, probe, None, class));
        if global {
            global_seconds += dt;
        }
        timeline.seconds.push(dt);
        timeline.global_ok.push(global);
        let mut full = true;
        for (id, menu) in &job.shards {
            for (ix, ids) in menu.iter().enumerate() {
                let by_class: Vec<bool> = (0..4)
                    .map(|class| succeeds(job, &states, probe, Some(ids), class))
                    .collect();
                let all = by_class.iter().all(|ok| *ok);
                timeline.candidates.get_mut(id).unwrap()[ix]
                    .local_ok
                    .push(all);
                if ix == job.selected[id] {
                    if all {
                        *single_seconds.get_mut(id).unwrap() += dt;
                    } else {
                        full = false;
                    }
                    for (class, ok) in CLASSES.iter().zip(by_class) {
                        if ok {
                            *class_seconds.get_mut(id).unwrap().get_mut(*class).unwrap() += dt;
                        }
                    }
                }
            }
        }
        if full {
            full_seconds += dt;
        }
    }
    let per_shard: BTreeMap<_, _> = single_seconds
        .into_iter()
        .map(|(id, s)| (id, s / eligible))
        .collect();
    Ok(JointRecord {
        model: "deterministic_replay_zero_service_time_no_repair".into(),
        horizon_seconds: job.horizon_seconds,
        eligible_start_seconds: eligible,
        censored_seconds: job.deadline_seconds,
        single_access: per_shard.values().sum::<f64>() / per_shard.len() as f64,
        full_access: full_seconds / eligible,
        global_access: global_seconds / eligible,
        per_shard,
        class_success: class_seconds
            .into_iter()
            .map(|(id, classes)| {
                (
                    id,
                    classes
                        .into_iter()
                        .map(|(name, s)| (name, s / eligible))
                        .collect(),
                )
            })
            .collect(),
        timeline,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn toy() -> JointJob {
        serde_json::from_value(json!({
            "mode": "joint", "horizon_seconds": 100.0, "deadline_seconds": 1.0,
            "schema_lease_seconds": 5.0, "pd_election_seconds": 2.0,
            "hosts": {
                "p1": {"region":"r1","az":"a1"}, "p2": {"region":"r2","az":"a2"},
                "p3": {"region":"r3","az":"a3"}, "f": {"region":"r3","az":"a3"},
                "u1": {"region":"r1","az":"a1"}, "u2": {"region":"r2","az":"a2"},
                "u3": {"region":"r3","az":"a3"}, "v1": {"region":"r1","az":"a1"},
                "v2": {"region":"r2","az":"a2"}, "v3": {"region":"r3","az":"a3"},
                "m1": {"region":"r1","az":"a1"}, "m2": {"region":"r2","az":"a2"},
                "m3": {"region":"r3","az":"a3"}
            },
            "pd_members": ["p1","p2","p3"], "pd_leader":"p1",
            "frontends": {"sql":"f"}, "frontend":"sql",
            "system_regions": [["m1","m2","m3"]],
            "shards": {"a": [["u1","u2","u3"], ["p1","u1"]],
                       "b": [["v1","v2","v3"]]},
            "selected": {"a": 0, "b": 0}
        }))
        .unwrap()
    }
    fn event(time: f64, scope: &str, target: &str, down: bool) -> FaultEvent {
        FaultEvent {
            time,
            scope: scope.into(),
            target: Some(target.into()),
            down,
        }
    }
    fn approx(got: f64, expected: f64) {
        assert!((got - expected).abs() < 1e-10, "{got} != {expected}");
    }

    #[test]
    fn pd_quorum_outage_affects_both_shards_once_but_not_point_or_stale() {
        let mut job = toy();
        job.events = vec![
            event(20., "host", "p1", true),
            event(20., "host", "p2", true),
            event(30., "host", "p1", false),
            event(30., "host", "p2", false),
        ];
        let r = replay(&job).unwrap();
        for a in r.per_shard.values() {
            approx(*a, 90. / 99.);
        }
        approx(r.single_access, 90. / 99.);
        approx(r.full_access, 90. / 99.);
        approx(r.class_success["a"]["point_read"], 1.);
        approx(r.class_success["a"]["stale_read"], 1.);
        approx(r.class_success["a"]["normal_write"], 90. / 99.);
        approx(r.censored_seconds, 1.);
    }

    #[test]
    fn one_shard_and_overlapping_shards_contribute_union_once() {
        let mut job = toy();
        // Two replicas of a down together; b only overlaps during [45, 50).
        job.events = vec![
            event(40., "host", "u1", true),
            event(40., "host", "u2", true),
            event(50., "host", "u1", false),
            event(50., "host", "u2", false),
            event(45., "host", "v1", true),
            event(45., "host", "v2", true),
            event(55., "host", "v1", false),
            event(55., "host", "v2", false),
        ];
        let r = replay(&job).unwrap();
        approx(r.per_shard["a"], 90. / 99.);
        approx(r.per_shard["b"], 90. / 99.);
        approx(r.full_access, 85. / 99.); // [40,54) after deadline retries
        approx(r.single_access, 90. / 99.);
        assert!(r.timeline.candidates["a"][1].local_ok.iter().any(|ok| !ok));
    }

    #[test]
    fn region_and_host_events_share_physical_failures_and_election_delay() {
        let mut job = toy();
        job.events = vec![
            event(20., "region", "r1", true),
            event(30., "region", "r1", false),
        ];
        let r = replay(&job).unwrap();
        // p1 (leader) down, but PD quorum remains; election ends at 22.
        approx(r.global_access, 98. / 99.); // [20,21) fails a 1s deadline
                                            // Candidate [p1,u1] loses both physical hosts in r1.
        assert!(r.timeline.candidates["a"][1].local_ok.iter().any(|ok| !ok));
        approx(r.per_shard["a"], 98. / 99.); // PD election, selected user quorum intact
        approx(r.per_shard["b"], 98. / 99.);
        job.events = vec![
            event(20., "host", "p1", true),
            event(30., "host", "p1", false),
        ];
        let r = replay(&job).unwrap();
        assert!(r.timeline.candidates["a"][1].local_ok.iter().any(|ok| !ok));
    }

    #[test]
    fn schema_lease_expiry_and_deadline_boundary_are_separate() {
        let mut job = toy();
        // Metadata quorum fails at 20; cached lease lasts to 25, then
        // requests may wait until metadata recovers at 30.
        job.events = vec![
            event(20., "host", "m1", true),
            event(20., "host", "m2", true),
            event(30., "host", "m1", false),
            event(30., "host", "m2", false),
        ];
        let r = replay(&job).unwrap();
        approx(r.global_access, 95. / 99.); // [25,29)
        approx(r.class_success["a"]["point_read"], 95. / 99.);
        approx(r.class_success["a"]["stale_read"], 1.);
        assert_eq!(r.timeline.seconds.iter().sum::<f64>(), 99.);
        job.events = vec![
            event(20., "host", "p2", true),
            event(20., "host", "p3", true),
            event(20.5, "host", "p2", false),
            event(20.5, "host", "p3", false),
        ];
        approx(replay(&job).unwrap().full_access, 1.); // quorum outage shorter than deadline
    }

    #[test]
    fn invalid_horizon_replicas_and_events_are_rejected_without_panics() {
        let mut job = toy();
        job.deadline_seconds = 100.;
        assert!(replay(&job).unwrap_err().contains("deadline"));
        job.deadline_seconds = 1.;
        job.shards.get_mut("a").unwrap()[0] = vec!["u1".into(), "u1".into()];
        assert!(replay(&job).unwrap_err().contains("distinct"));
        job = toy();
        job.events = vec![event(-1., "host", "u1", true)];
        assert!(replay(&job).unwrap_err().contains("horizon"));
    }
}
