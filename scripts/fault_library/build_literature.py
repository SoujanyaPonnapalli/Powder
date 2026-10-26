"""Literature sources, the disk pool, and the composed AWS on-demand profile (Step D).

    python scripts/fault_library/build_literature.py      (run after build_ford10.py)

Every number below was confirmed on 2026-10-20/21 from the primary document
(V) or a reliable secondary summary (S); unconfirmed recollections (U) are
recorded only to document that they were rejected. See plan §7.11.

Outputs:
  data/fault_library/sources/literature.json
  data/fault_library/profiles/component/disks.json   (per-source + disk/pooled)
  data/fault_library/profiles/node/host-reference.json (host/vishwanath10)
  data/fault_library/profiles/node/aws-ondemand.json (cloud/aws-ondemand-nvme)
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from powder.faults.fit import nb_frailty_from_p_and_conditional_mean, random_effects_pool  # noqa: E402

LIB = ROOT / "data/fault_library"
YEAR_H = 8766.0


def tri(lo, c, hi):
    return {"low": lo, "central": c, "high": hi}


def r8(x):
    if isinstance(x, float):
        return float(f"{x:.8g}")
    if isinstance(x, dict):
        return {k: r8(v) for k, v in x.items()}
    if isinstance(x, list):
        return [r8(v) for v in x]
    return x


def M(mid, verbatim, value, unit, status, loc, covers=(), conversion=None, notes=None):
    d = {"id": mid, "metric_verbatim": verbatim, "value": value, "unit": unit, "status": status,
         "location": loc, "covers": list(covers)}
    if conversion:
        d["conversion"] = conversion
    if notes:
        d["notes"] = notes
    return d


SOURCES = [
    {"id": "backblaze-drive-stats", "kind": "source", "title": "Backblaze Drive Stats (2025 annual; SSD edition mid-2023)",
     "url": "https://www.backblaze.com/blog/backblaze-drive-stats-for-2025/", "accessed": "2026-10-20",
     "population": "Backblaze data-centre drives (HDD data drives; SSD boot drives)",
     "measurements": [
         M("hdd-2025-afr", "2025 annual AFR declined to 1.36%", 0.0136, "failures per drive-year", "V",
           "Drive Stats for 2025", ["disk"], "AFR = failures / drive-days x 365, a rate: lambda = AFR / 8766 per hour"),
         M("hdd-2025-drives", "349,462 hard drives used to store data", 349462, "drives", "V", "Drive Stats for 2025"),
         M("hdd-lifetime-afr", "lifetime AFR held steady at 1.30%", 0.0130, "failures per drive-year", "V", "Drive Stats for 2025", ["disk"]),
         M("ssd-2023-q1-afr", "SSD boot drives Q1 2023 AFR 0.96%", 0.0096, "failures per drive-year", "V",
           "https://www.backblaze.com/blog/ssd-edition-2023-mid-year-drive-stats-review/", ["disk"]),
         M("ssd-2023-q2-afr", "SSD boot drives Q2 2023 AFR 1.05%", 0.0105, "failures per drive-year", "V",
           "https://www.backblaze.com/blog/ssd-edition-2023-mid-year-drive-stats-review/", ["disk"]),
         M("ssd-2023-drives", "3,144 SSDs in the storage servers (June 30, 2023)", 3144, "drives", "V",
           "https://www.backblaze.com/blog/ssd-edition-2023-mid-year-drive-stats-review/"),
     ]},
    {"id": "schroeder-gibson-fast07", "kind": "source",
     "title": "Disk failures in the real world: What does an MTTF of 1,000,000 hours mean to you?",
     "url": "https://www.usenix.org/legacy/events/fast07/tech/schroeder/schroeder.pdf", "accessed": "2026-10-20",
     "population": ">100,000 drives, several large production systems",
     "measurements": [
         M("weibull-k", "time between replacements ... Weibull distribution with a shape parameter between 0.7 and 0.8",
           [0.7, 0.8], "shape", "V", "FAST'07 §5", ["disk", "hw_reboot", "hw_host_loss"]),
         M("arr-2-4pct", "One study [29] reports ARR (annual replacement rate) for disks between 2% and 4%",
           [0.02, 0.04], "replacements per disk-year", "V", "quoted in Ford et al. OSDI'10 p.3 ([29] = Schroeder & Gibson)",
           ["disk"], "rate; central = geometric mean 2.83%"),
     ]},
    {"id": "jiang-fast08", "kind": "source", "title": "Are disks the dominant contributor for storage failures?",
     "url": "https://www.usenix.org/legacy/event/fast08/tech/jiang.html", "accessed": "2026-10-21",
     "population": "storage subsystems (shelves, enclosures, interconnects, disks)",
     "measurements": [
         M("subsystem-afr", "AFR generally between 2% and 4%, but for some storage systems values ranging between 3.9% and 8.3%",
           {"typical": [0.02, 0.04], "some_systems": [0.039, 0.083]}, "failures per subsystem-year", "V",
           "quoted in Ford et al. OSDI'10 p.3 ([19] = Jiang et al.)", [],
           notes="storage subsystem, not a single drive: recorded, not pooled into disk/pooled")]},
    {"id": "vishwanath-nagappan-socc10", "kind": "source", "title": "Characterizing Cloud Computing Hardware Reliability",
     "url": "https://www.microsoft.com/en-us/research/wp-content/uploads/2010/06/socc088-vishwanath.pdf",
     "local_path": "docs/reading/Characterizing Cloud Computing Hardware Reliability.pdf",
     "accessed": "2026-10-21", "population": ">100,000 Microsoft servers, 14 months of part replacements (2010)",
     "measurements": [
         M("disk-top-failure", "hard disks are the number one failing component", True, "text", "V", "SoCC'10 §3"),
         M("disk-events-top5", "between 11% and 25% of all servers see at least 1 disk event in the given period (14 months, largest clusters)",
           [0.11, 0.25], "fraction of servers / 14 months", "V", "SoCC'10 §3", ["disk"],
           notes="per server (multi-disk), not per drive: not pooled into disk/pooled"),
         M("afr-8pct", "We saw a total of 20 replacements in the 14 month period. However, these replacements were all "
           "contained in around 9 machines. 9 machine failures in 14 months translates to an annual failure rate (AFR) "
           "of around 8%", {"replacements_per_100": 20, "machines_per_100": 9, "afr": 0.08},
           "per 100 servers / 14 months", "V", "SoCC'10 §2.2 (PDF read 2026-10-21)", ["hw_host_loss", "disk"]),
         M("repairs-per-repaired", "The average number of repairs seen by a repaired machine is 2 ... around 5 machines "
           "failed only once. Thus, around 4 machines had more than 1 failure", {"mean": 2, "once": 5, "repeat": 4},
           "per 100 servers / 14 months", "V", "SoCC'10 §2.2, §4.2"),
         M("disk-share", "a majority (78%) were for hard disks ... 5% due to memory and the rest (18%)",
           {"disk": 0.78, "memory": 0.05, "raid": 0.04, "other": 0.18}, "share of replacements", "V", "SoCC'10 §3",
           notes="RAID share by subtraction of the quoted shares (raid ~4%)"),
         M("disk-afr", "hard disks are the number one failing component with an AFR for 2.7%", 0.027,
           "failures per disk-year", "V", "SoCC'10 §3", ["disk"]),
         M("raid-afr", "For overall RAID controllers the failure rate was 0.7% AFR", 0.007, "per controller-year", "V", "SoCC'10 §3"),
         M("repeat-timing", "20% of all repeat failures happen within a day of the first failure, and 50% of all repeat "
           "failures happen within 2 weeks", {"within_day": 0.2, "within_2_weeks": 0.5}, "fraction of repeat failures",
           "V", "SoCC'10 §4.2"),
     ]},
    {"id": "aws-ec2-sla", "kind": "source", "title": "Amazon Compute Service Level Agreement",
     "url": "https://aws.amazon.com/compute/sla/", "accessed": "2026-10-20", "population": "EC2 SLA",
     "measurements": [
         M("instance-level", "Instance-Level Uptime Percentage of at least 99.5%", 0.995, "monthly uptime", "V", "Compute SLA",
           ["hw_reboot", "hw_host_loss", "sw"], "cap: unplanned downtime <= 0.005 x 8766 = 43.8 h per instance-year"),
         M("region-level", "Monthly Uptime Percentage of at least 99.99% (multi-AZ)", 0.9999, "monthly uptime", "V", "Compute SLA"),
     ]},
    {"id": "azure-vm-docs", "kind": "source", "title": "Azure: Understand VM reboots; VM SLA",
     "url": "https://learn.microsoft.com/en-us/azure/virtual-machines/understand-vm-reboots", "accessed": "2026-10-20",
     "population": "Azure VMs", "measurements": [
         M("pause-18mo", "no more than once every 18 months for general purpose VM sizes ... pause the VM for about 30 seconds",
           {"max_rate_per_18_months": 1, "pause_s": 30}, "events", "V", "Understand VM reboots", ["pause"]),
         M("single-vm-sla", "Single Instance Virtual Machine using Premium SSD ... at least 99.9%", 0.999, "monthly uptime", "V",
           "https://www.azure.cn/en-us/support/sla/virtual-machines/"),
     ]},
    {"id": "narya-afpd", "kind": "source", "title": "Narya (OSDI'20), AIR KPI (arXiv 1910.12200), AFPD (2025)",
     "url": "https://www.usenix.org/conference/osdi20/presentation/levy", "accessed": "2026-10-20",
     "population": "Azure fleet", "measurements": [
         M("air-definition", "AIR = interruptions in T / total VM uptime in T x 365 x 100 (interruptions per 100 VM-years)",
           "definition", "text", "V", "Narya OSDI'20 slides", conversion="lambda = AIR / (100 x 8766) per hour"),
         M("narya-reduction", "Narya on average reduces VM interruptions by 26% ... oracle 35%", {"narya": 0.26, "oracle": 0.35},
           "relative reduction", "V", "Narya OSDI'20 abstract"),
         M("afpd-reduction", "improving hardware-related VM interruption prevention by 36%", 0.36, "relative", "V", "AFPD (IEEE 2025) abstract"),
     ]},
    {"id": "birke-dsn14", "kind": "source", "title": "Failure Analysis of Virtual and Physical Machines: Patterns, Causes and Characteristics (DSN'14)",
     "url": "https://ieeexplore.ieee.org/document/6903562/",
     "local_path": "docs/reading/Failure_Analysis_of_Virtual_and_Physical_Machines_Patterns_Causes_and_Characteristics.pdf",
     "accessed": "2026-10-21", "population": "4292 VMs and 5129 PMs, five commercial (IBM) datacenter subsystems, one year; 2759 crash tickets",
     "measurements": [
         M("population", "Our dataset consists of 2759 crash tickets spread across 4292 VMs and 5129 PMs", {"vms": 4292, "pms": 5129, "crash_tickets": 2759}, "count", "V", "DSN'14 §III"),
         M("weekly-rate", "PMs have higher failure rates than VMs roughly by 40%, i.e., 0.005 vs. 0.003", {"pm": 0.005, "vm": 0.003},
           "failures per server-week", "V", "DSN'14 §IV-A, Fig 2", ["hw_reboot", "sw"], "rate_per_hour = value / 168"),
         M("weekly-random-prob", "Table V random weekly failure probability: PM .0062, VM .0038 (all systems)", {"pm": 0.0062, "vm": 0.0038},
           "P(fail at least once in a week)", "V", "DSN'14 Table V", ["hw_reboot", "sw"], "rate_per_hour = -ln(1-p)/168"),
         M("recurrent-week", "Table V recurrent failure probability within a week: PM .22 (35.5x), VM .16 (42.1x)",
           {"pm": 0.22, "vm": 0.16, "pm_ratio": 35.5, "vm_ratio": 42.1}, "P(fail again within a week | failure)", "V", "DSN'14 Table V"),
         M("interfailure-gamma", "inter-failure times ... Gamma: PM alpha=0.29 beta=119.60; VM alpha=0.44 beta=84.28 (days); VM mean 37.22 days",
           {"pm": {"alpha": 0.29, "beta_days": 119.6}, "vm": {"alpha": 0.44, "beta_days": 84.28}}, "days", "V", "DSN'14 Fig 3"),
         M("repair-lognormal", "repair times ... Log-normal: VM mu=1.14 sigma=2.88, PM mu=1.80 sigma=2.07 (hours); mean repair times 38.5 (PM) and 19.6 (VM) hours",
           {"vm": {"mu_h": 1.14, "sigma": 2.88, "mean_h": 19.6}, "pm": {"mu_h": 1.80, "sigma": 2.07, "mean_h": 38.5}}, "hours", "V", "DSN'14 Fig 4"),
         M("repair-by-class", "Table IV mean/median repair hours: HW 80.1/8.28, Net 67.6/8.97, Power 12.17/0.83, Reboot 18.03/2.27, SW 30.0/22.37",
           {"hw": [80.1, 8.28], "net": [67.6, 8.97], "power": [12.17, 0.83], "reboot": [18.03, 2.27], "sw": [30.0, 22.37]}, "hours (mean, median)", "V", "DSN'14 Table IV"),
         M("vm-reboot-share", "a significant percentage of VM failures, i.e., roughly 35%, are caused by unexpected reboots", 0.35,
           "fraction of VM failures", "V", "DSN'14 §IV-C"),
         M("vm-lower", "VMs have lower failure rates and lower probability of failure recurrences than PMs", True, "text", "V",
           "as summarised by a citing paper; consistent with Table V absolute values"),
     ]},
    {"id": "lanl-dsn06", "kind": "source", "title": "A large-scale study of failures in high-performance computing systems (DSN'06)",
     "url": "https://www.pdl.cmu.edu/PDL-FTP/associated/dsn06.pdf", "accessed": "2026-10-20",
     "population": "22 LANL HPC systems, 1996-2005, ~23,000 failures", "measurements": [
         M("weibull-k", "decreasing hazard rates with Weibull shape parameter of 0.7-0.8", [0.7, 0.8], "shape", "V", "DSN'06 abstract",
           ["hw_reboot", "hw_host_loss", "sw"]),
         M("repair-lognormal", "repair times are well modeled by a lognormal distribution", True, "text", "V", "TDSC'09 abstract"),
         M("rates", "failure rates ... ranging from 20-1000 failures per year", [20, 1000], "failures per system-year", "V", "TDSC'09 abstract")]},
]


def disk_profiles():
    """Per-source disk components and the random-effects pool."""
    hdd_n = 349462 * 0.0136  # ~ failures in 2025 (drives x annual AFR)
    ssd_rate = 0.5 * (0.0096 + 0.0105)
    ssd_n = 3144 * 0.5 * ssd_rate  # half a year of exposure
    entries_ = [
        ("disk/backblaze-hdd-2025", 0.0136, 1.0 / hdd_n, ["backblaze-drive-stats#hdd-2025-afr", "backblaze-drive-stats#hdd-2025-drives"],
         "HDD data drives, 2025 annual AFR (a rate)."),
        ("disk/backblaze-ssd-2023", ssd_rate, 1.0 / ssd_n, ["backblaze-drive-stats#ssd-2023-q1-afr",
                                                              "backblaze-drive-stats#ssd-2023-q2-afr",
                                                              "backblaze-drive-stats#ssd-2023-drives"],
         "SSD boot drives, mean of Q1/Q2 2023 AFR; closest public proxy for datacentre NVMe."),
        ("disk/schroeder-gibson-fast07", math.sqrt(0.02 * 0.04), 0.25, ["schroeder-gibson-fast07#arr-2-4pct"],
         "Field ARR 2-4% (via Ford et al.); event count unknown -> log-variance 0.25."),
        ("disk/vishwanath10", 0.027, 0.25, ["vishwanath-nagappan-socc10#disk-afr"],
         "Microsoft fleet disk AFR 2.7% (2010); event count not given -> log-variance 0.25."),
        ("disk/ford10-table2", 1 / math.sqrt(10 * 50), 0.25, ["ford10#table2-disk-mttf"],
         "Google disk MTTF 10-50 years (application-visible); event count unknown -> log-variance 0.25."),
    ]
    comps = []
    for cid, rate_yr, var, refs, note in entries_:
        if cid == "disk/ford10-table2":
            continue  # disk/ford10 already exists (build_ford10.py); pooled below references it
        comps.append({"id": cid, "kind": "component", "component": "disk", "version": 1, "derived_from": refs,
                      "built_by": "scripts/fault_library/build_literature.py",
                      "rate_per_hour": tri(rate_yr / YEAR_H, rate_yr / YEAR_H, rate_yr / YEAR_H),
                      "pool_input": {"rate_per_year": rate_yr, "log_variance": var}, "notes": note})
    pooled = random_effects_pool([e[1] for e in entries_], [e[2] for e in entries_])
    comps.append({"id": "disk/pooled", "kind": "component", "component": "disk", "version": 1,
                  "derived_from": ["disk/backblaze-hdd-2025", "disk/backblaze-ssd-2023",
                                   "disk/schroeder-gibson-fast07", "disk/vishwanath10", "disk/ford10"],
                  "built_by": "scripts/fault_library/build_literature.py",
                  "rate_per_hour": tri(pooled.low / YEAR_H, pooled.central / YEAR_H, pooled.high / YEAR_H),
                  "pool": {"method": "DerSimonian-Laird random effects on log rates; low/high = 90% prediction interval "
                                     "with Student-t, k-2 dof (Higgins et al. 2009)",
                           "tau2": pooled.tau2, "se": pooled.se, "k": pooled.k,
                           "per_year": {"low": pooled.low, "central": pooled.central, "high": pooled.high}},
                  "notes": ("Datacentre NVMe on i4i/m6id may differ from these HDD/boot-SSD fleets; the between-source "
                            "spread (tau2) widens the band accordingly.")})
    return comps, pooled


def main():
    (LIB / "sources/literature.json").write_text(json.dumps(r8(SOURCES), indent=2) + "\n")
    comps, pooled = disk_profiles()
    (LIB / "profiles/component/disks.json").write_text(json.dumps(r8(comps), indent=2) + "\n")

    # host/vishwanath10: server repair process with NB frailty
    mu14, a = nb_frailty_from_p_and_conditional_mean(0.09, 20 / 9)
    rate_yr = mu14 * 12 / 14
    vish = {"id": "host/vishwanath10", "kind": "node", "version": 1, "tier": "reference",
            "tags": ["microsoft", "2010", "bare-metal", "frailty", "recoverability-unknown"],
            "derived_from": ["vishwanath-nagappan-socc10#afr-8pct", "vishwanath-nagappan-socc10#repairs-per-repaired"],
            "built_by": "scripts/fault_library/build_literature.py",
            "notes": ("Physical server hardware repairs (2010, HDD-heavy servers): 20 replacements on ~9 of 100 servers per "
                      "14 months (PDF, status V). All repairs are treated as host loss (data lost on "
                      "local disks): pessimistic. frailty_a from P(N>=1)=0.09 and E[N|N>=1]=2.22 via a gamma-Poisson "
                      "model: strong lemon effect."),
            "causes": {"hw_host_loss": {"rate_per_hour": tri(rate_yr / YEAR_H, rate_yr / YEAR_H, rate_yr / YEAR_H)}},
            "frailty_a": a, "shape": {"type": "exponential"}}

    # cloud/aws-ondemand-nvme (plan §7.4, §7.9)
    # Transient causes come from Ford et al. only (user decision 2026-10-26): Ford's unplanned reboots and
    # node restarts are recoverable by definition (the node responds again), whereas Vishwanath's non-disk
    # repairs and LANL's records do not say whether data survived, so they are not used.
    ford = json.loads((LIB / "profiles/node/ford10.json").read_text())
    ford_all = next(p for p in ford if p["id"] == "cluster/ford10-google-storage")
    ford_ge15 = next(p for p in ford if p["id"] == "cluster/ford10-google-storage-ge15min")
    f_lm = tri(0.5, 0.25, 0.0)     # share of host reboots hidden from the instance by live migration / live update
    reboot = {k: ford_all["causes"]["hw_reboot"]["rate_per_hour"][k] * (1 - f_lm[k]) for k in f_lm}
    sw = dict(ford_ge15["causes"]["sw"]["rate_per_hour"])   # >=15-min node restarts (short ones: planned downtime)
    host_loss = tri(0.0, 0.0, 0.0)  # no source with known recoverability: a documented gap (plan §0)
    pause = tri(0.0, 0.0, 1.0 / (18 * 30.4375 * 24))  # Azure bound, used only at the pessimistic end
    aws = {"id": "cloud/aws-ondemand-nvme", "kind": "node", "version": 1, "tier": "cloud",
           "tags": ["aws", "on-demand", "local-nvme", "derived"],
           "derived_from": ["aws-ec2-sla#instance-level",
                            "azure-vm-docs#pause-18mo", "narya-afpd#narya-reduction", "narya-afpd#afpd-reduction",
                            "ford10#derived-rate-node_restart", "ford10#derived-rate-unplanned_reboot",
                            "ford10#digitized-unplanned_reboot", "ford10#digitized-node_restart", "disk/pooled"],
           "built_by": "scripts/fault_library/build_literature.py",
           "notes": ("No public source gives a modern on-demand VM interruption rate (plan §7.13); this profile is "
                     "derived, using only sources whose recoverability is known. Host reboots: Ford's unplanned "
                     "reboots (all events; Table 1 cell range as low/high) x (1 - f_LM), f_LM in {0.5, 0.25, 0} "
                     "(Narya/AFPD: live migration and prediction hide a large share). Software: Ford's >=15-min node "
                     "restarts (short automatic restarts are left to planned downtime and exec #10). Host loss "
                     "(non-disk faults that lose the host): 0 - no source says which host faults lose local data "
                     "(Vishwanath's non-disk repairs and LANL were excluded for this reason); a known optimistic gap. "
                     "Pause: Azure's 30 s at most once per 18 months, pessimistic end only. Disk: disk/pooled. "
                     "Durations: Ford unplanned-reboot CDF (hw_reboot), Ford node-restart CDF >=15 min (sw). The "
                     "pessimistic end is clipped to the AWS instance SLA allowance if it exceeds it. Replace with "
                     "exec #10 measurements as @v2."),
           "assumptions": {"f_LM": f_lm, "host_loss": "0 (no source with known recoverability)",
                           "hw_reboot_source": "cluster/ford10-google-storage", "sw_source": "cluster/ford10-google-storage-ge15min"},
           "causes": {
               "hw_reboot": {"rate_per_hour": reboot, "duration": ford_all["causes"]["hw_reboot"]["duration"]},
               "hw_host_loss": {"rate_per_hour": host_loss},
               "sw": {"rate_per_hour": sw, "duration": ford_ge15["causes"]["sw"]["duration"]},
               "pause": {"rate_per_hour": pause, "duration": {"type": "constant", "value": 30.0}},
               "disk": {"rate_per_hour": next(c for c in comps if c["id"] == "disk/pooled")["rate_per_hour"]},
           },
           "shape": {"type": "exponential"},
           "shape_variant": {"type": "weibull", "k": tri(0.7, 0.75, 0.8),
                             "source": ["schroeder-gibson-fast07#weibull-k"]}}
    # vm/birke14-*: enterprise VM / PM crash tickets (IBM datacenters, 1 year)
    def birke(kind, wk_rate, wk_prob, med_h, mean_h, ratio, gamma_a):
        c = wk_rate / 168.0                    # failures per server-hour (Fig 2 mean weekly rate)
        h = -math.log1p(-wk_prob) / 168.0      # Table V P(>=1 failure in a week) -> rate
        sigma = math.sqrt(2 * math.log(mean_h / med_h))  # lognormal matching the reported median and mean
        mu = math.log(med_h * 3600.0)
        return {"id": f"vm/birke14-{kind}", "kind": "node", "version": 1, "tier": "reference",
                "tags": ["ibm", "enterprise", "2012-2013", kind, "recoverability-unknown"],
                "derived_from": ["birke-dsn14#weekly-rate", "birke-dsn14#weekly-random-prob", "birke-dsn14#repair-lognormal",
                                 "birke-dsn14#recurrent-week", "birke-dsn14#interfailure-gamma"],
                "built_by": "scripts/fault_library/build_literature.py",
                "notes": (f"{kind.upper()} crash tickets (server unresponsive/unreachable), all causes, as one transient cause "
                          "(the paper gives no data-loss rate). NOT INSTANTIATED as a machine class (user decision 2026-10-26): "
                          "crash tickets mean 'server unresponsive/unreachable' and the paper never says whether data "
                          "survived, so the transient/permanent split is unknown. Kept for reference only. Rate central = mean weekly failure rate (Fig 2), high = "
                          "Table V weekly failure probability. Duration: lognormal matched to the reported median "
                          f"({med_h:.2f} h) and mean ({mean_h} h); the paper's own fit (sigma ~2-2.9) has a much larger mean "
                          "than the data, so it is not used. Repair = ticket open to close, including queueing. "
                          f"Recurrence within a week is {ratio}x the random weekly probability; frailty_a = 1/(ratio - 1) "
                          "(gamma-Poisson: E[Z | failed] = 1 + 1/a), opt-in. Inter-failure Gamma shape "
                          f"{gamma_a} (bursty) is recorded in shape_variant."),
                "causes": {"hw_reboot": {"rate_per_hour": tri(c, c, h),
                                         "duration": {"type": "lognormal", "mu": mu, "sigma": sigma}}},
                "frailty_a": 1.0 / (ratio - 1.0),
                "shape": {"type": "exponential"},
                "shape_variant": {"type": "gamma", "alpha": gamma_a, "source": "birke-dsn14#interfailure-gamma"}}
    birke_vm = birke("vm", 0.003, 0.0038, math.exp(1.14), 19.6, 42.1, 0.44)
    birke_pm = birke("pm", 0.005, 0.0062, math.exp(1.80), 38.5, 35.5, 0.29)
    (LIB / "profiles/node/host-reference.json").write_text(json.dumps(r8([vish, birke_vm, birke_pm]), indent=2) + "\n")
    # SLA clip (plan §7.4): the instance SLA (99.5%) allows <= 43.8 h of unplanned downtime per year;
    # if the pessimistic end exceeds it, scale the high transient rates down to the cap and log it.
    from powder.faults.dists import from_spec as _fs
    from powder.faults.schema import pick as _pick
    cap_h = 0.005 * YEAR_H
    def _h(c):
        spec = aws["causes"][c]
        return spec["rate_per_hour"]["high"] * YEAR_H * _fs(_pick(spec["duration"], "high", "duration")).mean / 3600
    hi_h = sum(_h(c) for c in ("hw_reboot", "sw", "pause"))
    if hi_h > cap_h:  # scale the Ford-derived causes only; the Azure pause bound is kept
        f = (0.99 * cap_h - _h("pause")) / (_h("hw_reboot") + _h("sw"))
        for c in ("hw_reboot", "sw"):
            r = aws["causes"][c]["rate_per_hour"]
            r["high"] = max(r["central"], r["high"] * f)
        aws["assumptions"]["sla_clip"] = {"pessimistic_h_per_year_before": hi_h, "cap_h_per_year": cap_h, "scale": f}
        print(f"SLA clip: pessimistic unplanned downtime {hi_h:.1f} h/yr > {cap_h:.1f}; high rates x {f:.3f}", file=sys.stderr)
    (LIB / "profiles/node/aws-ondemand.json").write_text(json.dumps(r8([aws]), indent=2) + "\n")

    # SLA cap check (pessimistic end)
    from powder.faults.dists import from_spec
    from powder.faults.schema import pick
    down_h = 0.0
    for c in ("hw_reboot", "sw", "pause"):
        spec = aws["causes"][c]
        d = from_spec(pick(spec["duration"], "high", "duration"))
        down_h += spec["rate_per_hour"]["high"] * YEAR_H * d.mean / 3600
    rebuild_h = 1.0
    down_h += (host_loss["high"] + aws["causes"]["disk"]["rate_per_hour"]["high"]) * YEAR_H * rebuild_h
    print(json.dumps(r8({"disk_pool_per_year": pooled.__dict__, "vishwanath": {"mu_14mo": mu14, "frailty_a": a},
                         "aws_rates_per_year": {c: {k: v * YEAR_H for k, v in s["rate_per_hour"].items()}
                                                for c, s in aws["causes"].items()},
                         "pessimistic_unplanned_downtime_h_per_year": down_h,
                         "aws_instance_sla_cap_h_per_year": 0.005 * YEAR_H,
                         "within_sla_cap": down_h <= 0.005 * YEAR_H}), indent=1))


if __name__ == "__main__":
    main()
