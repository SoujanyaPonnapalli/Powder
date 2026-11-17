# These results use the removed weekly planned-downtime cadence

Every file in this directory was produced before 2026-11-04, when the library switched planned downtime from an
assumed weekly cadence (`planned_downtime/weekly-ford10`) to Ford's measured cadence
(`planned_downtime/ford10-rate`, one planned reboot per ~72 days per node).

The weekly entries were deleted on 2026-11-17: `weekly-ford10`, `weekly-ford10-rolling`, the v1
`scenario/canonical-*` scenarios, `scenario/canonical-central-az-aware@v2` and `scenario/reference-ford10-weekly`.
Files here that name them (`az-frontiers.json`, `az-mc-scores.jsonl`, `az-24h-oneshot.json`, and the
`horizons-*.jsonl` runs of `scripts/experiment_horizon.py`) can only be reproduced at commit `1b52ccf`:

    git worktree add /tmp/powder-weekly 1b52ccf

Rerunning the scripts at HEAD uses Ford's cadence. Planned downtime then falls from about 0.2% to about 0.02% of
node time, so availability comes out roughly one nine higher. Do not cite these files as current results.
