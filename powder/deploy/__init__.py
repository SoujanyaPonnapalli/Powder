"""TiDB deployment and experiment library (October Goals, execution order #7).

Pure logic only: specs, run state, network planning, cost estimates, TiUP topology rendering and the
availability / repartitioning metric math. Nothing here calls AWS or touches a cluster; the scripts in
``infra/tidb/`` do that and import this package. See ``docs/plans/exec-07-tidb-deployment.md``.
"""
