this is a repo for a December OSDI submission.

We are going to work through all of the tasks in October Goals.md

Each chat will detail a specific action item to take. We will make a plan that will not block future tasks and then commit code that advances the project.

## Where things are

- Plans live in `docs/plans/`. Exec #1 (fault + price library): `docs/plans/exec-01-canonical-failure-data.md`.
  Its §0 lists the decisions taken; its §8 is the progress checklist (unattended runs work from it).
- Fault + price library: code in `powder/faults/`, build scripts in `scripts/fault_library/`,
  entries in `data/fault_library/`, generated catalogue in `docs/fault_library.md`.
  Rebuild and test everything with `bash scripts/fault_library/build_all.sh`.
- Library tests are NumPy-only: `python scripts/fault_library/run_tests.py` works without pytest/SciPy.
- Monte Carlo engine: Rust, `rust/` (`cargo build --release --manifest-path rust/Cargo.toml`).
- AWS work VM (data refresh + unattended Claude sessions): `infra/aws/README.md`.
- TiDB deployment + experiments on AWS (exec #7): plan `docs/plans/exec-07-tidb-deployment.md` (§12 checklist),
  scripts and README in `infra/tidb/`, library in `powder/deploy/`, pinned TiDB in `third_party/tidb/VERSIONS` and
  `third_party/pd` (submodule: `git submodule update --init --depth 1 third_party/pd`). Results: `outputs/tidb/<handle>/`.
  Tests: `~/.cache/powder/venv` has moto; `python -m pytest tests/test_deploy_*.py` (moto tests skip without it) and
  `go test ./...` in `infra/tidb/loadgen`. Deploys cost money: never run them unattended.

## Conventions

- Every number in the library cites a source record measurement with status V (primary) or S (reliable secondary);
  U (unconfirmed) can never be cited. Entries used in results are immutable: change them by adding `@v2`.
  Exception (user decision 2026-11-17): the weekly planned-downtime cadence was deleted. Ford's cadence
  (`planned_downtime/ford10-rate`) is the only one; use the `@v2` canonical scenarios and `az-aware@v3`.
  Weekly-era results (`outputs/exec-02/`) reproduce only at commit `1b52ccf`.
- Experiments target AWS: all US regions, one identical 8 vCPU x86 instance type (`i4i.2xlarge`) with local NVMe;
  heterogeneity is emulated by injecting each machine class's faults (`powder.faults.trace`).
- Commit as `Reginald Frank <reginaldfrank77@berkeley.edu>`; never push to `main`.
