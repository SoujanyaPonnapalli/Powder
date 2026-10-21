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

## Conventions

- Every number in the library cites a source record measurement with status V (primary) or S (reliable secondary);
  U (unconfirmed) can never be cited. Entries used in results are immutable: change them by adding `@v2`.
- Experiments target AWS: all US regions, one identical 8 vCPU x86 instance type (`i4i.2xlarge`) with local NVMe;
  heterogeneity is emulated by injecting each machine class's faults (`powder.faults.trace`).
- Commit as `Reginald Frank <reginaldfrank77@berkeley.edu>`; never push to `main`.
