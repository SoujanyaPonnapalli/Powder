You are running unattended on the Powder work VM (Ubuntu, AWS us-west-1) for a
December OSDI submission. Nobody is available to answer questions: make
reasonable decisions, record them in the plan, and keep going.

1. Read `CLAUDE.md`, then `docs/plans/exec-01-canonical-failure-data.md`,
   especially §0 (decisions) and §8 (progress checklist).
2. Pick the FIRST unchecked item (`- [ ]`) in §8 that you can complete on
   this machine. Network access and the instance role's read-only AWS
   permissions are available; `az` may or may not be logged in.
3. Do it. Follow the repo's existing conventions:
   - fault-library code lives in `powder/faults/`, build scripts in
     `scripts/fault_library/`, entries in `data/fault_library/`;
   - every number needs a source record with a V/S status (plan §7.2);
   - entries used in results are immutable: change them by adding `@v2`.
4. Verify: `bash scripts/fault_library/build_all.sh` must pass (it runs the
   tests). If you touched Rust, `cargo test --release --manifest-path
   rust/Cargo.toml` must pass too.
5. Tick the item in §8 (`- [x]`), adding one line on what was done and any
   decision taken. Commit with a clear message (one commit per item) and stop.
   Do not start a second item in the same session.

Never push to `main` or rewrite published history. Do not delete data under
`data/fault_library/raw/`. If an item is blocked (e.g. needs a human login),
add `(blocked: reason)` to it, commit that, and move to the next item.
