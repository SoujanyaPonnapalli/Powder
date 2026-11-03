"""Stage 5: deploy -> install -> experiment(s) -> collect -> teardown, with teardown guaranteed (plan §6).

The handle is chosen and printed before anything is created. Teardown runs in ``finally``: after success, after a
failure in any stage (a half-failed deploy has resources too), and on SIGINT/SIGTERM. While tearing down, further
SIGINT/SIGTERM are ignored so a second Ctrl-C cannot abort the cleanup halfway. A failed or incomplete teardown is
loud (non-zero exit and the exact command to finish it) even if the experiment succeeded. ``--keep-infra`` leaves
everything up and prints the hourly burn.
"""

from __future__ import annotations

import json
import signal
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import aws, teardown as td
from .deploy import deploy as _deploy
from .experiment import run_experiment as _run_experiment
from .install import install as _install
from .rundir import RunDir
from .spec import ClusterSpec, ExperimentConfig

OK, FAIL, INTERRUPTED = 0, 1, 130


@dataclass
class Stages:
    """The three stage functions; replaced in tests."""
    deploy: Callable = None
    install: Callable = None
    experiment: Callable = None


class Interrupted(KeyboardInterrupt):
    pass


@contextmanager
def interruptible():
    """Turn SIGTERM into KeyboardInterrupt so ``finally`` blocks (teardown) run."""
    def handler(signum, frame):
        raise Interrupted(f"signal {signum}")
    old = {s: signal.signal(s, handler) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        yield
    finally:
        for s, h in old.items():
            signal.signal(s, h)


@contextmanager
def uninterruptible():
    old = {s: signal.signal(s, signal.SIG_IGN) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        yield
    finally:
        for s, h in old.items():
            signal.signal(s, h)


def teardown_handle(clients: aws.Clients, handle: str, rundir: RunDir, out: Callable[[str], None], sleep=time.sleep) -> bool:
    """Delete everything of ``handle``; True only if it succeeded and ``verify`` finds nothing left."""
    ok = True
    try:
        runs = td.discover(clients, handle=handle)
        run = runs.get(handle)
        if run is not None:
            res = td.teardown_run(clients, run, sleep=sleep, log=out)
            ok &= not res["errors"]
            if rundir.exists():
                rundir.teardown_path.write_text(json.dumps({"handle": handle, "at": aws.utcnow(), **res}, indent=1) + "\n")
        left = [x for x in td.verify(clients) if x.startswith(handle + " ")]
        if left:
            out("LEFTOVERS:\n  " + "\n  ".join(left))
            ok = False
        if rundir.state_path.exists():
            st = rundir.load_state()
            for rec in st.instances.values():
                rec.state = "terminated"
            st.save(rundir.state_path)
    except Exception as e:                      # teardown must report, never raise
        out(f"teardown error: {e}")
        ok = False
    return ok


def run_all(spec: ClusterSpec, experiments: list[ExperimentConfig], rundir: RunDir, clients: aws.Clients,
            runner_factory: Callable, operator_cidr: str, quotas: dict | None = None, yes: bool = False,
            keep_infra: bool = False, environ=None, out: Callable[[str], None] = print, sleep=time.sleep,
            stages: Stages | None = None, self_test: bool = False) -> int:
    from .deploy import check_spend_allowed
    check_spend_allowed(yes, environ)                      # before anything: a refusal creates (and tears down) nothing
    st = stages or Stages()
    st.deploy = st.deploy or (lambda spec, rd, runner_factory: _deploy(
        spec, rd, clients, runner_factory, operator_cidr, quotas=quotas, yes=yes, environ=environ, log=out, sleep=sleep))
    st.install = st.install or (lambda spec, rd, runner: _install(spec, rd, runner, log=out, sleep=sleep))
    st.experiment = st.experiment or (lambda spec, cfg, rd, runner: _run_experiment(spec, cfg, rd, runner, log=out, sleep=sleep))

    out(f"deployment handle: {spec.handle}")
    code, failure = OK, None
    started = time.monotonic()
    try:
        with interruptible():
            rundir.ensure()
            state = st.deploy(spec, rundir, runner_factory)
            runner = runner_factory(rundir.load_state())
            if rundir.load_state().stages.get("install") != "done":
                st.install(spec, rundir, runner)
            else:
                out("install already done; skipping")
            for cfg in experiments:
                st.experiment(spec, cfg, rundir, runner)
            if self_test:
                from .selftest import run_self_test
                _, verdict = run_self_test(spec, rundir, runner)
                out(f"metric self-test: {'PASSED' if verdict['passed'] else 'FAILED'}")
                if not verdict["passed"]:
                    raise RuntimeError("metric self-test failed (see selftest.json)")
    except KeyboardInterrupt:
        code, failure = INTERRUPTED, "interrupted"
        out(f"[{spec.handle}] interrupted: tearing down")
    except Exception as e:
        code, failure = FAIL, f"{type(e).__name__}: {e}"
        out(f"[{spec.handle}] FAILED: {failure}")
    finally:
        hours = (time.monotonic() - started) / 3600
        if keep_infra:
            burn = rundir.load_state().est_hourly_usd if rundir.state_path.exists() else 0.0
            out(f"[{spec.handle}] --keep-infra: leaving the deployment up, ~${burn:.2f}/h. Tear down with: "
                f"infra/tidb/teardown_infra.py {spec.handle}")
        else:
            with uninterruptible():
                clean = teardown_handle(clients, spec.handle, rundir, out, sleep=sleep)
            if not clean:
                out(f"[{spec.handle}] TEARDOWN INCOMPLETE: resources may still be billing. Finish with: "
                    f"infra/tidb/teardown_infra.py {spec.handle}   (then --verify)")
                if code == OK:                  # a successful experiment does not excuse leaked, billing resources
                    code = FAIL
            else:
                out(f"[{spec.handle}] torn down ({hours:.2f} h).")
    out(f"deployment handle: {spec.handle}  exit={code}" + (f"  ({failure})" if failure else ""))
    return code
