"""Command-line entry points for the five scripts in ``infra/tidb/`` (thin wrappers import these).

Exit codes: 0 ok, 1 failure, 2 invalid input/spec, 3 spend not allowed, 4 refused (nothing selected / unsafe).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

from . import aws, plan as planmod, teardown as td
from .deploy import DeployError, SpendNotAllowed, deploy, fetch_quotas
from .install import InstallError, install
from .remote import RemoteError, SshRunner
from .rundir import DEFAULT_ROOT, RunDir, existing_handles
from .spec import ClusterSpec, SpecError, parse_spec, resolve_handle
from .state import RunState

OK, FAIL, BAD_INPUT, NO_SPEND, REFUSED = 0, 1, 2, 3, 4


def ssh_runner_factory(rundir: RunDir) -> Callable[[RunState], SshRunner]:
    def make(state: RunState) -> SshRunner:
        hosts = {n: (r.private_ip, r.public_ip) for n, r in state.instances.items()}
        return SshRunner(Path(state.key_path or rundir.key_path), hosts)
    return make


def load_run_spec(rundir: RunDir) -> ClusterSpec:
    """The spec a run was deployed from (the copy in its directory), bound to the directory's handle."""
    if not rundir.spec_path.exists():
        raise SpecError(f"{rundir.spec_path} not found: has handle {rundir.handle} been deployed from this machine?")
    spec = parse_spec(rundir.spec_path)
    spec.handle = rundir.handle
    return spec


# ------------------------------------------------------------------ deploy_infra
def deploy_main(argv: list[str] | None = None, clients: aws.Clients | None = None, runner_factory=None,
                environ=None, out=print, sleep=None) -> int:
    ap = argparse.ArgumentParser(prog="deploy_infra.py", description="Create the AWS infrastructure for a run.")
    ap.add_argument("--spec", required=True, type=Path)
    ap.add_argument("--handle", help="explicit deployment handle (default: the spec's, else a random AdjectiveNoun)")
    ap.add_argument("--out-root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--operator-cidr", help="CIDR allowed SSH/Grafana/Prometheus (default: this machine's public IP /32)")
    ap.add_argument("--yes", action="store_true", help="consent to creating billable AWS resources")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and estimated cost; create nothing")
    ap.add_argument("--resume", action="store_true", help="continue an existing run (needs --handle)")
    ap.add_argument("--skip-quota-check", action="store_true")
    a = ap.parse_args(argv)

    try:
        spec = parse_spec(a.spec)
    except SpecError as e:
        print(e, file=sys.stderr)
        return BAD_INPUT
    if a.handle:
        spec.handle = a.handle

    if a.dry_run:
        p = planmod.build_plan(spec, operator_cidr=a.operator_cidr or "203.0.113.1/32")
        out(planmod.format_plan(p))
        return FAIL if p["problems"] else OK

    clients = clients or aws.Clients()
    try:
        from .deploy import check_spend_allowed
        check_spend_allowed(a.yes, environ)
        if a.resume:
            if spec.handle.lower() == "auto" or not RunDir(a.out_root, spec.handle).exists():
                print("--resume needs an existing run: pass --handle <Handle> of a directory under the out-root",
                      file=sys.stderr)
                return BAD_INPUT
        else:
            taken = existing_handles(a.out_root) | aws.live_handles(clients)
            spec = resolve_handle(spec, taken)
        rundir = RunDir(a.out_root, spec.handle)
        out(f"deployment handle: {spec.handle}")
        rundir.ensure()
        if not rundir.spec_path.exists():
            rundir.spec_path.write_text(a.spec.read_text())
        rundir.spec_json.write_text(json.dumps(spec.to_dict(), indent=1, sort_keys=True) + "\n")
        cidr = a.operator_cidr or aws.detect_operator_cidr()
        quotas = None if a.skip_quota_check else fetch_quotas(clients, spec.regions)
        factory = runner_factory or ssh_runner_factory(rundir)
        kw = {"sleep": sleep} if sleep else {}
        state = deploy(spec, rundir, clients, factory, cidr, quotas=quotas, yes=a.yes, environ=environ, log=out, **kw)
    except SpendNotAllowed as e:
        print(e, file=sys.stderr)
        return NO_SPEND
    except (DeployError, aws.AwsError) as e:
        print(f"deploy failed: {e}", file=sys.stderr)
        print(f"inspect or clean up with: teardown_infra.py {spec.handle}", file=sys.stderr)
        return FAIL
    out(f"deployed {spec.handle}: {len(state.instances)} instances, ~${state.est_hourly_usd:.2f}/h")
    out(f"tear down with: infra/tidb/teardown_infra.py {spec.handle}")
    return OK


# ------------------------------------------------------------------ teardown_infra
def teardown_main(argv: list[str] | None = None, clients: aws.Clients | None = None, input_fn=input, out=print,
                  sleep=None, lib=None) -> int:
    ap = argparse.ArgumentParser(prog="teardown_infra.py", description="List and delete deployments by handle.")
    ap.add_argument("handles", nargs="*", help="handle(s) to tear down")
    ap.add_argument("--all", action="store_true", help="every Project=powder deployment")
    ap.add_argument("--older-than", metavar="DURATION", help="deployments created more than e.g. 6h ago")
    ap.add_argument("--list", action="store_true", help="show deployments still holding AWS resources")
    ap.add_argument("--verify", action="store_true", help="exit non-zero if anything tagged Project=powder remains")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--yes", action="store_true", help="skip the typed confirmation")
    ap.add_argument("--out-root", type=Path, default=DEFAULT_ROOT)
    a = ap.parse_args(argv)
    clients = clients or aws.Clients()

    if a.list:
        out(td.format_list(td.discover(clients), a.out_root, lib=lib))
        return OK
    if a.verify and not (a.handles or a.all or a.older_than):
        left = td.verify(clients)
        out("clean: nothing tagged Project=powder remains" if not left else "LEFTOVERS:\n  " + "\n  ".join(left))
        return OK if not left else FAIL

    from .spec import parse_duration
    try:
        older = parse_duration(a.older_than) if a.older_than else None
    except SpecError as e:
        print(e, file=sys.stderr)
        return BAD_INPUT
    runs = td.discover(clients)
    try:
        chosen = td.select(runs, a.handles or None, a.all, older)
    except td.SelectionError as e:
        print(e, file=sys.stderr)
        return REFUSED
    if not chosen:
        out("nothing matches; nothing to do")
        return OK
    out("selected for teardown:")
    out(td.format_list({h: runs[h] for h in chosen}, a.out_root, lib=lib))
    if a.dry_run:
        return OK
    if not a.yes:
        answer = input_fn(f"type 'yes' to delete {len(chosen)} deployment(s): ").strip().lower()
        if answer != "yes":
            out("aborted")
            return REFUSED

    kw = {"sleep": sleep} if sleep else {}
    failed = False
    for h in chosen:
        out(f"tearing down {h} ...")
        res = td.teardown_run(clients, runs[h], log=out, **kw)
        failed |= bool(res["errors"])
        rd = RunDir(a.out_root, h)
        if rd.exists():
            rd.teardown_path.write_text(json.dumps({"handle": h, "at": aws.utcnow(), **res}, indent=1) + "\n")
            if rd.state_path.exists():
                st = rd.load_state()
                for rec in st.instances.values():
                    rec.state = "terminated"
                st.save(rd.state_path)
        out(f"  deleted: {res['deleted']}")
    left = [x for x in td.verify(clients) if any(x.startswith(h + " ") for h in chosen)]
    if left:
        out("LEFTOVERS:\n  " + "\n  ".join(left))
    return FAIL if (failed or left) else OK


# ------------------------------------------------------------------ install_tidb
def install_main(argv: list[str] | None = None, runner_factory=None, out=print, sleep=None) -> int:
    ap = argparse.ArgumentParser(prog="install_tidb.py", description="Install and start TiDB on a deployed run (TiUP).")
    ap.add_argument("--handle", required=True, help="the deployment to install on")
    ap.add_argument("--out-root", type=Path, default=DEFAULT_ROOT)
    a = ap.parse_args(argv)
    rd = RunDir(a.out_root, a.handle)
    if not rd.exists():
        print(f"no run directory {rd.path}: deploy first (deploy_infra.py)", file=sys.stderr)
        return BAD_INPUT
    try:
        spec = load_run_spec(rd)
        state = rd.load_state()
        runner = (runner_factory or ssh_runner_factory(rd))(state)
        kw = {"sleep": sleep} if sleep else {}
        install(spec, rd, runner, log=out, **kw)
    except (SpecError, InstallError, RemoteError) as e:
        print(f"install failed: {e}", file=sys.stderr)
        return FAIL if not isinstance(e, SpecError) else BAD_INPUT
    except Exception as e:                    # StateError etc.
        print(f"install failed: {e}", file=sys.stderr)
        return FAIL
    out(f"installed TiDB on {a.handle}")
    return OK
