"""Stage 1: create the run's AWS infrastructure (plan §6, §6.1). Library behind ``deploy_infra.py``.

Order: spend guard -> preflight (quotas, limits, offered AZs) -> key pair -> network + peering -> instances ->
wait running -> wait for SSH and ``bootstrap.done`` on every node -> ``state.json`` complete. Any failure leaves
``state.json`` describing exactly what exists and raises; **nothing is torn down here** (the orchestrator decides).
"""

from __future__ import annotations

import os
import time
from typing import Callable, Mapping

from . import aws, aws_net, aws_nodes, netplan, pricing
from .remote import RemoteError, Runner, wait_for
from .rundir import RunDir
from .spec import ClusterSpec, plan_nodes
from .state import RunState, utcnow

SPEND_ENV = "POWDER_ALLOW_SPEND"


class DeployError(RuntimeError):
    pass


class SpendNotAllowed(DeployError):
    pass


def check_spend_allowed(yes: bool, environ: Mapping[str, str] | None = None) -> None:
    """Creating AWS resources needs an explicit ``--yes`` or ``POWDER_ALLOW_SPEND=1`` (plan §9)."""
    environ = os.environ if environ is None else environ
    if not (yes or environ.get(SPEND_ENV) == "1"):
        raise SpendNotAllowed(
            "refusing to create AWS resources without explicit consent: pass --yes or set "
            f"{SPEND_ENV}=1. Review the estimate with --dry-run first.")


def fetch_quotas(clients: aws.Clients, regions: list[str]) -> dict[str, int]:
    """On-demand Standard vCPU quota per region (applied value, else the AWS default)."""
    out = {}
    svc, code = aws.EC2_STANDARD_VCPU_QUOTA
    for r in regions:
        sq = clients.get("service-quotas", r)
        try:
            q = sq.get_service_quota(ServiceCode=svc, QuotaCode=code)["Quota"]["Value"]
        except Exception:
            q = sq.get_aws_default_service_quota(ServiceCode=svc, QuotaCode=code)["Quota"]["Value"]
        out[r] = int(q)
    return out


def preflight(spec: ClusterSpec, est: pricing.CostEstimate, quotas: dict[str, int] | None) -> list[str]:
    problems = pricing.check_limits(est, spec.limits)
    if quotas is not None:
        problems += pricing.check_quotas(est, quotas)
    return problems


def wait_bootstrap(runner: Runner, state: RunState, timeout: float = 900.0, interval: float = 10.0,
                   sleep=time.sleep, log: Callable[[str], None] = lambda m: None) -> None:
    """Controller first (direct SSH), then every node over it: wait for /var/lib/powder/bootstrap.done."""
    order = ["controller"] + [n for n in state.instances if n != "controller"]
    for name in order:
        wait_for(lambda n=name: runner.run(n, "test -f /var/lib/powder/bootstrap.done", timeout=30, check=False).returncode == 0,
                 timeout=timeout, interval=interval, what=f"{name} bootstrap", sleep=sleep)
        log(f"  {name}: bootstrapped")


def deploy(spec: ClusterSpec, rundir: RunDir, clients: aws.Clients, runner_factory: Callable[[RunState], Runner],
           operator_cidr: str, quotas: dict[str, int] | None = None, yes: bool = False,
           environ: Mapping[str, str] | None = None, lib=None, log: Callable[[str], None] = print,
           sleep=time.sleep, status_checks: bool = True, bootstrap_timeout: float = 900.0) -> RunState:
    check_spend_allowed(yes, environ)
    nodes = plan_nodes(spec)
    est = pricing.estimate(spec, lib, nodes=nodes)
    problems = preflight(spec, est, quotas)
    if problems:
        raise DeployError("preflight failed:\n  " + "\n  ".join(problems))

    rundir.ensure()
    state = rundir.load_state() if rundir.state_path.exists() else RunState(spec.handle, spec.spec_hash())
    if state.spec_hash != spec.spec_hash():
        raise DeployError(f"{rundir.state_path} was created from a different spec ({state.spec_hash} != "
                          f"{spec.spec_hash()}); change slots through a new handle or edit the state deliberately")
    state.est_hourly_usd = est.hourly_total
    state.set_stage("infra", "partial")
    save = lambda: state.save(rundir.state_path)          # noqa: E731
    save()
    owner = aws.caller_owner(clients)
    ctx = aws_net.Ctx(clients, spec.handle, owner, state.created_at, state, save, sleep, log)

    log(f"[{spec.handle}] generating SSH key")
    public_key = aws_nodes.generate_ssh_key(rundir.key_path)
    state.key_path = str(rundir.key_path)

    azs = {r: aws_nodes.choose_azs(ctx, r, nodes) for r in spec.regions}
    plan = netplan.plan_network(spec.regions, azs, operator_cidr)
    problems = netplan.validate_plan(plan)
    if problems:
        raise DeployError("network plan invalid: " + "; ".join(problems))

    for r in spec.regions:
        aws_nodes.ensure_key_pair(ctx, r, public_key)
    log(f"[{spec.handle}] network: {len(plan.regions)} VPCs, {len(plan.peerings)} peerings")
    aws_net.ensure_network(ctx, plan)
    log(f"[{spec.handle}] launching {len(nodes)} instances (~${est.hourly_total:.2f}/h)")
    aws_nodes.launch_all(ctx, nodes, azs)
    log(f"[{spec.handle}] waiting for instances to run")
    aws_nodes.wait_running(ctx, status_checks=status_checks)

    runner = runner_factory(state)
    log(f"[{spec.handle}] waiting for SSH and node bootstrap")
    try:
        wait_bootstrap(runner, state, timeout=bootstrap_timeout, sleep=sleep, log=log)
    except RemoteError as e:
        state.set_stage("infra", "partial")
        save()
        raise DeployError(f"nodes did not finish bootstrapping: {e}") from e
    state.set_stage("infra", "done")
    save()
    rundir.update_manifest(handle=spec.handle, spec_hash=spec.spec_hash(), tidb_version=spec.tidb_version,
                           created_at=state.created_at, deployed_at=utcnow(), est_hourly_usd=est.hourly_total,
                           regions=spec.regions, azs=azs)
    return state
