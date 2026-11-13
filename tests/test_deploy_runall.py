"""run_all: teardown is guaranteed on success, failure, interruption and a half-failed deploy (exec #7 step 6)."""

import json
import os
import signal

import pytest

moto = pytest.importorskip("moto")
import boto3  # noqa: E402

from powder.deploy import aws, aws_nodes, cli, runall, teardown as td  # noqa: E402
from powder.deploy.remote import Result  # noqa: E402
from powder.deploy.rundir import RunDir  # noqa: E402
from powder.deploy.spec import ExperimentConfig, parse_spec  # noqa: E402
import test_deploy_aws as T  # noqa: E402  (moto fixtures and helpers shared with the deploy tests)

aws_env = T.aws_env


class Rec:
    def __init__(self):
        self.calls = []


def stages(rec, install=None, experiment=None):
    def inst(spec, rd, runner):
        rec.calls.append("install")
        if install:
            install()
        st = rd.load_state()
        st.set_stage("install", "done")
        st.save(rd.state_path)

    def exp(spec, cfg, rd, runner):
        rec.calls.append(f"experiment:{cfg.name}")
        if experiment:
            experiment()
    return runall.Stages(install=inst, experiment=exp)


def go(aws_env, rec=None, install=None, experiment=None, keep=False, handle="OrangePanda", **kw):
    spec = T.small_spec(handle)
    rd = RunDir(aws_env / "out", handle)
    out = []
    rec = rec or Rec()
    st = stages(rec, install, experiment)
    base = dict(clients=aws.Clients(), runner_factory=lambda s: T.FakeRunner(), operator_cidr="203.0.113.7/32", yes=True,
                keep_infra=keep, out=out.append, sleep=lambda s: None, stages=st)
    base.update(kw)
    code = runall.run_all(spec, [ExperimentConfig(name="smoke")], rd, **base)
    return code, out, rec, rd


def live(clients=None):
    return td.discover(clients or aws.Clients())


@pytest.fixture(autouse=True)
def fast_deploy(monkeypatch):
    # run_all's default deploy waits on real AWS status checks; the moto tests skip those
    orig = runall._deploy
    monkeypatch.setattr(runall, "_deploy", lambda *a, **k: orig(*a, status_checks=False, **k))


def test_success_runs_stages_in_order_then_tears_everything_down(aws_env):
    code, out, rec, rd = go(aws_env)
    assert code == 0 and rec.calls == ["install", "experiment:smoke"]
    assert live() == {} and json.loads(rd.teardown_path.read_text())["errors"] == []
    assert out[0] == "deployment handle: OrangePanda" and out[-1].startswith("deployment handle: OrangePanda  exit=0")
    assert any("torn down" in l for l in out)


def test_stage_failure_still_tears_down_and_reports(aws_env):
    def boom():
        raise RuntimeError("tiup deploy exploded")
    code, out, rec, rd = go(aws_env, install=boom)
    assert code == 1 and rec.calls == ["install"]                                   # experiments never ran
    assert live() == {}
    assert any("FAILED: RuntimeError: tiup deploy exploded" in l for l in out) and out[-1].endswith("(RuntimeError: tiup deploy exploded)")


def test_half_failed_deploy_leaves_nothing_billing(aws_env, monkeypatch):
    real, n = aws_nodes.launch_node, {"c": 0}

    def flaky(ctx, node, *a, **k):
        n["c"] += 1
        if n["c"] == 5:
            raise RuntimeError("capacity")
        return real(ctx, node, *a, **k)
    monkeypatch.setattr(aws_nodes, "launch_node", flaky)
    code, out, rec, rd = go(aws_env)
    assert code == 1 and rec.calls == []                                            # never got past deploy
    assert live() == {}                                                              # but the VPCs and 4 instances are gone


def test_ctrl_c_during_an_experiment_tears_down(aws_env):
    def sigint():
        raise KeyboardInterrupt
    code, out, rec, rd = go(aws_env, experiment=sigint)
    assert code == runall.INTERRUPTED == 130 and live() == {}
    assert any("interrupted: tearing down" in l for l in out)


def test_sigterm_is_handled_like_ctrl_c(aws_env):
    code, out, rec, rd = go(aws_env, experiment=lambda: os.kill(os.getpid(), signal.SIGTERM))
    assert code == 130 and live() == {}


def test_a_second_interrupt_during_teardown_does_not_abort_it(aws_env, monkeypatch):
    real = td.teardown_run

    def hit_while_tearing_down(*a, **k):
        os.kill(os.getpid(), signal.SIGINT)                                          # impatient user presses Ctrl-C again
        os.kill(os.getpid(), signal.SIGTERM)
        return real(*a, **k)
    monkeypatch.setattr(td, "teardown_run", hit_while_tearing_down)
    code, out, rec, rd = go(aws_env, experiment=lambda: (_ for _ in ()).throw(KeyboardInterrupt))
    assert code == 130 and live() == {}                                              # cleanup completed anyway


def test_signal_handlers_are_restored(aws_env):
    before = signal.getsignal(signal.SIGINT)
    go(aws_env)
    assert signal.getsignal(signal.SIGINT) is before


def test_keep_infra_leaves_it_up_and_says_how_to_remove_it(aws_env):
    code, out, rec, rd = go(aws_env, keep=True)
    assert code == 0 and live()["OrangePanda"].instance_count() == 16
    msg = next(l for l in out if "--keep-infra" in l)
    assert "teardown_infra.py OrangePanda" in msg and "/h." in msg and msg.split("~$")[1][0].isdigit()   # states the burn


def test_incomplete_teardown_is_loud_even_when_the_experiment_succeeded(aws_env, monkeypatch):
    monkeypatch.setattr(td, "teardown_run", lambda *a, **k: {"deleted": {}, "errors": ["us-east-1 delete VPC: DependencyViolation"]})
    code, out, rec, rd = go(aws_env)
    assert code == 1 and rec.calls == ["install", "experiment:smoke"]
    msg = " ".join(out)
    assert "TEARDOWN INCOMPLETE" in msg and "teardown_infra.py OrangePanda" in msg


def test_teardown_exception_is_reported_not_raised(aws_env, monkeypatch):
    monkeypatch.setattr(td, "teardown_run", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("api down")))
    code, out, rec, rd = go(aws_env)
    assert code == 1 and any("teardown error: api down" in l for l in out)


def test_spend_guard_blocks_before_anything_is_created(aws_env):
    spec = T.small_spec("OrangePanda")
    with pytest.raises(Exception, match="refusing"):
        runall.run_all(spec, [], RunDir(aws_env / "out", "OrangePanda"), aws.Clients(), lambda s: T.FakeRunner(),
                       "203.0.113.7/32", yes=False, environ={}, out=lambda m: None)
    assert live() == {}


def test_existing_install_is_skipped_on_resume(aws_env):
    rec = Rec()
    go(aws_env, rec=rec, keep=True)
    assert rec.calls == ["install", "experiment:smoke"]
    rec2 = Rec()
    go(aws_env, rec=rec2, keep=True)                                                  # same handle, install already done
    assert rec2.calls == ["experiment:smoke"]


# ------------------------------------------------------------------ CLI
def write_spec(aws_env):
    import yaml
    d = T.small_spec("auto").to_dict()
    f = aws_env / "spec.yaml"
    f.write_text(yaml.safe_dump({"slots": {s["name"]: {"machine": f"{s['machine']}@{s['region']}"} for s in d["slots"]},
                                 "limits": {"max_hourly_usd": 50}}))
    return f


def test_cli_requires_consent_and_an_experiment(aws_env, monkeypatch):
    monkeypatch.setattr(aws, "detect_operator_cidr", lambda: "203.0.113.7/32")
    f, root = write_spec(aws_env), str(aws_env / "out")
    exp = "infra/tidb/experiments/smoke.yaml"
    assert cli.run_all_main(["--spec", str(f), "--out-root", root], out=lambda m: None) == cli.BAD_INPUT          # no experiment
    assert cli.run_all_main(["--spec", str(f), "--experiment", exp, "--out-root", root], environ={}, out=lambda m: None) == cli.NO_SPEND
    assert live() == {}


def test_cli_end_to_end_with_teardown(aws_env, monkeypatch):
    monkeypatch.setattr(aws, "detect_operator_cidr", lambda: "203.0.113.7/32")
    f, root = write_spec(aws_env), str(aws_env / "out")
    rec, out = Rec(), []
    code = cli.run_all_main(["--spec", str(f), "--experiment", "infra/tidb/experiments/smoke.yaml", "--yes", "--out-root", root,
                             "--skip-quota-check"], clients=aws.Clients(), runner_factory=lambda s: T.FakeRunner(),
                            out=out.append, sleep=lambda s: None, stages=stages(rec))
    assert code == 0 and rec.calls == ["install", "experiment:smoke"] and live() == {}
    handle = out[0].split(": ")[1]
    assert (aws_env / "out" / handle / "spec.yaml").exists() and (aws_env / "out" / handle / "teardown.json").exists()


def test_cli_dry_run_creates_nothing(aws_env):
    out = []
    assert cli.run_all_main(["--spec", "infra/tidb/specs/smoke.yaml", "--experiment", "infra/tidb/experiments/smoke.yaml",
                             "--dry-run"], out=out.append) == 0
    assert "AWS bill" in out[0] and any("experiment smoke: 600s" in l for l in out) and live() == {}
