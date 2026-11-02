"""Tests for infra/tidb/pin_versions.py (pure manifest logic; no network)."""

import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "pin_versions", Path(__file__).resolve().parents[1] / "infra" / "tidb" / "pin_versions.py"
)
pv = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pv)


def entry(sha="a" * 64, length=10, url="/x.tar.gz", yanked=False):
    return {"hashes": {"sha256": sha}, "length": length, "url": url, "yanked": yanked}


def manifest(entries, platform="linux/amd64"):
    return {"signed": {"platforms": {platform: entries}}}


def test_semver_accepts_plain_releases_only():
    assert pv.semver("v8.5.8") == (8, 5, 8)
    assert pv.semver("v9.0.0-beta.1") is None
    assert pv.semver("v9.0.0-beta.2.pre-nightly") is None
    assert pv.semver("8.5.8") is None


def test_latest_release_skips_yanked_prerelease_and_orders_numerically():
    entries = {
        "v1.9.6": entry(),
        "v1.17.1": entry(),          # numerically greatest, not lexicographically
        "v1.18.0": entry(yanked=True),
        "v2.0.0-beta.1": entry(),
    }
    assert pv.latest_release(entries) == "v1.17.1"


def test_latest_release_errors_when_nothing_released():
    with pytest.raises(ValueError):
        pv.latest_release({"v1.0.0": entry(yanked=True), "v2.0.0-rc.1": entry()})


def test_pick_entry_rejects_yanked_missing_and_unhashed():
    m = manifest({"v8.5.8": entry(yanked=True), "v8.5.9": entry()})
    with pytest.raises(ValueError, match="yanked"):
        pv.pick_entry(m, "pd", "v8.5.8")
    with pytest.raises(ValueError, match="not published"):
        pv.pick_entry(m, "pd", "v1.2.3")
    bad = manifest({"v8.5.8": {"url": "/x", "length": 1, "hashes": {}}})
    with pytest.raises(ValueError, match="sha256"):
        pv.pick_entry(bad, "pd", "v8.5.8")
    nolen = manifest({"v8.5.8": {"url": "/x", "hashes": {"sha256": "a"}}})
    with pytest.raises(ValueError, match="length"):
        pv.pick_entry(nolen, "pd", "v8.5.8")
    with pytest.raises(ValueError, match="platform"):
        pv.pick_entry(manifest({"v8.5.8": entry()}, platform="darwin/arm64"), "pd", "v8.5.8")


def build_manifests(tidb_version="v8.5.8"):
    ms = {}
    for i, c in enumerate(pv.AT_TIDB_VERSION):
        ms[c] = manifest({tidb_version: entry(sha=f"{i:064x}", url=f"/{c}-{tidb_version}.tar.gz")})
    for c in pv.AT_LATEST:
        ms[c] = manifest({"v1.16.5": entry(sha="b" * 64), "v1.17.1": entry(sha="c" * 64)})
    return ms


def test_rows_render_parse_round_trip_and_tiup_pin():
    rows = pv.rows_for("v8.5.8", build_manifests())
    assert [r[0] for r in rows] == pv.AT_TIDB_VERSION + pv.AT_LATEST
    assert {r[1] for r in rows if r[0] in pv.AT_TIDB_VERSION} == {"v8.5.8"}
    assert {r[1] for r in rows if r[0] in pv.AT_LATEST} == {"v1.17.1"}   # latest by default
    pinned = pv.rows_for("v8.5.8", build_manifests(), {"tiup": "v1.16.5"})
    assert dict((r[0], r[1]) for r in pinned)["tiup"] == "v1.16.5"       # explicit pin honoured
    text = pv.render(rows, 123, "2026-01-01")
    assert pv.parse(text) == rows


def test_parse_ignores_comments_and_rejects_malformed_lines():
    assert pv.parse("# only a comment\n\n") == []
    with pytest.raises(ValueError):
        pv.parse("pd v8.5.8 linux/amd64 deadbeef\n")


def test_committed_versions_file_is_well_formed():
    rows = pv.parse(pv.DEFAULT_OUT.read_text())
    by = {r[0]: r for r in rows}
    assert set(by) == set(pv.AT_TIDB_VERSION + pv.AT_LATEST)
    assert len({by[c][1] for c in pv.AT_TIDB_VERSION}) == 1       # one TiDB version across components
    for _, _, plat, sha, size, url in rows:
        assert plat == "linux/amd64" and len(sha) == 64 and int(sha, 16) >= 0
        assert size > 0 and url.startswith("/") and url.endswith(".tar.gz")
