"""The clock offset that made `submit_age_ms` look wrong in both directions.

Request timestamps are stamped by the platform, and this laptop's clock drifts.
Reporting an age without that offset understates our latency (it was negative at
p50 on 2026-11-15), and keeping a stale offset overstates it (it made every
request look 1 s older, so the loop degraded and skipped requests that were on
time).

Two properties matter:

* the measurement is robust to the round trip (bracket + midpoint) and reports
  failure as `None`, never as `0.0` (which means "clocks agree");
* `_refresh_clock` stores whatever it measures, unconditionally, and keeps the
  previous value when the measurement fails.
"""
from __future__ import annotations

import sys
from types import SimpleNamespace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import common as util
from src.state import LiveState


class _Stub:
    def __init__(self, remote: float | None, fail: bool = False) -> None:
        self.remote = remote
        self.fail = fail
        self.calls = 0

    def fetch(self, query: str):
        self.calls += 1
        if self.fail:
            raise RuntimeError("api down")
        assert query == "clock_now"
        return [{"ms": int(self.remote * 1000)}]


def test_offset_is_remote_minus_local(monkeypatch) -> None:
    monkeypatch.setattr(util.time, "time", lambda: 1_000_000.0)
    src = _Stub(remote=1_000_001.25)
    assert util.measure_clock_offset(src) == pytest.approx(1.25, abs=1e-6)


def test_single_sample(monkeypatch) -> None:
    monkeypatch.setattr(util.time, "time", lambda: 2_000_000.0)
    src = _Stub(remote=1_999_998.5)
    assert util.measure_clock_offset(src, samples=1) == pytest.approx(-1.5, abs=1e-6)
    assert src.calls == 1


def test_failure_is_none_not_zero(monkeypatch) -> None:
    monkeypatch.setattr(util.time, "time", lambda: 1_000_000.0)
    assert util.measure_clock_offset(_Stub(remote=None, fail=True)) is None


def _live(offset_source: _Stub) -> SimpleNamespace:
    # `_refresh_clock` only touches these attributes.
    return SimpleNamespace(
        source=offset_source,
        verbose=False,
        _last_clock=0.0,
        clock_offset_s=1.01,
        _stats={"clock_failures": 0},
        _stats_lock=__import__("threading").Lock(),
    )


def test_refresh_clock_tracks_a_shrinking_offset(monkeypatch) -> None:
    """Regression: the offset must update when it gets *smaller*.

    The first implementation only stored `|offset| > 0.05`, so after Windows
    Time resynced the machine the stale +1 s stayed applied for six hours: every
    age was inflated by a second and requests were degraded/skipped for nothing.
    """
    monkeypatch.setattr(util.time, "time", lambda: 1_000_000.0)
    stub = _live(_Stub(remote=1_000_000.02))
    LiveState._refresh_clock(stub)
    assert stub.clock_offset_s == pytest.approx(0.02, abs=1e-6)


def test_refresh_clock_keeps_previous_on_failure(monkeypatch) -> None:
    monkeypatch.setattr(util.time, "time", lambda: 1_000_000.0)
    stub = _live(_Stub(remote=None, fail=True))
    LiveState._refresh_clock(stub)
    assert stub.clock_offset_s == pytest.approx(1.01)
    assert stub._stats["clock_failures"] == 1


def test_refresh_clock_ignores_implausible_offset(monkeypatch) -> None:
    monkeypatch.setattr(util.time, "time", lambda: 1_000_000.0)
    stub = _live(_Stub(remote=1_000_030.0))
    LiveState._refresh_clock(stub)
    assert stub.clock_offset_s == pytest.approx(1.01)
    assert stub._stats["clock_failures"] == 1
