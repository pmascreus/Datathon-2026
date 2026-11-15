"""The evaluation sample must be reproducible, and comparisons must refuse to lie.

Two properties are load-bearing for every number in `docs/`:

* a fixed `(cut, salt, limit)` asks the API for the *same* sessions, so a
  baseline and a candidate bundle see identical rows;
* `compare_runs.py` refuses to pair two reports that sampled different
  sessions, because that difference would be sampling noise, not a gain.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import eval_sessions

import compare as compare_runs


class _Stub:
    name = "rawtree"

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def fetch_df(self, query: str, **params):
        self.calls.append((query, params))
        return pd.DataFrame()


def test_salt_selects_the_deterministic_query() -> None:
    src = _Stub()
    eval_sessions(src, "2026-11-14", limit=6000, salt="v1")
    query, params = src.calls[0]
    assert query == "history_sessions_test_salted"
    assert params["salt"] == "v1"
    assert params["limit"] == 6000


def test_no_salt_keeps_the_random_query() -> None:
    src = _Stub()
    eval_sessions(src, "2026-11-14", limit=6000)
    assert src.calls[0][0] == "history_sessions_test"


def test_salt_is_sanitised_and_never_empty() -> None:
    src = _Stub()
    eval_sessions(src, "2026-11-14", limit=10, salt="v1'; DROP TABLE x")
    assert src.calls[0][1]["salt"] == "v1DROPTABLEx"
    with pytest.raises(ValueError):
        eval_sessions(_Stub(), "2026-11-14", limit=10, salt="!!!")


def _report(models: str, salt: str, rows: dict[str, list]) -> dict:
    return {
        "cut": "2026-11-14", "sample_salt": salt, "n_sessions": len(rows),
        "models_tag": models,
        "policies": {"deployed": {"rows": rows}},
    }


def _write(tmp_path: Path, name: str, payload: dict) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_compare_pairs_by_request_id(tmp_path: Path, monkeypatch, capsys) -> None:
    a = _report("a", "v1", {"r1": [1.0, 1.0, 1.0], "r2": [0.0, 0.0, 0.0]})
    b = _report("b", "v1", {"r1": [0.0, 0.0, 0.0], "r2": [0.0, 0.0, 0.0]})
    ref, cand = _write(tmp_path, "a.json", a), _write(tmp_path, "b.json", b)
    monkeypatch.setattr(sys, "argv", ["compare_runs", str(ref), str(cand), "--boot", "50"])
    compare_runs.main()
    out = capsys.readouterr().out
    assert "sessions=2" in out
    assert "| purchase_hit |" in out


def test_compare_refuses_different_samples(tmp_path: Path, monkeypatch) -> None:
    a = _report("a", "v1", {"r1": [1.0, 1.0, 1.0]})
    b = _report("b", "v2", {"r1": [1.0, 1.0, 1.0]})
    ref, cand = _write(tmp_path, "a.json", a), _write(tmp_path, "b.json", b)
    monkeypatch.setattr(sys, "argv", ["compare_runs", str(ref), str(cand)])
    with pytest.raises(SystemExit) as exc:
        compare_runs.main()
    assert "did not sample the same sessions" in str(exc.value)


def test_compare_refuses_missing_rows(tmp_path: Path, monkeypatch) -> None:
    a = _report("a", "v1", {"r1": [1.0, 1.0, 1.0]})
    b = {"cut": "2026-11-14", "sample_salt": "v1", "n_sessions": 1,
         "models_tag": "b", "policies": {"deployed": {}}}
    ref, cand = _write(tmp_path, "a.json", a), _write(tmp_path, "b.json", b)
    monkeypatch.setattr(sys, "argv", ["compare_runs", str(ref), str(cand)])
    with pytest.raises(SystemExit) as exc:
        compare_runs.main()
    assert "--keep-rows" in str(exc.value)
