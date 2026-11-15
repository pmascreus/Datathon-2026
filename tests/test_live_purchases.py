"""Unit tests for the live user-level purchase feed (no API, no artefacts).

The feed exists because `team_25.feedback` has no `user_id`: without it the
per-user state (`purchased_before`, `recent_purchase_age_h`, the short
re-purchase penalty) is frozen at the training cut for the whole round. These
tests pin the three properties the serving loop relies on:

* a purchase is dated by `last_ts`, and ages are measured against an explicit
  reference time, never the wall clock (that is what makes the offline replay
  honest);
* a product that is not in the catalogue is ignored instead of crashing;
* a failed refresh keeps the previous state and increments a counter.

    python -m pytest tests -q
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import Config
from src.state import LiveState
from src.store import BanditState, Item, Models


def _toy_models() -> Models:
    catalog = {
        f"p_{i:03d}": Item(
            product_id=f"p_{i:03d}", category="c", subcategory="s", brand="b",
            price_eur=10.0 + i, is_active=True, created_ts=1_700_000_000.0,
            text=f"c s b {i}",
        )
        for i in range(20)
    }
    popularity = {pid: 1.0 / (i + 2) for i, pid in enumerate(catalog)}
    popular_sorted = sorted(catalog, key=lambda p: (-popularity[p], p))
    return Models(
        tag="toy", directory=ROOT / "models" / "toy", catalog=catalog,
        base_ids=list(catalog), popularity=popularity,
        pop_rank={p: i for i, p in enumerate(popular_sorted)},
        category_rate={}, item_item=None, als=None, content=None,
        bandit=BanditState(), profiles={}, recent_purchases={}, purchased_items={},
        ranker=None, ranker_features=[], ranker_calibration=None, meta={},
        price_median=30.0, popular_sorted=popular_sorted,
        max_popularity=max(popularity.values()),
    )


def test_ingest_dates_purchase_at_its_own_timestamp() -> None:
    models = _toy_models()
    now = 1_700_100_000.0
    added = models.ingest_purchases(
        [{"user_id": "u1", "product_id": "p_001", "last_ts": "2023-11-14 21:00:00"}],
        now=now,
    )
    assert added == 1
    recent = models.purchased_recently("u1", now)
    assert "p_001" in recent
    assert recent["p_001"] > 1.0  # hours of age, measured against `now`
    assert "p_001" in models.purchased_items["u1"]


def test_ingest_ignores_unknown_products_and_is_idempotent() -> None:
    models = _toy_models()
    rows = [
        {"user_id": "u1", "product_id": "nope", "last_ts": "2023-11-14 21:00:00"},
        {"user_id": "", "product_id": "p_002", "last_ts": "2023-11-14 21:00:00"},
        {"user_id": "u1", "product_id": "p_002", "last_ts": "2023-11-14 21:00:00"},
    ]
    assert models.ingest_purchases(rows, now=1_700_100_000.0) == 1
    assert models.ingest_purchases(rows, now=1_700_100_000.0) == 0
    assert set(models.purchased_items["u1"]) == {"p_002"}
    assert "nope" not in models.recent_purchases.get("u1", {})


def test_ingest_keeps_the_newest_timestamp_for_a_pair() -> None:
    models = _toy_models()
    now = 1_700_100_000.0
    models.ingest_purchases(
        [{"user_id": "u1", "product_id": "p_003", "last_ts": "2023-11-10 00:00:00"}], now=now
    )
    first = models.recent_purchases["u1"]["p_003"]
    models.ingest_purchases(
        [{"user_id": "u1", "product_id": "p_003", "last_ts": "2023-11-14 00:00:00"}], now=now
    )
    assert models.recent_purchases["u1"]["p_003"] > first


def test_purchase_outside_the_window_is_hidden_from_the_penalty() -> None:
    models = _toy_models()
    now = 1_700_100_000.0
    models.ingest_purchases(
        [{"user_id": "u1", "product_id": "p_004", "last_ts": "2023-10-01 00:00:00"}],
        now=now,
        window_h=168.0,
    )
    # 45 days old: pruned for the touched user, so the penalty cannot fire.
    assert models.recent_purchases.get("u1", {}) == {}


class _FakeSource:
    name = "fake"

    def __init__(self, rows: list[dict], boom: bool = False) -> None:
        self.rows = rows
        self.boom = boom

    def fetch(self, query: str, **params: object) -> list[dict]:
        if self.boom:
            raise RuntimeError("api down")
        return list(self.rows)

    def fetch_df(self, query: str, **params: object):  # pragma: no cover - not used here
        raise NotImplementedError

    def insert(self, table: str, rows: list[dict]) -> None:  # pragma: no cover
        raise NotImplementedError


def test_refresh_failure_keeps_previous_state_and_counts() -> None:
    models = _toy_models()
    rows = [{"user_id": "u1", "product_id": "p_005", "last_ts": time.time() - 3600}]
    live = LiveState(_FakeSource(rows), models, Config(), verbose=False)
    live._refresh_purchases()
    assert "p_005" in models.recent_purchases["u1"]
    assert live.stats()["rows"] == 1

    live.source = _FakeSource(rows, boom=True)
    live._refresh_purchases()
    assert live.stats()["failures"] == 1
    assert "p_005" in models.recent_purchases["u1"]
