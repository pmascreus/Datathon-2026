"""Fast unit tests for the pure decision core (no API, no artefacts on disk).

These guard the invariants that matter under deadline pressure: a list is always
exactly 10 distinct available products, repairs never lose validity, and the MMR
head-preselect does not change the top slot (the one that matters most for
purchases).

    python -m pytest tests -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import PHASE_POLICIES, Config
from src.policy import fallback_items, recommend
from src.retrieval import Session
from src.state import static_snapshot
from src.store import BanditState, Item, Models

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from serve import SeenJournal, validate  # noqa: E402


def _toy_models(n_categories: int = 10, n_items: int = 500) -> Models:
    catalog = {}
    popularity = {}
    by_category: dict[str, list[str]] = {}
    by_subcategory: dict[str, list[str]] = {}
    for i in range(n_items):
        pid = f"p_{i:04d}"
        cat = f"c{i % n_categories}"
        sub = f"s{i % (n_categories * 3)}"
        catalog[pid] = Item(
            product_id=pid, category=cat, subcategory=sub, brand=f"b{i % 7}",
            price_eur=10.0 + i % 50, is_active=True, created_ts=1_700_000_000.0,
            text=f"{cat} {sub} b{i % 7}",
        )
        popularity[pid] = 1.0 / (i + 2)
        by_category.setdefault(cat, []).append(pid)
        by_subcategory.setdefault(sub, []).append(pid)
    popular_sorted = sorted(catalog, key=lambda p: (-popularity[p], p))
    return Models(
        tag="toy", directory=ROOT / "models" / "toy", catalog=catalog,
        base_ids=list(catalog), popularity=popularity,
        pop_rank={p: i for i, p in enumerate(popular_sorted)},
        category_rate={}, item_item=None, als=None, content=None,
        bandit=BanditState(), profiles={}, recent_purchases={}, purchased_items={},
        ranker=None, ranker_features=[], ranker_calibration=None, meta={},
        price_median=30.0, by_category=by_category, by_subcategory=by_subcategory,
        popular_sorted=popular_sorted, max_popularity=max(popularity.values()),
    )


def _session(user: str = "u1", views: tuple[str, ...] = ()) -> Session:
    return Session(
        request_id="r1", user_id=user, is_new_user=False, ts=1_700_000_000.0,
        device="web", region="Catalunya", page="home",
        views=tuple((pid, 5.0 * k) for k, pid in enumerate(views)), request={},
    )


@pytest.mark.parametrize("name", ["p0_baseline", "p1_item_item", "p2_als", "p3_content", "p_fast", "default"])
def test_every_policy_returns_ten_distinct_available(name: str) -> None:
    models = _toy_models()
    live = static_snapshot(models)
    cfg = PHASE_POLICIES[name] if name in PHASE_POLICIES else Config()
    decision = recommend(_session(views=("p_0001", "p_0002")), models, live, cfg,
                         np.random.default_rng(0))
    assert len(decision.items) == 10
    assert len(set(decision.items)) == 10
    assert all(pid in live.available for pid in decision.items)
    assert decision.source == "model", decision.notes


def test_mmr_pool_keeps_the_top_slot() -> None:
    """Preselecting the MMR pool must not change the most valuable position."""
    models = _toy_models()
    live = static_snapshot(models)
    session = _session(views=("p_0001", "p_0002", "p_0003"))
    cfg_full = Config(mmr_lambda=0.16, mmr_pool=0)
    cfg_head = Config(mmr_lambda=0.16, mmr_pool=40)
    a = recommend(session, models, live, cfg_full, np.random.default_rng(0))
    b = recommend(session, models, live, cfg_head, np.random.default_rng(0))
    assert a.items[0] == b.items[0]
    assert len(b.items) == 10


def test_validate_repairs_invalid_and_duplicate_items() -> None:
    models = _toy_models(n_items=12)
    live = static_snapshot(models)
    raw = ["p_0000", "p_0000", "missing", "p_0001"]
    items, repaired = validate(raw, models, live)
    assert repaired is True
    assert len(items) == 10
    assert len(set(items)) == 10
    assert all(pid in live.available for pid in items)


def test_fallback_is_always_valid() -> None:
    models = _toy_models(n_items=8)
    live = static_snapshot(models)
    items = fallback_items(models, live)
    assert len(items) == 8  # catalogue smaller than 10: still the best we can do
    assert len(set(items)) == 8


def test_seen_journal_survives_reload(tmp_path) -> None:
    path = tmp_path / "seen.txt"
    journal = SeenJournal(path=path)
    journal.add("r_a")
    journal.add("r_b")
    journal.add("r_a")
    journal.flush()
    again = SeenJournal(path=path)
    assert again.seen("r_a") and again.seen("r_b")
    assert not again.seen("r_c")
    assert len(again) == 2
