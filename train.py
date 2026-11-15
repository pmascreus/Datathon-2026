"""Build every artefact the recommender needs, for one temporal cut.

    python training/train_model.py                       # production tag "prod"
    python training/train_model.py --until 2026-10-23    # tag "2026-10-23" (offline eval)

What is written to `models/<tag>/`:

    catalog.pkl        catalogue as of the cut, with the text used by content
    popularity.pkl     purchase-count score, rank, per-subcategory base rate
    item_item.pkl      Phase 1 co-view / co-cart / co-purchase matrices
    als.pkl            Phase 2 implicit factors (users + items)
    content.pkl        Phase 3 TF-IDF -> SVD item embeddings
    users.pkl          long-term profile, recent purchases, purchased items
    bandit.pkl         Phase 7 Beta priors/counts from historical impressions
    ranker.txt         Phase 4 LightGBM P(purchase | impression)
    ranker_meta.pkl    feature names, calibrator, validation metrics
    meta.json          tag, cut, dataset sizes, timings

Temporal discipline (the only rule that matters for the jury)
-------------------------------------------------------------
* Artefacts for tag `C` use interactions with `ts < C`.
* The ranker is trained on sessions in `[C - 7d, C)`; every feature for those
  sessions is built with artefacts trained on interactions `< C - 7d` (and a
  catalogue as of `C`, exactly like serving, where the catalogue is always
  fresher than the interaction models). Therefore no session ever sees its own
  outcome or another session from its own week.
* Evaluation uses tag `C` on sessions `>= C`, never touched by training.

No invented columns: only the `shop` tables documented in the official README,
read through the SQL in `src/queries.py`.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# implicit's CPU ALS warns (and slows down) when OpenBLAS spawns its own pool.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src import data as datasets
from src.config import Config
from src.data import Source, make_source
from src.ranking import build_training_set, train_ranker
from src.state import static_snapshot
from src.store import (
    AlsModel, BanditState, CHANNELS, ContentModel, DEFAULT_TOP_K, Item,
    ItemItemModel, Models, _catalog_from_frame,
)

MODELS_DIR = ROOT / "models"

EVENT_WEIGHT = {"view": 1.0, "add_to_cart": 3.0, "purchase": 6.0}
PRIOR_STRENGTH = 40.0
RECENT_PURCHASE_DAYS = 30


@dataclass
class Components:
    tag: str
    cut: str | None
    catalog_df: pd.DataFrame
    catalog: dict[str, Item]
    base_ids: list[str]
    popularity: dict[str, float]
    pop_rank: dict[str, int]
    category_rate: dict[str, float]
    item_item: ItemItemModel | None
    als: AlsModel | None
    content: ContentModel | None
    profiles: dict[str, list[tuple[str, float]]]
    recent_purchases: dict[str, dict[str, float]]
    purchased_items: dict[str, list[str]]
    bandit: BanditState
    timings: dict[str, float] = field(default_factory=dict)
    sizes: dict[str, int] = field(default_factory=dict)
    dataset: str = "rawtree-real"
    ranker: Any | None = None
    ranker_metrics: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Component builders
# ---------------------------------------------------------------------------

def build_components(
    source: Source,
    *,
    tag: str,
    interaction_cut: str | None,
    catalog_cut: str | None,
    cfg: Config,
    seed: int = 0,
    verbose: bool = True,
    user_item_top_k: int = 15,
    profile_days: int = 30,
    prior_days: int = 14,
) -> Components:
    timings: dict[str, float] = {}
    sizes: dict[str, int] = {}

    t0 = time.time()
    catalog_df = datasets.catalog(source, before=catalog_cut)
    if catalog_df.empty:
        raise RuntimeError("empty catalogue: cannot train")
    catalog = _catalog_from_frame(catalog_df)
    base_ids = list(catalog.keys())
    timings["catalog_s"] = time.time() - t0
    sizes["catalog"] = len(base_ids)

    # The interaction window is explicit: `now` in production, the cut date for
    # an offline tag. `now()` inside the SQL would silently leak the future into
    # an offline training run, so it is never used.
    end_dt = pd.Timestamp(interaction_cut, tz="UTC") if interaction_cut else pd.Timestamp.now(tz="UTC")
    end = end_dt.isoformat()
    start_profile = (end_dt - pd.Timedelta(days=profile_days)).isoformat()
    start_prior = (end_dt - pd.Timedelta(days=prior_days)).isoformat()

    t0 = time.time()
    user_item = datasets.user_item(source, start=start_profile, end=end, top_k=user_item_top_k)
    timings["history_s"] = time.time() - t0
    sizes["user_item"] = int(len(user_item))

    # ---- popularity + category priors -------------------------------------
    t0 = time.time()
    counts = datasets.item_counts(source, before=interaction_cut)
    popularity, pop_rank = _popularity(counts, base_ids)
    category_rate = _category_rate(counts, catalog_df)
    timings["popularity_s"] = time.time() - t0

    # ---- item-item ---------------------------------------------------------
    t0 = time.time()
    cooc = {
        ch: datasets.cooccurrence(source, ch, before=interaction_cut, top_k=DEFAULT_TOP_K)
        for ch in CHANNELS
    }
    item_item = ItemItemModel.build(cooc, base_ids, top_k=DEFAULT_TOP_K)
    timings["item_item_s"] = time.time() - t0
    sizes["cooccurrence_rows"] = int(sum(len(df) for df in cooc.values()))

    # ---- ALS ---------------------------------------------------------------
    t0 = time.time()
    als = AlsModel.build(
        user_item, base_ids,
        factors=cfg_als_factors(), iterations=cfg_als_iterations(),
        regularization=0.05, alpha=15.0, seed=seed, verbose=False,
    )
    timings["als_s"] = time.time() - t0
    sizes["als_users"] = len(als.user_ids)

    # ---- content -----------------------------------------------------------
    t0 = time.time()
    content = ContentModel.build(catalog_df, seed=seed)
    timings["content_s"] = time.time() - t0

    # ---- user state --------------------------------------------------------
    t0 = time.time()
    profiles, purchased_items = _profiles(user_item)
    recent_purchases = _recent_purchases_frame(
        datasets.recent_purchases(source, start=start_profile, end=end)
    )
    timings["users_s"] = time.time() - t0
    sizes["profiles"] = len(profiles)

    # ---- bandit priors -----------------------------------------------------
    t0 = time.time()
    stats = datasets.impression_stats(source, start=start_prior, end=end)
    bandit = _bandit_priors_from_stats(stats, catalog_df)
    timings["bandit_s"] = time.time() - t0
    try:
        sizes.update({f"session_{k}": v for k, v in
                      datasets.session_stats(source, start=start_prior, end=end).items()})
    except Exception:
        pass

    if verbose:
        print(
            f"[train:{tag}] items={len(base_ids)} user_item={len(user_item)} "
            f"users={len(profiles)} timings={{{', '.join(f'{k}={v:.1f}' for k, v in timings.items())}}}"
        )
    return Components(
        tag=tag, cut=interaction_cut, catalog_df=catalog_df, catalog=catalog,
        base_ids=base_ids, popularity=popularity, pop_rank=pop_rank,
        category_rate=category_rate, item_item=item_item, als=als, content=content,
        profiles=profiles, recent_purchases=recent_purchases,
        purchased_items=purchased_items, bandit=bandit,
        timings=timings, sizes=sizes,
        dataset="rawtree-real",
    )


def cfg_als_factors() -> int:
    return 48


def cfg_als_iterations() -> int:
    return 12


def _popularity(counts: pd.DataFrame, base_ids: list[str]) -> tuple[dict[str, float], dict[str, int]]:
    if counts.empty:
        return {p: 0.0 for p in base_ids}, {p: i for i, p in enumerate(base_ids)}
    counts_map = {str(p): int(n) for p, n in zip(counts.product_id, counts.purchases)}
    ordered = counts.sort_values(["purchases", "product_id"], ascending=[False, False])
    rank = {str(p): i for i, p in enumerate(ordered.product_id)}
    popularity = {
        p: (1.0 / (rank[p] + 1) if counts_map.get(p, 0) > 0 else 0.0)
        for p in base_ids
    }
    # Products with no purchase keep a stable tail rank.
    for p in base_ids:
        rank.setdefault(p, len(rank))
    return popularity, rank


def _category_rate(counts: pd.DataFrame, catalog_df: pd.DataFrame) -> dict[str, float]:
    if counts.empty:
        return {}
    merged = counts.merge(catalog_df[["product_id", "subcategory"]], on="product_id", how="left")
    merged = merged.dropna(subset=["subcategory"])
    g = merged.groupby("subcategory")[["views", "purchases"]].sum()
    return {
        str(sub): float((row.purchases + 2.0) / (row.views + PRIOR_STRENGTH))
        for sub, row in g.iterrows()
    }


def _profiles(user_item: pd.DataFrame) -> tuple[dict[str, list[tuple[str, float]]], dict[str, list[str]]]:
    if user_item.empty:
        return {}, {}
    df = user_item.copy()
    df["weight"] = (
        df["views"] * EVENT_WEIGHT["view"]
        + df["carts"] * EVENT_WEIGHT["add_to_cart"]
        + df["purchases"] * EVENT_WEIGHT["purchase"]
    ).astype(float)
    df["user_id"] = df["user_id"].astype(str)
    df["product_id"] = df["product_id"].astype(str)
    df = df[df.weight > 0]
    if df.empty:
        return {}, {}
    df = df.sort_values(["user_id", "weight", "last_ts"], ascending=[True, False, False])
    totals = df.groupby("user_id", sort=False).weight.sum()
    keep_users = totals[totals >= 2].index
    df = df[df.user_id.isin(keep_users)]

    profiles: dict[str, list[tuple[str, float]]] = {}
    for user, group in df.groupby("user_id", sort=False):
        head = group.head(50)
        profiles[str(user)] = [
            (str(p), float(w)) for p, w in zip(head.product_id, head.weight)
        ]

    purchased: dict[str, list[str]] = {}
    bought = df[df.purchases > 0].sort_values(["user_id", "last_ts"], ascending=[True, False])
    for user, group in bought.groupby("user_id", sort=False):
        purchased[str(user)] = [str(p) for p in group.product_id.head(100)]
    return profiles, purchased


def _recent_purchases_frame(purchases: pd.DataFrame) -> dict[str, dict[str, float]]:
    if purchases is None or purchases.empty:
        return {}
    out: dict[str, dict[str, float]] = {}
    for row in purchases.itertuples():
        ts = pd.Timestamp(row.last_ts)
        out.setdefault(str(row.user_id), {})[str(row.product_id)] = float(ts.timestamp())
    return out


def _bandit_priors_from_stats(stats: pd.DataFrame, catalog_df: pd.DataFrame) -> BanditState:
    sub_of = dict(zip(catalog_df.product_id.astype(str), catalog_df.subcategory.astype(str)))
    cat_alpha: dict[str, float] = {}
    cat_beta: dict[str, float] = {}
    state = BanditState()
    if stats is None or stats.empty:
        return state
    df = stats.copy()
    df["pid"] = df.product_id.astype(str)
    df["subcategory"] = df.pid.map(sub_of)
    by_sub = df.dropna(subset=["subcategory"]).groupby("subcategory")[["impressions", "purchases"]].sum()
    for sub, row in by_sub.iterrows():
        rate = (float(row.purchases) + 2.0) / (float(row.impressions) + PRIOR_STRENGTH)
        cat_alpha[str(sub)] = rate * PRIOR_STRENGTH
        cat_beta[str(sub)] = (1.0 - rate) * PRIOR_STRENGTH
    state.cat_alpha = cat_alpha
    state.cat_beta = cat_beta
    for row in df.itertuples():
        n = int(row.impressions)
        if n < 3:
            continue
        pid = str(row.product_id)
        sub = sub_of.get(pid, "")
        a0, b0 = state.prior(sub)
        k = float(row.purchases)
        state.alpha[pid] = a0 + k
        state.beta[pid] = b0 + max(0.0, float(n) - k)
        state.impressions[pid] = n
        state.purchases[pid] = int(k)
    return state


# ---------------------------------------------------------------------------
# In-memory Models (for building ranker training features)
# ---------------------------------------------------------------------------

def models_from(components: Components, directory: Path | None = None) -> Models:
    prices = [it.price_eur for it in components.catalog.values() if it.price_eur > 0]
    models = Models(
        tag=components.tag,
        directory=directory or (MODELS_DIR / components.tag),
        catalog=components.catalog,
        base_ids=components.base_ids,
        popularity=components.popularity,
        pop_rank=components.pop_rank,
        category_rate=components.category_rate,
        item_item=components.item_item,
        als=components.als,
        content=components.content,
        bandit=components.bandit,
        profiles=components.profiles,
        recent_purchases=components.recent_purchases,
        purchased_items=components.purchased_items,
        ranker=None,
        ranker_features=[],
        ranker_calibration=None,
        meta={"cut": components.cut, "dataset": components.dataset},
        price_median=float(np.median(prices)) if prices else 30.0,
    )
    models._rebuild_indices()
    models.max_popularity = max([float(v) for v in components.popularity.values()] + [1.0])
    return models


# ---------------------------------------------------------------------------
# Ranker (Phase 4)
# ---------------------------------------------------------------------------

def train_phase_ranker(
    source: Source,
    *,
    cut: str,
    cfg: Config,
    seed: int,
    days: int = 7,
    max_sessions: int = 6000,
    verbose: bool = True,
    metric: str = "auc",
    min_rounds: int = 1,
    label: str = "purchase",
) -> tuple[Any, dict[str, Any], Components, float]:
    inner_cut = datasets.cut_minus(cut, days)
    t0 = time.time()
    inner = build_components(
        source, tag=f"{cut}-features", interaction_cut=inner_cut, catalog_cut=cut,
        cfg=cfg, seed=seed, verbose=verbose,
    )
    models_inner = models_from(inner)
    # Products created during the label week have no collaborative history in
    # the inner bundle, which is exactly the cold-start state we want to learn.
    cut_ts = pd.Timestamp(cut, tz="UTC").timestamp()
    new_ids = [
        pid for pid, item in inner.catalog.items()
        if item.created_ts and item.created_ts >= cut_ts - days * 86400
    ]
    live = static_snapshot(models_inner, new_ids=new_ids)
    sessions = datasets.ranker_sessions(source, start=inner_cut, end=cut)
    if verbose:
        print(f"[ranker] label window [{inner_cut}, {cut}) -> {len(sessions)} sessions")
    rng = np.random.default_rng(seed)
    X, y, w, groups, names = build_training_set(
        sessions, models_inner, live, cfg, rng,
        max_sessions=max_sessions, negatives_per_session=40, label=label,
    )
    if len(y) == 0:
        raise RuntimeError("no ranker training rows: increase the label window")
    ranker = train_ranker(X, y, w, groups, seed=seed, verbose=verbose,
                          metric=metric, min_rounds=min_rounds)
    metrics = dict(ranker.metrics)
    metrics["inner_cut"] = inner_cut
    metrics["label_sessions"] = int(len(sessions))
    metrics["feature_seconds"] = time.time() - t0
    return ranker, metrics, inner, len(new_ids)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_components(components: Components, directory: Path, *, extra_meta: dict[str, Any] | None = None) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / "catalog.pkl", "wb") as f:
        pickle.dump(components.catalog_df, f, protocol=5)
    with open(directory / "popularity.pkl", "wb") as f:
        pickle.dump({
            "score": components.popularity,
            "rank": components.pop_rank,
            "category_rate": components.category_rate,
        }, f, protocol=5)
    if components.item_item is not None:
        components.item_item.save(directory / "item_item.pkl")
    if components.als is not None:
        components.als.save(directory / "als.pkl")
    if components.content is not None:
        components.content.save(directory / "content.pkl")
    with open(directory / "users.pkl", "wb") as f:
        pickle.dump({
            "profiles": components.profiles,
            "recent_purchases": components.recent_purchases,
            "purchased_items": components.purchased_items,
        }, f, protocol=5)
    with open(directory / "bandit.pkl", "wb") as f:
        pickle.dump(components.bandit.snapshot(), f, protocol=5)
    if components.ranker is not None:
        components.ranker.save(directory)
    meta = {
        "tag": components.tag,
        "cut": components.cut,
        "dataset": components.dataset,
        "sizes": components.sizes,
        "timings": components.timings,
        "ranker": components.ranker_metrics,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if extra_meta:
        meta.update(extra_meta)
    (directory / "meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("rawtree",), default="rawtree")
    parser.add_argument("--until", default=None, help="YYYY-MM-DD cut; default = production tag 'prod'")
    parser.add_argument("--tag", default=None, help="artefact directory name (default: cut or 'prod')")
    parser.add_argument("--out", default=str(MODELS_DIR), help="models root")
    parser.add_argument("--skip-ranker", action="store_true")
    parser.add_argument("--skip-als", action="store_true", help="ablation builds only")
    parser.add_argument("--ranker-days", type=int, default=7)
    parser.add_argument("--max-ranker-sessions", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-ranker-labels", action="store_true", help="build bundle without ranker")
    parser.add_argument("--ranker-metric", default="auc",
                        choices=("auc", "average_precision"),
                        help="early-stopping metric; average_precision is the honest "
                             "one when positives are scarce")
    parser.add_argument("--ranker-min-rounds", type=int, default=1,
                        help="floor on the number of boosting rounds (AUC on ~70 "
                             "positives can peak at round 1 by luck)")
    parser.add_argument("--ranker-label", default="purchase",
                        choices=("purchase", "click"),
                        help="positive label: purchase (the scored objective) or "
                             "click (2.5x more positives, a measured proxy)")

    args = parser.parse_args()

    source = _make_source(args.source)
    cut = args.until
    if cut is None:
        # Production: learn from everything that exists, but cut the label week
        # exactly at the last available history date so the ranker split stays
        # temporal and reproducible.
        cut = datasets.max_history_date(source)
    tag = args.tag or (cut if args.until else "prod")
    cfg = Config()

    started = time.time()
    print(f"[train] source={source.name} cut={cut} tag={tag}")
    components = build_components(
        source, tag=tag, interaction_cut=cut if args.until else None,
        catalog_cut=cut if args.until else None, cfg=cfg, seed=args.seed,
    )
    if args.skip_als:
        components.als = None

    if not args.skip_ranker and not args.no_ranker_labels:
        ranker, metrics, _inner, _n_new = train_phase_ranker(
            source, cut=cut, cfg=cfg, seed=args.seed, days=args.ranker_days,
            max_sessions=args.max_ranker_sessions,
            metric=args.ranker_metric, min_rounds=args.ranker_min_rounds,
            label=args.ranker_label,
        )
        components.ranker = ranker
        components.ranker_metrics = metrics

    out_dir = Path(args.out) / tag
    save_components(components, out_dir, extra_meta={"config": cfg.to_dict()})
    total = time.time() - started
    print(f"[train] wrote {out_dir} in {total:.1f}s")
    print(f"[train] timings: {json.dumps(components.timings, default=str)}")
    if components.ranker_metrics:
        print(f"[train] ranker: {json.dumps(components.ranker_metrics, default=str)}")


def _make_source(kind: str) -> Source:
    return make_source(kind)


if __name__ == "__main__":
    main()
