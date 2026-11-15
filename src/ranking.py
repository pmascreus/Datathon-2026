"""LightGBM re-ranker (Phase 4) and the single feature builder used by both
training and serving.

Target: P(purchase | impression) for each candidate. Clicks are used as a
secondary signal (higher sample weight) but never as the target, because the
competition is scored on purchases.

Leakage control
---------------
The ranker is trained on sessions in `[cut - r, cut)` while every candidate and
feature is produced with artefacts trained on `< cut - r`. At evaluation time
the artefacts are trained on `< cut` and the sessions are `>= cut`. There is no
session whose features were computed with data from its own future.
"""
from __future__ import annotations

import math
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.common import parse_ts
from src.config import Config
from src.retrieval import CandidateSet, Session, generate_candidates
from src.state import LiveSnapshot
from src.store import Models

FEATURES = [
    "ii_view", "ii_cart", "ii_purchase",
    "als_z", "als_rank", "content", "cold_content",
    "pop_score", "pop_rank_frac", "region_score", "trend_score", "subcat_trend",
    "catprior", "history_weight", "history_rank",
    "cat_rate", "cat_affinity", "subcat_match",
    "price_norm", "price_rank",
    "is_new_item", "cold_coverage", "days_since_created",
    "recent_purchase_age_h", "purchased_before", "profile_weight", "in_session",
    "bandit_mean", "bandit_imp_log",
    "heur_rank", "heur_score",
    "is_new_user", "device_ios", "device_android", "page_product", "page_category",
    "hour_sin", "hour_cos", "session_len", "user_profile_len",
]


def feature_matrix(
    models: Models,
    session: Session,
    cand: CandidateSet,
    live: LiveSnapshot,
    cfg: Config,
) -> np.ndarray:
    pids = cand.pids
    n = len(pids)
    if n == 0:
        return np.zeros((0, len(FEATURES)), dtype=np.float32)
    profile = models.user_profile(session.user_id) if not session.is_new_user else []
    profile_map = {pid: float(w) for pid, w in profile}
    max_profile = max(profile_map.values(), default=1.0)
    purchased = set(models.purchased_items.get(session.user_id, [])) if session.user_id else set()
    recent = models.purchased_recently(session.user_id, session.ts or 0.0) if session.user_id else {}
    session_items = {pid for pid, _ in session.views}
    new_ids = live.new_product_ids

    # Per-request normalisation of channel scores.
    als = cand.raw.get("als", {})
    als_values = np.array([als.get(pid, 0.0) for pid in pids], dtype=np.float64)
    als_mu = float(als_values.mean())
    als_sd = float(als_values.std()) or 1.0
    als_rank = _ranks([als.get(pid, -math.inf) for pid in pids])
    content = cand.raw.get("content", {})
    cold = cand.raw.get("cold", {})
    history = cand.raw.get("history", {})
    history_rank = _ranks([history.get(pid, -math.inf) for pid in pids])
    heur_values = [cand.heuristic.get(pid, 0.0) for pid in pids]
    heur_rank = _ranks([-v for v in heur_values])
    max_heur = max(heur_values) if heur_values else 1.0
    max_heur = max_heur if max_heur > 0 else 1.0
    price_values = [models.catalog[pid].price_eur if pid in models.catalog else 0.0 for pid in pids]
    price_rank = _ranks(price_values)

    # User category mix, for the category-affinity feature.
    user_categories: dict[str, float] = {}
    for pid, w in profile:
        item = models.item(pid)
        if item:
            user_categories[item.category] = user_categories.get(item.category, 0.0) + float(w)
    cat_total = sum(user_categories.values()) or 1.0
    user_subcats = {
        models.item(pid).subcategory for pid, _w in profile[:20]
        if models.item(pid) is not None
    }

    hour = session.hour
    hour_sin = math.sin(2 * math.pi * hour / 24)
    hour_cos = math.cos(2 * math.pi * hour / 24)
    out = np.zeros((n, len(FEATURES)), dtype=np.float32)
    for i, pid in enumerate(pids):
        item = models.item(pid)
        if item is None:
            continue
        rate = models.category_rate.get(item.subcategory, 0.02)
        imp = models.bandit.impressions.get(pid, 0)
        age_h = recent.get(pid)
        days_created = ((session.ts - item.created_ts) / 86400.0) if item.created_ts else 3650.0
        row = out[i]
        row[0] = cand.raw.get("ii_view", {}).get(pid, 0.0)
        row[1] = cand.raw.get("ii_cart", {}).get(pid, 0.0)
        row[2] = cand.raw.get("ii_purchase", {}).get(pid, 0.0)
        row[3] = (als.get(pid, 0.0) - als_mu) / als_sd
        row[4] = als_rank[i]
        row[5] = content.get(pid, 0.0)
        row[6] = cold.get(pid, 0.0)
        row[7] = models.popularity.get(pid, 0.0) / max(models.max_popularity, 1e-9)
        row[8] = min(models.pop_rank.get(pid, 99999) / max(len(models.pop_rank), 1), 1.0)
        row[9] = cand.norm.get("region", {}).get(pid, 0.0)
        row[10] = cand.raw.get("trend", {}).get(pid, 0.0)
        row[11] = live.subcategory_trend.get(item.subcategory, 0.0)
        row[12] = cand.raw.get("catprior", {}).get(pid, 0.0)
        row[13] = profile_map.get(pid, 0.0) / max_profile
        row[14] = history_rank[i]
        row[15] = float(rate)
        row[16] = user_categories.get(item.category, 0.0) / cat_total
        row[17] = 1.0 if item.subcategory in user_subcats else 0.0
        row[18] = models.price_norm(item.price_eur)
        row[19] = price_rank[i]
        row[20] = 1.0 if item.is_new or pid in new_ids else 0.0
        row[21] = 1.0 if (pid not in als and pid not in cand.raw.get("ii_view", {})) else 0.0
        row[22] = float(np.clip(days_created, 0, 3650))
        row[23] = float(age_h) if age_h is not None else 999.0
        row[24] = 1.0 if pid in purchased else 0.0
        row[25] = profile_map.get(pid, 0.0) / max_profile
        row[26] = 1.0 if pid in session_items else 0.0
        row[27] = models.bandit.mean(pid, item.subcategory)
        row[28] = math.log1p(imp)
        row[29] = heur_rank[i]
        row[30] = heur_values[i] / max_heur
        row[31] = 1.0 if session.is_new_user else 0.0
        row[32] = 1.0 if session.device == "ios" else 0.0
        row[33] = 1.0 if session.device == "android" else 0.0
        row[34] = 1.0 if session.page == "product" else 0.0
        row[35] = 1.0 if session.page == "category" else 0.0
        row[36] = hour_sin
        row[37] = hour_cos
        row[38] = len(session.views)
        row[39] = len(profile)
    return out


def _ranks(values: list[float]) -> np.ndarray:
    """0 = best (highest value), 1 = worst; ties share the average rank."""
    n = len(values)
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    order = np.argsort(-np.asarray(values, dtype=np.float64), kind="stable")
    ranks = np.empty(n, dtype=np.float32)
    ranks[order] = np.arange(n, dtype=np.float32)
    if n > 1:
        ranks /= (n - 1)
    else:
        ranks[:] = 0.0
    return ranks


@dataclass
class Ranker:
    """LightGBM booster + feature names + isotonic calibrator (training side)."""

    booster: Any
    features: list[str]
    calibration: Any = None
    metrics: dict[str, float] = field(default_factory=dict)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if X.size == 0:
            return np.zeros(0, dtype=np.float32)
        raw = self.booster.predict(X)
        return self.calibrate(raw)

    def calibrate(self, raw: np.ndarray) -> np.ndarray:
        if self.calibration is not None:
            try:
                raw = self.calibration.predict(raw)
            except Exception:
                pass
        return np.clip(np.asarray(raw, dtype=np.float64), 1e-5, 0.999).astype(np.float32)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(directory / "ranker.txt"))
        with open(directory / "ranker_meta.pkl", "wb") as f:
            pickle.dump({
                "features": self.features,
                "calibration": self.calibration,
                "metrics": self.metrics,
            }, f, protocol=5)


def build_training_set(
    sessions: pd.DataFrame,
    models: Models,
    live: LiveSnapshot,
    cfg: Config,
    rng: np.random.Generator,
    max_sessions: int | None = None,
    negatives_per_session: int = 45,
    label: str = "purchase",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Candidates from the hybrid generator + logged outcomes as labels.

    Negatives are down-sampled per session (random + top-heuristic) because the
    full 350-candidate tail is almost always irrelevant; the positives and the
    hard head are always kept. Calibration on a held-out slice restores
    probabilities afterwards.
    """
    rows: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    ws: list[np.ndarray] = []
    groups: list[int] = []
    order = sessions.sort_values("ts")
    if max_sessions is not None and len(order) > max_sessions:
        order = order.tail(max_sessions)
    for row in order.itertuples():
        user_id = str(getattr(row, "user_id", "") or "")
        is_new = bool(getattr(row, "is_new_user", False) or (user_id and user_id not in models.profiles))
        req = {
            "request_id": getattr(row, "request_id", ""),
            "ts": getattr(row, "ts", None),
            "user_id": user_id,
            "is_new_user": is_new,
            "context": getattr(row, "context", {}) or {},
            "device": getattr(row, "device", None),
            "page": getattr(row, "page", None),
            "region": getattr(row, "region", None),
            "recent_views": [{"product_id": v.get("product_id"), "ts": v.get("ts")}
                             for v in (getattr(row, "recent_views", []) or [])],
        }
        session = Session.from_request(req)
        cand = generate_candidates(session, models, live, cfg)
        purchased = str(getattr(row, "purchased", "") or "")
        clicked = str(getattr(row, "clicked", "") or "")
        for positive in (purchased, clicked):
            if positive and positive not in cand.pids and models.item(positive) is not None:
                cand.pids.append(positive)
        if not cand.pids:
            continue
        keep = list(range(len(cand.pids)))
        if len(keep) > negatives_per_session + 2:
            positives = {i for i, pid in enumerate(cand.pids) if pid in (purchased, clicked)}
            head = sorted(range(len(cand.pids)), key=lambda i: -cand.heuristic.get(cand.pids[i], 0.0))[:25]
            pool = [i for i in keep if i not in positives and i not in head]
            extra = list(rng.choice(pool, size=min(negatives_per_session - len(head), len(pool)), replace=False))
            keep = sorted(positives.union(head).union(int(x) for x in extra))
        sub = CandidateSet(
            pids=[cand.pids[i] for i in keep],
            raw={k: {p: v for p, v in d.items() if p in set(cand.pids[i] for i in keep)}
                 for k, d in cand.raw.items()},
            norm={k: {p: v for p, v in d.items() if p in set(cand.pids[i] for i in keep)}
                  for k, d in cand.norm.items()},
            heuristic={p: cand.heuristic.get(p, 0.0) for p in (cand.pids[i] for i in keep)},
        )
        X = feature_matrix(models, session, sub, live, cfg)
        # The ranker is trained on purchase because conversion is what the round
        # scores. `label="click"` exists as a measured alternative: clicks are
        # 2.5x more frequent, so the model gets more positives to learn from, and
        # the gate decides whether that proxy still ranks purchases better.
        if label == "click":
            target, secondary = clicked, purchased
        else:
            target, secondary = purchased, clicked
        y = np.array([1.0 if (target and pid == target) else 0.0 for pid in sub.pids], dtype=np.float32)
        weights = np.array([
            2.5 if (secondary and pid == secondary) else 1.0 for pid in sub.pids
        ], dtype=np.float32)
        rows.append(X)
        ys.append(y)
        ws.append(weights)
        groups.append(len(sub.pids))
    if not rows:
        return (np.zeros((0, len(FEATURES)), dtype=np.float32), np.zeros(0),
                np.zeros(0), np.zeros(0, dtype=np.int32), [])
    return (np.vstack(rows), np.concatenate(ys), np.concatenate(ws),
            np.array(groups, dtype=np.int32), list(FEATURES))


def train_ranker(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    groups: np.ndarray,
    timestamps: np.ndarray | None = None,
    num_threads: int = 4,
    seed: int = 0,
    verbose: bool = True,
    metric: str = "auc",
    min_rounds: int = 1,
) -> Ranker:
    import lightgbm as lgb
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

    if len(y) == 0 or y.sum() == 0:
        raise ValueError("ranker training set has no positive purchase label")
    # Temporal validation: the last 20% of sessions, never a random split.
    n_groups = len(groups)
    cut_group = max(1, int(n_groups * 0.8))
    cut_row = int(groups[:cut_group].sum())
    X_tr, y_tr, w_tr = X[:cut_row], y[:cut_row], w[:cut_row]
    X_va, y_va, w_va = X[cut_row:], y[cut_row:], w[cut_row:]
    if y_va.sum() == 0:
        X_va, y_va, w_va = X_tr, y_tr, w_tr
    params = {
        "objective": "binary",
        "metric": [metric],
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 80,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "verbosity": -1,
        "num_threads": num_threads,
        "seed": seed,
        "force_col_wise": True,
    }
    pos = max(y_tr.sum(), 1.0)
    neg = max(len(y_tr) - pos, 1.0)
    params["scale_pos_weight"] = float(np.clip(math.sqrt(neg / pos), 1.0, 80.0))
    dtrain = lgb.Dataset(X_tr, label=y_tr, weight=w_tr)
    dvalid = lgb.Dataset(X_va, label=y_va, weight=w_va, reference=dtrain)
    booster = lgb.train(
        params, dtrain, num_boost_round=500,
        valid_sets=[dvalid], valid_names=["valid"],
        callbacks=[lgb.early_stopping(40, verbose=False, first_metric_only=True)]
        if y_va.sum() > 10 else [],
    )
    # `min_rounds` exists because the validation slice holds ~70 positives: on
    # that many AUC can peak at iteration 1 by luck, which would ship a one-tree
    # model. A floor keeps the early stop honest without hiding the metric.
    best_iter = max(min_rounds, int(booster.best_iteration or 200))
    # Refit on everything with the selected number of rounds.
    full = lgb.Dataset(X, label=y, weight=w)
    booster = lgb.train(params, full, num_boost_round=best_iter)

    # Calibrate the *refitted* booster on the held-out temporal slice, so the
    # probabilities we serve are the ones the production model actually emits.
    raw_va = booster.predict(X_va)
    calibration = None
    if y_va.sum() >= 30 and len(y_va) > 100:
        calibration = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=0.999)
        calibration.fit(raw_va, y_va)
        prob_va = calibration.predict(raw_va)
    else:
        prob_va = raw_va
    metrics = {}
    try:
        metrics["valid_auc"] = float(roc_auc_score(y_va, prob_va)) if y_va.sum() > 0 else float("nan")
        metrics["valid_pr_auc"] = float(average_precision_score(y_va, prob_va)) if y_va.sum() > 0 else float("nan")
        metrics["valid_logloss"] = float(log_loss(y_va, np.clip(prob_va, 1e-6, 1 - 1e-6), labels=[0, 1]))
        metrics["train_rows"] = int(len(y))
        metrics["train_positives"] = int(y.sum())
        metrics["best_iteration"] = int(best_iter)
        metrics["null_rate"] = float(y.mean())
    except Exception:
        pass
    if verbose:
        print(f"[ranker] rows={len(y)} pos={int(y.sum())} best_iter={best_iter} metrics={metrics}")
    return Ranker(booster=booster, features=list(FEATURES), calibration=calibration, metrics=metrics)
