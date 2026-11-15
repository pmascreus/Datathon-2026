"""The single decision function used by both the live loop and offline evaluation.

Order of operations (each step is a config switch, so a phase report is just a
config diff):

1. hybrid candidate generation (`src.retrieval`)
2. LightGBM P(purchase | impression) if the bundle has a ranker
3. blend with the normalised heuristic, only to keep a sane ordering for the
   tiny fraction of candidates the ranker has never seen
4. business adjustments: short recent-purchase penalty, small revenue tilt
5. optional Thompson exploration on the purchase posterior
6. MMR diversification (content similarity) + availability validation
7. deterministic fallback that is valid by construction

Nothing here reads future data: every input is either static artefacts, the
current request, or the live snapshot published by `src.state`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

from src.retrieval import CandidateSet, Session, generate_candidates
from src.config import Config
from src.state import LiveSnapshot
from src.store import Models
from src.ranking import feature_matrix


@dataclass
class Decision:
    items: list[str]
    scores: dict[str, float] = field(default_factory=dict)
    n_candidates: int = 0
    used_ranker: bool = False
    source: str = "model"
    notes: dict[str, Any] = field(default_factory=dict)

    @property
    def valid(self) -> bool:
        return len(self.items) == 10 and len(set(self.items)) == 10


def fallback_items(
    models: Models,
    live: LiveSnapshot | None,
    k: int = 10,
    exclude: Iterable[str] = (),
    require_available: bool = True,
) -> list[str]:
    """Always-valid list: live fallback pool, then all-time best sellers.

    The live pool is already availability-filtered; the catalogue tail makes
    sure we can still answer if the stream reports an unusual stock state.
    """
    skip = set(exclude)
    available = live.available if (live is not None and require_available) else None
    out: list[str] = []
    seen: set[str] = set()

    def _take(pid: str) -> None:
        if pid in skip or pid in seen:
            return
        item = models.catalog.get(pid)
        if item is None or not item.is_active:
            return
        if available is not None and pid not in available:
            return
        seen.add(pid)
        out.append(pid)

    if live is not None:
        for pid in live.fallback_pool:
            if len(out) >= k:
                break
            _take(pid)
    for pid in models.popular_sorted:
        if len(out) >= k:
            break
        _take(pid)
    if len(out) < k:
        for pid, item in models.catalog.items():
            if len(out) >= k:
                break
            if item.is_active:
                _take(pid)
    return out


def recommend(
    session: Session,
    models: Models,
    live: LiveSnapshot,
    cfg: Config,
    rng: np.random.Generator | None = None,
) -> Decision:
    cand = generate_candidates(session, models, live, cfg)
    if not cand.pids:
        items = fallback_items(models, live)
        return Decision(items=items, source="fallback", notes={"reason": "no_candidates"})

    pids = cand.pids
    n = len(pids)
    heur = np.array([cand.heuristic.get(pid, 0.0) for pid in pids], dtype=np.float64)
    heur_max = float(heur.max()) if n else 1.0
    heur_max = heur_max if heur_max > 0 else 1.0
    heur_norm = heur / heur_max

    used_ranker = False
    probs: np.ndarray | None = None
    score = heur_norm.copy()
    if cfg.use_ranker and models.ranker is not None:
        X = feature_matrix(models, session, cand, live, cfg)
        probs = models.ranker_predict(X)
        if probs.size == n:
            used_ranker = True
            p_max = float(probs.max()) if n else 1.0
            p_max = p_max if p_max > 0 else 1.0
            p_norm = probs / p_max
            blend = float(np.clip(cfg.ranker_blend, 0.0, 1.0))
            score = blend * p_norm + (1.0 - blend) * heur_norm

    # ---- business adjustments (secondary to P(purchase)) -------------------
    if cfg.recent_purchase_delta > 0 and session.user_id:
        recent = models.purchased_recently(
            session.user_id, session.ts or 0.0, cfg.recent_purchase_window_h
        )
        if recent:
            for i, pid in enumerate(pids):
                age_h = recent.get(pid)
                if age_h is None:
                    continue
                penalty = cfg.recent_purchase_delta * math.exp(-age_h / max(cfg.recent_purchase_tau_h, 1e-6))
                score[i] *= max(0.0, 1.0 - penalty)

    if cfg.revenue_kappa > 0:
        median = max(models.price_median, 1.0)
        for i, pid in enumerate(pids):
            item = models.item(pid)
            if item is None:
                continue
            tilt = math.log1p(max(item.price_eur, 1.0)) / math.log1p(median)
            score[i] *= 1.0 + cfg.revenue_kappa * min(tilt, 4.0)

    # ---- optional exploration (Phase 7) ------------------------------------
    if cfg.exploration_eps > 0 and rng is not None and rng.random() < cfg.exploration_eps:
        theta = np.array([
            models.bandit.sample(pid, _subcat(models, pid), rng) for pid in pids
        ], dtype=np.float64)
        t_max = float(theta.max()) if n else 1.0
        t_max = t_max if t_max > 0 else 1.0
        w = 0.35
        score = (1.0 - w) * score + w * (theta / t_max)

    score_map = {pid: float(score[i]) for i, pid in enumerate(pids)}
    if cfg.mmr_lambda > 0:
        chosen = _mmr(pids, score_map, models, k=10, lam=cfg.mmr_lambda, pool=cfg.mmr_pool)
    else:
        chosen = [pid for pid, _ in sorted(score_map.items(), key=lambda kv: (-kv[1], kv[0]))[:10]]

    # ---- validity: exactly 10 distinct, active, available ------------------
    items: list[str] = []
    seen: set[str] = set()
    for pid in chosen:
        if len(items) >= 10:
            break
        if pid in seen or pid not in live.available:
            continue
        item = models.item(pid)
        if item is None or not item.is_active:
            continue
        seen.add(pid)
        items.append(pid)
    if len(items) < 10:
        items.extend(fallback_items(models, live, k=10 - len(items), exclude=seen))
    items = items[:10]
    source = "model" if len(items) == 10 else "fallback"
    # Keep the ranking score of the final list for telemetry / IPS analysis.
    final_scores = {pid: score_map.get(pid, 0.0) for pid in items}
    return Decision(
        items=items,
        scores=final_scores,
        n_candidates=n,
        used_ranker=used_ranker,
        source=source,
        notes={
            "prob_max": float(probs.max()) if probs is not None and probs.size else 0.0,
            "heur_max": heur_max,
        },
    )


def _subcat(models: Models, pid: str) -> str:
    item = models.item(pid)
    return item.subcategory if item else ""


def _mmr(
    pids: list[str],
    score: dict[str, float],
    models: Models,
    k: int,
    lam: float,
    pool: int = 0,
) -> list[str]:
    """Greedy maximal marginal relevance over content embeddings.

    `lam=0` is pure relevance, `lam=1` is pure diversity; Phase 6 validated a
    small value (~0.16), just enough to avoid ten near-identical products.

    `pool > 0` first keeps only the best `pool` candidates by relevance: MMR is
    a re-ranking step, so the products it can promote must already be plausible.
    That turns an O(candidates x k x k) Python loop into a 60x60 similarity
    matrix (one matmul) plus a tiny greedy scan, and it is what makes the live
    policy ~10x faster with no measurable quality change.
    """
    if k <= 1 or not pids:
        return sorted(pids, key=lambda p: -score.get(p, 0.0))[:k]
    if pool and len(pids) > pool:
        pids = sorted(pids, key=lambda p: (-score.get(p, 0.0), p))[:pool]
    lo = min(score.values())
    hi = max(score.values())
    span = hi - lo
    rel = [(0.0 if span < 1e-12 else (score.get(p, lo) - lo) / span) for p in pids]

    emb = getattr(models.content, "embeddings", None) if models.content is not None else None
    index = getattr(models.content, "index", {}) if models.content is not None else {}
    n = len(pids)
    sim = np.zeros((n, n), dtype=np.float32)
    if emb is not None:
        pos = {p: i for i, p in enumerate(pids)}
        have = [p for p in pids if p in index]
        if len(have) > 1:
            rows = [pos[p] for p in have]
            e = np.asarray(emb[[index[p] for p in have]], dtype=np.float32)
            sim[np.ix_(rows, rows)] = e @ e.T
    else:
        sub = [_subcat(models, p) for p in pids]
        for i in range(n):
            for j in range(i + 1, n):
                v = 1.0 if sub[i] == sub[j] else 0.0
                sim[i, j] = sim[j, i] = v

    selected: list[int] = []
    remaining = np.ones(n, dtype=bool)
    while len(selected) < min(k, n):
        if selected:
            mx = sim[:, selected].max(axis=1)
        else:
            mx = np.zeros(n, dtype=np.float32)
        value = (1.0 - lam) * np.asarray(rel, dtype=np.float32) - lam * mx
        value[~remaining] = -np.inf
        i = int(np.argmax(value))
        if not np.isfinite(value[i]):
            break
        remaining[i] = False
        selected.append(i)
    return [pids[i] for i in selected]
