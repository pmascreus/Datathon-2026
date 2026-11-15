"""Hybrid candidate generation: popularity, item-item, ALS, content, trends,
user history and category priors.

Every channel returns candidates in its own scale; we keep both the raw score
(useful for LightGBM) and a bounded reciprocal-rank score (useful for the
heuristic blend, so one extreme co-occurrence count cannot dominate). The union
is capped, availability-filtered and always includes cold/new products that
semantic similarity puts near the session.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

from src.common import as_list, parse_ts
from src.config import Config
from src.state import LiveSnapshot
from src.store import Models

CHANNEL_WEIGHTS = {
    "ii_view": "w_item_view",
    "ii_cart": "w_item_cart",
    "ii_purchase": "w_item_purchase",
    "als": "w_als",
    "content": "w_content",
    "pop": "w_popularity",
    "region": "w_region_trend",
    "trend": "w_product_trend",
    "history": "w_user_history",
    "catprior": "w_category_prior",
    "cold": "w_cold",
}


@dataclass(frozen=True)
class Session:
    request_id: str
    user_id: str
    is_new_user: bool
    ts: float
    device: str
    region: str
    page: str
    views: tuple[tuple[str, float], ...]      # (product_id, age_seconds), newest first
    request: dict[str, Any]

    @classmethod
    def from_request(cls, req: dict[str, Any]) -> "Session":
        request_ts = parse_ts(req.get("ts")) or 0.0
        context = req.get("context") or {}
        if isinstance(context, str):
            import json

            try:
                context = json.loads(context)
            except json.JSONDecodeError:
                context = {}
        if not isinstance(context, dict):
            context = {}
        # RawTree flattens nested objects: live rows carry `device`, `page` and
        # `region` as top-level keys, but a JSON `context` blob is accepted too.
        if not context:
            context = {
                "device": req.get("device", "web"),
                "region": req.get("region", ""),
                "page": req.get("page", "home"),
            }
        views: dict[str, float] = {}
        for raw in as_list(req.get("recent_views")):
            if not isinstance(raw, dict):
                continue
            pid = str(raw.get("product_id", "") or "")
            if not pid:
                continue
            ts = parse_ts(raw.get("ts"))
            age = max(0.0, request_ts - ts) if ts else 0.0
            views[pid] = min(age, views.get(pid, 1e12))
        ordered = sorted(views.items(), key=lambda kv: kv[1])
        return cls(
            request_id=str(req.get("request_id", "")),
            user_id=str(req.get("user_id", "") or ""),
            is_new_user=bool(req.get("is_new_user", False)),
            ts=request_ts,
            device=str(context.get("device", "web")),
            region=str(context.get("region", "")),
            page=str(context.get("page", "home")),
            views=tuple(ordered[:10]),
            request=req,
        )

    def view_weights(self, decay: float) -> list[tuple[str, float]]:
        return [(pid, decay ** k) for k, (pid, _age) in enumerate(self.views)]

    @property
    def hour(self) -> int:
        import datetime

        return datetime.datetime.fromtimestamp(self.ts or 0, tz=datetime.timezone.utc).hour


@dataclass
class CandidateSet:
    pids: list[str]
    raw: dict[str, dict[str, float]] = field(default_factory=dict)
    norm: dict[str, dict[str, float]] = field(default_factory=dict)
    heuristic: dict[str, float] = field(default_factory=dict)

    def add_channel(self, name: str, scores: dict[str, float], weight: float,
                    available: frozenset[str], top: int) -> None:
        filtered = {
            pid: float(score) for pid, score in scores.items()
            if pid in available
        }
        if not filtered:
            return
        ordered = sorted(filtered.items(), key=lambda kv: (-kv[1], kv[0]))[:top]
        raw = dict(ordered)
        norm = {pid: 1.0 / (rank + 1) for rank, (pid, _) in enumerate(ordered)}
        self.raw[name] = raw
        self.norm[name] = norm
        for pid, value in norm.items():
            self.heuristic[pid] = self.heuristic.get(pid, 0.0) + weight * value


def generate_candidates(
    session: Session,
    models: Models,
    live: LiveSnapshot,
    cfg: Config,
) -> CandidateSet:
    available = live.available
    seeds = session.view_weights(cfg.session_decay)
    profile = models.user_profile(session.user_id) if not session.is_new_user else []
    # A user with a long profile but no session activity still gets personalisation.
    profile_seeds = [(p, w) for p, w in profile[:12]]
    # What the shopper bought in the last hours is the strongest "next basket"
    # signal we can get during a round, because `team_25.feedback` has no
    # user_id. Off in the validated version: a measured experiment.
    purchase_seeds: list[tuple[str, float]] = []
    if cfg.use_purchase_seeds and session.user_id:
        recent = models.purchased_recently(
            session.user_id, session.ts or 0.0, cfg.recent_purchase_window_h
        )
        purchase_seeds = [
            (pid, cfg.purchase_seed_weight * math.exp(-age_h / max(cfg.recent_purchase_tau_h, 1e-6)))
            for pid, age_h in sorted(recent.items(), key=lambda kv: kv[1])[:5]
        ]
    combined = seeds + [(p, 0.35 * w) for p, w in profile_seeds] + purchase_seeds
    ii_seeds = seeds + purchase_seeds
    top = max(20, cfg.per_channel_k)

    cand = CandidateSet(pids=[])
    weights = {name: float(getattr(cfg, field_name)) for name, field_name in CHANNEL_WEIGHTS.items()}

    if cfg.use_item_item and models.item_item and ii_seeds:
        for channel in ("view", "add_to_cart", "purchase"):
            key = {"view": "ii_view", "add_to_cart": "ii_cart", "purchase": "ii_purchase"}[channel]
            cand.add_channel(key, models.item_item.scores(ii_seeds, channel, top=top),
                             weights[key], available, top)

    if cfg.use_als and models.als:
        vector = None
        if session.user_id and not session.is_new_user:
            vector = models.als.user_vector(session.user_id)
        if vector is None:
            vector = models.als.vector_from_items(combined or seeds)
        cand.add_channel("als", models.als.scores(vector, top=max(60, top)),
                         weights["als"], available, top)

    if cfg.use_content and models.content:
        vector = models.content.vector_for(combined or seeds)
        cand.add_channel("content", models.content.scores(vector, top=max(60, top)),
                         weights["content"], available, top)
        # Cold-start guarantee: new products that are semantically close. Only
        # the new subset is scored, so this channel stays O(new products).
        if live.new_product_ids:
            cold_scores = models.content.scores_subset(vector, live.new_product_ids)
            cand.add_channel("cold", cold_scores, weights["cold"], available, 20)

    if cfg.use_popularity:
        pop_scores = {}
        for pid in models.popular_sorted:
            if pid in available:
                pop_scores[pid] = models.popularity.get(pid, 0.0)
                if len(pop_scores) >= max(80, top):
                    break
        cand.add_channel("pop", pop_scores, weights["pop"], available, top)

    if cfg.use_trends:
        region_scores = {
            pid: 1.0 / (rank + 1)
            for rank, pid in enumerate(live.region_trending.get(session.region, ()))
        }
        cand.add_channel("region", region_scores, weights["region"], available, 25)
        trend_scores = {
            pid: value for pid, value in sorted(live.product_trend.items(), key=lambda kv: -kv[1])[:40]
        }
        cand.add_channel("trend", trend_scores, weights["trend"], available, 30)

    if cfg.use_user_history and profile:
        hist = {pid: w for pid, w in profile[:40]}
        cand.add_channel("history", hist, weights["history"], available, 40)

    if cfg.use_category_prior:
        cand.add_channel("catprior", _category_prior_scores(session, models, live, profile),
                         weights["catprior"], available, 50)

    # ---- union + cap -------------------------------------------------------
    ordered_channels = {name: dict(sorted(scores.items(), key=lambda kv: -kv[1]))
                        for name, scores in cand.raw.items()}
    selected: list[str] = []
    seen: set[str] = set()
    # 1) every channel's head, so no signal is lost by one dominant channel
    for name in ordered_channels:
        for pid in list(ordered_channels[name])[:40]:
            if pid not in seen:
                seen.add(pid)
                selected.append(pid)
    # 2) cold products near the session, even if their popularity is tiny
    for pid in list(ordered_channels.get("cold", {}))[:10]:
        if pid not in seen:
            seen.add(pid)
            selected.append(pid)
    # 3) fill by heuristic
    for pid, _ in sorted(cand.heuristic.items(), key=lambda kv: (-kv[1], kv[0])):
        if len(selected) >= cfg.max_candidates:
            break
        if pid not in seen:
            seen.add(pid)
            selected.append(pid)
    # 4) absolute safety: fallback pool
    for pid in live.fallback_pool:
        if len(selected) >= max(10, cfg.max_candidates):
            break
        if pid not in seen:
            seen.add(pid)
            selected.append(pid)

    selected = selected[: cfg.max_candidates]
    cand.pids = selected
    return cand


def _category_prior_scores(
    session: Session,
    models: Models,
    live: LiveSnapshot,
    profile: list[tuple[str, float]],
) -> dict[str, float]:
    """Products in the subcategories the shopper is already interested in.

    It is the only channel that can put a brand-new product (no ALS, no
    co-occurrence) in front of the ranker with a sensible prior.
    """
    subcats: dict[str, float] = {}
    for pid, age in session.views[:6]:
        item = models.item(pid)
        if item:
            subcats[item.subcategory] = max(subcats.get(item.subcategory, 0.0), 1.0 / (1 + age / 600))
    for pid, w in profile[:10]:
        item = models.item(pid)
        if item:
            subcats[item.subcategory] = max(subcats.get(item.subcategory, 0.0), 0.4 * w)
    if not subcats:
        return {}
    out: dict[str, float] = {}
    for subcat, weight in sorted(subcats.items(), key=lambda kv: -kv[1])[:4]:
        products = models.by_subcategory.get(subcat, [])
        if not products:
            continue
        # Cold/new products first (they have no other channel), then popularity.
        ranked = sorted(
            products,
            key=lambda p: (
                0 if p in live.new_product_ids else 1,
                -models.popularity.get(p, 0.0),
                p,
            ),
        )
        for rank, pid in enumerate(ranked[:24]):
            if pid in live.available:
                out[pid] = max(out.get(pid, 0.0), weight / (rank + 1))
    return out
