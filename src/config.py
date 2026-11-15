"""One immutable configuration object for serving and offline evaluation.

Every weight used by `recommend()` lives here so that a policy can be described
by a config diff instead of by editing code. The defaults are the ones that
survived the incremental offline validation; `PHASE_POLICIES` below keeps
the losing variants for the jury.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = ROOT / "config" / "recommender.json"


@dataclass(frozen=True)
class Config:
    # ---- candidate channels -------------------------------------------------
    use_item_item: bool = True
    use_als: bool = True
    use_content: bool = True
    use_popularity: bool = True
    use_trends: bool = True
    use_user_history: bool = True
    use_category_prior: bool = True

    w_item_view: float = 1.0
    w_item_cart: float = 1.4
    w_item_purchase: float = 2.0
    w_als: float = 1.6
    w_content: float = 0.9
    w_popularity: float = 0.7
    w_region_trend: float = 0.35
    w_product_trend: float = 0.5
    w_user_history: float = 1.1
    w_category_prior: float = 0.4
    w_cold: float = 0.9

    session_decay: float = 0.7          # older session views weigh less
    profile_decay_days: float = 21.0    # long-term profile half-life
    max_candidates: int = 350
    per_channel_k: int = 80

    # ---- ranking ------------------------------------------------------------
    use_ranker: bool = True
    ranker_blend: float = 0.85          # ranker vs heuristic probability blend

    # ---- calibration / business rules --------------------------------------
    mmr_lambda: float = 0.16
    mmr_pool: int = 60                   # MMR only re-ranks the head (0 = all)
    revenue_kappa: float = 0.04          # secondary objective, never dominates p(buy)
    recent_purchase_delta: float = 0.55
    recent_purchase_tau_h: float = 36.0
    recent_purchase_window_h: float = 168.0

    # ---- live user-level purchase feed (shop.events, last few hours) --------
    # Off by default: `versio4` (the validated version) does not use it. It is
    # enabled per deployment by config/versio5.json.
    use_live_purchases: bool = False
    live_purchases_refresh_s: float = 180.0
    live_purchases_hours: int = 3         # query lookback; must exceed refresh_s

    # ---- live purchases as extra candidate seeds (versio5 experiment) ------
    use_purchase_seeds: bool = False
    purchase_seed_weight: float = 0.5

    # ---- exploration (Thompson sampling) -----------------------------------
    exploration_eps: float = 0.0        # off until the live A/B shows a gain
    exploration_slots: tuple[int, int] = (6, 9)

    # ---- reliability --------------------------------------------------------
    compute_budget_ms: float = 900.0
    fallback_age_s: float = 2.2
    hard_deadline_s: float = 2.50
    stale_state_s: float = 180.0
    # Request timestamps come from the platform, so deadlines are in platform
    # time. This machine's clock runs ~1 s behind (measured in the original repo, DECISIONS §14);
    # deciding in local time made the loop think it had a second more than it
    # did, so the degrade/skip checks add the measured offset. Set to false to
    # fall back to the old (lenient) behaviour without a code change.
    use_clock_offset: bool = True

    # ---- serving loop -------------------------------------------------------
    poll_interval_s: float = 0.04
    batch_max: int = 50
    batch_flush_age_s: float = 0.05
    feedback_refresh_s: float = 20.0
    stock_refresh_s: float = 1.0
    trend_refresh_s: float = 15.0
    catalog_refresh_s: float = 30.0

    # ---- evaluation ---------------------------------------------------------
    ips_weight_clip: float = 50.0       # 1/propensity capped; documented in the original repo's DECISIONS.md
    eval_test_days: int = 7
    eval_ranker_days: int = 7

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def with_(
        self,
        *,
        drop_channels: tuple[str, ...] = (),
        **overrides: Any,
    ) -> "Config":
        """Return a copy with channels disabled and/or fields overridden.

        `drop_channels` uses the short names used in the phase report
        ("als", "content", "item_item", "trends", "user_history", "ranker").
        """
        mapping = {
            "item_item": "use_item_item",
            "als": "use_als",
            "content": "use_content",
            "popularity": "use_popularity",
            "trends": "use_trends",
            "user_history": "use_user_history",
            "category_prior": "use_category_prior",
            "ranker": "use_ranker",
        }
        extra: dict[str, Any] = {}
        for name in drop_channels:
            key = mapping.get(name)
            if key is None:
                raise KeyError(f"unknown channel to drop: {name}")
            extra[key] = False
        unknown = set(overrides) - set(self.to_dict())
        if unknown:
            raise KeyError(f"unknown config fields: {sorted(unknown)}")
        return replace(self, **extra, **overrides)

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        """Load JSON overrides, then environment overrides (for deployment)."""
        cfg = cls()
        target = Path(path) if path else DEFAULT_CONFIG_PATH
        if target.exists():
            raw = json.loads(target.read_text(encoding="utf-8"))
            unknown = set(raw) - set(cfg.to_dict())
            if unknown:
                raise KeyError(f"{target}: unknown config keys {sorted(unknown)}")
            cfg = replace(cfg, **raw)
        env_map = {
            "RT_EXPLORATION_EPS": ("exploration_eps", float),
            "RT_MMR_LAMBDA": ("mmr_lambda", float),
            "RT_REVENUE_KAPPA": ("revenue_kappa", float),
            "RT_USE_RANKER": ("use_ranker", lambda v: v.lower() not in ("0", "false", "no")),
            "RT_COMPUTE_BUDGET_MS": ("compute_budget_ms", float),
            "RT_USE_LIVE_PURCHASES": ("use_live_purchases", lambda v: v.lower() not in ("0", "false", "no")),
            "RT_LIVE_PURCHASES_REFRESH_S": ("live_purchases_refresh_s", float),
            "RT_USE_CLOCK_OFFSET": ("use_clock_offset", lambda v: v.lower() not in ("0", "false", "no")),
        }
        env_overrides = {}
        for env_name, (field_name, caster) in env_map.items():
            if env_name in os.environ:
                env_overrides[field_name] = caster(os.environ[env_name])
        if env_overrides:
            cfg = replace(cfg, **env_overrides)
        return cfg


PHASE_POLICIES: dict[str, Config] = {
    # Cumulative policies that mirror the assignment phases. Every step only
    # adds one component, so the phase report is a clean before/after diff.
    "p0_baseline": Config(
        use_item_item=False, use_als=False, use_content=False, use_popularity=True,
        use_trends=False, use_user_history=False, use_category_prior=False,
        use_ranker=False, mmr_lambda=0.0, revenue_kappa=0.0, recent_purchase_delta=0.0,
    ),
    # Phase 1: item-item collaborative filtering.
    "p1_item_item": Config(
        use_item_item=True, use_als=False, use_content=False, use_popularity=True,
        use_trends=False, use_user_history=False, use_category_prior=False,
        use_ranker=False, mmr_lambda=0.0, revenue_kappa=0.0, recent_purchase_delta=0.0,
    ),
    # Phase 2: ALS + long-term user history.
    "p2_als": Config(
        use_item_item=True, use_als=True, use_content=False, use_popularity=True,
        use_trends=False, use_user_history=True, use_category_prior=False,
        use_ranker=False, mmr_lambda=0.0, revenue_kappa=0.0, recent_purchase_delta=0.0,
    ),
    # Phase 3: content-based candidates + cold-start backbone.
    "p3_content": Config(
        use_item_item=True, use_als=True, use_content=True, use_popularity=True,
        use_trends=False, use_user_history=True, use_category_prior=True,
        use_ranker=False, mmr_lambda=0.0, revenue_kappa=0.0, recent_purchase_delta=0.0,
    ),
    # Phase 4: hybrid candidates + LightGBM P(purchase) re-ranker.
    "p4_ranker": Config(
        use_item_item=True, use_als=True, use_content=True, use_popularity=True,
        use_trends=False, use_user_history=True, use_category_prior=True,
        use_ranker=True, mmr_lambda=0.0, revenue_kappa=0.0, recent_purchase_delta=0.0,
    ),
    # Phase 5: live trends (validated live; offline the trend signal is empty).
    "p5_trends": Config(
        use_item_item=True, use_als=True, use_content=True, use_popularity=True,
        use_trends=True, use_user_history=True, use_category_prior=True,
        use_ranker=True, mmr_lambda=0.0, revenue_kappa=0.0, recent_purchase_delta=0.0,
    ),
    # Phase 6: cold-start robustness (stronger cold channel + category prior).
    "p6_cold": Config(
        use_item_item=True, use_als=True, use_content=True, use_popularity=True,
        use_trends=True, use_user_history=True, use_category_prior=True,
        use_ranker=True, w_cold=2.2, w_category_prior=0.7,
        mmr_lambda=0.0, revenue_kappa=0.0, recent_purchase_delta=0.0,
    ),
    # Phase 7: MMR diversification + small revenue tilt + short re-purchase penalty.
    "p7_diversify": Config(
        use_item_item=True, use_als=True, use_content=True, use_popularity=True,
        use_trends=True, use_user_history=True, use_category_prior=True,
        use_ranker=True, w_cold=2.2, w_category_prior=0.7,
        mmr_lambda=0.16, revenue_kappa=0.04, recent_purchase_delta=0.55,
    ),
    # Slow path: the same personalised decision core, but without the LightGBM
    # ranker and with MMR limited to the head. ~10x cheaper than `default`, so
    # it is what the live loop uses when the remaining budget is too small for
    # the full model. It is still personalised (ALS/content/session/category),
    # unlike the best-seller pool, and it is gated offline against the baseline.
    "p_fast": Config(
        use_item_item=True, use_als=True, use_content=True, use_popularity=True,
        use_trends=True, use_user_history=True, use_category_prior=True,
        use_ranker=False, w_cold=1.2, w_category_prior=0.6,
        mmr_lambda=0.06, mmr_pool=40, revenue_kappa=0.04, recent_purchase_delta=0.55,
    ),
    # Phase 8: online Thompson exploration on the purchase posterior.
    "p8_bandit": Config(
        use_item_item=True, use_als=True, use_content=True, use_popularity=True,
        use_trends=True, use_user_history=True, use_category_prior=True,
        use_ranker=True, w_cold=2.2, w_category_prior=0.7,
        mmr_lambda=0.16, revenue_kappa=0.04, recent_purchase_delta=0.55,
        exploration_eps=0.05,
    ),
}
