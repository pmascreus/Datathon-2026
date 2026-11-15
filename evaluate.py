"""Offline phase evaluation on a temporal cut, against the real API.

The evaluator is deliberately dumb about *how* a list is produced: every policy
is just `row -> list[str]`, and `recommend()` is the only code path that learns
anything. That keeps serving and offline numbers comparable.

    # real API, model artefacts trained with --until 2026-11-07
    python training/eval_offline.py --source rawtree --models 2026-11-07 \
        --cut 2026-11-07 --limit 3000

Temporal discipline: sessions are read with `ts >= cut`, artefacts with
`ts < cut`. Nothing in the loop can see a session's own outcome or the future.

Policies whose name ends in `_live` (`deployed_live`, `p4_ranker_live`, ...) get
the live user-level purchase feed replayed: before each session, every purchase
whose `first_ts` is earlier than the session `ts` is ingested into a private copy
of the model state. That is the same contract as `LiveState._refresh_purchases`,
so `deployed` vs `deployed_live` answers "does the feed help?" with the serving
code itself. A `_seeds` suffix additionally uses the shopper's recent purchases
as candidate seeds (`use_purchase_seeds`).

Metrics are the official estimator (hit rate and IPS/SNIPS on slot 1, with the
documented propensity clipping) plus purchase/click hit rate, NDCG@10,
cold-start slices, list validity, logged revenue on hits and local inference
latency. IPS in this dataset is systematically low because our list rarely
matches the logged slot 1; it is reported for comparability, not as the
selection metric.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src import data as datasets
from src.common import parse_ts
from src.config import PHASE_POLICIES, Config
from src.data import make_source
from src.policy import recommend
from src.retrieval import Session
from src.state import static_snapshot
from src.store import Models

REPORTS = ROOT / "reports"


def _request_from_row(row: Any, profiles: dict[str, Any]) -> dict[str, Any]:
    """Normalise an offline session row into a live-shaped request.

    Real `history_sessions` has no `is_new_user`, so a user absent from the
    training profiles is treated as cold. That is the same inference the live
    loop makes when `requests.is_new_user` is missing.
    """
    user_id = str(getattr(row, "user_id", "") or "")
    is_new = bool(getattr(row, "is_new_user", False))
    if not is_new and user_id:
        is_new = user_id not in profiles
    return {
        "request_id": str(getattr(row, "request_id", "")),
        "ts": getattr(row, "ts", None),
        "user_id": user_id,
        "is_new_user": is_new,
        "device": getattr(row, "device", None),
        "page": getattr(row, "page", None),
        "region": getattr(row, "region", None),
        "context": getattr(row, "context", None),
        "recent_views": list(getattr(row, "recent_views", []) or []),
    }


def make_policy(
    name: str,
    cfg: Config,
    models: Models,
    snapshot: Any,
    seed: int = 7,
) -> Callable[[Any], list[str]]:
    rng = np.random.default_rng(seed)

    def policy(row: Any) -> list[str]:
        req = _request_from_row(row, models.profiles)
        session = Session.from_request(req)
        decision = recommend(session, models, snapshot, cfg, rng=rng)
        return decision.items

    policy.__name__ = name
    return policy


def make_live_feed_policy(
    name: str,
    cfg: Config,
    models: Models,
    snapshot: Any,
    purchases: pd.DataFrame,
    seed: int = 7,
) -> Callable[[Any], list[str]]:
    """`recommend()` plus a chronological replay of the live purchase feed.

    The bundle's per-user state is frozen at the cut, exactly like the served
    model before the round starts. This policy walks the sessions in `ts` order
    and, before each one, ingests every purchase that happened earlier — the
    same contract as `LiveState._refresh_purchases`, so the comparison against
    the identical policy without the feed is a real A/B of the feature.
    """
    live_models: Models = dataclasses.replace(
        models,
        recent_purchases={u: dict(v) for u, v in models.recent_purchases.items()},
        purchased_items={u: list(v) for u, v in models.purchased_items.items()},
    )
    profiles = models.profiles
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = (
        [
            {"user_id": r["user_id"], "product_id": r["product_id"], "last_ts": r["first_ts"]}
            for r in purchases.to_dict("records")
        ]
        if not purchases.empty else []
    )
    starts = (
        purchases["first_ts"].astype("int64").to_numpy() / 1e9
        if not purchases.empty else np.zeros(0, dtype=float)
    )
    state = {"ptr": 0, "ingested": 0, "sessions": 0}

    def policy(row: Any) -> list[str]:
        ts = parse_ts(getattr(row, "ts", None)) or 0.0
        if state["ptr"] < len(rows):
            batch = []
            while state["ptr"] < len(rows) and starts[state["ptr"]] < ts:
                batch.append(rows[state["ptr"]])
                state["ptr"] += 1
            if batch:
                live_models.ingest_purchases(
                    batch, window_h=cfg.recent_purchase_window_h, now=ts
                )
                state["ingested"] += len(batch)
        state["sessions"] += 1
        req = _request_from_row(row, profiles)
        session = Session.from_request(req)
        decision = recommend(session, live_models, snapshot, cfg, rng=rng)
        return decision.items

    policy.__name__ = name
    policy.feed_stats = state  # type: ignore[attr-defined]
    return policy


def resolve_policy(name: str) -> tuple[Config, bool, bool]:
    """`deployed_live_seeds` -> (Config.load(), feed=True, seeds=True).

    Suffixes compose: `_live` replays the purchase feed, `_seeds` also uses the
    shopper's recent purchases as candidate seeds.
    """
    seeds = name.endswith("_seeds")
    base = name[: -len("_seeds")] if seeds else name
    live = base.endswith("_live")
    base = base[: -len("_live")] if live else base
    if base == "default":
        cfg = Config()
    elif base == "deployed":
        cfg = Config.load()
    elif base in PHASE_POLICIES:
        cfg = PHASE_POLICIES[base]
    else:
        raise KeyError(
            f"unknown policy {name!r}; use a name from {sorted(PHASE_POLICIES)}, "
            "'default' or 'deployed', optionally with '_live' and/or '_seeds'"
        )
    if seeds:
        cfg = cfg.with_(use_purchase_seeds=True)
    return cfg, live, seeds


def paired_bootstrap(
    ref: list[float | None],
    other: list[float | None],
    n_boot: int = 1000,
    seed: int = 0,
) -> dict[str, float]:
    """Mean per-session delta (`other - ref`) with a 95% bootstrap interval.

    Metrics here are means over ~6k sessions but the interesting denominators
    are tiny (≈100 buyers, ≈220 clickers). A paired bootstrap on the *same*
    sessions is the honest way to say whether a difference is real.
    """
    pairs = [(b, a) for a, b in zip(ref, other) if a is not None and b is not None]
    if not pairs:
        return {}
    diffs = np.array([b - a for b, a in pairs], dtype=float)
    rng = np.random.default_rng(seed)
    means = np.array(
        [diffs[rng.integers(0, len(diffs), len(diffs))].mean() for _ in range(n_boot)]
    )
    return {
        "delta": round(float(diffs.mean()), 5),
        "ci_low": round(float(np.percentile(means, 2.5)), 5),
        "ci_high": round(float(np.percentile(means, 97.5)), 5),
        "n": int(len(diffs)),
        "win_rate": round(float((diffs > 0).mean()), 3),
    }


def best_seller_policy(
    models: Models,
    active: set[str],
    k: int = 10,
) -> Callable[[Any], list[str]]:
    best = [p for p in models.popular_sorted if p in active][:k]

    def policy(_row: Any) -> list[str]:
        return best

    return policy


def evaluate(
    source_kind: str,
    models_tag: str,
    cut: str,
    limit: int,
    policies: list[str],
    out: Path | None,
    seed: int = 7,
    variants: dict[str, dict[str, Any]] | None = None,
    salt: str | None = None,
    keep_rows: bool = False,
    blank_history: bool = False,
) -> dict[str, Any]:
    source = make_source(source_kind)
    models = Models.load(models_tag, verbose=False)
    test = datasets.eval_sessions(source, cut, limit=limit, salt=salt)
    if test.empty:
        raise RuntimeError(f"no evaluation sessions at cut {cut}")

    if blank_history:
        # Cold-start simulation. The history table has no `is_new_user` and no
        # sessions for shoppers who have never been seen, so the only way to
        # measure the cold path offline is to take real sessions and remove
        # everything we know about the shopper: identity, profile and the
        # in-session view trail. The labels stay, so the outcomes are real.
        test = test.copy()
        test["user_id"] = ""
        test["recent_views"] = [[] for _ in range(len(test))]
        test["is_new_user"] = True
        if "_cold" in test:
            test["_cold"] = True
        print("[eval] blank-history: identity, profile and view trail removed "
              f"from all {len(test)} sessions (labels kept)", flush=True)

    active = {pid for pid, item in models.catalog.items() if item.is_active}
    snapshot = static_snapshot(models, new_ids=())
    # The replay policy needs chronological order; every metric here is a mean
    # over rows, so the order is invariant for the other policies (they all see
    # the same sample either way).
    test = test.sort_values("ts").reset_index(drop=True)

    replay: dict[str, pd.DataFrame] = {}

    def replay_purchases() -> pd.DataFrame:
        if "df" not in replay:
            end = (test.ts.max() + pd.Timedelta(days=1)).isoformat()
            replay["df"] = datasets.replay_purchases(source, cut, end)
            print(f"[eval] live feed replay: {len(replay['df'])} purchases, "
                  f"[{cut}, {end}]", flush=True)
        return replay["df"]

    baseline = best_seller_policy(models, active)
    results: dict[str, Any] = {
        "dataset": "rawtree-real",
        "source": source_kind,
        "models_tag": models_tag,
        "cut": cut,
        "sample_salt": salt,
        "n_sessions": int(len(test)),
        "catalog_items": len(models.catalog),
        "catalog_active": len(active),
        "cold_simulation": "blank_history" if blank_history else None,
        "policies": {},
    }

    t0 = time.time()
    results["policies"]["baseline_best_sellers"] = evaluate_policy(
        baseline, test, clip=Config().ips_weight_clip
    )
    print(f"[eval] baseline_best_sellers {time.time() - t0:.1f}s")

    started_all = time.time()
    row_arrays: dict[str, dict[str, list]] = {}

    for name in policies:
        cfg, use_feed, use_seeds = resolve_policy(name)
        if cfg.use_ranker and models.ranker is None:
            cfg = cfg.with_(use_ranker=False)
        t0 = time.time()
        if use_feed:
            fn = make_live_feed_policy(name, cfg, models, snapshot,
                                       replay_purchases(), seed=seed)
        else:
            fn = make_policy(name, cfg, models, snapshot, seed=seed)
        metrics = evaluate_policy(fn, test, clip=cfg.ips_weight_clip, keep_rows=True)
        rows = metrics.pop("_rows", None)
        if rows:
            row_arrays[name] = rows
            if keep_rows:
                # Keyed by request_id so `training/compare_runs.py` can pair two
                # bundles that were evaluated on the same deterministic sample.
                metrics["rows"] = {
                    rid: [rows["purchase_hit"][i], rows["click_hit"][i], rows["ndcg_purchase"][i]]
                    for i, rid in enumerate(test.request_id)
                }
        metrics["ranker_used"] = bool(cfg.use_ranker and models.ranker is not None)
        metrics["config"] = cfg.to_dict()
        metrics["experiment"] = {"live_feed": use_feed, "purchase_seeds": use_seeds}
        if use_feed:
            metrics["live_feed"] = dict(getattr(fn, "feed_stats", {}))
        results["policies"][name] = metrics
        print(f"[eval] {name:16s} purchase_hit={metrics['purchase_hit_rate@10']:.4f} "
              f"p95={metrics.get('compute_ms', {}).get('p95', float('nan')):.2f}ms "
              f"({time.time() - t0:.1f}s)")

    results["elapsed_s"] = round(time.time() - started_all, 2)
    for name, overrides in (variants or {}).items():
        # Ablations are one field off the *deployed* config, not off the code
        # defaults: otherwise a `ranker_blend` sweep would silently also flip
        # `mmr_lambda` back to 0.16 and confound the result.
        cfg = Config.load().with_(**overrides)
        if cfg.use_ranker and models.ranker is None:
            cfg = cfg.with_(use_ranker=False)
        t0 = time.time()
        metrics = evaluate_policy(
            make_policy(name, cfg, models, snapshot, seed=seed),
            test,
            clip=cfg.ips_weight_clip,
            keep_rows=True,
        )
        rows = metrics.pop("_rows", None)
        if rows:
            row_arrays[name] = rows
        metrics["variant_overrides"] = dict(overrides)
        metrics["ranker_used"] = bool(cfg.use_ranker and models.ranker is not None)
        results["policies"][name] = metrics
        print(f"[eval] {name:16s} purchase_hit={metrics['purchase_hit_rate@10']:.4f} "
              f"cold_hit={metrics.get('cold_purchase_hit_rate@10', float('nan')):.4f} "
              f"ndcg={metrics['ndcg@10_purchase']:.4f} "
              f"({time.time() - t0:.1f}s)")
    # Pair every policy and variant against the first policy of the run, on the
    # same sessions. Must run last: variants join `row_arrays` in the loop above.
    ref_name = next(iter(row_arrays), None)
    if ref_name:
        for name, rows in row_arrays.items():
            if name == ref_name:
                continue
            results["policies"][name][f"paired_vs_{ref_name}"] = {
                metric: paired_bootstrap(row_arrays[ref_name][metric], rows[metric])
                for metric in ("purchase_hit", "click_hit", "ndcg_purchase")
            }
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
        print(f"[eval] wrote {out}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("rawtree",), default="rawtree")
    parser.add_argument("--models", required=True, help="artefact tag, e.g. 2026-11-07 or prod")
    parser.add_argument("--cut", required=True, help="evaluation cut (sessions >= cut)")
    parser.add_argument("--limit", type=int, default=3000)
    parser.add_argument("--policies", nargs="+", default=list(PHASE_POLICIES))
    parser.add_argument("--variants", default=None,
                        help="JSON file {name: {config field overrides}} for ablations")
    parser.add_argument("--out", default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--salt", default="v1",
                        help="deterministic sample id; same (cut, salt, limit) = same sessions")
    parser.add_argument("--random-sample", action="store_true",
                        help="old behaviour: ORDER BY rand() (not reproducible)")
    parser.add_argument("--keep-rows", action="store_true",
                        help="store per-request outcomes so two runs can be paired")
    parser.add_argument("--blank-history", action="store_true",
                        help="cold-start simulation: strip user_id, recent_views and "
                             "profile from every session before scoring (labels kept)")
    args = parser.parse_args()

    variants = None
    if args.variants:
        variants = json.loads(Path(args.variants).read_text(encoding="utf-8"))
    out = Path(args.out) if args.out else (
        REPORTS / f"offline_{args.source}_{args.cut}.json"
    )
    evaluate(args.source, args.models, args.cut, args.limit, args.policies, out,
             args.seed, variants, None if args.random_sample else args.salt,
             args.keep_rows, args.blank_history)


# =====================================================================
# Metrics: hit rate, NDCG@10, IPS/SNIPS on slot 1, validity, cold slices
# =====================================================================
from src.common import summarize  # noqa: E402


import math
import time
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd


Policy = Callable[[Any], list[str]]


def _clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)


def _as_list(value: Any) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def ndcg_at_k(recs: Iterable[list[str]], relevant: Iterable[str], k: int = 10) -> float:
    """Binary NDCG@k. One relevant item per impression, so IDCG = 1."""
    total = 0.0
    n = 0
    for rec, rel in zip(recs, relevant):
        rel = _clean(rel)
        if not rel:
            continue
        n += 1
        try:
            rank = rec.index(rel) + 1
        except ValueError:
            continue
        if rank <= k:
            total += 1.0 / math.log2(rank + 1)
    return total / max(n, 1)


def evaluate_policy(
    policy: Policy,
    df: pd.DataFrame,
    clip: float = 50.0,
    measure_latency: bool = True,
    keep_rows: bool = False,
) -> dict[str, Any]:
    recs: list[list[str]] = []
    latencies: list[float] = []
    errors = 0
    for row in df.itertuples():
        started = time.perf_counter()
        try:
            rec = list(policy(row))
        except Exception:
            rec = []
            errors += 1
        if measure_latency:
            latencies.append((time.perf_counter() - started) * 1000.0)
        recs.append(rec)

    clicked = df.clicked.map(_clean).to_numpy()
    purchased = df.purchased.map(_clean).to_numpy()
    has_click = clicked != ""
    has_purchase = purchased != ""
    shown = df.shown.map(_as_list).to_list()
    first_shown = np.array([s[0] if s else "" for s in shown], dtype=object)
    prop0 = np.array([
        float(p[0]) if len(p) else 1.0 for p in df.propensity.map(_as_list)
    ])
    match = np.array([bool(r) and r[0] == s for r, s in zip(recs, first_shown)])
    weights = np.where(match, 1.0 / np.clip(prop0, 1e-9, None), 0.0)
    clipped = (weights > clip) & match
    weights = np.minimum(weights, clip)

    reward_click = ((clicked == first_shown) & has_click).astype(float)
    reward_buy = ((purchased == first_shown) & has_purchase).astype(float)

    def _hit(targets: np.ndarray, mask: np.ndarray) -> float:
        if mask.sum() == 0:
            return float("nan")
        idx = np.nonzero(mask)[0]
        return float(np.mean([targets[i] in recs[i] for i in idx]))

    def _valid(rec: list[str]) -> bool:
        return len(rec) == 10 and len(set(rec)) == 10

    valid_rate = float(np.mean([_valid(r) for r in recs])) if recs else 0.0
    cold = _cold_mask(df)
    out: dict[str, Any] = {
        "n": int(len(df)),
        "errors": errors,
        "click_hit_rate@10": _hit(clicked, has_click),
        "purchase_hit_rate@10": _hit(purchased, has_purchase),
        "ndcg@10_click": ndcg_at_k(recs, clicked),
        "ndcg@10_purchase": ndcg_at_k(recs, purchased),
        "ips_ctr_slot1": float(np.mean(weights * reward_click)),
        "snips_ctr_slot1": float((weights * reward_click).sum() / max(weights.sum(), 1e-9)),
        "ips_purchase_slot1": float(np.mean(weights * reward_buy)),
        "snips_purchase_slot1": float((weights * reward_buy).sum() / max(weights.sum(), 1e-9)),
        "slot1_matches": int(match.sum()),
        "slot1_match_rate": float(match.mean()) if len(match) else 0.0,
        "ips_clipped_share": float(clipped.sum() / max(match.sum(), 1)),
        "valid_list_rate": valid_rate,
        "observed_click_rate": float(has_click.mean()),
        "observed_purchase_rate": float(has_purchase.mean()),
        "logged_revenue_per_request": float(pd.to_numeric(df.get("revenue_eur", 0), errors="coerce").fillna(0).mean()),
    }
    # Revenue can only be measured counterfactually in the live simulator;
    # offline we report the logged revenue of impressions our list would hit.
    if "revenue_eur" in df:
        rev = pd.to_numeric(df["revenue_eur"], errors="coerce").fillna(0.0).to_numpy()
        hit_purchase = np.array([bool(p) and p in r for p, r in zip(purchased, recs)])
        out["revenue_on_hits_per_request"] = float((rev * hit_purchase).mean())
    if measure_latency:
        out["compute_ms"] = summarize(latencies)
    if keep_rows:
        # Per-row indicators so the caller can run a paired bootstrap: comparing
        # two policies on the same sessions is the only way to tell a +3 point
        # difference from 3 lucky buyers.
        buy_row = [
            (1.0 if purchased[i] and purchased[i] in recs[i] else 0.0) if has_purchase[i] else None
            for i in range(len(recs))
        ]
        click_row = [
            (1.0 if clicked[i] and clicked[i] in recs[i] else 0.0) if has_click[i] else None
            for i in range(len(recs))
        ]
        ndcg_row: list[float | None] = []
        for i, rec in enumerate(recs):
            rel = _clean(purchased[i])
            if not rel:
                ndcg_row.append(None)
                continue
            try:
                rank = rec.index(rel) + 1
            except ValueError:
                ndcg_row.append(0.0)
                continue
            ndcg_row.append(1.0 / math.log2(rank + 1) if rank <= 10 else 0.0)
        out["_rows"] = {
            "purchase_hit": buy_row,
            "click_hit": click_row,
            "ndcg_purchase": ndcg_row,
        }
    if cold.sum() > 0:
        sub = df[cold]
        sub_recs = [r for r, c in zip(recs, cold) if c]
        sub_clicked = clicked[cold]
        sub_purchased = purchased[cold]
        sub_mask_c = sub_clicked != ""
        sub_mask_p = sub_purchased != ""
        out.update({
            "cold_n": int(cold.sum()),
            "cold_click_hit_rate@10": _hit(sub_clicked, sub_mask_c),
            "cold_purchase_hit_rate@10": _hit(sub_purchased, sub_mask_p),
            "cold_ndcg@10_purchase": ndcg_at_k(sub_recs, sub_purchased),
            "cold_observed_purchase_rate": float(sub_mask_p.mean()),
        })
    return out


def _cold_mask(df: pd.DataFrame) -> np.ndarray:
    if "is_new_user" in df:
        is_new = df.is_new_user.fillna(False).astype(bool).to_numpy()
    else:
        is_new = np.zeros(len(df), dtype=bool)
    if "recent_views" not in df:
        return is_new
    # Offline proxy: sessions that mention a product created after the training
    # cut. The live definition comes from `catalog_updates` (`new_product`).
    if "_cold" in df:
        return df["_cold"].to_numpy(dtype=bool) | is_new
    return is_new


def evaluate_many(
    policies: dict[str, Policy],
    df: pd.DataFrame,
    clip: float = 50.0,
) -> pd.DataFrame:
    rows = {}
    for name, policy in policies.items():
        rows[name] = evaluate_policy(policy, df, clip=clip)
    return pd.DataFrame(rows).T


def compare(new: dict[str, Any], ref: dict[str, Any], metric: str = "purchase_hit_rate@10") -> dict[str, float]:
    base = ref.get(metric, float("nan"))
    cur = new.get(metric, float("nan"))
    return {
        "metric": metric,
        "reference": base,
        "candidate": cur,
        "absolute": cur - base,
        "relative_pct": 100.0 * (cur / base - 1.0) if base and not math.isnan(base) else float("nan"),
    }


if __name__ == "__main__":
    main()

