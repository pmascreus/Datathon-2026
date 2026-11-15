"""training/train_model.py: Precalcula matrius direccionals, perfils sense recompres,
estadístiques de conversió i entrena el model classificador (ranker.pkl)."""
from __future__ import annotations

import argparse
import math
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.rawtree_client import sql

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

EVENT_WEIGHT = {"view": 1.0, "add_to_cart": 3.5, "purchase": 6.0}


def best_sellers(until: str | None = None) -> list[str]:
    cut = f" AND ts < '{until}'" if until else ""
    rows = sql(f"""
        SELECT product_id::String AS product_id, count() AS n
        FROM shop.history_events
        WHERE event_type::String = 'purchase'{cut}
        GROUP BY product_id""")
    ranked = sorted(((int(r["n"]), r["product_id"]) for r in rows), reverse=True)
    return [p for _, p in ranked]


def fetch_catalog_meta() -> dict[str, dict]:
    rows = sql("""
        SELECT product_id::String AS product_id, category::String AS category,
               subcategory::String AS subcategory, brand::String AS brand,
               price_eur::Float64 AS price_eur
        FROM shop.catalog""")
    return {r["product_id"]: r for r in rows}


def directional_view_to_buy(until: str | None = None, top_k: int = 25) -> dict[str, list[tuple[str, float]]]:
    """Cross-correlation direccional: l'usuari mira 'p' i en la mateixa sessió compra
    o afegeix al cistell 'q' (amb p != q)."""
    cut = f" AND ts < '{until}'" if until else ""
    scores: defaultdict[str, Counter] = defaultdict(Counter)
    rows = sql(f"""
        SELECT a.product_id AS p, b.product_id AS q,
               countIf(b.etype = 'purchase') * 4.0 + countIf(b.etype = 'add_to_cart') * 1.5 AS score
        FROM (
            SELECT session_id::String AS session_id, product_id::String AS product_id
            FROM shop.history_events
            WHERE event_type::String = 'view'{cut}
            GROUP BY session_id, product_id
        ) AS a
        INNER JOIN (
            SELECT session_id::String AS session_id, product_id::String AS product_id, event_type::String AS etype
            FROM shop.history_events
            WHERE event_type::String IN ('purchase', 'add_to_cart'){cut}
        ) AS b ON a.session_id = b.session_id
        WHERE a.product_id != b.product_id
        GROUP BY p, q
        ORDER BY score DESC
        LIMIT {top_k} BY p""")
    for r in rows:
        scores[r["p"]][r["q"]] += float(r["score"])
    print(f"  view_to_buy direccional: {len(rows)} parelles")
    return {p: c.most_common(top_k) for p, c in scores.items()}


def co_occurrence(until: str | None = None, top_k: int = 25) -> dict[str, list[tuple[str, float]]]:
    cut = f" AND ts < '{until}'" if until else ""
    scores: defaultdict[str, Counter] = defaultdict(Counter)
    for event_type, weight in EVENT_WEIGHT.items():
        rows = sql(f"""
            SELECT a.product_id AS p, b.product_id AS q, count() AS n
            FROM (SELECT session_id::String AS session_id, product_id::String AS product_id
                  FROM shop.history_events WHERE event_type::String = '{event_type}'{cut}) AS a
            INNER JOIN (SELECT session_id::String AS session_id, product_id::String AS product_id
                        FROM shop.history_events WHERE event_type::String = '{event_type}'{cut}) AS b
              ON a.session_id = b.session_id
            WHERE a.product_id != b.product_id
            GROUP BY p, q ORDER BY n DESC LIMIT {top_k} BY p""")
        for r in rows:
            scores[r["p"]][r["q"]] += weight * float(r["n"])
        print(f"  co-ocurrència '{event_type}': {len(rows)} parelles")
    return {p: c.most_common(top_k) for p, c in scores.items()}


def user_profiles_and_purchases(until: str | None = None) -> tuple[dict[str, list[tuple[str, float]]], dict[str, set[str]]]:
    """Retorna (profiles, user_bought). Separem el que l'usuari ja ha comprat perquè
    no li tornem a recomanar el mateix producte."""
    cut = f" AND ts < '{until}'" if until else ""
    bought_rows = sql(f"""
        SELECT user_id::String AS user_id, product_id::String AS product_id
        FROM shop.history_events
        WHERE event_type::String = 'purchase'{cut}
        GROUP BY user_id, product_id""")
    user_bought: dict[str, set[str]] = defaultdict(set)
    for r in bought_rows:
        user_bought[r["user_id"]].add(r["product_id"])

    min_n = {"view": 2, "add_to_cart": 1, "purchase": 1}
    profiles: defaultdict[str, Counter] = defaultdict(Counter)
    for event_type, weight in EVENT_WEIGHT.items():
        rows = sql(f"""
            SELECT user_id::String AS user_id, product_id::String AS product_id, count() AS n
            FROM shop.history_events
            WHERE event_type::String = '{event_type}'{cut}
            GROUP BY user_id, product_id
            HAVING n >= {min_n[event_type]}
            ORDER BY n DESC LIMIT 40 BY user_id""")
        for r in rows:
            profiles[r["user_id"]][r["product_id"]] += weight * float(r["n"])
    clean_profiles = {u: c.most_common(40) for u, c in profiles.items()}
    return clean_profiles, dict(user_bought)


def item_conversion_stats(until: str | None = None) -> dict[str, dict[str, float]]:
    """Calcula CTR i CVR històrics de cada producte."""
    cut = f" AND ts < '{until}'" if until else ""
    rows = sql(f"""
        SELECT product_id::String AS product_id,
               countIf(event_type::String = 'view') AS views,
               countIf(event_type::String = 'add_to_cart') AS carts,
               countIf(event_type::String = 'purchase') AS purchases
        FROM shop.history_events
        WHERE 1=1{cut}
        GROUP BY product_id""")
    stats: dict[str, dict[str, float]] = {}
    for r in rows:
        v = float(r["views"])
        c = float(r["carts"])
        p = float(r["purchases"])
        stats[r["product_id"]] = {
            "views": v,
            "purchases": p,
            "cvr": (p + 2.0) / (v + 100.0),
            "cart_rate": (c + 2.0) / (v + 80.0),
            "pop_log": math.log1p(p),
        }
    return stats


def extract_candidate_features(
    cand: str,
    recent_views: list[dict],
    user_id: str | None,
    context: dict,
    catalog: dict[str, dict],
    co_purchases: dict[str, list[tuple[str, float]]],
    view_to_buy: dict[str, list[tuple[str, float]]],
    user_profiles: dict[str, list[tuple[str, float]]],
    user_bought: dict[str, set[str]],
    item_stats: dict[str, dict[str, float]],
    emb_sim_fn=None,
) -> list[float]:
    viewed_ids = [v["product_id"] for v in recent_views if isinstance(v, dict) and "product_id" in v]

    f_coview = 0.0
    f_v2b = 0.0
    f_emb = 0.0
    f_same_subcat = 0.0
    f_same_cat = 0.0
    f_same_brand = 0.0
    session_prices = []

    cand_meta = catalog.get(cand) or {}
    cand_sub = cand_meta.get("subcategory", "")
    cand_cat = cand_meta.get("category", "")
    cand_brand = cand_meta.get("brand", "")
    cand_price = float(cand_meta.get("price_eur") or 0.0)

    for k, pid in enumerate(reversed(viewed_ids[:6])):
        decay = 0.72 ** k
        p_meta = catalog.get(pid) or {}
        if p_meta.get("price_eur"):
            session_prices.append(float(p_meta["price_eur"]))
        if cand_sub and p_meta.get("subcategory") == cand_sub:
            f_same_subcat += decay
        if cand_cat and p_meta.get("category") == cand_cat:
            f_same_cat += decay
        if cand_brand and p_meta.get("brand") == cand_brand:
            f_same_brand += decay

        for rank, (q, w) in enumerate(co_purchases.get(pid, []), start=1):
            if q == cand:
                f_coview += decay * (1.0 + 0.15 * math.log1p(w)) / rank
                break

        for rank, (q, w) in enumerate(view_to_buy.get(pid, []), start=1):
            if q == cand:
                f_v2b += decay * (1.0 + 0.2 * math.log1p(w)) / rank
                break

        if emb_sim_fn is not None and k < 2:
            f_emb += decay * emb_sim_fn(pid, cand)

    f_user_interest = 0.0
    f_user_complement = 0.0
    bought_set = user_bought.get(user_id, set()) if user_id else set()
    if user_id and user_id in user_profiles:
        for rank_u, (p, w) in enumerate(user_profiles[user_id][:15], start=1):
            if p == cand and cand not in bought_set:
                f_user_interest += math.log1p(w) / rank_u
            for rank_r, (q, _) in enumerate(view_to_buy.get(p, [])[:6], start=1):
                if q == cand:
                    f_user_complement += 1.0 / (rank_u * rank_r)
                    break

    if session_prices and cand_price > 0:
        avg_p = sum(session_prices) / len(session_prices)
        f_price_diff = abs(cand_price - avg_p) / max(avg_p, 10.0)
    else:
        f_price_diff = 0.3

    st = item_stats.get(cand) or {}
    f_cvr = st.get("cvr", 0.015)
    f_cart_rate = st.get("cart_rate", 0.025)
    f_pop = st.get("pop_log", 1.0)

    page = (context or {}).get("page", "") if isinstance(context, dict) else ""
    f_is_prod_page = 1.0 if page == "product" else 0.0

    return [
        f_v2b,
        f_coview,
        f_emb,
        f_same_subcat,
        f_same_cat,
        f_same_brand,
        f_user_interest,
        f_user_complement,
        f_price_diff,
        f_cvr,
        f_cart_rate,
        f_pop + 0.1 * f_is_prod_page,
    ]


def train_ranker_model(
    until: str | None,
    catalog: dict[str, dict],
    co_purchases: dict[str, list[tuple[str, float]]],
    view_to_buy: dict[str, list[tuple[str, float]]],
    user_profiles: dict[str, list[tuple[str, float]]],
    user_bought: dict[str, set[str]],
    item_stats: dict[str, dict[str, float]],
) -> HistGradientBoostingClassifier:
    cut = f" AND ts < '{until}'" if until else ""
    print("descarregant sessions d'entrenament de shop.history_sessions...")
    sessions = sql(f"""
        SELECT user_id::String AS user_id, context, recent_views, shown,
               clicked::String AS clicked, purchased::String AS purchased
        FROM shop.history_sessions
        WHERE clicked IS NOT NULL AND clicked::String != ''{cut}
        ORDER BY rand()
        LIMIT 30000""")

    X, y, sample_weight = [], [], []
    for s in sessions:
        rv = s.get("recent_views") or []
        uid = s.get("user_id")
        ctx = s.get("context") or {}
        shown = s.get("shown") or []
        clicked = s.get("clicked") or ""
        purchased = s.get("purchased") or ""

        for pid in shown:
            feats = extract_candidate_features(
                pid, rv, uid, ctx, catalog, co_purchases, view_to_buy,
                user_profiles, user_bought, item_stats, emb_sim_fn=None
            )
            X.append(feats)
            if pid == purchased:
                y.append(1)
                sample_weight.append(4.0)
            elif pid == clicked:
                y.append(1)
                sample_weight.append(1.0)
            else:
                y.append(0)
                sample_weight.append(1.0)

    X_arr = np.array(X, dtype=np.float32)
    y_arr = np.array(y, dtype=np.int32)
    w_arr = np.array(sample_weight, dtype=np.float32)

    print(f"entrenant HistGradientBoostingClassifier sobre {len(X_arr)} exemples...")
    clf = HistGradientBoostingClassifier(
        max_iter=120,
        learning_rate=0.06,
        max_depth=5,
        min_samples_leaf=40,
        random_state=42,
    )
    clf.fit(X_arr, y_arr, sample_weight=w_arr)
    return clf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--until", default=None, help="YYYY-MM-DD per validació offline")
    args = parser.parse_args()

    print("1/6 catàleg i best sellers...")
    catalog = fetch_catalog_meta()
    best = best_sellers(args.until)

    print("2/6 cross-correlation direccional (view -> buy)...")
    v2b = directional_view_to_buy(args.until)

    print("3/6 co-ocurrència general...")
    covisit = co_occurrence(args.until)

    print("4/6 perfils d'usuari i exclusió de compres prèvies...")
    profiles, bought = user_profiles_and_purchases(args.until)

    print("5/6 estadístiques de conversió per producte...")
    istats = item_conversion_stats(args.until)

    print("6/6 entrenant classificador ranker...")
    ranker = train_ranker_model(args.until, catalog, covisit, v2b, profiles, bought, istats)

    suffix = f"_{args.until}" if args.until else ""
    artifacts = {
        "best_sellers": best,
        "co_purchases": covisit,
        "view_to_buy": v2b,
        "user_profiles": profiles,
        "user_bought": bought,
        "item_stats": istats,
        "catalog_meta": catalog,
        "ranker": ranker,
    }
    for name, obj in artifacts.items():
        with open(MODELS_DIR / f"{name}{suffix}.pkl", "wb") as f:
            pickle.dump(obj, f)

    print(f"Tot guardat a {MODELS_DIR} (suffix='{suffix}')!")


if __name__ == "__main__":
    main()