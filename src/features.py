"""src/features.py: Senyals en viu refrescats en un fil de fons (1s estoc, 2s nous productes,
15s tendències regionals, descomptes i feedback)."""
from __future__ import annotations

import os
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.rawtree_client import sql

TEAM = os.environ.get("TEAM_ID", "")
UNAVAILABLE_REFRESH_S = 1    # estoc: cada 1s
NEW_PRODUCTS_REFRESH_S = 2   # productes nous: cada 2s per al cold start
TRENDING_REFRESH_S = 15      # tendències, catàleg i feedback: cada 15s


def fetch_unavailable() -> set[str]:
    rows = sql("""
        SELECT product_id::String AS product_id
        FROM shop.catalog_updates
        WHERE change::String IN ('out_of_stock', 'back_in_stock')
        GROUP BY product_id
        HAVING argMax(change::String, ts::DateTime64(3)) = 'out_of_stock'""", timeout=10)
    return {r["product_id"] for r in rows}


def fetch_new_products() -> dict[str, dict]:
    rows = sql("""
        SELECT product_id::String AS product_id, category::String AS category,
               subcategory::String AS subcategory, brand::String AS brand,
               description::String AS description, price_eur::Float64 AS price_eur,
               toString(ts) AS created_at
        FROM shop.catalog_updates
        WHERE change::String = 'new_product'""", timeout=10)
    return {r["product_id"]: r for r in rows}


def fetch_region_trending(top_k: int = 25) -> dict[str, list[str]]:
    rows = sql(f"""
        SELECT region::String AS region, product_id::String AS product_id,
               countIf(event_type = 'purchase') * 6 +
               countIf(event_type = 'add_to_cart') * 3 +
               countIf(event_type = 'view') AS score
        FROM shop.events
        WHERE ts > now() - INTERVAL 1 HOUR
        GROUP BY region, product_id
        ORDER BY score DESC
        LIMIT {top_k} BY region""", timeout=15)
    out: dict[str, list[str]] = {}
    for r in rows:
        out.setdefault(r["region"], []).append(r["product_id"])
    return out


def fetch_catalog_state() -> tuple[dict[str, dict], set[str], set[str]]:
    rows = sql("""
        SELECT product_id,
               argMax(category, v)    AS category,
               argMax(subcategory, v) AS subcategory,
               argMax(brand, v)       AS brand,
               argMax(price_eur, v)   AS price_eur,
               argMin(price_eur, v)   AS initial_price,
               argMax(is_active, v)   AS is_active,
               argMax(description, v) AS description
        FROM (
            SELECT product_id::String AS product_id, category::String AS category,
                   subcategory::String AS subcategory, brand::String AS brand,
                   price_eur::Float64 AS price_eur, is_active::Bool AS is_active,
                   description::String AS description, '' AS v
            FROM shop.catalog
            UNION ALL
            SELECT product_id::String, category::String, subcategory::String, brand::String,
                   price_eur::Float64, is_active::Bool, description::String, toString(ts)
            FROM shop.catalog_updates
        )
        GROUP BY product_id""", timeout=20)
    cat, inactive, discounted = {}, set(), set()
    for r in rows:
        pid = r["product_id"]
        cat[pid] = r
        if not r.get("is_active", True):
            inactive.add(pid)
        if float(r.get("price_eur") or 0) < float(r.get("initial_price") or 0) * 0.95:
            discounted.add(pid)
    return cat, inactive, discounted


def fetch_feedback_rates() -> dict[str, float]:
    if not TEAM:
        return {}
    rows = sql(f"""
        SELECT item AS product_id,
               count() AS shown_n,
               countIf(clicked = item) AS clicks,
               countIf(purchased = item) AS purchases
        FROM (
            SELECT arrayJoin(shown) AS item, clicked::String AS clicked, purchased::String AS purchased
            FROM {TEAM}.feedback
            WHERE request_id != '__init__' AND ts > now() - INTERVAL 2 HOUR
        )
        GROUP BY product_id
        HAVING shown_n >= 5""", timeout=15)
    rates: dict[str, float] = {}
    for r in rows:
        shown_n = float(r["shown_n"])
        reward = float(r["purchases"]) * 3.0 + float(r["clicks"]) * 0.4
        rates[r["product_id"]] = (reward + 1.0) / (shown_n + 40.0)
    return rates


@dataclass
class LiveState:
    unavailable: set[str] = field(default_factory=set)
    region_trending: dict[str, list[str]] = field(default_factory=dict)
    new_products: dict[str, dict] = field(default_factory=dict)
    catalog: dict[str, dict] = field(default_factory=dict)
    discounted: set[str] = field(default_factory=set)
    feedback_rates: dict[str, float] = field(default_factory=dict)
    _stop: bool = field(default=False, repr=False)

    def refresh_once(self) -> None:
        try:
            cat, inactive, disc = fetch_catalog_state()
            self.catalog = cat
            self.discounted = disc
            self.unavailable = fetch_unavailable() | inactive
        except Exception as e:
            print(f"[features] primer refresc d'estoc/catàleg ha fallat: {e}")
        for name, fn in (("new_products", fetch_new_products),
                         ("region_trending", fetch_region_trending),
                         ("feedback_rates", fetch_feedback_rates)):
            try:
                setattr(self, name, fn())
            except Exception as e:
                print(f"[features] primer refresc de {name} ha fallat: {e}")

    def start_background_refresh(self) -> threading.Thread:
        def loop() -> None:
            last_new = last_trend = 0.0
            inactive: set[str] = set()
            while not self._stop:
                now = time.time()
                try:
                    self.unavailable = fetch_unavailable() | inactive
                except Exception as e:
                    print(f"[features] refresc d'estoc ha fallat: {e}")
                if now - last_new >= NEW_PRODUCTS_REFRESH_S:
                    last_new = now
                    try:
                        self.new_products = fetch_new_products()
                    except Exception as e:
                        print(f"[features] refresc de productes nous ha fallat: {e}")
                if now - last_trend >= TRENDING_REFRESH_S:
                    last_trend = now
                    try:
                        cat, inactive, disc = fetch_catalog_state()
                        self.catalog = cat
                        self.discounted = disc
                        self.region_trending = fetch_region_trending()
                        self.feedback_rates = fetch_feedback_rates()
                    except Exception as e:
                        print(f"[features] refresc de tendències ha fallat: {e}")
                time.sleep(UNAVAILABLE_REFRESH_S)

        t = threading.Thread(target=loop, daemon=True)
        t.start()
        return t

    def stop(self) -> None:
        self._stop = True