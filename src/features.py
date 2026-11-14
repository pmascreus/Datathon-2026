"""Senyals en viu, de canvi ràpid: es refresquen cada pocs segons en un fil de fons,
mai es consulten dins del camí de resposta de 3 segons. src/recommender.py els llegeix
com a atributs normals d'un objecte LiveState compartit.

Tot el que calculeu de dades *històriques* (co-compres, perfils d'usuari, embeddings)
va a training/, no aquí: aquí només hi ha coses que canvien minut a minut mentre el
stream corre (shop.events, shop.requests, shop.catalog_updates).
"""
from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.rawtree_client import sql

UNAVAILABLE_REFRESH_S = 1    # l'estoc canvia ràpid: comproveu-ho sovint (com l'starter)
TRENDING_REFRESH_S = 15      # agregats d'última hora / 15 min es mouen més a poc a poc


def fetch_unavailable() -> set[str]:
    """Productes fora d'estoc ara mateix (mateixa query que starter/recommender.py:
    l'últim canvi d'estoc de cada producte és 'out_of_stock')."""
    rows = sql("""
        SELECT product_id::String AS product_id
        FROM shop.catalog_updates
        WHERE change::String IN ('out_of_stock', 'back_in_stock')
        GROUP BY product_id
        HAVING argMax(change::String, ts::DateTime64(3)) = 'out_of_stock'""")
    return {r["product_id"] for r in rows}


def fetch_region_trending(top_k: int = 20) -> dict[str, list[str]]:
    """Productes més venuts/vistos per regió a l'última hora, per compres i després
    per vistes (adaptació de queries.sql #2, agrupant per regió en lloc de categoria)."""
    rows = sql(f"""
        SELECT region::String AS region, product_id::String AS product_id,
               countIf(event_type = 'purchase') AS purchases,
               countIf(event_type = 'view')     AS views
        FROM shop.events
        WHERE ts > now() - INTERVAL 1 HOUR
        GROUP BY region, product_id
        ORDER BY purchases DESC, views DESC
        LIMIT {top_k} BY region""")
    out: dict[str, list[str]] = {}
    for r in rows:
        out.setdefault(r["region"], []).append(r["product_id"])
    return out


def fetch_new_products() -> dict[str, dict]:
    """Productes afegits durant l'esdeveniment (change = 'new_product'), amb el seu
    text, perquè es puguin vectoritzar al vol per al cold start (veure build_embeddings.py)."""
    rows = sql("""
        SELECT product_id::String AS product_id, category::String AS category,
               subcategory::String AS subcategory, brand::String AS brand,
               description::String AS description
        FROM shop.catalog_updates
        WHERE change::String = 'new_product'""")
    return {r["product_id"]: r for r in rows}


@dataclass
class LiveState:
    """Instantània en memòria de tot el que canvia en viu. Un sol objecte, compartit
    entre el bucle principal (que el llegeix) i el fil de fons (que l'actualitza)."""

    unavailable: set[str] = field(default_factory=set)
    region_trending: dict[str, list[str]] = field(default_factory=dict)
    new_products: dict[str, dict] = field(default_factory=dict)
    _stop: bool = field(default=False, repr=False)

    def refresh_once(self) -> None:
        """Càrrega inicial síncrona, abans de començar a respondre peticions."""
        try:
            self.unavailable = fetch_unavailable()
        except Exception as e:
            print(f"[features] primer refresc d'estoc ha fallat: {e}")
        try:
            self.region_trending = fetch_region_trending()
            self.new_products = fetch_new_products()
        except Exception as e:
            print(f"[features] primer refresc de tendències ha fallat: {e}")

    def start_background_refresh(self) -> threading.Thread:
        """Bucle que corre per sempre en el seu propi fil: ràpid per a l'estoc, més
        lent per a les tendències. Si una consulta falla, es manté l'últim valor conegut
        (igual que fa l'starter amb out_of_stock())."""

        def loop() -> None:
            last_trend = 0.0
            while not self._stop:
                try:
                    self.unavailable = fetch_unavailable()
                except Exception as e:
                    print(f"[features] refresc d'estoc ha fallat: {e}")
                if time.time() - last_trend >= TRENDING_REFRESH_S:
                    last_trend = time.time()
                    try:
                        self.region_trending = fetch_region_trending()
                        self.new_products = fetch_new_products()
                    except Exception as e:
                        print(f"[features] refresc de tendències ha fallat: {e}")
                time.sleep(UNAVAILABLE_REFRESH_S)

        t = threading.Thread(target=loop, daemon=True)
        t.start()
        return t

    def stop(self) -> None:
        self._stop = True
