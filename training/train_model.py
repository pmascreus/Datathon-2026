"""Precalcula co-compres, perfils d'usuari i best-sellers a partir de les taules
d'històric (shop.history_events, shop.history_sessions).

Correu-ho un cop abans que comenci el stream (i torneu-ho a córrer de tant en tant
durant la ronda de pràctiques si voleu refrescar-ho amb dades més recents). Això
NOMÉS llegeix shop.history_* (estàtic, 60 dies de passat) — mai shop.events/shop.requests,
que només existeixen amb el stream en marxa. Els senyals en viu (estoc, tendència
d'última hora) no es calculen aquí: veure src/features.py.

    python training/train_model.py                        # amb tot l'històric (producció)
    python training/train_model.py --until 2026-11-10      # només abans d'aquesta data
                                                             # (per validar offline amb
                                                             # notebooks/evaluacio.ipynb)
"""
from __future__ import annotations

import argparse
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.rawtree_client import sql

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

# pes de cada tipus d'esdeveniment en construir co-ocurrències i perfils d'usuari:
# una compra pesa molt més que un simple view
EVENT_WEIGHT = {"view": 1.0, "add_to_cart": 3.0, "purchase": 6.0}


def best_sellers(until: str | None = None) -> list[str]:
    """Best-sellers de sempre (empat: product_id més alt primer). Mateixa definició
    que starter/recommender.py, així serveix de fallback compatible."""
    cut = f" AND ts < '{until}'" if until else ""
    rows = sql(f"""
        SELECT product_id::String AS product_id, count() AS n
        FROM shop.history_events
        WHERE event_type = 'purchase'{cut}
        GROUP BY product_id""")
    ranked = sorted(((int(r["n"]), r["product_id"]) for r in rows), reverse=True)
    return [p for _, p in ranked]


def co_occurrence(until: str | None = None, top_k: int = 20) -> dict[str, list[tuple[str, float]]]:
    """Per a cada producte p, els seus top_k productes relacionats q, puntuats per
    quantes vegades apareixen a la mateixa sessió, ponderat perquè una parella amb
    compra conti molt més que una parella amb només vistes."""
    cut = f" AND ts < '{until}'" if until else ""
    scores: defaultdict[str, Counter] = defaultdict(Counter)
    for event_type, weight in EVENT_WEIGHT.items():
        rows = sql(f"""
            SELECT a.product_id AS p, b.product_id AS q, count() AS n
            FROM (SELECT session_id::String AS session_id, product_id::String AS product_id
                  FROM shop.history_events WHERE event_type = '{event_type}'{cut}) AS a
            INNER JOIN (SELECT session_id::String AS session_id, product_id::String AS product_id
                        FROM shop.history_events WHERE event_type = '{event_type}'{cut}) AS b
              ON a.session_id = b.session_id
            WHERE a.product_id != b.product_id
            GROUP BY p, q ORDER BY n DESC LIMIT {top_k} BY p""")
        for r in rows:
            scores[r["p"]][r["q"]] += weight * r["n"]
        print(f"  co-ocurrència des de '{event_type}': {len(rows)} parelles")
    return {p: c.most_common(top_k) for p, c in scores.items()}


def user_profiles(until: str | None = None, min_events: int = 2) -> dict[str, list[tuple[str, float]]]:
    """Per a cada usuari amb almenys min_events esdeveniments, els productes que ha
    vist/afegit/comprat, ponderats igual que co_occurrence."""
    cut = f" AND ts < '{until}'" if until else ""
    # 'view' té tants esdeveniments que el GROUP BY sol no cap en un resultat (massa
    # parelles usuari-producte vistes només un cop, pur soroll). Exigim almenys 2
    # repeticions només per a 'view'; una sola compra o afegit al cistell sí compta.
    min_n = {"view": 2, "add_to_cart": 1, "purchase": 1}
    profiles: defaultdict[str, Counter] = defaultdict(Counter)
    for event_type, weight in EVENT_WEIGHT.items():
        rows = sql(f"""
            SELECT user_id::String AS user_id, product_id::String AS product_id, count() AS n
            FROM shop.history_events
            WHERE event_type = '{event_type}'{cut}
            GROUP BY user_id, product_id
            HAVING n >= {min_n[event_type]}
            ORDER BY n DESC LIMIT 50 BY user_id""")
        for r in rows:
            profiles[r["user_id"]][r["product_id"]] += weight * r["n"]
        print(f"  perfils des de '{event_type}': {len(rows)} parelles (usuari, producte)")
    return {u: c.most_common(50) for u, c in profiles.items() if sum(c.values()) >= min_events}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--until", default=None,
                         help="YYYY-MM-DD: només usa historial anterior a aquesta data (validació offline)")
    args = parser.parse_args()

    print("best sellers...")
    best = best_sellers(args.until)
    print(f"  {len(best)} productes amb almenys una venda")

    print("co-ocurrència...")
    covisit = co_occurrence(args.until)
    print(f"  {len(covisit)} productes amb relacionats")

    print("perfils d'usuari...")
    profiles = user_profiles(args.until)
    print(f"  {len(profiles)} usuaris amb perfil")

    suffix = f"_{args.until}" if args.until else ""
    with open(MODELS_DIR / f"best_sellers{suffix}.pkl", "wb") as f:
        pickle.dump(best, f)
    with open(MODELS_DIR / f"co_purchases{suffix}.pkl", "wb") as f:
        pickle.dump(covisit, f)
    with open(MODELS_DIR / f"user_profiles{suffix}.pkl", "wb") as f:
        pickle.dump(profiles, f)
    print(f"guardat a {MODELS_DIR} (suffix='{suffix}')")


if __name__ == "__main__":
    main()