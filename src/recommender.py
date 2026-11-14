"""El vostre recomanador en viu: pull -> predict -> push, mateixa forma que
starter/recommender.py, però recommend() combina co-visita de sessió, perfil llarg
d'usuari, tendència regional i veïns per embedding (cold start) en lloc de només
best-sellers.

IMPORTANT: abans de desplegar un canvi a recommend(), valideu-lo a
notebooks/evaluacio.ipynb. Aquest fitxer només és el que corre contra el stream real.

    python src/recommender.py
"""
from __future__ import annotations

import pickle
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.features import LiveState
from src.rawtree_client import Record, insert, sql

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

NEW_REQUESTS = """
SELECT request_id, ts, user_id, is_new_user, context, recent_views
FROM shop.requests
ORDER BY ts DESC
LIMIT 50"""

# Pes de cada senyal en l'ordenació final. Ajusteu-los contrastant-los a
# notebooks/evaluacio.ipynb (hit_rate@10 / SNIPS_ctr_slot1), no a ull a producció.
W_SESSION_COVIEW = 2.0
W_USER_PROFILE = 1.0
W_REGION_TREND = 0.5
W_EMBEDDING = 1.5
SESSION_DECAY = 0.7   # vistes més antigues de la sessió pesen menys (com coview() de l'starter)


class Models:
    """Tot el que ha precalculat training/, carregat un sol cop en arrencar."""

    def __init__(self) -> None:
        self.best_sellers: list[str] = self._load("best_sellers.pkl", default=[])
        self.co_purchases: dict[str, list[tuple[str, float]]] = self._load("co_purchases.pkl", default={})
        self.user_profiles: dict[str, list[tuple[str, float]]] = self._load("user_profiles.pkl", default={})

        self.embeddings: np.ndarray | None = None
        self.product_ids: list[str] = []
        self.id_to_row: dict[str, int] = {}
        emb_path, ids_path = MODELS_DIR / "embeddings.npy", MODELS_DIR / "product_ids.pkl"
        if emb_path.exists() and ids_path.exists():
            self.embeddings = np.load(emb_path)
            with open(ids_path, "rb") as f:
                self.product_ids = pickle.load(f)
            self.id_to_row = {p: i for i, p in enumerate(self.product_ids)}

        print(f"[models] {len(self.best_sellers)} best sellers, "
              f"{len(self.co_purchases)} productes amb co-compres, "
              f"{len(self.user_profiles)} perfils d'usuari, "
              f"embeddings={'si' if self.embeddings is not None else 'no'}")

    @staticmethod
    def _load(name: str, default: Any) -> Any:
        path = MODELS_DIR / name
        if not path.exists():
            print(f"[models] {name} no existeix encara, correu training/train_model.py primer "
                  f"-> faig servir un valor buit mentrestant")
            return default
        with open(path, "rb") as f:
            return pickle.load(f)

    def embedding_neighbors(self, product_id: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Productes més similars per text (categoria/marca/descripció). És el que
        dona cobertura a productes sense historial (cold start)."""
        if self.embeddings is None or product_id not in self.id_to_row:
            return []
        row = self.id_to_row[product_id]
        sims = self.embeddings @ self.embeddings[row]
        best_idx = np.argpartition(-sims, top_k + 1)[: top_k + 1]
        out = [(self.product_ids[i], float(sims[i])) for i in best_idx if self.product_ids[i] != product_id]
        out.sort(key=lambda x: -x[1])
        return out[:top_k]


def recommend(req: Record, models: Models, live: LiveState) -> list[str]:
    """10 product IDs per a una petició. Aquesta és la funció a seguir millorant
    (i validant a notebooks/evaluacio.ipynb) — tota la resta del fitxer és infraestructura."""
    score: Counter[str] = Counter()

    # 1a. co-visita dins la sessió actual (gratis: ja ve a la petició)
    recent_views = req.get("recent_views") or []
    for k, v in enumerate(reversed(recent_views)):
        pid = v["product_id"]
        related = models.co_purchases.get(pid, [])
        for rank, (q, _w) in enumerate(related, start=1):
            score[q] += W_SESSION_COVIEW * (SESSION_DECAY ** k) / rank
        if not related:  # producte sense historial -> cold start via embeddings
            for q, sim in models.embedding_neighbors(pid):
                score[q] += W_EMBEDDING * sim

    # 1b. historial llarg de l'usuari (si en té)
    user_id = req.get("user_id")
    if not req.get("is_new_user") and user_id in models.user_profiles:
        for p, n in models.user_profiles[user_id]:
            score[p] += W_USER_PROFILE * n

    # 2. tendència de la seva regió, ara mateix
    region = (req.get("context") or {}).get("region")
    for rank, p in enumerate(live.region_trending.get(region, []), start=1):
        score[p] += W_REGION_TREND / rank

    # filtre d'estoc obligatori + completar amb best-sellers si falten 10
    unavailable = live.unavailable
    ranked = [p for p, _ in score.most_common(30) if p not in unavailable]
    fallback = [p for p in models.best_sellers if p not in unavailable]
    items = (ranked + [p for p in fallback if p not in ranked])[:10]
    return items


def pull(seen: set[str]) -> list[Record]:
    """Peticions noves de shop.requests no contestades encara."""
    fresh = []
    for req in sql(NEW_REQUESTS):
        if req["request_id"] not in seen:
            seen.add(req["request_id"])
            fresh.append(req)
    return fresh


def predict(requests_: list[Record], models: Models, live: LiveState, version: str) -> list[Record]:
    return [{"request_id": req["request_id"],
             "items": recommend(req, models, live),
             "model_version": version} for req in requests_]


def push(answers: list[Record]) -> None:
    insert("submissions", answers)
    print(f"contestades {len(answers)}")


def main(model_version: str = "v1-covisit-profile-region") -> None:
    models = Models()
    live = LiveState()
    live.refresh_once()               # no contesteu abans de tenir una primera foto
    live.start_background_refresh()   # a partir d'aquí es manté sola per sempre

    seen: set[str] = set()
    while True:
        try:
            answers = predict(pull(seen), models, live, model_version)
            if answers:
                push(answers)
        except Exception as e:  # seguiu corrent: els compradors arriben mentre dormiu
            print(f"error: {e}")
            time.sleep(2)
        time.sleep(0.3)


if __name__ == "__main__":
    main()
