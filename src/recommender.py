"""src/recommender.py: Combina l'índex viu d'embeddings i exploració Cold-Start
(de recommender_3.py) amb Candidate Retrieval + Classificador ML (ranker.pkl)."""
from __future__ import annotations

import math
import pickle
import sys
import time
import zlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.features import LiveState
from src.rawtree_client import Record, insert, sql
from training.train_model import extract_candidate_features

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

NEW_REQUESTS = """
SELECT request_id, ts, user_id, is_new_user, context, recent_views
FROM shop.requests
ORDER BY ts DESC
LIMIT 50"""

# Exploració de productes nous (Cold Start)
EXPLORE_RATE = 0.20       # 20% de peticions exploren un producte nou
EXPLORE_SLOT = 2          # posició 3 (0-based)
EXPLORE_MIN_SIM = 0.15    # similitud mínima amb la sessió


class Models:
    """Artefactes de training/ + índex d'embeddings que creix en viu amb sync_new_products()."""

    def __init__(self, suffix: str = "") -> None:
        self.best_sellers: list[str] = self._load(f"best_sellers{suffix}.pkl", default=[])
        self.co_purchases: dict[str, list[tuple[str, float]]] = self._load(f"co_purchases{suffix}.pkl", default={})
        self.view_to_buy: dict[str, list[tuple[str, float]]] = self._load(f"view_to_buy{suffix}.pkl", default={})
        self.user_profiles: dict[str, list[tuple[str, float]]] = self._load(f"user_profiles{suffix}.pkl", default={})
        self.user_bought: dict[str, set[str]] = self._load(f"user_bought{suffix}.pkl", default={})
        self.item_stats: dict[str, dict[str, float]] = self._load(f"item_stats{suffix}.pkl", default={})
        self.catalog_meta: dict[str, dict] = self._load(f"catalog_meta{suffix}.pkl", default={})
        self.ranker = self._load(f"ranker{suffix}.pkl", default=None)

        self.vectorizer = self._load("vectorizer.pkl", default=None)
        self.svd = self._load("svd.pkl", default=None)

        self.best_by_subcat: dict[str, list[str]] = defaultdict(list)
        for pid in self.best_sellers:
            sub = (self.catalog_meta.get(pid) or {}).get("subcategory")
            if sub:
                self.best_by_subcat[sub].append(pid)

        self._index: tuple[np.ndarray | None, list[str], dict[str, int]] = self._load_index()
        self._indexed_new: set[str] = set()

        print(f"[models] {len(self.best_sellers)} best sellers, "
              f"{len(self.co_purchases)} co-purchases, "
              f"{len(self.view_to_buy)} view->buy, "
              f"{len(self._index[1])} embeddings indexats, "
              f"ranker={'si' if self.ranker is not None else 'no'}")

    @staticmethod
    def _load(name: str, default: Any) -> Any:
        path = MODELS_DIR / name
        if not path.exists():
            return default
        with open(path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def _load_index():
        emb, ids_path = MODELS_DIR / "embeddings.npy", MODELS_DIR / "product_ids.pkl"
        if not (emb.exists() and ids_path.exists()):
            return None, [], {}
        matrix = np.load(emb)
        with open(ids_path, "rb") as f:
            ids = pickle.load(f)
        return matrix, ids, {p: i for i, p in enumerate(ids)}

    # ---------- embeddings en viu (de recommender_3.py) ----------

    def embed_text(self, category: str, subcategory: str, brand: str, description: str) -> np.ndarray | None:
        if self.vectorizer is None or self.svd is None:
            return None
        text = " ".join(filter(None, [category, subcategory, brand, description]))
        vec = self.svd.transform(self.vectorizer.transform([text]))[0]
        return vec / max(float(np.linalg.norm(vec)), 1e-9)

    def sync_new_products(self, new_products: dict[str, dict]) -> int:
        matrix, ids, id_to_row = self._index
        fresh = [p for p in new_products if p not in id_to_row and p not in self._indexed_new]
        if not fresh:
            return 0
        vecs, added = [], []
        for pid in fresh:
            r = new_products[pid]
            v = self.embed_text(r.get("category", ""), r.get("subcategory", ""),
                                r.get("brand", ""), r.get("description", ""))
            self._indexed_new.add(pid)
            if v is not None:
                vecs.append(v)
                added.append(pid)
            self.catalog_meta.setdefault(pid, r)
        if not vecs:
            return 0
        dim = len(vecs[0])
        base = matrix if matrix is not None else np.zeros((0, dim))
        new_matrix = np.vstack([base, np.array(vecs)])
        all_ids = ids + added
        self._index = (new_matrix, all_ids, {p: i for i, p in enumerate(all_ids)})
        print(f"[embeddings] {len(added)} productes nous indexats en viu")
        return len(added)

    def is_indexed(self, pid: str) -> bool:
        return pid in self._index[2]

    def emb_sim(self, pid1: str, pid2: str) -> float:
        matrix, _ids, id_to_row = self._index
        if matrix is None or pid1 not in id_to_row or pid2 not in id_to_row:
            return 0.0
        return float(matrix[id_to_row[pid1]] @ matrix[id_to_row[pid2]])

    def neighbors(self, pid: str, top_k: int = 12) -> list[tuple[str, float]]:
        matrix, ids, id_to_row = self._index
        if matrix is None or pid not in id_to_row:
            return []
        sims = matrix @ matrix[id_to_row[pid]]
        k = min(top_k + 1, len(ids))
        best = np.argpartition(-sims, k - 1)[:k]
        out = [(ids[i], float(sims[i])) for i in best if ids[i] != pid]
        out.sort(key=lambda x: -x[1])
        return out[:top_k]

    def similarity_to(self, pid: str, others: list[str]) -> float:
        matrix, _ids, id_to_row = self._index
        if matrix is None or pid not in id_to_row:
            return 0.0
        vals = [float(matrix[id_to_row[pid]] @ matrix[id_to_row[o]])
                for o in others if o != pid and o in id_to_row]
        return float(np.mean(vals)) if vals else 0.0


def place(items: list[str], pick: str, slot: int) -> list[str]:
    items = [p for p in items if p != pick]
    items.insert(min(slot, len(items)), pick)
    return items[:10]


def recommend(req: Record, models: Models, live: LiveState) -> list[str]:
    recent_views = req.get("recent_views") or []
    recent = [v["product_id"] for v in recent_views if "product_id" in v]
    user_id = req.get("user_id")
    context = req.get("context") or {}
    region = context.get("region")

    # Excloem productes sense estoc, vistos ara mateix i JA COMPRATS per l'usuari
    bought_by_user = models.user_bought.get(user_id, set()) if (user_id and not req.get("is_new_user")) else set()
    exclude = set(live.unavailable) | set(recent) | bought_by_user

    catalog = live.catalog if live.catalog else models.catalog_meta
    candidate_pool: Counter[str] = Counter()

    # ETAPA 1: Generació de candidats (~45 productes)
    for k, pid in enumerate(reversed(recent[:6])):
        decay = 0.72 ** k
        # 1a. Cross-correlation direccional (View -> Buy)
        for rank, (q, w) in enumerate(models.view_to_buy.get(pid, [])[:15], start=1):
            if q not in exclude:
                candidate_pool[q] += 3.5 * decay * (1.0 + 0.15 * math.log1p(w)) / rank

        # 1b. Co-ocurrència de sessió
        related = models.co_purchases.get(pid, [])
        for rank, (q, w) in enumerate(related[:15], start=1):
            if q not in exclude:
                candidate_pool[q] += 2.2 * decay * (1.0 + 0.1 * math.log1p(w)) / rank

        # 1c. Veïns d'embeddings (sempre per als últims 2 vistos o si és producte nou / sense historial)
        if k < 2 or not related or pid in live.new_products:
            for q, sim in models.neighbors(pid, top_k=12):
                if q not in exclude and sim > 0.20:
                    # Bonus si el veí trobat és un producte nou (Cold Start!)
                    bonus_new = 1.35 if q in live.new_products else 1.0
                    candidate_pool[q] += 2.2 * decay * sim * bonus_new

        # 1d. Top vendes de la mateixa subcategoria
        sub = (catalog.get(pid) or {}).get("subcategory")
        if sub:
            for rank, q in enumerate(models.best_by_subcat.get(sub, [])[:8], start=1):
                if q not in exclude:
                    candidate_pool[q] += 0.8 * decay / rank

    # 2. Historial de l'usuari (sense recomanar el que ja ha comprat!)
    if not req.get("is_new_user") and user_id in models.user_profiles:
        for rank_u, (p, w) in enumerate(models.user_profiles[user_id][:12], start=1):
            if p not in exclude:
                candidate_pool[p] += 1.4 * math.log1p(w) / rank_u
            for rank_r, (q, _) in enumerate(models.view_to_buy.get(p, [])[:5], start=1):
                if q not in exclude:
                    candidate_pool[q] += 0.9 / (rank_u * rank_r)

    # 3. Tendència de la regió (molt més pes si l'usuari és nou i no té recent_views)
    trend_list = live.region_trending.get(region, [])
    for rank, p in enumerate(trend_list[:15], start=1):
        if p not in exclude:
            candidate_pool[p] += (2.5 if not recent else 0.8) / rank

    if len(candidate_pool) < 35:
        for rank, p in enumerate(models.best_sellers[:45], start=1):
            if p not in exclude and p not in candidate_pool:
                candidate_pool[p] += 0.3 / rank
                if len(candidate_pool) >= 45:
                    break

    candidates = [p for p, _ in candidate_pool.most_common(45)]
    if not candidates:
        return [p for p in models.best_sellers if p not in live.unavailable][:10]

    # ETAPA 2: Ranking amb el model classificador (ranker.pkl) + senyals en viu
    if models.ranker is not None:
        X_cand = np.array([
            extract_candidate_features(
                cand, recent_views, user_id, context, catalog,
                models.co_purchases, models.view_to_buy, models.user_profiles,
                models.user_bought, models.item_stats,
                emb_sim_fn=models.emb_sim
            )
            for cand in candidates
        ], dtype=np.float32)
        ml_probs = models.ranker.predict_proba(X_cand)[:, 1]
    else:
        ml_probs = np.zeros(len(candidates), dtype=np.float32)

    trend_rank_map = {p: r for r, p in enumerate(trend_list, start=1)}
    final_scores: list[tuple[float, str]] = []
    for idx, cand in enumerate(candidates):
        base_h = math.log1p(candidate_pool[cand])
        prob = float(ml_probs[idx])
        live_bonus = 0.0
        if cand in trend_rank_map:
            live_bonus += 0.15 / trend_rank_map[cand]
        if cand in live.discounted:
            live_bonus += 0.08
        if cand in live.feedback_rates:
            live_bonus += 1.5 * live.feedback_rates[cand]

        combined = (0.65 * prob + 0.35 * (base_h / 4.0) if models.ranker is not None else base_h) + live_bonus
        final_scores.append((combined, cand))

    final_scores.sort(reverse=True)

    # Diversificació per subcategoria (màxim 4 de la mateixa subcategoria)
    items: list[str] = []
    sub_counts: Counter[str] = Counter()
    for _, cand in final_scores:
        sub = (catalog.get(cand) or {}).get("subcategory", "")
        if sub and sub_counts[sub] >= 4:
            continue
        items.append(cand)
        exclude.add(cand)
        if sub:
            sub_counts[sub] += 1
        if len(items) == 10:
            break

    if len(items) < 10:
        items += [p for p in models.best_sellers if p not in exclude and p not in items]
        items = items[:10]

    # ETAPA 3: Exploració controlada de productes nous (de recommender_3.py)
    new_ok = [p for p in live.new_products if p not in live.unavailable and p not in recent and models.is_indexed(p)]
    if new_ok and recent and not any(p in live.new_products for p in items):
        if zlib.crc32(str(req.get("request_id")).encode()) % 100 < EXPLORE_RATE * 100:
            best_new = max(new_ok, key=lambda p: models.similarity_to(p, recent))
            if models.similarity_to(best_new, recent) >= EXPLORE_MIN_SIM:
                items = place(items, best_new, EXPLORE_SLOT)

    return items[:10]


def is_valid(items: list[str], unavailable: set[str]) -> bool:
    return len(items) == 10 and len(set(items)) == 10 and not any(p in unavailable for p in items)


def pull(seen: set[str]) -> list[Record]:
    fresh = []
    for req in sql(NEW_REQUESTS, timeout=10):
        if req["request_id"] not in seen:
            seen.add(req["request_id"])
            fresh.append(req)
    return fresh


def predict(batch: list[Record], models: Models, live: LiveState, version: str) -> list[Record]:
    answers = []
    for req in batch:
        items = recommend(req, models, live)
        if not is_valid(items, live.unavailable):
            items = [p for p in models.best_sellers if p not in live.unavailable][:10]
        answers.append({"request_id": req["request_id"], "items": items, "model_version": version})
    return answers


def push(answers: list[Record]) -> None:
    insert("submissions", answers)
    print(f"contestades {len(answers)}")


def main(model_version: str = "v3-hybrid-ranker-coldstart") -> None:
    models = Models()
    live = LiveState()
    live.refresh_once()
    live.start_background_refresh()
    models.sync_new_products(live.new_products)

    seen: set[str] = set()
    while True:
        try:
            models.sync_new_products(live.new_products)
            answers = predict(pull(seen), models, live, model_version)
            if answers:
                push(answers)
        except Exception as e:
            print(f"error: {e}")
            time.sleep(2)
        time.sleep(0.3)


if __name__ == "__main__":
    main()