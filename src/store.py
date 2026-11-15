"""Everything loaded once from models/<tag>/ and kept in RAM.

    catalog.pkl / popularity.pkl / item_item.pkl / als.pkl / content.pkl
    users.pkl / bandit.pkl / ranker.txt + ranker_meta.pkl / meta.json

The three recommender models (ALS factors, item-item matrices, TF-IDF/SVD
embeddings) live here next to the container that owns them, because they
share one lifecycle: built together for a cut, saved together, loaded
together, read-only while serving.
"""
from __future__ import annotations

import json
import pickle
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import sparse

from src.common import parse_ts

# =====================================================================
# ALS: implicit matrix factorisation (Phase 2)
# =====================================================================
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse

EVENT_WEIGHT = {"view": 1.0, "add_to_cart": 3.0, "purchase": 6.0}


@dataclass
class AlsModel:
    user_ids: list[str]
    item_ids: list[str]
    user_factors: np.ndarray
    item_factors: np.ndarray
    user_index: dict[str, int]
    item_index: dict[str, int]
    factors: int = 48
    alpha: float = 15.0

    @classmethod
    def build(
        cls,
        user_item: pd.DataFrame,
        item_ids: list[str],
        factors: int = 48,
        iterations: int = 15,
        regularization: float = 0.05,
        alpha: float = 15.0,
        seed: int = 0,
        verbose: bool = True,
    ) -> "AlsModel":
        from implicit.als import AlternatingLeastSquares

        item_index = {p: i for i, p in enumerate(item_ids)}
        rows = user_item.copy()
        rows["product_id"] = rows["product_id"].astype(str)
        rows = rows[rows.product_id.isin(item_index)]
        user_ids = sorted(rows.user_id.astype(str).unique().tolist())
        user_index = {u: i for i, u in enumerate(user_ids)}
        if not user_ids:
            return cls(user_ids=[], item_ids=item_ids,
                       user_factors=np.zeros((0, factors), dtype=np.float32),
                       item_factors=np.zeros((len(item_ids), factors), dtype=np.float32),
                       user_index={}, item_index=item_index, factors=factors, alpha=alpha)
        weight = (
            rows.get("views", 0) * EVENT_WEIGHT["view"]
            + rows.get("carts", 0) * EVENT_WEIGHT["add_to_cart"]
            + rows.get("purchases", 0) * EVENT_WEIGHT["purchase"]
        ).astype(np.float32)
        keep = weight > 0
        r = rows.user_id.map(user_index)[keep].to_numpy(np.int32)
        c = rows.product_id.map(item_index)[keep].to_numpy(np.int32)
        v = weight[keep].to_numpy(np.float32)
        Cui = sparse.csr_matrix((v, (r, c)), shape=(len(user_ids), len(item_ids)))
        model = AlternatingLeastSquares(
            factors=factors,
            regularization=regularization,
            iterations=iterations,
            random_state=seed,
            alpha=alpha,
            use_gpu=False,
            calculate_training_loss=False,
        )
        model.fit(Cui, show_progress=verbose)
        return cls(
            user_ids=user_ids,
            item_ids=item_ids,
            user_factors=np.asarray(model.user_factors, dtype=np.float32),
            item_factors=np.asarray(model.item_factors, dtype=np.float32),
            user_index=user_index,
            item_index=item_index,
            factors=factors,
            alpha=alpha,
        )

    def user_vector(self, user_id: str) -> np.ndarray | None:
        i = self.user_index.get(user_id)
        if i is None:
            return None
        return self.user_factors[i]

    def vector_from_items(self, seeds: Iterable[tuple[str, float]]) -> np.ndarray | None:
        rows, weights = [], []
        for pid, w in seeds:
            j = self.item_index.get(pid)
            if j is not None and w > 0:
                rows.append(j)
                weights.append(float(w))
        if not rows:
            return None
        vec = self.item_factors[rows].T @ np.asarray(weights, dtype=np.float32)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else None

    def scores(self, user_vector: np.ndarray, top: int = 120) -> dict[str, float]:
        if user_vector is None or self.item_factors.size == 0:
            return {}
        raw = self.item_factors @ user_vector.astype(np.float32)
        k = min(top, len(raw))
        idx = np.argpartition(-raw, k - 1)[:k]
        idx = idx[np.argsort(-raw[idx])]
        return {self.item_ids[i]: float(raw[i]) for i in idx}

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=5)

    @classmethod
    def load(cls, path: Path) -> "AlsModel":
        with open(path, "rb") as f:
            return pickle.load(f)

    def memory_mb(self) -> float:
        return (self.user_factors.nbytes + self.item_factors.nbytes) / 1e6


# =====================================================================
# Item-item: co-view / co-cart / co-purchase (Phase 1)
# =====================================================================
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse

CHANNELS = ("view", "add_to_cart", "purchase")
DEFAULT_TOP_K = 64


@dataclass
class ItemItemModel:
    item_ids: list[str]
    index: dict[str, int]
    matrices: dict[str, sparse.csr_matrix]

    @classmethod
    def build(
        cls,
        cooc: dict[str, pd.DataFrame],
        item_ids: list[str],
        top_k: int = DEFAULT_TOP_K,
    ) -> "ItemItemModel":
        index = {p: i for i, p in enumerate(item_ids)}
        matrices: dict[str, sparse.csr_matrix] = {}
        for channel in CHANNELS:
            df = cooc.get(channel)
            if df is None or len(df) == 0:
                matrices[channel] = sparse.csr_matrix((len(item_ids), len(item_ids)), dtype=np.float32)
                continue
            p = df["p"].astype(str).map(index)
            q = df["q"].astype(str).map(index)
            n = pd.to_numeric(df["n"], errors="coerce").fillna(0.0)
            keep = p.notna() & q.notna() & (n > 0)
            mat = sparse.coo_matrix(
                (n[keep].to_numpy(np.float32), (p[keep].astype(int), q[keep].astype(int))),
                shape=(len(item_ids), len(item_ids)),
            ).tocsr()
            mat = _row_normalise(mat)
            mat = _prune_rows(mat, top_k)
            matrices[channel] = mat.astype(np.float32)
        return cls(item_ids=item_ids, index=index, matrices=matrices)

    def scores(
        self,
        seeds: Iterable[tuple[str, float]],
        channel: str,
        top: int = 80,
    ) -> dict[str, float]:
        mat = self.matrices.get(channel)
        if mat is None:
            return {}
        rows: list[int] = []
        weights: list[float] = []
        for pid, w in seeds:
            i = self.index.get(pid)
            if i is not None and w > 0:
                rows.append(i)
                weights.append(float(w))
        if not rows:
            return {}
        sub = mat[rows]
        scores = np.asarray(sub.multiply(np.asarray(weights, dtype=np.float32)[:, None]).sum(axis=0)).ravel()
        if not np.any(scores):
            return {}
        k = min(top, int(np.count_nonzero(scores)))
        if k <= 0:
            return {}
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[scores[idx] > 0]
        idx = idx[np.argsort(-scores[idx])]
        return {self.item_ids[i]: float(scores[i]) for i in idx}

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump({
                "item_ids": self.item_ids,
                "index": self.index,
                "matrices": {k: v for k, v in self.matrices.items()},
            }, f, protocol=5)

    @classmethod
    def load(cls, path: Path) -> "ItemItemModel":
        with open(path, "rb") as f:
            data = pickle.load(f)
        return cls(item_ids=data["item_ids"], index=data["index"], matrices=data["matrices"])


def _row_normalise(mat: sparse.csr_matrix) -> sparse.csr_matrix:
    sums = np.asarray(mat.sum(axis=1)).ravel()
    inv = np.where(sums > 0, 1.0 / np.maximum(sums, 1e-12), 0.0)
    return sparse.diags(inv.astype(np.float32)) @ mat


def _prune_rows(mat: sparse.csr_matrix, top_k: int) -> sparse.csr_matrix:
    """Keep the `top_k` strongest entries per row (after normalisation)."""
    rows, cols, vals = [], [], []
    indptr = mat.indptr
    data = mat.data
    indices = mat.indices
    for r in range(mat.shape[0]):
        start, end = indptr[r], indptr[r + 1]
        if end <= start:
            continue
        row_vals = data[start:end]
        row_cols = indices[start:end]
        if end - start > top_k:
            top = np.argpartition(-row_vals, top_k)[:top_k]
            row_vals = row_vals[top]
            row_cols = row_cols[top]
        rows.extend([r] * len(row_vals))
        cols.extend(row_cols.tolist())
        vals.extend(row_vals.tolist())
    return sparse.csr_matrix((np.asarray(vals, dtype=np.float32), (rows, cols)), shape=mat.shape)


# =====================================================================
# Content: TF-IDF -> SVD embeddings (Phase 3)
# =====================================================================
import pickle
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

N_COMPONENTS = 80


@dataclass
class ContentModel:
    item_ids: list[str]
    embeddings: np.ndarray
    vectorizer: TfidfVectorizer
    svd: TruncatedSVD
    index: dict[str, int] = field(init=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def __post_init__(self) -> None:
        self.index = {p: i for i, p in enumerate(self.item_ids)}

    @classmethod
    def build(
        cls,
        catalog: pd.DataFrame,
        n_components: int = N_COMPONENTS,
        seed: int = 0,
    ) -> "ContentModel":
        item_ids = catalog.product_id.astype(str).tolist()
        texts = catalog.text.fillna("").astype(str).tolist()
        vectorizer = TfidfVectorizer(
            max_features=30_000,
            ngram_range=(1, 2),
            min_df=2,
            stop_words="english",
            sublinear_tf=True,
        )
        tfidf = vectorizer.fit_transform(texts)
        n_components = int(min(n_components, max(2, min(tfidf.shape) - 1)))
        svd = TruncatedSVD(n_components=n_components, random_state=seed)
        embeddings = svd.fit_transform(tfidf).astype(np.float32)
        embeddings = _normalise(embeddings)
        return cls(item_ids=item_ids, embeddings=embeddings, vectorizer=vectorizer, svd=svd)

    def embed_texts(self, texts: Iterable[str]) -> np.ndarray:
        with self._lock:
            matrix = self.vectorizer.transform(list(texts))
            vecs = self.svd.transform(matrix).astype(np.float32)
        return _normalise(vecs)

    def add_items(self, item_ids: Iterable[str], texts: Iterable[str]) -> list[str]:
        """Embed and register new products; returns the ids actually added."""
        ids = [str(p) for p in item_ids]
        texts = list(texts)
        new = [i for i, p in enumerate(ids) if p not in self.index]
        if not new:
            return []
        vecs = self.embed_texts([texts[i] for i in new])
        with self._lock:
            base = len(self.item_ids)
            self.embeddings = np.vstack([self.embeddings, vecs])
            for k, i in enumerate(new):
                self.index[ids[i]] = base + k
            self.item_ids.extend([ids[i] for i in new])
        return [ids[i] for i in new]

    def vector_for(self, seeds: Iterable[tuple[str, float]]) -> np.ndarray | None:
        rows, weights = [], []
        for pid, w in seeds:
            i = self.index.get(pid)
            if i is not None and w > 0:
                rows.append(i)
                weights.append(float(w))
        if not rows:
            return None
        vec = self.embeddings[rows].T @ np.asarray(weights, dtype=np.float32)
        norm = np.linalg.norm(vec)
        return (vec / norm).astype(np.float32) if norm > 0 else None

    def scores(self, vector: np.ndarray | None, top: int = 100) -> dict[str, float]:
        if vector is None:
            return {}
        raw = self.embeddings @ vector.astype(np.float32)
        k = min(top, len(raw))
        idx = np.argpartition(-raw, k - 1)[:k]
        idx = idx[np.argsort(-raw[idx])]
        return {self.item_ids[i]: float(raw[i]) for i in idx}

    def scores_subset(self, vector: np.ndarray | None, item_ids: Iterable[str]) -> dict[str, float]:
        """Cosine scores only for `item_ids` (cold-start guarantee, O(subset))."""
        if vector is None:
            return {}
        rows: list[int] = []
        ids: list[str] = []
        for pid in item_ids:
            j = self.index.get(str(pid))
            if j is not None:
                rows.append(j)
                ids.append(str(pid))
        if not rows:
            return {}
        raw = self.embeddings[rows] @ vector.astype(np.float32)
        return {pid: float(score) for pid, score in zip(ids, raw)}

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump({
                "item_ids": self.item_ids,
                "embeddings": self.embeddings,
                "vectorizer": self.vectorizer,
                "svd": self.svd,
            }, f, protocol=5)

    @classmethod
    def load(cls, path: Path) -> "ContentModel":
        with open(path, "rb") as f:
            data = pickle.load(f)
        return cls(item_ids=data["item_ids"], embeddings=data["embeddings"],
                   vectorizer=data["vectorizer"], svd=data["svd"])

    def memory_mb(self) -> float:
        return self.embeddings.nbytes / 1e6


def _normalise(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.clip(norms, 1e-9, None)


# ---------------------------------------------------------------------------
# Legacy pickle paths. `als.pkl`, `itemitem.pkl` and `content.pkl` were written
# by `pickle.dump(self)` while the classes lived in `src.als_model`,
# `src.itemitem` and `src.content`. Those three modules are merged into this
# one, so unpickling would fail on the import. Registering the old paths as
# aliases keeps every already-trained bundle loadable, byte for byte, instead of
# making people rebuild artefacts whose output we validated.
# ---------------------------------------------------------------------------
import sys as _sys
import types as _types

for _name, _cls in (
    ("src.als_model", AlsModel),
    ("src.itemitem", ItemItemModel),
    ("src.content", ContentModel),
):
    _mod = _types.ModuleType(_name)
    setattr(_mod, _cls.__name__, _cls)
    _sys.modules.setdefault(_name, _mod)

# =====================================================================
# Models: the container
# =====================================================================
import json
import pickle
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"


@dataclass(frozen=True)
class Item:
    product_id: str
    category: str
    subcategory: str
    brand: str
    price_eur: float
    is_active: bool
    created_ts: float
    text: str
    is_new: bool = False



@dataclass
class BanditState:
    """Beta-Bernoulli posterior over purchase given impression.

    Priors are empirical-Bayes per subcategory (purchase rate with a strength
    of 40 impressions), so a brand-new product starts at its category's rate
    instead of at 0.
    """

    alpha: dict[str, float] = field(default_factory=dict)
    beta: dict[str, float] = field(default_factory=dict)
    cat_alpha: dict[str, float] = field(default_factory=dict)
    cat_beta: dict[str, float] = field(default_factory=dict)
    impressions: dict[str, int] = field(default_factory=dict)
    purchases: dict[str, int] = field(default_factory=dict)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def prior(self, subcategory: str) -> tuple[float, float]:
        return (
            float(self.cat_alpha.get(subcategory, 1.5)),
            float(self.cat_beta.get(subcategory, 38.5)),
        )

    def posterior(self, pid: str, subcategory: str) -> tuple[float, float]:
        with self._lock:
            if pid in self.alpha:
                return self.alpha[pid], self.beta[pid]
        return self.prior(subcategory)

    def mean(self, pid: str, subcategory: str) -> float:
        a, b = self.posterior(pid, subcategory)
        return a / max(a + b, 1e-9)

    def sample(self, pid: str, subcategory: str, rng: np.random.Generator) -> float:
        """Thompson draw: one Beta sample, used only when exploration is on."""
        a, b = self.posterior(pid, subcategory)
        return float(rng.beta(a, b))

    def update(self, pid: str, subcategory: str, purchased: bool) -> None:
        a0, b0 = self.prior(subcategory)
        with self._lock:
            self.alpha.setdefault(pid, a0)
            self.beta.setdefault(pid, b0)
            self.impressions[pid] = self.impressions.get(pid, 0) + 1
            if purchased:
                self.alpha[pid] += 1.0
                self.purchases[pid] = self.purchases.get(pid, 0) + 1
            else:
                self.beta[pid] += 1.0

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "alpha": dict(self.alpha), "beta": dict(self.beta),
                "cat_alpha": dict(self.cat_alpha), "cat_beta": dict(self.cat_beta),
                "impressions": dict(self.impressions), "purchases": dict(self.purchases),
            }


@dataclass
class Models:
    tag: str
    directory: Path
    catalog: dict[str, Item]
    base_ids: list[str]
    popularity: dict[str, float]
    pop_rank: dict[str, int]
    category_rate: dict[str, float]
    item_item: ItemItemModel | None
    als: AlsModel | None
    content: ContentModel | None
    bandit: BanditState
    profiles: dict[str, list[tuple[str, float]]]
    recent_purchases: dict[str, dict[str, float]]
    purchased_items: dict[str, list[str]]
    ranker: Any | None
    ranker_features: list[str]
    ranker_calibration: Any | None
    meta: dict[str, Any]
    price_median: float
    by_category: dict[str, list[str]] = field(default_factory=dict)
    by_subcategory: dict[str, list[str]] = field(default_factory=dict)
    popular_sorted: list[str] = field(default_factory=list)
    max_popularity: float = 1.0
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    # ------------------------------------------------------------------ load --
    @classmethod
    def load(
        cls,
        tag: str = "prod",
        root: Path | None = None,
        require_ranker: bool = False,
        verbose: bool = True,
    ) -> "Models":
        root = root or MODELS_DIR
        directory = root / tag
        if not (directory / "catalog.pkl").exists():
            raise FileNotFoundError(
                f"no artefacts in {directory}; run training/train_model.py first"
            )
        catalog_df = _pickle(directory / "catalog.pkl")
        catalog = _catalog_from_frame(catalog_df)
        base_ids = list(catalog.keys())
        pop_data = _pickle(directory / "popularity.pkl", default={})
        popularity = pop_data.get("score", {})
        pop_rank = pop_data.get("rank", {p: i for i, p in enumerate(base_ids)})
        category_rate = pop_data.get("category_rate", {})
        item_item = ItemItemModel.load(directory / "item_item.pkl") if (directory / "item_item.pkl").exists() else None
        als = AlsModel.load(directory / "als.pkl") if (directory / "als.pkl").exists() else None
        content = ContentModel.load(directory / "content.pkl") if (directory / "content.pkl").exists() else None
        users = _pickle(directory / "users.pkl", default={})
        bandit_data = _pickle(directory / "bandit.pkl", default={})
        bandit = BanditState(
            alpha=bandit_data.get("alpha", {}),
            beta=bandit_data.get("beta", {}),
            cat_alpha=bandit_data.get("cat_alpha", {}),
            cat_beta=bandit_data.get("cat_beta", {}),
            impressions=bandit_data.get("impressions", {}),
            purchases=bandit_data.get("purchases", {}),
        )
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8")) if (directory / "meta.json").exists() else {}
        ranker = None
        ranker_features: list[str] = []
        calibration = None
        ranker_meta = _pickle(directory / "ranker_meta.pkl", default={})
        if (directory / "ranker.txt").exists() and ranker_meta:
            import lightgbm as lgb

            ranker = lgb.Booster(model_file=str(directory / "ranker.txt"))
            ranker_features = ranker_meta.get("features", [])
            calibration = ranker_meta.get("calibration")
        elif require_ranker:
            raise FileNotFoundError(f"ranker requested but {directory / 'ranker.txt'} is missing")
        prices = [it.price_eur for it in catalog.values() if it.price_eur > 0]
        price_median = float(np.median(prices)) if prices else 30.0
        models = cls(
            tag=tag, directory=directory, catalog=catalog, base_ids=base_ids,
            popularity=popularity, pop_rank=pop_rank, category_rate=category_rate,
            item_item=item_item, als=als, content=content, bandit=bandit,
            profiles=users.get("profiles", {}),
            recent_purchases=users.get("recent_purchases", {}),
            purchased_items=users.get("purchased_items", {}),
            ranker=ranker, ranker_features=ranker_features,
            ranker_calibration=calibration, meta=meta,
            price_median=price_median,
            max_popularity=max([float(v) for v in popularity.values()] + [1.0]),
        )
        models._rebuild_indices()
        if verbose:
            print(
                f"[models:{tag}] {len(catalog)} items, "
                f"item-item={'yes' if item_item else 'no'}, "
                f"als={'yes' if als else 'no'} ({als.memory_mb():.1f} MB), "
                f"content={'yes' if content else 'no'} ({content.memory_mb():.1f} MB), "
                f"ranker={'yes' if ranker else 'no'}, "
                f"users={len(models.profiles)}, cut={meta.get('cut', 'all')}"
            )
        return models

    # --------------------------------------------------------------- runtime --
    def ensure_items(self, rows: Iterable[dict[str, Any]]) -> list[str]:
        """Register products discovered in `shop.catalog_updates`.

        New products get a content embedding immediately (same fitted
        vectoriser/SVD) so they can be shown in the same request. Item-item and
        ALS simply have no history for them, which is the honest cold-start
        state, not an error.
        """
        rows = list(rows)
        if not rows:
            return []
        items: list[dict[str, Any]] = []
        with self._lock:
            for row in rows:
                pid = str(row.get("product_id", ""))
                if not pid or pid in self.catalog:
                    continue
                created = pd.to_datetime(row.get("created_at"), utc=True, errors="coerce")
                created_ts = float(created.timestamp()) if not pd.isna(created) else 0.0
                text = " ".join(str(row.get(c, "") or "") for c in ("category", "subcategory", "brand", "description")).strip()
                self.catalog[pid] = Item(
                    product_id=pid,
                    category=str(row.get("category", "") or ""),
                    subcategory=str(row.get("subcategory", "") or ""),
                    brand=str(row.get("brand", "") or ""),
                    price_eur=float(row.get("price_eur", 0) or 0),
                    is_active=bool(row.get("is_active", True)),
                    created_ts=created_ts,
                    text=text,
                    is_new=True,
                )
                items.append({"product_id": pid, "text": text})
            if items and self.content is not None:
                self.content.add_items([i["product_id"] for i in items], [i["text"] for i in items])
            for entry in items:
                self.add_to_indices(entry["product_id"])
                self.add_popularity(entry["product_id"])
        return [i["product_id"] for i in items]

    def add_popularity(self, pid: str) -> None:
        if pid not in self.popularity:
            self.popularity[pid] = 1.0 / (len(self.popularity) + 2)
            self.pop_rank[pid] = len(self.pop_rank)

    def _rebuild_indices(self) -> None:
        by_cat: dict[str, list[str]] = {}
        by_sub: dict[str, list[str]] = {}
        for pid, item in self.catalog.items():
            by_cat.setdefault(item.category, []).append(pid)
            by_sub.setdefault(item.subcategory, []).append(pid)
        self.by_category = by_cat
        self.by_subcategory = by_sub
        self.popular_sorted = sorted(
            self.catalog, key=lambda p: (-self.popularity.get(p, 0.0), p)
        )

    def add_to_indices(self, pid: str) -> None:
        item = self.catalog.get(pid)
        if item is None:
            return
        self.by_category.setdefault(item.category, []).append(pid)
        self.by_subcategory.setdefault(item.subcategory, []).append(pid)
        self.popular_sorted.append(pid)

    def item(self, pid: str) -> Item | None:
        return self.catalog.get(pid)

    def sync_catalog(self, rows: Iterable[dict[str, Any]]) -> list[str]:
        """Apply catalog_updates to already-known products (price, stock, text)
        and register genuinely new ones. Returns the ids that were new."""
        rows = list(rows)
        new_ids: list[str] = []
        for row in rows:
            pid = str(row.get("product_id", ""))
            if not pid:
                continue
            existing = self.catalog.get(pid)
            if existing is None:
                new_ids.extend(self.ensure_items([row]))
                continue
            created = pd.to_datetime(row.get("created_at"), utc=True, errors="coerce")
            updated = Item(
                product_id=pid,
                category=str(row.get("category", existing.category) or existing.category),
                subcategory=str(row.get("subcategory", existing.subcategory) or existing.subcategory),
                brand=str(row.get("brand", existing.brand) or existing.brand),
                price_eur=float(row.get("price_eur", existing.price_eur) or existing.price_eur),
                is_active=bool(row.get("is_active", existing.is_active)),
                created_ts=existing.created_ts if pd.isna(created) else float(created.timestamp()),
                text=existing.text,
                is_new=existing.is_new,
            )
            self.catalog[pid] = updated
        return new_ids

    def price_norm(self, price: float) -> float:
        return float(np.log1p(max(price, 1.0)) / np.log1p(max(self.price_median, 1.0)))

    def user_profile(self, user_id: str) -> list[tuple[str, float]]:
        return self.profiles.get(user_id, [])

    def ranker_predict(self, X: np.ndarray) -> np.ndarray:
        """Calibrated purchase probability for a feature matrix.

        Keeping the booster and the calibrator as plain attributes (instead of
        importing `src.ranker` here) avoids a circular import: training owns the
        `Ranker` wrapper, serving only needs this one call.
        """
        if self.ranker is None or X.size == 0:
            return np.zeros(0, dtype=np.float32)
        raw = self.ranker.predict(X)
        if self.ranker_calibration is not None:
            try:
                raw = self.ranker_calibration.predict(raw)
            except Exception:
                pass
        return np.clip(np.asarray(raw, dtype=np.float64), 1e-5, 0.999).astype(np.float32)

    def purchased_recently(self, user_id: str, ts_now: float, window_h: float = 168.0) -> dict[str, float]:
        with self._lock:
            items = list(self.recent_purchases.get(user_id, {}).items())
        out = {}
        for pid, ts in items:
            age_h = (ts_now - ts) / 3600.0
            if 0 <= age_h <= window_h:
                out[pid] = age_h
        return out

    def ingest_purchases(
        self,
        rows: Iterable[dict[str, Any]],
        window_h: float = 168.0,
        now: float | None = None,
    ) -> int:
        """Merge live (user, product) purchases into the per-user state.

        `team_25.feedback` carries no `user_id`, so `apply_feedback()` can never
        update a shopper's own state. This is the path that keeps
        `purchased_before`, `recent_purchase_age_h` and the short re-purchase
        penalty alive during a round: `shop.events` does have `user_id`.

        `now` is the reference time — the wall clock live, the session timestamp
        when replaying offline — so the age of a purchase is always measured
        against the moment the list is built.

        Only the users in `rows` are pruned; for everyone else the window filter
        in `purchased_recently()` already hides stale entries, which keeps the
        refresh O(purchases) instead of O(all users).

        Returns the number of newly seen (user, product) pairs.
        """
        reference = time.time() if now is None else float(now)
        cutoff = reference - max(float(window_h), 1.0) * 3600.0
        added = 0
        touched: dict[str, dict[str, float]] = {}
        with self._lock:
            for row in rows:
                user_id = str(row.get("user_id", "") or "")
                pid = str(row.get("product_id", "") or "")
                if not user_id or not pid or pid not in self.catalog:
                    continue
                ts = parse_ts(row.get("last_ts"))
                if ts is None:
                    ts = reference
                target = self.recent_purchases.setdefault(user_id, {})
                known = target.get(pid)
                if known is None:
                    added += 1
                if known is None or ts > known:
                    target[pid] = ts
                bought = self.purchased_items.setdefault(user_id, [])
                if pid not in bought:
                    bought.append(pid)
                touched[user_id] = target
            for target in touched.values():
                stale = [pid for pid, ts in target.items() if ts < cutoff]
                for pid in stale:
                    target.pop(pid, None)
        return added

    def memory_mb(self) -> dict[str, float]:
        return {
            "item_item": sum(m.data.nbytes for m in (self.item_item.matrices.values() if self.item_item else []) ) / 1e6,
            "als": self.als.memory_mb() if self.als else 0.0,
            "content": self.content.memory_mb() if self.content else 0.0,
            "catalog": len(self.catalog) * 0.0005,
        }

    # ------------------------------------------------------------- feedback --
    def apply_feedback(self, rows: Iterable[dict[str, Any]], profiles_window_days: float = 30.0) -> int:
        """Online update from team-served feedback only.

        Beta posteriors use every impression; user recent purchases use only
        purchases that our own list produced, so we never reinforce products we
        did not show.

        Note: `team_25.feedback` has no `user_id`, so in production the
        user-level branch below never fires. User state is fed by
        `ingest_purchases()` from `shop.events` (see `LiveState`).
        """
        n = 0
        now = time.time()
        for row in rows:
            if str(row.get("served_by", "team")) != "team":
                continue
            purchased = str(row.get("purchased", "") or "")
            shown = row.get("shown") or []
            user_id = str(row.get("user_id", "") or "")
            for pid in shown:
                item = self.catalog.get(str(pid))
                if item is None:
                    continue
                self.bandit.update(str(pid), item.subcategory, str(pid) == purchased)
                n += 1
            if purchased:
                item = self.catalog.get(purchased)
                if item is not None and user_id:
                    with self._lock:
                        target = self.recent_purchases.setdefault(user_id, {})
                        target[purchased] = now
                        cutoff = now - profiles_window_days * 86400
                        for pid in [p for p, ts in target.items() if ts < cutoff]:
                            target.pop(pid, None)
        return n


def _pickle(path: Path, default: Any = None) -> Any:
    if not path.exists():
        if default is not None:
            return default
        raise FileNotFoundError(f"missing artefact: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def _catalog_from_frame(df: pd.DataFrame) -> dict[str, Item]:
    out: dict[str, Item] = {}
    for row in df.itertuples():
        created = pd.to_datetime(getattr(row, "created_at", None), utc=True, errors="coerce")
        out[str(row.product_id)] = Item(
            product_id=str(row.product_id),
            category=str(getattr(row, "category", "") or ""),
            subcategory=str(getattr(row, "subcategory", "") or ""),
            brand=str(getattr(row, "brand", "") or ""),
            price_eur=float(getattr(row, "price_eur", 0) or 0),
            is_active=bool(getattr(row, "is_active", True)),
            created_ts=float(created.timestamp()) if not pd.isna(created) else 0.0,
            text=str(getattr(row, "text", "") or ""),
        )
    return out

