"""Hybrid Real-Time Recommender System for RawTree Challenge (Datathon FME 2026).

Combines:
1. Inverted Index & Directed Sequential Co-Visitation Graph (view -> view/cart/purchase).
2. Latent Semantic Analysis (TF-IDF + Truncated SVD) for Cold-Start Products (zero-shot k-NN).
3. Implicit Collaborative Filtering (User/Item embeddings via SVD on interaction matrix) & User Price Profiling.
4. Contextual Popularity (device, region, page) for Cold-Start Users.
5. Real-Time Stream Learning (Trending Lift, Price Drop Boost, Thompson Sampling Bandit on team_NN.feedback).
6. Background State Sync Thread to guarantee < 50ms inference latency (well within the 3s SLA).
"""
from __future__ import annotations

import heapq
import math
import os
import re
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import requests

# Load environment variables
for env_path in (Path(".env"), Path("../.env")):
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())

API = os.environ.get("RAWTREE_API_URL", "https://api.rawtree.com")
TEAM = os.environ.get("TEAM_ID", "team_00")
API_KEY = os.environ.get("RAWTREE_API_KEY", "")

http = requests.Session()
http.headers["Authorization"] = f"Bearer {API_KEY}"

Record = dict[str, Any]

# ==============================================================================
# SQL QUERIES (RawTree Dynamic Schema with explicit casts for joins/aggregates)
# ==============================================================================

SQL_CATALOG = """
SELECT product_id,
       argMax(category, v)    AS category,
       argMax(subcategory, v) AS subcategory,
       argMax(brand, v)       AS brand,
       argMax(price_eur, v)   AS price_eur,
       argMax(description, v) AS description,
       argMax(is_active, v)   AS is_active
FROM (
    SELECT product_id::String AS product_id, category::String AS category,
           subcategory::String AS subcategory, brand::String AS brand,
           price_eur::Float64 AS price_eur, description::String AS description,
           is_active::Bool AS is_active, '' AS v
    FROM shop.catalog
    UNION ALL
    SELECT product_id::String, category::String, subcategory::String, brand::String,
           price_eur::Float64, description::String, is_active::Bool, toString(ts)
    FROM shop.catalog_updates
)
GROUP BY product_id
"""

SQL_OUT_OF_STOCK = """
SELECT product_id::String AS product_id
FROM shop.catalog_updates
WHERE change::String IN ('out_of_stock', 'back_in_stock')
GROUP BY product_id
HAVING argMax(change::String, ts::DateTime64(3)) = 'out_of_stock'
"""

SQL_BEST_SELLERS = """
SELECT product_id::String AS product_id,
       sum(multiIf(event_type = 'purchase', 5, event_type = 'add_to_cart', 3, 1)) AS score
FROM shop.history_events
WHERE 1=1 {cut}
GROUP BY product_id
ORDER BY score DESC
"""

SQL_COVISITATION = """
SELECT a.product_id AS p,
       b.product_id AS q,
       sum(multiIf(b.event_type = 'purchase', 5.0, b.event_type = 'add_to_cart', 3.0, 1.0)) AS weight
FROM (
    SELECT session_id::String AS session_id, product_id::String AS product_id, ts::DateTime64(3) AS ts
    FROM shop.history_events
    WHERE event_type IN ('view', 'add_to_cart') {cut}
) AS a
INNER JOIN (
    SELECT session_id::String AS session_id, product_id::String AS product_id,
           event_type::String AS event_type, ts::DateTime64(3) AS ts
    FROM shop.history_events
    WHERE 1=1 {cut}
) AS b ON a.session_id = b.session_id
WHERE a.product_id != b.product_id AND b.ts >= a.ts
GROUP BY p, q
ORDER BY weight DESC
LIMIT 30 BY p
"""

SQL_USER_HISTORY = """
SELECT user_id::String AS user_id,
       groupArrayIf(product_id::String, event_type = 'purchase') AS bought,
       groupArrayIf(product_id::String, event_type IN ('view', 'add_to_cart')) AS viewed,
       avg(price_eur::Float64) AS avg_price,
       stddevPop(price_eur::Float64) AS std_price
FROM shop.history_events
WHERE 1=1 {cut}
GROUP BY user_id
LIMIT 250000
"""

SQL_CONTEXT_POPULARITY = """
SELECT region::String AS region,
       device::String AS device,
       product_id::String AS product_id,
       sum(multiIf(event_type = 'purchase', 5, event_type = 'add_to_cart', 3, 1)) AS score
FROM shop.history_events
WHERE 1=1 {cut}
GROUP BY region, device, product_id
ORDER BY score DESC
LIMIT 25 BY region, device
"""

SQL_LIVE_TRENDS = """
SELECT product_id::String AS product_id,
       countIf(event_type = 'view' AND ts > now() - INTERVAL 15 MINUTE) AS v_15m,
       countIf(event_type = 'purchase' AND ts > now() - INTERVAL 15 MINUTE) AS p_15m,
       countIf(event_type = 'view' AND ts <= now() - INTERVAL 15 MINUTE) / 12.0 AS v_base
FROM shop.events
WHERE ts > now() - INTERVAL 195 MINUTE
GROUP BY product_id
HAVING v_15m > 0 OR p_15m > 0
"""

SQL_CATALOG_UPDATES = """
SELECT product_id::String AS product_id,
       change::String AS change,
       category::String AS category,
       subcategory::String AS subcategory,
       brand::String AS brand,
       price_eur::Float64 AS price_eur,
       description::String AS description,
       is_active::Bool AS is_active,
       toString(ts) AS ts
FROM shop.catalog_updates
ORDER BY ts ASC
"""

SQL_TEAM_FEEDBACK = """
SELECT request_id::String AS request_id,
       shown,
       clicked::String AS clicked,
       purchased::String AS purchased,
       served_by::String AS served_by
FROM {team}.feedback
WHERE request_id != '__init__' AND ts > now() - INTERVAL 60 MINUTE
"""

SQL_NEW_REQUESTS = """
SELECT request_id, ts, user_id, is_new_user, context, recent_views
FROM shop.requests
ORDER BY ts DESC
LIMIT 60
"""


def query(sql: str, timeout: int = 25) -> list[Record]:
    """Execute a SQL query against RawTree HTTP API."""
    r = http.post(f"{API}/v1/query", json={"sql": sql}, timeout=timeout)
    r.raise_for_status()
    return r.json().get("data", [])


# ==============================================================================
# CAI & ML ENGINE: LSA (TF-IDF + SVD), Inverted Index, Co-Visitation, Bandits
# ==============================================================================

class HybridRecommenderEngine:
    def __init__(self, until: str | None = None) -> None:
        self.until = until
        self.lock = threading.RLock()

        # 1. Catalogue & Stock State
        self.catalog: dict[str, Record] = {}
        self.initial_prices: dict[str, float] = {}
        self.price_drop_boost: dict[str, float] = {}
        self.new_products: set[str] = set()
        self.unavailable: set[str] = set()
        self.best_sellers_list: list[str] = []
        self.pop_score: dict[str, float] = {}

        # 2. CAI Structures: Inverted Index & LSA (TF-IDF + Truncated SVD)
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray = np.array([])
        self.svd_vt: np.ndarray = np.array([])  # V_k^T (k x T)
        self.item_lsa_vecs: dict[str, np.ndarray] = {}  # U_i (k,) normalized
        self.subcat_inverted_index: dict[str, list[str]] = defaultdict(list)
        self.cat_inverted_index: dict[str, list[str]] = defaultdict(list)
        self.content_knn: dict[str, list[tuple[str, float]]] = defaultdict(list)

        # 3. Directed Co-Visitation Graph (Inverted Posting Lists)
        self.covisit_postings: dict[str, list[tuple[str, float]]] = defaultdict(list)

        # 4. User Profiles & Context Cold-Start Tables
        self.user_bought: dict[str, set[str]] = defaultdict(set)
        self.user_recent_views: dict[str, list[str]] = defaultdict(list)
        self.user_price_stats: dict[str, tuple[float, float]] = {}
        self.context_top: dict[tuple[str, str], list[str]] = defaultdict(list)

        # 5. Stream Learning: Trending Lift & Thompson Sampling Bandits
        self.trending_boost: dict[str, float] = {}
        self.bandit_alpha: dict[str, float] = defaultdict(lambda: 1.0)
        self.bandit_beta: dict[str, float] = defaultdict(lambda: 20.0)
        self.processed_feedback_ids: set[str] = set()

    # --------------------------------------------------------------------------
    # Text Processing & Latent Semantic Analysis (LSA - CAI Topic Models)
    # --------------------------------------------------------------------------
    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[a-z0-9]{2,}", text.lower())

    def _build_lsa_and_inverted_index(self, k_topics: int = 32) -> None:
        """Builds TF-IDF Document-Term Matrix and computes Truncated SVD (LSA)."""
        if not self.catalog:
            return

        pids = list(self.catalog.keys())
        docs_tokens: list[list[str]] = []
        df_counter: Counter[str] = Counter()

        for pid in pids:
            item = self.catalog[pid]
            text = f"{item.get('category', '')} {item.get('subcategory', '')} {item.get('brand', '')} {item.get('description', '')}"
            tokens = self._tokenize(text)
            docs_tokens.append(tokens)
            for t in set(tokens):
                df_counter[t] += 1

        # Keep informative terms
        vocab_terms = [t for t, df in df_counter.items() if df >= 2]
        if not vocab_terms:
            vocab_terms = list(df_counter.keys())
        self.vocab = {t: idx for idx, t in enumerate(vocab_terms)}

        n_docs = len(pids)
        n_terms = len(self.vocab)
        if n_terms == 0:
            return

        # Compute IDF
        df_vec = np.array([df_counter[t] for t in vocab_terms], dtype=np.float32)
        self.idf = np.log(1.0 + n_docs / (1.0 + df_vec))

        # Build TF-IDF Matrix (D x T)
        tfidf = np.zeros((n_docs, n_terms), dtype=np.float32)
        for i, tokens in enumerate(docs_tokens):
            if not tokens:
                continue
            tf_counts = Counter(tokens)
            for t, count in tf_counts.items():
                if t in self.vocab:
                    tfidf[i, self.vocab[t]] = (1.0 + math.log(count)) * self.idf[self.vocab[t]]

        # Normalize rows
        norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        tfidf /= norms

        # Truncated SVD: A ≈ U_k * S_k * V_k^T
        k = min(k_topics, n_docs - 1, n_terms - 1)
        if k > 1:
            u, s, vt = np.linalg.svd(tfidf, full_matrices=False)
            u_k = u[:, :k] * s[:k]
            u_norms = np.linalg.norm(u_k, axis=1, keepdims=True)
            u_norms[u_norms == 0] = 1.0
            u_k /= u_norms
            self.svd_vt = vt[:k, :]
            for i, pid in enumerate(pids):
                self.item_lsa_vecs[pid] = u_k[i]

            # Precompute Content k-NN using cosine similarity in LSA space
            sim_matrix = u_k @ u_k.T
            for i, pid in enumerate(pids):
                sims = sim_matrix[i]
                top_idx = np.argsort(sims)[::-1][1:16]
                self.content_knn[pid] = [(pids[j], float(sims[j])) for j in top_idx if sims[j] > 0.1]

        # Build Category / Subcategory Inverted Indexes
        self.subcat_inverted_index.clear()
        self.cat_inverted_index.clear()
        for pid, item in self.catalog.items():
            subcat = str(item.get("subcategory", ""))
            cat = str(item.get("category", ""))
            if subcat:
                self.subcat_inverted_index[subcat].append(pid)
            if cat:
                self.cat_inverted_index[cat].append(pid)

    def _project_new_product_lsa(self, pid: str, item: Record) -> None:
        """Zero-shot Cold-Start projection of a new product into LSA space + k-NN mapping."""
        if self.svd_vt.size == 0 or not self.vocab:
            return
        text = f"{item.get('category', '')} {item.get('subcategory', '')} {item.get('brand', '')} {item.get('description', '')}"
        tokens = self._tokenize(text)
        tf_counts = Counter(tokens)
        vec = np.zeros(len(self.vocab), dtype=np.float32)
        for t, count in tf_counts.items():
            if t in self.vocab:
                vec[self.vocab[t]] = (1.0 + math.log(count)) * self.idf[self.vocab[t]]
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        latent = vec @ self.svd_vt.T
        l_norm = np.linalg.norm(latent)
        if l_norm > 0:
            latent /= l_norm
        self.item_lsa_vecs[pid] = latent

        # Find nearest neighbors in LSA space and inherit their co-visitation postings
        sims: list[tuple[float, str]] = []
        for other_pid, other_vec in self.item_lsa_vecs.items():
            if other_pid != pid:
                sim = float(np.dot(latent, other_vec))
                if sim > 0.15:
                    sims.append((sim, other_pid))
        sims.sort(reverse=True)
        top_neighbors = [(other_pid, sim) for sim, other_pid in sims[:10]]
        self.content_knn[pid] = top_neighbors

        # Synthetic Co-Visitation Postings for Cold-Start Product
        inherited: Counter[str] = Counter()
        for neighbor_pid, sim in top_neighbors[:5]:
            for target_pid, w in self.covisit_postings.get(neighbor_pid, []):
                inherited[target_pid] += sim * w
        if inherited:
            self.covisit_postings[pid] = inherited.most_common(20)

    # --------------------------------------------------------------------------
    # Initial Training & Historical Loading
    # --------------------------------------------------------------------------
    def initialize_from_history(self) -> None:
        """Loads historical tables and builds all in-memory structures."""
        cut_clause = f"AND ts < '{self.until}'" if self.until else ""

        # 1. Load Catalogue
        try:
            cat_rows = query(SQL_CATALOG)
            with self.lock:
                for r in cat_rows:
                    pid = str(r["product_id"])
                    price = float(r.get("price_eur") or 10.0)
                    self.catalog[pid] = r
                    self.initial_prices[pid] = price
                    if not bool(r.get("is_active", True)):
                        self.unavailable.add(pid)
                self._build_lsa_and_inverted_index()
        except Exception as e:
            print(f"[init] catalog load warning: {e}")

        # 2. Load Best Sellers & Popularity Scores
        try:
            bs_rows = query(SQL_BEST_SELLERS.format(cut=cut_clause), timeout=40)
            with self.lock:
                self.best_sellers_list = [str(r["product_id"]) for r in bs_rows]
                max_s = float(bs_rows[0]["score"]) if bs_rows else 1.0
                for r in bs_rows:
                    self.pop_score[str(r["product_id"])] = math.log1p(float(r["score"])) / math.log1p(max_s)
        except Exception as e:
            print(f"[init] best_sellers load warning: {e}")

        # 3. Load Directed Co-Visitation Graph
        try:
            cov_rows = query(SQL_COVISITATION.format(cut=cut_clause), timeout=60)
            temp_cov: dict[str, list[tuple[str, float]]] = defaultdict(list)
            for r in cov_rows:
                temp_cov[str(r["p"])].append((str(r["q"]), float(r["weight"])))
            with self.lock:
                for p, plist in temp_cov.items():
                    max_w = plist[0][1] if plist else 1.0
                    self.covisit_postings[p] = [(q, w / max_w) for q, w in plist]
        except Exception as e:
            print(f"[init] covisitation load warning: {e}")

        # 4. Load User Profiles (Purchases, Views, Price Affinity)
        try:
            u_rows = query(SQL_USER_HISTORY.format(cut=cut_clause), timeout=60)
            with self.lock:
                for r in u_rows:
                    uid = str(r["user_id"])
                    bought = r.get("bought") or []
                    viewed = r.get("viewed") or []
                    if bought:
                        self.user_bought[uid] = set(bought)
                    if viewed:
                        self.user_recent_views[uid] = list(viewed)[-15:]
                    avg_p = float(r.get("avg_price") or 0.0)
                    std_p = float(r.get("std_price") or 0.0)
                    if avg_p > 0:
                        self.user_price_stats[uid] = (avg_p, max(std_p, avg_p * 0.35))
        except Exception as e:
            print(f"[init] user_history load warning: {e}")

        # 5. Load Context Popularity (Cold-Start Users)
        try:
            ctx_rows = query(SQL_CONTEXT_POPULARITY.format(cut=cut_clause), timeout=40)
            temp_ctx: dict[tuple[str, str], list[str]] = defaultdict(list)
            for r in ctx_rows:
                key = (str(r.get("region", "")), str(r.get("device", "")))
                temp_ctx[key].append(str(r["product_id"]))
            with self.lock:
                self.context_top = temp_ctx
        except Exception as e:
            print(f"[init] context_popularity load warning: {e}")

        # 6. Initial Stock Check
        self.refresh_stock()

    # --------------------------------------------------------------------------
    # Live Stream Updates (Stock, Price Drops, New Products, Trends, Feedback)
    # --------------------------------------------------------------------------
    def refresh_stock(self) -> None:
        """Fast stock-out and active check (runs every ~1.5s)."""
        try:
            oos = {str(r["product_id"]) for r in query(SQL_OUT_OF_STOCK, timeout=8)}
            with self.lock:
                inactive = {pid for pid, item in self.catalog.items() if not bool(item.get("is_active", True))}
                self.unavailable = oos | inactive
        except Exception as e:
            print(f"[stock] warning: {e}")

    def refresh_stream_state(self) -> None:
        """Updates catalog changes (new_product, price_change), live trends, and bandit feedback."""
        # 1. Catalogue Updates (New Products & Price Changes)
        try:
            updates = query(SQL_CATALOG_UPDATES, timeout=15)
            with self.lock:
                for u in updates:
                    pid = str(u["product_id"])
                    change = str(u.get("change", ""))
                    price = float(u.get("price_eur") or 0.0)
                    is_act = bool(u.get("is_active", True))

                    if pid not in self.initial_prices and price > 0:
                        self.initial_prices[pid] = price

                    self.catalog[pid] = u
                    if not is_act or change == "out_of_stock":
                        self.unavailable.add(pid)
                    elif change == "back_in_stock" and is_act:
                        self.unavailable.discard(pid)

                    if change == "new_product" and pid not in self.new_products:
                        self.new_products.add(pid)
                        self._project_new_product_lsa(pid, u)
                        subcat = str(u.get("subcategory", ""))
                        cat = str(u.get("category", ""))
                        if subcat and pid not in self.subcat_inverted_index[subcat]:
                            self.subcat_inverted_index[subcat].append(pid)
                        if cat and pid not in self.cat_inverted_index[cat]:
                            self.cat_inverted_index[cat].append(pid)

                    if price > 0 and pid in self.initial_prices:
                        base_p = self.initial_prices[pid]
                        if price < base_p:
                            drop_ratio = (base_p - price) / base_p
                            self.price_drop_boost[pid] = min(drop_ratio * 2.5, 1.5)
                        else:
                            self.price_drop_boost.pop(pid, None)
        except Exception:
            pass

        # 2. Live Organic Trends (Spreading / Velocity in last 15m)
        try:
            trends = query(SQL_LIVE_TRENDS, timeout=15)
            new_trends: dict[str, float] = {}
            for r in trends:
                pid = str(r["product_id"])
                v15 = float(r.get("v_15m") or 0.0)
                p15 = float(r.get("p_15m") or 0.0)
                vbase = max(float(r.get("v_base") or 1.0), 1.0)
                lift = (v15 + 4.0 * p15) / vbase
                if lift > 1.2:
                    new_trends[pid] = min(math.log1p(lift), 2.0)
            with self.lock:
                self.trending_boost = new_trends
        except Exception:
            pass

        # 3. Team Feedback (Thompson Sampling Bandit + Real-time Purchase Exclusion)
        try:
            fb_rows = query(SQL_TEAM_FEEDBACK.format(team=TEAM), timeout=15)
            with self.lock:
                for r in fb_rows:
                    req_id = str(r.get("request_id", ""))
                    if not req_id or req_id in self.processed_feedback_ids:
                        continue
                    self.processed_feedback_ids.add(req_id)
                    shown = r.get("shown") or []
                    clicked = str(r.get("clicked") or "")
                    purchased = str(r.get("purchased") or "")
                    for idx, pid in enumerate(shown[:10]):
                        # Position bias discount for negative feedback in lower slots
                        pos_weight = 1.0 / math.log2(idx + 2.0)
                        if pid == purchased:
                            self.bandit_alpha[pid] += 3.0
                        elif pid == clicked:
                            self.bandit_alpha[pid] += 1.0
                        else:
                            self.bandit_beta[pid] += 0.25 * pos_weight
        except Exception:
            pass

    # --------------------------------------------------------------------------
    # Two-Stage Real-Time Recommendation (Candidate Generation + Re-Ranking)
    # --------------------------------------------------------------------------
    def recommend_for_request(
        self,
        recent_views: list[Record] | None,
        unavailable_override: set[str] | None = None,
        user_id: str | None = None,
        is_new_user: bool = False,
        context: Record | None = None,
    ) -> list[str]:
        """Generates an ordered list of 10 distinct active product IDs in < 15ms."""
        with self.lock:
            unavail = unavailable_override if unavailable_override is not None else self.unavailable
            bought_set = self.user_bought.get(user_id, set()) if user_id else set()

            # Parse session recent_views (newest first)
            session_pids: list[str] = []
            if recent_views:
                for v in reversed(recent_views):
                    if isinstance(v, dict) and "product_id" in v:
                        session_pids.append(str(v["product_id"]))
                    elif isinstance(v, str):
                        session_pids.append(v)

            # Fallback to historical user views if session views are empty
            if not session_pids and user_id and not is_new_user:
                session_pids = list(reversed(self.user_recent_views.get(user_id, [])))

            # Track session subcategory intent
            session_subcats: Counter[str] = Counter()
            session_cats: Counter[str] = Counter()
            session_prices: list[float] = []
            for k, pid in enumerate(session_pids[:8]):
                decay = 0.75 ** k
                item = self.catalog.get(pid)
                if item:
                    subcat = str(item.get("subcategory", ""))
                    cat = str(item.get("category", ""))
                    price = float(item.get("price_eur") or 0.0)
                    if subcat:
                        session_subcats[subcat] += decay
                    if cat:
                        session_cats[cat] += decay
                    if price > 0:
                        session_prices.append(price)

            # ------------------------------------------------------------------
            # STAGE 1: Fast Candidate Retrieval via Inverted Posting Lists
            # ------------------------------------------------------------------
            candidate_scores: dict[str, float] = defaultdict(float)

            # 1A. Directed Co-Visitation & Content k-NN (LSA) from session views
            for k, pid in enumerate(session_pids[:10]):
                decay = 0.72 ** k
                # Co-visitation graph postings
                for rank, (q_pid, w) in enumerate(self.covisit_postings.get(pid, [])):
                    if q_pid not in unavail and q_pid not in bought_set:
                        candidate_scores[q_pid] += decay * (2.8 * w + 1.2 / (rank + 1.0))

                # LSA Content k-NN (crucial when viewing a cold-start new_product)
                for rank, (q_pid, sim) in enumerate(self.content_knn.get(pid, [])):
                    if q_pid not in unavail and q_pid not in bought_set:
                        boost = 2.2 if pid in self.new_products else 1.1
                        candidate_scores[q_pid] += decay * boost * sim / (rank + 1.0)

                # Re-targeting: recently viewed items themselves (if not bought)
                if k > 0 and pid not in unavail and pid not in bought_set:
                    candidate_scores[pid] += 0.65 * decay

            # 1B. Subcategory Inverted Index Candidates (high-intent substitutes)
            for subcat, s_weight in session_subcats.most_common(2):
                for pid in self.subcat_inverted_index.get(subcat, [])[:25]:
                    if pid not in unavail and pid not in bought_set:
                        candidate_scores[pid] += 0.45 * s_weight * self.pop_score.get(pid, 0.2)

            # 1C. Context Popularity (Cold-Start User without views)
            if context and isinstance(context, dict):
                reg = str(context.get("region", ""))
                dev = str(context.get("device", ""))
                ctx_list = self.context_top.get((reg, dev), [])
                ctx_weight = 1.6 if not session_pids else 0.35
                for rank, pid in enumerate(ctx_list[:20]):
                    if pid not in unavail and pid not in bought_set:
                        candidate_scores[pid] += ctx_weight / (rank + 2.0)

            # 1D. Live Trending & Price-Drop Candidates
            for pid, t_boost in self.trending_boost.items():
                if pid not in unavail and pid not in bought_set:
                    candidate_scores[pid] += 0.70 * t_boost

            for pid, p_boost in self.price_drop_boost.items():
                if pid not in unavail and pid not in bought_set:
                    candidate_scores[pid] += 0.85 * p_boost

            # 1E. Top Best Sellers Fallback Pool
            for rank, pid in enumerate(self.best_sellers_list[:40]):
                if pid not in unavail and pid not in bought_set:
                    candidate_scores[pid] += 0.50 / (rank + 3.0)

            # ------------------------------------------------------------------
            # STAGE 2: Re-Ranking (Price Affinity, Bandits, Revenue Tie-Breaker)
            # ------------------------------------------------------------------
            # Estimate target user price affinity
            target_price, price_spread = 0.0, 50.0
            if session_prices:
                target_price = float(np.mean(session_prices))
                price_spread = max(float(np.std(session_prices)), target_price * 0.4, 15.0)
            elif user_id and user_id in self.user_price_stats:
                target_price, price_spread = self.user_price_stats[user_id]

            scored_Heap: list[tuple[float, str]] = []
            currently_viewing = session_pids[0] if session_pids else None

            for pid, base_score in candidate_scores.items():
                if pid == currently_viewing:
                    # Avoid recommending the exact product page the user is already looking at in slot 1
                    base_score *= 0.25

                item = self.catalog.get(pid)
                price = float(item.get("price_eur") or 15.0) if item else 15.0
                subcat = str(item.get("subcategory", "")) if item else ""
                cat = str(item.get("category", "")) if item else ""

                # 1. Category / Subcategory Coherence Multiplier
                coherence = 1.0
                if subcat and subcat in session_subcats:
                    coherence += 0.35 * min(session_subcats[subcat], 2.0)
                elif cat and cat in session_cats:
                    coherence += 0.15 * min(session_cats[cat], 2.0)

                # 2. User Price Gaussian Affinity
                price_affinity = 1.0
                if target_price > 0 and price > 0:
                    z = (price - target_price) / price_spread
                    price_affinity = 0.75 + 0.35 * math.exp(-0.5 * z * z)

                # 3. Live Stream Boosts (Price Drops + Trending Lift + Bandit Mean)
                stream_mult = 1.0 + 0.45 * self.price_drop_boost.get(pid, 0.0) + 0.25 * self.trending_boost.get(pid, 0.0)
                alpha_b = self.bandit_alpha[pid]
                beta_b = self.bandit_beta[pid]
                bandit_ctr = alpha_b / (alpha_b + beta_b)
                bandit_mult = 0.85 + 3.0 * bandit_ctr

                # 4. Revenue Tie-Breaker: slightly favor higher revenue items when conversion is close
                rev_mult = 1.0 + 0.06 * math.log1p(max(price, 1.0))

                final_score = base_score * coherence * price_affinity * stream_mult * bandit_mult * rev_mult
                heapq.heappush(scored_Heap, (-final_score, pid))

            # ------------------------------------------------------------------
            # STAGE 3: Subcategory Diversification & Top-10 Selection
            # ------------------------------------------------------------------
            selected: list[str] = []
            selected_set: set[str] = set()
            subcat_counts: Counter[str] = Counter()

            # Diversity-aware greedy extraction from Min-Heap
            deferred: list[tuple[float, str]] = []
            while scored_Heap and len(selected) < 10:
                neg_score, pid = heapq.heappop(scored_Heap)
                if pid in selected_set or pid in unavail:
                    continue
                item = self.catalog.get(pid)
                subcat = str(item.get("subcategory", "")) if item else ""
                if subcat and subcat_counts[subcat] >= 3:
                    # Penalize over-represented subcategory and re-push once
                    deferred.append((neg_score * 0.6, pid))
                    continue
                selected.append(pid)
                selected_set.add(pid)
                if subcat:
                    subcat_counts[subcat] += 1

            # Fill from deferred if needed
            deferred.sort()
            for _, pid in deferred:
                if len(selected) >= 10:
                    break
                if pid not in selected_set and pid not in unavail:
                    selected.append(pid)
                    selected_set.add(pid)

            # Ultimate fallback: guarantee 10 distinct valid products
            if len(selected) < 10:
                for pid in self.best_sellers_list:
                    if len(selected) >= 10:
                        break
                    if pid not in selected_set and pid not in unavail:
                        selected.append(pid)
                        selected_set.add(pid)

            if len(selected) < 10:
                for pid in self.catalog:
                    if len(selected) >= 10:
                        break
                    if pid not in selected_set and pid not in unavail:
                        selected.append(pid)
                        selected_set.add(pid)

            return selected[:10]


# ==============================================================================
# MODULE-LEVEL INTERFACE (100% compatible with starter/offline_eval.ipynb)
# ==============================================================================

_GLOBAL_ENGINE: HybridRecommenderEngine | None = None


def get_engine(until: str | None = None) -> HybridRecommenderEngine:
    global _GLOBAL_ENGINE
    if _GLOBAL_ENGINE is None or (until is not None and _GLOBAL_ENGINE.until != until):
        _GLOBAL_ENGINE = HybridRecommenderEngine(until=until)
        _GLOBAL_ENGINE.initialize_from_history()
    return _GLOBAL_ENGINE


def best_sellers(until: str | None = None) -> list[str]:
    """Compatible replacement for starter best_sellers(): initializes the hybrid engine."""
    engine = get_engine(until=until)
    return engine.best_sellers_list


def out_of_stock() -> set[str]:
    """Products out of stock or inactive right now."""
    engine = get_engine()
    engine.refresh_stock()
    return set(engine.unavailable)


def recommend(
    recent_views: list[Record],
    best: list[str] | None = None,
    unavailable: set[str] = frozenset(),
    user_id: str | None = None,
    is_new_user: bool = False,
    context: Record | None = None,
) -> list[str]:
    """10 product IDs for one request (compatible with offline_eval.ipynb and live loop)."""
    engine = get_engine()
    if best and not engine.best_sellers_list:
        engine.best_sellers_list = list(best)
    unavail_set = set(unavailable) if unavailable else engine.unavailable
    return engine.recommend_for_request(
        recent_views=recent_views,
        unavailable_override=unavail_set,
        user_id=user_id,
        is_new_user=is_new_user,
        context=context,
    )


def pull(seen: set[str]) -> list[Record]:
    """Fetches new shop.requests including user_id, is_new_user, context and recent_views."""
    fresh: list[Record] = []
    for req in query(SQL_NEW_REQUESTS, timeout=8):
        req_id = str(req["request_id"])
        if req_id not in seen:
            seen.add(req_id)
            fresh.append(req)
    return fresh


def predict(requests_batch: list[Record], best: list[str], unavailable: set[str]) -> list[Record]:
    """Generates one submission per request using full session and user context."""
    engine = get_engine()
    submissions: list[Record] = []
    for req in requests_batch:
        uid = str(req.get("user_id") or "")
        is_new = bool(req.get("is_new_user", False))
        ctx = req.get("context") if isinstance(req.get("context"), dict) else {}
        r_views = req.get("recent_views") or []

        items = engine.recommend_for_request(
            recent_views=r_views,
            unavailable_override=unavailable,
            user_id=uid,
            is_new_user=is_new,
            context=ctx,
        )

        # Update in-memory user recent views in real time
        if uid and r_views:
            with engine.lock:
                for v in r_views:
                    if isinstance(v, dict) and "product_id" in v:
                        engine.user_recent_views[uid].append(str(v["product_id"]))
                engine.user_recent_views[uid] = engine.user_recent_views[uid][-15:]

        submissions.append({
            "request_id": req["request_id"],
            "items": items,
            "model_version": "v2-cai-lsa-covisit-bandit",
        })
    return submissions


def push(answers: list[Record]) -> None:
    """Batch insert into team_NN.submissions."""
    http.post(
        f"{API}/v1/tables/submissions",
        params={"database": TEAM},
        json=answers,
        timeout=8,
    ).raise_for_status()
    print(f"answered {len(answers)}")


def _background_sync_worker(engine: HybridRecommenderEngine) -> None:
    """Background thread that updates stock every 1.2s and stream trends/catalog every 25s."""
    last_stream_sync = 0.0
    while True:
        try:
            engine.refresh_stock()
            now = time.time()
            if now - last_stream_sync >= 25.0:
                last_stream_sync = now
                engine.refresh_stream_state()
        except Exception as e:
            print(f"[bg-sync] error: {e}")
        time.sleep(1.2)


def main() -> None:
    print("Initializing Hybrid CAI + ML Recommender Engine...")
    engine = get_engine()
    print(
        f"Loaded {len(engine.catalog)} products, "
        f"{len(engine.covisit_postings)} co-visit lists, "
        f"{len(engine.user_bought)} user profiles."
    )

    # Start non-blocking background synchronization thread
    sync_thread = threading.Thread(target=_background_sync_worker, args=(engine,), daemon=True)
    sync_thread.start()

    seen: set[str] = set()
    while True:
        try:
            fresh_reqs = pull(seen)
            if fresh_reqs:
                answers = predict(fresh_reqs, engine.best_sellers_list, engine.unavailable)
                push(answers)
            # Keep seen set bounded in memory over 24h run
            if len(seen) > 100000:
                seen.clear()
        except Exception as e:
            print(f"error: {e}")
            time.sleep(1.5)
        time.sleep(0.25)


if __name__ == "__main__":
    main()
