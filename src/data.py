"""RawTree access: HTTP client, every SQL statement, the Source, and the
readers that turn rows into DataFrames. One file because these four things
change together: a query, its parameters and its parsing.

Semantic parameters (`before`, `since`, `start`, `end`) are translated into
SQL fragments here, so call sites never build SQL. `before` is exclusive.
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import pandas as pd
import requests
from requests.adapters import HTTPAdapter, Retry

from src.common import parse_ts

# =====================================================================
# HTTP client (official starter shape: sql / sql_df / insert)
# =====================================================================
import os
import threading
import time
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter, Retry


def _load_env() -> None:
    here = Path(__file__).resolve().parent.parent
    for folder in (Path("."), Path(".."), here):
        env_file = folder / ".env"
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                if not line.strip() or line.strip().startswith("#"):
                    continue
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())
            return


_load_env()

API = os.environ.get("RAWTREE_API_URL", "https://api.rawtree.com").rstrip("/")
TEAM = os.environ.get("TEAM_ID", "")

# Reads are idempotent: retry transient failures. Inserts are retried only on
# connection errors by the caller, because a retried insert can duplicate a row.
_read_retry = Retry(
    total=3,
    connect=3,
    read=2,
    backoff_factor=0.2,
    status_forcelist=(429, 500, 502, 503, 504),
    allowed_methods=frozenset({"POST"}),
)
_http = requests.Session()
_http.headers["Authorization"] = f"Bearer {os.environ.get('RAWTREE_API_KEY', '')}"
_http.mount("https://", HTTPAdapter(max_retries=_read_retry, pool_maxsize=16))
_http.mount("http://", HTTPAdapter(max_retries=_read_retry, pool_maxsize=16))

Record = dict[str, Any]
"""RawTree returns rows as JSON objects: {"column": value, ...}"""

_stats_lock = threading.Lock()
_stats = {"queries": 0, "query_ms": 0.0, "inserts": 0, "insert_ms": 0.0, "errors": 0}


def stats() -> dict[str, float]:
    """Cheap counters for the telemetry line; never used on the hot path."""
    with _stats_lock:
        return dict(_stats)


def _bump(name: str, value: float = 1.0) -> None:
    with _stats_lock:
        _stats[name] = _stats.get(name, 0.0) + value


def sql(q: str, timeout: int = 600, attempts: int = 4) -> list[Record]:
    """Read query. Returns one dict per row.

    Reads are idempotent, so a dropped chunked response ("Response ended
    prematurely"), a timeout or a 5xx is retried with backoff. The adapter's
    Retry does not cover failures that happen while the body is being read,
    which is exactly what the API does on large aggregations.
    """
    started = time.perf_counter()
    last: Exception | None = None
    try:
        for attempt in range(1, max(1, attempts) + 1):
            try:
                r = _http.post(f"{API}/v1/query", json={"sql": q}, timeout=timeout)
                r.raise_for_status()
                data = r.json()["data"]
                _bump("queries")
                return data
            except (requests.exceptions.RequestException, ValueError, KeyError) as e:
                last = e
                _bump("errors")
                if attempt >= attempts:
                    break
                wait = min(0.5 * (2 ** (attempt - 1)), 8.0)
                print(f"[rawtree] query failed ({type(e).__name__}), retry {attempt}/{attempts - 1} in {wait:.1f}s", flush=True)
                time.sleep(wait)
        raise last if last is not None else RuntimeError("query failed")
    finally:
        _bump("query_ms", (time.perf_counter() - started) * 1000.0)


def sql_df(q: str, timeout: int = 600):
    """Same as sql(), as a pandas DataFrame (import deferred)."""
    import pandas as pd

    return pd.DataFrame(sql(q, timeout=timeout))


def insert(table: str, rows: list[Record], chunk_size: int = 500) -> None:
    """Batched insert into a team_NN table.

    The serving loop may hand us dozens of answers at once; chunking keeps the
    request below RawTree's body limits while still using few HTTP calls.
    """
    if not rows:
        return
    started = time.perf_counter()
    try:
        for start in range(0, len(rows), max(1, chunk_size)):
            chunk = rows[start : start + chunk_size]
            r = _http.post(
                f"{API}/v1/tables/{table}",
                params={"database": TEAM},
                json=chunk,
                timeout=10,
            )
            r.raise_for_status()
        _bump("inserts")
    except Exception:
        _bump("errors")
        raise
    finally:
        _bump("insert_ms", (time.perf_counter() - started) * 1000.0)


# =====================================================================
# SQL: every statement the system sends
# =====================================================================
QUERIES: dict[str, str] = {
    # ---------------------------------------------------------------- live ----
    "requests_pull": """
        SELECT request_id, ts, user_id, is_new_user,
               context.device::String AS device,
               context.page::String   AS page,
               context.region::String AS region,
               recent_views
        FROM shop.requests
        ORDER BY ts DESC
        LIMIT {limit}""",
    "unavailable": """
        SELECT product_id::String AS product_id
        FROM shop.catalog_updates
        WHERE change::String IN ('out_of_stock', 'back_in_stock')
        GROUP BY product_id
        HAVING argMax(change::String, ts::DateTime64(3)) = 'out_of_stock'""",
    "catalog_current": """
        SELECT product_id,
               argMax(category, v)    AS category,
               argMax(subcategory, v) AS subcategory,
               argMax(brand, v)       AS brand,
               argMax(price_eur, v)   AS price_eur,
               argMax(description, v) AS description,
               argMax(created_at, v)  AS created_at,
               argMax(is_active, v)   AS is_active
        FROM (
            SELECT product_id::String AS product_id, category::String AS category,
                   subcategory::String AS subcategory, brand::String AS brand,
                   price_eur::Float64 AS price_eur, description::String AS description,
                   created_at::DateTime64(3) AS created_at, is_active::Bool AS is_active,
                   '' AS v
            FROM shop.catalog
            UNION ALL
            SELECT product_id::String, category::String, subcategory::String, brand::String,
                   price_eur::Float64, description::String, created_at::DateTime64(3),
                   is_active::Bool, toString(ts)
            FROM shop.catalog_updates
        )
        GROUP BY product_id""",
    "region_trending": """
        SELECT region::String AS region, product_id::String AS product_id,
               countIf(event_type = 'purchase') AS purchases,
               countIf(event_type = 'view')     AS views
        FROM shop.events
        WHERE ts > now() - INTERVAL 1 HOUR
        GROUP BY region, product_id
        ORDER BY purchases DESC, views DESC
        LIMIT {limit} BY region""",
    "product_trending": """
        SELECT product_id::String AS product_id,
               countIf(ts > now() - INTERVAL 15 MINUTE) AS views_15m,
               countIf(ts <= now() - INTERVAL 15 MINUTE) / 12 AS avg_views_15m,
               countIf(ts > now() - INTERVAL 15 MINUTE)
                   / greatest(countIf(ts <= now() - INTERVAL 15 MINUTE) / 12, 1) AS lift
        FROM shop.events
        WHERE event_type = 'view' AND ts > now() - INTERVAL 195 MINUTE
        GROUP BY product_id
        ORDER BY lift DESC
        LIMIT {limit}""",
    "subcategory_trending": """
        SELECT c.subcategory::String AS subcategory,
               countIf(e.ts > now() - INTERVAL 15 MINUTE) AS views_15m,
               countIf(e.ts > now() - INTERVAL 15 MINUTE)
                   / greatest(countIf(e.ts <= now() - INTERVAL 15 MINUTE) / 12, 1) AS lift
        FROM shop.events AS e
        INNER JOIN shop.catalog AS c ON c.product_id::String = e.product_id::String
        WHERE e.event_type = 'view' AND e.ts > now() - INTERVAL 195 MINUTE
        GROUP BY subcategory
        ORDER BY lift DESC
        LIMIT {limit}""",
    "new_products": """
        SELECT product_id::String AS product_id, category::String AS category,
               subcategory::String AS subcategory, brand::String AS brand,
               price_eur::Float64 AS price_eur, description::String AS description,
               created_at::DateTime64(3) AS created_at, ts::DateTime64(3) AS ts
        FROM shop.catalog_updates
        WHERE change::String = 'new_product'""",
    # Live user-level purchase feed. `team_25.feedback` has no user_id, so this
    # is the only way the serving loop can learn that a shopper bought something
    # *during* the round. Aggregated per (user, product) so the payload stays
    # small; `hours` is an integer literal validated by the caller.
    "live_purchases": """
        SELECT user_id::String AS user_id, product_id::String AS product_id,
               max(ts::DateTime64(3)) AS last_ts, count() AS n
        FROM shop.events
        WHERE event_type = 'purchase'
          AND ts > now() - INTERVAL {hours} HOUR
        GROUP BY user_id, product_id""",
    "feedback_since": """
        SELECT request_id, ts, shown, served_by, fallback_reason, latency_ms,
               clicked, purchased, revenue_eur, model_version
        FROM {team}.feedback
        WHERE ts > toDateTime64('{since}', 9) AND request_id != '__init__'
        ORDER BY ts ASC
        LIMIT {limit}""",
    # The platform's clock, not ours. Request timestamps are stamped on the
    # platform side, so an unsynchronised local clock silently shifts every
    # measured age (this machine runs ~1 s behind, see docs/DECISIONS.md §14).
    "clock_now": """
        SELECT toUnixTimestamp64Milli(now64(3)) AS ms""",

    # ------------------------------------------------------------- history ----
    "history_max_ts": """
        SELECT toDate(max(ts::DateTime64(3))) AS d FROM shop.history_sessions""",
    "history_best_sellers": """
        SELECT product_id::String AS product_id, count() AS n
        FROM shop.history_events
        WHERE event_type = 'purchase'{cut}
        GROUP BY product_id
        ORDER BY n DESC""",
    "history_item_counts": """
        SELECT product_id::String AS product_id,
               countIf(event_type = 'view')       AS views,
               countIf(event_type = 'add_to_cart') AS carts,
               countIf(event_type = 'purchase')   AS purchases,
               argMax(price_eur::Float64, ts::DateTime64(3)) AS last_price,
               max(ts::DateTime64(3))             AS last_ts
        FROM shop.history_events
        WHERE 1 = 1{cut}
        GROUP BY product_id""",
    "history_cooccurrence": """
        SELECT a.product_id::String AS p, b.product_id::String AS q, count() AS n
        FROM (SELECT session_id::String AS session_id, product_id::String AS product_id
              FROM shop.history_events
              WHERE event_type = '{event_type}'{cut}) AS a
        INNER JOIN (SELECT session_id::String AS session_id, product_id::String AS product_id
                    FROM shop.history_events
                    WHERE event_type = '{event_type}'{cut}) AS b
          ON a.session_id = b.session_id
        WHERE a.product_id != b.product_id
        GROUP BY p, q
        ORDER BY n DESC
        LIMIT {top_k} BY p""",
    "history_user_item": """
        SELECT user_id::String AS user_id, product_id::String AS product_id,
               sumIf(1.0, event_type = 'view')        AS views,
               sumIf(1.0, event_type = 'add_to_cart') AS carts,
               sumIf(1.0, event_type = 'purchase')    AS purchases,
               max(ts::DateTime64(3))                 AS last_ts
        FROM shop.history_events
        WHERE user_id != ''
          AND ts >= toDateTime64('{start}', 3)
          AND ts <  toDateTime64('{end}', 3)
        GROUP BY user_id, product_id
        ORDER BY (views + 3 * carts + 6 * purchases) DESC
        LIMIT {top_k} BY user_id""",
    "history_recent_purchases": """
        SELECT user_id::String AS user_id, product_id::String AS product_id,
               max(ts::DateTime64(3)) AS last_ts
        FROM shop.history_events
        WHERE event_type = 'purchase'
          AND ts >= toDateTime64('{start}', 3)
          AND ts <  toDateTime64('{end}', 3)
        GROUP BY user_id, product_id""",
    # `shown` is a Dynamic column holding a nullable array; RawTree/ClickHouse
    # rejects `arrayJoin(shown)` with "Nullable type inside Variant" (Code 36).
    # Casting to `Array(String)` first materialises a concrete array and works.
    "history_impression_stats": """
        SELECT pid AS product_id,
               count() AS impressions,
               countIf(purchased = pid) AS purchases
        FROM (
            SELECT purchased::String AS purchased,
                   arrayJoin(shown::Array(String)) AS pid
            FROM shop.history_sessions
            WHERE ts >= toDateTime64('{start}', 3)
              AND ts <  toDateTime64('{end}', 3)
        )
        WHERE pid != ''
        GROUP BY product_id""",
    "history_session_stats": """
        SELECT count() AS sessions,
               countIf(purchased::String != '') AS purchases,
               count(DISTINCT user_id::String) AS users
        FROM shop.history_sessions
        WHERE ts >= toDateTime64('{start}', 3)
          AND ts <  toDateTime64('{end}', 3)""",
    "catalog_asof": """
        SELECT product_id,
               argMax(category, v)    AS category,
               argMax(subcategory, v) AS subcategory,
               argMax(brand, v)       AS brand,
               argMax(price_eur, v)   AS price_eur,
               argMax(description, v) AS description,
               argMax(created_at, v)  AS created_at,
               argMax(is_active, v)   AS is_active
        FROM (
            SELECT product_id::String AS product_id, category::String AS category,
                   subcategory::String AS subcategory, brand::String AS brand,
                   price_eur::Float64 AS price_eur, description::String AS description,
                   created_at::DateTime64(3) AS created_at, is_active::Bool AS is_active,
                   '' AS v
            FROM shop.catalog
            WHERE created_at::DateTime64(3) < toDateTime64('{cut}', 3)
            UNION ALL
            SELECT product_id::String, category::String, subcategory::String, brand::String,
                   price_eur::Float64, description::String, created_at::DateTime64(3),
                   is_active::Bool, toString(ts)
            FROM shop.catalog_updates
            WHERE ts::DateTime64(3) < toDateTime64('{cut}', 3)
        )
        GROUP BY product_id""",
    "stock_asof": """
        SELECT product_id::String AS product_id
        FROM shop.catalog_updates
        WHERE change::String IN ('out_of_stock', 'back_in_stock')
          AND ts::DateTime64(3) < toDateTime64('{cut}', 3)
        GROUP BY product_id
        HAVING argMax(change::String, ts::DateTime64(3)) = 'out_of_stock'""",

    # ----------------------------------------------------------- evaluation ----
    # Same shape as the live feed but bounded by the evaluation window, so the
    # offline replay can ingest each purchase only after its own timestamp.
    "history_purchases_window": """
        SELECT user_id::String AS user_id, product_id::String AS product_id,
               min(ts::DateTime64(3)) AS first_ts,
               max(ts::DateTime64(3)) AS last_ts,
               count() AS n
        FROM shop.history_events
        WHERE event_type = 'purchase'
          AND ts >= toDateTime64('{start}', 3)
          AND ts <  toDateTime64('{end}', 3)
        GROUP BY user_id, product_id""",
    "history_sessions_test": """
        SELECT request_id, ts, user_id,
               context.device::String AS device,
               context.page::String   AS page,
               context.region::String AS region,
               recent_views, shown, propensity, clicked, purchased, revenue_eur
        FROM shop.history_sessions
        WHERE ts >= '{cut}'
        ORDER BY rand()
        LIMIT {limit}""",
    # Same rows, deterministic sample: `sipHash64(request_id + salt)` picks the
    # same sessions for every run with the same (cut, salt, limit). `ORDER BY
    # rand()` cannot do that, so two bundles evaluated on one cut would see two
    # different samples and could not be compared.
    "history_sessions_test_salted": """
        SELECT request_id, ts, user_id,
               context.device::String AS device,
               context.page::String   AS page,
               context.region::String AS region,
               recent_views, shown, propensity, clicked, purchased, revenue_eur
        FROM shop.history_sessions
        WHERE ts >= '{cut}'
        ORDER BY sipHash64(concat(request_id, '{salt}'))
        LIMIT {limit}""",
    "history_sessions_window": """
        SELECT request_id, ts, user_id,
               context.device::String AS device,
               context.page::String   AS page,
               context.region::String AS region,
               recent_views, shown, propensity, clicked, purchased, revenue_eur
        FROM shop.history_sessions
        WHERE ts >= '{start}' AND ts < '{end}'
        ORDER BY ts ASC
        LIMIT {limit}""",
    "history_sessions_window_sample": """
        SELECT request_id, ts, user_id,
               context.device::String AS device,
               context.page::String   AS page,
               context.region::String AS region,
               recent_views, shown, propensity, clicked, purchased, revenue_eur
        FROM shop.history_sessions
        WHERE ts >= '{start}' AND ts < '{end}'
        ORDER BY rand()
        LIMIT {limit}""",
    "history_events_since": """
        SELECT ts, event_type, user_id, session_id, product_id, price_eur, device, region
        FROM shop.history_events
        WHERE ts >= '{start}'{cut}
        ORDER BY ts ASC
        LIMIT {limit}""",

    # ------------------------------------------------------------- training ----
    "history_sessions_all": """
        SELECT request_id, ts, user_id,
               context.device::String AS device,
               context.page::String   AS page,
               context.region::String AS region,
               recent_views, shown, propensity, clicked, purchased, revenue_eur
        FROM shop.history_sessions
        WHERE 1 = 1{cut}
        ORDER BY ts ASC""",
    "history_events_all": """
        SELECT ts, event_type, user_id, session_id, product_id, price_eur, device, region
        FROM shop.history_events
        WHERE 1 = 1{cut}
        ORDER BY ts ASC""",
    "health": "SELECT 1 AS ok",
}


# =====================================================================
# Source: the only data-access boundary
# =====================================================================
from typing import Any, Protocol, runtime_checkable


Record = dict[str, Any]


@runtime_checkable
class Source(Protocol):
    name: str

    def fetch(self, query: str, **params: Any) -> list[Record]: ...

    def fetch_df(self, query: str, **params: Any): ...

    def insert(self, table: str, rows: list[Record]) -> None: ...


class RawTreeSource:
    """Production source. Every read is the SQL in src/queries.py."""

    name = "rawtree"

    def __init__(self, team: str | None = None, client: Any = None) -> None:
        self.team = team or rawtree_client.TEAM
        self._client = client or rawtree_client

    def fetch(self, query: str, **params: Any) -> list[Record]:
        if query not in QUERIES:
            raise KeyError(f"unknown query {query!r}; add it to src/queries.py")
        params = self._sql_params(query, params)
        params.setdefault("team", self.team)
        sql = QUERIES[query].format(**params)
        return self._client.sql(sql)

    def fetch_df(self, query: str, **params: Any):
        import pandas as pd

        return pd.DataFrame(self.fetch(query, **params))

    @staticmethod
    def _sql_params(query: str, params: dict[str, Any]) -> dict[str, Any]:
        """Translate semantic params (before / since) into SQL fragments.

        Keeping this translation here means call sites stay readable: they pass
        semantic params (`before`, `since`) and never build SQL fragments.
        """
        out = dict(params)
        before = out.pop("before", None)
        if before is not None:
            if query in ("catalog_asof", "stock_asof"):
                out["cut"] = str(before)
            elif query in ("history_sessions_test", "history_sessions_test_salted"):
                out["cut"] = str(before)
            else:
                out.setdefault("cut", f" AND ts < '{before}'")
        if query == "feedback_since":
            since = out.get("since")
            if hasattr(since, "isoformat"):
                out["since"] = since.isoformat()
        fragment_queries = {
            "history_best_sellers", "history_item_counts", "history_cooccurrence",
            "history_user_item", "history_user_sessions", "history_sessions_all",
            "history_events_all", "history_events_since",
        }
        if query in fragment_queries:
            out.setdefault("cut", "")
        return out

    def insert(self, table: str, rows: list[Record]) -> None:
        self._client.insert(table, rows)


def make_source(kind: str = "rawtree", **kwargs: Any) -> Source:
    if kind == "rawtree":
        return RawTreeSource(**kwargs)
    raise ValueError(f"unknown source kind: {kind!r}; the only source is 'rawtree'")


# =====================================================================
# Readers: rows -> DataFrames
# =====================================================================
from typing import Any

import pandas as pd



def catalog(source: Source, before: str | None = None) -> pd.DataFrame:
    df = source.fetch_df("catalog_asof" if before else "catalog_current", **({"before": before} if before else {}))
    if df.empty:
        return df
    df = df.copy()
    df["price_eur"] = pd.to_numeric(df.price_eur, errors="coerce").fillna(0.0)
    df["is_active"] = df.is_active.fillna(True).astype(bool)
    df["product_id"] = df.product_id.astype(str)
    for col in ("category", "subcategory", "brand", "description"):
        df[col] = df.get(col, "").fillna("").astype(str)
    df["text"] = (df.category + " " + df.subcategory + " " + df.brand + " " + df.description).str.strip()
    return df


def events(source: Source, before: str | None = None) -> pd.DataFrame:
    df = source.fetch_df("history_events_all", before=before) if before else source.fetch_df("history_events_all")
    if df.empty:
        return df
    df = df.copy()
    df["ts"] = pd.to_datetime(df.ts, utc=True, errors="coerce")
    df["price_eur"] = pd.to_numeric(df.price_eur, errors="coerce").fillna(0.0)
    return df.dropna(subset=["ts"])


def best_seller_ids(source: Source, before: str | None = None) -> list[str]:
    """Official ordering: purchases desc, then product_id desc (starter parity)."""
    df = source.fetch_df("history_best_sellers", **({"before": before} if before else {}))
    if df.empty:
        return []
    df = df.copy()
    df["n"] = pd.to_numeric(df.n, errors="coerce").fillna(0).astype(int)
    df = df.sort_values(["n", "product_id"], ascending=[False, False])
    return df.product_id.astype(str).tolist()


def item_counts(source: Source, before: str | None = None) -> pd.DataFrame:
    df = source.fetch_df("history_item_counts", **({"before": before} if before else {}))
    if df.empty:
        return df
    for col in ("views", "carts", "purchases"):
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    return df


def cooccurrence(source: Source, event_type: str, before: str | None = None, top_k: int = 20) -> pd.DataFrame:
    return source.fetch_df(
        "history_cooccurrence", event_type=event_type, top_k=top_k,
        **({"before": before} if before else {}),
    )


def user_item(
    source: Source,
    start: str | None = None,
    end: str | None = None,
    before: str | None = None,
    top_k: int = 15,
) -> pd.DataFrame:
    """Weighted user x item aggregation, top-k per user.

    Real RawTree has 8.8M distinct user-product pairs over 60 days; fetching
    all of them is wasteful and unnecessary. The SQL keeps the strongest 15
    products per user in the chosen window, which is enough for ALS and for a
    long-term profile while keeping the payload small.
    """
    if start is None or end is None:
        if before is None:
            raise ValueError("user_item needs start/end or before")
        start, end = window_bounds(before, days=30)
    df = source.fetch_df("history_user_item", start=start, end=end, top_k=top_k)
    if df.empty:
        return df
    for col in ("views", "carts", "purchases"):
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    return df


def recent_purchases(source: Source, start: str, end: str) -> pd.DataFrame:
    df = source.fetch_df("history_recent_purchases", start=start, end=end)
    if df.empty:
        return df
    df["last_ts"] = pd.to_datetime(df.last_ts, utc=True, errors="coerce")
    return df.dropna(subset=["last_ts"])


def live_purchases(source: Source, hours: int = 3) -> pd.DataFrame:
    """Live (user, product) purchases from the last `hours`, as a DataFrame."""
    df = source.fetch_df("live_purchases", hours=int(hours))
    if df.empty:
        return df
    df = df.copy()
    df["last_ts"] = pd.to_datetime(df.last_ts, utc=True, errors="coerce")
    return df.dropna(subset=["last_ts"])


def replay_purchases(source: Source, start: str, end: str) -> pd.DataFrame:
    """Purchases inside the evaluation window, ordered by first sighting.

    `first_ts` is when the pair became visible; the offline replay ingests each
    row only after a session's own timestamp, so nothing leaks forward.
    """
    df = source.fetch_df("history_purchases_window", start=start, end=end)
    if df.empty:
        return df
    df = df.copy()
    for col in ("first_ts", "last_ts"):
        df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    return df.dropna(subset=["first_ts"]).sort_values("first_ts").reset_index(drop=True)


def impression_stats(source: Source, start: str, end: str) -> pd.DataFrame:
    df = source.fetch_df("history_impression_stats", start=start, end=end)
    if df.empty:
        return df
    for col in ("impressions", "purchases"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    return df


def session_stats(source: Source, start: str, end: str) -> dict[str, int]:
    rows = source.fetch("history_session_stats", start=start, end=end)
    if not rows:
        return {}
    return {k: int(v) for k, v in rows[0].items()}


def eval_sessions(
    source: Source,
    cut: str,
    limit: int = 100_000,
    salt: str | None = None,
) -> pd.DataFrame:
    """Last-`eval_test_days` impressions, with the cold-start proxy flag.

    `salt` switches to the deterministic sampler: the same `(cut, salt, limit)`
    always returns the same sessions, which is what makes two bundles comparable
    on one cut (and a second salt a genuine out-of-sample check).
    """
    if salt is None:
        test = source.fetch_df("history_sessions_test", before=cut, limit=limit)
    else:
        safe = "".join(c for c in str(salt) if c.isalnum() or c in "_-")
        if not safe:
            raise ValueError(f"invalid sampling salt: {salt!r}")
        test = source.fetch_df(
            "history_sessions_test_salted", before=cut, limit=limit, salt=safe
        )
    if test.empty:
        return test
    test = test.copy()
    test["ts"] = pd.to_datetime(test.ts, utc=True, errors="coerce")
    if "is_new_user" not in test:
        # RawTree's history_sessions does not expose is_new_user; it only exists
        # on live requests. Offline, cold users are inferred from the absence
        # of a pre-cut profile (see `add_cold_flag` / the evaluator).
        test["is_new_user"] = False
    for col in ("shown", "propensity", "recent_views"):
        test[col] = test[col].apply(lambda v: v if isinstance(v, list) else [])
    for col in ("clicked", "purchased"):
        test[col] = test[col].fillna("").astype(str)
    test["revenue_eur"] = pd.to_numeric(test.revenue_eur, errors="coerce").fillna(0.0)
    new_ids = set(catalog(source, before=cut).loc[
        lambda d: pd.to_datetime(d.created_at, utc=True, errors="coerce") >= pd.Timestamp(cut, tz="UTC"),
        "product_id",
    ])
    return add_cold_flag(test, new_ids)


def ranker_sessions(
    source: Source,
    start: str,
    end: str,
    limit: int = 40_000,
    sample: bool = True,
) -> pd.DataFrame:
    """Sessions in `[start, end)`, randomly sampled when the window is large.

    A 7-day real window has ~220k sessions; scoring candidates for all of them
    would take hours. The sample is representative and is re-sorted by `ts` so
    the temporal validation split stays honest.
    """
    query = "history_sessions_window_sample" if sample else "history_sessions_window"
    df = source.fetch_df(query, start=start, end=end, limit=limit)
    if df.empty:
        return df
    df = df.copy()
    df["ts"] = pd.to_datetime(df.ts, utc=True, errors="coerce")
    if "is_new_user" not in df:
        df["is_new_user"] = False
    for col in ("shown", "propensity", "recent_views"):
        if col in df:
            df[col] = df[col].apply(lambda v: v if isinstance(v, list) else [])
    for col in ("clicked", "purchased"):
        if col in df:
            df[col] = df[col].fillna("").astype(str)
    if "revenue_eur" in df:
        df["revenue_eur"] = pd.to_numeric(df.revenue_eur, errors="coerce").fillna(0.0)
    return df.sort_values("ts").reset_index(drop=True)


def max_history_date(source: Source) -> str:
    rows = source.fetch("history_max_ts")
    if not rows:
        raise RuntimeError("shop.history_sessions is empty")
    return str(pd.Timestamp(rows[0]["d"]).date())


def cut_from_source(source: Source, days: int = 7) -> str:
    last = pd.Timestamp(max_history_date(source))
    return str((last - pd.Timedelta(days=days)).date())


def cut_minus(cut: str, days: int) -> str:
    return str((pd.Timestamp(cut) - pd.Timedelta(days=days)).date())


def window_bounds(cut: str, days: int = 7) -> tuple[str, str]:
    """[cut-days, cut) as ISO timestamps, for history_sessions_window."""
    end = pd.Timestamp(cut, tz="UTC")
    start = end - pd.Timedelta(days=days)
    return start.isoformat(), end.isoformat()


def row_ts(row: Any) -> float | None:
    return parse_ts(getattr(row, "ts", None))


# =====================================================================
# Cold-start flag (used by loaders and the evaluator)
# =====================================================================
def _as_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def add_cold_flag(df: pd.DataFrame, new_product_ids) -> pd.DataFrame:
    """Mark offline impressions that match the challenge's cold-start rule.

    Live, `cold` is decided by the stream (`is_new_user` or a view of a product
    added via catalog_updates). Offline we use products whose creation date is
    after the training cut as the proxy.
    """
    new_ids = set(new_product_ids)
    out = df.copy()
    has_new = out.recent_views.map(
        lambda views: any((v or {}).get("product_id") in new_ids for v in _as_list(views))
    )
    out["_cold"] = out.is_new_user.fillna(False).astype(bool) | has_new
    return out

