"""Live state: everything that changes while the stream runs.

One immutable `LiveSnapshot` is published atomically by a background thread;
inference only reads the current reference, so a slow RawTree query can never
delay an answer. The thread refreshes:

* stock every second (the cheapest, most time-critical signal),
* the catalogue every 30 s (prices, new products, availability),
* trends every 15 s (last 15 min vs the previous 3 h, per product/subcategory),
* user purchases every 3 min (`shop.events`), because feedback has no `user_id`.

The purchase feed is not part of the snapshot: it mutates `Models` in place,
under the model lock, so the ranker's `purchased_before` / age features and the
re-purchase penalty see what shoppers bought minutes ago.

Failures keep the previous state: a stale stock list is better than no list,
and the fallback pool is always valid by construction.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.common import AtomicBox, clean_str, measure_clock_offset, now_ts
from src.config import Config
from src.data import Source
from src.store import Models


@dataclass(frozen=True)
class LiveSnapshot:
    available: frozenset[str]
    unavailable: frozenset[str]
    region_trending: dict[str, tuple[str, ...]]
    product_trend: dict[str, float]
    subcategory_trend: dict[str, float]
    new_product_ids: frozenset[str]
    fallback_pool: tuple[str, ...]
    prices: dict[str, float]
    built_at: float
    source: str = "live"

    def is_available(self, pid: str) -> bool:
        return pid in self.available

    def trend_of(self, pid: str) -> float:
        return float(self.product_trend.get(pid, 0.0))

    def region_rank(self, region: str, pid: str) -> float:
        ranked = self.region_trending.get(region, ())
        try:
            return 1.0 / (ranked.index(pid) + 1)
        except ValueError:
            return 0.0


def build_snapshot(
    models: Models,
    unavailable: Iterable[str],
    region_rows: list[dict[str, Any]],
    product_rows: list[dict[str, Any]],
    subcategory_rows: list[dict[str, Any]],
    source: str = "live",
) -> LiveSnapshot:
    unavailable_set = frozenset(str(p) for p in unavailable)
    available = frozenset(
        pid for pid, item in models.catalog.items()
        if item.is_active and pid not in unavailable_set
    )
    regions: dict[str, list[str]] = {}
    for row in region_rows:
        region = clean_str(row.get("region"))
        pid = clean_str(row.get("product_id"))
        if region and pid and pid in available:
            regions.setdefault(region, []).append(pid)
    region_trending = {r: tuple(p) for r, p in regions.items()}

    product_trend = _lift_map(product_rows, "product_id", available)
    subcategory_trend = _lift_map(subcategory_rows, "subcategory", None)
    new_ids = frozenset(pid for pid, item in models.catalog.items() if item.is_new)
    fallback = _fallback_pool(models, available)
    return LiveSnapshot(
        available=available,
        unavailable=unavailable_set,
        region_trending=region_trending,
        product_trend=product_trend,
        subcategory_trend=subcategory_trend,
        new_product_ids=new_ids,
        fallback_pool=fallback,
        prices={pid: models.catalog[pid].price_eur for pid in available},
        built_at=now_ts(),
        source=source,
    )


def _lift_map(rows: list[dict[str, Any]], key: str, universe: frozenset[str] | None) -> dict[str, float]:
    """Turn lift values into a 0..1 score by reciprocal rank (bounded, robust)."""
    pairs: list[tuple[str, float]] = []
    for row in rows:
        name = clean_str(row.get(key))
        if not name:
            continue
        if universe is not None and name not in universe:
            continue
        try:
            lift = float(row.get("lift", 0.0))
        except (TypeError, ValueError):
            lift = 0.0
        if lift >= 1.25:
            pairs.append((name, lift))
    pairs.sort(key=lambda kv: (-kv[1], kv[0]))
    return {name: 1.0 / (rank + 1) for rank, (name, _) in enumerate(pairs[:300])}


def _fallback_pool(models: Models, available: frozenset[str], size: int = 100) -> tuple[str, ...]:
    ranked = sorted(
        (pid for pid in available if pid in models.popularity),
        key=lambda pid: (-models.popularity.get(pid, 0.0), pid),
    )
    if len(ranked) < 10:
        extra = sorted(pid for pid in available if pid not in set(ranked))
        ranked.extend(extra)
    return tuple(ranked[:size])


def static_snapshot(
    models: Models,
    catalog: pd.DataFrame | None = None,
    region_trending: dict[str, list[str]] | None = None,
    product_trend: dict[str, float] | None = None,
    subcategory_trend: dict[str, float] | None = None,
    unavailable: Iterable[str] = (),
    new_ids: Iterable[str] = (),
) -> LiveSnapshot:
    """Offline evaluator's frozen equivalent of LiveState."""
    unavailable_set = frozenset(str(p) for p in unavailable)
    available = frozenset(
        pid for pid, item in models.catalog.items()
        if item.is_active and pid not in unavailable_set
    )
    region = {r: tuple(ps) for r, ps in (region_trending or {}).items()}
    return LiveSnapshot(
        available=available,
        unavailable=unavailable_set,
        region_trending=region,
        product_trend=dict(product_trend or {}),
        subcategory_trend=dict(subcategory_trend or {}),
        new_product_ids=frozenset(str(p) for p in new_ids),
        fallback_pool=_fallback_pool(models, available),
        prices={pid: models.catalog[pid].price_eur for pid in available},
        built_at=now_ts(),
        source="offline",
    )


class LiveState:
    def __init__(self, source: Source, models: Models, config: Config | None = None, verbose: bool = True) -> None:
        self.source = source
        self.models = models
        self.config = config or Config()
        self.verbose = verbose
        self._box = AtomicBox(self._empty_snapshot())
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_catalog = 0.0
        self._last_trends = 0.0
        self._last_purchases = 0.0
        self._last_clock = 0.0
        # Platform clock minus local clock. It is applied to every age we report
        # *and* to the degrade/skip decisions, because request timestamps are the
        # platform's; it is re-measured every minute because it moves (see
        # `_refresh_clock` and docs/DECISIONS.md §14).
        self.clock_offset_s = 0.0
        self._stats: dict[str, float] = {
            "rows": 0, "new_pairs": 0, "failures": 0, "clock_failures": 0,
            "last_ts": 0.0,
        }
        self._stats_lock = threading.Lock()
        self._refresh_lock = threading.Lock()

    def _empty_snapshot(self) -> LiveSnapshot:
        available = frozenset(p for p, item in self.models.catalog.items() if item.is_active)
        return LiveSnapshot(
            available=available, unavailable=frozenset(), region_trending={},
            product_trend={}, subcategory_trend={}, new_product_ids=frozenset(),
            fallback_pool=_fallback_pool(self.models, available),
            prices={p: self.models.catalog[p].price_eur for p in available},
            built_at=now_ts(), source="empty",
        )

    def snapshot(self) -> LiveSnapshot:
        return self._box.get()

    # compatibility with the original team code / notebook
    @property
    def unavailable(self) -> set[str]:
        return set(self._box.get().unavailable)

    @property
    def region_trending(self) -> dict[str, list[str]]:
        return {r: list(v) for r, v in self._box.get().region_trending.items()}

    @property
    def new_products(self) -> dict[str, Any]:
        return {p: {"product_id": p} for p in self._box.get().new_product_ids}

    def refresh_once(self) -> None:
        """Synchronous first refresh before answering anything."""
        self._refresh(
            stock=True, catalog=True, trends=True,
            purchases=self.config.use_live_purchases,
        )
        self._refresh_clock()

    def stats(self) -> dict[str, float]:
        """Counters for the telemetry line (rows read, pairs learned, failures)."""
        with self._stats_lock:
            return dict(self._stats)

    def _bump_stats(self, **deltas: float) -> None:
        with self._stats_lock:
            for key, value in deltas.items():
                self._stats[key] = self._stats.get(key, 0) + value

    def start_background_refresh(self) -> threading.Thread:
        if self._thread and self._thread.is_alive():
            return self._thread

        def loop() -> None:
            while not self._stop.is_set():
                now = time.time()
                try:
                    self._refresh(
                        stock=True,
                        catalog=(now - self._last_catalog >= self.config.catalog_refresh_s),
                        trends=(now - self._last_trends >= self.config.trend_refresh_s),
                        purchases=(
                            self.config.use_live_purchases
                            and now - self._last_purchases >= self.config.live_purchases_refresh_s
                        ),
                    )
                    if now - self._last_clock >= 60.0:
                        self._refresh_clock()
                except Exception as e:
                    if self.verbose:
                        print(f"[live] refresh failed, keeping previous snapshot: {e}")
                self._stop.wait(self.config.stock_refresh_s)

        self._thread = threading.Thread(target=loop, daemon=True, name="live-state")
        self._thread.start()
        return self._thread

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def _refresh(self, stock: bool, catalog: bool, trends: bool, purchases: bool = False) -> None:
        with self._refresh_lock:
            unavailable = set(self._box.get().unavailable)
            region_rows: list[dict[str, Any]] = []
            product_rows: list[dict[str, Any]] = []
            subcategory_rows: list[dict[str, Any]] = []
            trends_ok = True
            prev = self._box.get()
            if stock:
                try:
                    unavailable = {str(r["product_id"]) for r in self.source.fetch("unavailable")}
                except Exception as e:
                    if self.verbose:
                        print(f"[live] stock query failed: {e}")
            # The common case is a 1 s stock poll with an unchanged stock set.
            # Rebuilding the 5k-item snapshot would burn GIL time that the
            # workers need, so publish nothing and keep the current reference.
            if stock and not catalog and not trends and not purchases \
                    and frozenset(unavailable) == prev.unavailable:
                return
            if purchases:
                self._refresh_purchases()
            if catalog:
                self._last_catalog = time.time()
                try:
                    rows = self.source.fetch("catalog_current")
                    new_ids = self.models.sync_catalog(rows)
                    if new_ids and self.verbose:
                        print(f"[live] catalogue sync: {len(new_ids)} new products, {len(rows)} rows")
                except Exception as e:
                    if self.verbose:
                        print(f"[live] catalogue query failed: {e}")
            if trends:
                self._last_trends = time.time()
                try:
                    region_rows = self.source.fetch("region_trending", limit=20)
                    product_rows = self.source.fetch("product_trending", limit=200)
                    subcategory_rows = self.source.fetch("subcategory_trending", limit=40)
                except Exception as e:
                    if self.verbose:
                        print(f"[live] trend query failed: {e}")
                    trends_ok = False
            snap = build_snapshot(self.models, unavailable, region_rows,
                                  product_rows, subcategory_rows)
            if not trends_ok:
                from dataclasses import replace

                snap = replace(
                    snap,
                    region_trending=prev.region_trending,
                    product_trend=prev.product_trend,
                    subcategory_trend=prev.subcategory_trend,
                )
            self._box.set(snap)

    def _refresh_clock(self) -> None:
        """Re-measure platform-minus-local clock offset (telemetry + deadlines).

        Request timestamps come from the platform, so every age we compute is off
        by the difference between the two clocks. That difference moves: it was
        +1.0 s until Windows Time resynced this laptop overnight, after which it
        is ~0.0 s. The update must therefore be **unconditional** — an earlier
        version only stored the value when `|offset| > 0.05`, which froze the
        stale +1 s and made every request look a second older than it was (the
        loop then degraded and skipped requests that were perfectly on time).

        A failed measurement keeps the previous value: with no information, the
        last known offset is a better guess than zero.
        """
        self._last_clock = time.time()
        offset = measure_clock_offset(self.source)
        if offset is None:
            with self._stats_lock:
                self._stats["clock_failures"] = self._stats.get("clock_failures", 0) + 1
            return
        if abs(offset) > 10.0:
            # A wild value is a measurement artefact, not a clock that jumped ten
            # seconds: applying it would skip every request in the round.
            with self._stats_lock:
                self._stats["clock_failures"] = self._stats.get("clock_failures", 0) + 1
            return
        previous = self.clock_offset_s
        self.clock_offset_s = offset
        if self.verbose and abs(offset - previous) > 0.25:
            print(f"[live] clock offset: platform is {offset:+.3f}s ahead of this machine")

    def _refresh_purchases(self) -> None:
        """Pull user-level purchases and merge them into the model state.

        Called with `_refresh_lock` held. A failure is counted and swallowed:
        losing one refresh only means the previous purchase state stays, which
        is exactly what a stale-but-valid feed looks like.
        """
        self._last_purchases = time.time()
        try:
            rows = self.source.fetch(
                "live_purchases", hours=int(self.config.live_purchases_hours)
            )
        except Exception as e:
            self._bump_stats(failures=1)
            if self.verbose:
                print(f"[live] purchase feed failed, keeping previous state: {e}")
            return
        added = self.models.ingest_purchases(
            rows, window_h=self.config.recent_purchase_window_h
        )
        self._bump_stats(rows=len(rows), new_pairs=added, last_ts=time.time())
        if self.verbose and added:
            print(f"[live] purchase feed: {len(rows)} rows, {added} new user-product pairs")
