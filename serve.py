"""Live serving loop: pull -> recommend -> push, one thread, one job.

The decision itself is `src.policy.recommend()`, shared with `evaluate.py`, so
what we measured offline is exactly what runs here. This file is only plumbing.

Why a single thread is enough (measured, not assumed): compute is ~18 ms p50 /
~45 ms p95, a pull is ~200 ms and an insert of a batch is ~300 ms. A serial
loop with batches of 50 answers 50 requests in ~1.4 s, inside the 2.5 s
deadline. The previous design used a poller, 4 workers and a writer because the
first version issued one HTTP call per request and serialised two network
round-trips with inference; that problem is gone.

Guarantees kept, because they are what makes the round survivable:

* every list is re-validated (10 distinct, active, available) and repaired from
  the always-valid fallback pool: a model bug can never produce an invalid row;
* a request older than `hard_deadline_s` is dropped (the platform already
  answered it), and one that cannot be computed in time is answered with the
  fallback list instead of being computed too late;
* a persisted seen-journal prevents duplicate submissions across restarts;
* ages and deadlines use the platform's clock (measured offset), not ours.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.common import iso_ms, parse_ts, summarize
from src.config import PHASE_POLICIES, Config
from src.data import make_source
from src.policy import fallback_items, recommend
from src.retrieval import Session
from src.state import LiveSnapshot, LiveState
from src.store import Models

STATE_DIR = ROOT / "state"
SEEN_PATH = STATE_DIR / "seen_requests.txt"
TELEMETRY_PATH = STATE_DIR / "telemetry.jsonl"


class SeenJournal:
    """Request ids we already answered, persisted so a restart never repeats.

    The in-memory set is the hot path; the file is append-only and compacted
    when it grows past twice the cap. A restart keeps the last `max_items` ids,
    far more than the platform's retry window.
    """

    def __init__(self, path: Path = SEEN_PATH, max_items: int = 400_000) -> None:
        self.path = path
        self.max_items = max_items
        self._ids: set[str] = set()
        self._order: deque[str] = deque()
        self._dirty = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            for line in self.path.read_text(encoding="utf-8").splitlines():
                rid = line.strip()
                if rid:
                    self._remember(rid)
        except OSError:
            pass

    def _remember(self, rid: str) -> None:
        if rid in self._ids:
            return
        self._ids.add(rid)
        self._order.append(rid)
        if len(self._order) > self.max_items:
            self._ids.discard(self._order.popleft())

    def seen(self, rid: str) -> bool:
        return rid in self._ids

    def add(self, rid: str) -> None:
        if rid in self._ids:
            return
        self._remember(rid)
        self._dirty += 1

    def flush(self) -> None:
        if self._dirty == 0:
            return
        try:
            if self._dirty > self.max_items:
                self.path.write_text("\n".join(self._order) + "\n", encoding="utf-8")
            else:
                with self.path.open("a", encoding="utf-8") as f:
                    for rid in list(self._order)[-self._dirty:]:
                        f.write(rid + "\n")
            self._dirty = 0
        except OSError as e:
            print(f"[serve] seen journal write failed: {e}")

    def __len__(self) -> int:
        return len(self._ids)


def validate(items: list[str], models: Models, snapshot: LiveSnapshot) -> tuple[list[str], bool]:
    """Exactly 10 distinct products that are active and available right now.

    `repaired=True` means the model list was short or contained something
    illegal and the fallback pool filled the gap: the honest "model served" vs
    "fallback served" telemetry.
    """
    out: list[str] = []
    seen: set[str] = set()
    for pid in items:
        if pid in seen:
            continue
        item = models.item(pid)
        if item is None or not item.is_active:
            continue
        if pid not in snapshot.available:
            continue
        seen.add(pid)
        out.append(pid)
        if len(out) >= 10:
            break
    repaired = len(out) < 10
    if repaired:
        out.extend(fallback_items(models, snapshot, k=10 - len(out), exclude=seen))
    return out[:10], repaired


def answer(
    req: dict[str, Any],
    models: Models,
    live: LiveState,
    cfg: Config,
    rng: np.random.Generator,
    version: str,
    degrade: bool = False,
    fast_cfg: Config | None = None,
) -> tuple[dict[str, Any], bool, float]:
    """One request -> one submission row. Never raises."""
    snapshot = live.snapshot()
    started = time.perf_counter()
    used_fallback = False
    try:
        session = Session.from_request(req)
        active_cfg = fast_cfg if (degrade and fast_cfg is not None) else cfg
        decision = recommend(session, models, snapshot, active_cfg, rng=rng)
        items = decision.items
        used_fallback = decision.source == "fallback"
    except Exception as e:  # a policy bug must degrade to a valid answer
        print(f"[serve] recommend failed for {req.get('request_id')}: {e!r}", flush=True)
        items = fallback_items(models, snapshot)
        used_fallback = True
    items, repaired = validate(items, models, snapshot)
    compute_ms = (time.perf_counter() - started) * 1000.0
    row = {
        "request_id": str(req.get("request_id", "")),
        "items": items,
        "model_version": version,
        "_ts": parse_ts(req.get("ts")) or 0.0,
    }
    return row, (used_fallback or repaired), compute_ms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="2026-11-13r", help="artefact tag under models/")
    parser.add_argument("--version", default="versio4", help="model_version written to submissions")
    parser.add_argument("--duration", type=float, default=0.0, help="stop after N seconds (0 = forever)")
    parser.add_argument("--limit", type=int, default=50, help="rows per pull")
    parser.add_argument("--degrade-at", type=float, default=2.0,
                        help="requests older than this use the cheap personalised path")
    parser.add_argument("--dry-run", action="store_true", help="build answers but do not submit")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = Config.load()
    fast_cfg = PHASE_POLICIES["p_fast"]
    source = make_source("rawtree")
    models = Models.load(args.models)
    live = LiveState(source, models, cfg)
    live.refresh_once()
    live.start_background_refresh()
    seen = SeenJournal()
    rng = np.random.default_rng(args.seed)

    print(
        f"[serve] version={args.version} models={args.models} "
        f"items={len(models.catalog)} available={len(live.snapshot().available)} "
        f"ranker={'yes' if models.ranker else 'no'} dry_run={args.dry_run}",
        flush=True,
    )

    # ---- counters and rolling estimates ------------------------------------
    stats = {
        "pulled": 0, "answered": 0, "submitted": 0, "late_skipped": 0,
        "degraded": 0, "deadline_fallback": 0, "fallback": 0, "errors": 0,
        "insert_failures": 0, "feedback_rows": 0,
    }
    compute_ms: list[float] = []
    push_ms: list[float] = []
    submit_age_ms: list[float] = []
    visibility_age_ms: list[float] = []
    lat = {"compute_ms": 20.0, "push_ms": 350.0}
    started_at = time.time()

    def platform_age(age: float) -> float:
        """Age on the platform's clock, which is the clock the deadline uses."""
        if not cfg.use_clock_offset:
            return age
        return age + float(getattr(live, "clock_offset_s", 0.0) or 0.0)

    def lat_update(key: str, value: float) -> None:
        # Clamp one outlier so a single slow request cannot make the loop think
        # the machine is 10x slower than it is.
        current = lat[key]
        lat[key] = 0.85 * current + 0.15 * min(value, max(4.0 * current, 1.0))

    def over_budget(age: float) -> bool:
        """True when running the full model would likely cross the deadline."""
        if age > args.degrade_at:
            return True
        needed = (lat["compute_ms"] + lat["push_ms"]) / 1000.0
        return (cfg.hard_deadline_s - age) < 1.5 * needed + 0.10

    def flush(batch: list[dict[str, Any]]) -> bool:
        """Insert a batch; returns False on failure (the batch is kept)."""
        if not batch:
            return True
        started = time.perf_counter()
        sent_at = time.time()
        payload = [{k: v for k, v in row.items() if not k.startswith("_")} for row in batch]
        ages = [platform_age(sent_at - float(row.get("_ts") or sent_at)) * 1000.0 for row in batch]
        try:
            if args.dry_run:
                stats["submitted"] += len(batch)
            else:
                source.insert("submissions", payload)
                stats["submitted"] += len(batch)
                seen.flush()
            push_ms.append((time.perf_counter() - started) * 1000.0)
            submit_age_ms.extend(ages)
            lat_update("push_ms", push_ms[-1])
            return True
        except Exception as e:
            print(f"[serve] insert failed, will retry: {e!r}", flush=True)
            stats["errors"] += 1
            stats["insert_failures"] += 1
            if len(batch) > 400:  # bound memory if the API stays down
                drop = len(batch) - 400
                del batch[:drop]
                print(f"[serve] dropping {drop} unsent answers", flush=True)
            return False

    last_summary = time.time()
    last_feedback = time.time()
    feedback_since = iso_ms(time.time() - 3600)
    batch: list[dict[str, Any]] = []
    last_flush = time.time()

    try:
        while True:
            now = time.time()
            if args.duration and (now - started_at) >= args.duration:
                break

            # ---- pull ------------------------------------------------------
            try:
                requests_ = source.fetch("requests_pull", limit=args.limit)
            except Exception as e:
                print(f"[serve] pull failed: {e!r}", flush=True)
                stats["errors"] += 1
                time.sleep(cfg.poll_interval_s)
                continue
            stats["pulled"] += len(requests_)

            # ---- recommend (oldest first) ---------------------------------
            for req in sorted(requests_, key=lambda r: str(r.get("ts") or "")):
                rid = str(req.get("request_id", ""))
                if not rid or seen.seen(rid):
                    continue
                now = time.time()
                req_ts = parse_ts(req.get("ts")) or now
                age = platform_age(now - req_ts)
                if age > cfg.hard_deadline_s:
                    seen.add(rid)
                    stats["late_skipped"] += 1
                    continue
                seen.add(rid)
                visibility_age_ms.append(age * 1000.0)

                if cfg.hard_deadline_s - age < 2.0 * lat["compute_ms"] / 1000.0:
                    # Even the cheap path would land late: answer with the
                    # always-valid pool instead of computing a stale list.
                    stats["deadline_fallback"] += 1
                    row = {
                        "request_id": rid,
                        "items": fallback_items(models, live.snapshot()),
                        "model_version": args.version,
                        "_ts": req_ts,
                    }
                    batch.append(row)
                    seen.add(rid)
                    continue

                degrade = over_budget(age)
                row, fb, ms = answer(req, models, live, cfg, rng, args.version,
                                     degrade=degrade, fast_cfg=fast_cfg)
                compute_ms.append(ms)
                lat_update("compute_ms", ms)
                stats["answered"] += 1
                stats["fallback"] += 1 if fb else 0
                stats["degraded"] += 1 if degrade else 0
                batch.append(row)

                if len(batch) >= cfg.batch_max or (time.time() - last_flush) >= cfg.batch_flush_age_s:
                    if flush(batch):
                        batch.clear()
                    last_flush = time.time()

            # ---- push ------------------------------------------------------
            if batch:
                if flush(batch):
                    batch.clear()
                last_flush = time.time()

            # ---- feedback (bandit posteriors only) -------------------------
            now = time.time()
            if now - last_feedback >= cfg.feedback_refresh_s:
                last_feedback = now
                try:
                    rows = source.fetch("feedback_since", since=feedback_since, limit=5000)
                    if rows:
                        models.apply_feedback(rows)
                        stats["feedback_rows"] += len(rows)
                        feedback_since = str(rows[-1].get("ts") or feedback_since)
                except Exception as e:
                    print(f"[serve] feedback poll failed: {e!r}", flush=True)

            # ---- telemetry -------------------------------------------------
            if now - last_summary >= 15.0:
                last_summary = now
                summary = {
                    "ts": iso_ms(now),
                    "version": args.version,
                    **stats,
                    "seen": len(seen),
                    "fallback_rate": stats["fallback"] / max(stats["answered"], 1),
                    "compute_ms": summarize(compute_ms[-5000:]),
                    "push_ms": summarize(push_ms[-500:]),
                    "submit_age_ms": summarize(submit_age_ms[-2000:]),
                    "visibility_age_ms": summarize(visibility_age_ms[-2000:]),
                    "clock_offset_s": round(float(getattr(live, "clock_offset_s", 0.0) or 0.0), 3),
                    "clock_failures": int(live.stats().get("clock_failures", 0)),
                    "lat_estimate_ms": dict(lat),
                    "live_purchases": live.stats(),
                    "uptime_s": round(now - started_at, 1),
                }
                compute_ms[:] = compute_ms[-5000:]
                push_ms[:] = push_ms[-500:]
                submit_age_ms[:] = submit_age_ms[-2000:]
                visibility_age_ms[:] = visibility_age_ms[-2000:]
                print(f"[serve] {json.dumps(summary, default=str)}", flush=True)
                try:
                    TELEMETRY_PATH.parent.mkdir(parents=True, exist_ok=True)
                    with TELEMETRY_PATH.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(summary, default=str) + "\n")
                except OSError:
                    pass

            if not requests_:
                time.sleep(cfg.poll_interval_s)
    except KeyboardInterrupt:
        print("[serve] shutting down", flush=True)
    finally:
        if batch:
            flush(batch)
        seen.flush()
        live.stop()
        print(f"[serve] final {json.dumps(stats, default=str)}", flush=True)


if __name__ == "__main__":
    main()
