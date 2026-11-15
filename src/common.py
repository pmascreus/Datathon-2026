"""Small, dependency-free helpers shared by training/, src/ and tests/.

Nothing here knows about recommender logic: parsing timestamps, robust string
normalisation, atomic snapshots and percentile summaries.
"""
from __future__ import annotations

import math
import threading
import time
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence


def now_ts() -> float:
    """Unix seconds, UTC. The only clock used for deadlines."""
    return time.time()


def measure_clock_offset(source: Any, samples: int = 3) -> float | None:
    """Platform clock minus local clock, in seconds (median of `samples`).

    Request timestamps are stamped by the platform. If the local clock drifts,
    every age we compute and report is shifted by that drift: this machine ran
    ~1 s behind for a while, which made `submit_age_ms` look negative and
    understated. Each sample brackets the query with local time and uses the
    midpoint, so the round-trip cancels out.

    Returns `None` when the clock cannot be measured (API error, no rows). That
    is deliberately *not* `0.0`: zero means "clocks agree", and collapsing the
    two would let a transient API error silently switch the correction off.
    """
    offsets: list[float] = []
    for _ in range(max(int(samples), 1)):
        try:
            t0 = time.time()
            rows = source.fetch("clock_now")
            t1 = time.time()
            remote = float(rows[0]["ms"]) / 1000.0
        except Exception:
            return None
        offsets.append(remote - 0.5 * (t0 + t1))
    offsets.sort()
    return offsets[len(offsets) // 2]


def parse_ts(value: Any) -> float | None:
    """Parse a RawTree/ISO timestamp into unix seconds.

    RawTree returns timestamps as ISO strings ("2026-11-14T16:02:11.482Z") but
    pandas may hand us datetime objects or epoch numbers. Returns None instead
    of raising: a malformed request must not take the serving loop down.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        v = float(value)
        # Heuristic: some exports store epoch milliseconds instead of seconds.
        return v / 1000.0 if v > 1e11 else v
    if isinstance(value, datetime):
        dt = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    text = str(value).strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except ValueError:
        return None


def iso_ms(ts: float | None = None) -> str:
    """UTC ISO string with millisecond precision (telemetry timestamps)."""
    dt = datetime.fromtimestamp(now_ts() if ts is None else ts, tz=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def clean_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)


def as_list(value: Any) -> list:
    """RawTree arrays arrive as lists, but be forgiving about None/strings."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        f = float(value)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def minmax(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi - lo < 1e-12:
        return [0.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def rank_normalize(scores: dict[str, float], top: int | None = None) -> dict[str, float]:
    """Map a channel's raw scores to (0, 1] using reciprocal rank.

    Reciprocal rank is deliberately used instead of min-max: raw co-occurrence
    counts have heavy tails, and min-max would let one product with an extreme
    count flatten every other signal in the same channel.
    """
    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    if top is not None:
        ordered = ordered[:top]
    return {pid: 1.0 / (rank + 1) for rank, (pid, _) in enumerate(ordered)}


def percentile(values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile (q in [0, 1]); no numpy dependency needed."""
    if not values:
        return float("nan")
    data = sorted(values)
    if len(data) == 1:
        return float(data[0])
    pos = max(0.0, min(1.0, q)) * (len(data) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(data[lo])
    return float(data[lo] + (data[hi] - data[lo]) * (pos - lo))


def summarize(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean": float(sum(values) / len(values)),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": float(max(values)),
    }


class AtomicBox:
    """Publish an immutable snapshot with a single reference assignment.

    Inference threads only ever read `get()`; the background syncer builds a
    new object and calls `set()`. No lock is taken on the hot path, so a slow
    refresh can never block a request.
    """

    __slots__ = ("_value", "_lock")

    def __init__(self, value: Any) -> None:
        self._value = value
        self._lock = threading.Lock()

    def get(self) -> Any:
        return self._value

    def set(self, value: Any) -> None:
        with self._lock:
            self._value = value


def dedupe_keep_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out
