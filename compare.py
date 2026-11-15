"""Paired comparison of two offline reports (two bundles, same sessions).

`eval_offline.py` compares *policies* inside one run. Comparing two bundles
needs two runs, and `ORDER BY rand()` would hand each run a different sample, so
the difference would be sampling noise. With `--keep-rows` plus a fixed
`--salt`, both runs sample the same sessions and this script pairs them by
`request_id` and bootstraps the per-session delta.

    python training/eval_offline.py --models 2026-11-13r --cut 2026-11-14 \
        --limit 6000 --policies deployed --keep-rows --out reports/a.json
    python training/eval_offline.py --models 2026-11-14r --cut 2026-11-14 \
        --limit 6000 --policies deployed --keep-rows --out reports/b.json
    python training/compare_runs.py reports/a.json reports/b.json \
        --ref-policy deployed --cand-policy deployed

Exit code 0 always: the verdict is for the reader (and for `docs/`), not for a
pipeline. The script refuses to compare runs that did not sample the same
sessions, because that is the one mistake that manufactures a fake winner.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

METRICS = ("purchase_hit", "click_hit", "ndcg_purchase")


def load(path: Path, policy: str) -> tuple[dict[str, Any], dict[str, list[Any]]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    entry = report.get("policies", {}).get(policy)
    if entry is None:
        raise SystemExit(f"{path}: no policy {policy!r} (has: {list(report.get('policies', {}))})")
    rows = entry.get("rows")
    if not rows:
        raise SystemExit(f"{path}: policy {policy!r} has no per-request rows; rerun with --keep-rows")
    return report, rows


def sample_key(report: dict[str, Any]) -> tuple[str, str | None, int]:
    return (str(report.get("cut")), report.get("sample_salt"), int(report.get("n_sessions") or 0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ref", type=Path, help="baseline report")
    parser.add_argument("cand", type=Path, help="candidate report")
    parser.add_argument("--ref-policy", default="deployed")
    parser.add_argument("--cand-policy", default="deployed")
    parser.add_argument("--out", type=Path, default=None, help="write the comparison as JSON")
    parser.add_argument("--boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    ref_report, ref_rows = load(args.ref, args.ref_policy)
    cand_report, cand_rows = load(args.cand, args.cand_policy)

    if sample_key(ref_report) != sample_key(cand_report):
        raise SystemExit(
            "refusing to compare: the two runs did not sample the same sessions\n"
            f"  ref : cut={ref_report.get('cut')} salt={ref_report.get('sample_salt')} "
            f"n={ref_report.get('n_sessions')}\n"
            f"  cand: cut={cand_report.get('cut')} salt={cand_report.get('sample_salt')} "
            f"n={cand_report.get('n_sessions')}"
        )

    shared = sorted(set(ref_rows) & set(cand_rows))
    if len(shared) < 0.99 * max(len(ref_rows), len(cand_rows)):
        raise SystemExit(
            f"refusing to compare: only {len(shared)} shared request_ids "
            f"({len(ref_rows)} vs {len(cand_rows)})"
        )

    from evaluate import paired_bootstrap

    print(f"ref : {args.ref}  models={ref_report.get('models_tag')} policy={args.ref_policy}")
    print(f"cand: {args.cand}  models={cand_report.get('models_tag')} policy={args.cand_policy}")
    print(f"cut={ref_report.get('cut')} salt={ref_report.get('sample_salt')} "
          f"sessions={len(shared)}")
    print()
    print("| metric | ref | cand | delta | 95% CI | win rate | n |")
    print("|---|---|---|---|---|---|---|")

    comparison: dict[str, Any] = {
        "ref": str(args.ref), "cand": str(args.cand),
        "ref_models": ref_report.get("models_tag"),
        "cand_models": cand_report.get("models_tag"),
        "cut": ref_report.get("cut"), "salt": ref_report.get("sample_salt"),
        "sessions": len(shared), "metrics": {},
    }
    for metric in METRICS:
        a = [ref_rows[r][METRICS.index(metric)] for r in shared]
        b = [cand_rows[r][METRICS.index(metric)] for r in shared]
        paired = [i for i in range(len(shared)) if a[i] is not None and b[i] is not None]
        ref_mean = sum(a[i] for i in paired) / max(len(paired), 1)
        cand_mean = sum(b[i] for i in paired) / max(len(paired), 1)
        stats = paired_bootstrap(a, b, n_boot=args.boot, seed=args.seed)
        comparison["metrics"][metric] = {**stats, "ref_mean": round(ref_mean, 5),
                                         "cand_mean": round(cand_mean, 5)}
        print(f"| {metric} | {ref_mean:.4f} | {cand_mean:.4f} | "
              f"{stats.get('delta', float('nan')):+.4f} | "
              f"[{stats.get('ci_low', float('nan')):+.4f}, {stats.get('ci_high', float('nan')):+.4f}] | "
              f"{stats.get('win_rate')} | {stats.get('n')} |")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
