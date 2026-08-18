"""LLM metric 3 of 3 — LATENCY DISTRIBUTION.

Replaces the single 2026-05-31 spike figure ("warm ~6-9s") with a measured distribution. The
reasoner's measurement hook wraps the ``httpx.post`` call in ``perf_counter``, so what is timed
is the round trip to Ollama — the model's own generation time plus local HTTP overhead. It does
NOT include the one-off cold model load, and it does not include rules-layer time.

Reports median, p95, min, max and the count of calls that hit the 30 s ``_TIMEOUT_SECONDS``
ceiling. Timed-out calls ARE included in the distribution (they are real latency the user would
experience) and are also counted separately.

p95 uses the nearest-rank method. At the default N this is degenerate — see the note the script
prints — so pass --repeats to enlarge the sample if the extra minutes are affordable.

The cache is bypassed: a cache hit does no network call and would record no latency at all.

Run:  python backend/scripts/eval_llm_latency.py [--repeats 1] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

os.environ["AEGIS_USE_LLM"] = "true"
os.environ["AEGIS_USE_CAPTIONER"] = "false"

from app.config import get_settings  # noqa: E402

get_settings.cache_clear()

from app.fusion import llm_reasoner  # noqa: E402
from app.fusion.llm_reasoner import _TIMEOUT_SECONDS, reason_over_text  # noqa: E402
from tests.eval.llm_cases import load_caption_scene_cases, reasoner_inputs  # noqa: E402


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Nearest-rank percentile: the smallest value at or above pct of the sample."""
    rank = max(1, math.ceil(pct / 100 * len(sorted_values)))
    return sorted_values[rank - 1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure LLM latency distribution.")
    parser.add_argument("--repeats", type=int, default=1, help="passes over the case set")
    parser.add_argument("--json", type=str, default="", help="write full results to this path")
    args = parser.parse_args()

    cases = load_caption_scene_cases()
    rows: list[dict] = []

    for pass_no in range(args.repeats):
        for case in cases:
            caption, scene, on_screen = reasoner_inputs(case)
            sink: list[llm_reasoner.CallRecord] = []
            with llm_reasoner.record_calls(sink, bypass_cache=True):
                reason_over_text(caption, scene, on_screen)
            rec = sink[0]
            rows.append({
                "id": case["id"],
                "pass": pass_no + 1,
                "latency_s": rec.latency_s,
                "available": rec.available,
                "is_timeout": rec.is_timeout,
                "validity": rec.validity,
            })

    latencies = sorted(r["latency_s"] for r in rows if r["latency_s"] is not None)
    timeouts = [r for r in rows if r["is_timeout"]]
    succeeded = sorted(r["latency_s"] for r in rows if r["available"])

    n = len(latencies)
    print(
        f"\n=== LLM latency — model={get_settings().ollama_model}, "
        f"N={n} calls, timeout ceiling {_TIMEOUT_SECONDS:.0f}s ==="
    )
    print(f"{'id':<12}{'pass':>5}{'latency_s':>11}  {'outcome':<14}")
    print("-" * 44)
    for r in rows:
        lat = f"{r['latency_s']:.2f}" if r["latency_s"] is not None else "-"
        outcome = "TIMEOUT" if r["is_timeout"] else r["validity"]
        print(f"{r['id']:<12}{r['pass']:>5}{lat:>11}  {outcome:<14}")
    print("-" * 44)

    if n:
        print(f"  all calls (n={n}):")
        print(f"    median : {statistics.median(latencies):6.2f} s")
        print(f"    p95    : {_percentile(latencies, 95):6.2f} s   (nearest-rank)")
        print(f"    min    : {latencies[0]:6.2f} s")
        print(f"    max    : {latencies[-1]:6.2f} s")
    if succeeded:
        print(f"  successful calls only (n={len(succeeded)}):")
        print(f"    median : {statistics.median(succeeded):6.2f} s")
        print(f"    min    : {succeeded[0]:6.2f} s")
        print(f"    max    : {succeeded[-1]:6.2f} s")
    print(f"  timeouts at the {_TIMEOUT_SECONDS:.0f}s ceiling: {len(timeouts)}/{len(rows)}")
    for r in timeouts:
        print(f"    {r['id']} (pass {r['pass']})")

    print(
        f"\n  N={n}. At this N the p95 is the largest or second-largest single observation, so "
        f"it carries no distributional weight — it is reported for completeness, not as a "
        f"stable tail estimate. Timing is the Ollama round trip on an i5-8350U with no CUDA, "
        f"excluding the one-off cold model load."
    )

    if args.json:
        out = {
            "model": get_settings().ollama_model,
            "timeout_ceiling_s": _TIMEOUT_SECONDS,
            "n_calls": len(rows),
            "median_s": statistics.median(latencies) if latencies else None,
            "p95_s": _percentile(latencies, 95) if latencies else None,
            "min_s": latencies[0] if latencies else None,
            "max_s": latencies[-1] if latencies else None,
            "timeout_count": len(timeouts),
            "rows": rows,
        }
        Path(args.json).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
