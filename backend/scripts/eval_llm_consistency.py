"""LLM metric 2 of 3 — SELF-CONSISTENCY across repeated runs.

The report asserts that phi3:mini is non-deterministic; nothing measures it. This runs the SAME
input k times (default 5) with the input-hash cache bypassed — without that bypass every run
after the first returns the identical stored verdict and consistency would be a meaningless
1.00 — and reports, per input, the modal verdict and the agreement rate for ``same_subject``
and ``same_tone``.

Agreement rate = (count of the modal value) / k, over the runs that returned a verdict.
1.00 means every run agreed; 0.60 on k=5 means the model changed its mind twice. Runs that
failed (timeout / unparseable) are excluded from the agreement denominator and reported
separately, because "the model did not answer" is not "the model disagreed with itself".

Default inputs are the three hard synonym cases (where the LLM's semantic judgement is the
whole value proposition) plus mm01, the flood/dry-street case that timed out in both the
2026-06-01 and 2026-08-17 comparison runs — so a systematic failure shows up as a repeated one.

Run:  python backend/scripts/eval_llm_consistency.py [--k 5] [--ids mm01,hard_kc01] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

os.environ["AEGIS_USE_LLM"] = "true"
os.environ["AEGIS_USE_CAPTIONER"] = "false"

from app.config import get_settings  # noqa: E402

get_settings.cache_clear()

from app.fusion import llm_reasoner  # noqa: E402
from app.fusion.llm_reasoner import reason_over_text  # noqa: E402
from tests.eval.llm_cases import reasoner_inputs, select  # noqa: E402

DEFAULT_IDS = ["mm01", "hard_kc01", "hard_kc02", "hard_kc03"]


def _agreement(values: list) -> tuple[object, float | None]:
    """Modal value and its share of the given values. (None, None) when there are none."""
    if not values:
        return None, None
    counts = Counter(map(repr, values))
    modal_repr, modal_count = counts.most_common(1)[0]
    modal = next(v for v in values if repr(v) == modal_repr)
    return modal, modal_count / len(values)


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure LLM self-consistency over k runs.")
    parser.add_argument("--k", type=int, default=5, help="runs per input (default 5)")
    parser.add_argument("--ids", type=str, default="", help="comma-separated case ids")
    parser.add_argument("--json", type=str, default="", help="write full results to this path")
    args = parser.parse_args()

    ids = [i.strip() for i in args.ids.split(",") if i.strip()] or DEFAULT_IDS
    cases = select(ids)
    k = args.k
    rows: list[dict] = []

    for case in cases:
        caption, scene, on_screen = reasoner_inputs(case)
        runs: list[dict] = []
        for _ in range(k):
            sink: list[llm_reasoner.CallRecord] = []
            with llm_reasoner.record_calls(sink, bypass_cache=True):
                verdict = reason_over_text(caption, scene, on_screen)
            rec = sink[0] if sink else None
            runs.append({
                "available": verdict.available,
                "same_subject": verdict.same_subject,
                "same_tone": verdict.same_tone,
                "explanation": verdict.explanation,
                "validity": rec.validity if rec else None,
                "latency_s": round(rec.latency_s, 3) if rec and rec.latency_s is not None else None,
            })

        ok = [r for r in runs if r["available"]]
        subj_modal, subj_rate = _agreement([r["same_subject"] for r in ok])
        tone_modal, tone_rate = _agreement([r["same_tone"] for r in ok])
        rows.append({
            "id": case["id"],
            "caption": caption,
            "scene": scene,
            "k": k,
            "returned": len(ok),
            "failed": k - len(ok),
            "same_subject_modal": subj_modal,
            "same_subject_agreement": subj_rate,
            "same_tone_modal": tone_modal,
            "same_tone_agreement": tone_rate,
            "runs": runs,
        })

    print(f"\n=== LLM self-consistency — model={get_settings().ollama_model}, k={k} runs/input ===")
    header = f"{'id':<12}{'ret/k':>7}  {'subject':>9} {'agree':>6}   {'tone':>6} {'agree':>6}"
    print(header)
    print("-" * len(header))
    for r in rows:
        sr = f"{r['same_subject_agreement']:.2f}" if r["same_subject_agreement"] is not None else "  -"
        tr = f"{r['same_tone_agreement']:.2f}" if r["same_tone_agreement"] is not None else "  -"
        print(
            f"{r['id']:<12}{r['returned']}/{r['k']:<5}  {str(r['same_subject_modal']):>9} {sr:>6}"
            f"   {str(r['same_tone_modal']):>6} {tr:>6}"
        )
    print("-" * len(header))

    rates = [r["same_subject_agreement"] for r in rows if r["same_subject_agreement"] is not None]
    if rates:
        print(f"  mean same_subject agreement: {sum(rates) / len(rates):.2f} over {len(rates)} input(s)")
    total_failed = sum(r["failed"] for r in rows)
    print(f"  failed runs (excluded from agreement): {total_failed}/{len(rows) * k}")
    for r in rows:
        if r["failed"]:
            print(f"    {r['id']}: {r['failed']}/{k} runs returned no verdict")

    print(
        f"\n  {len(rows)} input(s) x k={k}. Agreement on k=5 can only take the values "
        f"0.40/0.60/0.80/1.00 for a binary field, so this is a mechanism illustration, "
        f"not a population estimate."
    )

    if args.json:
        Path(args.json).write_text(
            json.dumps({"model": get_settings().ollama_model, "k": k, "rows": rows}, indent=2),
            encoding="utf-8",
        )
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
