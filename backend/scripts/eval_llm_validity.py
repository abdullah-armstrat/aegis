"""LLM metric 1 of 3 — OUTPUT VALIDITY RATE.

How often does phi3:mini actually return the well-formed JSON the prompt asks for? Until now
this was unanswerable after the fact: the raw response string is parsed and discarded, so only
the *parsed* verdict survived. This script turns on the reasoner's measurement hook
(``llm_reasoner.record_calls``), which retains the raw string and records how the tolerant
parser resolved each field, then classifies every call:

  clean       — parsed directly; every key exact, every bool a real JSON bool
  salvaged    — a key only matched by _find_key's 4-letter stem fallback (the "explanrance" case)
  coerced     — a bool arrived as a string ("true"/"yes") and needed _coerce_bool
  unparseable — fell to the unavailable path (timeout, HTTP error, non-JSON, not an object)

Runs the full production path (build_scorecard -> _llm_caption_scene_flag -> reason_over_text)
over the labelled caption↔scene cases, so the flag verdict is collected alongside the validity
outcome — giving completion rate and ground-truth agreement from the same run.

The input-hash cache is BYPASSED: a cache hit returns a stored verdict with no raw response and
no network call, so it would measure nothing. Nothing is written back to the cache either.

Run:  python backend/scripts/eval_llm_validity.py [--json out.json]
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
os.environ["AEGIS_USE_CAPTIONER"] = "false"  # scenes are injected; don't load BLIP

from app.config import get_settings  # noqa: E402

get_settings.cache_clear()

from app.fusion import llm_reasoner  # noqa: E402
from app.fusion.scorecard import build_scorecard  # noqa: E402
from app.models import FlagStatus  # noqa: E402
from tests.eval.llm_cases import FLAG, bundle_for, load_caption_scene_cases  # noqa: E402

_STATUS_TO_PRED = {
    FlagStatus.FIRED: "fired",
    FlagStatus.CLEAR: "clear",
    FlagStatus.NOT_ASSESSED: "not_assessed",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure LLM output validity rate.")
    parser.add_argument("--json", type=str, default="", help="write full results to this path")
    args = parser.parse_args()

    cases = load_caption_scene_cases()
    rows: list[dict] = []

    for case in cases:
        sink: list[llm_reasoner.CallRecord] = []
        with llm_reasoner.record_calls(sink, bypass_cache=True):
            card = build_scorecard(bundle_for(case))

        llm_flag = next(
            (f for f in card.flags if f.type.value == FLAG and f.source == "llm"), None
        )
        # Exactly one reasoner call per scorecard build; guard rather than assume.
        assert len(sink) == 1, f"{case['id']}: expected 1 reasoner call, saw {len(sink)}"
        rec = sink[0]

        expected = case["expected"][FLAG]  # "fire" | "clear"
        predicted = _STATUS_TO_PRED[llm_flag.status] if llm_flag else "missing"
        agrees = (
            None
            if predicted == "not_assessed"
            else (predicted == "fired") == (expected == "fire")
        )

        rows.append({
            "id": case["id"],
            "validity": rec.validity,
            "available": rec.available,
            "latency_s": round(rec.latency_s, 3) if rec.latency_s is not None else None,
            "error_type": rec.error_type,
            "is_timeout": rec.is_timeout,
            "expected": expected,
            "llm_predicted": predicted,
            "agrees_with_ground_truth": agrees,
            "missing_keys": rec.detail.get("missing_keys", []),
            "extra_keys": rec.detail.get("extra_keys", []),
            "key_resolution": rec.detail.get("key_resolution", {}),
            "bool_resolution": rec.detail.get("bool_resolution", {}),
            "raw_response": rec.raw_response,
        })

    n = len(rows)
    counts = Counter(r["validity"] for r in rows)
    completed = [r for r in rows if r["available"]]
    judged = [r for r in rows if r["agrees_with_ground_truth"] is not None]
    agreed = [r for r in judged if r["agrees_with_ground_truth"]]

    print(f"\n=== LLM output validity — model={get_settings().ollama_model}, N={n} calls ===")
    print(f"{'id':<12}{'validity':<14}{'latency_s':>10}  {'expected':<9}{'llm':<14}agrees")
    print("-" * 74)
    for r in rows:
        lat = f"{r['latency_s']:.2f}" if r["latency_s"] is not None else "-"
        agrees = "-" if r["agrees_with_ground_truth"] is None else str(r["agrees_with_ground_truth"])
        print(
            f"{r['id']:<12}{r['validity']:<14}{lat:>10}  {r['expected']:<9}"
            f"{r['llm_predicted']:<14}{agrees}"
        )
    print("-" * 74)
    for outcome in ("clean", "salvaged", "coerced", "unparseable"):
        c = counts.get(outcome, 0)
        print(f"  {outcome:<12} {c}/{n}  ({c / n:.0%})")
    other = {k: v for k, v in counts.items() if k not in
             {"clean", "salvaged", "coerced", "unparseable"}}
    if other:
        print(f"  other outcomes (not in the taxonomy): {dict(other)}")

    well_formed = counts.get("clean", 0)
    usable = n - counts.get("unparseable", 0)
    print(f"\n  raw JSON well-formed (clean)     : {well_formed}/{n} = {well_formed / n:.1%}")
    print(f"  usable after tolerant parsing    : {usable}/{n} = {usable / n:.1%}")
    print(f"  completion rate (available=True) : {len(completed)}/{n} = {len(completed) / n:.1%}")
    if judged:
        print(
            f"  ground-truth agreement           : {len(agreed)}/{len(judged)} "
            f"= {len(agreed) / len(judged):.1%}  (of calls that returned a verdict)"
        )
    missing_any = [r["id"] for r in rows if r["missing_keys"]]
    if missing_any:
        print(f"  NOTE: calls with an absent key  : {missing_any} (neither salvaged nor coerced)")

    print(f"\n  N={n}. This is a mechanism illustration, not a population estimate.")

    if args.json:
        out = {
            "model": get_settings().ollama_model,
            "n_calls": n,
            "counts": dict(counts),
            "completion_rate": len(completed) / n,
            "ground_truth_agreement": (len(agreed) / len(judged)) if judged else None,
            "rows": rows,
        }
        Path(args.json).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
