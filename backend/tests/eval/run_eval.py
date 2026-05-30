"""Evaluation harness — runs the hand-built test set through the real pipeline and reports
flag-level precision / recall / F1 (SSOT §5.3).

What it does, honestly and reproducibly:
  * Loads the labelled examples (``test_set/examples.json``).
  * Ensures each referenced image exists, generating a tiny blank PNG if missing — so the set
    is self-contained and the run is reproducible without committing binaries.
  * For each example, builds the Evidence Bundle with the *real* extractors, optionally injects
    controlled ``scene_descriptions`` (standing in for the not-yet-integrated captioner so the
    caption↔scene *rule* can be scored), then runs the *real* fusion core.
  * Compares each labelled flag's predicted status to its expected outcome and accumulates a
    confusion matrix per flag type (NOT_ASSESSED excluded from P/R, counted as coverage —
    ADR-009).
  * Runs both configurations — "rules-only" and "rules+llm" — so the baseline comparison the
    Prelim wants (SSOT §5.3) is a single command.

Every number this prints comes from an actual executed run of the pipeline; the harness invents
nothing. Usage:
    python tests/eval/run_eval.py                 # both configs, human-readable
    python tests/eval/run_eval.py --json out.json # also write full results
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Make the backend package importable however this is launched.
_BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from app.adapters.image_adapter import build_bundle  # noqa: E402
from app.fusion.scorecard import build_scorecard  # noqa: E402
from app.models import FlagStatus, SceneDescription  # noqa: E402
from tests.eval.metrics import build_report  # noqa: E402

_TEST_SET_DIR = Path(__file__).resolve().parent / "test_set"
_IMAGES_DIR = _TEST_SET_DIR / "images"

_STATUS_TO_PRED = {
    FlagStatus.FIRED: "fired",
    FlagStatus.CLEAR: "clear",
    FlagStatus.NOT_ASSESSED: "not_assessed",
}


def _load_examples() -> list[dict]:
    data = json.loads((_TEST_SET_DIR / "examples.json").read_text(encoding="utf-8"))
    return data["examples"]


def _ensure_image(filename: str) -> bytes:
    _IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    path = _IMAGES_DIR / filename
    if not path.exists():
        from PIL import Image

        Image.new("RGB", (64, 32), "white").save(path, format="PNG")
    return path.read_bytes()


def _predict(example: dict, use_llm: bool) -> dict[str, str]:
    """Run the pipeline for one example and return {flag_type: predicted_status_str}."""
    os.environ["AEGIS_USE_LLM"] = "true" if use_llm else "false"
    from app.config import get_settings

    get_settings.cache_clear()

    image_bytes = _ensure_image(example["image"])
    bundle = build_bundle(
        image_bytes,
        caption=example.get("caption") or None,
        source_ref=example.get("source_ref") or example["image"],
    )
    if example.get("inject_scene"):
        bundle.scene_descriptions = [SceneDescription(text=t) for t in example["inject_scene"]]

    card = build_scorecard(bundle)
    # Rules emit one flag per type; the LLM may add one more. Keep the rules' deterministic
    # verdict as the scored one for reproducibility.
    predicted: dict[str, str] = {}
    for flag in card.flags:
        if flag.source == "rules" or flag.type.value not in predicted:
            predicted[flag.type.value] = _STATUS_TO_PRED[flag.status]
    return predicted


def evaluate(use_llm: bool):
    """Run every example for one configuration; return (EvalReport, per-example records)."""
    examples = _load_examples()
    pairs_by_flag: dict[str, list[tuple[str, str]]] = {}
    records: list[dict] = []

    for ex in examples:
        predicted = _predict(ex, use_llm=use_llm)
        row = {"id": ex["id"], "category": ex["category"], "flags": {}}
        for flag_type, expected in ex.get("expected", {}).items():
            pred = predicted.get(flag_type, "not_assessed")
            pairs_by_flag.setdefault(flag_type, []).append((expected, pred))
            row["flags"][flag_type] = {"expected": expected, "predicted": pred}
        records.append(row)

    config = "rules+llm" if use_llm else "rules-only"
    return build_report(config, pairs_by_flag), records


def _fmt(v: float | None) -> str:
    return "  n/a" if v is None else f"{v:0.2f}"


def _print_report(report) -> None:
    print(f"\n=== configuration: {report.config} ===")
    header = f"{'flag':<28}{'P':>6}{'R':>6}{'F1':>6}{'cov':>6}   (tp/fp/fn/tn, n/a)"
    print(header)
    print("-" * len(header))
    for flag, m in sorted(report.per_flag.items()):
        c = m.confusion
        print(
            f"{flag:<28}{_fmt(m.precision):>6}{_fmt(m.recall):>6}{_fmt(m.f1):>6}"
            f"{_fmt(m.coverage):>6}   ({c.tp}/{c.fp}/{c.fn}/{c.tn}, {c.not_assessed})"
        )
    o = report.overall
    print("-" * len(header))
    print(
        f"{'OVERALL (micro)':<28}{_fmt(o.precision):>6}{_fmt(o.recall):>6}{_fmt(o.f1):>6}"
        f"{_fmt(o.coverage):>6}   ({o.confusion.tp}/{o.confusion.fp}/{o.confusion.fn}/"
        f"{o.confusion.tn}, {o.confusion.not_assessed})"
    )


def _report_to_dict(report) -> dict:
    def m2d(m):
        c = m.confusion
        return {
            "precision": m.precision, "recall": m.recall, "f1": m.f1, "coverage": m.coverage,
            "tp": c.tp, "fp": c.fp, "fn": c.fn, "tn": c.tn, "not_assessed": c.not_assessed,
        }

    return {
        "config": report.config,
        "per_flag": {k: m2d(v) for k, v in report.per_flag.items()},
        "overall": m2d(report.overall),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Aegis flag-level evaluation.")
    parser.add_argument("--json", type=str, default="", help="write full results to this JSON path")
    parser.add_argument(
        "--config", choices=["rules-only", "rules+llm", "both"], default="both",
        help="which configuration(s) to run (default: both)",
    )
    args = parser.parse_args()

    configs = (
        [False] if args.config == "rules-only"
        else [True] if args.config == "rules+llm"
        else [False, True]
    )

    print(
        "Note: caption_content_mismatch is scored at the RULE level using controlled "
        "scene descriptions (inject_scene); the real captioner is not yet integrated."
    )
    out = {"reports": [], "examples": None}
    for use_llm in configs:
        report, records = evaluate(use_llm=use_llm)
        _print_report(report)
        out["reports"].append(_report_to_dict(report))
        out["examples"] = records

    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
