"""Compare the rules and the LLM on the caption-scene check, over the labelled examples.

run_eval.py scores only the rules verdict (the LLM flag only adds to it), so it cannot show what
the LLM adds. This script reads both flags the pipeline already makes for caption_content_mismatch
(rules source and llm source) and scores each against the same labels. It changes no matching logic.

The question: the rules over-fire on synonym pairs (word overlap cannot see "automobile" = "car"),
giving precision 0.50 on the 19-case set. Does the LLM clear those and still catch the real
mismatches?

phi3:mini is not deterministic, so counts can change between runs. Every value is written to JSON
and reported as from a run on a given date, not pinned in a test (the rules numbers stay pinned in
test_eval_harness.py). It also records whether the LLM was reachable, so an "LLM unavailable" run
shows up instead of being scored as not_assessed.

Run:  python backend/scripts/compare_rules_vs_llm.py
"""

import json
import os
import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

# LLM on for this whole script.
os.environ["AEGIS_USE_LLM"] = "true"
os.environ["AEGIS_USE_CAPTIONER"] = "false"  # we inject scenes; don't load BLIP here

from app.config import get_settings  # noqa: E402

get_settings.cache_clear()

from app.fusion.scorecard import build_scorecard  # noqa: E402
from app.models import (  # noqa: E402
    EvidenceBundle,
    FlagStatus,
    Meta,
    Modality,
    SceneDescription,
)
from tests.eval.metrics import Confusion, metrics_for, update_confusion  # noqa: E402

FLAG = "caption_content_mismatch"
_STATUS_TO_PRED = {
    FlagStatus.FIRED: "fired",
    FlagStatus.CLEAR: "clear",
    FlagStatus.NOT_ASSESSED: "not_assessed",
}

EXAMPLES_PATH = _BACKEND_DIR / "tests" / "eval" / "test_set" / "examples.json"
assert EXAMPLES_PATH.exists(), f"GUARD FAILED: missing {EXAMPLES_PATH}"

examples = json.loads(EXAMPLES_PATH.read_text(encoding="utf-8"))["examples"]
cases = [e for e in examples if FLAG in e.get("expected", {}) and e.get("inject_scene")]
assert cases, "GUARD FAILED: no caption_content_mismatch examples with inject_scene found"


def _verdicts(example: dict) -> tuple[str, str, str]:
    """Return (rules_pred, llm_pred, llm_explanation) for one example."""
    bundle = EvidenceBundle(
        scene_descriptions=[SceneDescription(text=t) for t in example["inject_scene"]],
        caption=example["caption"],
        extractor_status={"captioner": FlagStatus.FIRED},
        meta=Meta(modality=Modality.IMAGE, source_ref=example["id"]),
    )
    card = build_scorecard(bundle)
    rules_pred = llm_pred = "missing"
    llm_expl = ""
    for f in card.flags:
        if f.type.value != FLAG:
            continue
        if f.source == "rules":
            rules_pred = _STATUS_TO_PRED[f.status]
        elif f.source == "llm":
            llm_pred = _STATUS_TO_PRED[f.status]
            llm_expl = f.evidence
    return rules_pred, llm_pred, llm_expl


rules_conf = Confusion()
llm_conf = Confusion()
rows = []
llm_unreachable = 0

for ex in cases:
    expected = ex["expected"][FLAG]  # "fire" or "clear"
    rules_pred, llm_pred, llm_expl = _verdicts(ex)
    if llm_pred == "not_assessed":
        llm_unreachable += 1
    update_confusion(rules_conf, expected, rules_pred)
    update_confusion(llm_conf, expected, llm_pred)
    rows.append({
        "id": ex["id"],
        "caption": ex["caption"],
        "scene": ex["inject_scene"],
        "expected": expected,
        "rules_pred": rules_pred,
        "llm_pred": llm_pred,
        "llm_explanation": llm_expl,
        "llm_recovers": (rules_pred == "fired" and expected == "clear" and llm_pred == "clear"),
    })

rules_m = metrics_for(FLAG, rules_conf)
llm_m = metrics_for(FLAG, llm_conf)


def _m(m):
    c = m.confusion
    return {
        "precision": m.precision, "recall": m.recall, "f1": m.f1,
        "tp": c.tp, "fp": c.fp, "fn": c.fn, "tn": c.tn, "not_assessed": c.not_assessed,
    }


out = {
    "flag": FLAG,
    "n_cases": len(cases),
    "llm_unreachable_count": llm_unreachable,
    "ollama_model": get_settings().ollama_model,
    "rules": _m(rules_m),
    "llm": _m(llm_m),
    "per_example": rows,
}
out_path = _BACKEND_DIR / "scripts" / "_rules_vs_llm.json"
out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
print(f"wrote {out_path}")
print(f"n_cases={len(cases)} llm_unreachable={llm_unreachable}")
