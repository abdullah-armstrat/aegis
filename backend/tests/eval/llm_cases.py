"""Shared case loading for the LLM-specific evaluation scripts (SSOT §5.3).

The three LLM metrics — output validity, self-consistency, latency — all run over the same
caption↔scene pairs the rules-vs-LLM comparison uses: the labelled examples that carry an
``inject_scene`` field, so the scene text is controlled rather than whatever BLIP produced.
Keeping the loader here (rather than copied into three scripts) means all four measurements
are demonstrably over the same inputs.

Pure data access — no model calls, no I/O beyond reading the manifest.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.models import EvidenceBundle, FlagStatus, Meta, Modality, SceneDescription

_EXAMPLES_PATH = Path(__file__).resolve().parent / "test_set" / "examples.json"

FLAG = "caption_content_mismatch"


def load_caption_scene_cases() -> list[dict]:
    """The labelled caption↔scene cases (those with a controlled ``inject_scene``)."""
    examples = json.loads(_EXAMPLES_PATH.read_text(encoding="utf-8"))["examples"]
    cases = [e for e in examples if FLAG in e.get("expected", {}) and e.get("inject_scene")]
    assert cases, f"GUARD FAILED: no {FLAG} examples with inject_scene in {_EXAMPLES_PATH}"
    return cases


def select(ids: list[str]) -> list[dict]:
    """The subset of caption↔scene cases with the given ids, in the order requested."""
    by_id = {c["id"]: c for c in load_caption_scene_cases()}
    missing = [i for i in ids if i not in by_id]
    assert not missing, f"GUARD FAILED: unknown case id(s) {missing}"
    return [by_id[i] for i in ids]


def reasoner_inputs(case: dict) -> tuple[str, list[str], list[str]]:
    """The exact (caption, scene_descriptions, on_screen_text) the reasoner would receive."""
    return case["caption"], list(case["inject_scene"]), []


def bundle_for(case: dict) -> EvidenceBundle:
    """An EvidenceBundle equivalent to what the pipeline builds for this case."""
    return EvidenceBundle(
        scene_descriptions=[SceneDescription(text=t) for t in case["inject_scene"]],
        caption=case["caption"],
        extractor_status={"captioner": FlagStatus.FIRED},
        meta=Meta(modality=Modality.IMAGE, source_ref=case["id"]),
    )
