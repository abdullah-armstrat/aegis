"""Loads the caption-vs-scene cases used by the LLM evaluation scripts.

These are the labelled examples with an ``inject_scene`` field, so the scene text is fixed
rather than whatever BLIP gives. Keeping the loader in one place means the validity,
consistency, latency and rules-vs-LLM scripts all use the same inputs.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.models import EvidenceBundle, FlagStatus, Meta, Modality, SceneDescription

_EXAMPLES_PATH = Path(__file__).resolve().parent / "test_set" / "examples.json"

FLAG = "caption_content_mismatch"


def load_caption_scene_cases() -> list[dict]:
    """Return the labelled caption-vs-scene cases that have an ``inject_scene``."""
    examples = json.loads(_EXAMPLES_PATH.read_text(encoding="utf-8"))["examples"]
    cases = [e for e in examples if FLAG in e.get("expected", {}) and e.get("inject_scene")]
    assert cases, f"GUARD FAILED: no {FLAG} examples with inject_scene in {_EXAMPLES_PATH}"
    return cases


def select(ids: list[str]) -> list[dict]:
    """Return the cases with the given ids, in the order given."""
    by_id = {c["id"]: c for c in load_caption_scene_cases()}
    missing = [i for i in ids if i not in by_id]
    assert not missing, f"GUARD FAILED: unknown case id(s) {missing}"
    return [by_id[i] for i in ids]


def reasoner_inputs(case: dict) -> tuple[str, list[str], list[str]]:
    """Return (caption, scene_descriptions, on_screen_text) as the reasoner would get them."""
    return case["caption"], list(case["inject_scene"]), []


def bundle_for(case: dict) -> EvidenceBundle:
    """Build the same EvidenceBundle the pipeline would build for this case."""
    return EvidenceBundle(
        scene_descriptions=[SceneDescription(text=t) for t in case["inject_scene"]],
        caption=case["caption"],
        extractor_status={"captioner": FlagStatus.FIRED},
        meta=Meta(modality=Modality.IMAGE, source_ref=case["id"]),
    )
