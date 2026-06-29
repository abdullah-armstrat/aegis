"""Tests for the deterministic fusion rules (ADR-004).

Each rule is tested across all three honest outcomes where applicable (ADR-009). Bundles are
constructed directly (no extractors run), so these are fast, deterministic, and pin the rule
logic precisely — exactly the reproducibility the rules layer exists to provide.
"""

from app.fusion.rules import (
    CAPTION_SCENE_OVERLAP_THRESHOLD,
    EMOTIONAL_INTENSITY_THRESHOLD,
    caption_scene_mismatch_rule,
    emotional_framing_rule,
    recycled_context_rule,
    run_rules,
)
from app.models import (
    EvidenceBundle,
    FlagStatus,
    FlagType,
    Meta,
    Modality,
    SceneDescription,
    Sentiment,
    WebMatch,
)


def _bundle(**kwargs) -> EvidenceBundle:
    kwargs.setdefault("meta", Meta(modality=Modality.IMAGE, source_ref="t.jpg"))
    return EvidenceBundle(**kwargs)


# --- emotional framing ---

def test_emotional_framing_fires_on_high_intensity():
    b = _bundle(
        sentiment=Sentiment(label="negative", score=0.99, source="caption"),
        extractor_status={"sentiment": FlagStatus.FIRED},
    )
    flag = emotional_framing_rule(b)
    assert flag.type == FlagType.EMOTIONAL_FRAMING
    assert flag.status == FlagStatus.FIRED
    assert flag.what_to_check


def test_emotional_framing_clear_on_low_intensity():
    b = _bundle(
        sentiment=Sentiment(label="positive", score=0.55, source="caption"),
        extractor_status={"sentiment": FlagStatus.FIRED},
    )
    assert emotional_framing_rule(b).status == FlagStatus.CLEAR


def test_emotional_framing_does_not_fire_on_confident_positive():
    """Regression: a confidently-POSITIVE caption must NOT fire (direction matters, not just
    confidence), and must never be described as 'strongly negative'. This is the bug found in
    demo testing where 'a quiet afternoon at the park' (positive 1.00) fired the flag."""
    b = _bundle(
        sentiment=Sentiment(label="positive", score=1.00, source="caption"),
        extractor_status={"sentiment": FlagStatus.FIRED},
    )
    flag = emotional_framing_rule(b)
    assert flag.status == FlagStatus.CLEAR
    assert "negative" not in flag.plain_explanation or "not read as strongly negative" in flag.plain_explanation


def test_emotional_framing_not_assessed_without_sentiment():
    b = _bundle(extractor_status={"sentiment": FlagStatus.NOT_ASSESSED})
    assert emotional_framing_rule(b).status == FlagStatus.NOT_ASSESSED


# --- recycled context ---

def test_recycled_context_fires_with_matches():
    b = _bundle(
        web_matches=[WebMatch(url="https://e.com/2019", published_date="2019-03-04")],
        extractor_status={"reverse_image": FlagStatus.FIRED},
    )
    flag = recycled_context_rule(b)
    assert flag.status == FlagStatus.FIRED
    assert "2019-03-04" in flag.evidence  # earliest date surfaced


def test_recycled_context_clear_when_searched_empty():
    b = _bundle(web_matches=[], extractor_status={"reverse_image": FlagStatus.CLEAR})
    assert recycled_context_rule(b).status == FlagStatus.CLEAR


def test_recycled_context_not_assessed_when_unknown():
    b = _bundle(web_matches=[], extractor_status={"reverse_image": FlagStatus.NOT_ASSESSED})
    assert recycled_context_rule(b).status == FlagStatus.NOT_ASSESSED


def test_recycled_context_not_assessed_when_status_missing():
    """Regression: a bundle where the reverse-image extractor never ran (no status key) and
    has no matches must be NOT_ASSESSED, never CLEAR — empty must not read as 'searched, none
    recycled' (ADR-009). This bug was caught by the scorecard 'nothing fires' test."""
    b = _bundle(web_matches=[], extractor_status={})
    assert recycled_context_rule(b).status == FlagStatus.NOT_ASSESSED


# --- caption <-> scene mismatch ---

def test_caption_scene_fires_on_low_overlap():
    b = _bundle(
        caption="Massive flood devastates the city centre this morning",
        scene_descriptions=[SceneDescription(text="a dry empty street on a sunny day")],
    )
    flag = caption_scene_mismatch_rule(b)
    assert flag.type == FlagType.CAPTION_CONTENT_MISMATCH
    assert flag.status == FlagStatus.FIRED


def test_caption_scene_clear_on_high_overlap():
    b = _bundle(
        caption="a dry empty street on a sunny day with parked cars",
        scene_descriptions=[SceneDescription(text="a dry empty street on a sunny day")],
    )
    assert caption_scene_mismatch_rule(b).status == FlagStatus.CLEAR


def test_caption_scene_not_assessed_without_scene():
    """Before the captioner is wired (no scene_descriptions) this is NOT_ASSESSED, not a
    silent pass — the honest behaviour (ADR-009)."""
    b = _bundle(caption="anything")
    assert caption_scene_mismatch_rule(b).status == FlagStatus.NOT_ASSESSED


# --- the worked mismatch example end-to-end through the rules ---

def test_run_rules_on_built_mismatch_example():
    """The canonical demo case: recycled flood image + wrong fresh caption. Rules alone should
    raise the caption↔scene mismatch AND the recycled-context flag (ADR-004 baseline)."""
    b = _bundle(
        caption="URGENT: massive flood hitting the city right now, share immediately!",
        scene_descriptions=[SceneDescription(text="a calm dry residential street, parked cars")],
        sentiment=Sentiment(label="negative", score=0.97, source="caption"),
        web_matches=[
            WebMatch(url="https://news.example.com/2019/flood", published_date="2019-03-04")
        ],
        extractor_status={
            "sentiment": FlagStatus.FIRED,
            "reverse_image": FlagStatus.FIRED,
        },
    )
    flags = run_rules(b)
    fired = {f.type for f in flags if f.status == FlagStatus.FIRED}
    assert FlagType.CAPTION_CONTENT_MISMATCH in fired
    assert FlagType.RECYCLED_CONTEXT in fired
    assert FlagType.EMOTIONAL_FRAMING in fired
    # Exactly one flag per rule, always.
    assert len(flags) == 3
