"""Tests for building the scorecard, in rules-only and rules+LLM mode.

The LLM is stubbed. The LLM only adds a flag on top of the three rule flags, and there is
no verdict field.
"""

from app.config import get_settings
from app.fusion import scorecard as sc
from app.fusion.llm_reasoner import ReasonerVerdict
from app.models import (
    EvidenceBundle,
    FlagStatus,
    Meta,
    Modality,
    SceneDescription,
    WebMatch,
)


def _mismatch_bundle() -> EvidenceBundle:
    return EvidenceBundle(
        caption="URGENT massive flood hitting the city now, share!",
        scene_descriptions=[SceneDescription(text="a calm dry residential street, parked cars")],
        web_matches=[WebMatch(url="https://e.com/2019/flood", published_date="2019-03-04")],
        extractor_status={"reverse_image": FlagStatus.FIRED},
        meta=Meta(modality=Modality.IMAGE, source_ref="flood.jpg"),
    )


def test_rules_only_scorecard(monkeypatch):
    get_settings.cache_clear()
    monkeypatch.setenv("AEGIS_USE_LLM", "false")
    get_settings.cache_clear()

    card = sc.build_scorecard(_mismatch_bundle())

    # Just the three rule flags.
    assert len(card.flags) == 3
    assert all(f.source == "rules" for f in card.flags)
    assert card.modality == Modality.IMAGE
    assert card.summary  # a neutral summary is there
    assert "verdict" not in sc.Scorecard.model_fields
    get_settings.cache_clear()


def test_rules_plus_llm_is_additive(monkeypatch):
    """With use_llm=true the LLM adds one flag; the three rule flags remain."""
    get_settings.cache_clear()
    monkeypatch.setenv("AEGIS_USE_LLM", "true")
    get_settings.cache_clear()
    # Fake reasoner that says the caption and scene don't match.
    monkeypatch.setattr(
        sc,
        "reason_over_text",
        lambda caption, scene, ocr: ReasonerVerdict(
            same_subject=False, same_tone=False, explanation="Different things.", available=True
        ),
    )

    card = sc.build_scorecard(_mismatch_bundle())

    assert len(card.flags) == 4  # 3 rules + 1 LLM
    llm_flags = [f for f in card.flags if f.source == "llm"]
    assert len(llm_flags) == 1
    assert llm_flags[0].status == FlagStatus.FIRED
    get_settings.cache_clear()


def test_llm_unavailable_adds_not_assessed_not_silence(monkeypatch):
    """If the LLM is on but unavailable, its flag is shown as NOT_ASSESSED, not dropped."""
    get_settings.cache_clear()
    monkeypatch.setenv("AEGIS_USE_LLM", "true")
    get_settings.cache_clear()
    monkeypatch.setattr(
        sc,
        "reason_over_text",
        lambda caption, scene, ocr: ReasonerVerdict(None, None, "LLM unavailable: down", available=False),
    )

    card = sc.build_scorecard(_mismatch_bundle())
    llm_flags = [f for f in card.flags if f.source == "llm"]
    assert len(llm_flags) == 1
    assert llm_flags[0].status == FlagStatus.NOT_ASSESSED
    get_settings.cache_clear()


def test_summary_is_neutral_when_nothing_fires(monkeypatch):
    get_settings.cache_clear()
    monkeypatch.setenv("AEGIS_USE_LLM", "false")
    get_settings.cache_clear()
    # A bundle where every rule is NOT_ASSESSED (no inputs).
    bare = EvidenceBundle(meta=Meta(modality=Modality.IMAGE, source_ref="x"))
    card = sc.build_scorecard(bare)
    assert card.summary
    assert all(f.status == FlagStatus.NOT_ASSESSED for f in card.flags)
    get_settings.cache_clear()
