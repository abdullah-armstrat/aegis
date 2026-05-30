"""Tests for the scorecard assembler.

Cover the two evaluation configurations the project compares (SSOT §5.3): rules-only
(use_llm=false) and rules+LLM (use_llm=true). The LLM is stubbed so these stay fast and
deterministic — the reasoner has its own tests. Asserts the additive contract (ADR-004): the
LLM adds a flag, never replaces the rule flags, and the scorecard carries no verdict (ADR-002).
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
    Sentiment,
    WebMatch,
)


def _mismatch_bundle() -> EvidenceBundle:
    return EvidenceBundle(
        caption="URGENT massive flood hitting the city now, share!",
        scene_descriptions=[SceneDescription(text="a calm dry residential street, parked cars")],
        sentiment=Sentiment(label="negative", score=0.97, source="caption"),
        web_matches=[WebMatch(url="https://e.com/2019/flood", published_date="2019-03-04")],
        extractor_status={"sentiment": FlagStatus.FIRED, "reverse_image": FlagStatus.FIRED},
        meta=Meta(modality=Modality.IMAGE, source_ref="flood.jpg"),
    )


def test_rules_only_scorecard(monkeypatch):
    get_settings.cache_clear()
    monkeypatch.setenv("AEGIS_USE_LLM", "false")
    get_settings.cache_clear()

    card = sc.build_scorecard(_mismatch_bundle())

    # Exactly the three rule flags, all sourced from rules.
    assert len(card.flags) == 3
    assert all(f.source == "rules" for f in card.flags)
    assert card.modality == Modality.IMAGE
    assert card.summary  # neutral overview present
    assert "verdict" not in sc.Scorecard.model_fields
    get_settings.cache_clear()


def test_rules_plus_llm_is_additive(monkeypatch):
    """With use_llm=true the LLM adds one flag; the three rule flags remain (ADR-004)."""
    get_settings.cache_clear()
    monkeypatch.setenv("AEGIS_USE_LLM", "true")
    get_settings.cache_clear()
    # Stub the reasoner: an explicit mismatch judgement.
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
    """If the LLM is on but unavailable, its flag is NOT_ASSESSED — visible, not dropped."""
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
