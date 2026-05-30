"""Contract tests: the Evidence Bundle / Scorecard models validate as designed.

These lock the shape that everything downstream depends on (ADR-003) and assert the
'explain, don't verdict' rule (ADR-002) by construction — there is no trust-score field.
"""

from app.models import (
    EvidenceBundle,
    Flag,
    FlagType,
    Meta,
    Modality,
    Scorecard,
    SceneDescription,
    Sentiment,
    Severity,
    WebMatch,
)


def test_minimal_image_bundle():
    bundle = EvidenceBundle(meta=Meta(modality=Modality.IMAGE, source_ref="ex1.jpg"))
    assert bundle.meta.modality == Modality.IMAGE
    assert bundle.scene_descriptions == []
    assert bundle.transcript is None


def test_full_image_bundle_roundtrip():
    bundle = EvidenceBundle(
        scene_descriptions=[SceneDescription(text="a flooded street", confidence=0.9)],
        on_screen_text=["BREAKING"],
        caption="Floods hit the city today",
        sentiment=Sentiment(label="negative", score=0.8, source="caption"),
        web_matches=[
            WebMatch(
                url="https://example.com/2019-article",
                published_date="2019-03-04",
                context="Unrelated 2019 flood coverage",
            )
        ],
        meta=Meta(modality=Modality.IMAGE, source_ref="ex2.jpg"),
    )
    # Round-trips through JSON without loss.
    assert EvidenceBundle.model_validate_json(bundle.model_dump_json()) == bundle


def test_flag_has_explanation_and_what_to_check():
    flag = Flag(
        type=FlagType.RECYCLED_CONTEXT,
        severity=Severity.HIGH,
        evidence="web_matches[0] dated 2019",
        plain_explanation="This image also appears in a 2019 article in a different context.",
        what_to_check="Search for the image's earliest appearance before trusting the caption.",
    )
    assert flag.plain_explanation
    assert flag.what_to_check
    # 'Explain, don't verdict': the Flag model carries no trust/confidence verdict field.
    assert "trust" not in Flag.model_fields


def test_scorecard_has_no_verdict_field():
    card = Scorecard(modality=Modality.IMAGE, flags=[])
    assert card.flags == []
    assert "verdict" not in Scorecard.model_fields
