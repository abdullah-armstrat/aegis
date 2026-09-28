"""Tests for the EvidenceBundle and Scorecard models.

They fix the data shape the rest of the app relies on, and check there is no trust score or
verdict field, since the app explains flags rather than giving a verdict.
"""

import json

import pytest
from pydantic import ValidationError

from app.models import (
    SCHEMA_VERSION,
    EvidenceBundle,
    Flag,
    FlagStatus,
    FlagType,
    Meta,
    Modality,
    SchemaVersionError,
    Scorecard,
    SceneDescription,
    Severity,
    WebMatch,
    read_bundle,
    read_scorecard,
)


def test_minimal_image_bundle():
    bundle = EvidenceBundle(meta=Meta(modality=Modality.IMAGE, source_ref="ex1.jpg"))
    assert bundle.meta.modality == Modality.IMAGE
    assert bundle.scene_descriptions == []
    assert bundle.transcript is None
    # Versioned so older stored shapes can be told apart.
    assert bundle.schema_version == SCHEMA_VERSION


def test_full_image_bundle_roundtrip():
    bundle = EvidenceBundle(
        scene_descriptions=[SceneDescription(text="a flooded street", confidence=0.9)],
        on_screen_text=["BREAKING"],
        caption="Floods hit the city today",
        web_matches=[
            WebMatch(
                url="https://example.com/2019-article",
                published_date="2019-03-04",
                context="Unrelated 2019 flood coverage",
            )
        ],
        meta=Meta(modality=Modality.IMAGE, source_ref="ex2.jpg"),
    )
    # Survives a JSON round trip unchanged.
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
    # Status defaults to FIRED.
    assert flag.status == FlagStatus.FIRED
    # No trust or confidence field on a flag.
    assert "trust" not in Flag.model_fields


def test_flag_distinguishes_clear_from_not_assessed():
    """'Checked and clear' and 'could not check' are different statuses."""
    clear = Flag(
        type=FlagType.CAPTION_CONTENT_MISMATCH,
        status=FlagStatus.CLEAR,
        severity=Severity.INFO,
        evidence="caption and scene caption overlap on key terms",
        plain_explanation="The caption matches what the image shows.",
        what_to_check="No action needed for this check.",
    )
    not_assessed = Flag(
        type=FlagType.CAPTION_CONTENT_MISMATCH,
        status=FlagStatus.NOT_ASSESSED,
        severity=Severity.INFO,
        evidence="no on-screen text could be extracted",
        plain_explanation="This check could not be run because no text was found in the image.",
        what_to_check="Try a clearer image so on-screen text can be read.",
    )
    assert clear.status == FlagStatus.CLEAR
    assert not_assessed.status == FlagStatus.NOT_ASSESSED
    assert clear.status != not_assessed.status


def test_scorecard_has_no_verdict_field():
    card = Scorecard(modality=Modality.IMAGE, flags=[])
    assert card.flags == []
    assert card.schema_version == SCHEMA_VERSION
    assert "verdict" not in Scorecard.model_fields


def test_no_flag_type_or_field_is_declared_without_being_produced():
    """The generated-image hint was never produced by any check, so it was removed."""
    assert "ai_generation_hint" not in {t.value for t in FlagType}
    assert "ai_gen_hint" not in EvidenceBundle.model_fields


def test_a_stored_bundle_or_scorecard_of_the_current_version_is_read():
    bundle = EvidenceBundle(caption="x", meta=Meta(modality=Modality.IMAGE))
    assert read_bundle(bundle.model_dump_json()) == bundle
    card = Scorecard(modality=Modality.IMAGE)
    assert read_scorecard(card.model_dump()) == card


@pytest.mark.parametrize("reader, model, what", [
    ("bundle", EvidenceBundle, "evidence bundle"),
    ("scorecard", Scorecard, "scorecard"),
])
def test_a_different_schema_version_is_refused_with_a_clear_error(reader, model, what):
    read = read_bundle if reader == "bundle" else read_scorecard
    stored = (EvidenceBundle(meta=Meta(modality=Modality.IMAGE)) if reader == "bundle"
              else Scorecard(modality=Modality.IMAGE)).model_dump()
    stored["schema_version"] = "1.0"
    with pytest.raises(SchemaVersionError) as refused:
        read(json.dumps(stored))
    message = str(refused.value)
    assert f"This {what} has schema version '1.0'" in message
    assert f"reads version '{SCHEMA_VERSION}' only" in message
    # Validating the model directly refuses it for the same reason.
    with pytest.raises(ValidationError, match="has schema version '1.0'"):
        model.model_validate(stored)


def test_a_stored_bundle_without_a_version_is_refused():
    stored = EvidenceBundle(meta=Meta(modality=Modality.IMAGE)).model_dump()
    del stored["schema_version"]
    with pytest.raises(SchemaVersionError, match="carries no schema version"):
        read_bundle(stored)
