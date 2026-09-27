"""Pydantic data contracts shared across the pipeline."""

from app.models.bundle import (
    SCHEMA_VERSION,
    AiGenHint,
    CaptionMatch,
    EvidenceBundle,
    Flag,
    FlagStatus,
    FlagType,
    Keyframe,
    KeyframeView,
    Meta,
    Modality,
    Scorecard,
    SceneDescription,
    Severity,
    TranscriptSegment,
    WebMatch,
)

__all__ = [
    "SCHEMA_VERSION",
    "AiGenHint",
    "CaptionMatch",
    "EvidenceBundle",
    "Flag",
    "FlagStatus",
    "FlagType",
    "Keyframe",
    "KeyframeView",
    "Meta",
    "Modality",
    "Scorecard",
    "SceneDescription",
    "Severity",
    "TranscriptSegment",
    "WebMatch",
]
