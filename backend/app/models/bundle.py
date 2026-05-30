"""The Evidence Bundle contract — the spine of the whole system (SSOT §3.2).

Every modality adapter (image, video) emits an :class:`EvidenceBundle`; the fusion core
consumes it and emits a :class:`Scorecard` of explained :class:`Flag` s. Building this
contract before any extractor is a deliberate decision (ADR-003): everything downstream
depends on the bundle shape, so it is defined and stable first.

Design rule reflected here (ADR-002): a Flag carries a plain-language explanation and a
"what to check" prompt. There is no trust score or verdict field anywhere — by design.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- enums


class Modality(str, Enum):
    """Source modality that produced a bundle."""

    IMAGE = "image"
    VIDEO = "video"


class Severity(str, Enum):
    """How strongly a flag should be surfaced. Not a probability of 'fakeness'."""

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class FlagType(str, Enum):
    """The catalogue of flags Aegis can raise (SSOT §1.4)."""

    AUDIO_VISUAL_MISMATCH = "audio_visual_mismatch"
    CAPTION_CONTENT_MISMATCH = "caption_content_mismatch"
    RECYCLED_CONTEXT = "recycled_context"
    EMOTIONAL_FRAMING = "emotional_framing"
    AI_GENERATION_HINT = "ai_generation_hint"


# ------------------------------------------------------- evidence-bundle components


class SceneDescription(BaseModel):
    """A caption / object description for an image or a single video keyframe."""

    text: str
    confidence: float | None = None
    frame_timestamp: float | None = Field(
        default=None, description="Seconds into the video; None for a still image."
    )


class Sentiment(BaseModel):
    """Emotional-intensity reading over the available text (SSOT §3.2)."""

    label: str
    score: float = Field(description="Model confidence / intensity in [0, 1].")
    source: str = Field(description="Which text was scored, e.g. 'caption' or 'transcript'.")


class WebMatch(BaseModel):
    """A reverse-image-search hit — where else this image has appeared.

    For the Preliminary Report these come from a cached fixture (ADR-007); the shape is
    identical to a live API result so a real backend can drop in later.
    """

    url: str
    title: str | None = None
    published_date: str | None = Field(
        default=None, description="ISO date the match was published, if known."
    )
    context: str | None = Field(
        default=None, description="Short description of the matching page's context."
    )


class AiGenHint(BaseModel):
    """An optional, explicitly weak AI-generation signal (SSOT §1.4 — never conclusive)."""

    indicative: bool
    confidence: float = Field(description="Weak signal strength in [0, 1]; treat with caution.")


class Meta(BaseModel):
    """Provenance for the bundle."""

    modality: Modality
    frame_timestamps: list[float] = Field(default_factory=list)
    source_ref: str | None = Field(
        default=None, description="Filename / URL / id of the analysed item."
    )


class EvidenceBundle(BaseModel):
    """Normalised evidence emitted by every adapter and consumed by the fusion core.

    The fields mirror SSOT §3.2 exactly. ``caption`` is the user-supplied caption for the
    image path; ``transcript`` is the spoken-audio transcript for the video path.
    """

    scene_descriptions: list[SceneDescription] = Field(default_factory=list)
    on_screen_text: list[str] = Field(default_factory=list)
    caption: str | None = Field(
        default=None, description="User-supplied caption (image path)."
    )
    transcript: str | None = Field(
        default=None, description="Whisper transcript (video path)."
    )
    sentiment: Sentiment | None = None
    web_matches: list[WebMatch] = Field(default_factory=list)
    ai_gen_hint: AiGenHint | None = None
    meta: Meta


# ------------------------------------------------------------------- scorecard side


class Flag(BaseModel):
    """A single explained finding. Note: no trust score — explain, don't verdict (ADR-002)."""

    type: FlagType
    severity: Severity
    evidence: str = Field(description="The concrete bundle fields that triggered this flag.")
    plain_explanation: str = Field(description="Plain-language statement of the finding.")
    what_to_check: str = Field(description="A prompt teaching the user what to verify themselves.")
    source: str = Field(
        default="rules", description="Which fusion layer raised it: 'rules' or 'llm'."
    )


class Scorecard(BaseModel):
    """The system's output: a list of explained flags, never a verdict."""

    flags: list[Flag] = Field(default_factory=list)
    summary: str | None = Field(
        default=None, description="Optional neutral, non-verdict overview of the findings."
    )
    modality: Modality
    source_ref: str | None = None
