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

# Version of the Evidence Bundle / Scorecard contract (ADR-010). The bundle is the one
# structure the whole system pivots on, and it will grow when the video path lands
# (temporal/transcript fields). Stamping a version lets the eval harness and cached
# fixtures tell which shape they are dealing with later. Bump on any breaking change.
SCHEMA_VERSION = "1.0"


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


class FlagStatus(str, Enum):
    """The outcome of *attempting* a given check (ADR-009).

    The crucial distinction is between FIRED, CLEAR, and NOT_ASSESSED. A check that
    silently does not fire because its input was missing (no on-screen text, blurry image,
    LLM offloaded/unavailable) must not look like a check that ran and found consistency —
    that would be a false sense of safety, the exact harm 'explain, don't verdict' exists to
    avoid (ADR-002). So fusion emits a Flag for every check it considered, carrying its
    status, rather than only appending Flags that fired.
    """

    FIRED = "fired"            # the check ran and the finding is present
    CLEAR = "clear"            # the check ran and found nothing to flag
    NOT_ASSESSED = "not_assessed"  # the check could not run (missing input / extractor down)


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
    hash_distance: int | None = Field(
        default=None,
        description="pHash Hamming distance (of 64 bits) between the upload and the matched "
        "image-history entry. None when the match did not come from a hash lookup.",
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
    posted_date: str | None = Field(
        default=None,
        description="ISO date (YYYY-MM-DD) the post claims to have been published, if the user "
        "gave one. Recycled context compares prior appearances against it.",
    )
    image_phash: str | None = Field(
        default=None, description="64-bit pHash of the analysed image, as 16 hex characters."
    )


class EvidenceBundle(BaseModel):
    """Normalised evidence emitted by every adapter and consumed by the fusion core.

    The fields mirror SSOT §3.2 exactly. ``caption`` is the user-supplied caption for the
    image path; ``transcript`` is the spoken-audio transcript for the video path.
    """

    schema_version: str = Field(
        default=SCHEMA_VERSION, description="Contract version that produced this bundle (ADR-010)."
    )
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
    extractor_status: dict[str, FlagStatus] = Field(
        default_factory=dict,
        description=(
            "Per-extractor outcome (e.g. {'ocr': 'clear'}). Lets fusion tell an empty field "
            "that was CLEARED from one that could not be assessed, so absence is never read "
            "as consistency (ADR-009)."
        ),
    )
    meta: Meta


# ------------------------------------------------------------------- scorecard side


class Flag(BaseModel):
    """A single check's result. Note: no trust score — explain, don't verdict (ADR-002).

    A Flag is emitted for every check fusion *considered*, not only those that fired; its
    ``status`` says whether it FIRED, came back CLEAR, or could NOT_ASSESSED (ADR-009). The
    explanatory fields are required when ``status`` is FIRED; for CLEAR / NOT_ASSESSED they
    carry a short note on what was (or could not be) checked.
    """

    type: FlagType
    status: FlagStatus = Field(
        default=FlagStatus.FIRED, description="Outcome of the check: fired / clear / not_assessed."
    )
    severity: Severity
    evidence: str = Field(description="The concrete bundle fields that triggered this flag.")
    plain_explanation: str = Field(description="Plain-language statement of the finding.")
    what_to_check: str = Field(description="A prompt teaching the user what to verify themselves.")
    source: str = Field(
        default="rules", description="Which fusion layer raised it: 'rules' or 'llm'."
    )


class Scorecard(BaseModel):
    """The system's output: the result of every check considered, never a verdict.

    ``flags`` includes CLEAR and NOT_ASSESSED entries, not only those that fired (ADR-009),
    so the UI can show honestly what was checked, what was clear, and what could not be
    assessed.
    """

    schema_version: str = Field(
        default=SCHEMA_VERSION, description="Contract version that produced this scorecard (ADR-010)."
    )
    flags: list[Flag] = Field(default_factory=list)
    summary: str | None = Field(
        default=None, description="Optional neutral, non-verdict overview of the findings."
    )
    modality: Modality
    source_ref: str | None = None
