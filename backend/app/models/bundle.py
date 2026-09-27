"""The Evidence Bundle contract — the spine of the whole system.

Every modality adapter (image, video) emits an :class:`EvidenceBundle`; the fusion core
consumes it and emits a :class:`Scorecard` of explained :class:`Flag` s. Building this
contract before any extractor is a deliberate decision: everything downstream
depends on the bundle shape, so it is defined and stable first.

Design rule reflected here: a Flag carries a plain-language explanation and a
"what to check" prompt. There is no trust score or verdict field anywhere — by design.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator

# Version of the Evidence Bundle / Scorecard contract. The bundle is the one structure the whole
# system pivots on. Stamping a version lets a reader tell which shape it holds, and a bundle or
# scorecard with any other version is refused when it is read (see ``read_bundle``). Bump on any
# breaking change. 2.1: the never-produced AI-generation hint was removed from the flag types and
# the bundle.
SCHEMA_VERSION = "2.1"


class SchemaVersionError(ValueError):
    """A bundle or scorecard was written with a different version of the contract."""


def _check_version(value: str, what: str) -> str:
    if value != SCHEMA_VERSION:
        raise SchemaVersionError(
            f"This {what} has schema version {value!r}, but this version of Aegis reads version "
            f"{SCHEMA_VERSION!r} only. Create it again with the current version."
        )
    return value


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
    """The catalogue of flags Aegis can raise."""

    AUDIO_VISUAL_MISMATCH = "audio_visual_mismatch"
    CAPTION_CONTENT_MISMATCH = "caption_content_mismatch"
    RECYCLED_CONTEXT = "recycled_context"
    EMOTIONAL_FRAMING = "emotional_framing"


class FlagStatus(str, Enum):
    """The outcome of *attempting* a given check.

    The crucial distinction is between FIRED, CLEAR, and NOT_ASSESSED. A check that
    silently does not fire because its input was missing (no on-screen text, blurry image,
    LLM offloaded/unavailable) must not look like a check that ran and found consistency —
    that would be a false sense of safety, the exact harm 'explain, don't verdict' exists to
    avoid. So fusion emits a Flag for every check it considered, carrying its
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


class WebMatch(BaseModel):
    """A reverse-image-search hit — where else this image has appeared.

    These come from the offline image history index; the shape matches a live reverse-image
    API result, so a live backend can drop in later.
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
    keypoint_inliers: int | None = Field(
        default=None,
        description="Number of image keypoints that line up geometrically with the matched "
        "entry, when the match came from keypoint matching rather than the hash.",
    )
    frame_timestamp: float | None = Field(
        default=None, description="Seconds into the video of the keyframe that matched; None for an image."
    )
    found_by: str | None = Field(
        default=None, description="'index' for the local image history index, 'web' for the live web search."
    )
    match_kind: str | None = Field(
        default=None, description="For a web match: 'full' or 'partial' copy of the image on the page."
    )
    date_source: str | None = Field(
        default=None,
        description="Where a web match's date came from: 'htmldate' (the page's own metadata), 'wayback' "
        "(the Wayback Machine's first capture), or None when the page could not be dated.",
    )


class CaptionMatch(BaseModel):
    """How closely the caption matches the image in meaning (cosine similarities, -1 to 1)."""

    image_similarity: float | None = Field(
        default=None, description="CLIP similarity between the image itself and the caption."
    )
    text_similarity: float | None = Field(
        default=None,
        description="Similarity between the caption and the scene description plus on-screen text.",
    )
    image_model: str | None = Field(default=None, description="CLIP model behind image_similarity.")
    text_model: str | None = Field(default=None, description="Method behind text_similarity.")
    caption_truncated: bool = Field(
        default=False,
        description="True when the caption was longer than CLIP's 77-token limit and was cut to fit.",
    )
    detail: str = Field(default="", description="Why a similarity is missing, when one is.")


class Keyframe(BaseModel):
    """One frame taken from a video, with what was read from it."""

    timestamp: float = Field(description="Seconds into the video.")
    phash: str | None = Field(default=None, description="64-bit pHash of the frame, as 16 hex characters.")
    on_screen_text: list[str] = Field(default_factory=list, description="Text OCR read in the frame.")
    caption_similarity: float | None = Field(
        default=None, description="CLIP similarity between this frame and the caption."
    )
    thumbnail: str | None = Field(default=None, description="Small JPEG of the frame, as a data URL.")


class TranscriptSegment(BaseModel):
    """One stretch of speech, as the speech recogniser segmented it."""

    start: float = Field(description="Seconds into the video.")
    end: float = Field(description="Seconds into the video.")
    text: str
    frame_timestamps: list[float] = Field(
        default_factory=list, description="Frames the segment's words were compared with."
    )
    picture_similarity: float | None = Field(
        default=None, description="Highest CLIP similarity between the segment's text and those frames."
    )
    best_frame: float | None = Field(default=None, description="The frame that scored highest.")


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
    duration_s: float | None = Field(default=None, description="Length of the video; None for an image.")
    has_audio: bool | None = Field(default=None, description="Whether the video has an audio track.")


class EvidenceBundle(BaseModel):
    """Normalised evidence emitted by every adapter and consumed by the fusion core.

    ``caption`` is the user-supplied caption for the
    image path; ``transcript`` is the spoken-audio transcript for the video path.
    """

    schema_version: str = Field(
        default=SCHEMA_VERSION,
        description="Contract version that produced this bundle, so a reader can tell which "
        "shape it holds.",
    )
    scene_descriptions: list[SceneDescription] = Field(default_factory=list)
    on_screen_text: list[str] = Field(default_factory=list)
    caption: str | None = Field(
        default=None, description="User-supplied caption (image path)."
    )
    transcript: str | None = Field(
        default=None, description="Whisper transcript (video path)."
    )
    transcript_segments: list[TranscriptSegment] = Field(
        default_factory=list, description="The transcript with start and end times (video path)."
    )
    keyframes: list[Keyframe] = Field(default_factory=list, description="Frames taken from the video.")
    web_matches: list[WebMatch] = Field(default_factory=list)
    caption_match: CaptionMatch | None = None
    extractor_status: dict[str, FlagStatus] = Field(
        default_factory=dict,
        description=(
            "Per-extractor outcome (e.g. {'ocr': 'clear'}). Lets fusion tell an empty field "
            "that was CLEARED from one that could not be assessed, so absence is never read "
            "as consistency."
        ),
    )
    extractor_detail: dict[str, str] = Field(
        default_factory=dict,
        description="Why an extractor produced nothing, when it did not (e.g. 'The video has no audio track.').",
    )
    meta: Meta

    @field_validator("schema_version")
    @classmethod
    def _same_version(cls, value: str) -> str:
        return _check_version(value, "evidence bundle")


# ------------------------------------------------------------------- scorecard side


class Flag(BaseModel):
    """A single check's result. Note: no trust score — explain, don't verdict.

    A Flag is emitted for every check fusion *considered*, not only those that fired; its
    ``status`` says whether it FIRED, came back CLEAR, or could NOT_ASSESSED. The
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
    timestamps: list[float] = Field(
        default_factory=list, description="Moments in the video, in seconds, that the flag cites."
    )


class KeyframeView(BaseModel):
    """A keyframe as the interface shows it."""

    timestamp: float
    thumbnail: str | None = None


class Scorecard(BaseModel):
    """The system's output: the result of every check considered, never a verdict.

    ``flags`` includes CLEAR and NOT_ASSESSED entries, not only those that fired,
    so the UI can show honestly what was checked, what was clear, and what could not be
    assessed.
    """

    schema_version: str = Field(
        default=SCHEMA_VERSION,
        description="Contract version that produced this scorecard, so a reader can tell "
        "which shape it holds.",
    )
    flags: list[Flag] = Field(default_factory=list)
    summary: str | None = Field(
        default=None, description="Optional neutral, non-verdict overview of the findings."
    )
    modality: Modality
    source_ref: str | None = None
    duration_s: float | None = None
    keyframes: list[KeyframeView] = Field(default_factory=list)

    @field_validator("schema_version")
    @classmethod
    def _same_version(cls, value: str) -> str:
        return _check_version(value, "scorecard")


def _read(model, raw: str | bytes | dict, what: str):
    """Validate stored JSON as ``model``, refusing it first if its version is missing or different."""
    import json

    data = json.loads(raw) if isinstance(raw, (str, bytes)) else raw
    if not isinstance(data, dict):
        raise SchemaVersionError(f"This {what} is not a JSON object.")
    if "schema_version" not in data:
        raise SchemaVersionError(
            f"This {what} carries no schema version, so its shape is unknown; this version of Aegis "
            f"reads version {SCHEMA_VERSION!r} only."
        )
    _check_version(data["schema_version"], what)
    return model.model_validate(data)


def read_bundle(raw: str | bytes | dict) -> EvidenceBundle:
    """Read a stored evidence bundle. Raises SchemaVersionError for a missing or different version."""
    return _read(EvidenceBundle, raw, "evidence bundle")


def read_scorecard(raw: str | bytes | dict) -> Scorecard:
    """Read a stored scorecard. Raises SchemaVersionError for a missing or different version."""
    return _read(Scorecard, raw, "scorecard")
