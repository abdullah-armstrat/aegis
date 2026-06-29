"""Deterministic fusion rules — the reproducible heart of the analysis (ADR-004).

Each rule reads the Evidence Bundle and returns exactly one :class:`Flag` describing the
*outcome of attempting that check* — FIRED, CLEAR, or NOT_ASSESSED (ADR-009). Crucially, a
rule emits a flag even when it cannot run, so "couldn't check" is never silently dropped and
never mistaken for "checked and consistent". These rules are deterministic and explainable,
so they form the baseline the "rules only" vs "rules + LLM" evaluation compares against.

Thresholds live here as named constants; they are deliberately simple and tunable, and the
eval harness will inform their final values. No rule outputs a trust verdict (ADR-002) — only
a typed, explained flag with a "what to check" prompt.
"""

from __future__ import annotations

import re

from app.models import (
    EvidenceBundle,
    Flag,
    FlagStatus,
    FlagType,
    Severity,
)

# --- Tunable thresholds (eval harness will inform final values) ---
EMOTIONAL_INTENSITY_THRESHOLD = 0.90  # sentiment confidence above which framing is "high intensity"
CAPTION_SCENE_OVERLAP_THRESHOLD = 0.15  # min content-word overlap before caption↔scene is "consistent"

# Very small English stop-word list — enough to stop overlap being dominated by glue words.
_STOPWORDS = frozenset(
    """a an the this that these those is are was were be been being of to in on at for with
    and or but if then than so as it its from by into over under up down out about no not just
    we you they he she i them his her their our your my me""".split()
)


def _content_words(text: str) -> set[str]:
    """Lower-cased alphabetic content words (stop-words and short tokens removed)."""
    tokens = re.findall(r"[a-zA-Z]+", text.lower())
    return {t for t in tokens if len(t) > 2 and t not in _STOPWORDS}


def emotional_framing_rule(bundle: EvidenceBundle) -> Flag:
    """Flags a strongly negative tone in the caption.

    Keys solely on the sentiment model's polarity confidence (``bundle.sentiment.score`` from
    distilbert SST-2). That measures negative *polarity*, NOT manipulation specifically, so the
    user-facing wording is scoped honestly to "strongly negative tone" rather than claiming a
    manipulation technique (interim Option A; see ADR-014 for the planned marker-based fix).
    Firing logic and threshold are unchanged.
    """
    status = bundle.extractor_status.get("sentiment")
    if bundle.sentiment is None or status == FlagStatus.NOT_ASSESSED:
        return Flag(
            type=FlagType.EMOTIONAL_FRAMING,
            status=FlagStatus.NOT_ASSESSED,
            severity=Severity.INFO,
            evidence="No sentiment reading was available.",
            plain_explanation="The tone of the caption could not be assessed.",
            what_to_check="Consider for yourself whether the wording reads as strongly negative or one-sided.",
        )

    intensity = bundle.sentiment.score
    is_negative = bundle.sentiment.label.lower() == "negative"
    # Fire ONLY on a confidently-NEGATIVE reading. High confidence in a *positive* label is not
    # a negative tone and must never be described as one (direction matters, not just intensity).
    if is_negative and intensity >= EMOTIONAL_INTENSITY_THRESHOLD:
        return Flag(
            type=FlagType.EMOTIONAL_FRAMING,
            status=FlagStatus.FIRED,
            severity=Severity.MEDIUM,
            evidence=(
                f"Sentiment '{bundle.sentiment.label}' at confidence {intensity:.2f} "
                f"(≥ {EMOTIONAL_INTENSITY_THRESHOLD:.2f}) over the {bundle.sentiment.source}."
            ),
            plain_explanation=(
                "This caption reads as strongly negative in tone. A negative tone is not "
                "dishonest in itself, but strongly negative framing can discourage a reader "
                "from taking a second look — so it is worth pausing on."
            ),
            what_to_check="Try restating the claim in plain, neutral words. Does it still stand on its own?",
        )

    # Clear: either the reading is not negative, or negative but below the confidence threshold.
    reason = (
        f"Sentiment '{bundle.sentiment.label}' at confidence {intensity:.2f} is not a "
        f"strongly-negative reading (fires only on 'negative' ≥ {EMOTIONAL_INTENSITY_THRESHOLD:.2f})."
    )
    return Flag(
        type=FlagType.EMOTIONAL_FRAMING,
        status=FlagStatus.CLEAR,
        severity=Severity.INFO,
        evidence=reason,
        plain_explanation="The caption does not read as strongly negative in tone.",
        what_to_check="No strong negative tone detected for this caption.",
    )


def recycled_context_rule(bundle: EvidenceBundle) -> Flag:
    """The image appearing elsewhere (especially earlier) suggests recycled context.

    Highest-value, fully reliable flag (recency comes from the web index, not a model). For
    the Prelim the matches come from the cached fixture (ADR-007).
    """
    status = bundle.extractor_status.get("reverse_image")
    # CLEAR is only safe to report when the extractor *explicitly* ran and found nothing.
    # A missing status (extractor never ran) or NOT_ASSESSED must not read as "nothing
    # recycled" — that would be a false reassurance (ADR-009).
    if not bundle.web_matches and status != FlagStatus.CLEAR:
        return Flag(
            type=FlagType.RECYCLED_CONTEXT,
            status=FlagStatus.NOT_ASSESSED,
            severity=Severity.INFO,
            evidence="Reverse-image search could not be performed for this image.",
            plain_explanation="Whether this image has appeared elsewhere before could not be checked.",
            what_to_check="Run the image through a reverse-image search to see where else it appears.",
        )

    if bundle.web_matches:
        dated = [m for m in bundle.web_matches if m.published_date]
        earliest = min((m.published_date for m in dated), default=None)
        when = f" The earliest known appearance is dated {earliest}." if earliest else ""
        return Flag(
            type=FlagType.RECYCLED_CONTEXT,
            status=FlagStatus.FIRED,
            severity=Severity.HIGH,
            evidence=(
                f"{len(bundle.web_matches)} prior web appearance(s) found, e.g. "
                f"{bundle.web_matches[0].url}.{when}"
            ),
            plain_explanation=(
                "This image has appeared elsewhere on the web, possibly in an unrelated or earlier "
                "context. Recycled imagery is a common way old material is passed off as new."
            ),
            what_to_check="Compare the dates and contexts of the earlier appearances with this post's claim.",
        )

    return Flag(
        type=FlagType.RECYCLED_CONTEXT,
        status=FlagStatus.CLEAR,
        severity=Severity.INFO,
        evidence="No prior web appearances were found for this image.",
        plain_explanation="No earlier appearances of this image were found in the searched index.",
        what_to_check="Absence of matches is not proof of originality; the index is not exhaustive.",
    )


def caption_scene_mismatch_rule(bundle: EvidenceBundle) -> Flag:
    """Does the caption describe what the image actually shows? (caption ↔ scene)

    Compares the user caption against the scene description(s) from the captioner by
    content-word overlap. Low overlap suggests the caption and the visible content are about
    different things. Requires both a caption and scene descriptions; otherwise NOT_ASSESSED
    (e.g. before the captioner is wired in).
    """
    scene_text = " ".join(s.text for s in bundle.scene_descriptions)
    if not bundle.caption or not scene_text.strip():
        missing = "caption" if not bundle.caption else "scene description"
        return Flag(
            type=FlagType.CAPTION_CONTENT_MISMATCH,
            status=FlagStatus.NOT_ASSESSED,
            severity=Severity.INFO,
            evidence=f"Cannot compare: no {missing} available.",
            plain_explanation="Whether the caption matches the image content could not be assessed.",
            what_to_check="Look at the image yourself and ask whether the caption fits what you see.",
        )

    caption_words = _content_words(bundle.caption)
    scene_words = _content_words(scene_text)
    if not caption_words or not scene_words:
        return Flag(
            type=FlagType.CAPTION_CONTENT_MISMATCH,
            status=FlagStatus.NOT_ASSESSED,
            severity=Severity.INFO,
            evidence="Caption or scene description had no comparable content words.",
            plain_explanation="There was not enough text to compare the caption against the image.",
            what_to_check="Look at the image yourself and ask whether the caption fits what you see.",
        )

    overlap = len(caption_words & scene_words) / len(caption_words)
    if overlap < CAPTION_SCENE_OVERLAP_THRESHOLD:
        return Flag(
            type=FlagType.CAPTION_CONTENT_MISMATCH,
            status=FlagStatus.FIRED,
            severity=Severity.HIGH,
            evidence=(
                f"Caption↔scene content-word overlap {overlap:.0%} is below "
                f"{CAPTION_SCENE_OVERLAP_THRESHOLD:.0%}. Scene: \"{scene_text.strip()}\"."
            ),
            plain_explanation=(
                "The caption and what the image appears to show have little in common, which can "
                "indicate the image does not actually depict what the caption claims."
            ),
            what_to_check="Check whether the image genuinely shows what the caption says it does.",
        )

    return Flag(
        type=FlagType.CAPTION_CONTENT_MISMATCH,
        status=FlagStatus.CLEAR,
        severity=Severity.INFO,
        evidence=f"Caption↔scene content-word overlap {overlap:.0%} meets the {CAPTION_SCENE_OVERLAP_THRESHOLD:.0%} threshold.",
        plain_explanation="The caption is broadly consistent with what the image appears to show.",
        what_to_check="Consistency here is about wording overlap, not a guarantee of accuracy.",
    )


# Order is the order flags are presented in the scorecard.
ALL_RULES = (
    caption_scene_mismatch_rule,
    recycled_context_rule,
    emotional_framing_rule,
)


def run_rules(bundle: EvidenceBundle) -> list[Flag]:
    """Run every deterministic rule and return one Flag per rule (ADR-004, ADR-009)."""
    return [rule(bundle) for rule in ALL_RULES]
