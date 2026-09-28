"""Scorecard assembly: turns an Evidence Bundle into the flags the user sees.

The rules always run. When ``use_llm`` is on, the LLM adds its own flag (``source="llm"``) as a
second opinion and never overrides the rules, so "rules only" vs "rules + LLM" is just that
setting. The :class:`Scorecard` holds the flags and a neutral summary line, with no trust score.
"""

from __future__ import annotations

from app.config import get_settings
from app.fusion.llm_reasoner import reason_over_text
from app.fusion.rules import run_rules
from app.fusion.severity import Evidence, severity
from app.models import (
    EvidenceBundle,
    Flag,
    FlagStatus,
    FlagType,
    KeyframeView,
    Modality,
    Scorecard,
    Severity,
)


def _llm_caption_scene_flag(bundle: EvidenceBundle) -> Flag:
    """The LLM's second opinion on whether the caption and the scene description agree.

    Returns NOT_ASSESSED if the LLM is off, unreachable or gives no clear answer.
    """
    scene = [s.text for s in bundle.scene_descriptions]
    verdict = reason_over_text(bundle.caption, scene, bundle.on_screen_text)

    if not verdict.available:
        return Flag(
            type=FlagType.CAPTION_CONTENT_MISMATCH,
            status=FlagStatus.NOT_ASSESSED,
            severity=Severity.INFO,
            evidence=verdict.explanation or "LLM reasoner unavailable.",
            plain_explanation="The language model could not give its second opinion on the caption.",
            what_to_check="Look at the image yourself and ask whether the caption fits what you see.",
            source="llm",
        )

    # None means uncertain, so only an explicit False counts as a mismatch
    if verdict.same_subject is False:
        return Flag(
            type=FlagType.CAPTION_CONTENT_MISMATCH,
            status=FlagStatus.FIRED,
            severity=severity("llm", Evidence.INDIRECT),
            evidence=f"LLM judged the supplied texts to describe different things. {verdict.explanation}".strip(),
            plain_explanation=(
                "A language model read only the words taken from the post. It judged that the caption and "
                "the picture's description are about different things."
            ),
            what_to_check="Check whether the image genuinely shows what the caption says it does.",
            source="llm",
        )

    if verdict.same_subject is True:
        return Flag(
            type=FlagType.CAPTION_CONTENT_MISMATCH,
            status=FlagStatus.CLEAR,
            severity=Severity.INFO,
            evidence=f"LLM judged the supplied texts to describe the same thing. {verdict.explanation}".strip(),
            plain_explanation="A language model compared the caption with a description of the picture. It judged them to be about the same thing.",
            what_to_check="The model can be wrong. Rely more on the other checks.",
            source="llm",
        )

    # uncertain (None)
    return Flag(
        type=FlagType.CAPTION_CONTENT_MISMATCH,
        status=FlagStatus.NOT_ASSESSED,
        severity=Severity.INFO,
        evidence=f"LLM did not return a clear judgement. {verdict.explanation}".strip(),
        plain_explanation="The language model could not decide whether the caption fits the picture.",
        what_to_check="Look at the image yourself and ask whether the caption fits what you see.",
        source="llm",
    )


def _summarise(flags: list[Flag]) -> str:
    """One neutral summary line: how many points to check and how many checks could not run."""
    fired = sum(1 for f in flags if f.status == FlagStatus.FIRED)
    not_assessed = sum(1 for f in flags if f.status == FlagStatus.NOT_ASSESSED)
    if fired == 0:
        base = "No flags were raised by the checks that could be run."
    else:
        base = f"{fired} point{'s' if fired != 1 else ''} to check before you share this post."
    if not_assessed:
        base += f" {not_assessed} check{'s' if not_assessed != 1 else ''} could not be performed."
    return base


def build_scorecard(bundle: EvidenceBundle) -> Scorecard:
    """Build the Scorecard: the rules always, plus the LLM's opinion on images when enabled."""
    flags: list[Flag] = list(run_rules(bundle))

    # The LLM stays out of video runs: the laptop cannot hold it alongside the video models.
    if get_settings().use_llm and bundle.meta.modality == Modality.IMAGE:
        flags.append(_llm_caption_scene_flag(bundle))

    return Scorecard(
        flags=flags,
        summary=_summarise(flags),
        modality=bundle.meta.modality,
        source_ref=bundle.meta.source_ref,
        duration_s=bundle.meta.duration_s,
        keyframes=[KeyframeView(timestamp=k.timestamp, thumbnail=k.thumbnail) for k in bundle.keyframes],
    )
