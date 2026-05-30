"""Scorecard assembler — turns an Evidence Bundle into the system's output.

Runs the deterministic rules (always) and, when enabled and available, the LLM reasoner as an
*additive* second opinion (ADR-004). The rules are the reproducible baseline; the LLM never
overrides them — it contributes its own separately-sourced flag (``source="llm"``) so the
"rules only" vs "rules + LLM" comparison is a simple matter of toggling ``use_llm`` (the
top-band evaluation move, SSOT §5.3).

The output is a :class:`Scorecard`: a list of explained, typed flags — each FIRED / CLEAR /
NOT_ASSESSED (ADR-009) — and a neutral, non-verdict summary line (ADR-002). There is no trust
score anywhere.
"""

from __future__ import annotations

from app.config import get_settings
from app.fusion.llm_reasoner import reason_over_text
from app.fusion.rules import run_rules
from app.models import EvidenceBundle, Flag, FlagStatus, FlagType, Scorecard, Severity


def _llm_caption_scene_flag(bundle: EvidenceBundle) -> Flag:
    """An LLM-sourced second opinion on caption↔scene agreement (additive to the rules).

    Reasons over supplied text only (ADR-005). Returns a NOT_ASSESSED flag if the LLM is
    disabled/unreachable or lacks the inputs — never a fabricated finding.
    """
    scene = [s.text for s in bundle.scene_descriptions]
    verdict = reason_over_text(bundle.caption, scene, bundle.on_screen_text)

    if not verdict.available:
        return Flag(
            type=FlagType.CAPTION_CONTENT_MISMATCH,
            status=FlagStatus.NOT_ASSESSED,
            severity=Severity.INFO,
            evidence=verdict.explanation or "LLM reasoner unavailable.",
            plain_explanation="The language model's second opinion on caption↔image agreement was not available.",
            what_to_check="Look at the image yourself and ask whether the caption fits what you see.",
            source="llm",
        )

    # same_subject is None => uncertain; treat only an explicit False as a mismatch signal.
    if verdict.same_subject is False:
        return Flag(
            type=FlagType.CAPTION_CONTENT_MISMATCH,
            status=FlagStatus.FIRED,
            severity=Severity.MEDIUM,
            evidence=f"LLM judged the supplied texts to describe different things. {verdict.explanation}".strip(),
            plain_explanation=(
                "A language model reading only the extracted text judged the caption and the image's "
                "description to be about different things."
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
            plain_explanation="A language model judged the caption and the image description broadly consistent.",
            what_to_check="Model agreement is not a guarantee; trust the deterministic checks more.",
            source="llm",
        )

    # Uncertain (None): be honest about it.
    return Flag(
        type=FlagType.CAPTION_CONTENT_MISMATCH,
        status=FlagStatus.NOT_ASSESSED,
        severity=Severity.INFO,
        evidence=f"LLM did not return a clear judgement. {verdict.explanation}".strip(),
        plain_explanation="The language model could not clearly judge caption↔image agreement.",
        what_to_check="Look at the image yourself and ask whether the caption fits what you see.",
        source="llm",
    )


def _summarise(flags: list[Flag]) -> str:
    """A neutral, non-verdict one-line overview (ADR-002 — describes, never decides)."""
    fired = sum(1 for f in flags if f.status == FlagStatus.FIRED)
    not_assessed = sum(1 for f in flags if f.status == FlagStatus.NOT_ASSESSED)
    if fired == 0:
        base = "No flags were raised by the checks that could be run."
    else:
        base = f"{fired} point{'s' if fired != 1 else ''} to check before trusting this content."
    if not_assessed:
        base += f" {not_assessed} check{'s' if not_assessed != 1 else ''} could not be performed."
    return base


def build_scorecard(bundle: EvidenceBundle) -> Scorecard:
    """Assemble the Scorecard: deterministic rules always; LLM second opinion when enabled."""
    flags: list[Flag] = list(run_rules(bundle))

    if get_settings().use_llm:
        flags.append(_llm_caption_scene_flag(bundle))

    return Scorecard(
        flags=flags,
        summary=_summarise(flags),
        modality=bundle.meta.modality,
        source_ref=bundle.meta.source_ref,
    )
