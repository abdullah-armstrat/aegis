"""Severity: how prominently a finding is shown, by one stated rule per level (ADR-045).

Severity says how much weight the evidence behind a finding can bear. Two things decide it: the
kind of evidence, and how often the check raised a finding where there was nothing to find (its
false-alarm rate, as measured in the project's evaluations). It is not a probability that a post
is misleading, and no level is named or described with a verdict word.

  Direct evidence   something the user can open and see for themselves: the same image on
                    another page, with that page's date.
  Indirect signal   a pattern that suggests something without showing it: a model's similarity
                    score, or wording markers.

  high    Direct evidence that is complete: the same image found on a page dated before the
          stated posting date, by a check whose measured false-alarm rate is under 5%.
  medium  Direct evidence that is incomplete (the same image found, but no posting date was
          given, or the pages that could be earlier carry no date), by a check whose measured
          false-alarm rate is under 5%; or an indirect signal whose measured false-alarm rate is
          10% or less.
  low     Any other finding: an indirect signal whose false-alarm rate is above 10% or not yet
          measured, or direct evidence from a check whose rate is 5% or more or not yet measured.
  info    The check raised no finding: it was clear, or it could not be assessed.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from app.models import FlagStatus, Severity

DIRECT_LIMIT = 0.05    # direct evidence: false-alarm rate under this
INDIRECT_LIMIT = 0.10  # indirect signal: false-alarm rate at or under this


class Evidence(str, Enum):
    DIRECT_COMPLETE = "direct, complete"
    DIRECT_INCOMPLETE = "direct, incomplete"
    INDIRECT = "indirect"


@dataclass(frozen=True)
class FalseAlarms:
    """A check's measured false-alarm rate: findings raised where there was nothing to find."""

    raised: int
    cases: int
    measured_on: str

    @property
    def rate(self) -> float:
        return self.raised / self.cases


# Each check's false-alarm rate, from the evaluation named (the project's decision log). None: not
# yet measured, which puts any finding of that check at "low".
FALSE_ALARMS: dict[str, FalseAlarms | None] = {
    # The image history lookup: no unrelated pair matched by the hash or by keypoints (WP-1b,
    # ADR-022). The live web search likewise matched none of 15 never-posted photos (WP-4, ADR-033).
    "recycled_context": FalseAlarms(0, 9977, "unrelated image pairs matched, WP-1b"),
    # Truthful captions flagged on the 165 fresh VERITE pairs (ADR-024; unchanged by ADR-041).
    "caption_image": FalseAlarms(3, 48, "truthful fresh VERITE pairs flagged, picture only"),
    "caption_meaning": FalseAlarms(8, 48, "truthful fresh VERITE pairs flagged, meaning"),
    "caption_overlap": FalseAlarms(41, 48, "truthful fresh VERITE pairs flagged, word overlap"),
    "caption_video": None,        # no set of captioned videos has been evaluated
    # Matching lines flagged on the held-out-side dataset E clips, with word timestamps (ADR-040).
    "speech_picture": FalseAlarms(8, 50, "matching lines flagged, held-out dataset E clips"),
    "emotional_framing": None,    # dataset F is evaluated in WP-6
    "llm": None,                  # not measured on real pairs
}


def severity(check: str, evidence: Evidence, status: FlagStatus = FlagStatus.FIRED) -> Severity:
    """The level for a finding of ``check`` resting on ``evidence``, by the rule above."""
    if status != FlagStatus.FIRED:
        return Severity.INFO
    measured = FALSE_ALARMS.get(check)
    rate = measured.rate if measured else None
    if evidence == Evidence.INDIRECT:
        return Severity.MEDIUM if rate is not None and rate <= INDIRECT_LIMIT else Severity.LOW
    if rate is None or rate >= DIRECT_LIMIT:
        return Severity.LOW
    return Severity.HIGH if evidence == Evidence.DIRECT_COMPLETE else Severity.MEDIUM
