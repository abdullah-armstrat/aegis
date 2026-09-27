"""Severity: how prominently a finding is shown, by one stated rule per level (ADR-045, ADR-048).

Severity says how much weight the evidence behind a finding can bear. Two things decide it: the
kind of evidence, and how often the check raised a finding where there was nothing to find (its
false-alarm rate). It is not a probability that a post is misleading, and no level is named or
described with a verdict word.

The false-alarm rate used is the upper end of the 95% Wilson interval of what was measured, not the
measured share itself, because some checks rest on very few cases: a check with no false alarm in
15 cases could still have one in five.

  Direct evidence   something the user can open and see for themselves: the same image on
                    another page, with that page's date.
  Indirect signal   a pattern that suggests something without showing it: a model's similarity
                    score, or wording markers.

  high    Direct evidence that is complete: the same image found on a page dated before the
          stated posting date, by a check whose false-alarm rate is under 5%.
  medium  Direct evidence that is incomplete (the same image found, but no posting date was
          given, or the pages that could be earlier carry no date), by a check whose false-alarm
          rate is under 5%; or an indirect signal whose false-alarm rate is 10% or less.
  low     Any other finding: an indirect signal whose false-alarm rate is above 10% or not yet
          measured, or direct evidence from a check whose rate is 5% or more or not yet measured.
  info    The check raised no finding: it was clear, or it could not be assessed.

One cap on top: a finding that rests on the live web search is medium at most, because a web
page's date is read from the page and the spot-check found 3 of 12 checkable page dates wrong.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from app.models import FlagStatus, Severity

DIRECT_LIMIT = 0.05    # direct evidence: false-alarm rate under this
INDIRECT_LIMIT = 0.10  # indirect signal: false-alarm rate at or under this
_Z = 1.959964          # 95%
_ORDER = [Severity.INFO, Severity.LOW, Severity.MEDIUM, Severity.HIGH]


class Evidence(str, Enum):
    DIRECT_COMPLETE = "direct, complete"
    DIRECT_INCOMPLETE = "direct, incomplete"
    INDIRECT = "indirect"


@dataclass(frozen=True)
class FalseAlarms:
    """A check's measured false alarms: findings raised where there was nothing to find."""

    raised: int
    cases: int
    measured_on: str

    @property
    def rate(self) -> float:
        return self.raised / self.cases

    @property
    def upper(self) -> float:
        """The upper end of the 95% Wilson interval of the rate."""
        n, p = self.cases, self.raised / self.cases
        centre = (p + _Z * _Z / (2 * n)) / (1 + _Z * _Z / n)
        half = _Z * math.sqrt(p * (1 - p) / n + _Z * _Z / (4 * n * n)) / (1 + _Z * _Z / n)
        return min(1.0, centre + half)


# Each check's false alarms, from the evaluation named (the project's decision log). None: not yet
# measured, which puts any finding of that check at "low".
FALSE_ALARMS: dict[str, FalseAlarms | None] = {
    # The image history index: no unrelated pair matched by the hash or by keypoints (WP-1b, ADR-022).
    "recycled_index": FalseAlarms(0, 9977, "unrelated image pairs matched, WP-1b"),
    # The live web search: none of 15 never-posted photos was found on any page (WP-4, ADR-033).
    "recycled_web": FalseAlarms(0, 15, "never-posted photos found on a page, WP-4 dataset D"),
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

# The most a check's findings can reach, whatever its rate.
CAPS: dict[str, Severity] = {"recycled_web": Severity.MEDIUM}


def severity(check: str, evidence: Evidence, status: FlagStatus = FlagStatus.FIRED) -> Severity:
    """The level for a finding of ``check`` resting on ``evidence``, by the rule above."""
    if status != FlagStatus.FIRED:
        return Severity.INFO
    measured = FALSE_ALARMS.get(check)
    rate = measured.upper if measured else None
    if evidence == Evidence.INDIRECT:
        level = Severity.MEDIUM if rate is not None and rate <= INDIRECT_LIMIT else Severity.LOW
    elif rate is None or rate >= DIRECT_LIMIT:
        level = Severity.LOW
    else:
        level = Severity.HIGH if evidence == Evidence.DIRECT_COMPLETE else Severity.MEDIUM
    cap = CAPS.get(check)
    return min(level, cap, key=_ORDER.index) if cap else level
