"""Severity: how prominently a finding is shown.

It depends on the kind of evidence and on how often the check raised false alarms in the
evaluation (the upper end of the 95% Wilson interval). It is not a probability that the post is
misleading.

Direct evidence is something the user can open, like the same image on a dated page. An indirect
signal is a similarity score or wording markers.
  high    direct and complete (a page dated before the posting date), rate under 5%
  medium  direct but incomplete (no posting date, or undated pages), rate under 5%,
          or an indirect signal with a rate of 10% or less
  low     anything else, including checks that have not been measured
  info    the check was clear or could not run
Live web search findings are capped at medium, since page dates are sometimes wrong.
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
    """The kind of evidence a finding rests on."""

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
        """Upper end of the 95% Wilson interval of the rate.

        Used instead of the rate itself because some checks rest on few cases: no false alarm in
        15 cases could still mean one in five.
        """
        n, p = self.cases, self.raised / self.cases
        centre = (p + _Z * _Z / (2 * n)) / (1 + _Z * _Z / n)
        half = _Z * math.sqrt(p * (1 - p) / n + _Z * _Z / (4 * n * n)) / (1 + _Z * _Z / n)
        return min(1.0, centre + half)


# Each check's false alarms in the evaluation. None means not measured yet, which puts any
# finding of that check at "low".
FALSE_ALARMS: dict[str, FalseAlarms | None] = {
    # history index: no unrelated image pair matched by the hash or by keypoints
    "recycled_index": FalseAlarms(0, 9977, "unrelated image pairs matched, WP-1b"),
    # live web search: none of 15 never-posted photos was found on any page
    "recycled_web": FalseAlarms(0, 15, "never-posted photos found on a page, WP-4 dataset D"),
    # truthful captions flagged among the 165 unseen VERITE pairs
    "caption_image": FalseAlarms(3, 48, "truthful fresh VERITE pairs flagged, picture only"),
    "caption_meaning": FalseAlarms(8, 48, "truthful fresh VERITE pairs flagged, meaning"),
    "caption_overlap": FalseAlarms(41, 48, "truthful fresh VERITE pairs flagged, word overlap"),
    "caption_video": None,        # no set of captioned videos has been evaluated
    # matching lines flagged on the held-out dataset E clips, with the current Whisper settings
    "speech_picture": FalseAlarms(9, 50, "matching lines flagged, held-out dataset E clips"),
    # dataset F texts labelled honest that the flag fired on
    "emotional_framing": FalseAlarms(5, 49, "honest dataset F texts flagged"),
    "llm": None,                  # not measured on real pairs
}

# The highest level a check can reach. Web page dates are read from the pages themselves, and a
# spot-check found 3 of 12 checkable dates wrong, so the live web search stops at medium.
CAPS: dict[str, Severity] = {"recycled_web": Severity.MEDIUM}


def severity(check: str, evidence: Evidence, status: FlagStatus = FlagStatus.FIRED) -> Severity:
    """Return the level for a finding of ``check`` resting on ``evidence``, using the rules at the
    top of this module and the rates in ``FALSE_ALARMS``, then any cap in ``CAPS``."""
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
