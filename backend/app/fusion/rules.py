"""Deterministic fusion rules — the reproducible heart of the analysis.

Each rule reads the Evidence Bundle and returns exactly one :class:`Flag` describing the
*outcome of attempting that check* — FIRED, CLEAR, or NOT_ASSESSED. Crucially, a
rule emits a flag even when it cannot run, so "couldn't check" is never silently dropped and
never mistaken for "checked and consistent". These rules are deterministic and explainable,
so they form the baseline the "rules only" vs "rules + LLM" evaluation compares against.

Thresholds live here as named constants; they are deliberately simple and tunable, and the
eval harness will inform their final values. No rule outputs a trust verdict — only
a typed, explained flag with a "what to check" prompt.
"""

from __future__ import annotations

import re
from datetime import date

from app.models import (
    EvidenceBundle,
    Flag,
    FlagStatus,
    FlagType,
    Severity,
)

# --- Tunable thresholds (eval harness will inform final values) ---
CAPTION_SCENE_OVERLAP_THRESHOLD = 0.15  # min content-word overlap before caption↔scene is "consistent"

# --- Emotional-framing marker thresholds ---
# Set from reasoning about what each marker means, BEFORE measuring on the labelled set, so the
# rule is not tuned to its own evaluation. Each is a presentation property of the caption text.
CAPS_RATIO_THRESHOLD = 0.15         # >=15% of words shouted is a deliberate stylistic choice
EXCLAMATION_ABSOLUTE_THRESHOLD = 2  # "!!" or more is emphasis, not punctuation
EXCLAMATION_DENSITY_THRESHOLD = 0.05  # or 1 per 20 words in a longer caption
MIN_MARKERS_TO_FIRE = 2             # fire on a COMBINATION: one marker alone is ordinary style

# Very small English stop-word list — enough to stop overlap being dominated by glue words.
_STOPWORDS = frozenset(
    """a an the this that these those is are was were be been being of to in on at for with
    and or but if then than so as it its from by into over under up down out about no not just
    we you they he she i them his her their our your my me""".split()
)

# Urgency / sensational lexicon: phrasing that pressures the reader to act (share, hurry, be
# amazed) rather than to check. Grouped by rhetorical function, not cherry-picked per example.
_URGENCY_PATTERNS: tuple[tuple[str, str], ...] = (
    ("urgency word", r"\b(urgent|breaking|alert|warning)\b"),
    ("share pressure", r"\b(share|repost|forward|spread)\b[^.!?]{0,30}\b(before|now|immediately|quickly|everyone)\b"),
    ("deletion threat", r"\bbefore\b[^.!?]{0,25}\b(deleted|delete|removed|taken down|banned)\b"),
    ("disbelief hook", r"\byou\b[^.!?]{0,15}\b(won't|will not|wont)\b[^.!?]{0,10}\bbelieve\b"),
    ("sensational adjective", r"\b(shocking|terrifying|horrifying|unbelievable|insane|devastating)\b"),
    ("must-see framing", r"\b(must see|must watch|you need to see|everyone needs to see)\b"),
)

# In-/out-group cues: language that constructs an "us against them" frame. Bare pronouns ("they",
# "us") are deliberately EXCLUDED — they are far too common in ordinary reporting to be a marker
# on their own; only group-framing constructions count.
_GROUP_FRAMING_PATTERNS: tuple[tuple[str, str], ...] = (
    ("concealment claim", r"\bthey\b[^.!?]{0,20}\b(hiding|hide|covering up|don't want|dont want|do not want)\b"),
    ("addressed concealment", r"\b(hiding|kept) from you\b|\bdon't want you to know\b|\bdont want you to know\b"),
    ("silencing claim", r"\b(silence|silencing|censor|censoring|suppress)\b[^.!?]{0,10}\b(us|them|this|it)\b"),
    ("awakening appeal", r"\b(wake up|open your eyes|do your own research)\b"),
    ("othering label", r"\b(these people|the elites|mainstream media|sheeple)\b"),
)


def _content_words(text: str) -> set[str]:
    """Lower-cased alphabetic content words (stop-words and short tokens removed)."""
    tokens = re.findall(r"[a-zA-Z]+", text.lower())
    return {t for t in tokens if len(t) > 2 and t not in _STOPWORDS}


def _normalise(text: str) -> str:
    """Lower-case and straighten typographic apostrophes so the lexicons match reliably."""
    return text.replace("’", "'").lower()


def find_manipulation_markers(text: str) -> list[str]:
    """Return a human-readable description of each manipulation marker present in ``text``.

    The four markers are the ALL-CAPS ratio, exclamation
    density, an urgency/sensational lexicon, and in-/out-group framing cues. Each returned
    string is evidence the user can verify by looking at the caption, which is the whole point
    of replacing the sentiment-polarity proxy: the rule now measures what its name claims.

    Returns an empty list when no marker is present. Deterministic and side-effect free.
    """
    normalised = _normalise(text)
    words = re.findall(r"[A-Za-z']+", text)
    markers: list[str] = []

    # 1. ALL-CAPS ratio. Single letters ("I", "A") are excluded — they are not shouting.
    shoutable = [w for w in words if len(w) >= 2]
    caps = [w for w in shoutable if w.isupper()]
    if shoutable:
        ratio = len(caps) / len(shoutable)
        if ratio >= CAPS_RATIO_THRESHOLD:
            markers.append(
                f"ALL-CAPS ratio {ratio:.0%} ({len(caps)} of {len(shoutable)} words: "
                f"{', '.join(caps[:4])})"
            )

    # 2. Exclamation density: either a run of them, or a sustained rate in a longer caption.
    bangs = text.count("!")
    density = bangs / len(words) if words else 0.0
    if bangs >= EXCLAMATION_ABSOLUTE_THRESHOLD or (
        bangs and density >= EXCLAMATION_DENSITY_THRESHOLD
    ):
        markers.append(f"{bangs} exclamation mark(s) across {len(words)} words")

    # 3. Urgency / sensational lexicon.
    urgency_hits = [label for label, pattern in _URGENCY_PATTERNS if re.search(pattern, normalised)]
    if urgency_hits:
        markers.append(f"urgency/sensational language ({', '.join(urgency_hits)})")

    # 4. In-/out-group framing.
    group_hits = [label for label, pattern in _GROUP_FRAMING_PATTERNS if re.search(pattern, normalised)]
    if group_hits:
        markers.append(f"in-/out-group framing ({', '.join(group_hits)})")

    return markers


def emotional_framing_rule(bundle: EvidenceBundle) -> Flag:
    """Flags manipulation markers in the caption's *presentation*.

    Measures four deterministic, explainable markers computed from the caption text itself —
    ALL-CAPS ratio, exclamation density, an urgency/sensational lexicon, and in-/out-group
    framing — and fires only on a COMBINATION of at least ``MIN_MARKERS_TO_FIRE`` of them. The
    specific markers found are reported as evidence, so the finding is checkable by eye.

    This replaces the previous sentiment-polarity proxy, which keyed on distilbert SST-2's
    negative-class confidence. That measured negative *polarity*, not manipulation, and so
    labelled sober-but-critical posts as manipulative: a measured, critical real-world news post
    was flagged that way in testing on 2026-06-29. ``bundle.sentiment`` is deliberately
    no longer consulted by this rule.

    NOT_ASSESSED when there is no caption text to measure — markers are properties of text, so
    absent text means the check could not run, never a silent pass.
    """
    caption = (bundle.caption or "").strip()
    if not caption or not re.search(r"[A-Za-z]", caption):
        return Flag(
            type=FlagType.EMOTIONAL_FRAMING,
            status=FlagStatus.NOT_ASSESSED,
            severity=Severity.INFO,
            evidence="No caption text was available to examine for manipulation markers.",
            plain_explanation="How this post is worded could not be assessed.",
            what_to_check="Read the wording yourself: is it pressuring you to share rather than to check?",
        )

    markers = find_manipulation_markers(caption)

    if len(markers) >= MIN_MARKERS_TO_FIRE:
        return Flag(
            type=FlagType.EMOTIONAL_FRAMING,
            status=FlagStatus.FIRED,
            severity=Severity.MEDIUM,
            evidence=(
                f"{len(markers)} of 4 manipulation markers present: " + "; ".join(markers) + "."
            ),
            plain_explanation=(
                "This caption is written using several techniques that pressure a reader to "
                "react and share rather than to check: shouting, heavy punctuation, urgency "
                "wording, or an 'us against them' frame. These are features of how the message "
                "is presented, not evidence that it is false."
            ),
            what_to_check=(
                "Try restating the claim without the capitals, exclamation marks and urgency "
                "wording. Does it still stand on its own?"
            ),
        )

    # Clear: fewer than the required combination. Name the single marker if there was one, so
    # "clear" is never an unexplained pass.
    if markers:
        reason = (
            f"Only 1 of 4 manipulation markers present ({markers[0]}); "
            f"{MIN_MARKERS_TO_FIRE} are required to fire."
        )
    else:
        reason = (
            "None of the 4 manipulation markers were present (ALL-CAPS ratio, exclamation "
            "density, urgency/sensational lexicon, in-/out-group framing)."
        )
    return Flag(
        type=FlagType.EMOTIONAL_FRAMING,
        status=FlagStatus.CLEAR,
        severity=Severity.INFO,
        evidence=reason,
        plain_explanation="This caption is not written in a high-pressure or sensational style.",
        what_to_check="Wording alone says nothing about accuracy; the other checks cover the content.",
    )


def _parse_iso(value: str | None) -> date | None:
    try:
        return date.fromisoformat(value) if value else None
    except ValueError:
        return None


def recycled_context_rule(bundle: EvidenceBundle) -> Flag:
    """Has this image appeared before the post claims to be from?

    The lookup matches the image by content against the image history index. The posting date,
    when the user gives one, decides whether a match actually makes the post look recycled:

      Situation                                                   Status
      lookup could not run                                        not_assessed
      lookup ran, no match                                        clear
      match, posting date given, an appearance is earlier         fired, citing the earliest date
      match, posting date given, nothing earlier                  clear
      match, no posting date                                      fired, saying a date would allow
                                                                  a comparison

    One case the table above does not cover: a match whose appearances are not all dated, and
    none of the dated ones is earlier. That comparison is incomplete, so it is reported as fired
    with the gap named, never as clear: an incomplete check must not reassure. An
    appearance on the posting date itself is not "earlier": it may be the post being checked.
    """
    status = bundle.extractor_status.get("reverse_image")
    if not bundle.web_matches and status != FlagStatus.CLEAR:
        return Flag(
            type=FlagType.RECYCLED_CONTEXT,
            status=FlagStatus.NOT_ASSESSED,
            severity=Severity.INFO,
            evidence="The image could not be looked up in the image history index.",
            plain_explanation="Whether this image has appeared elsewhere before could not be checked.",
            what_to_check="Run the image through a reverse-image search to see where else it appears.",
        )

    if not bundle.web_matches:
        return Flag(
            type=FlagType.RECYCLED_CONTEXT,
            status=FlagStatus.CLEAR,
            severity=Severity.INFO,
            evidence="The image history index holds no image close enough to this one to be a copy.",
            plain_explanation="No earlier appearances of this image were found in the searched index.",
            what_to_check="Absence of matches is not proof of originality; the index is not exhaustive.",
        )

    matches = bundle.web_matches
    dated = [(d, m) for m in matches if (d := _parse_iso(m.published_date)) is not None]
    earliest = min(dated, key=lambda pair: pair[0]) if dated else None
    distances = [m.hash_distance for m in matches if m.hash_distance is not None]
    closeness = f" Closest match differs by {min(distances)} of 64 hash bits." if distances else ""
    found = f"{len(matches)} earlier appearance(s) of a matching image found, e.g. {matches[0].url}."
    posted = _parse_iso(bundle.meta.posted_date)

    if posted is None:
        when = (f" The earliest known appearance is dated {earliest[0].isoformat()}."
                if earliest else " None of the appearances is dated.")
        return Flag(
            type=FlagType.RECYCLED_CONTEXT,
            status=FlagStatus.FIRED,
            severity=Severity.HIGH,
            evidence=found + when + closeness,
            plain_explanation=(
                "This image has appeared elsewhere before. No posting date was given, so it is not "
                "possible to say whether those appearances came before this post; adding the date "
                "the post was published would allow that comparison."
            ),
            what_to_check="Compare the dates and contexts of the earlier appearances with this post's claim.",
        )

    earlier = [(d, m) for d, m in dated if d < posted]
    if earlier:
        first_date, first = min(earlier, key=lambda pair: pair[0])
        return Flag(
            type=FlagType.RECYCLED_CONTEXT,
            status=FlagStatus.FIRED,
            severity=Severity.HIGH,
            evidence=(
                f"The image appeared on {first_date.isoformat()} ({first.url}), before the stated "
                f"posting date of {posted.isoformat()}. {len(earlier)} of {len(matches)} known "
                f"appearance(s) predate the post.{closeness}"
            ),
            plain_explanation=(
                "This image was online before this post says it was published. Old images "
                "presented as new are a common way of misleading about when or where something "
                "happened."
            ),
            what_to_check="Open the earlier appearance and compare what it said the image showed.",
        )

    undated = len(matches) - len(dated)
    if undated:
        return Flag(
            type=FlagType.RECYCLED_CONTEXT,
            status=FlagStatus.FIRED,
            severity=Severity.HIGH,
            evidence=(
                f"{found} None of the dated appearances is earlier than {posted.isoformat()}, but "
                f"{undated} appearance(s) carry no date, so the comparison is incomplete.{closeness}"
            ),
            plain_explanation=(
                "This image has appeared elsewhere, and some of those appearances have no date, so "
                "it cannot be confirmed that this post came first."
            ),
            what_to_check="Check when the undated pages first showed this image.",
        )

    return Flag(
        type=FlagType.RECYCLED_CONTEXT,
        status=FlagStatus.CLEAR,
        severity=Severity.INFO,
        evidence=(
            f"{len(matches)} matching appearance(s) found, all dated on or after the stated posting "
            f"date of {posted.isoformat()} (earliest {earliest[0].isoformat()}).{closeness}"
        ),
        plain_explanation=(
            "Copies of this image exist elsewhere, but none is known from before this post's date, "
            "which is consistent with this being where it first appeared."
        ),
        what_to_check="Absence of an earlier appearance is not proof of originality; the index is not exhaustive.",
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
    """Run every deterministic rule and return one Flag per rule, whatever its outcome."""
    return [rule(bundle) for rule in ALL_RULES]
