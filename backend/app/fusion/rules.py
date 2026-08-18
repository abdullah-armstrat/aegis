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
CAPTION_SCENE_OVERLAP_THRESHOLD = 0.15  # min content-word overlap before caption↔scene is "consistent"

# --- Emotional-framing marker thresholds (ADR-014 Option B) ---
# Set from reasoning about what each marker means, BEFORE measuring on the labelled set, so the
# rule is not tuned to its own evaluation. Each is a presentation property of the caption text.
CAPS_RATIO_THRESHOLD = 0.15         # >=15% of words shouted is a deliberate stylistic choice
EXCLAMATION_ABSOLUTE_THRESHOLD = 2  # "!!" or more is emphasis, not punctuation
EXCLAMATION_DENSITY_THRESHOLD = 0.05  # or 1 per 20 words in a longer caption
MIN_MARKERS_TO_FIRE = 2             # fire on a COMBINATION, never a single marker (ADR-014)

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

    The four markers are those specified in ADR-014 Option B: ALL-CAPS ratio, exclamation
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
    """Flags manipulation markers in the caption's *presentation* (ADR-014 Option B).

    Measures four deterministic, explainable markers computed from the caption text itself —
    ALL-CAPS ratio, exclamation density, an urgency/sensational lexicon, and in-/out-group
    framing — and fires only on a COMBINATION of at least ``MIN_MARKERS_TO_FIRE`` of them. The
    specific markers found are reported as evidence, so the finding is checkable by eye.

    This replaces the previous sentiment-polarity proxy, which keyed on distilbert SST-2's
    negative-class confidence. That measured negative *polarity*, not manipulation, and so
    labelled sober-but-critical posts as manipulative (the construct-validity gap evidenced by
    the [name] worked example, DEVLOG 2026-06-29). ``bundle.sentiment`` is deliberately
    no longer consulted by this rule.

    NOT_ASSESSED when there is no caption text to measure — markers are properties of text, so
    absent text means the check could not run, never a silent pass (ADR-009).
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
