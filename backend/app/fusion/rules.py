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

from app.config import get_settings
from app.extractors.caption_match import read_from_image
from app.extractors.video import clock
from app.models import (
    EvidenceBundle,
    Flag,
    FlagStatus,
    FlagType,
    Modality,
    Severity,
)

# --- Tunable thresholds (eval harness will inform final values) ---
CAPTION_SCENE_OVERLAP_THRESHOLD = 0.15  # min content-word overlap before caption↔scene is "consistent"

# --- Caption vs picture by meaning ---
# Fitted on the calibration part of the VERITE sample: the pair of thresholds that flags at most
# 1 in 10 truthful captions there while catching the most images used out of context. The flag
# fires only when BOTH similarities are below their thresholds. On the held-out part it flagged
# 10 of 60 truthful captions and caught 13 of 61 images used out of context.
CAPTION_IMAGE_SIMILARITY_THRESHOLD = 0.3772  # CLIP ViT-B/32: the picture against the caption
CAPTION_TEXT_SIMILARITY_THRESHOLD = 0.7274   # spaCy: the caption against scene description + on-screen text

# --- Caption vs picture, picture similarity only (the default method) ---
# Fitted on all 300 pairs of the VERITE sample: flags at most 10 of its 100 truthful captions
# while catching the most images used out of context. Stored to 5 decimals because two sampled
# pairs score 0.27745 and 0.27747; this value decides every one of the 300 exactly as fitted.
# On 165 VERITE pairs no earlier evaluation had touched, it flagged 3 of 48 truthful captions and
# caught 26 of 69 images used out of context and 7 of 48 miscaptioned ones.
CAPTION_IMAGE_ONLY_THRESHOLD = 0.27746  # CLIP ViT-B/32: the picture against the caption

# --- Speech vs picture (video) ---
# Fitted on the fitting side of dataset E's swap test (true narration lines against lines taken
# from other footage): flags at most 1 in 10 true lines there while catching the most swapped
# ones. On the held-out side it flagged 8 of 50 true lines and caught 42 of 50 swapped ones.
SPEECH_PICTURE_THRESHOLD = 0.2354  # CLIP ViT-B/32: a stretch of speech against its frames

# What the speech check can and cannot see, stated on every result it gives.
_SPEECH_LIMIT = (
    "This compares only the kind of scene: it cannot catch a wrong detail, such as a wrong colour, "
    "count, name or place, in speech that fits the scene."
)

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
    of measuring markers rather than sentiment: the rule measures what its name claims.

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
    was flagged that way in testing on 2026-06-29. The sentiment extractor has since been
    removed, since nothing read it.

    NOT_ASSESSED when there is no caption text to measure — markers are properties of text, so
    absent text means the check could not run, never a silent pass. For a video the markers are
    run on the caption, the speech and each keyframe's on-screen text (see below).
    """
    if bundle.meta.modality == Modality.VIDEO:
        return _video_emotional_framing(bundle)
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


def _video_emotional_framing(bundle: EvidenceBundle) -> Flag:
    """The markers on each text a video carries: caption, speech, and each keyframe's on-screen
    text, each judged on its own. Fires when any one source shows the combination of markers."""
    sources: list[tuple[str, str, list[float]]] = []
    if bundle.caption and re.search(r"[A-Za-z]", bundle.caption):
        sources.append(("the caption", bundle.caption.strip(), []))
    if bundle.transcript_segments:
        marked = [s.start for s in bundle.transcript_segments if find_manipulation_markers(s.text)]
        sources.append(("the speech", " ".join(s.text for s in bundle.transcript_segments), marked))
    for k in bundle.keyframes:
        text = " ".join(k.on_screen_text)
        if re.search(r"[A-Za-z]", text):
            sources.append((f"the on-screen text at {clock(k.timestamp)}", text, [k.timestamp]))

    missing = []
    if not bundle.caption:
        missing.append("there is no caption")
    if not bundle.transcript_segments:
        reason = bundle.extractor_detail.get("speech", "No speech was found.")
        missing.append("speech was not checked (" + reason[0].lower() + reason[1:].rstrip(".") + ")")
    if not any(k.on_screen_text for k in bundle.keyframes):
        missing.append("no on-screen text was found in the keyframes")
    gaps = f" Not checked: {'; '.join(missing)}." if missing else ""

    if not sources:
        return Flag(
            type=FlagType.EMOTIONAL_FRAMING,
            status=FlagStatus.NOT_ASSESSED,
            severity=Severity.INFO,
            evidence=f"No text was available to examine for manipulation markers.{gaps}",
            plain_explanation="How this video's words are presented could not be assessed.",
            what_to_check="Listen and read yourself: is it pressuring you to share rather than to check?",
        )

    results = [(label, find_manipulation_markers(text), times) for label, text, times in sources]
    firing = [r for r in results if len(r[1]) >= MIN_MARKERS_TO_FIRE]
    if firing:
        return Flag(
            type=FlagType.EMOTIONAL_FRAMING,
            status=FlagStatus.FIRED,
            severity=Severity.MEDIUM,
            evidence=" ".join(f"In {label}, {len(m)} of 4 manipulation markers: {'; '.join(m)}."
                              for label, m, _ in firing) + gaps,
            plain_explanation=(
                "Some of the words in this video use several techniques that pressure a viewer to react "
                "and share rather than to check: shouting, heavy punctuation, urgency wording, or an "
                "'us against them' frame. These are features of how the message is presented, not "
                "evidence that it is false."
            ),
            what_to_check="Restate the claim without the urgency and the capitals. Does it still stand on its own?",
            timestamps=sorted({t for _, _, times in firing for t in times}),
        )
    checked = "; ".join(f"{label}: {len(m)} marker{'s' if len(m) != 1 else ''}" for label, m, _ in results)
    return Flag(
        type=FlagType.EMOTIONAL_FRAMING,
        status=FlagStatus.CLEAR,
        severity=Severity.INFO,
        evidence=f"Checked {checked}; {MIN_MARKERS_TO_FIRE} in one source are required to fire.{gaps}",
        plain_explanation="The video's words are not presented in a high-pressure or sensational style.",
        what_to_check="Wording alone says nothing about accuracy; the other checks cover the content.",
    )


def _parse_iso(value: str | None) -> date | None:
    try:
        return date.fromisoformat(value) if value else None
    except ValueError:
        return None


def recycled_context_rule(bundle: EvidenceBundle) -> Flag:
    """Has this image, or a frame of this video, appeared before the post claims to be from?

    For a video every keyframe is looked up; the flag names the frames that matched and cites
    their times.
    """
    if bundle.meta.modality != Modality.VIDEO:
        return _recycled_context(bundle)
    status = bundle.extractor_status.get("reverse_image")
    if not bundle.web_matches and status != FlagStatus.CLEAR:
        return Flag(
            type=FlagType.RECYCLED_CONTEXT,
            status=FlagStatus.NOT_ASSESSED,
            severity=Severity.INFO,
            evidence=("The keyframes could not all be looked up in the image history index. "
                      + bundle.extractor_detail.get("reverse_image", "")).strip(),
            plain_explanation="Whether this video's frames have appeared elsewhere before could not be checked.",
            what_to_check="Take a screenshot of a key moment and run it through a reverse-image search.",
        )
    if not bundle.web_matches:
        return Flag(
            type=FlagType.RECYCLED_CONTEXT,
            status=FlagStatus.CLEAR,
            severity=Severity.INFO,
            evidence=f"None of the {len(bundle.keyframes)} keyframes is close enough to an image in the "
                     "image history index to be a copy.",
            plain_explanation="No earlier appearances of this video's frames were found in the searched index.",
            what_to_check="Absence of matches is not proof of originality; the index is not exhaustive.",
        )
    flag = _recycled_context(bundle)
    times = sorted({m.frame_timestamp for m in bundle.web_matches if m.frame_timestamp is not None})
    many = len(times) != 1
    flag.evidence = (f"The keyframe{'s' if many else ''} at {', '.join(clock(t) for t in times)} "
                     f"match{'' if many else 'es'} a known image. " + flag.evidence)
    flag.plain_explanation = flag.plain_explanation.replace("This image", "A frame from this video").replace(
        "this image", "a frame from this video")
    flag.timestamps = times
    return flag


def _recycled_context(bundle: EvidenceBundle) -> Flag:
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
    inliers = [m.keypoint_inliers for m in matches if m.keypoint_inliers is not None]
    if distances:
        closeness = f" Closest match differs by {min(distances)} of 64 hash bits."
    elif inliers:
        closeness = (f" Found by {max(inliers)} image details that line up with a known image, "
                     "which still works when a picture has been cropped or framed in a screenshot.")
    else:
        closeness = ""
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

    ``caption_match_method`` selects how: by the picture's similarity to the caption (CLIP), by
    that and the caption's similarity to the scene description (CLIP and spaCy), by the earlier
    content-word overlap, or not at all.
    """
    method = get_settings().caption_match_method
    if bundle.meta.modality == Modality.VIDEO:
        return _caption_check_off(bundle) if method == "off" else _video_caption_rule(bundle)
    if method == "image":
        return _caption_image_rule(bundle)
    if method == "meaning":
        return _caption_meaning_rule(bundle)
    if method == "off":
        return _caption_check_off(bundle)
    return _caption_overlap_rule(bundle)


# What the meaning checks can and cannot see, stated on every result they give.
_MEANING_LIMIT = (
    "This compares only the kind of scene: it cannot catch a wrong name, place or date in a "
    "caption that fits the scene."
)

# Why the check can be switched off, shown on the scorecard when it is.
CAPTION_CHECK_OFF_REASON = "The caption-vs-picture check is switched off in this configuration."


def _caption_check_off(bundle: EvidenceBundle) -> Flag:
    """The check is switched off: say so, rather than leave it out or call it clear."""
    return Flag(
        type=FlagType.CAPTION_CONTENT_MISMATCH,
        status=FlagStatus.NOT_ASSESSED,
        severity=Severity.INFO,
        evidence=CAPTION_CHECK_OFF_REASON,
        plain_explanation="Whether the picture fits the caption was not checked.",
        what_to_check="Look at the image yourself and ask whether the caption fits what you see.",
    )


def _video_caption_rule(bundle: EvidenceBundle) -> Flag:
    """The caption must be a reasonable match for at least one keyframe, by the same CLIP score
    and threshold as the picture-only check on images. BLIP and spaCy play no part for video."""
    scored = [k for k in bundle.keyframes if k.caption_similarity is not None]
    if not bundle.caption or not scored:
        reason = ("there is no caption." if not bundle.caption
                  else bundle.extractor_detail.get("caption_match", "No keyframe could be compared with the caption."))
        return Flag(
            type=FlagType.CAPTION_CONTENT_MISMATCH,
            status=FlagStatus.NOT_ASSESSED,
            severity=Severity.INFO,
            evidence=f"Cannot compare: {reason[0].lower() + reason[1:]}",
            plain_explanation="Whether the video shows the kind of scene the caption describes could not be assessed.",
            what_to_check="Watch the video yourself and ask whether the caption fits what you see.",
        )
    best = max(scored, key=lambda k: k.caption_similarity)
    low = best.caption_similarity < CAPTION_IMAGE_ONLY_THRESHOLD
    evidence = (
        f"Of {len(scored)} keyframes, the best match for the caption is the one at {clock(best.timestamp)}, "
        f"a {'weak' if low else 'reasonable'} match (similarity {best.caption_similarity:.2f}; under "
        f"{CAPTION_IMAGE_ONLY_THRESHOLD:.2f} counts as weak)."
    )
    if low:
        return Flag(
            type=FlagType.CAPTION_CONTENT_MISMATCH,
            status=FlagStatus.FIRED,
            severity=Severity.MEDIUM,
            evidence=evidence + " No keyframe is a reasonable match.",
            plain_explanation=(
                "None of the video's keyframes seems to show the kind of scene the caption describes. "
                f"{_MEANING_LIMIT}"
            ),
            what_to_check="Find where the video first appeared and check what it actually shows, and when and where.",
            timestamps=[best.timestamp],
        )
    return Flag(
        type=FlagType.CAPTION_CONTENT_MISMATCH,
        status=FlagStatus.CLEAR,
        severity=Severity.INFO,
        evidence=evidence,
        plain_explanation=f"At least one keyframe seems to show the kind of scene the caption describes. {_MEANING_LIMIT}",
        what_to_check="Check the names, places and dates in the caption against a trusted source; this check cannot.",
        timestamps=[best.timestamp],
    )


def _similarity_not_assessed(bundle: EvidenceBundle) -> Flag:
    """NOT_ASSESSED for a similarity check, with the reason the score is missing."""
    match = bundle.caption_match
    if not bundle.caption:
        reason = "no caption available."
    elif match is not None and match.detail:
        reason = match.detail
    else:
        reason = "the similarity models did not run."
    return Flag(
        type=FlagType.CAPTION_CONTENT_MISMATCH,
        status=FlagStatus.NOT_ASSESSED,
        severity=Severity.INFO,
        evidence=f"Cannot compare: {reason}",
        plain_explanation="Whether the picture fits the kind of scene the caption describes could not be assessed.",
        what_to_check="Look at the image yourself and ask whether the caption fits what you see.",
    )


def _caption_image_rule(bundle: EvidenceBundle) -> Flag:
    """Fires when the picture itself matches the caption weakly (CLIP similarity).

    Uses no scene description, so it does not depend on the captioner.
    """
    match = bundle.caption_match
    if not bundle.caption or match is None or match.image_similarity is None:
        return _similarity_not_assessed(bundle)

    low = match.image_similarity < CAPTION_IMAGE_ONLY_THRESHOLD
    evidence = (
        f"The picture is a {'weak' if low else 'reasonable'} match for the caption (similarity "
        f"{match.image_similarity:.2f}; under {CAPTION_IMAGE_ONLY_THRESHOLD:.2f} counts as weak)."
    )
    if match.caption_truncated:
        evidence += " The caption was longer than the model can read, so only its beginning was compared."
    if low:
        return Flag(
            type=FlagType.CAPTION_CONTENT_MISMATCH,
            status=FlagStatus.FIRED,
            severity=Severity.MEDIUM,
            evidence=evidence,
            plain_explanation=(
                "The picture seems to show a different kind of scene from the one the caption "
                f"describes. {_MEANING_LIMIT}"
            ),
            what_to_check=(
                "Find where the picture first appeared and check what it actually shows, and when "
                "and where it was taken."
            ),
        )
    return Flag(
        type=FlagType.CAPTION_CONTENT_MISMATCH,
        status=FlagStatus.CLEAR,
        severity=Severity.INFO,
        evidence=evidence,
        plain_explanation=f"The picture seems to show the kind of scene the caption describes. {_MEANING_LIMIT}",
        what_to_check="Check the names, places and dates in the caption against a trusted source; this check cannot.",
    )


def _caption_meaning_rule(bundle: EvidenceBundle) -> Flag:
    """Fires when the picture and the caption seem to be about different kinds of scene.

    Reads two similarities from ``bundle.caption_match``: the picture against the caption (CLIP),
    and the caption against what the other extractors read from the picture (spaCy). On the
    held-out part of the VERITE sample it caught about 1 in 5 mismatched pairs of either kind
    while flagging 1 in 6 truthful ones. A false caption usually describes the same kind of scene
    with a wrong detail, which a similarity score cannot see, and the explanation says so.
    """
    match = bundle.caption_match
    if not bundle.caption or match is None or match.image_similarity is None or match.text_similarity is None:
        return _similarity_not_assessed(bundle)

    image_low = match.image_similarity < CAPTION_IMAGE_SIMILARITY_THRESHOLD
    text_low = match.text_similarity < CAPTION_TEXT_SIMILARITY_THRESHOLD
    seen = read_from_image([s.text for s in bundle.scene_descriptions], bundle.on_screen_text)
    source = "The description of the picture" + (" and the text on it" if bundle.on_screen_text else "")

    def strength(low: bool) -> str:
        return "weak" if low else "reasonable"

    evidence = (
        f"The picture itself is a {strength(image_low)} match for the caption (similarity "
        f"{match.image_similarity:.2f}; under {CAPTION_IMAGE_SIMILARITY_THRESHOLD:.2f} counts as weak). "
        f"{source}, \"{seen}\", is a {strength(text_low)} match for the caption "
        f"(similarity {match.text_similarity:.2f}; under {CAPTION_TEXT_SIMILARITY_THRESHOLD:.2f} counts as weak)."
    )
    if match.caption_truncated:
        evidence += " The caption was longer than the model can read, so only its beginning was compared."

    if image_low and text_low:
        return Flag(
            type=FlagType.CAPTION_CONTENT_MISMATCH,
            status=FlagStatus.FIRED,
            severity=Severity.MEDIUM,
            evidence=evidence + " Both matches are weak, which is when this check is raised.",
            plain_explanation=(
                "The picture seems to show a different kind of scene from the one the caption "
                f"describes. {_MEANING_LIMIT}"
            ),
            what_to_check=(
                "Find where the picture first appeared and check what it actually shows, and when "
                "and where it was taken."
            ),
        )

    return Flag(
        type=FlagType.CAPTION_CONTENT_MISMATCH,
        status=FlagStatus.CLEAR,
        severity=Severity.INFO,
        evidence=evidence + " This check is raised only when both matches are weak.",
        plain_explanation=f"The picture seems to show the kind of scene the caption describes. {_MEANING_LIMIT}",
        what_to_check="Check the names, places and dates in the caption against a trusted source; this check cannot.",
    )


def _caption_overlap_rule(bundle: EvidenceBundle) -> Flag:
    """Caption ↔ scene by content-word overlap, the earlier method.

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


def audio_visual_mismatch_rule(bundle: EvidenceBundle) -> Flag:
    """Does what is said match what is shown at that moment? (video only)

    Each stretch of speech was scored against the frame at its midpoint and any keyframe inside
    it, with CLIP; its score is the highest. A stretch below the threshold fires the flag, which
    cites its time. No audio track or no speech makes the check not assessed, with the reason.
    """
    segments = bundle.transcript_segments
    if not segments:
        reason = bundle.extractor_detail.get("speech_picture") or bundle.extractor_detail.get(
            "speech", "No speech was found in the audio.")
        return Flag(
            type=FlagType.AUDIO_VISUAL_MISMATCH,
            status=FlagStatus.NOT_ASSESSED,
            severity=Severity.INFO,
            evidence=f"Cannot compare speech with the picture: {reason[0].lower() + reason[1:]}",
            plain_explanation="Whether what is said matches what is shown could not be assessed.",
            what_to_check="Watch with the sound on and ask whether the words fit the pictures.",
        )
    scored = [seg for seg in segments if seg.picture_similarity is not None]
    if not scored:
        reason = bundle.extractor_detail.get("speech_picture", "The picture-matching model did not run.")
        return Flag(
            type=FlagType.AUDIO_VISUAL_MISMATCH,
            status=FlagStatus.NOT_ASSESSED,
            severity=Severity.INFO,
            evidence=f"Cannot compare speech with the picture: {reason[0].lower() + reason[1:]}",
            plain_explanation="Whether what is said matches what is shown could not be assessed.",
            what_to_check="Watch with the sound on and ask whether the words fit the pictures.",
        )
    low = [seg for seg in scored if seg.picture_similarity < SPEECH_PICTURE_THRESHOLD]
    if low:
        moments = ", ".join(clock(seg.start) for seg in low)
        return Flag(
            type=FlagType.AUDIO_VISUAL_MISMATCH,
            status=FlagStatus.FIRED,
            severity=Severity.MEDIUM,
            evidence=" ".join(
                f"At {clock(seg.start)}, \"{seg.text}\" is a weak match for the picture at that moment "
                f"(similarity {seg.picture_similarity:.2f}; under {SPEECH_PICTURE_THRESHOLD:.2f} counts as weak)."
                for seg in low) + f" {len(scored) - len(low)} of {len(scored)} stretches of speech match reasonably.",
            plain_explanation=(
                f"At {moments}, the spoken line seems to describe a different scene from the picture at "
                f"that moment. {_SPEECH_LIMIT}"
            ),
            what_to_check="Watch those moments: does the picture show what is being said?",
            timestamps=[seg.start for seg in low],
        )
    weakest = min(scored, key=lambda seg: seg.picture_similarity)
    return Flag(
        type=FlagType.AUDIO_VISUAL_MISMATCH,
        status=FlagStatus.CLEAR,
        severity=Severity.INFO,
        evidence=(f"All {len(scored)} stretches of speech are a reasonable match for the picture at that moment; "
                  f"the weakest, at {clock(weakest.start)}, scores {weakest.picture_similarity:.2f} "
                  f"(under {SPEECH_PICTURE_THRESHOLD:.2f} counts as weak)."),
        plain_explanation=f"What is said seems to fit the kind of scene shown at each moment. {_SPEECH_LIMIT}",
        what_to_check="Check any names, numbers and places that are spoken against a trusted source; this check cannot.",
        timestamps=[weakest.start],
    )


# Order is the order flags are presented in the scorecard.
ALL_RULES = (
    caption_scene_mismatch_rule,
    recycled_context_rule,
    emotional_framing_rule,
)


def run_rules(bundle: EvidenceBundle) -> list[Flag]:
    """Run every deterministic rule and return one Flag per rule, whatever its outcome.

    Speech vs picture applies only to video, so an image scorecard does not carry it.
    """
    flags = [rule(bundle) for rule in ALL_RULES]
    if bundle.meta.modality == Modality.VIDEO:
        flags.append(audio_visual_mismatch_rule(bundle))
    return flags
