"""Deterministic fusion rules.

Each rule reads the Evidence Bundle and returns one :class:`Flag`: FIRED, CLEAR or NOT_ASSESSED.
A rule returns a flag even when it cannot run, so "could not check" is never mistaken for
"checked and fine". These rules are the "rules only" baseline in the evaluation, and their
thresholds are the named constants below. No rule gives a trust verdict, only an explained flag.
"""

from __future__ import annotations

import re
from datetime import date

from app.config import get_settings
from app.extractors.caption_match import read_from_image
from app.extractors.video import clock
from app.fusion.severity import Evidence, severity
from app.models import (
    EvidenceBundle,
    Flag,
    FlagStatus,
    FlagType,
    Modality,
    Severity,
)

# --- Caption vs scene description by word overlap (the older method) ---
CAPTION_SCENE_OVERLAP_THRESHOLD = 0.15  # fires below this share of caption words found in the scene

# --- Caption vs picture by meaning ---
# Fitted on the calibration part of the VERITE sample: the pair that flags at most 1 in 10
# truthful captions while catching the most out-of-context images. The flag fires only when BOTH
# similarities are below. On the held-out part it flagged 10 of 60 truthful captions and caught
# 13 of 61 out-of-context images.
CAPTION_IMAGE_SIMILARITY_THRESHOLD = 0.3772  # CLIP ViT-B/32: the picture against the caption
CAPTION_TEXT_SIMILARITY_THRESHOLD = 0.7274   # spaCy: the caption against scene description + on-screen text

# --- Caption vs picture, picture similarity only (the default method) ---
# Fitted on all 300 VERITE sample pairs: flags at most 10 of the 100 truthful captions while
# catching the most out-of-context images. Kept to 5 decimals because two pairs score 0.27745 and
# 0.27747. On 165 unseen VERITE pairs it flagged 3 of 48 truthful captions and caught 26 of 69
# out-of-context images and 7 of 48 miscaptioned ones.
CAPTION_IMAGE_ONLY_THRESHOLD = 0.27746  # CLIP ViT-B/32: the picture against the caption

# --- Speech vs picture (video) ---
# Fitted on the fitting half of dataset E's swap test (true narration lines vs lines taken from
# other footage): flags at most 1 in 10 true lines while catching the most swapped ones. On the
# held-out half it flagged 8 of 50 true lines and caught 42 of 50 swapped ones.
SPEECH_PICTURE_THRESHOLD = 0.2354  # CLIP ViT-B/32: a stretch of speech against its frames

# What the speech check cannot see, added to every result it gives.
_SPEECH_LIMIT = (
    "This only compares the kind of scene. It cannot catch a wrong detail, such as a colour, a number, "
    "a name or a place, in words that fit the scene."
)

# --- Emotional-framing marker thresholds ---
# Set from what each marker means, before looking at the labelled set, so the rule is not tuned
# to its own evaluation. Each one is about how the caption is written, not what it says.
CAPS_RATIO_THRESHOLD = 0.15         # 15% or more of words in capitals is a clear style choice
EXCLAMATION_ABSOLUTE_THRESHOLD = 2  # "!!" or more is emphasis, not punctuation
EXCLAMATION_DENSITY_THRESHOLD = 0.05  # or 1 per 20 words
MIN_MARKERS_TO_FIRE = 2             # one marker alone is ordinary style, so two are needed

# Small English stop-word list, so the overlap is not dominated by filler words.
_STOPWORDS = frozenset(
    """a an the this that these those is are was were be been being of to in on at for with
    and or but if then than so as it its from by into over under up down out about no not just
    we you they he she i them his her their our your my me""".split()
)

# Urgency and sensational phrases that push the reader to act (share, hurry, be amazed) rather
# than check. Grouped by what they do, not picked to fit particular examples.
_URGENCY_PATTERNS: tuple[tuple[str, str], ...] = (
    ("urgency word", r"\b(urgent|breaking|alert|warning)\b"),
    ("share pressure", r"\b(share|repost|forward|spread)\b[^.!?]{0,30}\b(before|now|immediately|quickly|everyone)\b"),
    ("deletion threat", r"\bbefore\b[^.!?]{0,25}\b(deleted|delete|removed|taken down|banned)\b"),
    ("disbelief hook", r"\byou\b[^.!?]{0,15}\b(won't|will not|wont)\b[^.!?]{0,10}\bbelieve\b"),
    ("sensational adjective", r"\b(shocking|terrifying|horrifying|unbelievable|insane|devastating)\b"),
    ("must-see framing", r"\b(must see|must watch|you need to see|everyone needs to see)\b"),
)

# "Us against them" phrases. Bare pronouns such as "they" or "us" are left out because they are
# too common in ordinary reporting to mean anything on their own.
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
    """Lower-case and straighten curly apostrophes so the patterns match."""
    return text.replace("’", "'").lower()


def find_manipulation_markers(text: str) -> list[str]:
    """Return a readable description of each manipulation marker found in ``text``.

    The four markers are the share of words in capitals, exclamation marks, the urgency list and
    the "us against them" list. Each description is something the user can check by reading the
    caption. Returns an empty list if there are none.
    """
    normalised = _normalise(text)
    words = re.findall(r"[A-Za-z']+", text)
    markers: list[str] = []

    # 1. ALL-CAPS ratio. Single letters like "I" and "A" are not shouting, so they are skipped.
    shoutable = [w for w in words if len(w) >= 2]
    caps = [w for w in shoutable if w.isupper()]
    if shoutable:
        ratio = len(caps) / len(shoutable)
        if ratio >= CAPS_RATIO_THRESHOLD:
            markers.append(
                f"words in capitals: {ratio:.0%} ({len(caps)} of {len(shoutable)} words: "
                f"{', '.join(caps[:4])})"
            )

    # 2. Exclamation marks: two or more, or at least 1 per 20 words.
    bangs = text.count("!")
    density = bangs / len(words) if words else 0.0
    if bangs >= EXCLAMATION_ABSOLUTE_THRESHOLD or (
        bangs and density >= EXCLAMATION_DENSITY_THRESHOLD
    ):
        markers.append(f"{bangs} exclamation mark{'s' if bangs != 1 else ''} in {len(words)} words")

    # 3. Urgency and sensational phrases.
    urgency_hits = [m.group(0) for _, pattern in _URGENCY_PATTERNS if (m := re.search(pattern, normalised))]
    if urgency_hits:
        markers.append("words from the urgency list: " + ", ".join(f'"{h}"' for h in urgency_hits))

    # 4. "Us against them" phrases.
    group_hits = [m.group(0) for _, pattern in _GROUP_FRAMING_PATTERNS if (m := re.search(pattern, normalised))]
    if group_hits:
        markers.append("listed phrases: " + ", ".join(f'"{h}"' for h in group_hits))

    return markers


def emotional_framing_rule(bundle: EvidenceBundle) -> Flag:
    """Flag manipulation markers in how the caption is written.

    Fires when at least ``MIN_MARKERS_TO_FIRE`` of the four markers are present, and lists them as
    evidence. This replaced an earlier sentiment score (distilbert SST-2), which measured negative
    tone rather than manipulation and flagged a calm but critical news post in testing on
    2026-06-29. NOT_ASSESSED when there is no caption text. Videos use ``_video_emotional_framing``.
    """
    if bundle.meta.modality == Modality.VIDEO:
        return _video_emotional_framing(bundle)
    caption = (bundle.caption or "").strip()
    if not caption or not re.search(r"[A-Za-z]", caption):
        return Flag(
            type=FlagType.EMOTIONAL_FRAMING,
            status=FlagStatus.NOT_ASSESSED,
            severity=Severity.INFO,
            evidence="There is no caption text to check for a shouting style.",
            plain_explanation="Whether the caption is written in a shouting style could not be checked.",
            what_to_check="Read the wording yourself. Is it shouting, or telling you to share at once?",
        )

    markers = find_manipulation_markers(caption)

    if len(markers) >= MIN_MARKERS_TO_FIRE:
        return Flag(
            type=FlagType.EMOTIONAL_FRAMING,
            status=FlagStatus.FIRED,
            severity=severity("emotional_framing", Evidence.INDIRECT),
            evidence=f"{len(markers)} of 4 style markers present: " + "; ".join(markers) + ".",
            plain_explanation=(
                f"This caption is written in a shouting style. {_style_summary(markers)} "
                "This is about how it is written, not about what it says."
            ),
            what_to_check="Read the caption without the capitals and exclamation marks. Then look at what it actually says.",
        )

    # CLEAR: still name the single marker if there was one, so the result is explained
    if markers:
        reason = f"Only 1 of 4 style markers present ({markers[0]}); {MIN_MARKERS_TO_FIRE} are needed."
    else:
        reason = ("None of the 4 style markers is present (words in capitals, exclamation marks, words "
                  "from the urgency list, listed phrases).")
    return Flag(
        type=FlagType.EMOTIONAL_FRAMING,
        status=FlagStatus.CLEAR,
        severity=Severity.INFO,
        evidence=reason,
        plain_explanation="This caption is not written in a shouting style.",
        what_to_check="How words are written says nothing about what they claim. The other checks look at the content.",
    )


def _style_summary(markers: list[str]) -> str:
    """The style markers found, in plain words."""
    parts = []
    for m in markers:
        if m.startswith("words in capitals"):
            parts.append("Many words are in capitals.")
        elif "exclamation mark" in m:
            parts.append("There are several exclamation marks.")
        elif m.startswith("words from the urgency list"):
            parts.append("It uses words from a list of urgent words: " + m.split(": ", 1)[1] + ".")
        elif m.startswith("listed phrases"):
            parts.append("It uses phrases from a list: " + m.split(": ", 1)[1] + ".")
    return " ".join(parts)


def _video_emotional_framing(bundle: EvidenceBundle) -> Flag:
    """Run the markers on each text in a video: caption, speech and each keyframe's on-screen text.

    Each source is judged on its own, and the flag fires if any one of them has enough markers.
    """
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
    if not bundle.keyframes:
        missing.append("no frame of the video could be read")
    elif bundle.extractor_status.get("ocr") == FlagStatus.NOT_ASSESSED:
        missing.append("the text in the keyframes could not be read")
    elif not any(k.on_screen_text for k in bundle.keyframes):
        missing.append("no on-screen text was found in the keyframes")
    gaps = f" Not checked: {'; '.join(missing)}." if missing else ""

    if not sources:
        return Flag(
            type=FlagType.EMOTIONAL_FRAMING,
            status=FlagStatus.NOT_ASSESSED,
            severity=Severity.INFO,
            evidence=f"There is no text to check for a shouting style.{gaps}",
            plain_explanation="Whether the video's words are written in a shouting style could not be checked.",
            what_to_check="Listen and read yourself. Is it shouting, or telling you to share at once?",
        )

    results = [(label, find_manipulation_markers(text), times) for label, text, times in sources]
    firing = [r for r in results if len(r[1]) >= MIN_MARKERS_TO_FIRE]
    if firing:
        return Flag(
            type=FlagType.EMOTIONAL_FRAMING,
            status=FlagStatus.FIRED,
            severity=severity("emotional_framing", Evidence.INDIRECT),
            evidence=" ".join(f"In {label}, {len(m)} of 4 style markers: {'; '.join(m)}."
                              for label, m, _ in firing) + gaps,
            plain_explanation=(
                f"Some of the words in this video are written in a shouting style. {_style_summary(firing[0][1])} "
                "This is about how they are written, not about what they say."
            ),
            what_to_check="Read and listen without the capitals and the urgency. Then look at what is actually said.",
            timestamps=sorted({t for _, _, times in firing for t in times}),
        )
    checked = "; ".join(f"{label}: {len(m)} marker{'s' if len(m) != 1 else ''}" for label, m, _ in results)
    return Flag(
        type=FlagType.EMOTIONAL_FRAMING,
        status=FlagStatus.CLEAR,
        severity=Severity.INFO,
        evidence=f"Checked {checked}; {MIN_MARKERS_TO_FIRE} in one source are required to fire.{gaps}",
        plain_explanation="The video's words are not written in a shouting style.",
        what_to_check="How words are written says nothing about what they claim. The other checks look at the content.",
    )


def _parse_iso(value: str | None) -> date | None:
    """Parse an ISO date, or return None if it is missing or invalid."""
    try:
        return date.fromisoformat(value) if value else None
    except ValueError:
        return None


def recycled_context_rule(bundle: EvidenceBundle) -> Flag:
    """Has this image, or a frame of this video, appeared before the post's date?

    For a video every keyframe is looked up, and the flag gives the times of the frames that matched.
    """
    if bundle.meta.modality != Modality.VIDEO:
        return _recycled_context(bundle)
    status = bundle.extractor_status.get("reverse_image")
    if not bundle.web_matches and status != FlagStatus.CLEAR:
        return Flag(
            type=FlagType.RECYCLED_CONTEXT,
            status=FlagStatus.NOT_ASSESSED,
            severity=Severity.INFO,
            evidence=bundle.extractor_detail.get(
                "reverse_image", "The keyframes could not all be searched for earlier copies."),
            plain_explanation="We could not check whether this video's frames have been online before.",
            what_to_check="Take a screenshot of a key moment and run it through a reverse-image search.",
        )
    if not bundle.web_matches:
        return Flag(
            type=FlagType.RECYCLED_CONTEXT,
            status=FlagStatus.CLEAR,
            severity=Severity.INFO,
            evidence=f"None of the {len(bundle.keyframes)} keyframes is close enough to an image in the "
                     "image history index to be a copy.",
            plain_explanation="No earlier copies of this video's frames were found in the index we searched.",
            what_to_check="Finding no copies does not prove the video is new. The index does not hold every image.",
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


# A page's date is the page's, not the picture's: a Wikipedia article can be years older than a
# photo added to it later. Added to every result that cites a page.
_PAGE_DATE_LIMIT = "A page can be older or newer than the image on it."


def _recycled_check(pages) -> str:
    """Name the lookup a finding rests on, for its severity.

    That is the history index if any of the pages came from it, otherwise the live web search.
    """
    return "recycled_index" if any(m.found_by != "web" for m in pages) else "recycled_web"


def _dated_by(match) -> str:
    """How a web page's date was found, for the evidence."""
    if match.date_source == "htmldate":
        return "; date from the page's own metadata"
    if match.date_source == "wayback":
        return "; date of the Wayback Machine's first capture of the page"
    return ""


def _recycled_context(bundle: EvidenceBundle) -> Flag:
    """Has this image been found on a page dated before the post?

    No lookup gives NOT_ASSESSED and no match gives CLEAR. With a match, it fires if a page is
    dated before the posting date, and also fires (as incomplete) if no posting date was given or
    some pages are undated. Otherwise it is CLEAR. A page dated on the posting date itself does not
    count as earlier, since it may be the post being checked.
    """
    status = bundle.extractor_status.get("reverse_image")
    web_note = bundle.extractor_detail.get("web_search", "")
    if not bundle.web_matches and status != FlagStatus.CLEAR:
        return Flag(
            type=FlagType.RECYCLED_CONTEXT,
            status=FlagStatus.NOT_ASSESSED,
            severity=Severity.INFO,
            evidence=bundle.extractor_detail.get(
                "reverse_image", "The image could not be looked up in the image history index."),
            plain_explanation="We could not check whether this image has been online before.",
            what_to_check="Run the image through a reverse-image search to see where else it appears.",
        )

    if not bundle.web_matches:
        return Flag(
            type=FlagType.RECYCLED_CONTEXT,
            status=FlagStatus.CLEAR,
            severity=Severity.INFO,
            evidence="The image history index holds no image close enough to this one to be a copy."
                     + (f" {web_note}" if web_note else ""),
            plain_explanation=("No earlier copies of this image were found in our index or on the web."
                               if web_note else "No earlier copies of this image were found in the index we searched."),
            what_to_check="Finding no copies does not prove the image is new. The index does not hold every image.",
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
    found = f"{len(matches)} page(s) showing a matching image found, e.g. {matches[0].url}."
    if any(m.found_by == "web" for m in matches):
        on_web = sum(m.found_by == "web" for m in matches)
        found += f" {on_web} of them {'is a web page' if on_web == 1 else 'are web pages'} found by the live web search."
    elif web_note:
        found += f" {web_note}"
    posted = _parse_iso(bundle.meta.posted_date)

    if posted is None:
        when = (f" The earliest date among them: found on a page dated {earliest[0].isoformat()} "
                f"({earliest[1].url}{_dated_by(earliest[1])})." if earliest else " None of the pages is dated.")
        return Flag(
            type=FlagType.RECYCLED_CONTEXT,
            status=FlagStatus.FIRED,
            severity=severity(_recycled_check(matches), Evidence.DIRECT_INCOMPLETE),
            evidence=found + when + closeness,
            plain_explanation=(
                "This image has been found on other pages. No posting date was given, so we cannot say "
                "whether those pages came before this post. Add the date the post was published to "
                f"compare them. {_PAGE_DATE_LIMIT}"
            ),
            what_to_check="Open the pages and compare their dates, and what they say the image shows, with this post's claim.",
        )

    earlier = [(d, m) for d, m in dated if d < posted]
    if earlier:
        first_date, first = min(earlier, key=lambda pair: pair[0])
        return Flag(
            type=FlagType.RECYCLED_CONTEXT,
            status=FlagStatus.FIRED,
            severity=severity(_recycled_check([m for _, m in earlier]), Evidence.DIRECT_COMPLETE),
            evidence=(
                f"Found on a page dated {first_date.isoformat()} ({first.url}{_dated_by(first)}), before the "
                f"stated posting date of {posted.isoformat()}. {len(earlier)} of {len(matches)} page(s) "
                f"showing the image are dated before the post.{closeness}"
            ),
            plain_explanation=(
                "A copy of this image was found on a page dated before this post says it was published. "
                "Old pictures shown as new are a common way to mislead people about when or where "
                f"something happened. {_PAGE_DATE_LIMIT}"
            ),
            what_to_check="Open that page: check when the image was added to it, and what it said the image showed.",
        )

    undated = len(matches) - len(dated)
    if undated:
        return Flag(
            type=FlagType.RECYCLED_CONTEXT,
            status=FlagStatus.FIRED,
            severity=severity(_recycled_check(matches), Evidence.DIRECT_INCOMPLETE),
            evidence=(
                f"{found} None of the dated pages is dated before {posted.isoformat()}, but "
                f"{undated} page(s) carry no date, so the comparison is incomplete.{closeness}"
            ),
            plain_explanation=(
                "This image has been found on other pages. Some of them have no date, so we cannot tell "
                f"whether this post came first. {_PAGE_DATE_LIMIT}"
            ),
            what_to_check="Check when the undated pages first showed this image.",
        )

    return Flag(
        type=FlagType.RECYCLED_CONTEXT,
        status=FlagStatus.CLEAR,
        severity=Severity.INFO,
        evidence=(
            f"{len(matches)} page(s) showing a matching image found, all dated on or after the stated "
            f"posting date of {posted.isoformat()} (the earliest dated {earliest[0].isoformat()}).{closeness}"
        ),
        plain_explanation=(
            "Copies of this image were found, but none on a page dated before this post's date. "
            f"{_PAGE_DATE_LIMIT}"
        ),
        what_to_check="Not finding an earlier page does not prove the image is new. The search does not cover every page.",
    )


def caption_scene_mismatch_rule(bundle: EvidenceBundle) -> Flag:
    """Does the caption fit what the image shows?

    ``caption_match_method`` picks how: CLIP picture similarity ("image"), that plus spaCy
    similarity to the scene description ("meaning"), word overlap ("overlap"), or not at all
    ("off"). Videos always use the keyframe CLIP check unless it is off.
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


# What the meaning checks cannot see, added to every result they give.
_MEANING_LIMIT = (
    "This only compares the kind of scene. It cannot catch a wrong name, place or date in a "
    "caption that fits the scene."
)

# Shown on the scorecard when the check is switched off.
CAPTION_CHECK_OFF_REASON = "The caption-vs-picture check is switched off in this configuration."


def _caption_check_off(bundle: EvidenceBundle) -> Flag:
    """The check is switched off, so report NOT_ASSESSED rather than leaving it out."""
    return Flag(
        type=FlagType.CAPTION_CONTENT_MISMATCH,
        status=FlagStatus.NOT_ASSESSED,
        severity=Severity.INFO,
        evidence=CAPTION_CHECK_OFF_REASON,
        plain_explanation="Whether the picture fits the caption was not checked.",
        what_to_check="Look at the image yourself and ask whether the caption fits what you see.",
    )


def _video_caption_rule(bundle: EvidenceBundle) -> Flag:
    """Fires unless the caption is a reasonable match for at least one keyframe.

    Uses the same CLIP score and threshold as the picture-only check on images.
    """
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
            severity=severity("caption_video", Evidence.INDIRECT),
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
        what_to_check="This check cannot test names, places or dates. Look them up in a source you rely on.",
        timestamps=[best.timestamp],
    )


def _similarity_not_assessed(bundle: EvidenceBundle) -> Flag:
    """NOT_ASSESSED flag for a similarity check, giving the reason the score is missing."""
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
    """Fires when the picture itself is a weak match for the caption (CLIP similarity).

    It does not use the scene description, so it does not need the captioner.
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
            severity=severity("caption_image", Evidence.INDIRECT),
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
        what_to_check="This check cannot test names, places or dates. Look them up in a source you rely on.",
    )


def _caption_meaning_rule(bundle: EvidenceBundle) -> Flag:
    """Fires when both the picture and what was read from it are weak matches for the caption.

    Uses CLIP for the picture and spaCy for the scene description and on-screen text. On the
    held-out VERITE part it caught about 1 in 5 mismatched pairs (of either kind) while flagging
    1 in 6 truthful ones. A false caption often describes the right kind of scene with a wrong
    detail, which a similarity score cannot see, and the explanation says so.
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
            severity=severity("caption_meaning", Evidence.INDIRECT),
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
        what_to_check="This check cannot test names, places or dates. Look them up in a source you rely on.",
    )


def _caption_overlap_rule(bundle: EvidenceBundle) -> Flag:
    """Caption vs scene description by shared content words (the older method).

    Fires when the share of caption words that also appear in the BLIP description is below the
    threshold. NOT_ASSESSED without both a caption and a scene description.
    """
    scene_text = " ".join(s.text for s in bundle.scene_descriptions)
    if not bundle.caption or not scene_text.strip():
        if not bundle.caption:
            reason = "no caption available."
        else:
            reason = bundle.extractor_detail.get("captioner", "no description of the picture available.")
        return Flag(
            type=FlagType.CAPTION_CONTENT_MISMATCH,
            status=FlagStatus.NOT_ASSESSED,
            severity=Severity.INFO,
            evidence=f"Cannot compare: {reason}",
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
            severity=severity("caption_overlap", Evidence.INDIRECT),
            evidence=(
                f"Caption↔scene content-word overlap {overlap:.0%} is below "
                f"{CAPTION_SCENE_OVERLAP_THRESHOLD:.0%}. Scene: \"{scene_text.strip()}\"."
            ),
            plain_explanation=(
                "The caption and what the picture seems to show share few words. This can mean the "
                "picture does not show what the caption says."
            ),
            what_to_check="Check whether the image genuinely shows what the caption says it does.",
        )

    return Flag(
        type=FlagType.CAPTION_CONTENT_MISMATCH,
        status=FlagStatus.CLEAR,
        severity=Severity.INFO,
        evidence=f"Caption↔scene content-word overlap {overlap:.0%} meets the {CAPTION_SCENE_OVERLAP_THRESHOLD:.0%} threshold.",
        plain_explanation="The caption is broadly consistent with what the image appears to show.",
        what_to_check="This only counts shared words. It cannot tell you if the caption is right about the picture.",
    )


def audio_visual_mismatch_rule(bundle: EvidenceBundle) -> Flag:
    """Does what is said match what is shown at that moment? (video only)

    Each stretch of speech has a CLIP score against its frames (the best one is kept). Any stretch
    below the threshold fires the flag, citing its time. No audio or no speech gives NOT_ASSESSED.
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
        reason = bundle.extractor_detail.get(
            "speech_picture", "The model that compares what is said with the picture did not run.")
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
            severity=severity("speech_picture", Evidence.INDIRECT),
            evidence=" ".join(
                f"At {clock(seg.start)}, \"{seg.text}\" is a weak match for the picture at that moment "
                f"(similarity {seg.picture_similarity:.2f}; under {SPEECH_PICTURE_THRESHOLD:.2f} counts as weak)."
                for seg in low) + f" {len(scored) - len(low)} of {len(scored)} stretches of speech match reasonably.",
            plain_explanation=(
                f"At {moments}, the words spoken seem to describe a different scene from the picture. "
                f"{_SPEECH_LIMIT}"
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
        what_to_check="This check cannot test spoken names, numbers or places. Look them up in a source you rely on.",
        timestamps=[weakest.start],
    )


# in the order the flags appear on the scorecard
ALL_RULES = (
    caption_scene_mismatch_rule,
    recycled_context_rule,
    emotional_framing_rule,
)


def run_rules(bundle: EvidenceBundle) -> list[Flag]:
    """Run every rule and return one Flag per rule, whatever its outcome.

    The speech vs picture rule is only added for video.
    """
    flags = [rule(bundle) for rule in ALL_RULES]
    if bundle.meta.modality == Modality.VIDEO:
        flags.append(audio_visual_mismatch_rule(bundle))
    return flags
