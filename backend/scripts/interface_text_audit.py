"""Text measures for the interface review, over the words the backend writes.

Runs every branch of every check (the rules and the LLM layer) on bundles built for it, then:
  reading level   Flesch-Kincaid grade of each distinct explanation and "what to check" line
                  (Kincaid et al., 1975): 0.39 * words/sentence + 11.8 * syllables/word - 15.59.
                  Syllables are vowel groups (a e i o u y) per word, less a silent final "e"
                  (not in "-le"), at least one. Numbers count as one word of one syllable.
  verdict words   every line the backend writes (explanations, what to check, evidence) is
                  scanned for the listed words; each hit is listed with its context, not judged.

Prints one JSON object.
Run (from the repo root):  python backend/scripts/interface_text_audit.py
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

VERDICT_WORDS = ("safe", "unsafe", "fake", "true", "false", "verified", "unverified", "real", "genuine",
                 "authentic", "trustworthy", "untrustworthy", "hoax", "debunked", "misinformation",
                 "disinformation", "correct", "incorrect", "accurate", "inaccurate", "legit", "lie", "lies",
                 "trust", "reliable", "unreliable", "credible", "proven", "confirmed")
_VERDICT = re.compile(r"\b(" + "|".join(VERDICT_WORDS) + r")\b", re.IGNORECASE)


def syllables(word: str) -> int:
    word = word.lower()
    if not re.search(r"[a-z]", word):
        return 1
    count = len(re.findall(r"[aeiouy]+", word))
    if word.endswith("e") and not word.endswith("le") and count > 1:
        count -= 1
    return max(1, count)


def fk_grade(text: str) -> dict:
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if re.search(r"\w", s)]
    words = re.findall(r"[A-Za-z0-9'’-]+", text)
    if not sentences or not words:
        return {"grade": None, "words": 0, "sentences": 0, "syllables": 0}
    syl = sum(syllables(w) for w in words)
    grade = 0.39 * len(words) / len(sentences) + 11.8 * syl / len(words) - 15.59
    return {"grade": round(grade, 1), "words": len(words), "sentences": len(sentences), "syllables": syl}


def all_flags() -> list:
    """Every branch of every check, on bundles built for it."""
    from app.config import get_settings
    from app.fusion import rules, scorecard
    from app.fusion.llm_reasoner import ReasonerVerdict
    from app.models import (CaptionMatch, EvidenceBundle, FlagStatus, Keyframe, Meta, Modality,
                            SceneDescription, TranscriptSegment, WebMatch)

    def image(**kw):
        kw.setdefault("meta", Meta(modality=Modality.IMAGE))
        return EvidenceBundle(**kw)

    def video(**kw):
        kw.setdefault("meta", Meta(modality=Modality.VIDEO, duration_s=30.0, has_audio=True))
        return EvidenceBundle(**kw)

    t = rules.CAPTION_IMAGE_ONLY_THRESHOLD
    page = lambda d, **kw: WebMatch(url=f"https://news.example.com/{d or 'x'}", published_date=d, **kw)  # noqa: E731
    found = {"reverse_image": FlagStatus.FIRED}
    flags = []
    previous = os.environ.get("AEGIS_CAPTION_MATCH_METHOD")
    for method in ("image", "meaning", "overlap", "off"):
        os.environ["AEGIS_CAPTION_MATCH_METHOD"] = method
        get_settings.cache_clear()
        cm = rules.caption_scene_mismatch_rule
        flags += [
            cm(image()),
            cm(image(caption="A flood in the city", caption_match=CaptionMatch(detail="The picture is nearly blank."))),
            cm(image(caption="A flood in the city", caption_match=CaptionMatch(image_similarity=t - 0.1, text_similarity=0.1),
                     scene_descriptions=[SceneDescription(text="a dry street")])),
            cm(image(caption="A flood in the city", caption_match=CaptionMatch(image_similarity=t + 0.2, text_similarity=0.9),
                     scene_descriptions=[SceneDescription(text="a flooded city street")])),
            cm(image(caption="the of and", scene_descriptions=[SceneDescription(text="a b")])),
            cm(video(caption="A flood", keyframes=[Keyframe(timestamp=3.0, caption_similarity=t - 0.1)])),
            cm(video(caption="A flood", keyframes=[Keyframe(timestamp=3.0, caption_similarity=t + 0.1)])),
            cm(video(keyframes=[Keyframe(timestamp=3.0)])),
        ]
    if previous is None:
        os.environ.pop("AEGIS_CAPTION_MATCH_METHOD", None)
    else:
        os.environ["AEGIS_CAPTION_MATCH_METHOD"] = previous
    get_settings.cache_clear()

    rc = rules.recycled_context_rule
    for modality in ("image", "video"):
        make = image if modality == "image" else video
        stamp = {"frame_timestamp": 3.0} if modality == "video" else {}
        dated, undated = page("2019-03-04", **stamp), page(None, **stamp)
        posted = lambda d: Meta(modality=Modality.IMAGE if modality == "image" else Modality.VIDEO, posted_date=d)  # noqa: E731
        flags += [
            rc(make(extractor_status={"reverse_image": FlagStatus.NOT_ASSESSED}, keyframes=[Keyframe(timestamp=3.0)])),
            rc(make(extractor_status={"reverse_image": FlagStatus.CLEAR}, keyframes=[Keyframe(timestamp=3.0)])),
            rc(make(web_matches=[dated], extractor_status=found)),
            rc(make(web_matches=[undated], extractor_status=found)),
            rc(make(web_matches=[dated], extractor_status=found, meta=posted("2024-01-01"))),
            rc(make(web_matches=[page("2025-01-01", **stamp), undated], extractor_status=found, meta=posted("2024-01-01"))),
            rc(make(web_matches=[page("2025-01-01", **stamp)], extractor_status=found, meta=posted("2024-01-01"))),
        ]
    flags.append(rc(image(extractor_status={"reverse_image": FlagStatus.CLEAR},
                          extractor_detail={"web_search": "The web search found no page showing this image."})))

    ef = rules.emotional_framing_rule
    flags += [ef(image()), ef(image(caption="SHOCKING!! Share before they delete it!!")),
              ef(image(caption="The bus was late!!")), ef(image(caption="The bus was late."))]
    seg = lambda s, sim: TranscriptSegment(start=s, end=s + 3, text="A barge on a river.", picture_similarity=sim, frame_timestamps=[s])  # noqa: E731
    flags += [ef(video(extractor_detail={"speech": "The video has no audio track."})),
              ef(video(caption="SHOCKING!! Share before they delete it!!")),
              ef(video(caption="A calm river.", transcript_segments=[seg(1.0, 0.3)]))]

    av = rules.audio_visual_mismatch_rule
    flags += [av(video(extractor_detail={"speech": "No speech was found in the audio."})),
              av(video(transcript_segments=[TranscriptSegment(start=1.0, end=3.0, text="A barge.")])),
              av(video(transcript_segments=[seg(1.0, rules.SPEECH_PICTURE_THRESHOLD - 0.05), seg(9.0, 0.3)])),
              av(video(transcript_segments=[seg(1.0, 0.3)]))]

    for verdict in (ReasonerVerdict(None, None, "LLM unavailable.", available=False),
                    ReasonerVerdict(False, True, "They differ.", available=True),
                    ReasonerVerdict(True, True, "They agree.", available=True),
                    ReasonerVerdict(None, None, "Unclear.", available=True)):
        original = scorecard.reason_over_text
        scorecard.reason_over_text = lambda *a, v=verdict: v
        try:
            flags.append(scorecard._llm_caption_scene_flag(image(caption="A flood")))
        finally:
            scorecard.reason_over_text = original
    return flags


def main() -> None:
    flags = all_flags()
    lines = {}
    for f in flags:
        for field in ("plain_explanation", "what_to_check"):
            text = getattr(f, field)
            lines.setdefault(text, {"field": field, "check": f.type.value, "status": f.status.value})
    reading = [{**meta, "text": text, **fk_grade(text)} for text, meta in lines.items()]
    grades = [r["grade"] for r in reading if r["grade"] is not None]
    written = sorted({s for f in flags for s in (f.plain_explanation, f.what_to_check, f.evidence)})
    hits = []
    for text in written:
        for m in _VERDICT.finditer(text):
            hits.append({"word": m.group(0), "context": text[max(0, m.start() - 60):m.end() + 60]})
    out = {
        "reading_level": {
            "formula": "Flesch-Kincaid grade = 0.39 * words/sentence + 11.8 * syllables/word - 15.59",
            "syllable_rule": "vowel groups (a e i o u y) per word, less a silent final e (not -le), at least 1",
            "target": "grade 8 or lower",
            "lines": len(reading), "at_or_below_8": sum(g <= 8 for g in grades), "above_8": sum(g > 8 for g in grades),
            "median": sorted(grades)[len(grades) // 2] if grades else None, "max": max(grades) if grades else None,
            "items": sorted(reading, key=lambda r: -(r["grade"] or 0)),
        },
        "backend_verdict_words": {"words": VERDICT_WORDS, "lines_scanned": len(written), "hits": hits},
    }
    print(json.dumps(out, indent=1))  # ASCII escapes: safe on any console encoding


if __name__ == "__main__":
    main()
