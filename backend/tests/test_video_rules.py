"""Tests for the rules on hand-built video bundles (no extractors run)."""

import pytest

from app.config import get_settings
from app.fusion.rules import (
    CAPTION_IMAGE_ONLY_THRESHOLD,
    caption_scene_mismatch_rule,
    emotional_framing_rule,
    recycled_context_rule,
)
from app.models import (
    EvidenceBundle,
    FlagStatus,
    Keyframe,
    Meta,
    Modality,
    TranscriptSegment,
    WebMatch,
)

LOW = CAPTION_IMAGE_ONLY_THRESHOLD - 0.05
HIGH = CAPTION_IMAGE_ONLY_THRESHOLD + 0.05


def _video(**kwargs) -> EvidenceBundle:
    kwargs.setdefault("meta", Meta(modality=Modality.VIDEO, duration_s=30.0, has_audio=True))
    return EvidenceBundle(**kwargs)


@pytest.fixture(autouse=True)
def _picture_method(monkeypatch):
    get_settings.cache_clear()
    monkeypatch.setenv("AEGIS_CAPTION_MATCH_METHOD", "image")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_caption_fires_when_no_keyframe_matches_and_cites_the_best_frame():
    b = _video(caption="A busy market", keyframes=[Keyframe(timestamp=3.0, caption_similarity=LOW - 0.02),
                                                  Keyframe(timestamp=12.0, caption_similarity=LOW)])
    flag = caption_scene_mismatch_rule(b)
    assert flag.status == FlagStatus.FIRED
    assert flag.timestamps == [12.0]
    assert "0:12" in flag.evidence
    assert "cannot catch a wrong name, place or date" in flag.plain_explanation


def test_caption_clear_when_at_least_one_keyframe_matches():
    b = _video(caption="A busy market", keyframes=[Keyframe(timestamp=3.0, caption_similarity=LOW),
                                                  Keyframe(timestamp=20.0, caption_similarity=HIGH)])
    flag = caption_scene_mismatch_rule(b)
    assert flag.status == FlagStatus.CLEAR
    assert flag.timestamps == [20.0]


def test_caption_not_assessed_without_a_caption_and_says_why():
    flag = caption_scene_mismatch_rule(_video(keyframes=[Keyframe(timestamp=3.0)]))
    assert flag.status == FlagStatus.NOT_ASSESSED
    assert "no caption" in flag.evidence


def test_recycled_matches_cite_the_frames_that_matched():
    b = _video(web_matches=[WebMatch(url="https://e.com/a", published_date="2019-03-04", hash_distance=3,
                                     frame_timestamp=14.5)],
               extractor_status={"reverse_image": FlagStatus.FIRED},
               keyframes=[Keyframe(timestamp=2.0), Keyframe(timestamp=14.5)])
    flag = recycled_context_rule(b)
    assert flag.status == FlagStatus.FIRED
    assert flag.timestamps == [14.5]
    assert flag.evidence.startswith("The keyframe at 0:14 matches a known image.")
    assert "A frame from this video" in flag.plain_explanation


def test_recycled_clear_counts_the_keyframes_looked_up():
    b = _video(extractor_status={"reverse_image": FlagStatus.CLEAR},
               keyframes=[Keyframe(timestamp=2.0), Keyframe(timestamp=9.0)])
    flag = recycled_context_rule(b)
    assert flag.status == FlagStatus.CLEAR
    assert "None of the 2 keyframes" in flag.evidence


def test_framing_is_judged_per_source_and_cites_the_frame():
    b = _video(caption="Harbour at dawn",
               keyframes=[Keyframe(timestamp=6.0, on_screen_text=["SHARE BEFORE THEY DELETE IT!!"])],
               transcript_segments=[TranscriptSegment(start=0, end=4, text="Boats leave the harbour.")])
    flag = emotional_framing_rule(b)
    assert flag.status == FlagStatus.FIRED
    assert flag.timestamps == [6.0]
    assert "the on-screen text at 0:06" in flag.evidence


def test_framing_reads_the_speech_and_cites_its_segment():
    b = _video(transcript_segments=[
        TranscriptSegment(start=0, end=3, text="The tide is out."),
        TranscriptSegment(start=5, end=9, text="Urgent! Share this now before they delete it!!"),
    ])
    flag = emotional_framing_rule(b)
    assert flag.status == FlagStatus.FIRED
    assert flag.timestamps == [5]
    assert "In the speech" in flag.evidence


def test_framing_clear_names_what_it_could_not_check():
    b = _video(caption="Harbour at dawn", extractor_detail={"speech": "The video has no audio track."})
    flag = emotional_framing_rule(b)
    assert flag.status == FlagStatus.CLEAR
    assert "speech was not checked (the video has no audio track)" in flag.evidence


# --- speech vs picture -----------------------------------------------------------------------

from app.extractors.video import segment_frame_times  # noqa: E402
from app.fusion.rules import SPEECH_PICTURE_THRESHOLD, audio_visual_mismatch_rule, run_rules  # noqa: E402
from app.models import FlagType  # noqa: E402


def test_linking_takes_the_midpoint_and_every_keyframe_inside():
    assert segment_frame_times(10.0, 16.0, [2.0, 11.0, 15.5, 20.0]) == [13.0, 11.0, 15.5]
    assert segment_frame_times(10.0, 16.0, [2.0, 20.0]) == [13.0]
    # A keyframe exactly at the midpoint is not counted twice; the ends are inside.
    assert segment_frame_times(10.0, 16.0, [10.0, 13.0, 16.0]) == [13.0, 10.0, 16.0]


def _spoken(*scores):
    return _video(transcript_segments=[
        TranscriptSegment(start=5.0 * i, end=5.0 * i + 3, text=f"line {i}", picture_similarity=s,
                          frame_timestamps=[5.0 * i + 1.5])
        for i, s in enumerate(scores)])


def test_speech_fires_on_a_weak_stretch_and_cites_its_moment():
    flag = audio_visual_mismatch_rule(_spoken(SPEECH_PICTURE_THRESHOLD + 0.05, SPEECH_PICTURE_THRESHOLD - 0.05))
    assert flag.type == FlagType.AUDIO_VISUAL_MISMATCH
    assert flag.status == FlagStatus.FIRED
    assert flag.timestamps == [5.0]
    assert "At 0:05" in flag.evidence and '"line 1"' in flag.evidence
    assert "seem to describe a different scene from the picture" in flag.plain_explanation
    assert "cannot catch a wrong detail" in flag.plain_explanation


def test_speech_clear_when_every_stretch_matches_and_still_states_the_limit():
    flag = audio_visual_mismatch_rule(_spoken(SPEECH_PICTURE_THRESHOLD + 0.05, SPEECH_PICTURE_THRESHOLD + 0.01))
    assert flag.status == FlagStatus.CLEAR
    assert flag.timestamps == [5.0]
    assert "cannot catch a wrong detail" in flag.plain_explanation


@pytest.mark.parametrize("reason", ["The video has no audio track.", "No speech was found in the audio."])
def test_speech_not_assessed_without_speech_and_says_why(reason):
    flag = audio_visual_mismatch_rule(_video(extractor_detail={"speech": reason, "speech_picture": reason}))
    assert flag.status == FlagStatus.NOT_ASSESSED
    assert reason[1:].rstrip(".") in flag.evidence


def test_speech_check_runs_for_video_only():
    assert FlagType.AUDIO_VISUAL_MISMATCH in {f.type for f in run_rules(_spoken(0.3))}
    image = EvidenceBundle(meta=Meta(modality=Modality.IMAGE))
    assert FlagType.AUDIO_VISUAL_MISMATCH not in {f.type for f in run_rules(image)}


def test_framing_names_why_on_screen_text_was_not_checked():
    """The evidence tells apart no frames, unreadable text and no text at all."""
    from app.models import FlagStatus as S

    no_frames = emotional_framing_rule(_video(caption="A calm river."))
    assert "no frame of the video could be read" in no_frames.evidence
    unread = emotional_framing_rule(_video(caption="A calm river.", keyframes=[Keyframe(timestamp=2.0)],
                                           extractor_status={"ocr": S.NOT_ASSESSED}))
    assert "the text in the keyframes could not be read" in unread.evidence
    absent = emotional_framing_rule(_video(caption="A calm river.", keyframes=[Keyframe(timestamp=2.0)],
                                           extractor_status={"ocr": S.CLEAR}))
    assert "no on-screen text was found in the keyframes" in absent.evidence
