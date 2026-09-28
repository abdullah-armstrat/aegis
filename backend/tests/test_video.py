"""Tests for the video path: keyframe sampling, the video bundle and the upload limits.

Test videos are generated with ffmpeg's built-in sources (solid colours, tones, noise), so no
real footage is needed.
"""

import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import main
from app.adapters.video_adapter import build_video_bundle
from app.extractors.video import (
    KEYFRAME_CAP,
    VideoError,
    even_timestamps,
    keyframes_from_scenes,
    probe,
    sample_keyframes,
)
from app.fusion.scorecard import build_scorecard
from app.models import SCHEMA_VERSION, FlagStatus, FlagType, Modality

client = TestClient(main.app)


def _video(path: Path, colours: list[str], seconds: float = 2.0, audio: str | None = None) -> Path:
    """A small video of solid-colour scenes, optionally with an audio track."""
    args = ["ffmpeg", "-v", "error", "-y"]
    for c in colours:
        args += ["-f", "lavfi", "-i", f"color=c={c}:s=160x120:r=10:d={seconds}"]
    n = len(colours)
    if audio:
        args += ["-f", "lavfi", "-t", str(seconds * n), "-i", audio]
    graph = "".join(f"[{i}:v]" for i in range(n)) + f"concat=n={n}:v=1:a=0[v]"
    args += ["-filter_complex", graph, "-map", "[v]"]
    if audio:
        args += ["-map", f"{n}:a", "-c:a", "aac"]
    args += ["-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)]
    subprocess.run(args, check=True)
    return path


# --- keyframe sampler ------------------------------------------------------------------------

def test_even_sampling_takes_the_middle_of_equal_intervals():
    assert even_timestamps(20.0) == [3.333, 10.0, 16.667]
    assert even_timestamps(3.0) == [1.5]
    assert len(even_timestamps(600.0)) == KEYFRAME_CAP


def test_scene_middles_are_keyframes_and_no_cut_means_even_sampling():
    times, how = keyframes_from_scenes([(0, 4), (4, 10), (10, 12)], 12)
    assert (times, how) == ([2.0, 7.0, 11.0], "scene cuts")
    assert keyframes_from_scenes([], 20.0) == (even_timestamps(20.0), "even")
    assert keyframes_from_scenes([(0, 20.0)], 20.0) == (even_timestamps(20.0), "even")


def test_more_scenes_than_the_cap_keeps_the_longest_in_time_order():
    scenes = [(i, i + 1) for i in range(10)] + [(10, 20), (20, 25)]
    times, _ = keyframes_from_scenes(scenes, 25)
    assert len(times) == KEYFRAME_CAP
    assert times == sorted(times)
    assert 15.0 in times and 22.5 in times


def test_sampler_finds_the_cuts_in_a_real_video(tmp_path):
    path = _video(tmp_path / "cuts.mp4", ["red", "blue", "green"], seconds=2.0)
    times, how = sample_keyframes(path, probe(path).duration_s)
    assert how == "scene cuts"
    assert [round(t) for t in times] == [1, 3, 5]


def test_sampler_samples_evenly_when_there_is_no_cut(tmp_path):
    path = _video(tmp_path / "still.mp4", ["gray"], seconds=16.0)
    times, how = sample_keyframes(path, probe(path).duration_s)
    assert how == "even"
    assert times == even_timestamps(16.0)


# --- the bundle ------------------------------------------------------------------------------

def test_no_audio_track_leaves_the_speech_checks_not_assessed_and_says_why(tmp_path):
    path = _video(tmp_path / "silent.mp4", ["red", "blue"], seconds=2.0)
    bundle = build_video_bundle(path, caption=None, source_ref="silent.mp4")
    assert bundle.meta.modality == Modality.VIDEO
    assert bundle.meta.has_audio is False
    assert bundle.extractor_status["speech"] == FlagStatus.NOT_ASSESSED
    assert bundle.extractor_detail["speech"] == "The video has no audio track."
    assert [round(k.timestamp) for k in bundle.keyframes] == [1, 3]
    assert all(k.thumbnail and k.thumbnail.startswith("data:image/jpeg;base64,") for k in bundle.keyframes)
    card = build_scorecard(bundle)
    framing = next(f for f in card.flags if f.type == FlagType.EMOTIONAL_FRAMING)
    assert framing.status == FlagStatus.NOT_ASSESSED
    assert "no audio track" in framing.evidence
    assert [round(k.timestamp) for k in card.keyframes] == [1, 3]


def test_a_corrupt_file_is_refused_by_the_probe(tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"\x00\x01not a video at all" * 100)
    with pytest.raises(VideoError, match="could not be read as a video"):
        probe(bad)


# --- the upload limits -----------------------------------------------------------------------

def _post(path: Path, content_type: str = "video/mp4", caption: str = ""):
    with open(path, "rb") as fh:
        return client.post("/analyze/video", files={"video": (path.name, fh, content_type)},
                           data={"caption": caption})


def test_api_refuses_a_file_that_is_not_a_video_type(tmp_path):
    txt = tmp_path / "notes.txt"
    txt.write_text("hello")
    resp = _post(txt, content_type="text/plain")
    assert resp.status_code == 415
    assert "mp4, mov or webm" in resp.json()["detail"]


def test_api_refuses_a_corrupt_video_with_a_clear_message(tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"\x00\x01not a video at all" * 100)
    resp = _post(bad)
    assert resp.status_code == 400
    assert resp.json()["detail"] == "The file could not be read as a video."


def test_api_refuses_a_clip_over_60_seconds(tmp_path):
    long = _video(tmp_path / "long.mp4", ["gray"], seconds=61.0)
    resp = _post(long)
    assert resp.status_code == 400
    assert "the limit is 60 s" in resp.json()["detail"]
    assert "61.0 s long" in resp.json()["detail"]


def test_api_refuses_a_file_over_the_size_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(main.settings, "video_max_mb", 0)
    path = _video(tmp_path / "small.mp4", ["red"], seconds=1.0)
    resp = _post(path)
    assert resp.status_code == 413
    assert "0 MB limit" in resp.json()["detail"]


def test_api_returns_a_video_scorecard(tmp_path):
    path = _video(tmp_path / "ok.mp4", ["red", "blue"], seconds=2.0)
    resp = _post(path)
    assert resp.status_code == 200
    body = resp.json()
    assert body["modality"] == "video"
    assert body["schema_version"] == SCHEMA_VERSION
    assert len(body["keyframes"]) == 2
    assert {f["type"] for f in body["flags"]} >= {"caption_content_mismatch", "recycled_context", "emotional_framing"}


# --- speech (loads Whisper) ------------------------------------------------------------------

@pytest.mark.slow
def test_noise_only_audio_is_no_speech_not_invented_words(tmp_path):
    path = _video(tmp_path / "noise.mp4", ["gray"], seconds=12.0,
                  audio="anoisesrc=color=brown:amplitude=0.05:seed=3")
    bundle = build_video_bundle(path, caption=None)
    assert bundle.meta.has_audio is True
    assert bundle.transcript_segments == []
    assert bundle.extractor_status["speech"] == FlagStatus.NOT_ASSESSED
    assert bundle.extractor_detail["speech"] == "No speech was found in the audio."


@pytest.mark.slow
def test_clip_stays_loaded_between_videos_and_whisper_is_loaded_for_each(tmp_path, monkeypatch):
    from app.config import get_settings
    from app.extractors import caption_match, speech

    path = _video(tmp_path / "tone.mp4", ["red", "blue"], seconds=2.0, audio="sine=frequency=440")
    loads = []
    real = speech.load_model
    monkeypatch.setattr(speech, "load_model", lambda name: loads.append(name) or real(name))
    caption_match._clip.cache_clear()
    for _ in range(2):
        build_video_bundle(path, caption="A red screen, then a blue one.", source_ref="tone.mp4")
    clip = caption_match._clip.cache_info()
    assert (clip.misses, clip.currsize) == (1, 1)  # loaded once, kept for the second video
    assert loads == [get_settings().whisper_model] * 2  # Whisper loaded once per video


def test_speech_times_come_from_the_first_and_last_word():
    """Whisper's segment times drift after a pause but its word times don't, so words are used."""
    from app.extractors.speech import run_model

    class FakeWhisper:
        def transcribe(self, audio, **options):
            self.options = options
            return {"segments": [
                {"start": 0.0, "end": 7.0, "text": " The barge approaches.", "no_speech_prob": 0.1,
                 "words": [{"word": " The", "start": 1.02, "end": 1.2}, {"word": " approaches.", "start": 1.9, "end": 2.98}]},
                {"start": 11.0, "end": 20.0, "text": " A tugboat pushes.", "no_speech_prob": 0.2, "words": []},
                {"start": 20.0, "end": 22.0, "text": " hum", "no_speech_prob": 0.9,
                 "words": [{"word": " hum", "start": 20.5, "end": 21.0}]},
            ]}

    from app.extractors.speech import DECODING

    model = FakeWhisper()
    segments = run_model(model, Path("audio.wav"))
    assert model.options["word_timestamps"] is True
    assert {k: model.options[k] for k in DECODING} == DECODING  # the chosen decoding settings
    run_model(model, Path("audio.wav"), condition_on_previous_text=False)
    assert model.options["condition_on_previous_text"] is False  # a setting can be overridden
    assert [(s.start, s.end, s.text) for s in segments] == [
        (1.02, 2.98, "The barge approaches."),  # from its words
        (11.0, 20.0, "A tugboat pushes."),      # no words: Whisper's own times
    ]                                           # the third is not speech (no-speech probability 0.9)


def test_the_same_audio_always_gives_the_same_transcript():
    """Whisper's temperature fallback uses PyTorch's random generator, which is seeded before
    each transcription."""
    import torch

    from app.extractors.speech import run_model

    class Sampling:
        def transcribe(self, audio, **options):
            draw = torch.rand(1).item()
            return {"segments": [{"start": 0.0, "end": 1.0, "text": f" {draw:.8f}", "no_speech_prob": 0.1, "words": []}]}

    first = run_model(Sampling(), Path("audio.wav"))[0].text
    torch.rand(7)  # something else uses the generator in between
    assert run_model(Sampling(), Path("audio.wav"))[0].text == first


@pytest.mark.parametrize("reader, marker, message", [
    ("grab_frame", "-frames:v", "could not be read"),
    ("extract_audio", "-vn", "The audio track could not be read."),
])
def test_an_ffmpeg_timeout_is_a_read_failure_not_a_crash(tmp_path, monkeypatch, reader, marker, message):
    import app.extractors.video as video

    real = subprocess.run

    def slow(args, *a, **kw):
        if marker in args:
            raise subprocess.TimeoutExpired(args, kw.get("timeout", 60))
        return real(args, *a, **kw)

    monkeypatch.setattr(subprocess, "run", slow)
    path = _video(tmp_path / "clip.mp4", ["red"], seconds=2.0, audio="sine=frequency=440")
    with pytest.raises(VideoError, match=message):
        if reader == "grab_frame":
            video.grab_frame(path, 1.0)
        else:
            video.extract_audio(path, tmp_path / "a.wav")
