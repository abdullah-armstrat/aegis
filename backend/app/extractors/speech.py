"""Speech: transcribe a WAV file with openai-whisper into segments with start and end times.

The model is read from Whisper's cache (checksum checked, downloaded only if
``allow_model_downloads`` is set), used for one call and then released, since the laptop cannot
hold every model at once. Decoding is English on the CPU. Segments that are probably not speech
or have no words are dropped, so background noise gives "no speech" rather than made-up words.
"""

from __future__ import annotations

import gc
import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path

from app.models import FlagStatus

WHISPER_CACHE = Path(os.path.expanduser("~/.cache/whisper"))
NO_SPEECH_PROB = 0.6  # Whisper's own default no-speech threshold
# Extra decoding settings, chosen on long files of LibriSpeech tuning utterances with pauses and
# trailing silence (not the test sets). Without conditioning each 30 s window on the previous
# text, Whisper made up no sentences there (2 with it), for a word error rate 0.05 points higher.
DECODING = {"condition_on_previous_text": False, "hallucination_silence_threshold": None}
# Whisper retries a poor decode by sampling at a higher temperature, using PyTorch's random
# generator. Seeding it before each run makes the same audio always give the same transcript.
DECODING_SEED = 20260928


@dataclass
class Segment:
    """One stretch of speech; times are in seconds from the start of the audio."""

    start: float
    end: float
    text: str
    no_speech_prob: float


@dataclass
class SpeechResult:
    """Result of transcribing a file. ``model`` is the Whisper model name used."""

    segments: list[Segment] = field(default_factory=list)
    status: FlagStatus = FlagStatus.NOT_ASSESSED
    detail: str = ""
    model: str = ""


def whisper_checkpoint(name: str) -> Path:
    """Return the cached Whisper checkpoint after checking it against OpenAI's published SHA-256.

    Raises FileNotFoundError if the file is missing and ValueError if the checksum is wrong.
    """
    import whisper

    url = whisper._MODELS[name]
    path = WHISPER_CACHE / os.path.basename(url)
    if not path.is_file():
        raise FileNotFoundError(f"Whisper {name} weights are not in {WHISPER_CACHE}")
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    if digest.hexdigest() != url.split("/")[-2]:
        raise ValueError(f"Whisper {name} weights at {path} do not match the published checksum")
    return path


def load_model(name: str):
    """Load a Whisper model from the checked local file.

    A missing file is only downloaded from OpenAI when ``allow_model_downloads`` is set.
    """
    import whisper

    from app.config import get_settings

    try:
        source = str(whisper_checkpoint(name))
    except FileNotFoundError:
        if not get_settings().allow_model_downloads:
            raise
        source = name  # a model name makes the whisper package download it into WHISPER_CACHE
    return whisper.load_model(source, device="cpu", download_root=str(WHISPER_CACHE))


def _times(segment: dict) -> tuple[float, float]:
    """A segment's start and end, taken from its first and last word.

    Whisper's own segment times drift by several seconds after a long pause, so speech would be
    matched to the wrong frame. They are only used when the segment has no word timings.
    """
    words = segment.get("words") or []
    if words:
        return float(words[0]["start"]), float(words[-1]["end"])
    return float(segment["start"]), float(segment["end"])


def run_model(model, audio: Path, **decoding) -> list[Segment]:
    """Transcribe with the fixed decoding settings and keep only the segments that are speech.

    ``decoding`` overrides ``DECODING`` (used when comparing settings).
    """
    import torch

    options = {**DECODING, **decoding}
    torch.manual_seed(DECODING_SEED)
    out = model.transcribe(str(audio), language="en", fp16=False, verbose=None, word_timestamps=True, **options)
    kept = []
    for s in out.get("segments", []):
        if s["text"].strip() and float(s["no_speech_prob"]) <= NO_SPEECH_PROB:
            start, end = _times(s)
            kept.append(Segment(round(start, 2), round(end, 2), s["text"].strip(), float(s["no_speech_prob"])))
    return kept


def transcribe(wav: Path, model_name: str) -> SpeechResult:
    """Transcribe ``wav``. The model is loaded, used once and released. Never raises."""
    try:
        model = load_model(model_name)
    except (ImportError, FileNotFoundError, ValueError, OSError, RuntimeError) as exc:
        return SpeechResult(detail="The speech recogniser could not be loaded.", model=model_name)
    try:
        segments = run_model(model, wav)
    except Exception:  # noqa: BLE001 - any decoding failure is "could not assess"
        return SpeechResult(detail="The speech recogniser stopped before it finished.", model=model_name)
    finally:
        del model
        gc.collect()
    if not segments:
        return SpeechResult(status=FlagStatus.CLEAR, detail="No speech was found in the audio.", model=model_name)
    return SpeechResult(segments=segments, status=FlagStatus.FIRED, model=model_name)
