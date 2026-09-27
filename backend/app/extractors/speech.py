"""Speech: transcribe a WAV with openai-whisper into segments with start and end times.

The model is read from Whisper's own cache folder, checked against the checksum OpenAI publishes,
and not downloaded unless ``allow_model_downloads`` is set. It is loaded for one call and released afterwards, because the laptop
cannot hold every model at once. Decoding is English on the CPU with Whisper's default settings.

Each segment's start and end are its first word's start and its last word's end, from Whisper's
word-level timestamps: Whisper's own segment times drift by several seconds after a long pause,
so a line of speech would be compared with the picture at the wrong moment.

A segment Whisper itself rates as probably not speech (its no-speech probability above 0.6,
Whisper's own default threshold) is dropped, as is one with no words, so background noise alone
comes back as "no speech" rather than as invented words. Never raises: a failure is a result with
NOT_ASSESSED and a reason.
"""

from __future__ import annotations

import gc
import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path

from app.models import FlagStatus

WHISPER_CACHE = Path(os.path.expanduser("~/.cache/whisper"))
NO_SPEECH_PROB = 0.6
# Decoding settings beyond the language and word timestamps, chosen on long files of LibriSpeech
# tuning utterances with pauses and trailing silence, never on the test sets (ADR-049): without
# conditioning each 30-second window on the text before it, Whisper invented no sentences there
# (2 with it), at a word error rate 0.05 points higher.
DECODING = {"condition_on_previous_text": False, "hallucination_silence_threshold": None}
# When a decode looks poor, Whisper retries by sampling at a higher temperature, drawing from
# PyTorch's random generator; seeding it before each transcription makes the same audio always give
# the same transcript (ADR-049).
DECODING_SEED = 20260928


@dataclass
class Segment:
    start: float
    end: float
    text: str
    no_speech_prob: float


@dataclass
class SpeechResult:
    segments: list[Segment] = field(default_factory=list)
    status: FlagStatus = FlagStatus.NOT_ASSESSED
    detail: str = ""
    model: str = ""


def whisper_checkpoint(name: str) -> Path:
    """The cached checkpoint for a Whisper model, checked against OpenAI's published checksum."""
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
    """A Whisper model from the verified local file (a file path never triggers a download).

    Only when ``allow_model_downloads`` is set does a missing file come from OpenAI's server.
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
    """A segment's start and end from its first and last word; Whisper's own times if it has no words."""
    words = segment.get("words") or []
    if words:
        return float(words[0]["start"]), float(words[-1]["end"])
    return float(segment["start"]), float(segment["end"])


def run_model(model, audio: Path, **decoding) -> list[Segment]:
    """Transcribe with the fixed decoding settings and keep only segments that are speech.

    ``decoding`` overrides ``DECODING``, for comparing settings.
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
    """Segments of speech in ``wav``; the model is loaded, used once and released."""
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
