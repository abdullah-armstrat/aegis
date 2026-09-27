"""Video adapter — turns a video file + caption into an :class:`EvidenceBundle`.

Stages, in order, each loading at most one model and releasing it before the next:

  keyframes      scene cuts (PySceneDetect), the middle frame of each scene, at most 8; even
                 intervals when there are no cuts
  ocr            Tesseract on every keyframe (an external program, no model in this process)
  reverse_image  the image history lookup (hash, then keypoints) on every keyframe
  speech         the audio track through Whisper, loaded for this stage only
  clip           CLIP ViT-B/32, loaded for this stage only: the caption against every keyframe

BLIP and spaCy are never loaded here, and the LLM is not used for video. Every item carries its
time in the video. A stage that cannot run records NOT_ASSESSED and the reason in
``extractor_detail``, so the rules can say why a check did not run.
"""

from __future__ import annotations

import gc
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from app.config import get_settings
from app.extractors.caption_match import IMAGE_MODEL
from app.extractors.ocr import extract_on_screen_text
from app.extractors.reverse_image import find_web_matches
from app.extractors.speech import transcribe
from app.extractors.video import (
    VideoError,
    extract_audio,
    grab_frame,
    probe,
    sample_keyframes,
    thumbnail,
)
from app.models import EvidenceBundle, FlagStatus, Keyframe, Meta, Modality, TranscriptSegment


@contextmanager
def _stage(record: list | None, name: str):
    """Time a stage and sample this process's peak memory while it runs (when recording)."""
    if record is None:
        yield
        return
    import psutil

    proc, peak, done = psutil.Process(), [0], threading.Event()

    def sample():
        while not done.is_set():
            peak[0] = max(peak[0], proc.memory_info().rss)
            done.wait(0.05)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    started = time.perf_counter()
    try:
        yield
    finally:
        done.set()
        sampler.join()
        peak[0] = max(peak[0], proc.memory_info().rss)
        record.append({"stage": name, "seconds": round(time.perf_counter() - started, 2),
                       "peak_mb": round(peak[0] / 2**20, 1)})


def _clip_scores(texts: list[str], frames: dict[float, bytes]) -> dict[tuple[int, float], float]:
    """CLIP similarity of each text with each frame. Loads CLIP once and releases it afterwards."""
    from io import BytesIO

    from PIL import Image

    from app.extractors import caption_match

    try:
        text_vecs = [caption_match.clip_text_vector(t, IMAGE_MODEL) for t in texts]
        scores = {}
        for t, png in frames.items():
            with Image.open(BytesIO(png)) as img:
                vec = caption_match.clip_image_vector(img.convert("RGB"), IMAGE_MODEL)
            for i, tv in enumerate(text_vecs):
                scores[(i, t)] = float(vec @ tv)
        return scores
    finally:
        caption_match._clip.cache_clear()
        gc.collect()


def build_video_bundle(
    path: Path,
    caption: str | None,
    source_ref: str | None = None,
    posted_date: str | None = None,
    stages: list | None = None,
) -> EvidenceBundle:
    """Run the video stages on a file and assemble the bundle. Raises VideoError if unreadable."""
    settings = get_settings()
    info = probe(path)
    status: dict[str, FlagStatus] = {}
    detail: dict[str, str] = {}

    with _stage(stages, "keyframes"):
        times, how = sample_keyframes(path, info.duration_s)
        frames: dict[float, bytes] = {}
        for t in times:
            try:
                frames[t] = grab_frame(path, t)
            except VideoError:
                continue
        status["keyframes"] = FlagStatus.FIRED if frames else FlagStatus.NOT_ASSESSED
        detail["keyframes"] = f"{len(frames)} keyframes by {how}" if frames else "No frame could be read from the video."
        keyframes = {t: Keyframe(timestamp=t, thumbnail=thumbnail(png)) for t, png in frames.items()}

    with _stage(stages, "ocr"):
        ocr_status = []
        for t, png in frames.items():
            result = extract_on_screen_text(png)
            keyframes[t].on_screen_text = result.lines
            ocr_status.append(result.status)
        if FlagStatus.FIRED in ocr_status:
            status["ocr"] = FlagStatus.FIRED
        elif ocr_status and all(s == FlagStatus.CLEAR for s in ocr_status):
            status["ocr"] = FlagStatus.CLEAR
            detail["ocr"] = "No on-screen text was found in the keyframes."
        else:
            status["ocr"] = FlagStatus.NOT_ASSESSED
            detail["ocr"] = "Text could not be read from the keyframes."

    with _stage(stages, "reverse_image"):
        matches, lookups = [], []
        for t, png in frames.items():
            result = find_web_matches(png)
            keyframes[t].phash = result.phash
            lookups.append(result.status)
            matches += [m.model_copy(update={"frame_timestamp": t}) for m in result.matches]
        if matches:
            status["reverse_image"] = FlagStatus.FIRED
        elif lookups and all(s == FlagStatus.CLEAR for s in lookups):
            status["reverse_image"] = FlagStatus.CLEAR
        else:
            status["reverse_image"] = FlagStatus.NOT_ASSESSED
            detail["reverse_image"] = (f"{lookups.count(FlagStatus.CLEAR)} of {len(frames)} keyframes could be "
                                       "looked up in the image history index." if frames else "No keyframes.")

    with _stage(stages, "speech"):
        segments: list[TranscriptSegment] = []
        if not info.has_audio:
            status["speech"] = FlagStatus.NOT_ASSESSED
            detail["speech"] = "The video has no audio track."
        else:
            with tempfile.TemporaryDirectory() as tmp:
                wav = Path(tmp) / "audio.wav"
                try:
                    extract_audio(path, wav)
                    result = transcribe(wav, settings.whisper_model)
                    status["speech"] = result.status
                    if result.detail:
                        detail["speech"] = result.detail
                    segments = [TranscriptSegment(start=s.start, end=s.end, text=s.text) for s in result.segments]
                except VideoError as exc:
                    status["speech"], detail["speech"] = FlagStatus.NOT_ASSESSED, str(exc)
            if status["speech"] == FlagStatus.CLEAR:
                status["speech"] = FlagStatus.NOT_ASSESSED  # audio, but no speech to check

    with _stage(stages, "clip"):
        texts = [caption.strip()] if caption and caption.strip() else []
        if settings.caption_match_method == "off":
            status["caption_match"], detail["caption_match"] = FlagStatus.NOT_ASSESSED, "The check is switched off."
        elif not texts:
            status["caption_match"], detail["caption_match"] = FlagStatus.NOT_ASSESSED, "There is no caption."
        elif not frames:
            status["caption_match"], detail["caption_match"] = FlagStatus.NOT_ASSESSED, "No keyframes could be read."
        else:
            try:
                scores = _clip_scores(texts, frames)
                for t in frames:
                    keyframes[t].caption_similarity = scores[(0, t)]
                status["caption_match"] = FlagStatus.FIRED
            except (FileNotFoundError, OSError, ValueError, ImportError, RuntimeError) as exc:
                status["caption_match"] = FlagStatus.NOT_ASSESSED
                detail["caption_match"] = f"The picture-matching model could not run: {exc}"

    ordered = [keyframes[t] for t in sorted(keyframes)]
    seen, on_screen = set(), []
    for k in ordered:
        for line in k.on_screen_text:
            if line not in seen:
                seen.add(line)
                on_screen.append(line)
    return EvidenceBundle(
        caption=caption,
        on_screen_text=on_screen,
        transcript=" ".join(s.text for s in segments) or None,
        transcript_segments=segments,
        keyframes=ordered,
        web_matches=matches,
        extractor_status=status,
        extractor_detail=detail,
        meta=Meta(modality=Modality.VIDEO, frame_timestamps=[k.timestamp for k in ordered],
                  source_ref=source_ref, posted_date=posted_date,
                  duration_s=round(info.duration_s, 2), has_audio=info.has_audio),
    )
