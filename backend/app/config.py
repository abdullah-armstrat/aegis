"""Application configuration.

All settings are read from environment variables (or a local ``.env`` file, which is
git-ignored). Secrets never live in code. See ``.env.example`` for the full list.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed application settings, populated from the environment."""

    model_config = SettingsConfigDict(
        env_prefix="AEGIS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- CORS ---
    # Comma-separated string in the environment; parsed to a list below.
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    # --- Feature flags ---
    # "index": content-matched image history index, offline (the default). "cache" is the
    # earlier name for the offline mode and is accepted as an alias. "api" is reserved for a
    # live web lookup and reports NOT_ASSESSED until one exists.
    reverse_image_mode: str = "index"
    use_llm: bool = False              # rules-only fusion until the LLM is wired in
    use_captioner: bool = True         # BLIP scene captioning, local on CPU; false skips it
    # How the caption is compared with the picture. "image": CLIP similarity between the picture
    # and the caption; the flag fires when it is low. "meaning": that, and spaCy similarity
    # between the caption and the scene description plus on-screen text; fires when both are low.
    # "overlap": the earlier content-word overlap between the caption and the scene description.
    # "off": the check does not run and the scorecard says so. Only the models the chosen method
    # needs are loaded. "image" is the default because it was the only method that, on VERITE
    # pairs no earlier evaluation had touched, flagged few truthful captions (3 of 48) while
    # catching images used out of context (26 of 69).
    caption_match_method: Literal["image", "meaning", "overlap", "off"] = "image"

    # --- Video ---
    # Whisper model for the video path's speech. Base, by the tiny-vs-base comparison: word error
    # rate 4.67% against 7.77% on 100 LibriSpeech utterances, a gap above the 2-point margin that
    # would have let the faster tiny model win.
    whisper_model: Literal["tiny", "base"] = "base"
    # Upload limits. 100 MB holds a minute of phone video at ordinary bitrates; longer clips are
    # refused because every stage's cost grows with length on this CPU-only machine.
    video_max_mb: int = 100
    video_max_seconds: float = 60.0

    # --- Local LLM reasoner (Ollama) ---
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "phi3:mini"

    # --- Extractors ---
    # Absolute path to the Tesseract binary. It is not on PATH on the dev machine, so the
    # OCR extractor points pytesseract at it explicitly. Empty => rely on PATH.
    tesseract_cmd: str = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    # --- Recycled context: image history index ---
    # Path to the image history index. Empty => the packaged default in app/data/.
    image_index_path: str = ""
    # A match is a pHash Hamming distance at or below this (out of 64 bits). PROVISIONAL:
    # chosen from the robustness curve on openly licensed sample images; to be fixed once the
    # project's own dated originals and distractor images are collected.
    phash_match_threshold: int = 10
    # Also compare the hash of the upload's horizontal mirror, so flipped re-posts match.
    # PROVISIONAL, same basis as the threshold.
    phash_mirror_lookup: bool = True
    # Second stage, run only when the hash finds nothing: ORB keypoints confirmed by a RANSAC
    # homography, which finds cropped, bordered and screenshot-framed copies the hash misses.
    keypoint_matching: bool = True
    # A match needs at least this many geometrically agreeing keypoint pairs. PROVISIONAL: a
    # floor well above the highest score seen between unrelated images (6) in the sample-image
    # comparison; to be confirmed once the project's own originals and distractors are collected.
    keypoint_min_inliers: int = 12

    @property
    def cors_origin_list(self) -> list[str]:
        """CORS origins as a clean list."""
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()
