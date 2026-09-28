"""Application settings.

Everything is read from ``AEGIS_*`` environment variables or a local, git-ignored ``.env`` file,
so no secrets are kept in code. ``.env.example`` lists them all.
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
    # comma-separated in the environment, split into a list below
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    # --- Feature flags ---
    # "local" (default): match against the offline image history index ("index" and "cache" are
    # older names for it). "live": each image upload also searches the web with Google Cloud
    # Vision (needs GOOGLE_VISION_API_KEY); with the key set, a single upload can still ask for it.
    # Video keyframes only ever use the local index. "off": no lookup, and the scorecard says so.
    reverse_image_mode: str = "local"
    # Where live web results are cached, by the image's SHA-256. Empty => data/live_cache/.
    live_cache_dir: str = ""
    use_llm: bool = False              # the LLM second opinion (Ollama), off by default
    # By default models are only loaded from files already on this machine, with no network
    # access. True lets a missing model be downloaded (CLIP and Whisper from OpenAI, BLIP from
    # the Hugging Face Hub).
    allow_model_downloads: bool = False
    use_captioner: bool = True         # BLIP scene captioning, local on CPU; false skips it
    # How the caption is compared with the picture. "image": CLIP picture-caption similarity,
    # fires when low. "meaning": that plus spaCy similarity between the caption and the scene
    # description and on-screen text, fires when both are low. "overlap": the older word overlap
    # with the scene description. "off": not run. Only the models the method needs are loaded.
    # "image" is the default because on unseen VERITE pairs it flagged few truthful captions
    # (3 of 48) while still catching 26 of 69 out-of-context images.
    caption_match_method: Literal["image", "meaning", "overlap", "off"] = "image"

    # --- Video ---
    # Whisper model for the video speech. "base" beat "tiny" on 100 LibriSpeech utterances (word
    # error rate 4.67% vs 7.77%). The faster tiny model would have been kept if within 2 points.
    whisper_model: Literal["tiny", "base"] = "base"
    # Upload limits. 100 MB is about a minute of phone video; longer clips are refused because
    # every stage gets slower with length on this CPU-only machine.
    video_max_mb: int = 100
    video_max_seconds: float = 60.0

    # --- Local LLM reasoner (Ollama) ---
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "phi3:mini"

    # --- Extractors ---
    # Full path to the Tesseract binary, since it is not on PATH on the dev machine.
    # Empty => rely on PATH.
    tesseract_cmd: str = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    # --- Recycled context: image history index ---
    # Path to the image history index. Empty => the packaged default in app/data/.
    image_index_path: str = ""
    # A match is a pHash Hamming distance at or below this (out of 64 bits). Provisional: taken
    # from the robustness curve on openly licensed sample images, to be re-checked once the
    # project's own dated originals and distractor images are collected.
    phash_match_threshold: int = 10
    # Also hash the horizontal mirror of the upload so flipped re-posts match (provisional too).
    phash_mirror_lookup: bool = True
    # Second stage, only when the hash finds nothing: ORB keypoints checked with a RANSAC
    # homography, which finds cropped, bordered and screenshot-framed copies the hash misses.
    keypoint_matching: bool = True
    # A match needs at least this many geometrically agreeing keypoint pairs. Provisional: well
    # above the highest count seen between unrelated sample images (6), to be confirmed on the
    # project's own originals and distractors.
    keypoint_min_inliers: int = 12

    @property
    def cors_origin_list(self) -> list[str]:
        """CORS origins as a list."""
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    """Return the settings, read once and cached."""
    return Settings()
