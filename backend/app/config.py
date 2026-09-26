"""Application configuration.

All settings are read from environment variables (or a local ``.env`` file, which is
git-ignored). Secrets never live in code. See ``.env.example`` for the full list.
"""

from __future__ import annotations

from functools import lru_cache

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

    @property
    def cors_origin_list(self) -> list[str]:
        """CORS origins as a clean list."""
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()
