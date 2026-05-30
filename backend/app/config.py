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
    reverse_image_mode: str = "cache"  # "cache" (Prelim default) | "api"
    use_llm: bool = False              # rules-only fusion until the LLM is wired in
    use_captioner: bool = True         # BLIP scene captioning (ADR-013); set false to skip

    # --- Heavy-inference offload (ADR-008) ---
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "phi3:mini"

    # --- Extractors ---
    # Absolute path to the Tesseract binary. It is not on PATH on the dev machine, so the
    # OCR extractor points pytesseract at it explicitly. Empty => rely on PATH.
    tesseract_cmd: str = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    # Path to the cached reverse-image fixture used when reverse_image_mode == "cache"
    # (ADR-007). Empty => the extractor uses its packaged default location.
    reverse_image_cache_path: str = ""

    @property
    def cors_origin_list(self) -> list[str]:
        """CORS origins as a clean list."""
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()
