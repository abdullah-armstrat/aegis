"""Perceptual hashing — the content fingerprint behind the recycled-context check (WP-1).

A perceptual hash summarises what an image *looks like* rather than its exact bytes, so a
recompressed, resized or lightly cropped copy lands a small Hamming distance from the original,
while an unrelated image lands far away. That is what lets the recycled-context lookup match
an image by its content instead of by the uploaded filename, which any re-post changes.

Uses pHash from ``imagehash`` at its default ``hash_size=8``: a 64-bit hash from the low
frequencies of a 32x32 DCT of the greyscale image. Hashes travel as 16-character hex strings.

Never raises: unreadable bytes become ``None`` with a reason, so the caller can report the
check as NOT_ASSESSED instead of silently treating "could not hash" as "no match" (ADR-009).
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

HASH_BITS = 64


@dataclass
class PHashResult:
    """A computed hash, or ``None`` with the reason it could not be computed."""

    hash_hex: str | None
    mirrored_hex: str | None = None  # hash of the horizontal mirror image
    detail: str = ""


def phash_of_image(image) -> str:
    """64-bit pHash of a PIL image as a 16-char hex string."""
    import imagehash

    return str(imagehash.phash(image.convert("RGB")))


def compute_phash(image_bytes: bytes) -> PHashResult:
    """Hash raw image bytes, plus the hash of their horizontal mirror.

    The mirror hash exists because pHash is not flip-invariant: a mirrored re-post lands about
    half the bits away from its original. Whether the lookup uses it is a configuration choice
    measured by the robustness harness, not assumed.
    """
    try:
        from PIL import Image, ImageOps, UnidentifiedImageError
    except ImportError as exc:  # pragma: no cover - environment guard
        return PHashResult(None, detail=f"Hashing deps missing: {exc}")

    try:
        image = Image.open(BytesIO(image_bytes))
        image.load()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        return PHashResult(None, detail=f"Image could not be read: {exc}")

    try:
        return PHashResult(
            hash_hex=phash_of_image(image),
            mirrored_hex=phash_of_image(ImageOps.mirror(image.convert("RGB"))),
        )
    except Exception as exc:  # noqa: BLE001 - any hashing failure is "could not assess"
        return PHashResult(None, detail=f"Hashing error: {exc}")


def hamming(a_hex: str, b_hex: str) -> int:
    """Number of differing bits between two hex-encoded hashes of equal length."""
    if len(a_hex) != len(b_hex):
        raise ValueError(f"hash lengths differ: {len(a_hex)} vs {len(b_hex)}")
    return bin(int(a_hex, 16) ^ int(b_hex, 16)).count("1")
