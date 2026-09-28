"""Perceptual hashing, used by the recycled-context lookup to match images by content.

A recompressed, resized or lightly cropped copy stays a small Hamming distance from the original,
while unrelated images are far apart. We use ``imagehash.phash`` at its default size: 64 bits
from the low frequencies of a 32x32 DCT, stored as 16-character hex. Unreadable bytes give
``None`` with a reason, so the caller can report NOT_ASSESSED rather than "no match".
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
    """Hash raw image bytes, and also their horizontal mirror.

    pHash does not survive a flip (a mirrored copy is about half the bits away), so the mirror
    hash lets flipped re-posts match. The ``phash_mirror_lookup`` setting decides if it is used.
    """
    try:
        from PIL import Image, ImageOps, UnidentifiedImageError
    except ImportError as exc:  # pragma: no cover - environment guard
        return PHashResult(None, detail="The picture could not be processed to search for earlier copies.")

    try:
        image = Image.open(BytesIO(image_bytes))
        image.load()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        return PHashResult(None, detail="The picture could not be read, so earlier copies were not searched for.")

    try:
        return PHashResult(
            hash_hex=phash_of_image(image),
            mirrored_hex=phash_of_image(ImageOps.mirror(image.convert("RGB"))),
        )
    except Exception:  # noqa: BLE001 - any hashing failure is "could not assess"
        return PHashResult(None, detail="The picture could not be processed to search for earlier copies.")


def hamming(a_hex: str, b_hex: str) -> int:
    """Number of differing bits between two hex-encoded hashes of equal length."""
    if len(a_hex) != len(b_hex):
        raise ValueError(f"hash lengths differ: {len(a_hex)} vs {len(b_hex)}")
    return bin(int(a_hex, 16) ^ int(b_hex, 16)).count("1")
