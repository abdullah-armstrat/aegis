"""Build the image history index the recycled-context lookup searches.

Until dataset A is collected, the shipped index is ILLUSTRATIVE. This script draws four synthetic
images — author-generated, deterministic from fixed seeds, no third-party content — into
``data/illustrative/`` and registers two of them in ``backend/app/data/image_history_index.json``
with the histories the earlier filename-keyed fixture held. Those sources are fictional
pages on example.com, a domain reserved for examples by RFC 2606; they were fictional in the
fixture too. The other two images are deliberately left out of the index, so they demonstrate the
"lookup ran, no match" path.

  flood_illustrative.png    registered, earliest appearance 2019-03-04 (was flood_recycled_2019.jpg)
  protest_illustrative.png  registered, earliest appearance 2017-06-22 (was protest_recycled.jpg)
  sunset_illustrative.png   not registered                              (was consistent_sunset.jpg)
  cat_illustrative.png      not registered                              (was studio_cat.jpg)

Rerunning regenerates byte-identical images and hashes; ``test_image_index.py`` checks that the
committed index still matches the committed images.

Run:  python backend/scripts/build_image_index.py
"""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

_BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from PIL import Image, ImageDraw, ImageFilter  # noqa: E402

from app.extractors.phash import hamming, phash_of_image  # noqa: E402

IMAGES_DIR = _BACKEND_DIR.parent / "data" / "illustrative"
INDEX_PATH = _BACKEND_DIR / "app" / "data" / "image_history_index.json"
W, H = 480, 320

# Histories carried over verbatim from tests/eval/fixtures/reverse_image_cache.json.
FLOOD_SOURCES = [
    {"url": "https://news.example.com/2019/03/severe-flooding-region-a",
     "title": "Severe flooding hits Region A", "published_date": "2019-03-04",
     "context": "Original 2019 flood coverage in a different country and event."},
    {"url": "https://factcheck.example.org/2021/recycled-flood-image",
     "title": "Fact check: this flood photo is from 2019, not this week",
     "published_date": "2021-08-17",
     "context": "Fact-checking article noting the image has been recycled."},
]
PROTEST_SOURCES = [
    {"url": "https://archive.example.com/2017/city-b-rally", "title": "Thousands rally in City B",
     "published_date": "2017-06-22",
     "context": "Earlier appearance of the same crowd photo for a different rally."},
]


def _texture(seed: int, scale: int) -> np.ndarray:
    """Smooth seeded noise in [0, 1], H x W: gives each image its own large-scale structure."""
    rng = np.random.default_rng(seed)
    small = rng.random((H // scale + 2, W // scale + 2))
    img = Image.fromarray((small * 255).astype(np.uint8)).resize((W, H), Image.Resampling.BICUBIC)
    return np.asarray(img.filter(ImageFilter.GaussianBlur(scale / 2)), dtype=float) / 255.0


def _flood() -> Image.Image:
    t = _texture(11, 40)
    arr = np.zeros((H, W, 3))
    horizon = int(H * 0.38)
    arr[:horizon] = [150, 170, 185]  # overcast sky
    water = np.stack([110 + 50 * t, 95 + 40 * t, 70 + 30 * t], axis=-1)  # muddy water
    arr[horizon:] = water[horizon:]
    img = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    for x, w, h in [(30, 70, 120), (120, 55, 90), (300, 90, 140), (410, 50, 80)]:
        d.rectangle([x, horizon - h // 2, x + w, horizon + 25], fill=(90, 80, 75))  # buildings
    return img


def _protest() -> Image.Image:
    rng = np.random.default_rng(22)
    img = Image.new("RGB", (W, H), (120, 118, 112))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, int(H * 0.25)], fill=(190, 195, 200))
    for _ in range(420):  # crowd
        x, y = rng.integers(0, W), rng.integers(int(H * 0.3), H)
        r = int(rng.integers(5, 11))
        c = tuple(int(v) for v in rng.integers(40, 230, 3))
        d.ellipse([x - r, y - r, x + r, y + r], fill=c)
    for x in (60, 250, 390):  # banners
        d.rectangle([x, int(H * 0.2), x + 90, int(H * 0.33)], fill=(235, 235, 225))
    return img


def _sunset() -> Image.Image:
    y = np.linspace(0, 1, H)[:, None]
    t = _texture(33, 60)
    arr = np.zeros((H, W, 3))
    arr[..., 0] = 250 - 90 * y + 10 * t
    arr[..., 1] = 140 - 100 * y + 10 * t
    arr[..., 2] = 90 + 60 * y
    arr[int(H * 0.62):] = [30, 30, 60]  # sea
    img = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
    ImageDraw.Draw(img).ellipse([W - 170, int(H * 0.40), W - 90, int(H * 0.40) + 80],
                                fill=(255, 210, 120))
    return img


def _cat() -> Image.Image:
    t = _texture(44, 30)
    arr = np.stack([200 + 30 * t] * 3, axis=-1)  # studio backdrop
    img = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    d.ellipse([120, 150, 300, 300], fill=(60, 55, 50))  # body
    d.ellipse([230, 90, 330, 190], fill=(60, 55, 50))  # head
    d.polygon([(238, 110), (250, 60), (272, 100)], fill=(60, 55, 50))  # ears
    d.polygon([(290, 100), (312, 60), (322, 112)], fill=(60, 55, 50))
    d.rectangle([60, 290, 420, 300], fill=(150, 140, 130))  # studio floor line
    return img


IMAGES = {
    "flood_illustrative.png": (_flood, "illustrative-flood", FLOOD_SOURCES),
    "protest_illustrative.png": (_protest, "illustrative-protest", PROTEST_SOURCES),
    "sunset_illustrative.png": (_sunset, None, None),
    "cat_illustrative.png": (_cat, None, None),
}


def main() -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    entries = []
    for filename, (draw, entry_id, sources) in IMAGES.items():
        img = draw()
        img.save(IMAGES_DIR / filename, format="PNG", optimize=True)
        # Hash what a user would upload: the saved file, read back.
        hashes[filename] = phash_of_image(Image.open(IMAGES_DIR / filename))
        if entry_id:
            entries.append({
                "id": entry_id,
                "phash": hashes[filename],
                "earliest_date": min(s["published_date"] for s in sources),
                "image": f"data/illustrative/{filename}",
                "sources": sources,
            })

    index = {
        "_about": (
            "Image history index for the recycled-context check (ADR-017). Each entry is a known "
            "image: its 64-bit pHash (16 hex chars), the earliest date it is known to have "
            "appeared, and the pages it appeared on. An upload matches an entry when their pHash "
            "Hamming distance is at or below the configured threshold. THESE ENTRIES ARE "
            "ILLUSTRATIVE: the images are author-generated by backend/scripts/build_image_index.py "
            "and the sources are fictional example.com pages (RFC 2606) carried over from the "
            "pre-WP-1 fixture. Real entries come from dataset A."
        ),
        "entries": entries,
    }
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    INDEX_PATH.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {len(IMAGES)} images to {IMAGES_DIR}")
    for name, h in hashes.items():
        size = (IMAGES_DIR / name).stat().st_size // 1024
        print(f"  {name:<26} phash={h}  {size} KB")
    print(f"wrote index with {len(entries)} entries to {INDEX_PATH}")
    print("pairwise Hamming distances (want all well above the match threshold):")
    for a, b in combinations(hashes, 2):
        print(f"  {a:<26} {b:<26} {hamming(hashes[a], hashes[b])}")


if __name__ == "__main__":
    main()
