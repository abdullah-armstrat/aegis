"""Build dataset A: 40 photos with known dates, for the recycled-context evaluation.

Source: the NASA Image and Video Library (images-api.nasa.gov). Wikimedia Commons did not resolve
from this machine, so it was not used. NASA media are generally public domain (not subject to
copyright in the United States); items with any third-party copyright or credit are refused.

The 40 items were picked by hand: everyday scenes (weather, transport, animals, landscapes) with no
identifiable people. Eight have little texture (fog, plain sky, snow) because keypoint matching
struggles with those and the evaluation should show it.

For each item the script:
  1. reads NASA's record and the file metadata, and stops if anything names a copyright, a
     courtesy credit or a rights holder other than NASA;
  2. downloads the original (or the large rendition if the original is not JPEG/PNG or is over
     15 MB) into data/A_originals/, which git ignores;
  3. records NASA's date created, the EXIF capture date if the file has one, and the SHA-256.
The manifest goes to data/labels/A_originals.csv. Run build_image_index.py afterwards to index them.

Run:  python backend/scripts/get_dataset_a.py
"""

from __future__ import annotations

import csv
import hashlib
import re
import sys
from datetime import date
from io import BytesIO
from pathlib import Path

import httpx
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
IMAGES = ROOT / "data" / "A_originals"
MANIFEST = ROOT / "data" / "labels" / "A_originals.csv"
API = "https://images-api.nasa.gov"
USER_AGENT = ("aegis-dataset-builder/1.0 (University of London CM3070 student project; "
              "contact 189360416+abdullah-armstrat@users.noreply.github.com)")
LICENCE = "Public domain: NASA media, not subject to copyright in the United States"
LICENCE_URL = "https://www.nasa.gov/nasa-brand-center/images-and-media/"
MAX_ORIGINAL_BYTES = 15 * 2**20
CENTRES = {"KSC": "Kennedy Space Center", "JSC": "Johnson Space Center", "ARC": "Ames Research Center",
           "GRC": "Glenn Research Center", "GSFC": "Goddard Space Flight Center", "HQ": "NASA Headquarters",
           "LRC": "Langley Research Center", "MSFC": "Marshall Space Flight Center",
           "AFRC": "Armstrong Flight Research Center"}

# (NASA id, topic, low texture)
SELECTION = [
    ("KSC-2013-2815", "weather", False),       # storm cloud over Launch Complex 39
    ("KSC-2011-2584", "weather", False),       # thunderstorm cloud over the Vehicle Assembly Building
    ("KSC-2009-5073", "weather", False),       # rainbow behind the Vehicle Assembly Building
    ("KSC-2014-4413", "weather", False),       # rainbow over the turn basin
    ("KSC-2010-4566", "weather", False),       # morning rainbow over trees and buildings
    ("KSC-08pd2424", "weather", False),        # wind and rain from Tropical Storm Fay
    ("ast-01-042", "weather", False),          # cloud patterns over the Pacific, from orbit
    ("s31-77-078", "weather", False),          # thunderstorm over the Texas Gulf Coast, from orbit
    ("KSC-2009-1010", "weather", True),        # lighthouse in fog
    ("KSC-2009-6804", "weather", True),        # dense fog over Launch Complex 39
    ("KSC-2009-1011", "weather", True),        # sunrise glow over a foggy tree line
    ("KSC-08pd2426", "weather", True),         # windows boarded with plywood before a storm
    ("S77-28200", "transport", False),         # carrier aircraft and escorts above an orbiter on a lakebed
    ("KSC-98PC-632", "transport", False),      # helium tank rail car on the track
    ("KSC-98PC-572", "transport", False),      # railroad train on the track
    ("KSC01PD-1615", "transport", False),      # railroad locomotive
    ("KSC-98PC-629", "transport", False),      # train passing a launch pad
    ("S89-33292", "transport", False),         # aircraft lined up on an apron
    ("S82-28715", "transport", False),         # aircraft over clouds
    ("S62-09048", "transport", False),         # aircraft carrier from the air
    ("S89-51980", "transport", True),          # rollout along a road in morning fog
    ("KSC-2009-2111", "animals", False),       # wild pigs in a grassy field
    ("KSC-07pd0148", "animals", False),        # nine-banded armadillo
    ("KSC-2009-2848", "animals", False),       # snowy egrets in water
    ("KSC-2009-2846", "animals", False),       # roseate spoonbill wading
    ("KSC-2009-1911", "animals", False),       # Florida scrub jay
    ("KSC-07pd3143", "animals", False),        # dolphin surfacing
    ("KSC-2010-5875", "animals", False),       # wood stork wading at dawn
    ("KSC-02pd0542", "animals", False),        # raccoon in tall grass
    ("KSC-2011-2639", "weather", True),        # dead trees under an overcast sky (a far-off eagle)
    ("KSC-2009-3155", "animals", True),        # white bird in flight against plain blue sky
    ("sts054-152-189", "landscapes", False),   # Sahara Desert, from orbit
    ("sts068-220-033", "landscapes", False),   # river through desert in China, from orbit
    ("SL2-04-018", "landscapes", False),       # Colorado River and Grand Canyon, from orbit
    ("s36-81-054", "landscapes", False),       # Great Salt Lake, from orbit
    ("ast-30-2601", "landscapes", False),      # Cascade Mountains, from orbit
    ("KSC-2009-1363", "landscapes", False),    # fog clearing over the turn basin at sunrise
    ("KSC-2011-2953", "landscapes", False),    # sunrise over the Vehicle Assembly Building and water
    ("sts001-012-0350", "landscapes", False),  # snow-covered mountain ridges, from orbit
    ("sts059-219-065", "landscapes", True),    # snow-covered island in dark sea, from orbit
]

THIRD_PARTY = re.compile(r"copyright|©|\(c\)|all rights reserved|courtesy", re.I)
RIGHTS_FIELDS = re.compile(r"copyright|rights|usageterms|credit|byline|owner", re.I)
PHOTO_CREDIT = re.compile(r"(?:photo|image) credit:\s*([^\n.]+)", re.I)


def third_party_marks(record: dict, meta: dict) -> list[str]:
    """Anything suggesting the item is not NASA's own public-domain media."""
    marks = []
    for text in [str(record.get(k, "")) for k in ("description", "title", "photographer", "secondary_creator")]:
        if THIRD_PARTY.search(text):
            marks.append(f"record text: {THIRD_PARTY.search(text).group(0)!r}")
    for key, value in meta.items():
        value = str(value).strip()
        if THIRD_PARTY.search(value):
            marks.append(f"{key} mentions {THIRD_PARTY.search(value).group(0)!r}")
        elif RIGHTS_FIELDS.search(key) and value and key != "AVAIL:Owner" and "nasa" not in value.lower():
            marks.append(f"{key} = {value[:60]!r}")
    return marks


def author(record: dict) -> str:
    parts = [f"NASA {CENTRES.get(record.get('center', ''), record.get('center', ''))}".strip()]
    for field in ("photographer", "secondary_creator"):
        if record.get(field):
            parts.append(str(record[field]).replace("_", " "))
    credit = PHOTO_CREDIT.search(str(record.get("description", "")))
    if credit and credit.group(1).strip() not in parts:
        parts.append(credit.group(1).strip())
    return "; ".join(dict.fromkeys(parts))


def exif_capture_date(data: bytes) -> str:
    """DateTimeOriginal from the file's own EXIF, as ISO date and time, or '' if it has none."""
    with Image.open(BytesIO(data)) as img:
        exif = img.getexif()
        raw = exif.get_ifd(0x8769).get(36867) or exif.get(36867)
    if not raw:
        return ""
    return re.sub(r"^(\d{4}):(\d{2}):(\d{2})", r"\1-\2-\3", str(raw).strip()).replace(" ", "T")


def fetch(client: httpx.Client, nasa_id: str) -> dict:
    items = client.get(f"{API}/search", params={"nasa_id": nasa_id}).json()["collection"]["items"]
    record = next((i["data"][0] for i in items if i["data"][0]["nasa_id"] == nasa_id), None)
    if record is None:
        sys.exit(f"{nasa_id}: not found in the NASA library")
    meta = client.get(client.get(f"{API}/metadata/{nasa_id}").json()["location"]).json()
    marks = third_party_marks(record, meta)
    if marks:
        sys.exit(f"{nasa_id}: possible third-party rights, refused: {marks}")
    assets = [i["href"] for i in client.get(f"{API}/asset/{nasa_id}").json()["collection"]["items"]]
    original = next((a for a in assets if "~orig." in a), None)
    large = next((a for a in assets if "~large." in a), None)
    url, rendition = original, "original"
    if original is None or not re.search(r"\.(jpe?g|png)$", original, re.I):
        url, rendition = large, "large"
    else:
        head = client.head(original)
        if int(head.headers.get("content-length", 0)) > MAX_ORIGINAL_BYTES:
            url, rendition = large, "large"
    if url is None:
        sys.exit(f"{nasa_id}: no usable image file")
    resp = client.get(url.replace("http://", "https://"))
    resp.raise_for_status()
    return {"record": record, "data": resp.content, "url": url.replace("http://", "https://"),
            "rendition": rendition}


def main() -> None:
    ids = [s[0] for s in SELECTION]
    if len(SELECTION) != 40 or len(set(ids)) != 40:
        sys.exit("the selection must be 40 distinct items")
    IMAGES.mkdir(parents=True, exist_ok=True)
    rows = []
    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=60, follow_redirects=True) as client:
        for n, (nasa_id, topic, low_texture) in enumerate(SELECTION, 1):
            got = fetch(client, nasa_id)
            record, data = got["record"], got["data"]
            with Image.open(BytesIO(data)) as img:
                img.load()
                width, height, fmt = img.width, img.height, img.format
            ext = {"JPEG": "jpg", "PNG": "png"}[fmt]
            name = f"A{n:02d}_{nasa_id}.{ext}"
            (IMAGES / name).write_bytes(data)
            rows.append({
                "file": name, "nasa_id": nasa_id, "topic": topic, "low_texture": "yes" if low_texture else "no",
                "earliest_date": str(record["date_created"])[:10],
                "exif_capture_date": exif_capture_date(data),
                "sha256": hashlib.sha256(data).hexdigest(),
                "width": width, "height": height, "rendition": got["rendition"],
                "source_url": f"https://images.nasa.gov/details/{nasa_id}", "file_url": got["url"],
                "author": author(record), "licence": LICENCE, "licence_url": LICENCE_URL,
                "title": " ".join(str(record.get("title", "")).split()),
                "nasa_description": " ".join(str(record.get("description", "")).split()),
                "access_date": date.today().isoformat(),
            })
            print(f"  {name:<34} {topic:<10} {rows[-1]['earliest_date']}  exif {rows[-1]['exif_capture_date'] or '-':<19} "
                  f"{width}x{height} {got['rendition']}")
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} images to {IMAGES.relative_to(ROOT)} and {MANIFEST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
