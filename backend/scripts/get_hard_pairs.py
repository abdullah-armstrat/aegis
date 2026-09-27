"""Fresh hard cases for image matching: 20 pairs of different NASA photos of the same subject.

The procedure was fixed before anything was downloaded (the project's decision log, ADR-056). For
each subject the NASA Image and Video Library is searched (images only) and its results are taken in
the order given, skipping any item in dataset A, any item whose record or file metadata names a
third-party right (dataset A's rule, from get_dataset_a.py), any item without a JPEG or PNG file,
and any photo within 10 bits of the first one taken (the same photo); the first two left form the
pair. NASA's large rendition is downloaded where there is one, else the original.

Files go to data/hard_pairs/ (ignored); the manifest to data/labels/hard_pairs.csv.

Run from the repo root: python backend/scripts/get_hard_pairs.py
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

_BACKEND = Path(__file__).resolve().parents[1]
for path in (_BACKEND, _BACKEND / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from get_dataset_a import (  # noqa: E402
    API, LICENCE, LICENCE_URL, RIGHTS_FIELDS, THIRD_PARTY, USER_AGENT, author)

from app.extractors.phash import hamming, phash_of_image  # noqa: E402

ROOT = _BACKEND.parent
OUT_DIR = ROOT / "data" / "hard_pairs"
MANIFEST = ROOT / "data" / "labels" / "hard_pairs.csv"
SUBJECTS = [
    "Vehicle Assembly Building", "Launch Complex 39A", "Launch Complex 39B", "crawler-transporter",
    "Space Shuttle Atlantis", "Space Shuttle Endeavour", "Orion spacecraft", "Space Launch System rocket",
    "Hubble Space Telescope", "International Space Station", "Kennedy Space Center Visitor Complex",
    "Operations and Checkout Building", "Mission Control", "Saturn V", "Neutral Buoyancy Laboratory",
    "Pegasus barge", "Shuttle Landing Facility", "mobile launcher", "Space Station Processing Facility",
    "SOFIA aircraft",
]
SAME_PHOTO_BITS = 10
MAX_CANDIDATES = 100  # per subject: the search's first page, in its order
NASA = re.compile(r"\bnasa\b|nasa\.gov|\bPAO\b", re.I)


def third_party_rights(record: dict, meta: dict) -> list[str]:
    """Dataset A's test, clarified (ADR-056): NASA's own notices, the colour profile's copyright and
    a copyright flag set to False name no third party and are not counted."""
    marks = []
    for text in [str(record.get(k, "")) for k in ("description", "title", "photographer", "secondary_creator")]:
        if THIRD_PARTY.search(text):
            marks.append(f"record text: {THIRD_PARTY.search(text).group(0)!r}")
    for key, value in meta.items():
        value = str(value).strip()
        if not value or key == "AVAIL:Owner" or key.startswith("ICC_Profile:"):
            continue
        if key.endswith("CopyrightFlag") and value.lower() == "false":
            continue
        if (THIRD_PARTY.search(value) or RIGHTS_FIELDS.search(key)) and not NASA.search(value):
            marks.append(f"{key} = {value[:60]!r}")
    return marks


def candidate(client: httpx.Client, record: dict) -> dict | None:
    """The item's file and record, or None when the procedure skips it."""
    nasa_id = record["nasa_id"]
    meta = client.get(client.get(f"{API}/metadata/{nasa_id}").json()["location"]).json()
    if third_party_rights(record, meta):
        return None
    assets = [i["href"] for i in client.get(f"{API}/asset/{nasa_id}").json()["collection"]["items"]]
    url = next((a for a in assets if "~large." in a and re.search(r"\.(jpe?g|png)$", a, re.I)), None) \
        or next((a for a in assets if "~orig." in a and re.search(r"\.(jpe?g|png)$", a, re.I)), None)
    if url is None:
        return None
    url = url.replace("http://", "https://")
    resp = client.get(url)
    resp.raise_for_status()
    with Image.open(BytesIO(resp.content)) as img:
        img.load()
        phash = phash_of_image(img.convert("RGB"))
    return {"record": record, "data": resp.content, "url": url, "phash": phash}


def main() -> None:
    with open(ROOT / "data" / "labels" / "A_originals.csv", encoding="utf-8") as fh:
        in_a = {r["nasa_id"] for r in csv.DictReader(fh)}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=60, follow_redirects=True) as client:
        for n, subject in enumerate(SUBJECTS, 1):
            items = client.get(f"{API}/search", params={"q": subject, "media_type": "image"}).json()
            chosen = []
            for item in items["collection"]["items"][:MAX_CANDIDATES]:
                record = item["data"][0]
                if record["nasa_id"] in in_a or any(c["record"]["nasa_id"] == record["nasa_id"] for c in chosen):
                    continue
                found = candidate(client, record)
                if found is None:
                    continue
                if chosen and hamming(found["phash"], chosen[0]["phash"]) <= SAME_PHOTO_BITS:
                    continue
                chosen.append(found)
                if len(chosen) == 2:
                    break
            if len(chosen) < 2:
                sys.exit(f"{subject}: fewer than two usable photos in the first {MAX_CANDIDATES} results")
            for side, c in zip("ab", chosen):
                r = c["record"]
                suffix = Path(c["url"]).suffix.lower()
                file = f"P{n:02d}{side}_{re.sub(r'[^A-Za-z0-9._-]', '_', r['nasa_id'])}{suffix}"
                (OUT_DIR / file).write_bytes(c["data"])
                rows.append({"pair": f"P{n:02d}", "side": side, "subject": subject, "nasa_id": r["nasa_id"],
                             "title": r.get("title", ""), "date_created": str(r.get("date_created", ""))[:10],
                             "file": file, "phash": c["phash"], "sha256": hashlib.sha256(c["data"]).hexdigest(),
                             "source_url": f"https://images.nasa.gov/details/{r['nasa_id']}", "file_url": c["url"],
                             "author": author(r), "licence": LICENCE, "licence_url": LICENCE_URL,
                             "access_date": date.today().isoformat()})
            print(f"  {subject}: {chosen[0]['record']['nasa_id']} / {chosen[1]['record']['nasa_id']} "
                  f"({hamming(chosen[0]['phash'], chosen[1]['phash'])} bits apart)", flush=True)
    with open(MANIFEST, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} photos ({len(rows) // 2} pairs) to {OUT_DIR.relative_to(ROOT)} and {MANIFEST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
