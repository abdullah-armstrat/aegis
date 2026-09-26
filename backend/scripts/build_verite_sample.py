"""Draw the dataset C sample from VERITE and download its images from their original URLs.

VERITE (Papadopoulos et al., 2024) pairs real-world captions with images, labelled "true",
"miscaptioned" or "out-of-context". Its annotation files come from the authors' official GitHub
repository at a fixed commit; the images are not redistributed there, so each one is fetched from
the URL the authors recorded, exactly as their own preparation script does.

What this script does:
  1. Fetches VERITE.csv and VERITE_articles.csv from the pinned commit into data/verite/.
  2. Shuffles each label's pairs with a fixed seed. The first 100 are the draw; the rest of the
     shuffled list is that label's reserve, used strictly in order.
  3. Downloads every image the draw needs. A download counts only if it decodes as an image of at
     least 64 px on each side (an HTML error page or a tiny placeholder does not).
  4. Replaces each pair whose image failed with the next reserve pair of the same label whose
     image downloads. If a label's reserve runs out, it reports the shortfall rather than padding.
  5. Writes the sample (IDs and labels only) to data/labels/C_verite_sample.csv, the seed and
     counts to data/labels/C_verite_sample_meta.json, and every download attempt to
     data/verite/download_log.csv.

Images, captions and URLs stay in data/verite/, which git ignores. Only IDs, labels, the seed and
this script are committed. The truthful and miscaptioned pair of one article share an image, so
the sample records each pair's article ID for any later split that must keep articles together.

Run:  python backend/scripts/build_verite_sample.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

import httpx
from PIL import Image

REPO = "stevejpapad/image-text-verification"
COMMIT = "74adfa1c8e13a027bfc1146c775958e832bd1089"
SEED = 20260927
PER_LABEL = 100
LABELS = ["true", "miscaptioned", "out-of-context"]  # VERITE's own label names
MIN_SIDE = 64
USER_AGENT = "aegis-verite-sampler/1.0 (academic research; one-off download of the VERITE images)"

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "data" / "verite"
IMAGES = WORK / "images"
OUT_CSV = ROOT / "data" / "labels" / "C_verite_sample.csv"
OUT_META = ROOT / "data" / "labels" / "C_verite_sample_meta.json"
LOG_CSV = WORK / "download_log.csv"
EXT = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp", "GIF": "gif", "BMP": "bmp", "MPO": "jpg"}


def fetch_annotations(client: httpx.Client) -> dict[str, str]:
    """Download the two annotation files at the pinned commit; return their SHA-256."""
    WORK.mkdir(parents=True, exist_ok=True)
    digests = {}
    for name in ("VERITE.csv", "VERITE_articles.csv"):
        url = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/VERITE/{name}"
        resp = client.get(url)
        resp.raise_for_status()
        (WORK / name).write_bytes(resp.content)
        digests[name] = hashlib.sha256(resp.content).hexdigest()
    return digests


def load_pairs() -> list[dict]:
    """Every VERITE pair with its row ID, label, article ID and image URL."""
    with open(WORK / "VERITE_articles.csv", encoding="utf-8", newline="") as fh:
        articles = {row["id"]: row for row in csv.DictReader(fh)}
    pairs = []
    with open(WORK / "VERITE.csv", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            kind, article_id = re.fullmatch(r"images/(true|false)_(\d+)\.jpg", row["image_path"]).groups()
            pairs.append({
                "verite_row": int(row[""]),
                "label": row["label"],
                "article_id": int(article_id),
                "image": f"{kind}_{article_id}",
                "url": articles[article_id][f"{kind}_url"].strip(),
            })
    return pairs


def download(client: httpx.Client, image: str, url: str) -> dict:
    """Fetch one image; keep it only if it decodes and is not a tiny placeholder."""
    rec = {"image": image, "url": url, "ok": False, "reason": "", "http_status": "",
           "bytes": 0, "width": "", "height": "", "file": ""}
    if not url:
        rec["reason"] = "no URL recorded"
        return rec
    try:
        resp = client.get(url)
        rec["http_status"] = resp.status_code
        if resp.status_code != 200:
            rec["reason"] = f"HTTP {resp.status_code}"
            return rec
        data = resp.content
        rec["bytes"] = len(data)
        with Image.open(BytesIO(data)) as img:
            img.load()
            fmt, (w, h) = img.format, img.size
        rec.update(width=w, height=h)
        if min(w, h) < MIN_SIDE:
            rec["reason"] = f"too small ({w}x{h}), likely a placeholder"
            return rec
        name = f"{image}.{EXT.get(fmt, 'img')}"
        (IMAGES / name).write_bytes(data)
        rec.update(ok=True, file=name)
    except httpx.TimeoutException:
        rec["reason"] = "timeout"
    except httpx.HTTPError as exc:
        rec["reason"] = f"connection error: {type(exc).__name__}"
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        rec["reason"] = f"not a readable image: {type(exc).__name__}"
    return rec


def main() -> None:
    IMAGES.mkdir(parents=True, exist_ok=True)
    timeout = httpx.Timeout(30.0, connect=15.0)
    with httpx.Client(headers={"User-Agent": USER_AGENT}, follow_redirects=True, timeout=timeout) as client:
        digests = fetch_annotations(client)
        pairs = load_pairs()

        rng = random.Random(SEED)
        order = {}
        for label in LABELS:
            items = sorted((p for p in pairs if p["label"] == label), key=lambda p: p["verite_row"])
            rng.shuffle(items)
            order[label] = items

        results: dict[str, dict] = {}

        def get(images: list[tuple[str, str]]) -> None:
            todo = [(i, u) for i, u in dict(images).items() if i not in results]
            with ThreadPoolExecutor(max_workers=6) as pool:
                for rec in pool.map(lambda iu: download(client, *iu), todo):
                    results[rec["image"]] = rec

        # Primary draw first, all labels at once; reserves are fetched only when needed.
        get([(p["image"], p["url"]) for label in LABELS for p in order[label][:PER_LABEL]])

        chosen, summary = [], {}
        for label in LABELS:
            picked, failed_primary, failed_reserve, position = [], [], [], 0
            items = order[label]
            while len(picked) < PER_LABEL and position < len(items):
                pair = items[position]
                if pair["image"] not in results:
                    get([(pair["image"], pair["url"])])
                ok = results[pair["image"]]["ok"]
                if ok:
                    picked.append({**pair, "selection": "draw" if position < PER_LABEL else "reserve"})
                else:
                    (failed_primary if position < PER_LABEL else failed_reserve).append(pair["verite_row"])
                position += 1
            chosen += picked
            summary[label] = {
                "available_in_verite": len(items),
                "drawn": PER_LABEL,
                "draw_failed": len(failed_primary),
                "reserve_used": sum(1 for p in picked if p["selection"] == "reserve"),
                "reserve_failed": len(failed_reserve),
                "final": len(picked),
                "shortfall": PER_LABEL - len(picked),
                "failed_verite_rows": sorted(failed_primary + failed_reserve),
            }

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["verite_row", "article_id", "label", "image", "selection"])
        for p in sorted(chosen, key=lambda p: (LABELS.index(p["label"]), p["verite_row"])):
            w.writerow([p["verite_row"], p["article_id"], p["label"], p["image"], p["selection"]])

    attempts = sorted(results.values(), key=lambda r: r["image"])
    with open(LOG_CSV, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(attempts[0]))
        w.writeheader()
        w.writerows(attempts)

    arts = Counter(p["article_id"] for p in chosen if p["label"] in ("true", "miscaptioned"))
    meta = {
        "source": f"https://github.com/{REPO}", "commit": COMMIT, "annotation_sha256": digests,
        "seed": SEED, "per_label": PER_LABEL, "min_image_side_px": MIN_SIDE,
        "labels": summary,
        "images_attempted": len(attempts),
        "images_ok": sum(r["ok"] for r in attempts),
        "failure_reasons": dict(Counter(re.sub(r"\(.*\)", "", r["reason"]).strip()
                                        for r in attempts if not r["ok"])),
        "articles_with_both_true_and_miscaptioned_in_sample": sum(1 for n in arts.values() if n > 1),
    }
    OUT_META.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print(f"VERITE commit {COMMIT[:7]}, seed {SEED}")
    for label, s in summary.items():
        print(f"  {label:<15} final {s['final']}/{PER_LABEL}  draw failed {s['draw_failed']}  "
              f"reserve used {s['reserve_used']}  reserve failed {s['reserve_failed']}  "
              f"shortfall {s['shortfall']}")
    print(f"  images attempted {meta['images_attempted']}, downloaded {meta['images_ok']}")
    print(f"  failure reasons: {meta['failure_reasons']}")
    print(f"wrote {OUT_CSV.relative_to(ROOT)}, {OUT_META.relative_to(ROOT)}, {LOG_CSV.relative_to(ROOT)}")
    if any(s["shortfall"] for s in summary.values()):
        sys.exit("a label is short of its target; see the counts above")


if __name__ == "__main__":
    main()
