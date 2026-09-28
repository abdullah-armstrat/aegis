"""Draw the dataset C sample from VERITE and download its images from their original URLs.

VERITE (Papadopoulos et al., 2024) pairs real captions with images labelled "true", "miscaptioned"
or "out-of-context". The annotation files (Apache-2.0) come from the authors' GitHub repository at
a pinned commit. The images are not in that repository, so each is fetched from the URL the authors
recorded, as their own preparation script does.

Steps:
  1. Fetch VERITE.csv and VERITE_articles.csv at the pinned commit into data/verite/.
  2. Shuffle each label's pairs with a fixed seed: the first 100 are the draw, the rest are that
     label's reserve, used in order.
  3. Download the draw's images. A download counts only if it decodes as an image of at least
     64 px on each side (so not an HTML error page or a tiny placeholder).
  4. Replace each pair whose image failed with the next reserve pair of that label whose image
     downloads; if the reserve runs out, report the shortfall instead of padding.
  5. Write the sample (IDs and labels only) to data/labels/C_verite_sample.csv, the seed and counts
     to data/labels/C_verite_sample_meta.json, and every attempt to data/verite/download_log.csv.

data/verite/ (images, captions, URLs) is git-ignored; only IDs, labels, the seed and this script are
committed. The true and miscaptioned pairs of one article share an image, so the sample keeps each
pair's article ID for any split that must keep articles together.

--fresh builds the fresh set instead: every VERITE pair whose article is not in the sample, for
testing on pairs no earlier evaluation has touched. Each image is tried once and a failed pair is
dropped (there are no reserves). Writes data/labels/C_verite_fresh.csv (IDs and labels only),
data/labels/C_verite_fresh_meta.json and data/verite/fresh_download_log.csv.

Run:  python backend/scripts/build_verite_sample.py [--fresh]
"""

from __future__ import annotations

import argparse
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
FRESH_CSV = ROOT / "data" / "labels" / "C_verite_fresh.csv"
FRESH_META = ROOT / "data" / "labels" / "C_verite_fresh_meta.json"
FRESH_LOG = WORK / "fresh_download_log.csv"
EXT = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp", "GIF": "gif", "BMP": "bmp", "MPO": "jpg"}


def fetch_annotations(client: httpx.Client) -> dict[str, str]:
    """Download the two annotation files at the pinned commit and return their SHA-256."""
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


def failure_reasons(attempts: list[dict]) -> dict[str, int]:
    """Count failed downloads by reason, ignoring the details in brackets."""
    return dict(Counter(re.sub(r"\(.*\)", "", r["reason"]).strip() for r in attempts if not r["ok"]))


def build_fresh() -> None:
    """Build the fresh set: every pair whose article is not in the sample and whose image downloads."""
    recorded = json.loads(OUT_META.read_text(encoding="utf-8"))["annotation_sha256"]
    for name, digest in recorded.items():
        if hashlib.sha256((WORK / name).read_bytes()).hexdigest() != digest:
            sys.exit(f"{name} differs from the file the sample was drawn from")
    with open(OUT_CSV, encoding="utf-8", newline="") as fh:
        sampled = {int(row["article_id"]) for row in csv.DictReader(fh)}
    fresh = [p for p in load_pairs() if p["article_id"] not in sampled]
    wanted = {p["image"]: p["url"] for p in fresh}

    IMAGES.mkdir(parents=True, exist_ok=True)
    timeout = httpx.Timeout(30.0, connect=15.0)
    with httpx.Client(headers={"User-Agent": USER_AGENT}, follow_redirects=True, timeout=timeout) as client:
        with ThreadPoolExecutor(max_workers=6) as pool:
            results = {rec["image"]: rec for rec in pool.map(lambda iu: download(client, *iu), sorted(wanted.items()))}
    kept = [p for p in fresh if results[p["image"]]["ok"]]

    with open(FRESH_CSV, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["verite_row", "article_id", "label", "image"])
        for p in sorted(kept, key=lambda p: (LABELS.index(p["label"]), p["verite_row"])):
            w.writerow([p["verite_row"], p["article_id"], p["label"], p["image"]])
    attempts = sorted(results.values(), key=lambda r: r["image"])
    with open(FRESH_LOG, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(attempts[0]))
        w.writeheader()
        w.writerows(attempts)
    labels = {label: {"in_fresh_articles": sum(p["label"] == label for p in fresh),
                      "image_failed": sum(p["label"] == label and not results[p["image"]]["ok"] for p in fresh),
                      "kept": sum(p["label"] == label for p in kept)} for label in LABELS}
    meta = {"source": f"https://github.com/{REPO}", "commit": COMMIT, "annotation_sha256": recorded,
            "excluded_sample_articles": len(sampled), "fresh_articles": len({p["article_id"] for p in fresh}),
            "kept_articles": len({p["article_id"] for p in kept}), "labels": labels,
            "images_attempted": len(attempts), "images_ok": sum(r["ok"] for r in attempts),
            "failure_reasons": failure_reasons(attempts)}
    FRESH_META.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print(f"fresh set: {len(fresh)} pairs from {meta['fresh_articles']} articles not in the sample")
    for label, c in labels.items():
        print(f"  {label:<15} {c['kept']}/{c['in_fresh_articles']} kept, {c['image_failed']} image failed")
    print(f"  images attempted {meta['images_attempted']}, downloaded {meta['images_ok']}")
    print(f"  failure reasons: {meta['failure_reasons']}")
    print(f"wrote {FRESH_CSV.relative_to(ROOT)}, {FRESH_META.relative_to(ROOT)}, {FRESH_LOG.relative_to(ROOT)}")


def main() -> None:
    """Draw the sample, or build the fresh set with --fresh."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh", action="store_true", help="build the fresh set instead of the sample")
    if parser.parse_args().fresh:
        return build_fresh()
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

        # Fetch the main draw for all labels first; reserve images only when they are needed.
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
        "failure_reasons": failure_reasons(attempts),
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
