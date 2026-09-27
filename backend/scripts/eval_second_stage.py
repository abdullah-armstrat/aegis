"""WP-1b comparison: which second-stage matcher recovers what the perceptual hash misses?

Stage one is the shipped lookup: 64-bit pHash with mirror lookup, match at a Hamming distance of
10 or less. A second stage runs only on a lookup where stage one found nothing. Variants:

  S0  pHash only (the current app)
  S1  pHash, then trim-and-rehash
  S2  pHash, then ORB + RANSAC
  S3  pHash, then CLIP image similarity
  S4  pHash, then trim-and-rehash, then ORB + RANSAC
  S5  pHash, then CLIP shortlist (top 3 known images) verified by ORB + RANSAC

Known images (the index): the 11 openly licensed originals of the robustness harness.
Genuine queries: each original under the 15 re-post transformations (165 queries).
Impostor queries: every original and copy compared with the other originals' entries, plus the
VERITE images in data/verite/images as unrelated real photographs, each also bordered and
screenshot-framed, because framed images are exactly the queries the second stage sees.

ORB and CLIP thresholds are set from the impostor scores measured here: the lowest value above
every impostor, with a floor for ORB. Setting a threshold on the data it is then reported on is
optimistic, so every number is PROVISIONAL - SAMPLE IMAGES until the dated originals and the
distractor set are collected and split.

Run: python backend/scripts/eval_second_stage.py [--with-clip] --json out.json
     (run once without CLIP and once with it: the peak memory of each run is reported)
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import psutil  # noqa: E402
from PIL import Image  # noqa: E402

from app.extractors.phash import hamming, phash_of_image  # noqa: E402
from scripts.eval_hash_robustness import _load_samples  # noqa: E402
from tests.eval.candidate_matchers import (  # noqa: E402
    ClipEmbedder,
    orb_features,
    orb_inliers,
    trim_to_content,
)
from tests.eval.image_transforms import TRANSFORMS, border, screenshot  # noqa: E402

HASH_T = 10
ORB_FLOOR = 12
SHORTLIST = 3
VERITE_DIR = _BACKEND.parent / "data" / "verite" / "images"


def features(img: Image.Image, clip: ClipEmbedder | None) -> dict:
    """Every feature any variant needs, with the time each took."""
    f, t = {}, {}
    s = time.perf_counter()
    f["h"] = phash_of_image(img)
    f["hm"] = phash_of_image(img.transpose(Image.Transpose.FLIP_LEFT_RIGHT))
    t["hash"] = time.perf_counter() - s
    s = time.perf_counter()
    trimmed = trim_to_content(img)
    f["th"] = phash_of_image(trimmed)
    f["thm"] = phash_of_image(trimmed.transpose(Image.Transpose.FLIP_LEFT_RIGHT))
    t["trim"] = time.perf_counter() - s
    s = time.perf_counter()
    f["orb"] = orb_features(img)
    t["orb_features"] = time.perf_counter() - s
    if clip is not None:
        s = time.perf_counter()
        f["clip"] = clip.embed(img)
        t["clip"] = time.perf_counter() - s
    f["t"] = t
    return f


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-clip", action="store_true")
    parser.add_argument("--json", default="")
    args = parser.parse_args()
    proc = psutil.Process()

    originals, _ = _load_samples()
    clip = ClipEmbedder() if args.with_clip else None

    index = {name: features(img, clip) for name, img in originals.items()}
    queries = []  # (origin or None, kind, features)
    for name, img in originals.items():
        queries.append((name, "original", features(img, clip)))
        for tname, fn in TRANSFORMS.items():
            queries.append((name, tname, features(fn(img), clip)))
    distractors = sorted(VERITE_DIR.glob("*")) if VERITE_DIR.exists() else []
    for path in distractors:
        img = Image.open(path).convert("RGB")
        for kind, fn in (("distractor", lambda i: i), ("distractor+border", border),
                         ("distractor+screenshot", screenshot)):
            queries.append((None, kind, features(fn(img), clip)))

    # Pairwise scores, with ORB matching time per pair.
    orb_time_per_pair = []
    scores = []
    for origin, kind, q in queries:
        row = {"origin": origin, "kind": kind, "t": q["t"], "per_entry": {}}
        for name, e in index.items():
            s = time.perf_counter()
            inl = orb_inliers(q["orb"], e["orb"])
            orb_time_per_pair.append(time.perf_counter() - s)
            row["per_entry"][name] = {
                "d1": min(hamming(q["h"], e["h"]), hamming(q["hm"], e["h"])),
                "dt": min(hamming(q["th"], e["h"]), hamming(q["thm"], e["h"])),
                "orb": inl,
                "clip": float(q["clip"] @ e["clip"]) if clip is not None else None,
            }
        scores.append(row)

    # Thresholds from the impostor scores (pairs whose entry is not the query's origin).
    imp = [s for r in scores for n, s in r["per_entry"].items() if n != r["origin"]]
    max_imp_orb = max(s["orb"] for s in imp)
    orb_k = max(ORB_FLOOR, max_imp_orb + 1)
    clip_t = None
    if clip is not None:
        max_imp_clip = max(s["clip"] for s in imp)
        clip_t = math.ceil((max_imp_clip + 0.005) * 100) / 100

    def lookup(row, variant) -> set[str]:
        pe = row["per_entry"]
        hit = {n for n, s in pe.items() if s["d1"] <= HASH_T}
        if hit or variant == "S0":
            return hit
        trim = {n for n, s in pe.items() if s["dt"] <= HASH_T}
        orb = {n for n, s in pe.items() if s["orb"] >= orb_k}
        if variant == "S1":
            return trim
        if variant == "S2":
            return orb
        if variant == "S4":
            return trim or orb
        ranked = sorted(pe, key=lambda n: pe[n]["clip"], reverse=True)
        if variant == "S3":
            return {n for n in pe if pe[n]["clip"] >= clip_t}
        if variant == "S5":
            return {n for n in ranked[:SHORTLIST] if pe[n]["orb"] >= orb_k}
        raise ValueError(variant)

    def lookup_time(row, variant) -> float:
        t = row["t"]
        total = t["hash"]
        if variant == "S0" or any(s["d1"] <= HASH_T for s in row["per_entry"].values()):
            return total
        orb_all = t["orb_features"] + len(index) * statistics.mean(orb_time_per_pair)
        orb_short = t["orb_features"] + SHORTLIST * statistics.mean(orb_time_per_pair)
        if variant == "S1":
            return total + t["trim"]
        if variant == "S2":
            return total + orb_all
        if variant == "S3":
            return total + t["clip"]
        if variant == "S4":
            trim_hit = any(s["dt"] <= HASH_T for s in row["per_entry"].values())
            return total + t["trim"] + (0 if trim_hit else orb_all)
        if variant == "S5":
            return total + t["clip"] + orb_short
        raise ValueError(variant)

    variants = ["S0", "S1", "S2", "S4"] + (["S3", "S5"] if clip is not None else [])
    impostor_pairs = sum(len(index) - (1 if r["origin"] else 0) for r in scores)
    out = {"label": "PROVISIONAL - sample images", "with_clip": clip is not None,
           "n_index": len(index), "n_genuine": sum(1 for r in scores if r["origin"] and r["kind"] != "original"),
           "n_distractor_images": len(distractors), "impostor_pairs": impostor_pairs,
           "thresholds": {"hash": HASH_T, "orb_min_inliers": orb_k, "clip_min_similarity": clip_t,
                          "max_impostor_orb_inliers": max_imp_orb,
                          "max_impostor_clip": round(max_imp_clip, 4) if clip is not None else None},
           "variants": {}}
    for v in variants:
        per_t = {}
        for tname in TRANSFORMS:
            rows = [r for r in scores if r["kind"] == tname]
            per_t[tname] = sum(r["origin"] in lookup(r, v) for r in rows) / len(rows)
        genuine = [r for r in scores if r["origin"] and r["kind"] != "original"]
        false_pairs = sum(len(lookup(r, v) - {r["origin"]}) for r in scores)
        times = sorted(lookup_time(r, v) * 1000 for r in scores)
        out["variants"][v] = {
            "match_rate": per_t,
            "overall_match_rate": sum(r["origin"] in lookup(r, v) for r in genuine) / len(genuine),
            "false_match_pairs": false_pairs,
            "false_match_rate": false_pairs / impostor_pairs,
            "lookup_ms_mean": round(statistics.mean(times), 1),
            "lookup_ms_p95": round(times[max(0, math.ceil(0.95 * len(times)) - 1)], 1),
        }
    genuine_orb = {t: statistics.median(r["per_entry"][r["origin"]]["orb"] for r in scores if r["kind"] == t)
                   for t in TRANSFORMS}
    out["genuine_orb_median_inliers"] = genuine_orb
    if clip is not None:
        out["genuine_clip_median"] = {t: round(statistics.median(r["per_entry"][r["origin"]]["clip"] for r in scores
                                                              if r["kind"] == t), 4) for t in TRANSFORMS}
    mem = proc.memory_info()
    out["peak_memory_mb"] = round(getattr(mem, "peak_wset", mem.rss) / 2**20, 1)

    th = out["thresholds"]
    print(f"\n=== WP-1b second stage — {out['label']} — CLIP {'on' if clip else 'off'} ===")
    print(f"index {out['n_index']} originals | genuine queries {out['n_genuine']} | "
          f"distractor images {out['n_distractor_images']} (x3 framings) | impostor pairs {impostor_pairs}")
    print(f"thresholds: hash <= {HASH_T}; ORB >= {th['orb_min_inliers']} inliers "
          f"(max impostor {th['max_impostor_orb_inliers']})"
          + (f"; CLIP >= {th['clip_min_similarity']} (max impostor {th['max_impostor_clip']})" if clip else ""))
    head = f"{'transform':<12}" + "".join(f"{v:>7}" for v in variants)
    print(head); print("-" * len(head))
    for tname in TRANSFORMS:
        print(f"{tname:<12}" + "".join(f"{out['variants'][v]['match_rate'][tname]:>7.2f}" for v in variants))
    print("-" * len(head))
    for label, key, fmt in (("ALL copies", "overall_match_rate", "{:>7.2f}"),
                            ("false pairs", "false_match_pairs", "{:>7}"),
                            ("ms mean", "lookup_ms_mean", "{:>7}"), ("ms p95", "lookup_ms_p95", "{:>7}")):
        print(f"{label:<12}" + "".join(fmt.format(out['variants'][v][key]) for v in variants))
    print(f"peak memory of this run: {out['peak_memory_mb']} MB")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
