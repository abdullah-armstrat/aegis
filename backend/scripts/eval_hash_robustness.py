"""Check whether a re-posted copy of an image still matches its original by pHash.

Each original's pHash goes into a throwaway index, then every re-post transformation from
tests/eval/image_transforms.py is applied and the Hamming distance back to the original recorded.
Two curves over the threshold come out: the match rate per transformation (want high) and the
false-match rate on impostor pairs, i.e. a copy of one original against a different original
(want about 0).

Two lookup variants are measured:
  plain   - distance between the query hash and the entry hash;
  mirror  - the smaller of that and the distance from the query's horizontal mirror, so flipped
            re-posts can be found (pHash is not flip-invariant).

Options:
  --source samples   (default) openly licensed images bundled with scikit-image, loaded at run
                     time and never copied into the repository; licences are in SAMPLE_IMAGES.
                     Numbers from this source are only provisional.
  --source dataset   originals from data/labels/A_originals.csv and distractors from
                     B_distractors.csv, image files under --images-dir (git-ignored).

Run:  python backend/scripts/eval_hash_robustness.py [--json out.json]
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from PIL import Image, ImageOps  # noqa: E402

from app.extractors.phash import hamming, phash_of_image  # noqa: E402
from tests.eval.image_transforms import TRANSFORMS  # noqa: E402

_REPO = _BACKEND_DIR.parent
THRESHOLDS = [4, 6, 8, 10, 12, 14, 16, 18, 20]

# Picked from scikit-image's bundled data: each has an open licence in its docstring, no people
# and no medical images. Licence text is quoted from scikit-image 0.26.0 (read 2026-09-27).
# `cat` is left out because it is the same image as `chelsea`.
SAMPLE_IMAGES: dict[str, dict[str, str]] = {
    "brick": {"source": "CC0Textures (Bricks25) via skimage.data",
              "licence": "licensed under the Creative Commons CC0 License"},
    "chelsea": {"source": "Stefan van der Walt via skimage.data",
                "licence": "No copyright restrictions. CC0 by the photographer"},
    "clock": {"source": "Stefan van der Walt via skimage.data",
              "licence": "Released into the public domain by the photographer"},
    "coffee": {"source": "Rachel Michetti, courtesy of Pikolo Espresso Bar, via skimage.data",
               "licence": "No copyright restrictions. CC0 by the photographer"},
    "coins": {"source": "Brooklyn Museum Collection via skimage.data",
              "licence": "No known copyright restrictions"},
    "grass": {"source": "DeviantArt (linolafett, Grass-01) via skimage.data",
              "licence": "licensed under the Creative Commons CC0 License"},
    "gravel": {"source": "CC0Textures (Gravel04) via skimage.data",
               "licence": "licensed under the Creative Commons CC0 License"},
    "horse": {"source": "openclipart (Andreas Preuss, marauder) via skimage.data",
              "licence": "No copyright restrictions. CC0 given by owner"},
    "hubble_deep_field": {"source": "NASA / HubbleSite via skimage.data",
                          "licence": "may be freely used in the public domain"},
    "rocket": {"source": "SpaceX Photos (DSCOVR launch) via skimage.data",
               "licence": "released in the public domain"},
    "text": {"source": "Wikipedia (File:Corner.png) via skimage.data",
             "licence": "No known copyright restrictions, released into the public domain"},
}


def _load_samples() -> tuple[dict[str, Image.Image], dict[str, Image.Image]]:
    from skimage import data

    originals = {name: Image.fromarray(getattr(data, name)()).convert("RGB") for name in SAMPLE_IMAGES}
    return originals, {}  # distractors: the other originals and their copies (see below)


def _load_dataset(images_dir: Path) -> tuple[dict[str, Image.Image], dict[str, Image.Image]]:
    labels = _REPO / "data" / "labels"

    def read(csv_name: str) -> dict[str, Image.Image]:
        with open(labels / csv_name, newline="", encoding="utf-8") as fh:
            rows = [r for r in csv.DictReader(fh) if r.get("file")]
        return {r["file"]: Image.open(images_dir / r["file"]).convert("RGB") for r in rows}

    return read("A_originals.csv"), read("B_distractors.csv")


def _hashes(img: Image.Image) -> tuple[str, str]:
    """(hash, hash of the horizontal mirror)."""
    return phash_of_image(img), phash_of_image(ImageOps.mirror(img))


def _distance(query: tuple[str, str], entry_hash: str, mirror: bool) -> int:
    d = hamming(query[0], entry_hash)
    return min(d, hamming(query[1], entry_hash)) if mirror else d


def _rate(values: list[int], t: int) -> float:
    return sum(v <= t for v in values) / len(values) if values else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure pHash robustness to re-post edits.")
    parser.add_argument("--source", choices=["samples", "dataset"], default="samples")
    parser.add_argument("--images-dir", type=Path, default=_REPO / "data" / "raw")
    parser.add_argument("--json", type=str, default="", help="write full results to this path")
    args = parser.parse_args()

    originals, distractors = (
        _load_samples() if args.source == "samples" else _load_dataset(args.images_dir)
    )
    label = "PROVISIONAL - sample images" if args.source == "samples" else "dataset A/B"

    index = {name: phash_of_image(img) for name, img in originals.items()}
    queries: list[tuple[str, str, tuple[str, str]]] = []  # (origin, transform, hashes)
    for name, img in originals.items():
        queries.append((name, "original", _hashes(img)))
        for tname, fn in TRANSFORMS.items():
            queries.append((name, tname, _hashes(fn(img))))
    for dname, img in distractors.items():
        queries.append((f"distractor:{dname}", "distractor", _hashes(img)))

    results: dict = {"label": label, "n_originals": len(originals),
                     "n_distractor_images": len(distractors), "thresholds": THRESHOLDS,
                     "variants": {}}

    for variant, mirror in (("plain", False), ("mirror", True)):
        genuine: dict[str, list[int]] = {t: [] for t in TRANSFORMS}
        impostor: list[int] = []
        for origin, tname, qh in queries:
            for entry_name, entry_hash in index.items():
                d = _distance(qh, entry_hash, mirror)
                if origin == entry_name:
                    if tname in genuine:
                        genuine[tname].append(d)
                else:
                    impostor.append(d)
        results["variants"][variant] = {
            "per_transform": {
                t: {"distances": v, "median": statistics.median(v), "max": max(v),
                    "match_rate": {str(th): _rate(v, th) for th in THRESHOLDS}}
                for t, v in genuine.items()
            },
            "impostor_pairs": len(impostor),
            "impostor_min_distance": min(impostor),
            "impostor_median_distance": statistics.median(impostor),
            "impostor_p01_distance": sorted(impostor)[max(0, len(impostor) // 100 - 1)],
            "false_match_rate": {str(th): _rate(impostor, th) for th in THRESHOLDS},
            "false_matches": {str(th): sum(d <= th for d in impostor) for th in THRESHOLDS},
        }

    # ---- report ----
    print(f"\n=== pHash robustness - {label} ===")
    print(f"originals: {len(originals)}   transforms: {len(TRANSFORMS)}   "
          f"distractor images: {len(distractors)}")
    for variant in ("plain", "mirror"):
        v = results["variants"][variant]
        print(f"\n--- lookup variant: {variant} ---")
        head = f"{'transform':<12}{'med':>5}{'max':>5}  " + "".join(f"T<={t:<3}" for t in THRESHOLDS)
        print(head)
        print("-" * len(head))
        for t, m in v["per_transform"].items():
            rates = "".join(f"{m['match_rate'][str(th)]:>6.2f}" for th in THRESHOLDS)
            print(f"{t:<12}{m['median']:>5.0f}{m['max']:>5}  {rates}")
        print("-" * len(head))
        allgen = [d for m in v["per_transform"].values() for d in m["distances"]]
        print(f"{'ALL copies':<12}{statistics.median(allgen):>5.0f}{max(allgen):>5}  "
              + "".join(f"{_rate(allgen, th):>6.2f}" for th in THRESHOLDS))
        print(f"{'FALSE match':<22}"
              + "".join(f"{v['false_match_rate'][str(th)]:>6.3f}" for th in THRESHOLDS)
              + f"   ({v['impostor_pairs']} impostor pairs)")
        print(f"{'impostor distance':<22}min {v['impostor_min_distance']}, 1st percentile "
              f"{v['impostor_p01_distance']}, median {v['impostor_median_distance']:.0f} "
              f"(64-bit hashes: unrelated images centre near 32)")

    if args.source == "samples":
        print("\nSample images (loaded from scikit-image at run time; not in the repository):")
        for name, p in SAMPLE_IMAGES.items():
            print(f"  {name:<18} {p['source']}  -  \"{p['licence']}\"")
        results["provenance"] = SAMPLE_IMAGES

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
