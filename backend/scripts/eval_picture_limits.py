"""WP-5: the picture check's limits (nearly blank, mostly text) on the VERITE pairs.

The definitions were fixed before anything was measured (the project's decision log): nearly blank
is a greyscale standard deviation under 10; mostly text is the words OCR reads covering more than
40% of the picture. Both are measured by the app's own functions.

  measure   Both measures for every image of the 300-pair sample and of the fresh set, with the
            app's own OCR. Cached in data/verite/features/picture_limits.json (ignored).
  report    Pairs and distinct images excluded, per label and reason, and the picture-only rule
            and the meaning rule (thresholds unchanged) on both sets before and after, through the
            app's rule code, with excluded pairs not assessed.

Run from the repo root: python backend/scripts/eval_picture_limits.py measure|report
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))
sys.path.insert(0, str(_BACKEND / "scripts"))

from eval_caption_match import CACHE, ROOT, cached, load_pairs, load_scores, wilson  # noqa: E402

LIMITS = CACHE / "picture_limits.json"
OUT = ROOT / "results" / "wp5_picture_limits.json"
LABELS = ("true", "out-of-context", "miscaptioned")


def do_measure() -> None:
    from app.extractors.caption_match import picture_limit, picture_measures
    from app.extractors.ocr import extract_on_screen_text

    done = json.loads(LIMITS.read_text(encoding="utf-8")) if LIMITS.exists() else {}
    for pair_set in ("sample", "fresh"):
        images = {p["image"]: p["file"] for p in load_pairs(pair_set)}
        for n, (image, path) in enumerate(sorted(images.items()), 1):
            if image in done:
                continue
            data = path.read_bytes()
            lines = extract_on_screen_text(data).lines
            spread, share = picture_measures(data, lines)
            done[image] = {"grey_std": spread, "text_share": share, "ocr_lines": len(lines),
                           "limit": picture_limit(data, lines)}
            if n % 25 == 0:
                LIMITS.write_text(json.dumps(done, indent=1), encoding="utf-8")
                print(f"  {pair_set}: {n}/{len(images)}")
    LIMITS.write_text(json.dumps(done, indent=1), encoding="utf-8")
    print(f"measured {len(done)} images")


def _reason(limit: str | None) -> str | None:
    if limit is None:
        return None
    return "nearly blank" if "nearly blank" in limit else "mostly text"


def _rate(k: int, n: int) -> dict:
    return {"k": k, "n": n, "rate": round(k / n, 3) if n else None, "ci": wilson(k, n) if n else None}


def do_report() -> None:
    import os

    from app.config import get_settings
    from app.fusion import rules
    import math

    from app.models import CaptionMatch, EvidenceBundle, FlagStatus, Meta, Modality, SceneDescription

    limits = json.loads(LIMITS.read_text(encoding="utf-8"))
    res = {"threshold": rules.CAPTION_IMAGE_ONLY_THRESHOLD, "definitions": {
        "nearly_blank": "greyscale standard deviation under 10",
        "mostly_text": "OCR word boxes cover more than 40% of the image"}}
    for pair_set in ("sample", "fresh"):
        pairs = load_pairs(pair_set)
        scores, _, _ = load_scores(pairs, pair_set, arches=("ViT-B/32",))
        feats = json.loads(cached("blip_ocr.json", pair_set).read_text(encoding="utf-8"))
        excluded = {p["row"]: _reason(limits[p["image"]]["limit"]) for p in pairs}

        def status(p, after: bool, method: str) -> FlagStatus:
            limit = limits[p["image"]]["limit"] if after else None
            text = scores["txt:spaCy"][p["row"]]
            if limit:
                match = CaptionMatch(detail=limit)
            elif method == "image":
                match = CaptionMatch(image_similarity=scores["img:ViT-B/32"][p["row"]])
            else:
                match = CaptionMatch(image_similarity=scores["img:ViT-B/32"][p["row"]],
                                     text_similarity=None if math.isnan(text) else text)
            f = feats[p["image"]]
            bundle = EvidenceBundle(caption=p["caption"], caption_match=match, on_screen_text=f["ocr"],
                                    scene_descriptions=[SceneDescription(text=f["scene"])] if f["scene"] else [],
                                    meta=Meta(modality=Modality.IMAGE))
            return rules.caption_scene_mismatch_rule(bundle).status

        entry = {"pairs": len(pairs), "per_label": {}, "rule": {}}
        for lab in LABELS:
            sub = [p for p in pairs if p["label"] == lab]
            images = {p["image"] for p in sub}
            entry["per_label"][lab] = {
                "pairs": len(sub), "images": len(images),
                "pairs_excluded": sum(excluded[p["row"]] is not None for p in sub),
                "pairs_nearly_blank": sum(excluded[p["row"]] == "nearly blank" for p in sub),
                "pairs_mostly_text": sum(excluded[p["row"]] == "mostly text" for p in sub),
                "images_excluded": len({p["image"] for p in sub if excluded[p["row"]]}),
                "excluded_images": sorted({p["image"] for p in sub if excluded[p["row"]]})}
        all_images = {p["image"] for p in pairs}
        entry["images"] = len(all_images)
        entry["images_excluded"] = sorted(i for i in all_images if limits[i]["limit"])
        for method, when in (("image", "before"), ("image", "after"), ("meaning", "before"), ("meaning", "after")):
            os.environ["AEGIS_CAPTION_MATCH_METHOD"] = method
            get_settings.cache_clear()
            st = {p["row"]: status(p, when == "after", method) for p in pairs}
            row = {}
            for lab, name in (("true", "truthful_flagged"), ("out-of-context", "out_of_context_caught"),
                              ("miscaptioned", "miscaptioned_caught")):
                sub = [p for p in pairs if p["label"] == lab]
                fired = sum(st[p["row"]] == FlagStatus.FIRED for p in sub)
                assessed = [p for p in sub if st[p["row"]] != FlagStatus.NOT_ASSESSED]
                row[name] = {**_rate(fired, len(sub)), "not_assessed": len(sub) - len(assessed),
                             "of_assessed": _rate(fired, len(assessed))}
            entry["rule"].setdefault(method, {})[when] = row
        res[pair_set] = entry
    get_settings.cache_clear()
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")

    for pair_set in ("sample", "fresh"):
        e = res[pair_set]
        print(f"== {pair_set}: {e['pairs']} pairs, {e['images']} images, {len(e['images_excluded'])} images excluded")
        for lab, v in e["per_label"].items():
            print(f"   {lab:<15} pairs excluded {v['pairs_excluded']}/{v['pairs']} (blank {v['pairs_nearly_blank']}, "
                  f"text {v['pairs_mostly_text']}); images {v['images_excluded']}/{v['images']}")
        for method, runs in e["rule"].items():
            for when, r in runs.items():
                print(f"   {method:<8}{when:<7}" + "; ".join(f"{k} {v['k']}/{v['n']} (n/a {v['not_assessed']})"
                                                       for k, v in r.items()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["measure", "report"])
    args = parser.parse_args()
    do_measure() if args.step == "measure" else do_report()


if __name__ == "__main__":
    main()
