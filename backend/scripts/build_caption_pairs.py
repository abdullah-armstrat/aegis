"""Caption regression cases on real pictures from dataset A (the project's decision log, ADR-042).

The rule was fixed before any case was scored:
  * 8 photos, 2 per topic (weather, transport, animals, landscapes), drawn with
    random.Random(20260927) from each topic's nasa_ids in sorted order.
  * The matching caption is the first sentence of the photo's NASA description, after removing a
    leading dateline (everything up to and including the first run of dashes set off by spaces,
    when that run starts within the first 60 characters).
  * The mismatched caption is the matching caption of the photo two places later in the list
    ordered by topic then draw, wrapping round: always from a different topic.
  * Expected: the matching caption clear, the mismatched one fired.

  build     writes tests/eval/test_set/caption_pairs_A.json (ids, files and captions; the photos
            stay out of the repository)
  score     runs each case through the app's image path with the default settings and writes
            results/wp5_caption_pairs_A.json

Run from the repo root: python backend/scripts/build_caption_pairs.py build|score
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

ROOT = _BACKEND.parent
MANIFEST = ROOT / "data" / "labels" / "A_originals.csv"
PHOTOS = ROOT / "data" / "A_originals"
CASES = _BACKEND / "tests" / "eval" / "test_set" / "caption_pairs_A.json"
OUT = ROOT / "results" / "wp5_caption_pairs_A.json"
SEED = 20260927
TOPICS = ("weather", "transport", "animals", "landscapes")
PER_TOPIC = 2

# A dateline ends at dashes set off by spaces: two or more hyphens, or en or em dashes. A hyphen
# inside a word or a date range ("AST-01-042", "13-19 Jan.") is not one.
_DASHES = re.compile(r"\s(?:-{2,}|[\u2013\u2014](?:\s*[\u2013\u2014])*)\s")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def first_sentence(description: str) -> str:
    """The description's first sentence, without a leading dateline."""
    text = " ".join(description.split())
    dash = _DASHES.search(text)
    if dash and dash.start() < 60:
        text = text[dash.end():]
    return _SENTENCE_END.split(text, maxsplit=1)[0].strip()


def build() -> None:
    rows = list(csv.DictReader(open(MANIFEST, encoding="utf-8")))
    rng = random.Random(SEED)
    chosen = []
    for topic in TOPICS:
        ids = sorted(r["nasa_id"] for r in rows if r["topic"] == topic)
        chosen += [next(r for r in rows if r["nasa_id"] == i) for i in rng.sample(ids, PER_TOPIC)]
    captions = [first_sentence(r["nasa_description"]) for r in chosen]
    cases = []
    for n, r in enumerate(chosen):
        other = (n + 2) % len(chosen)
        cases.append({"id": f"{r['nasa_id']}-own", "nasa_id": r["nasa_id"], "file": r["file"], "topic": r["topic"],
                      "caption": captions[n], "caption_from": r["nasa_id"], "expected": "clear"})
        cases.append({"id": f"{r['nasa_id']}-other", "nasa_id": r["nasa_id"], "file": r["file"], "topic": r["topic"],
                      "caption": captions[other], "caption_from": chosen[other]["nasa_id"], "expected": "fire"})
    CASES.write_text(json.dumps({
        "_about": ("Caption regression cases on real pictures from dataset A (public-domain NASA photos, "
                   "kept out of the repository; data/A_originals/). Built by backend/scripts/"
                   "build_caption_pairs.py: 2 photos per topic drawn with seed 20260927; each photo with "
                   "the first sentence of its own NASA description (expected clear) and with that of a "
                   "photo from another topic (expected fire)."),
        "cases": cases}, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    for c in cases:
        print(f"{c['id']:<22} {c['expected']:<5} {c['caption'][:90]}")


def score() -> None:
    import os

    for key in [k for k in os.environ if k.startswith("AEGIS_")]:
        del os.environ[key]  # the production defaults
    from app.adapters.image_adapter import build_bundle
    from app.config import get_settings
    from app.fusion.scorecard import build_scorecard

    get_settings.cache_clear()
    settings = get_settings()
    cases = json.loads(CASES.read_text(encoding="utf-8"))["cases"]
    rows = []
    for c in cases:
        bundle = build_bundle((PHOTOS / c["file"]).read_bytes(), caption=c["caption"], source_ref=c["file"])
        flag = next(f for f in build_scorecard(bundle).flags if f.type.value == "caption_content_mismatch")
        sim = bundle.caption_match.image_similarity if bundle.caption_match else None
        rows.append({"id": c["id"], "expected": c["expected"], "status": flag.status.value,
                     "similarity": None if sim is None else round(sim, 4),
                     "as_expected": {"clear": "clear", "fire": "fired"}[c["expected"]] == flag.status.value,
                     "evidence": flag.evidence})
        print(f"{c['id']:<22} expected {c['expected']:<5} got {flag.status.value:<12} "
              f"similarity {'-' if sim is None else f'{sim:.4f}'}")
    out = {"method": settings.caption_match_method, "cases": len(rows),
           "as_expected": sum(r["as_expected"] for r in rows),
           "matching_clear": sum(r["status"] == "clear" for r in rows if r["expected"] == "clear"),
           "mismatched_fired": sum(r["status"] == "fired" for r in rows if r["expected"] == "fire"),
           "not_assessed": sum(r["status"] == "not_assessed" for r in rows), "rows": rows}
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"{out['as_expected']}/{out['cases']} as expected: matching captions clear {out['matching_clear']}/8, "
          f"mismatched fired {out['mismatched_fired']}/8, not assessed {out['not_assessed']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["build", "score"])
    args = parser.parse_args()
    build() if args.step == "build" else score()


if __name__ == "__main__":
    main()
