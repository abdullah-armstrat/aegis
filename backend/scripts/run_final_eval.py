"""WP-6: the final evaluation. Every step writes its numbers to results/; with no step named, all run.

Each step's plan was fixed in the project's decision log before it first ran, and no rule or
threshold is changed because of a result.

  matching-a        image matching on the 40 dataset A photos: match rate per transformation, hash
                    alone and with the keypoint stage, and false matches against every downloaded
                    VERITE image (ADR-051)
  date-check        the recycled-context date check on the A photos under seven posting-date
                    conditions (ADR-052)
  framing-f         emotional framing on the 99 SemEval texts of dataset F, as written and
                    lowercased (ADR-053)
  fault-injection   every extractor switched off, timed out or broken in turn (ADR-054)
  web-archive       the WP-4 web lookup replayed from saved page and archive answers (ADR-050)

Run from the repo root: python backend/scripts/run_final_eval.py [step ...]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from collections import Counter
from datetime import date, timedelta
from io import BytesIO
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
for path in (_BACKEND, _BACKEND / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

ROOT = _BACKEND.parent
OUT = ROOT / "results"
LABELS = ROOT / "data" / "labels"
A_DIR = ROOT / "data" / "A_originals"
VERITE_DIR = ROOT / "data" / "verite" / "images"
SEED = 20260928


def wilson(k: int, n: int, z: float = 1.959964) -> list:
    if n == 0:
        return [None, None]
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(max(0.0, centre - half), 3) + 0.0, round(min(1.0, centre + half), 3)]


def rate(k: int, n: int) -> dict:
    return {"k": k, "n": n, "rate": round(k / n, 3) if n else None, "ci": wilson(k, n)}


def production_settings(**overrides) -> None:
    """The app's shipped settings (no AEGIS_ overrides from the environment), plus any given here."""
    for key in [k for k in os.environ if k.startswith("AEGIS_")]:
        del os.environ[key]
    os.environ.pop("GOOGLE_VISION_API_KEY", None)
    for key, value in overrides.items():
        os.environ[f"AEGIS_{key.upper()}"] = str(value)
    from app.config import get_settings

    get_settings.cache_clear()


def a_photos() -> list[dict]:
    return list(csv.DictReader(open(LABELS / "A_originals.csv", encoding="utf-8")))


def write(name: str, data: dict) -> None:
    OUT.mkdir(exist_ok=True)
    (OUT / name).write_text(json.dumps(data, indent=1, default=str), encoding="utf-8")
    print(f"wrote results/{name}")


# ------------------------------------------------------------------------------ Part A
def _png(img) -> bytes:
    buf = BytesIO()
    img.convert("RGB").save(buf, format="PNG", compress_level=1)
    return buf.getvalue()


def matching_a() -> None:
    """ADR-051: the shipped lookup on transformed A photos and on unrelated VERITE images."""
    from PIL import Image

    from app.extractors import reverse_image
    from tests.eval.image_transforms import TRANSFORMS

    production_settings()
    index = reverse_image.load_index(str(reverse_image._index_path()))
    entry_of = {s.url: e.id for e in index for s in e.sources}

    def lookup(data: bytes, keypoints: bool) -> set[str]:
        production_settings(keypoint_matching=str(keypoints).lower())
        result = reverse_image.find_local_matches(data)
        if result.status.value == "not_assessed":
            raise SystemExit(f"a lookup could not run: {result.detail}")
        return {entry_of[m.url] for m in result.matches}

    photos = a_photos()
    low = {r["nasa_id"] for r in photos if r["low_texture"] == "yes"}
    rows, started = [], time.perf_counter()
    for n, r in enumerate(photos, 1):
        own = f"nasa-{r['nasa_id']}"
        with Image.open(A_DIR / r["file"]) as img:
            original = img.convert("RGB")
        for name, transform in TRANSFORMS.items():
            data = _png(transform(original))
            for stage, keypoints in (("hash", False), ("shipped", True)):
                found = lookup(data, keypoints)
                rows.append({"photo": r["nasa_id"], "low_texture": r["nasa_id"] in low, "transform": name,
                             "stage": stage, "matched": own in found, "wrong": sorted(found - {own})})
        print(f"  {n}/40 {r['nasa_id']} ({time.perf_counter() - started:.0f} s)", flush=True)

    unrelated = []
    files = sorted(p for p in VERITE_DIR.iterdir() if p.is_file())
    for n, path in enumerate(files, 1):
        data = path.read_bytes()
        try:
            from PIL import Image as _I

            with _I.open(BytesIO(data)) as im:
                im.load()
        except Exception:  # noqa: BLE001 - an unreadable download is not a query
            unrelated.append({"image": path.stem, "unreadable": True})
            continue
        entry = {"image": path.stem}
        for stage, keypoints in (("hash", False), ("shipped", True)):
            entry[stage] = sorted(lookup(data, keypoints))
        unrelated.append(entry)
        if n % 50 == 0:
            print(f"  VERITE {n}/{len(files)}", flush=True)
    production_settings()

    def table(subset) -> dict:
        out = {}
        for name in TRANSFORMS:
            out[name] = {}
            for stage in ("hash", "shipped"):
                rs = [x for x in subset if x["transform"] == name and x["stage"] == stage]
                out[name][stage] = {**rate(sum(x["matched"] for x in rs), len(rs)),
                                    "wrong_matches": sum(bool(x["wrong"]) for x in rs)}
        for stage in ("hash", "shipped"):
            rs = [x for x in subset if x["stage"] == stage]
            out.setdefault("all transformations", {})[stage] = {**rate(sum(x["matched"] for x in rs), len(rs)),
                                                               "wrong_matches": sum(bool(x["wrong"]) for x in rs)}
        return out

    readable = [u for u in unrelated if not u.get("unreadable")]
    false = {}
    for stage in ("hash", "shipped"):
        hits = [u for u in readable if u[stage]]
        pairs = sum(len(u[stage]) for u in readable)
        false[stage] = {"images": rate(len(hits), len(readable)), "pairs": rate(pairs, len(readable) * len(index)),
                        "matches": [{"image": u["image"], "entries": u[stage]} for u in hits]}
    res = {"thresholds": {"phash_bits": 10, "mirror": True, "keypoint_min_inliers": 12},
           "index_entries": len(index), "photos": len(photos), "low_texture": sorted(low),
           "all_40": table(rows), "low_texture_8": table([x for x in rows if x["low_texture"]]),
           "other_32": table([x for x in rows if not x["low_texture"]]),
           "unrelated": {"verite_files": len(files), "readable": len(readable), "false_matches": false},
           "rows": rows}
    write("wp6_matching_A.json", res)
    print(f"{'transformation':<16} {'hash':>18} {'with keypoints':>18}")
    for name, v in res["all_40"].items():
        print(f"{name:<16} {v['hash']['k']:>3}/{v['hash']['n']:<3} {v['hash']['rate']:.3f}      "
              f"{v['shipped']['k']:>3}/{v['shipped']['n']:<3} {v['shipped']['rate']:.3f}  wrong {v['shipped']['wrong_matches']}")
    for stage, v in false.items():
        print(f"false matches, {stage}: images {v['images']['k']}/{v['images']['n']}, pairs {v['pairs']['k']}/{v['pairs']['n']}")


# ------------------------------------------------------------------------------ Part B
DATE_CONDITIONS = [  # (name, days from the index date or None, expected status, expected severity)
    ("365 days before", -365, "clear", "info"),
    ("1 day before", -1, "clear", "info"),
    ("on the index date", 0, "clear", "info"),
    ("1 day after", 1, "fired", "high"),
    ("365 days after", 365, "fired", "high"),
    ("no posting date", None, "fired", "medium"),
    ("lookup switched off", None, "not_assessed", "info"),
]


def date_check() -> None:
    """ADR-052: the recycled-context status for posting dates around each A photo's index date."""
    from app.adapters.image_adapter import build_bundle
    from app.extractors import reverse_image
    from app.fusion.rules import recycled_context_rule

    production_settings()
    index = {e.id: e for e in reverse_image.load_index(str(reverse_image._index_path()))}
    rows = []
    for n, r in enumerate(a_photos(), 1):
        entry = index[f"nasa-{r['nasa_id']}"]
        indexed = date.fromisoformat(entry.earliest_date)
        data = (A_DIR / r["file"]).read_bytes()
        production_settings()
        bundle = build_bundle(data, caption=None, source_ref=r["file"])
        production_settings(reverse_image_mode="off")
        off_bundle = build_bundle(data, caption=None, source_ref=r["file"])
        production_settings()
        for name, days, expected, severity in DATE_CONDITIONS:
            if name == "lookup switched off":
                case = off_bundle
            else:
                posted = None if days is None else (indexed + timedelta(days=days)).isoformat()
                case = bundle.model_copy(update={"meta": bundle.meta.model_copy(update={"posted_date": posted})})
            flag = recycled_context_rule(case)
            checks = {"status": flag.status.value == expected, "severity": flag.severity.value == severity,
                      "reason": bool(flag.evidence.strip())}
            if expected == "fired" and days is not None:
                checks["cites_index_date"] = flag.evidence.startswith(f"Found on a page dated {indexed.isoformat()}")
            if name == "no posting date":
                checks["asks_for_date"] = "posting date" in flag.plain_explanation
            rows.append({"photo": r["nasa_id"], "index_date": indexed.isoformat(), "condition": name,
                         "status": flag.status.value, "severity": flag.severity.value, "expected": expected,
                         "correct": all(checks.values()), "checks": checks, "evidence": flag.evidence})
        print(f"  {n}/40 {r['nasa_id']}", flush=True)
    per = {}
    for name, _, expected, severity in DATE_CONDITIONS:
        rs = [x for x in rows if x["condition"] == name]
        per[name] = {"expected": expected, "expected_severity": severity,
                     **rate(sum(x["correct"] for x in rs), len(rs)),
                     "statuses": dict(Counter(x["status"] for x in rs))}
    wrong = [x for x in rows if not x["correct"]]
    off_reason = next(x["evidence"] for x in rows if x["condition"] == "lookup switched off")
    write("wp6_date_check.json", {"conditions": per, "mismatches": wrong, "off_reason": off_reason, "rows": rows})
    for name, v in per.items():
        print(f"  {name:<22} expected {v['expected']:<13} correct {v['k']}/{v['n']} {v['ci']}  {v['statuses']}")
    print(f"  mismatches: {len(wrong)}; lookup off says: {off_reason}")


# ------------------------------------------------------------------------------ Part C
MARKER_KINDS = (("capitals", "ALL-CAPS ratio"), ("exclamation", "exclamation mark"),
                ("urgency", "urgency/sensational"), ("group framing", "in-/out-group"))  # words each marker's text contains
F_GROUPS = ["calm_honest", "calm_manipulative", "loud_honest", "loud_manipulative"]


def _f1(sample: list[dict]) -> float:
    tp = sum(x["manipulative"] and x["status"] == "fired" for x in sample)
    fp = sum(not x["manipulative"] and x["status"] == "fired" for x in sample)
    fn = sum(x["manipulative"] and x["status"] != "fired" for x in sample)
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def framing_f() -> None:
    """ADR-053: the emotional-framing rule on dataset F, as written and lowercased."""
    from app.fusion.rules import emotional_framing_rule, find_manipulation_markers
    from app.models import EvidenceBundle, Meta, Modality

    production_settings()
    with open(ROOT / "data" / "F_captions" / "captions.csv", encoding="utf-8") as fh:
        texts = {r["caption_id"]: r["text"] for r in csv.DictReader(fh)}
    with open(LABELS / "F_harder_captions.csv", encoding="utf-8") as fh:
        items = [{**r, "text": texts[r["caption_id"]]} for r in csv.DictReader(fh)]
    res = {"texts": len(items), "groups": dict(Counter(r["group"] for r in items))}
    for variant in ("as written", "lowercased"):
        rows = []
        for r in items:
            text = r["text"].lower() if variant == "lowercased" else r["text"]
            flag = emotional_framing_rule(EvidenceBundle(caption=text, meta=Meta(modality=Modality.IMAGE)))
            markers = find_manipulation_markers(text)
            rows.append({"id": r["caption_id"], "group": r["group"], "manipulative": r["manipulative"] == "yes",
                         "status": flag.status.value,
                         "markers": {kind: any(words in m for m in markers) for kind, words in MARKER_KINDS}})
        assessed = [x for x in rows if x["status"] != "not_assessed"]
        tp = sum(x["manipulative"] and x["status"] == "fired" for x in assessed)
        fp = sum(not x["manipulative"] and x["status"] == "fired" for x in assessed)
        fn = sum(x["manipulative"] and x["status"] != "fired" for x in assessed)
        rng = random.Random(SEED)
        draws = sorted(_f1([rng.choice(assessed) for _ in assessed]) for _ in range(2000))
        res[variant] = {
            "not_assessed": len(rows) - len(assessed), "tp": tp, "fp": fp, "fn": fn,
            "tn": len(assessed) - tp - fp - fn,
            "precision": rate(tp, tp + fp), "recall": rate(tp, tp + fn),
            "f1": round(_f1(assessed), 3), "f1_ci": [round(draws[49], 3), round(draws[1949], 3)],
            "fire_rate": {g: rate(sum(x["status"] == "fired" for x in rows if x["group"] == g),
                                  sum(x["group"] == g for x in rows)) for g in F_GROUPS},
            "markers_present": {g: {kind: sum(x["markers"][kind] for x in rows if x["group"] == g)
                                    for kind, _ in MARKER_KINDS} for g in F_GROUPS},
            "rows": rows}
    write("wp6_framing_F.json", res)
    for variant in ("as written", "lowercased"):
        v = res[variant]
        print(f"  {variant}: P {v['precision']['k']}/{v['precision']['n']} {v['precision']['ci']}  "
              f"R {v['recall']['k']}/{v['recall']['n']} {v['recall']['ci']}  F1 {v['f1']} {v['f1_ci']}  "
              f"n/a {v['not_assessed']}")
        for g in F_GROUPS:
            fr = v["fire_rate"][g]
            print(f"     {g:<18} fired {fr['k']}/{fr['n']} {fr['ci']}  markers {v['markers_present'][g]}")


# ------------------------------------------------------------------------------ Part D
def fault_injection() -> None:
    """ADR-054: every extractor switched off, timed out or broken in turn."""
    import eval_fault_injection

    res = eval_fault_injection.run()
    write("wp6_fault_injection.json", res)
    s = res["summary"]
    print(f"  {s['passed']}/{s['faults']} faults passed; failed: {s['failed']}")


STEPS = {"matching-a": matching_a, "date-check": date_check, "framing-f": framing_f,
         "fault-injection": fault_injection}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("steps", nargs="*", choices=[[], *STEPS], help="steps to run (default: all)")
    args = parser.parse_args()
    for step in args.steps or list(STEPS):
        print(f"== {step}")
        STEPS[step]()


if __name__ == "__main__":
    main()
