"""WP-6: the final evaluation. Every step writes its numbers to results/; with no step named, all run.

Each step's plan was fixed in the project's decision log before it first ran, and no rule or
threshold is changed because of a result.

  matching-a        image matching on the 40 dataset A photos: match rate per transformation, hash
                    alone and with the keypoint stage, and false matches against every downloaded
                    VERITE image (ADR-051)
  hard-pairs        20 pairs of different NASA photos of the same subject: wrong matches with and
                    without the alignment check (ADR-056)
  date-check        the recycled-context date check on the A photos under seven posting-date
                    conditions (ADR-052)
  framing-f         emotional framing on the 99 SemEval texts of dataset F, as written and
                    lowercased (ADR-053)
  framing-llm       the local LLM on the same 99 texts, next to the wording check (ADR-058)
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


def hard_pairs() -> None:
    """ADR-056: 20 pairs of different NASA photos of one subject, as their own index; every photo under
    the 15 transformations, with the alignment check and with it replaced by acceptance."""
    import tempfile
    from unittest import mock

    from PIL import Image

    from app.extractors import reverse_image
    from tests.eval.image_transforms import TRANSFORMS

    with open(LABELS / "hard_pairs.csv", encoding="utf-8") as fh:
        photos = list(csv.DictReader(fh))
    folder = ROOT / "data" / "hard_pairs"
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        entries = []
        for p in photos:
            try:
                dated = date.fromisoformat(p["date_created"]).isoformat()
            except ValueError:
                dated = "1970-01-01"
            entries.append({"id": f"{p['pair']}{p['side']}", "phash": p["phash"], "earliest_date": dated,
                            "image": str(folder / p["file"]),
                            "sources": [{"url": p["source_url"], "published_date": dated}]})
        index = Path(tmp) / "hard_pairs_index.json"
        index.write_text(json.dumps({"entries": entries}), encoding="utf-8")
        entry_of = {e["sources"][0]["url"]: e["id"] for e in entries}
        for n, p in enumerate(photos, 1):
            own = f"{p['pair']}{p['side']}"
            partner = f"{p['pair']}{'b' if p['side'] == 'a' else 'a'}"
            with Image.open(folder / p["file"]) as img:
                original = img.convert("RGB")
            for name, transform in TRANSFORMS.items():
                data = _png(transform(original))
                for variant in ("without alignment", "with alignment"):
                    production_settings(image_index_path=str(index))
                    reverse_image.load_index.cache_clear()
                    with mock.patch.object(reverse_image, "_aligned_distance", lambda *a: 0) \
                            if variant == "without alignment" else mock.patch.object(reverse_image, "HASH_BITS", 64):
                        result = reverse_image.find_local_matches(data)
                    found = {entry_of[m.url] for m in result.matches}
                    rows.append({"photo": own, "pair": p["pair"], "transform": name, "variant": variant,
                                 "method": result.method, "own": own in found, "partner": partner in found,
                                 "other": sorted(found - {own})})
            print(f"  {n}/{len(photos)} {own}", flush=True)
    production_settings()
    reverse_image.load_index.cache_clear()
    res = {"pairs": len(photos) // 2, "photos": len(photos), "queries_per_variant": len(photos) * len(TRANSFORMS)}
    for variant in ("without alignment", "with alignment"):
        rs = [r for r in rows if r["variant"] == variant]
        res[variant] = {
            "queries_with_another_photo": rate(sum(bool(r["other"]) for r in rs), len(rs)),
            "pairs_with_a_partner_match": rate(len({r["pair"] for r in rs if r["partner"]}), len(photos) // 2),
            "own_photo_found": rate(sum(r["own"] for r in rs), len(rs)),
            "own_by_transformation": {t: sum(r["own"] for r in rs if r["transform"] == t) for t in TRANSFORMS},
            "wrong_by_transformation": {t: sum(bool(r["other"]) for r in rs if r["transform"] == t) for t in TRANSFORMS},
            "wrong_pairs": sorted({r["pair"] for r in rs if r["partner"]})}
    res["rows"] = rows
    write("wp6_hard_pairs.json", res)
    for variant in ("without alignment", "with alignment"):
        v = res[variant]
        print(f"  {variant}: queries matching another photo {v['queries_with_another_photo']['k']}/"
              f"{v['queries_with_another_photo']['n']}; pairs with a partner match {v['pairs_with_a_partner_match']['k']}/"
              f"{v['pairs_with_a_partner_match']['n']}; own photo found {v['own_photo_found']['k']}/{v['own_photo_found']['n']}")


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
MARKER_KINDS = (("capitals", "words in capitals"), ("exclamation", "exclamation mark"),
                ("urgency", "urgency list"), ("listed phrases", "listed phrases"))  # words each marker's text contains
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


FRAMING_LLM_PROMPT = """\
Here is the text printed on an image shared on social media:

\"\"\"{text}\"\"\"

Does this text use any of these persuasion techniques?
- Loaded language: words or phrases with strong emotional meaning, used to influence the reader.
- Appeal to fear or prejudice: building support for an idea by raising fear, anxiety or prejudice.
- Exaggeration or minimisation: making something seem bigger or smaller than it really is.

Answer with a single JSON object and nothing else: {{"uses_technique": true}} or {{"uses_technique": false}}
"""


def framing_llm() -> None:
    """ADR-058: the local LLM asked about the three techniques on each dataset F text, next to the
    wording check. Pre-registered; nothing is changed because of the result."""
    import httpx

    from app.config import get_settings

    production_settings()
    settings = get_settings()
    with open(ROOT / "data" / "F_captions" / "captions.csv", encoding="utf-8") as fh:
        texts = {r["caption_id"]: r["text"] for r in csv.DictReader(fh)}
    with open(LABELS / "F_harder_captions.csv", encoding="utf-8") as fh:
        items = [{**r, "text": texts[r["caption_id"]]} for r in csv.DictReader(fh)]
    rows = []
    for n, r in enumerate(items, 1):
        started = time.perf_counter()
        answer, raw = None, None
        try:
            resp = httpx.post(f"{settings.ollama_host}/api/generate", timeout=180, json={
                "model": settings.ollama_model, "prompt": FRAMING_LLM_PROMPT.format(text=r["text"]),
                "stream": False, "format": "json", "options": {"temperature": 0, "seed": SEED}})
            resp.raise_for_status()
            raw = resp.json()["response"]
            value = json.loads(raw).get("uses_technique")
            if isinstance(value, bool):
                answer = value
            elif str(value).strip().lower() in ("true", "false"):
                answer = str(value).strip().lower() == "true"
        except (httpx.HTTPError, ValueError, KeyError, AttributeError) as exc:
            raw = raw or type(exc).__name__
        rows.append({"id": r["caption_id"], "group": r["group"], "manipulative": r["manipulative"] == "yes",
                     "answer": answer, "raw": raw, "seconds": round(time.perf_counter() - started, 2),
                     "status": "not_assessed" if answer is None else ("fired" if answer else "clear")})
        if n % 10 == 0:
            print(f"  {n}/{len(items)}", flush=True)
    answered = [x for x in rows if x["answer"] is not None]
    tp = sum(x["manipulative"] and x["answer"] for x in answered)
    fp = sum(not x["manipulative"] and x["answer"] for x in answered)
    fn = sum(x["manipulative"] and not x["answer"] for x in answered)
    rng = random.Random(SEED)
    draws = sorted(_f1([rng.choice(answered) for _ in answered]) for _ in range(2000)) if answered else [0.0] * 2000
    words = json.loads((OUT / "wp6_framing_F.json").read_text(encoding="utf-8"))["as written"] \
        if (OUT / "wp6_framing_F.json").exists() else None
    res = {"model": settings.ollama_model, "texts": len(rows), "unreadable_answers": len(rows) - len(answered),
           "tp": tp, "fp": fp, "fn": fn, "tn": len(answered) - tp - fp - fn,
           "precision": rate(tp, tp + fp), "recall": rate(tp, tp + fn),
           "f1": round(_f1(answered), 3), "f1_ci": [round(draws[49], 3), round(draws[1949], 3)],
           "yes_rate": {g: rate(sum(bool(x["answer"]) for x in rows if x["group"] == g),
                                sum(x["group"] == g for x in rows)) for g in F_GROUPS},
           "mean_seconds": round(sum(x["seconds"] for x in rows) / len(rows), 2),
           "wording_check": None if words is None else {k: words[k] for k in ("precision", "recall", "f1", "f1_ci",
                                                                               "fire_rate")},
           "rows": rows}
    write("wp6_framing_llm.json", res)
    print(f"  LLM: P {res['precision']['k']}/{res['precision']['n']} {res['precision']['ci']}  "
          f"R {res['recall']['k']}/{res['recall']['n']} {res['recall']['ci']}  F1 {res['f1']} {res['f1_ci']}  "
          f"unreadable {res['unreadable_answers']}  {res['mean_seconds']} s each")
    for g in F_GROUPS:
        print(f"     {g:<18} yes {res['yes_rate'][g]['k']}/{res['yes_rate'][g]['n']} {res['yes_rate'][g]['ci']}")


# ------------------------------------------------------------------------------ Part D
def fault_injection() -> None:
    """ADR-054: every extractor switched off, timed out or broken in turn."""
    import eval_fault_injection

    res = eval_fault_injection.run()
    write("wp6_fault_injection.json", res)
    s = res["summary"]
    print(f"  {s['passed']}/{s['faults']} faults passed; failed: {s['failed']}")


# ------------------------------------------------------------------------------ Part E
WEB_RUNS = (("first live run (ADR-033)", "wp4_web_report_run1.json"),
            ("fixed live run (ADR-034)", "wp4_web_report_run2.json"),
            ("saved inputs, dating before ADR-043", "wp4_web_report_saved_before.json"),
            ("saved inputs, current dating", "wp4_web_report_saved_after.json"))


def web_archive() -> None:
    """ADR-050: the WP-4 evaluation replayed from saved page and archive answers only."""
    import eval_web_lookup

    production_settings()  # no key: a Vision call is impossible, the cached responses are used
    eval_web_lookup.dates_from_inputs()
    runs = {}
    for name, file in WEB_RUNS:
        path = OUT / file
        runs[name] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    pages = json.loads((OUT / "wp4_page_dates_saved.json").read_text(encoding="utf-8"))
    write("wp6_web_archive.json", {"runs": runs, "saved_inputs": {k: pages[k] for k in (
        "pages", "page_fetch_failures", "page_status", "transitions")}})
    for name, r in runs.items():
        if r is None:
            print(f"  {name}: not on this machine")
            continue
        a, v = r["A"], r["VERITE"]
        print(f"  {name:<38} A found {a['found'][0]}/{a['found'][1]} dated {a['dated'][0]}/{a['dated'][1]} "
              f"errors {a['dating_errors'][0]}/{a['dating_errors'][1]}  VERITE dated {v['dated'][0]}/{v['dated'][1]}  "
              f"sources {r['date_sources']}")


STEPS = {"matching-a": matching_a, "hard-pairs": hard_pairs, "date-check": date_check, "framing-f": framing_f, "framing-llm": framing_llm,
         "fault-injection": fault_injection, "web-archive": web_archive}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("steps", nargs="*", choices=[[], *STEPS], help="steps to run (default: all)")
    args = parser.parse_args()
    for step in args.steps or list(STEPS):
        print(f"== {step}")
        STEPS[step]()


if __name__ == "__main__":
    main()
