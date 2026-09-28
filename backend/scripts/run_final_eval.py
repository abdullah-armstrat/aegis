"""WP-6: the final evaluation. One command regenerates every number and chart the report uses.

    python backend/scripts/run_final_eval.py              every step, then the numbers and charts
    python backend/scripts/run_final_eval.py --rebuild    the same, rebuilding every cached model output
    python backend/scripts/run_final_eval.py STEP ...     only the steps named

Each step's plan was fixed in the project's decision log before it first ran, and no rule or
threshold is changed because of a result. Everything is written to results/: one JSON file per
step, results/report_numbers.csv (every figure the report may cite, with its interval and the file
and step it comes from) and one PNG chart per result in results/figures/.

Slow model outputs are replayed from data/final_eval_cache/ (git-ignored): BLIP descriptions, OCR
lines and CLIP and spaCy scores of the VERITE images, the LLM's answers, Whisper's transcripts, the
video path's evidence bundles, and CLIP scores of the swap test and of keyframes against captions.
A missing entry is built, item by item where the step allows, so an interrupted build resumes.
--rebuild deletes the cache of every step being run first. Everything else runs live every time.

  synthetic         the 19 hand-built examples against their pinned counts, and the 16 dataset A
                    caption cases (ADR-042)
  matching-a        image matching on the 40 dataset A photos: match rate per transformation, hash
                    alone and with the keypoint stage, and false matches against every downloaded
                    VERITE image (ADR-051)
  hard-pairs        20 pairs of different NASA photos of the same subject: wrong matches with and
                    without the alignment check (ADR-056)
  date-check        the recycled-context date check on the A photos under seven posting-date
                    conditions (ADR-052)
  verite-features   BLIP, OCR, CLIP, spaCy and the picture limits on every VERITE image (cached)
  verite-llm        the LLM's answer on every held-out and fresh VERITE pair, and repeats (cached)
  verite-heldout    caption vs picture on the held-out VERITE pairs: the calibrated rule, word
                    overlap, the LLM and always-fire, McNemar tests and AUCs (ADR-023)
  verite-fresh      the fresh VERITE confirmation: picture only, meaning and overlap (ADR-024)
  picture-limits    the nearly-blank and mostly-text limits on both sets, before and after (ADR-041)
  llm-verite        the LLM on real pairs: validity, latency and consistency (ADR-067)
  ablations         captioner off, LLM on and off, embeddings against word overlap (ADR-067)
  framing-f         emotional framing on the 99 SemEval texts of dataset F, as written and
                    lowercased (ADR-053)
  framing-llm       the local LLM on the same 99 texts, next to the wording check (ADR-058)
  speech-swap       speech against picture: the swap test (ADR-030)
  video-pipeline    the whole video path on the 34 dataset E clips (ADR-031, ADR-049)
  caption-video     each clip against its own caption and a swapped one (ADR-067)
  whisper           Whisper tiny against base, and decoding settings a, b and c (ADR-029, ADR-049)
  fault-injection   every extractor switched off, timed out or broken in turn (ADR-054)
  web-archive       the WP-4 web lookup replayed from saved page and archive answers (ADR-050)
  performance       time per stage and peak memory: image path, video path, each model
  interface-review  the interface review's round B measures on the current build; round A as frozen
  no-verdict        verdict words in every output the evaluation produced (ADR-067)
  readability       reading grade of every explanation and "what to check" line it produced
  numbers           results/report_numbers.csv, from the step files
  charts            results/figures/*.png, from the step files

Run from the repo root.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
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
CLIPS = ROOT / "data" / "E_videos" / "clips"
FRONTEND = ROOT / "frontend"
CACHE = ROOT / "data" / "final_eval_cache"      # slow model outputs, replayed unless --rebuild
SEEN_DIR = ROOT / "data" / "final_eval_outputs"  # every flag each step made, for the two audits
SEED = 20260928

# The cache folders each step builds; --rebuild deletes those of the steps being run.
CACHES = {"verite-features": ["verite"], "verite-llm": ["llm"], "framing-llm": ["framing_llm"],
          "speech-swap": ["speech_swap"], "video-pipeline": ["video_bundles"], "caption-video": ["caption_video"],
          "whisper": ["whisper"]}


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


def _env() -> dict:
    """The environment for a child process: no AEGIS_ overrides and no web-search key."""
    return {k: v for k, v in os.environ.items() if not k.startswith("AEGIS_") and k != "GOOGLE_VISION_API_KEY"}


def _child(task: str) -> None:
    """Build one cache (or take one measurement) in a separate process, so its models and memory
    stay apart from the rest of the run."""
    print(f"  building in a separate process: {task}", flush=True)
    subprocess.run([sys.executable, str(Path(__file__).resolve()), "--child", task], check=True, env=_env(), cwd=ROOT)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _save(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    tmp.replace(path)


# ------------------------------------------------------------------------------ outputs seen
class _Seen:
    """Every flag, scorecard summary and LLM explanation a step creates, and the texts its bundles
    were built from, for the no-verdict audit and the readability check."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.flags: dict[tuple, int] = {}
        self.inputs: set[str] = set()
        self.llm: set[str] = set()
        self.summaries: set[str] = set()

    def flag(self, f) -> None:
        key = (f.type.value, f.status.value, f.source, f.plain_explanation, f.what_to_check, f.evidence)
        self.flags[key] = self.flags.get(key, 0) + 1

    def bundle(self, b) -> None:
        texts = [b.caption, b.transcript, *(s.text for s in b.scene_descriptions), *b.on_screen_text,
                 *(s.text for s in b.transcript_segments), *(t for k in b.keyframes for t in k.on_screen_text),
                 *(v for m in b.web_matches for v in (m.url, m.title, m.context))]
        self.inputs.update(t.strip() for t in texts if t and t.strip())

    def card(self, c) -> None:
        for f in c.flags:
            self.flag(f)
        if c.summary:
            self.summaries.add(c.summary)

    def verdict(self, v) -> None:
        if v.available and v.explanation:  # an unavailable verdict's text is the app's own reason
            self.llm.add(v.explanation)

    def save(self, step: str) -> None:
        flags = [{"check": k[0], "status": k[1], "source": k[2], "plain_explanation": k[3], "what_to_check": k[4],
                  "evidence": k[5], "count": n} for k, n in self.flags.items()]
        _save(SEEN_DIR / f"{step}.json", {"step": step, "flags": flags, "summaries": sorted(self.summaries),
                                          "inputs": sorted(self.inputs), "llm_explanations": sorted(self.llm)})


SEEN = _Seen()


def _record_outputs() -> None:
    """Hook the models, so every flag, scorecard, LLM verdict and bundle made in this process is seen."""
    from app.fusion import llm_reasoner, scorecard
    from app.models import EvidenceBundle, Flag, Scorecard

    def after(cls, hook) -> None:
        init = cls.__init__

        def wrapped(self, *args, **kwargs):
            init(self, *args, **kwargs)
            hook(self)

        cls.__init__ = wrapped

    after(Flag, SEEN.flag)
    after(EvidenceBundle, SEEN.bundle)
    after(Scorecard, SEEN.card)
    after(llm_reasoner.ReasonerVerdict, SEEN.verdict)
    run_rules = scorecard.run_rules

    def recorded(bundle):
        SEEN.bundle(bundle)  # a bundle read back from a cache was never constructed here
        return run_rules(bundle)

    scorecard.run_rules = recorded


# ------------------------------------------------------------------------------ regression
PINNED = {"emotional_framing": (3, 0, 0, 3, 1), "recycled_context": (3, 0, 0, 2, 1),
          "caption_content_mismatch": (3, 3, 0, 2, 0), "overall": (9, 3, 0, 7, 2)}  # tests/test_eval_harness.py


def synthetic() -> None:
    """The 19 hand-built examples, rules only, against the counts pinned in the test suite (word
    overlap, as pinned), and with the shipped caption check; then the 16 dataset A caption cases."""
    import build_caption_pairs
    from tests.eval.run_eval import evaluate

    def table(report) -> dict:
        out = {}
        for name, m in [*report.per_flag.items(), ("overall", report.overall)]:
            c = m.confusion
            out[name] = {"tp": c.tp, "fp": c.fp, "fn": c.fn, "tn": c.tn, "not_assessed": c.not_assessed,
                         "precision": m.precision, "recall": m.recall, "f1": m.f1}
        return out

    production_settings(caption_match_method="overlap")
    pinned_run = table(evaluate(use_llm=False)[0])
    production_settings()
    shipped_run = table(evaluate(use_llm=False)[0])
    matches = {k: tuple(pinned_run[k][f] for f in ("tp", "fp", "fn", "tn", "not_assessed")) == v for k, v in PINNED.items()}
    with tempfile.TemporaryDirectory() as tmp:
        build_caption_pairs.OUT = Path(tmp) / "cases.json"
        production_settings()
        build_caption_pairs.score()
        cases = json.loads(build_caption_pairs.OUT.read_text(encoding="utf-8"))
    write("wp6_synthetic.json", {"examples": 19, "word_overlap_as_pinned": pinned_run, "pinned": PINNED,
                                 "equal_to_pinned": matches, "all_equal": all(matches.values()),
                                 "shipped_caption_check": shipped_run, "caption_cases_A": cases})
    print(f"  19 examples: every pinned count equal: {all(matches.values())} {matches}")


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


# ------------------------------------------------------------------------------ VERITE (WP-2 rows)
VERITE_PHASES = (("sample", "blip_ocr"), ("sample", "clip:ViT-B/32"), ("sample", "clip:RN50"), ("sample", "spacy"),
                 ("fresh", "blip_ocr"), ("fresh", "clip:ViT-B/32"), ("fresh", "spacy"))
PHASE_FILES = {"blip_ocr": ("blip_ocr.json",), "clip:ViT-B/32": ("clip_vitb32.npz", "clip_vitb32.json"),
               "clip:RN50": ("clip_rn50.npz", "clip_rn50.json"), "spacy": ("spacy.json",)}
LLM_DIR = CACHE / "llm"
LLM_REPEATS = 4          # each consistency pair is asked this many more times
CONSISTENCY_PER_LABEL = 10


def _ecm(folder: Path | None = None):
    """The WP-2 evaluation code, reading and writing its features in the final evaluation's cache."""
    import eval_caption_match as ecm

    ecm.CACHE = folder or CACHE / "verite"
    ecm.CACHE.mkdir(parents=True, exist_ok=True)
    return ecm


def _limits_complete(ecm) -> bool:
    limits = _load(ecm.CACHE / "picture_limits.json")
    return all(p["image"] in limits for s in ("sample", "fresh") for p in ecm.load_pairs(s))


def verite_features() -> None:
    """Every model output the VERITE rows read, each model in its own process (ADR-022's layout)."""
    ecm = _ecm()
    for pair_set, phase in VERITE_PHASES:
        if all(ecm.cached(f, pair_set).exists() for f in PHASE_FILES[phase]):
            print(f"  {pair_set} {phase}: replayed from the cache")
        else:
            _child(f"verite|{pair_set}|{phase}")
    if _limits_complete(ecm):
        print("  picture limits: replayed from the cache")
    else:
        _child("verite-limits")


def _ask_llm(caption: str, scene: str, ocr: list[str]) -> dict:
    """One real call to the LLM reasoner, as the app makes it, with the input cache bypassed."""
    from app.fusion import llm_reasoner

    sink: list = []
    with llm_reasoner.record_calls(sink, bypass_cache=True):
        v = llm_reasoner.reason_over_text(caption, [scene], ocr)
    rec = sink[0] if sink else None
    return {"available": v.available, "same_subject": v.same_subject, "same_tone": v.same_tone,
            "explanation": v.explanation,
            "latency_s": None if rec is None or rec.latency_s is None else round(rec.latency_s, 3),
            "validity": rec.validity if rec else None, "timeout": bool(rec and rec.is_timeout)}


def _consistency_pairs(held: list[dict]) -> list[int]:
    rng = random.Random(SEED)
    rows = []
    for label in ("true", "miscaptioned", "out-of-context"):
        rows += rng.sample(sorted(p["row"] for p in held if p["label"] == label), CONSISTENCY_PER_LABEL)
    return rows


def _verite_sets(ecm) -> dict[str, list[dict]]:
    return {"heldout": [p for p in ecm.load_pairs("sample") if p["split"] == "held-out"],
            "fresh": ecm.load_pairs("fresh")}


def verite_llm() -> None:
    """The LLM's answer on every held-out and fresh pair, and the repeats for consistency (cached)."""
    from app.fusion.llm_reasoner import warm_up

    ecm = _ecm()
    production_settings(use_llm="true")
    sets = _verite_sets(ecm)
    # As the app does at start-up (ADR-037), so the first question does not wait for the model to load.
    print(f"  LLM warm-up: {warm_up()}", flush=True)
    for name, pairs in sets.items():
        feats = json.loads(ecm.cached("blip_ocr.json", "sample" if name == "heldout" else "fresh")
                           .read_text(encoding="utf-8"))
        path = LLM_DIR / f"verite_{name}.json"
        done = _load(path)
        todo = [p for p in pairs if str(p["row"]) not in done]
        print(f"  {name}: {len(pairs) - len(todo)} answers replayed, {len(todo)} to ask", flush=True)
        for n, p in enumerate(todo, 1):
            f = feats[p["image"]]
            done[str(p["row"])] = _ask_llm(p["caption"], f["scene"], f["ocr"])
            _save(path, done)
            if n % 10 == 0:
                print(f"    {n}/{len(todo)}", flush=True)
    feats = json.loads(ecm.cached("blip_ocr.json", "sample").read_text(encoding="utf-8"))
    by_row = {p["row"]: p for p in sets["heldout"]}
    path = LLM_DIR / "verite_repeats.json"
    done = _load(path)
    for n, row in enumerate(_consistency_pairs(sets["heldout"]), 1):
        p, answers = by_row[row], done.get(str(row), [])
        while len(answers) < LLM_REPEATS:
            answers.append(_ask_llm(p["caption"], feats[p["image"]]["scene"], feats[p["image"]]["ocr"]))
            done[str(row)] = answers
            _save(path, done)
        if n % 10 == 0:
            print(f"  repeats: {n}/{3 * CONSISTENCY_PER_LABEL} pairs", flush=True)
    production_settings()


def verite_heldout() -> None:
    """ADR-023: the held-out run, by its own code, on the rebuilt features and LLM answers."""
    # Its word-overlap baseline goes through the app's caption rule, which follows the configured
    # method; overlap was the default when this run was made (ADR-024 later made it "image").
    production_settings(caption_match_method="overlap")
    with tempfile.TemporaryDirectory() as tmp:
        view = Path(tmp)  # the layout the WP-2 code reads: features and the held-out answers together
        for f in (CACHE / "verite").iterdir():
            shutil.copy(f, view / f.name)
        shutil.copy(LLM_DIR / "verite_heldout.json", view / "llm.json")
        ecm = _ecm(view)
        ecm.do_evaluate(ecm.load_pairs(), str(OUT / "wp6_verite_heldout.json"))
    _ecm()
    production_settings()


def verite_fresh() -> None:
    """ADR-024: the fresh-set confirmation, by its own code."""
    ecm = _ecm()
    production_settings()
    ecm.do_confirm(str(OUT / "wp6_verite_fresh.json"))
    production_settings()


def picture_limits() -> None:
    """ADR-041: pairs excluded by the picture limits, and the rules before and after, on both sets."""
    import eval_picture_limits as epl

    ecm = _ecm()
    epl.LIMITS = ecm.CACHE / "picture_limits.json"
    epl.OUT = OUT / "wp6_picture_limits.json"
    production_settings()
    epl.do_report()
    production_settings()


def llm_verite() -> None:
    """ADR-067: the LLM on real pairs. Validity and latency of every first answer (held-out and
    fresh), consistency on 30 held-out pairs asked five times, and the LLM flag each answer makes."""
    from app.fusion import scorecard
    from app.fusion.llm_reasoner import ReasonerVerdict
    from app.models import EvidenceBundle, FlagStatus, Meta, Modality, SceneDescription

    ecm = _ecm()
    production_settings(use_llm="true")
    sets = _verite_sets(ecm)
    answers = {name: json.loads((LLM_DIR / f"verite_{name}.json").read_text(encoding="utf-8")) for name in sets}
    res = {"model": None, "calls": {}, "flag": {}}
    from app.config import get_settings

    res["model"] = get_settings().ollama_model
    original = scorecard.reason_over_text
    firsts = []
    try:
        for name, pairs in sets.items():
            feats = json.loads(ecm.cached("blip_ocr.json", "sample" if name == "heldout" else "fresh")
                               .read_text(encoding="utf-8"))
            statuses = {}
            for p in pairs:
                a = answers[name][str(p["row"])]
                firsts.append(a)
                verdict = ReasonerVerdict(a["same_subject"], a["same_tone"], a["explanation"] or "", available=a["available"])
                scorecard.reason_over_text = lambda *args, v=verdict: v
                f = feats[p["image"]]
                bundle = EvidenceBundle(caption=p["caption"], on_screen_text=f["ocr"], meta=Meta(modality=Modality.IMAGE),
                                        scene_descriptions=[SceneDescription(text=f["scene"])] if f["scene"] else [])
                statuses[p["row"]] = scorecard._llm_caption_scene_flag(bundle).status
            table = {}
            for lab in ("true", "out-of-context", "miscaptioned"):
                sub = [p for p in pairs if p["label"] == lab]
                table[lab] = {"fired": rate(sum(statuses[p["row"]] == FlagStatus.FIRED for p in sub), len(sub)),
                              "not_assessed": sum(statuses[p["row"]] == FlagStatus.NOT_ASSESSED for p in sub)}
            res["flag"][name] = table
    finally:
        scorecard.reason_over_text = original
        production_settings()

    lat = sorted(a["latency_s"] for a in firsts if a["latency_s"] is not None)
    res["calls"] = {
        "n": len(firsts), "validity": dict(Counter(a["validity"] for a in firsts)),
        "answered": rate(sum(bool(a["available"]) and a["same_subject"] is not None for a in firsts), len(firsts)),
        "clean_json": rate(sum(a["validity"] == "clean" for a in firsts), len(firsts)),
        "timeouts": sum(a["timeout"] for a in firsts),
        "latency_s": {"median": lat[len(lat) // 2] if len(lat) % 2 else round((lat[len(lat) // 2 - 1] + lat[len(lat) // 2]) / 2, 3),
                      "p95_nearest_rank": lat[math.ceil(0.95 * len(lat)) - 1], "min": lat[0], "max": lat[-1],
                      "mean": round(sum(lat) / len(lat), 3), "n": len(lat)},
        "latencies": lat}
    repeats = json.loads((LLM_DIR / "verite_repeats.json").read_text(encoding="utf-8"))
    label_of = {p["row"]: p["label"] for p in sets["heldout"]}
    rows = []
    for row in _consistency_pairs(sets["heldout"]):
        runs = [answers["heldout"][str(row)], *repeats[str(row)]]
        values = [a["same_subject"] if a["available"] else None for a in runs]
        rows.append({"row": row, "label": label_of[row], "same_subject": values,
                     "all_answered": None not in values, "consistent": None not in values and len(set(values)) == 1})
    res["consistency"] = {"pairs": len(rows), "runs_per_pair": 1 + LLM_REPEATS,
                          "consistent": rate(sum(r["consistent"] for r in rows), len(rows)),
                          "pairs_with_an_unanswered_call": sum(not r["all_answered"] for r in rows),
                          "by_label": {lab: rate(sum(r["consistent"] for r in rows if r["label"] == lab),
                                                 sum(r["label"] == lab for r in rows))
                                       for lab in ("true", "miscaptioned", "out-of-context")},
                          "rows": rows}
    write("wp6_llm_verite.json", res)
    c = res["calls"]
    print(f"  {c['n']} calls: answered {c['answered']['k']}/{c['n']}, clean JSON {c['clean_json']['k']}, "
          f"timeouts {c['timeouts']}; latency median {c['latency_s']['median']} s, p95 {c['latency_s']['p95_nearest_rank']} s, "
          f"max {c['latency_s']['max']} s; consistent {res['consistency']['consistent']['k']}/{len(rows)}")


def ablations() -> None:
    """ADR-067: captioner off, the LLM on and off, and embeddings against word overlap, on the
    held-out and fresh pairs, with the shipped picture limits."""
    from app.extractors.caption_match import read_from_image, spacy_similarity
    from app.fusion import rules
    from app.models import CaptionMatch, EvidenceBundle, FlagStatus, Meta, Modality, SceneDescription

    ecm = _ecm()
    limits = json.loads((ecm.CACHE / "picture_limits.json").read_text(encoding="utf-8"))
    res = {"note": ("Held-out pairs are among the 300 sampled pairs the picture-only threshold was fitted on; "
                    "the fresh pairs were never used to fit anything."), "sets": {}}
    for name, pairs in _verite_sets(ecm).items():
        pair_set = "sample" if name == "heldout" else "fresh"
        feats = json.loads(ecm.cached("blip_ocr.json", pair_set).read_text(encoding="utf-8"))
        scores, _, _ = ecm.load_scores(pairs, pair_set, arches=("ViT-B/32",))
        llm = json.loads((LLM_DIR / f"verite_{name}.json").read_text(encoding="utf-8"))

        def status(p, method: str, captioner: bool = True) -> FlagStatus:
            f = feats[p["image"]]
            scene = [SceneDescription(text=f["scene"])] if captioner and f["scene"] else []
            limit = limits[p["image"]]["limit"]
            if method == "overlap":
                match = None
            elif limit:
                match = CaptionMatch(detail=limit)
            elif method == "image":
                match = CaptionMatch(image_similarity=scores["img:ViT-B/32"][p["row"]])
            elif captioner:
                t = scores["txt:spaCy"][p["row"]]
                match = CaptionMatch(image_similarity=scores["img:ViT-B/32"][p["row"]],
                                     text_similarity=None if math.isnan(t) else t)
            else:  # as the app measures it with no scene description: the on-screen text alone
                seen = read_from_image([], f["ocr"])
                match = CaptionMatch(image_similarity=scores["img:ViT-B/32"][p["row"]],
                                     text_similarity=spacy_similarity(p["caption"], seen) if seen else None,
                                     detail="" if seen else "There was no scene description or on-screen text "
                                                            "to compare with the caption.")
            bundle = EvidenceBundle(caption=p["caption"], scene_descriptions=scene, on_screen_text=f["ocr"],
                                    caption_match=match, meta=Meta(modality=Modality.IMAGE))
            return rules.caption_scene_mismatch_rule(bundle).status

        def table(subset, fired: dict, assessed: dict | None = None) -> dict:
            out = {}
            for lab, key in (("true", "truthful_flagged"), ("out-of-context", "out_of_context_caught"),
                             ("miscaptioned", "miscaptioned_caught")):
                sub = [p for p in subset if p["label"] == lab]
                out[key] = rate(sum(fired[p["row"]] for p in sub), len(sub))
                if assessed is not None:
                    out[key]["not_assessed"] = sum(not assessed[p["row"]] for p in sub)
            return out

        entry = {"pairs": len(pairs), "captioner": {}}
        for method in ("image", "meaning", "overlap"):
            production_settings(caption_match_method=method)
            for on in (True, False):
                st = {p["row"]: status(p, method, on) for p in pairs}
                entry["captioner"].setdefault(method, {})["on" if on else "off"] = table(
                    pairs, {r: s == FlagStatus.FIRED for r, s in st.items()},
                    {r: s != FlagStatus.NOT_ASSESSED for r, s in st.items()})
        production_settings()
        picture = {p["row"]: status(p, "image") == FlagStatus.FIRED for p in pairs}
        answered = [p for p in pairs if llm[str(p["row"])]["available"] and llm[str(p["row"])]["same_subject"] is not None]
        says_different = {p["row"]: llm[str(p["row"])]["same_subject"] is False for p in answered}
        entry["llm"] = {"answered": len(answered), "unanswered": len(pairs) - len(answered),
                        "picture check alone": table(answered, picture),
                        "picture check or LLM": table(answered, {r: picture[r] or says_different[r] for r in says_different}),
                        "LLM alone": table(answered, says_different)}
        res["sets"][name] = entry
    write("wp6_ablations.json", res)
    for name, e in res["sets"].items():
        for method, runs in e["captioner"].items():
            for when, t in runs.items():
                print(f"  {name:<8} {method:<8} captioner {when:<4} " + "; ".join(
                    f"{k} {v['k']}/{v['n']} (n/a {v['not_assessed']})" for k, v in t.items()))
        for cfg in ("picture check alone", "picture check or LLM", "LLM alone"):
            t = e["llm"][cfg]
            print(f"  {name:<8} {cfg:<22} " + "; ".join(f"{k} {v['k']}/{v['n']}" for k, v in t.items()))


# ------------------------------------------------------------------------------ Part C
MARKER_KINDS = (("capitals", "words in capitals"), ("exclamation", "exclamation mark"),
                ("urgency", "urgency list"), ("listed phrases", "listed phrases"))  # words each marker's text contains
F_GROUPS = ["calm_honest", "calm_manipulative", "loud_honest", "loud_manipulative"]


def _f1(sample: list[dict]) -> float:
    tp = sum(x["manipulative"] and x["status"] == "fired" for x in sample)
    fp = sum(not x["manipulative"] and x["status"] == "fired" for x in sample)
    fn = sum(x["manipulative"] and x["status"] != "fired" for x in sample)
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def _f_items() -> list[dict]:
    with open(ROOT / "data" / "F_captions" / "captions.csv", encoding="utf-8") as fh:
        texts = {r["caption_id"]: r["text"] for r in csv.DictReader(fh)}
    with open(LABELS / "F_harder_captions.csv", encoding="utf-8") as fh:
        return [{**r, "text": texts[r["caption_id"]]} for r in csv.DictReader(fh)]


def framing_f() -> None:
    """ADR-053: the emotional-framing rule on dataset F, as written and lowercased."""
    from app.fusion.rules import emotional_framing_rule, find_manipulation_markers
    from app.models import EvidenceBundle, Meta, Modality

    production_settings()
    items = _f_items()
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
    wording check. Pre-registered; nothing is changed because of the result. Answers are cached."""
    import httpx

    from app.config import get_settings

    production_settings()
    settings = get_settings()
    items = _f_items()
    path = CACHE / "framing_llm" / "answers.json"
    saved = _load(path)
    todo = [r for r in items if r["caption_id"] not in saved]
    print(f"  {len(items) - len(todo)} answers replayed, {len(todo)} to ask", flush=True)
    for n, r in enumerate(todo, 1):
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
        saved[r["caption_id"]] = {"answer": answer, "raw": raw, "seconds": round(time.perf_counter() - started, 2)}
        _save(path, saved)
        if n % 10 == 0:
            print(f"  {n}/{len(todo)}", flush=True)
    rows = []
    for r in items:
        a = saved[r["caption_id"]]
        rows.append({"id": r["caption_id"], "group": r["group"], "manipulative": r["manipulative"] == "yes",
                     "answer": a["answer"], "raw": a["raw"], "seconds": a["seconds"],
                     "status": "not_assessed" if a["answer"] is None else ("fired" if a["answer"] else "clear")})
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


# ------------------------------------------------------------------------------ video (WP-3 rows)
def speech_swap() -> None:
    """ADR-030: the swap test, by its own code, on CLIP scores rebuilt in the cache."""
    import eval_speech_picture as esp
    from app.fusion.rules import SPEECH_PICTURE_THRESHOLD

    esp.CACHE = CACHE / "speech_swap"
    if (esp.CACHE / "speech_picture_scores.json").exists():
        print("  scores replayed from the cache")
    else:
        _child("speech-swap")
    with tempfile.TemporaryDirectory() as tmp:
        esp.OUT = Path(tmp)
        esp.do_fit_and_heldout()
        res = json.loads((Path(tmp) / "wp3_speech_picture_swap.json").read_text(encoding="utf-8"))
    res["app_threshold"] = SPEECH_PICTURE_THRESHOLD
    res["fitted_equals_app_threshold"] = round(res["threshold"], 4) == round(SPEECH_PICTURE_THRESHOLD, 4)
    write("wp6_speech_swap.json", res)


def video_pipeline() -> None:
    """ADR-031 and ADR-049: the whole video path on every dataset E clip. Each clip's evidence
    bundle is cached; the rules and the scorecard run on it live."""
    import eval_speech_picture as esp
    from app.fusion.rules import audio_visual_mismatch_rule
    from app.fusion.scorecard import build_scorecard
    from app.models import EvidenceBundle

    folder = CACHE / "video_bundles"
    clips = esp.clips()
    if all((folder / f"{cid}.json").exists() for cid in clips):
        print("  bundles replayed from the cache")
    else:
        _child("video-bundles")
    production_settings()
    cards = {}
    with tempfile.TemporaryDirectory() as tmp:
        esp.CACHE = esp.OUT = Path(tmp)
        runs = esp._pipeline_cache("")
        runs.mkdir(parents=True)
        for cid in sorted(clips):
            bundle = EvidenceBundle.model_validate_json((folder / f"{cid}.json").read_text(encoding="utf-8"))
            card = build_scorecard(bundle)
            flag = audio_visual_mismatch_rule(bundle)
            (runs / f"{cid}.json").write_text(json.dumps({
                "clip_id": cid, "flag_status": flag.status.value, "flag_timestamps": flag.timestamps,
                "flag_evidence": flag.evidence, "speech_status": bundle.extractor_status.get("speech"),
                "detail": bundle.extractor_detail,
                "segments": [s.model_dump() for s in bundle.transcript_segments]}, indent=1), encoding="utf-8")
            cards[cid] = {"summary": card.summary,
                          "flags": {f.type.value: f.status.value for f in card.flags if f.source == "rules"}}
        esp.analyse_pipeline("")
        res = json.loads((Path(tmp) / "wp3_speech_picture_pipeline.json").read_text(encoding="utf-8"))
    res["scorecards"] = cards
    write("wp6_video_pipeline.json", res)


def _video_captions() -> dict[str, dict]:
    """ADR-067: each clip's caption (the first sentence of its source footage's NASA description) and
    a swapped one (the same from a source of another topic, drawn with a fixed seed)."""
    from build_caption_pairs import first_sentence

    footage = {r["footage"]: r for r in csv.DictReader(open(LABELS / "E_footage.csv", encoding="utf-8"))}
    # The two clips that show a dataset A photo as a still have that photo as their source.
    footage.update({r["file"]: r for r in csv.DictReader(open(LABELS / "A_originals.csv", encoding="utf-8"))})
    clips = list(csv.DictReader(open(LABELS / "E_video_clips.csv", encoding="utf-8")))
    topic = {c["source"]: c["topic"] for c in clips}
    rng = random.Random(SEED)
    partner = {s: rng.choice(sorted(o for o in topic if topic[o] != topic[s])) for s in sorted(topic)}
    caption = {s: first_sentence(footage[s]["nasa_description"]) for s in topic}
    return {c["clip_id"]: {"file": c["file"], "source": c["source"], "topic": c["topic"],
                           "partner": partner[c["source"]], "true": caption[c["source"]],
                           "swapped": caption[partner[c["source"]]]} for c in clips}


def caption_video() -> None:
    """ADR-067: the video caption check on each clip with its own caption and a swapped one. CLIP
    scores of each keyframe against both are cached; the app's rule decides live."""
    from app.fusion import rules
    from app.models import EvidenceBundle, FlagStatus, Keyframe, Meta, Modality

    captions = _video_captions()
    path = CACHE / "caption_video" / "similarities.json"
    sims = _load(path)
    if all(cid in sims and sims[cid]["captions"] == [c["true"], c["swapped"]] for cid, c in captions.items()):
        print("  keyframe scores replayed from the cache")
    else:
        _child("caption-video")
        sims = _load(path)
    production_settings()
    rows = []
    for cid, c in sorted(captions.items()):
        s = sims[cid]
        for kind in ("true", "swapped"):
            frames = [Keyframe(timestamp=t, caption_similarity=v) for t, v in zip(s["keyframes"], s[kind])]
            bundle = EvidenceBundle(caption=c[kind], keyframes=frames, extractor_status={"caption_match": FlagStatus.FIRED},
                                    meta=Meta(modality=Modality.VIDEO, duration_s=s["duration_s"],
                                              frame_timestamps=s["keyframes"]))
            flag = rules.caption_scene_mismatch_rule(bundle)
            rows.append({"clip": cid, "source": c["source"], "topic": c["topic"], "caption_kind": kind,
                         "caption": c[kind], "status": flag.status.value,
                         "best_similarity": round(max(s[kind]), 4) if s[kind] else None, "keyframes": len(s[kind])})
    res = {"threshold": rules.CAPTION_IMAGE_ONLY_THRESHOLD, "clips": len(captions),
           "sources": len({c["source"] for c in captions.values()}),
           "partners": {c["source"]: c["partner"] for c in captions.values()}}
    for kind, name in (("true", "own_caption_flagged"), ("swapped", "swapped_caption_flagged")):
        rs = [r for r in rows if r["caption_kind"] == kind]
        res[name] = {**rate(sum(r["status"] == "fired" for r in rs), len(rs)),
                     "not_assessed": sum(r["status"] == "not_assessed" for r in rs)}
    true = [r["best_similarity"] for r in rows if r["caption_kind"] == "true"]
    swapped = [r["best_similarity"] for r in rows if r["caption_kind"] == "swapped"]
    res["auc_best_similarity"] = round(sum((s < t) + 0.5 * (s == t) for t in true for s in swapped)
                                       / (len(true) * len(swapped)), 3)
    res["rows"] = rows
    write("wp6_caption_video.json", res)
    print(f"  own caption flagged {res['own_caption_flagged']['k']}/{res['clips']} {res['own_caption_flagged']['ci']}; "
          f"swapped flagged {res['swapped_caption_flagged']['k']}/{res['clips']} {res['swapped_caption_flagged']['ci']}; "
          f"AUC {res['auc_best_similarity']}")


WHISPER_ON = ("tuning_long", "e", "original")


def whisper() -> None:
    """ADR-029 and ADR-049, again with the app's current decoding: tiny against base, and decoding
    settings a, b and c. Transcripts (with their times) are cached; the scores run live."""
    import eval_whisper as ew

    folder = CACHE / "whisper"
    for model in ("tiny", "base"):
        if (folder / f"wp3_whisper_{model}.json").exists():
            print(f"  {model}: replayed from the cache")
        else:
            _child(f"whisper-run|{model}")
    if all((folder / f"wp5_whisper_{s}_{on}.json").exists() for s in ew.SETTINGS for on in WHISPER_ON):
        print("  decoding settings: replayed from the cache")
    else:
        _child("whisper-decoding")
    with tempfile.TemporaryDirectory() as tmp:
        for f in folder.glob("*.json"):
            shutil.copy(f, Path(tmp) / f.name)
        ew.OUT = Path(tmp)
        ew.compare()
        ew.choose_long()
        comparison = json.loads((Path(tmp) / "wp3_whisper_comparison.json").read_text(encoding="utf-8"))
        choice = json.loads((Path(tmp) / "wp5_whisper_decoding_choice_long.json").read_text(encoding="utf-8"))
    decoding = {}
    for s in ew.SETTINGS:
        for on in WHISPER_ON:
            d = json.loads((folder / f"wp5_whisper_{s}_{on}.json").read_text(encoding="utf-8"))
            decoding.setdefault(s, {"options": d["options"]})[on] = {
                k: d[k] for k in ("items", "wer_pct", "invented_sentences", "items_with_invented", "mean_s")}
    write("wp6_whisper.json", {"models": comparison, "decoding": decoding, "decoding_choice_long": choice,
                               "same_choices_as_before": {"model base (ADR-029)": comparison["decision"]["chosen"] == "base",
                                                          "decoding b (ADR-049)": choice["chosen"] == "b"}})


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


def _archive_answers_saved() -> tuple[int, int]:
    """Pages whose archive availability answer is saved, of all the pages the replay dates."""
    import httpx

    import eval_web_lookup
    from app.extractors import web_lookup as lookup

    urls = eval_web_lookup.page_urls()
    return sum(1 for u in urls if eval_web_lookup._saved(
        str(httpx.URL(lookup.WAYBACK_AVAILABLE, params={"url": u, "timestamp": "19900101"})))), len(urls)


def web_archive() -> None:
    """ADR-050: the WP-4 evaluation replayed from saved page and archive answers only. Until every
    archive answer is saved (archive.org limits the fetch), the replay is left out and says so."""
    import eval_web_lookup

    production_settings()  # no key: a Vision call is impossible, the cached responses are used
    saved, pages_total = _archive_answers_saved()
    complete, reason = saved == pages_total, f"archive answers saved for {saved} of {pages_total} pages"
    if complete:
        try:
            eval_web_lookup.dates_from_inputs()
        except SystemExit as exc:  # a second-try archive answer is still missing
            complete, reason = False, str(exc)
    if not complete:
        print(f"  the replay from saved inputs waits for the archive fetch: {reason}")
    runs = {}
    for name, file in WEB_RUNS:
        path = OUT / file
        replay = file.startswith("wp4_web_report_saved")
        runs[name] = json.loads(path.read_text(encoding="utf-8")) if path.exists() and (complete or not replay) else None
    res = {"archive_answers_saved": {"pages": saved, "of": pages_total}, "replay_complete": complete, "runs": runs}
    if complete:
        pages = json.loads((OUT / "wp4_page_dates_saved.json").read_text(encoding="utf-8"))
        res["saved_inputs"] = {k: pages[k] for k in ("pages", "page_fetch_failures", "page_status", "transitions")}
    else:
        res["waiting_for"] = reason
    write("wp6_web_archive.json", res)
    for name, r in runs.items():
        if r is None:
            print(f"  {name}: {'waits for the archive fetch' if 'saved inputs' in name else 'not on this machine'}")
            continue
        a, v = r["A"], r["VERITE"]
        print(f"  {name:<38} A found {a['found'][0]}/{a['found'][1]} dated {a['dated'][0]}/{a['dated'][1]} "
              f"errors {a['dating_errors'][0]}/{a['dating_errors'][1]}  VERITE dated {v['dated'][0]}/{v['dated'][1]}  "
              f"sources {r['date_sources']}")


# ------------------------------------------------------------------------------ performance
PERF_IMAGES = 31  # as in WP-2's timing: the first call loads the models, 30 are warm
PERF_CLIP = "E15.mp4"
PERF_CLIP_CAPTION = "A barge arrives at the space centre"  # as in WP-3's and WP-5's timings


def performance() -> None:
    """Time per stage and peak memory: the image path (shipped settings, and with the LLM on), the
    video path on a 60 s clip (fresh, then warm), and every model on its own, each in its own process."""
    parts = []
    for config in ("shipped", "llm"):
        _child(f"performance-image|{config}")
        part = OUT / f"wp6_performance_image_{config}.json"
        parts.append(json.loads(part.read_text(encoding="utf-8")))
        part.unlink()
    write("wp6_performance_image.json", {"configurations": parts})
    subprocess.run([sys.executable, str(_BACKEND / "scripts" / "time_video_path.py"), str(CLIPS / PERF_CLIP),
                    PERF_CLIP_CAPTION, "--runs", "2", "--out", "wp6_performance_video.json"],
                   check=True, env=_env(), cwd=ROOT, stdout=subprocess.DEVNULL)
    print("wrote results/wp6_performance_video.json")
    subprocess.run([sys.executable, str(_BACKEND / "scripts" / "probe_models.py"), "--json",
                    str(OUT / "wp6_model_probe.json")], check=True, env=_env(), cwd=ROOT)


def _performance_image(config: str) -> None:
    """The image path on the first 31 VERITE images of the sample, stage by stage, in this process."""
    import statistics

    import psutil

    from app.adapters import image_adapter
    from app.fusion import llm_reasoner, scorecard

    production_settings(**({"use_llm": "true"} if config == "llm" else {}))
    ecm = _ecm()
    chosen = sorted({p["image"]: p for p in ecm.load_pairs("sample")}.items())[:PERF_IMAGES]
    names = {"extract_on_screen_text": "OCR", "find_web_matches": "image lookup", "describe_scene": "BLIP captioner",
             "measure_caption_fit": "CLIP caption check"}
    current: list[tuple[str, float]] = []

    def timed(label: str, fn):
        def wrapped(*args, **kwargs):
            s = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                current.append((label, time.perf_counter() - s))
        return wrapped

    for name, label in names.items():
        setattr(image_adapter, name, timed(label, getattr(image_adapter, name)))
    scorecard.reason_over_text = timed("LLM", scorecard.reason_over_text)
    proc = psutil.Process()
    runs = []
    with llm_reasoner.record_calls([], bypass_cache=True):  # every LLM call is a real one
        for image, p in chosen:
            data = p["file"].read_bytes()
            current.clear()
            s = time.perf_counter()
            bundle = image_adapter.build_bundle(data, p["caption"])
            built = time.perf_counter() - s
            s = time.perf_counter()
            scorecard.build_scorecard(bundle)
            fused = time.perf_counter() - s
            stages = {}
            for label, seconds in current:
                stages[label] = stages.get(label, 0.0) + seconds
            stages["rules and scorecard"] = fused - stages.get("LLM", 0.0)
            runs.append({"image": image, "stages_s": stages, "total_s": built + fused,
                         "rss_mb": round(proc.memory_info().rss / 2**20, 1)})

    def summary(values: list[float]) -> dict:
        ms = sorted(1000 * v for v in values)
        return {"mean_ms": round(statistics.mean(ms), 1), "median_ms": round(statistics.median(ms), 1),
                "p95_ms": round(ms[math.ceil(0.95 * len(ms)) - 1], 1), "n": len(ms)}

    first, warm = runs[0], runs[1:]
    labels = list(dict.fromkeys(k for r in runs for k in r["stages_s"]))
    mem = proc.memory_info()
    out = {"configuration": config, "images": len(runs),
           "settings": {"caption_match_method": "image", "use_llm": config == "llm"},
           "first_call_s": {k: round(v, 3) for k, v in first["stages_s"].items()} | {"total": round(first["total_s"], 3)},
           "warm": {k: summary([r["stages_s"].get(k, 0.0) for r in warm]) for k in labels}
           | {"total": summary([r["total_s"] for r in warm])},
           "peak_memory_mb": round(getattr(mem, "peak_wset", mem.rss) / 2**20, 1), "rows": runs}
    (OUT / f"wp6_performance_image_{config}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


# ------------------------------------------------------------------------------ interface review
def _review_summary(d: dict) -> dict:
    acc = {k: v["counts_by_impact"] for k, v in d["accessibility"].items()}
    return {
        "screens": len(acc),
        "accessibility_serious_or_critical": sum(v["serious"] + v["critical"] for v in acc.values()),
        "accessibility_critical": sum(v["critical"] for v in acc.values()),
        "accessibility_by_impact": {i: sum(v[i] for v in acc.values()) for i in ("critical", "serious", "moderate", "minor")},
        "targets_under_24": sum(len(v["under_24x24"]) for v in d["click_targets"].values()),
        "targets_under_44": sum(v["under_44x44"] for v in d["click_targets"].values()),
        "controls_measured": sum(v["controls"] for v in d["click_targets"].values()),
        "text_under_16px_max_share": max(v["share_under_16px"] for v in d["text_size"].values()),
        "text_under_16px_chars": sum(v["chars_under_16px"] for v in d["text_size"].values()),
        "text_chars": sum(v["chars"] for v in d["text_size"].values()),
        "reading_level": {k: d["reading_level"][k] for k in ("lines", "at_or_below_8", "above_8", "median", "max")},
        "steps_per_task": {t: v["steps"] for t, v in d["steps_per_task"].items()},
        "predicted_time_s": {t: v["seconds"] for t, v in d["predicted_time"]["tasks"].items()},
        "measured_waits_s": {t: v["response_s"] for t, v in d["predicted_time"]["tasks"].items()},
        "scrolls": {size: {t: [s["scrolls_needed"] for s in steps] for t, steps in tasks.items()}
                    for size, tasks in d["controls_visible"].items()},
        "screens_of_scrolling": {size: sum(s["scrolls_needed"] or 0 for steps in tasks.values() for s in steps)
                                 for size, tasks in d["controls_visible"].items()},
        "ui_verdict_words": dict(Counter(h["word"] for h in d["no_verdict_wording"]["ui_hits"])),
        "backend_verdict_words": d["no_verdict_wording"].get("backend", {}).get("hits"),
        "generated": d["generated"]}


def interface_review() -> None:
    """The interface review's round B measures on the current build; round A is read as frozen."""
    npx = shutil.which("npx") or shutil.which("npx.cmd")
    if not npx:
        raise SystemExit("npx is not on PATH: install Node.js and run npm install in frontend/")
    subprocess.run([npx, "playwright", "test", "-c", "playwright.review.config.js"], cwd=FRONTEND,
                   env={**_env(), "ROUND": "B"}, check=True)
    rounds = {r: json.loads((OUT / f"interface_review_{r}.json").read_text(encoding="utf-8")) for r in "AB"}
    write("wp6_interface_review.json", {"A": _review_summary(rounds["A"]), "B": _review_summary(rounds["B"]),
                                        "files": {"A": "results/interface_review_A.json (frozen)",
                                                  "B": "results/interface_review_B.json"}})


# ------------------------------------------------------------------------------ audits
_QUOTED = re.compile(r'"[^"]*"|“[^”]*”')
_WEB = re.compile(r"(?:https?://|www\.)\S+")
_CAPITALS = re.compile(r"(words in capitals: \d+% \(\d+ of \d+ words: )[^)]*\)")


def _seen_files() -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(SEEN_DIR.glob("*.json"))]


def _own_words(text: str, llm: list[str], inputs: list[str]) -> str:
    """The app's own words in a line: without the LLM's explanations, the texts the flag was built
    from, anything in double quotes, web addresses and the capitals the style marker lists."""
    for s in llm:
        if s in text:
            text = text.replace(s, " ")
    for s in inputs:
        if s in text:
            text = re.sub(r"(?<!\w)" + re.escape(s) + r"(?!\w)", " ", text)
    text = _CAPITALS.sub(r"\1...)", text)
    text = _QUOTED.sub(" ", text)
    text = _WEB.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def _removals(data: dict) -> tuple[list[str], list[str]]:
    llm = sorted(data["llm_explanations"], key=len, reverse=True)
    inputs = sorted((s for s in data["inputs"] if len(s) >= 20), key=len, reverse=True)
    return llm, inputs


def no_verdict() -> None:
    """ADR-067: the interface review's 29 verdict words in every output the evaluation produced."""
    from interface_text_audit import _VERDICT, VERDICT_WORDS, all_flags

    lines, llm_texts = {}, {}
    for data in _seen_files():
        llm, inputs = _removals(data)
        texts = [(f"{f['check']} ({f['source']}, {f['status']})", field, f[field]) for f in data["flags"]
                 for field in ("plain_explanation", "what_to_check", "evidence")]
        texts += [("scorecard", "summary", s) for s in data["summaries"]]
        for where, field, text in texts:
            entry = lines.setdefault(text, {"where": where, "field": field, "steps": set(),
                                            "own": _own_words(text, llm, inputs)})
            entry["steps"].add(data["step"])
        for s in data["llm_explanations"]:
            llm_texts.setdefault(s, set()).add(data["step"])

    def hits(text: str) -> list:
        return [(m.group(0).lower(), text[max(0, m.start() - 60):m.end() + 60]) for m in _VERDICT.finditer(text)]

    own_hits, content_hits = [], Counter()
    for text, e in lines.items():
        found = hits(e["own"])
        own_hits += [{"word": w, "where": e["where"], "field": e["field"], "steps": sorted(e["steps"]), "context": c}
                     for w, c in found]
        own_words = Counter(w for w, _ in found)
        for w, n in Counter(w for w, _ in hits(text)).items():
            content_hits[w] += max(0, n - own_words.get(w, 0))
    llm_hits = [{"word": w, "steps": sorted(steps), "context": c} for text, steps in llm_texts.items() for w, c in hits(text)]
    enumerated = all_flags()
    written = sorted({s for f in enumerated for s in (f.plain_explanation, f.what_to_check, f.evidence)})
    review = json.loads((OUT / "interface_review_B.json").read_text(encoding="utf-8"))["no_verdict_wording"]
    res = {"words": list(VERDICT_WORDS),
           "evaluation_outputs": {"steps": sorted({s for e in lines.values() for s in e["steps"]}),
                                  "distinct_lines": len(lines), "own_word_hits": len(own_hits),
                                  "own_word_hits_by_word": dict(Counter(h["word"] for h in own_hits)),
                                  "hits_in_quoted_or_input_text": dict(content_hits), "own_hits": own_hits},
           "llm_explanations": {"distinct": len(llm_texts), "hits": len(llm_hits),
                                "texts_with_a_hit": len({h["context"] for h in llm_hits}),
                                "by_word": dict(Counter(h["word"] for h in llm_hits)), "rows": llm_hits},
           "every_branch_of_every_check": {"lines": len(written),
                                           "hits": sum(len(hits(t)) for t in written)},
           "interface_round_B": {"hits": len(review["ui_hits"]),
                                 "by_word": dict(Counter(h["word"] for h in review["ui_hits"])),
                                 "screens": sorted({h["screen"] for h in review["ui_hits"]})}}
    write("wp6_no_verdict.json", res)
    print(f"  {len(lines)} distinct lines from {len(res['evaluation_outputs']['steps'])} steps: "
          f"{len(own_hits)} verdict words in the app's own words {res['evaluation_outputs']['own_word_hits_by_word']}; "
          f"{sum(content_hits.values())} in quoted or input text; LLM explanations {len(llm_texts)} with "
          f"{len(llm_hits)} hits {res['llm_explanations']['by_word']}")


def readability() -> None:
    """ADR-067: Flesch-Kincaid grade of every distinct explanation and "what to check" line the
    evaluation produced, in the app's own words."""
    from interface_text_audit import fk_grade

    lines = {}
    for data in _seen_files():
        llm, inputs = _removals(data)
        for f in data["flags"]:
            for field in ("plain_explanation", "what_to_check"):
                text = _own_words(f[field], llm, inputs)
                e = lines.setdefault(text, {"field": field, "check": f["check"], "status": f["status"], "steps": set()})
                e["steps"].add(data["step"])
    items = [{**meta, "steps": sorted(meta["steps"]), "text": text, **fk_grade(text)} for text, meta in lines.items()]
    grades = sorted(i["grade"] for i in items if i["grade"] is not None)
    res = {"formula": "Flesch-Kincaid grade = 0.39 * words/sentence + 11.8 * syllables/word - 15.59",
           "target": "grade 8 or lower", "lines": len(items), "graded": len(grades),
           "at_or_below_8": sum(g <= 8 for g in grades), "above_8": sum(g > 8 for g in grades),
           "median": grades[len(grades) // 2] if grades else None, "max": grades[-1] if grades else None,
           "share_at_or_below_8": rate(sum(g <= 8 for g in grades), len(grades)),
           "above_8_lines": [i for i in items if i["grade"] is not None and i["grade"] > 8],
           "grades": grades}
    write("wp6_readability.json", res)
    print(f"  {res['graded']} lines: {res['at_or_below_8']} at grade 8 or below, median {res['median']}, max {res['max']}")


def numbers() -> None:
    import final_eval_report

    final_eval_report.write_numbers()


def charts() -> None:
    import final_eval_report

    final_eval_report.draw_charts()


# ------------------------------------------------------------------------------ child processes
def _run_child(task: str) -> None:
    kind, *rest = task.split("|")
    production_settings()
    if kind == "verite":
        pair_set, phase = rest
        ecm = _ecm()
        pairs = ecm.load_pairs(pair_set)
        if phase == "blip_ocr":
            ecm.phase_blip_ocr(pairs, pair_set)
        elif phase.startswith("clip:"):
            ecm.phase_clip(pairs, phase[5:], pair_set)
        else:
            ecm.phase_spacy(pairs, pair_set)
    elif kind == "verite-limits":
        import eval_picture_limits as epl

        epl.LIMITS = _ecm().CACHE / "picture_limits.json"
        epl.do_measure()
    elif kind == "speech-swap":
        import eval_speech_picture as esp

        esp.CACHE = CACHE / "speech_swap"
        esp.do_scores()
    elif kind == "video-bundles":
        import eval_speech_picture as esp
        from app.adapters.video_adapter import build_video_bundle

        folder = CACHE / "video_bundles"
        folder.mkdir(parents=True, exist_ok=True)
        for cid, c in sorted(esp.clips().items()):
            out = folder / f"{cid}.json"
            if out.exists():
                continue
            started = time.perf_counter()
            bundle = build_video_bundle(CLIPS / c["file"], None, c["file"])
            tmp = out.with_suffix(".tmp")
            tmp.write_text(bundle.model_dump_json(), encoding="utf-8")
            tmp.replace(out)
            print(f"  {cid}: {len(bundle.transcript_segments)} segments ({time.perf_counter() - started:.0f} s)", flush=True)
    elif kind == "caption-video":
        from app.adapters.video_adapter import _clip_vectors
        from app.extractors.video import VideoError, grab_frame, probe, sample_keyframes

        path = CACHE / "caption_video" / "similarities.json"
        done = _load(path)
        for cid, c in sorted(_video_captions().items()):
            if cid in done and done[cid]["captions"] == [c["true"], c["swapped"]]:
                continue
            clip = CLIPS / c["file"]
            info = probe(clip)
            times, how = sample_keyframes(clip, info.duration_s)
            frames = {}
            for t in times:
                try:
                    frames[t] = grab_frame(clip, t)
                except VideoError:
                    continue
            text_vecs, frame_vecs = _clip_vectors([c["true"], c["swapped"]], frames)
            keys = sorted(frames)
            done[cid] = {"captions": [c["true"], c["swapped"]], "duration_s": info.duration_s, "sampler": how,
                         "keyframes": keys, "true": [float(frame_vecs[t] @ text_vecs[0]) for t in keys],
                         "swapped": [float(frame_vecs[t] @ text_vecs[1]) for t in keys]}
            _save(path, done)
            print(f"  {cid}: {len(keys)} keyframes", flush=True)
    elif kind == "whisper-run":
        import eval_whisper as ew

        ew.OUT = CACHE / "whisper"
        ew.OUT.mkdir(parents=True, exist_ok=True)
        ew.run(rest[0])
    elif kind == "whisper-decoding":
        import eval_whisper as ew

        ew.OUT = CACHE / "whisper"
        ew.OUT.mkdir(parents=True, exist_ok=True)
        for s in ew.SETTINGS:
            for on in WHISPER_ON:
                if not (ew.OUT / f"wp5_whisper_{s}_{on}.json").exists():
                    ew.decoding(s, on)
    elif kind == "performance-image":
        _performance_image(rest[0])
    else:
        raise SystemExit(f"unknown child task {task!r}")


STEPS = {"synthetic": synthetic, "matching-a": matching_a, "hard-pairs": hard_pairs, "date-check": date_check,
         "verite-features": verite_features, "verite-llm": verite_llm, "verite-heldout": verite_heldout,
         "verite-fresh": verite_fresh, "picture-limits": picture_limits, "llm-verite": llm_verite,
         "ablations": ablations, "framing-f": framing_f, "framing-llm": framing_llm, "speech-swap": speech_swap,
         "video-pipeline": video_pipeline, "caption-video": caption_video, "whisper": whisper,
         "fault-injection": fault_injection, "web-archive": web_archive, "performance": performance,
         "interface-review": interface_review, "no-verdict": no_verdict, "readability": readability,
         "numbers": numbers, "charts": charts}
NOT_AUDITED = {"no-verdict", "readability", "numbers", "charts"}  # they read the outputs; they make none


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("steps", nargs="*", help="steps to run (default: all)")
    parser.add_argument("--rebuild", action="store_true",
                        help="delete the cached model outputs of the steps being run and build them again from the models")
    parser.add_argument("--child", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child:
        return _run_child(args.child)
    unknown = [s for s in args.steps if s not in STEPS]
    if unknown:
        parser.error(f"unknown step(s): {', '.join(unknown)}; choose from {', '.join(STEPS)}")
    steps = args.steps or list(STEPS)
    if args.rebuild:
        for step in steps:
            for folder in CACHES.get(step, []):
                if (CACHE / folder).exists():
                    shutil.rmtree(CACHE / folder)
                    print(f"deleted the cache data/final_eval_cache/{folder}")
    _record_outputs()
    log = {"started": time.strftime("%Y-%m-%dT%H:%M:%S"), "rebuild": args.rebuild, "steps": {}}
    for step in steps:
        print(f"== {step}", flush=True)
        SEEN.reset()
        started = time.perf_counter()
        STEPS[step]()
        if step not in NOT_AUDITED:
            SEEN.save(step)
        log["steps"][step] = round(time.perf_counter() - started, 1)
        print(f"   ({step}: {log['steps'][step]:.0f} s)", flush=True)
    if not args.steps:
        write("wp6_run.json", log)


if __name__ == "__main__":
    main()
