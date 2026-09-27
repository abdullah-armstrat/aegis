"""WP-4 evaluation: the live reverse-image lookup on datasets A, D and a VERITE sample.

The definitions were fixed before any live call (the project's decision log): a match is a page
listing a full or partial copy; an image is found online with at least one; dates are the app's
own (htmldate, else the Wayback Machine's first capture); the earliest dated page is the earliest
known appearance.

  run --set A|D|VERITE   look every image up through the app's web_lookup (the cache first, then
                         one live call per image not yet cached). A and VERITE cache into the
                         committed data/live_cache_eval/; D, the user's own photos kept outside the
                         repository, caches into the ignored data/live_cache/. Stops at the first
                         failed live call.
  run --set X --redate   the same, after dropping the cached page dates of the set's images, so
                         every page is dated again by the current code; the cached Vision
                         responses are reused, so no live call is needed (run it without the key).
  report                 the tables, with Wilson 95% intervals, and the date-source split.

Run from the repo root: python backend/scripts/eval_web_lookup.py run --set A
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from statistics import median

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

ROOT = _BACKEND.parent
OUT = ROOT / "results"
EVAL_CACHE = "data/live_cache_eval"
LOCAL_CACHE = "data/live_cache"
D_FOLDER = Path(os.path.expanduser("~")) / "Desktop" / "aegis_D"
D_LANDMARKS = {"D12"}
SEED = 20260927
VERITE_N = 50


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    ph = k / n
    centre = (ph + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (round(max(0.0, centre - half), 3) + 0.0, round(min(1.0, centre + half), 3))


def images(which: str) -> list[dict]:
    if which == "A":
        rows = csv.DictReader(open(ROOT / "data" / "labels" / "A_originals.csv", encoding="utf-8"))
        return [{"id": r["nasa_id"], "path": ROOT / "data" / "A_originals" / r["file"],
                 "nasa_date": r["earliest_date"]} for r in rows]
    if which == "D":
        if not D_FOLDER.is_dir():
            sys.exit(f"{D_FOLDER} is not there")
        files = sorted(D_FOLDER.glob("*.jpeg")) + sorted(D_FOLDER.glob("*.jpg"))
        return [{"id": f"D{i:02d}", "path": f, "landmark": f"D{i:02d}" in D_LANDMARKS}
                for i, f in enumerate(files, 1)]
    on_disk = {p.stem: p for p in (ROOT / "data" / "verite" / "images").iterdir()}
    pool = set()
    for name in ("C_verite_sample.csv", "C_verite_fresh.csv"):
        for r in csv.DictReader(open(ROOT / "data" / "labels" / name, encoding="utf-8")):
            if r["label"] == "miscaptioned" and r["image"] in on_disk:
                pool.add(r["image"])
    chosen = random.Random(SEED).sample(sorted(pool), VERITE_N)
    return [{"id": img, "path": on_disk[img]} for img in chosen]


def run(which: str, redate: bool = False) -> None:
    os.environ["AEGIS_LIVE_CACHE_DIR"] = LOCAL_CACHE if which == "D" else EVAL_CACHE
    from app.config import get_settings
    get_settings.cache_clear()
    import hashlib

    from app.extractors import web_lookup as lookup
    from app.extractors.web_lookup import calls_this_month, web_lookup
    from app.fusion.rules import recycled_context_rule
    from app.models import EvidenceBundle, FlagStatus, Meta, Modality

    before = calls_this_month()
    rows = []
    for item in images(which):
        if redate:
            sha = hashlib.sha256(item["path"].read_bytes()).hexdigest()
            entry = lookup._load(sha)
            if "vision" not in entry:
                sys.exit(f"{item['id']}: no cached Vision response, so re-dating would need a live call")
            entry.pop("page_dates", None)
            entry.pop("dated_at", None)
            lookup._save(sha, entry)
        result = web_lookup(item["path"].read_bytes())
        if result.status == FlagStatus.NOT_ASSESSED:
            sys.exit(f"{item['id']}: the lookup could not run, stopping: {result.detail}")
        dated = [m for m in result.matches if m.published_date]
        earliest = min(dated, key=lambda m: m.published_date) if dated else None
        row = {"id": item["id"], "live_call": result.live_call, "status": result.status.value,
               "found": result.pages_with_matches > 0, "pages_with_matches": result.pages_with_matches,
               "pages_dated_attempted": len(result.matches),
               "date_sources": [m.date_source for m in result.matches],
               "earliest_date": earliest.published_date if earliest else None,
               "earliest_source": earliest.date_source if earliest else None,
               "earliest_url": earliest.url if earliest else None}
        if which == "A":
            row["nasa_date"] = item["nasa_date"]
            bundle = EvidenceBundle(web_matches=result.matches,
                                    extractor_status={"reverse_image": result.status},
                                    extractor_detail={"web_search": result.detail},
                                    meta=Meta(modality=Modality.IMAGE, posted_date=date.today().isoformat()))
            row["fires_if_posted_today"] = recycled_context_rule(bundle).status == FlagStatus.FIRED
        if which == "D":
            row["landmark"] = item["landmark"]
        rows.append(row)
        print(f"  {item['id']:<24} {'live ' if result.live_call else 'cache'} {result.status.value:<6} "
              f"pages {result.pages_with_matches:>2}  earliest {row['earliest_date']} ({row['earliest_source']})")
    OUT.mkdir(exist_ok=True)
    out = {"set": which, "images": len(rows), "live_calls": sum(r["live_call"] for r in rows),
           "calls_this_month_before": before, "calls_this_month_after": calls_this_month(), "rows": rows}
    (OUT / f"wp4_web_{which}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"{which}: {len(rows)} images, {out['live_calls']} live calls "
          f"(month count {before} -> {out['calls_this_month_after']})")


def report() -> None:
    res = {}
    sources = Counter()
    for which in ("A", "D", "VERITE"):
        path = OUT / f"wp4_web_{which}.json"
        if not path.exists():
            continue
        d = json.loads(path.read_text(encoding="utf-8"))
        rows = d["rows"]
        for r in rows:
            sources.update(s or "none" for s in r["date_sources"])
        n = len(rows)
        found = sum(r["found"] for r in rows)
        dated = sum(r["earliest_date"] is not None for r in rows)
        entry = {"images": n, "live_calls": d["live_calls"],
                 "found": (found, n, wilson(found, n)), "dated": (dated, n, wilson(dated, n))}
        if which == "A":
            both = [r for r in rows if r["earliest_date"]]
            errors = [r for r in both if r["earliest_date"] < r["nasa_date"]]
            gaps = [(date.fromisoformat(r["earliest_date"]) - date.fromisoformat(r["nasa_date"])).days for r in both]
            fires = sum(r["fires_if_posted_today"] for r in rows)
            entry.update({"dating_errors": (len(errors), len(both), wilson(len(errors), len(both))),
                          "dating_error_ids": [(r["id"], r["earliest_date"], r["nasa_date"], r["earliest_source"])
                                               for r in errors],
                          "gap_days_median": median(gaps) if gaps else None,
                          "fires_if_posted_today": (fires, n, wilson(fires, n))})
        if which == "D":
            for label, sub in (("landmark", [r for r in rows if r["landmark"]]),
                               ("everyday", [r for r in rows if not r["landmark"]])):
                k = sum(r["found"] for r in sub)
                entry[f"false_alarms_{label}"] = (k, len(sub), wilson(k, len(sub)))
        res[which] = entry
    res["date_sources"] = dict(sources)
    res["live_calls_total"] = sum(v["live_calls"] for k, v in res.items() if isinstance(v, dict) and "live_calls" in v)
    (OUT / "wp4_web_report.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["run", "report"])
    parser.add_argument("--set", dest="which", choices=["A", "D", "VERITE"])
    parser.add_argument("--redate", action="store_true", help="date every page again from the cached responses")
    args = parser.parse_args()
    run(args.which, args.redate) if args.step == "run" else report()


if __name__ == "__main__":
    main()
