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
  report [--tag T]       the tables, with Wilson 95% intervals, and the date-source split.

Re-dating from saved inputs, so that a change to the dating code is measured apart from the
network (the project's decision log, ADR-043):
  inputs --what pages    fetch every page the evaluation dates, once, one at a time with a pause,
                         into the ignored data/live_cache/raw/ (whatever the page answers is kept,
                         a failure included)
  inputs --what archive  every Wayback Machine request the dating code can make for those pages,
                         once, one at a time, 15 s apart; stops at the first refusal (429, or any
                         failure of the availability API) without saving it, and picks up where it
                         stopped when run again (to be completed in WP-6)
  metadata-from-inputs   the fix of ADR-043 alone: the page-metadata dating before it and the
                         current one on the same saved page answers (no network). The fix changes
                         only this step, so this is its whole effect
  dates-from-inputs      once every archive answer is saved: the whole dating, before ADR-043 and
                         now, on the saved inputs only; each run's tables, and the current dates
                         written into the committed cache

Run from the repo root: python backend/scripts/eval_web_lookup.py run --set A
"""

from __future__ import annotations

import argparse
import base64
import csv
import gzip
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path
from statistics import median

import httpx

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

ROOT = _BACKEND.parent
OUT = ROOT / "results"
EARLIER = ROOT / "data" / "earlier_runs"  # tracked records of runs that cannot be made again
EVAL_CACHE = "data/live_cache_eval"
LOCAL_CACHE = "data/live_cache"
D_FOLDER = Path(os.path.expanduser("~")) / "Desktop" / "aegis_D"
D_LANDMARKS = {"D12"}
SEED = 20260927
VERITE_N = 50
RAW = ROOT / "data" / "live_cache" / "raw"   # ignored: saved answers of pages and the archive
PAGE_PAUSE_S = 1.0
ARCHIVE_PAUSE_S = 15.0  # 5 s still met a 429 after 251 requests (ADR-043)
ARCHIVE_HOSTS = {"archive.org", "web.archive.org"}


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


def run(which: str, redate: bool = False, tag: str = "") -> None:
    os.environ["AEGIS_LIVE_CACHE_DIR"] = LOCAL_CACHE if which == "D" else EVAL_CACHE
    from app.config import get_settings
    get_settings.cache_clear()
    
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
    (OUT / f"wp4_web_{which}{'_' + tag if tag else ''}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"{which}: {len(rows)} images, {out['live_calls']} live calls "
          f"(month count {before} -> {out['calls_this_month_after']})")


def report(tag: str = "") -> None:
    res = {}
    sources = Counter()
    for which in ("A", "D", "VERITE"):
        path = OUT / f"wp4_web_{which}{'_' + tag if tag else ''}.json"
        if not path.exists():
            path = OUT / f"wp4_web_{which}.json"  # D has no pages to date, so it is not re-run
        if not path.exists():
            path = EARLIER / f"wp4_web_{which}.json"  # D's live run, kept since the photos are not public
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
    (OUT / f"wp4_web_report{'_' + tag if tag else ''}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


# ------------------------------------------------------------------------------ saved inputs
class Refused(RuntimeError):
    """The archive refused a request; nothing is saved for it and the step stops."""


def _raw_path(url: str) -> Path:
    return RAW / f"{hashlib.sha256(url.encode('utf-8')).hexdigest()}.json.gz"


def _saved(url: str) -> dict | None:
    try:
        return json.loads(gzip.decompress(_raw_path(url).read_bytes()))
    except OSError:
        return None


def _save_raw(url: str, answer: dict) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    answer = {"url": url, "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **answer}
    _raw_path(url).write_bytes(gzip.compress(json.dumps(answer).encode("utf-8")))


def _response(saved: dict, request):
    import httpx

    if "error" in saved:
        raise httpx.ConnectError(f"saved answer: {saved['error']}", request=request)
    headers = {k: v for k, v in saved["headers"].items() if v is not None}
    return httpx.Response(saved["status"], headers=headers, content=base64.b64decode(saved["content"]),
                          request=request)


class _Inputs(httpx.BaseTransport):
    """An httpx transport answering from the saved inputs; with ``fetch``, it fetches and saves
    what is not saved yet, one request at a time, pausing first."""

    def __init__(self, fetch: bool = False):
        self.fetch, self.misses, self.fetched = fetch, [], 0
        self.real = httpx.HTTPTransport() if fetch else None

    def handle_request(self, request):
        import httpx

        url = str(request.url)
        saved = _saved(url)
        if saved is None and not self.fetch:
            self.misses.append(url)
            raise httpx.ConnectError("not in the saved inputs", request=request)
        if saved is None:
            archive = request.url.host in ARCHIVE_HOSTS
            availability = archive and "cdx" not in request.url.path
            time.sleep(ARCHIVE_PAUSE_S if archive else PAGE_PAUSE_S)
            self.fetched += 1
            try:
                resp = self.real.handle_request(request)
                content = resp.read()
            except httpx.HTTPError as exc:
                if availability:
                    raise Refused(f"the archive's availability API failed: {type(exc).__name__}") from exc
                _save_raw(url, {"error": type(exc).__name__})
                raise
            if archive and (resp.status_code == 429 or (availability and resp.status_code != 200)):
                raise Refused(f"the archive answered HTTP {resp.status_code} to {url[:90]}")
            saved = {"status": resp.status_code, "content": base64.b64encode(content).decode("ascii"),
                     "headers": {"content-type": resp.headers.get("content-type"),
                                 "location": resp.headers.get("location")}}
            _save_raw(url, saved)
        return _response(saved, request)

def _use_inputs(lookup, transport) -> None:
    import httpx

    lookup._client = lambda timeout: httpx.Client(transport=transport, timeout=timeout, follow_redirects=True,
                                                  headers={"User-Agent": lookup.USER_AGENT})


def page_urls() -> list[str]:
    """Every page the evaluation dates: the first MAX_PAGES matching pages of each A and VERITE image."""
    os.environ["AEGIS_LIVE_CACHE_DIR"] = EVAL_CACHE
    from app.config import get_settings

    get_settings.cache_clear()
    from app.extractors import web_lookup as lookup

    urls = []
    for which in ("A", "VERITE"):
        for item in images(which):
            entry = lookup._load(hashlib.sha256(item["path"].read_bytes()).hexdigest())
            urls += [p["url"] for p in lookup.matching_pages(entry["vision"])[:lookup.MAX_PAGES]]
    return sorted(set(urls))


def fetch_inputs(what: str) -> None:
    from app.extractors import web_lookup as lookup

    urls = page_urls()
    transport = _Inputs(fetch=True)
    _use_inputs(lookup, transport)
    try:
        for n, url in enumerate(urls, 1):
            if what == "pages":
                try:
                    with lookup._client(lookup.PAGE_TIMEOUT_S) as client:
                        client.get(url)
                except Exception:  # noqa: BLE001 - the failure is saved as this page's answer
                    pass
            else:
                lookup.date_from_wayback(url)
            if n % 25 == 0:
                print(f"  {what}: {n}/{len(urls)} ({transport.fetched} fetched this run)", flush=True)
    except Refused as exc:
        sys.exit(f"stopped after {transport.fetched} requests this run: {exc}")
    print(f"{what}: all {len(urls)} pages have their saved answers ({transport.fetched} fetched this run)")


def _date_from_html_before_adr_043(html: str) -> str | None:
    """web_lookup.date_from_html as it was before ADR-043 (commit 621468d)."""
    from htmldate import find_date

    return find_date(html, extensive_search=False, original_date=True, outputformat="%Y-%m-%d",
                     max_date=date.today().isoformat())


def metadata_from_inputs() -> None:
    """ADR-043's fix alone: both versions of the page-metadata dating on the same saved pages."""
    from app.extractors import web_lookup as lookup

    urls = page_urls()
    transport = _Inputs(fetch=False)
    _use_inputs(lookup, transport)
    rows, answers = [], Counter()
    for url in urls:
        try:
            with lookup._client(lookup.PAGE_TIMEOUT_S) as client:
                resp = client.get(url)
            answers[str(resp.status_code)] += 1
            html = resp.text if resp.status_code == 200 and "html" in resp.headers.get("content-type", "html") else None
        except httpx.HTTPError as exc:
            answers[type(exc).__name__] += 1
            html = None
        before = _date_from_html_before_adr_043(html) if html else None
        after = lookup.date_from_html(html) if html else None
        rows.append({"url": url, "before": before, "after": after})
    if transport.misses:
        sys.exit(f"{len(transport.misses)} requests had no saved answer, e.g. {transport.misses[0][:100]}")
    changed = [r for r in rows if r["before"] != r["after"]]
    out = {"pages": len(rows), "page_answers": dict(answers),
           "dated_before": sum(r["before"] is not None for r in rows),
           "dated_after": sum(r["after"] is not None for r in rows),
           "moved_earlier": sum(bool(r["before"] and r["after"] and r["after"] < r["before"]) for r in changed),
           "moved_later": sum(bool(r["before"] and r["after"] and r["after"] > r["before"]) for r in changed),
           "lost": sum(bool(r["before"] and not r["after"]) for r in changed),
           "gained": sum(bool(r["after"] and not r["before"]) for r in changed),
           "changed": changed}
    (OUT / "wp4_htmldate_saved_pages.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "changed"}, indent=1))
    for r in changed:
        print(f"  {r['before']} -> {r['after']}  {r['url'][:90]}")


def dates_from_inputs() -> None:
    from app.extractors import web_lookup as lookup

    current = lookup.date_from_html
    urls = page_urls()
    dates = {}
    for version, dater in (("before", _date_from_html_before_adr_043), ("after", current)):
        transport = _Inputs(fetch=False)
        _use_inputs(lookup, transport)
        lookup.date_from_html = dater
        for which in ("A", "VERITE"):
            run(which, redate=True, tag=f"saved_{version}")
        dates[version] = {url: lookup.date_page(url) for url in urls}
        if transport.misses:
            sys.exit(f"{len(transport.misses)} requests had no saved answer, e.g. {transport.misses[0][:100]}")
        report(f"saved_{version}")
    lookup.date_from_html = current
    kinds, changed = Counter(), []
    for url in urls:
        b, a = dates["before"][url], dates["after"][url]
        key = f"{b['source'] or 'none'} -> {a['source'] or 'none'}"
        if b["date"] != a["date"]:
            changed.append({"url": url, "before": b, "after": a})
            key += " (date changed)"
        kinds[key] += 1
    raw = [_saved(u) or {} for u in urls]
    out = {"pages": len(urls), "page_fetch_failures": sum("error" in r for r in raw),
           "page_status": dict(Counter(str(r.get("status", "error")) for r in raw)),
           "transitions": dict(kinds), "changed": changed, "dates": dates}
    (OUT / "wp4_page_dates_saved.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("pages", "page_fetch_failures", "page_status", "transitions")}, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["run", "report", "inputs", "metadata-from-inputs", "dates-from-inputs"])
    parser.add_argument("--set", dest="which", choices=["A", "D", "VERITE"])
    parser.add_argument("--redate", action="store_true", help="date every page again from the cached responses")
    parser.add_argument("--tag", default="", help="run/report: a suffix for the results files")
    parser.add_argument("--what", choices=["pages", "archive"], help="inputs: which answers to fetch")
    args = parser.parse_args()
    if args.step == "run":
        run(args.which, args.redate, args.tag)
    elif args.step == "report":
        report(args.tag)
    elif args.step == "inputs":
        fetch_inputs(args.what)
    elif args.step == "metadata-from-inputs":
        metadata_from_inputs()
    else:
        dates_from_inputs()


if __name__ == "__main__":
    main()
