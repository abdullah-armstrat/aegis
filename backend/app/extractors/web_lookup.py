"""Live reverse-image lookup: Google Cloud Vision web detection, with each page dated.

Only pages with a full or partial copy of the image count, not "visually similar" ones. Vision
gives no dates, so each page is dated by the publication date it declares (htmldate) or else the
Wayback Machine's first capture. Results are cached by the image's SHA-256 and live calls are
capped at ``MONTHLY_LIMIT`` a month. Any failure gives NOT_ASSESSED with the reason. The API key
goes in a request header, not the URL, so it cannot end up in a log or error message.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

import httpx

from app.config import get_settings
from app.models import FlagStatus, WebMatch

KEY_ENV = "GOOGLE_VISION_API_KEY"
VISION_URL = "https://vision.googleapis.com/v1/images:annotate"
WAYBACK_AVAILABLE = "https://archive.org/wayback/available"
WAYBACK_CDX = "https://web.archive.org/cdx/search/cdx"
MONTHLY_LIMIT = 900    # live Vision calls allowed per calendar month
MAX_RESULTS = 20       # asked of Vision for each list
MAX_PAGES = 10         # pages dated per image, in Vision's order
VISION_TIMEOUT_S = 20.0
PAGE_TIMEOUT_S = 10.0
USER_AGENT = ("aegis-lookup/1.0 (University of London CM3070 student project; dates pages that show "
              "an image; contact 189360416+abdullah-armstrat@users.noreply.github.com)")
_REPO_ROOT = Path(__file__).resolve().parents[3]
_USAGE_LOCK = threading.Lock()


@dataclass
class WebLookupResult:
    """Result of a live web lookup; ``detail`` gives the reason when it is not FIRED."""

    matches: list[WebMatch] = field(default_factory=list)
    status: FlagStatus = FlagStatus.NOT_ASSESSED
    detail: str = ""
    live_call: bool = False        # True when this lookup spent a Vision call
    pages_with_matches: int = 0    # pages Vision listed with a full or partial copy


def live_available() -> bool:
    """True if a live lookup is possible, i.e. the key is set on the server."""
    return bool(os.environ.get(KEY_ENV))


def _cache_dir() -> Path:
    configured = get_settings().live_cache_dir
    path = Path(configured) if configured else Path("data/live_cache")
    return path if path.is_absolute() else _REPO_ROOT / path


def _usage_file() -> Path:
    return _REPO_ROOT / "data" / "live_cache" / "usage.json"


def calls_this_month() -> int:
    """Number of live Vision calls made this calendar month, from the usage file."""
    try:
        return json.loads(_usage_file().read_text(encoding="utf-8")).get(date.today().strftime("%Y-%m"), 0)
    except (OSError, ValueError):
        return 0


def _count_call() -> None:
    """Add one to this month's count in the usage file."""
    path = _usage_file()
    with _USAGE_LOCK:
        try:
            usage = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            usage = {}
        month = date.today().strftime("%Y-%m")
        usage[month] = usage.get(month, 0) + 1
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(usage, indent=1), encoding="utf-8")
        tmp.replace(path)


def _load(sha: str) -> dict:
    """The cached entry for an image hash, or {} if there is none."""
    try:
        return json.loads((_cache_dir() / f"{sha}.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save(sha: str, entry: dict) -> None:
    folder = _cache_dir()
    folder.mkdir(parents=True, exist_ok=True)
    tmp = folder / f"{sha}.tmp"
    tmp.write_text(json.dumps(entry, indent=1, sort_keys=True), encoding="utf-8")
    tmp.replace(folder / f"{sha}.json")


def _client(timeout: float) -> httpx.Client:
    return httpx.Client(timeout=timeout, follow_redirects=True, headers={"User-Agent": USER_AGENT})


def _call_vision(image_bytes: bytes) -> dict:
    """Make one Vision web-detection request. Raises RuntimeError with a plain reason on failure."""
    key = os.environ.get(KEY_ENV, "")
    body = {"requests": [{"image": {"content": base64.b64encode(image_bytes).decode("ascii")},
                          "features": [{"type": "WEB_DETECTION", "maxResults": MAX_RESULTS}]}]}
    try:
        with _client(VISION_TIMEOUT_S) as client:
            resp = client.post(VISION_URL, json=body, headers={"x-goog-api-key": key})
    except httpx.TimeoutException as exc:
        raise RuntimeError("the web search timed out") from exc
    except httpx.HTTPError as exc:
        raise RuntimeError("the web search could not reach Google") from exc
    _count_call()
    if resp.status_code == 429:
        raise RuntimeError("Google refused the web search because too many searches were made")
    if resp.status_code in (401, 403):
        raise RuntimeError("Google refused the web search because the server's key was not accepted")
    if resp.status_code != 200:
        raise RuntimeError("Google could not carry out the web search")
    answer = resp.json().get("responses", [{}])[0]
    if "error" in answer:
        raise RuntimeError("Google could not search this image")
    return answer


def matching_pages(vision: dict) -> list[dict]:
    """Pages showing a full or partial copy of the image, in Vision's order, with the match kind."""
    pages = []
    for page in vision.get("webDetection", {}).get("pagesWithMatchingImages", []):
        if page.get("fullMatchingImages"):
            kind = "full"
        elif page.get("partialMatchingImages"):
            kind = "partial"
        else:
            continue  # listed only for visually similar images: another picture, not this one
        if page.get("url"):
            pages.append({"url": page["url"], "title": _plain(page.get("pageTitle", "")), "kind": kind})
    return pages


def _plain(title: str) -> str:
    """Strip HTML tags and extra whitespace from a page title."""
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", title)).strip()


def _without_fallback_dates(html: str):
    """Parse the page and drop the meta tags htmldate falls back on (modified date, copyright year).

    Otherwise a page whose publication date htmldate could not parse (one was in Portuguese) got
    its last-modified date instead. This way only a declared publication date counts.
    """
    from htmldate.core import ITEMPROP_ATTRS_MODIFIED, NAME_MODIFIED, PROPERTY_MODIFIED
    from htmldate.utils import load_html

    tree = load_html(html)
    if tree is None:
        return None
    for meta in list(tree.iter("meta")):
        name, prop = (meta.get("name") or "").lower(), (meta.get("property") or "").lower()
        itemprop, equiv = (meta.get("itemprop") or "").lower(), (meta.get("http-equiv") or "").lower()
        if (name in NAME_MODIFIED or prop in PROPERTY_MODIFIED or itemprop in ITEMPROP_ATTRS_MODIFIED
                or itemprop == "copyrightyear" or equiv == "last-modified"):
            meta.drop_tree()
    return tree


def date_from_html(html: str) -> str | None:
    """The publication date the page declares in its metadata, found with htmldate, or None.

    htmldate's extensive text search is off because it read numbers in scripts as years (a
    `"404": 2000` retry setting on Facebook came back as 2000-01-01).
    """
    from htmldate import find_date

    tree = _without_fallback_dates(html)
    if tree is None:
        return None
    return find_date(tree, extensive_search=False, original_date=True, outputformat="%Y-%m-%d",
                     max_date=date.today().isoformat())


def _stamp(ts: str) -> str | None:
    """Turn a Wayback timestamp (YYYYMMDD...) into YYYY-MM-DD, or None if it is not one."""
    return f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}" if len(ts) >= 8 and ts[:8].isdigit() else None


def date_from_wayback(url: str) -> str | None:
    """Date of the Wayback Machine's earliest capture of the page, or None.

    Tries the availability API first (the capture closest to 1990, so the earliest), then the
    first successful capture from the CDX API.
    """
    try:
        with _client(PAGE_TIMEOUT_S) as client:
            resp = client.get(WAYBACK_AVAILABLE, params={"url": url, "timestamp": "19900101"})
        if resp.status_code == 200:
            closest = resp.json().get("archived_snapshots", {}).get("closest") or {}
            if closest.get("available") and str(closest.get("status")) == "200":
                found = _stamp(str(closest.get("timestamp", "")))
                if found:
                    return found
    except (httpx.HTTPError, ValueError):
        pass
    params = {"url": url, "output": "json", "limit": "1", "fl": "timestamp", "filter": "statuscode:200"}
    try:
        with _client(PAGE_TIMEOUT_S) as client:
            resp = client.get(WAYBACK_CDX, params=params)
        if resp.status_code == 200:
            rows = resp.json()
            if len(rows) > 1 and rows[1]:
                return _stamp(rows[1][0])
    except (httpx.HTTPError, ValueError):
        pass
    return None


def date_page(url: str) -> dict:
    """Date a page. Returns {'date': 'YYYY-MM-DD' or None, 'source': 'htmldate' | 'wayback' | None}."""
    try:
        with _client(PAGE_TIMEOUT_S) as client:
            resp = client.get(url)
        if resp.status_code == 200 and "html" in resp.headers.get("content-type", "html"):
            found = date_from_html(resp.text)
            if found:
                return {"date": found, "source": "htmldate"}
    except Exception:  # noqa: BLE001 - an unreadable page falls through to the archive
        pass
    found = date_from_wayback(url)
    return {"date": found, "source": "wayback" if found else None}


def web_lookup(image_bytes: bytes) -> WebLookupResult:
    """Search the web for this image, using the cache and making a live call only when needed."""
    sha = hashlib.sha256(image_bytes).hexdigest()
    entry = _load(sha)
    live_call = False
    if "vision" not in entry:
        if not live_available():
            return WebLookupResult(detail="The web search is not available: no Google Cloud Vision key is set on the server.")
        if calls_this_month() >= MONTHLY_LIMIT:
            return WebLookupResult(detail=f"The web search was not run: this month's limit of {MONTHLY_LIMIT} "
                                          "live lookups has been reached.")
        try:
            entry["vision"] = _call_vision(image_bytes)
        except RuntimeError as exc:
            return WebLookupResult(detail=f"The web search could not run: {exc}.", live_call=True)
        entry["requested_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        live_call = True
        _save(sha, entry)

    pages = matching_pages(entry["vision"])[:MAX_PAGES]
    dates = entry.setdefault("page_dates", {})
    todo = [p["url"] for p in pages if p["url"] not in dates]
    if todo:
        with ThreadPoolExecutor(max_workers=5) as pool:
            for url, found in zip(todo, pool.map(date_page, todo)):
                dates[url] = found
        entry["dated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        _save(sha, entry)

    matches = [
        WebMatch(url=p["url"], title=p["title"] or None, published_date=dates[p["url"]]["date"],
                 date_source=dates[p["url"]]["source"], match_kind=p["kind"], found_by="web",
                 context=f"Web page showing a {p['kind']} copy of the image (Google Cloud Vision).")
        for p in pages
    ]
    if not matches:
        return WebLookupResult(status=FlagStatus.CLEAR, live_call=live_call,
                               detail="The web search found no page showing this image.")
    return WebLookupResult(matches=matches, status=FlagStatus.FIRED, live_call=live_call,
                           pages_with_matches=len(matching_pages(entry["vision"])))
