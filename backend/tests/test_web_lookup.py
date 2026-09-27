"""The live web lookup, with every network call replaced by a stand-in: no key is used and no
Vision call is spent. What is pinned: only full and partial matches count, dates and their sources
are recorded, results replay from the cache, the monthly limit holds, every failure is NOT_ASSESSED
with a reason, and the key never appears in a result, a reason or the cache."""

import json
from datetime import date
from io import BytesIO

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import main
from app.config import get_settings
from app.extractors import reverse_image, web_lookup
from app.fusion.rules import recycled_context_rule
from app.models import EvidenceBundle, FlagStatus, Meta, Modality

FAKE_KEY = "test-key-not-real-123"

VISION = {"responses": [{"webDetection": {
    "pagesWithMatchingImages": [
        {"url": "https://old.example.org/2015/story", "pageTitle": "<b>Old</b> story",
         "fullMatchingImages": [{"url": "https://old.example.org/a.jpg"}]},
        {"url": "https://blog.example.net/post", "pageTitle": "A blog",
         "partialMatchingImages": [{"url": "https://blog.example.net/b.jpg"}]},
        {"url": "https://similar.example.com/page", "pageTitle": "Looks alike",
         "visuallySimilarImages": [{"url": "https://similar.example.com/c.jpg"}]},
    ],
    "visuallySimilarImages": [{"url": "https://elsewhere.example.com/d.jpg"}],
}}]}
DATES = {"https://old.example.org/2015/story": {"date": "2015-06-01", "source": "htmldate"},
         "https://blog.example.net/post": {"date": "2019-02-03", "source": "wayback"}}


def _png(seed: int = 1) -> bytes:
    buf = BytesIO()
    Image.new("RGB", (64, 48), (seed * 40 % 255, 90, 160)).save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture
def live(monkeypatch, tmp_path):
    """A fake key, a temporary cache and usage file, and a counter of calls made to Google."""
    get_settings.cache_clear()
    monkeypatch.setenv("AEGIS_LIVE_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv(web_lookup.KEY_ENV, FAKE_KEY)
    monkeypatch.setattr(web_lookup, "_usage_file", lambda: tmp_path / "usage.json")
    monkeypatch.setattr(web_lookup, "date_page", lambda url: DATES.get(url, {"date": None, "source": None}))
    calls = []

    def serve(response):
        def handler(request: httpx.Request):
            calls.append(request)
            if isinstance(response, Exception):
                raise response
            return response

        monkeypatch.setattr(web_lookup, "_client", lambda timeout: httpx.Client(transport=httpx.MockTransport(handler)))

    serve(httpx.Response(200, json=VISION))
    yield {"calls": calls, "serve": serve, "tmp": tmp_path}
    get_settings.cache_clear()


def test_only_full_and_partial_matches_count_and_each_date_keeps_its_source(live):
    result = web_lookup.web_lookup(_png())
    assert result.status == FlagStatus.FIRED and result.live_call
    assert [m.url for m in result.matches] == ["https://old.example.org/2015/story", "https://blog.example.net/post"]
    assert [m.match_kind for m in result.matches] == ["full", "partial"]
    assert [(m.published_date, m.date_source) for m in result.matches] == [("2015-06-01", "htmldate"),
                                                                          ("2019-02-03", "wayback")]
    assert result.matches[0].title == "Old story"
    assert all(m.found_by == "web" for m in result.matches)
    request = live["calls"][0]
    assert request.headers["x-goog-api-key"] == FAKE_KEY
    assert FAKE_KEY not in str(request.url)
    assert json.loads(request.content)["requests"][0]["features"][0]["type"] == "WEB_DETECTION"


def test_a_repeat_lookup_replays_from_the_cache_without_a_call(live):
    first = web_lookup.web_lookup(_png())
    live["serve"](httpx.ConnectError("offline"))
    again = web_lookup.web_lookup(_png())
    assert not again.live_call
    assert [m.model_dump() for m in again.matches] == [m.model_dump() for m in first.matches]
    assert len(live["calls"]) == 1 + 0  # the second lookup never reached the network
    assert web_lookup.calls_this_month() == 1
    cached = list((live["tmp"] / "cache").glob("*.json"))
    assert len(cached) == 1 and FAKE_KEY not in cached[0].read_text(encoding="utf-8")


def test_no_key_is_not_assessed_and_spends_nothing(live, monkeypatch):
    monkeypatch.delenv(web_lookup.KEY_ENV)
    result = web_lookup.web_lookup(_png())
    assert result.status == FlagStatus.NOT_ASSESSED
    assert "no Google Cloud Vision key" in result.detail
    assert live["calls"] == []


def test_the_monthly_limit_refuses_further_live_calls(live):
    (live["tmp"] / "usage.json").write_text(json.dumps({date.today().strftime("%Y-%m"): 900}))
    result = web_lookup.web_lookup(_png())
    assert result.status == FlagStatus.NOT_ASSESSED
    assert "limit of 900" in result.detail
    assert live["calls"] == []


@pytest.mark.parametrize("failure, reason", [
    (httpx.ReadTimeout("slow"), "timed out"),
    (httpx.ConnectError("no route"), "could not reach Google"),
    (httpx.Response(429, json={"error": {"message": "Quota exceeded"}}), "HTTP 429"),
    (httpx.Response(403, json={"error": {"message": f"API key {FAKE_KEY} not valid"}}), "HTTP 403"),
])
def test_every_failure_is_not_assessed_with_a_reason_and_never_shows_the_key(live, failure, reason):
    live["serve"](failure)
    result = web_lookup.web_lookup(_png())
    assert result.status == FlagStatus.NOT_ASSESSED
    assert reason in result.detail
    assert FAKE_KEY not in result.detail


def test_a_failed_web_search_never_turns_into_clear(live, monkeypatch):
    live["serve"](httpx.ReadTimeout("slow"))
    monkeypatch.setattr(reverse_image, "find_local_matches",
                        lambda b: reverse_image.ReverseImageResult(status=FlagStatus.CLEAR, phash="0" * 16))
    result = reverse_image.find_web_matches(_png(), search_web=True)
    assert result.status == FlagStatus.NOT_ASSESSED
    assert "The web search could not run: the web search timed out." in result.detail


def test_without_search_web_the_default_mode_stays_local(live, monkeypatch):
    monkeypatch.setattr(reverse_image, "find_local_matches",
                        lambda b: reverse_image.ReverseImageResult(status=FlagStatus.CLEAR, phash="0" * 16))
    assert reverse_image.find_web_matches(_png()).status == FlagStatus.CLEAR
    assert live["calls"] == []


def test_the_earliest_dated_page_drives_the_unchanged_date_check(live):
    matches = web_lookup.web_lookup(_png()).matches
    bundle = EvidenceBundle(web_matches=matches, extractor_status={"reverse_image": FlagStatus.FIRED},
                            extractor_detail={"web_search": ""},
                            meta=Meta(modality=Modality.IMAGE, posted_date="2024-05-01"))
    flag = recycled_context_rule(bundle)
    assert flag.status == FlagStatus.FIRED
    assert "Found on a page dated 2015-06-01" in flag.evidence
    assert "date from the page's own metadata" in flag.evidence


def _archive(monkeypatch, pages, available=None, cdx=None):
    """Stand-ins for the pages, the Wayback availability API and the CDX API."""
    seen = []

    def handler(request: httpx.Request):
        seen.append(request.url.host)
        url = request.url.params.get("url")
        if request.url.host == "archive.org":
            if available is None:
                return httpx.Response(503)
            snap = available.get(url)
            closest = {"available": True, "status": "200", "timestamp": snap} if snap else None
            return httpx.Response(200, json={"archived_snapshots": {"closest": closest} if closest else {}})
        if request.url.host == "web.archive.org":
            if cdx is None:
                return httpx.Response(503)
            return httpx.Response(200, json=[["timestamp"], [cdx[url]]] if url in (cdx or {}) else [])
        return httpx.Response(200, text=pages[str(request.url)], headers={"content-type": "text/html"})

    monkeypatch.setattr(web_lookup, "_client", lambda timeout: httpx.Client(transport=httpx.MockTransport(handler)))
    return seen


def test_a_page_is_dated_from_its_metadata_then_from_the_earliest_archive_capture(monkeypatch):
    pages = {"https://a.example/x": '<html><head><meta property="article:published_time" '
                                    'content="2018-04-05T10:00:00Z"></head><body>x</body></html>',
             "https://b.example/y": "<html><body>no date here</body></html>"}
    _archive(monkeypatch, pages, available={"https://b.example/y": "20200102030405"})
    assert web_lookup.date_page("https://a.example/x") == {"date": "2018-04-05", "source": "htmldate"}
    assert web_lookup.date_page("https://b.example/y") == {"date": "2020-01-02", "source": "wayback"}


def test_the_cdx_api_is_the_second_try_when_the_availability_api_fails(monkeypatch):
    pages = {"https://b.example/y": "<html><body>no date here</body></html>"}
    seen = _archive(monkeypatch, pages, available=None, cdx={"https://b.example/y": "20190708090000"})
    assert web_lookup.date_page("https://b.example/y") == {"date": "2019-07-08", "source": "wayback"}
    assert seen == ["b.example", "archive.org", "web.archive.org"]


# A made-up page in the shape that produced the fault: no date in its metadata, and an inline script
# holding retry settings like Facebook's, whose "2000" htmldate's extensive text search read as a year.
FACEBOOK_LIKE_PAGE = (
    '<!DOCTYPE html><html><head><title>A post - Example social site</title></head><body>'
    '<div role="main"><p>Look at this picture of the harbour.</p></div>'
    '<script>window.config = {"network_retry_intervals_json": '
    '"{\\"0\\": 1000, \\"404\\": 2000, \\"502\\": 1000, \\"429\\": 2000}"};</script>'
    '</body></html>'
)


def test_numbers_in_page_scripts_are_never_read_as_a_year():
    from htmldate import find_date

    # The page does reproduce the fault when the extensive search is on...
    assert find_date(FACEBOOK_LIKE_PAGE, original_date=True, outputformat="%Y-%m-%d") == "2000-01-01"
    # ...and the app's dating, on metadata only, finds no date in it.
    assert web_lookup.date_from_html(FACEBOOK_LIKE_PAGE) is None


def test_a_script_only_page_falls_through_to_the_archive(monkeypatch):
    _archive(monkeypatch, {"https://social.example/post/1": FACEBOOK_LIKE_PAGE},
             available={"https://social.example/post/1": "20210315120000"})
    assert web_lookup.date_page("https://social.example/post/1") == {"date": "2021-03-15", "source": "wayback"}


# A made-up page in the shape of the one the spot-check found misdated: its publication date is in
# a meta tag htmldate cannot parse (Portuguese) and in its structured data, and its modification
# date is in a meta tag htmldate can parse, which it used to fall back on.
MODIFIED_FALLBACK_PAGE = (
    '<!DOCTYPE html><html><head><title>Visita guiada</title>'
    '<meta property="article:modified_time" content="2018-11-26T14:27:56+00:00">'
    '<meta itemprop="datePublished" content="5 de maio de 2018">'
    '<meta itemprop="dateModified" content="26 de novembro de 2018">'
    '<script type="application/ld+json">{"@context": "https://schema.org", "@type": "Article", '
    '"datePublished": "2018-05-06T00:25:12+00:00", "dateModified": "2018-11-26T14:27:56+00:00"}</script>'
    '</head><body><p>Uma visita guiada.</p></body></html>'
)


def test_a_page_is_dated_by_its_publication_not_its_last_modification():
    from htmldate import find_date

    # htmldate alone, asked for the original date, falls back to the modification date...
    assert find_date(MODIFIED_FALLBACK_PAGE, extensive_search=False, original_date=True,
                     outputformat="%Y-%m-%d") == "2018-11-26"
    # ...and the app reads the publication date the page declares.
    assert web_lookup.date_from_html(MODIFIED_FALLBACK_PAGE) == "2018-05-06"


def test_a_page_with_only_a_modification_date_goes_to_the_archive(monkeypatch):
    page = ('<html><head><meta property="article:modified_time" content="2024-02-03T10:00:00Z">'
            '<meta itemprop="copyrightYear" content="2023"></head><body>x</body></html>')
    assert web_lookup.date_from_html(page) is None
    _archive(monkeypatch, {"https://m.example/p": page}, available={"https://m.example/p": "20190102030405"})
    assert web_lookup.date_page("https://m.example/p") == {"date": "2019-01-02", "source": "wayback"}


def test_health_says_whether_live_lookup_is_possible(monkeypatch):
    client = TestClient(main.app)
    monkeypatch.delenv(web_lookup.KEY_ENV, raising=False)
    assert client.get("/health").json()["config"]["live_lookup_available"] is False
    monkeypatch.setenv(web_lookup.KEY_ENV, FAKE_KEY)
    body = client.get("/health").json()
    assert body["config"]["live_lookup_available"] is True
    assert FAKE_KEY not in json.dumps(body)


def test_the_api_passes_the_web_search_choice_to_the_lookup(monkeypatch):
    from app.adapters import image_adapter

    seen = []
    monkeypatch.setattr(image_adapter, "find_web_matches",
                        lambda b, search_web=False: seen.append(search_web)
                        or reverse_image.ReverseImageResult(status=FlagStatus.CLEAR))
    client = TestClient(main.app)
    for flag in ("true", "false"):
        client.post("/analyze", files={"image": ("x.png", _png(), "image/png")},
                    data={"caption": "", "search_web": flag})
    assert seen == [True, False]
