"""Fault injection: switch off, time out or break each extractor in turn and record every flag.

Each fault is one run through the app's API. The checks that depend on the broken extractor are
named in advance; each must come back not assessed with a reason (never clear), and the request
must still complete. A check that still has other inputs passes if its evidence names the missing
one. Nothing here changes the app; a failure is only reported.

Posts: a dataset A photo that is in the image history index, the same photo framed as a phone
screenshot (the hash misses it, the keypoint stage finds it), a made-up picture that is in no index
(for the web search), and dataset E clip E09.

Run: python backend/scripts/run_final_eval.py fault-injection (this module has no command line)
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from unittest import mock

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

ROOT = _BACKEND.parent
PHOTO = ROOT / "data" / "A_originals" / "A01_KSC-2013-2815.jpg"
CAPTION = "Storm clouds gather over the space centre this afternoon."
CLIP = ROOT / "data" / "E_videos" / "clips" / "E09.mp4"
CLIP_CAPTION = "A hurricane seen from the space station"
CHECKS = {"caption": "caption_content_mismatch", "recycled": "recycled_context",
          "framing": "emotional_framing", "speech": "audio_visual_mismatch"}

# Added after the first run: each affected check's reason must name the real cause in plain words
# and must not contain developer text (see DEVELOPER below).
CAUSES = {
    "I04": "switched off", "I05": "could not be processed", "I06": "index could not be read",
    "I07": "cropped or framed", "I08": "cropped or framed", "I09": "no Google Cloud Vision key",
    "I10": "timed out", "I11": "Google could not carry out", "I12": "switched off",
    "I13": "compares the caption with the picture", "I14": "compares the caption with the picture",
    "I15": "switched off", "I16": "describes the picture", "I17": "describes the picture",
    "I18": "compares the caption with the picture", "I19": "compares the caption with the picture",
    "I20": "could not be reached", "I21": "took too long", "I22": "could not be read",
    "V01": "frame", "V02": "frame", "V05": "index could not be read", "V06": "audio track",
    "V07": "audio track", "V08": "stopped before it finished", "V09": "could not be loaded",
    "V10": "compares words with the picture", "V11": "compares words with the picture", "V12": "switched off",
}
# For these faults, the words a partly assessed check's evidence must contain (replaces Fault.partial).
PARTIAL = {"V01": {"framing": "no frame of the video could be read"},
           "V02": {"framing": "no frame of the video could be read"},
           "V03": {"framing": "the text in the keyframes could not be read"},
           "V04": {"framing": "the text in the keyframes could not be read"},
           "V06": {"framing": "the audio track could not be read"},
           "V07": {"framing": "the audio track could not be read"},
           "V08": {"framing": "speech recogniser"}, "V09": {"framing": "speech recogniser"}}
# Signs of developer text in a reason: exception names, HTTP codes, settings, file paths, braces.
DEVELOPER = re.compile(r"Error|Exception|Errno|Traceback|HTTP \d|_mode|=|[A-Za-z]:[\\/]|[{}]")


def _raise(exc):
    """A stand-in function that raises exc whatever it is called with."""
    def _f(*args, **kwargs):
        raise exc
    return _f


@dataclass
class Fault:
    """One fault: the post, the extractor, how it fails, and the checks it should affect."""
    id: str
    post: str          # photo | framed | unindexed | video
    extractor: str
    kind: str          # off | time out | broken
    how: str
    affected: list[str]
    partial: dict[str, str] = field(default_factory=dict)  # check -> words its evidence must contain
    settings: dict = field(default_factory=dict)
    patches: list = field(default_factory=list)            # (object, attribute, replacement)
    search_web: bool = False
    key: bool = False
    web: object = None                                      # the stand-in web answer, if any
    clear_caches: tuple = ()


def _ffmpeg_timeout(marker: str):
    """A subprocess.run that times out on calls whose arguments contain marker."""
    real = subprocess.run

    def run(args, *a, **kw):
        if marker in args:
            raise subprocess.TimeoutExpired(args, kw.get("timeout", 60))
        return real(args, *a, **kw)
    return run


def faults() -> list[Fault]:
    """Every fault: image posts (I01-I22), then the video (V01-V12)."""
    import httpx
    import pytesseract

    from app.adapters import video_adapter
    from app.extractors import caption_match, captioner, phash, reverse_image, speech, video
    from app.extractors.ocr import OcrResult
    from app.fusion import llm_reasoner
    from app.models import FlagStatus

    missing = str(ROOT / "no-such-folder" / "missing")
    ocr_failed = OcrResult(status=FlagStatus.NOT_ASSESSED, detail="OCR engine error: broken for the test")
    tess_timeout = _raise(RuntimeError("Tesseract process timeout"))
    tess_broken = _raise(pytesseract.TesseractError(1, "broken for the test"))
    return [
        Fault("I01", "photo", "OCR", "off", "Tesseract's program is not where the setting points",
              [], settings={"tesseract_cmd": missing + ".exe"}),
        Fault("I02", "photo", "OCR", "time out", "Tesseract times out", [],
              patches=[(pytesseract, "image_to_string", tess_timeout)]),
        Fault("I03", "photo", "OCR", "broken", "Tesseract fails", [],
              patches=[(pytesseract, "image_to_string", tess_broken)]),
        Fault("I04", "photo", "image lookup", "off", "lookup mode 'off'", ["recycled"],
              settings={"reverse_image_mode": "off"}),
        Fault("I05", "photo", "image lookup (hash)", "broken", "the hash computation fails", ["recycled"],
              patches=[(phash, "phash_of_image", _raise(RuntimeError("hash failed for the test")))]),
        Fault("I06", "photo", "image lookup (index)", "broken", "the index file is missing", ["recycled"],
              settings={"image_index_path": missing + ".json"}),
        Fault("I07", "framed", "image lookup (keypoints)", "time out", "the keypoint stage times out", ["recycled"],
              patches=[(reverse_image, "_keypoint_matches", _raise(TimeoutError("keypoint stage timed out")))]),
        Fault("I08", "framed", "image lookup (keypoints)", "broken", "the keypoint stage fails", ["recycled"],
              patches=[(reverse_image, "_keypoint_matches", _raise(RuntimeError("keypoint stage failed")))]),
        Fault("I09", "unindexed", "web search", "off", "no Google Cloud Vision key on the server", ["recycled"],
              search_web=True, key=False),
        Fault("I10", "unindexed", "web search", "time out", "Google's answer times out", ["recycled"],
              search_web=True, key=True, web=httpx.ReadTimeout("slow")),
        Fault("I11", "unindexed", "web search", "broken", "Google answers HTTP 500", ["recycled"],
              search_web=True, key=True, web=httpx.Response(500, json={"error": {"message": "backend error"}})),
        Fault("I12", "photo", "CLIP", "off", "caption check switched off", ["caption"],
              settings={"caption_match_method": "off"}),
        Fault("I13", "photo", "CLIP", "time out", "CLIP times out on the picture", ["caption"],
              patches=[(caption_match, "clip_image_vector", _raise(TimeoutError("CLIP timed out")))]),
        Fault("I14", "photo", "CLIP", "broken", "CLIP's weights are missing", ["caption"],
              patches=[(caption_match, "clip_checkpoint", _raise(FileNotFoundError("CLIP weights missing")))],
              clear_caches=("clip",)),
        Fault("I15", "photo", "BLIP captioner", "off", "captioner switched off (word overlap)", ["caption"],
              settings={"caption_match_method": "overlap", "use_captioner": "false"}),
        Fault("I16", "photo", "BLIP captioner", "time out", "BLIP times out (word overlap)", ["caption"],
              settings={"caption_match_method": "overlap"},
              patches=[(captioner, "_get_model", _raise(TimeoutError("BLIP timed out")))]),
        Fault("I17", "photo", "BLIP captioner", "broken", "BLIP fails to load (word overlap)", ["caption"],
              settings={"caption_match_method": "overlap"},
              patches=[(captioner, "_get_model", _raise(RuntimeError("BLIP failed to load")))]),
        Fault("I18", "photo", "spaCy", "time out", "spaCy times out (meaning)", ["caption"],
              settings={"caption_match_method": "meaning", "use_captioner": "true"},
              patches=[(caption_match, "_spacy", _raise(TimeoutError("spaCy timed out")))]),
        Fault("I19", "photo", "spaCy", "broken", "spaCy's model is missing (meaning)", ["caption"],
              settings={"caption_match_method": "meaning", "use_captioner": "true"},
              patches=[(caption_match, "_spacy", _raise(OSError("spaCy model missing")))]),
        Fault("I20", "photo", "LLM", "off", "the LLM switched on but its server not running", ["llm"],
              settings={"use_llm": "true"},
              patches=[(llm_reasoner.httpx, "post", _raise(httpx.ConnectError("connection refused")))]),
        Fault("I21", "photo", "LLM", "time out", "the LLM times out", ["llm"], settings={"use_llm": "true"},
              patches=[(llm_reasoner.httpx, "post", _raise(httpx.ReadTimeout("slow")))]),
        Fault("I22", "photo", "LLM", "broken", "the LLM answers with something that is not JSON", ["llm"],
              settings={"use_llm": "true"},
              patches=[(llm_reasoner.httpx, "post", lambda *a, **k: httpx.Response(
                  200, json={"response": "not json at all"}, request=httpx.Request("POST", "http://llm")))]),
        Fault("V01", "video", "frame reading", "time out", "ffmpeg times out reading frames",
              ["caption", "recycled", "speech"], partial={"framing": "on-screen text"},
              patches=[(subprocess, "run", _ffmpeg_timeout("-frames:v"))]),
        Fault("V02", "video", "frame reading", "broken", "no frame can be read",
              ["caption", "recycled", "speech"], partial={"framing": "on-screen text"},
              patches=[(video_adapter, "grab_frame", _raise(video.VideoError("The frame could not be read.")))]),
        Fault("V03", "video", "OCR", "broken", "Tesseract fails on every keyframe", [],
              partial={"framing": "on-screen text"}, patches=[(pytesseract, "image_to_string", tess_broken)]),
        Fault("V04", "video", "OCR", "time out", "Tesseract times out on every keyframe", [],
              partial={"framing": "on-screen text"},
              patches=[(video_adapter, "extract_on_screen_text", lambda png: ocr_failed)]),
        Fault("V05", "video", "image lookup (index)", "broken", "the index file is missing", ["recycled"],
              settings={"image_index_path": missing + ".json"}),
        Fault("V06", "video", "audio track", "time out", "ffmpeg times out reading the audio", ["speech"],
              partial={"framing": "speech"}, patches=[(subprocess, "run", _ffmpeg_timeout("-vn"))]),
        Fault("V07", "video", "audio track", "broken", "the audio track cannot be read", ["speech"],
              partial={"framing": "speech"},
              patches=[(video_adapter, "extract_audio", _raise(video.VideoError("The audio track could not be read.")))]),
        Fault("V08", "video", "Whisper", "time out", "Whisper times out", ["speech"], partial={"framing": "speech"},
              patches=[(speech, "run_model", _raise(TimeoutError("Whisper timed out")))]),
        Fault("V09", "video", "Whisper", "broken", "Whisper's weights are missing", ["speech"],
              partial={"framing": "speech"},
              patches=[(speech, "load_model", _raise(FileNotFoundError("Whisper weights missing")))]),
        Fault("V10", "video", "CLIP", "time out", "CLIP times out", ["caption", "speech"],
              patches=[(caption_match, "clip_image_vector", _raise(TimeoutError("CLIP timed out")))]),
        Fault("V11", "video", "CLIP", "broken", "CLIP's weights are missing", ["caption", "speech"],
              patches=[(caption_match, "clip_checkpoint", _raise(FileNotFoundError("CLIP weights missing")))],
              clear_caches=("clip",)),
        Fault("V12", "video", "CLIP", "off", "caption check switched off", ["caption"],
              settings={"caption_match_method": "off"}),
    ]


def _posts() -> dict[str, tuple[bytes, str]]:
    """The three image posts as (bytes, file name); the made-up picture is seeded random blocks."""
    import numpy as np
    from PIL import Image

    from tests.eval.image_transforms import screenshot

    photo = PHOTO.read_bytes()
    with Image.open(BytesIO(photo)) as img:
        framed = screenshot(img.convert("RGB"))
    buf = BytesIO()
    framed.save(buf, "PNG")
    rng = np.random.default_rng(20260928)
    blocks = (rng.random((30, 40, 3)) * 255).astype(np.uint8)
    unindexed = BytesIO()
    Image.fromarray(blocks).resize((640, 480), Image.Resampling.NEAREST).save(unindexed, "PNG")
    return {"photo": (photo, "photo.jpg"), "framed": (buf.getvalue(), "framed.png"),
            "unindexed": (unindexed.getvalue(), "made-up.png")}


@contextmanager
def _applied(fault: Fault | None, tmp: Path):
    """Default settings plus the fault's settings and patches for one run, restored afterwards."""
    import httpx

    from app.config import get_settings
    from app.extractors import caption_match, captioner, reverse_image, web_lookup

    saved = dict(os.environ)
    for key in [k for k in os.environ if k.startswith("AEGIS_")]:
        del os.environ[key]
    os.environ.pop("GOOGLE_VISION_API_KEY", None)
    os.environ["AEGIS_LIVE_CACHE_DIR"] = str(tmp / "live_cache")
    for key, value in (fault.settings if fault else {}).items():
        os.environ[f"AEGIS_{key.upper()}"] = str(value)
    if fault and fault.key:
        os.environ["GOOGLE_VISION_API_KEY"] = "fault-injection-not-a-real-key"
    get_settings.cache_clear()
    reverse_image.load_index.cache_clear()
    captioner._get_model.cache_clear()
    caption_match._spacy.cache_clear()
    if fault and "clip" in fault.clear_caches:
        caption_match._clip.cache_clear()
    with ExitStack() as stack:
        stack.enter_context(mock.patch.object(web_lookup, "_usage_file", lambda: tmp / "usage.json"))
        if fault and fault.web is not None:
            answer = fault.web

            def handler(request):
                if isinstance(answer, Exception):
                    raise answer
                return answer
            stack.enter_context(mock.patch.object(
                web_lookup, "_client", lambda timeout: httpx.Client(transport=httpx.MockTransport(handler))))
        for obj, attr, replacement in (fault.patches if fault else []):
            stack.enter_context(mock.patch.object(obj, attr, replacement))
        try:
            yield
        finally:
            os.environ.clear()
            os.environ.update(saved)
            get_settings.cache_clear()
            reverse_image.load_index.cache_clear()
            if fault and "clip" in fault.clear_caches:
                caption_match._clip.cache_clear()


def _submit(client, post: str, posts: dict, search_web: bool) -> tuple[int, list[dict]]:
    """Send one post to the API; return the HTTP status and the flags."""
    if post == "video":
        resp = client.post("/analyze/video", files={"video": ("E09.mp4", CLIP.read_bytes(), "video/mp4")},
                           data={"caption": CLIP_CAPTION})
    else:
        data, name = posts[post]
        form = {"caption": CAPTION}
        if search_web:
            form["search_web"] = "true"
        resp = client.post("/analyze", files={"image": (name, data, "image/png")}, data=form)
    body = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
    return resp.status_code, body.get("flags", []) if isinstance(body, dict) else []


def _flag(flags: list[dict], check: str) -> dict | None:
    if check == "llm":
        return next((f for f in flags if f.get("source") == "llm"), None)
    return next((f for f in flags if f["type"] == CHECKS[check] and f.get("source", "rules") == "rules"), None)


def run() -> dict:
    """Run a baseline for each post, then every fault; return the records and a summary."""
    from fastapi.testclient import TestClient

    from app import main

    client = TestClient(main.app, raise_server_exceptions=False)
    posts = _posts()
    out = {"posts": {"photo": PHOTO.name, "framed": "the photo framed as a phone screenshot",
                     "unindexed": "a made-up picture in no index", "video": CLIP.name},
           "baselines": {}, "faults": []}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        baseline_settings = {"photo": {}, "framed": {}, "unindexed": {}, "video": {}}
        for post in baseline_settings:
            with _applied(None, tmp):
                code, flags = _submit(client, post, posts, search_web=False)
            out["baselines"][post] = {"code": code, "flags": {f"{f['type']}/{f.get('source', 'rules')}": f["status"]
                                                              for f in flags}}
            print(f"  baseline {post}: {code} {out['baselines'][post]['flags']}", flush=True)
        for fault in faults():
            with _applied(fault, tmp):
                code, flags = _submit(client, fault.post, posts, fault.search_web)
            problems = []
            if code != 200:
                problems.append(f"the request failed with HTTP {code}")
            cause = CAUSES.get(fault.id)
            for check in fault.affected:
                f = _flag(flags, check)
                if f is None:
                    problems.append(f"{check}: no flag")
                elif f["status"] == "clear":
                    problems.append(f"{check}: reported clear")
                elif f["status"] != "not_assessed":
                    problems.append(f"{check}: reported {f['status']}")
                elif not f["evidence"].strip():
                    problems.append(f"{check}: not assessed without a reason")
                else:
                    if cause and cause not in f["evidence"]:
                        problems.append(f"{check}: the reason does not name the cause ({cause})")
                    if DEVELOPER.search(f["evidence"]):
                        problems.append(f"{check}: developer text in the reason: {DEVELOPER.search(f['evidence']).group(0)}")
            for check, words in PARTIAL.get(fault.id, fault.partial).items():
                f = _flag(flags, check)
                if code == 200 and (f is None or (f["status"] != "not_assessed" and words not in f["evidence"])):
                    problems.append(f"{check}: does not name the missing input ({words})")
            record = {"id": fault.id, "post": fault.post, "extractor": fault.extractor, "kind": fault.kind,
                      "how": fault.how, "affected": fault.affected, "partial": fault.partial, "http": code,
                      "flags": [{"check": f["type"], "source": f.get("source", "rules"), "status": f["status"],
                                 "evidence": f["evidence"]} for f in flags],
                      "passed": not problems, "problems": problems}
            out["faults"].append(record)
            print(f"  {fault.id} {fault.extractor:<26} {fault.kind:<9} {'pass' if not problems else 'FAIL'} "
                  f"{'; '.join(problems)}", flush=True)
    out["summary"] = {"faults": len(out["faults"]), "passed": sum(r["passed"] for r in out["faults"]),
                      "failed": [r["id"] for r in out["faults"] if not r["passed"]]}
    return out
