"""Checks that by default the app works offline and loads models only from local files.

The slow tests run the image and video flows in a new process where any outside connection or
name lookup is blocked and recorded, using the real defaults. No attempts are allowed.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from app.config import get_settings

BACKEND = Path(__file__).resolve().parents[1]

CHILD = r'''
import json, socket, subprocess, sys, tempfile
from io import BytesIO
from pathlib import Path

attempts = []
LOCAL = {"127.0.0.1", "::1", "localhost"}
_connect, _connect_ex, _getaddrinfo = socket.socket.connect, socket.socket.connect_ex, socket.getaddrinfo


def _host(address):
    return address[0] if isinstance(address, tuple) else str(address)


def connect(self, address):
    if _host(address) not in LOCAL:
        attempts.append(f"connect {_host(address)}")
        raise OSError("outbound connections are blocked in this test")
    return _connect(self, address)


def connect_ex(self, address):
    if _host(address) not in LOCAL:
        attempts.append(f"connect {_host(address)}")
        return 111
    return _connect_ex(self, address)


def getaddrinfo(host, *args, **kwargs):
    if host is not None and str(host) not in LOCAL:
        attempts.append(f"resolve {host}")
        raise socket.gaierror("name lookups are blocked in this test")
    return _getaddrinfo(host, *args, **kwargs)


socket.socket.connect, socket.socket.connect_ex, socket.getaddrinfo = connect, connect_ex, getaddrinfo

# The guard works: an outbound connection fails and is recorded.
try:
    socket.create_connection(("example.com", 80), timeout=2)
    guard = "not blocked"
except OSError:
    guard = "blocked"
guard_seen = list(attempts)
attempts.clear()

sys.path.insert(0, sys.argv[1])
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from app import main

picture = Image.new("RGB", (320, 240))
draw = ImageDraw.Draw(picture)
for x in range(320):
    draw.line([(x, 0), (x, 239)], fill=(x * 255 // 319, 120, 255 - x * 255 // 319))
draw.rectangle([40, 60, 140, 160], fill=(220, 30, 30))
draw.ellipse([180, 70, 280, 170], fill=(30, 60, 220))
buf = BytesIO()
picture.save(buf, "PNG")

out = {"guard": guard, "guard_seen": guard_seen}
with TestClient(main.app) as client:
    out["health"] = client.get("/health").json()["config"]
    card = client.post("/analyze", files={"image": ("shapes.png", buf.getvalue(), "image/png")},
                       data={"caption": "A red square and a blue circle on a coloured background."})
    out["image"] = {"code": card.status_code, "flags": card.json().get("flags")}
    with tempfile.TemporaryDirectory() as tmp:
        if sys.argv[2] != "video":
            raise SystemExit(print("RESULT " + json.dumps({**out, "attempts": attempts})))
        clip = Path(tmp) / "clip.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=320x240:r=10:d=6",
                        "-f", "lavfi", "-i", "sine=frequency=440:duration=6", "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(clip)], check=True)
        card = client.post("/analyze/video", files={"video": ("clip.mp4", clip.read_bytes(), "video/mp4")},
                           data={"caption": "A colourful moving test pattern."})
    out["video"] = {"code": card.status_code, "flags": card.json().get("flags")}
out["attempts"] = attempts
print("RESULT " + json.dumps(out))
'''


def _flag(flags, kind):
    return next(f for f in flags if f["type"] == kind)


def _run_offline(video: bool, **settings) -> dict:
    """Run the child script with default settings plus any AEGIS_ ones given; return its report."""
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("AEGIS_") and k not in {"HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
                                                     "HF_DATASETS_OFFLINE", "GOOGLE_VISION_API_KEY"}}
    env.update(settings)
    run = subprocess.run([sys.executable, "-c", CHILD, str(BACKEND), "video" if video else "image"],
                         cwd=BACKEND, env=env, capture_output=True, text=True, timeout=900)
    lines = [ln for ln in run.stdout.splitlines() if ln.startswith("RESULT ")]
    assert run.returncode == 0 and lines, run.stderr[-3000:]
    return json.loads(lines[-1][len("RESULT "):])


@pytest.mark.slow
def test_the_default_image_and_video_flows_open_no_network_connection():
    out = _run_offline(video=True)

    assert out["guard"] == "blocked" and out["guard_seen"] == ["resolve example.com"]
    assert out["health"]["allow_model_downloads"] is False
    assert out["health"]["reverse_image_mode"] == "local" and out["health"]["use_llm"] is False
    assert out["attempts"] == []

    # CLIP scored the picture and keyframes, and Whisper ran on the audio (just a tone, so it
    # may find no speech, but it ran).
    assert out["image"]["code"] == 200
    picture = _flag(out["image"]["flags"], "caption_content_mismatch")
    assert picture["status"] in ("fired", "clear") and "similarity" in picture["evidence"]
    assert out["video"]["code"] == 200
    video_picture = _flag(out["video"]["flags"], "caption_content_mismatch")
    assert video_picture["status"] in ("fired", "clear") and "similarity" in video_picture["evidence"]
    speech = _flag(out["video"]["flags"], "audio_visual_mismatch")
    assert "could not run" not in speech["evidence"]


@pytest.mark.slow
def test_blip_stays_offline_when_a_setting_makes_the_image_flow_load_it():
    """Word overlap needs BLIP, which must still load from the local cache only."""
    out = _run_offline(video=False, AEGIS_CAPTION_MATCH_METHOD="overlap")
    assert out["guard"] == "blocked"
    assert out["attempts"] == []
    picture = _flag(out["image"]["flags"], "caption_content_mismatch")
    assert picture["status"] in ("fired", "clear") and "overlap" in picture["evidence"]


def test_blip_is_read_from_local_files_unless_downloads_are_allowed(monkeypatch):
    transformers = pytest.importorskip("transformers")
    huggingface_hub = pytest.importorskip("huggingface_hub")
    from app.extractors import captioner

    seen = []

    def record(source, **kwargs):
        seen.append((source, kwargs.get("local_files_only")))
        return type("Stub", (), {"eval": lambda self: None})()

    def cached_folder(name, local_files_only=False):
        assert local_files_only is True
        return "/cache/blip-snapshot"

    monkeypatch.setattr(huggingface_hub, "snapshot_download", cached_folder)
    monkeypatch.setattr(transformers.BlipProcessor, "from_pretrained", record)
    monkeypatch.setattr(transformers.BlipForConditionalGeneration, "from_pretrained", record)
    for allow, expected in (("false", ("/cache/blip-snapshot", True)),
                            ("true", ("Salesforce/blip-image-captioning-base", False))):
        monkeypatch.setenv("AEGIS_ALLOW_MODEL_DOWNLOADS", allow)
        get_settings.cache_clear()
        captioner._get_model.cache_clear()
        captioner._get_model()
        assert seen[-2:] == [expected, expected]
    captioner._get_model.cache_clear()
    get_settings.cache_clear()


def test_a_missing_clip_or_whisper_file_is_downloaded_only_when_allowed(monkeypatch):
    clip = pytest.importorskip("clip")
    whisper = pytest.importorskip("whisper")
    from app.extractors import caption_match, speech

    def missing(name):
        raise FileNotFoundError(f"{name} is not on disk")

    loaded = []
    monkeypatch.setattr(caption_match, "clip_checkpoint", missing)
    monkeypatch.setattr(speech, "whisper_checkpoint", missing)
    monkeypatch.setattr(clip, "load", lambda source, **kw: loaded.append(("clip", source)) or (
        type("M", (), {"eval": lambda self: None})(), None))
    monkeypatch.setattr(whisper, "load_model", lambda source, **kw: loaded.append(("whisper", source)))

    monkeypatch.setenv("AEGIS_ALLOW_MODEL_DOWNLOADS", "false")
    get_settings.cache_clear()
    caption_match._clip.cache_clear()
    with pytest.raises(FileNotFoundError):
        caption_match._clip("ViT-B/32")
    with pytest.raises(FileNotFoundError):
        speech.load_model("base")
    assert loaded == []

    monkeypatch.setenv("AEGIS_ALLOW_MODEL_DOWNLOADS", "true")
    get_settings.cache_clear()
    caption_match._clip("ViT-B/32")
    speech.load_model("base")
    assert loaded == [("clip", "ViT-B/32"), ("whisper", "base")]
    caption_match._clip.cache_clear()
    get_settings.cache_clear()
