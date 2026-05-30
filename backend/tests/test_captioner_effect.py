"""Before/after of the BLIP captioner on the caption↔scene check, encoded as assertions.

This pins the captioner's *effect* so it is execution-verified by the test's pass/fail rather
than transcribed from a screen. On one real (synthetic) beach image:

  BEFORE (captioner off): caption↔scene is NOT_ASSESSED — no scene description exists.
  AFTER  (captioner on):  BLIP describes the scene, so the rule becomes assessable, a matching
                          caption is CLEAR and a mismatching caption FIRES.

Marked ``slow`` because it loads BLIP. The labelled eval set scores caption↔scene at the rule
level with injected scenes (a deterministic stand-in); this test is the live evidence that the
real captioner moves the check from "could not assess" to a correct verdict.
"""

from io import BytesIO

import pytest

pytest.importorskip("transformers")
pytest.importorskip("torch")

from PIL import Image, ImageDraw  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.models import FlagStatus  # noqa: E402


def _beach_png() -> bytes:
    img = Image.new("RGB", (384, 256), "skyblue")
    d = ImageDraw.Draw(img)
    d.rectangle([0, 170, 384, 256], fill="khaki")  # sand
    d.ellipse([300, 20, 360, 80], fill="yellow")    # sun
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _status(image_bytes: bytes, caption: str, use_captioner: bool, monkeypatch) -> FlagStatus:
    get_settings.cache_clear()
    monkeypatch.setenv("AEGIS_USE_CAPTIONER", "true" if use_captioner else "false")
    monkeypatch.setenv("AEGIS_USE_LLM", "false")
    get_settings.cache_clear()
    from app.adapters.image_adapter import build_bundle
    from app.fusion.rules import caption_scene_mismatch_rule

    bundle = build_bundle(image_bytes, caption=caption, source_ref="beach_demo.png")
    return caption_scene_mismatch_rule(bundle).status


@pytest.mark.slow
def test_captioner_before_after_caption_scene(monkeypatch):
    img = _beach_png()
    matching = "a sunny beach with the sun in a blue sky"
    mismatching = "a snowy mountain village at night in winter"

    # BEFORE: no captioner -> the check cannot run.
    before = _status(img, matching, use_captioner=False, monkeypatch=monkeypatch)
    assert before == FlagStatus.NOT_ASSESSED

    # AFTER: captioner on -> the check runs. A matching caption is not a mismatch...
    after_match = _status(img, matching, use_captioner=True, monkeypatch=monkeypatch)
    assert after_match in {FlagStatus.CLEAR, FlagStatus.FIRED}  # it RAN (not NOT_ASSESSED)
    assert after_match == FlagStatus.CLEAR

    # ...and a clearly unrelated caption fires the mismatch.
    after_mismatch = _status(img, mismatching, use_captioner=True, monkeypatch=monkeypatch)
    assert after_mismatch == FlagStatus.FIRED

    get_settings.cache_clear()
