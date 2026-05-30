"""Live before/after demonstration of the BLIP captioner's effect on caption↔scene.

The labelled eval set scores caption↔scene at the RULE level using injected scene text on
blank images (a deterministic stand-in from when the captioner wasn't integrated). Running the
live captioner on those blank images would be meaningless, so the genuine before/after is shown
here on a real (synthetic) scene image:

  BEFORE: captioner off  -> caption↔scene is NOT_ASSESSED (no scene description).
  AFTER:  captioner on   -> BLIP describes the scene, and the rule fires/clears against it.

Two captions are tested against one beach image: a matching one (expect clear/consistent) and a
mismatching one (expect fired). Results are written to a JSON file to be read back; no number is
transcribed from the terminal. Run: python backend/scripts/demo_captioner_effect.py
"""

import json
import os
import sys
from io import BytesIO
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from PIL import Image, ImageDraw  # noqa: E402


def _beach_png() -> bytes:
    img = Image.new("RGB", (384, 256), "skyblue")
    d = ImageDraw.Draw(img)
    d.rectangle([0, 170, 384, 256], fill="khaki")  # sand
    d.ellipse([300, 20, 360, 80], fill="yellow")    # sun
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _caption_scene_status(image_bytes: bytes, caption: str, use_captioner: bool):
    os.environ["AEGIS_USE_CAPTIONER"] = "true" if use_captioner else "false"
    os.environ["AEGIS_USE_LLM"] = "false"
    from app.config import get_settings

    get_settings.cache_clear()
    from app.adapters.image_adapter import build_bundle
    from app.fusion.rules import caption_scene_mismatch_rule

    bundle = build_bundle(image_bytes, caption=caption, source_ref="beach_demo.png")
    flag = caption_scene_mismatch_rule(bundle)
    scene = [s.text for s in bundle.scene_descriptions]
    return {"status": flag.status.value, "scene_descriptions": scene, "evidence": flag.evidence}


def main() -> None:
    img = _beach_png()
    matching = "a sunny beach with the sun in a blue sky"
    mismatching = "a snowy mountain village at night in winter"

    result = {
        "before_captioner_off": {
            "matching_caption": _caption_scene_status(img, matching, use_captioner=False),
        },
        "after_captioner_on": {
            "matching_caption": _caption_scene_status(img, matching, use_captioner=True),
            "mismatching_caption": _caption_scene_status(img, mismatching, use_captioner=True),
        },
    }
    out = _BACKEND_DIR / "scripts" / "_captioner_effect.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
