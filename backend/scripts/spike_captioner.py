"""Time the BLIP captioner on this machine and write the results to a JSON file.

The limit was set before measuring, as for the LLM spike: a warm caption of 20 s or less per image
means BLIP stays local, otherwise it would move to a hosted service. Numbers are written to a file
and read back, not copied from the terminal.

Two simple drawn scenes give BLIP something to describe. It times the cold load and first caption,
then two warm captions.

Run:  python backend/scripts/spike_captioner.py
"""

import json
import sys
import time
from io import BytesIO
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from PIL import Image, ImageDraw  # noqa: E402

_MODEL = "Salesforce/blip-image-captioning-base"


def _scene_image(kind: str) -> Image.Image:
    """A simple drawn scene for BLIP to caption (used for timing, not accuracy)."""
    img = Image.new("RGB", (384, 256), "skyblue")
    d = ImageDraw.Draw(img)
    if kind == "beach":
        d.rectangle([0, 170, 384, 256], fill="khaki")        # sand
        d.ellipse([300, 20, 360, 80], fill="yellow")          # sun
    else:
        d.rectangle([0, 180, 384, 256], fill="green")         # grass
        d.rectangle([150, 90, 230, 180], fill="saddlebrown")  # a house-ish block
        d.polygon([(140, 90), (240, 90), (190, 40)], fill="red")  # roof
    return img


def main() -> None:
    result: dict = {"model": _MODEL, "threshold_warm_s": 20}
    try:
        import torch
        from transformers import BlipForConditionalGeneration, BlipProcessor

        torch.set_num_threads(max(1, (torch.get_num_threads() or 4)))

        t0 = time.time()
        processor = BlipProcessor.from_pretrained(_MODEL)
        model = BlipForConditionalGeneration.from_pretrained(_MODEL)
        model.eval()
        result["load_s"] = round(time.time() - t0, 1)

        def caption(img: Image.Image) -> tuple[str, float]:
            t = time.time()
            inputs = processor(img, return_tensors="pt")
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=30)
            text = processor.decode(out[0], skip_special_tokens=True)
            return text, round(time.time() - t, 1)

        # The first caption is the cold one (includes any lazy set-up); then two warm ones.
        c0_text, c0_s = caption(_scene_image("beach"))
        c1_text, c1_s = caption(_scene_image("house"))
        c2_text, c2_s = caption(_scene_image("beach"))

        result.update({
            "cold_first_caption_s": c0_s,
            "warm1_s": c1_s,
            "warm2_s": c2_s,
            "warm_avg_s": round((c1_s + c2_s) / 2, 1),
            "sample_captions": {"beach": c0_text, "house": c1_text},
            "ok": True,
        })
        result["verdict"] = (
            "local-viable" if result["warm_avg_s"] <= result["threshold_warm_s"] else "offload"
        )
    except Exception as exc:  # noqa: BLE001 - record the failure in the output
        result["ok"] = False
        result["error"] = f"{type(exc).__name__}: {exc}"

    out_path = _BACKEND_DIR / "scripts" / "_captioner_spike.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
