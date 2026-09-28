"""Download each model once, then measure it loading with the network blocked.

Covers every model the project uses or compares. All come without the Hugging Face Hub except
BLIP, which is read from the local cache only:

  clip-vit-b-32   OpenAI CLIP ViT-B/32 (image and text encoders in one checkpoint)
  clip-rn50       OpenAI CLIP RN50
  clip-vit-b-16   OpenAI CLIP ViT-B/16
  spacy-md        spaCy en_core_web_md word vectors
  whisper-tiny    openai-whisper tiny
  whisper-base    openai-whisper base
  blip            BLIP-base captioner, local cache only

Each model runs in its own child process so one model's memory does not inflate the next. The
network is blocked before any library is imported, so a successful load means local files only.
Reported per model: import time, load time, one small inference, and peak working set (psutil,
Windows) with its rise over the baseline.

Run:
  python backend/scripts/probe_models.py --download          # fetch weights once (needs network)
  python backend/scripts/probe_models.py --json out.json     # measure all, network blocked
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

MODELS = ["clip-vit-b-32", "clip-rn50", "clip-vit-b-16", "spacy-md",
          "whisper-tiny", "whisper-base", "blip"]
CLIP_NAMES = {"clip-vit-b-32": "ViT-B/32", "clip-rn50": "RN50", "clip-vit-b-16": "ViT-B/16"}
WHISPER_NAMES = {"whisper-tiny": "tiny", "whisper-base": "base"}
BLIP_ID = "Salesforce/blip-image-captioning-base"


def block_network() -> None:
    """Refuse every outgoing connection and name lookup in this process."""

    def refuse(*_args, **_kwargs):
        raise OSError("network blocked by probe_models.py")

    socket.socket.connect = refuse
    socket.socket.connect_ex = refuse
    socket.create_connection = refuse
    socket.getaddrinfo = refuse
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def network_is_blocked() -> bool:
    try:
        socket.create_connection(("github.com", 443), timeout=5)
    except OSError:
        return True
    return False


def measure(name: str) -> dict:
    """Load one model with the network blocked; return timings and peak memory."""
    import psutil

    proc = psutil.Process()
    baseline = proc.memory_info().rss
    block_network()
    out = {"model": name, "network_blocked": network_is_blocked()}

    t0 = time.perf_counter()
    if name in CLIP_NAMES:
        import clip
        import torch
        from PIL import Image
        t1 = time.perf_counter()
        model, preprocess = clip.load(CLIP_NAMES[name], device="cpu")
        model.eval()
        t2 = time.perf_counter()
        with torch.no_grad():
            image = preprocess(Image.new("RGB", (320, 240), "skyblue")).unsqueeze(0)
            text = clip.tokenize(["a blue sky", "a plate of food"])
            img_f = model.encode_image(image)
            t3 = time.perf_counter()
            txt_f = model.encode_text(text)
            t4 = time.perf_counter()
            sims = (img_f @ txt_f.T).softmax(dim=-1)[0].tolist()
        out.update(image_encode_s=round(t3 - t2, 3), text_encode_s=round(t4 - t3, 3),
                   sanity=f"softmax over ['a blue sky','a plate of food'] = {[round(s, 3) for s in sims]}")
        infer_end = t4
    elif name == "spacy-md":
        import spacy
        t1 = time.perf_counter()
        nlp = spacy.load("en_core_web_md")
        t2 = time.perf_counter()
        a, b, c = (nlp(s) for s in ("a car on the street", "an automobile parked by a road",
                                     "a plate of pasta"))
        infer_end = time.perf_counter()
        out.update(sanity=f"sim(car, automobile)={a.similarity(b):.3f}  sim(car, pasta)={a.similarity(c):.3f}")
    elif name in WHISPER_NAMES:
        import numpy as np
        import whisper
        t1 = time.perf_counter()
        model = whisper.load_model(WHISPER_NAMES[name], device="cpu")
        t2 = time.perf_counter()
        result = model.transcribe(np.zeros(16000 * 2, dtype=np.float32), fp16=False, language="en")
        infer_end = time.perf_counter()
        out.update(sanity=f"transcribed 2 s of silence: {result['text'].strip()!r}")
    elif name == "blip":
        import torch
        from PIL import Image
        from transformers import BlipForConditionalGeneration, BlipProcessor
        t1 = time.perf_counter()
        processor = BlipProcessor.from_pretrained(BLIP_ID, local_files_only=True)
        model = BlipForConditionalGeneration.from_pretrained(BLIP_ID, local_files_only=True)
        model.eval()
        t2 = time.perf_counter()
        with torch.no_grad():
            ids = model.generate(**processor(Image.new("RGB", (320, 240), "skyblue"),
                                             return_tensors="pt"), max_new_tokens=20)
        infer_end = time.perf_counter()
        out.update(sanity=f"caption of a plain sky-blue image: {processor.decode(ids[0], skip_special_tokens=True)!r}")
    else:
        raise SystemExit(f"unknown model {name}")

    mem = proc.memory_info()
    peak = getattr(mem, "peak_wset", mem.rss)
    out.update(
        import_s=round(t1 - t0, 2),
        load_s=round(t2 - t1, 2),
        first_inference_s=round(infer_end - t2, 2),
        baseline_mb=round(baseline / 2**20, 1),
        peak_mb=round(peak / 2**20, 1),
        peak_rise_mb=round((peak - baseline) / 2**20, 1),
    )
    return out


def download() -> None:
    """Fetch each model's weights once from its maker's own server."""
    import clip
    import whisper
    for name, arch in CLIP_NAMES.items():
        path = clip.load(arch, device="cpu", jit=False)[0]
        print(f"downloaded or found: {name}")
        del path
    for name, size in WHISPER_NAMES.items():
        whisper.load_model(size, device="cpu")
        print(f"downloaded or found: {name}")
    import spacy
    spacy.load("en_core_web_md")
    print("found: spacy-md (installed as a package)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--one", choices=MODELS, help="measure a single model (internal)")
    parser.add_argument("--json", default="")
    args = parser.parse_args()

    if args.download:
        download()
        return
    if args.one:
        print(json.dumps(measure(args.one)))
        return

    rows = []
    for name in MODELS:
        run = subprocess.run([sys.executable, __file__, "--one", name], capture_output=True, text=True)
        line = next((l for l in reversed(run.stdout.splitlines()) if l.startswith("{")), None)
        if run.returncode != 0 or line is None:
            err = (run.stderr.strip().splitlines() or ["no output"])[-1]
            rows.append({"model": name, "error": err})
            print(f"{name:<14} FAILED: {err}")
            continue
        row = json.loads(line)
        rows.append(row)
        print(f"{name:<14} net-blocked={row['network_blocked']}  import {row['import_s']:>5}s  "
              f"load {row['load_s']:>5}s  1st inference {row['first_inference_s']:>5}s  "
              f"peak {row['peak_mb']:>7} MB (+{row['peak_rise_mb']} MB)  | {row['sanity']}")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
