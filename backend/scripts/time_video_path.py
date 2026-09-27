"""WP-3: time and peak memory of each stage of the video path, on one clip, in a fresh process.

The clip runs end to end exactly as an upload does (probe, keyframes, OCR, image history lookup,
Whisper, CLIP, rules). Each stage's peak is the highest resident memory sampled every 50 ms while
it ran; memory is not handed back to the operating system between stages, so a later stage's peak
includes what earlier ones left behind, and the rise over the previous peak is reported too.

Run from the repo root:  python backend/scripts/time_video_path.py <clip.mp4> ["caption"]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

ROOT = _BACKEND.parent


def main() -> None:
    import psutil

    clip = Path(sys.argv[1])
    caption = sys.argv[2] if len(sys.argv) > 2 else None
    proc = psutil.Process()
    baseline_mb = proc.memory_info().rss / 2**20

    from app.adapters.video_adapter import build_video_bundle
    from app.extractors.video import probe
    from app.fusion.scorecard import build_scorecard

    imported_mb = proc.memory_info().rss / 2**20
    stages: list[dict] = []
    started = time.perf_counter()
    s = time.perf_counter()
    info = probe(clip)
    probe_s = time.perf_counter() - s
    bundle = build_video_bundle(clip, caption, clip.name, stages=stages)
    s = time.perf_counter()
    card = build_scorecard(bundle)
    rules_s = time.perf_counter() - s
    total_s = time.perf_counter() - started
    prev = imported_mb
    for st in stages:
        st["rise_mb"] = round(max(0.0, st["peak_mb"] - prev), 1)
        prev = max(prev, st["peak_mb"])
    mem = proc.memory_info()
    out = {"clip": clip.name, "duration_s": round(info.duration_s, 2), "has_audio": info.has_audio,
           "baseline_mb": round(baseline_mb, 1), "after_imports_mb": round(imported_mb, 1),
           "probe_s": round(probe_s, 2), "stages": stages, "rules_and_scorecard_s": round(rules_s, 3),
           "total_s": round(total_s, 2),
           "process_peak_mb": round(getattr(mem, "peak_wset", mem.rss) / 2**20, 1),
           "keyframes": len(bundle.keyframes), "segments": len(bundle.transcript_segments),
           "flags": {f.type.value: f.status.value for f in card.flags}}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "wp3_timing_60s.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
