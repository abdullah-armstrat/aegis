"""Time and peak memory of each stage of the video path, on one clip.

The clip goes through the same steps as an upload (probe, keyframes, OCR, image history lookup,
Whisper, CLIP, rules), possibly several times in one process: the first run is fresh, later ones
are warm (models from earlier runs stay loaded). A stage's peak is the highest resident memory,
sampled every 50 ms. Memory is not given back between stages, so a later peak includes what
earlier stages left; the rise over the previous peak is reported as well.

Run (from the repo root):
  python backend/scripts/time_video_path.py <clip.mp4> ["caption"] [--runs 2] [--out name.json]
"""

from __future__ import annotations

import argparse
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

    parser = argparse.ArgumentParser()
    parser.add_argument("clip")
    parser.add_argument("caption", nargs="?")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--out", default="wp3_timing_60s.json")
    args = parser.parse_args()
    clip = Path(args.clip)
    proc = psutil.Process()
    baseline_mb = proc.memory_info().rss / 2**20

    from app.adapters.video_adapter import build_video_bundle
    from app.extractors import reverse_image
    from app.extractors.video import probe
    from app.fusion.scorecard import build_scorecard

    index_path = reverse_image._index_path()
    stored = reverse_image.keypoints_path(index_path) if hasattr(reverse_image, "keypoints_path") else None
    imported_mb = proc.memory_info().rss / 2**20
    runs = []
    for n in range(args.runs):
        stages: list[dict] = []
        started = time.perf_counter()
        s = time.perf_counter()
        info = probe(clip)
        probe_s = time.perf_counter() - s
        bundle = build_video_bundle(clip, args.caption, clip.name, stages=stages)
        s = time.perf_counter()
        card = build_scorecard(bundle)
        rules_s = time.perf_counter() - s
        total_s = time.perf_counter() - started
        prev = imported_mb if n == 0 else runs[-1]["end_mb"]
        for st in stages:
            st["rise_mb"] = round(max(0.0, st["peak_mb"] - prev), 1)
            prev = max(prev, st["peak_mb"])
        runs.append({"run": "fresh process" if n == 0 else f"warm ({n + 1} in the same process)",
                     "probe_s": round(probe_s, 2), "stages": stages, "rules_and_scorecard_s": round(rules_s, 3),
                     "total_s": round(total_s, 2), "end_mb": round(proc.memory_info().rss / 2**20, 1),
                     "keyframes": len(bundle.keyframes), "segments": len(bundle.transcript_segments),
                     "flags": {f.type.value: f.status.value for f in card.flags}})
    mem = proc.memory_info()
    out = {"clip": clip.name, "duration_s": round(info.duration_s, 2), "has_audio": info.has_audio,
           "baseline_mb": round(baseline_mb, 1), "after_imports_mb": round(imported_mb, 1),
           "stored_keypoints_at_start": bool(stored and stored.is_file()) if stored else None,
           "runs": runs, "process_peak_mb": round(getattr(mem, "peak_wset", mem.rss) / 2**20, 1)}
    if args.runs == 1:  # same layout as the earlier single-run results
        out.update({k: v for k, v in runs[0].items() if k not in ("run", "end_mb")})
        del out["runs"]
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
