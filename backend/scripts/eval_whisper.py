"""WP-3 model comparison: openai-whisper tiny against base.

The rule was fixed before any transcription ran: the model with the lower word error rate on the
100 LibriSpeech utterances wins; if the two are within 2 percentage points, the faster one wins
(lower mean transcription time per dataset E clip).

Word error rate: reference and hypothesis both go through Whisper's EnglishTextNormalizer, then
word-level edit distance; the rate is total edits over total reference words. Decoding and
filtering are the app's own (app/extractors/speech.py). Dataset E audio is taken from each clip
the way the app takes it.

Run one model per process, then compare:
  python backend/scripts/eval_whisper.py run --model tiny
  python backend/scripts/eval_whisper.py run --model base
  python backend/scripts/eval_whisper.py compare
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

ROOT = _BACKEND.parent
LABELS = ROOT / "data" / "labels"
OUT = ROOT / "results"
MARGIN_PP = 2.0


def edits(ref: list[str], hyp: list[str]) -> int:
    """Word-level Levenshtein distance."""
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1]


def run(model_name: str, out_name: str | None = None) -> None:
    import psutil
    from whisper.normalizers import EnglishTextNormalizer

    from app.extractors.speech import load_model, run_model
    from app.extractors.video import extract_audio

    norm = EnglishTextNormalizer()
    started = time.perf_counter()
    model = load_model(model_name)
    load_s = time.perf_counter() - started

    def score(items):
        rows = []
        for item_id, audio, reference in items:
            s = time.perf_counter()
            segments = run_model(model, audio)
            seconds = time.perf_counter() - s
            hyp = " ".join(seg.text for seg in segments)
            ref_words, hyp_words = norm(reference).split(), norm(hyp).split()
            rows.append({"id": item_id, "seconds": round(seconds, 3), "ref_words": len(ref_words),
                         "edits": edits(ref_words, hyp_words), "hypothesis": hyp})
        return rows

    speech = list(csv.DictReader(open(LABELS / "E_speech.csv", encoding="utf-8")))
    libri = score([(r["utterance_id"], ROOT / "data" / "E_speech" / r["file"], r["transcript"]) for r in speech])
    clips = [r for r in csv.DictReader(open(LABELS / "E_video_clips.csv", encoding="utf-8")) if r["has_speech"] == "yes"]
    with tempfile.TemporaryDirectory() as tmp:
        items = []
        for r in clips:
            wav = Path(tmp) / f"{r['clip_id']}.wav"
            extract_audio(ROOT / "data" / "E_videos" / "clips" / r["file"], wav)
            items.append((r["clip_id"], wav, r["script"]))
        e_rows = score(items)
    mem = psutil.Process().memory_info()
    out = {"model": model_name, "load_s": round(load_s, 2),
           "peak_memory_mb": round(getattr(mem, "peak_wset", mem.rss) / 2**20, 1),
           "librispeech": libri, "dataset_e": e_rows}
    OUT.mkdir(exist_ok=True)
    (OUT / (out_name or f"wp3_whisper_{model_name}.json")).write_text(json.dumps(out, indent=1), encoding="utf-8")
    for name, rows in (("LibriSpeech", libri), ("dataset E", e_rows)):
        wer = 100 * sum(r["edits"] for r in rows) / sum(r["ref_words"] for r in rows)
        print(f"{model_name} {name}: WER {wer:.2f}% over {len(rows)} items, "
              f"mean {statistics.mean(r['seconds'] for r in rows):.2f} s each")


def compare() -> None:
    res = {}
    for m in ("tiny", "base"):
        d = json.loads((OUT / f"wp3_whisper_{m}.json").read_text(encoding="utf-8"))
        summary = {"load_s": d["load_s"], "peak_memory_mb": d["peak_memory_mb"]}
        for name in ("librispeech", "dataset_e"):
            rows = d[name]
            summary[name] = {
                "items": len(rows), "ref_words": sum(r["ref_words"] for r in rows),
                "edits": sum(r["edits"] for r in rows),
                "wer_pct": round(100 * sum(r["edits"] for r in rows) / sum(r["ref_words"] for r in rows), 2),
                "mean_s": round(statistics.mean(r["seconds"] for r in rows), 3),
                "median_s": round(statistics.median(r["seconds"] for r in rows), 3),
            }
        res[m] = summary
    gap = res["tiny"]["librispeech"]["wer_pct"] - res["base"]["librispeech"]["wer_pct"]
    if abs(gap) < MARGIN_PP:
        chosen = min(("tiny", "base"), key=lambda m: res[m]["dataset_e"]["mean_s"])
        why = (f"LibriSpeech word error rates differ by {abs(gap):.2f} points (< {MARGIN_PP}), so the faster "
               f"per dataset E clip wins")
    else:
        chosen = "base" if gap > 0 else "tiny"
        why = f"lower LibriSpeech word error rate by {abs(gap):.2f} points"
    res["decision"] = {"chosen": chosen, "why": why, "tiny_minus_base_pp": round(gap, 2)}
    (OUT / "wp3_whisper_comparison.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"{'':<6} {'LibriSpeech WER':>16} {'E WER':>8} {'E mean s/clip':>14} {'load s':>7} {'peak MB':>8}")
    for m in ("tiny", "base"):
        r = res[m]
        print(f"{m:<6} {r['librispeech']['wer_pct']:>15.2f}% {r['dataset_e']['wer_pct']:>7.2f}% "
              f"{r['dataset_e']['mean_s']:>14.2f} {r['load_s']:>7.2f} {r['peak_memory_mb']:>8.0f}")
    print(f"chosen: {chosen} ({why})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["run", "compare"])
    parser.add_argument("--model", choices=["tiny", "base"])
    parser.add_argument("--out", help="write the run to this file in results/ instead of wp3_whisper_<model>.json")
    args = parser.parse_args()
    if args.step == "run":
        run(args.model, args.out)
    else:
        compare()


if __name__ == "__main__":
    main()
