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

Decoding settings for base, chosen on a tuning set, never on the test sets (ADR-047):
  python backend/scripts/eval_whisper.py decoding --setting a|b|c --on tuning|tuning_long|original|e
  python backend/scripts/eval_whisper.py choose          (ADR-047, on the short tuning set)
  python backend/scripts/eval_whisper.py choose-long     (ADR-049, on the long tuning files)
An invented sentence is a Whisper segment more than half of whose words are insertions in the word
alignment against the reference.
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


def inserted(ref: list[str], hyp: list[str]) -> list[bool]:
    """For each hypothesis word, whether the word alignment (Levenshtein) makes it an insertion."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
    flags, i, j = [False] * m, n, m
    while i > 0 or j > 0:  # from the end: where a tie allows, the later words are the extra ones
        if j > 0 and d[i][j] == d[i][j - 1] + 1:
            flags[j - 1] = True
            j -= 1
        elif i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            i, j = i - 1, j - 1
        else:
            i -= 1
    return flags


SETTINGS = {
    "a": {"condition_on_previous_text": True, "hallucination_silence_threshold": None},
    "b": {"condition_on_previous_text": False, "hallucination_silence_threshold": None},
    "c": {"condition_on_previous_text": False, "hallucination_silence_threshold": 2.0},
}


def decoding(setting: str, on: str) -> None:
    """Whisper base with one decoding setting, on the tuning set, the original 100 or dataset E."""
    from whisper.normalizers import EnglishTextNormalizer

    from app.extractors.speech import load_model, run_model
    from app.extractors.video import extract_audio

    norm = EnglishTextNormalizer()
    model = load_model("base")
    with tempfile.TemporaryDirectory() as tmp:
        if on == "tuning_long":
            rows = csv.DictReader(open(LABELS / "E_speech_tuning_long.csv", encoding="utf-8"))
            items = [(r["file_id"], ROOT / "data" / "E_speech" / "tuning_long" / r["file"], r["transcript"])
                     for r in rows]
        elif on == "tuning":
            rows = csv.DictReader(open(LABELS / "E_speech_tuning.csv", encoding="utf-8"))
            items = [(r["utterance_id"], ROOT / "data" / "E_speech" / "tuning" / f"{r['utterance_id']}_silence.wav",
                      r["transcript"]) for r in rows]
        elif on == "original":
            rows = csv.DictReader(open(LABELS / "E_speech.csv", encoding="utf-8"))
            items = [(r["utterance_id"], ROOT / "data" / "E_speech" / r["file"], r["transcript"]) for r in rows]
        else:
            items = []
            for r in csv.DictReader(open(LABELS / "E_video_clips.csv", encoding="utf-8")):
                if r["has_speech"] == "yes":
                    wav = Path(tmp) / f"{r['clip_id']}.wav"
                    extract_audio(ROOT / "data" / "E_videos" / "clips" / r["file"], wav)
                    items.append((r["clip_id"], wav, r["script"]))
        out_rows = []
        for item_id, audio, reference in items:
            s = time.perf_counter()
            segments = run_model(model, audio, **SETTINGS[setting])
            seconds = time.perf_counter() - s
            ref_words = norm(reference).split()
            hyp_words = norm(" ".join(seg.text for seg in segments)).split()
            per_segment = [norm(seg.text).split() for seg in segments]
            flags = inserted(ref_words, [w for words in per_segment for w in words])
            invented, at = [], 0
            for seg, words in zip(segments, per_segment):
                if words and sum(flags[at:at + len(words)]) > len(words) / 2:
                    invented.append(seg.text)
                at += len(words)
            out_rows.append({"id": item_id, "seconds": round(seconds, 3), "ref_words": len(ref_words),
                             "edits": edits(ref_words, hyp_words), "hypothesis": " ".join(seg.text for seg in segments),
                             "segments": [seg.text for seg in segments], "invented": invented})
    wer = 100 * sum(r["edits"] for r in out_rows) / sum(r["ref_words"] for r in out_rows)
    out = {"model": "base", "setting": setting, "options": SETTINGS[setting], "on": on, "items": len(out_rows),
           "wer_pct": round(wer, 2), "invented_sentences": sum(len(r["invented"]) for r in out_rows),
           "items_with_invented": sum(bool(r["invented"]) for r in out_rows),
           "mean_s": round(statistics.mean(r["seconds"] for r in out_rows), 3), "rows": out_rows}
    OUT.mkdir(exist_ok=True)
    (OUT / f"wp5_whisper_{setting}_{on}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"setting {setting} on {on}: WER {wer:.2f}% over {len(out_rows)} items, invented sentences "
          f"{out['invented_sentences']} in {out['items_with_invented']} items, mean {out['mean_s']:.2f} s")


def choose() -> None:
    """The lowest tuning-set error rate; a tie at two decimals to fewer invented sentences, then (a)."""
    runs = {k: json.loads((OUT / f"wp5_whisper_{k}_tuning.json").read_text(encoding="utf-8")) for k in SETTINGS}
    order = sorted(SETTINGS, key=lambda k: (runs[k]["wer_pct"], runs[k]["invented_sentences"], k != "a"))
    res = {"rule": "lowest tuning-set WER; ties at two decimals to fewer invented sentences, then (a)",
           "tuning": {k: {f: runs[k][f] for f in ("wer_pct", "invented_sentences", "items_with_invented", "mean_s")}
                      for k in SETTINGS},
           "chosen": order[0], "options": SETTINGS[order[0]]}
    (OUT / "wp5_whisper_decoding_choice.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


def choose_long() -> None:
    """Seeded, on the long tuning files: the fewest invented sentences, then the lowest error rate;
    ties go to (a), so another setting is kept only if it helps (ADR-049)."""
    runs = {k: json.loads((OUT / f"wp5_whisper_{k}_tuning_long.json").read_text(encoding="utf-8")) for k in SETTINGS}
    order = sorted(SETTINGS, key=lambda k: (runs[k]["invented_sentences"], runs[k]["wer_pct"], k != "a"))
    res = {"rule": "fewest invented sentences, then lowest WER; ties to (a)",
           "tuning_long": {k: {f: runs[k][f] for f in ("wer_pct", "invented_sentences", "items_with_invented", "mean_s")}
                           for k in SETTINGS},
           "chosen": order[0], "options": SETTINGS[order[0]]}
    (OUT / "wp5_whisper_decoding_choice_long.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


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
    parser.add_argument("step", choices=["run", "compare", "decoding", "choose", "choose-long"])
    parser.add_argument("--model", choices=["tiny", "base"])
    parser.add_argument("--out", help="write the run to this file in results/ instead of wp3_whisper_<model>.json")
    parser.add_argument("--setting", choices=list(SETTINGS), help="decoding: which setting")
    parser.add_argument("--on", choices=["tuning", "tuning_long", "original", "e"], help="decoding: which items")
    args = parser.parse_args()
    if args.step == "decoding":
        decoding(args.setting, args.on)
    elif args.step == "choose":
        choose()
    elif args.step == "choose-long":
        choose_long()
    elif args.step == "run":
        run(args.model, args.out)
    else:
        compare()


if __name__ == "__main__":
    main()
