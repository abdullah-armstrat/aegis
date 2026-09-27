"""WP-3: does what is said match what is shown? The swap test and the full pipeline on dataset E.

The method was fixed before any score was computed (the project's decision log):

  pairs     Split dataset E's sources 40/60 (seeded; no source on both sides) and pair each true
            line ("matches") with a partner line drawn, seeded, from a different source on the
            same side. Written to data/labels/E_speech_picture_pairs.csv. No scoring.
  scores    For every true line: its frames (the frame at its midpoint plus any keyframe inside
            it, keyframes from the app's own sampler); CLIP ViT-B/32 image score of its own text
            (true pair) and of its partner's text (swapped pair) against each frame; a pair's
            score is the highest. Cached in data/E_videos/features/ (ignored).
  fit       On the fitting side: among thresholds flagging at most 10% of true lines, the one
            catching the most swapped pairs; ties to fewer true lines flagged, then the lower one.
  heldout   True lines flagged and swapped pairs caught with Wilson 95% intervals, and the AUC with
            a bootstrap interval resampling sources.
  pipeline  The whole app on all 34 clips, Whisper base transcripts; each ground-truth line matched
            to the Whisper segment overlapping it most. Per-line and per-clip tables, held-out-side
            clips first, then all clips. Drift: each line against the segment that transcribes it
            (the most similar text, difflib ratio at least 0.5), start and end differences.
            ``--tag`` keeps a re-run apart: its own cache folder (pipeline_<tag>) and results file.

Run from the repo root: python backend/scripts/eval_speech_picture.py <step>
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

ROOT = _BACKEND.parent
LABELS = ROOT / "data" / "labels"
CLIPS = ROOT / "data" / "E_videos" / "clips"
CACHE = ROOT / "data" / "E_videos" / "features"
PAIRS = LABELS / "E_speech_picture_pairs.csv"
OUT = ROOT / "results"
SEED = 20260927
FIT_SHARE = 0.40
FALSE_ALARM_CAP = 0.10
BOOT_REPS = 2000


def clips() -> dict[str, dict]:
    return {r["clip_id"]: r for r in csv.DictReader(open(LABELS / "E_video_clips.csv", encoding="utf-8"))}


def lines() -> list[dict]:
    return list(csv.DictReader(open(LABELS / "E_video_segments.csv", encoding="utf-8")))


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    ph = k / n
    centre = (ph + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (round(centre - half, 3), round(centre + half, 3))


# ------------------------------------------------------------------------------ pairs
def do_pairs() -> None:
    clip = clips()
    true = [r for r in lines() if r["label"] == "matches"]
    true.sort(key=lambda r: (r["clip_id"], int(r["line_index"])))
    source_of = {cid: c["source"] for cid, c in clip.items()}
    sources = sorted({source_of[r["clip_id"]] for r in true})
    rng = random.Random(SEED)
    order = list(sources)
    rng.shuffle(order)
    per_source = {s: sum(source_of[r["clip_id"]] == s for r in true) for s in sources}
    fitting, count = set(), 0
    for s in order:
        if count >= FIT_SHARE * len(true):
            break
        fitting.add(s)
        count += per_source[s]
    side = {s: "fitting" if s in fitting else "held-out" for s in sources}
    rows = []
    for r in true:
        s = source_of[r["clip_id"]]
        candidates = [c for c in true if side[source_of[c["clip_id"]]] == side[s] and source_of[c["clip_id"]] != s]
        partner = rng.choice(candidates)
        rows.append({"clip_id": r["clip_id"], "line_index": r["line_index"], "source": s, "side": side[s],
                     "partner_clip_id": partner["clip_id"], "partner_line_index": partner["line_index"],
                     "partner_source": source_of[partner["clip_id"]]})
    with open(PAIRS, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for sd in ("fitting", "held-out"):
        ss = sorted(s for s in sources if side[s] == sd)
        print(f"{sd:<9} {sum(r['side'] == sd for r in rows):>3} true lines from {len(ss)} sources: {', '.join(ss)}")
    print(f"wrote {PAIRS.relative_to(ROOT)}")


# ------------------------------------------------------------------------------ scores
def do_scores() -> None:
    """True and swapped scores for every pair, through the app's own sampler, linking and CLIP."""
    from io import BytesIO

    from PIL import Image

    from app.extractors import caption_match
    from app.extractors.caption_match import IMAGE_MODEL
    from app.extractors.video import grab_frame, probe, sample_keyframes, segment_frame_times

    clip = clips()
    by_key = {(r["clip_id"], r["line_index"]): r for r in lines()}
    pairs = list(csv.DictReader(open(PAIRS, encoding="utf-8")))
    keyframes = {}
    for cid in sorted({p["clip_id"] for p in pairs}):
        path = CLIPS / clip[cid]["file"]
        keyframes[cid], _ = sample_keyframes(path, probe(path).duration_s)
    text_vec = {}
    rows = []
    for p in pairs:
        line = by_key[(p["clip_id"], p["line_index"])]
        partner = by_key[(p["partner_clip_id"], p["partner_line_index"])]
        times = segment_frame_times(float(line["start_s"]), float(line["end_s"]), keyframes[p["clip_id"]])
        frame_vecs = []
        for t in times:
            with Image.open(BytesIO(grab_frame(CLIPS / clip[p["clip_id"]]["file"], t))) as img:
                frame_vecs.append(caption_match.clip_image_vector(img.convert("RGB"), IMAGE_MODEL))
        for text in (line["text"], partner["text"]):
            if text not in text_vec:
                text_vec[text] = caption_match.clip_text_vector(text, IMAGE_MODEL)
        rows.append({**p, "frames": times,
                     "true_score": max(float(v @ text_vec[line["text"]]) for v in frame_vecs),
                     "swapped_score": max(float(v @ text_vec[partner["text"]]) for v in frame_vecs)})
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / "speech_picture_scores.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"scored {len(rows)} true and {len(rows)} swapped pairs; "
          f"{sum(len(r['frames']) for r in rows)} frames in all")


def _grid(values) -> np.ndarray:
    u = np.unique(values)
    return np.concatenate([[u[0] - 1e-6], (u[:-1] + u[1:]) / 2, [u[-1] + 1e-6]])


def fit_threshold(rows) -> float:
    """At most 10% of true lines flagged; then most swapped caught; ties to fewer true, lower t."""
    true = np.array([r["true_score"] for r in rows])
    swapped = np.array([r["swapped_score"] for r in rows])
    cap = math.floor(FALSE_ALARM_CAP * len(true))
    best = None
    for t in _grid(np.concatenate([true, swapped])):
        fa = int((true < t).sum())
        if fa > cap:
            continue
        rank = (int((swapped < t).sum()), -fa, -t)
        if best is None or rank > best[0]:
            best = (rank, float(t))
    return best[1]


def auc(true, swapped) -> float:
    """Chance that a swapped pair scores lower than a true one (ties count half)."""
    t, s = np.asarray(true)[:, None], np.asarray(swapped)[None, :]
    return float(((s < t).sum() + 0.5 * (s == t).sum()) / (t.size * s.size))


def do_fit_and_heldout() -> None:
    rows = json.loads((CACHE / "speech_picture_scores.json").read_text(encoding="utf-8"))
    fit = [r for r in rows if r["side"] == "fitting"]
    held = [r for r in rows if r["side"] == "held-out"]
    t = fit_threshold(fit)

    def table(rs):
        n = len(rs)
        fa = sum(r["true_score"] < t for r in rs)
        caught = sum(r["swapped_score"] < t for r in rs)
        return {"pairs": n, "true_flagged": fa, "true_flagged_rate": round(fa / n, 3), "true_flagged_ci": wilson(fa, n),
                "swapped_caught": caught, "swapped_caught_rate": round(caught / n, 3),
                "swapped_caught_ci": wilson(caught, n),
                "auc": round(auc([r["true_score"] for r in rs], [r["swapped_score"] for r in rs]), 3)}

    res = {"threshold": t, "fitting": table(fit), "held_out": table(held)}
    rng = np.random.default_rng(SEED)
    sources = sorted({r["source"] for r in held})
    by_source = {s: [r for r in held if r["source"] == s] for s in sources}
    draws = []
    for _ in range(BOOT_REPS):
        pick = [r for s in rng.choice(sources, size=len(sources), replace=True) for r in by_source[s]]
        draws.append(auc([r["true_score"] for r in pick], [r["swapped_score"] for r in pick]))
    res["held_out"]["auc_ci"] = (round(float(np.percentile(draws, 2.5)), 3), round(float(np.percentile(draws, 97.5)), 3))
    OUT.mkdir(exist_ok=True)
    (OUT / "wp3_speech_picture_swap.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"threshold (fitted on {len(fit)} fitting pairs): {t!r}")
    for name in ("fitting", "held_out"):
        r = res[name]
        print(f"{name:<9} true flagged {r['true_flagged']}/{r['pairs']} = {r['true_flagged_rate']:.3f} {r['true_flagged_ci']}  "
              f"swapped caught {r['swapped_caught']}/{r['pairs']} = {r['swapped_caught_rate']:.3f} {r['swapped_caught_ci']}  "
              f"AUC {r['auc']:.3f}" + (f" {r['auc_ci']}" if "auc_ci" in r else ""))


# ------------------------------------------------------------------------------ whole pipeline
def _pipeline_cache(tag: str) -> Path:
    return CACHE / ("pipeline" + (f"_{tag}" if tag else ""))


def do_pipeline(tag: str = "") -> None:
    """The app on every clip (no caption), then the line-level and clip-level tables."""
    from app.adapters.video_adapter import build_video_bundle
    from app.fusion.rules import audio_visual_mismatch_rule

    cache = _pipeline_cache(tag)
    cache.mkdir(parents=True, exist_ok=True)
    for cid, c in sorted(clips().items()):
        out = cache / f"{cid}.json"
        if out.exists():
            continue
        bundle = build_video_bundle(CLIPS / c["file"], None, c["file"])
        flag = audio_visual_mismatch_rule(bundle)
        out.write_text(json.dumps({
            "clip_id": cid, "flag_status": flag.status.value, "flag_timestamps": flag.timestamps,
            "flag_evidence": flag.evidence, "speech_status": bundle.extractor_status.get("speech"),
            "detail": bundle.extractor_detail,
            "segments": [s.model_dump() for s in bundle.transcript_segments]}, indent=1), encoding="utf-8")
        print(f"  {cid}: {flag.status.value}, {len(bundle.transcript_segments)} segments")
    analyse_pipeline(tag)


def _plain(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def drift(runs: dict) -> dict:
    """Each script line against the Whisper segment that transcribes it: how far apart they start and end."""
    rows = []
    for line in lines():
        scored = [(SequenceMatcher(None, _plain(line["text"]), _plain(s["text"])).ratio(), s)
                  for s in runs[line["clip_id"]]["segments"]]
        ratio, seg = max(scored, key=lambda pair: pair[0]) if scored else (0.0, None)
        row = {"clip_id": line["clip_id"], "line_index": int(line["line_index"]), "text": line["text"]}
        if seg is None or ratio < 0.5:
            rows.append({**row, "segment": None})
            continue
        rows.append({**row, "segment": seg["text"], "ratio": round(ratio, 3),
                     "start_drift": round(abs(seg["start"] - float(line["start_s"])), 2),
                     "end_drift": round(abs(seg["end"] - float(line["end_s"])), 2)})
    found = [r for r in rows if r["segment"] is not None]
    starts = sorted(r["start_drift"] for r in found)
    return {"lines": len(rows), "no_transcribing_segment": len(rows) - len(found),
            "start_over_1s": sum(r["start_drift"] > 1 for r in found),
            "end_over_1s": sum(r["end_drift"] > 1 for r in found),
            "either_over_1s": sum(r["start_drift"] > 1 or r["end_drift"] > 1 for r in found),
            "start_over_2s": sum(r["start_drift"] > 2 for r in found),
            "median_start_drift": starts[len(starts) // 2] if len(starts) % 2 else
            round((starts[len(starts) // 2 - 1] + starts[len(starts) // 2]) / 2, 3),
            "max_start_drift": starts[-1],
            "over_1s": [r for r in found if r["start_drift"] > 1 or r["end_drift"] > 1],
            "untranscribed": [r for r in rows if r["segment"] is None]}


def transcripts(runs: dict) -> dict:
    """Whisper's word error rate and invented sentences on each clip with speech, from the same
    segments the check scored (the definitions of eval_whisper.py)."""
    from whisper.normalizers import EnglishTextNormalizer

    from eval_whisper import edits, inserted

    norm = EnglishTextNormalizer()
    rows = []
    for cid, c in sorted(clips().items()):
        if c["has_speech"] != "yes":
            continue
        texts = [seg["text"] for seg in runs[cid]["segments"]]
        ref = norm(c["script"]).split()
        per_segment = [norm(t).split() for t in texts]
        flags = inserted(ref, [w for words in per_segment for w in words])
        invented, at = [], 0
        for text, words in zip(texts, per_segment):
            if words and sum(flags[at:at + len(words)]) > len(words) / 2:
                invented.append(text)
            at += len(words)
        rows.append({"clip_id": cid, "edits": edits(ref, norm(" ".join(texts)).split()), "ref_words": len(ref),
                     "invented": invented})
    return {"clips": len(rows),
            "wer_pct": round(100 * sum(r["edits"] for r in rows) / sum(r["ref_words"] for r in rows), 2),
            "invented_sentences": sum(len(r["invented"]) for r in rows),
            "clips_with_invented": [r["clip_id"] for r in rows if r["invented"]], "rows": rows}


def analyse_pipeline(tag: str = "") -> None:
    from app.fusion.rules import SPEECH_PICTURE_THRESHOLD

    clip = clips()
    side_of_source = {p["source"]: p["side"] for p in csv.DictReader(open(PAIRS, encoding="utf-8"))}
    runs = {cid: json.loads((_pipeline_cache(tag) / f"{cid}.json").read_text(encoding="utf-8")) for cid in clip}

    def fired(seg):
        return seg["picture_similarity"] is not None and seg["picture_similarity"] < SPEECH_PICTURE_THRESHOLD

    matched = []
    for line in lines():
        start, end = float(line["start_s"]), float(line["end_s"])
        overlaps = [(min(end, s["end"]) - max(start, s["start"]), s) for s in runs[line["clip_id"]]["segments"]]
        overlaps = [o for o in overlaps if o[0] > 0]
        best = max(overlaps, key=lambda o: o[0])[1] if overlaps else None
        matched.append({**line, "side": side_of_source[clip[line["clip_id"]]["source"]],
                        "segment": best, "flagged": bool(best and fired(best))})

    def per_line(rows):
        out = {}
        for label in ("matches", "different_scene", "wrong_detail"):
            rs = [r for r in rows if r["label"] == label]
            k = sum(r["flagged"] for r in rs)
            out[label] = {"flagged": k, "lines": len(rs), "rate": round(k / len(rs), 3) if rs else None,
                          "ci": wilson(k, len(rs)), "no_segment": sum(r["segment"] is None for r in rs)}
        return out

    def per_clip(clip_ids):
        mism = [c for c in clip_ids if clip[c]["type"] == "narrated_mismatch"]
        match = [c for c in clip_ids if clip[c]["type"] == "narrated_match"]
        correct = [c for c in mism if any(r["flagged"] and r["label"] != "matches"
                                          for r in matched if r["clip_id"] == c)]
        false = [c for c in match if any(fired(s) for s in runs[c]["segments"])]
        return {"mismatched_clips": len(mism), "with_a_correct_flag": len(correct),
                "matching_clips": len(match), "with_a_false_flag": len(false),
                "correct_clips": correct, "false_flag_clips": false}

    held_clips = [c for c in clip if side_of_source[clip[c]["source"]] == "held-out"]
    res = {"threshold": SPEECH_PICTURE_THRESHOLD, "whisper_model": "base",
           "held_out_clips": {"clips": len(held_clips), "lines": per_line([r for r in matched if r["side"] == "held-out"]),
                              "clip_level": per_clip(held_clips)},
           "all_clips": {"clips": len(clip), "lines": per_line(matched), "clip_level": per_clip(list(clip))},
           "no_speech_clips": {c: runs[c]["flag_status"] + ": " + runs[c]["flag_evidence"]
                               for c in clip if clip[c]["has_speech"] == "no"},
           "drift": drift(runs), "transcripts": transcripts(runs)}
    name = f"wp5_speech_picture_pipeline_{tag}.json" if tag else "wp3_speech_picture_pipeline.json"
    (OUT / name).write_text(json.dumps(res, indent=2), encoding="utf-8")
    for name in ("held_out_clips", "all_clips"):
        r = res[name]
        print(f"== {name} ({r['clips']} clips)")
        for label, v in r["lines"].items():
            print(f"   {label:<16} flagged {v['flagged']:>2}/{v['lines']:<3} {v['rate']} {v['ci']}  (no Whisper segment: {v['no_segment']})")
        cl = r["clip_level"]
        print(f"   mismatched clips with a correct flag: {cl['with_a_correct_flag']}/{cl['mismatched_clips']}; "
              f"matching clips with a false flag: {cl['with_a_false_flag']}/{cl['matching_clips']}")
    for c, v in res["no_speech_clips"].items():
        print(f"   {c}: {v}")
    t = res["transcripts"]
    print(f"transcripts of {t['clips']} clips with speech: WER {t['wer_pct']}%, invented sentences "
          f"{t['invented_sentences']} in {t['clips_with_invented']}")
    d = res["drift"]
    print(f"drift over {d['lines']} lines: start > 1 s {d['start_over_1s']}, end > 1 s {d['end_over_1s']}, "
          f"either {d['either_over_1s']}, start > 2 s {d['start_over_2s']}; median start "
          f"{d['median_start_drift']} s, largest {d['max_start_drift']} s; untranscribed {d['no_transcribing_segment']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["pairs", "scores", "fit", "pipeline", "analyse"])
    parser.add_argument("--tag", default="", help="pipeline/analyse: keep a re-run in its own cache and file")
    args = parser.parse_args()
    if args.step in ("pipeline", "analyse"):
        (do_pipeline if args.step == "pipeline" else analyse_pipeline)(args.tag)
    else:
        {"pairs": do_pairs, "scores": do_scores, "fit": do_fit_and_heldout}[args.step]()


if __name__ == "__main__":
    main()
