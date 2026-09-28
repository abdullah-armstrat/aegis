"""WP-2: does the caption match the picture? Compared on the VERITE sample (dataset C).

The mismatch flag should fire for "miscaptioned" and "out-of-context" pairs and stay clear for
"true" ones. Candidates:

  image score  cosine similarity between the image and the caption, from OpenAI CLIP: ViT-B/32 or RN50
  text score   similarity between the caption and what the app extracted from the image (the BLIP
               scene description plus OCR text): CLIP's text encoder or spaCy en_core_web_md vectors
  rules        image alone, text alone, both low (fire if both scores are under their thresholds),
               either low (fire if either is)

Steps, each cached so the expensive ones run once:

  split     Stratified by label, grouped by article, 40% calibration / 60% held-out, fixed seed.
            Writes data/labels/C_verite_split.csv (IDs only).
  features  --phase blip_ocr | clip:ViT-B/32 | clip:RN50 | spacy. One model per process, cached in
            data/verite/features/ (git-ignored). CLIP text is cut at 77 tokens and every cut is counted.
  llm       The local LLM reasoner's verdict on every held-out pair (Ollama must be running).
  calibrate Calibration pairs only. Each candidate's threshold flags at most 10% of truthful
            pairs and, within that, catches the most out-of-context pairs; the candidate catching
            the most is chosen, ties by time then memory. Held-out labels are not read.
  timing    Time per image of the check alone and of the whole pipeline, on this machine.
  fit       All 300 sampled pairs, used only for fitting: the picture-only threshold, flagging at
            most 10% of the truthful pairs.
  confirm   The single run on the fresh set (every VERITE pair whose article is not in the sample;
            features computed with --set fresh): the picture-only rule, the meaning rule and word
            overlap, scored through the app's own rule code, and the default chosen from them.
  evaluate  The single held-out run: for the chosen rule, word overlap, the LLM and always-fire,
            the share of truthful pairs flagged and recall on out-of-context and miscaptioned
            pairs, with Wilson 95% intervals; exact McNemar tests against word overlap and the
            LLM; AUC of every candidate score with an article-level bootstrap interval; the paired
            true-vs-miscaptioned comparison on the same image. Writes a JSON report.

The selection rule was fixed before the held-out run was made: picking by F1 was
degenerate here, because two thirds of the pairs are mismatches and firing on everything already
scores F1 0.80. It is kept below only to reproduce that finding.

Run from the repo root:  python backend/scripts/eval_caption_match.py <step> [...]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

ROOT = _BACKEND.parent
SAMPLE = ROOT / "data" / "labels" / "C_verite_sample.csv"
FRESH = ROOT / "data" / "labels" / "C_verite_fresh.csv"
SPLIT = ROOT / "data" / "labels" / "C_verite_split.csv"
VERITE = ROOT / "data" / "verite"
CACHE = VERITE / "features"
SEED = 20260928
MISMATCH = {"miscaptioned", "out-of-context"}
CLIP_ARCHES = {"ViT-B/32": "vitb32", "RN50": "rn50"}


# ------------------------------------------------------------------------------ data
def load_pairs(pair_set: str = "sample") -> list[dict]:
    """The 300 sampled pairs, or the fresh set (every pair whose article is not in the sample),
    joined with their caption and local image file."""
    with open(VERITE / "VERITE.csv", encoding="utf-8", newline="") as fh:
        captions = {int(r[""]): r["caption"] for r in csv.DictReader(fh)}
    files = {p.stem: p for p in (VERITE / "images").iterdir()}
    with open(SAMPLE if pair_set == "sample" else FRESH, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    split = {}
    if pair_set == "sample" and SPLIT.exists():
        with open(SPLIT, encoding="utf-8", newline="") as fh:
            split = {int(r["verite_row"]): r["split"] for r in csv.DictReader(fh)}
    return [{
        "row": int(r["verite_row"]), "article": int(r["article_id"]), "label": r["label"],
        "image": r["image"], "file": files[r["image"]], "caption": captions[int(r["verite_row"])],
        "y": int(r["label"] in MISMATCH), "split": split.get(int(r["verite_row"]), pair_set),
    } for r in rows]


def cached(name: str, pair_set: str = "sample") -> Path:
    """A feature cache file; the fresh set's files carry a prefix so the two sets never mix."""
    return CACHE / (name if pair_set == "sample" else f"{pair_set}_{name}")


def key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def scene_text(feat: dict) -> str:
    """What the app read from the image: the scene description plus any on-screen text."""
    from app.extractors.caption_match import read_from_image

    return read_from_image([feat["scene"]] if feat["scene"] else [], feat["ocr"])


# ------------------------------------------------------------------------------ split
def do_split() -> None:
    from sklearn.model_selection import StratifiedGroupKFold

    pairs = load_pairs()
    labels = [p["label"] for p in pairs]
    groups = [p["article"] for p in pairs]
    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    fold_of = {}
    for k, (_, test_idx) in enumerate(folds.split(np.zeros(len(pairs)), labels, groups)):
        for i in test_idx:
            fold_of[i] = k
    with open(SPLIT, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["verite_row", "article_id", "label", "split"])
        for i, p in sorted(enumerate(pairs), key=lambda ip: ip[1]["row"]):
            w.writerow([p["row"], p["article"], p["label"], "calibration" if fold_of[i] < 2 else "held-out"])
    pairs = load_pairs()
    for s in ("calibration", "held-out"):
        sub = [p for p in pairs if p["split"] == s]
        by = {l: sum(p["label"] == l for p in sub) for l in ("true", "miscaptioned", "out-of-context")}
        print(f"{s:<12} {len(sub):>3} pairs  {by}  articles {len({p['article'] for p in sub})}")
    cal = {p["article"] for p in pairs if p["split"] == "calibration"}
    held = {p["article"] for p in pairs if p["split"] == "held-out"}
    print(f"articles on both sides: {len(cal & held)}   wrote {SPLIT.relative_to(ROOT)}")


# ------------------------------------------------------------------------------ features
def phase_blip_ocr(pairs, pair_set: str = "sample") -> None:
    from app.extractors.captioner import describe_scene
    from app.extractors.ocr import extract_on_screen_text

    out, images = {}, {p["image"]: p["file"] for p in pairs}
    for n, (image, path) in enumerate(sorted(images.items()), 1):
        data = path.read_bytes()
        s = time.perf_counter(); cap = describe_scene(data); t_blip = time.perf_counter() - s
        s = time.perf_counter(); ocr = extract_on_screen_text(data); t_ocr = time.perf_counter() - s
        out[image] = {"scene": " ".join(d.text for d in cap.scene_descriptions),
                      "scene_status": cap.status.value, "ocr": ocr.lines, "ocr_status": ocr.status.value,
                      "t_blip": t_blip, "t_ocr": t_ocr}
        if n % 25 == 0:
            print(f"  {n}/{len(images)} images")
    cached("blip_ocr.json", pair_set).write_text(json.dumps(out, indent=1), encoding="utf-8")


def phase_clip(pairs, arch: str, pair_set: str = "sample") -> None:
    """CLIP vectors for every image, caption and scene text, through the app's own functions."""
    from PIL import Image

    from app.extractors.caption_match import (
        clip_context_length,
        clip_image_vector,
        clip_text_vector,
        clip_token_count,
    )

    feats = json.loads(cached("blip_ocr.json", pair_set).read_text(encoding="utf-8"))
    limit = clip_context_length(arch)

    def encode_text(texts: list[str]) -> tuple[np.ndarray, list[float]]:
        vecs, times = [], []
        for t in texts:
            s = time.perf_counter()
            vecs.append(clip_text_vector(t, arch) if t else np.full(512 if arch == "ViT-B/32" else 1024, np.nan))
            times.append(time.perf_counter() - s)
        return np.array(vecs), times

    images = sorted({p["image"]: p["file"] for p in pairs}.items())
    img_vecs, img_times = [], []
    for _, path in images:
        with Image.open(path) as im:
            im.load()
            picture = im.convert("RGB")
        s = time.perf_counter()
        img_vecs.append(clip_image_vector(picture, arch))
        img_times.append(time.perf_counter() - s)
    captions = sorted({p["caption"] for p in pairs})
    cap_vecs, cap_times = encode_text(captions)
    scenes = [scene_text(feats[i]) for i, _ in images]
    scene_vecs, scene_times = encode_text(scenes)
    tag = CLIP_ARCHES[arch]
    np.savez(cached(f"clip_{tag}.npz", pair_set), img=np.array(img_vecs), cap=cap_vecs, scene=scene_vecs)
    meta = {"arch": arch, "images": [i for i, _ in images], "captions": [key(c) for c in captions],
            "t_image": img_times, "t_caption": cap_times, "t_scene": scene_times,
            "context_length": limit,
            "captions_cut": sum(clip_token_count(c) > limit for c in captions), "captions_total": len(captions),
            "pairs_with_cut_caption": sum(clip_token_count(p["caption"]) > limit for p in pairs),
            "scene_texts_cut": sum(clip_token_count(t) > limit for t in scenes if t),
            "scene_texts_empty": sum(not t for t in scenes), "scene_texts_total": len(scenes)}
    cached(f"clip_{tag}.json", pair_set).write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"  {arch}: captions cut {meta['captions_cut']}/{meta['captions_total']} "
          f"(pairs {meta['pairs_with_cut_caption']}/{len(pairs)}); scene texts cut "
          f"{meta['scene_texts_cut']}/{meta['scene_texts_total']}, empty {meta['scene_texts_empty']}")


def phase_spacy(pairs, pair_set: str = "sample") -> None:
    from app.extractors.caption_match import _spacy, spacy_similarity

    feats = json.loads(cached("blip_ocr.json", pair_set).read_text(encoding="utf-8"))
    _spacy()  # load outside the timed calls
    out = {}
    for p in pairs:
        seen = scene_text(feats[p["image"]])
        s = time.perf_counter()
        sim = spacy_similarity(p["caption"], seen) if seen else None
        out[str(p["row"])] = {"sim": float("nan") if sim is None else sim, "t": time.perf_counter() - s}
    cached("spacy.json", pair_set).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"  spaCy: {sum(math.isnan(v['sim']) for v in out.values())} pairs without a similarity")


def do_llm(pairs) -> None:
    """The LLM reasoner's verdict on every held-out pair (production code path)."""
    import os

    os.environ["AEGIS_USE_LLM"] = "true"
    from app.config import get_settings
    get_settings.cache_clear()
    from app.fusion import llm_reasoner

    feats = json.loads((CACHE / "blip_ocr.json").read_text(encoding="utf-8"))
    path = CACHE / "llm.json"
    out = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    todo = [p for p in pairs if p["split"] == "held-out" and str(p["row"]) not in out]
    for n, p in enumerate(todo, 1):
        f = feats[p["image"]]
        sink: list = []
        with llm_reasoner.record_calls(sink):
            v = llm_reasoner.reason_over_text(p["caption"], [f["scene"]], f["ocr"])
        rec = sink[0] if sink else None
        out[str(p["row"])] = {"available": v.available, "same_subject": v.same_subject,
                              "latency_s": rec.latency_s if rec else None,
                              "validity": rec.validity if rec else None}
        path.write_text(json.dumps(out, indent=1), encoding="utf-8")
        if n % 10 == 0:
            print(f"  {n}/{len(todo)} held-out pairs")
    done = [out[str(p["row"])] for p in pairs if p["split"] == "held-out"]
    print(f"  LLM verdicts: {sum(d['available'] and d['same_subject'] is not None for d in done)} usable "
          f"of {len(done)}")


# ------------------------------------------------------------------------------ evaluation
FALSE_ALARM_CAP = 0.10  # a threshold may flag at most this share of truthful calibration pairs
BOOT_REPS = 2000
SCORES = ["img:ViT-B/32", "img:RN50", "txt:CLIP ViT-B/32", "txt:CLIP RN50", "txt:spaCy"]
CANDIDATES = [("image alone", ["img:ViT-B/32"]), ("image alone", ["img:RN50"]),
              ("text alone", ["txt:CLIP ViT-B/32"]), ("text alone", ["txt:CLIP RN50"]),
              ("text alone", ["txt:spaCy"])]
for _img, _txt in (("img:ViT-B/32", "txt:CLIP ViT-B/32"), ("img:ViT-B/32", "txt:spaCy"),
                   ("img:RN50", "txt:CLIP RN50"), ("img:RN50", "txt:spaCy")):
    CANDIDATES += [("both low", [_img, _txt]), ("either low", [_img, _txt])]
PARTS_OF = {"img:ViT-B/32": {"ViT-B/32:image", "ViT-B/32:caption"}, "img:RN50": {"RN50:image", "RN50:caption"},
            "txt:CLIP ViT-B/32": {"ViT-B/32:caption", "ViT-B/32:scene"},
            "txt:CLIP RN50": {"RN50:caption", "RN50:scene"}, "txt:spaCy": {"spacy"}}
MODEL_OF = {"img:ViT-B/32": "clip-vit-b-32", "img:RN50": "clip-rn50", "txt:CLIP ViT-B/32": "clip-vit-b-32",
            "txt:CLIP RN50": "clip-rn50", "txt:spaCy": "spacy-md"}


def load_scores(pairs, pair_set: str = "sample", arches=tuple(CLIP_ARCHES)) -> tuple[dict, dict, dict]:
    """Every candidate score per pair (keyed by VERITE row), mean time per part, CLIP metadata."""
    spacy_sim = json.loads(cached("spacy.json", pair_set).read_text(encoding="utf-8"))
    scores, meta = {"txt:spaCy": {int(k): v["sim"] for k, v in spacy_sim.items()}}, {}
    part_time = {"spacy": float(np.mean([v["t"] for v in spacy_sim.values()]))}
    for arch in arches:
        tag = CLIP_ARCHES[arch]
        z = np.load(cached(f"clip_{tag}.npz", pair_set))
        m = json.loads(cached(f"clip_{tag}.json", pair_set).read_text(encoding="utf-8"))
        ii = {k: i for i, k in enumerate(m["images"])}
        ci = {k: i for i, k in enumerate(m["captions"])}
        scores[f"img:{arch}"] = {p["row"]: float(z["img"][ii[p["image"]]] @ z["cap"][ci[key(p["caption"])]])
                                 for p in pairs}
        scores[f"txt:CLIP {arch}"] = {p["row"]: float(z["scene"][ii[p["image"]]] @ z["cap"][ci[key(p["caption"])]])
                                      for p in pairs}
        for part in ("image", "caption", "scene"):
            part_time[f"{arch}:{part}"] = float(np.mean(m[f"t_{part}"]))
        meta[arch] = m
    return scores, part_time, meta


def predict(rule: str, parts: list[str], th: list[float], subset, scores) -> np.ndarray:
    a = np.array([scores[parts[0]][p["row"]] for p in subset])
    if rule in ("image alone", "text alone"):
        return a < th[0]
    b = np.array([scores[parts[1]][p["row"]] for p in subset])
    return (a < th[0]) & (b < th[1]) if rule == "both low" else (a < th[0]) | (b < th[1])


def _grid(values) -> np.ndarray:
    """Midpoints between consecutive distinct values, plus one below the lowest and one above the highest."""
    u = np.unique(values)
    return np.concatenate([[u[0] - 1e-6], (u[:-1] + u[1:]) / 2, [u[-1] + 1e-6]])


def fit_capped(rule: str, parts: list[str], cal, scores) -> list[float]:
    """The pre-registered threshold: among thresholds flagging at most 10% of truthful
    calibration pairs, the one catching the most out-of-context pairs; ties to fewer truthful
    flagged, then more miscaptioned caught, then the lower threshold."""
    label = np.array([p["label"] for p in cal])
    true, ooc, mis = label == "true", label == "out-of-context", label == "miscaptioned"
    cap = math.floor(FALSE_ALARM_CAP * true.sum())
    a = np.array([scores[parts[0]][p["row"]] for p in cal])
    b = np.array([scores[parts[1]][p["row"]] for p in cal]) if len(parts) == 2 else None
    best = None
    for ta in _grid(a):
        low_a = a < ta
        for tb in (_grid(b) if b is not None else [None]):
            if tb is None:
                fire = low_a
            else:
                fire = (low_a & (b < tb)) if rule == "both low" else (low_a | (b < tb))
            fa = int(fire[true].sum())
            if fa > cap:
                continue
            rank = (int(fire[ooc].sum()), -fa, int(fire[mis].sum()), -ta, -(tb if tb is not None else 0.0))
            if best is None or rank > best[0]:
                best = (rank, [float(ta)] + ([float(tb)] if tb is not None else []))
    return best[1]


def counts(pred: np.ndarray, subset) -> dict:
    """Flags on truthful pairs and catches on each kind of mismatch, with Wilson 95% intervals."""
    label = np.array([p["label"] for p in subset])
    out = {}
    for name, lab in (("fires_on_truthful", "true"), ("recall_out_of_context", "out-of-context"),
                      ("recall_miscaptioned", "miscaptioned")):
        k, n = int(pred[label == lab].sum()), int((label == lab).sum())
        out[name] = {"k": k, "n": n, "rate": k / n if n else float("nan"), "ci": wilson(k, n)}
    return out


def calibrate(pairs, scores, part_time) -> tuple[list[dict], dict, list[dict]]:
    """Fit and rank every candidate on the calibration pairs only. Held-out labels are not read."""
    probe = {r["model"]: r for r in
             json.loads((ROOT / "data" / "earlier_runs" / "wp0_model_probe.json").read_text(encoding="utf-8"))}
    cal = [p for p in pairs if p["split"] == "calibration"]
    yc = [p["y"] for p in cal]
    table, f1_table = [], []
    for rule, parts in CANDIDATES:
        models = sorted({MODEL_OF[x] for x in parts})
        cost = {"added_ms_per_image": 1000 * sum(part_time[q] for q in set().union(*(PARTS_OF[x] for x in parts))),
                "models": models, "memory_rise_mb": sum(probe[x]["peak_rise_mb"] for x in models)}
        th = fit_capped(rule, parts, cal, scores)
        pred = predict(rule, parts, th, cal, scores)
        table.append({"rule": rule, "scores": parts, "thresholds": th, **counts(pred, cal), **cost})
        # Superseded criterion, kept so the first run's degenerate result can be reproduced.
        a = [scores[parts[0]][p["row"]] for p in cal]
        if len(parts) == 1:
            th_f1 = [fit_1d(a, yc)]
        else:
            b = [scores[parts[1]][p["row"]] for p in cal]
            th_f1 = list(fit_2d(a, b, yc, "both" if rule == "both low" else "either"))
        pred_f1 = predict(rule, parts, th_f1, cal, scores)
        f1_table.append({"rule": rule, "scores": parts, "thresholds": th_f1, **prf(yc, pred_f1),
                         "fires_on_truthful": float(pred_f1[~np.asarray(yc, bool)].mean()), **cost})
    table.sort(key=lambda r: (-r["recall_out_of_context"]["k"], r["added_ms_per_image"], r["memory_rise_mb"]))
    f1_table.sort(key=lambda r: (-round(r["f1"], 3), r["added_ms_per_image"], r["memory_rise_mb"]))
    return table, table[0], f1_table


def print_calibration(table, f1_table, n_cal) -> None:
    print(f"calibration: {n_cal} pairs. Threshold flags at most {FALSE_ALARM_CAP:.0%} of truthful pairs; "
          "ranked by out-of-context pairs caught, then time, then memory")
    print(f"{'rule':<11} {'scores':<32} {'thresholds':<16} {'truthful':>9} {'OOC':>7} {'mis':>7} {'ms':>6} {'MB':>6}")
    for r in table:
        def c(k):
            return f"{r[k]['k']}/{r[k]['n']}"
        print(f"{r['rule']:<11} {' + '.join(r['scores']):<32} {', '.join(f'{t:.3f}' for t in r['thresholds']):<16} "
              f"{c('fires_on_truthful'):>9} {c('recall_out_of_context'):>7} {c('recall_miscaptioned'):>7} "
              f"{r['added_ms_per_image']:>6.1f} {r['memory_rise_mb']:>6.0f}")
    top = f1_table[0]
    print(f"(superseded F1 criterion would pick: {top['rule']} {' + '.join(top['scores'])}, F1 {top['f1']:.3f}, "
          f"fires on {top['fires_on_truthful']:.1%} of truthful)")


def do_calibrate(pairs) -> None:
    scores, part_time, _ = load_scores(pairs)
    table, _, f1_table = calibrate(pairs, scores, part_time)
    print_calibration(table, f1_table, sum(p["split"] == "calibration" for p in pairs))


def auc(y, s) -> float:
    """Chance that a mismatched pair scores lower than a truthful one."""
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, -np.asarray(s)))


def cluster_bootstrap(stat, subset, reps: int = BOOT_REPS, seed: int = SEED) -> tuple[float, float, int]:
    """95% percentile interval of stat(resampled pairs), resampling whole articles."""
    rng = np.random.default_rng(seed)
    groups = np.array([p["article"] for p in subset])
    ids = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in ids}
    vals, skipped = [], 0
    for _ in range(reps):
        idx = np.concatenate([members[g] for g in rng.choice(ids, size=len(ids), replace=True)])
        v = stat([subset[i] for i in idx])
        if v is None:
            skipped += 1
        else:
            vals.append(v)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), skipped


def do_evaluate(pairs, out_json: str) -> None:
    """The one held-out run, made after the selection rule was fixed."""
    from app.fusion.rules import caption_scene_mismatch_rule
    from app.models import EvidenceBundle, FlagStatus, Meta, Modality, SceneDescription

    feats = json.loads((CACHE / "blip_ocr.json").read_text(encoding="utf-8"))
    llm = json.loads((CACHE / "llm.json").read_text(encoding="utf-8"))
    held = [p for p in pairs if p["split"] == "held-out"]
    missing = [p["row"] for p in held if str(p["row"]) not in llm]
    if missing:
        raise SystemExit(f"LLM verdicts missing for {len(missing)} held-out pairs: run the llm step first")

    scores, part_time, clip_meta = load_scores(pairs)
    table, chosen, f1_table = calibrate(pairs, scores, part_time)
    cal = [p for p in pairs if p["split"] == "calibration"]
    print_calibration(table, f1_table, len(cal))
    yh = np.array([p["y"] for p in held], bool)
    label = np.array([p["label"] for p in held])
    claim = label != "miscaptioned"  # truthful + out-of-context pairs: what the flag claims to catch

    def overlap_status(p):
        b = EvidenceBundle(caption=p["caption"],
                           scene_descriptions=[SceneDescription(text=feats[p["image"]]["scene"])],
                           meta=Meta(modality=Modality.IMAGE))
        return caption_scene_mismatch_rule(b).status

    new_pred = predict(chosen["rule"], chosen["scores"], chosen["thresholds"], held, scores)
    ov_status = [overlap_status(p) for p in held]
    ov_pred = np.array([s == FlagStatus.FIRED for s in ov_status])  # not assessed never fires
    ov_ok = np.array([s != FlagStatus.NOT_ASSESSED for s in ov_status])
    llm_ok = np.array([bool(llm[str(p["row"])]["available"]) and llm[str(p["row"])]["same_subject"] is not None
                       for p in held])
    llm_pred = np.array([llm[str(p["row"])]["same_subject"] is False for p in held])

    res = {"selection_rule": "fixed before this run; see the module docstring", "calibration_n": len(cal), "heldout_n": len(held),
           "calibration_table": table, "chosen": chosen, "superseded_f1_calibration_table": f1_table}
    res["heldout"] = {
        "chosen_rule": counts(new_pred, held),
        "word_overlap": {**counts(ov_pred, held), "not_assessed": int((~ov_ok).sum())},
        "always_fire": counts(np.ones(len(held), bool), held),
        "llm_answered_only": {**counts(llm_pred[llm_ok], [p for p, ok in zip(held, llm_ok) if ok]),
                              "answered": int(llm_ok.sum()), "unanswered": int((~llm_ok).sum())},
    }
    res["mcnemar"] = {}
    for name, other, ok in (("vs word overlap", ov_pred, ov_ok), ("vs LLM", llm_pred, llm_ok)):
        for scope, mask in (("truthful + out-of-context", claim), ("all held-out", np.ones(len(held), bool))):
            m = mask & ok
            res["mcnemar"][f"{name}, {scope}"] = {**mcnemar_exact((new_pred == yh)[m], (other == yh)[m]),
                                                  "pairs": int(m.sum()), "excluded_unanswered": int((mask & ~ok).sum())}

    res["auc"] = {}
    for sname in SCORES:
        res["auc"][sname] = {}
        for contrast, lab in (("true vs out-of-context", "out-of-context"), ("true vs miscaptioned", "miscaptioned")):
            sub = [p for p in held if p["label"] in ("true", lab)]

            def stat(ps, sname=sname):
                ys = [p["y"] for p in ps]
                return auc(ys, [scores[sname][p["row"]] for p in ps]) if 0 < sum(ys) < len(ys) else None

            lo, hi, skipped = cluster_bootstrap(stat, sub)
            res["auc"][sname][contrast] = {"auc": stat(sub), "ci": (lo, hi), "n": len(sub), "draws_skipped": skipped}

    by_art = {}
    for p in held:
        by_art.setdefault(p["article"], {})[p["label"]] = p
    duo = [v for v in by_art.values() if "true" in v and "miscaptioned" in v]
    assert all(v["true"]["image"] == v["miscaptioned"]["image"] for v in duo)
    res["paired_same_image"] = {"articles": len(duo)}
    for sname in SCORES:
        higher = sum(scores[sname][v["true"]["row"]] > scores[sname][v["miscaptioned"]["row"]] for v in duo)
        res["paired_same_image"][sname] = {"true_scores_higher": higher, "share": higher / len(duo)}

    n, o = res["heldout"]["chosen_rule"], res["heldout"]["word_overlap"]
    res["default_decision"] = {
        "new_fires_on_truthful": n["fires_on_truthful"]["k"], "overlap_fires_on_truthful": o["fires_on_truthful"]["k"],
        "new_catches_ooc": n["recall_out_of_context"]["k"], "overlap_catches_ooc": o["recall_out_of_context"]["k"],
    }
    res["default_decision"]["new_rule_becomes_default"] = (
        n["fires_on_truthful"]["k"] <= o["fires_on_truthful"]["k"]
        and n["recall_out_of_context"]["k"] > o["recall_out_of_context"]["k"])

    f1_rule = f1_table[0]
    f1_pred = predict(f1_rule["rule"], f1_rule["scores"], f1_rule["thresholds"], held, scores)
    res["superseded_f1_rule_heldout"] = {
        task: prf(yh[label != drop], f1_pred[label != drop])
        for task, drop in (("true vs out-of-context", "miscaptioned"), ("true vs miscaptioned", "out-of-context"))}
    res["truncation"] = {a: {k: clip_meta[a][k] for k in ("context_length", "captions_cut", "captions_total",
                                                          "pairs_with_cut_caption", "scene_texts_cut",
                                                          "scene_texts_total")}
                         for a in CLIP_ARCHES}

    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")

    def fmt(c):
        return f"{c['k']:>2}/{c['n']:<2} {c['rate']:.3f} ({c['ci'][0]:.3f}-{c['ci'][1]:.3f})"

    print(f"\nHELD-OUT ({len(held)} pairs). chosen: {chosen['rule']} {' + '.join(chosen['scores'])} "
          f"< {', '.join(f'{t:.3f}' for t in chosen['thresholds'])}")
    print(f"{'':<18} {'fires on truthful':<26} {'recall out-of-context':<26} {'recall miscaptioned':<26}")
    for name, c in res["heldout"].items():
        print(f"{name:<18} {fmt(c['fires_on_truthful']):<26} {fmt(c['recall_out_of_context']):<26} "
              f"{fmt(c['recall_miscaptioned']):<26}")
    print(f"word overlap not assessed: {o['not_assessed']}; "
          f"LLM unanswered: {res['heldout']['llm_answered_only']['unanswered']}")
    for k, v in res["mcnemar"].items():
        print(f"McNemar {k:<40} new only {v['only_new_correct']:>2}, other only {v['only_other_correct']:>2}, "
              f"p = {v['p_value']:.4f} ({v['pairs']} pairs, {v['excluded_unanswered']} excluded)")
    print("AUC (95% article bootstrap)")
    for sname, d in res["auc"].items():
        print(f"  {sname:<20} " + "   ".join(f"{c}: {v['auc']:.3f} ({v['ci'][0]:.3f}-{v['ci'][1]:.3f})"
                                           for c, v in d.items()))
    print(f"paired, same image ({len(duo)} articles): " + ", ".join(
        f"{s} {v['true_scores_higher']}/{len(duo)}" for s, v in res["paired_same_image"].items() if s != "articles"))
    print("default decision:", res["default_decision"])
    print(f"wrote {out_json}")


def prf(y, pred):
    y, pred = np.asarray(y, bool), np.asarray(pred, bool)
    tp = int((y & pred).sum())
    fp = int((~y & pred).sum())
    fn = int((y & ~pred).sum())
    tn = int((~y & ~pred).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": p, "recall": r, "f1": f}


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    ph = k / n
    centre = (ph + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (centre - half, centre + half)


def mcnemar_exact(correct_a, correct_b) -> dict:
    from scipy.stats import binomtest

    a, b = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    only_a, only_b = int((a & ~b).sum()), int((~a & b).sum())
    n = only_a + only_b
    p = float(binomtest(min(only_a, only_b), n, 0.5).pvalue) if n else 1.0
    return {"only_new_correct": only_a, "only_other_correct": only_b, "discordant": n, "p_value": p}


def fit_1d(scores, y):
    """Superseded criterion: threshold t maximising F1 for 'fire when score < t'."""
    s = np.asarray(scores)
    best = max(((prf(y, s < t)["f1"], -abs(t), t) for t in _grid(s)))
    return float(best[2])


def fit_2d(a, b, y, mode: str, grid: int = 60):
    """Superseded criterion: F1-best threshold pair on a quantile grid."""
    a, b = np.asarray(a), np.asarray(b)
    qa = np.concatenate([np.unique(np.quantile(a, np.linspace(0, 1, grid))), [a.max() + 1e-6]])
    qb = np.concatenate([np.unique(np.quantile(b, np.linspace(0, 1, grid))), [b.max() + 1e-6]])
    best = None
    for ta in qa:
        for tb in qb:
            pred = (a < ta) & (b < tb) if mode == "both" else (a < ta) | (b < tb)
            f = prf(y, pred)["f1"]
            if best is None or f > best[0]:
                best = (f, float(ta), float(tb))
    return best[1], best[2]


def do_timing(pairs, n_images: int, out_json: str) -> None:
    """Time per image on this machine: the caption-vs-picture check alone, and the whole image
    pipeline (all extractors and rules, LLM off) under each caption_match_method. The first call
    of each is reported separately because it includes loading the models."""
    import os
    import statistics

    import psutil

    os.environ["AEGIS_USE_LLM"] = "false"
    os.environ["AEGIS_USE_CAPTIONER"] = "true"
    from app.adapters.image_adapter import build_bundle, measure_caption_fit
    from app.config import get_settings
    from app.fusion.scorecard import build_scorecard

    feats = json.loads((CACHE / "blip_ocr.json").read_text(encoding="utf-8"))
    chosen = sorted({p["image"]: p for p in pairs}.items())[:n_images]
    work = [(p["file"].read_bytes(), p["caption"], feats[img]) for img, p in chosen]

    def timed(fn) -> tuple[float, list[float]]:
        times = []
        for item in work:
            s = time.perf_counter()
            fn(*item)
            times.append(time.perf_counter() - s)
        return times[0], times[1:]

    def summary(first, rest) -> dict:
        ms = sorted(1000 * t for t in rest)
        return {"first_call_ms": round(1000 * first, 1), "images": len(ms), "mean_ms": round(statistics.mean(ms), 1),
                "median_ms": round(statistics.median(ms), 1), "p95_ms": round(ms[math.ceil(0.95 * len(ms)) - 1], 1)}

    out = {"n_images": n_images}
    out["check_alone"] = summary(*timed(lambda b, c, f: measure_caption_fit(b, c, [f["scene"]], f["ocr"])))

    def pipeline(method: str, b: bytes, c: str) -> float:
        os.environ["AEGIS_CAPTION_MATCH_METHOD"] = method
        get_settings.cache_clear()
        s = time.perf_counter()
        build_scorecard(build_bundle(b, c))
        return time.perf_counter() - s

    # Both pipelines are warmed on the first image, then take turns going first, so neither
    # gains from running after the other.
    first = {m: pipeline(m, *work[0][:2]) for m in ("overlap", "meaning")}
    rest = {"overlap": [], "meaning": []}
    for i, (b, c, _) in enumerate(work[1:]):
        for m in (("overlap", "meaning") if i % 2 == 0 else ("meaning", "overlap")):
            rest[m].append(pipeline(m, b, c))
    for m in rest:
        out[f"pipeline_{m}"] = summary(first[m], rest[m])
    mem = psutil.Process().memory_info()
    out["peak_memory_mb"] = round(getattr(mem, "peak_wset", mem.rss) / 2**20, 1)
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


# ------------------------------------------------------------------------------ fresh-set test
FRESH_TRUTHFUL_CAP = 0.20  # a rule qualifies as the default only if it flags at most this share


def fit_image_only(sample, scores) -> float:
    """Image-only threshold on all sampled pairs, by the same rule as the held-out comparison:
    at most 10% of truthful pairs flagged, then the most out-of-context pairs caught."""
    return fit_capped("image alone", ["img:ViT-B/32"], sample, scores)[0]


def do_fit() -> None:
    """Fit the image-only threshold on the 300 sampled pairs. Reads nothing from the fresh set."""
    sample = load_pairs("sample")
    scores, _, _ = load_scores(sample, "sample")
    th = fit_image_only(sample, scores)
    c = counts(predict("image alone", ["img:ViT-B/32"], [th], sample, scores), sample)
    print(f"image only, fitted on {len(sample)} sampled pairs: fire when CLIP ViT-B/32 similarity < {th!r}")
    print("  on those pairs: " + ", ".join(f"{k} {v['k']}/{v['n']}" for k, v in c.items()))


def do_confirm(out_json: str) -> None:
    """The single fresh-set run: each rule as the app applies it, on pairs no evaluation touched."""
    import os

    from app.config import get_settings
    from app.fusion import rules
    from app.models import CaptionMatch, EvidenceBundle, FlagStatus, Meta, Modality, SceneDescription

    sample = load_pairs("sample")
    sample_scores, _, _ = load_scores(sample, "sample")
    fitted = fit_image_only(sample, sample_scores)
    fresh = load_pairs("fresh")
    feats = json.loads(cached("blip_ocr.json", "fresh").read_text(encoding="utf-8"))
    scores, part_time, clip_meta = load_scores(fresh, "fresh", arches=("ViT-B/32",))
    # The app stores the threshold rounded; it must decide exactly as the fitted one does.
    stored = rules.CAPTION_IMAGE_ONLY_THRESHOLD
    if not (predict("image alone", ["img:ViT-B/32"], [fitted], sample, sample_scores)
            == predict("image alone", ["img:ViT-B/32"], [stored], sample, sample_scores)).all():
        raise SystemExit("the app's stored image-only threshold decides differently on the sample")

    def bundle(p, method: str) -> EvidenceBundle:
        f = feats[p["image"]]
        text = scores["txt:spaCy"][p["row"]]
        match = None
        if method == "image":
            match = CaptionMatch(image_similarity=scores["img:ViT-B/32"][p["row"]])
        elif method == "meaning":
            match = CaptionMatch(image_similarity=scores["img:ViT-B/32"][p["row"]],
                                 text_similarity=None if math.isnan(text) else text)
        return EvidenceBundle(caption=p["caption"],
                              scene_descriptions=[SceneDescription(text=f["scene"])] if f["scene"] else [],
                              on_screen_text=f["ocr"], caption_match=match, meta=Meta(modality=Modality.IMAGE))

    methods = {"image only": "image", "meaning": "meaning", "word overlap": "overlap"}
    fired, assessed = {}, {}
    for name, method in methods.items():
        os.environ["AEGIS_CAPTION_MATCH_METHOD"] = method
        get_settings.cache_clear()
        status = [rules.caption_scene_mismatch_rule(bundle(p, method)).status for p in fresh]
        fired[name] = np.array([s == FlagStatus.FIRED for s in status])
        assessed[name] = np.array([s != FlagStatus.NOT_ASSESSED for s in status])
    get_settings.cache_clear()
    exact = predict("image alone", ["img:ViT-B/32"], [fitted], fresh, scores)
    if not (exact == fired["image only"]).all():
        raise SystemExit("the app's stored image-only threshold decides differently from the fitted one")

    label = np.array([p["label"] for p in fresh])
    y = np.array([p["y"] for p in fresh], bool)
    per_label = {lab: int((label == lab).sum()) for lab in ("true", "miscaptioned", "out-of-context")}
    res = {"fresh_pairs": len(fresh), "per_label": per_label, "articles": len({p["article"] for p in fresh}),
           "image_only_threshold": fitted, "stored_threshold": rules.CAPTION_IMAGE_ONLY_THRESHOLD,
           "meaning_thresholds": [rules.CAPTION_IMAGE_SIMILARITY_THRESHOLD, rules.CAPTION_TEXT_SIMILARITY_THRESHOLD],
           "overlap_threshold": rules.CAPTION_SCENE_OVERLAP_THRESHOLD, "rules": {}}
    for name in methods:
        res["rules"][name] = {**counts(fired[name], fresh), "not_assessed": int((~assessed[name]).sum())}

    res["auc_image_score"] = {}
    for contrast, lab in (("true vs out-of-context", "out-of-context"), ("true vs miscaptioned", "miscaptioned")):
        sub = [p for p in fresh if p["label"] in ("true", lab)]

        def stat(ps):
            ys = [p["y"] for p in ps]
            return auc(ys, [scores["img:ViT-B/32"][p["row"]] for p in ps]) if 0 < sum(ys) < len(ys) else None

        lo, hi, skipped = cluster_bootstrap(stat, sub)
        res["auc_image_score"][contrast] = {"auc": stat(sub), "ci": (lo, hi), "n": len(sub), "draws_skipped": skipped}

    claim = label != "miscaptioned"
    ok = claim & assessed["image only"] & assessed["meaning"]
    res["mcnemar_image_only_vs_meaning"] = {
        **mcnemar_exact((fired["image only"] == y)[ok], (fired["meaning"] == y)[ok]),
        "pairs": int(ok.sum()), "excluded_not_assessed": int((claim & ~ok).sum()),
        "note": "only_new_correct = image only right and meaning wrong"}

    # Default, as pre-registered: qualify at <= 20% of truthful fresh pairs flagged; the largest
    # gap between the share of out-of-context caught and the share of truthful flagged wins; ties
    # to fewer truthful flagged, then less added time per image.
    added_ms = {"word overlap": 0.0,
                "image only": 1000 * (part_time["ViT-B/32:image"] + part_time["ViT-B/32:caption"]),
                "meaning": 1000 * (part_time["ViT-B/32:image"] + part_time["ViT-B/32:caption"] + part_time["spacy"])}
    table = []
    for name in methods:
        r = res["rules"][name]
        table.append({"rule": name, "truthful_flagged": r["fires_on_truthful"]["rate"],
                      "ooc_caught": r["recall_out_of_context"]["rate"],
                      "gap": r["recall_out_of_context"]["rate"] - r["fires_on_truthful"]["rate"],
                      "qualifies": r["fires_on_truthful"]["rate"] <= FRESH_TRUTHFUL_CAP,
                      "added_ms_per_image": added_ms[name]})
    ranked = sorted((t for t in table if t["qualifies"]),
                    key=lambda t: (-t["gap"], t["truthful_flagged"], t["added_ms_per_image"]))
    res["default_decision"] = {"table": table, "default": ranked[0]["rule"] if ranked else "off"}

    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")

    def fmt(c):
        return f"{c['k']:>2}/{c['n']:<3} {c['rate']:.3f} ({c['ci'][0]:.3f}-{c['ci'][1]:.3f})"

    print(f"FRESH SET: {len(fresh)} pairs {per_label}, {res['articles']} articles")
    print(f"image only fitted on the sample: < {fitted:.4f} (stored {rules.CAPTION_IMAGE_ONLY_THRESHOLD})")
    print(f"{'rule':<13} {'truthful flagged':<27} {'out-of-context caught':<27} {'miscaptioned caught':<27} n/a")
    for name, r in res["rules"].items():
        print(f"{name:<13} {fmt(r['fires_on_truthful']):<27} {fmt(r['recall_out_of_context']):<27} "
              f"{fmt(r['recall_miscaptioned']):<27} {r['not_assessed']}")
    for c, v in res["auc_image_score"].items():
        print(f"image score AUC, {c}: {v['auc']:.3f} ({v['ci'][0]:.3f}-{v['ci'][1]:.3f}), n={v['n']}")
    m = res["mcnemar_image_only_vs_meaning"]
    print(f"McNemar image only vs meaning, truthful + out-of-context: image only right {m['only_new_correct']}, "
          f"meaning right {m['only_other_correct']}, p = {m['p_value']:.4f} ({m['pairs']} pairs)")
    for t in table:
        print(f"  {t['rule']:<13} truthful {t['truthful_flagged']:.3f}  OOC {t['ooc_caught']:.3f}  "
              f"gap {t['gap']:+.3f}  qualifies {t['qualifies']}")
    print(f"default: {res['default_decision']['default']}")
    print(f"wrote {out_json}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["split", "features", "llm", "calibrate", "evaluate", "timing",
                                         "fit", "confirm"])
    parser.add_argument("--phase", default="")
    parser.add_argument("--set", dest="pair_set", choices=["sample", "fresh"], default="sample",
                        help="features: which pairs to compute them for")
    parser.add_argument("--images", type=int, default=31, help="timing: images to time (the first loads models)")
    parser.add_argument("--json", default=str(ROOT / "results" / "wp2_caption_match.json"))
    args = parser.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    if args.step == "split":
        return do_split()
    if args.step == "fit":
        return do_fit()
    if args.step == "confirm":
        return do_confirm(str(Path(args.json).with_name("wp2_fresh_confirmation.json")))
    pairs = load_pairs(args.pair_set)
    if args.step == "features":
        if args.phase == "blip_ocr":
            phase_blip_ocr(pairs, args.pair_set)
        elif args.phase.startswith("clip:"):
            phase_clip(pairs, args.phase[5:], args.pair_set)
        elif args.phase == "spacy":
            phase_spacy(pairs, args.pair_set)
        else:
            raise SystemExit("--phase must be blip_ocr, clip:ViT-B/32, clip:RN50 or spacy")
    elif args.pair_set != "sample":
        raise SystemExit("only the features step runs on the fresh set; the fresh test is the confirm step")
    elif args.step == "llm":
        do_llm(pairs)
    elif args.step == "calibrate":
        do_calibrate(pairs)
    elif args.step == "timing":
        do_timing(pairs, args.images, str(Path(args.json).with_name("wp2_timing.json")))
    else:
        do_evaluate(pairs, args.json)


if __name__ == "__main__":
    main()
