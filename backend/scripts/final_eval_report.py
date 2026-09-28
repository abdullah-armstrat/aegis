"""The report's numbers and charts, built only from the step files in results/ (nothing is re-run).

  write_numbers()  results/report_numbers.csv: every figure the report may cite, with its value,
                   count, interval, and the file and step it comes from
  draw_charts()    results/figures/*.png at 300 dpi, one per result the report shows

These are the numbers and charts steps of run_final_eval.py; either can be run on its own once the
step files exist.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results"
FIGURES = OUT / "figures"
LABELS = (("true", "truthful_flagged", "Truthful pairs flagged"),
          ("out-of-context", "out_of_context_caught", "Out-of-context caught"),
          ("miscaptioned", "miscaptioned_caught", "Miscaptioned caught"))
WP2_KEYS = (("fires_on_truthful", "Truthful pairs flagged"), ("recall_out_of_context", "Out-of-context caught"),
            ("recall_miscaptioned", "Miscaptioned caught"))


def _read(name: str) -> dict:
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def _r3(x):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), 3)


# ------------------------------------------------------------------------------ numbers
class _Numbers:
    """The rows of report_numbers.csv, each tagged with the results file and step it came from."""

    FIELDS = ["id", "description", "value", "count", "interval_low", "interval_high", "interval", "file", "step"]

    def __init__(self) -> None:
        self.rows: list[dict] = []
        self.file = self.step = ""

    def source(self, file: str, step: str) -> dict:
        self.file, self.step = file, step
        return _read(file)

    def add(self, id_: str, description: str, value, count: str = "", ci=None, interval: str = "") -> None:
        low, high = (ci or (None, None))[:2]
        self.rows.append({"id": id_, "description": description,
                          "value": _r3(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else value,
                          "count": count, "interval_low": _r3(low), "interval_high": _r3(high),
                          "interval": interval if ci else "", "file": f"results/{self.file}", "step": self.step})

    def rate(self, id_: str, description: str, r: dict) -> None:
        """A {k, n, rate, ci} count with its Wilson 95% interval."""
        self.add(id_, description, r["k"] / r["n"] if r["n"] else None, f"{r['k']}/{r['n']}", r.get("ci"), "Wilson 95%")


def _synthetic(n: _Numbers) -> None:
    d = n.source("wp6_synthetic.json", "synthetic")
    n.add("synthetic.pinned_counts_equal", "19 synthetic examples: every confusion count equals the pinned one",
          "yes" if d["all_equal"] else "no")
    o = d["word_overlap_as_pinned"]["overall"]
    for m in ("precision", "recall", "f1"):
        n.add(f"synthetic.overall_{m}", f"19 synthetic examples, rules only (word overlap): overall {m}", o[m])
    c = d["caption_cases_A"]
    n.add("caption_cases.as_expected", "Dataset A caption cases decided as expected", c["as_expected"] / c["cases"],
          f"{c['as_expected']}/{c['cases']}")
    n.add("caption_cases.matching_clear", "Dataset A caption cases: matching captions clear", c["matching_clear"] / 8,
          f"{c['matching_clear']}/8")
    n.add("caption_cases.mismatched_fired", "Dataset A caption cases: mismatched captions flagged",
          c["mismatched_fired"] / 8, f"{c['mismatched_fired']}/8")


def _matching(n: _Numbers) -> None:
    d = n.source("wp6_matching_A.json", "matching-a")
    for t, v in d["all_40"].items():
        key = t.replace(" ", "_")
        n.rate(f"matching_a.{key}.hash", f"Dataset A, {t}: own photo found, hash alone", v["hash"])
        n.rate(f"matching_a.{key}.shipped", f"Dataset A, {t}: own photo found, as shipped (hash then keypoints)",
               v["shipped"])
        n.add(f"matching_a.{key}.shipped_wrong", f"Dataset A, {t}: queries that also returned another photo, as shipped",
              v["shipped"]["wrong_matches"])
    for group, label in (("low_texture_8", "8 low-texture photos"), ("other_32", "other 32 photos")):
        for stage in ("hash", "shipped"):
            n.rate(f"matching_a.{group}.{stage}", f"Dataset A, {label}: own photo found, {stage}",
                   d[group]["all transformations"][stage])
    for stage, v in d["unrelated"]["false_matches"].items():
        n.rate(f"matching_a.verite_false_images.{stage}", f"Unrelated VERITE images matched to an A photo, {stage}",
               v["images"])
        n.rate(f"matching_a.verite_false_pairs.{stage}", f"Unrelated image pairs matched, {stage}", v["pairs"])


def _hard_pairs(n: _Numbers) -> None:
    d = n.source("wp6_hard_pairs.json", "hard-pairs")
    for variant in ("without alignment", "with alignment"):
        key = variant.replace(" ", "_")
        v = d[variant]
        n.rate(f"hard_pairs.{key}.wrong_queries", f"Hard pairs, {variant}: queries that returned another photo",
               v["queries_with_another_photo"])
        n.rate(f"hard_pairs.{key}.partner_pairs", f"Hard pairs, {variant}: pairs where a photo matched its partner",
               v["pairs_with_a_partner_match"])
        n.rate(f"hard_pairs.{key}.own_found", f"Hard pairs, {variant}: own photo found", v["own_photo_found"])


def _date(n: _Numbers) -> None:
    d = n.source("wp6_date_check.json", "date-check")
    for name, v in d["conditions"].items():
        n.rate(f"date_check.{name.replace(' ', '_')}", f"Date check, posting date {name}: status as expected "
               f"({v['expected']}, severity {v['expected_severity']})", v)
    n.add("date_check.mismatches", "Date check: photo-condition cases not as expected", len(d["mismatches"]))


def _heldout(n: _Numbers) -> None:
    d = n.source("wp6_verite_heldout.json", "verite-heldout")
    ch = d["chosen"]
    n.add("verite_heldout.chosen_rule", "VERITE calibration: rule chosen before the held-out run",
          f"{ch['rule']}: {' + '.join(ch['scores'])} < {', '.join(f'{t:.4f}' for t in ch['thresholds'])}")
    names = {"chosen_rule": "calibrated rule", "word_overlap": "word overlap", "always_fire": "always fire",
             "llm_answered_only": "LLM (answered pairs)"}
    for key, label in names.items():
        for metric, text in WP2_KEYS:
            n.rate(f"verite_heldout.{key}.{metric}", f"VERITE held-out, {label}: {text.lower()}", d["heldout"][key][metric])
    n.add("verite_heldout.word_overlap.not_assessed", "VERITE held-out, word overlap: pairs not assessed",
          d["heldout"]["word_overlap"]["not_assessed"])
    n.add("verite_heldout.llm.unanswered", "VERITE held-out: pairs the LLM did not answer",
          d["heldout"]["llm_answered_only"]["unanswered"])
    for key, v in d["mcnemar"].items():
        n.add(f"verite_heldout.mcnemar.{key.replace(' ', '_').replace(',', '')}",
              f"VERITE held-out, exact McNemar p, calibrated rule {key} ({v['pairs']} pairs; "
              f"{v['only_new_correct']} vs {v['only_other_correct']} discordant)", v["p_value"])
    for score, contrasts in d["auc"].items():
        for contrast, v in contrasts.items():
            n.add(f"verite_heldout.auc.{score}.{contrast.replace(' ', '_')}", f"VERITE held-out AUC, {score}, {contrast}",
                  v["auc"], str(v["n"]), v["ci"], "article bootstrap 95% (2000)")
    for score, v in d["paired_same_image"].items():
        if score != "articles":
            n.add(f"verite_heldout.paired.{score}", f"VERITE held-out, same image: the true caption scores higher, {score}",
                  v["share"], f"{v['true_scores_higher']}/{d['paired_same_image']['articles']}")


def _fresh(n: _Numbers) -> None:
    d = n.source("wp6_verite_fresh.json", "verite-fresh")
    n.add("verite_fresh.pairs", "Fresh VERITE pairs", d["fresh_pairs"])
    n.add("verite_fresh.threshold", "Picture-only threshold, fitted on the 300 sampled pairs", d["image_only_threshold"])
    for rule, v in d["rules"].items():
        for metric, text in WP2_KEYS:
            n.rate(f"verite_fresh.{rule.replace(' ', '_')}.{metric}", f"Fresh VERITE, {rule}: {text.lower()}", v[metric])
        n.add(f"verite_fresh.{rule.replace(' ', '_')}.not_assessed", f"Fresh VERITE, {rule}: pairs not assessed",
              v["not_assessed"])
    for contrast, v in d["auc_image_score"].items():
        n.add(f"verite_fresh.auc.{contrast.replace(' ', '_')}", f"Fresh VERITE AUC of the picture score, {contrast}",
              v["auc"], str(v["n"]), v["ci"], "article bootstrap 95% (2000)")
    m = d["mcnemar_image_only_vs_meaning"]
    n.add("verite_fresh.mcnemar_image_vs_meaning", f"Fresh VERITE, exact McNemar p, picture only vs meaning "
          f"({m['pairs']} pairs; {m['only_new_correct']} vs {m['only_other_correct']} discordant)", m["p_value"])
    n.add("verite_fresh.default", "Fresh VERITE: default chosen by the pre-registered rule", d["default_decision"]["default"])


def _limits(n: _Numbers) -> None:
    d = n.source("wp6_picture_limits.json", "picture-limits")
    for pair_set in ("sample", "fresh"):
        e = d[pair_set]
        n.add(f"picture_limits.{pair_set}.images_excluded", f"Picture limits, {pair_set} set: images excluded",
              len(e["images_excluded"]), f"{len(e['images_excluded'])}/{e['images']}")
        for lab, v in e["per_label"].items():
            n.add(f"picture_limits.{pair_set}.{lab}.pairs_excluded", f"Picture limits, {pair_set} set, {lab}: pairs "
                  f"excluded (nearly blank {v['pairs_nearly_blank']}, mostly text {v['pairs_mostly_text']})",
                  v["pairs_excluded"], f"{v['pairs_excluded']}/{v['pairs']}")
        for method, runs in e["rule"].items():
            for when, r in runs.items():
                for key, v in r.items():
                    n.rate(f"picture_limits.{pair_set}.{method}.{when}.{key}",
                           f"Picture limits, {pair_set} set, {method} rule {when} the limits: {key.replace('_', ' ')}", v)


def _llm(n: _Numbers) -> None:
    d = n.source("wp6_llm_verite.json", "llm-verite")
    c = d["calls"]
    n.rate("llm_verite.answered", "LLM on VERITE (held-out and fresh): calls answered", c["answered"])
    n.rate("llm_verite.clean_json", "LLM on VERITE: answers that were clean JSON", c["clean_json"])
    n.add("llm_verite.timeouts", "LLM on VERITE: calls that timed out (30 s)", c["timeouts"], f"{c['timeouts']}/{c['n']}")
    for k in ("median", "p95_nearest_rank", "max", "mean"):
        n.add(f"llm_verite.latency_{k}_s", f"LLM on VERITE: latency, {k.replace('_', ' ')} (s)", c["latency_s"][k])
    n.rate("llm_verite.consistent", f"LLM on VERITE: pairs with the same answer in all {d['consistency']['runs_per_pair']} runs",
           d["consistency"]["consistent"])
    for lab, v in d["consistency"]["by_label"].items():
        n.rate(f"llm_verite.consistent.{lab}", f"LLM consistency, {lab} pairs", v)
    for name, table in d["flag"].items():
        for lab, v in table.items():
            n.rate(f"llm_verite.{name}.{lab}.fired", f"LLM flag on VERITE {name} pairs: fired on {lab} pairs", v["fired"])


def _ablations(n: _Numbers) -> None:
    d = n.source("wp6_ablations.json", "ablations")
    for name, e in d["sets"].items():
        for method, runs in e["captioner"].items():
            for when, t in runs.items():
                for key, v in t.items():
                    n.rate(f"ablation.{name}.{method}.captioner_{when}.{key}",
                           f"Ablation, {name}: {method} check, captioner {when}: {key.replace('_', ' ')} "
                           f"(not assessed {v['not_assessed']})", v)
        n.add(f"ablation.{name}.llm_answered", f"Ablation, {name}: pairs the LLM answered", e["llm"]["answered"],
              f"{e['llm']['answered']}/{e['pairs']}")
        for cfg, t in e["llm"].items():
            if not isinstance(t, dict):
                continue
            for key, v in t.items():
                n.rate(f"ablation.{name}.{cfg.replace(' ', '_')}.{key}",
                       f"Ablation, {name} (answered pairs): {cfg}: {key.replace('_', ' ')}", v)


def _framing(n: _Numbers) -> None:
    d = n.source("wp6_framing_F.json", "framing-f")
    for variant in ("as written", "lowercased"):
        v, key = d[variant], variant.replace(" ", "_")
        n.rate(f"framing_f.{key}.precision", f"Shouting-style check on dataset F, {variant}: precision", v["precision"])
        n.rate(f"framing_f.{key}.recall", f"Shouting-style check on dataset F, {variant}: recall", v["recall"])
        n.add(f"framing_f.{key}.f1", f"Shouting-style check on dataset F, {variant}: F1", v["f1"], "", v["f1_ci"],
              "bootstrap 95% (2000)")
        for g, r in v["fire_rate"].items():
            n.rate(f"framing_f.{key}.fired.{g}", f"Shouting-style check, {variant}: fired on {g.replace('_', ' ')} texts", r)
    d = n.source("wp6_framing_llm.json", "framing-llm")
    n.rate("framing_llm.precision", "LLM on dataset F: precision", d["precision"])
    n.rate("framing_llm.recall", "LLM on dataset F: recall", d["recall"])
    n.add("framing_llm.f1", "LLM on dataset F: F1", d["f1"], "", d["f1_ci"], "bootstrap 95% (2000)")
    n.add("framing_llm.unreadable", "LLM on dataset F: unreadable answers", d["unreadable_answers"])
    n.add("framing_llm.mean_seconds", "LLM on dataset F: mean seconds per text (build run)", d["mean_seconds"])
    for g, r in d["yes_rate"].items():
        n.rate(f"framing_llm.yes.{g}", f"LLM on dataset F: said yes on {g.replace('_', ' ')} texts", r)


def _video(n: _Numbers) -> None:
    d = n.source("wp6_speech_swap.json", "speech-swap")
    n.add("speech_swap.threshold", "Speech vs picture threshold, fitted on the fitting side", d["threshold"])
    for side in ("fitting", "held_out"):
        v = d[side]
        n.add(f"speech_swap.{side}.true_flagged", f"Swap test, {side.replace('_', '-')}: true lines flagged",
              v["true_flagged_rate"], f"{v['true_flagged']}/{v['pairs']}", v["true_flagged_ci"], "Wilson 95%")
        n.add(f"speech_swap.{side}.swapped_caught", f"Swap test, {side.replace('_', '-')}: swapped lines caught",
              v["swapped_caught_rate"], f"{v['swapped_caught']}/{v['pairs']}", v["swapped_caught_ci"], "Wilson 95%")
        n.add(f"speech_swap.{side}.auc", f"Swap test, {side.replace('_', '-')}: AUC", v["auc"], "", v.get("auc_ci"),
              "source bootstrap 95% (2000)")
    d = n.source("wp6_video_pipeline.json", "video-pipeline")
    for scope in ("held_out_clips", "all_clips"):
        e = d[scope]
        for lab, v in e["lines"].items():
            n.add(f"video.{scope}.{lab}.flagged", f"Video path, {scope.replace('_', ' ')}: {lab.replace('_', ' ')} lines flagged",
                  v["rate"], f"{v['flagged']}/{v['lines']}", v["ci"], "Wilson 95%")
        cl = e["clip_level"]
        n.add(f"video.{scope}.mismatched_clips_caught", f"Video path, {scope.replace('_', ' ')}: mismatched clips with "
              "a correct flag", cl["with_a_correct_flag"] / cl["mismatched_clips"],
              f"{cl['with_a_correct_flag']}/{cl['mismatched_clips']}")
        n.add(f"video.{scope}.matching_clips_false_flag", f"Video path, {scope.replace('_', ' ')}: matching clips with "
              "a false flag", cl["with_a_false_flag"] / cl["matching_clips"], f"{cl['with_a_false_flag']}/{cl['matching_clips']}")
    t, dr = d["transcripts"], d["drift"]
    n.add("video.whisper_wer_pct", "Video path: Whisper word error rate on dataset E clips with speech (%)", t["wer_pct"])
    n.add("video.invented_sentences", "Video path: invented sentences in the transcripts", t["invented_sentences"])
    n.add("video.drift_either_over_1s", "Video path: script lines whose segment start or end is over 1 s off",
          dr["either_over_1s"], f"{dr['either_over_1s']}/{dr['lines'] - dr['no_transcribing_segment']}")
    n.add("video.drift_median_start_s", "Video path: median start drift (s)", dr["median_start_drift"])
    d = n.source("wp6_caption_video.json", "caption-video")
    n.rate("caption_video.own_flagged", "Caption vs video: clips flagged with their own caption", d["own_caption_flagged"])
    n.rate("caption_video.swapped_flagged", "Caption vs video: clips flagged with a swapped caption",
           d["swapped_caption_flagged"])
    n.add("caption_video.auc", "Caption vs video: AUC of the best keyframe score, own vs swapped", d["auc_best_similarity"])


def _whisper(n: _Numbers) -> None:
    d = n.source("wp6_whisper.json", "whisper")
    for m in ("tiny", "base"):
        v = d["models"][m]
        for set_ in ("librispeech", "dataset_e"):
            n.add(f"whisper.{m}.{set_}.wer_pct", f"Whisper {m}, {set_.replace('_', ' ')}: word error rate (%)",
                  v[set_]["wer_pct"], f"{v[set_]['edits']}/{v[set_]['ref_words']} words")
            n.add(f"whisper.{m}.{set_}.mean_s", f"Whisper {m}, {set_.replace('_', ' ')}: mean seconds per item (build run)",
                  v[set_]["mean_s"])
        n.add(f"whisper.{m}.peak_mb", f"Whisper {m}: peak memory of its process (MB, build run)", v["peak_memory_mb"])
    n.add("whisper.chosen", "Whisper model chosen by the WP-3 rule", d["models"]["decision"]["chosen"])
    for s, v in d["decoding"].items():
        for on in ("tuning_long", "e", "original"):
            n.add(f"whisper.decoding_{s}.{on}.wer_pct", f"Whisper base, decoding {s}, {on}: word error rate (%)",
                  v[on]["wer_pct"])
            n.add(f"whisper.decoding_{s}.{on}.invented", f"Whisper base, decoding {s}, {on}: invented sentences",
                  v[on]["invented_sentences"])
    n.add("whisper.decoding_chosen", "Decoding chosen by the ADR-049 rule", d["decoding_choice_long"]["chosen"])


def _faults(n: _Numbers) -> None:
    d = n.source("wp6_fault_injection.json", "fault-injection")
    s = d["summary"]
    n.add("faults.passed", "Fault injection: faults whose checks came back not assessed with a plain reason",
          s["passed"] / s["faults"], f"{s['passed']}/{s['faults']}")
    for kind in sorted({f["kind"] for f in d["faults"]}):
        fs = [f for f in d["faults"] if f["kind"] == kind]
        n.add(f"faults.{kind.replace(' ', '_')}", f"Fault injection, extractor {kind}: passed",
              sum(f["passed"] for f in fs) / len(fs), f"{sum(f['passed'] for f in fs)}/{len(fs)}")


def _web(n: _Numbers) -> None:
    d = n.source("wp6_web_archive.json", "web-archive")
    a = d["archive_answers_saved"]
    n.add("web.archive_answers_saved", "Web lookup replay: pages with a saved archive answer "
          f"({'complete' if d['replay_complete'] else 'the replay waits for the rest'})", a["pages"] / a["of"],
          f"{a['pages']}/{a['of']}")
    for run, r in d["runs"].items():
        if r is None:
            continue
        key = {"first live run (ADR-033)": "run1", "fixed live run (ADR-034)": "run2",
               "saved inputs, dating before ADR-043": "saved_before", "saved inputs, current dating": "saved_after"}[run]
        for set_, fields in (("A", ("found", "dated", "dating_errors")), ("D", ("found",)), ("VERITE", ("found", "dated"))):
            if set_ not in r:
                continue
            for f in fields:
                k, total, ci = r[set_][f]
                n.add(f"web.{key}.{set_}.{f}", f"Web lookup, {run}, dataset {set_}: {f.replace('_', ' ')}",
                      k / total if total else None, f"{k}/{total}", ci, "Wilson 95%")
        for source, count in r["date_sources"].items():
            n.add(f"web.{key}.date_source.{source}", f"Web lookup, {run}: pages dated by {source}", count)


def _performance(n: _Numbers) -> None:
    d = n.source("wp6_performance_image.json", "performance")
    for cfg in d["configurations"]:
        c = cfg["configuration"]
        n.add(f"perf.image.{c}.first_s", f"Image path ({c}): first call, seconds (loads the models)", cfg["first_call_s"]["total"])
        for stage, v in cfg["warm"].items():
            n.add(f"perf.image.{c}.warm_median_ms.{stage.replace(' ', '_')}",
                  f"Image path ({c}), {cfg['images'] - 1} warm images: {stage}, median ms", v["median_ms"])
        n.add(f"perf.image.{c}.warm_p95_ms", f"Image path ({c}): whole path, 95th percentile ms", cfg["warm"]["total"]["p95_ms"])
        n.add(f"perf.image.{c}.peak_mb", f"Image path ({c}): peak memory of the process (MB)", cfg["peak_memory_mb"])
    d = n.source("wp6_performance_video.json", "performance")
    for i, run in enumerate(d["runs"]):
        when = "fresh" if i == 0 else "warm"
        n.add(f"perf.video.{when}.total_s", f"Video path, {d['clip']} ({d['duration_s']:.0f} s), {when}: total seconds",
              run["total_s"])
        for st in run["stages"]:
            n.add(f"perf.video.{when}.{st['stage']}_s", f"Video path, {when}: {st['stage']} stage, seconds", st["seconds"])
    n.add("perf.video.peak_mb", "Video path: peak memory of the process (MB)", d["process_peak_mb"])
    d = n.source("wp6_model_probe.json", "performance")
    for row in d:
        if "error" in row:
            n.add(f"perf.model.{row['model']}.error", f"Model probe, {row['model']}: failed", row["error"])
            continue
        n.add(f"perf.model.{row['model']}.load_s", f"Model {row['model']}: load time with the network blocked (s)", row["load_s"])
        n.add(f"perf.model.{row['model']}.peak_rise_mb", f"Model {row['model']}: rise in peak memory (MB)", row["peak_rise_mb"])


def _review(n: _Numbers) -> None:
    d = n.source("wp6_interface_review.json", "interface-review")
    for r in ("A", "B"):
        s = d[r]
        n.add(f"review.{r}.serious_or_critical", f"Interface review round {r}: serious or critical accessibility faults "
              f"({s['screens']} screens)", s["accessibility_serious_or_critical"])
        n.add(f"review.{r}.targets_under_24", f"Interface review round {r}: click targets under 24 x 24 px", s["targets_under_24"])
        n.add(f"review.{r}.targets_under_44", f"Interface review round {r}: click targets under 44 x 44 px",
              s["targets_under_44"], f"{s['targets_under_44']}/{s['controls_measured']}")
        n.add(f"review.{r}.text_under_16px_share", f"Interface review round {r}: largest share of a screen's text under 16 px",
              s["text_under_16px_max_share"])
        rl = s["reading_level"]
        n.add(f"review.{r}.reading_at_or_below_8", f"Interface review round {r}: explanation lines at grade 8 or below",
              rl["at_or_below_8"] / rl["lines"], f"{rl['at_or_below_8']}/{rl['lines']}")
        n.add(f"review.{r}.reading_median", f"Interface review round {r}: median reading grade", rl["median"])
        n.add(f"review.{r}.reading_max", f"Interface review round {r}: highest reading grade", rl["max"])
        for size, total in s["screens_of_scrolling"].items():
            n.add(f"review.{r}.scrolling.{size}", f"Interface review round {r}: screens of scrolling over T1-T5, {size}", total)
        for t, sec in s["predicted_time_s"].items():
            n.add(f"review.{r}.klm.{t}", f"Interface review round {r}: predicted time for {t} (s, with the measured wait)", sec)
        n.add(f"review.{r}.backend_verdict_words", f"Interface review round {r}: verdict words in the backend's lines",
              s["backend_verdict_words"])


def _audits(n: _Numbers) -> None:
    d = n.source("wp6_no_verdict.json", "no-verdict")
    e = d["evaluation_outputs"]
    n.add("no_verdict.lines", "No-verdict audit: distinct output lines scanned", e["distinct_lines"])
    n.add("no_verdict.own_word_hits", "No-verdict audit: verdict words in the app's own words", e["own_word_hits"])
    n.add("no_verdict.quoted_or_input_hits", "No-verdict audit: verdict words inside quoted or input text",
          sum(e["hits_in_quoted_or_input_text"].values()))
    n.add("no_verdict.llm_hits", f"No-verdict audit: verdict words in the LLM's explanations "
          f"({d['llm_explanations']['distinct']} distinct)", d["llm_explanations"]["hits"])
    n.add("no_verdict.every_branch_hits", "No-verdict audit: verdict words in every branch of every check",
          d["every_branch_of_every_check"]["hits"], f"{d['every_branch_of_every_check']['lines']} lines")
    n.add("no_verdict.interface_hits", "No-verdict audit: verdict words on the round B screens "
          f"({', '.join(f'{w} {c}' for w, c in d['interface_round_B']['by_word'].items())})", d["interface_round_B"]["hits"])
    d = n.source("wp6_readability.json", "readability")
    n.rate("readability.at_or_below_8", "Readability: explanation and what-to-check lines at grade 8 or below",
           d["share_at_or_below_8"])
    n.add("readability.median", "Readability: median grade", d["median"])
    n.add("readability.max", "Readability: highest grade", d["max"])


SECTIONS = (_synthetic, _matching, _hard_pairs, _date, _heldout, _fresh, _limits, _llm, _ablations, _framing, _video,
            _whisper, _faults, _web, _performance, _review, _audits)


def write_numbers() -> None:
    """Write results/report_numbers.csv; sections whose step file is missing are left out."""
    n, missing = _Numbers(), []
    for section in SECTIONS:
        try:
            section(n)
        except FileNotFoundError as exc:
            missing.append(Path(exc.filename).name)
    with open(OUT / "report_numbers.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=_Numbers.FIELDS)
        w.writeheader()
        w.writerows(n.rows)
    print(f"wrote results/report_numbers.csv: {len(n.rows)} figures")
    if missing:
        print(f"  not yet produced, so left out: {', '.join(missing)}")


# ------------------------------------------------------------------------------ charts
def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.dpi": 300, "savefig.bbox": "tight"})
    return plt


COLOURS = ["#1f4e79", "#c55a11", "#548235", "#7f6000", "#7030a0", "#595959"]


def _err(r: dict) -> tuple[float, list[float]]:
    """A rate and its Wilson interval as error-bar lengths."""
    v = r["k"] / r["n"] if r["n"] else 0.0
    lo, hi = r["ci"]
    return v, [max(0.0, v - lo), max(0.0, hi - v)]


def _grouped(ax, groups: list[str], series: dict[str, list[dict]], ylabel: str = "Share of pairs") -> None:
    """Bars per group, one per series, each a {k, n, ci} rate with its interval."""
    width = 0.8 / len(series)
    for i, (name, rates) in enumerate(series.items()):
        xs = [g + (i - (len(series) - 1) / 2) * width for g in range(len(groups))]
        vals, errs = zip(*(_err(r) for r in rates))
        ax.bar(xs, vals, width, yerr=list(zip(*errs)), capsize=2, label=name, color=COLOURS[i % len(COLOURS)])
    ax.set_xticks(range(len(groups)), groups)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))  # never over the bars


def _save(plt, fig, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / name)
    plt.close(fig)
    print(f"wrote results/figures/{name}")


def _fig_matching(plt) -> None:
    d = _read("wp6_matching_A.json")["all_40"]
    names = [t for t in d if t != "all transformations"]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    _grouped(ax, [t.replace("_", " ") for t in names],
             {"Hash alone": [d[t]["hash"] for t in names], "As shipped (hash, then keypoints)": [d[t]["shipped"] for t in names]},
             "Share of the 40 photos found")
    ax.tick_params(axis="x", rotation=45)
    ax.set_xlabel("Transformation of the photo")
    ax.set_title("Finding a re-posted copy of each dataset A photo")
    _save(plt, fig, "fig01_matching_A.png")


def _fig_hard_pairs(plt) -> None:
    d = _read("wp6_hard_pairs.json")
    variants = ("without alignment", "with alignment")
    fig, ax = plt.subplots(figsize=(6, 3.5))
    _grouped(ax, ["Queries returning another photo", "Pairs where a photo matched its partner"],
             {f"Keypoints {v}": [d[v]["queries_with_another_photo"], d[v]["pairs_with_a_partner_match"]] for v in variants},
             "Share")
    ax.set_ylim(0, 0.3)
    ax.set_title("Wrong matches between similar photos (20 pairs)")
    ax.set_xlabel("Measure")
    _save(plt, fig, "fig02_hard_pairs.png")


def _fig_date(plt) -> None:
    d = _read("wp6_date_check.json")["conditions"]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    names = list(d)
    vals, errs = zip(*(_err(d[c]) for c in names))
    ax.bar(range(len(names)), vals, yerr=list(zip(*errs)), capsize=2, color=COLOURS[0])
    ax.set_xticks(range(len(names)), [f"{c}\n({d[c]['expected'].replace('_', ' ')})" for c in names], fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Share of the 40 photos as expected")
    ax.set_xlabel("Posting date given (expected status)")
    ax.set_title("Recycled-context date check on dataset A")
    _save(plt, fig, "fig03_date_check.png")


def _fig_heldout(plt) -> None:
    d = _read("wp6_verite_heldout.json")["heldout"]
    names = {"chosen_rule": "Calibrated rule", "word_overlap": "Word overlap", "llm_answered_only": "LLM (answered)",
             "always_fire": "Always fire"}
    fig, ax = plt.subplots(figsize=(7, 3.8))
    _grouped(ax, [t for _, t in WP2_KEYS], {label: [d[k][m] for m, _ in WP2_KEYS] for k, label in names.items()})
    ax.set_xlabel("VERITE label")
    ax.set_title("Caption vs picture on the 180 held-out VERITE pairs")
    _save(plt, fig, "fig04_verite_heldout.png")


def _fig_fresh(plt) -> None:
    d = _read("wp6_verite_fresh.json")["rules"]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    _grouped(ax, [t for _, t in WP2_KEYS], {rule.capitalize(): [d[rule][m] for m, _ in WP2_KEYS] for rule in d})
    ax.set_xlabel("VERITE label")
    ax.set_title(f"Caption vs picture on the {_read('wp6_verite_fresh.json')['fresh_pairs']} fresh VERITE pairs")
    _save(plt, fig, "fig05_verite_fresh.png")


def _fig_auc(plt) -> None:
    d = _read("wp6_verite_heldout.json")["auc"]
    scores = list(d)
    contrasts = list(d[scores[0]])
    fig, ax = plt.subplots(figsize=(7, 3.8))
    width = 0.8 / len(contrasts)
    for i, c in enumerate(contrasts):
        xs = [s + (i - (len(contrasts) - 1) / 2) * width for s in range(len(scores))]
        vals = [d[s][c]["auc"] for s in scores]
        errs = [[v - d[s][c]["ci"][0] for s, v in zip(scores, vals)], [d[s][c]["ci"][1] - v for s, v in zip(scores, vals)]]
        ax.bar(xs, vals, width, yerr=errs, capsize=2, label=c.capitalize(), color=COLOURS[i])
    ax.axhline(0.5, color="#999999", linewidth=0.8, linestyle="--")
    ax.set_xticks(range(len(scores)), [s.replace("img:", "picture ").replace("txt:", "text ") for s in scores], fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("AUC (0.5 = chance)")
    ax.set_xlabel("Score")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("How well each score separates truthful from mismatched pairs (held-out)")
    _save(plt, fig, "fig06_verite_auc.png")


def _fig_limits(plt) -> None:
    d = _read("wp6_picture_limits.json")["fresh"]["rule"]["image"]
    keys = ("truthful_flagged", "out_of_context_caught", "miscaptioned_caught")
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    _grouped(ax, ["Truthful flagged", "Out-of-context caught", "Miscaptioned caught"],
             {"Before the limits": [d["before"][k] for k in keys], "After (blank or text pictures not assessed)":
              [d["after"][k] for k in keys]})
    ax.set_xlabel("VERITE label")
    ax.set_title("Picture check on the fresh pairs, before and after the picture limits")
    _save(plt, fig, "fig07_picture_limits.png")


def _fig_llm(plt) -> None:
    d = _read("wp6_llm_verite.json")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.5))
    a1.hist(d["calls"]["latencies"], bins=30, color=COLOURS[0])
    a1.axvline(d["calls"]["latency_s"]["median"], color=COLOURS[1], linestyle="--", label="median")
    a1.set_xlabel("Seconds per call")
    a1.set_ylabel("Calls")
    a1.set_title(f"LLM latency ({d['calls']['n']} VERITE calls)")
    a1.legend(frameon=False)
    labels = ("true", "out-of-context", "miscaptioned")
    _grouped(a2, list(labels), {f"{s.replace('heldout', 'held-out')} pairs": [d["flag"][s][lab]["fired"] for lab in labels]
                                for s in d["flag"]}, "Share of pairs the LLM flagged")
    a2.set_xlabel("VERITE label")
    a2.set_title("LLM flag by label")
    _save(plt, fig, "fig08_llm_verite.png")


def _fig_ablations(plt) -> None:
    d = _read("wp6_ablations.json")["sets"]["fresh"]
    keys = ("truthful_flagged", "out_of_context_caught", "miscaptioned_caught")
    groups = ["Truthful flagged", "Out-of-context caught", "Miscaptioned caught"]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.5, 7))
    _grouped(a1, groups, {f"{m}, captioner {w}": [d["captioner"][m][w][k] for k in keys]
                          for m in ("image", "meaning", "overlap") for w in ("on", "off")})
    a1.set_title("Captioner on and off (fresh pairs)")
    a1.set_xlabel("VERITE label")
    _grouped(a2, groups, {cfg[0].upper() + cfg[1:]: [d["llm"][cfg][k] for k in keys]
                          for cfg in d["llm"] if isinstance(d["llm"][cfg], dict)})
    a2.set_title(f"LLM on and off (fresh pairs the LLM answered: {d['llm']['answered']})")
    a2.set_xlabel("VERITE label")
    _save(plt, fig, "fig09_ablations.png")


def _fig_framing(plt) -> None:
    w, l = _read("wp6_framing_F.json")["as written"], _read("wp6_framing_llm.json")
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for i, (name, v) in enumerate((("Shouting-style check", w), ("Local LLM", l))):
        vals = [v["precision"]["rate"], v["recall"]["rate"], v["f1"]]
        lo = [v["precision"]["ci"][0], v["recall"]["ci"][0], v["f1_ci"][0]]
        hi = [v["precision"]["ci"][1], v["recall"]["ci"][1], v["f1_ci"][1]]
        xs = [x + (i - 0.5) * 0.38 for x in range(3)]
        ax.bar(xs, vals, 0.38, yerr=[[a - b for a, b in zip(vals, lo)], [b - a for a, b in zip(vals, hi)]], capsize=2,
               label=name, color=COLOURS[i])
    ax.set_xticks(range(3), ["Precision", "Recall", "F1"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_xlabel("Measure")
    ax.legend(frameon=False)
    ax.set_title("Manipulative wording on the 99 dataset F texts")
    _save(plt, fig, "fig10_framing_F.png")


def _fig_swap(plt) -> None:
    d = _read("wp6_speech_swap.json")
    h = d["held_out"]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    _grouped(ax, ["True lines flagged", "Swapped lines caught"],
             {"Held-out side": [{"k": h["true_flagged"], "n": h["pairs"], "ci": h["true_flagged_ci"]},
                                {"k": h["swapped_caught"], "n": h["pairs"], "ci": h["swapped_caught_ci"]}]},
             "Share of lines")
    ax.set_xlabel("Line")
    ax.set_title(f"Speech vs picture swap test (threshold {d['threshold']:.4f})")
    _save(plt, fig, "fig11_speech_swap.png")


def _fig_video(plt) -> None:
    d = _read("wp6_video_pipeline.json")
    labs = ("matches", "different_scene", "wrong_detail")
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    _grouped(ax, ["Line matches", "Different scene", "Wrong detail"],
             {scope.replace("_", " ").capitalize(): [{"k": d[scope]["lines"][l]["flagged"], "n": d[scope]["lines"][l]["lines"],
                                                       "ci": d[scope]["lines"][l]["ci"]} for l in labs]
              for scope in ("held_out_clips", "all_clips")}, "Share of script lines flagged")
    ax.set_xlabel("What the line says against the picture")
    ax.set_title("The whole video path on dataset E")
    _save(plt, fig, "fig12_video_pipeline.png")


def _fig_caption_video(plt) -> None:
    d = _read("wp6_caption_video.json")
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    _grouped(ax, ["Own caption", "Swapped caption"],
             {"Clips flagged": [d["own_caption_flagged"], d["swapped_caption_flagged"]]}, f"Share of the {d['clips']} clips")
    ax.set_xlabel("Caption given with the clip")
    ax.set_title("Caption vs video on dataset E")
    _save(plt, fig, "fig13_caption_video.png")


def _fig_whisper(plt) -> None:
    d = _read("wp6_whisper.json")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.5))
    sets = ("librispeech", "dataset_e")
    for i, m in enumerate(("tiny", "base")):
        a1.bar([x + (i - 0.5) * 0.38 for x in range(2)], [d["models"][m][s]["wer_pct"] for s in sets], 0.38,
               label=f"Whisper {m}", color=COLOURS[i])
    a1.set_xticks(range(2), ["LibriSpeech (100)", "Dataset E clips (32)"])
    a1.set_ylabel("Word error rate (%)")
    a1.set_xlabel("Test set")
    a1.legend(frameon=False)
    a1.set_title("Whisper tiny against base")
    ons = ("tuning_long", "e", "original")
    for i, s in enumerate(d["decoding"]):
        bars = a2.bar([x + (i - 1) * 0.27 for x in range(3)], [d["decoding"][s][on]["invented_sentences"] for on in ons],
                      0.27, label=f"Setting {s}", color=COLOURS[i])
        a2.bar_label(bars, fontsize=7)  # a zero stays visible
    a2.set_xticks(range(3), ["Long tuning files", "Dataset E", "LibriSpeech"])
    a2.set_ylabel("Invented sentences")
    a2.set_xlabel("Test set")
    a2.legend(frameon=False)
    a2.set_title("Whisper base decoding settings")
    _save(plt, fig, "fig14_whisper.png")


def _fig_faults(plt) -> None:
    d = _read("wp6_fault_injection.json")
    extractors = sorted({f["extractor"] for f in d["faults"]})
    passed = [sum(f["passed"] for f in d["faults"] if f["extractor"] == e) for e in extractors]
    total = [sum(f["extractor"] == e for f in d["faults"]) for e in extractors]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.bar(range(len(extractors)), total, color="#d9d9d9", label="Faults injected")
    ax.bar(range(len(extractors)), passed, 0.5, color=COLOURS[2], label="Passed")
    ax.set_xticks(range(len(extractors)), extractors, rotation=40, ha="right", fontsize=7)
    ax.set_ylabel("Faults")
    ax.set_xlabel("Extractor switched off, timed out or broken")
    ax.legend(frameon=False)
    ax.set_title(f"Fault injection: {d['summary']['passed']} of {d['summary']['faults']} passed")
    _save(plt, fig, "fig15_fault_injection.png")


def _fig_web(plt) -> None:
    d = _read("wp6_web_archive.json")["runs"]
    runs = [r for r in d if d[r] is not None]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    series = {"A photos found": [], "A photos dated": [], "VERITE images dated": []}
    for r in runs:
        for name, (s, f) in zip(series, (("A", "found"), ("A", "dated"), ("VERITE", "dated"))):
            k, n, ci = d[r][s][f]
            series[name].append({"k": k, "n": n, "ci": ci})
    _grouped(ax, [r.replace(", ", ",\n").replace(" (", "\n(") for r in runs], series, "Share of images")
    ax.tick_params(axis="x", labelsize=7)
    ax.set_xlabel("Run of the web lookup")
    ax.set_title("Live web lookup: earlier copies found and dated")
    _save(plt, fig, "fig16_web_lookup.png")


def _fig_performance(plt) -> None:
    img = _read("wp6_performance_image.json")["configurations"]
    vid = _read("wp6_performance_video.json")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    stages = [s for s in dict.fromkeys(k for c in img for k in c["warm"]) if s != "total"]
    bottom = [0.0] * len(img)
    for i, s in enumerate(stages):
        vals = [c["warm"].get(s, {}).get("median_ms", 0) / 1000 for c in img]
        a1.bar(range(len(img)), vals, bottom=bottom, label=s, color=COLOURS[i % len(COLOURS)])
        bottom = [b + v for b, v in zip(bottom, vals)]
    a1.set_xticks(range(len(img)), ["Shipped settings", "LLM on"])
    a1.set_ylabel("Median seconds per image (warm)")
    a1.set_xlabel("Configuration")
    a1.legend(frameon=False, fontsize=7)
    a1.set_title("Image path, time per stage")
    runs = vid["runs"]
    names = {"keyframes": "Keyframes", "ocr": "OCR", "reverse_image": "Image lookup", "speech": "Speech (Whisper)",
             "clip": "CLIP (caption and speech)"}
    vstages = [st["stage"] for st in runs[0]["stages"]]
    bottom = [0.0] * len(runs)
    for i, s in enumerate(vstages):
        vals = [next(st["seconds"] for st in r["stages"] if st["stage"] == s) for r in runs]
        a2.bar(range(len(runs)), vals, bottom=bottom, label=names.get(s, s), color=COLOURS[i % len(COLOURS)])
        bottom = [b + v for b, v in zip(bottom, vals)]
    a2.set_xticks(range(len(runs)), ["Fresh process", "Warm"])
    a2.set_ylabel("Seconds")
    a2.set_xlabel("Run")
    a2.legend(frameon=False, fontsize=7)
    a2.set_title(f"Video path, {vid['clip']} ({vid['duration_s']:.0f} s), time per stage")
    _save(plt, fig, "fig17_performance.png")
    probe = [r for r in _read("wp6_model_probe.json") if "error" not in r]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(range(len(probe)), [r["peak_rise_mb"] for r in probe], color=COLOURS[0])
    ax.set_xticks(range(len(probe)), [r["model"] for r in probe], rotation=30, ha="right")
    ax.set_ylabel("Rise in peak memory (MB)")
    ax.set_xlabel("Model, loaded alone with the network blocked")
    ax.set_title("Memory each model adds")
    _save(plt, fig, "fig18_model_memory.png")


def _fig_review(plt) -> None:
    d = _read("wp6_interface_review.json")
    measures = (("accessibility_serious_or_critical", "Serious or critical\naccessibility faults"),
                ("targets_under_44", "Click targets\nunder 44 px"), ("targets_under_24", "Click targets\nunder 24 px"))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
    for i, r in enumerate("AB"):
        bars = a1.bar([x + (i - 0.5) * 0.38 for x in range(len(measures))], [d[r][k] for k, _ in measures], 0.38,
                      label=f"Round {r}", color=COLOURS[i])
        a1.bar_label(bars, fontsize=7)  # round B's zeros stay visible
    a1.set_xticks(range(len(measures)), [t for _, t in measures], fontsize=8)
    a1.set_ylabel("Count over all screens")
    a1.set_xlabel("Measure")
    a1.legend(frameon=False)
    a1.set_title("Interface review: faults found")
    shares = []
    for r in "AB":
        rl = d[r]["reading_level"]
        shares.append([d[r]["text_under_16px_max_share"], 1 - rl["at_or_below_8"] / rl["lines"]])
    for i, r in enumerate("AB"):
        bars = a2.bar([x + (i - 0.5) * 0.38 for x in range(2)], shares[i], 0.38, label=f"Round {r}", color=COLOURS[i])
        a2.bar_label(bars, fmt="%.2f", fontsize=7)
    a2.set_xticks(range(2), ["Largest share of a screen's\ntext under 16 px", "Explanation lines\nabove grade 8"], fontsize=8)
    a2.set_ylim(0, 1)
    a2.set_ylabel("Share")
    a2.set_xlabel("Measure")
    a2.legend(frameon=False)
    a2.set_title("Interface review: text")
    _save(plt, fig, "fig19_interface_review.png")


def _fig_readability(plt) -> None:
    grades = _read("wp6_readability.json")["grades"]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(grades, bins=[x / 2 for x in range(-8, 26)], color=COLOURS[0])
    ax.axvline(8, color=COLOURS[1], linestyle="--", label="Target: grade 8")
    ax.set_xlabel("Flesch-Kincaid grade")
    ax.set_ylabel("Distinct lines")
    ax.legend(frameon=False)
    ax.set_title(f"Reading grade of the {len(grades)} explanation lines the evaluation produced")
    _save(plt, fig, "fig20_readability.png")


CHARTS = (_fig_matching, _fig_hard_pairs, _fig_date, _fig_heldout, _fig_fresh, _fig_auc, _fig_limits, _fig_llm,
          _fig_ablations, _fig_framing, _fig_swap, _fig_video, _fig_caption_video, _fig_whisper, _fig_faults, _fig_web,
          _fig_performance, _fig_review, _fig_readability)


def draw_charts() -> None:
    """Draw every chart into results/figures/; charts whose step file is missing are skipped."""
    plt = _plt()
    missing = []
    for chart in CHARTS:
        try:
            chart(plt)
        except FileNotFoundError as exc:
            missing.append(Path(exc.filename).name)
            plt.close("all")
    if missing:
        print(f"  charts left out, their step files are not yet produced: {', '.join(sorted(set(missing)))}")
