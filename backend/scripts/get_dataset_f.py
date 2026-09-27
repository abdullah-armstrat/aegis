"""Build dataset F: 100 meme texts labelled for manipulation and loudness, 25 in each group.

Source: SemEval-2021 Task 6 (Dimitrov et al., 2021), subtask 1: meme texts with human labels for
20 persuasion techniques, from the task's GitHub repository at a pinned commit ("free for general
research use"). Loudness comes from the NRC Emotion Intensity Lexicon (Mohammad, 2018; free for
non-commercial research and educational use). Neither uses Aegis's own markers. The definitions
below were fixed in the project's decision log before any text was read:

  manipulative  the human labels include Loaded Language, Appeal to fear/prejudice or
                Exaggeration/Minimisation
  honest        no technique labelled at all (texts with only other techniques are excluded)
  loud          at least one word whose highest emotion intensity in the lexicon is 0.75 or more;
                words are the lowercased text split on letters (inner apostrophes kept), looked up
                exactly, with no stemming
  groups        calm and honest, calm but manipulative, loud and honest, loud and manipulative;
                25 each, drawn with a fixed seed from the eligible texts sorted by split and id

Texts that are identical after collapsing whitespace are counted once. The texts and the lexicon
stay in the git-ignored data/F_captions/; the committed manifest data/labels/F_harder_captions.csv
holds ids, labels, groups and the loudness evidence, plus each text's SHA-256.

Run:  python backend/scripts/get_dataset_f.py
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import random
import re
import sys
import zipfile
from datetime import date
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "data" / "F_captions"
MANIFEST = ROOT / "data" / "labels" / "F_harder_captions.csv"
USER_AGENT = ("aegis-dataset-builder/1.0 (University of London CM3070 student project; "
              "contact 189360416+abdullah-armstrat@users.noreply.github.com)")

REPO = "di-dimitrov/SEMEVAL-2021-task6-corpus"
COMMIT = "796b32fa76b7a5814a48d1870d163058e81e4ce1"
SPLITS = ["training_set_task1.txt", "dev_set_task1.txt", "test_set_task1.txt"]
LEXICON_URL = "http://saifmohammad.com/WebDocs/Lexicons/NRC-Emotion-Intensity-Lexicon.zip"

MANIPULATIVE = {"Loaded Language", "Appeal to fear/prejudice", "Exaggeration/Minimisation"}
LOUD_CUTOFF = 0.75
SEED = 20260927
PER_GROUP = 25
GROUPS = ["calm_honest", "calm_manipulative", "loud_honest", "loud_manipulative"]
WORD = re.compile(r"[a-z]+(?:'[a-z]+)*")


def fetch(client: httpx.Client) -> tuple[list[dict], dict[str, tuple[float, str]], str]:
    """The labelled meme texts, the lexicon (word -> highest intensity and its emotion), its file name."""
    WORK.mkdir(parents=True, exist_ok=True)
    texts = []
    for split in SPLITS:
        url = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/data/{split}"
        raw = client.get(url)
        raw.raise_for_status()
        (WORK / split).write_bytes(raw.content)
        for item in json.loads(raw.content.decode("utf-8")):
            texts.append({"split": split.split("_set")[0], "id": str(item["id"]),
                          "labels": list(item["labels"]), "text": item["text"], "url": url})
    resp = client.get(LEXICON_URL)
    resp.raise_for_status()
    (WORK / "NRC-Emotion-Intensity-Lexicon.zip").write_bytes(resp.content)
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        name = next(n for n in zf.namelist()
                    if re.search(r"NRC-Emotion-Intensity-Lexicon-v1\.txt$", n) and "MACOSX" not in n)
        lines = zf.read(name).decode("utf-8").splitlines()
    lexicon: dict[str, tuple[float, str]] = {}
    for line in lines:
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        word, emotion, score = parts
        try:
            value = float(score)
        except ValueError:
            continue  # header line
        if value > lexicon.get(word, (-1.0, ""))[0]:
            lexicon[word] = (value, emotion)
    return texts, lexicon, name


def loudness(text: str, lexicon: dict[str, tuple[float, str]]) -> tuple[float, str, str, int]:
    """Highest word intensity in the text, the word, its emotion, and the number of words."""
    words = WORD.findall(text.lower())
    best = (0.0, "", "")
    for w in words:
        if w in lexicon and lexicon[w][0] > best[0]:
            best = (lexicon[w][0], w, lexicon[w][1])
    return best[0], best[1], best[2], len(words)


def main() -> None:
    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=60, follow_redirects=True) as client:
        texts, lexicon, lexicon_file = fetch(client)

    seen, unique = set(), []
    order = {s.split("_set")[0]: i for i, s in enumerate(SPLITS)}
    for t in sorted(texts, key=lambda t: (order[t["split"]], t["id"])):
        norm = " ".join(t["text"].split())
        if norm in seen:
            continue
        seen.add(norm)
        unique.append(t)

    eligible: dict[str, list[dict]] = {g: [] for g in GROUPS}
    for t in unique:
        labels = set(t["labels"])
        if labels & MANIPULATIVE:
            manipulative = True
        elif not labels:
            manipulative = False
        else:
            continue  # only other techniques: neither honest nor manipulative
        score, word, emotion, n_words = loudness(t["text"], lexicon)
        loud = score >= LOUD_CUTOFF
        t.update(manipulative=manipulative, loud=loud, score=score, word=word, emotion=emotion, n_words=n_words)
        eligible[f"{'loud' if loud else 'calm'}_{'manipulative' if manipulative else 'honest'}"].append(t)

    rng = random.Random(SEED)
    chosen, shortfall = [], {}
    for group in GROUPS:
        pool = eligible[group]
        picked = rng.sample(pool, PER_GROUP) if len(pool) >= PER_GROUP else list(pool)
        shortfall[group] = PER_GROUP - len(picked)
        chosen += [(group, t) for t in picked]

    rows, text_rows = [], []
    for n, (group, t) in enumerate(chosen, 1):
        cid = f"F{n:03d}"
        text_sha = hashlib.sha256(t["text"].encode("utf-8")).hexdigest()
        rows.append({
            "caption_id": cid, "semeval_split": t["split"], "semeval_id": t["id"], "group": group,
            "manipulative": "yes" if t["manipulative"] else "no", "loud": "yes" if t["loud"] else "no",
            "techniques": "; ".join(t["labels"]), "max_intensity": t["score"], "max_word": t["word"],
            "max_emotion": t["emotion"], "n_words": t["n_words"], "text_sha256": text_sha,
            "source_url": t["url"], "author": "Dimitrov et al. (2021), SemEval-2021 Task 6",
            "licence": "Free for general research use (task repository)",
            "loudness_source": f"NRC Emotion Intensity Lexicon v1 ({lexicon_file}); non-commercial research use",
            "access_date": date.today().isoformat(),
        })
        text_rows.append({"caption_id": cid, "semeval_id": t["id"], "text": t["text"], "text_sha256": text_sha})

    with open(MANIFEST, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(WORK / "captions.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(text_rows[0]))
        w.writeheader()
        w.writerows(text_rows)

    print(f"SemEval-2021 Task 6 subtask 1 at {COMMIT[:7]}: {len(texts)} texts, {len(unique)} after removing duplicates")
    print(f"lexicon: {len(lexicon)} words from {lexicon_file}")
    for group in GROUPS:
        print(f"  {group:<18} eligible {len(eligible[group]):>4}  sampled {PER_GROUP - shortfall[group]:>2}"
              f"  shortfall {shortfall[group]}")
    print(f"wrote {MANIFEST.relative_to(ROOT)} and {(WORK / 'captions.csv').relative_to(ROOT)}")
    if any(shortfall.values()):
        print("a group is short of 25; see above", file=sys.stderr)


if __name__ == "__main__":
    main()
