#!/usr/bin/env python3
"""
English emotion-concreteness statistics (Valence, Arousal, Dominance, Concreteness)
model-vs-human, parallel to the Mandarin script.

Outputs
-------
1. Console tables:
   • Global median V / A / D / C per category (human, each model × prompt)
   • Lexicon coverage (% of types / tokens matched) for VAD & Concreteness
   • % Concrete / % Abstract / % Unknown, using a concreteness threshold

2. JSON file  (english_per_cue_metrics.json) with per-cue metrics.
"""
import csv
import json
import os
import string
from collections import defaultdict
from statistics import median

import pandas as pd
from nltk.stem import WordNetLemmatizer

# ─── PATHS ────────────────────────────────────────────────────────────────────
PICKLE_PATH = "/data/gpfs/projects/punim2219/LM_with_SWOW/kabir/Data/swow_en_results_processed.pkl"
VAD_PATH    = "/data/gpfs/projects/punim2219/LM_with_SWOW/kabir/Data/VAD-emot-submit.csv"
CONC_PATH   = "/data/gpfs/projects/punim2219/LM_with_SWOW/kabir/Data/Concreteness_ratings_Brysbaert_et_al_BRM.txt"

THRESHOLD   = 3.0          # 1-5 scale (≥3 ⇒ “concrete”, <3 ⇒ “abstract”)

# ─── TEXT PRE-PROCESSING ──────────────────────────────────────────────────────
_LEMM   = WordNetLemmatizer()
_PUNCS  = string.punctuation + "“”‘’"

def preprocess(tok: str) -> str:
    """Lower-case, strip punctuation, lemmatise."""
    return _LEMM.lemmatize(tok.lower().strip(_PUNCS))

# ─── DATA LOADERS ─────────────────────────────────────────────────────────────
def load_from_pickle(path: str) -> dict:
    """
    Returns: cue → {ground_truth: [w], models: {model: {prompt: [w]}}}
    """
    df   = pd.read_pickle(path)
    data = {}
    for cue, grp in df.groupby("Cue Word"):
        gt_raw = grp.iloc[0]["Ground Truth Associated Words"] or ""
        gt     = [w.strip() for w in gt_raw.split(",") if w.strip()][:10]

        models = {}
        for (m, p), sub in grp.groupby(["Model Type", "Prompt Type"]):
            words = []
            for cell in sub["Parsed Associated Words"].dropna():
                words.extend([w.strip() for w in cell.split(",") if w.strip()])
            models.setdefault(m, {})[p] = words[:10]

        data[cue] = {"ground_truth": gt, "models": models}
        
    return data


def load_vad(path: str) -> dict:
    """
    English VAD lexicon — returns word → (V, A, D) on 1-9 scale.
    """
    vad = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            w = row["Word"].strip().lower()
            try:
                v = float(row["V.Mean.Sum"])
                a = float(row["A.Mean.Sum"])
                d = float(row["D.Mean.Sum"])
            except ValueError:
                continue
            vad[w] = (v, a, d)
    return vad


def load_concreteness(path: str) -> dict:
    """
    Brysbaert concreteness ratings — returns word → 1-5 (1=abstract, 5=concrete)
    """
    conc = {}
    with open(path, encoding="utf-8") as f:
        next(f)  # skip header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            w = parts[0].lower()
            try:
                conc[w] = float(parts[2])
            except ValueError:
                continue
    return conc

# ─── METRIC HELPERS ──────────────────────────────────────────────────────────
def compute_metrics(raws, vad, conc):
    vs, as_, ds, cs = [], [], [], []
    for raw in raws:
        w = preprocess(raw)
        if w in vad:
            v, a, d = vad[w]
            vs.append(v); as_.append(a); ds.append(d)
        if w in conc:
            cs.append(conc[w])
    return {
        "valence"     : median(vs)  if vs else None,
        "arousal"     : median(as_) if as_ else None,
        "dominance"   : median(ds)  if ds else None,
        "concreteness": median(cs)  if cs else None,
    }


def compute_threshold_percents(raws, conc, thr):
    """
    Returns %Concrete / %Abstract / %Unknown for a list of raw tokens.
    """
    total = len(raws)
    if total == 0:
        return None, None, None
    conc_cnt = abs_cnt = unk_cnt = 0
    for raw in raws:
        w = preprocess(raw)
        if w not in conc:
            unk_cnt += 1
        elif conc[w] >= thr:
            conc_cnt += 1
        else:
            abs_cnt += 1
    return conc_cnt / total * 100, abs_cnt / total * 100, unk_cnt / total * 100

# ─── AGGREGATION ─────────────────────────────────────────────────────────────
def aggregate(data, vad, conc):
    """
    per_cue[cue][category][prompt] → metric → value
      category = ground_truth | model name
      prompt   = Complex | Simple   (only for models; GT duplicated)
    Returns per_cue, global_medians
    """
    sums   = defaultdict(lambda: defaultdict(float))
    counts = defaultdict(lambda: defaultdict(int))
    per_cue = {}

    for cue, pay in data.items():
        per_cue[cue] = {}

        # Human baseline (duplicate for both prompts)
        gt_metrics = compute_metrics(pay["ground_truth"], vad, conc)
        per_cue[cue]["ground_truth"] = gt_metrics
        for m, v in gt_metrics.items():
            if v is not None:
                sums["ground_truth"][m] += v
                counts["ground_truth"][m] += 1

        # Models
        for model, outs in pay["models"].items():
            per_cue[cue].setdefault(model, {})
            for prompt in ("Complex", "Simple"):
                mets = compute_metrics(outs.get(prompt, []), vad, conc)
                per_cue[cue][model][prompt] = mets
                for k, v in mets.items():
                    if v is not None:
                        key = f"{prompt}_{k}"
                        sums[model][key] += v
                        counts[model][key] += 1

    global_medians = {
        cat: {m: sums[cat][m] / counts[cat][m] for m in sums[cat]}
        for cat in sums
    }
    return per_cue, global_medians

# ─── COVERAGE HELPERS ────────────────────────────────────────────────────────
def build_word_lists(data):
    """
    Returns dict category → list of raw tokens (duplicates kept).
    category:
        ground_truth
        {model}|Complex
        {model}|Simple
    """
    wl = defaultdict(list)
    for pay in data.values():
        wl["ground_truth"].extend(pay["ground_truth"])
        for m, outs in pay["models"].items():
            for p in ("Complex", "Simple"):
                wl[f"{m}|{p}"].extend(outs.get(p, []))
    return wl


def type_coverage(wl, lex):
    return {
        cat: None if not raws else
             len({preprocess(w) for w in raws if preprocess(w) in lex})
             / len({preprocess(w) for w in raws}) * 100
        for cat, raws in wl.items()
    }


def token_coverage(wl, lex):
    return {
        cat: None if not raws else
             sum(preprocess(w) in lex for w in raws) / len(raws) * 100
        for cat, raws in wl.items()
    }

# ─── MAIN ────────────────────────────────────────────────────────────────────
def main():
    data = load_from_pickle(PICKLE_PATH)
    vad  = load_vad(VAD_PATH)
    conc = load_concreteness(CONC_PATH)

    breakpoint()

    per_cue, global_meds = aggregate(data, vad, conc)

    # 1. Global medians -------------------------------------------------------
    print("=== Global Median Valence / Arousal / Dominance / Concreteness ===\n")
    for cat, mets in global_meds.items():
        print(f"-- {cat} --")
        for name, val in sorted(mets.items()):
            print(f"   {name:16s}: {val:5.3f}")
        print()

    # 2. Coverage (%) ---------------------------------------------------------
    wl       = build_word_lists(data)
    vad_type = type_coverage(wl, set(vad))
    vad_tok  = token_coverage(wl, set(vad))
    conc_type = type_coverage(wl, set(conc))
    conc_tok  = token_coverage(wl, set(conc))

    print("=== Lexicon Coverage & Emotional / Concrete Token % ===\n")
    print(f"{'Category':40s}  {'VAD Cov':>8s} {'VAD Tok%':>8s}   {'Conc Cov':>8s} {'Conc Tok%':>8s}")
    for cat in sorted(wl):
        print(
            f"{cat:40s}  "
            f"{(f'{vad_type[cat]:5.1f}%'  if vad_type[cat]  is not None else '   N/A'):>8s} "
            f"{(f'{vad_tok[cat]:5.1f}%'   if vad_tok[cat]   is not None else '   N/A'):>8s}   "
            f"{(f'{conc_type[cat]:5.1f}%' if conc_type[cat] is not None else '   N/A'):>8s} "
            f"{(f'{conc_tok[cat]:5.1f}%'  if conc_tok[cat]  is not None else '   N/A'):>8s}"
        )
    print()

    # 3. % Concrete / % Abstract / % Unknown ----------------------------------
    print(f"=== %Concrete / %Abstract / %Unknown  (threshold ≥{THRESHOLD}) ===\n")
    print(f"{'Category':40s} {'%Conc':>6s} {'%Abs':>6s} {'%Unk':>6s}")
    for cat, raws in wl.items():
        c, a, u = compute_threshold_percents(raws, conc, THRESHOLD)
        print(
            f"{cat:40s} "
            f"{(f'{c:5.1f}%' if c is not None else '  N/A'):>6s} "
            f"{(f'{a:5.1f}%' if a is not None else '  N/A'):>6s} "
            f"{(f'{u:5.1f}%' if u is not None else '  N/A'):>6s}"
        )
    print()

    # 4. Dump per-cue metrics --------------------------------------------------
    with open("english_per_cue_metrics.json", "w", encoding="utf-8") as f:
        json.dump(per_cue, f, indent=2, ensure_ascii=False)
    print("Per-cue metrics written to english_per_cue_metrics.json\n")


if __name__ == "__main__":
    main()
