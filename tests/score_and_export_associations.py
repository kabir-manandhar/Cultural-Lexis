#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enrich SWOW associations with Valence, Arousal, and Concreteness (lean schema).

Output per list (GT / each model×prompt):
{
  "words": [...],
  "valence": [...],        # floats or null (per word)
  "arousal": [...],        # floats or null
  "concreteness": [...],   # floats or null
  "medians": { "valence": v, "arousal": a, "concreteness": c }  # nulls if no matches
}
"""

import argparse
import csv
import json
import re
import string
from statistics import median
from typing import Dict, List, Tuple, Optional

import pandas as pd
from nltk.stem import WordNetLemmatizer

# ── Default paths (from your earlier script) ──────────────────────────────────
PICKLE_PATH = "/data/gpfs/projects/punim2219/LM_with_SWOW/kabir/Data/swow_en_results_processed.pkl"
VAD_PATH    = "/data/gpfs/projects/punim2219/LM_with_SWOW/kabir/Data/VAD-emot-submit.csv"
CONC_PATH   = "/data/gpfs/projects/punim2219/LM_with_SWOW/kabir/Data/Concreteness_ratings_Brysbaert_et_al_BRM.txt"

# ── Text helpers ──────────────────────────────────────────────────────────────
_LEMM  = WordNetLemmatizer()
_PUNCS = string.punctuation + "“”‘’"
_SPLIT = re.compile(r"[\s\-/]+")

def preprocess(tok: str) -> str:
    t = tok.lower().strip(_PUNCS)
    try:
        return _LEMM.lemmatize(t)
    except Exception:
        return t

# ── Loaders ──────────────────────────────────────────────────────────────────
def load_from_pickle(path: str) -> dict:
    df   = pd.read_pickle(path)
    data = {}
    for cue, grp in df.groupby("Cue Word"):
        gt_raw = grp.iloc[0]["Ground Truth Associated Words"] or ""
        gt = [w.strip() for w in re.split(r"[，,]", gt_raw) if w.strip()][:10]

        models = {}
        for (m, p), sub in grp.groupby(["Model Type", "Prompt Type"]):
            words = []
            for cell in sub["Parsed Associated Words"].dropna():
                words.extend([w.strip() for w in cell.split(",") if w.strip()])
            models.setdefault(m, {})[p] = words[:10]

        data[cue] = {"ground_truth": gt, "models": models}
    return data

def load_vad(path: str) -> Dict[str, Tuple[float, float, float]]:
    vad = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            w = row["Word"].strip().lower()
            try:
                v = float(row["V.Mean.Sum"]); a = float(row["A.Mean.Sum"]); d = float(row["D.Mean.Sum"])
            except ValueError:
                continue
            vad[w] = (v, a, d)
    return vad

def load_concreteness(path: str) -> Dict[str, float]:
    conc = {}
    with open(path, encoding="utf-8") as f:
        next(f)  # header
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

# ── Scoring ───────────────────────────────────────────────────────────────────
def lookup_scores(term: str,
                  vad: Dict[str, Tuple[float, float, float]],
                  conc: Dict[str, float]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Return (valence, arousal, concreteness) for a term.
    Strategy:
      1) Exact lookup on the preprocessed whole term.
      2) If OOV and term has spaces/hyphens/slashes, split and take median over matched tokens.
    """
    t = preprocess(term)

    # Whole-term first
    v = a = c = None
    if t in vad or t in conc:
        if t in vad:
            v, a, _ = vad[t]
        if t in conc:
            c = conc[t]
        return v, a, c

    # Token median fallback
    toks = [preprocess(x) for x in _SPLIT.split(term) if x.strip()]
    if len(toks) > 1:
        v_list, a_list, c_list = [], [], []
        for tok in toks:
            if tok in vad:
                vv, aa, _ = vad[tok]
                v_list.append(vv); a_list.append(aa)
            if tok in conc:
                c_list.append(conc[tok])
        v = median(v_list) if v_list else None
        a = median(a_list) if a_list else None
        c = median(c_list) if c_list else None
        if v is not None or a is not None or c is not None:
            return v, a, c

    return None, None, None

def score_list(raws: List[str],
               vad: Dict[str, Tuple[float, float, float]],
               conc: Dict[str, float]) -> dict:
    """
    Produce parallel lists for words and their scores, plus medians.
    """
    words = list(raws)  # preserve order
    v_list, a_list, c_list = [], [], []

    for t in words:
        v, a, c = lookup_scores(t, vad, conc)
        v_list.append(v)
        a_list.append(a)
        c_list.append(c)

    # medians on non-None values
    mv = median([x for x in v_list if x is not None]) if any(x is not None for x in v_list) else None
    ma = median([x for x in a_list if x is not None]) if any(x is not None for x in a_list) else None
    mc = median([x for x in c_list if x is not None]) if any(x is not None for x in c_list) else None

    return {
        "words": words,
        "valence": v_list,
        "arousal": a_list,
        "concreteness": c_list,
        "medians": {"valence": mv, "arousal": ma, "concreteness": mc}
    }

# ── Orchestration ─────────────────────────────────────────────────────────────
def enrich(data: dict,
           vad: Dict[str, Tuple[float, float, float]],
           conc: Dict[str, float]) -> dict:
    """
    per cue:
      ground_truth: {words, valence, arousal, concreteness, medians}
      models: model → { "Complex": {...}, "Simple": {...} }
    """
    out = {}
    for cue, payload in data.items():
        cue_obj = {"ground_truth": score_list(payload.get("ground_truth", []), vad, conc), "models": {}}
        for model, outs in payload.get("models", {}).items():
            cue_obj["models"][model] = {
                "Complex": score_list(outs.get("Complex", []), vad, conc),
                "Simple" : score_list(outs.get("Simple",  []), vad, conc),
            }
        out[cue] = cue_obj
    return out

def extract_medians_only(enriched: dict) -> dict:
    """
    Compact JSON with medians only, per cue / category / prompt.
    """
    slim = {}
    for cue, block in enriched.items():
        slim[cue] = {"ground_truth": block["ground_truth"]["medians"], "models": {}}
        for model, mp in block["models"].items():
            slim[cue]["models"][model] = {p: mp[p]["medians"] for p in ("Complex", "Simple")}
    return slim

# ── CLI ──────────────────────────────────────────────────────────────────────
def parse_args():
    ap = argparse.ArgumentParser(description="Enrich associations with V/A/C and medians (lean schema).")
    ap.add_argument("--pickle", default=PICKLE_PATH, help="Path to processed SWOW pickle.")
    ap.add_argument("--vad",    default=VAD_PATH,    help="Path to VAD-emot-submit.csv")
    ap.add_argument("--conc",   default=CONC_PATH,   help="Path to Brysbaert concreteness txt")
    ap.add_argument("--out_full",   default="english_associations_scored.json", help="Full enriched JSON")
    ap.add_argument("--out_medians", default="english_per_cue_medians.json",    help="Medians-only JSON")
    return ap.parse_args()

def main():
    args = parse_args()
    data = load_from_pickle(args.pickle)
    vad  = load_vad(args.vad)
    conc = load_concreteness(args.conc)

    enriched = enrich(data, vad, conc)

    with open(args.out_full, "w", encoding="utf-8") as f:
        json.dump(enriched, f, indent=2, ensure_ascii=False)
    print(f"✔ Full enriched JSON → {args.out_full}")

    if args.out_medians:
        med_only = extract_medians_only(enriched)
        with open(args.out_medians, "w", encoding="utf-8") as f:
            json.dump(med_only, f, indent=2, ensure_ascii=False)
        print(f"✔ Medians-only JSON → {args.out_medians}")

if __name__ == "__main__":
    main()
