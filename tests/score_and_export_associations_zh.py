#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enrich SWOW (Mandarin) associations with Valence (1–9), Arousal (1–9), and Concreteness (1–5, higher=more concrete).

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
import json
import re
import string
from statistics import median
from typing import Dict, List, Optional, Tuple

import pandas as pd

# ── Default paths (from your ZH script) ───────────────────────────────────────
PICKLE_PATH  = "/data/gpfs/projects/punim2219/LM_with_SWOW/kabir/Data/swow_zh_results_processed_ppo.pkl"
CH_VAD_PATH  = "/data/gpfs/projects/punim2219/LM_with_SWOW/kabir/Data/VAD-zh-emot-submit.csv"
CH_CONC_PATH = "/data/gpfs/projects/punim2219/LM_with_SWOW/kabir/Data/Concretness-ZH.xlsx"

# ── Rescaling helpers ─────────────────────────────────────────────────────────
def rescale_valence(v_zh: float) -> float:
    """Chinese valence −3…+3 → English 1…9."""
    return (v_zh + 3.0) / 6.0 * 8.0 + 1.0

def rescale_arousal(a_zh: float) -> float:
    """Chinese arousal 0…4 → English 1…9."""
    return a_zh / 4.0 * 8.0 + 1.0

def invert_concreteness(c_abs: float) -> float:
    """ZH conc: 1=concrete, 5=abstract → invert to 1=abstract, 5=concrete."""
    return 6.0 - c_abs

# ── Text helpers (lightweight for ZH) ─────────────────────────────────────────
# Keep Chinese punctuation + ASCII punctuation to strip at edges
_PUNCS = string.punctuation + "“”‘’，。？！；：（）「」『』—…、《》【】"
_SPLIT = re.compile(r"[\s\-/]+")

def preprocess(tok: str) -> str:
    return tok.strip(_PUNCS)

# ── Loaders ──────────────────────────────────────────────────────────────────
def load_from_pickle(path: str) -> dict:
    """
    Returns: cue → {ground_truth: [w], models: {model: {prompt: [w]}}}
    Keeps top-10 items per list as in your current code.
    """
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

def load_chinese_vad(path: str) -> Dict[str, Tuple[float, float]]:
    """
    Load ZH VAD CSV with columns: Word, Valence_Mean (−3..+3), Arousal_Mean (0..4).
    Return word → (valence_rescaled_1to9, arousal_rescaled_1to9).
    """
    df = pd.read_csv(path)
    vad = {}
    for _, row in df.iterrows():
        w = str(row.get("Word", "")).strip()
        if not w:
            continue
        try:
            v_res = rescale_valence(float(row["Valence_Mean"]))
            a_res = rescale_arousal(float(row["Arousal_Mean"]))
        except Exception:
            continue
        vad[w] = (v_res, a_res)
    return vad

def load_chinese_concreteness(path: str) -> Dict[str, float]:
    """
    Load ZH concreteness Excel where mean column is 1=concrete .. 5=abstract.
    Return word → concreteness (1=abstract .. 5=concrete).
    """
    df = pd.read_excel(path)
    # Find the mean column programmatically (matches your previous approach)
    mean_col = next(c for c in df.columns if "Mean" in c and "Word" not in c)
    conc = {}
    for _, r in df.iterrows():
        w = str(r["Word"]).strip()
        if not w:
            continue
        try:
            conc[w] = invert_concreteness(float(r[mean_col]))
        except Exception:
            continue
    return conc

# ── Scoring ───────────────────────────────────────────────────────────────────
def lookup_scores(term: str,
                  vad: Dict[str, Tuple[float, float]],
                  conc: Dict[str, float]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Return (valence_1to9, arousal_1to9, concreteness_1to5) for a term.
    Strategy:
      1) Exact lookup on the preprocessed whole term.
      2) If OOV and term has spaces/hyphens/slashes, split and take median over matched tokens.
    """
    t = preprocess(term)

    v = a = c = None
    if t in vad or t in conc:
        if t in vad:
            v, a = vad[t]
        if t in conc:
            c = conc[t]
        return v, a, c

    toks = [preprocess(x) for x in _SPLIT.split(term) if x.strip()]
    if len(toks) > 1:
        v_list, a_list, c_list = [], [], []
        for tok in toks:
            if tok in vad:
                vv, aa = vad[tok]
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
               vad: Dict[str, Tuple[float, float]],
               conc: Dict[str, float]) -> dict:
    """
    Produce parallel lists for words and their scores, plus medians.
    """
    words = list(raws)  # preserve original order (top-10 already enforced)
    v_list, a_list, c_list = [], [], []

    for t in words:
        v, a, c = lookup_scores(t, vad, conc)
        v_list.append(v)
        a_list.append(a)
        c_list.append(c)

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
           vad: Dict[str, Tuple[float, float]],
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
    ap = argparse.ArgumentParser(description="Enrich Mandarin associations with V/A/C and medians (lean schema).")
    ap.add_argument("--pickle", default=PICKLE_PATH, help="Path to processed SWOW ZH pickle.")
    ap.add_argument("--vad",    default=CH_VAD_PATH, help="Path to VAD-zh-emot-submit.csv")
    ap.add_argument("--conc",   default=CH_CONC_PATH, help="Path to Concretness-ZH.xlsx")
    ap.add_argument("--out_full",   default="mandarin_associations_scored.json", help="Full enriched JSON")
    ap.add_argument("--out_medians", default="mandarin_per_cue_medians.json",    help="Medians-only JSON")
    return ap.parse_args()

def main():
    args = parse_args()
    data = load_from_pickle(args.pickle)
    vad  = load_chinese_vad(args.vad)
    conc = load_chinese_concreteness(args.conc)

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
