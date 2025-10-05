#!/usr/bin/env python3
"""
Mandarin emotion–concreteness statistics, parallel to the English script,
with 1–9 rescaling for Valence/Arousal and %Conc/%Abs/%Unk.
"""
import re
import json
import string
import csv
import pandas as pd
from collections import defaultdict
from statistics import median
from nltk.stem import WordNetLemmatizer

# ─── PATHS & PARAMS ───────────────────────────────────────────────────────────
PICKLE_PATH  = "/data/gpfs/projects/punim2219/LM_with_SWOW/kabir/Data/swow_zh_results_processed_ppo.pkl"
CH_VAD_PATH  = "/data/gpfs/projects/punim2219/LM_with_SWOW/kabir/Data/VAD-zh-emot-submit.csv"
CH_CONC_PATH = "/data/gpfs/projects/punim2219/LM_with_SWOW/kabir/Data/Concretness-ZH.xlsx"
THRESHOLD    = 3.0  # concreteness cutoff on English‐scale 1–5

# ─── RESCALING ────────────────────────────────────────────────────────────────

def rescale_valence(v_zh: float) -> float:
    """Chinese valence −3…+3  → English 1…9"""
    return (v_zh + 3.0) / 6.0 * 8.0 + 1.0


def rescale_arousal(a_zh: float) -> float:
    """Chinese arousal 0…4  → English 1…9 (source lexicon uses 0‑4)."""
    return a_zh / 4.0 * 8.0 + 1.0


def invert_concreteness(c_abs: float) -> float:
    """Chinese concreteness lexicon: 1 = concrete, 5 = abstract.
    Invert so 1 = abstract, 5 = concrete (English orientation)."""
    return 6.0 - c_abs

# ─── PRE‑PROCESSING ───────────────────────────────────────────────────────────
_PUNCS = string.punctuation + "“”‘’，。？！；：（）「」『』—…"

def preprocess(tok: str) -> str:
    return tok.strip(_PUNCS)

# ─── DATA LOADERS ─────────────────────────────────────────────────────────────

def load_from_pickle(path: str) -> dict:
    """Return nested dict: cue → {ground_truth, models}"""
    df = pd.read_pickle(path)
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


def load_chinese_vad(path: str):
    """Return dict word → (valence, arousal) on 1‑9 scale and print ranges."""
    df = pd.read_csv(path)
    raw_v = df["Valence_Mean"].astype(float)
    raw_a = df["Arousal_Mean"].astype(float)
    print("Loaded Chinese VAD: raw valence min/max", raw_v.min(), raw_v.max())
    print("Loaded Chinese VAD: raw arousal  min/max", raw_a.min(), raw_a.max())

    vad = {}
    for _, row in df.iterrows():
        w = str(row.get("Word", "")).strip()
        if not w:
            continue
        try:
            v_res = rescale_valence(float(row["Valence_Mean"]))
            a_res = rescale_arousal(float(row["Arousal_Mean"]))
        except ValueError:
            continue
        vad[w] = (v_res, a_res)

    vals = [v for v, _ in vad.values()]
    aros = [a for _, a in vad.values()]
    print("Rescaled valence min/max", min(vals), max(vals))
    print("Rescaled arousal  min/max", min(aros), max(aros))
    print()
    return vad


def load_chinese_concreteness(path: str):
    """Return dict word → concreteness (English orientation) and print ranges."""
    df = pd.read_excel(path)
    mean_col = next(c for c in df.columns if "Mean" in c and "Word" not in c)
    raw_vals = df[mean_col].astype(float)
    print("Loaded Chinese concreteness: raw mean min/max", raw_vals.min(), raw_vals.max())

    conc = {}
    for _, r in df.iterrows():
        w = str(r["Word"]).strip()
        if not w:
            continue
        try:
            conc[w] = invert_concreteness(float(r[mean_col]))
        except ValueError:
            continue

    c_vals = list(conc.values())
    print("Inverted concreteness min/max", min(c_vals), max(c_vals))
    print()
    return conc

# ─── METRIC HELPERS ──────────────────────────────────────────────────────────

def compute_metrics(raws, vad, conc):
    vs, as_, cs = [], [], []
    for raw in raws:
        w = preprocess(raw)
        if w in vad:
            v, a = vad[w]
            vs.append(v); as_.append(a)
        if w in conc:
            cs.append(conc[w])
    return {
        "valence"     : median(vs)  if vs else None,
        "arousal"     : median(as_) if as_ else None,
        "concreteness": median(cs)  if cs else None,
    }


def compute_threshold_percents(raws, conc, thr):
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
    sums, cnts, per_cue = defaultdict(lambda: defaultdict(float)), defaultdict(lambda: defaultdict(int)), {}
    for cue, pay in data.items():
        per_cue[cue] = {}
        # ground truth
        gt = compute_metrics(pay["ground_truth"], vad, conc)
        per_cue[cue]["ground_truth"] = gt
        for k, v in gt.items():
            if v is not None:
                sums["ground_truth"][k] += v
                cnts["ground_truth"][k] += 1
        # models
        for model, outs in pay["models"].items():
            per_cue[cue].setdefault(model, {})
            for p in ("Complex","Simple"):
                m = compute_metrics(outs.get(p, []), vad, conc)
                per_cue[cue][model][p] = m
                for k, v in m.items():
                    if v is not None:
                        sums[model][f"{p}_{k}"] += v
                        cnts[model][f"{p}_{k}"] += 1
    global_avgs = {
        cat: {m: sums[cat][m] / cnts[cat][m] for m in sums[cat]}
        for cat in sums
    }
    return per_cue, global_avgs

# ─── COVERAGE & PERCENTAGES ─────────────────────────────────────────────────
def build_word_lists(data):
    wl = defaultdict(list)
    for pay in data.values():
        wl["ground_truth"].extend(pay["ground_truth"])
        for m, outs in pay["models"].items():
            for p in ("Complex","Simple"):
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
    vad  = load_chinese_vad(CH_VAD_PATH)
    conc = load_chinese_concreteness(CH_CONC_PATH)

    breakpoint()

    per_cue, global_avgs = aggregate(data, vad, conc)

    # 1. Global medians
    print("=== Global median Valence/Arousal & Concreteness by Category ===\n")
    for cat, mets in global_avgs.items():
        print(f"-- {cat} --")
        for name, val in mets.items():
            if val is not None:
                print(f"   {name:12s}: {val:5.3f}")
            else:
                print(f"   {name:12s}:   None")
        print()

    # 2. Coverage & token %
    wl      = build_word_lists(data)
    v_type  = type_coverage(wl, set(vad))
    v_token = token_coverage(wl, set(vad))
    c_type  = type_coverage(wl, set(conc))
    c_token = token_coverage(wl, set(conc))

    print("=== Coverage & Emotional/Concrete Response % ===\n")
    print(f"{'Category':40s}  {'VAD Cov':>8s} {'Emo %':>8s}   {'Conc Cov':>8s} {'Conc %':>8s}")
    for cat in sorted(wl):
        print(
            f"{cat:40s}  "
            f"{(f'{v_type[cat]:5.1f}%' if v_type[cat] is not None else '   N/A'):>8s} "
            f"{(f'{v_token[cat]:5.1f}%' if v_token[cat] is not None else '   N/A'):>8s}   "
            f"{(f'{c_type[cat]:5.1f}%' if c_type[cat] is not None else '   N/A'):>8s} "
            f"{(f'{c_token[cat]:5.1f}%' if c_token[cat] is not None else '   N/A'):>8s}"
        )
    print()

    # 3. % Conc / % Abs / % Unk
    print(f"=== % Conc / % Abs / % Unk (@ threshold {THRESHOLD}) ===\n")
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

    # dump per-cue detail if desired
    with open("mandarin_per_cue_metrics.json","w",encoding="utf-8") as out:
        json.dump(per_cue, out, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
