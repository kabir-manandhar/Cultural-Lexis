#!/usr/bin/env python3
"""
Create violin+box plots for English Valence, Arousal, Dominance & Concreteness
across each model × prompt (Complex / Simple).

Outputs
-------
• 14 HTML files (one per prompt × metric) in OUT_DIR
• per_cue_metrics_en.json  – cue-level medians used in hypothesis tests
"""

import csv, json, os, re, string
from collections import defaultdict
from statistics import median

import pandas as pd
import plotly.express as px
import plotly.colors as pc
from nltk.stem import WordNetLemmatizer

# ─── PATHS ────────────────────────────────────────────────────────────────
PICKLE_PATH = "/data/gpfs/projects/punim2219/LM_with_SWOW/kabir/Data/swow_en_results_processed.pkl"
VAD_PATH    = "/data/gpfs/projects/punim2219/LM_with_SWOW/kabir/Data/VAD-emot-submit.csv"
CONC_PATH   = "/data/gpfs/projects/punim2219/LM_with_SWOW/kabir/Data/Concreteness_ratings_Brysbaert_et_al_BRM.txt"
OUT_DIR     = "/data/projects/punim2219/LM_with_SWOW/kabir/Data/plots"

MODEL_MAP = {
    "Qwen/Qwen2.5-7B-Instruct"              : "qwen-vanilla",
    "meta-llama/Meta-Llama-3.1-8B-Instruct" : "llama-vanilla",
    "sukai/qwen_ppo_us"                     : "qwen-ppo",
    "sukai/llama_ppo_us"                    : "llama-ppo",
    "sukai/qwen_swow_us"                    : "qwen-sft",
    "sukai/llama_swow_us"                   : "llama-sft",
}

# ─── HELPERS ──────────────────────────────────────────────────────────────
_LEM   = WordNetLemmatizer()
_PUNCS = string.punctuation + "“”‘’"

def preprocess(tok: str) -> str:
    return _LEM.lemmatize(tok.lower().strip(_PUNCS))

# ─── DATA LOADERS ─────────────────────────────────────────────────────────
def load_from_pickle(path: str):
    df = pd.read_pickle(path)
    data = {}
    for cue, grp in df.groupby("Cue Word"):
        gt_raw = grp.iloc[0]["Ground Truth Associated Words"] or ""
        gt = [w.strip() for w in gt_raw.split(",") if w.strip()][:10]

        models = defaultdict(dict)
        for (m, p), sub in grp.groupby(["Model Type", "Prompt Type"]):
            words = []
            for cell in sub["Parsed Associated Words"].dropna():
                words.extend([w.strip() for w in cell.split(",") if w.strip()])
            models[m][p] = words[:10]

        data[cue] = {"ground_truth": gt, "models": dict(models)}
    return data


def load_vad(path: str):
    vad = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                vad[row["Word"].lower()] = (
                    float(row["V.Mean.Sum"]),
                    float(row["A.Mean.Sum"]),
                    float(row["D.Mean.Sum"]),
                )
            except ValueError:
                continue
    return vad


def load_conc(path: str):
    conc = {}
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            try:
                conc[parts[0].lower()] = float(parts[2])
            except ValueError:
                continue
    return conc

# ─── METRICS ─────────────────────────────────────────────────────────────
def compute_metrics(tokens, vad, conc):
    vs, as_, ds, cs = [], [], [], []
    for raw in tokens:
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

def aggregate(data, vad, conc):
    per_cue = {}
    for cue, payload in data.items():
        per_cue[cue] = {}

        base = compute_metrics(payload["ground_truth"], vad, conc)
        per_cue[cue]["ground_truth"] = {"Complex": base, "Simple": base}

        for m, prm in payload["models"].items():
            per_cue[cue][m] = {p: compute_metrics(w, vad, conc)
                               for p, w in prm.items()}
    return per_cue

def flatten(per_cue):
    rows = []
    for cue, models in per_cue.items():
        for long, prm_dict in models.items():
            short = "human" if long == "ground_truth" else MODEL_MAP.get(long)
            if short is None:
                continue
            for prompt, mets in prm_dict.items():
                for metric, val in mets.items():
                    if val is None:
                        continue
                    rows.append(
                        dict(Cue=cue, Model=short, Prompt=prompt,
                             Metric=metric, Value=val)
                    )
    return pd.DataFrame(rows)

# ─── PLOTTING ────────────────────────────────────────────────────────────
def darker(col, factor=0.55):
    if col.startswith("#"):
        r, g, b = pc.hex_to_rgb(col)
    else:
        r, g, b = map(int, re.findall(r"\d+", col))
    r, g, b = [max(0, int(x * factor)) for x in (r, g, b)]
    return f"rgb({r},{g},{b})"

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    data = load_from_pickle(PICKLE_PATH)
    vad  = load_vad(VAD_PATH)
    conc = load_conc(CONC_PATH)

    per_cue = aggregate(data, vad, conc)
    df      = flatten(per_cue)

    order = ["human", "qwen-vanilla", "llama-vanilla",
             "qwen-ppo", "llama-ppo", "qwen-sft", "llama-sft"]
    metrics = ["valence", "arousal", "dominance", "concreteness"]

    for prompt in ("Complex", "Simple"):
        sub_prompt = df.query("Prompt == @prompt")
        for metric in metrics:
            sub = sub_prompt.query("Metric == @metric")

            fig = px.violin(
                sub, x="Model", y="Value", color="Model",
                category_orders={"Model": order},
                box=True, points=False,
                color_discrete_sequence=px.colors.qualitative.Plotly,
                title=f"{prompt} prompt – {metric.title()} (EN)",
            )

            for tr in fig.data:
                if tr.type != "violin":
                    continue
                base = tr.line.color or tr.marker.color or tr.fillcolor
                tr.update(
                    line_width=1.2,
                    meanline_visible=False,
                    box_visible=True,
                    box_fillcolor="rgba(0,0,0,0)",
                    box_line_width=4,
                    box_line_color=darker(base),
                )

            fig.update_layout(
                width=1920, height=1080,
                plot_bgcolor="white", showlegend=False,
                font=dict(size=32),
                xaxis=dict(title="",tickfont=dict(size=30), gridcolor="#E5E5E5"),
                yaxis=dict(title=metric.title(), tickfont=dict(size=26),
                           gridcolor="#E5E5E5", dtick=1),
            )

            fname = f"violin_{prompt.lower()}_{metric}_en.html"
            fig.write_html(os.path.join(OUT_DIR, fname))
            print("Saved:", fname)

    with open(os.path.join(OUT_DIR, "per_cue_metrics_en.json"), "w", encoding="utf-8") as f:
        json.dump(per_cue, f, indent=2, ensure_ascii=False)
    print("Per-cue metrics written to per_cue_metrics_en.json")

if __name__ == "__main__":
    main()
