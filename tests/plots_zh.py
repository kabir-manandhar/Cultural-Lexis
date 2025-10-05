#!/usr/bin/env python3
"""
Create violin+box plots for Mandarin (ZH) Valence, Arousal and Concreteness
across each model × prompt (Complex / Simple).

Differences from the English version
-----------------------------------
* Dominance is **omitted** (no reliable ZH lexicon), so only three metrics.
* Chinese VAD and Concreteness lexicons use different scales – we rescale
  Valence & Arousal to 1‑9 and invert Concreteness to English‑style 1 (abstract)
  → 5 (concrete) before plotting.
* Output filenames carry a “_zh” suffix, e.g. `violin_complex_valence_zh.html`.
"""

import csv, json, os, re, string
from collections import defaultdict
from statistics import median

import pandas as pd
import plotly.express as px
import plotly.colors as pc
import re
import plotly.io as pio          # <-- for static image export


# ─── PATHS ────────────────────────────────────────────────────────────────
PICKLE_PATH  = "/data/gpfs/projects/punim2219/LM_with_SWOW/kabir/Data/swow_zh_results_processed.pkl"
ZH_VAD_PATH  = "/data/gpfs/projects/punim2219/LM_with_SWOW/kabir/Data/VAD-zh-emot-submit.csv"
ZH_CONC_PATH = "/data/gpfs/projects/punim2219/LM_with_SWOW/kabir/Data/Concretness-ZH.xlsx"
OUT_DIR      = "/data/projects/punim2219/LM_with_SWOW/kabir/Data/plots"

MODEL_MAP = {
    "Qwen/Qwen2.5-7B-Instruct":              "qwen-vanilla",
    "meta-llama/Meta-Llama-3.1-8B-Instruct": "llama-vanilla",
    "sukai/qwen_ppo_zh":                     "qwen-ppo",
    "sukai/llama_ppo_zh":                    "llama-ppo",
    "sukai/qwen_swow_zh":                    "qwen-sft",
    "sukai/llama_swow_zh":                   "llama-sft",
}

# ─── RESCALING FUNCTIONS ────────────────────────────────────────────────

def rescale_valence(v_ch: float) -> float:
    """Chinese −3…+3 → English‑style 1…9"""
    return (v_ch + 3.0) / 6.0 * 8.0 + 1.0


def rescale_arousal(a_ch: float) -> float:
    """Chinese 0…4 (or 1…5) → 1…9"""
    return a_ch / 4.0 * 8.0 + 1.0  # maps 0→1, 4→9


def invert_concreteness(c_abs: float) -> float:
    """ZH 1 (concrete) → 5 (abstract) becomes EN 1 (abstract) → 5 (concrete)"""
    return 6.0 - c_abs

# ─── I/O HELPERS ─────────────────────────────────────────────────────────
_PUNCS = string.punctuation + "“”‘’，。？！；：（）「」『』—…"

def clean(tok: str) -> str:
    return tok.strip(_PUNCS)


def load_from_pickle(path: str):
    """Return dict cue → {ground_truth, models} identical to EN loader."""
    df = pd.read_pickle(path)
    data = {}
    for cue, grp in df.groupby("Cue Word"):
        gt_raw = grp.iloc[0]["Ground Truth Associated Words"] or ""
        gt = [w.strip() for w in re.split(r"[，,]", gt_raw) if w.strip()][:10]

        models: dict[str, dict[str, list[str]]] = defaultdict(dict)
        for (model, prompt), sub in grp.groupby(["Model Type", "Prompt Type"]):
            words: list[str] = []
            for cell in sub["Parsed Associated Words"].dropna():
                words.extend([w.strip() for w in cell.split(",") if w.strip()])
            models[model][prompt] = words[:10]

        data[cue] = {"ground_truth": gt, "models": dict(models)}
    return data


def load_zh_vad(path: str):
    vad = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            w = row.get("Word", "").strip()
            try:
                v_raw = float(row["Valence_Mean"])
                a_raw = float(row["Arousal_Mean"])
            except (KeyError, ValueError):
                continue
            if w:
                vad[w] = (rescale_valence(v_raw), rescale_arousal(a_raw))
    return vad


def load_zh_conc(path: str):
    df = pd.read_excel(path)
    mean_col = next(c for c in df.columns if "Mean" in c and "Word" not in c)
    conc = {}
    for _, row in df.iterrows():
        w = str(row["Word"]).strip()
        if not w:
            continue
        try:
            conc[w] = invert_concreteness(float(row[mean_col]))
        except ValueError:
            continue
    return conc

# ─── METRIC COMPUTATION ────────────────────────────────────────────────

def compute_metrics(tokens: list[str], vad: dict, conc: dict):
    vs, as_, cs = [], [], []
    for raw in tokens:
        w = clean(raw)
        if w in vad:
            v, a = vad[w]
            vs.append(v); as_.append(a)
        if w in conc:
            cs.append(conc[w])
    return {
        "valence":      median(vs)   if vs else None,
        "arousal":      median(as_)  if as_ else None,
        "concreteness": median(cs)   if cs else None,
    }

# ─── AGGREGATION & FLATTENING ───────────────────────────────────────────

def aggregate(data: dict, vad: dict, conc: dict):
    per_cue = {}
    for cue, payload in data.items():
        per_cue[cue] = {}

        base = compute_metrics(payload["ground_truth"], vad, conc)
        per_cue[cue]["ground_truth"] = {"Complex": base, "Simple": base}

        for model, prm in payload["models"].items():
            per_cue[cue][model] = {p: compute_metrics(w, vad, conc) for p, w in prm.items()}
    return per_cue


def flatten(per_cue: dict):
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
                    rows.append({
                        "Cue": cue,
                        "Model": short,
                        "Prompt": prompt,
                        "Metric": metric,
                        "Value": val,
                    })
    return pd.DataFrame(rows)

# ─── MAIN ───────────────────────────────────────────────────────────────

def darker(c: str, factor: float = 0.55) -> str:
    """
    Return a darker version of a Plotly colour.
    `factor` < 1.0 darkens; e.g. 0.55 → 55 % of original brightness.
    Accepts hex (“#1f77b4”) or “rgb(31,119,180)”.
    """
    if c.startswith("#"):                          # hex → rgb tuple
        r, g, b = pc.hex_to_rgb(c)
    else:                                          # rgb(...)  → tuple
        r, g, b = map(int, re.findall(r"\d+", c))
    r, g, b = [max(0, int(x * factor)) for x in (r, g, b)]
    return f"rgb({r},{g},{b})"

def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    data     = load_from_pickle(PICKLE_PATH)
    zh_vad   = load_zh_vad(ZH_VAD_PATH)
    zh_conc  = load_zh_conc(ZH_CONC_PATH)
    per_cue  = aggregate(data, zh_vad, zh_conc)
    df       = flatten(per_cue)

    model_order = [
        "human", "qwen-vanilla", "llama-vanilla",
        "qwen-ppo", "llama-ppo", "qwen-sft", "llama-sft",
    ]
    metrics = ["valence", "arousal", "concreteness"]   # no dominance

    for prompt in ("Complex", "Simple"):
        df_p = df.query("Prompt == @prompt")
        for metric in metrics:
            sub = df_p.query("Metric == @metric")

            fig = px.violin(
                sub,
                x="Model",
                y="Value",
                color="Model",
                category_orders={"Model": model_order},
                box=True,          # inner box-plot (median & IQR)
                points=False,      # hide individual cue points
                color_discrete_sequence=px.colors.qualitative.Plotly,
                title=f"{prompt} prompt – {metric.title()} (ZH)",
            )

            for tr in fig.data:
                if tr.type != "violin":
                    continue
                base = tr.line.color or tr.marker.color or tr.fillcolor
                tr.update(
                    line_width=1.2,
                    meanline_visible=False,
                    box_visible=True,
                    box_fillcolor="rgba(0,0,0,0)",         # transparent box
                    box_line_width=4,
                    box_line_color=darker(base, 0.55),      # darker shade
                )

            # -------- overall layout & typography ------------
            fig.update_layout(
                width=1920, height=1080,
                plot_bgcolor="white", showlegend=False,
                font=dict(size=45),             # default text size
                title_font_size=30,

                xaxis=dict(
                    title="",
                    title_font_size=50,
                    tickfont=dict(size=32),
                    gridcolor="#E5E5E5",
                ),
                yaxis=dict(
                    title=metric.title(),
                    title_font_size=30,
                    tickfont=dict(size=24),
                    gridcolor="#E5E5E5",
                    dtick=1,                  # 0.5-unit ticks/gridlines
                ),
            )

            html_name = f"violin_{prompt.lower()}_{metric}_zh.html"
            fig.write_html(os.path.join(OUT_DIR, html_name))
            print("Saved:", html_name)

    # optionally keep per-cue medians
    with open(os.path.join(OUT_DIR, "per_cue_metrics_zh.json"), "w", encoding="utf-8") as f:
        json.dump(per_cue, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()