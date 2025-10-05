#!/usr/bin/env python3
"""
Cross-cultural “tension-set” pipeline  –  HYBRID DISTANCE
=========================================================
Hybrid distance  = 0.5·JSD  +  0.5·(EMD / (bins−1))
(See docstring of generate_tension_csv).

Outputs
-------
* tension_set_top50.csv                 – global ranking (US vs CN)
* <model>_top50.json                    – model answers + all metrics
* <model>_hybrid_bias_top50.pdf         – scatter plot with question IDs
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance

# ---------------------------------------------------------------------------
# Paths & configuration
# ---------------------------------------------------------------------------
ROOT_READ  = Path("/data/gpfs/projects/punim2219/LM_with_SWOW")
SURVEY_JSON = ROOT_READ / "kabir/Data/WV_Bench/question_answer.json"

ROOT_WRITE  = Path("/data/projects/punim2219/LM_with_SWOW/kabir/Data/output")
PLOT_DIR    = ROOT_WRITE / "cross_country"
TENSION_CSV = ROOT_WRITE / "tension_set_top50.csv"
K_TOP       = 50

ROOT_WRITE.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Model-output paths  (edit if needed)
# ---------------------------------------------------------------------------
OUTPUT_FILES: Dict[str, str] = {
    "WVS_US_llama_vanilla":  "/data/gpfs/projects/punim2219/LM_with_SWOW/chunhua/Data/output/llama/WVS_US_llama_vanilla.json",
    "WVS_US_llama_sft":      "/data/gpfs/projects/punim2219/LM_with_SWOW/chunhua/Data/output/llama/WVS_US_llama_us_sft.json",
    "WVS_US_qwen_vanilla":   "/data/gpfs/projects/punim2219/LM_with_SWOW/chunhua/Data/output/qwen/WVS_US_qwen_vanilla.json",
    "WVS_US_qwen_sft":       "/data/gpfs/projects/punim2219/LM_with_SWOW/chunhua/Data/output/qwen/WVS_US_qwen_us_sft.json",
    "WVS_ZH_llama_vanilla":  "/data/gpfs/projects/punim2219/LM_with_SWOW/chunhua/Data/output/llama/WVS_ZH_llama_vanilla.json",
    "WVS_ZH_llama_sft":      "/data/gpfs/projects/punim2219/LM_with_SWOW/chunhua/Data/output/llama/WVS_ZH_llama_sft.json",
    "WVS_ZH_qwen_vanilla":   "/data/gpfs/projects/punim2219/LM_with_SWOW/chunhua/Data/output/qwen/WVS_ZH_qwen_vanilla.json",
    "WVS_ZH_qwen_sft":       "/data/gpfs/projects/punim2219/LM_with_SWOW/chunhua/Data/output/qwen/WVS_ZH_qwen_sft.json",
}

# ---------------------------------------------------------------------------
# Distance helpers
# ---------------------------------------------------------------------------
try:
    from evaluate import get_jensen_shannon_distance as _js, get_earth_mover_distance as _emd   # type: ignore
    def js_distance(p: List[float], q: List[float]) -> float: return _js (p, q)
    def em_distance(p: List[float], q: List[float]) -> float: return _emd(p, q)
except ImportError:
    def js_distance(p, q): return float(jensenshannon(np.asarray(p), np.asarray(q), base=2))
    def em_distance(p, q): return float(wasserstein_distance(p, q))

def emd_norm(p, q):                        # scaled to [0,1]
    bins = len(p)
    return em_distance(p, q) / (bins-1) if bins > 1 else 0.0
def combo_distance(p, q):
    return 0.5*js_distance(p, q) + 0.5*emd_norm(p, q)

# ---------------------------------------------------------------------------
# Phase 1 – global tension-set
# ---------------------------------------------------------------------------
def generate_tension_csv(path=TENSION_CSV, k=K_TOP):
    df = pd.read_json(SURVEY_JSON)

    def vec(row, key): return [v for _, v in sorted(row[key].items())]
    df["us_vec"] = df.apply(lambda r: vec(r, "us_score"),    axis=1)
    df["cn_vec"] = df.apply(lambda r: vec(r, "china_score"), axis=1)
    df = df[df.us_vec.str.len()==df.cn_vec.str.len()].reset_index(drop=True)

    df["js_us_cn"]  = df.apply(lambda r: js_distance (r.us_vec, r.cn_vec), axis=1)
    df["emd_us_cn"] = df.apply(lambda r: emd_norm    (r.us_vec, r.cn_vec), axis=1)
    df["combo"]     = 0.5*df.js_us_cn + 0.5*df.emd_us_cn

    cols = ["Id","question","question_instruction","choices",
            "china_score","us_score","js_us_cn","emd_us_cn","combo"]
    top = (df.sort_values("combo", ascending=False)
             .head(k)[cols].reset_index(drop=True))
    top.to_csv(path, index=False)
    return top

# ---------------------------------------------------------------------------
# Phase 2 – per-model analysis
# ---------------------------------------------------------------------------
def prepare_dataframe(model_json: Path) -> pd.DataFrame:
    df_model = pd.read_json(model_json)
    df_gold  = pd.read_json(SURVEY_JSON)

    df = df_gold.copy()
    df["model_vec"] = df_model["choice_values"]
    ok = (df.model_vec.str.len()==df.us_score.apply(len)) & \
         (df.model_vec.str.len()==df.china_score.apply(len))
    return df[ok].reset_index(drop=True)

def add_metrics(df):
    df["us_vec"] = df.us_score.apply(lambda d: list(d.values()))
    df["cn_vec"] = df.china_score.apply(lambda d: list(d.values()))

    df["combo_us"] = df.apply(lambda r: combo_distance(r.model_vec, r.us_vec), axis=1)
    df["combo_cn"] = df.apply(lambda r: combo_distance(r.model_vec, r.cn_vec), axis=1)
    df["bias"]     = df.combo_cn - df.combo_us           # + = nearer US, – = nearer CN
    return df

def plot_scatter(df_top, model_name):
    fig, ax = plt.subplots(figsize=(6,6))

    sc = ax.scatter(df_top.combo_us, df_top.combo_cn,
                    c=df_top.bias, cmap="coolwarm",
                    edgecolor="black", s=45)

    # annotate each point with the WVS Id
    for _, r in df_top.iterrows():
        ax.text(r.combo_us+0.006, r.combo_cn+0.006, r.Id,
                fontsize=5, alpha=0.6)

    ax.plot([0,1],[0,1],ls="--",c="gray",alpha=.5)
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.set_xlabel("Hybrid distance  (model ↔ US)")
    ax.set_ylabel("Hybrid distance  (model ↔ CN)")
    ax.set_title(f"{model_name}   ·   top-{K_TOP} divergent questions")

    ax.text(0.05,0.95,f"Closer to CN: {(df_top.combo_us>df_top.combo_cn).sum()}",
            transform=ax.transAxes)
    ax.text(0.05,0.90,f"Closer to US: {(df_top.combo_us<df_top.combo_cn).sum()}",
            transform=ax.transAxes)

    fig.colorbar(sc, ax=ax, label="Bias (CN – US)")
    fig.tight_layout()

    pdf_name = f"{model_name}_hybrid_bias_top{K_TOP}.pdf"
    fig.savefig(PLOT_DIR/pdf_name)
    plt.close(fig)
    return pdf_name

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # always rebuild
    if TENSION_CSV.exists(): TENSION_CSV.unlink()
    tension_set = generate_tension_csv()
    print(f"✔ tension_set_top50.csv written ({len(tension_set)} items)")

    tension_ids = set(tension_set.Id)

    for model_name, model_path in OUTPUT_FILES.items():
        mp = Path(model_path)
        if not mp.exists():
            print(f"⚠  {model_name}: file not found – skipped")
            continue

        df = add_metrics(prepare_dataframe(mp))
        df_top = df[df.Id.isin(tension_ids)].copy()

        json_name = f"{model_name}_top{K_TOP}.json"
        df_top.to_json(ROOT_WRITE/json_name, orient="records", indent=2)
        pdf_name  = plot_scatter(df_top, model_name)

        print(f"✔ {model_name}: {json_name}, {pdf_name}")
