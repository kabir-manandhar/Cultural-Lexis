#!/usr/bin/env python3
"""
Hybrid-distance tension-set analysis – EN composite plots
=========================================================
Produces:

• tension_set_top50.csv                (global US-CN gap)
• <model>_top50.json                   (per-model metrics)
• EN_qwen_vanilla_vs_sft_shift.png     (vanilla ↔ SFT, Qwen)
• EN_llama_vanilla_vs_sft_shift.png    (vanilla ↔ SFT, Llama)
"""

from __future__ import annotations
import json, math
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance

# ─────────────────────────────────────────────────────────────────────────────
# Paths & configuration
# ─────────────────────────────────────────────────────────────────────────────
ROOT_READ   = Path("/data/gpfs/projects/punim2219/LM_with_SWOW")
SURVEY_JSON = ROOT_READ / "kabir/Data/WV_Bench/question_answer.json"

ROOT_WRITE  = Path("/data/projects/punim2219/LM_with_SWOW/kabir/Data/output")
PLOT_DIR    = ROOT_WRITE / "cross_country"
TENSION_CSV = ROOT_WRITE / "tension_set_top50.csv"
K_TOP       = 50

ROOT_WRITE.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Update this block for EN model outputs
OUTPUT_FILES_EN: Dict[str,str] = {
    "WVS_EN_qwen_vanilla":  "/data/gpfs/projects/punim2219/LM_with_SWOW/chunhua/Data/output/qwen/WVS_US_qwen_vanilla.json",
    "WVS_EN_qwen_sft":      "/data/gpfs/projects/punim2219/LM_with_SWOW/chunhua/Data/output/qwen/WVS_US_qwen_us_sft.json",
    "WVS_EN_llama_vanilla": "/data/gpfs/projects/punim2219/LM_with_SWOW/chunhua/Data/output/llama/WVS_US_llama_vanilla.json",
    "WVS_EN_llama_sft":     "/data/gpfs/projects/punim2219/LM_with_SWOW/chunhua/Data/output/llama/WVS_US_llama_us_sft.json",
}

# ─────────────────────────────────────────────────────────────────────────────
# Distances & helpers
# ─────────────────────────────────────────────────────────────────────────────
def js_distance(p, q):  return float(jensenshannon(p, q, base=2))
def em_distance(p, q):  return float(wasserstein_distance(p, q))
def emd_norm(p, q):     return em_distance(p, q) / (len(p)-1) if len(p) > 1 else 0.
def combo(p, q):        return 0.5*js_distance(p, q) + 0.5*emd_norm(p, q)

def truncate(a: List[float], b: List[float]) -> tuple[List[float], List[float]]:
    m = min(len(a), len(b))
    return a[:m], b[:m]

# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 – tension set
# ─────────────────────────────────────────────────────────────────────────────
def generate_tension_csv() -> pd.DataFrame:
    df = pd.read_json(SURVEY_JSON)

    def vec(row, key): return [v for _, v in sorted(row[key].items())]
    df["us_vec"] = df.apply(lambda r: vec(r, "us_score"), axis=1)
    df["cn_vec"] = df.apply(lambda r: vec(r, "china_score"), axis=1)
    df = df[df.us_vec.str.len() == df.cn_vec.str.len()].reset_index(drop=True)

    df["js_us_cn"]  = df.apply(lambda r: js_distance(r.us_vec, r.cn_vec), axis=1)
    df["emd_us_cn"] = df.apply(lambda r: emd_norm   (r.us_vec, r.cn_vec), axis=1)
    df["combo"]     = 0.5*df.js_us_cn + 0.5*df.emd_us_cn

    keep = ["Id","question","question_instruction","choices",
            "china_score","us_score","js_us_cn","emd_us_cn","combo"]

    top = (df.sort_values("combo", ascending=False)
             .head(K_TOP)[keep].reset_index(drop=True))
    top.to_csv(TENSION_CSV, index=False)
    return top

# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 – model-specific distances
# ─────────────────────────────────────────────────────────────────────────────
def model_dataframe(model_json: Path, tension_ids: set[str]) -> pd.DataFrame:
    df_model = pd.read_json(model_json)
    df_gold  = pd.read_json(SURVEY_JSON)

    df = df_gold[df_gold.Id.isin(tension_ids)].copy()
    df["model_vec"] = df_model.loc[df.index, "choice_values"]

    df["us_vec"] = df.us_score.apply(lambda d: list(d.values()))
    df["cn_vec"] = df.china_score.apply(lambda d: list(d.values()))

    js_u, js_c, cmb_u, cmb_c = [], [], [], []
    for mv, uv, cv in zip(df.model_vec, df.us_vec, df.cn_vec):
        mv_u, uv_t = truncate(mv, uv)
        mv_c, cv_t = truncate(mv, cv)
        js_u.append(js_distance(mv_u, uv_t))
        js_c.append(js_distance(mv_c, cv_t))
        cmb_u.append(combo(mv_u, uv_t))
        cmb_c.append(combo(mv_c, cv_t))

    df["js_model_us"] = js_u
    df["js_model_cn"] = js_c
    df["combo_us"]    = cmb_u
    df["combo_cn"]    = cmb_c
    df["bias"]        = df.combo_cn - df.combo_us
    return df

# ─────────────────────────────────────────────────────────────────────────────
# Composite plot with shared color scale
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# Composite plot with shared colour-bar
# ─────────────────────────────────────────────────────────────────────────────
def composite_plot(tag_left: str, tag_right: str,
                   df_left: pd.DataFrame, df_right: pd.DataFrame,
                   title_left: str, title_right: str) -> None:

    # wider canvas → room for colour-bar
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), sharex=True, sharey=True)

    # leave 12 % of the width on the right for the bar
    fig.subplots_adjust(wspace=0.25, right=0.88)

    # global colour scale (symmetric around 0)
    all_bias = pd.concat([df_left["bias"], df_right["bias"]])
    vmax = max(all_bias.abs());  vmin = -vmax

    scatters = []
    for ax, df, ttl in zip(
            axes, [df_left, df_right], [title_left, title_right]):

        sc = ax.scatter(df.combo_us, df.combo_cn,
                        c=df.bias, cmap='coolwarm',
                        edgecolor='black', s=46,
                        vmin=vmin, vmax=vmax)
        scatters.append(sc)

        # question IDs (keep them)
        for _, r in df.iterrows():
            ax.text(r.combo_us + .006, r.combo_cn + .006,
                    r.Id, fontsize=5, alpha=.6)

        ax.plot([0, 0.7], [0, 0.7], ls='--', c='gray', alpha=.4)

        # give dots breathing-room
        pad = 0.02
        ax.set_xlim(-pad, 0.70 + pad)
        ax.set_ylim(-pad, 0.70 + pad)
        ax.set_xlabel("Distance to US Human Responses")
        ax.set_ylabel("Distance to Chinese Human Responses")
        ax.set_title(ttl, fontsize=11)

        n_cn = (df.combo_us > df.combo_cn).sum()
        n_us = (df.combo_us < df.combo_cn).sum()
        ax.text(.05, .92, f"Closer to CN: {n_cn}", transform=ax.transAxes)
        ax.text(.05, .86, f"Closer to US: {n_us}", transform=ax.transAxes)

    # add colour-bar in reserved strip
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])   # [left, bottom, width, height]
    cbar = fig.colorbar(scatters[0], cax=cbar_ax)
    cbar.set_label("Bias (CN – US)", fontsize=9)
    cbar.ax.tick_params(labelsize=7)

    fig.suptitle(f"{title_left}   vs   {title_right}", fontsize=14, y=.985)

    out_png = PLOT_DIR / f"{tag_left}_vs_{tag_right}.png"
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"✔ composite plot written → {out_png.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Extra – Optional CSV output for review
# ─────────────────────────────────────────────────────────────────────────────
def write_vector_csv(tension_ids: set[str], dfs: dict[str, pd.DataFrame]) -> None:
    df_gold = pd.read_json(SURVEY_JSON)

    def model_vec(tag: str, qid: str) -> list[float]:
        row = dfs[tag].loc[dfs[tag].Id == qid, "model_vec"]
        return row.iloc[0] if not row.empty else []

    rows = []
    for qid in sorted(tension_ids):
        g = df_gold[df_gold.Id == qid].iloc[0]

        us_vec = [v for _, v in sorted(g["us_score"].items())]
        cn_vec = [v for _, v in sorted(g["china_score"].items())]

        rows.append({
            "Id": qid,
            "question": g["question"],
            "question_instruction": g["question_instruction"],
            "choices": g["choices"],
            "us_vec":      json.dumps(us_vec),
            "cn_vec":      json.dumps(cn_vec),
            "qwen_vanilla":   json.dumps(model_vec("WVS_EN_qwen_vanilla",  qid)),
            "llama_vanilla":  json.dumps(model_vec("WVS_EN_llama_vanilla", qid)),
            "qwen_sft":       json.dumps(model_vec("WVS_EN_qwen_sft",      qid)),
            "llama_sft":      json.dumps(model_vec("WVS_EN_llama_sft",     qid)),
        })

    df_out = pd.DataFrame(rows)
    out_csv = ROOT_WRITE / "tension_set_vectors_EN.csv"
    df_out.to_csv(out_csv, index=False)
    print(f"✔ probability-vector table → {out_csv.name}")

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Rebuilding global top-50 tension set for EN…")
    tension_ids = set(generate_tension_csv().Id)

    dfs_en: dict[str, pd.DataFrame] = {}
    for tag, path in OUTPUT_FILES_EN.items():
        df = model_dataframe(Path(path), tension_ids)
        dfs_en[tag] = df
        out_json = ROOT_WRITE / f"{tag}_top{K_TOP}.json"
        df.to_json(out_json, orient="records", indent=2)

    composite_plot("EN_qwen_vanilla", "EN_qwen_sft",
                   dfs_en["WVS_EN_qwen_vanilla"], dfs_en["WVS_EN_qwen_sft"],
                   "Qwen-vanilla", "Qwen-SFT (on US SWOW)")

    composite_plot("EN_llama_vanilla", "EN_llama_sft",
                   dfs_en["WVS_EN_llama_vanilla"], dfs_en["WVS_EN_llama_sft"],
                   "Llama-vanilla ", "Llama-SFT (on US SWOW)")

    write_vector_csv(tension_ids, dfs_en)
