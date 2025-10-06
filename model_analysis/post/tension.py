# model_analysis/post/tension.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance


# ---------- small helpers ----------

def _DATA_ROOT() -> Path:
    # repo-relative default data root (override by passing explicit paths)
    return Path(__file__).resolve().parents[2] / "data"

def _to_vec(x) -> List[float]:
    if isinstance(x, dict):
        # sort by key to ensure stable option order
        return [v for _, v in sorted(x.items())]
    if isinstance(x, (list, tuple, np.ndarray, pd.Series)):
        return list(x)
    return []

def _truncate(a: List[float], b: List[float]) -> Tuple[List[float], List[float]]:
    m = min(len(a), len(b))
    return a[:m], b[:m]

def js_distance(p, q) -> float:
    return float(jensenshannon(p, q, base=2))

def em_distance(p, q) -> float:
    return float(wasserstein_distance(p, q))

def emd_norm(p, q) -> float:
    return em_distance(p, q) / (len(p) - 1) if len(p) > 1 else 0.0

def combo_distance(p, q) -> float:
    # hybrid: average of JS and normalized EMD
    return 0.5 * js_distance(p, q) + 0.5 * emd_norm(p, q)


# ---------- config dataclass ----------

@dataclass
class TensionConfig:
    language: str                     # "en" or "zh"
    survey_json: Path                 # WV survey with us_score / china_score
    model_outputs: Dict[str, Path]    # tag -> model result json (has choice_values)
    out_dir: Path                     # where to save csv/json/plots
    k_top: int = 50
    plot_pairs: Optional[List[Tuple[str, str]]] = None  # e.g. [("qwen_vanilla","qwen_sft"), ...]
    plot_prefix: Optional[str] = None                   # "EN" / "ZH" in file names
    image_format: str = "png"                           # "png" or "pdf"

    def __post_init__(self):
        self.language = "zh" if str(self.language).lower().startswith("zh") else "en"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        if self.plot_prefix is None:
            self.plot_prefix = "ZH" if self.language == "zh" else "EN"
        # If no explicit plot pairs, try to infer common (vanilla, sft) for qwen & llama
        if not self.plot_pairs:
            tags = list(self.model_outputs.keys())
            def find_pair(prefix: str) -> Optional[Tuple[str, str]]:
                van = next((t for t in tags if "vanilla" in t and prefix in t), None)
                sft = next((t for t in tags if "sft" in t and prefix in t), None)
                return (van, sft) if (van and sft) else None
            pairs = []
            for prefix in ("qwen", "llama"):
                p = find_pair(prefix)
                if p:
                    pairs.append(p)
            self.plot_pairs = pairs


# ---------- phase 1: build global tension set ----------

def build_tension_set(survey_json: Path, k_top: int, out_csv: Path) -> pd.DataFrame:
    df = pd.read_json(survey_json)

    def vec(row, key): return [v for _, v in sorted(row[key].items())]

    df["us_vec"] = df.apply(lambda r: vec(r, "us_score"), axis=1)
    df["cn_vec"] = df.apply(lambda r: vec(r, "china_score"), axis=1)
    df = df[df.us_vec.str.len() == df.cn_vec.str.len()].reset_index(drop=True)

    df["js_us_cn"]  = df.apply(lambda r: js_distance(r.us_vec, r.cn_vec), axis=1)
    df["emd_us_cn"] = df.apply(lambda r: emd_norm   (r.us_vec, r.cn_vec), axis=1)
    df["combo"]     = 0.5 * df["js_us_cn"] + 0.5 * df["emd_us_cn"]

    keep = ["Id", "question", "question_instruction", "choices",
            "china_score", "us_score", "js_us_cn", "emd_us_cn", "combo"]

    top = (df.sort_values("combo", ascending=False)
             .head(k_top)[keep].reset_index(drop=True))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    top.to_csv(out_csv, index=False)
    return top


# ---------- phase 2: per-model distances on the tension set ----------

def model_dataframe(model_json: Path, survey_json: Path, tension_ids: Iterable[str]) -> pd.DataFrame:
    df_m = pd.read_json(model_json)
    df_g = pd.read_json(survey_json)

    # keep only the selected questions (by Id), preserve a stable order
    df_sel = df_g[df_g["Id"].isin(set(tension_ids))].copy()

    # pick model vectors aligned by Id if available; otherwise fall back to row index alignment
    has_id = "Id" in df_m.columns
    mvecs = []
    for _, row in df_sel.iterrows():
        qid = row["Id"]
        if has_id:
            mrow = df_m[df_m["Id"] == qid]
            mv = _to_vec(mrow.iloc[0]["choice_values"]) if not mrow.empty else []
        else:
            # align by original index position in the full gold df
            orig_idx = int(df_g.index[df_g["Id"] == qid][0])
            mv = _to_vec(df_m.iloc[orig_idx]["choice_values"]) if orig_idx < len(df_m) else []
        mvecs.append(mv)

    df = df_sel.reset_index(drop=True)
    df["model_vec"] = mvecs
    df["us_vec"] = df["us_score"].apply(_to_vec)
    df["cn_vec"] = df["china_score"].apply(_to_vec)

    js_u, js_c, cmb_u, cmb_c = [], [], [], []
    for mv, uv, cv in zip(df["model_vec"], df["us_vec"], df["cn_vec"]):
        mv_u, uv_t = _truncate(mv, uv)
        mv_c, cv_t = _truncate(mv, cv)
        js_u.append(js_distance(mv_u, uv_t))
        js_c.append(js_distance(mv_c, cv_t))
        cmb_u.append(combo_distance(mv_u, uv_t))
        cmb_c.append(combo_distance(mv_c, cv_t))

    df["js_model_us"] = js_u
    df["js_model_cn"] = js_c
    df["combo_us"]    = cmb_u
    df["combo_cn"]    = cmb_c
    df["bias"]        = df["combo_cn"] - df["combo_us"]
    return df


# ---------- plotting ----------

def composite_plot(
    tag_left: str, tag_right: str,
    df_left: pd.DataFrame, df_right: pd.DataFrame,
    title_left: str, title_right: str,
    out_dir: Path, prefix: str, image_format: str = "png"
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), sharex=True, sharey=True)
    fig.subplots_adjust(wspace=0.25, right=0.88)

    all_bias = pd.concat([df_left["bias"], df_right["bias"]])
    vmax = float(all_bias.abs().max()) if len(all_bias) else 1.0
    vmin = -vmax

    scatters = []
    for ax, df, ttl in zip(axes, [df_left, df_right], [title_left, title_right]):
        sc = ax.scatter(df["combo_us"], df["combo_cn"],
                        c=df["bias"], cmap="coolwarm",
                        edgecolor="black", s=46,
                        vmin=vmin, vmax=vmax)
        scatters.append(sc)

        # annotate with question Ids
        for _, r in df.iterrows():
            ax.text(float(r["combo_us"]) + 0.006,
                    float(r["combo_cn"]) + 0.006,
                    str(r["Id"]), fontsize=5, alpha=.6)

        ax.plot([0, 0.7], [0, 0.7], ls='--', c='gray', alpha=.4)

        pad = 0.02
        ax.set_xlim(-pad, 0.70 + pad)
        ax.set_ylim(-pad, 0.70 + pad)
        ax.set_xlabel("Distance to US Human Responses")
        ax.set_ylabel("Distance to Chinese Human Responses")
        ax.set_title(ttl, fontsize=11)

        n_cn = int((df["combo_us"] > df["combo_cn"]).sum())
        n_us = int((df["combo_us"] < df["combo_cn"]).sum())
        ax.text(.05, .92, f"Closer to CN: {n_cn}", transform=ax.transAxes)
        ax.text(.05, .86, f"Closer to US: {n_us}", transform=ax.transAxes)

    # shared colour bar
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(scatters[0], cax=cbar_ax)
    cbar.set_label("Bias (CN – US)", fontsize=9)
    cbar.ax.tick_params(labelsize=7)

    fig.suptitle(f"{title_left}   vs   {title_right}", fontsize=14, y=.985)

    out_path = out_dir / f"{prefix}_{tag_left}_vs_{tag_right}.{image_format}"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return out_path


# ---------- optional: dump vectors CSV for review ----------

def write_vector_csv(
    tension_ids: Iterable[str],
    survey_json: Path,
    dfs_by_tag: Dict[str, pd.DataFrame],
    out_csv: Path
) -> None:
    df_gold = pd.read_json(survey_json)
    rows = []
    for qid in sorted(tension_ids):
        g = df_gold[df_gold["Id"] == qid].iloc[0]
        us_vec = _to_vec(g["us_score"])
        cn_vec = _to_vec(g["china_score"])

        row = {
            "Id": qid,
            "question": g["question"],
            "question_instruction": g.get("question_instruction", ""),
            "choices": g.get("choices", []),
            "us_vec": json.dumps(us_vec),
            "cn_vec": json.dumps(cn_vec),
        }
        for tag, dfm in dfs_by_tag.items():
            mv = dfm.loc[dfm["Id"] == qid, "model_vec"]
            row[tag] = json.dumps(list(mv.iloc[0]) if not mv.empty else [])
        rows.append(row)

    pd.DataFrame(rows).to_csv(out_csv, index=False)


# ---------- orchestration ----------

def run_tension_analysis(cfg: TensionConfig) -> Dict[str, object]:
    # 1) tension set
    tension_csv = cfg.out_dir / f"tension_set_top{cfg.k_top}.csv"
    top_df = build_tension_set(cfg.survey_json, cfg.k_top, tension_csv)
    tension_ids = set(top_df["Id"])

    # 2) per-model frames + save their JSONs
    dfs: Dict[str, pd.DataFrame] = {}
    for tag, mpath in cfg.model_outputs.items():
        dfm = model_dataframe(Path(mpath), cfg.survey_json, tension_ids)
        dfs[tag] = dfm
        (cfg.out_dir / f"{tag}_top{cfg.k_top}.json").write_text(
            dfm.to_json(orient="records", indent=2), encoding="utf-8"
        )

    # 3) composite plots for requested pairs
    plot_paths: List[Path] = []
    for left, right in cfg.plot_pairs:
        if left in dfs and right in dfs:
            # nicer titles
            def pretty(t: str) -> str:
                t = t.replace("_", " ").replace("-", " ")
                return t.replace("vanilla", "vanilla").replace("sft", "SFT")
            p = composite_plot(
                left, right,
                dfs[left], dfs[right],
                title_left=pretty(left), title_right=pretty(right),
                out_dir=cfg.out_dir, prefix=cfg.plot_prefix, image_format=cfg.image_format
            )
            print(f"✔ wrote plot → {p.name}")
            plot_paths.append(p)

    # 4) optional vectors CSV
    vec_csv = cfg.out_dir / f"tension_set_vectors_{cfg.plot_prefix}.csv"
    write_vector_csv(tension_ids, cfg.survey_json, dfs, vec_csv)
    print(f"✔ probability-vector table → {vec_csv.name}")

    return {
        "tension_csv": tension_csv,
        "per_model_jsons": [cfg.out_dir / f"{t}_top{cfg.k_top}.json" for t in dfs],
        "plots": plot_paths,
        "vectors_csv": vec_csv,
    }


# ---------- example (not executed when imported) ----------

if __name__ == "__main__":
    # Example usage (edit paths or call from your Typer CLI later)
    DATA = _DATA_ROOT()

    # EN example
    # cfg = TensionConfig(
    #     language="en",
    #     survey_json=DATA / "WV_Bench/question_answer.json",
    #     model_outputs={
    #         "WVS_EN_qwen_vanilla":  DATA / "output/qwen/WVS_US_qwen_vanilla.json",
    #         "WVS_EN_qwen_sft":      DATA / "output/qwen/WVS_US_qwen_us_sft.json",
    #         "WVS_EN_llama_vanilla": DATA / "output/llama/WVS_US_llama_vanilla.json",
    #         "WVS_EN_llama_sft":     DATA / "output/llama/WVS_US_llama_us_sft.json",
    #     },
    #     out_dir=DATA / "cross_country/en",
    #     k_top=50,
    # )

    # ZH example
    # cfg = TensionConfig(
    #     language="zh",
    #     survey_json=DATA / "WV_Bench/question_answer.json",
    #     model_outputs={
    #         "WVS_ZH_qwen_vanilla":  DATA / "output/qwen/WVS_ZH_qwen_vanilla.json",
    #         "WVS_ZH_qwen_sft":      DATA / "output/qwen/WVS_ZH_qwen_sft.json",
    #         "WVS_ZH_llama_vanilla": DATA / "output/llama/WVS_ZH_llama_vanilla.json",
    #         "WVS_ZH_llama_sft":     DATA / "output/llama/WVS_ZH_llama_sft.json",
    #     },
    #     out_dir=DATA / "cross_country/zh",
    #     k_top=50,
    #     image_format="png",
    # )

    # results = run_tension_analysis(cfg)
    # print(results)
