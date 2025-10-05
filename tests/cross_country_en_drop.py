#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EN Tension Set (leakage-filtered, topic-only exact match)
=========================================================
Drops an item only if a training cue (from EN *.jsonl "input") appears
as an exact phrase (word-boundary, case-insensitive) inside the *topic
span* of the English prompt. We ignore answer options and boilerplate.

Outputs (in output_en_noleak/):
- train_cues_en.json
- dropped_due_to_cues_en.csv            (Id, question, prompt_instruction, topic_en, matched_cue)
- kept_after_filter_en.csv
- tension_set_top50_en_noleak.csv
- tension_set_vectors_en_noleak.csv
- <tag>_topK_en_noleak.json
- cross_country_en_noleak/*.png
"""

from __future__ import annotations
import json, re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance

# ─────────────────────────────────────────────────────────────────────────────
# Paths & configuration
# ─────────────────────────────────────────────────────────────────────────────
ROOT_READ   = Path("/data/gpfs/projects/punim2219/LM_with_SWOW")
# NOTE: If your file is plural (questions_answers.json), update this path.
SURVEY_JSON = ROOT_READ / "kabir/Data/WV_Bench/question_answer.json"

# EN train jsonl directory (provided)
TRAIN_JSONL_DIR = Path(
    "/data/gpfs/projects/punim2219/LM_with_SWOW/sukaih/cultural-lexis-finetune-llms/data/03_primary/llm_swow_finetune_dataset/swow_en/train"
)

ROOT_WRITE  = Path("/data/projects/punim2219/LM_with_SWOW/kabir/Data/output_en_noleak")
PLOT_DIR    = ROOT_WRITE / "cross_country_en_noleak"
TENSION_CSV = ROOT_WRITE / "tension_set_top50_en_noleak.csv"
VECTORS_CSV = ROOT_WRITE / "tension_set_vectors_en_noleak.csv"
DROP_LOG    = ROOT_WRITE / "dropped_due_to_cues_en.csv"
KEEP_LOG    = ROOT_WRITE / "kept_after_filter_en.csv"
CUES_JSON   = ROOT_WRITE / "train_cues_en.json"
K_TOP       = 50

# Safer cue policy (avoid tiny/function words nuking everything)
MIN_CUE_LEN = 2                # set to 3 if you want to be stricter
SINGLE_CHAR_POLICY = "drop"    # "drop" | "whitelist"
SINGLE_CHAR_WHITELIST_FILE: Optional[Path] = None  # if using "whitelist", one char per line

ROOT_WRITE.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# EN model outputs (as provided)
OUTPUT_FILES_EN: Dict[str,str] = {
    "WVS_EN_qwen_vanilla":  "/data/gpfs/projects/punim2219/LM_with_SWOW/chunhua/Data/output/qwen/WVS_US_qwen_vanilla.json",
    "WVS_EN_qwen_sft":      "/data/gpfs/projects/punim2219/LM_with_SWOW/chunhua/Data/output/qwen/WVS_US_qwen_us_sft.json",
    "WVS_EN_llama_vanilla": "/data/gpfs/projects/punim2219/LM_with_SWOW/chunhua/Data/output/llama/WVS_US_llama_vanilla.json",
    "WVS_EN_llama_sft":     "/data/gpfs/projects/punim2219/LM_with_SWOW/chunhua/Data/output/llama/WVS_US_llama_us_sft.json",
}

# ─────────────────────────────────────────────────────────────────────────────
# Distances & helpers
# ─────────────────────────────────────────────────────────────────────────────
def js_distance(p, q) -> float:  return float(jensenshannon(p, q, base=2))
def em_distance(p, q) -> float:  return float(wasserstein_distance(p, q))
def emd_norm(p, q) -> float:     return em_distance(p, q) / (len(p)-1) if len(p) > 1 else 0.
def combo(p, q) -> float:        return 0.5*js_distance(p, q) + 0.5*emd_norm(p, q)

def truncate(a: List[float], b: List[float]) -> Tuple[List[float], List[float]]:
    m = min(len(a), len(b));  return a[:m], b[:m]

# ─────────────────────────────────────────────────────────────────────────────
# Load cues from EN jsonl + save to JSON
# ─────────────────────────────────────────────────────────────────────────────
def _load_whitelist(path: Optional[Path]) -> set[str]:
    if path and path.exists():
        return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
    return set()

def load_cues_en(train_dir: Path, min_len: int = 2) -> List[str]:
    cues: set[str] = set()
    files = sorted(train_dir.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No .jsonl files found under {train_dir}")
    whitelist = _load_whitelist(SINGLE_CHAR_WHITELIST_FILE)

    total_seen = 0; singles_seen = 0
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                cue = str(obj.get("input", "")).strip()
                if not cue:
                    continue
                total_seen += 1
                if len(cue) >= min_len:
                    cues.add(cue)
                elif len(cue) == 1:
                    singles_seen += 1
                    if SINGLE_CHAR_POLICY == "whitelist" and cue in whitelist:
                        cues.add(cue)

    sorted_cues = sorted(cues, key=lambda x: (len(x), x.lower()))
    CUES_JSON.write_text(json.dumps(sorted_cues, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Cues(EN): kept {len(sorted_cues)} after policy (min_len={min_len}, singles={SINGLE_CHAR_POLICY}); "
          f"seen={total_seen}, single-char seen={singles_seen}. Saved → {CUES_JSON.name}")
    return sorted_cues

# ─────────────────────────────────────────────────────────────────────────────
# Build cue regex (word-boundary, case-insensitive) in batches
# ─────────────────────────────────────────────────────────────────────────────
_WORDCHARS = re.compile(r"[A-Za-z]")

def build_cue_patterns_en(cues: List[str], batch_size: int = 600) -> List[re.Pattern]:
    """
    For alphabetic cues, use case-insensitive word-boundary match: \b...\b
    For non-alphabetic cues, just escape and match literally (still case-insensitive).
    """
    parts: List[str] = []
    for c in cues:
        esc = re.escape(c)
        if _WORDCHARS.search(c):
            parts.append(rf"\b{esc}\b")
        else:
            parts.append(esc)
    # chunk to avoid giant regexes
    return [re.compile("|".join(parts[i:i+batch_size]), flags=re.IGNORECASE)
            for i in range(0, len(parts), batch_size)]

# ─────────────────────────────────────────────────────────────────────────────
# Topic extraction (EN): use prompt_instruction; ignore "Options:"
# fallback to "question" (topic after ':' or '-' if present) then last clause
# ─────────────────────────────────────────────────────────────────────────────
SENT_SPLIT_EN = re.compile(r"[.!?]")

def _before_options_en(s: str) -> str:
    up = s.split("Options:")[0]
    return up.strip()

TOPIC_PATS_EN: List[re.Pattern] = [
    re.compile(r"–\s*(.+?)\s*$"),              # dash-topic (– Topic)
    re.compile(r"-\s*(.+?)\s*$"),              # hyphen-topic (- Topic)
    re.compile(r":\s*(.+?)\s*$"),              # trailing colon intro: "...: Topic"
    re.compile(r"Are you.*?\bmember\b.*?\bof\b\s+(.+?)\??", re.IGNORECASE),
    re.compile(r"How much (?:do you|you)\s+trust[: ]\s*(.+?)\??", re.IGNORECASE),
    re.compile(r"Confidence[: ]\s*(.+?)\s*$", re.IGNORECASE),
    re.compile(r"Not.*?want.*?neighbors?.*?\b(.+?)\b", re.IGNORECASE),
]

WS = re.compile(r"\s+")

def _last_clause_en(s: str) -> str:
    parts = [p.strip() for p in SENT_SPLIT_EN.split(s) if p.strip()]
    return parts[-1] if parts else s.strip()

def extract_topic_en(row: pd.Series) -> str:
    """
    1) Prefer 'prompt_instruction' (EN); strip after 'Options:'.
    2) Try targeted patterns to isolate the short topic phrase.
    3) Fallback to 'question' tail (after ':' or '-' if present).
    4) Fallback to last clause pre-Options.
    """
    text = ""
    for fld in ("prompt_instruction", "question_instruction", "question"):
        v = row.get(fld, None)
        if isinstance(v, str) and v.strip():
            text = v.strip()
            break
    if not text:
        return ""

    stem = _before_options_en(text)

    # try prompt-based patterns
    for pat in TOPIC_PATS_EN:
        m = pat.search(stem)
        if m:
            span = m.group(1).strip()
            return WS.sub(" ", span)

    # fallback: derive from 'question' after colon/hyphen
    q = row.get("question", "")
    if isinstance(q, str) and q.strip():
        q = q.strip()
        if ":" in q:
            tail = q.split(":", 1)[1].strip()
            if tail:
                return WS.sub(" ", tail)
        if "-" in q:
            tail = q.split("-", 1)[1].strip()
            if tail:
                return WS.sub(" ", tail)

    # final fallback: last meaningful clause before options
    return WS.sub(" ", _last_clause_en(stem))

# ─────────────────────────────────────────────────────────────────────────────
# Filtering (topic-only exact match)
# ─────────────────────────────────────────────────────────────────────────────
def drop_leaky_questions_topic_only_en(df_all: pd.DataFrame, cue_pats: List[re.Pattern]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    df_all = df_all.copy()
    df_all["topic_en"] = df_all.apply(extract_topic_en, axis=1)

    def find_match_in_topic(topic: str) -> Optional[str]:
        if not topic:
            return None
        for pat in cue_pats:
            m = pat.search(topic)
            if m:
                return m.group(0)
        return None

    matched = df_all["topic_en"].apply(find_match_in_topic)
    mask = matched.notna()

    # logs (include prompt_instruction for cross-check)
    if "prompt_instruction" not in df_all.columns:
        df_all["prompt_instruction"] = ""

    dropped = df_all.loc[mask, ["Id", "question", "prompt_instruction", "topic_en"]].copy()
    dropped["matched_cue"] = matched.loc[mask].values
    kept = df_all.loc[~mask].drop(columns=["topic_en"], errors="ignore").copy()

    stats = {
        "total_questions": int(len(df_all)),
        "dropped": int(mask.sum()),
        "kept": int((~mask).sum()),
        "pct_dropped": round(100.0 * float(mask.sum()) / max(1, len(df_all)), 2),
    }

    dropped.to_csv(DROP_LOG, index=False)
    kept.to_csv(KEEP_LOG, index=False)
    print(f"Leakage filter (EN, topic-only): dropped {stats['dropped']} / {stats['total_questions']} "
          f"({stats['pct_dropped']}%). Kept: {stats['kept']}. "
          f"Logs → {DROP_LOG.name}, {KEEP_LOG.name}")
    return kept.reset_index(drop=True), dropped.reset_index(drop=True), stats

# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 – filtered tension set (top-K by hybrid US–CN gap)
# ─────────────────────────────────────────────────────────────────────────────
def generate_tension_csv_filtered_en() -> pd.DataFrame:
    df_all = pd.read_json(SURVEY_JSON)

    cues = load_cues_en(TRAIN_JSONL_DIR, min_len=MIN_CUE_LEN)
    if cues:
        cue_pats = build_cue_patterns_en(cues)
        df, _dropped, _stats = drop_leaky_questions_topic_only_en(df_all, cue_pats)
    else:
        print("⚠ No EN cues retained after policy — skipping filtering.")
        df = df_all.copy()

    if df.empty:
        cols = ["Id","question","question_instruction","choices",
                "china_score","us_score","js_us_cn","emd_us_cn","combo"]
        pd.DataFrame(columns=cols).to_csv(TENSION_CSV, index=False)
        print("⚠ All questions filtered out. Wrote empty tension set and stopping Phase 1.")
        return pd.DataFrame(columns=["Id"])

    # distances
    def vec(row, key): return [v for _, v in sorted(row[key].items())]
    df["us_vec"] = df.apply(lambda r: vec(r, "us_score"),    axis=1)
    df["cn_vec"] = df.apply(lambda r: vec(r, "china_score"), axis=1)
    df = df[df.us_vec.str.len() == df.cn_vec.str.len()].reset_index(drop=True)

    df["js_us_cn"]  = df.apply(lambda r: js_distance(r.us_vec, r.cn_vec),  axis=1)
    df["emd_us_cn"] = df.apply(lambda r: emd_norm   (r.us_vec, r.cn_vec),  axis=1)
    df["combo"]     = 0.5*df.js_us_cn + 0.5*df.emd_us_cn

    keep_cols = ["Id","question","question_instruction","choices",
                 "china_score","us_score","js_us_cn","emd_us_cn","combo"]

    k = min(K_TOP, len(df))
    if k < K_TOP:
        print(f"⚠ Only {k} questions remain after filtering; writing top-{k}.")
    top = df.sort_values("combo", ascending=False).head(k)[keep_cols].reset_index(drop=True)
    top.to_csv(TENSION_CSV, index=False)
    print(f"✔ filtered EN tension set → {TENSION_CSV.name}")
    return top

# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 – model-specific distances
# ─────────────────────────────────────────────────────────────────────────────
def model_dataframe_en(model_json: Path, tension_ids: set[str]) -> pd.DataFrame:
    df_model = pd.read_json(model_json)
    df_gold  = pd.read_json(SURVEY_JSON)

    df = df_gold[df_gold.Id.isin(tension_ids)].copy()
    df["model_vec"] = df_model.loc[df.index, "choice_values"]

    df["us_vec"] = df.us_score.apply(lambda d: list(d.values()))
    df["cn_vec"] = df.china_score.apply(lambda d: list(d.values()))

    js_u, js_c, cmb_u, cmb_c = [], [], [], []
    for mv, uv, cv in zip(df["model_vec"], df["us_vec"], df["cn_vec"]):
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
# Plot – composite (shared colour-bar)
# ─────────────────────────────────────────────────────────────────────────────
def composite_plot(tag_left: str, tag_right: str,
                   df_left: pd.DataFrame, df_right: pd.DataFrame,
                   title_left: str, title_right: str) -> None:

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), sharex=True, sharey=True)
    fig.subplots_adjust(wspace=0.25, right=0.88)

    all_bias = pd.concat([df_left["bias"], df_right["bias"]])
    vmax = max(all_bias.abs());  vmin = -vmax

    scatters = []
    for ax, df, ttl in zip(axes, [df_left, df_right], [title_left, title_right]):
        sc = ax.scatter(df.combo_us, df.combo_cn,
                        c=df.bias, cmap='coolwarm',
                        edgecolor='black', s=46,
                        vmin=vmin, vmax=vmax)
        scatters.append(sc)

        for _, r in df.iterrows():
            ax.text(r.combo_us + .006, r.combo_cn + .006, r.Id, fontsize=5, alpha=.6)

        ax.plot([0, 0.7], [0, 0.7], ls='--', c='gray', alpha=.4)
        pad = 0.02
        ax.set_xlim(-pad, 0.70 + pad); ax.set_ylim(-pad, 0.70 + pad)
        ax.set_xlabel("Hybrid distance (model ↔ US)")
        ax.set_ylabel("Hybrid distance (model ↔ CN)")
        ax.set_title(ttl, fontsize=11)

        n_cn = (df.combo_us > df.combo_cn).sum()
        n_us = (df.combo_us < df.combo_cn).sum()
        ax.text(.05, .92, f"Closer to CN: {n_cn}", transform=ax.transAxes)
        ax.text(.05, .86, f"Closer to US: {n_us}", transform=ax.transAxes)

    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(scatters[0], cax=cbar_ax)
    cbar.set_label("Bias (CN – US)", fontsize=9)
    cbar.ax.tick_params(labelsize=7)

    out_png = PLOT_DIR / f"{tag_left}_vs_{tag_right}.png"
    fig.suptitle(f"{title_left}   vs   {title_right}", fontsize=14, y=.985)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"✔ composite plot written → {out_png.name}")

# ─────────────────────────────────────────────────────────────────────────────
# Vector CSV (inspection)
# ─────────────────────────────────────────────────────────────────────────────
def write_vector_csv_en(tension_ids: set[str], dfs: dict[str, pd.DataFrame]) -> None:
    df_gold = pd.read_json(SURVEY_JSON)

    def model_vec(tag: str, qid: str) -> List[float]:
        row = dfs[tag].loc[dfs[tag].Id == qid, "model_vec"]
        return row.iloc[0] if not row.empty else []

    rows = []
    for qid in sorted(tension_ids):
        g = df_gold[df_gold.Id == qid].iloc[0]
        us_vec = [v for _, v in sorted(g["us_score"].items())]
        cn_vec = [v for _, v in sorted(g["china_score"].items())]
        rows.append({
            "Id": qid,
            "question": g.get("question", ""),
            "question_instruction": g.get("question_instruction", ""),
            "choices": g.get("choices", {}),
            "us_vec":         json.dumps(us_vec),
            "cn_vec":         json.dumps(cn_vec),
            "qwen_vanilla":   json.dumps(model_vec("WVS_EN_qwen_vanilla",  qid)),
            "llama_vanilla":  json.dumps(model_vec("WVS_EN_llama_vanilla", qid)),
            "qwen_sft":       json.dumps(model_vec("WVS_EN_qwen_sft",      qid)),
            "llama_sft":      json.dumps(model_vec("WVS_EN_llama_sft",     qid)),
        })

    pd.DataFrame(rows).to_csv(VECTORS_CSV, index=False)
    print(f"✔ probability-vector table → {VECTORS_CSV.name}")

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Rebuilding leakage-filtered EN tension set (topic-only, exact-match)…")
    top_df = generate_tension_csv_filtered_en()
    tension_ids = set(top_df.Id)

    if not tension_ids:
        print("⚠ No questions left after filtering—skipping per-model metrics, plots, and vectors.")
    else:
        dfs_en: dict[str, pd.DataFrame] = {}
        for tag, path in OUTPUT_FILES_EN.items():
            df = model_dataframe_en(Path(path), tension_ids)
            dfs_en[tag] = df
            out_json = ROOT_WRITE / f"{tag}_top{len(tension_ids)}_en_noleak.json"
            df.to_json(out_json, orient="records", indent=2)
            print(f"✔ per-model metrics → {out_json.name}")

        composite_plot("EN_qwen_vanilla", "EN_qwen_sft",
                       dfs_en["WVS_EN_qwen_vanilla"], dfs_en["WVS_EN_qwen_sft"],
                       "Qwen-vanilla", "Qwen-SFT (on US SWOW)")
        composite_plot("EN_llama_vanilla", "EN_llama_sft",
                       dfs_en["WVS_EN_llama_vanilla"], dfs_en["WVS_EN_llama_sft"],
                       "Llama-vanilla", "Llama-SFT (on US SWOW)")

        write_vector_csv_en(tension_ids, dfs_en)
