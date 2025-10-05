#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ZH Tension Set (leakage-filtered, topic-only exact match)
=========================================================

Only drop a WVS question if a training cue (from ZH *.jsonl "input")
appears as an exact substring inside the **topic span** of the Chinese
question text. We **ignore options and boilerplate**.

Outputs (in output_noleak/):
- train_cues_zh.json
- dropped_due_to_cues_zh.csv
- kept_after_filter_zh.csv
- tension_set_top50_noleak.csv
- tension_set_vectors_noleak.csv
- <tag>_topK_noleak.json
- cross_country_noleak/*.png
- NEW: per-model tables (CSV + XLSX with row colors)
  • WVS_ZH_qwen_vanilla_table_noleak.csv / .xlsx
  • WVS_ZH_qwen_sft_table_noleak.csv / .xlsx
  • WVS_ZH_llama_vanilla_table_noleak.csv / .xlsx
  • WVS_ZH_llama_sft_table_noleak.csv / .xlsx
"""

from __future__ import annotations
import json, re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance

# ----------------------- paths & config --------------------------------------
ROOT_READ   = Path("/data/gpfs/projects/punim2219/LM_with_SWOW")
SURVEY_JSON = ROOT_READ / "kabir/Data/WV_Bench/question_answer.json"

TRAIN_JSONL_DIR = Path("/data/gpfs/projects/punim2219/LM_with_SWOW/sukaih/cultural-lexis-finetune-llms/data/03_primary/llm_swow_finetune_dataset/swow_zh/train")

ROOT_WRITE  = Path("/data/projects/punim2219/LM_with_SWOW/kabir/Data/output_noleak")
PLOT_DIR    = ROOT_WRITE / "cross_country_noleak"
TENSION_CSV = ROOT_WRITE / "tension_set_top50_noleak.csv"
VECTORS_CSV = ROOT_WRITE / "tension_set_vectors_noleak.csv"
DROP_LOG    = ROOT_WRITE / "dropped_due_to_cues_zh.csv"
KEEP_LOG    = ROOT_WRITE / "kept_after_filter_zh.csv"
CUES_JSON   = ROOT_WRITE / "train_cues_zh.json"
K_TOP       = 50

MIN_CUE_LEN = 2
SINGLE_CHAR_POLICY = "drop"    # "drop" | "whitelist"
SINGLE_CHAR_WHITELIST_FILE: Optional[Path] = None

ROOT_WRITE.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILES: Dict[str, str] = {
    "WVS_ZH_qwen_vanilla":  "/data/gpfs/projects/punim2219/LM_with_SWOW/chunhua/Data/output/qwen/WVS_ZH_qwen_vanilla.json",
    "WVS_ZH_qwen_sft":      "/data/gpfs/projects/punim2219/LM_with_SWOW/chunhua/Data/output/qwen/WVS_ZH_qwen_sft.json",
    "WVS_ZH_llama_vanilla": "/data/gpfs/projects/punim2219/LM_with_SWOW/chunhua/Data/output/llama/WVS_ZH_llama_vanilla.json",
    "WVS_ZH_llama_sft":     "/data/gpfs/projects/punim2219/LM_with_SWOW/chunhua/Data/output/llama/WVS_ZH_llama_sft.json",
}

# ----------------------- distances -------------------------------------------
def js_distance(p, q) -> float:  return float(jensenshannon(p, q, base=2))
def em_distance(p, q) -> float:  return float(wasserstein_distance(p, q))
def emd_norm(p, q) -> float:     return em_distance(p, q) / (len(p)-1) if len(p) > 1 else 0.
def combo(p, q) -> float:        return 0.5*js_distance(p, q) + 0.5*emd_norm(p, q)
def truncate(a: List[float], b: List[float]) -> Tuple[List[float], List[float]]:
    m = min(len(a), len(b));  return a[:m], b[:m]

# ----------------------- cues -------------------------------------------------
def _load_whitelist(path: Optional[Path]) -> set[str]:
    if path and path.exists():
        return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
    return set()

def load_cues(train_dir: Path, min_len: int = 2) -> List[str]:
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

    sorted_cues = sorted(cues, key=lambda x: (len(x), x))
    CUES_JSON.write_text(json.dumps(sorted_cues, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Cues: kept {len(sorted_cues)} after policy (min_len={min_len}, singles={SINGLE_CHAR_POLICY}); "
          f"seen={total_seen}, single-char seen={singles_seen}. Saved → {CUES_JSON.name}")
    return sorted_cues

def build_cue_patterns_exact(cues: List[str], batch_size: int = 800) -> List[re.Pattern]:
    parts = [re.escape(c) for c in sorted(cues, key=len, reverse=True)]
    return [re.compile("|".join(parts[i:i+batch_size])) for i in range(0, len(parts), batch_size)]

# ----------------------- topic extraction (ZH) --------------------------------
_SENT_SPLIT = re.compile(r"[。！？!?]")
WS = re.compile(r"\s+")

def _before_options(s: str) -> str:
    return s.split("选项")[0]

def _last_sentence(s: str) -> str:
    parts = [p.strip() for p in _SENT_SPLIT.split(s) if p.strip()]
    return parts[-1] if parts else s.strip()

PATTERNS: List[re.Pattern] = [
    re.compile(r"您觉得(.+?)在您的生活中"),
    re.compile(r"您是否不愿意与(.+?)做邻居"),
    re.compile(r"您认为在家应着重培养孩子的哪些品质？[-—–](.+)"),
    re.compile(r"您对(.+?)的信任(?:度|程度)?"),
    re.compile(r"请问您现在是这些组织的成员吗？(.+?)$"),
    re.compile(r"您是否信仰(.+?)？"),
    re.compile(r"您是否相信(.+?)？"),
    re.compile(r"您是否认为有(.+?)？"),
]

def extract_topic_span(row: pd.Series) -> str:
    text = ""
    for fld in ("prompt_instruction_zh", "question_instruction_zh", "question_zh"):
        v = row.get(fld, None)
        if isinstance(v, str) and v.strip():
            text = v.strip()
            break
    if not text:
        return ""
    stem = _before_options(text)
    for pat in PATTERNS:
        m = pat.search(stem)
        if m:
            span = m.group(1).strip()
            return WS.sub("", span)
    return WS.sub("", _last_sentence(stem))

# ----------------------- filtering -------------------------------------------
def drop_leaky_questions_topic_only(df_all: pd.DataFrame, cue_pats: List[re.Pattern]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    df_all = df_all.copy()
    df_all["topic_zh"] = df_all.apply(extract_topic_span, axis=1)

    def find_match_in_topic(topic: str):
        if not topic:
            return None
        for pat in cue_pats:
            m = pat.search(topic)
            if m:
                return m.group(0)
        return None

    matched = df_all["topic_zh"].apply(find_match_in_topic)
    mask = matched.notna()

    if "prompt_instruction_zh" not in df_all.columns:
        df_all["prompt_instruction_zh"] = ""

    dropped = df_all.loc[mask, ["Id", "question", "prompt_instruction_zh", "topic_zh"]].copy()
    dropped["matched_cue"] = matched.loc[mask].values
    kept = df_all.loc[~mask].drop(columns=["topic_zh"], errors="ignore").copy()

    stats = {
        "total_questions": int(len(df_all)),
        "dropped": int(mask.sum()),
        "kept": int((~mask).sum()),
        "pct_dropped": round(100.0 * float(mask.sum()) / max(1, len(df_all)), 2),
    }

    dropped.to_csv(DROP_LOG, index=False)
    kept.to_csv(KEEP_LOG, index=False)
    print(f"Leakage filter (topic-only): dropped {stats['dropped']} / {stats['total_questions']} "
          f"({stats['pct_dropped']}%). Kept: {stats['kept']}. "
          f"Logs → {DROP_LOG.name}, {KEEP_LOG.name}")
    return kept.reset_index(drop=True), dropped.reset_index(drop=True), stats

# ----------------------- Phase 1 – filtered top-K ----------------------------
def generate_tension_csv_filtered() -> pd.DataFrame:
    df_all = pd.read_json(SURVEY_JSON)

    cues = load_cues(TRAIN_JSONL_DIR, min_len=MIN_CUE_LEN)
    if cues:
        cue_pats = build_cue_patterns_exact(cues)
        df, _dropped, _stats = drop_leaky_questions_topic_only(df_all, cue_pats)
    else:
        print("⚠ No cues retained after policy — skipping filtering.")
        df = df_all.copy()

    if df.empty:
        cols = ["Id","question","question_instruction","choices",
                "china_score","us_score","js_us_cn","emd_us_cn","combo"]
        pd.DataFrame(columns=cols).to_csv(TENSION_CSV, index=False)
        print("⚠ All questions filtered out. Wrote empty tension set and stopping Phase 1.")
        return pd.DataFrame(columns=["Id"])

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
    print(f"✔ filtered tension set → {TENSION_CSV.name}")
    return top

# ----------------------- Phase 2 – per-model metrics -------------------------
def model_dataframe(model_json: Path, tension_ids: set[str]) -> pd.DataFrame:
    df_model = pd.read_json(model_json)
    df_gold  = pd.read_json(SURVEY_JSON)

    df = df_gold[df_gold.Id.isin(tension_ids)].copy()
    df["model_vec"] = df_model.loc[df.index, "choice_values"]  # relies on alignment

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

# ----------------------- plots ------------------------------------------------
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

# ----------------------- export vectors --------------------------------------
def write_vector_csv(tension_ids: set[str], dfs: dict[str, pd.DataFrame]) -> None:
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
            "us_vec":         json.dumps(us_vec, ensure_ascii=False),
            "cn_vec":         json.dumps(cn_vec, ensure_ascii=False),
            "qwen_vanilla":   json.dumps(model_vec("WVS_ZH_qwen_vanilla",  qid), ensure_ascii=False),
            "llama_vanilla":  json.dumps(model_vec("WVS_ZH_llama_vanilla", qid), ensure_ascii=False),
            "qwen_sft":       json.dumps(model_vec("WVS_ZH_qwen_sft",      qid), ensure_ascii=False),
            "llama_sft":      json.dumps(model_vec("WVS_ZH_llama_sft",     qid), ensure_ascii=False),
        })

    pd.DataFrame(rows).to_csv(VECTORS_CSV, index=False)
    print(f"✔ probability-vector table → {VECTORS_CSV.name}")

# ----------------------- NEW: per-model colored tables -----------------------
def _save_table_csv_and_xlsx(df_table: pd.DataFrame, basepath: Path, closer_col: str = "closer_to") -> None:
    """
    Save CSV (no colors) and XLSX (row colors via conditional formatting).
    Colors: CN=green, US=red, TIE=yellow.
    """
    # CSV (include helper columns)
    csv_path = basepath.with_suffix(".csv")
    df_table.to_csv(csv_path, index=False)
    print(f"✔ per-model table → {csv_path.name}")

    # XLSX with row coloring
    xlsx_path = basepath.with_suffix(".xlsx")
    with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as writer:
        df_table.to_excel(writer, index=False, sheet_name="table")
        wb  = writer.book
        ws  = writer.sheets["table"]

        # Formats
        fmt_cn  = wb.add_format({"bg_color": "#C6EFCE"})  # green-ish
        fmt_us  = wb.add_format({"bg_color": "#FFC7CE"})  # red-ish
        fmt_tie = wb.add_format({"bg_color": "#FFEB9C"})  # yellow

        # Determine range (A2: lastcol lastrow)
        nrows, ncols = df_table.shape
        if nrows > 0:
            from string import ascii_uppercase
            def excel_col(idx: int) -> str:
                # 0-based idx to Excel letters
                s = ""
                idx += 1
                while idx:
                    idx, r = divmod(idx-1, 26)
                    s = chr(65 + r) + s
                return s
            last_col_letter = excel_col(ncols-1)
            data_range = f"A2:{last_col_letter}{nrows+1}"

            # Find the closer_to column letter
            closer_idx = df_table.columns.get_loc(closer_col)
            closer_letter = excel_col(closer_idx)

            # Apply conditional formatting rules over full data range
            ws.conditional_format(data_range, {
                "type": "formula",
                "criteria": f'=${closer_letter}2="CN"',
                "format": fmt_cn
            })
            ws.conditional_format(data_range, {
                "type": "formula",
                "criteria": f'=${closer_letter}2="US"',
                "format": fmt_us
            })
            ws.conditional_format(data_range, {
                "type": "formula",
                "criteria": f'=${closer_letter}2="TIE"',
                "format": fmt_tie
            })
    print(f"✔ per-model colored Excel → {xlsx_path.name}")

# --- NEW helper: build a robust meta table from SURVEY_JSON -------------------
def build_meta_from_gold_zh() -> pd.DataFrame:
    """
    Return a metadata frame with ZH fields:
      Id, question_instruction_zh, choices_zh, china_score, us_score
    If any are missing in the source JSON, create empty placeholders.
    """
    g = pd.read_json(SURVEY_JSON)

    cols_needed = ["Id", "question_instruction_zh", "choices_zh", "china_score", "us_score"]
    for c in cols_needed:
        if c not in g.columns:
            # create sensible empty defaults
            if c in ("choices_zh", "china_score", "us_score"):
                g[c] = [{}] * len(g)
            else:
                g[c] = ""

    meta = g[cols_needed].copy()
    return meta

# --- Replace write_per_model_tables with this ZH-aware version ---------------
def write_per_model_tables(dfs: dict[str, pd.DataFrame]) -> None:
    """
    For each model df in `dfs`, write a detailed table including ZH fields:
      Id, question_instruction_zh, choices_zh, us_score, china_score,
      us_vec, cn_vec, model_vec,
      js_model_us, js_model_cn, combo_us, combo_cn, bias,
      closer_to (CN/US/TIE), row_color.
    Exports both CSV and color-formatted XLSX (CN row green, US row red, TIE yellow).
    """
    df_meta = build_meta_from_gold_zh()

    for tag, dfm in dfs.items():
        dfm = dfm.copy()

        # Attach ZH meta + gold distributions
        out = dfm.merge(df_meta, on="Id", how="left")

        # Who is it closer to?
        out["closer_to"] = out.apply(
            lambda r: "CN" if r["combo_cn"] < r["combo_us"]
            else ("US" if r["combo_us"] < r["combo_cn"] else "TIE"),
            axis=1
        )
        out["row_color"] = out["closer_to"].map({"CN": "green", "US": "red", "TIE": "yellow"})

        # Serialize vectors and gold dists for readability in CSV/XLSX
        out["us_vec"]     = out["us_vec"].apply(lambda v: json.dumps(v, ensure_ascii=False))
        out["cn_vec"]     = out["cn_vec"].apply(lambda v: json.dumps(v, ensure_ascii=False))
        out["model_vec"]  = out["model_vec"].apply(lambda v: json.dumps(v, ensure_ascii=False))
        out["us_score"]   = out["us_score"].apply(lambda d: json.dumps(d, ensure_ascii=False))
        out["china_score"]= out["china_score"].apply(lambda d: json.dumps(d, ensure_ascii=False))

        # Column order focused on ZH fields + gold dists
        cols = [
            "Id",
            "question_instruction_zh",
            "choices_zh",
            "us_score", "china_score",
            "us_vec", "cn_vec", "model_vec",
            "js_model_us", "js_model_cn", "combo_us", "combo_cn", "bias",
            "closer_to", "row_color",
        ]
        # keep only existing cols to be safe
        cols = [c for c in cols if c in out.columns]
        out = out[cols]

        base = ROOT_WRITE / f"{tag}_table_noleak"
        _save_table_csv_and_xlsx(out, base)
# ----------------------- main -------------------------------------------------
if __name__ == "__main__":
    print("Rebuilding leakage-filtered ZH tension set (topic-only, exact-match)…")
    top_df = generate_tension_csv_filtered()
    tension_ids = set(top_df.Id)

    if not tension_ids:
        print("⚠ No questions left after filtering—skipping per-model metrics, plots, and vectors.")
    else:
        dfs: dict[str, pd.DataFrame] = {}
        for tag, path in OUTPUT_FILES.items():
            df = model_dataframe(Path(path), tension_ids)
            dfs[tag] = df
            out_json = ROOT_WRITE / f"{tag}_top{len(tension_ids)}_noleak.json"
            df.to_json(out_json, orient="records", indent=2)
            print(f"✔ per-model metrics → {out_json.name}")

        composite_plot("ZH_qwen_vanilla", "ZH_qwen_sft",
                       dfs["WVS_ZH_qwen_vanilla"],  dfs["WVS_ZH_qwen_sft"],
                       "Qwen-vanilla", "Qwen-SFT (on Chinese SWOW)")
        composite_plot("ZH_llama_vanilla", "ZH_llama_sft",
                       dfs["WVS_ZH_llama_vanilla"], dfs["WVS_ZH_llama_sft"],
                       "Llama-vanilla", "Llama-SFT (on Chinese SWOW)")

        write_vector_csv(tension_ids, dfs)
        write_per_model_tables(dfs)
