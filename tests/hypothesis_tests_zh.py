#!/usr/bin/env python3
"""
Paired Wilcoxon tests on cue-level Valence, Arousal and Concreteness
(Mandarin SWOW).  Produces wilcoxon_results_zh.csv.
"""

import json, itertools, pandas as pd
from scipy.stats import wilcoxon

# ─── paths ───────────────────────────────────────────────────────────────
JSON_PATH = "/data/projects/punim2219/LM_with_SWOW/kabir/Data/plots/per_cue_metrics_zh.json"
OUT_CSV   = "/data/projects/punim2219/LM_with_SWOW/kabir/Data/plots/wilcoxon_results_zh.csv"

# ─── model long → short mapping  (keep only those you care about) ────────
MODEL_MAP = {
    "ground_truth"                          : "human",
    "meta-llama/Meta-Llama-3.1-8B-Instruct" : "llama-vanilla",
    "sukai/llama_ppo_zh"                    : "llama-ppo",
    "sukai/llama_swow_zh"                   : "llama-sft",
    "Qwen/Qwen2.5-7B-Instruct"              : "qwen-vanilla",
    "sukai/qwen_ppo_zh"                     : "qwen-ppo",
    "sukai/qwen_swow_zh"                    : "qwen-sft",
}

MODEL_COLS = [
    "human",
    "llama-vanilla", "llama-ppo", "llama-sft",
    "qwen-vanilla",  "qwen-ppo",  "qwen-sft",
]

# ─── flatten JSON → long DataFrame ───────────────────────────────────────
def flatten_json(path: str) -> pd.DataFrame:
    with open(path, encoding="utf-8") as f:
        nested = json.load(f)

    rows = []
    for cue, models in nested.items():
        for long_name, prompts in models.items():
            short = MODEL_MAP.get(long_name)
            if short is None:
                continue   # ignore any extra models

            if long_name == "ground_truth":             # already per-prompt dict
                for prompt in ("Complex", "Simple"):
                    for metric, val in prompts[prompt].items():
                        rows.append(dict(Cue=cue, Model=short,
                                         Prompt=prompt, Metric=metric,
                                         Value=val))
            else:
                for prompt, met_dict in prompts.items():
                    for metric, val in met_dict.items():
                        if val is not None:
                            rows.append(dict(Cue=cue, Model=short,
                                             Prompt=prompt, Metric=metric,
                                             Value=val))
    return pd.DataFrame(rows)

df = flatten_json(JSON_PATH)

# ─── Wilcoxon pairwise tests ─────────────────────────────────────────────
results = []
for prompt in ("Complex", "Simple"):
    for metric in ("valence", "arousal", "concreteness"):
        wide = (df[(df["Prompt"] == prompt) & (df["Metric"] == metric)]
                  .pivot(index="Cue", columns="Model", values="Value"))

        for m1, m2 in itertools.combinations(MODEL_COLS, 2):
            if not {"human", m1, m2}.issubset(wide.columns):
                results.append(dict(prompt=prompt, metric=metric,
                                    comparison=f"{m1} vs {m2}",
                                    W_stat=None, p_value=None))
                continue

            # ---------- decide what to compare ----------
            if "human" in (m1, m2):           # direct model ↔ human
                pair = wide[[m1, m2]].dropna()
                if len(pair) < 10:
                    stat, p = None, None        # too few cues, skip
                elif (pair[m1] == pair[m2]).all():
                    stat, p = 0.0, 1.0          # identical samples
                else:
                    stat, p = wilcoxon(pair[m1], pair[m2],
                                       alternative="two-sided")
            else:                              # which model is closer to human?
                pair = wide[[m1, m2, "human"]].dropna()
                if len(pair) < 10:
                    stat, p = None, None
                else:
                    d1 = (pair[m1] - pair["human"]).abs()
                    d2 = (pair[m2] - pair["human"]).abs()
                    if (d1 == d2).all():
                        stat, p = 0.0, 1.0
                    else:
                        # alternative='greater' asks if d1 > d2 (m2 closer)
                        stat, p = wilcoxon(d1, d2, alternative="greater")

            results.append(dict(prompt=prompt, metric=metric,
                                comparison=f"{m1} vs {m2}",
                                W_stat=stat, p_value=p))

pd.DataFrame(results).to_csv(OUT_CSV, index=False)
print(f"Saved Wilcoxon results → {OUT_CSV}")
