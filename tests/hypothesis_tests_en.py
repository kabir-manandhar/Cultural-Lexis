#!/usr/bin/env python3
"""
Paired Wilcoxon tests on cue-level emotion / concreteness medians
(English data).  Outputs wilcoxon_results_en.csv
"""
import json, itertools, pandas as pd
from scipy.stats import wilcoxon

# ---------- paths ----------
JSON_PATH = "/data/gpfs/projects/punim2219/LM_with_SWOW/kabir/Cultural-Lexis/english_per_cue_metrics.json"
OUT_CSV   = "/data/projects/punim2219/LM_with_SWOW/kabir/Data/plots/wilcoxon_results_en.csv"

# ---------- model name map ----------
MODEL_MAP = {
    "ground_truth"                          : "human",
    "meta-llama/Meta-Llama-3.1-8B-Instruct" : "llama-vanilla",
    "sukai/llama_ppo_us"                    : "llama-ppo",
    "sukai/llama_swow_us"                   : "llama-sft",
    "Qwen/Qwen2.5-7B-Instruct"              : "qwen-vanilla",
    "sukai/qwen_ppo_us"                     : "qwen-ppo",
    "sukai/qwen_swow_us"                    : "qwen-sft",
}

MODEL_COLS = [
    "human",
    "llama-vanilla", "llama-ppo", "llama-sft",
    "qwen-vanilla",  "qwen-ppo",  "qwen-sft",
]

# ---------- JSON → long DataFrame ----------
def flatten_json(path: str) -> pd.DataFrame:
    with open(path, encoding="utf-8") as f:
        nested = json.load(f)

    rows = []
    for cue, models in nested.items():
        for long_name, blob in models.items():
            short = MODEL_MAP.get(long_name)
            if short is None:
                continue

            # ── Human baseline ───────────────────────────────────────────
            if long_name == "ground_truth":
                # blob = {"valence": 5.5, "arousal": 4.2, …}
                for metric, val in blob.items():
                    if val is None:
                        continue
                    for prompt in ("Complex", "Simple"):
                        rows.append(
                            dict(Cue=cue, Model=short,
                                 Prompt=prompt, Metric=metric,
                                 Value=val)
                        )
                continue

            # ── Model outputs (have Complex/Simple keys) ────────────────
            for prompt, metric_dict in blob.items():
                if metric_dict is None:
                    continue
                for metric, val in metric_dict.items():
                    if val is None:
                        continue
                    rows.append(
                        dict(Cue=cue, Model=short,
                             Prompt=prompt, Metric=metric,
                             Value=val)
                    )
    return pd.DataFrame(rows)

df = flatten_json(JSON_PATH)

# ---------- Wilcoxon tests ----------
results = []
for prompt in ["Complex", "Simple"]:
    for metric in ["valence", "arousal", "dominance", "concreteness"]:
        wide = (df[(df["Prompt"] == prompt) & (df["Metric"] == metric)]
                  .pivot(index="Cue", columns="Model", values="Value"))

        for m1, m2 in itertools.combinations(MODEL_COLS, 2):
            if not {"human", m1, m2}.issubset(wide.columns):
                results.append(dict(prompt=prompt, metric=metric,
                                    comparison=f"{m1} vs {m2}",
                                    W_stat=None, p_value=None))
                continue

            if "human" in (m1, m2):          # direct model↔human
                pair = wide[[m1, m2]].dropna()
                if len(pair) < 10:            # too few cues → skip
                    stat = p = None
                else:
                    x, y = pair[m1], pair[m2]
                    if (x == y).all():
                        stat, p = 0.0, 1.0    # identical samples
                    else:
                        stat, p = wilcoxon(x, y, alternative="two-sided")
            else:                             # closeness to human
                pair = wide[[m1, m2, "human"]].dropna()
                if len(pair) < 10:
                    stat = p = None
                else:
                    d1 = (pair[m1] - pair["human"]).abs()
                    d2 = (pair[m2] - pair["human"]).abs()
                    if (d1 == d2).all():
                        stat, p = 0.0, 1.0
                    else:
                        # alternative='greater' → test d1 > d2
                        stat, p = wilcoxon(d1, d2, alternative="greater")

            results.append(dict(prompt=prompt, metric=metric,
                                comparison=f"{m1} vs {m2}",
                                W_stat=stat, p_value=p))

pd.DataFrame(results).to_csv(OUT_CSV, index=False)
print(f"Saved Wilcoxon results → {OUT_CSV}")
