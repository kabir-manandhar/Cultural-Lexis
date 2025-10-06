# model_analysis/post/hypothesis.py
from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from scipy.stats import wilcoxon


# --------- Defaults ---------

# Friendly short names; unknown models are slugged on the fly.
DEFAULT_MODEL_MAP: Dict[str, str] = {
    "ground_truth": "human",
    # Llama
    "meta-llama/Meta-Llama-3.1-8B-Instruct": "llama-vanilla",
    "sukai/llama_ppo_us": "llama-ppo",
    "sukai/llama_swow_us": "llama-sft",
    "sukai/llama_ppo_zh": "llama-ppo",
    "sukai/llama_swow_zh": "llama-sft",
    # Qwen
    "Qwen/Qwen2.5-7B-Instruct": "qwen-vanilla",
    "sukai/qwen_ppo_us": "qwen-ppo",
    "sukai/qwen_swow_us": "qwen-sft",
    "sukai/qwen_ppo_zh": "qwen-ppo",
    "sukai/qwen_swow_zh": "qwen-sft",
}

# Display ordering (if present)
ORDER = {
    "human": 0,
    "llama-vanilla": 1, "llama-ppo": 2, "llama-sft": 3,
    "qwen-vanilla": 4,  "qwen-ppo": 5,  "qwen-sft": 6,
}

# Which metrics to consider per language
LANG_METRICS = {
    "en": ("valence", "arousal", "dominance", "concreteness"),
    "zh": ("valence", "arousal", "concreteness"),
}


# --------- Helpers ---------

def _slug_model(name: str) -> str:
    return name.lower().replace("/", "-").replace("_", "-")


def _short_name(long: str, mapping: Dict[str, str]) -> str:
    return mapping.get(long, _slug_model(long))


def _flatten_per_cue_json(
    json_path: Path,
    language: str,
    model_map: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Robustly flattens the per-cue medians JSON into long-form rows:
    Cue, Model, Prompt, Metric, Value

    Accepts both shapes:
      A) per_cue[cue]["ground_truth"] = {metric: value, ...}
         per_cue[cue][MODEL]["Complex"/"Simple"] = {metric: value, ...}

      B) (older ZH variant) per_cue[cue]["ground_truth"]["Complex"/"Simple"] = {...}
    """
    model_map = model_map or DEFAULT_MODEL_MAP
    lang = "zh" if language.lower().startswith("zh") else "en"
    allowed_metrics = set(LANG_METRICS[lang])

    with json_path.open(encoding="utf-8") as f:
        nested = json.load(f)

    rows: List[dict] = []
    for cue, blob in nested.items():
        for long_name, payload in blob.items():
            short = _short_name(long_name, model_map)

            # --- ground truth ---
            if long_name == "ground_truth":
                # Case B: prompts exist under ground_truth
                if isinstance(payload, dict) and {"Complex", "Simple"} <= set(payload.keys()):
                    for prompt in ("Complex", "Simple"):
                        metrics = payload.get(prompt, {}) or {}
                        for metric, val in metrics.items():
                            if metric in allowed_metrics and val is not None:
                                rows.append(dict(Cue=cue, Model=short, Prompt=prompt, Metric=metric, Value=val))
                else:
                    # Case A: metrics dict (duplicate across prompts)
                    metrics = payload or {}
                    for metric, val in metrics.items():
                        if metric in allowed_metrics and val is not None:
                            for prompt in ("Complex", "Simple"):
                                rows.append(dict(Cue=cue, Model=short, Prompt=prompt, Metric=metric, Value=val))
                continue

            # --- models ---
            if isinstance(payload, dict) and any(p in payload for p in ("Complex", "Simple")):
                # Normal: prompts present
                for prompt, metrics in payload.items():
                    if prompt not in ("Complex", "Simple") or metrics is None:
                        continue
                    for metric, val in (metrics or {}).items():
                        if metric in allowed_metrics and val is not None:
                            rows.append(dict(Cue=cue, Model=short, Prompt=prompt, Metric=metric, Value=val))
            else:
                # Rare: model has no prompts → duplicate across both prompts
                metrics = payload or {}
                for metric, val in metrics.items():
                    if metric in allowed_metrics and val is not None:
                        for prompt in ("Complex", "Simple"):
                            rows.append(dict(Cue=cue, Model=short, Prompt=prompt, Metric=metric, Value=val))

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No usable rows parsed from {json_path} for language='{language}'.")
    return df


def _ordered_models(models: Iterable[str]) -> List[str]:
    return sorted(set(models), key=lambda m: ORDER.get(m, 999))


@dataclass
class WilcoxonConfig:
    language: str
    min_pairs: int = 10
    include_direct: bool = True      # model vs human
    include_closeness: bool = True   # compare |model-human| (is m2 closer than m1?)


# --------- Main API ---------

def run_hypothesis_tests(
    per_cue_json: Path,
    out_csv: Path,
    *,
    language: str,
    model_map: Optional[Dict[str, str]] = None,
    config: Optional[WilcoxonConfig] = None,
) -> pd.DataFrame:
    """
    Performs paired Wilcoxon tests on cue-level medians.

    per_cue_json : JSON created by affect_conc.run_affect_concreteness(..., write_json=...)
    out_csv      : where to save the results CSV
    language     : "en" or "zh" (determines metric set)
    model_map    : optional mapping long-name → short-name
    config       : optional overrides (min_pairs, include_direct, include_closeness)

    Returns a DataFrame with columns:
      prompt, metric, comparison, W_stat, p_value
    """
    cfg = config or WilcoxonConfig(language=language)
    df = _flatten_per_cue_json(Path(per_cue_json), language=language, model_map=model_map)

    results: List[dict] = []
    all_models = _ordered_models(df["Model"].unique())
    metrics = LANG_METRICS["zh" if cfg.language.lower().startswith("zh") else "en"]

    for prompt in ("Complex", "Simple"):
        for metric in metrics:
            wide = (
                df[(df["Prompt"] == prompt) & (df["Metric"] == metric)]
                .pivot(index="Cue", columns="Model", values="Value")
            )

            # Build the comparison set from the available columns
            present_models = [m for m in all_models if m in wide.columns]
            if "human" not in present_models:
                # nothing meaningful to compare
                continue

            for m1, m2 in itertools.combinations(present_models, 2):
                # Direct model↔human
                if cfg.include_direct and ("human" in (m1, m2)):
                    pair = wide[[m1, m2]].dropna()
                    if len(pair) < cfg.min_pairs:
                        stat = p = None
                    else:
                        x, y = pair[m1], pair[m2]
                        if (x == y).all():
                            stat, p = 0.0, 1.0
                        else:
                            stat, p = wilcoxon(x, y, alternative="two-sided")

                    results.append(dict(
                        prompt=prompt, metric=metric,
                        comparison=f"{m1} vs {m2}",
                        W_stat=stat, p_value=p
                    ))
                    continue

                # Closeness to human (which is nearer?)
                if cfg.include_closeness and "human" in wide.columns:
                    cols = [c for c in (m1, m2, "human") if c in wide.columns]
                    if len(cols) < 3:
                        results.append(dict(
                            prompt=prompt, metric=metric,
                            comparison=f"{m1} vs {m2}",
                            W_stat=None, p_value=None
                        ))
                        continue

                    pair = wide[cols].dropna()
                    if len(pair) < cfg.min_pairs:
                        stat = p = None
                    else:
                        d1 = (pair[m1] - pair["human"]).abs()
                        d2 = (pair[m2] - pair["human"]).abs()
                        if (d1 == d2).all():
                            stat, p = 0.0, 1.0
                        else:
                            # alternative='greater' asks if d1 > d2 (i.e., m2 closer to human than m1)
                            stat, p = wilcoxon(d1, d2, alternative="greater")

                    results.append(dict(
                        prompt=prompt, metric=metric,
                        comparison=f"{m1} vs {m2}",
                        W_stat=stat, p_value=p
                    ))

    out_df = pd.DataFrame(results)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    print(f"Saved Wilcoxon results → {out_csv}")
    return out_df
