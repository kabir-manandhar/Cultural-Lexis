from __future__ import annotations

import csv
import json
import re
import string
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Dict, List, Optional, Tuple

import pandas as pd

# -----------------------------------------------------------------------------
# Defaults & small helpers
# -----------------------------------------------------------------------------

# Repo-relative data root with optional override via DATA_ROOT
DATA_ROOT = Path(Path(__file__).resolve().parents[2] / "data")

# Punctuation sets
PUNCS_EN = string.punctuation + "“”‘’"
PUNCS_ZH = string.punctuation + "“”‘’，。？！；：（）「」『』—…、《》【】"

# Split tokens if OOV (space, hyphen, slash)
_SPLIT = re.compile(r"[\s\-/]+")

def _pre_en(tok: str) -> str:
    """Lower + strip punctuation (keep simple to avoid external models)."""
    return tok.lower().strip(PUNCS_EN)

def _pre_zh(tok: str) -> str:
    """Strip mixed punctuation; Chinese words are not lemmatized."""
    return tok.strip(PUNCS_ZH)

def _ensure_path(p: Optional[Path], desc: str) -> Path:
    if p is None:
        raise FileNotFoundError(
            f"Path for {desc} was not provided and no default exists. "
            "Please pass an explicit file path."
        )
    if not p.exists():
        raise FileNotFoundError(f"{desc} not found at: {p}")
    return p

def _default_paths(language: str) -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
    """
    Best-effort defaults under DATA_ROOT for (pickle, vad, concr).
    You can override all via function args / CLI flags.
    """
    lang = language.lower()
    if lang.startswith("zh"):
        return (
            (DATA_ROOT / "swow_zh_results_processed.pkl"),
            (DATA_ROOT / "VAD-zh-emot-submit.csv"),
            (DATA_ROOT / "Concreteness-ZH.xlsx"),
        )
    else:
        return (
            (DATA_ROOT / "swow_en_results_processed.pkl"),
            (DATA_ROOT / "VAD-emot-submit.csv"),
            (DATA_ROOT / "Concreteness_ratings_Brysbaert_et_al_BRM.txt"),
        )

# -----------------------------------------------------------------------------
# SWOW pickle loader
# -----------------------------------------------------------------------------

def load_swow_pickle(path: Path) -> Dict[str, dict]:
    """
    Expected columns:
      - "Cue Word"
      - "Ground Truth Associated Words" (comma or '，' separated)
      - "Model Type", "Prompt Type" (for models)
      - "Parsed Associated Words"
    Returns: cue -> { ground_truth: [w], models: {model: {prompt: [w]}} }
    """
    df = pd.read_pickle(path)
    data = {}
    for cue, grp in df.groupby("Cue Word"):
        gt_raw = grp.iloc[0].get("Ground Truth Associated Words", "") or ""
        if "，" in str(gt_raw):
            gt = [w.strip() for w in str(gt_raw).split("，") if w.strip()]
        else:
            gt = [w.strip() for w in str(gt_raw).split(",") if w.strip()]
        gt = gt[:10]

        models: Dict[str, Dict[str, List[str]]] = {}
        if {"Model Type", "Prompt Type"}.issubset(grp.columns):
            for (m, p), sub in grp.groupby(["Model Type", "Prompt Type"]):
                words: List[str] = []
                for cell in sub.get("Parsed Associated Words", pd.Series(dtype=object)).dropna():
                    words.extend([w.strip() for w in str(cell).split(",") if w.strip()])
                models.setdefault(str(m), {})[str(p)] = words[:10]

        data[str(cue)] = {"ground_truth": gt, "models": models}

    return data

# -----------------------------------------------------------------------------
# Lexicon loaders
# -----------------------------------------------------------------------------

def load_vad_en(path: Path) -> Dict[str, Tuple[float, float, float]]:
    """English VAD: CSV columns Word, V.Mean.Sum, A.Mean.Sum, D.Mean.Sum (1–9)."""
    vad: Dict[str, Tuple[float, float, float]] = {}
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            w = row["Word"].strip().lower()
            try:
                v = float(row["V.Mean.Sum"]); a = float(row["A.Mean.Sum"]); d = float(row["D.Mean.Sum"])
            except Exception:
                continue
            vad[w] = (v, a, d)
    return vad

def load_conc_en(path: Path) -> Dict[str, float]:
    """Brysbaert concreteness TSV: word -> rating in [1,5] (1=abstract, 5=concrete)."""
    conc: Dict[str, float] = {}
    with path.open(encoding="utf-8") as f:
        next(f, None)  # skip header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            w = parts[0].strip().lower()
            try:
                conc[w] = float(parts[2])
            except Exception:
                continue
    return conc

def _rescale_valence_zh(v_zh: float) -> float:
    # Chinese valence −3…+3  → English-like 1…9
    return (v_zh + 3.0) / 6.0 * 8.0 + 1.0

def _rescale_arousal_zh(a_zh: float) -> float:
    # Chinese arousal 0…4  → English-like 1…9
    return a_zh / 4.0 * 8.0 + 1.0

def load_vad_zh(path: Path) -> Dict[str, Tuple[float, float]]:
    """Chinese VAD CSV columns: Word, Valence_Mean (−3..+3), Arousal_Mean (0..4)."""
    df = pd.read_csv(path)
    vad: Dict[str, Tuple[float, float]] = {}
    for _, row in df.iterrows():
        w = str(row.get("Word", "")).strip()
        if not w:
            continue
        try:
            v_res = _rescale_valence_zh(float(row["Valence_Mean"]))
            a_res = _rescale_arousal_zh(float(row["Arousal_Mean"]))
        except Exception:
            continue
        vad[w] = (v_res, a_res)
    return vad

def load_conc_zh(path: Path) -> Dict[str, float]:
    """
    Chinese concreteness Excel:
      * Mean column often named like "...Mean..."
      * Original: 1 = concrete, 5 = abstract
      * Invert to English orientation: 1 = abstract, 5 = concrete
    """
    df = pd.read_excel(path)
    mean_col = next(
        (c for c in df.columns if "mean" in c.lower() and "word" not in c.lower()),
        None,
    )
    if mean_col is None:
        for c in df.columns:
            if c.lower() != "word" and pd.api.types.is_numeric_dtype(df[c]):
                mean_col = c; break
    if mean_col is None:
        raise ValueError("Could not locate a numeric mean column in the Chinese concreteness file.")

    conc: Dict[str, float] = {}
    for _, r in df.iterrows():
        w = str(r.get("Word", "")).strip()
        if not w:
            continue
        try:
            conc[w] = 6.0 - float(r[mean_col])  # invert to English orientation
        except Exception:
            continue
    return conc

# -----------------------------------------------------------------------------
# Metrics (medians only)
# -----------------------------------------------------------------------------

def compute_metrics(
    raws: List[str],
    language: str,
    vad_en: Optional[Dict[str, Tuple[float, float, float]]] = None,
    vad_zh: Optional[Dict[str, Tuple[float, float]]] = None,
    conc: Optional[Dict[str, float]] = None,
) -> Dict[str, Optional[float]]:
    """Median Valence/Arousal/(Dominance if EN) and Concreteness for tokens."""
    lang = language.lower()
    pre = _pre_zh if lang.startswith("zh") else _pre_en

    vs: List[float] = []
    ars: List[float] = []
    doms: List[float] = []
    cs: List[float] = []

    for raw in raws:
        w = pre(raw)
        if not w:
            continue
        if lang.startswith("zh"):
            if vad_zh and w in vad_zh:
                v, a = vad_zh[w]
                vs.append(v); ars.append(a)
        else:
            if vad_en and w in vad_en:
                v, a, d = vad_en[w]
                vs.append(v); ars.append(a); doms.append(d)
        if conc and w in conc:
            cs.append(conc[w])

    return {
        "valence": median(vs) if vs else None,
        "arousal": median(ars) if ars else None,
        "dominance": (median(doms) if doms else None) if not lang.startswith("zh") else None,
        "concreteness": median(cs) if cs else None,
    }

def compute_threshold_percents(
    raws: List[str],
    language: str,
    conc: Dict[str, float],
    threshold: float,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """%Concrete / %Abstract / %Unknown given concreteness threshold (1–5)."""
    pre = _pre_zh if language.lower().startswith("zh") else _pre_en
    total = len(raws)
    if total == 0:
        return None, None, None

    conc_cnt = abs_cnt = unk_cnt = 0
    for raw in raws:
        w = pre(raw)
        if not w or w not in conc:
            unk_cnt += 1
        elif conc[w] >= threshold:
            conc_cnt += 1
        else:
            abs_cnt += 1

    return (
        conc_cnt / total * 100.0,
        abs_cnt / total * 100.0,
        unk_cnt / total * 100.0,
    )

# -----------------------------------------------------------------------------
# Aggregation & coverage (medians)
# -----------------------------------------------------------------------------

def build_word_lists(data: Dict[str, dict]) -> Dict[str, List[str]]:
    """
    Returns category -> list of raw tokens (duplicates kept).
    category:
       ground_truth
       {model}|Complex
       {model}|Simple
    """
    wl = defaultdict(list)
    for pay in data.values():
        wl["ground_truth"].extend(pay["ground_truth"])
        for m, outs in pay["models"].items():
            for p in ("Complex", "Simple"):
                wl[f"{m}|{p}"].extend(outs.get(p, []))
    return wl

def type_coverage(
    wl: Dict[str, List[str]],
    language: str,
    lexicon_words: set[str],
) -> Dict[str, Optional[float]]:
    pre = _pre_zh if language.lower().startswith("zh") else _pre_en
    out: Dict[str, Optional[float]] = {}
    for cat, raws in wl.items():
        if not raws:
            out[cat] = None
            continue
        types = {pre(w) for w in raws if pre(w)}
        matched = {w for w in types if w in lexicon_words}
        out[cat] = (len(matched) / len(types) * 100.0) if types else None
    return out

def token_coverage(
    wl: Dict[str, List[str]],
    language: str,
    lexicon_words: set[str],
) -> Dict[str, Optional[float]]:
    pre = _pre_zh if language.lower().startswith("zh") else _pre_en
    out: Dict[str, Optional[float]] = {}
    for cat, raws in wl.items():
        if not raws:
            out[cat] = None
            continue
        matched = sum(1 for w in raws if pre(w) in lexicon_words)
        out[cat] = matched / len(raws) * 100.0
    return out

def aggregate(
    data: Dict[str, dict],
    language: str,
    vad_en: Optional[Dict[str, Tuple[float, float, float]]] = None,
    vad_zh: Optional[Dict[str, Tuple[float, float]]] = None,
    conc: Optional[Dict[str, float]] = None,
) -> Tuple[Dict[str, dict], Dict[str, Dict[str, float]]]:
    """
    per_cue[cue][category][prompt] → metric → value
        category = ground_truth | model name
        prompt   = Complex | Simple (only for models)
    Returns (per_cue, global_means)
    """
    sums = defaultdict(lambda: defaultdict(float))
    counts = defaultdict(lambda: defaultdict(int))
    per_cue: Dict[str, dict] = {}

    for cue, pay in data.items():
        per_cue[cue] = {}

        # Ground truth
        gt_metrics = compute_metrics(pay["ground_truth"], language, vad_en=vad_en, vad_zh=vad_zh, conc=conc)
        per_cue[cue]["ground_truth"] = gt_metrics
        for m, v in gt_metrics.items():
            if v is not None:
                sums["ground_truth"][m] += v
                counts["ground_truth"][m] += 1

        # Models
        for model, outs in pay["models"].items():
            per_cue[cue].setdefault(model, {})
            for prompt in ("Complex", "Simple"):
                mets = compute_metrics(outs.get(prompt, []), language, vad_en=vad_en, vad_zh=vad_zh, conc=conc)
                per_cue[cue][model][prompt] = mets
                for k, v in mets.items():
                    if v is not None:
                        key = f"{prompt}_{k}"
                        sums[model][key] += v
                        counts[model][key] += 1

    global_means = {cat: {m: (sums[cat][m] / counts[cat][m]) for m in sums[cat]} for cat in sums}
    return per_cue, global_means

# -----------------------------------------------------------------------------
# Export/enrichment (per-word lists + medians)
# -----------------------------------------------------------------------------

def lookup_scores(
    term: str,
    language: str,
    vad_en: Optional[Dict[str, Tuple[float, float, float]]],
    vad_zh: Optional[Dict[str, Tuple[float, float]]],
    conc: Dict[str, float],
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Return (valence_1to9, arousal_1to9, concreteness_1to5) for a term.
    Strategy:
      1) Exact lookup on the preprocessed whole term.
      2) If OOV and term has spaces/hyphens/slashes, split and take median over matched tokens.
    """
    pre = _pre_zh if language.lower().startswith("zh") else _pre_en
    t = pre(term)

    v = a = c = None
    if language.lower().startswith("zh"):
        if t in (vad_zh or {}) or t in conc:
            if vad_zh and t in vad_zh:
                v, a = vad_zh[t]
            if t in conc:
                c = conc[t]
            return v, a, c
    else:
        if t in (vad_en or {}) or t in conc:
            if vad_en and t in vad_en:
                v, a, _ = vad_en[t]
            if t in conc:
                c = conc[t]
            return v, a, c

    toks = [pre(x) for x in _SPLIT.split(term) if x.strip()]
    if len(toks) > 1:
        v_list, a_list, c_list = [], [], []
        for tok in toks:
            if language.lower().startswith("zh"):
                if vad_zh and tok in vad_zh:
                    vv, aa = vad_zh[tok]
                    v_list.append(vv); a_list.append(aa)
            else:
                if vad_en and tok in vad_en:
                    vv, aa, _ = vad_en[tok]
                    v_list.append(vv); a_list.append(aa)
            if tok in conc:
                c_list.append(conc[tok])
        v = median(v_list) if v_list else None
        a = median(a_list) if a_list else None
        c = median(c_list) if c_list else None
        if v is not None or a is not None or c is not None:
            return v, a, c

    return None, None, None

def score_list(
    raws: List[str],
    language: str,
    vad_en: Optional[Dict[str, Tuple[float, float, float]]],
    vad_zh: Optional[Dict[str, Tuple[float, float]]],
    conc: Dict[str, float],
) -> dict:
    """
    Produce parallel lists for words and their scores, plus medians.
    Only V/A/C are exported (dominance is omitted to match previous export schema).
    """
    words = list(raws)  # preserve original order
    v_list, a_list, c_list = [], [], []

    for t in words:
        v, a, c = lookup_scores(t, language=language, vad_en=vad_en, vad_zh=vad_zh, conc=conc)
        v_list.append(v)
        a_list.append(a)
        c_list.append(c)

    mv = median([x for x in v_list if x is not None]) if any(x is not None for x in v_list) else None
    ma = median([x for x in a_list if x is not None]) if any(x is not None for x in a_list) else None
    mc = median([x for x in c_list if x is not None]) if any(x is not None for x in c_list) else None

    return {
        "words": words,
        "valence": v_list,
        "arousal": a_list,
        "concreteness": c_list,
        "medians": {"valence": mv, "arousal": ma, "concreteness": mc}
    }

def enrich_lists(
    data: dict,
    language: str,
    vad_en: Optional[Dict[str, Tuple[float, float, float]]],
    vad_zh: Optional[Dict[str, Tuple[float, float]]],
    conc: Dict[str, float],
) -> dict:
    """
    For each cue:
      ground_truth: {words, valence, arousal, concreteness, medians}
      models: model → { "Complex": {...}, "Simple": {...} }
    """
    out = {}
    for cue, payload in data.items():
        cue_obj = {
            "ground_truth": score_list(payload.get("ground_truth", []), language, vad_en, vad_zh, conc),
            "models": {}
        }
        for model, outs in payload.get("models", {}).items():
            cue_obj["models"][model] = {
                "Complex": score_list(outs.get("Complex", []), language, vad_en, vad_zh, conc),
                "Simple":  score_list(outs.get("Simple",  []), language, vad_en, vad_zh, conc),
            }
        out[cue] = cue_obj
    return out

def extract_medians_only(enriched: dict) -> dict:
    """
    Compact JSON with medians only, per cue / category / prompt.
    """
    slim = {}
    for cue, block in enriched.items():
        slim[cue] = {"ground_truth": block["ground_truth"]["medians"], "models": {}}
        for model, mp in block["models"].items():
            slim[cue]["models"][model] = {p: mp[p]["medians"] for p in ("Complex", "Simple")}
    return slim

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def run_affect_concreteness(
    language: str,
    swow_pickle: Optional[Path] = None,
    vad_file: Optional[Path] = None,
    concreteness_file: Optional[Path] = None,
    threshold: float = 3.0,
    write_json: Optional[Path] = None,
    *,
    out_enriched: Optional[Path] = None,
    out_medians: Optional[Path] = None,
) -> Dict[str, object]:
    """
    Unified runner for EN/ZH emotion (valence/arousal/(dominance)) and concreteness analysis.

    Side-effects (optional):
      - write_json: per-cue medians detail used by the paper's tables (old behavior)
      - out_enriched: full per-word V/A/C lists for GT and each model×prompt
      - out_medians: medians-only JSON (compact)
    """
    lang = language.lower()
    d_swow, d_vad, d_conc = _default_paths(lang)
    swow_pickle = _ensure_path(swow_pickle or d_swow, "SWOW processed pickle")
    vad_file = _ensure_path(vad_file or d_vad, "VAD lexicon")
    concreteness_file = _ensure_path(concreteness_file or d_conc, "Concreteness lexicon")

    # Load inputs
    data = load_swow_pickle(swow_pickle)
    if lang.startswith("zh"):
        vad_zh = load_vad_zh(vad_file)
        vad_en = None
        conc = load_conc_zh(concreteness_file)
    else:
        vad_en = load_vad_en(vad_file)
        vad_zh = None
        conc = load_conc_en(concreteness_file)

    # Aggregate medians for console/table
    per_cue, global_means = aggregate(data, language=lang, vad_en=vad_en, vad_zh=vad_zh, conc=conc)

    # Coverage
    wl = build_word_lists(data)
    vad_words = set((vad_zh or {}).keys()) if lang.startswith("zh") else set((vad_en or {}).keys())
    conc_words = set(conc.keys())

    vad_type = type_coverage(wl, language=lang, lexicon_words=vad_words)
    vad_token = token_coverage(wl, language=lang, lexicon_words=vad_words)
    conc_type = type_coverage(wl, language=lang, lexicon_words=conc_words)
    conc_token = token_coverage(wl, language=lang, lexicon_words=conc_words)

    # % Conc / % Abs / % Unk
    percents = {}
    for cat, raws in wl.items():
        c, a, u = compute_threshold_percents(raws, language=lang, conc=conc, threshold=threshold)
        percents[cat] = {"conc%": c, "abs%": a, "unk%": u}

    # Optional legacy per-cue dump (medians detail by category/prompt)
    if write_json:
        write_json.parent.mkdir(parents=True, exist_ok=True)
        write_json.write_text(json.dumps(per_cue, indent=2, ensure_ascii=False), encoding="utf-8")

    # Optional enriched exports
    enriched = None
    if out_enriched or out_medians:
        enriched = enrich_lists(data, language=lang, vad_en=vad_en, vad_zh=vad_zh, conc=conc)

    if out_enriched:
        out_enriched.parent.mkdir(parents=True, exist_ok=True)
        out_enriched.write_text(json.dumps(enriched, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"✔ Full enriched JSON → {out_enriched}")

    if out_medians:
        med_only = extract_medians_only(enriched if enriched is not None else enrich_lists(data, language=lang, vad_en=vad_en, vad_zh=vad_zh, conc=conc))
        out_medians.parent.mkdir(parents=True, exist_ok=True)
        out_medians.write_text(json.dumps(med_only, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"✔ Medians-only JSON → {out_medians}")

    # Minimal console view
    print("\n=== Global Means ===")
    for cat, mets in global_means.items():
        pretty = ", ".join(f"{k}: {v:.3f}" for k, v in sorted(mets.items()))
        print(f"- {cat}: {pretty}")

    print("\n=== Coverage (type/token) ===")
    cats = sorted(wl.keys())
    for cat in cats:
        vt = vad_type.get(cat); vtok = vad_token.get(cat)
        ct = conc_type.get(cat); ctok = conc_token.get(cat)
        print(
            f"- {cat:40s}  VAD(type)={vt:.1f}%  VAD(tok)={vtok:.1f}%   "
            f"CONC(type)={ct:.1f}%  CONC(tok)={ctok:.1f}%"
        )

    print(f"\n=== %Conc / %Abs / %Unk (threshold={threshold}) ===")
    for cat in cats:
        p = percents[cat]
        def _fmt(x): return f"{x:.1f}%" if x is not None else "N/A"
        print(f"- {cat:40s}  Conc={_fmt(p['conc%'])}  Abs={_fmt(p['abs%'])}  Unk={_fmt(p['unk%'])}")

    return {
        "per_cue": per_cue,
        "global_means": global_means,
        "coverage": {
            "vad_type": vad_type, "vad_token": vad_token,
            "conc_type": conc_type, "conc_token": conc_token,
        },
        "percents": percents,
        "enriched_written": {
            "full": str(out_enriched) if out_enriched else None,
            "medians": str(out_medians) if out_medians else None,
        }
    }
