from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Dict

import pandas as pd
import spacy
from vllm import LLM, SamplingParams

# -------------------------------------------------------------------
# Paths & constants
# -------------------------------------------------------------------

# Repo-relative data root (override with env var DATA_ROOT)
# Defaults to "<repo_root>/data" assuming this file lives at: repo/language_modelling/src/swow_utils.py
DATA_ROOT = Path(os.getenv("DATA_ROOT", Path(__file__).resolve().parents[2] / "data"))

# Use repo-relative paths instead of hard-coded absolutes
SWOW_DATA_PATH = DATA_ROOT / "SWOWEN.spellchecked.27-06-2022.csv"
CUE_ASSOCIATIONS_JSON_PATH = DATA_ROOT / "cue_associations_dict.json"

# Parts of speech we consider for cultural context
# Includes nouns, adjectives, verbs, adverbs, and proper nouns
CULTURAL_POS_TAGS = {"NOUN", "PROPN", "ADJ", "VERB", "ADV"}


# -------------------------------------------------------------------
# SWOW aggregation & IO
# -------------------------------------------------------------------

def aggregate_swow_associations(
    csv_path: str | Path = SWOW_DATA_PATH,
    output_path: str | Path = CUE_ASSOCIATIONS_JSON_PATH,
) -> None:
    """
    Aggregate SWOW cue→responses into a dict and save to JSON.

    Args:
        csv_path: Path to SWOW CSV (expects columns cue, R1, R2, R3 if present).
        output_path: Path to write the aggregated JSON.
    """
    csv_path = Path(csv_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    df = df[df["cue"].notna()]

    association_dict: Dict[str, List[str]] = defaultdict(list)
    response_cols = [c for c in ("R1", "R2", "R3") if c in df.columns]

    for _, row in df.iterrows():
        cue = str(row["cue"]).strip().lower()
        responses = [row[c] for c in response_cols if pd.notna(row[c])]
        responses = [str(r).strip().lower() for r in responses if isinstance(r, str)]
        if cue:
            association_dict[cue].extend(responses)

    # Sort each cue's responses by frequency (most common first)
    sorted_dict = {
        cue: [resp for resp, _ in Counter(resps).most_common()]
        for cue, resps in association_dict.items()
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(sorted_dict, f, indent=4, ensure_ascii=False)

    print(f"✅ Aggregated and saved to {output_path}")


def load_swow_data(csv_path: str | Path = SWOW_DATA_PATH) -> pd.DataFrame:
    """
    Load the SWOW dataset.

    Args:
        csv_path: Path to the SWOW CSV.

    Returns:
        DataFrame with SWOW word associations.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"SWOW dataset not found at {csv_path}")
    return pd.read_csv(csv_path)


def get_top_associations(
    word: str,
    top_n: int = 10,
    associations_json: str | Path = CUE_ASSOCIATIONS_JSON_PATH,
    csv_path: str | Path = SWOW_DATA_PATH,
) -> List[str]:
    """
    Retrieve top-N associations for a cue from a pre-aggregated JSON.
    If the JSON does not exist, build it from the CSV.

    Args:
        word: Cue word.
        top_n: Number of associations to return.
        associations_json: Path to the aggregated JSON file.
        csv_path: Path to the SWOW CSV (used if JSON needs to be built).

    Returns:
        List of associations (possibly empty).
    """
    associations_json = Path(associations_json)
    if not associations_json.exists():
        aggregate_swow_associations(csv_path=csv_path, output_path=associations_json)

    with associations_json.open("r", encoding="utf-8") as f:
        cue_dict = json.load(f)

    return cue_dict.get(word.lower(), [])[:top_n]


# -------------------------------------------------------------------
# Keyword extraction
# -------------------------------------------------------------------

def extract_keywords_with_llm(question: str, options: str, llm: LLM) -> List[str]:
    """
    Use an LLM to extract culturally relevant keywords from a question string,
    and filter out any words that appear verbatim in the options.

    Args:
        question: The question string (may include an 'Options:' section).
        options: The raw options text string for filtering.
        llm: vLLM model instance.

    Returns:
        List of extracted keywords (lowercased, filtered).
    """
    instruction = (
        "Extract culturally significant keywords from the following question. "
        "Return them as a comma-separated list."
    )
    prompt = f"{instruction}\nQuestion: {question}"

    sampling_params = SamplingParams(max_tokens=30, temperature=0.0, top_p=1.0)

    outputs = llm.generate([prompt], sampling_params=sampling_params)
    keyword_string = outputs[0].outputs[0].text.strip()
    raw_keywords = [k.strip().lower() for k in keyword_string.split(",") if k.strip()]

    # Prepare a naive set of words from options to filter (punctuation-light)
    option_words = set(
        options.lower()
        .replace(";", " ")
        .replace("(", " ")
        .replace(")", " ")
        .replace("-", " ")
        .split()
    )

    filtered_keywords = [kw for kw in raw_keywords if kw not in option_words]
    print("🔑 Extracted Keywords (filtered):", filtered_keywords)
    return filtered_keywords


def extract_relevant_keywords(question: str) -> List[str]:
    """
    Extract keywords from the question using POS tags, excluding any words
    that appear in the options section.

    Args:
        question: The full question text (may include an 'Options:' section).

    Returns:
        List of culturally relevant keywords (lowercased, filtered).
    """
    nlp = spacy.load("en_core_web_sm")

    # Split into question and options
    parts = question.split("Options:")
    question_text = parts[0].strip()
    options_text = parts[1].strip() if len(parts) > 1 else ""

    # Basic token set from options to exclude (punctuation-light)
    option_words = set(
        options_text.lower()
        .replace(";", " ")
        .replace("(", " ")
        .replace(")", " ")
        .replace("-", " ")
        .split()
    )

    # POS tagging on the main question
    doc = nlp(question_text)
    keywords = [t.text.lower() for t in doc if t.pos_ in CULTURAL_POS_TAGS]

    # Deduplicate and filter out any keyword also present in options
    filtered = sorted({kw for kw in keywords if kw not in option_words})
    print("🔑 Filtered Keywords:", filtered)
    return filtered


# -------------------------------------------------------------------
# SWOW-based augmentation
# -------------------------------------------------------------------

def augment_with_swow(question: str, country_name: str, llm: LLM) -> str:
    """
    Augment the question with SWOW-based contextual associations.

    Steps:
      1) Extract keywords from the question (POS-based).
      2) For each keyword, fetch top associations from SWOW.
      3) Append a short cultural context note to the question.

    Args:
        question: The input question (with or without 'Options:').
        country_name: Country for context label in the appended note.
        llm: vLLM instance (present for parity; not used by this function).

    Returns:
        The augmented question string.
    """
    keywords = extract_relevant_keywords(question)
    keyword_association_map: Dict[str, List[str]] = {}

    for kw in keywords:
        assoc = get_top_associations(kw)
        if assoc:
            keyword_association_map[kw] = assoc

    if keyword_association_map:
        formatted = "\n".join(f"{k}: {', '.join(v)}" for k, v in keyword_association_map.items())
        question = (
            f"{question}\n\nFor cultural context in {country_name}, "
            f"consider these associations:\n{formatted}"
        )

    return question
