import spacy
import pandas as pd
import os
from vllm import LLM
from vllm import SamplingParams
import json
from collections import defaultdict, Counter
from vllm.sampling_params import GuidedDecodingParams


# Load spaCy model for POS tagging

# Define the path to the SWOW dataset
SWOW_DATA_PATH = "/data/gpfs/projects/punim2219/LM_with_SWOW/kabir/Data/SWOWEN.spellchecked.27-06-2022.csv"
CUE_ASSOCIATIONS_JSON_PATH = "/data/gpfs/projects/punim2219/LM_with_SWOW/kabir/Data/cue_associations_dict.json"


# Parts of speech we consider for cultural context
CULTURAL_POS_TAGS = {"NOUN", "PROPN", "ADJ", "VERB", "ADV"}  # Includes nouns, adjectives, verbs, and adverbs

def aggregate_swow_associations(
    csv_path: str = SWOW_DATA_PATH,
    output_path: str = CUE_ASSOCIATIONS_JSON_PATH
):
    """
    Aggregates SWOW cue-response associations and saves as a cue → list of responses dictionary.

    Args:
        csv_path (str): Path to SWOW CSV.
        output_path (str): Path to JSON output file.
    """
    df = pd.read_csv(csv_path)
    association_dict = defaultdict(list)
    df = df[df["cue"].notna()]
    response_cols = [col for col in ["R1", "R2", "R3"] if col in df.columns]

    for _, row in df.iterrows():
        cue = row["cue"].strip().lower()
        responses = [row[col] for col in response_cols if pd.notna(row[col])]
        responses = [r.strip().lower() for r in responses if isinstance(r, str)]
        association_dict[cue].extend(responses)

    sorted_dict = {
        cue: [resp for resp, _ in Counter(resps).most_common()]
        for cue, resps in association_dict.items()
    }

    with open(output_path, "w") as f:
        json.dump(sorted_dict, f, indent=4)

    print(f"✅ Aggregated and saved to {output_path}")

def extract_keywords_with_llm(question: str, options:str, llm: LLM) -> list:
    """
    Use an LLM to extract culturally relevant keywords from a question string.

    Args:
        question (str): The question string.
        llm (LLM): The vLLM model instance.

    Returns:
        list: Extracted keywords as a list of strings.
    """
    instruction = "Extract culturally significant keywords from the following question. Return them as a comma-separated list."
    prompt = f"{instruction}\nQuestion: {question}"
    

    sampling_params = SamplingParams(
        max_tokens=30,
        temperature=0.0,
        top_p=1.0
    )
    
    breakpoint()

    outputs = llm.generate([prompt], sampling_params=sampling_params)
    keyword_string = outputs[0].outputs[0].text.strip()
    raw_keywords = [k.strip().lower() for k in keyword_string.split(',') if k.strip()]

    # Prepare set of words from options to filter
    option_words = set(options.lower().replace(";", "").replace("(", "").replace(")", "").replace("-", "").split())

    # Filter out any keyword that appears in options
    filtered_keywords = [kw for kw in raw_keywords if kw not in option_words]

    print("🔑 Extracted Keywords (filtered):", filtered_keywords)
    return filtered_keywords


def extract_relevant_keywords(question: str) -> list:
    """
    Extracts relevant keywords from a question using POS tagging.
    
    Args:
        question (str): The input question text.
    
    Returns:
        list: A list of culturally relevant keywords.
    """
    nlp = spacy.load("en_core_web_sm")

    # Split into question and options
    parts = question.split("Options:")
    question_text = parts[0].strip()
    options_text = parts[1].strip() if len(parts) > 1 else ""

    # Extract option words (e.g., 'agree', 'disagree', etc.)
    option_words = set(options_text.lower().replace(";", "").replace("(", "").replace(")", "").replace("-", "").split())

    # POS tagging on the main question
    doc = nlp(question_text)
    keywords = [token.text.lower() for token in doc if token.pos_ in CULTURAL_POS_TAGS]

    # Filter out any keyword also present in the options
    filtered_keywords = [kw for kw in set(keywords) if kw not in option_words]

    print("🔑 Filtered Keywords:", filtered_keywords)
    return filtered_keywords


def load_swow_data() -> pd.DataFrame:
    """
    Loads the SWOW dataset into a pandas DataFrame.
    
    Returns:
        pd.DataFrame: A DataFrame containing SWOW word associations.
    """
    if not os.path.exists(SWOW_DATA_PATH):
        raise FileNotFoundError(f"SWOW dataset not found at {SWOW_DATA_PATH}")

    return pd.read_csv(SWOW_DATA_PATH)


def get_top_associations(word: str, top_n: int = 10) -> list:
    """
    Retrieves top N associations for a given cue word from the pre-aggregated SWOW JSON.

    Args:
        word (str): Cue word.
        top_n (int): Number of top associations to return.

    Returns:
        list: Top N associated responses or empty list if cue not found.
    """
    # If the file doesn't exist, build it
    if not os.path.exists(CUE_ASSOCIATIONS_JSON_PATH):
        aggregate_swow_associations()
        
    # Load the cue→association dict
    with open(CUE_ASSOCIATIONS_JSON_PATH, "r") as f:
        cue_dict = json.load(f)

    word = word.lower()
    return cue_dict.get(word, [])[:top_n]


def augment_with_swow(question: str, country_name: str, llm: LLM) -> str:
    """
    Augments the question with SWOW-based contextual associations using LLM-derived keywords.

    Args:
        question (str): The input question.
        country_name (str): Country for context.
        llm (LLM): The vLLM model instance for keyword extraction.

    Returns:
        str: Augmented question string.
    """
    print("Reached inside Augument Function")
    keywords = extract_relevant_keywords(question)

    keyword_association_map = {}

    for keyword in keywords:
        associations = get_top_associations(keyword)
        if associations:
            keyword_association_map[keyword] = associations

    if keyword_association_map:
        formatted_context = "\n".join(
            [f"{k}: {', '.join(v)}" for k, v in keyword_association_map.items()]
        )
        question += f"\n\nFor cultural context in {country_name}, consider these associations:\n{formatted_context}"

    print("Updated Question:", question)
    return question

