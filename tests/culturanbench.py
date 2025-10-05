import os
import json
import torch
import gc
from collections import defaultdict

from tqdm import tqdm
from datasets import load_dataset
from vllm import LLM
from vllm.sampling_params import SamplingParams, GuidedDecodingParams

# -----------------------------------------------------------------------------
# 0. Zero‐shot instruction templates
# -----------------------------------------------------------------------------
BASE_INSTRUCTION = """\
To answer the following multiple-choice question, you should choose one option only among A, B, C, D.
Instruction: You must select one option among A, B, C, D. Do not output anything else."""
MULTI_SELECT_INSTRUCTION = "Select the options with all applicable statements."

# -----------------------------------------------------------------------------
# 1. Authentication & model utilities
# -----------------------------------------------------------------------------
def get_hf_auth_token(
    token_file_path: str = "/data/gpfs/projects/punim2219/LM_with_SWOW/kabir/Cultural-Lexis/language_modelling/hf_token.txt"
) -> str:
    """
    Reads the Hugging Face token from a text file.
    """
    if not os.path.exists(token_file_path):
        raise FileNotFoundError(f"Token file not found: {token_file_path}")
    with open(token_file_path, "r") as f:
        return f.readline().strip()

def clear_gpu_memory():
    """
    Clear GPU memory.
    """
    torch.cuda.empty_cache()
    gc.collect()

def load_llm(model_path: str, dtype: str = "bfloat16") -> LLM:
    """
    Load a vLLM model with GPU support.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please check your GPU setup.")
    return LLM(model=model_path, dtype=dtype)

# -----------------------------------------------------------------------------
# 2. Prompt builder & LLM runner
# -----------------------------------------------------------------------------
def build_prompt(ex: dict) -> str:
    """
    Build the zero‐shot prompt for a CulturalBench example.
    ex must contain:
      - prompt_question
      - prompt_option_a, _b, _c, _d
      - optionally 'multi_select' (bool)
    """
    parts = [BASE_INSTRUCTION]
    if ex.get("multi_select", False):
        parts.append(MULTI_SELECT_INSTRUCTION)
    parts.append(f"\nQuestion: {ex['prompt_question']}")
    parts.append(f"A. {ex['prompt_option_a']}")
    parts.append(f"B. {ex['prompt_option_b']}")
    parts.append(f"C. {ex['prompt_option_c']}")
    parts.append(f"D. {ex['prompt_option_d']}")
    return "\n".join(parts)

def configure_params() -> SamplingParams:
    """
    Configure sampling so that the model can only output exactly one of "A","B","C","D".
    """
    guided = GuidedDecodingParams(choice=["A", "B", "C", "D"])g
    return SamplingParams(
        guided_decoding=guided,
        max_tokens=1,
        temperature=0.0    )

def predict_choice(llm: LLM, prompt: str) -> str:
    """
    Run vLLM on the given prompt and return exactly the uppercase letter "A"/"B"/"C"/"D".
    """
    out = llm.generate(prompts=prompt, sampling_params=configure_params())
    return out[0].outputs[0].text.strip().upper()

# -----------------------------------------------------------------------------
# 3. Country→Region mapping (hard‐coded from the CulturalBench “Statistics” table)
# -----------------------------------------------------------------------------
COUNTRY_TO_REGION = {
    # North America (N = 27)
    "Canada":                   "North America",
    "United States":            "North America",

    # South America (N = 150)
    "Argentina":                "South America",
    "Brazil":                   "South America",
    "Chile":                    "South America",
    "Mexico":                   "South America",
    "Peru":                     "South America",

    # East Europe (N = 115)
    "Czech Republic":           "East Europe",
    "Poland":                   "East Europe",
    "Romania":                  "East Europe",
    "Ukraine":                  "East Europe",
    "Russia":                   "East Europe",

    # South Europe (N = 76)
    "Spain":                    "South Europe",
    "Italy":                    "South Europe",

    # West Europe (N = 96)
    "France":                   "West Europe",
    "Germany":                  "West Europe",
    "Netherlands":              "West Europe",
    "United Kingdom":           "West Europe",

    # Africa (N = 134)
    "Egypt":                    "Africa",
    "Morocco":                  "Africa",
    "Nigeria":                  "Africa",
    "South Africa":             "Africa",
    "Zimbabwe":                 "Africa",

    # Middle East/West Asia (N = 127)
    "Iran":                     "Middle East/West Asia",
    "Israel":                   "Middle East/West Asia",
    "Lebanon":                  "Middle East/West Asia",
    "Saudi Arabia":             "Middle East/West Asia",
    "Turkey":                   "Middle East/West Asia",

    # South Asia (N = 106)
    "Bangladesh":               "South Asia",
    "India":                    "South Asia",
    "Nepal":                    "South Asia",
    "Pakistan":                 "South Asia",

    # Southeast Asia (N = 159)
    "Indonesia":                "Southeast Asia",
    "Malaysia":                 "Southeast Asia",
    "Philippines":              "Southeast Asia",
    "Singapore":                "Southeast Asia",
    "Thailand":                 "Southeast Asia",
    "Vietnam":                  "Southeast Asia",

    # East Asia (N = 211)
    "China":                    "East Asia",
    "Hong Kong":                "East Asia",
    "Japan":                    "East Asia",
    "South Korea":              "East Asia",
    "Taiwan":                   "East Asia",

    # Oceania (N = 26)
    "Australia":                "Oceania",
    "New Zealand":              "Oceania",
}

# -----------------------------------------------------------------------------
# 4. Main evaluation loop (per‐country and per‐region)
# -----------------------------------------------------------------------------
def main():
    # --- Settings (adjust these paths if necessary) ---
    MODEL_PATH = os.getenv(
        "MODEL_PATH",
        "/data/gpfs/projects/punim2219/chunhua/cache_dir/Meta-Llama-3.1-8B-Instruct/"
    )
    DATASET   = "kellycyy/CulturalBench"
    SPLIT     = "CulturalBench-Easy"
    CACHE_DIR = "/data/gpfs/projects/punim2219/LM_with_SWOW/kabir/huggingface_cache"

    # --- Auth & environment variables ---
    hf_token = get_hf_auth_token()
    os.environ["HF_TOKEN"]           = hf_token
    os.environ["HF_HOME"]            = CACHE_DIR
    os.environ["HF_DATASETS_CACHE"]  = CACHE_DIR
    os.environ["HF_HUB_CACHE"]       = CACHE_DIR
    os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR

    # --- Load model & data ---
    llm = load_llm(MODEL_PATH, dtype="bfloat16")
    ds  = load_dataset(DATASET, SPLIT, cache_dir=CACHE_DIR)
    test_set = ds["test"]

    # --- Accumulators for per‐country evaluation ---
    country_freqs    = defaultdict(int)
    country_corrects = defaultdict(int)

    # --- Iterate over every test example ---
    for ex in tqdm(test_set, desc="Evaluating CulturalBench"):
        prompt = build_prompt(ex)
        pred   = predict_choice(llm, prompt)  # returns "A"/"B"/"C"/"D"

        gold = ex["answer"]
        # Skip any example whose “answer” is not uppercase A‐D
        if not (isinstance(gold, str) and gold.upper() in {"A","B","C","D"}):
            continue

        country = ex["country"]
        country_freqs[country] += 1
        if pred == gold.upper():
            country_corrects[country] += 1

    # --- Compute per‐country frequencies & accuracies ---
    per_country_summary = {}
    print("\nPer‐country frequencies and accuracies:")
    for country in sorted(country_freqs.keys()):
        freq = country_freqs[country]
        corr = country_corrects[country]
        acc  = (corr / freq * 100) if freq > 0 else 0.0

        per_country_summary[country] = {
            "frequency": freq,
            "correct":   corr,
            "accuracy":  round(acc, 2)
        }

        print(f"{country:20s} → freq={freq:4d}, correct={corr:4d}, acc={acc:5.2f}%")

    # --- Aggregate into per‐region totals ---
    region_freqs    = defaultdict(int)
    region_corrects = defaultdict(int)

    for country, freq in country_freqs.items():
        region = COUNTRY_TO_REGION.get(country, None)
        if region is None:
            # If a country is not in our mapping, skip it (or you can raise an error).
            continue

        region_freqs[region]    += freq
        region_corrects[region] += country_corrects[country]

    # --- Compute per‐region accuracies from those totals ---
    per_region_summary = {}
    print("\nPer‐region frequencies and accuracies:")
    for region in sorted(region_freqs.keys()):
        freq = region_freqs[region]
        corr = region_corrects[region]
        acc  = (corr / freq * 100) if freq > 0 else 0.0

        per_region_summary[region] = {
            "frequency": freq,
            "correct":   corr,
            "accuracy":  round(acc, 2)
        }

        print(f"{region:20s} → freq={freq:4d}, correct={corr:4d}, acc={acc:5.2f}%")

    # --- Write out two JSON summaries: by country and by region ---
    output_dir = os.path.dirname(CACHE_DIR)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    country_json_path = os.path.join(output_dir, "culturalbench_country_results.json")
    region_json_path  = os.path.join(output_dir, "culturalbench_region_results.json")

    with open(country_json_path, "w", encoding="utf-8") as f:
        json.dump(per_country_summary, f, indent=2, ensure_ascii=False)
    with open(region_json_path, "w", encoding="utf-8") as f:
        json.dump(per_region_summary, f, indent=2, ensure_ascii=False)

    print(f"\nWrote per‐country results to: {country_json_path}")
    print(f"Wrote per‐region results to:  {region_json_path}")

    # Clean up GPU
    del llm
    clear_gpu_memory()

if __name__ == "__main__":
    main()
