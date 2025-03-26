import argparse
import warnings
import json
import os
from config.hf_auth import get_hf_auth_token
from src.q_and_a import answer_question  # Updated function for vLLM usage
from src.model_utils import load_llm, clear_gpu_memory  # Updated for vLLM; no tokenizer needed
import pandas as pd
from tqdm import tqdm
import logging

# Ignore all warnings
warnings.filterwarnings("ignore")

# Suppress INFO and WARNING messages from vLLM and other libraries.
logging.getLogger("vllm").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.basicConfig(level=logging.ERROR)


def load_questions_from_json(json_file: str):
    """Loads questions from the JSON file."""
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Extract the questions from the JSON structure
    questions = [item['question'] for item in data]
    return questions


def main():
    parser = argparse.ArgumentParser(description="Run question answering using vLLM.")
    parser.add_argument('--model_path', type=str, default='/data/gpfs/projects/punim2219/chunhua/cache_dir/Meta-Llama-3.1-8B-Instruct/', 
                        help='Path to the vLLM model directory')
    parser.add_argument('--cache_dir', type=str, default='/data/gpfs/projects/punim2219/LM_with_SWOW/kabir/Data/huggingface_cache', 
                        help='Cache directory')
    parser.add_argument('--output_file', type=str, default='/data/gpfs/projects/punim2219/LM_with_SWOW/kabir/Data/output/WV_answers.json', 
                        help='Output file path')
    parser.add_argument('--country_name', type=str, default='United States', 
                        help='Country context (e.g., "United States" or "China")')
    parser.add_argument('--input_file', type=str, default="/data/gpfs/projects/punim2219/LM_with_SWOW/kabir/Data/WV_Bench/us_question_metadata.json", 
                        help='Input JSON file path containing questions')

    parser.add_argument('--use_swow', action='store_true', 
                    help='Use SWOW-based augmentation for contextual awareness')
    
    args = parser.parse_args()

    # Setup authentication environment variables if applicable.
    hf_auth = get_hf_auth_token()
    os.environ['HF_TOKEN'] = hf_auth
    os.environ['HF_HOME'] = args.cache_dir
    os.environ['TRANSFORMERS_CACHE'] = args.cache_dir

    warnings.filterwarnings('ignore')
    clear_gpu_memory()

    # Load questions based on country
    if str(args.country_name).lower() == "united states":
        questions = load_questions_from_json(args.input_file)
    elif str(args.country_name).lower() == "china":
        df = pd.read_excel("/data/gpfs/projects/punim2219/LM_with_SWOW/kabir/Data/WV_Bench/WVS_Question_ZH_EN 1.xlsx")
        questions = [f"{row['question_zh']} Options: {row['options_zh']}" for _, row in df.iterrows()]
    else:
        raise ValueError("Country not supported. Please use 'United States' or 'China'.")

    # Load the vLLM model instance
    llm = load_llm(args.model_path, dtype="bfloat16")

    results = []

    # Loop over each question and generate an answer using vLLM
    for question in tqdm(questions):
        try:
            result = answer_question(question, llm, args.country_name, args.use_swow)
        except Exception as e:
            result = {"question": question, "answer": f"Error processing question: {str(e)}"}
        breakpoint()
        results.append(result)

    # Write the results to the specified output file in JSON format
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=4)

    # Clean up resources
    del llm
    clear_gpu_memory()


if __name__ == "__main__":
    main()
