import argparse
import warnings
import json
import os
from config.hf_auth import get_hf_auth_token
from src.q_and_a import answer_question  # Updated function from 'classify' to 'answer'
from src.model_utils import load_model_and_tokenizer, clear_gpu_memory
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Run question answering using a pre-trained model.")
    parser.add_argument('--model_id', type=str, default='meta-llama/Meta-Llama-3.1-8B-Instruct', help='Hugging Face model ID')
    parser.add_argument('--cache_dir', type=str, default='/data/gpfs/projects/punim2219/LM_with_SWOW/kabir/Data/huggingface_cache', help='Cache directory')
    parser.add_argument('--output_file', type=str, default='/data/gpfs/projects/punim2219/LM_with_SWOW/kabir/Data/output/sample_answers.json', help='Output file path')
    # Removed input_file argument since we're using a list of questions directly

    args = parser.parse_args()

    hf_auth = get_hf_auth_token()
    os.environ['HF_TOKEN'] = hf_auth
    os.environ['HF_HOME'] = args.cache_dir
    os.environ['TRANSFORMERS_CACHE'] = args.cache_dir

    warnings.filterwarnings('ignore')
    clear_gpu_memory()

    # Load the model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_id, args.cache_dir)

    # List of questions for testing
    questions = [
        "What are the main factors affecting inflation?",
        "How do interest rates impact the economy?",
        "What is the significance of monetary policy?",
        "How does consumer spending influence economic growth?",
        "What are the causes of unemployment?"
    ]

    results = []

    for question in questions:
        try:
            # Call the answer_question function to get the model's answer
            result = answer_question(question, model, tokenizer)
        except Exception as e:
            result = {"question": question, "answer": f"Error processing question: {str(e)}"}
        
        results.append(result)

    # Write the results to the output file
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=4)

    # Clean up model and clear GPU memory
    del model
    clear_gpu_memory()

if __name__ == "__main__":
    main()