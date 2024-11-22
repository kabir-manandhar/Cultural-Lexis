import argparse
import warnings
import json
import os
from config.hf_auth import get_hf_auth_token
from src.q_and_a import answer_question  # Updated function from 'classify' to 'answer'
from src.model_utils import load_model_and_tokenizer, clear_gpu_memory
import pandas as pd
from tqdm import tqdm

def load_questions_from_json(json_file: str):
    """Loads questions that start with 'Q' from the JSON file."""
    with open(json_file, 'r') as f:
        data = json.load(f)
            
    # Filter and return only questions that have keys starting with 'Q'
    questions = [item['question'] for item in data]
    
    return questions

def main():
    parser = argparse.ArgumentParser(description="Run question answering using a pre-trained model.")
    parser.add_argument('--model_id', type=str, default='meta-llama/Meta-Llama-3.1-8B-Instruct', help='Hugging Face model ID')
    parser.add_argument('--cache_dir', type=str, default='/data/gpfs/projects/punim2219/LM_with_SWOW/kabir/Data/huggingface_cache', help='Cache directory')
    parser.add_argument('--output_file', type=str, default='/data/gpfs/projects/punim2219/LM_with_SWOW/kabir/Data/output/WV_answers.json', help='Output file path')
    parser.add_argument('--country_name', type=str, default='United States', help='Country from')
    parser.add_argument('--input_file', type=str, default="/data/gpfs/projects/punim2219/LM_with_SWOW/kabir/Data/WV_Bench/question_metadata.json", help='Input JSON file path')  # JSON file with questions

    args = parser.parse_args()

    hf_auth = get_hf_auth_token()
    os.environ['HF_TOKEN'] = hf_auth
    os.environ['HF_HOME'] = args.cache_dir
    os.environ['TRANSFORMERS_CACHE'] = args.cache_dir

    warnings.filterwarnings('ignore')
    clear_gpu_memory()
    
    # Load and filter questions from the JSON file
    questions = load_questions_from_json(args.input_file)
    

    # Load the model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_id, args.cache_dir)
    
    results = []

    for question in tqdm(questions):
        try:
            # Call the answer_question function to get the model's answer
            result = answer_question(question, model, tokenizer,args.country_name)
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