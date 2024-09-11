import sys
import os
import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import your model loading function
from language_modelling.src.model_utils import load_model_and_tokenizer

# Test Model ID and Cache Directory
CACHE_DIR = '/data/gpfs/projects/punim2219/LM_with_SWOW/kabir/Data/huggingface_cache'  # Replace with your actual cache directory

@pytest.mark.parametrize("model_id", [
    'meta-llama/Meta-Llama-3.1-8B-Instruct'
])
def test_load_model_and_tokenizer(model_id):
    model, tokenizer = load_model_and_tokenizer(model_id, CACHE_DIR)
    
    assert model is not None, f"Model failed to load for {model_id}"
    assert tokenizer is not None, f"Tokenizer failed to load for {model_id}"

