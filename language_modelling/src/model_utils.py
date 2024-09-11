import os
import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM

def clear_gpu_memory():
    """Function to clear GPU memory."""
    torch.cuda.empty_cache()
    gc.collect()

def load_model_and_tokenizer(model_id: str, cache_dir: str) -> tuple:
    """Loads the model and tokenizer with GPU settings."""
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Please check your GPU setup.")

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,  # Use mixed precision
            cache_dir=cache_dir,
            device_map="auto"
        )
                
        tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
        
        return model, tokenizer
    except Exception as e:
        print("Error loading model and tokenizer:", e)
        raise
