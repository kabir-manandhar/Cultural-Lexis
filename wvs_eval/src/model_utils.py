import os
import torch
import gc
from vllm import LLM

def clear_gpu_memory():
    """Function to clear GPU memory."""
    torch.cuda.empty_cache()
    gc.collect()

def load_llm(model_path: str, dtype: str = "bfloat16") -> LLM:
    """Loads the model using vLLM with GPU settings.
    
    Args:
        model_path (str): Path to the model directory.
        dtype (str, optional): Data type for model weights. Options include "float32", "float16", or "bfloat16".
            Default is "bfloat16".
    
    Returns:
        LLM: An instance of the loaded vLLM model.
    """
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Please check your GPU setup.")
        
        # Initialize the LLM instance from vLLM
        llm = LLM(
            model=model_path,
            dtype=dtype
        )
        return llm
    except Exception as e:
        print("Error loading vLLM model:", e)
        raise
