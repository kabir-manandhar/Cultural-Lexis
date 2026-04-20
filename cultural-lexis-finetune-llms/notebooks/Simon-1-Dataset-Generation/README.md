## Training script

they are at `/data/gpfs/projects/punim2219/LM_with_SWOW/sukaih/cultural-lexis-finetune-llms/scripts/HLCP-8-SFT_Training/s_8_sft_llama3_on_mimic_swow_us_isolated_5_10_percent copy.sh`


Important: The saved checkpoints are LoRA Adapters!
This is the most important bit: The saved checkpoints are not full models; they are LoRA adapters.
This means you can't load them directly with AutoModel.from_pretrained(...). You first need to load the original base LLaMA-3 model and then merge these LoRA weights onto it.
Here's a snippet of how to do it. The key is to use the PeftModel.from_pretrained function to apply the adapter to the base model.

```python
# load lora model 
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

def apply_lora(model_name_or_path, lora_path, output_path, is_rm, bf16):
    print(f"Loading the base model from {model_name_or_path}")
    model_cls = AutoModelForCausalLM if not is_rm else AutoModelForSequenceClassification
    base = model_cls.from_pretrained(
        model_name_or_path, torch_dtype=torch.bfloat16 if bf16 else "auto", low_cpu_mem_usage=True
    )
    base_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    print(f"Loading the LoRA adapter from {lora_path}")
    # This step merges the LoRA adapter with the base model
    lora_model = PeftModel.from_pretrained(
        base,
        lora_path,
        torch_dtype=torch.bfloat16 if bf16 else "auto",
    )
    
    # Optional: If you want to save the merged model for easier loading later
    # lora_model.save_pretrained(output_path)
    # base_tokenizer.save_pretrained(output_path)
    
    return lora_model, base_tokenizer

# Example usage for one of our 100% models:
model, tokenizer = apply_lora(
    model_name_or_path="meta-llama/Meta-Llama-3.1-8B-Instruct",
    lora_path="<the path I gave you>",
    output_path="./merged_model", # Optional
    is_rm=False,
    bf16=True
)
```