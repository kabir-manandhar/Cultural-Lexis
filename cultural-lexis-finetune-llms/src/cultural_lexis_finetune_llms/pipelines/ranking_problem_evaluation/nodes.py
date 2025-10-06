"""
This is a boilerplate pipeline 'ranking_problem_evaluation'
generated using Kedro 0.19.9
"""

from pathlib import Path
import torch 
import os 
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator, DataLoaderConfiguration, DistributedType, PartialState

from cultural_lexis_finetune_llms.pipelines.ppo_further_training.reward_func import reward_func

from datetime import datetime
import pickle 
from copy import deepcopy
from accelerate.utils.random import set_seed
import time 
import sys
from transformers import TrainingArguments
import wandb
import pandas as pd 
import jsonlines
from tqdm.auto import tqdm
import wandb

WORKING_DIR = os.environ['WORKING_DIR']

#! NODE
def evaluate_ranking_problem(params):
    data_dir = os.path.join(WORKING_DIR, params['data_dir'])
    dataset_name_lst = params['dataset_name_lst']
    split_name = params['split_name']
    input_key = params['input_key']
    label_key = params['label_key']

    saved_model_dir = os.path.join(WORKING_DIR, params['saved_model_dir'])

    fine_tune_class_lst = params['fine_tune_class_lst']

    llama_raw_model_name = params['llama_raw']
    qwen_raw_model_name = params['qwen_raw']

    input_key = params['input_key']
    label_key = params['label_key']

    model_type = params['model_type'] # changeable 

    # load the jsonl test data

    eval_dict = {}

    for dataset_name in dataset_name_lst:
        jsonfile_data_fp = os.path.join(data_dir, dataset_name, f'{split_name}.jsonl')

        # load the model
        for fine_tune_class in fine_tune_class_lst:
            if fine_tune_class == 'raw':
                if model_type == 'llama3':
                    model_name = llama_raw_model_name
                elif model_type == 'qwen':
                    model_name = qwen_raw_model_name
            else:
                model_name = os.path.join(saved_model_dir, model_type, dataset_name, fine_tune_class)

            # init wandb
            wandb.init(
                project="llamafactory",
                name=f"{dataset_name}_{fine_tune_class}_{model_type}_ranking_problem_evaluation",
                tags=[f"{dataset_name}", f"{fine_tune_class}", f"{model_type}", "ranking_problem_evaluation"],
                config=dict(params),
            )

            eval_metrics = evaluate_model(model_name, jsonfile_data_fp, input_key, label_key, dataset_name)

            eval_dict[f"{dataset_name}_{fine_tune_class}_{model_type}"] = eval_metrics
            wandb.log(eval_metrics)
            
            # close wandb 
            wandb.finish()
            
    print(eval_dict)
    return eval_dict


def evaluate_model(model_name, jsonfile_data_fp, input_key, label_key, dataset_name):
    
    # Initialize Accelerator
    accelerator = Accelerator(mixed_precision="bf16")
    print(f"Evaluating {model_name} with Accelerator on device: {accelerator.device}")
    
    # load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = accelerator.prepare(model)
    
    tokenizer.pad_token = tokenizer.eos_token
    
    
    # load the test data
    with jsonlines.open(jsonfile_data_fp) as reader:
        jsonfile_data_lst = [row for row in reader]
        
    # Process in batches 
    batch_size = 4 
    reward_lst = []
    for i in tqdm(range(0, len(jsonfile_data_lst), batch_size), desc="Evaluating"):
        batch = jsonfile_data_lst[i:i+batch_size]
        
        # prepare the batch
        prompts = [tokenizer.apply_chat_template(row[input_key], tokenize=False, add_generation_prompt=True) for row in batch]
        
        labels = [row[label_key] for row in batch]
        
        # Tokenize and padding 
        inputs = tokenizer(prompts, padding_side='left', return_tensors='pt', padding=True)
        
        inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}
        
        # generate outputs for batch 
        with accelerator.autocast():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=1024,
                pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=False 
            )
            
        # decode the outputs
        generated_responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        reward_tensor = reward_func(
            queries=generated_responses,
            prompts=prompts,
            labels=labels,
            dataset_name=dataset_name
        )
        # convert back to cpu
        rewards = reward_tensor.detach().cpu().numpy().tolist()
        for reward in rewards:
            reward_lst.append(float(reward))

    average_reward = sum(reward_lst) / len(reward_lst)
    print(f"Average reward: {average_reward}")
    
    # Cleanup
    del model
    del tokenizer
    accelerator.free_memory()
    torch.cuda.empty_cache()
    del accelerator
    print(f"Released resources for {model_name}")
    
    return average_reward
