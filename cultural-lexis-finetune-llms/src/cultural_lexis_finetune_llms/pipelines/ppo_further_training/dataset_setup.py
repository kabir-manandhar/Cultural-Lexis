from transformers import AutoTokenizer
from datasets import load_dataset
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence

from transformers import Seq2SeqTrainingArguments
from llamafactory.hparams import DataArguments, ModelArguments
from llamafactory.data.parser import DatasetAttr
from llamafactory.data.loader import get_dataset as lf_get_dataset
from llamafactory.data.template import get_template_and_fix_tokenizer
from copy import deepcopy

TRAIN_CACHE_DIR = '/data/projects/punim0478/sukaih/save_to_dict_dataset_train'
EVAL_CACHE_DIR = '/data/projects/punim0478/sukaih/save_to_dict_dataset_eval'

def lf_ver_prepare_dataset(
    subset: str,
    split: str,
    template: str,
    lf_dataset_dir: str,
    model_args,
    training_args,
    tokenizer,
    stage : str = 'ppo',
):
    
    # ! data args follow the dataset_info.json of llamafactory 
    data_args = DataArguments(
        dataset = f'{subset}_{split}',
        eval_dataset = f'{subset}_test',
        dataset_dir = lf_dataset_dir,
        template=template,
        cutoff_len=2048,
    )

    template_object = get_template_and_fix_tokenizer(tokenizer, data_args)
    
    lf_dataset = lf_get_dataset(
        template = template_object, 
        model_args = model_args, 
        data_args = data_args, 
        training_args = training_args, 
        stage = stage,
        tokenizer = tokenizer, 
    )
    
    return lf_dataset['train_dataset'], lf_dataset['eval_dataset']


def get_cue_word(template:str, query_str:str):
    if template == 'llama3':
        end_special_token = "<|eot_id|>"
    elif template == 'qwen':
        end_special_token = "<|im_end|>"
    front_special_token_lst = [':', '：']
    
    # rindex the last occurence of the end_special_token
    end_special_token_idx = query_str.rindex(end_special_token)
    # get the string before the end_special_token
    query_str = query_str[:end_special_token_idx]
    
    cue_word = query_str[query_str.rindex("\n")+1:]
    
    
    # get the string after the front_special_token
    actual_front_special_token = None
    for front_special_token in front_special_token_lst:
        if front_special_token in cue_word:
            actual_front_special_token = front_special_token
            break
    if actual_front_special_token is None:
        pass
    else:
        cue_word = query_str[query_str.rindex(actual_front_special_token)+1:]
    cue_word = cue_word.strip()
    return cue_word
    

if __name__ == "__main__":
    subset = "swow_zh"
    split = "trl"
    template = "qwen"  # or 'llama3' | 'qwen'
    output_dir = os.path.join(
        os.environ["WORKING_DIR"], f"data/07_model_output/{template}/{subset}/ppo"
    )
    if template == "qwen": # TODO later can remove this 
        model_name = 'Qwen/Qwen2.5-7B-Instruct'
    elif template == "llama3":
        model_name = 'meta-llama/Meta-Llama-3.1-8B-Instruct'

    model_args = ModelArguments(model_name_or_path=model_name)
    
    training_args = Seq2SeqTrainingArguments(output_dir=output_dir) # TODO remove this later 
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    
    train_dataset, eval_dataset = lf_ver_prepare_dataset(
        subset=subset,
        split=split,
        template=template,
        lf_dataset_dir=os.path.join(os.environ["WORKING_DIR"], 'scripts/HLCP-8-SFT_Training/configs'),
        model_args=model_args,
        training_args=training_args,
        tokenizer=tokenizer,
    )
    
    # try to decode back the text 
    
    queries = train_dataset[:5]['input_ids']
    queries_text = tokenizer.batch_decode(queries)
    
    # apply the get_cue_word function
    
    cue_words = [get_cue_word(template, query) for query in queries_text]
    
    # save to disk
    # train_dataset.save_to_disk(TRAIN_CACHE_DIR)
    # eval_dataset.save_to_disk(EVAL_CACHE_DIR)
    # print("=== Done saving to disk")
    # import time
    # time.sleep(2)
