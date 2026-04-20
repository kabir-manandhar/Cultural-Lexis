"""
This is a boilerplate pipeline 'finetuning_evaluation'
generated using Kedro 0.19.9
"""
# ref: https://github.com/huggingface/accelerate/blob/main/examples/complete_nlp_example.py

from pathlib import Path
import torch 
import os 
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator, DataLoaderConfiguration, DistributedType, PartialState
from cultural_lexis_finetune_llms.pipelines.finetuning_evaluation.eval_score import eval_score, SWOW_EN_FEW_SHOT_EXAMPLE, SWOW_ZH_FEW_SHOT_EXAMPLE, eval_score_wordties
from datetime import datetime
import pickle 
from copy import deepcopy
from accelerate.utils.random import set_seed
import time 
import sys
from transformers import TrainingArguments
import wandb
import pandas as pd 
from trl import ModelConfig
from cultural_lexis_finetune_llms.pipelines.ppo_further_training.deprecated.dataset_setup import lf_ver_prepare_dataset, get_cue_word
from torch.nn.utils.rnn import pad_sequence

def evaluate_finetuned_model(finetune_eval_params):

    top_k = finetune_eval_params["top_k"]
    model_save_dir = finetune_eval_params["model_save_dir"]
    model_type = finetune_eval_params["model_type"]
    dataset_name = finetune_eval_params["dataset_name"]
    dataset_location = os.path.join(os.environ["WORKING_DIR"], finetune_eval_params["dataset_location"])
    word_freq_dict_fp = os.path.join(os.environ["WORKING_DIR"], finetune_eval_params["word_freq_dict_fp"])
    split = finetune_eval_params["split"]
    batch_size = finetune_eval_params["batch_size"]
    mixed_precision = finetune_eval_params["mixed_precision"]
    dataset_num_proc = finetune_eval_params["dataset_num_proc"]
    max_length = finetune_eval_params["max_length"]
    want_few_shot_example = True
    eval_metric_type = finetune_eval_params["eval_metric_type"]
    
    if eval_metric_type == "raw":
        used_eval_func = eval_score
    elif eval_metric_type == "wordties":
        used_eval_func = eval_score_wordties
        
    if_lora = finetune_eval_params['if_lora']

    dataloader_config = DataLoaderConfiguration()
    
    fine_tune_class_lst = finetune_eval_params["fine_tune_class_lst"]
    
    for fine_tune_class in fine_tune_class_lst:
        print(f"Fine-tuning class: {fine_tune_class}")
        if fine_tune_class == 'raw':
            if model_type == "llama3":
                model_path = "meta-llama/Meta-Llama-3.1-8B-Instruct"
            elif model_type == "qwen":
                model_path = "Qwen/Qwen2.5-7B-Instruct"
        else:
            model_path = os.path.join(model_save_dir, model_type, dataset_name, fine_tune_class)

        # --- Load Tokenizer ---
        model_path = os.path.join(model_save_dir, model_type, dataset_name, fine_tune_class)
        
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.padding_side = 'left'

        # --- Load Eval dataset ---

        model_args = ModelConfig()
        model_args.cache_dir = None 
        model_args.hf_hub_token = None 

        training_args = TrainingArguments(
            output_dir=os.path.join(os.environ["WORKING_DIR"], 'data/08_reporting/finetuning_evaluation'),
            per_device_eval_batch_size=batch_size,
        )
        training_args.predict_with_generate = True
        with PartialState().local_main_process_first():
            _, tokenized_eval_datasets = lf_ver_prepare_dataset(
                subset=dataset_name,
                split=split,
                template=model_type,
                lf_dataset_dir=os.path.join(os.environ["WORKING_DIR"], 'scripts/HLCP-8-SFT_Training/configs'),
                model_args=model_args,
                training_args=training_args,
                tokenizer=tokenizer,
                stage = 'sft',
            )

        accelerator = Accelerator(
            mixed_precision=mixed_precision,
            dataloader_config=dataloader_config,
            log_with="wandb",
        )

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        project_name = "llamafactory"
        run_name = f"{model_type}_{dataset_name}_sft_evaluation{'_wordties_eval' if eval_metric_type == 'wordties' else ''}_{fine_tune_class}{'_lora' if if_lora else ''}"
        accelerator.print("Init Logging")
        accelerator.init_trackers(project_name, dict(finetune_eval_params),
                                init_kwargs={
                                    "wandb":{
                                        "name": run_name,
                                    }
                                })

        # --- Load cache dict of word associations ---
        accelerator.print("Loading word frequency dictionary")

        with open(word_freq_dict_fp, 'rb') as f:
            word_freq_dict =  pickle.load(f)

        # eval_dataset = load_dataset(dataset_location, dataset_name, split=split)

        # def prepare_dataset(dataset, tokenizer):
        #     """pre-tokenize the dataset before training; only collate during training"""

        #     def tokenize(element):
        #         if want_few_shot_example:
        #             if dataset_name == "swow_en":
        #                 messages = [
        #                     {"role": "system", "content": element['system']},
        #                     {"role": "user", "content": element['instruction'] + '\n' + SWOW_EN_FEW_SHOT_EXAMPLE + element['input']},
        #                 ]
        #             elif dataset_name == "swow_zh":
        #                 messages = [
        #                     {"role": "system", "content": element['system']},
        #                     {"role": "user", "content": element['instruction'] + '\n'  + SWOW_ZH_FEW_SHOT_EXAMPLE + element['input']},
        #                 ]
        #             else:
        #                 raise ValueError("Dataset name not recognized")
        #         else:
        #             messages = [
        #                 {"role": "system", "content": element['system']},
        #                 {"role": "user", "content": element['instruction'] + '\n' + element['input']},
        #             ]
        #         input_ids = tokenizer.apply_chat_template(
        #             messages,
        #             padding=False,
        #             add_generation_prompt=True,
        #         )
        #         return {"input_ids": input_ids, "lengths": len(input_ids)}

        #     rm_col_names = deepcopy(list(dataset.column_names))
        #     rm_col_names.remove('input') # this is the cue word, used for scoring

        #     return dataset.map(
        #         tokenize,
        #         remove_columns=rm_col_names,
        #         num_proc=dataset_num_proc,
        #     )

        # with accelerator.main_process_first():
        #     tokenized_eval_datasets = prepare_dataset(eval_dataset, tokenizer)

        def collate_fn(examples):
            # they are in batch
            # drop labels, images, videos
            new_output = [
                {k: exp[k] for k in examples[0].keys() if k not in ["labels", "images", "videos"]}
                for exp in examples
            ]
            input_ids = [exp["input_ids"] for exp in new_output]
            attention_mask = [exp["attention_mask"] for exp in new_output]
            # decoder help to get back the text
            raw_text = tokenizer.batch_decode(input_ids, skip_special_tokens=False)
            lengths = [len(text) for text in raw_text]
            cue_word_lst = [get_cue_word(model_type, text) for text in raw_text]
            
            if want_few_shot_example:
                # recalculate the input_ids
                input_ids = []
                attention_mask = []
                for text in raw_text:
                    if dataset_name == "swow_en":
                        input_id = tokenizer.encode(SWOW_EN_FEW_SHOT_EXAMPLE + "\n" + text, return_tensors="pt")
                        # flatten the input_id
                        input_id = input_id.flatten()
                        input_ids.append(input_id)
                        attention_mask.append(torch.ones_like(input_id))
                    elif dataset_name == "swow_zh":
                        input_id = tokenizer.encode(SWOW_ZH_FEW_SHOT_EXAMPLE + "\n" + text, return_tensors="pt")
                        # flatten the input_id
                        input_id = input_id.flatten()
                        input_ids.append(input_id)
                        attention_mask.append(torch.ones_like(input_id))
                    else:
                        raise ValueError("Dataset name not recognized")

            # pad the input_ids
            padded_input_ids, padded_attention_mask = left_padding(input_ids, attention_mask, tokenizer)

            dict_output = {
                "lengths": lengths,
                "cue_words": cue_word_lst,
                "input_ids": padded_input_ids,
                "attention_mask": padded_attention_mask,
            }

            return dict_output

        eval_dataloader = DataLoader(
            tokenized_eval_datasets,
            collate_fn=collate_fn,
            shuffle=False,
            batch_size=batch_size,
        )

        set_seed(42)

        # --- Load Model ---
        accelerator.print("Loading Model")
        model = AutoModelForCausalLM.from_pretrained(model_path)
        model = model.to(accelerator.device)
        model.eval()

        model, eval_dataloader = accelerator.prepare(model, eval_dataloader, device_placement=[True, False])
        
        

        # now eval the model

        if eval_metric_type == "raw":
            total_score = 0
            total_count = 0
        elif eval_metric_type == "wordties":
            total_score = None 
            total_count = 0 

        df_lst = [] 
        for step, batch in enumerate(eval_dataloader):
            model_inputs = {k: v.to(accelerator.device) for k, v in batch.items() if k not in ["cue_words", "lengths"]}
            cue_words = batch["cue_words"]
            lengths = batch["lengths"]
            with torch.no_grad():
                outputs = model.generate(**model_inputs, max_length=max_length, pad_token_id=tokenizer.eos_token_id)
                generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                actual_generated_text = []
                for text in generated_text:
                    start_idx = text.rfind("assistant")
                    if start_idx == -1:
                        start_idx = 0
                    actual_generated_text.append(text[start_idx + len("assistant"):].strip())

                # calculate the score
                
                
                if eval_metric_type == "raw":
                    score, response_lst = used_eval_func(actual_generated_text, cue_words, word_freq_dict, top_k)
                    total_score += score.sum()
                    total_count += len(score)

                    # log the score
                    accelerator.log(
                        {
                            "intermediate score": score.mean(),
                            "average score": total_score / total_count,
                        },
                        step = step+1,
                    )
                    
                    
                elif eval_metric_type == "wordties":
                    output_eval_dict, response_lst = used_eval_func(actual_generated_text, cue_words, word_freq_dict)
                    if total_score is None:
                        total_score = output_eval_dict
                    else:
                        # add each key 
                        for key in total_score:
                            if total_score[key] is None:
                                total_score[key] = 0.0
                            temp_val = output_eval_dict.get(key, 0.0)   
                            if temp_val is None:
                                temp_val = 0.0
                            total_score[key] += temp_val
                    total_count += 1
                    
                    # log the score
                    log_dict = dict()
                    # add "intermediate" to the key
                    for key in output_eval_dict:
                        log_dict[f"intermediate_{key}"] = output_eval_dict[key]
                    # add "average" to the key
                    for key in total_score:
                        if total_score[key] is None:
                            nominator = 0.0
                        else:
                            nominator = total_score[key]
                        log_dict[f"average_{key}"] = nominator / total_count
                    accelerator.log(
                        log_dict,
                        step = step+1,
                    )
                    
                
                # log the generated text and actual generated text and cue words
                df = pd.DataFrame({
                    "full_text": generated_text,
                    "generated_text": actual_generated_text,
                    "cue_words": cue_words,
                    "response_list": response_lst,
                    })
                df_lst.append(df)
            df = pd.concat(df_lst)
            accelerator.print(f"Logging completions at step {step}")
            wandb.log({"completions": wandb.Table(dataframe=df)})

        
        wandb.finish()
        # clean up 
        del model
        del tokenizer
        accelerator.free_memory()
        torch.cuda.empty_cache()
        accelerator.end_training()
        del accelerator

        
    return None


def left_padding(input_ids, attention_mask, tokenizer):
    # pad the input_ids
    input_ids_tensors = [torch.tensor(seq) for seq in input_ids]
    reversed_input_ids = [torch.flip(seq, dims=[0]) for seq in input_ids_tensors]
    padded_reversed_input_ids = pad_sequence(reversed_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    padded_input_ids = torch.flip(padded_reversed_input_ids, dims=[1])
    

    attention_mask_tensors = [torch.tensor(seq) for seq in attention_mask]
    reversed_attention_mask = [torch.flip(seq, dims=[0]) for seq in attention_mask_tensors]
    padded_reversed_attention_mask = pad_sequence(reversed_attention_mask, batch_first=True, padding_value=0)
    padded_attention_mask = torch.flip(padded_reversed_attention_mask, dims=[1])
    
    return padded_input_ids, padded_attention_mask

    