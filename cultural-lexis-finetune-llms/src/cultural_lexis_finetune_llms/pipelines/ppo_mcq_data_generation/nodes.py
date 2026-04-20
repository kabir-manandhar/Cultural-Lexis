"""
This is a boilerplate pipeline 'ppo_mcq_data_generation'
generated using Kedro 0.19.9
"""

import os
from pathlib import Path 
from datasets import load_dataset
from cultural_lexis_finetune_llms.pipelines.ppo_mcq_data_generation.constant import MCQ_DATASET_TEMPLATE_EN, MCQ_DATASET_TEMPLATE_ZH, RANKING_DATASET_TEMPLATE_EN, RANKING_DATASET_TEMPLATE_ZH

import re 
import random
import numpy.random as npr
from tqdm.auto import tqdm
import jsonlines
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader
import logging 
import torch 
import transformers
import pandas as pd
from copy import deepcopy
from scipy.stats import spearmanr


WORKING_DIR = os.environ['WORKING_DIR']
logger = logging.getLogger(__name__)

# ! deprecated NODE 
def generate_mcq_data(
    ppo_mcq_params,
):
    
    num_choices = ppo_mcq_params["num_choices"]
    choice_ratio = ppo_mcq_params["choice_ratio"]
    top_k = ppo_mcq_params["top_k"]
    split_name = ppo_mcq_params["split_name"]
    
    dataset_name_lst = ppo_mcq_params["dataset_name_lst"]
    
    data_location = os.path.join(WORKING_DIR, ppo_mcq_params["data_location"])
    
    # -- Load the dataset
    dataset_lst = []
    for dataset_name in dataset_name_lst:
        dataset = load_dataset(data_location, dataset_name, split=split_name)
        dataset_lst.append(dataset)
    
    option_choice_dict = {
        "swow_en": " , ".join([chr(ord('A') + i) for i in range(num_choices)]),
        "swow_zh": "，".join([chr(ord('A') + i) for i in range(num_choices)]),
        "swow_us": " , ".join([chr(ord('A') + i) for i in range(num_choices)]),
    }
    
    option_choice_list = [chr(ord('A') + i) for i in range(num_choices)]
    
    for idx, dataset_name in enumerate(dataset_name_lst):
        dataset = dataset_lst[idx]
        update_dataset_lst = []
        for data_id, data in enumerate(tqdm(dataset, desc=f"Generating MCQ data for {dataset_name}")):
            cue_word = data['input']
            raw_associations = data['output']
            # step 1. get the ground truth answer
            gt_answer_choice = get_gt_association_word_string(dataset_name, raw_associations, top_k)
            
            # step 2: get the distractor
            distractor_lst = []
            while len(distractor_lst) < (num_choices - 1):
                # random choose from [0,1], # 0 means hard negative sample, 1 means inter cue word sample
                # use npr 
                choice = int(npr.choice([0, 1], p=choice_ratio))
                if choice == 0:
                    distractor = get_hard_negative_association_word_string(dataset_name, cue_word, top_k)
                else:
                    # randomly choose another data_id 
                    
                    another_data_id = int(npr.choice(range(len(dataset))))
                    if another_data_id == data_id:
                        another_data_id = (another_data_id + 1) % len(dataset)
                    another_data = dataset[another_data_id]
                    another_data_raw_associations = another_data['output']
                    distractor = get_gt_association_word_string(dataset_name, another_data_raw_associations, top_k)
                    
                if distractor is not None:
                    distractor_lst.append(distractor)
                    
            # step 3: shuffle the distractor list
            random.shuffle(distractor_lst)
            
            choice_ids_lst = [i for i in range(num_choices)] # gt_answer_choice is always the first one, id = 0
            
            random.shuffle(choice_ids_lst)
            
            choices_val = [] 
            
            # step 4: create the final choices dictionary 
            gt_option_letter = None 
            for order_id, choice_id in enumerate(choice_ids_lst):
                if choice_id == 0:
                    choice = gt_answer_choice
                    gt_option_letter = option_choice_list[order_id]
                else:
                    choice = distractor_lst[order_id - 1]
                option_letter = option_choice_list[order_id]
                choices_val.append({
                    "letter": option_letter,
                    "text": choice,
                })
            instruction_format = MCQ_DATASET_TEMPLATE_EN["instruction"] if dataset_name in ["swow_en", "swow_us"] else MCQ_DATASET_TEMPLATE_ZH["instruction"]
            
            instruction = instruction_format.format(option_choice=option_choice_dict[dataset_name])
            new_dataset_entry = {
                "system": MCQ_DATASET_TEMPLATE_EN["system"] if dataset_name in ["swow_en", "swow_us"] else MCQ_DATASET_TEMPLATE_ZH["system"],
                "instruction": instruction,
                "input": cue_word,
                "output": gt_option_letter,
                "choices": choices_val,
            }
            
            # append choices into the input
            
            choice_str = "\n".join(["    {}: {}".format(x["letter"], x["text"]) for x in choices_val])
            updated_input = f"""Cue word: {cue_word}

Choices:
{choice_str}
"""
            new_dataset_entry["input"] = updated_input
            # remove choices from the dict 
            new_dataset_entry.pop("choices")
            
            update_dataset_lst.append(new_dataset_entry)
            
        # save the new dataset
        save_fp = os.path.join(WORKING_DIR, f"data/03_primary/llm_swow_finetune_dataset/{dataset_name}/trl_mcq")
        
        Path(save_fp).mkdir(parents=True, exist_ok=True)
        
        save_fp = os.path.join(save_fp, f"chunk_0.jsonl")
        
        with jsonlines.open(save_fp, mode='w') as writer:
            writer.write_all(update_dataset_lst)
            
    return None





def get_gt_association_word_string(dataset_name, raw_associations, top_k, back_to_string=True):
    if dataset_name in ["swow_en", "swow_us"]:
        gt_associateion_split_1 = raw_associations.split(",")
        gt_association_split = gt_associateion_split_1
    elif dataset_name == "swow_zh":
        gt_association_split_1 = raw_associations.split("，")
        gt_association_split_2 = raw_associations.split(",")
        gt_association_split_3 = raw_associations.split("、")
        # choose the longest one
        if len(gt_association_split_1) >= len(gt_association_split_2) and len(gt_association_split_1) >= len(gt_association_split_3):
            gt_association_split = gt_association_split_1
        elif len(gt_association_split_2) >= len(gt_association_split_1) and len(gt_association_split_2) >= len(gt_association_split_3):
            gt_association_split = gt_association_split_2
        else:
            gt_association_split = gt_association_split_3
    
    # strip the space
    gt_association_split = [x.strip() for x in gt_association_split]
    # re sub multiple space to one space
    gt_association_split = [re.sub(r"\s+", " ", x) for x in gt_association_split]
    if top_k is None:
        gt_answer_choice =  gt_association_split
    else:
        gt_answer_choice = gt_association_split[:top_k]
    if back_to_string:
        # convert back to string
        if dataset_name in ["swow_en", "swow_us"]:
            gt_answer_choice = ", ".join(gt_answer_choice)
        elif dataset_name == "swow_zh":
            gt_answer_choice = "，".join(gt_answer_choice)
            
    return gt_answer_choice

def get_hard_negative_association_word_string(dataset_name, raw_associations, top_k):
    
    if dataset_name in ["swow_en", "swow_us"]:
        gt_associateion_split_1 = raw_associations.split(",")
        gt_association_split = gt_associateion_split_1
    elif dataset_name == "swow_zh":
        gt_association_split_1 = raw_associations.split("，")
        gt_association_split_2 = raw_associations.split(",")
        gt_association_split_3 = raw_associations.split("、")
        # choose the longest one
        if len(gt_association_split_1) >= len(gt_association_split_2) and len(gt_association_split_1) >= len(gt_association_split_3):
            gt_association_split = gt_association_split_1
        elif len(gt_association_split_2) >= len(gt_association_split_1) and len(gt_association_split_2) >= len(gt_association_split_3):
            gt_association_split = gt_association_split_2
        else:
            gt_association_split = gt_association_split_3
    
    # strip the space
    gt_association_split = [x.strip() for x in gt_association_split]
    # re sub multiple space to one space
    gt_association_split = [re.sub(r"\s+", " ", x) for x in gt_association_split]
    
    # only consider the after top_k
    answer_choice_distracor_list = gt_association_split[top_k:]
    
    # check if 
    
    # randomly select k words
    try:
        distractor = npr.choice(answer_choice_distracor_list, size=top_k, replace=False).tolist()
    except Exception as e:
        return None
    # convert back to string
    if dataset_name in ["swow_en", "swow_us"]:
        distractor = ", ".join(distractor)
    elif dataset_name == "swow_zh":
        distractor = "，".join(distractor)
        
    return distractor


def generate_ranking_data(
    ppo_ranking_params,
):
    top_k = ppo_ranking_params["top_k"]
    split_name = ppo_ranking_params["split_name"]
    # ! force the split_name to train because we are having direct compare between SFT and PPO 
    split_name = "train"
    num_question_per_cue_word = ppo_ranking_params["num_question_per_cue_word"]
    dataset_name_lst = ppo_ranking_params["dataset_name_lst"]
    
    data_location = os.path.join(WORKING_DIR, ppo_ranking_params["data_location"])
    
    dataset_lst = []
    for dataset_name in dataset_name_lst:
        dataset = load_dataset(data_location, dataset_name, split=split_name)
        dataset_lst.append(dataset)
    
    for idx, dataset_name in enumerate(dataset_name_lst):
        dataset = dataset_lst[idx]
        update_dataset_lst = []
        for data_id, data in enumerate(tqdm(dataset, desc=f"Generating Ranking question data for {dataset_name}")):
            cue_word = data['input']
            raw_associations = data['output']
            
            gt_answer_choice = get_gt_association_word_string(dataset_name, raw_associations, top_k=None, back_to_string=False)
            
            ques_count = 0 
            while ques_count < num_question_per_cue_word:
                # step 1, get select k index from the gt_answer_choice
                select_k_idx = npr.choice(len(gt_answer_choice), size=top_k, replace=False).tolist()
                select_k_idx = [int(x) for x in select_k_idx]
                select_k_idx = sorted(select_k_idx)
                
                # get the gt_answer_choice_selected
                
                k_associated_selected = [gt_answer_choice[i] for i in select_k_idx]
                # step 2, shuffle the gt_answer_choice
                shuffle_choice = deepcopy(k_associated_selected)
                random.shuffle(shuffle_choice)
                
                # step 3, form the instruction 
                instruction = RANKING_DATASET_TEMPLATE_EN["instruction"] if dataset_name in ["swow_en", "swow_us"] else RANKING_DATASET_TEMPLATE_ZH["instruction"]
                
                punc = ", " if dataset_name in ["swow_en", "swow_us"] else "，"
                cue_word_str = "Cue word: " if dataset_name in ["swow_en", "swow_us"] else "提示词："
                associated_word_str = "Associated words: " if dataset_name in ["swow_en", "swow_us"] else "关联词："
                a_word_size = f"{top_k} size" if dataset_name in ["swow_en", "swow_us"] else f"{top_k} 个"
                
                input_str = f"""{cue_word_str}{cue_word}
{a_word_size}{associated_word_str}{punc.join(shuffle_choice)}
"""
                
                new_dataset_entry = {
                    "system": RANKING_DATASET_TEMPLATE_EN["system"] if dataset_name in ["swow_en", "swow_us"] else RANKING_DATASET_TEMPLATE_ZH["system"],
                    "instruction": instruction,
                    "input": input_str,
                    "output": str(k_associated_selected),
                }
                
                update_dataset_lst.append(new_dataset_entry)
                
                ques_count += 1
                
        # save the new dataset
        # shuffle the update_dataset_lst
        random.shuffle(update_dataset_lst)
        save_fp = os.path.join(WORKING_DIR, f"data/03_primary/llm_swow_finetune_dataset/{dataset_name}/trl_ranking")
        
        Path(save_fp).mkdir(parents=True, exist_ok=True)
        
        save_fp = os.path.join(save_fp, f"chunk_0.jsonl")
        
        with jsonlines.open(save_fp, mode='w') as writer:
            writer.write_all(update_dataset_lst)
            
    return None




# ! deprecated NODE 
def mcq_test_model_output(
    ppo_mcq_params,
):
    model_params = ppo_mcq_params["model_params"]
    model_type = model_params["model_type"]
    test_vanilla = model_params["test_vanilla"]

    dataset_name_lst = ppo_mcq_params["dataset_name_lst"]
    
    batch_size = model_params["batch_size"]
    

    for dataset_name in dataset_name_lst:
        
        if test_vanilla:
            model_path = model_params["model_path"]['vanilla'][model_type]
        else:
            model_path = model_params["model_path"][dataset_name]["model_path"]
        ##########
        # WANDB
        ###########

        wandb.init(
            project="llamafactory",
            name=f"mcq_check_output_samples_before_training_{dataset_name}_{model_type}_{'vanilla' if test_vanilla else 'fine_tuned'}{'_lora' if 'lora' in model_path else ''}",
            tags=["ppo_mcq", "before_training"],
            config=dict(ppo_mcq_params),
        )
        
        
        

        ################
        # Model
        ################

        logger.info(f"Loading model from {model_path}")
        
        
        pipeline = transformers.pipeline(
            "text-generation",
            model=model_path,
            model_kwargs={
                'torch_dtype': 'auto',
            },
            device_map="auto",
            batch_size=batch_size,
        )
        
        # set pad_token_id 
        pipeline.tokenizer.pad_token_id = pipeline.tokenizer.eos_token_id
        
        # set pad_token_id for the model
        pipeline.model.config.pad_token_id = pipeline.tokenizer.eos_token_id
        
        
        messages = [
            {"role": "system", "content": "You are a chatbot who loves to speak with people about amazing stories!"},
            {"role": "user", "content": "Who are you?"},
        ]
        
        outputs = pipeline(
            messages,
            max_new_tokens=256,
        )
        
        print("Test common output")
        print(outputs)


        ################
        # Dataset
        ################
        
        logger.info(f"Loading dataset {dataset_name}")
        
        dataset = load_dataset(
            os.path.join(WORKING_DIR, "data/03_primary/llm_swow_finetune_dataset"),
            dataset_name,
            split="trl_mcq",  # TODO, change to trl_ranking
        )
        
        # collect GT_answer dict 
        gt_answer_dict = {}
        for data in tqdm(dataset, desc=f"Collecting GT answer dict for {dataset_name}"):
            gt_answer_dict[data['input']] = data['output']
        
        
        
        ################
        # Evaluation
        ################
        
        data_dict_lst = [] 
        
        batch_messages = []
        batch_GT_answer = []
        
        accuracy_lst = []
        count = 0
        
        first_time_log = True 
        
        for data in tqdm(dataset, desc=f"Evaluating model output for {dataset_name}"):
            messages = [
                {"role": "system", "content": data['system']},
                {"role": "user", "content": data['instruction'] + '\n' + data['input']},
            ]
            
            GT_answer = data['output']
            
            batch_messages.append(messages)
            batch_GT_answer.append(GT_answer)
            
            if len(batch_messages) % batch_size  == 0:
                
                outputs = pipeline(
                    batch_messages,
                    max_new_tokens=1000,
                    # pad_token_id=pipeline.tokenizer.eos_token_id,
                    pad_token_id=pipeline.tokenizer.eos_token_id,
                    batch_size=batch_size,
                )
                for idx, output in enumerate(outputs):
                    response = output[0]['generated_text'][-1]['content']
                    query = output[0]['generated_text'][-2]['content']
                    
                    try:
                        if dataset_name in ["swow_en", "swow_us"]:
                            answer_line = response.lower().rfind("answer")
                            answer = response[answer_line + len("answer"):]
                        elif dataset_name == "swow_zh":
                            answer_line = response.lower().rfind("答案")
                            answer = response[answer_line + len("答案"):]
                            
                        # also chunk before the first \n 
                        answer = answer.split("\n")[0]
                        answer = re.search(r"([A-Z])", answer).group(1)
                    except:
                        answer = None
                    
                    data_dict = {
                        "query": query,
                        "response": response,
                        "GT_answer": batch_GT_answer[idx],
                        "parsed_answer": answer,
                    }
                    data_dict_lst.append(data_dict)
                    
                    if answer == batch_GT_answer[idx]:
                        accuracy_lst.append(1.0)
                    else:
                        accuracy_lst.append(0.0)
                        
                    count += 1
                    
                batch_messages.clear()
                batch_GT_answer.clear()
            
                # log the average accuracy
                
                accuracy = sum(accuracy_lst) / len(accuracy_lst)
                wandb.log({"average accuracy": accuracy})
                
                if first_time_log: # log the first batch of data
                    wandb.log({"data": wandb.Table(dataframe=pd.DataFrame(data_dict_lst))})
                    first_time_log = False
            
        data_df = pd.DataFrame(data_dict_lst)
        wandb.log({"data_df": wandb.Table(dataframe=data_df)})
            
            
        # close wandb
        if wandb.run is not None:
            wandb.finish()


def string_to_rank_integer(predicted_ranking, ground_truth_ranking):

    # Create dictionaries to map words to their ground truth ranks
    word_to_rank = {word: i + 1 for i, word in enumerate(ground_truth_ranking)}
      # Convert predicted ranking to numerical ranks based on ground truth order

    predicted_ranks = [word_to_rank[word] for word in predicted_ranking]
    ground_truth_ranks = [i+1 for i in range(len(ground_truth_ranking))]
    return predicted_ranks, ground_truth_ranks

def string_to_rank_integer_with_penalty(predicted_ranking, ground_truth_ranking):
    """
    Converts string rankings to integer ranks, handling missing words by assigning the worst rank.
    """
    word_to_rank = {word: i + 1 for i, word in enumerate(ground_truth_ranking)}
    predicted_ranks = []
    penalty_rank = len(ground_truth_ranking) + 1 # Worst possible rank

    for word in predicted_ranking:
        if word in word_to_rank:
            predicted_ranks.append(word_to_rank[word])
        else:
            predicted_ranks.append(penalty_rank)  # Apply penalty

    # ground truth rank should be always [1,2,3,4,....]
    ground_truth_ranks = list(range(1, len(ground_truth_ranking) + 1))

    #IMPORTANT: Pad the predicted_ranks with the penalty rank until it's the same length as ground_truth_ranks
    while len(predicted_ranks) < len(ground_truth_ranks):
      predicted_ranks.append(penalty_rank)

    return predicted_ranks, ground_truth_ranks

def spearman_score(predicted_ranking, ground_truth_ranking):
    """
    Calculates Spearman's rank correlation coefficient.

    Args:
        predicted_ranking: A list of predicted ranks (integers).
        ground_truth_ranking: A list of ground truth ranks (integers).

    Returns:
        The Spearman correlation coefficient (float).
        The p-value (float). We usually don't use pvalue here.
    """
    correlation, pvalue = spearmanr(predicted_ranking, ground_truth_ranking)
    if pd.isnull(correlation):
        correlation = -1.0
    return correlation

    # Example:
    # predicted = ['abc', 'ddd', '1213', '4']  # Model's predicted ranking
    # ground_truth = ['abc', '1213', 'ddd', '4'] # Correct ranking
    # predicted_ranks, ground_truth_ranks = string_to_rank_integer(predicted, ground_truth)
    # score = spearman_score(predicted_ranks, ground_truth_ranks)
    # print(f"Spearman's Rank Correlation: {score:.3f}")

    # predicted = ['abc', 'ddd', '1213', '4']  # Model's predicted ranking
    # ground_truth = ['abc', 'ddd', '1213', '4'] # Correct ranking
    # predicted_ranks, ground_truth_ranks = string_to_rank_integer(predicted, ground_truth)
    # score = spearman_score(predicted_ranks, ground_truth_ranks)
    # print(f"Spearman's Rank Correlation: {score:.3f}")



    # predicted = ['abc', 'ddd', ]  # Model's predicted ranking
    # ground_truth = ['abc', 'ddd', '1213', '4'] # Correct ranking
    # predicted_ranks, ground_truth_ranks = string_to_rank_integer_with_penalty(predicted, ground_truth)
    # score = spearman_score(predicted_ranks, ground_truth_ranks)
    # print(f"Spearman's Rank Correlation: {score:.3f}")


    # predicted = ['n']  # Model's predicted ranking
    # ground_truth = ['abc', 'ddd', '1213', '4'] # Correct ranking
    # predicted_ranks, ground_truth_ranks = string_to_rank_integer_with_penalty(predicted, ground_truth)
    # score = spearman_score(predicted_ranks, ground_truth_ranks)
    # print(f"Spearman's Rank Correlation: {score:.3f}")
    
def ranking_test_model_output(
    ppo_ranking_params,
    model_params,
):
    model_type = model_params["model_type"]
    test_vanilla = model_params["test_vanilla"]
    
    dataset_name_lst = ppo_ranking_params["dataset_name_lst"]
    
    batch_size = model_params["batch_size"]
    

    for dataset_name in dataset_name_lst:
        
        if test_vanilla:
            model_path = model_params["model_path"]['vanilla'][model_type]
        else:
            model_path = model_params["model_path"][dataset_name]["model_path"]
            if "qwen" in model_path and dataset_name in ["swow_en", "swow_us"]:
                continue # skip the qwen model for swow_en
            # elif "llama" in model_path and dataset_name == "swow_zh":
            #     continue # skip the llama model for swow_zh
        ##########
        # WANDB
        ###########

        wandb.init(
            project="llamafactory",
            name=f"ranking_q_before_training_{dataset_name}_{model_type}_{'vanilla' if test_vanilla else 'fine_tuned'}{'_lora' if 'lora' in model_path else ''}",
            tags=["ppo_ranking", "before_training"],
            config=dict(ppo_ranking_params),
        )
        
        
        ################
        # Model
        ################

        logger.info(f"Loading model from {model_path}")
        
        
        pipeline = transformers.pipeline(
            "text-generation",
            model=model_path,
            model_kwargs={
                'torch_dtype': 'auto',
            },
            device_map="auto",
            batch_size=batch_size,
        )
        
        # set pad_token_id 
        pipeline.tokenizer.pad_token_id = pipeline.tokenizer.eos_token_id
        
        # set pad_token_id for the model
        pipeline.model.config.pad_token_id = pipeline.tokenizer.eos_token_id
        
        
        messages = [
            {"role": "system", "content": "You are a chatbot who loves to speak with people about amazing stories!"},
            {"role": "user", "content": "Who are you?"},
        ]
        
        outputs = pipeline(
            messages,
            max_new_tokens=256,
        )
        
        print("Test common output")
        print(outputs)
    
    
        ################
        # Dataset
        ################
        
        logger.info(f"Loading dataset {dataset_name}")
        
        dataset = load_dataset(
            os.path.join(WORKING_DIR, "data/03_primary/llm_swow_finetune_dataset"),
            dataset_name,
            split="trl_ranking",  # ! change to trl_ranking
        )
        
        # collect GT_answer dict 
        gt_answer_dict = {}
        for data in tqdm(dataset, desc=f"Collecting GT answer dict for {dataset_name}"):
            gt_answer_dict[data['input']] = eval(data['output'])
        
        ################
        # Evaluation
        ################
        
        data_dict_lst = [] 
        
        batch_messages = []
        batch_GT_answer = []
        
        spearmanr_score_lst = []
        count = 0
        first_time_log = True 
        for data in tqdm(dataset, desc=f"Evaluating model output for {dataset_name}"):
            messages = [
                {"role": "system", "content": data['system']},
                {"role": "user", "content": data['instruction'] + '\n' + data['input']},
            ]
            
            GT_answer = eval(data['output'])
            
            batch_messages.append(messages)
            batch_GT_answer.append(GT_answer)
            
            if len(batch_messages) % batch_size  == 0:
                
                outputs = pipeline(
                    batch_messages,
                    max_new_tokens=1000,
                    pad_token_id=pipeline.tokenizer.eos_token_id,
                    batch_size=batch_size,
                )
                for idx, output in enumerate(outputs):
                    response = output[0]['generated_text'][-1]['content']
                    query = output[0]['generated_text'][-2]['content']
                    
                    try:
                        if dataset_name in ["swow_en", "swow_us"]:
                            answer_line = response.lower().rfind("final ranking")
                            answer = response[answer_line + len("Final Ranking"):]
                        elif dataset_name == "swow_zh":
                            answer_line = response.lower().rfind("最终排名")
                            answer = response[answer_line + len("最终排名"):]
                            
                        found_lst = re.findall(r"(\d+)\s*[:：]\s*(.*)", answer)
                        parsed_dict= dict()
                        for rank, word in found_lst:
                            if int(rank) not in parsed_dict:
                                parsed_dict[int(rank)] = word.strip().replace('*', '')
                        # sort the dict by key
                        parsed_dict = sorted(parsed_dict.items(), key=lambda x: x[0])
                        parsed_dict = dict(parsed_dict)
                        answer = list(parsed_dict.values())
                    except:
                        answer = None
                    
                    data_dict = {
                        "query": query,
                        "response": response,
                        "GT_answer": str(batch_GT_answer[idx]),
                        "parsed_answer": str(answer),
                    }
                    data_dict_lst.append(data_dict)
                    
                    # calculate spearmanr score
                    try:
                        # remove answer that is not in the batch_GT_answer[idx]
                        local_batch_GT_answer_for_spearmanr = [str(x).lower() for x in batch_GT_answer[idx]]
                        answer_for_spearmanr = [x.lower() for x in answer if x.lower() in local_batch_GT_answer_for_spearmanr]
                        
                        predicted_ranks, ground_truth_ranks = string_to_rank_integer_with_penalty(answer_for_spearmanr, local_batch_GT_answer_for_spearmanr)
                        
                        spearmanr_score = spearman_score(predicted_ranks, ground_truth_ranks)
                    except Exception as e:
                        spearmanr_score = -1.0
                    
                    spearmanr_score_lst.append(spearmanr_score)
                        
                    count += 1
                    
                batch_messages.clear()
                batch_GT_answer.clear()
            
                # log the average accuracy
                
                avg_spearmanr_score = sum(spearmanr_score_lst) / len(spearmanr_score_lst)
                wandb.log({"average spearmanr score": avg_spearmanr_score})
                
                if first_time_log: # log the first batch of data
                    interm_data_df = pd.DataFrame(data_dict_lst)
                    wandb.log({"intermediate": wandb.Table(dataframe=interm_data_df)})
                    first_time_log = False
            
        data_df = pd.DataFrame(data_dict_lst)
        wandb.log({"data_df": wandb.Table(dataframe=data_df)})
            
            
        # close wandb
        if wandb.run is not None:
            wandb.finish()