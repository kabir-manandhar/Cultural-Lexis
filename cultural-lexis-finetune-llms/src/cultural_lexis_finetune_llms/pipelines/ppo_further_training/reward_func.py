import torch
import time 
import os 

import pandas as pd
from copy import deepcopy
from scipy.stats import spearmanr
import re


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

def reward_func(queries, prompts, labels, dataset_name=None):
    # queries is prompts + responses
    # labels is answers
    if dataset_name is None:
        dataset_name = os.environ['DATA_NAME']
    spearmanr_score_lst = []
    for idx, query in enumerate(queries):
        response = queries[idx][len(prompts[idx]) - 1:]
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
        except Exception as e:
            print('Error in parsing answer:', e)
            # raise e
            answer = None


        # calculate spearmanr score
        try:
            # get the GT 
            batch_GT_answer = eval(labels[idx])
            # remove answer that is not in the GT
            local_batch_GT_answer_for_spearmanr = [str(x).lower() for x in batch_GT_answer]
            
            answer_for_spearmanr = [x.lower() for x in answer if x.lower() in local_batch_GT_answer_for_spearmanr]
            
            predicted_ranks, ground_truth_ranks = string_to_rank_integer_with_penalty(answer_for_spearmanr, local_batch_GT_answer_for_spearmanr)
            
            spearmanr_score = spearman_score(predicted_ranks, ground_truth_ranks)
        except Exception as e:
            print('Error in calculating spearmanr score:', e)
            # raise e
            spearmanr_score = -1.0

        spearmanr_score_lst.append(spearmanr_score)
        
    # convert to tensor
    reward = torch.tensor(spearmanr_score_lst, dtype=torch.float32)
    # same shape as queries
    return reward


# ray job submit --address="http://127.0.0.1:8265" \
    # --runtime-env-json='{"working_dir": "/openrlhf"}' \
    # -- python3 -m openrlhf.cli.train_ppo_ray \
    # ...
    # --remote_rm_url /path/to/reward_func.py


