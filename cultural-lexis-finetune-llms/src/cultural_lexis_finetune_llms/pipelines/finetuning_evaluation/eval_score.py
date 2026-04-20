from thefuzz import fuzz
import numpy as np

from scipy.stats import spearmanr
import pandas as pd 
import re

def eval_score(
    response_text_lst: list,
    cue_word_lst: list,
    associated_word_freq_dict: dict, # structure {"cue_word": {"associated_word": freq}}
    top_k_val: int,
):
    """
    Evaluates the score of responses based on their association with cue words.
    This function calculates a score for each response by comparing the response words
    against a dictionary of known word associations and their frequencies. The score is
    based on how well the response words match the top-k most frequent associations.
    Parameters:
    ----------
    response_text_lst : list
        List of response texts, where each text contains comma-separated words
    cue_word_lst : list
        List of cue words corresponding to each response text
    associated_word_freq_dict : dict
        Dictionary with structure {"cue_word": {"associated_word": freq}}
        Contains the frequency of association between cue words and their associated words
    top_k_val : int
        Number of top frequent associations to consider for each cue word
    Returns:
    -------
    numpy.ndarray
        Array of scores (shape: (n,)) where n is the number of responses
        Each score is the sum of frequencies of matched words divided by 
        total frequency of top-k associated words
    Notes:
    -----
    - Handles both ASCII and non-ASCII text (using ',' and '，' as delimiters respectively)
    - Removes duplicate words in responses before scoring
    - Score range is [0, 1] where higher scores indicate better matches with frequent associations
    """
    
    score_lst = []
    output_res_lst = []
    # the response text need to split using ',' or '，' if it is in Chinese
    for response_text, cue_word in zip(response_text_lst, cue_word_lst):
        if response_text.isascii():
            # use , to split the response text
            response_words = [x.strip().lower() for x in response_text.split(',')]
        else:
            # use ， to split the response text
            response_words = [x.strip().lower() for x in response_text.split('，')]
            
        response_words_re = re.findall(r'^\d+\.\s+(.+)$', response_text, flags=re.MULTILINE) # just handle the llama vanilla case 
        
        if len(response_words_re) > len(response_words):
            response_words = response_words_re
            response_words = [word.strip().lower() for word in response_words]
        
        # make it set to remove duplicates
        new_list = []
        for word in response_words:
            if word not in new_list:
                new_list.append(word)
        response_words = new_list
        
        output_res_lst.append(" | ".join(response_words))

        cue_word = cue_word.strip()
        
        # get the top k associated words from the dictionary
        # sort the associated words by frequency
        associated_words = list(associated_word_freq_dict[cue_word].keys())
        associated_words = sorted(associated_words, key=lambda x: associated_word_freq_dict[cue_word][x], reverse=True)
        associated_words = associated_words[:top_k_val]
        
        # response words and associated words replace "  " with " "
        response_words_fmted = [word.replace("  ", " ").lower() for word in response_words if isinstance(word, str)]
        associated_words_fmted = [word.replace("  ", " ").lower() for word in associated_words if isinstance(word, str)]
        
        # get the sum of frequency
        sum_freq= 0
        for word in response_words_fmted:
            if word in associated_words_fmted:
                word_idx = associated_words_fmted.index(word)
                sum_freq += associated_word_freq_dict[cue_word][associated_words[word_idx]]
                
        # get the sum of frequency of all the associated words of the cue word
        total_sum_freq = 0
        for word in associated_words:
            total_sum_freq += associated_word_freq_dict[cue_word][word]
            
        score = sum_freq / total_sum_freq
        score_lst.append(score)
        
    return np.array(score_lst), output_res_lst # shape (n, )


def eval_score_wordties(
    response_text_lst: list,
    cue_word_lst: list,
    associated_word_freq_dict: dict,  # structure {"cue_word": {"associated_word": freq}}
):
    """
    Evaluates the responses using WordTies metrics: precision@k and pooled Spearman correlation.
    
    Parameters:
    ----------
    response_text_lst : list
        List of response texts, where each text contains comma-separated words.
    cue_word_lst : list
        List of cue words corresponding to each response text.
    associated_word_freq_dict : dict
        Dictionary with structure {"cue_word": {"associated_word": freq}} containing the 
        frequency of association between cue words and their associated words.
        
    Returns:
    -------
    dict
        A dictionary containing the average precision@k for k in [5,10,20,30,40,50] and 
        the pooled Spearman correlation with p-value.
    """
    # Initialize storage for precision@k results
    precisions_at_k = {k: [] for k in [5, 10, 20, 30, 40, 50]}
    all_gold_ranks = []
    all_pred_ranks = []
    
    output_res_lst = []
    
    for response_text, cue in zip(response_text_lst, cue_word_lst):
        cue = cue.strip().lower()
        if cue not in associated_word_freq_dict:
            continue  # Skip cues not present in the gold data
        
        # Split response text into words, handling ASCII and non-ASCII delimiters
        response_text_clean = response_text.strip()
        if response_text_clean.isascii():
            response_words = [word.strip().lower() for word in response_text_clean.split(',')]
        else:
            response_words_v1 = [word.strip().lower() for word in response_text_clean.split('，')]
            
            response_words_v2 = [word.strip().lower() for word in response_text_clean.split('、')]
            
            if len(response_words_v1) > len(response_words_v2):
                response_words = response_words_v1
            else:
                response_words = response_words_v2
            
        response_words_re = re.findall(r'^\d+\.\s+(.+)$', response_text, flags=re.MULTILINE) # just handle the llama vanilla case 
        
        if len(response_words_re) > len(response_words):
            response_words = response_words_re
            response_words = [word.strip().lower() for word in response_words]
            
        # make it set to remove duplicates
        new_list = []
        for word in response_words:
            if word not in new_list:
                new_list.append(word)
        response_words = new_list
        
        output_res_lst.append(" | ".join(response_words))
        
        # Deduplicate while preserving order
        seen = set()
        deduped_response_words = []
        for word in response_words:
            if word not in seen and word:
                seen.add(word)
                deduped_response_words.append(word)
        response_words = deduped_response_words
        
        # Get gold associations for the current cue
        gold_word_freq = associated_word_freq_dict.get(cue, {})
        gold_assocs = set(gold_word_freq.keys())
        
        # Calculate precision@k for each k
        for k in precisions_at_k:
            topk = response_words[:k]
            overlap = len([word for word in topk if word in gold_assocs])
            prec = overlap / k
            precisions_at_k[k].append(prec)
        
        # Prepare data for Spearman correlation
        overlapping_words = [word for word in response_words if word in gold_assocs]
        if len(overlapping_words) < 2:
            continue  # Not enough overlapping words for correlation
        
        # Sort gold words by frequency descending to determine ranks
        sorted_gold = sorted(gold_word_freq.items(), key=lambda x: (int(-x[1]), str(x[0])))
        gold_rank_dict = {word: rank + 1 for rank, (word, _) in enumerate(sorted_gold)}
        
        # Predicted ranks based on response order (1-based)
        predicted_rank_dict = {word: idx + 1 for idx, word in enumerate(response_words)}
        
        # Collect ranks for overlapping words
        gold_ranks = []
        pred_ranks = []
        for word in overlapping_words:
            gr = gold_rank_dict.get(word)
            pr = predicted_rank_dict.get(word)
            if gr is not None and pr is not None:
                gold_ranks.append(gr)
                pred_ranks.append(pr)
        
        if len(gold_ranks) >= 2:
            all_gold_ranks.extend(gold_ranks)
            all_pred_ranks.extend(pred_ranks)
    
    # Compute average precision@k
    avg_precisions = {}
    for k in precisions_at_k:
        if not precisions_at_k[k]:
            avg_prec = 0.0
        else:
            avg_prec = np.mean(precisions_at_k[k])
        avg_precisions[f'prec_at_{k}'] = round(avg_prec, 3)
    
    # Compute pooled Spearman correlation
    pooled_spearman = {'spearman': None, 'spearman_p': None}
    if len(all_gold_ranks) >= 2 and len(all_pred_ranks) >= 2:
        corr, p_value = spearmanr(all_gold_ranks, all_pred_ranks)
        pooled_spearman['spearman'] = round(corr, 3)
        pooled_spearman['spearman_p'] = round(p_value, 3)
    
    output_dict = {
        **avg_precisions,
        **pooled_spearman
    }
    
    return output_dict, output_res_lst
    
    
    

SWOW_EN_FEW_SHOT_EXAMPLE = """
Task:
petal
assistant: rose, pink, metal, bike, leaf, pedal, color, delicate, sepal, spring, bloom, stamen, pusher, pollen, plant, love, colorful, bud, purple, tulip, oval, beautiful, girl, pick, wedding, beauty, piano, falling, koala, scent, flowers, summer, boat, colors, fairy, drifting, beg, thorn, red, pot  pourri, iris, work, gas, bush, blade, daffodil, nature, power, to  the  metal, gentle, flower  growing, round, little  girl, silky, leaves

Task:
"""

SWOW_ZH_FEW_SHOT_EXAMPLE = """
Task:
挤
assistant: 人， 拥挤， 庙会， 电车， 人多， 难受， 窒息， 火车， 施力， 电梯， 蛋黄酱， 臭， 讨厌， 暖和， 旅游， 一号线， 柠檬， 牙膏， 罐头， 混乱， 出汗， 过节， 回家， 乱， 春运， 捷运， 不舒服， 胖， 时间， 摩肩接踵， 空气， 汗水， 吃橙子， 溢出， 疼痛， 压， 躁， 素质， 痘痘， 大城市， 白眼， 死人， 广场， 公园， 北京， 占， 车站， 公共交通工具， 性骚扰， 比肩接踵， 橡胶

Task:
"""