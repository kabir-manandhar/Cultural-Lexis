from thefuzz import fuzz
import numpy as np

SCALE_REWARD= 10.0

def swow_manual_reward_coverage(response_text_lst: list,
                       reference_response_lst: list):
    
    # response_text_list should have shorter length in its response than reference_response_list
    
    score_lst = []
    for response_text, reference_response in zip(response_text_lst, reference_response_lst):
        score = fuzz.token_sort_ratio(response_text, reference_response)
        # divide by 100 to get a score between 0 and 1
        score = score * SCALE_REWARD
        score_lst.append(score / 100.0)
        
    return np.array(score_lst) # shape (batch_size,)



def swow_manual_reward_frequency(
    response_text_lst: list,
    cue_word_lst: list,
    associated_word_freq_dict: dict
):
    score_lst = []
    # the response text need to split using ',' or '，' if it is in Chinese
    for response_text, cue_word in zip(response_text_lst, cue_word_lst):
        if response_text.isascii():
            # use , to split the response text
            response_words = [x.strip() for x in response_text.split(',')]
        else:
            # use ， to split the response text
            response_words = [x.strip() for x in response_text.split('，')]
        response_words = [x.replace('.', '') for x in response_words]
            
        # make it set to remove duplicates
        response_words = list(set(response_words))

        cue_word = cue_word.strip()

        # get the sum of frequency 
        sum_freq = sum([associated_word_freq_dict[cue_word].get(word, 0) for word in response_words])
        # get the sum of frequency of all the associated words of the cue word
        sum_all_freq = sum(associated_word_freq_dict[cue_word].values())

        # calculate the reward
        score = sum_freq / sum_all_freq
        score = score * SCALE_REWARD
        score_lst.append(score)

    return np.array(score_lst) # shape (n, )