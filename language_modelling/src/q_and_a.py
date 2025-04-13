import re
from tqdm import tqdm
from .prompt_utils import create_system_prompt
import json
import numpy as np
from vllm import SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from src.evaluate import earth_movers_distance 
from scipy.special import rel_entr
from scipy.special import kl_div
# from sklearn.metrics import mutual_info_score
from scipy.stats import entropy


def extract_options_from_question(question: str) -> list:
    """
    Extracts options from a question string.
    Assumes options are listed under 'Options:' followed by '- ' for each choice.

    Args:
        question (str): The input question string containing options.

    Returns:
        list: A list of extracted options.
    """
    match = re.search(r"Options:\s*\n((?:- .*\n)+)", question)
    if match:
        options_text = match.group(1)
        options = [opt.strip("- ").strip() for opt in options_text.strip().split("\n")]
        return options
    else:
        return []


def create_choice_map(options: list) -> dict:
    """
    Maps options to numbered choices (1, 2, 3, ...).

    Args:
        options (list): List of options extracted from the question.

    Returns:
        dict: Mapping of choice numbers to option text.
    """
    return {str(i + 1): option for i, option in enumerate(options)}


def build_prompt(question: str, options: dict, system_prompt: str) -> str:
    """
    Constructs the final prompt with the question and numbered options.

    Args:
        question (str): The original question.
        options (list): List of options.
        system_prompt (str): System prompt for context.

    Returns:
        str: Formatted prompt for the model.
    """
    question_text = question.split("Options:")[0].strip()

    # options_text = "\n".join([f"{i + 1}. {option}" for i, option in enumerate(options)])
    options_text = ', '.join([f"{key}: {value}" for key, value in options.items()])


    prompt = f"""
{system_prompt}

{question_text}

Options:
{options_text}

Please respond only with the option number (1-{len(options)}).
"""
    return prompt.strip()


def configure_sampling_params(choice_map: dict) -> SamplingParams:
    """
    Configures sampling parameters for the model, including guided decoding.

    Args:
        choice_map (dict): Mapping of choices (e.g., {'1': 'Yes', '2': 'No'}).

    Returns:
        SamplingParams: Sampling configuration for the LLM.
    """
    guided_params = GuidedDecodingParams(choice=list(choice_map.keys()))

    return SamplingParams(
        guided_decoding=guided_params,
        max_tokens=1,
        temperature=0.0,
        top_p=1.0,
        logprobs=len(choice_map)
    )


def get_choice_logprobs(token_logprobs, choice_map):
    """
    Extracts log probabilities for each choice from the model output.

    Args:
        token_logprobs (dict): Log probabilities from the LLM response.
        choice_map (dict): Mapping of choice numbers to options.

    Returns:
        dict: Log probabilities for each choice.
    """
    choice_logprobs = {str(key): float('-inf') for key in choice_map.keys()}

    for logprob_obj in token_logprobs.values():
        decoded_token = logprob_obj.decoded_token.strip()
        if decoded_token in choice_logprobs:
            choice_logprobs[decoded_token] = logprob_obj.logprob

    return choice_logprobs


def convert_logprobs_to_percentages(choice_logprobs: dict, choice_map: dict) -> dict:
    """
    Converts log probabilities to normalized percentages.

    Args:
        choice_logprobs (dict): Log probabilities for each choice.
        choice_map (dict): Mapping of choice numbers to options.

    Returns:
        dict: Normalized probabilities for each choice.
    """
    choice_probs = {choice: np.exp(logprob) for choice, logprob in choice_logprobs.items()}
    total_prob = sum(choice_probs.values())

    # return {choice_map[choice]: (prob / total_prob) * 100 for choice, prob in choice_probs.items()}
    choice_content_logprobs = {choice_map[choice]: (prob / total_prob)  for choice, prob in choice_probs.items()}

    choice_logprobs_normalized = {choice: (prob / total_prob)  for choice, prob in choice_probs.items()}

    return choice_logprobs_normalized, choice_content_logprobs


def display_probabilities(normalized_probs: dict):
    """
    Displays the choice probabilities in a readable format.

    Args:
        normalized_probs (dict): Normalized probabilities for each choice.
    """
    print("\n📊 Category Probabilities:")
    for category, prob in normalized_probs.items():
        prob = prob*100
        print(f"{category}: {prob:.2f}%")


def answer_question(question: str, llm, country_name: str, gt_answer: list, choice_map: dict) -> dict:
    """
    Generates an answer for the given question using vLLM with multiple-choice options.

    Args:
        question (str): The question to answer.
        llm: The vLLM model instance.
        country_name (str): The country context for the system prompt.

    Returns:
        dict: A dictionary containing the question, generated answer, and log probabilities.
    """

    #1. Get the choie map 
    if choice_map is None: 
        # 1. Extract options from the question
        options = extract_options_from_question(question)
        if not options:
            return {"question": question, "error": "No valid options found."}

        # 2. Create choice map (e.g., {'1': 'Yes', '2': 'No'})
        choice_map = create_choice_map(options)

    # 3. Build the prompt for the LLM
    system_prompt = create_system_prompt(country_name)
    prompt = build_prompt(question, options=choice_map, system_prompt=system_prompt)

    # 4. Configure sampling parameters
    sampling_params = configure_sampling_params(choice_map)

    # 5. Generate model response
    outputs = llm.generate(prompts=prompt, sampling_params=sampling_params)
    generated_choice = outputs[0].outputs[0].text.strip()

    # 6. Collect log probabilities
    token_logprobs = outputs[0].outputs[0].logprobs[0]
    choice_logprobs = get_choice_logprobs(token_logprobs, choice_map)

    # 7. Convert logprobs to probabilities
    choice_logprobs_normalized, choice_logprobs_normalized_content = convert_logprobs_to_percentages(choice_logprobs, choice_map)
    # normalized_probs = convert_logprobs_to_percentages(choice_logprobs, choice_map)
    
    # 8. Display results
    display_probabilities(choice_logprobs_normalized_content)
    # breakpoint()

    # 9. Evaluate the results 
    # Check if they have the same keys
    keys_match = set(choice_logprobs_normalized.keys()) == set(gt_answer.keys())

    # If keys match, get aligned values
    if keys_match:
        choice_values, gt_values = zip(*[(choice_logprobs_normalized[k], gt_answer[k]) for k in gt_answer])
    else:
        print("Warning: Dictionaries have different keys")
        # Get values only for shared keys
        shared_keys = set(choice_logprobs_normalized.keys()) & set(gt_answer.keys())
        choice_values, gt_values = zip(*[(choice_logprobs_normalized[k], gt_answer[k]) for k in shared_keys])
        
    # em_score = earth_movers_distance(list(gt_answer.values()), list(choice_logprobs_normalized_content.values())) #TODO: make the keys of the two dicts the same, ensuring the order of values are the same
    em_score = earth_movers_distance(gt_values, choice_values)
    print("== EM_SCORE", em_score)
    
    # Calculate KL divergence
    # Convert to numpy arrays and add small epsilon to avoid zeros in log calculation
    # breakpoint()
    # epsilon = 1e-10
    # gt_array = np.array(gt_values) + epsilon
    # choice_array = np.array(choice_values) + epsilon

    # # Calculate and print metrics
    # kl_gt_to_pred = entropy(gt_array, choice_array)
    # kl_pred_to_gt = entropy(choice_array, gt_array)

    # m_array = 0.5 * (gt_array + choice_array)
    # js_div = 0.5 * (entropy(gt_array, m_array) + entropy(choice_array, m_array))

    
    # gt_array = np.array(gt_values, dtype=np.float64)  # Ensure float64 for precision
    # choice_array = np.array(choice_values, dtype=np.float64)

    # # Better epsilon handling
    # epsilon = 1e-10
    # nonzero_mask = (gt_array <= 0) | (choice_array <= 0)
    # if np.any(nonzero_mask):
    #     # Replace zeros safely
    #     gt_array = np.maximum(gt_array, epsilon)
    #     choice_array = np.maximum(choice_array, epsilon)
        
    #     # Re-normalize after adding epsilon
    #     gt_array = gt_array / np.sum(gt_array)
    #     choice_array = choice_array / np.sum(choice_array)

    # try:
    #     # Calculate metrics with error checking
    #     kl_gt_to_pred = entropy(gt_array, choice_array)
    #     kl_pred_to_gt = entropy(choice_array, gt_array)
        
    #     # Calculate Jensen-Shannon divergence
    #     m_array = 0.5 * (gt_array + choice_array)
    #     js_div = 0.5 * (entropy(gt_array, m_array) + entropy(choice_array, m_array))
        
    #     print(f"EM: {em_score:.4f}, KL(GT→Pred): {kl_gt_to_pred:.4f}, " 
    #         f"KL(Pred→GT): {kl_pred_to_gt:.4f}, JS: {js_div:.4f}")
    # except Exception as e:
    #     print(f"Error in entropy calculation: {e}")
    #     print(f"gt_array: {gt_array}")
    #     print(f"choice_array: {choice_array}")
    # # breakpoint()


    # 10. Prepare result dictionary
    result = {
        "question": question,
        "survey_scores": gt_answer,
        "generated_choice": generated_choice,
        "generated_category": choice_map.get(generated_choice, "Unknown"),
        "choice_logprobs": choice_logprobs,
        "choice_values": choice_values,
        "gt_values": gt_values,
        "choice_logprobs_normalized_content": choice_logprobs_normalized_content,
        "earth_mover_distance": em_score,
        # "kl_gt_to_pred": kl_gt_to_pred, 
        # "kl_pred_to_gt": kl_pred_to_gt, 
        # "jensen_shannon_divergence": js_div
    }

    return result
