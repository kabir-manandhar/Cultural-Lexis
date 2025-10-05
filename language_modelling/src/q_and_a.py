import re
import json
import os 
os.environ["VLLM_DISABLE_USAGE_STATS"] = "1"
os.environ["VLLM_REPORT_USAGE"] = "false"
import numpy as np
from tqdm import tqdm
from vllm import SamplingParams, LLM
from vllm.sampling_params import GuidedDecodingParams
from .prompt_utils import create_system_prompt
from .swow_utils import augment_with_swow


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


def build_prompt(question: str, options: list, system_prompt: str, use_swow: bool, country_name: str, llm: LLM) -> str:
    """
    Constructs the final prompt with the question and numbered options.
    
    If `use_swow` is True, the question is augmented with cultural context.

    Args:
        question (str): The original question.
        options (list): List of options.
        system_prompt (str): System prompt for context.
        use_swow (bool): Whether to use SWOW-based augmentation.
        country_name (str): Country for cultural context.

    Returns:
        str: Formatted prompt for the model.
    """
    if use_swow:
        question = augment_with_swow(question, country_name, llm)  # Add cultural context
        
    question_text = question.split("Options:")[0].strip()
    options_text = "\n".join([f"{i + 1}. {option}" for i, option in enumerate(options)])

    return f"""
{system_prompt}

{question_text}

Options:
{options_text}

Please respond only with the option number (1-{len(options)}).
""".strip()


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
    choice_logprobs = {str(key): float('-inf') for key in choice_map}

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

    return {choice_map[choice]: (prob / total_prob) * 100 for choice, prob in choice_probs.items()}


def display_probabilities(normalized_probs: dict):
    """
    Displays the choice probabilities in a readable format.

    Args:
        normalized_probs (dict): Normalized probabilities for each choice.
    """
    print("\n📊 Category Probabilities:")
    for category, prob in normalized_probs.items():
        print(f"{category}: {prob:.2f}%")


def answer_question(question: str, llm: LLM, country_name: str, use_swow: bool) -> dict:
    """
    Generates an answer for the given question using vLLM with multiple-choice options.

    Args:
        question (str): The question to answer.
        llm: The vLLM model instance.
        country_name (str): The country context for the system prompt.
        use_swow (bool): Whether to augment the question with SWOW-based cultural context.

    Returns:
        dict: A dictionary containing the question, generated answer, and log probabilities.
    """

    # 1. Extract options from the question
    options = extract_options_from_question(question)
    if not options:
        return {"question": question, "error": "No valid options found."}

    # 2. Create choice map (e.g., {'1': 'Yes', '2': 'No'})
    choice_map = create_choice_map(options)

    # 3. Build the prompt for the LLM
    system_prompt = create_system_prompt(country_name)
    prompt = build_prompt(question, options, system_prompt, use_swow, country_name, llm)

    # 4. Configure sampling parameters
    sampling_params = configure_sampling_params(choice_map)

    # 5. Generate model response
    outputs = llm.generate(prompts=prompt, sampling_params=sampling_params)
    breakpoint()
    generated_choice = outputs[0].outputs[0].text.strip()

    # 6. Collect log probabilities
    token_logprobs = outputs[0].outputs[0].logprobs[0]
    choice_logprobs = get_choice_logprobs(token_logprobs, choice_map)

    # 7. Convert logprobs to probabilities
    normalized_probs = convert_logprobs_to_percentages(choice_logprobs, choice_map)

    # 8. Display results
    display_probabilities(normalized_probs)

    # 9. Prepare result dictionary
    result = {
        "question": question,
        "generated_choice": generated_choice,
        "generated_category": choice_map.get(generated_choice, "Unknown"),
        "choice_logprobs": choice_logprobs,
        "normalized_probs": normalized_probs
    }

    return result
