import re
from tqdm import tqdm
from .prompt_utils import create_system_prompt
import json
import numpy as np
from vllm import SamplingParams

# Optional: A helper function to process log probabilities from a vLLM response.
def process_logprobs(response, choices):
    """
    Extract and normalize log probabilities for specified choices from a vLLM response.
    
    Args:
        response: The response object from vLLM.generate (assumed to have a .logprobs attribute).
        choices (list): A list of valid choices (as strings) to look for.
        
    Returns:
        dict: Normalized probabilities for each choice.
    """
    
    token_logprobs = response.logprobs[0]
    probs_raw = []
    
    # Loop over each expected choice and extract its log probability
    for choice in choices:
        for logprob_obj in token_logprobs.values():
            if logprob_obj.decoded_token == str(choice):
                probs_raw.append(np.exp(logprob_obj.logprob))
                break

    total = sum(probs_raw)
    if total > 0:
        probs = [p / total for p in probs_raw]
    else:
        probs = [0] * len(choices)
    
    return dict(zip(choices, probs))

def answer_question(question: str, llm, country_name: str) -> dict:
    """
    Generates an answer for the given question using vLLM.
    
    Instead of handling tokenization manually, we construct a plain-text prompt by combining 
    the system prompt (context) and the user question, and then use vLLM to generate the answer.
    
    Args:
        question (str): The question to answer.
        llm: The vLLM model instance.
        country_name (str): The country context for which to build the system prompt.
    
    Returns:
        dict: A dictionary containing the original question, the generated answer, 
              and optionally, the processed log probability distribution.
    """
    
    # Create the system prompt using our prompt utility
    system_prompt = create_system_prompt(country_name)
    
    # Build a single prompt by combining the system prompt and user question
    prompt = f"{system_prompt}\nUser: {question}\nAssistant:"
    
    # Optionally, you can configure guided decoding if you want to constrain the output.
    # For example, if you want to force outputs to be one of a set of predetermined choices,
    # you can create guided decoding parameters.
    #
    # from vllm.sampling_params import GuidedDecodingParams
    # guided_params = GuidedDecodingParams(choice=["1", "2", "3", "4"]) 
    # For now, we'll leave guided_params as None:
    guided_params = None
    
    # Set up the sampling parameters for generation.
    # Adjust max_tokens as needed; here, we allow up to 256 tokens in the generated answer.
    sampling_params = SamplingParams(
        guided_decoding=guided_params,
        max_tokens=256,
        temperature=0.0,  # Deterministic generation
        top_p=1.0,
        logprobs=20  # Request log probabilities for the top 20 tokens
    )
    
    # Generate the answer using vLLM.
    outputs = llm.generate(
        prompts=prompt,
        sampling_params=sampling_params
    )
    
    # Extract the generated answer from the vLLM output.
    # Here we take the first generated output and strip any extra whitespace.
    answer = outputs[0].outputs[0].text.strip()
    
    # (Optional) If you are using guided decoding for tasks like multiple-choice questions,
    # you can process the log probabilities to compute the confidence for each allowed choice.
    # For example, if the valid choices are ["1", "2", "3", "4"]:
    breakpoint()
    # choices = ["1", "2", "3", "4"]
    # choice_distribution = process_logprobs(outputs[0].outputs[0], choices)
    #
    # For the current example, we omit the choice_distribution.
    
    result = {
        "question": question,
        "answer": answer,
        # "choice_distribution": choice_distribution  # Include this line if applicable
    }
    
    return result
