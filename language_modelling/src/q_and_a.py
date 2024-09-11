import re
from tqdm import tqdm
from .prompt_utils import create_system_prompt
import json

def answer_question(question: str, model, tokenizer) -> dict:
    """Generates an answer for the given question using the model and tokenizer."""
    
    # Create the system prompt (which is much simpler now)
    system_prompt = create_system_prompt()

    # Construct the message to send to the model
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    
    # Convert the message into token IDs
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    # Get the end-of-sequence token ID
    eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.convert_tokens_to_ids("")

    # Generate the model's response
    outputs = model.generate(
        input_ids,
        max_new_tokens=256,
        eos_token_id=eos_token_id,
        do_sample=False,
        temperature=0.0,
        top_p=1,
    )

    # Extract the model's response and decode it
    response = outputs[0][input_ids.shape[-1]:]
    answer = tokenizer.decode(response, skip_special_tokens=True)

    # Return the question and the generated answer in a dictionary
    result = {
        "question": question,
        "answer": answer.strip()
    }

    return result
