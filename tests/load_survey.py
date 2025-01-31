from datasets import load_dataset
import json
import ast

# Load the dataset
dataset = load_dataset("Anthropic/llm_global_opinions")

# Function to parse string representations of dictionaries
def parse_selections(selection_string):
    """Convert string representation of a defaultdict to a Python dict."""
    try:
        # Use `ast.literal_eval` to safely evaluate the string as a Python dictionary
        return ast.literal_eval(selection_string)
    except Exception as e:
        print(f"Error parsing selections: {e}")
        return {}

# Function to parse options
def parse_options(options_string):
    """Convert string representation of options to a Python list."""
    try:
        # Use `ast.literal_eval` to safely evaluate the string as a Python list
        return ast.literal_eval(options_string)
    except Exception as e:
        print(f"Error parsing options: {e}")
        return []

# Filter relevant questions for US and China from the WVS source
relevant_questions = [
    row for row in dataset['train'] 
    if 'China' in row['selections'] and 'United States' in row['selections'] and row['source'] == 'WVS'
]

# Extract data and create a list of dictionaries
questions_metadata = []
for row in relevant_questions:
    try:
        # Convert the 'selections' and 'options' strings to Python objects
        selections = parse_selections(row['selections'].split(',',1)[1].replace(')',''))
        options = parse_options(row['options'])
        
        # Extract the question and scores for US and China
        question_text = f"Question: {row['question']}\nOptions:\n- " + "\n- ".join(options)
        us_score = selections.get('United States')
        china_score = selections.get('China')
        
        # Create an entry for the question metadata
        question_entry = {
            "us_question": question_text,
            "us_score": us_score,
            "china_score": china_score
        }
        questions_metadata.append(question_entry)
    except Exception as e:
        print(f"Error processing row: {e}")

# Define the output file path
output_file_path = "/data/gpfs/projects/punim2219/LM_with_SWOW/kabir/Data/WV_Bench/us_question_metadata.json"

# Save the results to a JSON file
with open(output_file_path, 'w') as f:
    json.dump(questions_metadata, f, indent=4)

print(f"Saved {len(questions_metadata)} questions to {output_file_path}")
