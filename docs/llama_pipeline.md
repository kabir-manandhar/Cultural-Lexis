# Cultural Lexis Modelling Documentation 

## Project Structure

```python
cultural-lexis/
│
├── language_modelling/
│   │
│   ├── config/
│   │   ├── __init__.py
│   │   └── hf_auth.py        # Handles the reading of Hugging Face authentication token.
│   │
│   ├── src/
│   │   ├── __init__.py
│   │   ├── model_utils.py    # Contains functions related to model loading and GPU memory management.
│   │   ├── prompt_utils.py   # Contains the function for creating the system prompt.
│   │   └── classify.py       # Handles the classification of Q/A survey data.
│   │
│   ├── main.py               # Main entry point for the command-line interface.
│
└── preprocessing/            # Other unrelated preprocessing code (not relevant to this documentation).
```

## Using Poetry

This project uses Poetry for dependency management and environment setup. Poetry is a tool for dependency management and packaging in Python. It allows you to specify the project dependencies and manage virtual environments easily.

### Step-by-Step Guide to Using Poetry

- **Navigate to the Project Directory:** First, navigate to the language_modelling directory where the pyproject.toml and poetry.lock files are located:

    ```console
    cd /path/to/cultural-lexis/
    ```

- **Install Project Dependencies:** Since Poetry is already installed on the VM and you have the same Python version, you can install the dependencies specified in pyproject.toml using:

    ```console
    poetry install
    ```

- **Activate the Virtual Environment:** After installing the dependencies, you can activate the virtual environment created by Poetry. Poetry manages the environment automatically, but if you want to activate it manually, use:

    ```console
    poetry shell
    ```

- **Add new libraries:** You can add new libraries using the following code:

    ```console
    poetry add library
    ```

- **Run the Project:** You can run the project directly with Python, assuming the environment is activated:

    ```console
    python language_modelling/main.py
    ```


## Command-Line Interface

The main.py script is designed to be flexible and configurable through command-line arguments. This allows you to customize the behavior of the pipeline without modifying the code. Below are the key parameters that you can use when running the script:

### Parameters
1.  **--model_id**:

    - *Description*: The Hugging Face model ID to use for classification.
    - *Default Value*: **'meta-llama/Meta-Llama-3.1-8B-Instruct'**
    - *Usage*: If you want to use a different model, specify its ID here.

2. **--cache_dir**:
    - *Description*: Directory to store cached models and tokenizers. 
    - *Default Value*: **'/data/gpfs/projects/punim2219/LM_with_SWOW/kabir/Data/huggingface_cache'**
    - *Usage*: If you need to change the cache directory, specify the path here.

3. **--output_file**:

    - *Description*: File path to save the classification results in JSON format.
    - *Default Value*: **'/data/gpfs/projects/punim2219/LM_with_SWOW/kabir/Data/output/sample_answers.json'**
    - Usage: To save the results to a different location or filename, specify the path here.

### Example Usage: 

Here’s how you might run the script with custom parameters:

```python
python language_modelling/main.py \
    --model_id 'your-custom-model-id' \
    --cache_dir '/path/to/your/custom/cache' \
    --output_file '/path/to/your/output.json'
```


