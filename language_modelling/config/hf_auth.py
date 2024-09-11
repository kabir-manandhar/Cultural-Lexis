import os

def get_hf_auth_token(token_file_path: str = '/data/gpfs/projects/punim2219/LM_with_SWOW/kabir/Cultural-Lexis/language_modelling/hf_token.txt') -> str:
    """Reads the Hugging Face token from a text file."""
    if not os.path.exists(token_file_path):
        raise FileNotFoundError(f"Token file not found: {token_file_path}")
    
    with open(token_file_path, 'r') as file:
        hf_auth = file.readline().strip()
        
    return hf_auth
