import os
from pathlib import Path

# Default to a repo-local token file (ignored by git)
DEFAULT_TOKEN_FILE = Path(__file__).resolve().parent.parent / "hf_token.txt"

def get_hf_auth_token(token_file_path: str | None = None) -> str:
    """
    Returns a Hugging Face token.

    Priority:
    1) HF_TOKEN environment variable (if set)
    2) token_file_path (if provided and exists)
    3) language_modelling/hf_token.txt (if exists)

    Raises:
        FileNotFoundError if no token is found.
    """
    env_token = os.getenv("HF_TOKEN")
    if env_token:
        return env_token.strip()

    p = Path(token_file_path) if token_file_path else DEFAULT_TOKEN_FILE
    if p.exists():
        return p.read_text(encoding="utf-8").strip()

    raise FileNotFoundError(
        "Hugging Face token not found. Set HF_TOKEN env var or create "
        f"{DEFAULT_TOKEN_FILE} with your token."
    )
