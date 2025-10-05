from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional

import pandas as pd
import typer

# keep these env flags quiet for vLLM
os.environ["VLLM_DISABLE_USAGE_STATS"] = "1"
os.environ["VLLM_REPORT_USAGE"] = "false"

# local imports
from language_modelling.config.hf_auth import get_hf_auth_token
from language_modelling.src.q_and_a import answer_question
from language_modelling.src.model_utils import load_llm, clear_gpu_memory


app = typer.Typer(help="WVS evaluation runner (vLLM + constrained decoding)")


# ---------- helpers ----------

def _ensure_hf_env(cache_dir: Optional[Path]) -> None:
    """Ensure HF auth & caches are set in the environment."""
    # token: favor existing env, else read from local file
    if not os.getenv("HF_TOKEN"):
        try:
            os.environ["HF_TOKEN"] = get_hf_auth_token()
        except FileNotFoundError:
            # token is optional if loading a local model path
            pass

    if cache_dir:
        os.environ["HF_HOME"] = str(cache_dir)
        os.environ["TRANSFORMERS_CACHE"] = str(cache_dir)


def _load_questions_from_json(json_file: Path) -> List[str]:
    """Load questions from a JSON where each item has a `question` field."""
    data = json.loads(json_file.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of objects.")
    questions = []
    for item in data:
        if "question" in item and isinstance(item["question"], str):
            questions.append(item["question"])
        elif "question_text" in item and "options_text" in item:
            questions.append(_build_question_with_options(item["question_text"], item["options_text"]))
        else:
            raise ValueError("Each JSON item must contain `question` (or `question_text` + `options_text`).")
    return questions


def _build_question_with_options(question_text: str, options_text: str) -> str:
    """
    Build a question string containing an 'Options:' block that matches the extractor in q_and_a.py.
    We standardize to:
        Question...
        Options:
        - option A
        - option B
        ...
    """
    q = question_text.strip()
    opts = options_text.strip()

    # Already formatted as bullet list?
    if "\n-" in opts or opts.startswith("- "):
        bullet_block = opts
    else:
        # try simple splits if options are delimited
        # (fallback: treat whole string as one option)
        for delim in ["\n", ";", "；", "|", "/", " or ", " OR "]:
            if delim in opts:
                parts = [p.strip() for p in opts.split(delim) if p.strip()]
                break
        else:
            parts = [opts] if opts else []
        bullet_block = "\n".join(f"- {p}" for p in parts)

    return f"{q}\n\nOptions:\n{bullet_block}\n"


def _load_questions_auto(input_file: Path, country_name: str) -> List[str]:
    """
    Auto-detect by file extension:
      - .json / .jsonl : reads `question` per item (or question_text/options_text)
      - .xlsx / .xls   : expects language-specific columns and builds options block
      - .csv           : similar to xlsx with column names
    """
    suffix = input_file.suffix.lower()
    if suffix in {".json", ".jsonl"}:
        if suffix == ".jsonl":
            lines = input_file.read_text(encoding="utf-8").splitlines()
            items = [json.loads(x) for x in lines if x.strip()]
            tmpfile = input_file.with_suffix(".json.tmp")
            tmpfile.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
            qs = _load_questions_from_json(tmpfile)
            tmpfile.unlink(missing_ok=True)
            return qs
        return _load_questions_from_json(input_file)

    # tabular: pick sensible columns
    if suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(input_file)
    elif suffix == ".csv":
        df = pd.read_csv(input_file)
    else:
        raise ValueError(f"Unsupported input file type: {suffix}")

    # try language-specific column names first
    if country_name.lower() == "china":
        q_col_candidates = ["question_zh", "question", "question_text"]
        o_col_candidates = ["options_zh", "options", "options_text"]
    else:
        q_col_candidates = ["question_en", "question", "question_text"]
        o_col_candidates = ["options_en", "options", "options_text"]

    q_col = next((c for c in q_col_candidates if c in df.columns), None)
    o_col = next((c for c in o_col_candidates if c in df.columns), None)
    if not q_col:
        raise ValueError(f"Could not find a question column in {input_file}. Tried {q_col_candidates}")

    questions: List[str] = []
    if o_col:
        for _, row in df.iterrows():
            qtxt = str(row[q_col]) if pd.notna(row[q_col]) else ""
            otxt = str(row[o_col]) if pd.notna(row[o_col]) else ""
            if qtxt:
                questions.append(_build_question_with_options(qtxt, otxt))
    else:
        # assume the question column already embeds an Options: block
        for _, row in df.iterrows():
            qtxt = str(row[q_col]) if pd.notna(row[q_col]) else ""
            if qtxt:
                questions.append(qtxt)

    return questions


# ---------- CLI command ----------

@app.command("wvs")
def run_wvs(
    model_path: Path = typer.Option(..., help="Path to a local vLLM-compatible model directory"),
    input_file: Path = typer.Option(..., help="Questions file (.json/.jsonl/.xlsx/.xls/.csv)"),
    country_name: str = typer.Option(
        "United States",
        help="Country context (United States or China)",
        case_sensitive=False,
    ),
    output_file: Path = typer.Option(
        Path("outputs/WV_answers.json"),
        help="Where to save the model's answers (JSON list)",
    ),
    cache_dir: Optional[Path] = typer.Option(
        None, help="Hugging Face cache directory (sets HF_HOME/TRANSFORMERS_CACHE)"
    ),
    use_swow: bool = typer.Option(False, help="Augment prompts with SWOW-based context"),
    dtype: str = typer.Option(
        "bfloat16", help='Model dtype for vLLM weights: "float16", "bfloat16", or "float32"'
    ),
):
    """
    Run WVS multiple-choice evaluation with constrained decoding via vLLM.
    """
    # HF auth + caches
    _ensure_hf_env(cache_dir)

    # load questions
    questions = _load_questions_auto(input_file, country_name=country_name)

    # load model
    clear_gpu_memory()
    llm = load_llm(str(model_path), dtype=dtype)

    results = []
    for q in questions:
        try:
            res = answer_question(q, llm, country_name=country_name, use_swow=use_swow)
        except Exception as e:
            res = {"question": q, "error": f"{type(e).__name__}: {e}"}
        results.append(res)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(results, indent=4, ensure_ascii=False), encoding="utf-8")

    # cleanup
    del llm
    clear_gpu_memory()

    typer.echo(f"✅ Saved {len(results)} results to {output_file}")


if __name__ == "__main__":
    app()
