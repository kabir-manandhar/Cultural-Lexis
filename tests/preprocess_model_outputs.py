import csv
import json
from collections import defaultdict
from typing import Dict, List

def parse_associations_csv(
    csv_path: str,
    cue_col: str = "Ground Truth Associated Words",
    gt_col: str = "Ground Truth Associated Words",
    prompt_col: str = "Prompt Type",
    model_col: str = "Model Type",
    gen_col: str = "Generated Associated Words",
) -> Dict[str, Dict]:
    """
    Reads a CSV of word‐association outputs and returns a dict of the form:
      {
        cue: {
          "ground_truth": [...],
          "models": {
            model_name: {
              "Complex": [...],
              "Simple": [...]
            },
            ...
          }
        },
        ...
      }
    """
    data: Dict[str, Dict] = {}
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cue = row[cue_col].strip()
            # split the ground-truth list
            gt_list = [w.strip() for w in row[gt_col].split(',') if w.strip()]
            prompt_type = row[prompt_col].strip()       # e.g. "Complex" or "Simple"
            model_name  = row[model_col].strip()        # e.g. "sukai/qwen_swow_en"
            gen_list    = [w.strip() for w in row[gen_col].split(',') if w.strip()]

            # initialize cue entry
            if cue not in data:
                data[cue] = {
                    "ground_truth": gt_list,
                    "models": defaultdict(lambda: {"Complex": [], "Simple": []})
                }

            # you may want to overwrite or extend; here we extend
            data[cue]["models"][model_name][prompt_type].extend(gen_list)

    # convert inner defaultdicts back to normal dicts
    for cue, block in data.items():
        block["models"] = dict(block["models"])

    return data


def save_associations_json(data: Dict, json_path: str):
    """
    Dumps the nested association dict to json_path with pretty formatting.
    """
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# Example usage:
if __name__ == "__main__":
    csv_file  = "/data/gpfs/projects/punim2219/LM_with_SWOW/sukaih/cultural-lexis-finetune-llms/notebooks/HLCP-A-eval-12-models/swow_en_results.csv"
    out_json  = "/data/gpfs/projects/punim2219/LM_with_SWOW/kabir/Data/associations_processed.json"
    nested = parse_associations_csv(csv_file,
                                   cue_col="Cue Word",           # or whatever your column is named
                                   gt_col="Ground Truth Associated Words",
                                   prompt_col="Prompt Type",
                                   model_col="Model Type",
                                   gen_col="Generated Associated Words")
    save_associations_json(nested, out_json)
    print(f"Processed {len(nested)} cues → saved to {out_json}")
