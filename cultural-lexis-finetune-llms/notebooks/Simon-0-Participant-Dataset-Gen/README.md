# Simon-0 Participant Dataset Generation

This folder contains notebooks used to build participant-style SWOW-US datasets and convert them into Alpaca-style instruction tuning files.

## Files in this folder

### read_swow_us.ipynb
Purpose: prepare SWOW-US data from raw SWOW CSV and export both analysis-friendly and training-ready artifacts.

What it does:
- Loads SWOW data from `./SWOW-EN.R100.20180827.csv`.
- Filters rows to participants with:
  - `country == United States`
  - `nativeLanguage == United States`
- Aggregates cue-response columns (`R1`, `R2`, `R3`, etc.) into cue-association frequency tables.
- Cleans associations by removing:
  - missing markers (`#Missing`, `#Unknown`)
  - cue=association duplicates
  - hash-like hex noise strings
- Computes basic corpus statistics (number of cues, average associations per cue, weighted totals).
- Exports processed cue-association data to:
  - `./SWOW-US.R100.20180827.processed.xlsx`
- Builds Alpaca-format instruction data from cue and responses, including `NO MORE RESPONSE` padding when `R2` or `R3` is missing.
- Writes Alpaca JSONL output:
  - `alpaca_data_swow_us.jsonl`

Notes:
- The notebook also includes exploratory quality checks (NA pattern checks, row inspection, value counts).
- Some cells are exploratory or legacy comparisons with other files/paths.

### alpaca_format.ipynb
Purpose: split the Alpaca-formatted SWOW-US dataset into reproducible participant collections for train/test experiments.

What it does:
- Reads `alpaca_data_swow_us.jsonl`.
- Removes records whose `output` starts with `nan,`.
- Groups samples by cue (`input`) to avoid splitting the same cue across train/test.
- Uses seeded splitting:
  - Train/test split on cue keys: 80/20 (seed `1234`)
  - Five shuffled train variants using seeds: `42, 43, 44, 45, 46`
- For each seed run, creates one folder:
  - `participant_swow_collection_us_seed_<seed_plus_1>`
- In each folder it writes:
  - `train/chunk_25_percent_train.jsonl`
  - `train/chunk_50_percent_train.jsonl`
  - `train/chunk_75_percent_train.jsonl`
  - `train/chunk_100_percent_train.jsonl`
  - `test/test.jsonl`
  - `cue_meta_info.jsonl` (stores seed, cue partitions, and test cues)

Notes:
- Folder names use `seed + 1` while `cue_meta_info.jsonl` stores the original seed value.
- The notebook also has an older simple chunking section (`swow_us/chunk_*.jsonl`) that appears to be legacy.

## Typical execution order
1. Run `read_swow_us.ipynb` to produce `alpaca_data_swow_us.jsonl`.
2. Run `alpaca_format.ipynb` to generate train/test participant collections.

## Expected main outputs
- `SWOW-US.R100.20180827.processed.xlsx`
- `alpaca_data_swow_us.jsonl`
- `participant_swow_collection_us_seed_*/` directories with train/test JSONL files and cue metadata
