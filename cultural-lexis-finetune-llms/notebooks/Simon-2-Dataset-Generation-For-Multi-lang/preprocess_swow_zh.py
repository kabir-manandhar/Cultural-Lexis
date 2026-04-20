import pandas as pd
import re

HEX_PATTERN = re.compile(r'^[a-f0-9]{32}$', re.IGNORECASE) ## This regex matches a 32-character lowercase hexadecimal string for political sensitive words

ASSOCIATION_COLS = lambda df: [col for col in df.columns if re.match(r'^R\d+$', col)]

def mask_politically_sensitive(df, n_samples=10):
    """Replace hex-encoded (politically sensitive) responses with #Missing in place.
    Reports the proportion of affected response cells."""
    assoc_cols = ASSOCIATION_COLS(df)
    total_cells = df[assoc_cols].notna().sum().sum()

    mask = df[assoc_cols].apply(
        lambda col: col.map(lambda v: isinstance(v, str) and bool(HEX_PATTERN.match(v.strip())))
    )
    n_sensitive = mask.sum().sum()

    # Sample rows that contain at least one sensitive cell — print BEFORE replacing
    sensitive_rows = mask.any(axis=1)
    sample = df[sensitive_rows].head(n_samples)
    print(f"[Politically sensitive] Sample rows (n={len(sample)}):")
    print(sample[['cue'] + assoc_cols].to_string(index=True))
    print()

    df[assoc_cols] = df[assoc_cols].where(~mask, other='#Missing')

    print(f"[Politically sensitive] {n_sensitive:,} / {total_cells:,} response cells "
          f"({100 * n_sensitive / total_cells:.2f}%) replaced with #Missing.")
    return df


def drop_all_missing_rows(df, n_samples=10):
    """Remove rows where every association response is #Missing or NaN.
    Reports the proportion of affected rows."""
    assoc_cols = ASSOCIATION_COLS(df)
    is_missing = df[assoc_cols].apply(
        lambda col: col.map(lambda v: pd.isna(v) or v == '#Missing')
    )
    all_missing_mask = is_missing.all(axis=1)
    n_dropped = all_missing_mask.sum()

    # Sample rows that will be dropped — print BEFORE dropping
    sample = df[all_missing_mask].head(n_samples)
    print(f"[All-missing rows] Sample rows to be dropped (n={len(sample)}):")
    print(sample[['cue'] + assoc_cols].to_string(index=True))
    print()

    print(f"[All-missing rows] {n_dropped:,} / {len(df):,} rows "
          f"({100 * n_dropped / len(df):.2f}%) dropped (all responses #Missing or NaN).")

    return df[~all_missing_mask].reset_index(drop=True)


def calculate_statistics(df):
    """Report descriptive statistics on the preprocessed response DataFrame."""
    assoc_cols = ASSOCIATION_COLS(df)

    valid = df[assoc_cols].apply(
        lambda col: col.map(lambda v: pd.notna(v) and v not in ('#Missing', '#Unknown'))
    )

    responses_per_row = valid.sum(axis=1)
    all_valid_values  = df[assoc_cols].where(valid).stack()

    print(f"\n[Statistics]")
    print(f"  Unique cues:                      {df['cue'].nunique():,}")
    print(f"  Total rows:                       {len(df):,}")
    print(f"  Avg valid responses per row:      {responses_per_row.mean():.2f}")
    print(f"  Total responses (valid):          {valid.sum().sum():,}")
    print(f"  Unique association types:         {all_valid_values.nunique():,}")



# --- Pipeline ---
path        = "../data/swow/SWOWZH.R55.20230424.csv"
output_path = "../data/swow/SWOWZH.R55.20230424.processed.csv"

df = pd.read_csv(path)
print(f"Loaded {len(df):,} rows.\n")
print("Statistics before the preprocesing")
calculate_statistics(df)
df = mask_politically_sensitive(df, n_samples=50)
df = drop_all_missing_rows(df)
df.to_csv(output_path, encoding='utf-8')

print("Statistics after the preprocesing")
calculate_statistics(df)
df.head()