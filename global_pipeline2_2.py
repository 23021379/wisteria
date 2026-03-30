# SCRIPT 2: 2_stratification_and_finalizing.py (Overhauled)
"""
Overview:
- Loads the fully assembled data from Script 1.
- FIX: Creates the dense `postcode_property_analysis` subset by pivoting the data,
  ensuring all property types (D, S, T, F) are present for every postcode.
- FIX: Correctly categorizes and saves the remaining subsets (ONS, Environment, LSOA Time-Series).
"""
import pandas as pd
import os
import re

# --- Configuration ---
BASE_DIR = r"[REDACTED_BY_SCRIPT]"
INPUT_FILE = os.path.join(BASE_DIR, "[REDACTED_BY_SCRIPT]")
OUTPUT_DIR = BASE_DIR

# --- Step 1: Create the Postcode Property Analysis Subset ---
print("[REDACTED_BY_SCRIPT]")
try:
    # Define all columns related to the postcode analysis file
    postcode_analysis_cols = [
        'postcode', 'propertytype', 'sale_count', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]'
    ] + [f'{year}_sale_count' for year in range(1995, 2024)]

    df_full = pd.read_parquet(INPUT_FILE)
    df_ppa = df_full[postcode_analysis_cols].drop_duplicates()

    # Pivot the data to create the dense structure
    df_pivot = df_ppa.pivot_table(
        index='postcode',
        columns='propertytype',
        values=[c for c in postcode_analysis_cols if c not in ['postcode', 'propertytype']],
        aggfunc='first' # Use first since data is already unique
    )

    # Flatten the multi-level column index
    df_pivot.columns = ['_'.join(col).strip() for col in df_pivot.columns.values]
    df_pivot.reset_index(inplace=True)

    # Ensure all property types are represented, filling missing with 0
    all_property_types = ['D', 'S', 'T', 'F']
    for col_group in ['sale_count', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]'] + [f'{y}_sale_count' for y in range(1995, 2024)]:
        for p_type in all_property_types:
            col_name = f'[REDACTED_BY_SCRIPT]'
            if col_name not in df_pivot.columns:
                df_pivot[col_name] = 0

    # Reorder columns for readability
    sorted_cols = ['postcode']
    for p_type in all_property_types:
        sorted_cols.extend([f'sale_count_{p_type}', f'[REDACTED_BY_SCRIPT]', f'[REDACTED_BY_SCRIPT]'])
    for year in range(1995, 2024):
         for p_type in all_property_types:
            sorted_cols.append(f'[REDACTED_BY_SCRIPT]')

    df_pivot = df_pivot[sorted_cols]

    output_path = os.path.join(OUTPUT_DIR, "[REDACTED_BY_SCRIPT]")
    df_pivot.to_parquet(output_path, index=False)
    print(f"[REDACTED_BY_SCRIPT]")
    print(f"[REDACTED_BY_SCRIPT]")

except Exception as e:
    print(f"[REDACTED_BY_SCRIPT]")

# --- Step 2: Create Remaining Subsets ---
print("[REDACTED_BY_SCRIPT]")

# We already have df_full loaded
all_columns = set(df_full.columns)

# Define feature groups based on prefixes from Script 1
identifier_cols = {c for c in all_columns if c in ['postcode', 'propertytype', 'output_area_2011', 'output_area_2021', 'lsoa_2011', 'lsoa_2021', 'msoa_2021', 'lad_code', 'wz_code']}
ons_oac_cols = {c for c in all_columns if c.startswith(('ons_', 'oac_', 'pcc_'))}
environment_cols = {c for c in all_columns if c.startswith(('ahah_', 'ages_', 'veg_', 'bb_', 'feat_', 'oa_', 'lsoa_', 'msoa_'))}
lsoa_timeseries_cols = {c for c in all_columns if c.startswith(('q_price_', 'q_trans_'))}

# Create a dictionary for subset creation loop
subsets_to_create = {
    "subset_ons_oac": identifier_cols.union(ons_oac_cols),
    "[REDACTED_BY_SCRIPT]": identifier_cols.union(environment_cols),
    "[REDACTED_BY_SCRIPT]": identifier_cols.union(lsoa_timeseries_cols)
}

for name, cols in subsets_to_create.items():
    try:
        # Select only existing columns and drop duplicates based on main identifiers
        subset_df = df_full[list(cols)].drop_duplicates(subset=['postcode', 'propertytype'])
        output_path = os.path.join(OUTPUT_DIR, f"{name}.parquet")
        subset_df.to_parquet(output_path, index=False)
        print(f"[REDACTED_BY_SCRIPT]")
        print(f"[REDACTED_BY_SCRIPT]")
    except Exception as e:
        print(f"[REDACTED_BY_SCRIPT]")