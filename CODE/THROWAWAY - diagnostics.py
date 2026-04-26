import pandas as pd
import pyarrow.parquet as pq
import os
import sys

# --- Configuration ---
# Path to your large, feature-rich dataset
input_path = r"[REDACTED_BY_SCRIPT]"

# --- Step 1: Load Column Schema ---
print(f"[REDACTED_BY_SCRIPT]")
try:
    all_columns = pq.read_schema(input_path).names
    print(f"[REDACTED_BY_SCRIPT]")
except Exception as e:
    print(f"[REDACTED_BY_SCRIPT]")
    sys.exit(1)

# --- Step 2: Run the Same Flawed Categorization Logic ---
print("[REDACTED_BY_SCRIPT]")

identifier_cols = [col for col in ['postcode', 'output_area_2011', 'output_area_2021', 'lsoa_2011', 'lsoa_2021', 'msoa_2011', 'msoa_2021', 'lad_code', 'wz_code', 'propertytype'] if col in all_columns]
postcode_cols = [col for col in all_columns if col.startswith('hpi_adjusted_') or col == 'sale_count' or (col.endswith('_sale_count') and len(col) == 16)]
ons_oac_cols = [col for col in all_columns if col.startswith(('ons_', 'oac_'))]
environment_cols = [col for col in all_columns if col.startswith(('ahah_', 'ages_', 'veg_', 'bb_'))]
timeseries_cols = [col for col in all_columns if col.startswith(('price_median_', 'trans_count_'))]
interactions_cols = [col for col in all_columns if col.startswith('int_')]

# --- Step 3: Find the Difference ---
# Combine all categorized columns into a single set to get unique names
all_assigned_columns = set(
    identifier_cols + 
    postcode_cols + 
    ons_oac_cols + 
    environment_cols + 
    timeseries_cols + 
    interactions_cols
)

# Convert the master list of all columns to a set
all_columns_set = set(all_columns)

# Find the columns that are in the full list but not in the assigned list
orphan_columns = sorted(list(all_columns_set - all_assigned_columns))

# --- Step 4: Report the Findings ---
print("[REDACTED_BY_SCRIPT]")
print(f"[REDACTED_BY_SCRIPT]")
print(f"[REDACTED_BY_SCRIPT]")
print(f"Number of 'Orphan'[REDACTED_BY_SCRIPT]")
print("[REDACTED_BY_SCRIPT]")
for col in orphan_columns:
    print(f"  - {col}")