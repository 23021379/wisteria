"""
This code is a data stratification script that takes a large, cleaned housing dataset and splits it into smaller, more manageable subsets based on feature categories. The purpose is to break down an enormous dataset with hundreds of columns into focused, domain-specific files that are easier to work with for analysis and modeling.

The script begins by configuring file paths, pointing to the cleaned dataset that was created by the previous cleanup script. It reads only the schema of the Parquet file first, which is memory-efficient because it gets the column names without loading the actual data.

The core functionality revolves around defining five distinct feature groups. First, it identifies identifier columns like postcode, output areas, and property types that serve as keys for joining data. Second, it captures postcode-level features including house price index data, coordinates, sale counts, and price ranges. Third, it groups census and classification features that start with prefixes like "ons_", "oac_", and "classif_". Fourth, it collects environmental and accessibility features including health access data, property ages, vegetation indices, and broadband information. Finally, it identifies time-series features for historical market dynamics and interaction features that were engineered in previous scripts.

The script includes a verification step to ensure all columns from the original dataset are properly assigned to one of these categories. If any columns remain unassigned, it prints warnings so the user can review and potentially adjust the categorization logic.

The actual subset creation process is designed for memory efficiency. Rather than loading the entire massive dataset into memory at once, it reads only the specific columns needed for each subset. For each of the five subsets, it combines the identifier columns with the relevant feature group, reads just those columns from the source file, saves the subset as a new Parquet file, and immediately deletes the DataFrame from memory before moving to the next subset.

The five output files created are subset_1_postcode.parquet containing postcode-level market data, subset_2_ons_oac.parquet with census and area classification information, subset_3_environment.parquet holding environmental and accessibility metrics, subset_4_timeseries.parquet with historical price and transaction data, and subset_5_interactions.parquet containing all the engineered interaction features.

This stratification approach provides several benefits. It makes the data more manageable by breaking it into logical, domain-specific chunks. It improves analysis workflows because researchers can focus on specific aspects without loading unnecessary data. It enables more efficient modeling by allowing data scientists to experiment with different feature combinations. It also facilitates sharing and collaboration since different teams might only need specific subsets.

The memory management is particularly important here because the original dataset likely contains millions of rows with hundreds of columns, which could easily exceed available RAM if loaded entirely. By processing one subset at a time and immediately cleaning up memory, the script can handle very large datasets on standard hardware.
"""


import pandas as pd
import pyarrow.parquet as pq
import os
import sys
import traceback

# --- Configuration ---
# CORRECTED: The path now points to the CLEAN dataset created by Script 1
CLEAN_INPUT_FILE = r"[REDACTED_BY_SCRIPT]"
output_directory = os.path.dirname(CLEAN_INPUT_FILE)

# --- Step 1: Load Column Schema (Memory-Efficient) ---
print(f"[REDACTED_BY_SCRIPT]")
try:
    all_columns = set(pq.read_schema(CLEAN_INPUT_FILE).names)
    print(f"[REDACTED_BY_SCRIPT]")
except Exception as e:
    print(f"[REDACTED_BY_SCRIPT]")
    print(f"[REDACTED_BY_SCRIPT]")
    sys.exit(1)

# --- Step 2: Define CORRECTED Feature Groups ---
print("[REDACTED_BY_SCRIPT]")

# 1. Identifier columns
identifier_cols = {col for col in ['postcode', 'output_area_2011', 'output_area_2021', 'lsoa_2011', 'lsoa_2021', 'msoa_2011', 'msoa_2021', 'lad_code', 'wz_code', 'propertytype'] if col in all_columns}

# 2. Postcode-level features
postcode_cols = {col for col in all_columns if col.startswith(('hpi_adjusted_', 'weighted_')) or col in ['latitude', 'longitude', 'sale_count', 'min_price', 'max_price', 'price_range', 'oldest_sale_year', 'newest_sale_year']}

# 3. Census / OAC / Classification features
ons_oac_cols = {col for col in all_columns if col.startswith(('ons_', 'oac_', 'classif_'))}

# 4. Environmental, Accessibility & Property Age features
environment_cols = {col for col in all_columns if col.startswith(('ahah_', 'ages_', 'veg_', 'bb_'))}

# 5. Time-series features (Historical Market Dynamics)
# --- THIS IS THE CORRECTED LOGIC ---
timeseries_cols = {col for col in all_columns if col.startswith(('price_median_', 'trans_count_')) or (col.endswith('_sale_count') and col != 'sale_count')}

# 6. Interaction features (Engineered Relationships)
interactions_cols = {col for col in all_columns if col.startswith('int_')}

# --- Step 3: Verify All Columns are Assigned ---
all_assigned_columns = set.union(identifier_cols, postcode_cols, ons_oac_cols, environment_cols, timeseries_cols, interactions_cols)
unassigned_cols = all_columns - all_assigned_columns

if unassigned_cols:
    print("[REDACTED_BY_SCRIPT]")
    for col in sorted(list(unassigned_cols)):
        print(f"  - {col}")
else:
    print("[REDACTED_BY_SCRIPT]")


# --- Step 4: Create and Save Subsets by Reading Only Necessary Columns ---
print("[REDACTED_BY_SCRIPT]")

try:
    # --- Subset 1: Postcode Data ---
    subset_1_cols = sorted(list(identifier_cols.union(postcode_cols)))
    df_postcode = pd.read_parquet(CLEAN_INPUT_FILE, columns=subset_1_cols)
    output_path = os.path.join(output_directory, "[REDACTED_BY_SCRIPT]")
    df_postcode.to_parquet(output_path, index=False)
    print(f"[REDACTED_BY_SCRIPT]")
    del df_postcode

    # --- Subset 2: ONS / OAC Data ---
    subset_2_cols = sorted(list(identifier_cols.union(ons_oac_cols)))
    df_ons_oac = pd.read_parquet(CLEAN_INPUT_FILE, columns=subset_2_cols)
    output_path = os.path.join(output_directory, "[REDACTED_BY_SCRIPT]")
    df_ons_oac.to_parquet(output_path, index=False)
    print(f"[REDACTED_BY_SCRIPT]")
    del df_ons_oac

    # --- Subset 3: Environmental Data ---
    subset_3_cols = sorted(list(identifier_cols.union(environment_cols)))
    df_environment = pd.read_parquet(CLEAN_INPUT_FILE, columns=subset_3_cols)
    output_path = os.path.join(output_directory, "[REDACTED_BY_SCRIPT]")
    df_environment.to_parquet(output_path, index=False)
    print(f"[REDACTED_BY_SCRIPT]")
    del df_environment

    # --- Subset 4: Time-Series Data ---
    subset_4_cols = sorted(list(identifier_cols.union(timeseries_cols)))
    df_timeseries = pd.read_parquet(CLEAN_INPUT_FILE, columns=subset_4_cols)
    output_path = os.path.join(output_directory, "[REDACTED_BY_SCRIPT]")
    df_timeseries.to_parquet(output_path, index=False)
    print(f"[REDACTED_BY_SCRIPT]")
    del df_timeseries

    # --- Subset 5: Interaction Data ---
    subset_5_cols = sorted(list(identifier_cols.union(interactions_cols)))
    df_interactions = pd.read_parquet(CLEAN_INPUT_FILE, columns=subset_5_cols)
    output_path = os.path.join(output_directory, "[REDACTED_BY_SCRIPT]")
    df_interactions.to_parquet(output_path, index=False)
    print(f"[REDACTED_BY_SCRIPT]")
    del df_interactions

except Exception as e:
    print(f"[REDACTED_BY_SCRIPT]")
    traceback.print_exc()
    sys.exit(1)

print("[REDACTED_BY_SCRIPT]")