"""
Main Components
Configuration & Setup (Lines 1-20)
Sets up file paths and chunk size (500,000 rows per chunk)
Input: ml_ready_combined_housing_data.parquet (from previous script)
Additional data: Price analysis and coordinates lookup tables
Output: final_enriched_ml_dataset_with_interactions.parquet
Core Function: create_feature_interactions() (Lines 30-140)
This is the heart of the script, creating sophisticated interaction features:

1. Market Structure & Economic Profile
2. Relative Value Analysis (Lines 85-112)
3. Temporal Price Dynamics
4. Socio-Economic Interactions
Chunk Processing Architecture (Lines 140-220)
Step 1: Load Lookup Tables
Loads price analysis and coordinate data
Combines them into a single lookup table
Converts to numeric format
Step 2: Process in Chunks
The script uses PyArrow for efficient large dataset processing:

Step 3: Final Verification
Verifies the output file integrity
Reports final dataset dimensions
Key Technical Features
Memory Efficiency
Processes 500k rows at a time to avoid memory issues
Uses PyArrow for efficient Parquet I/O
Monitors memory usage per chunk
Robust Error Handling
Checks column existence before creating interactions
Gracefully handles missing data scenarios
Uses epsilon (1e-6) to prevent division by zero
"""



import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import os
import re
import sys
import traceback

# --- Configuration ---
BASE_DIR = r"[REDACTED_BY_SCRIPT]"
CHUNK_SIZE = 500_000  # Rows to process at a time. Adjust based on available RAM.

# INPUT FILE: The main, large dataset
ML_READY_INPUT_FILE = os.path.join(BASE_DIR, "[REDACTED_BY_SCRIPT]")

# NEW DATA FILES (Lookup Tables)
PRICE_ANALYSIS_FILE = os.path.join(BASE_DIR, "[REDACTED_BY_SCRIPT]")
COORDINATES_FILE = os.path.join(BASE_DIR, "[REDACTED_BY_SCRIPT]")

# FINAL OUTPUT FILE
FINAL_ENRICHED_FILE = os.path.join(BASE_DIR, "[REDACTED_BY_SCRIPT]")

# --- Helper Function ---
def clean_col_names(df):
    """[REDACTED_BY_SCRIPT]"""
    new_cols = []
    for col in df.columns:
        new_col = re.sub(r'[^0-9a-zA-Z_]+', '_', str(col))
        new_col = new_col.strip('_').lower()
        new_cols.append(new_col)
    df.columns = new_cols
    return df

# --- NEW: Feature Interaction Creation Function ---
def create_feature_interactions(df):
    """
    Creates a suite of new feature interactions from the existing data.
    The function is designed to be robust to missing columns.
    """
    print("[REDACTED_BY_SCRIPT]")
    # Epsilon to prevent division by zero errors
    epsilon = 1e-6

    # Helper function to check if all required columns exist for an interaction
    def check_cols(required_cols):
        return all(col in df.columns for col in required_cols)

    # --- Market Structure & Economic Profile ---
    try:
        # 1. Price Skewness Proxy
        req = ['[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]']
        if check_cols(req):
            df['int_price_skewness'] = df['[REDACTED_BY_SCRIPT]'] - df['[REDACTED_BY_SCRIPT]']

        # 2. Market Depth (Total Value)
        req = ['[REDACTED_BY_SCRIPT]', 'sale_count']
        if check_cols(req):
            df['int_market_depth'] = df['[REDACTED_BY_SCRIPT]'] * df['sale_count']

        # 3. Household Leverage Ratio
        req = ['[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]']
        if check_cols(req):
            df['[REDACTED_BY_SCRIPT]'] = df[req[0]] / (df[req[1]] + epsilon)

        # 4. Investor Saturation Index
        req = ['[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]']
        if check_cols(req):
            df['int_investor_saturation'] = df[req[0]] / (df[req[1]] + epsilon)

    except Exception as e:
        print(f"[REDACTED_BY_SCRIPT]")


    # --- Relative Value & Internal Market Structure (More Complex) ---
    try:
        # 5. Detached House Premium & other relative metrics
        # This requires grouping within the dataframe, which is more advanced.
        price_cols = ['postcode', 'propertytype', '[REDACTED_BY_SCRIPT]']
        if check_cols(price_cols):
            # Pivot to get prices for each property type in columns
            postcode_prices = df.groupby(['postcode', 'propertytype'])['[REDACTED_BY_SCRIPT]'].mean().unstack()
            
            # Ensure the required property type columns exist after pivot
            prop_type_cols = {'D', 'S', 'T', 'F'}
            if prop_type_cols.issubset(postcode_prices.columns):
                # Calculate the premium/discounts
                postcode_prices['[REDACTED_BY_SCRIPT]'] = postcode_prices['D'] / (postcode_prices[['S', 'T', 'F']].mean(axis=1) + epsilon)
                postcode_prices['int_flat_discount'] = postcode_prices['F'] / (postcode_prices[['D', 'S', 'T']].mean(axis=1) + epsilon)
                postcode_prices['int_step_up_cost_semi'] = postcode_prices['S'] / (postcode_prices['T'] + epsilon)

                # Merge the new interaction features back into the main dataframe
                df = df.merge(postcode_prices[['[REDACTED_BY_SCRIPT]', 'int_flat_discount', 'int_step_up_cost_semi']], on='postcode', how='left')

    except Exception as e:
        print(f"[REDACTED_BY_SCRIPT]")


    # --- Temporal Price & Transaction Dynamics ---
    try:
        # 6. Normalized Price Volatility
        price_median_cols = [c for c in df.columns if 'price_median_' in c and '_q' in c]
        if len(price_median_cols) > 1:
            df['[REDACTED_BY_SCRIPT]'] = df[price_median_cols].std(axis=1) / (df[price_median_cols].mean(axis=1) + epsilon)
        
        # 7. Price Momentum (Last year vs. year before)
        req = ['price_median_18q4', 'price_median_17q4']
        if check_cols(req):
            df['[REDACTED_BY_SCRIPT]'] = df[req[0]] / (df[req[1]] + epsilon)

        # 8. Post-2008 Growth Index
        q_09_cols = [c for c in df.columns if 'price_median_09' in c]
        q_18_cols = [c for c in df.columns if 'price_median_18' in c]
        if q_09_cols and q_18_cols:
            avg_09 = df[q_09_cols].mean(axis=1)
            avg_18 = df[q_18_cols].mean(axis=1)
            df['[REDACTED_BY_SCRIPT]'] = avg_18 / (avg_09 + epsilon)
            
    except Exception as e:
        print(f"[REDACTED_BY_SCRIPT]")

    # --- Socio-Economic & Market Interactions ---
    try:
        # 9. Affluent Illiquidity
        req = ['deprivation_score', 'sale_count']
        if check_cols(req): # Assuming deprivation_score is created/present
             df['int_affluent_illiquidity'] = (1 / (df['deprivation_score'] + epsilon)) * (1 / (df['sale_count'] + epsilon))

        # 10. Deprivation vs. Market Momentum
        req = ['deprivation_score', '[REDACTED_BY_SCRIPT]']
        if check_cols(req):
            df['[REDACTED_BY_SCRIPT]'] = df['deprivation_score'] * df['[REDACTED_BY_SCRIPT]']

        # 11. Single Parent vs. Housing Cost Burden
        req = ['[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]']
        if check_cols(req):
            df['[REDACTED_BY_SCRIPT]'] = df[req[0]] * df[req[1]]

    except Exception as e:
        print(f"[REDACTED_BY_SCRIPT]")

    return df


# --- Main Processing Function ---
def process_data_in_chunks():
    """
    Loads the main dataset in chunks, merges it with lookup tables,
    and saves the result incrementally to avoid memory errors.
    """
    print("[REDACTED_BY_SCRIPT]")

    # --- Step 1: Load and Prepare Lookup Tables ---
    print("[REDACTED_BY_SCRIPT]")
    try:
        df_prices = pd.read_csv(PRICE_ANALYSIS_FILE, dtype=str)
        df_prices = clean_col_names(df_prices)
        print(f"[REDACTED_BY_SCRIPT]")

        df_coords = pd.read_csv(COORDINATES_FILE, dtype=str)
        df_coords = clean_col_names(df_coords)
        if 'pcds' in df_coords.columns:
            df_coords.rename(columns={'pcds': 'postcode'}, inplace=True)
        print(f"[REDACTED_BY_SCRIPT]")

        print("[REDACTED_BY_SCRIPT]")
        df_lookup = pd.merge(df_prices, df_coords, on='postcode', how='outer')
        # Convert all lookup columns (except postcode) to numeric, coercing errors
        lookup_numeric_cols = [col for col in df_lookup.columns if col != 'postcode']
        for col in lookup_numeric_cols:
            df_lookup[col] = pd.to_numeric(df_lookup[col], errors='coerce')
        print(f"[REDACTED_BY_SCRIPT]")

    except Exception as e:
        print(f"[REDACTED_BY_SCRIPT]")
        sys.exit(1)

    # --- Step 2: Process Main Dataset in Chunks ---
    print(f"[REDACTED_BY_SCRIPT]")
    try:
        if not os.path.exists(ML_READY_INPUT_FILE):
            print(f"[REDACTED_BY_SCRIPT]'{ML_READY_INPUT_FILE}'")
            sys.exit(1)

        parquet_file = pq.ParquetFile(ML_READY_INPUT_FILE)
        total_rows = parquet_file.metadata.num_rows
        batch_iterator = parquet_file.iter_batches(batch_size=CHUNK_SIZE)

        if os.path.exists(FINAL_ENRICHED_FILE):
            os.remove(FINAL_ENRICHED_FILE)
            print(f"[REDACTED_BY_SCRIPT]'{os.path.basename(FINAL_ENRICHED_FILE)}'")

        print(f"[REDACTED_BY_SCRIPT]")
        writer = None

        for i, batch in enumerate(batch_iterator):
            print(f"[REDACTED_BY_SCRIPT]")
            df_main_chunk = batch.to_pandas()
            
            # MERGE STEP
            df_merged_chunk = pd.merge(df_main_chunk, df_lookup, on='postcode', how='left')
            df_merged_chunk = df_merged_chunk.loc[:,~df_merged_chunk.columns.duplicated()]

            # --- NEW: FEATURE INTERACTION STEP ---
            df_processed_chunk = create_feature_interactions(df_merged_chunk)

            # Convert to PyArrow Table for writing
            table = pa.Table.from_pandas(df_processed_chunk)

            # On the first chunk, create the writer with the final schema
            if writer is None:
                print(f"[REDACTED_BY_SCRIPT]")
                writer = pq.ParquetWriter(FINAL_ENRICHED_FILE, table.schema)

            writer.write_table(table)
            print(f"[REDACTED_BY_SCRIPT]")

        if writer:
            writer.close()

    except Exception as e:
        print(f"[REDACTED_BY_SCRIPT]")
        traceback.print_exc()
        sys.exit(1)

    # --- Step 3: Final Verification ---
    print("[REDACTED_BY_SCRIPT]")
    try:
        final_metadata = pq.read_metadata(FINAL_ENRICHED_FILE)
        print(f"[REDACTED_BY_SCRIPT]")
        print(f"[REDACTED_BY_SCRIPT]")
    except Exception as e:
        print(f"[REDACTED_BY_SCRIPT]")

if __name__ == "__main__":
    process_data_in_chunks()