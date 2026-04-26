"""
The main purpose is to take an existing machine learning dataset and dramatically expand its feature set by creating sophisticated interaction variables that capture complex relationships in housing market data.

The script is organized into several key components:

Configuration and Setup: The script defines paths for input and output files, with the main input being a previously processed ML-ready dataset. It also sets a chunk size of 500,000 rows to handle memory constraints when processing large datasets.

Helper Functions: There's a column name cleaning function that standardizes naming conventions, and the main feature creation function that generates hundreds of new interaction features.

The core feature creation function is extremely comprehensive and creates several categories of interaction features:

Market Structure and Economic Profile features include price skewness calculations that measure market inequality, market depth metrics that combine price and transaction volume, household leverage ratios comparing mortgaged to owned-outright properties, and investor saturation indices measuring rental property concentration.

Relative Value and Internal Market Structure features calculate property type premiums and discounts. For example, it determines how much more expensive detached houses are compared to other property types, or how much cheaper flats are relative to houses.

Temporal Price and Transaction Dynamics features analyze price movements over time, including normalized price volatility measures, transaction pattern analysis, price momentum calculations comparing recent periods, shock resilience metrics measuring recovery from economic downturns like 2008, and price curve convexity that captures acceleration or deceleration in price changes.

Socio-Economic and Demographic Interactions create features like family size mismatch indicators for households with children in small properties, retiree amenity access scores combining elderly populations with pharmacy access, digital professional demand metrics linking employment with broadband quality, deprivation-related interactions that measure how economic disadvantage affects various outcomes, and ethnic diversity measures using inverse Herfindahl-Hirschman Index calculations.

Environmental, Geographic and Property Interactions include historic pollution exposure for old buildings, apartment blue space premiums for flats near water, health hazard hotspots combining pollution and hospital access, garden deficit value for apartments in areas with limited green space, property age diversity within postcodes, green versus grey space ratios, geographic gradients linking location with wealth and building age, remoteness indices based on distance from major urban centers, and urban core proxy measures.

The chunk processing architecture is designed for memory efficiency. It loads lookup tables containing price analysis and coordinate data, then processes the main dataset in manageable chunks. For each chunk, it merges with lookup data, applies the comprehensive feature engineering, and writes results incrementally to avoid memory overflow.

The script includes robust error handling throughout, checking for column existence before creating interactions, using epsilon values to prevent division by zero errors, and gracefully handling missing data scenarios.
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

# INPUT FILE: The main, large dataset that you want to add features to
ML_READY_INPUT_FILE = os.path.join(BASE_DIR, "[REDACTED_BY_SCRIPT]")

# LOOKUP FILES: These were used to create the input file, but we need them again
# to access some of the raw data for the new interaction features.
PRICE_ANALYSIS_FILE = os.path.join(BASE_DIR, "[REDACTED_BY_SCRIPT]")
COORDINATES_FILE = os.path.join(BASE_DIR, "[REDACTED_BY_SCRIPT]")

# FINAL OUTPUT FILE: The new, feature-rich dataset
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

# --- NEW: The Full Feature Interaction Creation Function ---
def create_feature_interactions(df):
    """
    Creates a comprehensive suite of new feature interactions from the existing data.
    The function is designed to be robust to missing columns.
    """
    print("[REDACTED_BY_SCRIPT]")
    epsilon = 1e-6  # Epsilon to prevent division by zero errors
    original_cols = set(df.columns)

    # Helper function to check if all required columns exist for an interaction
    def check_cols(required_cols):
        return all(col in df.columns for col in required_cols)

    # --- Pre-computation Step ---
    # Create lists of column groups for easier calculations
    age_cols = [c for c in df.columns if c.startswith('ages_bp_')]
    price_median_cols = [c for c in df.columns if 'price_median_' in c]
    trans_count_cols = [c for c in df.columns if 'trans_count_' in c]
    ethnic_cols = [c for c in df.columns if '[REDACTED_BY_SCRIPT]' in c]

    # Create a single deprivation score for easier use
    dep_cols = ['ons_household_is_deprived_in_one_dimension', '[REDACTED_BY_SCRIPT]',
                '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]']
    if check_cols(dep_cols):
        df['[REDACTED_BY_SCRIPT]'] = (df[dep_cols[0]] * 1 + df[dep_cols[1]] * 2 + df[dep_cols[2]] * 3 + df[dep_cols[3]] * 4)

    # --- Market Structure & Economic Profile ---
    try:
        if check_cols(['[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]']): df['int_price_skewness'] = df['[REDACTED_BY_SCRIPT]'] - df['[REDACTED_BY_SCRIPT]']
        if check_cols(['[REDACTED_BY_SCRIPT]', 'sale_count']): df['int_market_depth'] = df['[REDACTED_BY_SCRIPT]'] * df['sale_count']
        if check_cols(['[REDACTED_BY_SCRIPT]', 'sale_count']): df['int_price_to_transaction_ratio'] = df['[REDACTED_BY_SCRIPT]'] / (df['sale_count'] + epsilon)
        if check_cols(['[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]']): df['[REDACTED_BY_SCRIPT]'] = df['[REDACTED_BY_SCRIPT]'] / (df['[REDACTED_BY_SCRIPT]'] + epsilon)
        if check_cols(['[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]']): df['int_investor_saturation'] = df['[REDACTED_BY_SCRIPT]'] / (df['[REDACTED_BY_SCRIPT]'] + epsilon)
    except Exception as e: print(f"[REDACTED_BY_SCRIPT]")

    # --- Relative Value & Internal Market Structure ---
    try:
        price_cols = ['postcode', 'propertytype', '[REDACTED_BY_SCRIPT]']
        if check_cols(price_cols):
            pivot = df.pivot_table(index='postcode', columns='propertytype', values='[REDACTED_BY_SCRIPT]', aggfunc='mean')
            prop_types_present = [pt for pt in ['D', 'S', 'T', 'F'] if pt in pivot.columns]
            
            if 'D' in prop_types_present and len([pt for pt in ['S', 'T', 'F'] if pt in prop_types_present]) > 0:
                pivot['[REDACTED_BY_SCRIPT]'] = pivot['D'] / (pivot[[pt for pt in ['S', 'T', 'F'] if pt in prop_types_present]].mean(axis=1) + epsilon)
            if 'F' in prop_types_present and len([pt for pt in ['D', 'S', 'T'] if pt in prop_types_present]) > 0:
                pivot['int_flat_discount'] = pivot['F'] / (pivot[[pt for pt in ['D', 'S', 'T'] if pt in prop_types_present]].mean(axis=1) + epsilon)
            if 'S' in prop_types_present and 'T' in prop_types_present:
                pivot['int_step_up_cost_semi'] = pivot['S'] / (pivot['T'] + epsilon)
            
            # Select only new columns to merge back
            new_pivot_cols = [c for c in pivot.columns if c.startswith('int_')]
            if new_pivot_cols:
                df = df.merge(pivot[new_pivot_cols], on='postcode', how='left')
    except Exception as e: print(f"[REDACTED_BY_SCRIPT]")

    # --- Temporal Price & Transaction Dynamics ---
    try:
        if len(price_median_cols) > 1:
            price_median_series = df[price_median_cols].replace(0, np.nan)
            df['[REDACTED_BY_SCRIPT]'] = price_median_series.std(axis=1) / (price_median_series.mean(axis=1) + epsilon)
            df['[REDACTED_BY_SCRIPT]'] = price_median_series.std(axis=1)
        if len(trans_count_cols) > 1:
            df['int_transaction_spikiness'] = df[trans_count_cols].kurtosis(axis=1)
        if check_cols(['price_median_18q4', 'price_median_17q4']): df['[REDACTED_BY_SCRIPT]'] = df['price_median_18q4'] / (df['price_median_17q4'] + epsilon)
        
        q_07_cols = [c for c in price_median_cols if '_07' in c]; q_10_cols = [c for c in price_median_cols if '_10' in c]
        if q_07_cols and q_10_cols: df['[REDACTED_BY_SCRIPT]'] = df[q_10_cols].mean(axis=1) / (df[q_07_cols].mean(axis=1) + epsilon)
        
        q_18_cols = [c for c in price_median_cols if '_18' in c]; q_13_cols = [c for c in price_median_cols if '_13' in c]
        q_08_cols = [c for c in price_median_cols if '_08' in c]
        if q_18_cols and q_13_cols and q_08_cols:
             price_18_avg = df[q_18_cols].mean(axis=1); price_13_avg = df[q_13_cols].mean(axis=1); price_08_avg = df[q_08_cols].mean(axis=1)
             df['[REDACTED_BY_SCRIPT]'] = (price_18_avg - price_13_avg) - (price_13_avg - price_08_avg)
    except Exception as e: print(f"[REDACTED_BY_SCRIPT]")

    # --- Socio-Economic & Demographic Interactions ---
    try:
        if check_cols(['[REDACTED_BY_SCRIPT]', 'ons_1_bedroom', 'ons_2_bedrooms']): df['[REDACTED_BY_SCRIPT]'] = df['[REDACTED_BY_SCRIPT]'] * (df['ons_1_bedroom'] + df['ons_2_bedrooms'])
        if check_cols(['[REDACTED_BY_SCRIPT]', 'ahah_ah4phar']): df['int_retiree_amenity_access'] = df['[REDACTED_BY_SCRIPT]'] * df['ahah_ah4phar']
        if check_cols(['[REDACTED_BY_SCRIPT]', 'bb_bba225_sfu']): df['[REDACTED_BY_SCRIPT]'] = df['[REDACTED_BY_SCRIPT]'] * df['bb_bba225_sfu']
        if check_cols(['[REDACTED_BY_SCRIPT]', 'ahah_ah4gamb_pct']): df['[REDACTED_BY_SCRIPT]'] = df['[REDACTED_BY_SCRIPT]'] * df['ahah_ah4gamb_pct']
        if check_cols(['[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]']): df['[REDACTED_BY_SCRIPT]'] = df['[REDACTED_BY_SCRIPT]'] * df['[REDACTED_BY_SCRIPT]']
        if check_cols(['[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]']): df['[REDACTED_BY_SCRIPT]'] = df['[REDACTED_BY_SCRIPT]'] * df['[REDACTED_BY_SCRIPT]']
        if check_cols(['[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]']): df['[REDACTED_BY_SCRIPT]'] = df['[REDACTED_BY_SCRIPT]'] * df['[REDACTED_BY_SCRIPT]']
        if check_cols(['[REDACTED_BY_SCRIPT]', 'ons_4_or_more_bedrooms']): df['int_generational_transfer_potential'] = df['[REDACTED_BY_SCRIPT]'] * df['ons_4_or_more_bedrooms']
        
        if len(ethnic_cols) > 0:
            sq_ethnic_props = df[ethnic_cols].fillna(0).pow(2)
            df['int_ethnic_diversity_inverse_hhi'] = 1 - sq_ethnic_props.sum(axis=1)

    except Exception as e: print(f"[REDACTED_BY_SCRIPT]")
    
    # --- Environmental, Geographic & Property Interactions ---
    try:
        if check_cols(['ages_bp_pre_1900', 'ahah_ah4no2']): df['[REDACTED_BY_SCRIPT]'] = df['ages_bp_pre_1900'] * df['ahah_ah4no2']
        if check_cols(['[REDACTED_BY_SCRIPT]', 'ahah_ah4blue']): df['[REDACTED_BY_SCRIPT]'] = df['[REDACTED_BY_SCRIPT]'] * (1 / (df['ahah_ah4blue'] + epsilon))
        if check_cols(['ahah_ah4no2_rnk', 'ahah_ah4hosp']): df['[REDACTED_BY_SCRIPT]'] = df['ahah_ah4no2_rnk'] * df['ahah_ah4hosp']
        if check_cols(['ahah_ah4gpas_pct', '[REDACTED_BY_SCRIPT]']): df['[REDACTED_BY_SCRIPT]'] = df['ahah_ah4gpas_pct'] * df['[REDACTED_BY_SCRIPT]']
        if len(age_cols) > 1: df['[REDACTED_BY_SCRIPT]'] = df[age_cols].std(axis=1)
        if check_cols(['veg_ndvi_mean', 'ahah_ah4pm10_pct']): df['int_green_vs_grey_space'] = df['veg_ndvi_mean'] / (df['ahah_ah4pm10_pct'] + epsilon)
        if check_cols(['latitude', '[REDACTED_BY_SCRIPT]']): df['int_latitude_wealth_gradient'] = df['latitude'] * df['[REDACTED_BY_SCRIPT]']
        if check_cols(['longitude', 'ages_bp_pre_1900']): df['int_longitude_age_gradient'] = df['longitude'] * df['ages_bp_pre_1900']
        if check_cols(['latitude', 'longitude']): df['int_remoteness_index'] = (df['latitude'] - 54.0)**2 + (df['longitude'] + 2.5)**2
        if check_cols(['ahah_ah4pm10_rnk', 'ahah_ah4hosp']): df['[REDACTED_BY_SCRIPT]'] = df['ahah_ah4pm10_rnk'] * (1 / (df['ahah_ah4hosp'] + epsilon))
    except Exception as e: print(f"[REDACTED_BY_SCRIPT]")

    # --- Final Count ---
    new_cols_count = len(set(df.columns) - original_cols)
    print(f"[REDACTED_BY_SCRIPT]")
    return df


# --- Main Processing Function ---
def process_data_in_chunks():
    """
    Loads the main dataset in chunks, merges it with lookup tables,
    creates a massive number of features, and saves the result incrementally.
    """
    print("[REDACTED_BY_SCRIPT]")

    # --- Step 1: Load and Prepare Lookup Tables ---
    print("[REDACTED_BY_SCRIPT]")
    try:
        df_prices = pd.read_csv(PRICE_ANALYSIS_FILE, dtype={'postcode': str})
        df_prices = clean_col_names(df_prices)
        
        df_coords = pd.read_csv(COORDINATES_FILE, dtype={'pcds': str})
        df_coords = clean_col_names(df_coords)
        if 'pcds' in df_coords.columns:
            df_coords.rename(columns={'pcds': 'postcode'}, inplace=True)

        print("[REDACTED_BY_SCRIPT]")
        df_lookup = pd.merge(df_prices, df_coords, on='postcode', how='outer')
        lookup_numeric_cols = [col for col in df_lookup.columns if col != 'postcode']
        for col in lookup_numeric_cols:
            df_lookup[col] = pd.to_numeric(df_lookup[col], errors='coerce')
        print(f"[REDACTED_BY_SCRIPT]")

    except Exception as e:
        print(f"[REDACTED_BY_SCRIPT]")
        traceback.print_exc()
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
            df_main_chunk = batch.to_pandas(split_blocks=True, self_destruct=True)
            
            # MERGE STEP
            df_merged_chunk = pd.merge(df_main_chunk, df_lookup, on='postcode', how='left')
            df_merged_chunk = df_merged_chunk.loc[:,~df_merged_chunk.columns.duplicated()]

            # --- NEW: FULL FEATURE INTERACTION STEP ---
            df_processed_chunk = create_feature_interactions(df_merged_chunk)

            # Convert to PyArrow Table for writing
            table = pa.Table.from_pandas(df_processed_chunk, preserve_index=False)

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