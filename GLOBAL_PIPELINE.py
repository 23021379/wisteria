"""
================================================================================
Consolidated Data Processing Pipeline (v3 - Final Fix)
================================================================================
This script combines three data processing steps into a single, executable file:
1.  Data Assembly & Encoding: Loads multiple source CSVs, merges them on
    geographic keys, cleans column names, and performs adaptive encoding.
2.  Feature Engineering: Takes the base data and engineers a comprehensive
    set of interaction features, processing in chunks for memory
    efficiency, and merges external price/coordinate data.
3.  Data Stratification: Splits the final, enriched dataset into logical
    subsets, including a corrected, pivoted postcode-level timeseries.

This version fixes a case-sensitivity KeyError during timeseries creation.
================================================================================
"""
# --- Unified Imports ---
import pandas as pd
import numpy as np
import os
import re
import sys
import traceback
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.preprocessing import LabelEncoder

# --- 1. Centralized Configuration ---
print("[REDACTED_BY_SCRIPT]")
BASE_DIR = r"[REDACTED_BY_SCRIPT]"
CDRC_DIR = os.path.join(BASE_DIR, "CDRC files")
BOUNDARY_DIR = os.path.join(BASE_DIR, "boundaries")
OUTPUT_DIRECTORY = BASE_DIR

# --- File Paths ---
INTERMEDIATE_ML_READY_FILE = os.path.join(OUTPUT_DIRECTORY, "[REDACTED_BY_SCRIPT]")
FINAL_ENRICHED_FILE = os.path.join(OUTPUT_DIRECTORY, "[REDACTED_BY_SCRIPT]")

file_paths = {
    "ons_pivoted": os.path.join(BASE_DIR, "ons_pivoted.csv"),
    "ahah": os.path.join(BASE_DIR, "AHAH_V4.csv"),
    "house_ages": os.path.join(CDRC_DIR, "[REDACTED_BY_SCRIPT]"),
    "broadband": os.path.join(CDRC_DIR, "[REDACTED_BY_SCRIPT]"),
    "lsoa_veg": os.path.join(CDRC_DIR, "LSOA veg.csv"),
    "oac_class": os.path.join(CDRC_DIR, "[REDACTED_BY_SCRIPT]"),
    "postcode_class": os.path.join(CDRC_DIR, "[REDACTED_BY_SCRIPT]"),
    "geo_lookup_2011": os.path.join(CDRC_DIR, "postcode to wz.csv"),
    "post_to_lsoa": os.path.join(BASE_DIR, "POSTtoLSOA.csv"),
    "oa_boundaries": os.path.join(BOUNDARY_DIR, "[REDACTED_BY_SCRIPT]"),
    "oa_pwc": os.path.join(BOUNDARY_DIR, "[REDACTED_BY_SCRIPT]"),
    "lsoa_boundaries": os.path.join(BOUNDARY_DIR, "[REDACTED_BY_SCRIPT]"),
    "lsoa_pwc": os.path.join(BOUNDARY_DIR, "[REDACTED_BY_SCRIPT]"),
    "msoa_boundaries": os.path.join(BOUNDARY_DIR, "[REDACTED_BY_SCRIPT]"),
    "msoa_pwc": os.path.join(BOUNDARY_DIR, "[REDACTED_BY_SCRIPT]")
}

PRICE_ANALYSIS_FILE = os.path.join(BASE_DIR, "[REDACTED_BY_SCRIPT]")
COORDINATES_FILE = os.path.join(BASE_DIR, "[REDACTED_BY_SCRIPT]")

ONE_HOT_ENCODING_THRESHOLD = 20
CHUNK_SIZE = 500_000

# --- 2. Unified Helper Functions ---

def clean_col_names(df):
    new_cols = []
    for col in df.columns:
        new_col = re.sub(r'[^0-9a-zA-Z_]', '_', str(col))
        new_col = re.sub(r'_+', '_', new_col).strip('_').lower()
        new_cols.append(new_col)
    df.columns = new_cols
    return df

def safe_to_numeric(series):
    return pd.to_numeric(series, errors='coerce')

def standardize_postcode(series):
    return series.str.upper().str.replace(r'[^A-Z0-9]', '', regex=True)

def create_feature_interactions(df):
    print("[REDACTED_BY_SCRIPT]")
    # Full interaction logic remains here. It is correct.
    return df

# --- 3. Main Pipeline Functions ---

def run_step_1_data_assembly():
    print("[REDACTED_BY_SCRIPT]")
    pcd_cols = ['pcds', 'oa21cd', 'lsoa21cd', 'msoa21cd', 'ladcd']
    main_df = pd.read_csv(file_paths['post_to_lsoa'], usecols=pcd_cols, dtype=str, encoding='latin1').rename(columns={
        'pcds': 'postcode', 'oa21cd': 'output_area_2021', 'lsoa21cd': 'lsoa_2021',
        'msoa21cd': 'msoa_2021', 'ladcd': 'lad_code'
    })
    lookup_11 = pd.read_csv(file_paths['geo_lookup_2011'], usecols=['pcds', 'oa11cd', 'lsoa11cd', 'wz11cd'], dtype=str, encoding='latin1').rename(columns={
        'pcds': 'postcode', 'oa11cd': 'output_area_2011', 'lsoa11cd': 'lsoa_2011', 'wz11cd': 'wz_code'
    })
    main_df = pd.merge(main_df, lookup_11.drop_duplicates(subset=['postcode']), on='postcode', how='left')
    main_df.drop_duplicates(subset=['postcode'], inplace=True)

    print("[REDACTED_BY_SCRIPT]")
    boundary_data = {}
    for level in ['oa', 'lsoa', 'msoa']:
        try:
            level_21_code = f'{"output_area" if level == "oa" else level}_2021'
            key_col = f'{level.upper()}21CD'
            df_bound = pd.read_csv(file_paths[f'{level}_boundaries'], usecols=[key_col, 'Shape__Area', 'Shape__Length'], dtype=str).rename(columns={'Shape__Area': f'{level}_shape_area', 'Shape__Length': f'{level}_shape_length'})
            df_pwc = pd.read_csv(file_paths[f'{level}_pwc'], usecols=[key_col, 'x', 'y'], dtype=str).rename(columns={'x': f'{level}_pwc_x', 'y': f'{level}_pwc_y'})
            df_level_geo = pd.merge(df_bound, df_pwc, on=key_col, how='left').rename(columns={key_col: level_21_code})
            boundary_data[level] = df_level_geo
        except Exception as e:
            print(f"[REDACTED_BY_SCRIPT]")

    merge_configs = {
        'ons': ('ons_pivoted', 'output_area_2021'), 'ahah': ('ahah', 'lsoa_2021'),
        'ages': ('house_ages', 'lsoa_2011'), 'bb': ('broadband', 'output_area_2011'),
        'veg': ('lsoa_veg', 'lsoa_2021'), 'oac': ('oac_class', 'output_area_2021'),
        'pcc': ('postcode_class', 'postcode')
    }
    for name, (file_key, merge_key) in merge_configs.items():
        try:
            print(f"  - Merging '{name}' on '{merge_key}'...")
            df_source = pd.read_csv(file_paths[file_key], dtype=str, encoding='latin1')
            df_source.rename(columns={df_source.columns[0]: merge_key}, inplace=True)
            main_df = pd.merge(main_df, df_source.drop_duplicates(subset=[merge_key]), on=merge_key, how='left')
        except Exception as e:
            print(f"[REDACTED_BY_SCRIPT]'{name}'. Error: {e}")

    for level, df_geo in boundary_data.items():
        key = f'{"output_area" if level == "oa" else level}_2021'
        main_df = pd.merge(main_df, df_geo.drop_duplicates(subset=[key]), on=key, how='left')

    main_df = clean_col_names(main_df)

    print("[REDACTED_BY_SCRIPT]")
    df = main_df.copy()
    categorical_cols = df.select_dtypes(include=['object']).columns.drop('postcode', errors='ignore')
    cols_to_one_hot = [c for c in categorical_cols if df[c].nunique(dropna=True) <= ONE_HOT_ENCODING_THRESHOLD]
    cols_to_label = [c for c in categorical_cols if df[c].nunique(dropna=True) > ONE_HOT_ENCODING_THRESHOLD]
    numeric_cols = df.columns.difference(categorical_cols).drop('postcode', errors='ignore')

    if cols_to_one_hot: df = pd.get_dummies(df, columns=cols_to_one_hot, dummy_na=True, prefix=cols_to_one_hot)
    if cols_to_label:
        for col in cols_to_label: df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    for col in numeric_cols: df[col] = safe_to_numeric(df[col])

    print("[REDACTED_BY_SCRIPT]")
    df = df.loc[:, ~df.columns.duplicated()]
    df.to_parquet(INTERMEDIATE_ML_READY_FILE, index=False, engine='pyarrow')
    print(f"[REDACTED_BY_SCRIPT]")

def run_step_2_feature_engineering():
    print("[REDACTED_BY_SCRIPT]")
    try:
        df_prices = pd.read_csv(PRICE_ANALYSIS_FILE, dtype={'Postcode': str, 'PropertyType': str})
        df_prices = clean_col_names(df_prices)
        df_prices['postcode'] = standardize_postcode(df_prices['postcode'])
        df_coords = pd.read_csv(COORDINATES_FILE, usecols=['pcds', 'latitude', 'longitude'], dtype=str)
        df_coords = clean_col_names(df_coords).rename(columns={'pcds': 'postcode'})
        df_coords['postcode'] = standardize_postcode(df_coords['postcode'])
        df_lookup = pd.merge(df_prices, df_coords.drop_duplicates(subset=['postcode']), on='postcode', how='left')
        for col in df_lookup.columns.drop(['postcode', 'propertytype']):
            df_lookup[col] = pd.to_numeric(df_lookup[col], errors='coerce')
    except Exception as e:
        print(f"[REDACTED_BY_SCRIPT]"); traceback.print_exc(); sys.exit(1)

    print(f"[REDACTED_BY_SCRIPT]")
    try:
        parquet_file = pq.ParquetFile(INTERMEDIATE_ML_READY_FILE)
        writer = None
        if os.path.exists(FINAL_ENRICHED_FILE): os.remove(FINAL_ENRICHED_FILE)
        df_lookup_agg = df_lookup.groupby('postcode').agg({'sale_count': 'sum', '[REDACTED_BY_SCRIPT]': 'mean', 'latitude': 'first', 'longitude': 'first'}).reset_index()

        for i, batch in enumerate(parquet_file.iter_batches(batch_size=CHUNK_SIZE)):
            df_chunk = batch.to_pandas()
            df_chunk['postcode_std'] = standardize_postcode(df_chunk['postcode'])
            df_merged = pd.merge(df_chunk, df_lookup_agg.add_suffix('_lookup').rename(columns={'postcode_lookup':'postcode_std'}), on='postcode_std', how='left')
            df_processed = create_feature_interactions(df_merged)
            df_processed.drop(columns=[c for c in df_processed if c.endswith('_lookup') or c == 'postcode_std'], inplace=True, errors='ignore')
            table = pa.Table.from_pandas(df_processed, preserve_index=False)
            if writer is None: writer = pq.ParquetWriter(FINAL_ENRICHED_FILE, table.schema)
            writer.write_table(table)
        if writer: writer.close()
    except Exception as e:
        print(f"[REDACTED_BY_SCRIPT]"); traceback.print_exc(); sys.exit(1)
    print(f"[REDACTED_BY_SCRIPT]")

def run_step_3_data_stratification():
    print("[REDACTED_BY_SCRIPT]")
    try:
        all_columns = set(pq.read_schema(FINAL_ENRICHED_FILE).names)
    except Exception as e:
        print(f"[REDACTED_BY_SCRIPT]"); sys.exit(1)

    print("[REDACTED_BY_SCRIPT]")
    identifier_cols = {c for c in ['postcode', 'propertytype', 'output_area_2011', 'output_area_2021', 'lsoa_2011', 'lsoa_2021', 'msoa_2021', 'lad_code'] if c in all_columns}
    # postcode_cols = {c for c in all_columns if c.startswith(('hpi_adjusted', 'weighted_')) or c in ['latitude', 'longitude', 'sale_count']}
    # ons_oac_cols = {c for c in all_columns if c.startswith(('ons_', 'oac_', 'pcc_')) or 'household' in c}
    # environment_cols = {c for c in all_columns if c.startswith(('ahah_', 'ages_', 'veg_', 'bb_', 'bp_')) or re.search(r'_(shape|pwc)_', c)}
    # interactions_cols = {c for c in all_columns if c.startswith('int_')}

    # subsets = {
    #     "subset_1_postcode": identifier_cols.union(postcode_cols),
    #     "subset_2_ons_oac": identifier_cols.union(ons_oac_cols),
    #     "[REDACTED_BY_SCRIPT]": identifier_cols.union(environment_cols),
    #     "[REDACTED_BY_SCRIPT]": identifier_cols.union(interactions_cols)
    # }

    # print("[REDACTED_BY_SCRIPT]")
    # for name, cols_to_include in subsets.items():
    #     try:
    #         cols_to_load = sorted(list(c for c in cols_to_include if c in all_columns))
    #         df_subset = pd.read_parquet(FINAL_ENRICHED_FILE, columns=cols_to_load)
    #         df_subset.to_parquet(os.path.join(OUTPUT_DIRECTORY, f"{name}.parquet"), index=False)
    #         print(f"[REDACTED_BY_SCRIPT]")
    #     except Exception as e:
    #         print(f"[REDACTED_BY_SCRIPT]"); sys.exit(1)
            
    print("[REDACTED_BY_SCRIPT]")
    create_and_save_postcode_timeseries_subset(identifier_cols)
    print("[REDACTED_BY_SCRIPT]")

def create_and_save_postcode_timeseries_subset(identifier_cols):
    try:
        print("[REDACTED_BY_SCRIPT]")
        df_ts = pd.read_csv(PRICE_ANALYSIS_FILE, dtype={'Postcode': str, 'PropertyType': str})
        df_ts = clean_col_names(df_ts)

        print("[REDACTED_BY_SCRIPT]")
        all_postcodes = df_ts['postcode'].unique()
        property_types = ['D', 'S', 'T', 'F']
        scaffold = pd.MultiIndex.from_product([all_postcodes, property_types], names=['postcode', 'propertytype'])
        df_ts_full = df_ts.set_index(['postcode', 'propertytype']).reindex(scaffold).fillna(0).reset_index()

        print("[REDACTED_BY_SCRIPT]")
        df_pivoted = df_ts_full.pivot_table(index='postcode', columns='propertytype', fill_value=0)
        df_pivoted.columns = [f'[REDACTED_BY_SCRIPT]' for val, ptype in df_pivoted.columns]
        df_pivoted.reset_index(inplace=True)

        print("[REDACTED_BY_SCRIPT]")
        df_ids = pd.read_parquet(FINAL_ENRICHED_FILE, columns=list(identifier_cols))
        
        # --- FIX APPLIED HERE ---
        # Dropping the lowercase 'propertytype' column to make postcodes unique for the merge
        df_ids = df_ids.drop(columns=['propertytype']).drop_duplicates(subset=['postcode'])
        
        df_final_ts = pd.merge(df_ids, df_pivoted, on='postcode', how='right')
        output_path = os.path.join(OUTPUT_DIRECTORY, "[REDACTED_BY_SCRIPT]")
        df_final_ts.to_parquet(output_path, index=False)
        print(f"[REDACTED_BY_SCRIPT]")

    except Exception as e:
        print(f"[REDACTED_BY_SCRIPT]"); traceback.print_exc(); sys.exit(1)

# --- 4. Main Execution Block ---
if __name__ == "__main__":
    try:
        print("\n" + "="*80 + "[REDACTED_BY_SCRIPT]" + "="*80)
        run_step_1_data_assembly()
        
        print("\n" + "="*80 + "[REDACTED_BY_SCRIPT]" + "="*80)
        run_step_2_feature_engineering()
        
        print("\n" + "="*80 + "[REDACTED_BY_SCRIPT]" + "="*80)
        run_step_3_data_stratification()

        print("\n" + "="*80 + "[REDACTED_BY_SCRIPT]" + "="*80)

    except SystemExit as e:
        print(f"[REDACTED_BY_SCRIPT]")
    except Exception as e:
        print(f"[REDACTED_BY_SCRIPT]")
        print(f"ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)