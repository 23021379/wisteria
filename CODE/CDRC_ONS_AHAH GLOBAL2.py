"""
Overview
The script combines multiple datasets related to UK housing, geography, demographics, and amenities into a single enriched dataset for analysis or machine learning.

Main Components
1. Configuration & Setup (Lines 1-34)
Sets up file paths for various data sources including:
ONS (Office for National Statistics) data
AHAH (Access to Healthy Assets & Hazards) data
CDRC (Consumer Data Research Centre) files
Boundary and centroid files for different geographic levels
2. Helper Functions (Lines 37-60)
clean_col_names(): Standardizes column names by removing special characters
safe_to_numeric(): Safely converts data to numeric format
calculate_trend_slope(): Calculates linear regression slopes for time series data
create_interaction(): Creates interaction features between variables
3. Data Loading & Geographic Mapping (Lines 63-75)
Loads postcode-to-geography lookups for both 2011 and 2021 boundaries
Creates mappings between postcodes and various geographic areas (Output Areas, LSOAs, MSOAs)
4. Boundary Data Processing (Lines 79-101)
Processes geographic boundary files to extract:
Shape area and perimeter for each geographic unit
Population-weighted centroids (PWC) coordinates
Calculates these for Output Areas (OA), Lower Super Output Areas (LSOA), and Middle Super Output Areas (MSOA)
5. Data Merging (Lines 104-126)
Systematically merges multiple datasets:
ONS data: Census and demographic information
AHAH data: Access to health assets and environmental hazards
Property ages: Building age distributions
Broadband data: Internet connectivity metrics
Vegetation data: NDVI (greenness) measures
Property prices & transactions: Housing market data
Classification data: Area and postcode classifications
6. Feature Engineering (Lines 131-186)
6.1 Compactness Ratios (Lines 134-143)
Calculates how "compact" or circular each geographic area is using the isoperimetric quotient
6.2 Broadband Features (Lines 146-153)
Creates features for latest broadband speeds and trends over time
6.3 Property Market Features (Lines 156-164)
Price volatility: Standard deviation of prices across quarters
Price growth: Year-over-year price changes
Market liquidity: Transaction volume relative to housing stock
6.4 Property Age Proportions (Lines 167-173)
Converts raw counts to proportions of properties in different age brackets
6.5 One-Hot Encoding (Lines 176-180)
Creates dummy variables for categorical features like area classifications
6.6 AHAH Inversions & Interactions (Lines 183-191)
Inverts AHAH scores (since lower values typically mean better access)
Creates interaction features between health access and area types
6.7 ONS Proportions & Interactions (Lines 194-206)
Converts census counts to proportions
Creates interaction features between demographic and geographic variables
7. Final Processing & Export (Lines 209-220)
Removes duplicate columns
Standardizes column names
Reorders columns (geography keys first, then features)
Saves the final dataset as a Parquet file
"""

import pandas as pd
import numpy as np
import os
import re
from scipy.stats import linregress

# --- Configuration ---
BASE_DIR = r"[REDACTED_BY_SCRIPT]"
CDRC_DIR = os.path.join(BASE_DIR, "CDRC files")
BOUNDARY_DIR = os.path.join(BASE_DIR, "boundaries") 
POST_TO_LSOA_FILE = os.path.join(BASE_DIR, "POSTtoLSOA.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "[REDACTED_BY_SCRIPT]")

# Dictionary of file paths
file_paths = {
    # Main data
    "ons_pivoted": os.path.join(BASE_DIR, "ons_pivoted.csv"),
    "ahah": os.path.join(BASE_DIR, "AHAH_V4.csv"),
    # CDRC data
    "house_ages": os.path.join(CDRC_DIR, "[REDACTED_BY_SCRIPT]"),
    "broadband": os.path.join(CDRC_DIR, "[REDACTED_BY_SCRIPT]"),
    "lsoa_veg": os.path.join(CDRC_DIR, "LSOA veg.csv"),
    "oac_class": os.path.join(CDRC_DIR, "[REDACTED_BY_SCRIPT]"),
    "postcode_class": os.path.join(CDRC_DIR, "[REDACTED_BY_SCRIPT]"),
    "quarterly_prices": os.path.join(CDRC_DIR, "[REDACTED_BY_SCRIPT]"),
    "[REDACTED_BY_SCRIPT]": os.path.join(CDRC_DIR, "[REDACTED_BY_SCRIPT]"),
    "geo_lookup_2011": os.path.join(CDRC_DIR, "postcode to wz.csv"),
    # NEW: Boundary and Centroid Files
    "oa_boundaries": os.path.join(BOUNDARY_DIR, "[REDACTED_BY_SCRIPT]"),
    "oa_pwc": os.path.join(BOUNDARY_DIR, "[REDACTED_BY_SCRIPT]"),
    "lsoa_boundaries": os.path.join(BOUNDARY_DIR, "[REDACTED_BY_SCRIPT]"),
    "lsoa_pwc": os.path.join(BOUNDARY_DIR, "[REDACTED_BY_SCRIPT]"),
    "msoa_boundaries": os.path.join(BOUNDARY_DIR, "[REDACTED_BY_SCRIPT]"),
    "msoa_pwc": os.path.join(BOUNDARY_DIR, "[REDACTED_BY_SCRIPT]"),
}


# --- Helper Functions (No changes here) ---
def clean_col_names(df, prefix=''):
    new_cols = []
    for col in df.columns:
        new_col = re.sub(r'[^0-9a-zA-Z_]', '_', str(col))
        new_col = re.sub(r'_+', '_', new_col).strip('_').lower()
        if prefix and not new_col.startswith(prefix):
            new_col = f"{prefix}{new_col}"
        new_cols.append(new_col)
    df.columns = new_cols
    return df

def safe_to_numeric(series):
    return pd.to_numeric(series, errors='coerce')

def calculate_trend_slope(df, value_cols_ordered, year_values_ordered):
    if not value_cols_ordered or not year_values_ordered or not all(c in df.columns for c in value_cols_ordered):
        return pd.Series([np.nan] * len(df), index=df.index)
    numeric_data = df[value_cols_ordered].apply(safe_to_numeric)
    def get_slope(row):
        valid_idx = row.notna()
        if valid_idx.sum() < 2: return np.nan
        try: return linregress(np.array(year_values_ordered)[valid_idx], row[valid_idx]).slope #type: ignore
        except: return np.nan
    return numeric_data.apply(get_slope, axis=1)

interaction_creation_log = []
def create_interaction(df, col1, col2, prefix=""):
    global interaction_creation_log
    if col1 in df.columns and col2 in df.columns:
        c1_short = re.sub(r'[^a-zA-Z0-9]', '', col1.split('_')[-1])[:15]
        c2_short = re.sub(r'[^a-zA-Z0-9]', '', col2.split('_')[-1])[:15]
        interaction_name = f"[REDACTED_BY_SCRIPT]".lower()
        df[interaction_name] = safe_to_numeric(df[col1]) * safe_to_numeric(df[col2])
        interaction_creation_log.append(interaction_name)

# --- Main Execution ---
print("[REDACTED_BY_SCRIPT]")
# Same as before
pcd_cols = ['pcds', 'oa21cd', 'lsoa21cd', 'msoa21cd', 'ladcd']
main_df = pd.read_csv(POST_TO_LSOA_FILE, usecols=pcd_cols, dtype=str, encoding='latin1').rename(columns={
    'pcds': 'postcode', 'oa21cd': 'output_area_2021', 'lsoa21cd': 'lsoa_2021',
    'msoa21cd': 'msoa_2021', 'ladcd': 'lad_code'
})
lookup_11 = pd.read_csv(file_paths['geo_lookup_2011'], usecols=['pcds', 'oa11cd', 'lsoa11cd', 'wz11cd'], dtype=str, encoding='latin1').rename(columns={
    'pcds': 'postcode', 'oa11cd': 'output_area_2011', 'lsoa11cd': 'lsoa_2011', 'wz11cd': 'wz_code'
})
main_df = pd.merge(main_df, lookup_11.drop_duplicates(subset=['postcode']), on='postcode', how='left')

print("[REDACTED_BY_SCRIPT]")

# --- Boundary Data Pre-processing ---
print("[REDACTED_BY_SCRIPT]")
boundary_data = {}
for level in ['oa', 'lsoa', 'msoa']:
    try:
        level_21_code = f'{level}_2021' if level != 'oa' else 'output_area_2021'
        key_col = f'{level.upper()}21CD'
        
        # Load boundaries (Area, Length)
        df_bound = pd.read_csv(file_paths[f'{level}_boundaries'], encoding='latin1', dtype=str)
        df_bound = df_bound[[key_col, 'Shape__Area', 'Shape__Length']].rename(columns={
            'Shape__Area': f'{level}_shape_area', 'Shape__Length': f'{level}_shape_length'
        })
        
        # Load Population Weighted Centroids (PWC)
        df_pwc = pd.read_csv(file_paths[f'{level}_pwc'], encoding='latin1', dtype=str)
        df_pwc = df_pwc[[key_col, 'x', 'y']].rename(columns={
            'x': f'{level}_pwc_x', 'y': f'{level}_pwc_y'
        })
        
        # Combine boundary and PWC data for the level
        df_level_geo = pd.merge(df_bound, df_pwc, on=key_col, how='left')
        df_level_geo = df_level_geo.rename(columns={key_col: level_21_code})
        boundary_data[level] = df_level_geo
        print(f"[REDACTED_BY_SCRIPT]")
    except Exception as e:
        print(f"[REDACTED_BY_SCRIPT]")

# --- Merge Main Data Sources ---
merge_configs = {
    'ons': ('ons_pivoted', 'output_area_2021', 'ons_'), 'ahah': ('ahah', 'lsoa_2021', 'ahah_'),
    'ages': ('house_ages', 'lsoa_2011', 'ages_'), 'bb': ('broadband', 'output_area_2011', 'bb_'),
    'veg': ('lsoa_veg', 'lsoa_2021', 'veg_'), 'prices': ('quarterly_prices', 'lsoa_2011', 'price_'),
    'trans': ('[REDACTED_BY_SCRIPT]', 'lsoa_2011', 'trans_'), 'oac': ('oac_class', 'output_area_2021', 'oac_'),
    'pcc': ('postcode_class', 'postcode', 'pcc_')
}
for name, (file_key, merge_key, prefix) in merge_configs.items():
    # ... (same merging logic as before) ...
    try:
        print(f"  - Loading and merging '{name}'...")
        df_source = pd.read_csv(file_paths[file_key], dtype=str, encoding='latin1')
        df_source = clean_col_names(df_source)
        df_source.rename(columns={df_source.columns[0]: merge_key}, inplace=True)
        df_source.columns = [merge_key] + [f"{prefix}{col}" for col in df_source.columns if col != merge_key]
        main_df = pd.merge(main_df, df_source.drop_duplicates(subset=[merge_key]), on=merge_key, how='left')
    except Exception as e:
        print(f"[REDACTED_BY_SCRIPT]'{name}'. Error: {e}")


# --- Merge Boundary Data ---
print("[REDACTED_BY_SCRIPT]")
for level, df_geo in boundary_data.items():
    key = f'{"output_area" if level == "oa" else level}_2021'
    print(f"[REDACTED_BY_SCRIPT]'{key}'...")
    main_df = pd.merge(main_df, df_geo.drop_duplicates(subset=[key]), on=key, how='left')

print(f"[REDACTED_BY_SCRIPT]")
df_fe = main_df.copy()

print("[REDACTED_BY_SCRIPT]")

# --- 5.0 NEW: Compactness Ratio ---
print("[REDACTED_BY_SCRIPT]")
for level in ['oa', 'lsoa', 'msoa']:
    area_col = f'{level}_shape_area'
    perimeter_col = f'{level}_shape_length'
    if area_col in df_fe.columns and perimeter_col in df_fe.columns:
        area = safe_to_numeric(df_fe[area_col])
        perimeter = safe_to_numeric(df_fe[perimeter_col]).replace(0, np.nan) # Avoid division by zero
        # Isoperimetric Quotient
        df_fe[f'[REDACTED_BY_SCRIPT]'] = (4 * np.pi * area) / (perimeter**2)
        print(f"    Created '{f'[REDACTED_BY_SCRIPT]'}'")

# --- All other feature engineering steps remain the same ---
# (5.1 Broadband, 5.2 Price/Tx, 5.3 Ages, 5.4 OHE, 5.5 AHAH, 5.6 ONS, 5.7 Final Interactions)
# This entire block is copied from the prompt and adapted for the new column names

# --- 5.1 Broadband Features ---
print("[REDACTED_BY_SCRIPT]")
try:
    bb_types = ['dow', 'uf', 'sfu']
    for bb_type in bb_types:
        latest_col, earliest_col = f'bb_bba225_{bb_type}', f'bb_bba166_{bb_type}'
        if latest_col in df_fe.columns:
            df_fe[f'feat_bb_latest_{bb_type}'] = safe_to_numeric(df_fe[latest_col])
            if earliest_col in df_fe.columns:
                df_fe[f'feat_bb_trend_{bb_type}'] = safe_to_numeric(df_fe[latest_col]) - safe_to_numeric(df_fe[earliest_col])
except Exception as e: print(f"[REDACTED_BY_SCRIPT]")

# --- 5.2 Price and Transaction Features ---
print("[REDACTED_BY_SCRIPT]")
try:
    price_cols_18 = [c for c in df_fe.columns if c.startswith('price_median_18q')]
    if len(price_cols_18) > 1: df_fe['[REDACTED_BY_SCRIPT]'] = df_fe[price_cols_18].apply(safe_to_numeric).std(axis=1)
    if 'price_median_18q4' in df_fe and 'price_median_17q4' in df_fe:
        df_fe['[REDACTED_BY_SCRIPT]'] = (safe_to_numeric(df_fe['price_median_18q4']) - safe_to_numeric(df_fe['price_median_17q4'])) / safe_to_numeric(df_fe['price_median_17q4']).replace(0, np.nan)
    trans_cols_18 = [c for c in df_fe.columns if c.startswith('trans_count_18q')]
    if trans_cols_18 and 'ages_all_properties' in df_fe:
        avg_trans_18 = df_fe[trans_cols_18].apply(safe_to_numeric).mean(axis=1)
        df_fe['[REDACTED_BY_SCRIPT]'] = avg_trans_18 / safe_to_numeric(df_fe['ages_all_properties']).replace(0, np.nan)
except Exception as e: print(f"[REDACTED_BY_SCRIPT]")

# --- 5.3 Property Age Features ---
print("[REDACTED_BY_SCRIPT]")
try:
    bp_cols = [c for c in df_fe.columns if c.startswith('ages_bp_')]
    if 'ages_all_properties' in df_fe.columns:
        total_props_numeric = safe_to_numeric(df_fe['ages_all_properties']).replace(0, np.nan)
        for col in bp_cols:
            if col != 'ages_all_properties': df_fe[f'feat_{col}_prop'] = safe_to_numeric(df_fe[col]) / total_props_numeric
except Exception as e: print(f"[REDACTED_BY_SCRIPT]")

# --- 5.4 One-Hot Encoding ---
print("[REDACTED_BY_SCRIPT]")
ohe_configs = {'ages_mode1_type': 'cat_ages_period', 'oac_supergroup': 'cat_oac_supergroup', 'oac_group': 'cat_oac_group', 'pcc_classification': 'cat_pcc'}
for col, prefix in ohe_configs.items():
    if col in df_fe.columns:
        df_fe = pd.get_dummies(df_fe, columns=[col], prefix=prefix, dummy_na=True, dtype=float)

# --- 5.5 AHAH Inversions & Interactions ---
print("[REDACTED_BY_SCRIPT]")
try:
    ahah_inv_short = ['blue','dent','gp','hosp','phar','leis','pubs','ffood','tob','gamb','no2','so2','pm10','e']
    for shortname in ahah_inv_short:
        col_name = f'[REDACTED_BY_SCRIPT]'
        if col_name in df_fe.columns: df_fe[f'[REDACTED_BY_SCRIPT]'] = 100 - safe_to_numeric(df_fe[col_name])
    create_interaction(df_fe, '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', "ahah_oac")
    create_interaction(df_fe, '[REDACTED_BY_SCRIPT]', 'veg_ndvi_median', "ahah_veg")
except Exception as e: print(f"    AHAH FE failed: {e}")

# --- 5.6 ONS Proportions & Interactions ---
print("[REDACTED_BY_SCRIPT]")
prop_suffix_ons = '_prop'
try:
    family_cols = [c for c in df_fe.columns if c.startswith('ons_') and 'family' in c]
    if family_cols:
        total_hh = df_fe[family_cols].apply(safe_to_numeric).sum(axis=1, min_count=1).replace(0, np.nan)
        for col in [c for c in df_fe.columns if c.startswith('ons_') and c not in family_cols]:
            df_fe[f"[REDACTED_BY_SCRIPT]"] = safe_to_numeric(df_fe[col]) / total_hh
    disabled_lot_cols = [c for c in df_fe.columns if 'limited_a_lot' in c and c.endswith(prop_suffix_ons)]
    if disabled_lot_cols:
        df_fe['[REDACTED_BY_SCRIPT]'] = df_fe[disabled_lot_cols].sum(axis=1, min_count=1)
    create_interaction(df_fe, f'[REDACTED_BY_SCRIPT]', f'[REDACTED_BY_SCRIPT]', "ons_ons")
    create_interaction(df_fe, f'[REDACTED_BY_SCRIPT]', 'cat_pcc_a', "ons_pcc")
except Exception as e: print(f"    ONS FE failed: {e}")


print("[REDACTED_BY_SCRIPT]")
final_df = df_fe.loc[:, ~df_fe.columns.duplicated()]
final_df = clean_col_names(final_df)
key_cols = ['postcode', 'output_area_2011', 'output_area_2021', 'lsoa_2011', 'lsoa_2021', 'msoa_2021', 'lad_code', 'wz_code']
key_cols_exist = [col for col in key_cols if col in final_df.columns]
other_cols = sorted([col for col in final_df.columns if col not in key_cols_exist])
final_df = final_df[key_cols_exist + other_cols]

final_df.to_parquet(OUTPUT_FILE, index=False, engine='pyarrow')

print(f"[REDACTED_BY_SCRIPT]")
print(f"[REDACTED_BY_SCRIPT]")