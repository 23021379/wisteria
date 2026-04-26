# SCRIPT 1: 1_data_assembly_and_base_features.py (Overhauled)
"""
Overview:
- Consolidates all data loading and merging, including the critical postcode_price_analysis.
- Creates all base engineered features (compactness, broadband trends, etc.)
- Performs ML encoding.
- Saves a single, comprehensive, and well-structured file.
"""
import pandas as pd
import numpy as np
import os
import re
from sklearn.preprocessing import LabelEncoder

# --- Configuration ---
BASE_DIR = r"[REDACTED_BY_SCRIPT]"
CDRC_DIR = os.path.join(BASE_DIR, "CDRC files")
BOUNDARY_DIR = os.path.join(BASE_DIR, "boundaries")

# Define all file paths
file_paths = {
    # Main data
    "ons_pivoted": os.path.join(BASE_DIR, "ons_pivoted.csv"),
    "ahah": os.path.join(BASE_DIR, "AHAH_V4.csv"),
    "[REDACTED_BY_SCRIPT]": os.path.join(BASE_DIR, "[REDACTED_BY_SCRIPT].csv"),
    "post_to_lsoa": os.path.join(BASE_DIR, "POSTtoLSOA.csv"),
    # CDRC data
    "house_ages": os.path.join(CDRC_DIR, "[REDACTED_BY_SCRIPT]"),
    "broadband": os.path.join(CDRC_DIR, "[REDACTED_BY_SCRIPT]"),
    "lsoa_veg": os.path.join(CDRC_DIR, "LSOA veg.csv"),
    "oac_class": os.path.join(CDRC_DIR, "[REDACTED_BY_SCRIPT]"),
    "postcode_class": os.path.join(CDRC_DIR, "[REDACTED_BY_SCRIPT]"),
    "quarterly_prices": os.path.join(CDRC_DIR, "[REDACTED_BY_SCRIPT]"),
    "[REDACTED_BY_SCRIPT]": os.path.join(CDRC_DIR, "[REDACTED_BY_SCRIPT]"),
    "geo_lookup_2011": os.path.join(CDRC_DIR, "postcode to wz.csv"),
    # Boundary and Centroid Files
    "oa_boundaries": os.path.join(BOUNDARY_DIR, "[REDACTED_BY_SCRIPT]"),
    "oa_pwc": os.path.join(BOUNDARY_DIR, "[REDACTED_BY_SCRIPT]"),
    "lsoa_boundaries": os.path.join(BOUNDARY_DIR, "[REDACTED_BY_SCRIPT]"),
    "lsoa_pwc": os.path.join(BOUNDARY_DIR, "[REDACTED_BY_SCRIPT]"),
    "msoa_boundaries": os.path.join(BOUNDARY_DIR, "[REDACTED_BY_SCRIPT]"),
    "msoa_pwc": os.path.join(BOUNDARY_DIR, "[REDACTED_BY_SCRIPT]"),
}
OUTPUT_FILE = os.path.join(BASE_DIR, "[REDACTED_BY_SCRIPT]")
ONE_HOT_ENCODING_THRESHOLD = 20

# --- Helper Functions ---
def clean_col_names(df, prefix=''):
    df.columns = df.columns.str.lower().str.replace(r'[^0-9a-z_]+', '_', regex=True).str.strip('_')
    if prefix:
        df.columns = [f"{prefix}_{col}" if not col.startswith(prefix) else col for col in df.columns]
    return df

def safe_to_numeric(series):
    return pd.to_numeric(series, errors='coerce')

def standardize_postcode(series):
    return series.str.upper().str.replace(r'[^A-Z0-9]', '', regex=True)

print("[REDACTED_BY_SCRIPT]")
# Load the price analysis file, which is at the Postcode-PropertyType level. This is our new base.
df_main = pd.read_csv(file_paths['[REDACTED_BY_SCRIPT]'])
df_main.columns = [c.lower() for c in df_main.columns] # Clean column names early
df_main.rename(columns={'postcode': 'postcode_raw', 'propertytype': 'propertytype'}, inplace=True)
df_main['postcode'] = standardize_postcode(df_main['postcode_raw'])

print("[REDACTED_BY_SCRIPT]")
# Load geographic lookups
df_geo_21 = pd.read_csv(file_paths['post_to_lsoa'], usecols=['pcds', 'oa21cd', 'lsoa21cd', 'msoa21cd', 'ladcd'], dtype=str, encoding='latin1')
df_geo_21['postcode'] = standardize_postcode(df_geo_21['pcds'])
df_geo_11 = pd.read_csv(file_paths['geo_lookup_2011'], usecols=['pcds', 'oa11cd', 'lsoa11cd', 'wz11cd'], dtype=str, encoding='latin1')
df_geo_11['postcode'] = standardize_postcode(df_geo_11['pcds'])

# Merge all geographic keys onto the main dataframe
df_main = pd.merge(df_main, df_geo_21.drop(columns='pcds').drop_duplicates(subset='postcode'), on='postcode', how='left')
df_main = pd.merge(df_main, df_geo_11.drop(columns='pcds').drop_duplicates(subset='postcode'), on='postcode', how='left')
df_main.rename(columns={'oa21cd': 'output_area_2021', 'lsoa21cd': 'lsoa_2021', 'msoa21cd': 'msoa_2021',
                        'oa11cd': 'output_area_2011', 'lsoa11cd': 'lsoa_2011', 'ladcd': 'lad_code'}, inplace=True)

# Define and execute merges for other data sources
merge_configs = {
    'ons': ('ons_pivoted', 'output_area_2021', 'ons'), 'ahah': ('ahah', 'lsoa_2021', 'ahah'),
    'ages': ('house_ages', 'lsoa_2011', 'ages'), 'bb': ('broadband', 'output_area_2011', 'bb'),
    'veg': ('lsoa_veg', 'lsoa_2021', 'veg'),
    'q_prices': ('quarterly_prices', 'lsoa_2011', 'q_price'),
    'q_trans': ('[REDACTED_BY_SCRIPT]', 'lsoa_2011', 'q_trans'),
    'oac': ('oac_class', 'output_area_2021', 'oac'),
    'pcc': ('postcode_class', 'postcode_raw', 'pcc') # Merge on raw postcode
}

for name, (key, merge_on, prefix) in merge_configs.items():
    print(f"  - Merging {name}...")
    df_source = pd.read_csv(file_paths[key], dtype=str, encoding='latin1')
    source_key_col = df_source.columns[0]
    df_source.rename(columns={source_key_col: merge_on}, inplace=True)
    df_source = clean_col_names(df_source, prefix)
    df_main = pd.merge(df_main, df_source.drop_duplicates(subset=[merge_on]), on=merge_on, how='left')

# Boundary Data
for level in ['oa', 'lsoa', 'msoa']:
    key_col_21 = f'{level.upper()}21CD'
    merge_on_col = f'{"output_area" if level == "oa" else level}_2021'
    df_bound = pd.read_csv(file_paths[f'{level}_boundaries'], usecols=[key_col_21, 'Shape__Area', 'Shape__Length'], dtype=str, encoding='latin1')
    df_pwc = pd.read_csv(file_paths[f'{level}_pwc'], usecols=[key_col_21, 'x', 'y'], dtype=str, encoding='latin1')
    df_geo = pd.merge(df_bound, df_pwc, on=key_col_21, how='left')
    df_geo = clean_col_names(df_geo, prefix=f'{level}')
    df_geo.rename(columns={f'[REDACTED_BY_SCRIPT]': merge_on_col}, inplace=True)
    df_main = pd.merge(df_main, df_geo.drop_duplicates(subset=[merge_on_col]), on=merge_on_col, how='left')

print("[REDACTED_BY_SCRIPT]")
# Compactness Ratios
for level in ['oa', 'lsoa', 'msoa']:
    area_col, perim_col = f'{level}_shape_area', f'{level}_shape_length'
    if area_col in df_main.columns and perim_col in df_main.columns:
        area = safe_to_numeric(df_main[area_col])
        perimeter = safe_to_numeric(df_main[perim_col]).replace(0, np.nan)
        df_main[f'[REDACTED_BY_SCRIPT]'] = (4 * np.pi * area) / (perimeter**2)

# One-Hot Encoding and Type Conversion
df = df_main.copy()
categorical_cols = df.select_dtypes(include=['object']).columns.drop(['postcode_raw', 'postcode', 'propertytype'], errors='ignore')
cols_to_ohe = [col for col in categorical_cols if df[col].nunique(dropna=False) <= ONE_HOT_ENCODING_THRESHOLD]
numeric_cols = df.columns.difference(categorical_cols).difference(['postcode_raw', 'postcode', 'propertytype'])

if cols_to_ohe:
    df = pd.get_dummies(df, columns=cols_to_ohe, dummy_na=True, prefix=cols_to_ohe)

for col in numeric_cols:
    df[col] = safe_to_numeric(df[col])

print("[REDACTED_BY_SCRIPT]")
df = df.loc[:, ~df.columns.duplicated()] # Drop duplicate columns
df.to_parquet(OUTPUT_FILE, index=False)
print(f"[REDACTED_BY_SCRIPT]")
print(f"[REDACTED_BY_SCRIPT]")