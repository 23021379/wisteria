import pandas as pd
import numpy as np
import os
import re
from scipy.stats import linregress # Added for trend calculation

# --- Configuration ---
MAIN_PROPERTY_FILE = '[REDACTED_BY_SCRIPT]'
POSTCODE_LOOKUP_CSV = 'POSTtoLSOA.csv'
AHAH_DATA_CSV = 'AHAH_v4.csv'
ONS_PIVOTED_DATA_CSV = 'ons_pivoted.csv'
CDRC_DATA_FOLDER = 'CDRC files'
# IMPORTANT: Make sure this file exists in CDRC_DATA_FOLDER and contains PCDS and classif columns
PCC_DATA_FILENAME = '[REDACTED_BY_SCRIPT]' # <--- CONFIRM OR REPLACE THIS FILENAME
FINAL_OUTPUT_CSV = '[REDACTED_BY_SCRIPT]' # Updated output name

POSTCODE_INDEX_COLUMNS_MAP = {
    'pcds': 'PCDS_raw', 'oa21cd': 'oa21cd', 'lsoa21cd': 'lsoa21cd',
    'msoa21cd': 'msoa21cd', 'ladcd': 'ladcd', 'lsoa21nm': 'lsoa21nm',
    'msoa21nm': 'msoa21nm', 'ladnm': 'ladnm', 'oa11cd': 'oa11cd',
    'lsoa11cd': 'lsoa11cd', 'wz11cd': 'wz11cd'
}
PC_LOOKUP_JOIN_KEY_CLEANED = '[REDACTED_BY_SCRIPT]'

# --- Helper Functions ---
def clean_postcode_for_join(pc_str):
    if pd.isna(pc_str): return np.nan
    return str(pc_str).upper().replace(" ", "")

def extract_clean_postcode_from_address(address_str):
    if pd.isna(address_str): return None
    address_string_upper = str(address_str).strip().upper()
    match = re.search(r'[REDACTED_BY_SCRIPT]', address_string_upper)
    if match: return f"[REDACTED_BY_SCRIPT]"
    parts = address_string_upper.split()
    if len(parts) >= 2:
        potential_pc = f"[REDACTED_BY_SCRIPT]"
        if re.fullmatch(r'[REDACTED_BY_SCRIPT]', potential_pc): return potential_pc
        if re.fullmatch(r'[REDACTED_BY_SCRIPT]', parts[-1]): return parts[-1]
    elif len(parts) == 1:
        if re.fullmatch(r'[REDACTED_BY_SCRIPT]', parts[0]): return parts[0]
    return None

def safe_to_numeric(series):
    return pd.to_numeric(series, errors='coerce')

def calculate_trend_slope(df, value_cols_ordered, year_values_ordered):
    trends = []
    if not value_cols_ordered: return [np.nan] * len(df)

    # Check columns exist before processing
    missing_cols = [col for col in value_cols_ordered if col not in df.columns]
    if missing_cols:
         print(f"[REDACTED_BY_SCRIPT]")
         return [np.nan] * len(df)

    # Ensure data is numeric before iterating
    for col in value_cols_ordered: df[col] = safe_to_numeric(df[col])

    for _, row in df[value_cols_ordered].iterrows():
        values = row.values # Already converted to numeric
        valid_idx = ~np.isnan(values) & ~np.isinf(values)
        if sum(valid_idx) < 2:
            trends.append(np.nan)
        else:
            curr_years = np.array(year_values_ordered)[valid_idx]
            curr_values = values[valid_idx]
            if not (np.isnan(curr_years).any() or np.isinf(curr_years).any() or \
                    np.isnan(curr_values).any() or np.isinf(curr_values).any()):
                try:
                    slope, _, _, _, _ = linregress(curr_years, curr_values)
                    trends.append(slope)
                except ValueError as lin_err:
                     print(f"[REDACTED_BY_SCRIPT]")
                     trends.append(np.nan)
            else:
                 print(f"[REDACTED_BY_SCRIPT]")
                 trends.append(np.nan)
    return trends

# --- Interaction Helper Function (Revised) ---
interaction_creation_log = [] # Global log for tracking interactions

def create_interaction(df, col1_name, col2_name, prefix=""):
    """[REDACTED_BY_SCRIPT]"""
    global interaction_creation_log
    if col1_name in df.columns and col2_name in df.columns:
        # Ensure base columns are numeric before interaction
        if not pd.api.types.is_numeric_dtype(df[col1_name]):
             # print(f"[REDACTED_BY_SCRIPT]") # Optional debug
             df[col1_name] = safe_to_numeric(df[col1_name])
        if not pd.api.types.is_numeric_dtype(df[col2_name]):
             # print(f"[REDACTED_BY_SCRIPT]") # Optional debug
             df[col2_name] = safe_to_numeric(df[col2_name])

        # Check again after conversion attempt
        if pd.api.types.is_numeric_dtype(df[col1_name]) and pd.api.types.is_numeric_dtype(df[col2_name]):
            # Create a more robust short name, removing suffixes and simplifying
            c1_short = re.sub(r'[REDACTED_BY_SCRIPT]', '', col1_name)
            c1_short = re.sub(r'^(ah4|chn|bb)_', '', c1_short) # Remove common prefixes
            c1_short = re.sub(r'^classif_', '', c1_short)
            c1_short = re.sub(r'[REDACTED_BY_SCRIPT]', 'BPcat_', c1_short) # Shorten BP
            c1_short = re.sub(r'[^A-Za-z0-9_]+', '', c1_short).strip('_')[:25]

            c2_short = re.sub(r'[REDACTED_BY_SCRIPT]', '', col2_name)
            c2_short = re.sub(r'^(ah4|chn|bb)_', '', c2_short)
            c2_short = re.sub(r'^classif_', '', c2_short)
            c2_short = re.sub(r'[REDACTED_BY_SCRIPT]', 'BPcat_', c2_short)
            c2_short = re.sub(r'[^A-Za-z0-9_]+', '', c2_short).strip('_')[:25]

            interaction_name = f"[REDACTED_BY_SCRIPT]".replace("__", "_")
            
            if len(interaction_name) > 80:
                 interaction_name = interaction_name[:40] + "_X_" + interaction_name[-40:]
            
            df[interaction_name] = df[col1_name] * df[col2_name]
            interaction_creation_log.append(interaction_name)
            return True
        else:
             # print(f"[REDACTED_BY_SCRIPT]") # Optional
             pass
    # else:
    #      print(f"[REDACTED_BY_SCRIPT]'{col1_name}' or '{col2_name}').") # Optional
    return False

# --- 1. Load Main Property Data ---
print(f"[REDACTED_BY_SCRIPT]")
try:
    df_main = pd.read_csv(MAIN_PROPERTY_FILE, low_memory=False)
    if '[REDACTED_BY_SCRIPT]' not in df_main.columns:
        print(f"FATAL: 'original_property_address'[REDACTED_BY_SCRIPT]"); exit()
    df_main['postcode_for_join'] = df_main['[REDACTED_BY_SCRIPT]'].apply(extract_clean_postcode_from_address)
    df_main.dropna(subset=['postcode_for_join'], inplace=True)
    print(f"[REDACTED_BY_SCRIPT]")
except FileNotFoundError: print(f"[REDACTED_BY_SCRIPT]"); exit()
except Exception as e: print(f"[REDACTED_BY_SCRIPT]"); exit()

# --- 2. Load Postcode Lookup ---
print(f"[REDACTED_BY_SCRIPT]")
try:
    pc_lookup_header = pd.read_csv(POSTCODE_LOOKUP_CSV, nrows=0, encoding='latin-1').columns
    actual_cols_pc = [col for col in POSTCODE_INDEX_COLUMNS_MAP.keys() if col in pc_lookup_header]
    if 'pcds' not in actual_cols_pc: print(f"FATAL: 'pcds'[REDACTED_BY_SCRIPT]"); exit()
    print(f"[REDACTED_BY_SCRIPT]") # Debug: show which keys are expected
    df_pc_lookup = pd.read_csv(POSTCODE_LOOKUP_CSV, usecols=actual_cols_pc, encoding='latin-1', low_memory=False)
    rename_map_pc = {k: v for k, v in POSTCODE_INDEX_COLUMNS_MAP.items() if k in actual_cols_pc}
    df_pc_lookup.rename(columns=rename_map_pc, inplace=True)
    df_pc_lookup[PC_LOOKUP_JOIN_KEY_CLEANED] = df_pc_lookup['PCDS_raw'].apply(clean_postcode_for_join)
    df_pc_lookup.drop_duplicates(subset=[PC_LOOKUP_JOIN_KEY_CLEANED], inplace=True)
    cols_to_keep_lookup = [PC_LOOKUP_JOIN_KEY_CLEANED] + [v for k,v in rename_map_pc.items() if k != 'pcds']
    df_pc_lookup = df_pc_lookup[cols_to_keep_lookup]
    print(f"[REDACTED_BY_SCRIPT]") # Debug: show available keys
except FileNotFoundError: print(f"[REDACTED_BY_SCRIPT]"); exit()
except Exception as e: print(f"[REDACTED_BY_SCRIPT]"); exit()

# --- 3. Merge Main Data with Postcode Lookup ---
print("[REDACTED_BY_SCRIPT]")
df_enriched = pd.merge(df_main, df_pc_lookup, left_on='postcode_for_join', right_on=PC_LOOKUP_JOIN_KEY_CLEANED, how='left')
df_enriched.drop(columns=[PC_LOOKUP_JOIN_KEY_CLEANED], inplace=True, errors='ignore')
key_check = POSTCODE_INDEX_COLUMNS_MAP.get('lsoa21cd', 'lsoa21cd')
if key_check in df_enriched.columns: print(f"[REDACTED_BY_SCRIPT]")
print(f"[REDACTED_BY_SCRIPT]")
print(f"[REDACTED_BY_SCRIPT]") # Debug: list columns

# --- 4. Load and Merge AHAH Data ---
print(f"[REDACTED_BY_SCRIPT]")
AHAH_KEY, MAIN_KEY_FOR_AHAH, AHAH_SUFFIX = 'LSOA21CD', 'lsoa21cd', '_ahah'
if MAIN_KEY_FOR_AHAH in df_enriched.columns:
    try:
        df_ahah_raw = pd.read_csv(AHAH_DATA_CSV, encoding='utf-8')
        if AHAH_KEY in df_ahah_raw.columns:
            df_ahah_raw.drop_duplicates(subset=[AHAH_KEY], inplace=True)
            ahah_rename = {col: col + AHAH_SUFFIX for col in df_ahah_raw.columns if col != AHAH_KEY}
            df_ahah_raw.rename(columns=ahah_rename, inplace=True)
            df_enriched = pd.merge(df_enriched, df_ahah_raw, left_on=MAIN_KEY_FOR_AHAH, right_on=AHAH_KEY, how='left')
            df_enriched.drop(columns=[AHAH_KEY], inplace=True, errors='ignore')
            print(f"[REDACTED_BY_SCRIPT]")
        else: print(f"WARNING: Key '{AHAH_KEY}'[REDACTED_BY_SCRIPT]")
    except FileNotFoundError: print(f"[REDACTED_BY_SCRIPT]")
    except Exception as e: print(f"[REDACTED_BY_SCRIPT]")
else: print(f"WARNING: Key '{MAIN_KEY_FOR_AHAH}'[REDACTED_BY_SCRIPT]")

# --- 5. Load and Merge ONS Pivoted Data ---
print(f"[REDACTED_BY_SCRIPT]")
ONS_KEY_OPTIONS, MAIN_KEY_FOR_ONS, ONS_SUFFIX = ['OA21_CODE', 'oa21cd'], 'oa21cd', '_ons'
if MAIN_KEY_FOR_ONS in df_enriched.columns:
    try:
        df_ons = pd.read_csv(ONS_PIVOTED_DATA_CSV, encoding='utf-8')
        oa_key_ons = next((k_opt for k_opt in ONS_KEY_OPTIONS if k_opt in df_ons.columns), None)
        if not oa_key_ons and len(df_ons.columns)>0 and df_ons.columns[0].strip().upper() in [k.upper() for k in ONS_KEY_OPTIONS]:
            df_ons.rename(columns={df_ons.columns[0]: 'OA21_CODE'}, inplace=True); oa_key_ons = 'OA21_CODE'
        
        if oa_key_ons:
            df_ons[oa_key_ons] = df_ons[oa_key_ons].astype(str).str.strip().str.upper()
            df_enriched[MAIN_KEY_FOR_ONS] = df_enriched[MAIN_KEY_FOR_ONS].astype(str).str.strip().str.upper()
            df_ons.drop_duplicates(subset=[oa_key_ons], inplace=True)
            ons_rename = {col: col + ONS_SUFFIX for col in df_ons.columns if col != oa_key_ons}
            df_ons.rename(columns=ons_rename, inplace=True)
            df_enriched = pd.merge(df_enriched, df_ons, left_on=MAIN_KEY_FOR_ONS, right_on=oa_key_ons, how='left')
            df_enriched.drop(columns=[oa_key_ons], inplace=True, errors='ignore')
            print(f"[REDACTED_BY_SCRIPT]")
        else: print(f"WARNING: OA key ({' or '[REDACTED_BY_SCRIPT]")
    except FileNotFoundError: print(f"[REDACTED_BY_SCRIPT]")
    except Exception as e: print(f"[REDACTED_BY_SCRIPT]")
else: print(f"WARNING: Key '{MAIN_KEY_FOR_ONS}'[REDACTED_BY_SCRIPT]")

# --- 6. Load and Merge CDRC Datasets ---
print(f"[REDACTED_BY_SCRIPT]")
cdrc_files_config = [
    {'filename': '[REDACTED_BY_SCRIPT]', 'key_cdrc': 'LA23CD', 'key_main': 'ladcd', 'suffix': '_oac_lad23'},
    {'filename': '[REDACTED_BY_SCRIPT]', 'key_cdrc': 'AREA_CODE', 'key_main': 'lsoa21cd', 'suffix': '_bop_lsoa21'},
    {'filename': '[REDACTED_BY_SCRIPT]', 'key_cdrc': 'oa11cd', 'key_main': 'oa11cd', 'suffix': '_ubb_oa11', 'requires_2011_codes': True},
    {'filename': '[REDACTED_BY_SCRIPT]', 'key_cdrc': 'area', 'key_main': 'lsoa11cd', 'suffix': '_chn_lsoa11'},
    {'filename': 'LSOA veg.csv', 'key_cdrc': 'LSOA21CD', 'key_main': 'lsoa21cd', 'suffix': '_rsvi_lsoa21'},
    {'filename': '[REDACTED_BY_SCRIPT]', 'key_cdrc': 'oa21cd', 'key_main': 'oa21cd', 'suffix': '_oac_oa21'},
    {'filename': PCC_DATA_FILENAME, 'key_cdrc': 'PCDS', 'key_main': 'postcode_for_join', 'suffix': '_pcc', 'clean_cdrc_key': True},
    {'filename': 'postcode to wz.csv', 'key_cdrc': 'oa11cd', 'key_main': 'oa11cd', 'suffix': '_oac11_oa11', 'requires_2011_codes': True, 'encoding': 'latin-1'},
    {'filename': '[REDACTED_BY_SCRIPT]', 'key_cdrc': 'lsoa_cd', 'key_main': 'lsoa21cd', 'suffix': '_lthp_median_lsoa'},
    {'filename': '[REDACTED_BY_SCRIPT]', 'key_cdrc': 'lsoa_cd', 'key_main': 'lsoa21cd', 'suffix': '_lthp_counts_lsoa'},
    {'filename': '[REDACTED_BY_SCRIPT]', 'key_cdrc': 'Workplace Zone Code', 'key_main': 'wz11cd', 'suffix': '_wz_oac11', 'requires_wz11_codes': True}
]
for config in cdrc_files_config:
    file_path = os.path.join(CDRC_DATA_FOLDER, config['filename'])
    key_cdrc, key_main, suffix = config['key_cdrc'], config['key_main'], config['suffix']
    print(f"[REDACTED_BY_SCRIPT]'filename'[REDACTED_BY_SCRIPT]'{key_main}')...")
    
    # **Explicit Key Check BEFORE trying merge**
    if key_main not in df_enriched.columns:
        required_flag_2011 = config.get('requires_2011_codes', False)
        required_flag_wz = config.get('requires_wz11_codes', False)
        missing_key_type = ""
        if required_flag_2011: missing_key_type = "2011 codes ('oa11cd' or 'lsoa11cd')"
        elif required_flag_wz: missing_key_type = "WZ codes ('wz11cd')"
        elif key_main == 'postcode_for_join': missing_key_type = "Cleaned Postcode ('postcode_for_join')"
        else: missing_key_type = f"Geo Key ('{key_main}')"

        print(f"    WARNING: Key '{key_main}'[REDACTED_BY_SCRIPT]")
        if missing_key_type:
            print(f"[REDACTED_BY_SCRIPT]")
        print(f"[REDACTED_BY_SCRIPT]'filename']}.")
        continue
        
    try:
        df_cdrc = pd.read_csv(file_path, low_memory=False, encoding=config.get('encoding', 'utf-8'))
        if key_cdrc not in df_cdrc.columns:
            print(f"    WARNING: Key '{key_cdrc}' not in {config['filename']}. Skipping.")
            continue

        # Handle PCC column check more carefully
        is_pcc_file = config['filename'] == PCC_DATA_FILENAME
        expected_pcc_col_name = f'classif{suffix}'
        
        if is_pcc_file and expected_pcc_col_name not in df_cdrc.columns:
             alt_classif_names = ['classification', 'Classif', 'pcc_classif', 'classif'] # Added 'classif' itself
             found_alt = False
             for alt_name in alt_classif_names:
                 if alt_name in df_cdrc.columns:
                     print(f"[REDACTED_BY_SCRIPT]'{alt_name}' in {config['filename']}. Renaming to '{expected_pcc_col_name}'.")
                     df_cdrc.rename(columns={alt_name: expected_pcc_col_name}, inplace=True)
                     found_alt = True
                     break
             if not found_alt:
                 print(f"[REDACTED_BY_SCRIPT]'{expected_pcc_col_name}'[REDACTED_BY_SCRIPT]'filename'[REDACTED_BY_SCRIPT]")
                 # Do not skip the merge, but the column won't be useful later
                 
        # Apply suffix rename BEFORE cleaning key (if cleaning needed)
        cdrc_rename = {col: col + suffix for col in df_cdrc.columns if col != key_cdrc and col != expected_pcc_col_name} # Avoid renaming the pcc col if it exists
        df_cdrc.rename(columns=cdrc_rename, inplace=True)

        # Clean the key column if specified
        if config.get('clean_cdrc_key', False):
            if key_cdrc in df_cdrc.columns:
                 df_cdrc[key_cdrc] = df_cdrc[key_cdrc].apply(clean_postcode_for_join)
            else:
                 print(f"    WARNING: Key '{key_cdrc}'[REDACTED_BY_SCRIPT]'filename']} before cleaning.")
                 continue # Skip if key isn't there

        df_cdrc.drop_duplicates(subset=[key_cdrc], inplace=True)
        
        df_enriched[key_main] = df_enriched[key_main].astype(str).str.strip()
        df_cdrc[key_cdrc] = df_cdrc[key_cdrc].astype(str).str.strip()
        
        shape_before = df_enriched.shape
        df_enriched = pd.merge(df_enriched, df_cdrc, left_on=key_main, right_on=key_cdrc, how='left', suffixes=('', '_cdrc_dup'))

        # Handle potential duplicate columns from merge if suffixes weren't unique
        dup_cols = [c for c in df_enriched.columns if c.endswith('_cdrc_dup')]
        if dup_cols:
            print(f"[REDACTED_BY_SCRIPT]")
            df_enriched.drop(columns=dup_cols, inplace=True)

        print(f"[REDACTED_BY_SCRIPT]'filename'[REDACTED_BY_SCRIPT]")

        # Drop the original CDRC key only if it's different from the main key and still exists
        if key_cdrc != key_main and key_cdrc in df_enriched.columns:
            df_enriched.drop(columns=[key_cdrc], inplace=True, errors='ignore')

    except FileNotFoundError: print(f"[REDACTED_BY_SCRIPT]")
    except Exception as e: print(f"[REDACTED_BY_SCRIPT]'filename']}: {e}")


# --- Copy before FE for safety ---
df_fe = df_enriched.copy()
print(f"[REDACTED_BY_SCRIPT]")

# --- 7. CDRC Feature Engineering ---
print(f"[REDACTED_BY_SCRIPT]")
interaction_creation_log = [] # Reset log

# --- 7.1 Churn Data ---
churn_suffix_fe = '_chn_lsoa11'
print(f"[REDACTED_BY_SCRIPT]")
try:
    churn_cols_pattern = rf'[REDACTED_BY_SCRIPT]'
    churn_cols = sorted([col for col in df_fe.columns if re.match(churn_cols_pattern, col)])
    if not churn_cols:
         print("[REDACTED_BY_SCRIPT]'lsoa11cd'.")
    else:
        if f'[REDACTED_BY_SCRIPT]' in churn_cols:
            df_fe.drop(columns=[f'[REDACTED_BY_SCRIPT]'], inplace=True, errors='ignore')
            churn_cols = [col for col in churn_cols if col != f'[REDACTED_BY_SCRIPT]']
            
        for col in churn_cols: df_fe[col] = safe_to_numeric(df_fe[col]) # Ensure numeric first
        
        latest_churn_col = f'[REDACTED_BY_SCRIPT]'
        if latest_churn_col in df_fe.columns: df_fe[f'[REDACTED_BY_SCRIPT]'] = df_fe[latest_churn_col]
        
        recent_years_int = list(range(2018, 2023)); recent_churn_cols = [f'[REDACTED_BY_SCRIPT]' for y in recent_years_int if f'[REDACTED_BY_SCRIPT]' in df_fe.columns]
        if len(recent_churn_cols) > 0:
            df_fe[f'[REDACTED_BY_SCRIPT]'] = df_fe[recent_churn_cols].mean(axis=1)
            df_fe[f'[REDACTED_BY_SCRIPT]'] = df_fe[recent_churn_cols].std(axis=1)
            
        trend_churn_cols_ordered = [f'[REDACTED_BY_SCRIPT]' for y in recent_years_int if f'[REDACTED_BY_SCRIPT]' in df_fe.columns]
        trend_years_ordered = [y for y in recent_years_int if f'[REDACTED_BY_SCRIPT]' in df_fe.columns]
        if len(trend_churn_cols_ordered) >= 2:
             df_fe[f'[REDACTED_BY_SCRIPT]'] = calculate_trend_slope(df_fe, trend_churn_cols_ordered, trend_years_ordered)
        else:
             print("[REDACTED_BY_SCRIPT]")
             
        print(f"[REDACTED_BY_SCRIPT]'latest' in c or 'avg' in c or 'std' in c or 'trend' in c)]}")
        # Optional: Drop raw churn cols
        # df_fe.drop(columns=[c for c in churn_cols if c in df_fe.columns], inplace=True, errors='ignore')
except Exception as e: print(f"[REDACTED_BY_SCRIPT]")

# --- 7.2 Broadband Data ---
bb_suffix_fe = '_ubb_oa11'
print(f"[REDACTED_BY_SCRIPT]")
try:
    bb_types, all_bb_cols = ['dow', 'uf', 'sfu'], [c for c in df_fe.columns if bb_suffix_fe in c and c.startswith('bba')]
    if not all_bb_cols:
         print("[REDACTED_BY_SCRIPT]'oa11cd'.")
    else:
        created_bb_features = []
        for col in all_bb_cols: df_fe[col] = safe_to_numeric(df_fe[col])
        for bb_type in bb_types:
            latest_col, earliest_col = f'[REDACTED_BY_SCRIPT]', f'[REDACTED_BY_SCRIPT]'
            latest_feat_name = f'[REDACTED_BY_SCRIPT]'
            trend_feat_name = f'[REDACTED_BY_SCRIPT]'
            if latest_col in df_fe.columns:
                df_fe[latest_feat_name] = df_fe[latest_col]; created_bb_features.append(latest_feat_name)
                if earliest_col in df_fe.columns:
                     df_fe[trend_feat_name] = df_fe[latest_col] - df_fe[earliest_col]; created_bb_features.append(trend_feat_name)
        print(f"[REDACTED_BY_SCRIPT]")
        # Optional: Drop raw bb cols
        # df_fe.drop(columns=[c for c in all_bb_cols if c in df_fe.columns], inplace=True, errors='ignore')
except Exception as e: print(f"[REDACTED_BY_SCRIPT]")

# --- 7.3 Quarterly House Prices & Transactions ---
price_suffix_fe, count_suffix_fe, bop_suffix_fe = '_lthp_median_lsoa', '_lthp_counts_lsoa', '_bop_lsoa21'
print(f"[REDACTED_BY_SCRIPT]")
created_price_tx_features = []
try:
    target_price_col = f'[REDACTED_BY_SCRIPT]'
    if target_price_col in df_fe.columns: df_fe['[REDACTED_BY_SCRIPT]'] = safe_to_numeric(df_fe[target_price_col]); created_price_tx_features.append('[REDACTED_BY_SCRIPT]')
    price_cols = [c for c in df_fe.columns if c.startswith('median_') and c.endswith(price_suffix_fe)]
    if not price_cols: print("[REDACTED_BY_SCRIPT]")
    else:
        for col in price_cols: df_fe[col] = safe_to_numeric(df_fe[col])
        c18q4,c17q4,c15q4 = f'[REDACTED_BY_SCRIPT]',f'[REDACTED_BY_SCRIPT]',f'[REDACTED_BY_SCRIPT]'
        feat_1yr_growth = f'[REDACTED_BY_SCRIPT]'; feat_3yr_growth = f'[REDACTED_BY_SCRIPT]'; feat_volatility = f'[REDACTED_BY_SCRIPT]'
        if c18q4 in df_fe and c17q4 in df_fe and df_fe[c17q4].replace(0,np.nan).notna().any(): df_fe[feat_1yr_growth]=(df_fe[c18q4]-df_fe[c17q4])/df_fe[c17q4].replace(0,np.nan); created_price_tx_features.append(feat_1yr_growth)
        if c18q4 in df_fe and c15q4 in df_fe and df_fe[c15q4].replace(0,np.nan).notna().any(): df_fe[feat_3yr_growth]=(df_fe[c18q4]-df_fe[c15q4])/df_fe[c15q4].replace(0,np.nan); created_price_tx_features.append(feat_3yr_growth)
        cols_18_p = [f'[REDACTED_BY_SCRIPT]' for q in range(1,5) if f'[REDACTED_BY_SCRIPT]' in df_fe.columns]
        if len(cols_18_p)>1: df_fe[feat_volatility]=df_fe[cols_18_p].std(axis=1); created_price_tx_features.append(feat_volatility)
        # Optional: drop raw price cols
        # df_fe.drop(columns=[c for c in price_cols if c != target_price_col and c in df_fe.columns], inplace=True, errors='ignore')

    count_cols = [c for c in df_fe.columns if c.startswith('count_') and c.endswith(count_suffix_fe)]
    if not count_cols: print("[REDACTED_BY_SCRIPT]")
    else:
        for col in count_cols: df_fe[col] = safe_to_numeric(df_fe[col])
        cols_18_c = [f'[REDACTED_BY_SCRIPT]' for q in range(1,5) if f'[REDACTED_BY_SCRIPT]' in df_fe.columns]
        feat_avg_tx = f'[REDACTED_BY_SCRIPT]'; feat_liquidity = f'[REDACTED_BY_SCRIPT]'
        if len(cols_18_c)>0: df_fe[feat_avg_tx]=df_fe[cols_18_c].mean(axis=1); created_price_tx_features.append(feat_avg_tx)
        all_props_col = f'[REDACTED_BY_SCRIPT]'
        if all_props_col in df_fe.columns and feat_avg_tx in df_fe.columns:
            apn=safe_to_numeric(df_fe[all_props_col]); df_fe[feat_liquidity]=df_fe[feat_avg_tx]/apn.replace(0,np.nan); created_price_tx_features.append(feat_liquidity)
            # Replace inf/-inf from division by zero or near-zero
            df_fe[feat_liquidity] = df_fe[feat_liquidity].replace([np.inf, -np.inf], np.nan)
        else: print(f"[REDACTED_BY_SCRIPT]")
        # Optional: drop raw count cols
        # df_fe.drop(columns=[c for c in count_cols if c in df_fe.columns], inplace=True, errors='ignore')
    print(f"[REDACTED_BY_SCRIPT]")
except Exception as e: print(f"[REDACTED_BY_SCRIPT]")

# --- 7.4 Property Age Counts ---
bop_suffix_fe = '_bop_lsoa21'; categorical_age_col_final_name = f'[REDACTED_BY_SCRIPT]'
print(f"[REDACTED_BY_SCRIPT]")
prop_age_prop_cols = []
try:
    bp_cols, all_props_col = [c for c in df_fe.columns if c.startswith('BP_') and c.endswith(bop_suffix_fe)], f'[REDACTED_BY_SCRIPT]'
    if not bp_cols or all_props_col not in df_fe.columns:
         print(f"[REDACTED_BY_SCRIPT]")
    else:
        df_fe[all_props_col]=safe_to_numeric(df_fe[all_props_col])
        for col in bp_cols:
            if col==all_props_col: continue
            df_fe[col]=safe_to_numeric(df_fe[col]);prop_col=col.replace(f'{bop_suffix_fe}',f'[REDACTED_BY_SCRIPT]')
            df_fe[prop_col]=df_fe[col]/df_fe[all_props_col].replace(0,np.nan)
            # Use assignment instead of inplace replace on slice
            df_fe[prop_col] = df_fe[prop_col].replace([np.inf,-np.inf],np.nan);
            prop_age_prop_cols.append(prop_col)

        sel_cat_age_orig=(f'[REDACTED_BY_SCRIPT]' if f'[REDACTED_BY_SCRIPT]' in df_fe.columns else f'[REDACTED_BY_SCRIPT]' if f'[REDACTED_BY_SCRIPT]' in df_fe.columns else None)
        if sel_cat_age_orig: df_fe[categorical_age_col_final_name]=df_fe[sel_cat_age_orig]; print(f"  Using '{sel_cat_age_orig}' for OHE.")
        else: print(f"[REDACTED_BY_SCRIPT]'dwe_medbp' or 'MODE1_TYPE').")
        print(f"[REDACTED_BY_SCRIPT]")
        # Optional: Drop raw BP counts
        # df_fe.drop(columns=[c for c in bp_cols if c!= all_props_col and c in df_fe.columns], inplace=True, errors='ignore')
except Exception as e: print(f"[REDACTED_BY_SCRIPT]")

# --- 7.5 Vegetation Indices (CDRC) ---
veg_suffix_fe = '_rsvi_lsoa21'
print(f"[REDACTED_BY_SCRIPT]")
try:
    to_keep_cdrc_veg=[f'[REDACTED_BY_SCRIPT]',f'[REDACTED_BY_SCRIPT]',f'[REDACTED_BY_SCRIPT]',f'[REDACTED_BY_SCRIPT]']
    to_drop_cdrc_veg=[c for c in df_fe.columns if veg_suffix_fe in c and c not in to_keep_cdrc_veg]
    pres_keep_cdrc_veg=[c for c in to_keep_cdrc_veg if c in df_fe.columns]
    if pres_keep_cdrc_veg:
        for col in pres_keep_cdrc_veg: df_fe[col]=safe_to_numeric(df_fe[col])
        print(f"[REDACTED_BY_SCRIPT]")
        cols_actually_dropped = [c for c in to_drop_cdrc_veg if c in df_fe.columns]
        if cols_actually_dropped:
             df_fe.drop(columns=cols_actually_dropped,inplace=True,errors='ignore')
             print(f"[REDACTED_BY_SCRIPT]")
    else: print(f"[REDACTED_BY_SCRIPT]")
except Exception as e: print(f"[REDACTED_BY_SCRIPT]")

# --- 7.6 OAC Data (CDRC OA21 level) & 7.7 PCC Data ---
oac_suffix_fe = '_oac_oa21'; pcc_suffix_fe = '_pcc'
supergroup_col_fe_oac21, group_col_fe_oac21 = f'[REDACTED_BY_SCRIPT]', f'[REDACTED_BY_SCRIPT]'
pcc_classif_col_fe = f'classif{pcc_suffix_fe}'
print(f"[REDACTED_BY_SCRIPT]")
print(f"[REDACTED_BY_SCRIPT]")
print(f"[REDACTED_BY_SCRIPT]")
print(f"[REDACTED_BY_SCRIPT]")
print(f"[REDACTED_BY_SCRIPT]") # ** Critical Check **

# --- 7.8 One-Hot Encoding for CDRC Categoricals ---
print("[REDACTED_BY_SCRIPT]")
cdrc_ohe_cols = []
if categorical_age_col_final_name in df_fe.columns: cdrc_ohe_cols.append(categorical_age_col_final_name)
if supergroup_col_fe_oac21 in df_fe.columns: cdrc_ohe_cols.append(supergroup_col_fe_oac21)
if group_col_fe_oac21 in df_fe.columns: cdrc_ohe_cols.append(group_col_fe_oac21)
if pcc_classif_col_fe in df_fe.columns: cdrc_ohe_cols.append(pcc_classif_col_fe)
else: print(f"[REDACTED_BY_SCRIPT]'{pcc_classif_col_fe}'[REDACTED_BY_SCRIPT]")
    
cdrc_ohe_cols = sorted(list(set(cdrc_ohe_cols)))
if cdrc_ohe_cols:
    print(f"[REDACTED_BY_SCRIPT]")
    for col in cdrc_ohe_cols: df_fe[col] = df_fe[col].fillna('Missing').astype(str)
    df_fe = pd.get_dummies(df_fe, columns=cdrc_ohe_cols, prefix=cdrc_ohe_cols, dummy_na=False, dtype=int) # Use dtype=int for OHE
    print(f"[REDACTED_BY_SCRIPT]")
else: print("[REDACTED_BY_SCRIPT]")

# --- 7.9 CDRC Feature Interactions ---
print("[REDACTED_BY_SCRIPT]")
interaction_creation_log = [] # Reset log
try:
    # Interaction: Build Period x PCC
    ohe_bp_cols = [c for c in df_fe.columns if c.startswith(f'[REDACTED_BY_SCRIPT]')]
    ohe_pcc_cols = [c for c in df_fe.columns if c.startswith(f'{pcc_classif_col_fe}_')]
    if not ohe_pcc_cols: print("[REDACTED_BY_SCRIPT]")
    elif not ohe_bp_cols: print("[REDACTED_BY_SCRIPT]")
    else:
        for bp_col in ohe_bp_cols:
            for pcc_col in ohe_pcc_cols: create_interaction(df_fe, bp_col, pcc_col, "CDRC_BP_PCC_")
            
    # Interactions: OAC x NDVI/Churn Trend
    ohe_oac_sg_cdrc = [c for c in df_fe.columns if c.startswith(f'[REDACTED_BY_SCRIPT]')]
    ndvi_med_cdrc = f'[REDACTED_BY_SCRIPT]'
    churn_trend_cdrc = f'[REDACTED_BY_SCRIPT]'
    if not ohe_oac_sg_cdrc: print("[REDACTED_BY_SCRIPT]")
    else:
        # Check if base columns for interaction exist before looping
        ndvi_exists = ndvi_med_cdrc in df_fe.columns
        churn_exists = churn_trend_cdrc in df_fe.columns
        if not ndvi_exists: print(f"[REDACTED_BY_SCRIPT]")
        if not churn_exists: print(f"[REDACTED_BY_SCRIPT]")
            
        for sg_col in ohe_oac_sg_cdrc:
            if ndvi_exists: create_interaction(df_fe, sg_col, ndvi_med_cdrc, "CDRC_OAC_NDVI_")
            if churn_exists: create_interaction(df_fe, sg_col, churn_trend_cdrc, "CDRC_OAC_Churn_")
            
    print(f"[REDACTED_BY_SCRIPT]")
except Exception as e: print(f"[REDACTED_BY_SCRIPT]")
print(f"--- CDRC FE Complete ---")

# --- 8. AHAH Feature Engineering ---
print(f"[REDACTED_BY_SCRIPT]")
AHAH_SUFFIX_FE = '_ahah'
interaction_creation_log = [] # Reset log
# --- 8.1 AHAH Column Selection ---
print(f"[REDACTED_BY_SCRIPT]")
try:
    ahah_ind_pct_short = ['blue','dent','gp','hosp','phar','leis','pubs','ffood','gpas','tob','gamb','no2','so2','pm10']
    ahah_dom_pct_short = ['h','g','e','r']
    ahah_keep_pct = [f'[REDACTED_BY_SCRIPT]'] + \
                    [f'[REDACTED_BY_SCRIPT]' for m in ahah_ind_pct_short] + \
                    [f'[REDACTED_BY_SCRIPT]' for d in ahah_dom_pct_short]
    actual_ahah_kept = [c for c in ahah_keep_pct if c in df_fe.columns]
    all_orig_ahah = [c for c in df_fe.columns if c.endswith(AHAH_SUFFIX_FE)]
    ahah_drop = [c for c in all_orig_ahah if c not in actual_ahah_kept]
    if actual_ahah_kept:
        for col in actual_ahah_kept: df_fe[col]=safe_to_numeric(df_fe[col])
        if ahah_drop: df_fe.drop(columns=ahah_drop,inplace=True,errors='ignore')
        print(f"[REDACTED_BY_SCRIPT]")
    else: print("[REDACTED_BY_SCRIPT]")
except Exception as e: print(f"[REDACTED_BY_SCRIPT]")

# --- 8.2 AHAH Hazard Inversion ---
print(f"[REDACTED_BY_SCRIPT]")
try:
    ahah_inv_short = ['blue','dent','gp','hosp','phar','leis','pubs','ffood','tob','gamb','no2','so2','pm10','e'] # 'e' is domain
    inv_count=0
    for shortname in ahah_inv_short:
        col_name = f'[REDACTED_BY_SCRIPT]'
        if col_name in df_fe.columns:
            df_fe[f'{col_name}_inverted'] = 100 - df_fe[col_name]; inv_count+=1
    if inv_count==0: print("[REDACTED_BY_SCRIPT]")
    else: print(f"[REDACTED_BY_SCRIPT]")
except Exception as e: print(f"[REDACTED_BY_SCRIPT]")

# --- 8.3 AHAH Interactions ---
print("[REDACTED_BY_SCRIPT]")
# Note: Interaction helper function defined globally at the start
try:
    # Define required base column names after FE and OHE
    oac_sg_rprof_col = f'[REDACTED_BY_SCRIPT]' 
    oac_sg_murb_col = f'[REDACTED_BY_SCRIPT]'  
    oac_sg_suburb_col = f'[REDACTED_BY_SCRIPT]' 
    oac_g_student_col = f'[REDACTED_BY_SCRIPT]'   
    pcc_urban_core_col = f'{pcc_classif_col_fe}_C'   
    prop_pre_1900_col_cdrc = f'[REDACTED_BY_SCRIPT]'
    bp_recent_ohe_col_cdrc = f'[REDACTED_BY_SCRIPT]' 
    cdrc_churn_trend = f'[REDACTED_BY_SCRIPT]'
    cdrc_price_growth = f'[REDACTED_BY_SCRIPT]'
    cdrc_ndvi_median = f'[REDACTED_BY_SCRIPT]'

    # AHAH x CDRC (OAC)
    create_interaction(df_fe, f'[REDACTED_BY_SCRIPT]', oac_sg_rprof_col, "AHAHxCDRC_OAC_")
    create_interaction(df_fe, f'[REDACTED_BY_SCRIPT]', oac_g_student_col, "AHAHxCDRC_OAC_")
    create_interaction(df_fe, f'[REDACTED_BY_SCRIPT]', oac_sg_murb_col, "AHAHxCDRC_OAC_")
    create_interaction(df_fe, f'[REDACTED_BY_SCRIPT]', oac_sg_suburb_col, "AHAHxCDRC_OAC_")
    
    # AHAH x CDRC (Property)
    create_interaction(df_fe, f'[REDACTED_BY_SCRIPT]', prop_pre_1900_col_cdrc, "AHAHxCDRC_Prop_")
    create_interaction(df_fe, f'[REDACTED_BY_SCRIPT]', pcc_urban_core_col, "AHAHxCDRC_PCC_")
    create_interaction(df_fe, f'[REDACTED_BY_SCRIPT]', bp_recent_ohe_col_cdrc, "AHAHxCDRC_Prop_")
    
    # AHAH x CDRC (Market/Veg)
    create_interaction(df_fe, f'[REDACTED_BY_SCRIPT]', cdrc_churn_trend, "AHAHxCDRC_Dyn_")
    create_interaction(df_fe, f'[REDACTED_BY_SCRIPT]', cdrc_price_growth, "AHAHxCDRC_Dyn_")
    create_interaction(df_fe, f'[REDACTED_BY_SCRIPT]', cdrc_ndvi_median, "AHAHxCDRC_Veg_")
    
    # AHAH x AHAH
    create_interaction(df_fe, f'[REDACTED_BY_SCRIPT]', f'[REDACTED_BY_SCRIPT]', "AHAHxAHAH_")
    create_interaction(df_fe, f'[REDACTED_BY_SCRIPT]', f'[REDACTED_BY_SCRIPT]', "AHAHxAHAH_")
    
    print(f"[REDACTED_BY_SCRIPT]")
except Exception as e: print(f"[REDACTED_BY_SCRIPT]")
print(f"--- AHAH FE Complete ---")

# --- 9. ONS Feature Engineering ---
print(f"[REDACTED_BY_SCRIPT]")
ONS_SUFFIX_FE = '_ons'
interaction_creation_log = [] # Reset log

# --- 9.1 ONS Total Households & Proportions ---
print(f"[REDACTED_BY_SCRIPT]")
ons_cols_for_proportions = []
total_hh_col_ons = f'[REDACTED_BY_SCRIPT]'
use_raw_ons_counts = False # Default to false
prop_created_count = 0
try:
    family_count_cols = [
        f'[REDACTED_BY_SCRIPT]', f'[REDACTED_BY_SCRIPT]',
        f'[REDACTED_BY_SCRIPT]', f'[REDACTED_BY_SCRIPT]',
        f'[REDACTED_BY_SCRIPT]', f'[REDACTED_BY_SCRIPT]',
        f'[REDACTED_BY_SCRIPT]' ]
    present_family_count_cols = [col for col in family_count_cols if col in df_fe.columns]

    if len(present_family_count_cols) > 3:
        print(f"[REDACTED_BY_SCRIPT]")
        for col in present_family_count_cols: df_fe[col] = safe_to_numeric(df_fe[col])
        df_fe[total_hh_col_ons] = df_fe[present_family_count_cols].sum(axis=1)

        if df_fe[total_hh_col_ons].isna().all() or (df_fe[total_hh_col_ons] <= 0).all():
            print(f"  WARNING: Derived '{total_hh_col_ons}'[REDACTED_BY_SCRIPT]"); use_raw_ons_counts = True
        else:
            print(f"  Derived '{total_hh_col_ons}'[REDACTED_BY_SCRIPT]")
            original_ons_count_cols = [c for c in df_fe.columns if c.endswith(ONS_SUFFIX_FE) and c != total_hh_col_ons and 'OA21_Code' not in c]
            prop_suffix = f'[REDACTED_BY_SCRIPT]' # Define suffix for new columns

            # Create proportions in a batch to potentially reduce fragmentation
            new_prop_cols_data = {}
            for col in original_ons_count_cols:
                 prop_col_name = col.replace(ONS_SUFFIX_FE, prop_suffix)
                 if prop_col_name not in df_fe.columns and col != total_hh_col_ons:
                     # Simple heuristic check if it looks like a count column
                     if not ('_prop' in col or '_pct' in col or '_rnk' in col or 'rating' in col.lower() or 'per room' in col.lower() or 'per bedroom' in col.lower()):
                        count_col_numeric = safe_to_numeric(df_fe[col])
                        # Calculate proportion, handle division by zero
                        proportion = count_col_numeric / df_fe[total_hh_col_ons].replace(0, np.nan)
                        proportion = proportion.replace([np.inf, -np.inf], np.nan)
                        new_prop_cols_data[prop_col_name] = proportion
                        ons_cols_for_proportions.append(prop_col_name)
                        prop_created_count += 1

            # Add all new columns at once
            if new_prop_cols_data:
                df_fe = pd.concat([df_fe, pd.DataFrame(new_prop_cols_data, index=df_fe.index)], axis=1)
                print(f"[REDACTED_BY_SCRIPT]")
            else:
                print(f"[REDACTED_BY_SCRIPT]")

    else: print(f"[REDACTED_BY_SCRIPT]"); use_raw_ons_counts = True
except Exception as e: print(f"[REDACTED_BY_SCRIPT]"); use_raw_ons_counts = True

if use_raw_ons_counts:
     print("[REDACTED_BY_SCRIPT]")
     # Base feature selection will look for columns ending _ons instead of _prop_ons
     prop_suffix = ONS_SUFFIX_FE # Use the raw suffix
else:
     prop_suffix = f'[REDACTED_BY_SCRIPT]' # Use the proportion suffix

# --- 9.2 ONS Category Compression & Selection ---
print(f"[REDACTED_BY_SCRIPT]")
final_ons_features = []
aggregated_ons_features = [] # Keep track of newly created aggregated features
try:
    # Disability / Health Aggregation - MORE ROBUST SEARCH
    print(f"[REDACTED_BY_SCRIPT]")
    dis_health_map_robust = {
        f'[REDACTED_BY_SCRIPT]': ('limited a little', prop_suffix),
        f'[REDACTED_BY_SCRIPT]': ('limited a lot', prop_suffix),
        f'[REDACTED_BY_SCRIPT]': ('[REDACTED_BY_SCRIPT]', prop_suffix)
    }
    
    # Generate potential component names (simplified for matching)
    potential_dis_cols = [c for c in df_fe.columns if prop_suffix in c and ('disabled' in c.lower() or 'health condition' in c.lower())]

    for agg_col, (search_phrase, suffix_to_use) in dis_health_map_robust.items():
        # Find columns matching the phrase and suffix
        present_comp = [c for c in potential_dis_cols if search_phrase in c.lower() and c.endswith(suffix_to_use)]
        
        if present_comp and len(present_comp) >= 1: # Found at least one component
             print(f"[REDACTED_BY_SCRIPT]")
             for comp_col in present_comp: df_fe[comp_col] = safe_to_numeric(df_fe[comp_col]) # Ensure numeric
             # Use min_count=1 to handle cases where only '1 person' or '2 or more people' exists
             df_fe[agg_col] = df_fe[present_comp].sum(axis=1, min_count=1)
             final_ons_features.append(agg_col)
             aggregated_ons_features.append(agg_col) # Track this new aggregate
             print(f"[REDACTED_BY_SCRIPT]")
        else:
            print(f"[REDACTED_BY_SCRIPT]'{search_phrase}' with suffix '{suffix_to_use}'[REDACTED_BY_SCRIPT]")

    # Selecting other key features (using prop_suffix identified earlier)
    key_ons_features_bases = [
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        'Mains gas only', 'Electric only', 'No central heating',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '2 people in household', '3 people in household', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '1 bedroom', '2 bedrooms', '3 bedrooms', '4 or more bedrooms',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]'
    ]

    for base in key_ons_features_bases:
         col_name = f"{base}{prop_suffix}"
         if col_name in df_fe.columns: final_ons_features.append(col_name)
         else: print(f"[REDACTED_BY_SCRIPT]'{base}' (expected as '{col_name}') not found.")

    # Create aggregate features (Diversity, Other Heating) using present components with prop_suffix
    ethnic_diversity_components = [f'[REDACTED_BY_SCRIPT]', f'[REDACTED_BY_SCRIPT]', f'[REDACTED_BY_SCRIPT]']
    present_ethnic_div = [c for c in ethnic_diversity_components if c in df_fe.columns]
    if present_ethnic_div: df_fe[f'[REDACTED_BY_SCRIPT]'] = df_fe[present_ethnic_div].sum(axis=1, min_count=1); final_ons_features.append(f'[REDACTED_BY_SCRIPT]'); aggregated_ons_features.append(f'[REDACTED_BY_SCRIPT]')

    ch_other_bases = ['Oil only', 'Solid fuel only', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
                       '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', 'Renewable energy only', 'Wood only', '[REDACTED_BY_SCRIPT]']
    ch_other_components = [f"{base}{prop_suffix}" for base in ch_other_bases]
    present_ch_other = [c for c in ch_other_components if c in df_fe.columns]
    if present_ch_other: df_fe[f'[REDACTED_BY_SCRIPT]'] = df_fe[present_ch_other].sum(axis=1, min_count=1); final_ons_features.append(f'[REDACTED_BY_SCRIPT]'); aggregated_ons_features.append(f'[REDACTED_BY_SCRIPT]')

    final_ons_features = sorted(list(set(final_ons_features))) # Unique sorted list
    print(f"[REDACTED_BY_SCRIPT]")

    # Drop other ONS columns - carefully drop only those not selected or newly aggregated
    all_current_ons_cols = [c for c in df_fe.columns if c.endswith(ONS_SUFFIX_FE) and c != total_hh_col_ons]
    ons_cols_to_drop_final = [c for c in all_current_ons_cols if c not in final_ons_features and c not in aggregated_ons_features] # Keep newly created aggregates
    
    # Also drop original components if they were aggregated and not explicitly selected
    components_to_potentially_drop = set()
    for components in dis_health_map_robust.values(): components_to_potentially_drop.update(components)
    components_to_potentially_drop.update(ethnic_diversity_components)
    components_to_potentially_drop.update(ch_other_components)
    
    for comp in components_to_potentially_drop:
        if comp in df_fe.columns and comp not in final_ons_features: # Only drop if not explicitly kept
            ons_cols_to_drop_final.append(comp)
            
    ons_cols_to_drop_final = list(set(ons_cols_to_drop_final)) # Unique list

    if ons_cols_to_drop_final:
        # Ensure we don't drop columns needed for interactions later
        cols_needed_later = [f'[REDACTED_BY_SCRIPT]'] # Add others if needed
        ons_cols_to_drop_final = [c for c in ons_cols_to_drop_final if c not in cols_needed_later]
        
        cols_actually_dropped_ons = [c for c in ons_cols_to_drop_final if c in df_fe.columns]
        if cols_actually_dropped_ons:
            df_fe.drop(columns=cols_actually_dropped_ons, inplace=True, errors='ignore')
            print(f"[REDACTED_BY_SCRIPT]")
            
    # Clean specific sparse/unhelpful columns explicitly
    for col_base in ['0 people in household', 'Does not apply']:
         for suffix in [ONS_SUFFIX_FE, prop_suffix]:
             col_to_drop = f"{col_base}{suffix}"
             if col_to_drop in df_fe.columns: df_fe.drop(columns=[col_to_drop],inplace=True,errors='ignore')

except Exception as e: print(f"[REDACTED_BY_SCRIPT]")

# --- 9.3 ONS Feature Interactions ---
print("[REDACTED_BY_SCRIPT]")
interaction_creation_log = [] # Reset log
# Use prop_suffix determined earlier (_prop_ons or _ons)
try:
    # ONS x ONS
    create_interaction(df_fe, f'[REDACTED_BY_SCRIPT]', f'[REDACTED_BY_SCRIPT]', "ONSxONS_")
    create_interaction(df_fe, f'[REDACTED_BY_SCRIPT]', f'[REDACTED_BY_SCRIPT]', "ONSxONS_")
    create_interaction(df_fe, f'[REDACTED_BY_SCRIPT]', f'[REDACTED_BY_SCRIPT]', "ONSxONS_")
    create_interaction(df_fe, f'[REDACTED_BY_SCRIPT]', f'[REDACTED_BY_SCRIPT]', "ONSxONS_")

    # ONS x CDRC
    oac_sg_affluent_cdrc = f'[REDACTED_BY_SCRIPT]'
    pcc_inner_city_cdrc = f'{pcc_classif_col_fe}_A'
    prop_pre_1900_cdrc = f'[REDACTED_BY_SCRIPT]'
    churn_trend_cdrc = f'[REDACTED_BY_SCRIPT]'
    price_volatility_cdrc = f'[REDACTED_BY_SCRIPT]'

    create_interaction(df_fe, f'[REDACTED_BY_SCRIPT]', oac_sg_affluent_cdrc, "ONSxCDRC_")
    create_interaction(df_fe, f'[REDACTED_BY_SCRIPT]', prop_pre_1900_cdrc, "ONSxCDRC_")
    create_interaction(df_fe, f'[REDACTED_BY_SCRIPT]', churn_trend_cdrc, "ONSxCDRC_")
    create_interaction(df_fe, f'[REDACTED_BY_SCRIPT]', pcc_inner_city_cdrc, "ONSxCDRC_")
    create_interaction(df_fe, f'[REDACTED_BY_SCRIPT]', price_volatility_cdrc, "ONSxCDRC_") # Use aggregated diversity

    # ONS x AHAH
    ahah_gp_access_inv = f'[REDACTED_BY_SCRIPT]'
    ahah_air_quality_inv = f'[REDACTED_BY_SCRIPT]'
    ahah_greenspace_pct = f'[REDACTED_BY_SCRIPT]'
    ahah_pubs_access_inv = f'[REDACTED_BY_SCRIPT]'
    # Use the aggregated ONS disability feature if it was created
    ons_disability_lot_agg = f'[REDACTED_BY_SCRIPT]'

    create_interaction(df_fe, f'[REDACTED_BY_SCRIPT]', ahah_gp_access_inv, "ONSxAHAH_")
    create_interaction(df_fe, ons_disability_lot_agg, ahah_air_quality_inv, "ONSxAHAH_") # Interaction uses the aggregate
    create_interaction(df_fe, f'[REDACTED_BY_SCRIPT]', ahah_greenspace_pct, "ONSxAHAH_")
    create_interaction(df_fe, f'[REDACTED_BY_SCRIPT]', ahah_pubs_access_inv, "ONSxAHAH_")

    print(f"[REDACTED_BY_SCRIPT]")
except Exception as e: print(f"[REDACTED_BY_SCRIPT]")

print(f"[REDACTED_BY_SCRIPT]")

# --- Mitigate Fragmentation before Final Steps ---
print("[REDACTED_BY_SCRIPT]")
df_enriched = df_fe.copy()
print(f"[REDACTED_BY_SCRIPT]")


# --- 10. Final Clean up ---
# Drop intermediate columns or columns used only for joining if still present
cols_to_drop_final = ['postcode_for_join', total_hh_col_ons] # Add others if needed
cols_exist_to_drop = [c for c in cols_to_drop_final if c in df_enriched.columns]
if cols_exist_to_drop:
     df_enriched.drop(columns=cols_exist_to_drop, inplace=True, errors='ignore')
     print(f"[REDACTED_BY_SCRIPT]")

print(f"[REDACTED_BY_SCRIPT]")
# print(f"[REDACTED_BY_SCRIPT]") # Uncomment for full list
print(f"[REDACTED_BY_SCRIPT]")

# --- 11. Final Save ---
print(f"[REDACTED_BY_SCRIPT]")
try:
    long_cols = [c for c in df_enriched.columns if len(c) > 100]
    if long_cols: print(f"[REDACTED_BY_SCRIPT]")

    df_enriched.to_csv(FINAL_OUTPUT_CSV, index=False)
    print(f"[REDACTED_BY_SCRIPT]")
    print(f"[REDACTED_BY_SCRIPT]")
except Exception as e:
    print(f"[REDACTED_BY_SCRIPT]")