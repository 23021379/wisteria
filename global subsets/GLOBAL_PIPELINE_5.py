import pandas as pd
import numpy as np
import re # For extracting year codes

# --- Configuration ---
# Replace with actual file paths
BASE_DIR = "[REDACTED_BY_SCRIPT]" # Make sure this is correct
POSTCODE_HIERARCHY_FILE = BASE_DIR+"[REDACTED_BY_SCRIPT]"
POSTCODE_GEOG_LOOKUP_FILE = BASE_DIR+"[REDACTED_BY_SCRIPT]"
BROADBAND_DATA_FILE = BASE_DIR+"[REDACTED_BY_SCRIPT]"
FINAL_OUTPUT_FILE = BASE_DIR+"[REDACTED_BY_SCRIPT]" # Changed output filename

CHUNK_SIZE = 50000
EPSILON = 1e-6

# --- Helper function to read CSVs with multiple encodings (no changes) ---
def read_csv_with_encodings(filepath, **kwargs):
    encodings_to_try = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
    low_memory_default = kwargs.pop('low_memory', False)
    for encoding in encodings_to_try:
        try:
            return pd.read_csv(filepath, encoding=encoding, low_memory=low_memory_default, **kwargs)
        except UnicodeDecodeError:
            print(f"[REDACTED_BY_SCRIPT]")
        except FileNotFoundError:
            print(f"[REDACTED_BY_SCRIPT]")
            return None
    print(f"[REDACTED_BY_SCRIPT]")
    return None

# --- Function to select Y1, Y2, Y3 broadband columns (no changes) ---
def get_broadband_year_columns(df_broadband):
    def extract_year_code(col_name):
        match = re.search(r'bba(\d+)_', col_name)
        return int(match.group(1)) if match else -1

    all_dow_cols = sorted([col for col in df_broadband.columns if col.startswith('bba') and col.endswith('_dow')], key=extract_year_code, reverse=True)
    all_uf_cols = sorted([col for col in df_broadband.columns if col.startswith('bba') and col.endswith('_uf')], key=extract_year_code, reverse=True)
    all_sfu_cols = sorted([col for col in df_broadband.columns if col.startswith('bba') and col.endswith('_sfu')], key=extract_year_code, reverse=True)

    min_len = min(len(all_dow_cols), len(all_uf_cols), len(all_sfu_cols))
    
    if min_len == 0:
        print("[REDACTED_BY_SCRIPT]'bbaYYX_'[REDACTED_BY_SCRIPT]")
        return None, [], [], []

    dow_cols = all_dow_cols[:min_len]
    uf_cols = all_uf_cols[:min_len]
    sfu_cols = all_sfu_cols[:min_len]

    col_map = {}
    if min_len >= 1:
        col_map['Dow_Y3'], col_map['UF_Y3'], col_map['SFU_Y3'] = dow_cols[0], uf_cols[0], sfu_cols[0]
    if min_len >= 2:
        col_map['Dow_Y2'], col_map['UF_Y2'], col_map['SFU_Y2'] = dow_cols[1], uf_cols[1], sfu_cols[1]
    elif min_len == 1: 
        col_map['Dow_Y2'], col_map['UF_Y2'], col_map['SFU_Y2'] = col_map['Dow_Y3'], col_map['UF_Y3'], col_map['SFU_Y3']
    if min_len >= 3:
        col_map['Dow_Y1'], col_map['UF_Y1'], col_map['SFU_Y1'] = dow_cols[2], uf_cols[2], sfu_cols[2]
    elif min_len > 0 : 
         col_map['Dow_Y1'], col_map['UF_Y1'], col_map['SFU_Y1'] = col_map['Dow_Y2'], col_map['UF_Y2'], col_map['SFU_Y2']
    
    dow_std_cols_list = dow_cols[:min(min_len, 3)]
    uf_std_cols_list = uf_cols[:min(min_len, 3)]
    sfu_std_cols_list = sfu_cols[:min(min_len, 3)]
            
    return col_map, dow_std_cols_list, uf_std_cols_list, sfu_std_cols_list

# --- Function to calculate feature interactions (no changes other than the UF_Y3_if_SmallUser correction) ---
def calculate_interactions(df, col_map, dow_cols_for_std, uf_cols_for_std, sfu_cols_for_std):
    Dow_Y3 = df[col_map['Dow_Y3']] if 'Dow_Y3' in col_map and col_map['Dow_Y3'] in df else pd.Series(np.nan, index=df.index)
    Dow_Y2 = df[col_map['Dow_Y2']] if 'Dow_Y2' in col_map and col_map['Dow_Y2'] in df else pd.Series(np.nan, index=df.index)
    Dow_Y1 = df[col_map['Dow_Y1']] if 'Dow_Y1' in col_map and col_map['Dow_Y1'] in df else pd.Series(np.nan, index=df.index)

    UF_Y3 = df[col_map['UF_Y3']] if 'UF_Y3' in col_map and col_map['UF_Y3'] in df else pd.Series(np.nan, index=df.index)
    UF_Y2 = df[col_map['UF_Y2']] if 'UF_Y2' in col_map and col_map['UF_Y2'] in df else pd.Series(np.nan, index=df.index)
    UF_Y1 = df[col_map['UF_Y1']] if 'UF_Y1' in col_map and col_map['UF_Y1'] in df else pd.Series(np.nan, index=df.index)

    SFU_Y3 = df[col_map['SFU_Y3']] if 'SFU_Y3' in col_map and col_map['SFU_Y3'] in df else pd.Series(np.nan, index=df.index)
    SFU_Y2 = df[col_map['SFU_Y2']] if 'SFU_Y2' in col_map and col_map['SFU_Y2'] in df else pd.Series(np.nan, index=df.index)
    SFU_Y1 = df[col_map['SFU_Y1']] if 'SFU_Y1' in col_map and col_map['SFU_Y1'] in df else pd.Series(np.nan, index=df.index)
    
    usertype = df['usertype_numeric']
    df_interactions = pd.DataFrame(index=df.index)

    # 1-5: Temporal Changes in Download Speed (Dow)
    df_interactions['Dow_AbsChange_Y3_Y2'] = Dow_Y3 - Dow_Y2
    df_interactions['Dow_RelChange_Y3_Y2'] = (Dow_Y3 - Dow_Y2) / (Dow_Y2 + EPSILON)
    df_interactions['Dow_AbsChange_Y3_Y1'] = Dow_Y3 - Dow_Y1
    df_interactions['Dow_RelChange_Y3_Y1'] = (Dow_Y3 - Dow_Y1) / (Dow_Y1 + EPSILON)
    num_intervals_dow = 0
    if 'Dow_Y3' in col_map and 'Dow_Y2' in col_map and not pd.Series.equals(df.get(col_map['Dow_Y3'], pd.Series(dtype='float64')), df.get(col_map['Dow_Y2'], pd.Series(dtype='float64'))): num_intervals_dow +=1
    if 'Dow_Y2' in col_map and 'Dow_Y1' in col_map and not pd.Series.equals(df.get(col_map['Dow_Y2'], pd.Series(dtype='float64')), df.get(col_map['Dow_Y1'], pd.Series(dtype='float64'))): num_intervals_dow +=1
    df_interactions['[REDACTED_BY_SCRIPT]'] = (Dow_Y3 - Dow_Y1) / (num_intervals_dow if num_intervals_dow > 0 else 1)

    # 6-10: Temporal Changes in Ultrafast Availability (UF)
    df_interactions['UF_AbsChange_Y3_Y2'] = UF_Y3 - UF_Y2
    df_interactions['UF_RelChange_Y3_Y2'] = (UF_Y3 - UF_Y2) / (UF_Y2 + EPSILON)
    df_interactions['UF_AbsChange_Y3_Y1'] = UF_Y3 - UF_Y1
    df_interactions['UF_RelChange_Y3_Y1'] = (UF_Y3 - UF_Y1) / (UF_Y1 + EPSILON)
    num_intervals_uf = 0
    if 'UF_Y3' in col_map and 'UF_Y2' in col_map and not pd.Series.equals(df.get(col_map['UF_Y3'], pd.Series(dtype='float64')), df.get(col_map['UF_Y2'], pd.Series(dtype='float64'))): num_intervals_uf +=1
    if 'UF_Y2' in col_map and 'UF_Y1' in col_map and not pd.Series.equals(df.get(col_map['UF_Y2'], pd.Series(dtype='float64')), df.get(col_map['UF_Y1'], pd.Series(dtype='float64'))): num_intervals_uf +=1
    df_interactions['[REDACTED_BY_SCRIPT]'] = (UF_Y3 - UF_Y1) / (num_intervals_uf if num_intervals_uf > 0 else 1)
    
    # 11-15: Temporal Changes in Superfast Availability (SFU)
    df_interactions['SFU_AbsChange_Y3_Y2'] = SFU_Y3 - SFU_Y2
    df_interactions['SFU_RelChange_Y3_Y2'] = (SFU_Y3 - SFU_Y2) / (SFU_Y2 + EPSILON)
    df_interactions['SFU_AbsChange_Y3_Y1'] = SFU_Y3 - SFU_Y1
    df_interactions['SFU_RelChange_Y3_Y1'] = (SFU_Y3 - SFU_Y1) / (SFU_Y1 + EPSILON)
    num_intervals_sfu = 0
    if 'SFU_Y3' in col_map and 'SFU_Y2' in col_map and not pd.Series.equals(df.get(col_map['SFU_Y3'], pd.Series(dtype='float64')), df.get(col_map['SFU_Y2'], pd.Series(dtype='float64'))): num_intervals_sfu +=1
    if 'SFU_Y2' in col_map and 'SFU_Y1' in col_map and not pd.Series.equals(df.get(col_map['SFU_Y2'], pd.Series(dtype='float64')), df.get(col_map['SFU_Y1'], pd.Series(dtype='float64'))): num_intervals_sfu +=1
    df_interactions['[REDACTED_BY_SCRIPT]'] = (SFU_Y3 - SFU_Y1) / (num_intervals_sfu if num_intervals_sfu > 0 else 1)

    df_interactions['UF_to_SFU_Ratio_Y3'] = UF_Y3 / (SFU_Y3 + EPSILON)
    df_interactions['Dow_per_UF_Point_Y3'] = Dow_Y3 / (UF_Y3 + EPSILON)
    df_interactions['[REDACTED_BY_SCRIPT]'] = Dow_Y3 / (SFU_Y3 + EPSILON)
    df_interactions['SFU_UF_Gap_Y3'] = SFU_Y3 - UF_Y3
    df_interactions['[REDACTED_BY_SCRIPT]'] = Dow_Y3 * UF_Y3
    df_interactions['Dow_Y3_if_LargeUser'] = Dow_Y3 * usertype
    df_interactions['UF_Y3_if_LargeUser'] = UF_Y3 * usertype
    df_interactions['SFU_Y3_if_LargeUser'] = SFU_Y3 * usertype
    df_interactions['Dow_Y3_if_SmallUser'] = Dow_Y3 * (1 - usertype)
    df_interactions['UF_Y3_if_SmallUser'] = UF_Y3 * (1 - usertype)
    df_interactions['[REDACTED_BY_SCRIPT]'] = (Dow_Y3 - Dow_Y2) - (Dow_Y2 - Dow_Y1)
    df_interactions['[REDACTED_BY_SCRIPT]'] = (UF_Y3 - UF_Y2) - (UF_Y2 - UF_Y1)
    df_interactions['[REDACTED_BY_SCRIPT]'] = (SFU_Y3 - SFU_Y2) - (SFU_Y2 - SFU_Y1)

    valid_dow_cols = [col for col in dow_cols_for_std if col in df.columns]
    if valid_dow_cols and len(valid_dow_cols) > 1: df_interactions['Dow_StdDev_AllYears'] = df[valid_dow_cols].std(axis=1, skipna=True)
    else: df_interactions['Dow_StdDev_AllYears'] = np.nan
    valid_uf_cols = [col for col in uf_cols_for_std if col in df.columns]
    if valid_uf_cols and len(valid_uf_cols) > 1: df_interactions['UF_StdDev_AllYears'] = df[valid_uf_cols].std(axis=1, skipna=True)
    else: df_interactions['UF_StdDev_AllYears'] = np.nan
    valid_sfu_cols = [col for col in sfu_cols_for_std if col in df.columns]
    if valid_sfu_cols and len(valid_sfu_cols) > 1: df_interactions['SFU_StdDev_AllYears'] = df[valid_sfu_cols].std(axis=1, skipna=True)
    else: df_interactions['SFU_StdDev_AllYears'] = np.nan

    rel_uf_growth_y3_y1 = (UF_Y3 - UF_Y1) / (UF_Y1 + EPSILON)
    rel_sfu_growth_y3_y1 = (SFU_Y3 - SFU_Y1) / (SFU_Y1 + EPSILON)
    df_interactions['[REDACTED_BY_SCRIPT]'] = rel_uf_growth_y3_y1 / (rel_sfu_growth_y3_y1 + EPSILON)
    df_interactions['[REDACTED_BY_SCRIPT]'] = (UF_Y3 + SFU_Y3) / 2
    df_interactions['[REDACTED_BY_SCRIPT]'] = (Dow_Y3 / 10) / (UF_Y3 + EPSILON) 
    df_interactions['[REDACTED_BY_SCRIPT]'] = UF_Y3 / (UF_Y1 + EPSILON)
    df_interactions['[REDACTED_BY_SCRIPT]'] = SFU_Y3 / (SFU_Y1 + EPSILON)
    recent_uf_growth_rel = (UF_Y3 - UF_Y2) / (UF_Y2 + EPSILON)
    df_interactions['[REDACTED_BY_SCRIPT]'] = Dow_Y3 * recent_uf_growth_rel
    df_interactions['[REDACTED_BY_SCRIPT]'] = (UF_Y3 - UF_Y1) / ((SFU_Y3 - SFU_Y1) + EPSILON)
    uf_growth_y3_y2 = UF_Y3 - UF_Y2
    df_interactions['[REDACTED_BY_SCRIPT]'] = np.select(
        [UF_Y3.ge(100 - EPSILON), uf_growth_y3_y2.le(EPSILON)], 
        [0, 999], 
        default=(100 - UF_Y3) / uf_growth_y3_y2 
    )
    df_interactions['[REDACTED_BY_SCRIPT]'] = (Dow_Y3 / 1000 * 0.5) + (UF_Y3 / 100 * 0.5)
    return df_interactions

# --- Main processing logic ---
def main():
    print("[REDACTED_BY_SCRIPT]")

    # 1. Load postcode -> lsoa11cd AND oa11cd mapping
    print(f"[REDACTED_BY_SCRIPT]")
    # Ensure oa11cd is loaded if it exists in the file, feature_meanings.txt says it does for postcode to wz.csv
    df_pcd_geog_map = read_csv_with_encodings(POSTCODE_GEOG_LOOKUP_FILE, usecols=['pcds', 'lsoa11cd', 'oa11cd'], low_memory=False)
    if df_pcd_geog_map is None: return
    
    df_pcd_geog_map.dropna(subset=['pcds', 'lsoa11cd', 'oa11cd'], inplace=True) # Drop if any of these essential keys are missing
    df_pcd_geog_map['pcds'] = df_pcd_geog_map['pcds'].astype(str).str.upper().str.replace(r'\s+', '', regex=True)
    df_pcd_geog_map['lsoa11cd'] = df_pcd_geog_map['lsoa11cd'].astype(str).str.strip()
    df_pcd_geog_map['oa11cd_from_lookup'] = df_pcd_geog_map['oa11cd'].astype(str).str.strip() # This will be the key for broadband merge
    df_pcd_geog_map.drop(columns=['oa11cd'], inplace=True) # Drop original oa11cd to avoid confusion if broadband file also has 'oa11cd'
    print(f"[REDACTED_BY_SCRIPT]")
    print(f"[REDACTED_BY_SCRIPT]'oa11cd_from_lookup'].unique()[:5]}")
    print(f"[REDACTED_BY_SCRIPT]'lsoa11cd'].unique()[:5]}")


    # 2. Load and pre-process broadband data
    print(f"[REDACTED_BY_SCRIPT]")
    df_broadband = read_csv_with_encodings(BROADBAND_DATA_FILE, low_memory=False)
    if df_broadband is None: return

    if df_broadband.columns[0].startswith('\ufeff'):
        df_broadband.rename(columns={df_broadband.columns[0]: df_broadband.columns[0].replace('\ufeff', '')}, inplace=True)
    
    # This is THE KEY for broadband data, expected to be OA11CD based on column name and feature_meanings.txt
    BROADBAND_DATA_JOIN_KEY = 'oa11cd' 
    if BROADBAND_DATA_JOIN_KEY not in df_broadband.columns:
        print(f"[REDACTED_BY_SCRIPT]'{BROADBAND_DATA_JOIN_KEY}'[REDACTED_BY_SCRIPT]")
        return
        
    df_broadband[BROADBAND_DATA_JOIN_KEY] = df_broadband[BROADBAND_DATA_JOIN_KEY].astype(str).str.strip() # Clean key
    df_broadband.drop_duplicates(subset=[BROADBAND_DATA_JOIN_KEY], inplace=True)
    print(f"[REDACTED_BY_SCRIPT]'{BROADBAND_DATA_JOIN_KEY}'[REDACTED_BY_SCRIPT]") # Crucial Debug Print
    
    broadband_col_map, dow_std_cols, uf_std_cols, sfu_std_cols = get_broadband_year_columns(df_broadband)
    if broadband_col_map is None: return
    print(f"[REDACTED_BY_SCRIPT]")

    columns_to_keep_broadband = [BROADBAND_DATA_JOIN_KEY] + [val for val in broadband_col_map.values() if val is not None]
    columns_to_keep_broadband.extend(d for d in dow_std_cols if d not in columns_to_keep_broadband and d is not None)
    columns_to_keep_broadband.extend(u for u in uf_std_cols if u not in columns_to_keep_broadband and u is not None)
    columns_to_keep_broadband.extend(s for s in sfu_std_cols if s not in columns_to_keep_broadband and s is not None)
    columns_to_keep_broadband = list(set(columns_to_keep_broadband))
    
    df_broadband_subset = df_broadband[columns_to_keep_broadband].copy()
    for col_name_key in ['Dow_Y3', 'Dow_Y2', 'Dow_Y1', 'UF_Y3', 'UF_Y2', 'UF_Y1', 'SFU_Y3', 'SFU_Y2', 'SFU_Y1']:
        if col_name_key in broadband_col_map and broadband_col_map[col_name_key] in df_broadband_subset.columns:
            actual_col_name = broadband_col_map[col_name_key]
            df_broadband_subset[actual_col_name] = pd.to_numeric(df_broadband_subset[actual_col_name], errors='coerce')

    print(f"[REDACTED_BY_SCRIPT]")
    chunk_num = 0
    all_processed_chunks = [] 
    
    pcd_chunk_iter = read_csv_with_encodings(
        POSTCODE_HIERARCHY_FILE, 
        usecols=['pcds', 'usertype'], 
        chunksize=CHUNK_SIZE,
        low_memory=False
    )
    if pcd_chunk_iter is None: return

    for df_pcd_chunk in pcd_chunk_iter:
        chunk_num += 1
        print(f"[REDACTED_BY_SCRIPT]")
        df_pcd_chunk['pcds'] = df_pcd_chunk['pcds'].astype(str).str.upper().str.replace(r'\s+', '', regex=True)

        # Map '1' to 1 (Large User) and '0' to 0 (Small User).
        # Everything else becomes NaN, which you can then fill.
        df_pcd_chunk['usertype_numeric'] = df_pcd_chunk['usertype'].map({1: 1, 0: 0})
        # Fill any non-0/1 values with 0 (default to Small User)
        df_pcd_chunk['usertype_numeric'].fillna(0, inplace=True)
        df_pcd_chunk['usertype_numeric'] = df_pcd_chunk['usertype_numeric'].astype(int)

        # Merge with Postcode Geography mapping (to get oa11cd_from_lookup and lsoa11cd)
        df_merged_chunk_intermediate = pd.merge(df_pcd_chunk, df_pcd_geog_map, on='pcds', how='left', sort=False)
        
        # Merge with broadband data using oa11cd_from_lookup and BROADBAND_DATA_JOIN_KEY (which is 'oa11cd' from broadband file)
        df_merged_chunk = pd.merge(df_merged_chunk_intermediate, df_broadband_subset, 
                                   left_on='oa11cd_from_lookup', right_on=BROADBAND_DATA_JOIN_KEY, how='left', sort=False)
        
        if BROADBAND_DATA_JOIN_KEY != 'oa11cd_from_lookup' and BROADBAND_DATA_JOIN_KEY in df_merged_chunk.columns:
             # This might happen if BROADBAND_DATA_JOIN_KEY was something other than 'oa11cd' and we want to avoid duplicate geo code columns
            df_merged_chunk.drop(columns=[BROADBAND_DATA_JOIN_KEY], inplace=True)
        
        if broadband_col_map.get('Dow_Y3') and broadband_col_map['Dow_Y3'] in df_merged_chunk.columns:
            non_null_dow_y3 = df_merged_chunk[broadband_col_map['Dow_Y3']].notna().sum()
            print(f"[REDACTED_BY_SCRIPT]")
            if non_null_dow_y3 == 0 and len(df_merged_chunk) > 0:
                 print(f"[REDACTED_BY_SCRIPT]'oa11cd_from_lookup'[REDACTED_BY_SCRIPT]'oa11cd_from_lookup'].unique()[:5]}")
        
        df_interactions = calculate_interactions(df_merged_chunk, broadband_col_map, dow_std_cols, uf_std_cols, sfu_std_cols)
        
        # Keep lsoa11cd for context, and oa11cd_from_lookup as it was the join key to broadband data
        base_cols_to_keep = ['pcds', 'lsoa11cd', 'oa11cd_from_lookup', 'usertype_numeric'] 
        
        original_broadband_cols_in_map = [broadband_col_map[col_key] for col_key in ['Dow_Y3', 'UF_Y3', 'SFU_Y3', 'Dow_Y2', 'UF_Y2', 'SFU_Y2', 'Dow_Y1', 'UF_Y1', 'SFU_Y1'] 
                                          if col_key in broadband_col_map and broadband_col_map[col_key] in df_merged_chunk.columns and broadband_col_map[col_key] is not None]
        
        df_final_chunk_cols = base_cols_to_keep[:]
        present_original_broadband_cols = [col for col in original_broadband_cols_in_map if col not in df_final_chunk_cols and col in df_merged_chunk.columns]
        df_final_chunk_cols.extend(present_original_broadband_cols)

        df_final_chunk = pd.concat(
            [df_merged_chunk[df_final_chunk_cols].reset_index(drop=True), 
             df_interactions.reset_index(drop=True)], 
            axis=1
        )
        
        all_processed_chunks.append(df_final_chunk)
        print(f"[REDACTED_BY_SCRIPT]")

    if not all_processed_chunks:
        print("[REDACTED_BY_SCRIPT]")
        return

    print("[REDACTED_BY_SCRIPT]")
    df_full_subset = pd.concat(all_processed_chunks, ignore_index=True)
    
    # Final check for column order based on user's sample output, if desired
    # This is complex to perfectly match dynamically if not all years are present etc.
    # For now, it includes base, original broadband, and then interactions.
    
    print(f"[REDACTED_BY_SCRIPT]")
    df_full_subset.to_csv(FINAL_OUTPUT_FILE, index=False)
    print(f"[REDACTED_BY_SCRIPT]")

    print("[REDACTED_BY_SCRIPT]")

if __name__ == '__main__':
    main()