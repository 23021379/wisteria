import pandas as pd
import numpy as np
import os

# --- Configuration: Define file paths ---
# !!! USER: Update these paths to your actual file locations !!!
BASE_DATA_PATH = "[REDACTED_BY_SCRIPT]" # Or your specific data directory
CDRC_PATH = os.path.join(BASE_DATA_PATH, "CDRC files") # If you have a specific CDRC directory
# Files for Subset 3
PCD_OA_LSOA_MSOA_LAD_LU_FILE = os.path.join(BASE_DATA_PATH, "[REDACTED_BY_SCRIPT]")
POSTCODE_TO_WZ_FILE = os.path.join(CDRC_PATH, "postcode to wz.csv")
POSTCODE_PRICE_ANALYSIS_FILE = os.path.join(BASE_DATA_PATH, "[REDACTED_BY_SCRIPT]")
ONS_PIVOTED_FILE = os.path.join(BASE_DATA_PATH, "ons_pivoted.csv")
HOUSE_AGES_LSOA_FILE = os.path.join(CDRC_PATH, "[REDACTED_BY_SCRIPT]")
QUARTERLY_PRICES_LSOA_FILE = os.path.join(CDRC_PATH, "[REDACTED_BY_SCRIPT]")
QUARTERLY_TRANSACTIONS_LSOA_FILE = os.path.join(CDRC_PATH, "[REDACTED_BY_SCRIPT]")
CHURN_LSOA_FILE = os.path.join(CDRC_PATH, "[REDACTED_BY_SCRIPT]")
LSOA_BOUNDARIES_FILE = os.path.join(BASE_DATA_PATH, "[REDACTED_BY_SCRIPT]") # For LSOA_Area_Ha
CHUNK_SIZE = 50000

LSOA_RECENT_PERIOD_SUFFIX = '18Q4' 
LSOA_5Y_AGO_PERIOD_SUFFIX = '13Q4' 
PC_SALES_CURRENT_YEAR = '2022' 
PC_SALES_PREVIOUS_YEAR = str(int(PC_SALES_CURRENT_YEAR) - 1)

def read_csv_with_encoding(file_path, usecols=None, dtype=None, low_memory=True):
    encodings_to_try = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
    if not os.path.exists(file_path):
        print(f"[REDACTED_BY_SCRIPT]")
        return None
    for encoding in encodings_to_try:
        try:
            df = pd.read_csv(file_path, encoding=encoding, usecols=usecols, dtype=dtype, low_memory=low_memory)
            if df.columns.size > 0: df.columns = df.columns.str.replace('\ufeff', '', regex=False)
            print(f"[REDACTED_BY_SCRIPT]")
            return df
        except UnicodeDecodeError: print(f"[REDACTED_BY_SCRIPT]")
        except Exception as e:
            print(f"[REDACTED_BY_SCRIPT]")
            return None 
    print(f"[REDACTED_BY_SCRIPT]")
    return None

def detect_file_encoding_for_iteration(file_path):
    encodings_to_try = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
    if not os.path.exists(file_path):
        print(f"[REDACTED_BY_SCRIPT]")
        return None
    for encoding in encodings_to_try:
        try:
            pd.read_csv(file_path, encoding=encoding, nrows=1)
            print(f"[REDACTED_BY_SCRIPT]")
            return encoding
        except (UnicodeDecodeError, StopIteration): continue
        except Exception as e:
            print(f"[REDACTED_BY_SCRIPT]")
            continue
    print(f"[REDACTED_BY_SCRIPT]")
    return None

print("[REDACTED_BY_SCRIPT]")
ons_housing_columns_to_try = [
    'Output Areas Code','[REDACTED_BY_SCRIPT]','[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]','[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]','[REDACTED_BY_SCRIPT]',
    'number_bedrooms_1_bedroom', 'number_bedrooms_2_bedrooms', 'number_bedrooms_3_bedrooms','number_bedrooms_4_or_more_bedrooms', 
    'number_rooms_1_room', 'number_rooms_2_rooms', 'number_rooms_3_rooms', 'number_rooms_4_rooms','number_rooms_5_rooms', 'number_rooms_6_rooms', 'number_rooms_7_rooms', 'number_rooms_8_or_more_rooms',
    '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]','[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]','[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]','[REDACTED_BY_SCRIPT]',
    'ownership_Lives_here_rent_free']
df_ons_pivoted_full = read_csv_with_encoding(ONS_PIVOTED_FILE, low_memory=False)
if df_ons_pivoted_full is not None:
    actual_ons_cols = [col for col in ons_housing_columns_to_try if col in df_ons_pivoted_full.columns]
    if 'Output Areas Code' in df_ons_pivoted_full.columns and 'Output Areas Code' not in actual_ons_cols:
        actual_ons_cols.insert(0, 'Output Areas Code')
    df_ons_pivoted_housing = df_ons_pivoted_full[actual_ons_cols].copy()
    df_ons_pivoted_housing.rename(columns={'Output Areas Code': 'oa21cd'}, inplace=True)
    for col in df_ons_pivoted_housing.columns:
        if col != 'oa21cd': df_ons_pivoted_housing[col] = pd.to_numeric(df_ons_pivoted_housing[col], errors='coerce').fillna(0)
else: df_ons_pivoted_housing = pd.DataFrame(columns=['oa21cd'] + [c for c in ons_housing_columns_to_try if c != 'Output Areas Code'])

print("[REDACTED_BY_SCRIPT]")

def standardize_lsoa_key(df, possible_keys, standard_key='lsoa_join_key'):
    """[REDACTED_BY_SCRIPT]"""
    if df is None or df.empty:
        return pd.DataFrame(columns=[standard_key])
    
    found_key = next((key for key in possible_keys if key in df.columns), None)
    
    if found_key:
        df.rename(columns={found_key: standard_key}, inplace=True)
        return df[[col for col in df.columns if col == standard_key or col not in possible_keys]]
    else:
        print(f"[REDACTED_BY_SCRIPT]")
        return pd.DataFrame(columns=[standard_key])

# Standardize House Ages
df_house_ages_raw = read_csv_with_encoding(HOUSE_AGES_LSOA_FILE)
if df_house_ages_raw is not None:
    numeric_cols = [col for col in df_house_ages_raw.columns if col not in ['AREA_CODE', 'AREA_NAME', 'MODE1_TYPE', 'MODE2_TYPE']]
    for col in numeric_cols: df_house_ages_raw[col] = pd.to_numeric(df_house_ages_raw[col], errors='coerce')
df_house_ages = standardize_lsoa_key(df_house_ages_raw, ['AREA_CODE', 'lsoa11cd', 'lsoa21cd'])

# Standardize Quarterly Prices
df_quarterly_prices_raw = read_csv_with_encoding(QUARTERLY_PRICES_LSOA_FILE)
if df_quarterly_prices_raw is not None:
    lsoa_price_col_recent = f'[REDACTED_BY_SCRIPT]'
    lsoa_price_col_5y_ago = f'[REDACTED_BY_SCRIPT]'
    cols_to_load_prices = ['lsoa_cd', lsoa_price_col_recent, lsoa_price_col_5y_ago]
    df_quarterly_prices_raw = df_quarterly_prices_raw[[col for col in cols_to_load_prices if col in df_quarterly_prices_raw.columns]]
    df_quarterly_prices_raw.rename(columns={lsoa_price_col_recent: '[REDACTED_BY_SCRIPT]', lsoa_price_col_5y_ago: '[REDACTED_BY_SCRIPT]'}, inplace=True)
    if '[REDACTED_BY_SCRIPT]' in df_quarterly_prices_raw.columns: df_quarterly_prices_raw['[REDACTED_BY_SCRIPT]'] = pd.to_numeric(df_quarterly_prices_raw['[REDACTED_BY_SCRIPT]'], errors='coerce')
    if '[REDACTED_BY_SCRIPT]' in df_quarterly_prices_raw.columns: df_quarterly_prices_raw['[REDACTED_BY_SCRIPT]'] = pd.to_numeric(df_quarterly_prices_raw['[REDACTED_BY_SCRIPT]'], errors='coerce')
df_quarterly_prices = standardize_lsoa_key(df_quarterly_prices_raw, ['lsoa_cd', 'lsoa11cd', 'lsoa21cd'])

# Standardize Quarterly Transactions
df_quarterly_transactions_raw = read_csv_with_encoding(QUARTERLY_TRANSACTIONS_LSOA_FILE)
if df_quarterly_transactions_raw is not None:
    lsoa_trans_col_recent = f'[REDACTED_BY_SCRIPT]'
    lsoa_trans_col_5y_ago = f'[REDACTED_BY_SCRIPT]'
    cols_to_load_trans = ['lsoa_cd', lsoa_trans_col_recent, lsoa_trans_col_5y_ago]
    df_quarterly_transactions_raw = df_quarterly_transactions_raw[[col for col in cols_to_load_trans if col in df_quarterly_transactions_raw.columns]]
    df_quarterly_transactions_raw.rename(columns={lsoa_trans_col_recent: '[REDACTED_BY_SCRIPT]', lsoa_trans_col_5y_ago: '[REDACTED_BY_SCRIPT]'}, inplace=True)
    if '[REDACTED_BY_SCRIPT]' in df_quarterly_transactions_raw.columns: df_quarterly_transactions_raw['[REDACTED_BY_SCRIPT]'] = pd.to_numeric(df_quarterly_transactions_raw['[REDACTED_BY_SCRIPT]'], errors='coerce')
    if '[REDACTED_BY_SCRIPT]' in df_quarterly_transactions_raw.columns: df_quarterly_transactions_raw['[REDACTED_BY_SCRIPT]'] = pd.to_numeric(df_quarterly_transactions_raw['[REDACTED_BY_SCRIPT]'], errors='coerce')
df_quarterly_transactions = standardize_lsoa_key(df_quarterly_transactions_raw, ['lsoa_cd', 'lsoa11cd', 'lsoa21cd'])

# Standardize Churn Data
df_churn_raw = read_csv_with_encoding(CHURN_LSOA_FILE)
if df_churn_raw is not None:
    LSOA_RECENT_CHURN_COL = 'chn2022'
    if LSOA_RECENT_CHURN_COL not in df_churn_raw.columns:
        churn_cols = sorted([col for col in df_churn_raw.columns if col.startswith('chn') and col[3:].isdigit()], reverse=True)
        if churn_cols: LSOA_RECENT_CHURN_COL = churn_cols[0]
    if 'area' in df_churn_raw.columns and LSOA_RECENT_CHURN_COL in df_churn_raw.columns:
        df_churn_raw = df_churn_raw[['area', LSOA_RECENT_CHURN_COL]].copy()
        df_churn_raw.rename(columns={LSOA_RECENT_CHURN_COL: 'LSOA_Churn_Recent'}, inplace=True)
        df_churn_raw['LSOA_Churn_Recent'] = pd.to_numeric(df_churn_raw['LSOA_Churn_Recent'], errors='coerce')
df_churn = standardize_lsoa_key(df_churn_raw, ['area', 'lsoa11cd', 'lsoa21cd'])

# Standardize LSOA Boundaries
df_lsoa_boundaries_raw = read_csv_with_encoding(LSOA_BOUNDARIES_FILE, usecols=['LSOA21CD', 'Shape__Area'])
if df_lsoa_boundaries_raw is not None:
    df_lsoa_boundaries_raw.rename(columns={'Shape__Area': 'LSOA_Shape_Area_sqm'}, inplace=True)
    df_lsoa_boundaries_raw['LSOA_Area_Ha'] = pd.to_numeric(df_lsoa_boundaries_raw['LSOA_Shape_Area_sqm'], errors='coerce') / 10000
df_lsoa_boundaries = standardize_lsoa_key(df_lsoa_boundaries_raw, ['LSOA21CD', 'lsoa21cd'])

print("[REDACTED_BY_SCRIPT]")
df_prices_pivoted_list = []
df_prices_pivoted = pd.DataFrame(columns=['pcds']) 
VALID_PROPERTY_TYPES = ['D', 'S', 'T', 'F', 'O'] 
if os.path.exists(POSTCODE_PRICE_ANALYSIS_FILE):
    ppa_encoding = detect_file_encoding_for_iteration(POSTCODE_PRICE_ANALYSIS_FILE)
    try:
        try:
            header_df = pd.read_csv(POSTCODE_PRICE_ANALYSIS_FILE, nrows=0, encoding=ppa_encoding)
            if header_df.columns.size > 0: header_df.columns = header_df.columns.str.replace('\ufeff', '', regex=False)
            preview_dtypes = {}; 
            if 'Postcode' in header_df.columns: preview_dtypes['Postcode'] = str
            if 'PropertyType' in header_df.columns: preview_dtypes['PropertyType'] = str
            preview_df = pd.read_csv(POSTCODE_PRICE_ANALYSIS_FILE, nrows=5, encoding=ppa_encoding, dtype=preview_dtypes, low_memory=False)
            if preview_df.columns.size > 0: preview_df.columns = preview_df.columns.str.replace('\ufeff', '', regex=False)
        except Exception as e_preview: print(f"[REDACTED_BY_SCRIPT]"); raise
        year_sale_count_cols = [col for col in preview_df.columns if col.endswith("_sale_count") and col[:-12].replace('_','').isdigit()]
        value_cols_to_pivot = ['[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', 'Sale_Count'] + year_sale_count_cols
        cols_to_read_for_pivot = []
        if 'Postcode' in preview_df.columns: cols_to_read_for_pivot.append('Postcode')
        if 'PropertyType' in preview_df.columns: cols_to_read_for_pivot.append('PropertyType')
        cols_to_read_for_pivot.extend([col for col in value_cols_to_pivot if col in preview_df.columns])
        if 'Postcode' not in cols_to_read_for_pivot or 'PropertyType' not in cols_to_read_for_pivot:
            print(f"CRITICAL: 'Postcode' or 'PropertyType'[REDACTED_BY_SCRIPT]")
        else:
            chunk_dtypes = {}
            if 'Postcode' in cols_to_read_for_pivot: chunk_dtypes['Postcode'] = str
            if 'PropertyType' in cols_to_read_for_pivot: chunk_dtypes['PropertyType'] = str
            for chunk_iter_count, chunk in enumerate(pd.read_csv(
                    POSTCODE_PRICE_ANALYSIS_FILE, chunksize=CHUNK_SIZE, usecols=cols_to_read_for_pivot, 
                    iterator=True, low_memory=False, encoding=ppa_encoding, dtype=chunk_dtypes)):
                if chunk.columns.size > 0: chunk.columns = chunk.columns.str.replace('\ufeff', '', regex=False)
                chunk['PropertyType'].fillna('UNKNOWN', inplace=True)
                chunk.dropna(subset=['Postcode'], inplace=True)
                if chunk.empty: continue
                chunk['Postcode'] = chunk['Postcode'].astype(str)
                chunk['PropertyType'] = chunk['PropertyType'].astype(str).str.strip().str.upper()
                chunk = chunk[chunk['PropertyType'].isin(VALID_PROPERTY_TYPES)]
                if chunk.empty: continue
                current_value_cols_in_chunk = [col for col in value_cols_to_pivot if col in chunk.columns]
                for val_col in current_value_cols_in_chunk: chunk.loc[:, val_col] = pd.to_numeric(chunk[val_col], errors='coerce')
                try:
                    chunk_pivoted = chunk.pivot_table(index='Postcode', columns='PropertyType', values=current_value_cols_in_chunk, aggfunc='mean')
                    chunk_pivoted.columns = ['_'.join(map(str, col)).strip() for col in chunk_pivoted.columns.values]
                    df_prices_pivoted_list.append(chunk_pivoted)
                except Exception as e: print(f"[REDACTED_BY_SCRIPT]")
            if df_prices_pivoted_list:
                # 1. Concatenate all the chunked pivots as before
                df_prices_pivoted_temp = pd.concat(df_prices_pivoted_list)
                
                # 2. Reset the index to turn the 'Postcode' index into a column
                df_prices_pivoted_temp.reset_index(inplace=True)

                # 3. *** THE FIX ***
                #    Group by the postcode and aggregate the data.
                #    Using .mean() will correctly combine data for postcodes split across chunks.
                #    If a postcode appeared only once, its values remain unchanged.
                print("[REDACTED_BY_SCRIPT]")
                df_prices_pivoted = df_prices_pivoted_temp.groupby('Postcode').mean().reset_index()
                print("[REDACTED_BY_SCRIPT]")

                # 4. Rename the column for the final merge
                df_prices_pivoted.rename(columns={'Postcode': 'pcds'}, inplace=True)
            else: print(f"[REDACTED_BY_SCRIPT]")
    except Exception as e: print(f"[REDACTED_BY_SCRIPT]")
else: print(f"[REDACTED_BY_SCRIPT]")
    
print("[REDACTED_BY_SCRIPT]")
df_postcode_to_wz = read_csv_with_encoding(POSTCODE_TO_WZ_FILE, usecols=['pcds', 'oa11cd', 'lsoa11cd'])
if df_postcode_to_wz is None: df_postcode_to_wz = pd.DataFrame(columns=['pcds', 'oa11cd', 'lsoa11cd'])

print("[REDACTED_BY_SCRIPT]")
processed_chunks_list = []
main_lu_cols = ['pcds', 'oa21cd', 'lsoa21cd']
if os.path.exists(PCD_OA_LSOA_MSOA_LAD_LU_FILE):
    main_lu_encoding = detect_file_encoding_for_iteration(PCD_OA_LSOA_MSOA_LAD_LU_FILE)
    try:
        # Pre-merge WZ data once to get the lsoa11cd mapping
        df_postcode_to_wz = read_csv_with_encoding(POSTCODE_TO_WZ_FILE, usecols=['pcds', 'oa11cd', 'lsoa11cd'])
        if df_postcode_to_wz is None: df_postcode_to_wz = pd.DataFrame(columns=['pcds', 'oa11cd', 'lsoa11cd'])

        for i, main_chunk in enumerate(pd.read_csv(
            PCD_OA_LSOA_MSOA_LAD_LU_FILE, chunksize=CHUNK_SIZE, usecols=main_lu_cols, 
            iterator=True, low_memory=False, encoding=main_lu_encoding)):
            print(f"[REDACTED_BY_SCRIPT]")
            if main_chunk.columns.size > 0: main_chunk.columns = main_chunk.columns.str.replace('\ufeff', '', regex=False)
            
            # Merge to get lsoa11cd first
            main_chunk = pd.merge(main_chunk, df_postcode_to_wz, on='pcds', how='left', sort=False)
            
            # --- THE FIX: Create a single, reliable LSOA join key ---
            # Coalesce lsoa11cd and lsoa21cd, preferring lsoa11cd if it exists
            main_chunk['lsoa_join_key'] = main_chunk['lsoa11cd'].fillna(main_chunk['lsoa21cd'])
            
            # FIX: This line was dropping rows where an LSOA key could not be formed.
            # Removing it will preserve all original rows from the input file.
            # main_chunk.dropna(subset=['lsoa_join_key'], inplace=True)

            # Now perform all subsequent merges using the standardized key
            main_chunk = pd.merge(main_chunk, df_prices_pivoted, on='pcds', how='left', sort=False)
            main_chunk = pd.merge(main_chunk, df_ons_pivoted_housing, on='oa21cd', how='left', sort=False)
            if not df_house_ages.empty: main_chunk = pd.merge(main_chunk, df_house_ages, on='lsoa_join_key', how='left', sort=False)
            if not df_quarterly_prices.empty: main_chunk = pd.merge(main_chunk, df_quarterly_prices, on='lsoa_join_key', how='left', sort=False)
            if not df_quarterly_transactions.empty: main_chunk = pd.merge(main_chunk, df_quarterly_transactions, on='lsoa_join_key', how='left', sort=False)
            if not df_churn.empty: main_chunk = pd.merge(main_chunk, df_churn, on='lsoa_join_key', how='left', sort=False)
            if not df_lsoa_boundaries.empty: main_chunk = pd.merge(main_chunk, df_lsoa_boundaries, on='lsoa_join_key', how='left', sort=False)
            
            print(f"[REDACTED_BY_SCRIPT]")

            # Feature Engineering logic follows... (kept as is, it's complex but not the source of duplicates)
            def safe_get_col(df, col_name_variants, default_val: float=0.0):
                if not isinstance(col_name_variants, list): col_name_variants = [col_name_variants]
                for col_variant_name in col_name_variants: 
                    if col_variant_name in df.columns:
                        return pd.to_numeric(df[col_variant_name], errors='coerce').fillna(default_val)
                return pd.Series(default_val, index=df.index, name=col_name_variants[0] if col_name_variants else "missing_col")

            property_type_codes = ['D', 'S', 'T', 'F'] 
            year_for_recent_sales = PC_SALES_CURRENT_YEAR
            prev_year_for_sales = PC_SALES_PREVIOUS_YEAR

            existing_oa_accom_cols = [col for col in ons_housing_columns_to_try if col.startswith('Accommodation_type_') and col in main_chunk.columns] # More dynamic
            if existing_oa_accom_cols: main_chunk['OA_TotalAccom'] = main_chunk[existing_oa_accom_cols].sum(axis=1, skipna=True)
            else: main_chunk['OA_TotalAccom'] = 0
            main_chunk['OA_TotalAccom_Safe'] = main_chunk['OA_TotalAccom'].replace(0, np.nan)

            for pt_code in property_type_codes:
                p_med_price_col = f'[REDACTED_BY_SCRIPT]'
                p_avg_price_col = f'[REDACTED_BY_SCRIPT]'
                term1 = safe_get_col(main_chunk, [p_med_price_col], np.nan) # Default to nan for prices
                term2 = safe_get_col(main_chunk, ['[REDACTED_BY_SCRIPT]'], np.nan) # Default to nan for prices
                main_chunk[f'[REDACTED_BY_SCRIPT]'] = term1 / term2.replace(0, np.nan)
                main_chunk[f'[REDACTED_BY_SCRIPT]'] = (safe_get_col(main_chunk, [p_avg_price_col], np.nan) - term1) / term1.replace(0, np.nan)
            main_chunk['[REDACTED_BY_SCRIPT]'] = safe_get_col(main_chunk, ['[REDACTED_BY_SCRIPT]'], np.nan) / safe_get_col(main_chunk, ['[REDACTED_BY_SCRIPT]'], np.nan).replace(0,np.nan)

            oa_accom_col_map = {'D': '[REDACTED_BY_SCRIPT]','S': '[REDACTED_BY_SCRIPT]',
                                'T': '[REDACTED_BY_SCRIPT]','F': '[REDACTED_BY_SCRIPT]'}
            for pt_code in property_type_codes:
                p_sale_count_recent_col = f'[REDACTED_BY_SCRIPT]'
                oa_accom_col_for_pt = oa_accom_col_map.get(pt_code)
                if oa_accom_col_for_pt:
                     main_chunk[f'[REDACTED_BY_SCRIPT]'] = safe_get_col(main_chunk, [p_sale_count_recent_col]) / safe_get_col(main_chunk, [oa_accom_col_for_pt], np.nan).replace(0,np.nan)
            
            main_chunk['[REDACTED_BY_SCRIPT]'] = (safe_get_col(main_chunk, ['[REDACTED_BY_SCRIPT]']) * 4) / safe_get_col(main_chunk, ['ALL_PROPERTIES'], np.nan).replace(0,np.nan)
            main_chunk['[REDACTED_BY_SCRIPT]'] = safe_get_col(main_chunk, ['LSOA_Churn_Recent']) / main_chunk['[REDACTED_BY_SCRIPT]'].replace(0,np.nan)
            main_chunk['[REDACTED_BY_SCRIPT]'] = safe_get_col(main_chunk, [f'[REDACTED_BY_SCRIPT]']) * safe_get_col(main_chunk, ['LSOA_Churn_Recent'])
            
            bedroom_cols_for_avg = { # Map ONS column name to bedroom count
                'number_bedrooms_1_bedroom': 1, 'number_bedrooms_2_bedrooms': 2, 
                'number_bedrooms_3_bedrooms': 3, 'number_bedrooms_4_or_more_bedrooms': 4.5 # Example for 4+
            }
            weighted_bedrooms_sum = pd.Series(0.0, index=main_chunk.index)
            for col_name, bed_count in bedroom_cols_for_avg.items():
                weighted_bedrooms_sum += safe_get_col(main_chunk, [col_name]) * bed_count
            main_chunk['OA_AvgBedrooms'] = weighted_bedrooms_sum / main_chunk['OA_TotalAccom_Safe']
            main_chunk['[REDACTED_BY_SCRIPT]'] = safe_get_col(main_chunk, ['[REDACTED_BY_SCRIPT]'],np.nan) / main_chunk['OA_AvgBedrooms'].replace(0,np.nan)
            
            main_chunk['[REDACTED_BY_SCRIPT]'] = (safe_get_col(main_chunk, ['BP_PRE_1900']) + safe_get_col(main_chunk, ['BP_1900_1918'])) / safe_get_col(main_chunk, ['ALL_PROPERTIES'], np.nan).replace(0,np.nan)
            bp_post_2000_cols_to_sum = ['BP_2000_2009', 'BP_2010_2014', 'BP_2015_LATER'] 
            existing_bp_post_2000 = [col for col in bp_post_2000_cols_to_sum if col in main_chunk.columns]
            main_chunk['[REDACTED_BY_SCRIPT]'] = main_chunk[existing_bp_post_2000].sum(axis=1, skipna=True) / safe_get_col(main_chunk, ['ALL_PROPERTIES'], np.nan).replace(0,np.nan)
            main_chunk['[REDACTED_BY_SCRIPT]'] = safe_get_col(main_chunk, ['[REDACTED_BY_SCRIPT]'],np.nan) * main_chunk.get('[REDACTED_BY_SCRIPT]', 0) # Use .get for safety
            main_chunk['[REDACTED_BY_SCRIPT]'] = safe_get_col(main_chunk, ['[REDACTED_BY_SCRIPT]'],np.nan) * main_chunk.get('[REDACTED_BY_SCRIPT]', 0)
            main_chunk['[REDACTED_BY_SCRIPT]'] = safe_get_col(main_chunk, [oa_accom_col_map.get('D')]) / main_chunk['OA_TotalAccom_Safe']
            main_chunk['OA_Flat_Stock_Ratio'] = safe_get_col(main_chunk, [oa_accom_col_map.get('F')]) / main_chunk['OA_TotalAccom_Safe']
            main_chunk['[REDACTED_BY_SCRIPT]'] = safe_get_col(main_chunk, ['[REDACTED_BY_SCRIPT]'],np.nan) * main_chunk.get('OA_Flat_Stock_Ratio',0)

            occupancy_minus_col = '[REDACTED_BY_SCRIPT]'
            main_chunk['OA_OvercrowdedRate'] = safe_get_col(main_chunk, [occupancy_minus_col]) / main_chunk['OA_TotalAccom_Safe']
            main_chunk['[REDACTED_BY_SCRIPT]'] = main_chunk.get('OA_OvercrowdedRate',0) * safe_get_col(main_chunk, ['[REDACTED_BY_SCRIPT]'],np.nan)
            owner_occupied_cols_sum = ['[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]']
            existing_owner_cols = [col for col in owner_occupied_cols_sum if col in main_chunk.columns] # Use distinct name
            main_chunk['[REDACTED_BY_SCRIPT]'] = main_chunk[existing_owner_cols].sum(axis=1, skipna=True) / main_chunk['OA_TotalAccom_Safe']
            main_chunk['[REDACTED_BY_SCRIPT]'] = safe_get_col(main_chunk, ['[REDACTED_BY_SCRIPT]'],np.nan) * main_chunk.get('[REDACTED_BY_SCRIPT]',0)
            private_rented_cols_sum = ['[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]']
            existing_private_rented_cols = [col for col in private_rented_cols_sum if col in main_chunk.columns] # Use distinct name
            main_chunk['OA_PrivateRentedRate'] = main_chunk[existing_private_rented_cols].sum(axis=1, skipna=True) / main_chunk['OA_TotalAccom_Safe']
            main_chunk['[REDACTED_BY_SCRIPT]'] = main_chunk.get('OA_PrivateRentedRate',0) * safe_get_col(main_chunk, ['LSOA_Churn_Recent'])
            main_chunk['[REDACTED_BY_SCRIPT]'] = safe_get_col(main_chunk, ['[REDACTED_BY_SCRIPT]'],np.nan) * main_chunk.get('[REDACTED_BY_SCRIPT]',0) 
            main_chunk['[REDACTED_BY_SCRIPT]'] = safe_get_col(main_chunk, ['[REDACTED_BY_SCRIPT]'],np.nan) * safe_get_col(main_chunk, ['MODE1_PC'], np.nan)
            bp_interwar_cols = ['BP_1919_1929', 'BP_1930_1939'] 
            existing_interwar_cols = [col for col in bp_interwar_cols if col in main_chunk.columns]
            main_chunk['[REDACTED_BY_SCRIPT]'] = safe_get_col(main_chunk, ['[REDACTED_BY_SCRIPT]'],np.nan) * (main_chunk[existing_interwar_cols].sum(axis=1, skipna=True) / safe_get_col(main_chunk, ['ALL_PROPERTIES'], np.nan).replace(0,np.nan))
            main_chunk['[REDACTED_BY_SCRIPT]'] = safe_get_col(main_chunk, ['LSOA_Churn_Recent']) * main_chunk.get('[REDACTED_BY_SCRIPT]',0)
            main_chunk['[REDACTED_BY_SCRIPT]'] = safe_get_col(main_chunk, ['number_bedrooms_4_or_more_bedrooms']) / main_chunk['OA_TotalAccom_Safe']
            main_chunk['[REDACTED_BY_SCRIPT]'] = safe_get_col(main_chunk, ['[REDACTED_BY_SCRIPT]'],np.nan) * main_chunk.get('[REDACTED_BY_SCRIPT]',0)
            main_chunk['[REDACTED_BY_SCRIPT]'] = safe_get_col(main_chunk, ['ALL_PROPERTIES']) / safe_get_col(main_chunk, ['LSOA_Area_Ha'], np.nan).replace(0,np.nan)
            main_chunk['[REDACTED_BY_SCRIPT]'] = safe_get_col(main_chunk, ['[REDACTED_BY_SCRIPT]'],np.nan) * main_chunk.get('[REDACTED_BY_SCRIPT]',0)

            price_recent = safe_get_col(main_chunk, ['[REDACTED_BY_SCRIPT]'], np.nan)
            price_5y_ago = safe_get_col(main_chunk, ['[REDACTED_BY_SCRIPT]'], np.nan)
            main_chunk['[REDACTED_BY_SCRIPT]'] = (price_recent - price_5y_ago) / price_5y_ago.replace(0, np.nan)
            trans_recent = safe_get_col(main_chunk, ['[REDACTED_BY_SCRIPT]'], np.nan)
            trans_5y_ago = safe_get_col(main_chunk, ['[REDACTED_BY_SCRIPT]'], np.nan)
            main_chunk['[REDACTED_BY_SCRIPT]'] = (trans_recent - trans_5y_ago) / trans_5y_ago.replace(0, np.nan)
            for pt_code in property_type_codes:
                sales_curr_year_col = f'[REDACTED_BY_SCRIPT]'
                sales_prev_year_col = f'[REDACTED_BY_SCRIPT]'
                sales_curr = safe_get_col(main_chunk, [sales_curr_year_col], np.nan)
                sales_prev = safe_get_col(main_chunk, [sales_prev_year_col], np.nan)
                main_chunk[f'[REDACTED_BY_SCRIPT]'] = (sales_curr - sales_prev) / sales_prev.replace(0, np.nan)
            
            bp_weights = {'[REDACTED_BY_SCRIPT]': 3, '[REDACTED_BY_SCRIPT]': -1 }
            weighted_sum_bp = pd.Series(0.0, index=main_chunk.index)
            for bp_col, weight in bp_weights.items():
                if bp_col in main_chunk: weighted_sum_bp += main_chunk[bp_col].fillna(0) * weight
            main_chunk['[REDACTED_BY_SCRIPT]'] = weighted_sum_bp
            median_of_lsoa_prices = main_chunk['[REDACTED_BY_SCRIPT]'].median()
            main_chunk['[REDACTED_BY_SCRIPT]'] = (
                safe_get_col(main_chunk, ['[REDACTED_BY_SCRIPT]'], np.nan) / median_of_lsoa_prices
            )
            avg_household_income = 35000  # You might want to make this dynamic based on local data
            main_chunk['[REDACTED_BY_SCRIPT]'] = (
                safe_get_col(main_chunk, ['[REDACTED_BY_SCRIPT]'], np.nan) / avg_household_income
            )
            pc_price_growth_sum = pd.Series(0.0, index=main_chunk.index)
            pc_price_growth_count = pd.Series(0, index=main_chunk.index)

            for pt_code in property_type_codes:
                # Get current and historical prices for each property type
                current_price = safe_get_col(main_chunk, [f'[REDACTED_BY_SCRIPT]'], np.nan)
                
                # Since we don'[REDACTED_BY_SCRIPT]'ll use LSOA historical as a base
                # and calculate relative performance
                lsoa_growth_rate = main_chunk.get('[REDACTED_BY_SCRIPT]', 0)
                
                # Estimate historical PC price using LSOA growth rate in reverse
                estimated_historical_pc_price = current_price / (1 + lsoa_growth_rate)
                
                # Calculate PC growth rate for this property type
                pc_growth_this_type = (current_price - estimated_historical_pc_price) / estimated_historical_pc_price.replace(0, np.nan)
                
                # Add to running sum (only for valid values)
                mask = ~pd.isna(pc_growth_this_type)
                pc_price_growth_sum[mask] += pc_growth_this_type[mask].fillna(0)
                pc_price_growth_count[mask] += 1

            # Calculate average PC price growth
            avg_pc_price_growth = pc_price_growth_sum / pc_price_growth_count.replace(0, np.nan)

            # Calculate relative growth (PC growth relative to LSOA growth)
            main_chunk['[REDACTED_BY_SCRIPT]'] = (
                avg_pc_price_growth - main_chunk.get('[REDACTED_BY_SCRIPT]', 0)
            )

            main_chunk['[REDACTED_BY_SCRIPT]'] = safe_get_col(main_chunk, ['ALL_PROPERTIES']) / (safe_get_col(main_chunk, ['[REDACTED_BY_SCRIPT]'], np.nan) * 4).replace(0,np.nan)
            main_chunk['[REDACTED_BY_SCRIPT]'] = safe_get_col(main_chunk, [f'[REDACTED_BY_SCRIPT]']) / safe_get_col(main_chunk, [f'[REDACTED_BY_SCRIPT]'], np.nan).replace(0,np.nan)
            main_chunk['[REDACTED_BY_SCRIPT]'] = (safe_get_col(main_chunk, ['number_bedrooms_1_bedroom']) + safe_get_col(main_chunk, ['number_bedrooms_2_bedrooms'])) / main_chunk['OA_TotalAccom_Safe']
            main_chunk['[REDACTED_BY_SCRIPT]'] = safe_get_col(main_chunk, ['[REDACTED_BY_SCRIPT]'],np.nan) * main_chunk.get('[REDACTED_BY_SCRIPT]',0)
            main_chunk['[REDACTED_BY_SCRIPT]'] = safe_get_col(main_chunk, ['[REDACTED_BY_SCRIPT]'],np.nan) / safe_get_col(main_chunk, ['LSOA_Churn_Recent'], np.nan).replace(0,np.nan)
            main_chunk['[REDACTED_BY_SCRIPT]'] = safe_get_col(main_chunk, ['[REDACTED_BY_SCRIPT]']) / main_chunk['OA_TotalAccom_Safe']
            main_chunk['[REDACTED_BY_SCRIPT]'] = safe_get_col(main_chunk, ['[REDACTED_BY_SCRIPT]'],np.nan) * main_chunk.get('[REDACTED_BY_SCRIPT]',0)
            main_chunk['[REDACTED_BY_SCRIPT]'] = safe_get_col(main_chunk, ['MODE1_VAL'], np.nan) / safe_get_col(main_chunk, ['MODE2_VAL'], np.nan).replace(0,np.nan)
            
            main_chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
            processed_chunks_list.append(main_chunk)
            print(f"[REDACTED_BY_SCRIPT]")
    except FileNotFoundError: print(f"[REDACTED_BY_SCRIPT]")
    except Exception as e:
        print(f"[REDACTED_BY_SCRIPT]")
        import traceback
        traceback.print_exc()

if processed_chunks_list:
    final_df_subset3 = pd.concat(processed_chunks_list, ignore_index=True)
    print("[REDACTED_BY_SCRIPT]")
    print(f"[REDACTED_BY_SCRIPT]")
    try:
        output_filename = "[REDACTED_BY_SCRIPT]"
        final_df_subset3.to_csv(output_filename, index=False)
        print(f"[REDACTED_BY_SCRIPT]")
    except Exception as e: print(f"[REDACTED_BY_SCRIPT]")
else: print("[REDACTED_BY_SCRIPT]")
