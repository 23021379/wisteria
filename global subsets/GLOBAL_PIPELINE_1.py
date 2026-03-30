import pandas as pd
import numpy as np

# --- Configuration (Keep as is) ---
CURRENT_YEAR = 2024
epsilon = 1e-6
CHUNK_SIZE = 50000

# --- Helper Functions (Keep as is) ---
# (Your helper functions are fine, no changes needed here)
def map_pcd_to_category_type(pcd_classif_code):
    if pd.isna(pcd_classif_code): return "Unknown"
    pcd_classif_code = str(pcd_classif_code).upper()
    if pcd_classif_code in ['A', 'C']: return "UrbanDense"
    elif pcd_classif_code in ['B', 'F', 'G']: return "Suburban"
    elif pcd_classif_code in ['D', 'E']: return "Rural"
    return "Other"

def map_oa_to_subgroup_nature(oa_subgroup_name):
    if pd.isna(oa_subgroup_name): return "Unknown"
    name_lower = str(oa_subgroup_name).lower()
    historic_keywords = ["legacy", "traditional", "established", "ageing", "mature", "seniors", "retirement"]
    modern_keywords = ["young", "new", "progression", "starters", "turnover", "burgeoning", "university", "graduate"]
    is_historic = any(keyword in name_lower for keyword in historic_keywords)
    is_modern = any(keyword in name_lower for keyword in modern_keywords)
    if is_historic and not is_modern: return "Historic"
    if is_modern and not is_historic: return "Modern"
    if is_historic and is_modern: return "MixedTempo"
    return "Neutral"

def map_to_wz_type(wz_supergroup_name):
    if pd.isna(wz_supergroup_name): return "Unknown"
    name_lower = str(wz_supergroup_name).lower()
    if "[REDACTED_BY_SCRIPT]" in name_lower: return "Industrial"
    elif "retail" in name_lower: return "Retail"
    elif "[REDACTED_BY_SCRIPT]" in name_lower: return "Commercial"
    elif "servants of society" in name_lower: return "PublicServiceFocus"
    elif "rural" in name_lower: return "Rural_WZ"
    elif "suburban services" in name_lower or "metro suburbs" in name_lower: return "Suburban_WZ_Service"
    return "Other_WZ"

def map_to_oa_density(oa_subgroup_name):
    if pd.isna(oa_subgroup_name): return "Unknown"
    name_lower = str(oa_subgroup_name).lower()
    high_density_keywords = ["flats", "university centric", "urbanite mix", "centrally located", "transient communities", "[REDACTED_BY_SCRIPT]", "[REDACTED_BY_SCRIPT]", "[REDACTED_BY_SCRIPT]"]
    low_density_keywords = ["spacious living", "rural amenity", "rural mix", "[REDACTED_BY_SCRIPT]", "[REDACTED_BY_SCRIPT]", "[REDACTED_BY_SCRIPT]"]
    if any(keyword in name_lower for keyword in high_density_keywords): return "[REDACTED_BY_SCRIPT]"
    if any(keyword in name_lower for keyword in low_density_keywords): return "[REDACTED_BY_SCRIPT]"
    return "[REDACTED_BY_SCRIPT]"

def map_to_pcd_rural_type(pcd_classif_code):
    if pd.isna(pcd_classif_code): return "Unknown"
    pcd_classif_code = str(pcd_classif_code).upper()
    if pcd_classif_code == 'E': return "DeepRural"
    elif pcd_classif_code == 'D': return "RuralResidential"
    return "[REDACTED_BY_SCRIPT]"

def map_to_broad_type(classification_input, level):
    if pd.isna(classification_input): return "Unknown"
    input_str = str(classification_input).lower()
    if level == 'PCD':
        if input_str in ['a', 'c']: return "Urban"
        elif input_str in ['b', 'f', 'g']: return "Suburban"
        elif input_str in ['d', 'e']: return "Rural"
    elif level == 'OA':
        if any(k in input_str for k in ["urban", "city", "student", "multicultural", "university", "cosmopolitan", "professionals in flats", "transient"]): return "Urban"
        elif any(k in input_str for k in ["suburban", "peri-urban", "suburbia", "outer suburbs", "established families"]): return "Suburban"
        elif any(k in input_str for k in ["rural", "village", "farming", "countryside", "spacious living", "agricultural"]): return "Rural"
        elif "legacy" in input_str or "baseline" in input_str: return "Mixed/Other"
    elif level == 'MSOA':
        if input_str == 'urban': return "Urban"
        if input_str == 'rural': return "Rural"
    return "Other"

def map_pcd_to_expected_oa_subgroup(pcd_classification_code):
    if pd.isna(pcd_classification_code): return None
    pcd_code = str(pcd_classification_code).upper()
    if pcd_code == 'A' or pcd_code == 'C': return ['3a1', '3a3', '3c1', '4a2', '4b3', '4c1', '4c2', '6c1', '8a2']
    elif pcd_code == 'B': return ['1a1', '1a2', '1b1', '1b2', '1c1', '1c2', '2a2', '2b1', '2b2', '5a2', '5a3']
    elif pcd_code == 'D' or pcd_code == 'E': return ['1a1', '1a2', '2b1', '2b2']
    elif pcd_code == 'F': return ['2a3', '5b1', '6b2', '7a1', '7b1']
    elif pcd_code == 'G': return ['1b1', '1b2', '2a2', '5a1', '5a2', '5a3', '6a1', '6a2']
    return None

def get_subgroup_code_with_max_index(row, lad_subgroup_cols):
    if not lad_subgroup_cols or not any(col in row.index for col in lad_subgroup_cols): return None
    existing_cols = [col for col in lad_subgroup_cols if col in row.index]
    if not existing_cols: return None
    if row[existing_cols].isnull().all(): return None
    numeric_series = pd.to_numeric(row[existing_cols], errors='coerce')
    if numeric_series.isnull().all(): return None
    return numeric_series.idxmax()

# --- Pre-load Ancillary DataFrames (FIXED) ---
def preload_ancillary_data(file_paths, base_file_problematic_col_idx=13, wz_file_problematic_col_idx=22):
    print("[REDACTED_BY_SCRIPT]")
    ancillary_data = {}

    try:
        # BNG Coordinates
        ancillary_data['df_coords_bng'] = pd.read_csv(file_paths['postcode_lookup_bng'])
        ancillary_data['df_coords_bng'].rename(columns={'postcode': 'pcds_bng_key', 'eastings': 'pcd_eastings', 'northings': 'pcd_northings'}, inplace=True)
        # --- FIX 1: Use robust regex to remove ALL whitespace for standardization ---
        ancillary_data['df_coords_bng']['pcds_bng_key'] = ancillary_data['df_coords_bng']['pcds_bng_key'].astype(str).str.replace(r'\s+', '', regex=True).str.strip().str.upper()
        print(f"[REDACTED_BY_SCRIPT]'df_coords_bng']['pcds_bng_key'[REDACTED_BY_SCRIPT]'df_coords_bng'].empty else 'N/A'}")

        # Lat/Lon Coordinates
        df_coords_latlon_temp = pd.read_csv(file_paths['postcode_lookup_latlon'])
        key_to_use_latlon = 'pcds' if 'pcds' in df_coords_latlon_temp.columns else 'postcode'
        
        df_coords_latlon_temp.rename(columns={key_to_use_latlon: 'pcds_latlon_key', 
                                             'latitude': 'pcd_latitude', 
                                             'longitude': 'pcd_longitude'}, inplace=True)
        # --- FIX 1: Use robust regex here as well ---
        df_coords_latlon_temp['pcds_latlon_key'] = df_coords_latlon_temp['pcds_latlon_key'].astype(str).str.replace(r'\s+', '', regex=True).str.strip().str.upper()
        ancillary_data['df_coords_latlon'] = df_coords_latlon_temp[['pcds_latlon_key', 'pcd_latitude', 'pcd_longitude']]
        print(f"[REDACTED_BY_SCRIPT]'df_coords_latlon']['pcds_latlon_key'[REDACTED_BY_SCRIPT]'df_coords_latlon'].empty else 'N/A'}")

        # --- The rest of your pre-loading is likely fine, as it uses non-postcode keys ---
        # (No changes needed below this line in this function)
        df_oa_class_temp = pd.read_csv(file_paths['oa_classification']).rename(columns={'oa21cd': 'oa21cd', 'supergroup': '[REDACTED_BY_SCRIPT]', 'group': 'OA21_GROUP_CODE', 'subgroup': 'OA21_SUBGROUP_CODE'})
        df_oa_class_temp.rename(columns=lambda x: x.replace('\ufeff', ''), inplace=True)
        df_oa_class_temp['[REDACTED_BY_SCRIPT]'] = df_oa_class_temp['[REDACTED_BY_SCRIPT]'].astype(str).str.strip()
        df_oa_class_temp['OA21_GROUP_CODE'] = df_oa_class_temp['OA21_GROUP_CODE'].astype(str).str.strip()
        df_oa_class_temp['OA21_SUBGROUP_CODE'] = df_oa_class_temp['OA21_SUBGROUP_CODE'].astype(str).str.strip()
        df_class_names = pd.read_csv(file_paths['[REDACTED_BY_SCRIPT]'])
        df_class_names.rename(columns=lambda x: x.replace('\ufeff', ''), inplace=True)
        df_class_names['Classification Code'] = df_class_names['Classification Code'].astype(str).str.strip()
        sg_names = df_class_names[df_class_names['Level Code'] == 'sg'][['Classification Code', 'Classification Name']].rename(columns={'Classification Code': '[REDACTED_BY_SCRIPT]', 'Classification Name': '[REDACTED_BY_SCRIPT]'})
        df_oa_class_temp = pd.merge(df_oa_class_temp, sg_names, on='[REDACTED_BY_SCRIPT]', how='left', sort=False)
        gr_names = df_class_names[df_class_names['Level Code'] == 'g'][['Classification Code', 'Classification Name']].rename(columns={'Classification Code': 'OA21_GROUP_CODE', 'Classification Name': 'OA21_GROUP_NAME'})
        df_oa_class_temp = pd.merge(df_oa_class_temp, gr_names, on='OA21_GROUP_CODE', how='left', sort=False)
        sub_names = df_class_names[df_class_names['Level Code'] == 'subg'][['Classification Code', 'Classification Name']].rename(columns={'Classification Code': 'OA21_SUBGROUP_CODE', 'Classification Name': 'OA21_SUBGROUP_NAME'})
        ancillary_data['df_oa_class_full'] = pd.merge(df_oa_class_temp, sub_names, on='OA21_SUBGROUP_CODE', how='left', sort=False)
        temp_wz_peek = pd.read_csv(file_paths['postcode_to_wz'], nrows=5, encoding='latin1')
        wz_problem_col_name = None
        if wz_file_problematic_col_idx < len(temp_wz_peek.columns):
            wz_problem_col_name = temp_wz_peek.columns[wz_file_problematic_col_idx]
        dtype_spec_wz = {wz_problem_col_name: str} if wz_problem_col_name else None
        try:
            df_pcd_wz_temp = pd.read_csv(file_paths['postcode_to_wz'], encoding='latin1', dtype=dtype_spec_wz)
        except UnicodeDecodeError:
            df_pcd_wz_temp = pd.read_csv(file_paths['postcode_to_wz'], encoding='cp1252', dtype=dtype_spec_wz)
        cols_to_select_pcd_wz = ['pcds', 'oa11cd', 'oac11nm', 'wz11cd', 'wzc11nm', 'lsoa11cd', 'msoa11cd', 'soac11nm', 'ladcd']
        cols_to_select_pcd_wz = [col for col in cols_to_select_pcd_wz if col in df_pcd_wz_temp.columns]
        df_pcd_wz_temp = df_pcd_wz_temp[cols_to_select_pcd_wz].rename(columns={'oac11nm': '[REDACTED_BY_SCRIPT]', 'wzc11nm': '[REDACTED_BY_SCRIPT]', 'soac11nm': 'MS[REDACTED_BY_SCRIPT]', 'ladcd': 'ladcd_from_wz_file'})
        # NOTE: This key also needs robust cleaning if you merge on it later.
        df_pcd_wz_temp['pcds_cleaned'] = df_pcd_wz_temp['pcds'].astype(str).str.replace(r'\s+', '', regex=True).str.strip().str.upper()
        try:
            df_wz_assign_temp = pd.read_csv(file_paths['wz_assignments'], encoding='latin1')
        except UnicodeDecodeError:
            df_wz_assign_temp = pd.read_csv(file_paths['wz_assignments'], encoding='cp1252')
        df_wz_assign_temp.rename(columns={'Workplace Zone Code': 'wz11cd', 'Supergroup Name': '[REDACTED_BY_SCRIPT]', 'Group Name': 'WZ11_GROUP_NAME'}, inplace=True)
        ancillary_data['df_pcd_wz_full'] = pd.merge(df_pcd_wz_temp, df_wz_assign_temp[['wz11cd', '[REDACTED_BY_SCRIPT]', 'WZ11_GROUP_NAME']], on='wz11cd', how='left', sort=False)
        ancillary_data['df_msoa_ruc'] = pd.read_csv(file_paths['msoa_ruc'])[['MSOA21CD', 'RUC21CD', 'RUC21NM', 'Urban_rura']].rename(columns={'MSOA21CD': 'msoa21cd', 'RUC21CD': 'MSOA21_RUC_CODE', 'RUC21NM': 'MSOA21_RUC_NAME', 'Urban_rura': '[REDACTED_BY_SCRIPT]'})
        df_lad_oac_temp = pd.read_csv(file_paths['lad_oac_indexscores'])
        df_lad_oac_temp.rename(columns=lambda x: x.replace('\ufeff', ''), inplace=True)
        sg_rename_map = {str(i): f'LAD_OAC_SG{i}_Index' for i in range(1, 9)}
        df_lad_oac_temp.rename(columns=sg_rename_map, inplace=True)
        original_grp_cols = [col for col in df_lad_oac_temp.columns if col.startswith('GRP')]
        grp_rename_map = {col: f'LAD_OAC_{col}_Index' for col in original_grp_cols}
        df_lad_oac_temp.rename(columns=grp_rename_map, inplace=True)
        original_sub_cols = [col for col in df_lad_oac_temp.columns if col.startswith('SUB')]
        sub_rename_map = {col: f'LAD_OAC_{col}_Index' for col in original_sub_cols}
        df_lad_oac_temp.rename(columns=sub_rename_map, inplace=True)
        ancillary_data['df_lad_oac'] = df_lad_oac_temp

        print("[REDACTED_BY_SCRIPT]")
        return ancillary_data
    except FileNotFoundError as e:
        print(f"[REDACTED_BY_SCRIPT]")
        return None
    except Exception as e:
        print(f"[REDACTED_BY_SCRIPT]")
        return None

# --- Main Data Processing Function (FIXED) ---
def process_chunk(df_chunk, ancillary_data):
    # Keep the original pcds column safe if needed, but we will mostly use the cleaned key.
    df_chunk['pcds_original'] = df_chunk['pcds'] 
    
    # --- FIX 1: Create a robust, standardized key in the chunk ---
    df_chunk['pcds_cleaned_key'] = df_chunk['pcds'].astype(str).str.replace(r'\s+', '', regex=True).str.upper()

    # --- UPDATE: Proactively drop existing coordinate columns BEFORE merging ---
    # This is the core fix. It prevents pandas from creating '_x' and '_y' columns.
    cols_to_drop = ['pcd_eastings', 'pcd_northings', 'pcd_latitude', 'pcd_longitude']
    df_chunk.drop(columns=cols_to_drop, inplace=True, errors='ignore')

    # --- Now the merges will work as expected, creating fresh columns ---
    if 'df_coords_bng' in ancillary_data and not ancillary_data['df_coords_bng'].empty:
        bng_data_to_merge = ancillary_data['df_coords_bng'][['pcds_bng_key', 'pcd_eastings', 'pcd_northings']]
        df_chunk = pd.merge(df_chunk, bng_data_to_merge, 
                            left_on='pcds_cleaned_key', 
                            right_on='pcds_bng_key', 
                            how='left')
        df_chunk.drop(columns=['pcds_bng_key'], inplace=True, errors='ignore')

    if 'df_coords_latlon' in ancillary_data and not ancillary_data['df_coords_latlon'].empty:
        latlon_data_to_merge = ancillary_data['df_coords_latlon'][['pcds_latlon_key', 'pcd_latitude', 'pcd_longitude']]
        df_chunk = pd.merge(df_chunk, latlon_data_to_merge,
                            left_on='pcds_cleaned_key',
                            right_on='pcds_latlon_key',
                            how='left')
        df_chunk.drop(columns=['pcds_latlon_key'], inplace=True, errors='ignore')

    # Add the check prints back to verify the fix
    print(f"[REDACTED_BY_SCRIPT]'pcd_eastings'[REDACTED_BY_SCRIPT]")
    print(f"[REDACTED_BY_SCRIPT]'pcd_latitude'[REDACTED_BY_SCRIPT]")
    
    # --- UPDATE: Implement the intelligent forward/backward fill imputation ---
    # This is the core of the fix. It uses the last known good coordinate to fill gaps.
    coords_to_impute = ['pcd_eastings', 'pcd_northings', 'pcd_latitude', 'pcd_longitude']
    for col in coords_to_impute:
        if col in df_chunk.columns:
            # First, forward-fill to propagate last valid observation forward
            df_chunk[col].fillna(method='ffill', inplace=True)
            # Then, backward-fill to handle any NaNs at the very start of the chunk
            df_chunk[col].fillna(method='bfill', inplace=True)
    
    print(f"[REDACTED_BY_SCRIPT]'pcd_latitude'[REDACTED_BY_SCRIPT]")
    
    # Clean up the temporary key from the chunk
    df_chunk.drop(columns=['pcds_cleaned_key'], inplace=True, errors='ignore')

    # --- The rest of your merges (unaffected by the postcode issue) ---
    # These merges use keys like 'oa21cd', 'msoa21cd', which are generally well-behaved.
    # The original logic here is a bit complex but should work.
    
    # OA Classification merge
    if 'df_oa_class_full' in ancillary_data:
        df_chunk = pd.merge(df_chunk, ancillary_data['df_oa_class_full'], on='oa21cd', how='left', suffixes=('', '_oa'))
        # This handles potential column name collisions if they already existed
        for col in ancillary_data['df_oa_class_full'].columns:
             if f"{col}_oa" in df_chunk.columns and col in df_chunk.columns:
                 df_chunk[col] = df_chunk[col].fillna(df_chunk[f"{col}_oa"])
                 df_chunk.drop(columns=[f"{col}_oa"], inplace=True)

    # Postcode to WZ Full (This one DOES use a postcode key, so needs care)
    if 'df_pcd_wz_full' in ancillary_data:
         # We need to create the cleaned key again for this merge
        df_chunk['pcds_cleaned_key'] = df_chunk['pcds'].astype(str).str.replace(r'\s+', '', regex=True).str.upper()
        df_chunk = pd.merge(df_chunk, ancillary_data['df_pcd_wz_full'],
                            left_on='pcds_cleaned_key',
                            right_on='pcds_cleaned', # Use the cleaned key we made in preload
                            how='left', suffixes=('', '_wz'))
        df_chunk.drop(columns=['pcds_cleaned_key', 'pcds_cleaned', 'pcds_wz'], inplace=True, errors='ignore')


    # MSOA RUC
    if 'df_msoa_ruc' in ancillary_data:
        df_chunk = pd.merge(df_chunk, ancillary_data['df_msoa_ruc'], on='msoa21cd', how='left', suffixes=('', '_msruc'))
        # Simplified collision handling
        for col in ancillary_data['df_msoa_ruc'].columns:
            if f"{col}_msruc" in df_chunk.columns and col in df_chunk.columns:
                 df_chunk.drop(columns=[col], inplace=True)
                 df_chunk.rename(columns={f"{col}_msruc": col}, inplace=True)

    # LAD OAC
    if 'df_lad_oac' in ancillary_data:
        df_chunk = pd.merge(df_chunk, ancillary_data['df_lad_oac'], left_on='ladcd', right_on='LA23CD', how='left', suffixes=('', '_ladoac'))
        df_chunk.drop(columns=['LA23CD', 'LA23CD_ladoac'], inplace=True, errors='ignore')


    print(f"[REDACTED_BY_SCRIPT]")
    df_chunk_fi = create_feature_interactions_for_chunk(df_chunk)
    return df_chunk_fi

# --- create_feature_interactions_for_chunk (Keep as is) ---
# Your feature interaction logic seems fine and operates on the columns
# that are created by the merges. No changes needed here.
def create_feature_interactions_for_chunk(df_fi):
    # (Your existing code here is fine)
    # Calculate Postcode_Age (handle NaNs in dointr)
    df_fi['dointr_year'] = pd.to_numeric(df_fi['dointr'], errors='coerce') // 100
    df_fi['Postcode_Age'] = CURRENT_YEAR - df_fi['dointr_year']
    # Corrected FutureWarning for fillna
    df_fi['Postcode_Age'] = df_fi['Postcode_Age'].fillna(df_fi['Postcode_Age'].median())

    # 1. Postcode-OA Classification Alignment Score
    df_fi['[REDACTED_BY_SCRIPT]'] = ((df_fi.get('[REDACTED_BY_SCRIPT]') == 'A') & (df_fi.get('[REDACTED_BY_SCRIPT]') == '1')).astype(int)

    # 2. OA Typicality in LAD (Supergroup Level)
    def get_lad_sg_index(row):
        sg_code = row.get('[REDACTED_BY_SCRIPT]')
        if pd.isna(sg_code): return np.nan
        sg_col_name = f'[REDACTED_BY_SCRIPT]".")[0]}_Index'
        return row.get(sg_col_name, np.nan)
    df_fi['[REDACTED_BY_SCRIPT]'] = df_fi.apply(get_lad_sg_index, axis=1)

    # 3. Urban Postcode in Rural MSOA Mismatch
    df_fi['[REDACTED_BY_SCRIPT]'] = df_fi.get('[REDACTED_BY_SCRIPT]', pd.Series(index=df_fi.index, dtype=object)).apply(map_pcd_to_category_type)
    df_fi['[REDACTED_BY_SCRIPT]'] = ((df_fi['[REDACTED_BY_SCRIPT]'] == "UrbanDense") & (df_fi.get('[REDACTED_BY_SCRIPT]') == "Rural")).astype(int)

    # 4. Large User Postcode in Non-Industrial/Commercial WZ
    df_fi['WZ_Type_Mapped'] = df_fi.get('[REDACTED_BY_SCRIPT]', pd.Series(index=df_fi.index, dtype=object)).apply(map_to_wz_type)
    df_fi['[REDACTED_BY_SCRIPT]'] = ((df_fi.get('usertype') == 1) & (~df_fi['WZ_Type_Mapped'].isin(["Industrial", "Commercial", "Retail"]))).astype(int)

    # 5. Postcode Age vs. OA Classification "Modernity"
    df_fi['[REDACTED_BY_SCRIPT]'] = df_fi.get('OA21_SUBGROUP_NAME', pd.Series(index=df_fi.index, dtype=object)).apply(map_oa_to_subgroup_nature)
    df_fi['[REDACTED_BY_SCRIPT]'] = (df_fi['Postcode_Age'] * (df_fi['[REDACTED_BY_SCRIPT]'] == "Historic").astype(int) - df_fi['Postcode_Age'] * (df_fi['[REDACTED_BY_SCRIPT]'] == "Modern").astype(int))

    # 6. LAD Specialization
    def get_lad_sub_sg_specialization(row):
        sub_code = row.get('OA21_SUBGROUP_CODE')
        sg_code = row.get('[REDACTED_BY_SCRIPT]')
        if pd.isna(sub_code) or pd.isna(sg_code): return np.nan
        sub_col_name = f'[REDACTED_BY_SCRIPT]'
        sg_col_name = f'[REDACTED_BY_SCRIPT]".")[0]}_Index'
        sub_index = pd.to_numeric(row.get(sub_col_name), errors='coerce')
        sg_index = pd.to_numeric(row.get(sg_col_name), errors='coerce')
        if pd.notna(sub_index) and pd.notna(sg_index) and sg_index > 0:
            return sub_index / (sg_index + epsilon)
        return np.nan
    df_fi['[REDACTED_BY_SCRIPT]'] = df_fi.apply(get_lad_sub_sg_specialization, axis=1)

    # 7. Consistency of 2011 OA Profile with 2021 MSOA Rurality
    df_fi['[REDACTED_BY_SCRIPT]'] = (df_fi.get('[REDACTED_BY_SCRIPT]', pd.Series(index=df_fi.index, dtype=object)).str.contains("[REDACTED_BY_SCRIPT]", case=False, na=False) & df_fi.get('MSOA21_RUC_NAME', pd.Series(index=df_fi.index, dtype=object)).str.contains("Rural", case=False, na=False)).astype(int)

    # 8. Log-Transformed LAD Prominence of Postcode's OA Subgroup
    def get_lad_sub_prominence(row):
        sub_code = row.get('OA21_SUBGROUP_CODE')
        if pd.isna(sub_code): return np.nan
        sub_col_name = f'[REDACTED_BY_SCRIPT]'
        val = row.get(sub_col_name)
        if pd.notna(val): return np.log(pd.to_numeric(val, errors='coerce') + 1)
        return np.nan
    df_fi['[REDACTED_BY_SCRIPT]'] = df_fi.apply(get_lad_sub_prominence, axis=1)

    # 9. Workplace Zone "Economic Hub" in High-Density Residential OA
    df_fi['[REDACTED_BY_SCRIPT]'] = df_fi['WZ_Type_Mapped'].isin(["Retail", "Commercial"]).astype(int) # Reuses WZ_Type_Mapped from #4
    df_fi['OA_Density_Mapped'] = df_fi.get('OA21_SUBGROUP_NAME', pd.Series(index=df_fi.index, dtype=object)).apply(map_to_oa_density)
    df_fi['[REDACTED_BY_SCRIPT]'] = ((df_fi['[REDACTED_BY_SCRIPT]'] == 1) & (df_fi['OA_Density_Mapped'] == "[REDACTED_BY_SCRIPT]")).astype(int)

    # 10. Relative "Newness"
    df_fi['Postcode_Recency_Score'] = 1 / (df_fi['Postcode_Age'] + 1)
    new_growth_subgroup_col = '[REDACTED_BY_SCRIPT]' # Example
    if new_growth_subgroup_col in df_fi.columns:
        df_fi['[REDACTED_BY_SCRIPT]'] = df_fi['Postcode_Recency_Score'] * pd.to_numeric(df_fi[new_growth_subgroup_col], errors='coerce').fillna(0)
    else:
        df_fi['[REDACTED_BY_SCRIPT]'] = 0

    # 11. MSOA Rurality Amplified by Specific Rural Postcode Type
    df_fi['[REDACTED_BY_SCRIPT]'] = df_fi.get('[REDACTED_BY_SCRIPT]', pd.Series(index=df_fi.index, dtype=object)).apply(map_to_pcd_rural_type)
    df_fi['FI_DeepRuralFocus'] = ((df_fi.get('[REDACTED_BY_SCRIPT]') == "Rural") & (df_fi['[REDACTED_BY_SCRIPT]'] == "DeepRural")).astype(int)

    # 12. Urban-Rural Transition Zone Indicator
    df_fi['[REDACTED_BY_SCRIPT]'] = df_fi.get('[REDACTED_BY_SCRIPT]', pd.Series(index=df_fi.index, dtype=object)).apply(lambda x: map_to_broad_type(x, 'OA'))
    df_fi['[REDACTED_BY_SCRIPT]'] = df_fi.get('[REDACTED_BY_SCRIPT]', pd.Series(index=df_fi.index, dtype=object)).apply(lambda x: map_to_broad_type(x, 'MSOA'))
    df_fi['FI_TransitionZone'] = (df_fi['[REDACTED_BY_SCRIPT]'] != df_fi['MS[REDACTED_BY_SCRIPT]']).astype(int)
    df_fi.loc[(df_fi['[REDACTED_BY_SCRIPT]'] == 'Other') | (df_fi['MS[REDACTED_BY_SCRIPT]'] == 'Other') | (df_fi['[REDACTED_BY_SCRIPT]'] == 'Unknown') | (df_fi['MS[REDACTED_BY_SCRIPT]'] == 'Unknown'), 'FI_TransitionZone'] = 0


    # 13. Ratio of Specific OA Group Index to "Average" Group Index in LAD
    lad_grp_cols = [col for col in df_fi.columns if col.startswith('LAD_OAC_GRP') and col.endswith('_Index')]
    if lad_grp_cols:
        df_fi['[REDACTED_BY_SCRIPT]'] = df_fi[lad_grp_cols].mean(axis=1, skipna=True)
        def get_oa_grp_vs_lad_avg(row):
            grp_code = row.get('OA21_GROUP_CODE')
            if pd.isna(grp_code): return np.nan
            grp_col_name = f'[REDACTED_BY_SCRIPT]'
            mean_val = row.get('[REDACTED_BY_SCRIPT]')
            grp_val = row.get(grp_col_name)
            if pd.notna(grp_val) and pd.notna(mean_val) and mean_val > 0:
                return pd.to_numeric(grp_val, errors='coerce') / (mean_val + epsilon)
            return np.nan
        df_fi['FI_OAGrp_vs_LADAvg'] = df_fi.apply(get_oa_grp_vs_lad_avg, axis=1)
    else:
        df_fi['FI_OAGrp_vs_LADAvg'] = np.nan
        df_fi['[REDACTED_BY_SCRIPT]'] = np.nan # Ensure column exists

    # 14. Combined Age and User Type for Postcode
    df_fi['[REDACTED_BY_SCRIPT]'] = df_fi['Postcode_Age'] * df_fi.get('usertype', pd.Series(index=df_fi.index)).fillna(0)

    # 15. Postcode Classification's Alignment with Dominant LAD OAC Subgroup Type
    lad_sub_cols = [col for col in df_fi.columns if col.startswith('LAD_OAC_SUB') and col.endswith('_Index')]
    if lad_sub_cols:
        df_fi['[REDACTED_BY_SCRIPT]'] = df_fi.apply(lambda row: get_subgroup_code_with_max_index(row, lad_sub_cols), axis=1)
        df_fi['[REDACTED_BY_SCRIPT]'] = df_fi['[REDACTED_BY_SCRIPT]'].astype(str).str.replace('LAD_OAC_SUB','', regex=False).str.replace('_Index','', regex=False)
        def check_alignment(row):
            expected_list = map_pcd_to_expected_oa_subgroup(row.get('[REDACTED_BY_SCRIPT]'))
            dominant_code = row.get('[REDACTED_BY_SCRIPT]')
            if expected_list is None or pd.isna(dominant_code) or dominant_code == 'None': return 0
            return 1 if dominant_code in expected_list else 0
        df_fi['[REDACTED_BY_SCRIPT]'] = df_fi.apply(check_alignment, axis=1)
    else:
        df_fi['[REDACTED_BY_SCRIPT]'] = 0
        df_fi['[REDACTED_BY_SCRIPT]'] = None
        df_fi['[REDACTED_BY_SCRIPT]'] = None


    # 16. Strength of WZ "Retail" Classification in MSOA designated "Urban Centre"
    df_fi['[REDACTED_BY_SCRIPT]'] = (df_fi.get('[REDACTED_BY_SCRIPT]', pd.Series(index=df_fi.index, dtype=object)).str.contains("Retail", case=False, na=False) & df_fi.get('MSOA21_RUC_NAME', pd.Series(index=df_fi.index, dtype=object)).str.contains("Urban Centre|Major Centre", case=False, na=False)).astype(int)

    # 17. Inverse of LAD Index for Postcode's OA Supergroup (Rarity Score)
    df_fi['FI_OA_Rarity_in_LAD'] = 1 / (pd.to_numeric(df_fi.get('[REDACTED_BY_SCRIPT]'), errors='coerce') + epsilon)

    # 18. Difference: LAD Index for Postcode'[REDACTED_BY_SCRIPT]'s OA Supergroup
    def get_lad_sub_index_raw(row):
        sub_code = row.get('OA21_SUBGROUP_CODE')
        if pd.isna(sub_code): return np.nan
        sub_col_name = f'[REDACTED_BY_SCRIPT]'
        val = row.get(sub_col_name)
        if pd.notna(val): return pd.to_numeric(val, errors='coerce')
        return np.nan
    df_fi['[REDACTED_BY_SCRIPT]'] = df_fi.apply(get_lad_sub_index_raw, axis=1)
    df_fi['[REDACTED_BY_SCRIPT]'] = df_fi['[REDACTED_BY_SCRIPT]'] - pd.to_numeric(df_fi.get('[REDACTED_BY_SCRIPT]'), errors='coerce')

    # 19. Hierarchical Consistency Score
    df_fi['[REDACTED_BY_SCRIPT]'] = df_fi.get('[REDACTED_BY_SCRIPT]', pd.Series(index=df_fi.index, dtype=object)).apply(lambda x: map_to_broad_type(x, 'PCD'))
    df_fi['[REDACTED_BY_SCRIPT]'] = ((df_fi['[REDACTED_BY_SCRIPT]'] == df_fi['[REDACTED_BY_SCRIPT]']).astype(int) + (df_fi['[REDACTED_BY_SCRIPT]'] == df_fi['MS[REDACTED_BY_SCRIPT]']).astype(int))
    df_fi.loc[(df_fi['[REDACTED_BY_SCRIPT]'] == 'Other') | (df_fi['[REDACTED_BY_SCRIPT]'] == 'Unknown'), '[REDACTED_BY_SCRIPT]'] = (df_fi['[REDACTED_BY_SCRIPT]'] == df_fi['MS[REDACTED_BY_SCRIPT]']).astype(int)
    df_fi.loc[(df_fi['[REDACTED_BY_SCRIPT]'] == 'Other') | (df_fi['[REDACTED_BY_SCRIPT]'] == 'Unknown'), '[REDACTED_BY_SCRIPT]'] = 0 # If middle is unknown, consistency is broken or hard to define

    # 20. Product of Postcode Age and MSOA Rurality
    df_fi['Is_MSOA_Rural'] = (df_fi.get('[REDACTED_BY_SCRIPT]') == "Rural").astype(int)
    df_fi['FI_OldPCD_RuralMSOA'] = df_fi['Postcode_Age'] * df_fi['Is_MSOA_Rural']
    
    # Clean up intermediate mapping columns
    cols_to_drop_intermediate = [
        'dointr_year', '[REDACTED_BY_SCRIPT]', 'WZ_Type_Mapped', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', 'OA_Density_Mapped', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', 'MS[REDACTED_BY_SCRIPT]', # Optional: keep if useful
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', 'Is_MSOA_Rural', 'Postcode_Recency_Score'
    ]
    df_fi.drop(columns=[col for col in cols_to_drop_intermediate if col in df_fi.columns], inplace=True, errors='ignore')
    return df_fi

# --- Main Execution (Keep as is) ---
if __name__ == '__main__':
    # ... your file paths and main loop ...
    # No changes are needed in this section.
    # Just run the script with the corrected functions above.
    file_paths = {
        'pcd_oa_lsoa_msoa_lad': '[REDACTED_BY_SCRIPT]',
        'postcode_lookup_bng': '[REDACTED_BY_SCRIPT]', # Contains eastings, northings
        'postcode_lookup_latlon': '[REDACTED_BY_SCRIPT]', # Contains lat, lon
        '[REDACTED_BY_SCRIPT]': '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]': '[REDACTED_BY_SCRIPT]',
        'oa_classification': '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]': '[REDACTED_BY_SCRIPT]',
        'postcode_to_wz': '[REDACTED_BY_SCRIPT]', # Contains oa11cd, oac11nm, wz11cd, wzc11nm, soac11nm
        'wz_assignments': '[REDACTED_BY_SCRIPT]',
        'msoa_ruc': '[REDACTED_BY_SCRIPT]',
        'lad_oac_indexscores': '[REDACTED_BY_SCRIPT]',
    }

    ancillary_data_store = preload_ancillary_data(file_paths)

    if ancillary_data_store:
        processed_chunks_list = []
        
        # Determine problematic column name for base file for dtype specification
        base_file_problematic_col_idx = 13 # As per your DtypeWarning
        try:
            temp_base_peek = pd.read_csv(file_paths['pcd_oa_lsoa_msoa_lad'], nrows=5, encoding='latin1')
            base_problem_col_name = None
            if base_file_problematic_col_idx < len(temp_base_peek.columns):
                base_problem_col_name = temp_base_peek.columns[base_file_problematic_col_idx]
            dtype_spec_base = {base_problem_col_name: str} if base_problem_col_name else None
        except Exception as e:
            print(f"[REDACTED_BY_SCRIPT]")
            dtype_spec_base = None


        print(f"[REDACTED_BY_SCRIPT]")
        chunk_iter_count = 0
        try:
            for df_base_chunk in pd.read_csv(file_paths['pcd_oa_lsoa_msoa_lad'], 
                                             chunksize=CHUNK_SIZE, 
                                             encoding='latin1',
                                             dtype=dtype_spec_base,
                                             low_memory=False if dtype_spec_base is None else True): # Use low_memory=False only if no dtype_spec
                chunk_iter_count += 1
                print(f"[REDACTED_BY_SCRIPT]")
                df_base_chunk.rename(columns=lambda x: x.replace('\ufeff', ''), inplace=True)
                
                if 'pcds' not in df_base_chunk.columns:
                    print("Critical Error: 'pcds'[REDACTED_BY_SCRIPT]")
                    processed_chunks_list = [] # Clear any previous chunks
                    break
                
                processed_chunk = process_chunk(df_base_chunk, ancillary_data_store)
                processed_chunks_list.append(processed_chunk)
                print(f"[REDACTED_BY_SCRIPT]")

        except UnicodeDecodeError: # Fallback encoding for chunk processing
            print("[REDACTED_BY_SCRIPT]")
            processed_chunks_list = [] # Reset
            chunk_iter_count = 0
            for df_base_chunk in pd.read_csv(file_paths['pcd_oa_lsoa_msoa_lad'], 
                                             chunksize=CHUNK_SIZE, 
                                             encoding='cp1252',
                                             dtype=dtype_spec_base,
                                             low_memory=False if dtype_spec_base is None else True):
                chunk_iter_count += 1
                print(f"[REDACTED_BY_SCRIPT]")
                df_base_chunk.rename(columns=lambda x: x.replace('\ufeff', ''), inplace=True)
                if 'pcds' not in df_base_chunk.columns:
                    print("Critical Error: 'pcds'[REDACTED_BY_SCRIPT]")
                    processed_chunks_list = []
                    break
                processed_chunk = process_chunk(df_base_chunk, ancillary_data_store)
                processed_chunks_list.append(processed_chunk)
                print(f"[REDACTED_BY_SCRIPT]")
        except Exception as e:
            print(f"[REDACTED_BY_SCRIPT]")
            processed_chunks_list = []


        if processed_chunks_list:
            print("[REDACTED_BY_SCRIPT]")
            final_df = pd.concat(processed_chunks_list, ignore_index=True)
            print("[REDACTED_BY_SCRIPT]")
            print(final_df.head())
            print(f"[REDACTED_BY_SCRIPT]")
            final_df.info()

            # Optional: Save to CSV
            try:
                final_df.to_csv("[REDACTED_BY_SCRIPT]", index=False)
                print("[REDACTED_BY_SCRIPT]")
            except Exception as e:
                print(f"[REDACTED_BY_SCRIPT]")
        else:
            print("[REDACTED_BY_SCRIPT]")
    else:
        print("[REDACTED_BY_SCRIPT]")