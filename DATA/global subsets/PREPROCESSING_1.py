import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import re 

# --- Configuration for Subset 1 ---
FILE_PATH_SUBSET1 = "[REDACTED_BY_SCRIPT]"  # Replace with the actual path to your subset1 file

SAMPLE_SIZE = 50000

COLS_TO_DROP_SUBSET1 = [
    'pcds_original','pcd7', 'pcd8', 'pcds',
    'oa21cd', 'lsoa21cd', 'msoa21cd', 'ladcd',
    'lsoa21nm', 'msoa21nm', 'ladnm', 'ladnmw', '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]', 'OA21_GROUP_NAME', 'OA21_SUBGROUP_NAME',
    'oa11cd', 'wz11cd', 'lsoa11cd', 'msoa11cd', 'ladcd_from_wz_file',
    'MSOA21_RUC_NAME'
]
INITIAL_CATEGORICAL_COLS_SUBSET1 = [
    'usertype', '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]', 'OA21_GROUP_CODE', 'OA21_SUBGROUP_CODE',
    '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', 'MS[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]', 'WZ11_GROUP_NAME',
    'MSOA21_RUC_CODE', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]'
]

COORDS_COLS = ['pcd_eastings', 'pcd_northings', 'pcd_latitude', 'pcd_longitude']

LOW_VARIANCE_THRESHOLD = 0.01
HIGH_CORRELATION_THRESHOLD = 0.95
OHE_LOW_VARIANCE_THRESHOLD_FOR_REMOVAL = 0.002

def load_data(file_path):
    encodings_to_try = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
    for encoding in encodings_to_try:
        try:
            df = pd.read_csv(file_path, low_memory=False, encoding=encoding)
            print(f"[REDACTED_BY_SCRIPT]")
            return df
        except UnicodeDecodeError:
            print(f"[REDACTED_BY_SCRIPT]")
        except FileNotFoundError:
            print(f"[REDACTED_BY_SCRIPT]")
            return None
    print("[REDACTED_BY_SCRIPT]")
    return None

def preprocess_subset1(df, initial_categorical_cols):
    print(f"[REDACTED_BY_SCRIPT]")
    df_processed = df.drop(columns=COLS_TO_DROP_SUBSET1, errors='ignore')
    
    df_processed['is_terminated'] = df_processed['doterm'].notna().astype(int)
    temp_initial_categorical_cols = list(initial_categorical_cols)
    if 'is_terminated' not in temp_initial_categorical_cols:
        temp_initial_categorical_cols.append('is_terminated')
        
    df_processed = df_processed.drop(columns=['dointr', 'doterm'], errors='ignore')

    for col in temp_initial_categorical_cols:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].astype(str)

    # --- START OF COORDINATE FIX ---

    # Identify all potential numerical and categorical columns
    all_numerical_cols = df_processed.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df_processed.select_dtypes(include=['object', 'category']).columns.tolist()

    # Explicitly separate coordinate columns from other numerical columns
    # Uses the COORDS_COLS list defined in the configuration section
    coords_cols = [col for col in all_numerical_cols if col in COORDS_COLS]
    numerical_cols = [col for col in all_numerical_cols if col not in COORDS_COLS]
    
    # Ensure all columns are covered (this logic is now updated)
    all_cols_set = set(df_processed.columns)
    processed_cols_set = set(numerical_cols + categorical_cols + coords_cols)
    if all_cols_set != processed_cols_set:
        missing_from_typed_lists = all_cols_set - processed_cols_set
        print(f"[REDACTED_BY_SCRIPT]")
        for m_col in missing_from_typed_lists:
            if m_col not in categorical_cols:
                categorical_cols.append(m_col)
                df_processed[m_col] = df_processed[m_col].astype(str)

    print(f"[REDACTED_BY_SCRIPT]")
    print(f"[REDACTED_BY_SCRIPT]")
    print(f"[REDACTED_BY_SCRIPT]")

    # Pipeline for standard numerical features (as discussed, 'median' is better)
    numerical_pipeline = Pipeline([
        ('imputer_num', SimpleImputer(strategy='median', add_indicator=True)),
        ('power_transform', PowerTransformer(method='yeo-johnson', standardize=False)),
        ('scaler_num', MinMaxScaler())
    ])
    
    # New, separate pipeline for coordinates (just imputes missing values with 0)
    coords_pipeline = Pipeline([
        ('imputer_coords', SimpleImputer(strategy='constant', fill_value=0))
    ])

    # Pipeline for categorical features
    categorical_pipeline = Pipeline([
        ('imputer_cat', SimpleImputer(strategy='constant', fill_value='Missing_Category')),
        ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='if_binary'))
    ])

    # Update ColumnTransformer to include the new coordinate pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_cols),
            ('cat', categorical_pipeline, categorical_cols),
            ('coords', coords_pipeline, coords_cols) # Added transformer for coordinates
        ],
        remainder='passthrough'
        # verbose_feature_names_out=False is not needed when using set_output
    )
    preprocessor.set_output(transform="pandas") # This is great! It handles column names robustly.

    # --- END OF COORDINATE FIX ---

    print("[REDACTED_BY_SCRIPT]")
    df_transformed = preprocessor.fit_transform(df_processed)
    
    # Store OHE names for later variance check
    # This introspection is a bit tricky but works
    ohe_feature_names = preprocessor.named_transformers_['cat'].named_steps['one_hot_encoder'].get_feature_names_out(categorical_cols).tolist()

    print(f"[REDACTED_BY_SCRIPT]")
    return df_transformed, ohe_feature_names

def get_low_variance_features_report(df, threshold):
    variances = df.var(ddof=0)
    low_variance_info = {feature: variances[feature] for feature in variances[variances < threshold].index}
    return low_variance_info

def get_collinearity_report(df, threshold):
    numeric_df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
    corr_matrix = numeric_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    highly_correlated_pairs = []
    for column in upper.columns:
        for feature in upper[upper[column] > threshold].index:
            highly_correlated_pairs.append((feature, column, upper.loc[feature, column])) # Swapped order for consistency
    return highly_correlated_pairs

def identify_features_to_remove(df, ohe_feature_names, low_variance_info, collinear_pairs):
    cols_to_remove = set()
    removal_reasons = {}

    # 1. User-specified low variance removals (fixed list)
    user_fixed_low_var_removals = [
        '[REDACTED_BY_SCRIPT]', 'FI_OA_Rarity_in_LAD',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]'
    ]
    for col in user_fixed_low_var_removals:
        if col in df.columns:
            cols_to_remove.add(col)
            removal_reasons[col] = "[REDACTED_BY_SCRIPT]"

    # 2. Conditional Low Variance Removals (OHE Features < 0.002)
    for feature, variance in low_variance_info.items():
        if feature in ohe_feature_names and variance < OHE_LOW_VARIANCE_THRESHOLD_FOR_REMOVAL:
            if feature not in cols_to_remove: # Avoid double-counting if already in user list
                 cols_to_remove.add(feature)
                 removal_reasons[feature] = f"[REDACTED_BY_SCRIPT]"
    
    # 3. Collinearity Removals
    # Note: drop='if_binary' in OHE handles usertype_0/1, is_terminated_0/1

    # Rule: Cascading NaNs (OHE _nan features)
    # Example: If '[REDACTED_BY_SCRIPT]' and 'OA21_GROUP_CODE_nan' are highly correlated, drop the latter.
    # This requires checking the collinear_pairs list.
    nan_hierarchies = [
        # (broader_nan, narrower_nan_to_drop_if_correlated)
        ('[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]'), # Assuming [REDACTED_BY_SCRIPT] results from NaN
        ('[REDACTED_BY_SCRIPT]', 'OA21_GROUP_CODE_nan'),
        ('OA21_GROUP_CODE_nan', '[REDACTED_BY_SCRIPT]'), # Could also be SUPERGROUP vs SUBGROUP
        ('[REDACTED_BY_SCRIPT]', 'WZ11_GROUP_NAME_nan'),
        ('MSOA21_RUC_CODE_nan', '[REDACTED_BY_SCRIPT]')
    ]
    for broader_nan, narrower_nan in nan_hierarchies:
        for f1, f2, corr in collinear_pairs:
            if {f1, f2} == {broader_nan, narrower_nan}:
                if narrower_nan in df.columns and narrower_nan not in cols_to_remove:
                    cols_to_remove.add(narrower_nan)
                    removal_reasons[narrower_nan] = f"[REDACTED_BY_SCRIPT]"
                break 
            # Check SUPERGROUP vs SUBGROUP for OA21
            if broader_nan == 'OA21_GROUP_CODE_nan' and narrower_nan == '[REDACTED_BY_SCRIPT]': # This rule is for GRP vs SUB
                 if {f1,f2} == {'[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]'}:
                      if '[REDACTED_BY_SCRIPT]' in df.columns and '[REDACTED_BY_SCRIPT]' not in cols_to_remove:
                           cols_to_remove.add('[REDACTED_BY_SCRIPT]')
                           removal_reasons['[REDACTED_BY_SCRIPT]'] = f"[REDACTED_BY_SCRIPT]"
                      break


    # Rule: Redundant Name/Classification Features (OHE)
    # WZ11_GROUP_NAME_X vs WZ11_CLASSIFICATION_NAME_raw_X
    raw_wz_cols_to_drop = set()
    for f1, f2, corr in collinear_pairs:
        if f1.startswith("WZ11_GROUP_NAME_") and f2.startswith("[REDACTED_BY_SCRIPT]"):
            suffix1 = f1.replace("WZ11_GROUP_NAME_", "")
            suffix2 = f2.replace("[REDACTED_BY_SCRIPT]", "")
            if suffix1 == suffix2:
                if f2 in df.columns and f2 not in cols_to_remove:
                    raw_wz_cols_to_drop.add(f2)
                    removal_reasons[f2] = f"[REDACTED_BY_SCRIPT]"
        elif f2.startswith("WZ11_GROUP_NAME_") and f1.startswith("[REDACTED_BY_SCRIPT]"):
            suffix1 = f1.replace("[REDACTED_BY_SCRIPT]", "")
            suffix2 = f2.replace("WZ11_GROUP_NAME_", "")
            if suffix1 == suffix2:
                if f1 in df.columns and f1 not in cols_to_remove:
                    raw_wz_cols_to_drop.add(f1)
                    removal_reasons[f1] = f"[REDACTED_BY_SCRIPT]"
    cols_to_remove.update(raw_wz_cols_to_drop)

    # Rule: "Pseudo" area features
    pseudo_oa11 = "[REDACTED_BY_SCRIPT]" # Assuming OHE name if spaces exist
    pseudo_wz11_raw = "[REDACTED_BY_SCRIPT]"
    pseudo_msoa11 = "[REDACTED_BY_SCRIPT]" # Original name
    
    # Need to handle actual OHE names for pseudo_msoa11 if it contains spaces/special chars
    # For simplicity, assume exact match for now or ensure these are correct OHE names
    
    pseudo_to_check = [pseudo_wz11_raw, pseudo_msoa11]
    for f1, f2, corr in collinear_pairs:
        if f1 == pseudo_oa11 and f2 in pseudo_to_check:
            if f2 in df.columns and f2 not in cols_to_remove:
                cols_to_remove.add(f2); removal_reasons[f2] = f"[REDACTED_BY_SCRIPT]"
        elif f2 == pseudo_oa11 and f1 in pseudo_to_check:
            if f1 in df.columns and f1 not in cols_to_remove:
                cols_to_remove.add(f1); removal_reasons[f1] = f"[REDACTED_BY_SCRIPT]"


    # Rule: LAD OAC Indices (Simplified: drop GRP if highly corr with SUB, drop SG if highly corr with GRP)
    lad_oac_indices_to_drop = set()
    # Regex to capture type (SG, GRP, SUB) and the code part (e.g., 1a, 1a1)
    lad_pattern = re.compile(r"[REDACTED_BY_SCRIPT]")

    for f1_orig, f2_orig, corr in collinear_pairs:
        match1 = lad_pattern.match(f1_orig)
        match2 = lad_pattern.match(f2_orig)

        if match1 and match2:
            type1, code1 = match1.groups()
            type2, code2 = match2.groups()

            # SUB vs GRP: if SUB'[REDACTED_BY_SCRIPT]'s code
            if type1 == "SUB" and type2 == "GRP" and code1.startswith(code2):
                if f2_orig in df.columns and f2_orig not in cols_to_remove: lad_oac_indices_to_drop.add(f2_orig)
            elif type2 == "SUB" and type1 == "GRP" and code2.startswith(code1):
                if f1_orig in df.columns and f1_orig not in cols_to_remove: lad_oac_indices_to_drop.add(f1_orig)
            
            # GRP vs SG: if GRP's code belongs to SG's code (e.g. GRP1a and SG1)
            # This requires SG codes to be just the number part, e.g. SG1 matching GRP1a
            if type1 == "GRP" and type2 == "SG" and code1.startswith(code2.replace("SG", "")): # code2 for SG is just the number
                 # Check if GRP is not already marked for removal by a SUB
                if f1_orig not in lad_oac_indices_to_drop:
                    if f2_orig in df.columns and f2_orig not in cols_to_remove: lad_oac_indices_to_drop.add(f2_orig)
            elif type2 == "GRP" and type1 == "SG" and code2.startswith(code1.replace("SG", "")):
                if f2_orig not in lad_oac_indices_to_drop:
                    if f1_orig in df.columns and f1_orig not in cols_to_remove: lad_oac_indices_to_drop.add(f1_orig)
    
    for col_to_drop in lad_oac_indices_to_drop:
        if col_to_drop not in cols_to_remove:
            cols_to_remove.add(col_to_drop)
            removal_reasons[col_to_drop] = "[REDACTED_BY_SCRIPT]"
            
    return list(cols_to_remove), removal_reasons

# --- Main Execution ---
if __name__ == '__main__':
    subset1_df_full = load_data(FILE_PATH_SUBSET1)

    if subset1_df_full is not None:
        if len(subset1_df_full) > SAMPLE_SIZE:
            print(f"[REDACTED_BY_SCRIPT]")
            subset1_df_to_process = subset1_df_full.sample(n=SAMPLE_SIZE, random_state=42).copy()
        else:
            subset1_df_to_process = subset1_df_full.copy()
        
        df_processed_subset1, ohe_cols = preprocess_subset1(subset1_df_to_process, INITIAL_CATEGORICAL_COLS_SUBSET1)
        
        print("[REDACTED_BY_SCRIPT]")
        #df_processed_subset1.fillna(-1, inplace=True) 

        low_var_info_report = get_low_variance_features_report(df_processed_subset1, LOW_VARIANCE_THRESHOLD)
        collinear_pairs_report = get_collinearity_report(df_processed_subset1, HIGH_CORRELATION_THRESHOLD)

        print("[REDACTED_BY_SCRIPT]")
        features_to_drop_final, reasons_for_dropping = identify_features_to_remove(
            df_processed_subset1,
            ohe_cols,
            low_var_info_report,
            collinear_pairs_report
        )
        
        features_to_drop_final_existing = [col for col in features_to_drop_final if col in df_processed_subset1.columns]
        
        df_final_subset1 = df_processed_subset1.copy() # Start with a copy
        if features_to_drop_final_existing:
            print(f"[REDACTED_BY_SCRIPT]")
            for col in sorted(features_to_drop_final_existing):
                 print(f"[REDACTED_BY_SCRIPT]'N/A')})")
            df_final_subset1.drop(columns=features_to_drop_final_existing, inplace=True, errors='ignore') # errors='ignore' in case a col was already removed implicitly
            print(f"[REDACTED_BY_SCRIPT]")
            print(f"[REDACTED_BY_SCRIPT]")
        else:
            print("[REDACTED_BY_SCRIPT]")
            # df_final_subset1 is already a copy of df_processed_subset1
            
        print("[REDACTED_BY_SCRIPT]")
        print(f"[REDACTED_BY_SCRIPT]")
        print(df_final_subset1.head())

        # --- SAVING THE PROCESSED DATAFRAME ---
        output_filename = "[REDACTED_BY_SCRIPT]"
        try:
            df_final_subset1.to_csv(output_filename, index=False)
            print(f"[REDACTED_BY_SCRIPT]")
        except Exception as e:
            print(f"[REDACTED_BY_SCRIPT]")