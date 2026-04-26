import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import re

# --- Configuration for Subset 3 ---
FILE_PATH_SUBSET3 = "[REDACTED_BY_SCRIPT]" # !! PLEASE UPDATE THIS PATH !!
OUTPUT_FILENAME_S3 = "[REDACTED_BY_SCRIPT]"
MANUAL_DROPS_FILE_S3 = "[REDACTED_BY_SCRIPT]" # Optional file for user to list additional drops

SAMPLE_SIZE = 50000 # As used in your examples
CHUNK_SIZE = 100000 # For potential future full processing (not implemented in this script's main)

# Columns to drop at the very beginning for Subset 3
# Includes identifiers, descriptive names, and identified placeholder columns
COLS_TO_DROP_SUBSET3 = [
    # Identifiers & Names
    'pcds', 'oa21cd', 'lsoa11cd', 'lsoa21cd', # Assuming these are primarily join keys
    'AREA_NAME', # LSOA Name
    # Placeholder columns (assuming they are not populated with actual data)
    'oa11cd','[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]'
]

# Initial categorical columns for Subset 3
INITIAL_CATEGORICAL_COLS_SUBSET3 = [
    'MODE1_TYPE', # e.g., 'BP_1930_1939'
    'MODE2_TYPE',
    # 'usertype' will be added if 'dointr'/'doterm' are present and processed
]

# Thresholds (consistent with your other scripts)
LOW_VARIANCE_THRESHOLD = 0.01 # For general numerical features post-scaling (for reporting)
HIGH_CORRELATION_THRESHOLD = 0.95 # For identifying highly correlated pairs
OHE_LOW_VARIANCE_THRESHOLD_FOR_REMOVAL = 0.002 # Specifically for OHE features to be auto-removed

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
    print(f"[REDACTED_BY_SCRIPT]")
    return None

def load_manual_drops(manual_drops_file):
    manual_drops = []
    try:
        with open(manual_drops_file, 'r') as f:
            manual_drops = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        if manual_drops:
            print(f"[REDACTED_BY_SCRIPT]")
    except FileNotFoundError:
        print(f"[REDACTED_BY_SCRIPT]")
    return manual_drops

def preprocess_data(df, cols_to_drop_initial, initial_categorical_cols_config):
    print(f"[REDACTED_BY_SCRIPT]")
    
    df_processed = df.drop(columns=cols_to_drop_initial, errors='ignore')
    print(f"[REDACTED_BY_SCRIPT]")

    # Feature Engineering: is_terminated from doterm (if doterm exists)
    # Also handle usertype if present (assuming it might come with postcode basics)
    temp_initial_categorical_cols = list(initial_categorical_cols_config)
    if 'doterm' in df_processed.columns:
        df_processed['is_terminated'] = df_processed['doterm'].notna().astype(int)
        print("Created 'is_terminated' feature.")
        if 'is_terminated' not in temp_initial_categorical_cols:
            temp_initial_categorical_cols.append('is_terminated')
    
    if 'usertype' in df_processed.columns: # Assuming usertype might be present
        if 'usertype' not in temp_initial_categorical_cols:
            temp_initial_categorical_cols.append('usertype')
            
    df_processed = df_processed.drop(columns=['dointr', 'doterm'], errors='ignore') # Drop original date cols

    for col in temp_initial_categorical_cols:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].astype(str)

    numerical_cols = df_processed.select_dtypes(include=np.number).columns.tolist()
    categorical_cols_from_dtype = df_processed.select_dtypes(include=['object', 'category']).columns.tolist()
    
    final_categorical_cols = []
    final_numerical_cols = list(numerical_cols)

    for col in temp_initial_categorical_cols:
        if col in df_processed.columns:
            if col not in final_categorical_cols:
                final_categorical_cols.append(col)
            if col in final_numerical_cols:
                final_numerical_cols.remove(col)
    
    for col in categorical_cols_from_dtype:
        if col not in final_categorical_cols:
            final_categorical_cols.append(col)
            
    all_cols_set = set(df_processed.columns)
    processed_cols_set = set(final_numerical_cols + final_categorical_cols)
    if all_cols_set != processed_cols_set:
        missing_from_typed_lists = all_cols_set - processed_cols_set
        print(f"[REDACTED_BY_SCRIPT]")
        for m_col in missing_from_typed_lists:
            if m_col not in final_numerical_cols and m_col not in final_categorical_cols:
                if pd.api.types.is_numeric_dtype(df_processed[m_col]):
                    final_numerical_cols.append(m_col)
                else:
                    final_categorical_cols.append(m_col)
                    df_processed[m_col] = df_processed[m_col].astype(str)

    print(f"[REDACTED_BY_SCRIPT]")
    print(f"[REDACTED_BY_SCRIPT]")

    numerical_pipeline = Pipeline([
        ('imputer_num', SimpleImputer(strategy='constant', fill_value=-1)),
        ('power_transform', PowerTransformer(method='yeo-johnson', standardize=False)),
        ('scaler_num', MinMaxScaler())
    ])
    categorical_pipeline = Pipeline([
        ('imputer_cat', SimpleImputer(strategy='constant', fill_value='Missing_Category')),
        ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='if_binary'))
    ])

    transformers_list = []
    if final_numerical_cols: # Check if list is not empty
        transformers_list.append(('num', numerical_pipeline, final_numerical_cols))
    if final_categorical_cols: # Check if list is not empty
        transformers_list.append(('cat', categorical_pipeline, final_categorical_cols))

    if not transformers_list:
        print("[REDACTED_BY_SCRIPT]")
        return df_processed, []

    preprocessor = ColumnTransformer(
        transformers=transformers_list,
        remainder='passthrough',
        verbose_feature_names_out=False
    )
    preprocessor.set_output(transform="pandas")

    print("[REDACTED_BY_SCRIPT]")
    df_transformed = preprocessor.fit_transform(df_processed)
    
    ohe_feature_names = []
    if final_categorical_cols and 'cat' in preprocessor.named_transformers_:
        try:
            ohe_feature_names = preprocessor.named_transformers_['cat'].named_steps['one_hot_encoder'].get_feature_names_out(final_categorical_cols).tolist()
        except Exception as e:
            print(f"[REDACTED_BY_SCRIPT]")

    print(f"[REDACTED_BY_SCRIPT]")
    return df_transformed, ohe_feature_names

def get_low_variance_features_report(df, threshold):
    variances = df.var(ddof=0)
    low_variance_info = {feature: variances[feature] for feature in variances[variances < threshold].index}
    return low_variance_info

def get_collinearity_report(df, threshold):
    numeric_df = df.copy()
    for col in numeric_df.columns:
        numeric_df[col] = pd.to_numeric(numeric_df[col], errors='coerce')
    numeric_df = numeric_df.fillna(0) 
    
    corr_matrix = numeric_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    highly_correlated_pairs = []
    for column in upper.columns:
        correlated_features_in_col = upper.index[upper[column] > threshold].tolist()
        for feature in correlated_features_in_col:
            highly_correlated_pairs.append((feature, column, upper.loc[feature, column]))
    return highly_correlated_pairs

def identify_features_to_remove_subset3(df, ohe_feature_names, low_variance_info, collinear_pairs, manual_drops):
    cols_to_remove = set(manual_drops) # Start with manually specified drops
    removal_reasons = {col: "[REDACTED_BY_SCRIPT]" for col in manual_drops}
    unresolved_collinear_pairs_for_review = []

    # 1. Low Variance OHE Features (from 'MODE1_TYPE', 'MODE2_TYPE', 'usertype', 'is_terminated')
    for feature, variance in low_variance_info.items():
        if feature in ohe_feature_names and variance < OHE_LOW_VARIANCE_THRESHOLD_FOR_REMOVAL:
            if feature not in cols_to_remove:
                 cols_to_remove.add(feature)
                 removal_reasons[feature] = f"[REDACTED_BY_SCRIPT]"
    
    # 2. General Low Variance Numerical Features (e.g., sparse YYYY_sale_count_... features)
    #    These are automatically removed if variance is extremely low (e.g., effectively zero after scaling)
    #    The LOW_VARIANCE_THRESHOLD is more for reporting; actual removal often happens if var is near 0.
    #    For Subset 3, many sales counts might become 0 variance.
    extremely_low_variance_threshold = 1e-9 # A very small number for auto-removal
    for feature, variance in low_variance_info.items():
        if feature not in ohe_feature_names and variance < extremely_low_variance_threshold:
            if feature not in cols_to_remove:
                cols_to_remove.add(feature)
                removal_reasons[feature] = f"[REDACTED_BY_SCRIPT]"

    # 3. Collinearity Removals for Subset 3
    #    Apply specific rules and report others for manual review.
    kept_due_to_priority = set() # To track features explicitly kept in a pair

    for f1, f2, corr_value in collinear_pairs:
        if f1 in cols_to_remove or f2 in cols_to_remove: # Skip if one is already marked
            continue
        if f1 in kept_due_to_priority or f2 in kept_due_to_priority: # Skip if one was already kept in a previous rule
            # if f1 was kept, and f2 is correlated, f2 should be considered for removal (if not already)
            if f1 in kept_due_to_priority and f2 not in cols_to_remove:
                 cols_to_remove.add(f2)
                 removal_reasons[f2] = f"[REDACTED_BY_SCRIPT]"
            elif f2 in kept_due_to_priority and f1 not in cols_to_remove:
                 cols_to_remove.add(f1)
                 removal_reasons[f1] = f"[REDACTED_BY_SCRIPT]"
            continue

        # Rule 1: LSOA Area (prefer Ha over sqm)
        if {f1, f2} == {'LSOA_Area_Ha', 'LSOA_Shape_Area_sqm'}:
            cols_to_remove.add('LSOA_Shape_Area_sqm')
            removal_reasons['LSOA_Shape_Area_sqm'] = f"[REDACTED_BY_SCRIPT]"
            kept_due_to_priority.add('LSOA_Area_Ha')
            continue

        # Rule 2: HPI Adjusted Prices (prefer Median over Avg)
        # Regex to match HPI_Adjusted_Avg_Price_X and HPI_Adjusted_Median_Price_X
        avg_price_match_f1 = re.match(r"[REDACTED_BY_SCRIPT]", f1)
        median_price_match_f1 = re.match(r"[REDACTED_BY_SCRIPT]", f1)
        avg_price_match_f2 = re.match(r"[REDACTED_BY_SCRIPT]", f2)
        median_price_match_f2 = re.match(r"[REDACTED_BY_SCRIPT]", f2)

        if avg_price_match_f1 and median_price_match_f2 and avg_price_match_f1.group(1) == median_price_match_f2.group(1):
            cols_to_remove.add(f1) # Drop Avg
            removal_reasons[f1] = f"[REDACTED_BY_SCRIPT]"
            kept_due_to_priority.add(f2)
            continue
        elif median_price_match_f1 and avg_price_match_f2 and median_price_match_f1.group(1) == avg_price_match_f2.group(1):
            cols_to_remove.add(f2) # Drop Avg
            removal_reasons[f2] = f"[REDACTED_BY_SCRIPT]"
            kept_due_to_priority.add(f1)
            continue
        
        # For other correlations (esp. FI features, LSOA stats), add to review list
        # Avoid automatic alphabetical drops for potentially complex/valuable FI features.
        unresolved_collinear_pairs_for_review.append((f1, f2, corr_value))

    print(f"[REDACTED_BY_SCRIPT]")
    if unresolved_collinear_pairs_for_review:
        print(f"[REDACTED_BY_SCRIPT]")
        print("[REDACTED_BY_SCRIPT]'subset3_manual_drops.txt'[REDACTED_BY_SCRIPT]")
        for f1, f2, corr_val in unresolved_collinear_pairs_for_review[:15]: # Print first 15
            # Check if they are already slated for removal for another reason
            if f1 not in cols_to_remove and f2 not in cols_to_remove:
                 print(f"[REDACTED_BY_SCRIPT]'{f1}' and '{f2}'[REDACTED_BY_SCRIPT]")
    else:
        print("[REDACTED_BY_SCRIPT]")
            
    return list(cols_to_remove), removal_reasons, unresolved_collinear_pairs_for_review

# --- Main Execution ---
if __name__ == '__main__':
    subset3_df_full = load_data(FILE_PATH_SUBSET3)
    manual_drops_s3 = load_manual_drops(MANUAL_DROPS_FILE_S3)

    if subset3_df_full is not None:
        # Initial drop of manually specified columns before sampling (if any)
        cols_to_drop_very_initial = list(COLS_TO_DROP_SUBSET3) # Start with base list
        # Add manual drops that are not already in COLS_TO_DROP_SUBSET3
        # for md_col in manual_drops_s3:
        #    if md_col not in cols_to_drop_very_initial :
        #        cols_to_drop_very_initial.append(md_col)
        # This is tricky because preprocess_data expects cols_to_drop_initial to be static.
        # Manual drops are better applied to the *processed* dataframe if they are from the collinearity review.
        # For now, manual_drops are only applied within identify_features_to_remove_subset3.

        if len(subset3_df_full) > SAMPLE_SIZE:
            print(f"[REDACTED_BY_SCRIPT]")
            subset3_df_to_process = subset3_df_full.sample(n=SAMPLE_SIZE, random_state=42).copy()
        else:
            subset3_df_to_process = subset3_df_full.copy()
        
        df_processed_s3, ohe_cols_s3 = preprocess_data(
            subset3_df_to_process, 
            COLS_TO_DROP_SUBSET3, # Pass the initial static list here
            INITIAL_CATEGORICAL_COLS_SUBSET3
        )
        
        print("[REDACTED_BY_SCRIPT]")
        df_processed_s3.fillna(-1, inplace=True)

        low_var_info_report_s3 = get_low_variance_features_report(df_processed_s3, LOW_VARIANCE_THRESHOLD)
        collinear_pairs_report_s3 = get_collinearity_report(df_processed_s3, HIGH_CORRELATION_THRESHOLD)

        print(f"[REDACTED_BY_SCRIPT]")
        reported_lv_count = 0
        for f, v in low_var_info_report_s3.items():
            # if f not in ohe_cols_s3 and v < LOW_VARIANCE_THRESHOLD : # General numerical
            #      print(f"  - {f}: {v:.6f}")
            #      reported_lv_count +=1
            # elif f in ohe_cols_s3 and v < OHE_LOW_VARIANCE_THRESHOLD_FOR_REMOVAL: # OHE specific (for removal)
            #      print(f"[REDACTED_BY_SCRIPT]")
            #      reported_lv_count +=1
            print(f"  - {f}: {v:.6f} {'(OHE)' if f in ohe_cols_s3 else ''}")
            reported_lv_count +=1
        if reported_lv_count == 0: print("[REDACTED_BY_SCRIPT]")


        print(f"[REDACTED_BY_SCRIPT]")
        if collinear_pairs_report_s3:
            for f1_rep, f2_rep, corr_rep in collinear_pairs_report_s3[:10]: # Print first 10
                print(f"[REDACTED_BY_SCRIPT]")
            if len(collinear_pairs_report_s3) > 10:
                print(f"[REDACTED_BY_SCRIPT]")
        else:
            print("[REDACTED_BY_SCRIPT]")


        print("[REDACTED_BY_SCRIPT]")
        features_to_drop_final_s3, reasons_for_dropping_s3, unresolved_pairs_s3 = identify_features_to_remove_subset3(
            df_processed_s3,
            ohe_cols_s3,
            low_var_info_report_s3,
            collinear_pairs_report_s3,
            manual_drops_s3 # Pass manual drops here
        )
        
        # Ensure only existing columns are attempted to be dropped
        features_to_drop_final_s3_existing = [col for col in features_to_drop_final_s3 if col in df_processed_s3.columns]
        
        df_final_sample_s3 = df_processed_s3.copy() 
        if features_to_drop_final_s3_existing:
            print(f"[REDACTED_BY_SCRIPT]")
            sorted_features_to_drop_s3 = sorted(list(set(features_to_drop_final_s3_existing))) 
            for col in sorted_features_to_drop_s3:
                 print(f"[REDACTED_BY_SCRIPT]'N/A')})")
            
            df_final_sample_s3.drop(columns=sorted_features_to_drop_s3, inplace=True, errors='ignore')
            print(f"[REDACTED_BY_SCRIPT]")
            print(f"[REDACTED_BY_SCRIPT]")
        else:
            print("[REDACTED_BY_SCRIPT]")
            
        print("[REDACTED_BY_SCRIPT]")
        print(f"[REDACTED_BY_SCRIPT]")
        print(df_final_sample_s3.head())

        try:
            df_final_sample_s3.to_csv(OUTPUT_FILENAME_S3, index=False)
            print(f"[REDACTED_BY_SCRIPT]")
            print(f"[REDACTED_BY_SCRIPT]")
            print(f"Review any 'Unresolved Highly Correlated Pairs'[REDACTED_BY_SCRIPT]")
        except Exception as e:
            print(f"[REDACTED_BY_SCRIPT]")