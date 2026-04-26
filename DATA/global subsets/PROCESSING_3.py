import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import re
import os

# --- Configuration for Subset 3 ---
FILE_PATH_SUBSET3 = "[REDACTED_BY_SCRIPT]"
OUTPUT_FULL_PROCESSED_FILENAME_S3 = "[REDACTED_BY_SCRIPT]"  # Changed to .parquet
MANUAL_DROPS_FILE_S3 = "[REDACTED_BY_SCRIPT]"

SAMPLE_SIZE = 50000
CHUNK_SIZE = 50000

# <<< IMPROVEMENT: Added COORDS_COLS list to handle them separately.
COORDS_COLS = ['pcd_eastings', 'pcd_northings', 'pcd_latitude', 'pcd_longitude']

COLS_TO_DROP_SUBSET3 = [
    'pcds', 'oa21cd', 'lsoa11cd', 'lsoa21cd', 'AREA_NAME', 'oa11cd',
    '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]',
    'dointr', 'doterm' # Handled by engineer_features_on_df
]
INITIAL_CATEGORICAL_COLS_SUBSET3 = ['MODE1_TYPE', 'MODE2_TYPE', 'usertype', 'is_terminated']

LOW_VARIANCE_THRESHOLD = 0.01
HIGH_CORRELATION_THRESHOLD = 0.95
OHE_LOW_VARIANCE_THRESHOLD_FOR_REMOVAL = 0.002
EXTREMELY_LOW_VARIANCE_THRESHOLD_FOR_AUTO_REMOVAL = 1e-9

def load_data_sample(file_path, sample_size_for_fitting):
    # This function is well-written, no changes needed.
    encodings_to_try = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
    for encoding in encodings_to_try:
        try:
            temp_df = pd.read_csv(file_path, low_memory=False, encoding=encoding, nrows=int(sample_size_for_fitting * 1.5))
            if len(temp_df) >= sample_size_for_fitting:
                df_sample = temp_df.sample(n=sample_size_for_fitting, random_state=42).copy()
            else:
                df_sample = temp_df.copy()
            print(f"[REDACTED_BY_SCRIPT]")
            return df_sample, encoding
        except (UnicodeDecodeError, FileNotFoundError, Exception) as e:
            print(f"[REDACTED_BY_SCRIPT]")
    print(f"[REDACTED_BY_SCRIPT]")
    return None, None

def load_manual_drops(manual_drops_file):
    # This function is well-written, no changes needed.
    try:
        with open(manual_drops_file, 'r') as f:
            manual_drops = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        if manual_drops:
            print(f"[REDACTED_BY_SCRIPT]")
        return manual_drops
    except FileNotFoundError:
        print(f"Manual drops file '{manual_drops_file}' not found.")
        return []

def engineer_features_on_df(df_input):
    """[REDACTED_BY_SCRIPT]"""
    df = df_input.copy()
    if 'doterm' in df.columns:
        df['is_terminated'] = df['doterm'].notna().astype(int)
    # The 'dointr' and 'doterm' columns are dropped via COLS_TO_DROP_SUBSET3
    return df

# <<< IMPROVEMENT: Combined multiple functions into one streamlined preprocessor creator.
def create_and_fit_preprocessor(df_sample, initial_categorical_cols, cols_to_drop):
    """
    Prepares sample data, defines pipelines, and fits a ColumnTransformer.
    Returns the fitted preprocessor and the prepared sample DataFrame used for fitting.
    """
    # 1. Prepare the data for fitting (feature engineering and initial drops)
    df_prepared = engineer_features_on_df(df_sample)
    df_prepared = df_prepared.drop(columns=cols_to_drop, errors='ignore')

    # 2. Identify column types from the prepared data
    all_numerical = df_prepared.select_dtypes(include=np.number).columns.tolist()
    coords_cols = [col for col in all_numerical if col in COORDS_COLS]
    numerical_cols = [col for col in all_numerical if col not in COORDS_COLS]
    categorical_cols = df_prepared.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Ensure explicitly defined categoricals are in the list and typed as string
    for col in initial_categorical_cols:
        if col in df_prepared.columns:
            if col not in categorical_cols:
                categorical_cols.append(col)
            if col in numerical_cols: # If a categorical was misidentified as numeric
                numerical_cols.remove(col)
            df_prepared[col] = df_prepared[col].astype(str)

    print(f"[REDACTED_BY_SCRIPT]")

    # 3. Define robust pipelines
    numerical_pipeline = Pipeline([
        ('imputer_num', SimpleImputer(strategy='median', add_indicator=True)),
        ('power_transform', PowerTransformer(method='yeo-johnson', standardize=False)),
        ('scaler_num', MinMaxScaler())
    ])
    coords_pipeline = Pipeline([
        ('imputer_coords', SimpleImputer(strategy='median', add_indicator=True))
    ])
    categorical_pipeline = Pipeline([
        ('imputer_cat', SimpleImputer(strategy='constant', fill_value='Missing_Category')),
        ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='if_binary'))
    ])

    # 4. Create and fit the preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_cols),
            ('cat', categorical_pipeline, categorical_cols),
            ('coords', coords_pipeline, coords_cols)
        ],
        remainder='passthrough',
        verbose_feature_names_out=False
    )
    preprocessor.set_output(transform="pandas")
    
    print(f"[REDACTED_BY_SCRIPT]")
    preprocessor.fit(df_prepared)
    
    return preprocessor, df_prepared

# Assuming get_low_variance_features_report, get_collinearity_report, 
# and identify_features_to_remove_subset3 are defined as in your original script.
# (Their logic is sound; they just need to receive correctly processed data).
def get_low_variance_features_report(df, threshold):
    variances = df.var(ddof=0)
    low_variance_info = {feature: variances[feature] for feature in variances[variances < threshold].index}
    return low_variance_info

def get_collinearity_report(df, threshold):
    numeric_df = df.copy()
    for col in numeric_df.columns:
        try:
            numeric_df[col] = pd.to_numeric(numeric_df[col], errors='coerce')
        except Exception:
            numeric_df = numeric_df.drop(columns=[col], errors='ignore')
    numeric_df = numeric_df.fillna(0)
    if numeric_df.empty or numeric_df.shape[1] == 0:
        return []
    corr_matrix = numeric_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    highly_correlated_pairs = []
    for column in upper.columns:
        correlated_features_in_col = upper.index[upper[column] > threshold].tolist()
        for feature in correlated_features_in_col:
            highly_correlated_pairs.append((feature, column, upper.loc[feature, column]))
    return highly_correlated_pairs

def identify_features_to_remove_subset3(df, ohe_feature_names, low_variance_info, collinear_pairs, manual_drops):
    cols_to_remove = set(manual_drops) 
    removal_reasons = {col: "[REDACTED_BY_SCRIPT]" for col in manual_drops}
    unresolved_collinear_pairs_for_review = []

    for feature, variance in low_variance_info.items():
        if feature in ohe_feature_names and variance < OHE_LOW_VARIANCE_THRESHOLD_FOR_REMOVAL:
            if feature not in cols_to_remove:
                 cols_to_remove.add(feature)
                 removal_reasons[feature] = f"[REDACTED_BY_SCRIPT]"
    
    for feature, variance in low_variance_info.items():
        if feature not in ohe_feature_names and variance < EXTREMELY_LOW_VARIANCE_THRESHOLD_FOR_AUTO_REMOVAL: # Using stricter threshold for auto-removal
            if feature not in cols_to_remove:
                cols_to_remove.add(feature)
                removal_reasons[feature] = f"[REDACTED_BY_SCRIPT]"

    kept_due_to_priority = set() 

    for f1, f2, corr_value in collinear_pairs:
        if f1 in cols_to_remove or f2 in cols_to_remove: 
            continue
        if f1 in kept_due_to_priority :
            if f2 not in cols_to_remove: # f1 kept, f2 is correlated to it
                 cols_to_remove.add(f2)
                 removal_reasons[f2] = f"[REDACTED_BY_SCRIPT]"
            continue # Move to next pair
        if f2 in kept_due_to_priority: # f2 kept, f1 is correlated to it
            if f1 not in cols_to_remove:
                 cols_to_remove.add(f1)
                 removal_reasons[f1] = f"[REDACTED_BY_SCRIPT]"
            continue # Move to next pair


        if {f1, f2} == {'LSOA_Area_Ha', 'LSOA_Shape_Area_sqm'}:
            if 'LSOA_Shape_Area_sqm' not in cols_to_remove:
                cols_to_remove.add('LSOA_Shape_Area_sqm')
                removal_reasons['LSOA_Shape_Area_sqm'] = f"[REDACTED_BY_SCRIPT]"
                kept_due_to_priority.add('LSOA_Area_Ha')
            continue

        avg_price_match_f1 = re.match(r"[REDACTED_BY_SCRIPT]", f1)
        median_price_match_f1 = re.match(r"[REDACTED_BY_SCRIPT]", f1)
        avg_price_match_f2 = re.match(r"[REDACTED_BY_SCRIPT]", f2)
        median_price_match_f2 = re.match(r"[REDACTED_BY_SCRIPT]", f2)

        if avg_price_match_f1 and median_price_match_f2 and avg_price_match_f1.group(1) == median_price_match_f2.group(1):
            if f1 not in cols_to_remove:
                cols_to_remove.add(f1); removal_reasons[f1] = f"[REDACTED_BY_SCRIPT]"
                kept_due_to_priority.add(f2)
            continue
        elif median_price_match_f1 and avg_price_match_f2 and median_price_match_f1.group(1) == avg_price_match_f2.group(1):
            if f2 not in cols_to_remove:
                cols_to_remove.add(f2); removal_reasons[f2] = f"[REDACTED_BY_SCRIPT]"
                kept_due_to_priority.add(f1)
            continue
        
        unresolved_collinear_pairs_for_review.append((f1, f2, corr_value))
            
    return list(cols_to_remove), removal_reasons, unresolved_collinear_pairs_for_review

# --- Main Execution ---
if __name__ == '__main__':
    print("[REDACTED_BY_SCRIPT]")
    sample_df, successful_encoding = load_data_sample(FILE_PATH_SUBSET3, SAMPLE_SIZE)

    if sample_df is None:
        exit()

    manual_drops_s3 = load_manual_drops(MANUAL_DROPS_FILE_S3)

    # 1. Create, prepare, and fit the preprocessor in one step
    preprocessor, df_prepared_for_fit = create_and_fit_preprocessor(
        sample_df, INITIAL_CATEGORICAL_COLS_SUBSET3, COLS_TO_DROP_SUBSET3
    )

    # 2. Transform the prepared sample data to get a DataFrame for analysis
    print("[REDACTED_BY_SCRIPT]")
    df_processed_sample_for_analysis = preprocessor.transform(df_prepared_for_fit)
    print(f"[REDACTED_BY_SCRIPT]")

    # <<< IMPROVEMENT: Removed fillna(-1) and added a proper NaN check.
    if df_processed_sample_for_analysis.isna().values.any():
        print("[REDACTED_BY_SCRIPT]")
        nan_cols = df_processed_sample_for_analysis.columns[df_processed_sample_for_analysis.isna().any()].tolist()
        print(f"[REDACTED_BY_SCRIPT]")
        print("[REDACTED_BY_SCRIPT]")

    # 3. Analyze processed sample to determine final columns to keep
    ohe_cols_s3 = preprocessor.named_transformers_['cat'].get_feature_names_out().tolist()

    low_var_info_report_s3 = get_low_variance_features_report(df_processed_sample_for_analysis, LOW_VARIANCE_THRESHOLD)
    collinear_pairs_report_s3 = get_collinearity_report(df_processed_sample_for_analysis, HIGH_CORRELATION_THRESHOLD)
    
    # Assuming identify_features_to_remove_subset3 is fully defined
    features_to_drop_from_analysis, reasons_s3, unresolved_s3 = identify_features_to_remove_subset3(
        df_processed_sample_for_analysis, ohe_cols_s3, low_var_info_report_s3, collinear_pairs_report_s3, manual_drops_s3
    )
    
    all_processed_cols = df_processed_sample_for_analysis.columns.tolist()
    final_columns_to_keep = [col for col in all_processed_cols if col not in features_to_drop_from_analysis]
    
    if not final_columns_to_keep:
        print("[REDACTED_BY_SCRIPT]")
        exit()
        
    print(f"[REDACTED_BY_SCRIPT]")
    # (Rest of your reporting and review logic)

    # --- Phase 2: Process Full Dataset in Chunks ---
    print("[REDACTED_BY_SCRIPT]")
    
    # Remove the file deletion logic since we're not appending chunks anymore
    # Store all processed chunks in a list for concatenation
    processed_chunks = []

    chunk_iter_full = pd.read_csv(FILE_PATH_SUBSET3, chunksize=CHUNK_SIZE, low_memory=False, encoding=successful_encoding)
    
    for i, chunk_df_raw in enumerate(chunk_iter_full):
        print(f"[REDACTED_BY_SCRIPT]")
        
        # <<< IMPROVEMENT: Consistent preparation and transformation flow
        chunk_df_prepared = engineer_features_on_df(chunk_df_raw)
        chunk_df_prepared = chunk_df_prepared.drop(columns=COLS_TO_DROP_SUBSET3, errors='ignore')
        
        chunk_df_transformed = preprocessor.transform(chunk_df_prepared)
        chunk_df_final_cols = chunk_df_transformed[final_columns_to_keep]
        
        # Append chunk to list instead of writing to CSV
        processed_chunks.append(chunk_df_final_cols)
        print(f"[REDACTED_BY_SCRIPT]")

    # Concatenate all chunks and save as Parquet
    print("[REDACTED_BY_SCRIPT]")
    final_df = pd.concat(processed_chunks, ignore_index=True)
    
    print(f"[REDACTED_BY_SCRIPT]")
    final_df.to_parquet(OUTPUT_FULL_PROCESSED_FILENAME_S3, index=False, engine='pyarrow')
    
    print(f"[REDACTED_BY_SCRIPT]")
    print(f"[REDACTED_BY_SCRIPT]")