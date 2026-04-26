import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import re

# --- Configuration ---
FILE_PATH_SUBSET1 = "[REDACTED_BY_SCRIPT]"  # Replace with the actual path to your subset1 file
OUTPUT_PROCESSED_FULL_FILE = "[REDACTED_BY_SCRIPT]"  # Changed to .parquet
CHUNK_SIZE = 50000  # Number of rows per chunk for processing the full file
N_ROWS_FOR_FITTING = 100000 # Number of rows to use for fitting transformers and identifying features (adjust based on memory and representativeness)

# Feature lists and thresholds (same as before)
COORDS_COLS = ['pcd_eastings', 'pcd_northings', 'pcd_latitude', 'pcd_longitude']
COLS_TO_DROP_INITIAL = [ # Renamed to avoid confusion
    'pcds_original','pcd7', 'pcd8', 'pcds',
    '[REDACTED_BY_SCRIPT]', # Added this based on previous debugging
    'oa21cd', 'lsoa21cd', 'msoa21cd', 'ladcd',
    'lsoa21nm', 'msoa21nm', 'ladnm', 'ladnmw',
    '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]', 'OA21_GROUP_NAME', 'OA21_SUBGROUP_NAME',
    'oa11cd', 'wz11cd', 'lsoa11cd', 'msoa11cd', 'ladcd_from_wz_file',
    'MSOA21_RUC_NAME',
    'dointr', 'doterm' # Will be handled by creating 'is_terminated' then dropped
]
INITIAL_CATEGORICAL_COLS = [ # Renamed
    'usertype', '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]', 'OA21_GROUP_CODE', 'OA21_SUBGROUP_CODE',
    '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', 'MS[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]', 'WZ11_GROUP_NAME',
    'MSOA21_RUC_CODE', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
    'is_terminated' # Will be created
]
LOW_VARIANCE_THRESHOLD_REPORT = 0.01 # For initial reporting
HIGH_CORRELATION_THRESHOLD_REPORT = 0.95 # For initial reporting
OHE_LOW_VARIANCE_THRESHOLD_FOR_REMOVAL = 0.002


def load_data_sample(file_path, n_rows):
    encodings_to_try = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
    for encoding in encodings_to_try:
        try:
            df = pd.read_csv(file_path, low_memory=False, encoding=encoding, nrows=n_rows)
            print(f"[REDACTED_BY_SCRIPT]")
            return df
        except UnicodeDecodeError: continue
        except FileNotFoundError: return None
    return None

def initial_feature_engineering_and_typing(df, initial_categorical_cols_list):
    """[REDACTED_BY_SCRIPT]"""
    df_processed = df.drop(columns=COLS_TO_DROP_INITIAL, errors='ignore')
    
    # Handle 'doterm' by checking if it was in original df before trying to access
    if 'doterm' in df.columns: # df refers to the original chunk here
         df_processed['is_terminated'] = df['doterm'].notna().astype(int)
    elif 'is_terminated' not in df_processed.columns: # If doterm was already dropped but is_terminated not made
         df_processed['is_terminated'] = 0 # Default if doterm was missing entirely

    # Convert initial categorical columns to string type
    for col in initial_categorical_cols_list:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].astype(str)
    return df_processed

def get_fitted_preprocessor_and_features(df_sample, initial_categorical_cols_list):
    """
    Fits the ColumnTransformer on the sample data and returns the fitted preprocessor
    and the list of all feature names it would generate.
    """
    df_sample_processed = initial_feature_engineering_and_typing(df_sample, initial_categorical_cols_list)

    # --- UPDATE: Explicitly ISOLATE coordinates from the start ---
    # This is the core of the fix. We remove the coordinate columns from the main dataframe
    # before any other processing happens. This makes it impossible for them to be sent
    # down the wrong pipeline.
    
    # 1. Keep a separate, clean copy of the coordinates
    coords_df = df_sample_processed[COORDS_COLS].copy()

    # 2. Drop them from the main processing dataframe
    df_sample_for_pipelines = df_sample_processed.drop(columns=COORDS_COLS)

    # Now, we derive the numerical and categorical lists from this "safe" dataframe
    numerical_cols = df_sample_for_pipelines.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df_sample_for_pipelines.select_dtypes(include=['object', 'category']).columns.tolist()

    # --- (The '[REDACTED_BY_SCRIPT]' check can remain the same, it will now operate on the safe df) ---
    all_cols_set = set(df_sample_for_pipelines.columns)
    processed_cols_set = set(numerical_cols + categorical_cols)
    if all_cols_set != processed_cols_set:
        missing_from_typed_lists = all_cols_set - processed_cols_set
        print(f"[REDACTED_BY_SCRIPT]")
        for m_col in missing_from_typed_lists:
            if m_col not in categorical_cols: categorical_cols.append(m_col)
            df_sample_for_pipelines[m_col] = df_sample_for_pipelines[m_col].astype(str)
    
    print(f"[REDACTED_BY_SCRIPT]")

    # Pipeline for standard numerical features (this is now safe)
    numerical_pipeline = Pipeline([
        ('imputer_num', SimpleImputer(strategy='median', add_indicator=True)),
        ('power_transform', PowerTransformer(method='yeo-johnson', standardize=False)),
        ('scaler_num', MinMaxScaler())
    ])

    # Pipeline for categorical features
    categorical_pipeline = Pipeline([
        ('imputer_cat', SimpleImputer(strategy='constant', fill_value='Missing_Category')),
        ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='if_binary'))
    ])

    # --- UPDATE: The ColumnTransformer no longer processes coordinates directly ---
    # It only handles the numerical and categorical columns. The coordinates will be
    # re-attached later.
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_cols),
            ('cat', categorical_pipeline, categorical_cols)
        ],
        remainder='passthrough',
        verbose_feature_names_out=False
    )

    print("[REDACTED_BY_SCRIPT]")
    # --- UPDATE: Fit on the dataframe that has had coordinates removed ---
    preprocessor.fit(df_sample_for_pipelines)

    all_generated_feature_names = preprocessor.get_feature_names_out()

    # Get OHE names (logic is the same, but applied to the fitted preprocessor)
    cat_transformer = preprocessor.named_transformers_['cat']
    ohe_step = cat_transformer.named_steps['one_hot_encoder']
    original_cat_cols = preprocessor.transformers_[1][2]
    ohe_feature_names = ohe_step.get_feature_names_out(original_cat_cols).tolist()
    
    # --- UPDATE: The list of ALL final features now must include the coordinate columns ---
    final_feature_list = all_generated_feature_names.tolist() + COORDS_COLS

    return preprocessor, final_feature_list, ohe_feature_names

# (get_low_variance_features_report and get_collinearity_report functions remain the same)
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
            highly_correlated_pairs.append((feature, column, upper.loc[feature, column]))
    return highly_correlated_pairs

# (identify_features_to_remove function remains largely the same, ensures it uses correct df columns)
def identify_features_to_remove(df_processed_sample, ohe_feature_names_from_sample, low_variance_info, collinear_pairs):
    cols_to_remove = set()
    removal_reasons = {}

    user_fixed_low_var_removals = [
        '[REDACTED_BY_SCRIPT]', 'FI_OA_Rarity_in_LAD',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]'
    ]
    for col in user_fixed_low_var_removals:
        if col in df_processed_sample.columns: # Check against processed sample columns
            cols_to_remove.add(col)
            removal_reasons[col] = "[REDACTED_BY_SCRIPT]"

    for feature, variance in low_variance_info.items():
        if feature in ohe_feature_names_from_sample and variance < OHE_LOW_VARIANCE_THRESHOLD_FOR_REMOVAL:
            if feature not in cols_to_remove and feature in df_processed_sample.columns:
                 cols_to_remove.add(feature)
                 removal_reasons[feature] = f"[REDACTED_BY_SCRIPT]"
    
    nan_hierarchies = [
        ('[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]'),
        ('[REDACTED_BY_SCRIPT]', 'OA21_GROUP_CODE_nan'),
        ('OA21_GROUP_CODE_nan', '[REDACTED_BY_SCRIPT]'),
        ('[REDACTED_BY_SCRIPT]', 'WZ11_GROUP_NAME_nan'),
        ('MSOA21_RUC_CODE_nan', '[REDACTED_BY_SCRIPT]')
    ]
    for broader_nan, narrower_nan in nan_hierarchies:
        for f1, f2, corr in collinear_pairs: # collinear_pairs from processed_sample
            if {f1, f2} == {broader_nan, narrower_nan}:
                if narrower_nan in df_processed_sample.columns and narrower_nan not in cols_to_remove:
                    cols_to_remove.add(narrower_nan)
                    removal_reasons[narrower_nan] = f"[REDACTED_BY_SCRIPT]"
                break
            if broader_nan == 'OA21_GROUP_CODE_nan' and narrower_nan == '[REDACTED_BY_SCRIPT]':
                 if {f1,f2} == {'[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]'}: # Check direct SUPER to SUB
                      if '[REDACTED_BY_SCRIPT]' in df_processed_sample.columns and '[REDACTED_BY_SCRIPT]' not in cols_to_remove:
                           cols_to_remove.add('[REDACTED_BY_SCRIPT]')
                           removal_reasons['[REDACTED_BY_SCRIPT]'] = f"[REDACTED_BY_SCRIPT]"
                      break
    
    raw_wz_cols_to_drop = set()
    for f1, f2, corr in collinear_pairs:
        is_f1_raw = f1.startswith("[REDACTED_BY_SCRIPT]")
        is_f2_raw = f2.startswith("[REDACTED_BY_SCRIPT]")
        is_f1_grp = f1.startswith("WZ11_GROUP_NAME_")
        is_f2_grp = f2.startswith("WZ11_GROUP_NAME_")

        if (is_f1_raw and is_f2_grp) or (is_f2_raw and is_f1_grp):
            raw_col = f1 if is_f1_raw else f2
            grp_col = f2 if is_f1_raw else f1
            suffix_raw = raw_col.replace("[REDACTED_BY_SCRIPT]", "")
            suffix_grp = grp_col.replace("WZ11_GROUP_NAME_", "")
            if suffix_raw == suffix_grp:
                if raw_col in df_processed_sample.columns and raw_col not in cols_to_remove:
                    raw_wz_cols_to_drop.add(raw_col)
                    removal_reasons[raw_col] = f"[REDACTED_BY_SCRIPT]"
    cols_to_remove.update(raw_wz_cols_to_drop)

    # Simplified "Pseudo" feature removal: assume OHE names are direct if no special chars
    pseudo_oa11 = next((col for col in df_processed_sample.columns if "[REDACTED_BY_SCRIPT]" in col and "Pseudo" in col), None)
    if pseudo_oa11:
        pseudo_others_to_drop = []
        for col_prefix in ["[REDACTED_BY_SCRIPT]", "[REDACTED_BY_SCRIPT]"]:
            potential_pseudo = next((col for col in df_processed_sample.columns if col_prefix in col and "Pseudo" in col), None)
            if potential_pseudo: pseudo_others_to_drop.append(potential_pseudo)
        
        for f1, f2, corr in collinear_pairs:
            for other_pseudo in pseudo_others_to_drop:
                if ({f1, f2} == {pseudo_oa11, other_pseudo}) and \
                   (other_pseudo in df_processed_sample.columns and other_pseudo not in cols_to_remove):
                    cols_to_remove.add(other_pseudo)
                    removal_reasons[other_pseudo] = f"[REDACTED_BY_SCRIPT]"
    
    lad_oac_indices_to_drop = set()
    lad_pattern = re.compile(r"[REDACTED_BY_SCRIPT]") # Updated pattern for robustness

    # Create a dictionary of features and their types/codes for easier lookup
    lad_features = {}
    for col_name in df_processed_sample.columns:
        match = lad_pattern.match(col_name)
        if match:
            lad_features[col_name] = {"type": match.group(1), "code": match.group(2)}

    for f1_orig, f2_orig, corr in collinear_pairs:
        if f1_orig in lad_features and f2_orig in lad_features:
            feat1_info = lad_features[f1_orig]
            feat2_info = lad_features[f2_orig]

            # Rule: SUB vs GRP - if SUB code starts with GRP code, GRP is parent
            if feat1_info["type"] == "SUB" and feat2_info["type"] == "GRP" and feat1_info["code"].startswith(feat2_info["code"]):
                if f2_orig not in cols_to_remove: lad_oac_indices_to_drop.add(f2_orig)
            elif feat2_info["type"] == "SUB" and feat1_info["type"] == "GRP" and feat2_info["code"].startswith(feat1_info["code"]):
                if f1_orig not in cols_to_remove: lad_oac_indices_to_drop.add(f1_orig)
            
            # Rule: GRP vs SG - if GRP code starts with SG code (SG code is just number, e.g., '1' for SG1)
            elif feat1_info["type"] == "GRP" and feat2_info["type"] == "SG" and feat1_info["code"].startswith(feat2_info["code"]):
                 if f1_orig not in lad_oac_indices_to_drop and f2_orig not in cols_to_remove: # Don't drop GRP if it's already a target from SUB
                    lad_oac_indices_to_drop.add(f2_orig)
            elif feat2_info["type"] == "GRP" and feat1_info["type"] == "SG" and feat2_info["code"].startswith(feat1_info["code"]):
                 if f2_orig not in lad_oac_indices_to_drop and f1_orig not in cols_to_remove:
                    lad_oac_indices_to_drop.add(f1_orig)
    
    for col_to_drop in lad_oac_indices_to_drop:
        if col_to_drop not in cols_to_remove:
            cols_to_remove.add(col_to_drop)
            removal_reasons[col_to_drop] = "[REDACTED_BY_SCRIPT]"
            
    return list(cols_to_remove), removal_reasons


def process_full_data_in_chunks(fitted_preprocessor, final_columns_to_keep, 
                                input_file_path, output_file_path, chunk_size, 
                                initial_categorical_cols_list):
    print(f"[REDACTED_BY_SCRIPT]'{input_file_path}'[REDACTED_BY_SCRIPT]")
    
    processed_chunks = []
    
    temp_initial_categorical_cols = list(initial_categorical_cols_list)
    if 'is_terminated' not in temp_initial_categorical_cols:
        temp_initial_categorical_cols.append('is_terminated')

    for chunk_num, df_chunk_raw in enumerate(pd.read_csv(input_file_path, chunksize=chunk_size, low_memory=False)):
        print(f"[REDACTED_BY_SCRIPT]")
        df_chunk_prepared = initial_feature_engineering_and_typing(df_chunk_raw, temp_initial_categorical_cols)
        
        # --- UPDATE: Apply the same coordinate isolation logic to each chunk ---
        # 1. Keep a separate, clean copy of the coordinates for this chunk
        coords_chunk_df = df_chunk_prepared[COORDS_COLS].copy()

        # 2. Drop them from the main processing dataframe for this chunk
        df_chunk_for_pipelines = df_chunk_prepared.drop(columns=COORDS_COLS)

        # 3. Transform the main data (which now contains NO coordinates)
        df_chunk_transformed_array = fitted_preprocessor.transform(df_chunk_for_pipelines)

        # 4. Convert the transformed data back to a DataFrame
        df_chunk_transformed = pd.DataFrame(
            df_chunk_transformed_array,
            columns=fitted_preprocessor.get_feature_names_out()
        )

        # 5. Reset index on both dataframes before joining to ensure alignment
        df_chunk_transformed.reset_index(drop=True, inplace=True)
        coords_chunk_df.reset_index(drop=True, inplace=True)

        # 6. Re-attach the original, untouched coordinates
        df_chunk_with_coords = pd.concat([df_chunk_transformed, coords_chunk_df], axis=1)

        # 7. Now, select the final columns from the fully reassembled dataframe
        # This ensures that the coordinate columns are present and correct.
        df_chunk_final_cols = df_chunk_with_coords[final_columns_to_keep]
        
        processed_chunks.append(df_chunk_final_cols)
        print(f"[REDACTED_BY_SCRIPT]")

    print("[REDACTED_BY_SCRIPT]")
    final_df = pd.concat(processed_chunks, ignore_index=True)
    
    print(f"[REDACTED_BY_SCRIPT]")
    # --- UPDATE: Add a final check to handle potential NaN values in coordinate columns ---
    # This is a defensive measure. If any coordinates were missing, we fill with 0.
    for col in COORDS_COLS:
        if col in final_df.columns:
            final_df[col] = final_df[col].fillna(0)
            
    final_df.to_parquet(output_file_path, index=False, engine='pyarrow')
    
    print(f"[REDACTED_BY_SCRIPT]'{output_file_path}'")
    print(f"[REDACTED_BY_SCRIPT]")

# --- Main Execution ---
if __name__ == '__main__':
    # 1. Load sample, fit preprocessor, identify features to keep/drop
    print(f"[REDACTED_BY_SCRIPT]")
    df_sample_for_fitting = load_data_sample(FILE_PATH_SUBSET1, N_ROWS_FOR_FITTING)

    if df_sample_for_fitting is not None:
        # Create a copy of INITIAL_CATEGORICAL_COLS for this phase
        initial_cat_cols_for_fit = list(INITIAL_CATEGORICAL_COLS)
        
        fitted_preprocessor, all_gen_names, ohe_names_from_sample = get_fitted_preprocessor_and_features(
            df_sample_for_fitting.copy(), # Use a copy for fitting
            initial_cat_cols_for_fit
        )
        
        # To run checks, we need a processed version of the sample
        print("[REDACTED_BY_SCRIPT]")
        temp_df_sample_prepared_for_transform = initial_feature_engineering_and_typing(df_sample_for_fitting.copy(), initial_cat_cols_for_fit)
        
        # --- FIX: Isolate coordinates before transforming the sample for checks ---
        # This mirrors the logic in the chunk processing function.
        # The preprocessor was not fitted on coordinate columns, so they must be removed before transform.
        df_sample_for_pipelines_check = temp_df_sample_prepared_for_transform.drop(columns=COORDS_COLS, errors='ignore')
        coords_sample_df_check = temp_df_sample_prepared_for_transform[COORDS_COLS].copy()

        df_processed_sample_for_checks_array = fitted_preprocessor.transform(df_sample_for_pipelines_check)

        # --- REVISED SECTION ---
        # Check for NaNs immediately after transformation
        if np.isnan(df_processed_sample_for_checks_array).any():
            print("[REDACTED_BY_SCRIPT]")
            # Find which columns are causing NaNs
            nan_cols_indices = np.where(np.isnan(df_processed_sample_for_checks_array).any(axis=0))[0]
            nan_cols_names = [all_gen_names[i] for i in nan_cols_indices]
            print(f"[REDACTED_BY_SCRIPT]")
            print("[REDACTED_BY_SCRIPT]")
            print("[REDACTED_BY_SCRIPT]")
            # You might want to halt execution here to fix the root cause
            # For now, we'll convert to a DataFrame but be aware of the issue.

        # --- FIX: Create DataFrame from the transformed part first ---
        df_processed_sample_for_checks = pd.DataFrame(
            df_processed_sample_for_checks_array,
            columns=fitted_preprocessor.get_feature_names_out()
        )

        # --- FIX: Re-attach the coordinate columns to the checked sample ---
        # This ensures the dataframe used for reporting has the same structure as the final output.
        df_processed_sample_for_checks.reset_index(drop=True, inplace=True)
        coords_sample_df_check.reset_index(drop=True, inplace=True)
        df_processed_sample_for_checks = pd.concat([df_processed_sample_for_checks, coords_sample_df_check], axis=1)


        print("[REDACTED_BY_SCRIPT]")
        low_var_info = get_low_variance_features_report(df_processed_sample_for_checks, LOW_VARIANCE_THRESHOLD_REPORT)
        collinear_pairs = get_collinearity_report(df_processed_sample_for_checks, HIGH_CORRELATION_THRESHOLD_REPORT)
        
        features_to_drop_identified, reasons = identify_features_to_remove(
            df_processed_sample_for_checks, ohe_names_from_sample, low_var_info, collinear_pairs
        )
        
        final_columns_to_keep = [col for col in all_gen_names if col not in features_to_drop_identified]
        
        # --- FIX: Also add back coordinate columns to the final list to keep ---
        # This ensures they are not accidentally dropped.
        for col in COORDS_COLS:
            if col not in final_columns_to_keep:
                final_columns_to_keep.append(col)

        print(f"[REDACTED_BY_SCRIPT]")
        print(f"[REDACTED_BY_SCRIPT]")
        if features_to_drop_identified:
             print("[REDACTED_BY_SCRIPT]")
             for i, col in enumerate(features_to_drop_identified[:5]):
                  print(f"[REDACTED_BY_SCRIPT]'N/A')})")


        # 2. Process full data in chunks using fitted preprocessor and selected features
        process_full_data_in_chunks(
            fitted_preprocessor,
            final_columns_to_keep,
            FILE_PATH_SUBSET1,
            OUTPUT_PROCESSED_FULL_FILE,
            CHUNK_SIZE,
            INITIAL_CATEGORICAL_COLS # Pass original list for consistency
        )
    else:
        print(f"[REDACTED_BY_SCRIPT]")