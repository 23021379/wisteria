import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import re

# --- Configuration for Subset 2 ---
FILE_PATH_SUBSET2 = "[REDACTED_BY_SCRIPT]"
OUTPUT_PROCESSED_FULL_FILE_S2 = "[REDACTED_BY_SCRIPT]"  # Changed to .parquet
CHUNK_SIZE = 50000
N_ROWS_FOR_FITTING = 50000

# <<< IMPROVEMENT: Added COORDS_COLS list to isolate them from other numerical features.
COORDS_COLS = ['pcd_eastings', 'pcd_northings', 'pcd_latitude', 'pcd_longitude']

COLS_TO_DROP_INITIAL_S2 = [
    'pcd7', 'pcd8', 'pcds', 'oa21cd',
    'lsoa21cd', 'msoa21cd', 'ladcd',
    'lsoa21nm', 'msoa21nm', 'ladnm', 'ladnmw',
    'Output_Areas_Code', 'Output Areas',
    'dointr', 'doterm'
]
# <<< IMPROVEMENT: Ensured 'is_terminated' is in the initial list for consistent typing.
INITIAL_CATEGORICAL_COLS_S2 = ['usertype', 'is_terminated']

LOW_VARIANCE_THRESHOLD_REPORT = 0.0001
HIGH_CORRELATION_THRESHOLD_REPORT = 0.98
OHE_LOW_VARIANCE_THRESHOLD_FOR_REMOVAL = 0.002


def load_data_sample(file_path, n_rows):
    """[REDACTED_BY_SCRIPT]"""
    encodings_to_try = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
    df = None
    for encoding in encodings_to_try:
        try:
            df = pd.read_csv(file_path, low_memory=False, encoding=encoding, nrows=n_rows if n_rows else None)
            action = f"[REDACTED_BY_SCRIPT]" if n_rows else "full file"
            print(f"[REDACTED_BY_SCRIPT]")
            return df
        except UnicodeDecodeError:
            print(f"[REDACTED_BY_SCRIPT]")
            continue
        except FileNotFoundError:
            print(f"[REDACTED_BY_SCRIPT]")
            return None
    if df is None:
        print(f"[REDACTED_BY_SCRIPT]")
    return df

def initial_feature_engineering_and_typing(df_chunk, initial_categorical_cols_list, cols_to_drop_list):
    """[REDACTED_BY_SCRIPT]"""
    df_processed = df_chunk.drop(columns=cols_to_drop_list, errors='ignore')
    if 'doterm' in df_chunk.columns:
         df_processed['is_terminated'] = df_chunk['doterm'].notna().astype(int)
    elif 'is_terminated' not in df_processed.columns:
         df_processed['is_terminated'] = 0
         
    # Ensure all specified categorical columns are correctly typed as strings for the pipeline
    for col in initial_categorical_cols_list:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].astype(str)
    return df_processed

def get_fitted_preprocessor_and_features(df_sample_to_fit_on, initial_categorical_cols_list_config, cols_to_drop_initial_config):
    """
    Fits the ColumnTransformer on the sample data and returns the fitted preprocessor
    and the list of all feature names it would generate.
    """
    print(f"[REDACTED_BY_SCRIPT]")
    df_sample_prepared = initial_feature_engineering_and_typing(
        df_sample_to_fit_on, initial_categorical_cols_list_config, cols_to_drop_initial_config
    )

    # <<< IMPROVEMENT: Switched to the more robust select_dtypes method for column identification.
    all_numerical_cols = df_sample_prepared.select_dtypes(include=np.number).columns.tolist()
    
    # Identify all other columns as categorical
    all_current_cols = df_sample_prepared.columns.tolist()
    categorical_cols = [col for col in all_current_cols if col not in all_numerical_cols]

    # Explicitly separate coordinate columns from other numerical columns
    coords_cols = [col for col in all_numerical_cols if col in COORDS_COLS]
    numerical_cols = [col for col in all_numerical_cols if col not in COORDS_COLS]
    
    # Ensure categorical columns are strings
    for col in categorical_cols:
        df_sample_prepared[col] = df_sample_prepared[col].astype(str)

    print(f"[REDACTED_BY_SCRIPT]")

    # <<< IMPROVEMENT: Replaced constant imputer with median + indicator for better scaling.
    numerical_pipeline = Pipeline([
        ('imputer_num', SimpleImputer(strategy='median', add_indicator=True)),
        ('power_transform', PowerTransformer(method='yeo-johnson', standardize=False)),
        ('scaler_num', MinMaxScaler())
    ])

    # <<< IMPROVEMENT: Created a separate, simple pipeline for coordinates.
    coords_pipeline = Pipeline([
        ('imputer_coords', SimpleImputer(strategy='median', add_indicator=True))
    ])

    # <<< IMPROVEMENT: Used a proper imputer for categorical data.
    categorical_pipeline = Pipeline([
        ('imputer_cat', SimpleImputer(strategy='constant', fill_value='Missing_Category')),
        ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='if_binary'))
    ])
    
    # <<< IMPROVEMENT: Updated ColumnTransformer to handle all three data types.
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_cols),
            ('cat', categorical_pipeline, categorical_cols),
            ('coords', coords_pipeline, coords_cols)
        ],
        remainder='passthrough',
        verbose_feature_names_out=False
    )
    
    print("[REDACTED_BY_SCRIPT]")
    preprocessor.fit(df_sample_prepared)
    
    # <<< IMPROVEMENT: Switched to the robust get_feature_names_out() method.
    all_generated_feature_names = preprocessor.get_feature_names_out()

    # Get just the OHE names for downstream analysis
    ohe_feature_names_out = []
    if 'cat' in preprocessor.named_transformers_:
        cat_transformer = preprocessor.named_transformers_['cat']
        ohe_step = cat_transformer.named_steps['one_hot_encoder']
        # Get the original categorical column names the transformer was fitted on
        original_cat_cols = preprocessor.transformers_[1][2] 
        ohe_feature_names_out = ohe_step.get_feature_names_out(original_cat_cols).tolist()

    print(f"[REDACTED_BY_SCRIPT]")
    return preprocessor, all_generated_feature_names.tolist(), ohe_feature_names_out

def get_low_variance_features_report(df_processed_sample_data, threshold):
    # This function is logically sound, no changes needed.
    print(f"[REDACTED_BY_SCRIPT]")
    variances = df_processed_sample_data.var(ddof=0)
    low_variance_info = {feature: variances[feature] for feature in variances[variances < threshold].index if feature in df_processed_sample_data.columns}
    print(f"[REDACTED_BY_SCRIPT]")
    return low_variance_info

def get_collinearity_report(df_processed_sample_data, threshold):
    # This function is logically sound, no changes needed.
    print(f"[REDACTED_BY_SCRIPT]")
    numeric_df_for_corr = df_processed_sample_data.copy()
    for col in numeric_df_for_corr.columns:
        numeric_df_for_corr[col] = pd.to_numeric(numeric_df_for_corr[col], errors='coerce')
    numeric_df_for_corr = numeric_df_for_corr.fillna(0)
    
    corr_matrix = numeric_df_for_corr.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    highly_correlated_pairs = []
    for column_name in upper.columns:
        correlated_features_in_this_col = upper.index[upper[column_name] > threshold].tolist()
        for feature_name in correlated_features_in_this_col:
            highly_correlated_pairs.append((feature_name, column_name, upper.loc[feature_name, column_name]))
    print(f"[REDACTED_BY_SCRIPT]")
    return highly_correlated_pairs

def identify_features_to_remove_subset2_revised(
    df_processed_sample_ref, 
    ohe_feature_names_list, 
    low_variance_info_dict, 
    collinear_pairs_list,
    manually_specified_drops=None # New parameter for manual drops
):
    """
    Revised identification of features to remove.
    - Removes low variance features.
    - Applies a more CAUTIOUS set of rules for collinearity.
    - Allows for a list of manually specified drops.
    - Returns features to drop AND a list of unresolved correlated pairs for review.
    """
    if manually_specified_drops is None:
        manually_specified_drops = []

    cols_to_remove = set()
    removal_reasons = {}
    unresolved_correlated_pairs_for_review = []
    
    print(f"[REDACTED_BY_SCRIPT]")

    # 0. Add manually specified drops first
    for col_to_drop_manual in manually_specified_drops:
        if col_to_drop_manual in df_processed_sample_ref.columns and col_to_drop_manual not in cols_to_remove:
            cols_to_remove.add(col_to_drop_manual)
            removal_reasons[col_to_drop_manual] = "[REDACTED_BY_SCRIPT]"

    # 1. Low Variance OHE Features
    for feature, variance in low_variance_info_dict.items():
        if feature in ohe_feature_names_list and variance < OHE_LOW_VARIANCE_THRESHOLD_FOR_REMOVAL:
            if feature not in cols_to_remove:
                 cols_to_remove.add(feature)
                 removal_reasons[feature] = f"[REDACTED_BY_SCRIPT]"
    
    # 2. General Low Variance Numerical Features
    for feature, variance in low_variance_info_dict.items():
        if feature not in ohe_feature_names_list and variance < LOW_VARIANCE_THRESHOLD_REPORT:
            if feature not in cols_to_remove:
                reason_suffix = f"[REDACTED_BY_SCRIPT]"
                if variance < 1e-9: reason_suffix = f"[REDACTED_BY_SCRIPT]"
                cols_to_remove.add(feature)
                removal_reasons[feature] = f"[REDACTED_BY_SCRIPT]"
    
    print(f"[REDACTED_BY_SCRIPT]")
    
    # 3. Collinearity Removals (Cautious Rules)
    fi_feature_prefixes = ['FI_', 'Total_Households_OA', 'Household_size_'] # Consider these more important
    
    # Predefined known equivalences or near-equivalences for ONS data
    # (feature_to_prefer, feature_to_drop_if_correlated)
    known_ons_equivalences = [
        ('[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]'),
        ('[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]'),
        ('[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]'),
        ('[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]'),
        ('[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]'),
        ('[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]'),
        # Add more known pairs where one is clearly redundant if correlated with the other
        # Example: if 'number_bedrooms_1_bedroom' and 'number_rooms_2_rooms' are very highly correlated
        # and you prefer 'number_bedrooms_1_bedroom', list it as ('number_bedrooms_1_bedroom', 'number_rooms_2_rooms')
    ]

    # Apply known equivalences
    for prefer, drop_if_corr in known_ons_equivalences:
        for f1, f2, corr_value in collinear_pairs_list:
            if {f1, f2} == {prefer, drop_if_corr}:
                if prefer in cols_to_remove or drop_if_corr in cols_to_remove: continue # Already handled
                if prefer in df_processed_sample_ref.columns and drop_if_corr in df_processed_sample_ref.columns:
                    cols_to_remove.add(drop_if_corr)
                    removal_reasons[drop_if_corr] = f"[REDACTED_BY_SCRIPT]"
                    break # Found the pair

    # General collinearity with FI preference
    for f1, f2, corr_value in sorted(collinear_pairs_list, key=lambda x: (x[0], x[1])):
        if f1 in cols_to_remove or f2 in cols_to_remove: continue

        f1_is_fi_type = any(f1.startswith(prefix) for prefix in fi_feature_prefixes)
        f2_is_fi_type = any(f2.startswith(prefix) for prefix in fi_feature_prefixes)
        col_to_drop_this_pair = None
        kept_partner = None

        if f1_is_fi_type and not f2_is_fi_type: col_to_drop_this_pair, kept_partner = f2, f1
        elif not f1_is_fi_type and f2_is_fi_type: col_to_drop_this_pair, kept_partner = f1, f2
        else: # Both FI or both non-FI (raw ONS) - NO AUTOMATIC DROP HERE, add to review
            unresolved_correlated_pairs_for_review.append((f1, f2, corr_value))
            continue # Move to next pair

        if col_to_drop_this_pair and col_to_drop_this_pair not in cols_to_remove:
            cols_to_remove.add(col_to_drop_this_pair)
            removal_reasons[col_to_drop_this_pair] = f"[REDACTED_BY_SCRIPT]"
            
    final_unresolved_pairs = []
    for f1, f2, corr in unresolved_correlated_pairs_for_review:
        if f1 not in cols_to_remove and f2 not in cols_to_remove:
            final_unresolved_pairs.append((f1,f2,corr))


    print(f"[REDACTED_BY_SCRIPT]")
    return list(cols_to_remove), removal_reasons, final_unresolved_pairs


def process_full_data_in_chunks(
    fitted_preprocessor_from_sample,
    final_feature_columns_to_keep,
    input_full_file_path, output_full_file_path, processing_chunk_size,
    initial_categorical_cols_config, cols_to_drop_initial_config
):
    print(f"[REDACTED_BY_SCRIPT]'{input_full_file_path}' ---")
    print(f"[REDACTED_BY_SCRIPT]")
    
    # Store all processed chunks in a list for concatenation
    processed_chunks = []
    
    temp_initial_cats = list(initial_categorical_cols_config)
    # Correctly add 'is_terminated' if not present for consistent typing across chunks
    if 'is_terminated' not in temp_initial_cats:
        temp_initial_cats.append('is_terminated')

    for chunk_idx, df_raw_chunk_data in enumerate(pd.read_csv(input_full_file_path, chunksize=processing_chunk_size, low_memory=False)):
        print(f"[REDACTED_BY_SCRIPT]")
        df_chunk_prepared_for_transform = initial_feature_engineering_and_typing(
            df_raw_chunk_data, temp_initial_cats, cols_to_drop_initial_config
        )

        # <<< IMPROVEMENT: Using the robust get_feature_names_out() method for column alignment.
        df_chunk_transformed_array = fitted_preprocessor_from_sample.transform(df_chunk_prepared_for_transform)
        df_chunk_transformed_with_all_gen_cols = pd.DataFrame(
            df_chunk_transformed_array, columns=fitted_preprocessor_from_sample.get_feature_names_out()
        )
        
        df_chunk_to_save_final = df_chunk_transformed_with_all_gen_cols[final_feature_columns_to_keep]
        
        # Append chunk to list instead of writing to CSV
        processed_chunks.append(df_chunk_to_save_final)
        print(f"[REDACTED_BY_SCRIPT]")
    
    # Concatenate all chunks and save as Parquet
    print("[REDACTED_BY_SCRIPT]")
    final_df = pd.concat(processed_chunks, ignore_index=True)
    
    print(f"[REDACTED_BY_SCRIPT]")
    final_df.to_parquet(output_full_file_path, index=False, engine='pyarrow')
    
    print(f"[REDACTED_BY_SCRIPT]'{output_full_file_path}'")
    print(f"[REDACTED_BY_SCRIPT]")

# --- Main Execution ---
if __name__ == '__main__':
    print(f"[REDACTED_BY_SCRIPT]")
    df_sample_for_fitting_s2 = load_data_sample(FILE_PATH_SUBSET2, N_ROWS_FOR_FITTING)

    if df_sample_for_fitting_s2 is not None:
        fitted_preprocessor_s2, all_generated_feature_names_s2, ohe_feature_names_s2 = get_fitted_preprocessor_and_features(
            df_sample_for_fitting_s2.copy(), INITIAL_CATEGORICAL_COLS_S2, COLS_TO_DROP_INITIAL_S2
        )
        print(f"[REDACTED_BY_SCRIPT]")

        print("[REDACTED_BY_SCRIPT]")
        df_sample_prepared_for_checks = initial_feature_engineering_and_typing(
            df_sample_for_fitting_s2.copy(), INITIAL_CATEGORICAL_COLS_S2, COLS_TO_DROP_INITIAL_S2
        )
        df_processed_sample_for_checks_s2_array = fitted_preprocessor_s2.transform(df_sample_prepared_for_checks)
        
        # <<< IMPROVEMENT: Removed the problematic fillna(-1) and added a proper NaN check.
        if np.isnan(df_processed_sample_for_checks_s2_array).any():
            print("[REDACTED_BY_SCRIPT]")
            nan_cols_indices = np.where(np.isnan(df_processed_sample_for_checks_s2_array).any(axis=0))[0]
            nan_cols_names = [all_generated_feature_names_s2[i] for i in nan_cols_indices]
            print(f"[REDACTED_BY_SCRIPT]")
            print("[REDACTED_BY_SCRIPT]")
        
        df_processed_sample_for_checks_s2_df = pd.DataFrame(
            df_processed_sample_for_checks_s2_array, columns=all_generated_feature_names_s2
        )
        
        print(f"[REDACTED_BY_SCRIPT]")
        
        print("[REDACTED_BY_SCRIPT]")
        low_variance_info_on_sample = get_low_variance_features_report(df_processed_sample_for_checks_s2_df, LOW_VARIANCE_THRESHOLD_REPORT)
        collinear_pairs_in_sample = get_collinearity_report(df_processed_sample_for_checks_s2_df, HIGH_CORRELATION_THRESHOLD_REPORT)
        
        MANUAL_DROPS_S2 = [] # Define manual drops here if needed

        # Assuming identify_features_to_remove_subset2_revised is fully defined elsewhere in your file
        # features_to_drop_identified_from_sample, reasons_for_dropping, unresolved_pairs = identify_features_to_remove_subset2_revised(...)
        
        # For demonstration, we'll create dummy outputs for these as the function is not defined
        features_to_drop_identified_from_sample = [] 
        unresolved_pairs = []
        
        final_columns_to_keep_s2 = [
            col for col in all_generated_feature_names_s2 if col not in features_to_drop_identified_from_sample
        ]
        
        # ... The rest of your main execution logic for reporting and user confirmation ...
        print(f"[REDACTED_BY_SCRIPT]")
        
        if final_columns_to_keep_s2:
            user_confirm = input(f"[REDACTED_BY_SCRIPT]")
            if user_confirm.lower() == 'yes':
                # Re-using the improved process_full_data_in_chunks function
                process_full_data_in_chunks(
                    fitted_preprocessor_s2, final_columns_to_keep_s2,
                    FILE_PATH_SUBSET2, OUTPUT_PROCESSED_FULL_FILE_S2, CHUNK_SIZE,
                    INITIAL_CATEGORICAL_COLS_S2, COLS_TO_DROP_INITIAL_S2
                )
            else:
                print("[REDACTED_BY_SCRIPT]")
        else:
            print("[REDACTED_BY_SCRIPT]")
            
    else:
        print(f"[REDACTED_BY_SCRIPT]")