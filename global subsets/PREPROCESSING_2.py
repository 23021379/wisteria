import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import re

# --- Configuration for Subset 2 ---
FILE_PATH_SUBSET2 = r"[REDACTED_BY_SCRIPT]" # Replace with path to your combined Subset 2 file

SAMPLE_SIZE = 50000 # Keep or adjust as needed

# Columns to drop at the very beginning for Subset 2
COLS_TO_DROP_SUBSET2 = [
    'pcd7', 'pcd8', 'pcds',
    'oa21cd', # <<< ADDED HERE
    'lsoa21cd', 'msoa21cd', 'ladcd',
    'lsoa21nm', 'msoa21nm', 'ladnm', 'ladnmw',
    'Output_Areas_Code', 'Output Areas' # ONS join keys, assuming merge happened
]

# Initial categorical columns for Subset 2
INITIAL_CATEGORICAL_COLS_SUBSET2 = [
    'usertype',
    # 'is_terminated' will be created and added to this list if handled as categorical for OHE
]

# Thresholds (can be kept or adjusted)
LOW_VARIANCE_THRESHOLD = 0.01 # For general numerical features post-scaling
HIGH_CORRELATION_THRESHOLD = 0.95
OHE_LOW_VARIANCE_THRESHOLD_FOR_REMOVAL = 0.002 # Specifically for OHE features

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

def preprocess_data(df, cols_to_drop_initial, initial_categorical_cols_config):
    print(f"[REDACTED_BY_SCRIPT]")
    
    # Drop initial set of columns
    df_processed = df.drop(columns=cols_to_drop_initial, errors='ignore')
    print(f"[REDACTED_BY_SCRIPT]")

    # Feature Engineering: is_terminated from doterm
    if 'doterm' in df_processed.columns:
        df_processed['is_terminated'] = df_processed['doterm'].notna().astype(int)
        print("Created 'is_terminated' feature.")
    
    # Add 'is_terminated' to a temporary list for categorical processing if it was created
    temp_initial_categorical_cols = list(initial_categorical_cols_config)
    if 'is_terminated' in df_processed.columns and 'is_terminated' not in temp_initial_categorical_cols:
        temp_initial_categorical_cols.append('is_terminated')
        
    # Drop date columns after use
    df_processed = df_processed.drop(columns=['dointr', 'doterm'], errors='ignore')

    # Ensure specified categorical columns are string type for consistent OHE
    for col in temp_initial_categorical_cols:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].astype(str)

    # Identify numerical and categorical columns
    numerical_cols = df_processed.select_dtypes(include=np.number).columns.tolist()
    categorical_cols_from_dtype = df_processed.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Ensure all columns from temp_initial_categorical_cols are in categorical_cols_from_dtype
    # and not accidentally in numerical_cols if they were numbers then cast to str (e.g. usertype '0','1')
    final_categorical_cols = []
    final_numerical_cols = list(numerical_cols) # Start with all numeric by dtype

    for col in temp_initial_categorical_cols:
        if col in df_processed.columns:
            if col not in final_categorical_cols:
                final_categorical_cols.append(col)
            if col in final_numerical_cols: # If it was picked as numeric but specified as cat
                final_numerical_cols.remove(col)
    
    # Add any other object/category columns not explicitly specified but found by dtype
    for col in categorical_cols_from_dtype:
        if col not in final_categorical_cols:
            final_categorical_cols.append(col)
            
    # Check for columns not assigned to either list
    all_cols_set = set(df_processed.columns)
    processed_cols_set = set(final_numerical_cols + final_categorical_cols)
    if all_cols_set != processed_cols_set:
        missing_from_typed_lists = all_cols_set - processed_cols_set
        print(f"[REDACTED_BY_SCRIPT]")
        # Decide how to handle these - for now, assume numerical if not handled otherwise
        for m_col in missing_from_typed_lists:
            if m_col not in final_numerical_cols and m_col not in final_categorical_cols:
                # Heuristic: if dtype is number, add to numerical, else categorical
                if pd.api.types.is_numeric_dtype(df_processed[m_col]):
                    final_numerical_cols.append(m_col)
                else:
                    final_categorical_cols.append(m_col)
                    df_processed[m_col] = df_processed[m_col].astype(str) # Ensure string for OHE

    print(f"[REDACTED_BY_SCRIPT]")
    print(f"[REDACTED_BY_SCRIPT]")

    # Define pipelines
    numerical_pipeline = Pipeline([
        ('imputer_num', SimpleImputer(strategy='constant', fill_value=-1)), # Using -1 for all numerical NaNs
        ('power_transform', PowerTransformer(method='yeo-johnson', standardize=False)), # standardize=False as MinMaxScaler comes next
        ('scaler_num', MinMaxScaler())
    ])
    categorical_pipeline = Pipeline([
        ('imputer_cat', SimpleImputer(strategy='constant', fill_value='Missing_Category')),
        ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='if_binary'))
    ])

    # Create preprocessor
    # Ensure no column is listed in both numerical and categorical transformers
    valid_numerical_cols = [col for col in final_numerical_cols if col in df_processed.columns]
    valid_categorical_cols = [col for col in final_categorical_cols if col in df_processed.columns]
    
    transformers_list = []
    if valid_numerical_cols:
        transformers_list.append(('num', numerical_pipeline, valid_numerical_cols))
    if valid_categorical_cols:
        transformers_list.append(('cat', categorical_pipeline, valid_categorical_cols))

    if not transformers_list:
        print("[REDACTED_BY_SCRIPT]")
        return df_processed, []

    preprocessor = ColumnTransformer(
        transformers=transformers_list,
        remainder='passthrough', # Keep any columns not explicitly handled
        verbose_feature_names_out=False
    )
    preprocessor.set_output(transform="pandas")

    print("[REDACTED_BY_SCRIPT]")
    df_transformed = preprocessor.fit_transform(df_processed)
    
    ohe_feature_names = []
    if valid_categorical_cols and 'cat' in preprocessor.named_transformers_:
        try:
            ohe_feature_names = preprocessor.named_transformers_['cat'].named_steps['one_hot_encoder'].get_feature_names_out(valid_categorical_cols).tolist()
        except Exception as e:
            print(f"[REDACTED_BY_SCRIPT]")


    print(f"[REDACTED_BY_SCRIPT]")
    return df_transformed, ohe_feature_names

def get_low_variance_features_report(df, threshold):
    variances = df.var(ddof=0) # ddof=0 for population variance
    low_variance_info = {feature: variances[feature] for feature in variances[variances < threshold].index}
    return low_variance_info

def get_collinearity_report(df, threshold):
    # Ensure all data is numeric for correlation matrix; coercing errors might indicate issues
    # Handle potential all-NaN columns that SimpleImputer might turn to -1, then PowerTransform, then Scale.
    # If a column became all NaN -> -1 -> (some constant after PT) -> (some constant after MinMax, usually 0 or 1)
    # These constant columns will have 0 variance and won't affect corr matrix much but are low var.
    numeric_df = df.copy()
    for col in numeric_df.columns:
        numeric_df[col] = pd.to_numeric(numeric_df[col], errors='coerce')
    numeric_df = numeric_df.fillna(0) # Or another strategy for NaNs post-to_numeric
    
    corr_matrix = numeric_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    highly_correlated_pairs = []
    for column in upper.columns: # Iterate through columns
        # Find features in this column that are highly correlated
        correlated_features_in_col = upper.index[upper[column] > threshold].tolist()
        for feature in correlated_features_in_col:
            highly_correlated_pairs.append((feature, column, upper.loc[feature, column]))
    return highly_correlated_pairs

def identify_features_to_remove_subset2(df, ohe_feature_names, low_variance_info, collinear_pairs):
    cols_to_remove = set()
    removal_reasons = {}

    # 1. Low Variance OHE Features (e.g., from 'usertype')
    for feature, variance in low_variance_info.items():
        if feature in ohe_feature_names and variance < OHE_LOW_VARIANCE_THRESHOLD_FOR_REMOVAL:
            if feature not in cols_to_remove:
                 cols_to_remove.add(feature)
                 removal_reasons[feature] = f"[REDACTED_BY_SCRIPT]"
    
    # 2. General Low Variance Features (Non-OHE, from main threshold)
    #    These are often more critical to review manually, but we can flag them.
    for feature, variance in low_variance_info.items():
        if feature not in ohe_feature_names and variance < LOW_VARIANCE_THRESHOLD: # Using the general threshold
            if feature not in cols_to_remove:
                cols_to_remove.add(feature)
                removal_reasons[feature] = f"[REDACTED_BY_SCRIPT]"


    # 3. Collinearity Removals for Subset 2 (ONS Data & FI Features)
    #    Priority: Keep FI_ features over raw counts if they are correlated.
    #    Keep one from a pair of highly correlated raw ONS counts.
    
    # Create a list of FI features to help prioritize
    fi_feature_prefixes = ['FI_', 'Total_Households_OA'] # Add other key calculated metrics if any
                                                    # Total_Households_OA is a raw sum but critical denominator
    
    # Store columns already decided to be kept to resolve conflicts
    kept_due_to_priority = set()

    for f1, f2, corr_value in collinear_pairs:
        # If one is already marked for removal, skip to avoid reprocessing or conflicts
        if f1 in cols_to_remove or f2 in cols_to_remove:
            continue
        # If one was already prioritized to be kept, and the other is now found correlated with it, remove the other
        if f1 in kept_due_to_priority and f2 not in cols_to_remove:
            cols_to_remove.add(f2)
            removal_reasons[f2] = f"[REDACTED_BY_SCRIPT]"
            continue
        if f2 in kept_due_to_priority and f1 not in cols_to_remove:
            cols_to_remove.add(f1)
            removal_reasons[f1] = f"[REDACTED_BY_SCRIPT]"
            continue

        f1_is_fi = any(f1.startswith(prefix) for prefix in fi_feature_prefixes)
        f2_is_fi = any(f2.startswith(prefix) for prefix in fi_feature_prefixes)

        # Rule 1: FI feature vs Non-FI feature
        if f1_is_fi and not f2_is_fi:
            cols_to_remove.add(f2)
            removal_reasons[f2] = f"[REDACTED_BY_SCRIPT]"
            kept_due_to_priority.add(f1)
        elif not f1_is_fi and f2_is_fi:
            cols_to_remove.add(f1)
            removal_reasons[f1] = f"[REDACTED_BY_SCRIPT]"
            kept_due_to_priority.add(f2)
        
        # Rule 2: Both are FI features or both are not (e.g., two raw ONS counts)
        # Simple heuristic: remove the one that comes later alphabetically to ensure one is dropped.
        # More sophisticated logic could be added here (e.g., based on sum of values, number of NaNs before imputation).
        elif (f1_is_fi and f2_is_fi) or (not f1_is_fi and not f2_is_fi):
            # Check for ONS "Does not apply" vs other categories from same group if possible
            # Example: '[REDACTED_BY_SCRIPT]' vs '[REDACTED_BY_SCRIPT]'
            f1_base = "_".join(f1.split('_')[:2]) # e.g., "[REDACTED_BY_SCRIPT]"
            f2_base = "_".join(f2.split('_')[:2])
            
            is_f1_dna = "Does_not_apply" in f1
            is_f2_dna = "Does_not_apply" in f2

            if f1_base == f2_base: # From the same ONS variable group
                if is_f1_dna and not is_f2_dna: # f1 is DNA, f2 is a category
                    cols_to_remove.add(f1) # Tentatively drop DNA if highly corr with a main category
                    removal_reasons[f1] = f"ONS 'Does_not_apply'[REDACTED_BY_SCRIPT]"
                    kept_due_to_priority.add(f2)
                elif not is_f1_dna and is_f2_dna: # f2 is DNA, f1 is a category
                    cols_to_remove.add(f2)
                    removal_reasons[f2] = f"ONS 'Does_not_apply'[REDACTED_BY_SCRIPT]"
                    kept_due_to_priority.add(f1)
                else: # Both are categories or both are DNA (unlikely for DNA vs DNA)
                    if f1 > f2: # Drop alphabetically later
                        cols_to_remove.add(f1)
                        removal_reasons[f1] = f"[REDACTED_BY_SCRIPT]"
                        kept_due_to_priority.add(f2)
                    else:
                        cols_to_remove.add(f2)
                        removal_reasons[f2] = f"[REDACTED_BY_SCRIPT]"
                        kept_due_to_priority.add(f1)
            else: # From different groups or general correlation
                if f1 > f2:
                    cols_to_remove.add(f1)
                    removal_reasons[f1] = f"[REDACTED_BY_SCRIPT]"
                    kept_due_to_priority.add(f2)
                else:
                    cols_to_remove.add(f2)
                    removal_reasons[f2] = f"[REDACTED_BY_SCRIPT]"
                    kept_due_to_priority.add(f1)
            
    # Remove any feature that was marked to be kept if its pair was also marked to be kept (resolve conflicts)
    # This scenario should be minimized by the `kept_due_to_priority` logic handling within the loop.

    # Specific check for `FI_HH_Adults_to_Children_Ratio` if it became constant - should be caught by low variance
    if '[REDACTED_BY_SCRIPT]' in low_variance_info and \
       low_variance_info['[REDACTED_BY_SCRIPT]'] < 1e-9 and \
       '[REDACTED_BY_SCRIPT]' not in cols_to_remove :
        cols_to_remove.add('[REDACTED_BY_SCRIPT]')
        removal_reasons['[REDACTED_BY_SCRIPT]'] = "[REDACTED_BY_SCRIPT]"
        
    return list(cols_to_remove), removal_reasons

# --- Main Execution ---
if __name__ == '__main__':
    # For Subset 2, you might need to load and merge if your input file isn't already combined.
    # Assuming FILE_PATH_SUBSET2 is a single CSV that has postcode data, ONS data, and FI features already.
    # If you need to merge, here's a conceptual placeholder:
    # df_lookup = load_data(POSTCODE_LOOKUP_FILE)
    # df_ons = load_data(ONS_PIVOTED_FILE)
    # if df_lookup is not None and df_ons is not None:
    #     # Ensure merge keys are correct, e.g., 'oa21cd' in lookup and 'Output_Areas_Code' in ONS
    #     df_merged_subset2_full = pd.merge(df_lookup, df_ons, left_on='oa21cd', right_on='Output_Areas_Code', how='left')
    #     # Here, you would also run your script to calculate FI_ features on df_merged_subset2_full
    #     # For now, we assume FILE_PATH_SUBSET2 already has all this.
    # else:
    #     df_merged_subset2_full = None
    
    df_merged_subset2_full = load_data(FILE_PATH_SUBSET2)

    if df_merged_subset2_full is not None:
        if 'Total_Households_OA' in df_merged_subset2_full.columns:
            zero_total_hh_count = (df_merged_subset2_full['Total_Households_OA'] == 0).sum()
            if zero_total_hh_count > 0:
                print(f"[REDACTED_BY_SCRIPT]")
        else:
            print("Warning: 'Total_Households_OA'[REDACTED_BY_SCRIPT]")


        if len(df_merged_subset2_full) > SAMPLE_SIZE:
            print(f"[REDACTED_BY_SCRIPT]")
            # Use .copy() to avoid SettingWithCopyWarning on the sample later
            subset2_df_to_process = df_merged_subset2_full.sample(n=SAMPLE_SIZE, random_state=42).copy()
        else:
            subset2_df_to_process = df_merged_subset2_full.copy()
        print(subset2_df_to_process['Total_Households_OA'].describe())
        print(subset2_df_to_process['Total_Households_OA'].isna().sum())
        print(subset2_df_to_process['[REDACTED_BY_SCRIPT]'].describe())
        print(subset2_df_to_process['[REDACTED_BY_SCRIPT]'].isna().sum())
        # Preprocess the sample
        df_processed_subset2, ohe_cols_s2 = preprocess_data(
            subset2_df_to_process, 
            COLS_TO_DROP_SUBSET2, 
            INITIAL_CATEGORICAL_COLS_SUBSET2
        )
        
        print("[REDACTED_BY_SCRIPT]")
        # Ensure no NaNs before variance/correlation - should be handled by imputer, but as a safeguard:
        df_processed_subset2.fillna(-1, inplace=True) # Or a more sophisticated check for NaNs

        low_var_info_report_s2 = get_low_variance_features_report(df_processed_subset2, LOW_VARIANCE_THRESHOLD)
        collinear_pairs_report_s2 = get_collinearity_report(df_processed_subset2, HIGH_CORRELATION_THRESHOLD)

        print(f"[REDACTED_BY_SCRIPT]")
        for f, v in low_var_info_report_s2.items():
            if f not in ohe_cols_s2 and v < LOW_VARIANCE_THRESHOLD : # check general LV threshold
                 print(f"  - {f}: {v:.6f}")
        
        print(f"[REDACTED_BY_SCRIPT]")
        for f, v in low_var_info_report_s2.items():
            if f in ohe_cols_s2 and v < OHE_LOW_VARIANCE_THRESHOLD_FOR_REMOVAL:
                 print(f"  - {f}: {v:.6f}")

        print(f"[REDACTED_BY_SCRIPT]")
        for f1, f2, corr in collinear_pairs_report_s2[:10]: # Print first 10
            print(f"[REDACTED_BY_SCRIPT]")
        if len(collinear_pairs_report_s2) > 10:
            print(f"[REDACTED_BY_SCRIPT]")


        print("[REDACTED_BY_SCRIPT]")
        features_to_drop_final_s2, reasons_for_dropping_s2 = identify_features_to_remove_subset2(
            df_processed_subset2,
            ohe_cols_s2,
            low_var_info_report_s2,
            collinear_pairs_report_s2
        )
        
        features_to_drop_final_s2_existing = [col for col in features_to_drop_final_s2 if col in df_processed_subset2.columns]
        
        df_final_sample_s2 = df_processed_subset2.copy() 
        if features_to_drop_final_s2_existing:
            print(f"[REDACTED_BY_SCRIPT]")
            # Sort for consistent output
            sorted_features_to_drop = sorted(list(set(features_to_drop_final_s2_existing))) 
            for col in sorted_features_to_drop:
                 print(f"[REDACTED_BY_SCRIPT]'N/A')})")
            
            df_final_sample_s2.drop(columns=sorted_features_to_drop, inplace=True, errors='ignore')
            print(f"[REDACTED_BY_SCRIPT]")
            print(f"[REDACTED_BY_SCRIPT]")
        else:
            print("[REDACTED_BY_SCRIPT]")
            
        print("[REDACTED_BY_SCRIPT]")
        print(f"[REDACTED_BY_SCRIPT]")
        print(df_final_sample_s2.head())

        output_filename_s2 = "[REDACTED_BY_SCRIPT]"
        try:
            df_final_sample_s2.to_csv(output_filename_s2, index=False)
            print(f"[REDACTED_BY_SCRIPT]")
            print(f"[REDACTED_BY_SCRIPT]")
            print(f"[REDACTED_BY_SCRIPT]")
        except Exception as e:
            print(f"[REDACTED_BY_SCRIPT]")