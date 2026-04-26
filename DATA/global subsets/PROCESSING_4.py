import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer, MinMaxScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import re
import os
import pyarrow as pa
import pyarrow.parquet as pq

# --- Configuration for Subset 4 ---
FILE_PATH_SUBSET4 = "[REDACTED_BY_SCRIPT]"
OUTPUT_FILE_FULL = "[REDACTED_BY_SCRIPT]"
SAMPLE_SIZE = 50000
CHUNK_SIZE_FULL_PROCESSING = 100000

COORDS_COLS = ['pcd_latitude', 'pcd_longitude', 'pcd_eastings', 'pcd_northings']
COLS_TO_DROP_SUBSET4 = ['pcds', 'LSOA21CD']
INITIAL_CATEGORICAL_COLS_SUBSET4 = []

MANUAL_DROPS_S4 = [
    '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
    'NDVI_MEDIAN', 'PCT_FILTERED', 'PXL_COUNT', '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]'
]

LOW_VARIANCE_THRESHOLD = 0.01
HIGH_CORRELATION_THRESHOLD = 0.95
OHE_LOW_VARIANCE_THRESHOLD_FOR_REMOVAL = 0.002

# Global dictionary to store clipping bounds from the sample
CLIP_BOUNDS = {}

def load_data_sample(file_path, sample_size):
    encodings_to_try = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
    for encoding in encodings_to_try:
        try:
            df = pd.read_csv(file_path, low_memory=False, encoding=encoding, nrows=sample_size + 20000)
            df_sample = df.sample(n=sample_size, random_state=42).copy() if len(df) > sample_size else df.copy()
            print(f"[REDACTED_BY_SCRIPT]")
            return df_sample, encoding
        except (UnicodeDecodeError, FileNotFoundError, Exception) as e:
            print(f"[REDACTED_BY_SCRIPT]")
    print(f"[REDACTED_BY_SCRIPT]")
    return None, None

def clip_data_func(X, bounds_dict, column_names):
    X_clipped = X.copy()
    for i, col_name in enumerate(column_names):
        if col_name in bounds_dict:
            min_val, max_val = bounds_dict[col_name]
            X_clipped[:, i] = np.clip(X_clipped[:, i], min_val, max_val)
    return X_clipped

def create_and_fit_preprocessor(df_sample, cols_to_drop_initial, initial_categorical_cols):
    global CLIP_BOUNDS
    df_prepared = df_sample.drop(columns=cols_to_drop_initial, errors='ignore')

    all_numerical = df_prepared.select_dtypes(include=np.number).columns.tolist()
    coords_cols = [col for col in all_numerical if col in COORDS_COLS]
    numerical_cols = [col for col in all_numerical if col not in COORDS_COLS]
    categorical_cols = df_prepared.select_dtypes(include=['object', 'category']).columns.tolist()
    
    for col in initial_categorical_cols:
        if col in df_prepared.columns and col not in categorical_cols:
            categorical_cols.append(col)
            if col in numerical_cols: numerical_cols.remove(col)
        if col in df_prepared.columns:
            df_prepared[col] = df_prepared[col].astype(str)

    print(f"[REDACTED_BY_SCRIPT]")

    temp_imputer = SimpleImputer(strategy='median')
    if numerical_cols:
        df_imputed_for_bounds = pd.DataFrame(temp_imputer.fit_transform(df_prepared[numerical_cols]), columns=numerical_cols)
        for col in numerical_cols:
            valid_data = df_imputed_for_bounds[col].dropna()
            if not valid_data.empty and valid_data.nunique() > 1:
                min_bound, max_bound = np.percentile(valid_data, [0.01, 99.99])
                CLIP_BOUNDS[col] = (min_bound, max_bound)
        print(f"[REDACTED_BY_SCRIPT]")

    clipper = FunctionTransformer(clip_data_func, kw_args={'bounds_dict': CLIP_BOUNDS, 'column_names': numerical_cols})

    numerical_pipeline = Pipeline([
        ('imputer_num', SimpleImputer(strategy='median', add_indicator=True)),
        ('clipper', clipper),
        ('power_transform', PowerTransformer(method='yeo-johnson', standardize=True)),
        ('scaler_num', MinMaxScaler())
    ])
    coords_pipeline = Pipeline([
        ('imputer_coords', SimpleImputer(strategy='median', add_indicator=True))
    ])
    categorical_pipeline = Pipeline([
        ('imputer_cat', SimpleImputer(strategy='constant', fill_value='Missing_Category')),
        ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='if_binary'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_cols),
            ('cat', categorical_pipeline, categorical_cols),
            ('coords', coords_pipeline, coords_cols)
        ],
        remainder='passthrough'
    )
    
    print("[REDACTED_BY_SCRIPT]")
    preprocessor.fit(df_prepared)
    
    return preprocessor, df_prepared

def get_low_variance_features_report(df, threshold):
    variances = df.var(ddof=0)
    return {feature: variances[feature] for feature in variances[variances < threshold].index}

def get_collinearity_report(df, threshold):
    numeric_df = df.copy()
    for col in numeric_df.columns:
        try: numeric_df[col] = pd.to_numeric(numeric_df[col])
        except ValueError: numeric_df[col] = 0 
    numeric_df = numeric_df.fillna(0) 
    corr_matrix = numeric_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    highly_correlated_pairs = []
    for column in upper.columns:
        correlated_features_in_col = upper.index[upper[column] > threshold].tolist()
        for feature in correlated_features_in_col:
            highly_correlated_pairs.append((feature, column, upper.loc[feature, column]))
    return highly_correlated_pairs

def get_ahah_metric_parts(feature_name):
    match = re.match(r"[REDACTED_BY_SCRIPT]", feature_name)
    if match:
        base, suffix_type = match.groups()[:2]
        return base, {"_rnk": "rank", "_pct": "percentile"}.get(suffix_type, "base")
    return feature_name, "other"

def get_veg_metric_parts(feature_name):
    match = re.match(r"[REDACTED_BY_SCRIPT]", feature_name, re.IGNORECASE)
    if match: return match.group(1).upper(), match.group(2).upper()
    return feature_name, "other"

def identify_features_to_remove_subset4(df, ohe_feature_names, low_variance_info, collinear_pairs, manual_drops):
    cols_to_remove = set(manual_drops)
    removal_reasons = {col: "Manual drop" for col in manual_drops}
    unresolved_correlated_pairs_for_review = []
    processed_in_pair = set()

    for feature, variance in low_variance_info.items():
        if feature in cols_to_remove: continue
        thresh = OHE_LOW_VARIANCE_THRESHOLD_FOR_REMOVAL if feature in ohe_feature_names else LOW_VARIANCE_THRESHOLD
        if variance < thresh:
            cols_to_remove.add(feature)
            removal_reasons[feature] = f"[REDACTED_BY_SCRIPT]"

    for f1_orig, f2_orig, corr_value in sorted(collinear_pairs, key=lambda x: x[2], reverse=True):
        f1, f2 = sorted((f1_orig, f2_orig))
        if (f1,f2) in processed_in_pair: continue
        processed_in_pair.add((f1,f2))
        if f1 in cols_to_remove or f2 in cols_to_remove: continue
        rule_applied = False
        base1, type1 = get_ahah_metric_parts(f1); base2, type2 = get_ahah_metric_parts(f2)
        if base1 == base2 and type1 != type2 and type1 != "other" and type2 != "other":
            to_drop, to_keep, reason_suffix = (None, None, None)
            if type1 == "base": to_drop, to_keep, reason_suffix = f2, f1, type2
            elif type2 == "base": to_drop, to_keep, reason_suffix = f1, f2, type1
            elif type1 == "percentile" and type2 == "rank": to_drop, to_keep, reason_suffix = f2, f1, "rank"
            elif type2 == "percentile" and type1 == "rank": to_drop, to_keep, reason_suffix = f1, f2, "rank"
            if to_drop: cols_to_remove.add(to_drop); removal_reasons[to_drop] = f"[REDACTED_BY_SCRIPT]"; rule_applied = True
        if rule_applied: continue
        f1_is_score = (type1 == "base" and (f1.endswith(('h','g','e','r','ahah'))) and len(f1) <= len("ah4ahah")+2)
        f2_is_score = (type2 == "base" and (f2.endswith(('h','g','e','r','ahah'))) and len(f2) <= len("ah4ahah")+2)
        if f1_is_score and not f2_is_score and type2 == "base": cols_to_remove.add(f1); removal_reasons[f1] = f"[REDACTED_BY_SCRIPT]"; rule_applied = True
        elif f2_is_score and not f1_is_score and type1 == "base": cols_to_remove.add(f2); removal_reasons[f2] = f"[REDACTED_BY_SCRIPT]"; rule_applied = True
        elif f1_is_score and f2_is_score:
            to_drop = f1 if len(f1) > len(f2) else f2; to_keep = f2 if len(f1) > len(f2) else f1
            cols_to_remove.add(to_drop); removal_reasons[to_drop] = f"[REDACTED_BY_SCRIPT]"; rule_applied = True
        if rule_applied: continue
        veg1, stat1 = get_veg_metric_parts(f1); veg2, stat2 = get_veg_metric_parts(f2)
        if stat1 == stat2 and stat1 != "other":
            if veg1 == "EVI" and veg2 == "NDVI": cols_to_remove.add(f1); removal_reasons[f1] = f"[REDACTED_BY_SCRIPT]"; rule_applied = True
            elif veg2 == "EVI" and veg1 == "NDVI": cols_to_remove.add(f2); removal_reasons[f2] = f"[REDACTED_BY_SCRIPT]"; rule_applied = True
        if rule_applied: continue
        unresolved_correlated_pairs_for_review.append((f1_orig, f2_orig, corr_value))
    final_unresolved_for_manual_review = []
    for u_f1, u_f2, u_corr in unresolved_correlated_pairs_for_review: 
        if u_f1 not in cols_to_remove and u_f2 not in cols_to_remove:
            final_unresolved_for_manual_review.append((u_f1, u_f2, u_corr))
    return list(cols_to_remove), removal_reasons, final_unresolved_for_manual_review

# --- Main Execution ---
if __name__ == '__main__':
    print("[REDACTED_BY_SCRIPT]")
    df_sample, encoding = load_data_sample(FILE_PATH_SUBSET4, SAMPLE_SIZE)
    
    if df_sample is not None:
        preprocessor, df_prepared_for_fit = create_and_fit_preprocessor(
            df_sample, COLS_TO_DROP_SUBSET4, INITIAL_CATEGORICAL_COLS_SUBSET4
        )

        transformed_array = preprocessor.transform(df_prepared_for_fit)
        
        num_cols_out = []
        cat_cols_out = []
        coords_cols_out = []

        original_num_cols = preprocessor.transformers_[0][2]
        original_cat_cols = preprocessor.transformers_[1][2]
        original_coords_cols = preprocessor.transformers_[2][2]
        
        if original_num_cols:
            imputer_num_step = preprocessor.named_transformers_['num'].named_steps['imputer_num']
            num_cols_out = imputer_num_step.get_feature_names_out(original_num_cols)

        if original_cat_cols:
            ohe_step = preprocessor.named_transformers_['cat'].named_steps['one_hot_encoder']
            cat_cols_out = ohe_step.get_feature_names_out(original_cat_cols)

        if original_coords_cols:
            imputer_coords_step = preprocessor.named_transformers_['coords'].named_steps['imputer_coords']
            coords_cols_out = imputer_coords_step.get_feature_names_out(original_coords_cols)

        transformed_features_cols = np.concatenate([num_cols_out, cat_cols_out]).tolist()
        if transformed_features_cols:
            transformed_features_array = transformed_array[:, :len(transformed_features_cols)]
            df_processed_sample_for_analysis = pd.DataFrame(transformed_features_array, columns=transformed_features_cols)

            if df_processed_sample_for_analysis.isna().values.any():
                print("[REDACTED_BY_SCRIPT]")
        else:
            df_processed_sample_for_analysis = pd.DataFrame()

        low_var_info = get_low_variance_features_report(df_processed_sample_for_analysis, LOW_VARIANCE_THRESHOLD)
        collinear_pairs = get_collinearity_report(df_processed_sample_for_analysis, HIGH_CORRELATION_THRESHOLD)
        features_to_drop_final, reasons, unresolved = identify_features_to_remove_subset4(
            df_processed_sample_for_analysis, cat_cols_out, low_var_info, collinear_pairs, MANUAL_DROPS_S4
        )
        
        final_ml_features_to_keep = [col for col in transformed_features_cols if col not in features_to_drop_final]
        
        # --- CORRECTED LINE ---
        # `coords_cols_out` is already a list or a NumPy array that can be converted to one.
        # We simply concatenate it with the other list of features.
        final_columns_to_keep = list(coords_cols_out) + final_ml_features_to_keep
        
        print(f"[REDACTED_BY_SCRIPT]")
        print(f"[REDACTED_BY_SCRIPT]")
        print(f"[REDACTED_BY_SCRIPT]")

        # --- Phase 2: Process full dataset in chunks ---
        print(f"[REDACTED_BY_SCRIPT]")
        if os.path.exists(OUTPUT_FILE_FULL):
            os.remove(OUTPUT_FILE_FULL)
            print(f"[REDACTED_BY_SCRIPT]")

        # Initialize variables for parquet writing
        parquet_writer = None
        parquet_schema = None

        for i, chunk_df in enumerate(pd.read_csv(FILE_PATH_SUBSET4, chunksize=CHUNK_SIZE_FULL_PROCESSING, low_memory=False, encoding=encoding)):
            print(f"[REDACTED_BY_SCRIPT]")
            
            chunk_prepared = chunk_df.drop(columns=COLS_TO_DROP_SUBSET4, errors='ignore')
            chunk_transformed_array = preprocessor.transform(chunk_prepared)

            all_output_cols = np.concatenate([num_cols_out, cat_cols_out, coords_cols_out]).tolist()
            df_chunk_transformed = pd.DataFrame(chunk_transformed_array, columns=all_output_cols)
            
            df_chunk_to_save = df_chunk_transformed[final_columns_to_keep]

            # Convert to PyArrow table
            table = pa.Table.from_pandas(df_chunk_to_save)
            
            if parquet_writer is None:
                # Create the parquet file and writer for the first chunk
                parquet_schema = table.schema
                parquet_writer = pq.ParquetWriter(OUTPUT_FILE_FULL, parquet_schema)
            
            # Write the chunk to the parquet file
            parquet_writer.write_table(table)
            print(f"[REDACTED_BY_SCRIPT]")

        # Close the parquet writer
        if parquet_writer is not None:
            parquet_writer.close()

        print(f"[REDACTED_BY_SCRIPT]")

    else:
        print("[REDACTED_BY_SCRIPT]")