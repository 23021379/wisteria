# -*- coding: utf-8 -*-
"""
Data Preprocessing and Analysis Script (Analysis Phase) - V3

Purpose:
- Processes a SAMPLE of the consolidated property dataset.
- Applies a comprehensive preprocessing pipeline including cleaning, feature
  engineering, scaling, and encoding.
- Performs feature selection based on low variance and high correlation.
- Generates detailed reports to guide manual feature removal decisions.
- NEW: Handles high-cardinality categorical features by dropping them.

This script is the "laboratory" phase, designed for analysis and tuning on a
sample. The decisions made here (especially in manual_drops.txt) will inform
the "production" script that processes the full dataset.
"""

import pandas as pd
import numpy as np
import re
import warnings
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PowerTransformer, MinMaxScaler, OneHotEncoder
from sklearn.feature_selection import VarianceThreshold

warnings.filterwarnings('ignore', category=FutureWarning)

# --- CONFIGURATION ---

# File Paths
FILE_PATH = "[REDACTED_BY_SCRIPT]"  # <--- CHANGE THIS TO YOUR CSV FILENAME
OUTPUT_FILENAME = "[REDACTED_BY_SCRIPT]"
MANUAL_DROPS_FILE = "[REDACTED_BY_SCRIPT]"
SAMPLE_SIZE = 50000
RANDOM_STATE = 42
HIGH_CORRELATION_THRESHOLD = 0.95
OHE_LOW_VARIANCE_THRESHOLD_FOR_REMOVAL = 0.002
NUMERICAL_LOW_VARIANCE_THRESHOLD_FOR_AUTO_REMOVAL = 1e-9
CARDINALITY_THRESHOLD = 25 # Increased slightly to keep more potentially useful features

# --- COLUMN CLASSIFICATION & INITIAL DROPS ---
# These lists will be used AFTER sanitizing column names
COLS_TO_DROP_INITIALLY_PATTERNS = [
    r'[REDACTED_BY_SCRIPT]', r'placeholder', r'\(empty\)_mp',
    r'[REDACTED_BY_SCRIPT]',
    r'[REDACTED_BY_SCRIPT]',
    r'[REDACTED_BY_SCRIPT]',
    r'property_floor_area_sqft',
    r'[REDACTED_BY_SCRIPT]',
    r'[REDACTED_BY_SCRIPT]',
    r'[REDACTED_BY_SCRIPT]',
]
COLS_TO_CONVERT_TO_NUMERIC_PATTERNS = [
    r'[REDACTED_BY_SCRIPT]', r'[REDACTED_BY_SCRIPT]',
    r'[REDACTED_BY_SCRIPT]',
    r'[REDACTED_BY_SCRIPT]',
    r'[REDACTED_BY_SCRIPT]',
    r'[REDACTED_BY_SCRIPT]',
]
INITIAL_CATEGORICAL_COLS_PATTERNS = [
    r'confidence_score_encoded', r'[REDACTED_BY_SCRIPT]',
    r'property_sub_type_code', r'property_tenure_code',
    r'[REDACTED_BY_SCRIPT]', r'[REDACTED_BY_SCRIPT]',
    r'[REDACTED_BY_SCRIPT]', r'epc_or_tax_band_encoded',
    r'[REDACTED_BY_SCRIPT]', r'geocoding_level',
]

def sanitize_columns(df):
    """[REDACTED_BY_SCRIPT]"""
    sanitized_columns = {}
    for col in df.columns:
        new_col = re.sub(r'[^A-Za-z0-9_]+', '_', col) # Replace non-alphanumeric with underscore
        new_col = re.sub(r'_+', '_', new_col) # Replace multiple underscores with one
        new_col = new_col.strip('_') # Remove leading/trailing underscores
        sanitized_columns[col] = new_col
    df.rename(columns=sanitized_columns, inplace=True)
    print("[REDACTED_BY_SCRIPT]")
    return df

def get_cols_by_pattern(all_cols, patterns):
    """[REDACTED_BY_SCRIPT]"""
    matched_cols = []
    for pat in patterns:
        for col in all_cols:
            if re.search(pat, col):
                matched_cols.append(col)
    return list(set(matched_cols)) # Return unique columns

def load_data(file_path):
    """[REDACTED_BY_SCRIPT]"""
    # (Implementation is the same, omitted for brevity)
    encodings_to_try = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
    for encoding in encodings_to_try:
        try:
            print(f"[REDACTED_BY_SCRIPT]")
            df = pd.read_csv(file_path, encoding=encoding, low_memory=False)
            print("[REDACTED_BY_SCRIPT]")
            return df
        except UnicodeDecodeError:
            print(f"[REDACTED_BY_SCRIPT]")
        except FileNotFoundError:
            print(f"[REDACTED_BY_SCRIPT]")
            return None
    print("[REDACTED_BY_SCRIPT]")
    return None

def load_manual_drops(file_path):
    """[REDACTED_BY_SCRIPT]"""
    # (Implementation is the same, omitted for brevity)
    try:
        with open(file_path, 'r') as f:
            # Sanitize manual drop names to match the new column names
            manual_drops = set()
            for line in f:
                if line.strip() and not line.startswith('#'):
                    new_col = re.sub(r'[^A-Za-z0-9_]+', '_', line.strip())
                    new_col = re.sub(r'_+', '_', new_col)
                    manual_drops.add(new_col.strip('_'))
            print(f"[REDACTED_BY_SCRIPT]")
            return manual_drops
    except FileNotFoundError:
        print(f"[REDACTED_BY_SCRIPT]'{file_path}'.")
        return set()

def preprocess_and_analyze(df):
    """[REDACTED_BY_SCRIPT]"""
    print("[REDACTED_BY_SCRIPT]")
    print(f"[REDACTED_BY_SCRIPT]")

    # --- Step 1: Sanitize, Clean, and Engineer ---
    df = sanitize_columns(df)
    
    # Get column lists based on patterns
    cols_to_drop_initially = get_cols_by_pattern(df.columns, COLS_TO_DROP_INITIALLY_PATTERNS)
    manual_drops = load_manual_drops(MANUAL_DROPS_FILE)
    
    df.drop(columns=cols_to_drop_initially + list(manual_drops), inplace=True, errors='ignore')
    print(f"[REDACTED_BY_SCRIPT]")
    
    cols_to_convert_to_numeric = get_cols_by_pattern(df.columns, COLS_TO_CONVERT_TO_NUMERIC_PATTERNS)
    for col in cols_to_convert_to_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    date_cols = ['last_sold_date_year_YYYY_hm', 'last_sold_date_month_MM_hm', 'last_sold_date_day_DD_hm']
    if all(col in df.columns for col in date_cols):
        print("Engineering 'days_since_last_sale' feature...")
        rename_map = {date_cols[0]: 'year', date_cols[1]: 'month', date_cols[2]: 'day'}
        date_df_to_convert = df[date_cols].rename(columns=rename_map)
        df['last_sold_date'] = pd.to_datetime(date_df_to_convert, errors='coerce')
        reference_date = pd.Timestamp('2025-01-01')
        df['days_since_last_sale'] = (reference_date - df['last_sold_date']).dt.days
        df.drop(columns=date_cols + ['last_sold_date'], inplace=True)
        print("[REDACTED_BY_SCRIPT]")
    
    print(f"[REDACTED_BY_SCRIPT]")

    # --- Step 2: Column Type Classification (with Cardinality Check) ---
    # (The rest of the function remains largely the same logic, adapted for sanitized names)
    print("[REDACTED_BY_SCRIPT]")
    
    potential_cat_cols = get_cols_by_pattern(df.columns, INITIAL_CATEGORICAL_COLS_PATTERNS)
    for col in df.select_dtypes(include=['object', 'category']).columns:
        if col not in potential_cat_cols:
            potential_cat_cols.append(col)

    final_categorical_cols = []
    high_cardinality_to_drop = []
    for col in potential_cat_cols:
        if df[col].nunique(dropna=False) > CARDINALITY_THRESHOLD:
            high_cardinality_to_drop.append(col)
        else:
            final_categorical_cols.append(col)

    if high_cardinality_to_drop:
        print(f"[REDACTED_BY_SCRIPT]")
        print(high_cardinality_to_drop)
        df.drop(columns=high_cardinality_to_drop, inplace=True)

    final_numerical_cols = [col for col in df.columns if col not in final_categorical_cols]

    for col in final_categorical_cols:
        df[col] = df[col].astype(str)

    print(f"[REDACTED_BY_SCRIPT]")
    print(f"[REDACTED_BY_SCRIPT]")
    
    # --- Step 3, 4, 5, 6... (The logic remains the same)
    # The pipelines and selection logic will now work on clean column names.
    print("[REDACTED_BY_SCRIPT]")
    
    numerical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value=-1)),
        ('transformer', PowerTransformer(method='yeo-johnson')),
        ('scaler', MinMaxScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Missing_Category')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='if_binary', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_pipeline, final_numerical_cols),
        ('cat', categorical_pipeline, final_categorical_cols)
    ], remainder='passthrough', verbose_feature_names_out=False)

    print("[REDACTED_BY_SCRIPT]")
    preprocessor.fit(df)
    processed_data = preprocessor.transform(df)
    
    all_feature_names = preprocessor.get_feature_names_out()

    df_processed = pd.DataFrame(processed_data, columns=all_feature_names, index=df.index)
    print(f"[REDACTED_BY_SCRIPT]")
    
    # --- Step 5: Feature Selection ---
    print("[REDACTED_BY_SCRIPT]")
    cols_to_remove = set()
    analysis_report = {}

    ohe_feature_names = [c for c in all_feature_names if c not in final_numerical_cols]

    # 5a. Low Variance
    ohe_variances = df_processed[ohe_feature_names].var()
    low_var_ohe_cols = ohe_variances[ohe_variances < OHE_LOW_VARIANCE_THRESHOLD_FOR_REMOVAL].index.tolist()
    cols_to_remove.update(low_var_ohe_cols)
    analysis_report['Low Variance (OHE)'] = low_var_ohe_cols

    selector = VarianceThreshold(threshold=NUMERICAL_LOW_VARIANCE_THRESHOLD_FOR_AUTO_REMOVAL)
    selector.fit(df_processed[final_numerical_cols])
    low_var_num_cols = [col for i, col in enumerate(final_numerical_cols) if not selector.get_support()[i]]
    cols_to_remove.update(low_var_num_cols)
    analysis_report['[REDACTED_BY_SCRIPT]'] = low_var_num_cols

    # 5b. High Correlation
    corr_matrix = df_processed[final_numerical_cols].corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    highly_correlated_pairs = []
    for col in upper_tri.columns:
        correlated_cols = upper_tri.index[upper_tri[col] > HIGH_CORRELATION_THRESHOLD].tolist()
        for correlated_col in correlated_cols:
            highly_correlated_pairs.append((col, correlated_col))

    kept_due_to_priority = set()
    unresolved_pairs = []
    correlated_to_remove = set()

    for f1, f2 in highly_correlated_pairs:
        pair = {f1, f2}
        if any(f in correlated_to_remove for f in pair) or any(f in kept_due_to_priority for f in pair if f != (pair - kept_due_to_priority).pop()):
            continue
        
        # Domain rules
        if '[REDACTED_BY_SCRIPT]' in pair:
            to_remove = next(p for p in pair if p != '[REDACTED_BY_SCRIPT]')
            correlated_to_remove.add(to_remove); kept_due_to_priority.add('[REDACTED_BY_SCRIPT]'); continue
        if '[REDACTED_BY_SCRIPT]' in pair:
            to_remove = next(p for p in pair if p != '[REDACTED_BY_SCRIPT]')
            correlated_to_remove.add(to_remove); kept_due_to_priority.add('[REDACTED_BY_SCRIPT]'); continue
        if '[REDACTED_BY_SCRIPT]' in pair and any(('safety' in p or 'Crime' in p) for p in pair):
            to_remove = next(p for p in pair if p != '[REDACTED_BY_SCRIPT]')
            correlated_to_remove.add(to_remove); kept_due_to_priority.add('[REDACTED_BY_SCRIPT]'); continue
        
        if f1 in kept_due_to_priority: correlated_to_remove.add(f2); continue
        if f2 in kept_due_to_priority: correlated_to_remove.add(f1); continue

        unresolved_pairs.append((f1, f2))
    
    cols_to_remove.update(correlated_to_remove)
    analysis_report['[REDACTED_BY_SCRIPT]'] = sorted(list(correlated_to_remove))


    # --- Step 6: Finalize and Report ---
    print("[REDACTED_BY_SCRIPT]")
    final_cols_to_remove = [col for col in list(cols_to_remove) if col in df_processed.columns]
    df_final = df_processed.drop(columns=final_cols_to_remove)
    
    print("[REDACTED_BY_SCRIPT]")
    # ... Reporting logic ...
    print(f"[REDACTED_BY_SCRIPT]")

    print("\n--- Final Summary ---")
    print(f"[REDACTED_BY_SCRIPT]")
    print(f"[REDACTED_BY_SCRIPT]")
    print(f"[REDACTED_BY_SCRIPT]")
    
    return df_final


def main():
    """[REDACTED_BY_SCRIPT]"""
    df_full = load_data(FILE_PATH)
    if df_full is None: return

    if len(df_full) > SAMPLE_SIZE:
        print(f"[REDACTED_BY_SCRIPT]")
        df_to_process = df_full.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)
    else:
        print("[REDACTED_BY_SCRIPT]")
        df_to_process = df_full.copy()

    df_processed_analyzed = preprocess_and_analyze(df_to_process)
    
    try:
        df_processed_analyzed.to_csv(OUTPUT_FILENAME, index=False)
        print(f"[REDACTED_BY_SCRIPT]'{OUTPUT_FILENAME}'")
    except Exception as e:
        print(f"[REDACTED_BY_SCRIPT]")

if __name__ == '__main__':
    main()
