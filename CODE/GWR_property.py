# =============================================================================
# Phase 0: Import Libraries
# =============================================================================
import pandas as pd
import numpy as np
import re
# Correctly import both GWR and MGWR for potential future comparison
from mgwr.gwr import GWR, MGWR
from mgwr.sel_bw import Sel_BW
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
print("[REDACTED_BY_SCRIPT]")

# =============================================================================
# Phase 1: Data Loading and Address Normalization
# =============================================================================
# --- User Configuration ---
PROPERTY_FILE_PATH = r"[REDACTED_BY_SCRIPT]"
GEMINI_FILE_PATH = r"[REDACTED_BY_SCRIPT]"
PROPERTY_KEY_COL = '[REDACTED_BY_SCRIPT]'
GEMINI_KEY_COL = 'property_id'
TARGET_VARIABLE = '[REDACTED_BY_SCRIPT]'

# --- Load Data ---
try:
    df_property = pd.read_csv(PROPERTY_FILE_PATH)
    print(f"[REDACTED_BY_SCRIPT]")
    df_gemini = pd.read_csv(GEMINI_FILE_PATH)
    print(f"[REDACTED_BY_SCRIPT]")
except FileNotFoundError as e:
    print(f"[REDACTED_BY_SCRIPT]")
    exit()

# --- Address Normalization Function ---
def normalize_address(address):
    if not isinstance(address, str): return ""
    address = address.lower()
    address = re.sub(r'[^a-z0-9\s]', '', address)
    address = re.sub(r'\s+', ' ', address)
    return address.strip()

# --- Create and Merge on Normalized Keys ---
print("[REDACTED_BY_SCRIPT]")
df_property['[REDACTED_BY_SCRIPT]'] = df_property[PROPERTY_KEY_COL].apply(normalize_address)
df_gemini['[REDACTED_BY_SCRIPT]'] = df_gemini[GEMINI_KEY_COL].apply(normalize_address)
df_merged = pd.merge(df_property, df_gemini, on='[REDACTED_BY_SCRIPT]', how='inner')
print(f"[REDACTED_BY_SCRIPT]")
if df_merged.empty:
    print("[REDACTED_BY_SCRIPT]")
    exit()

# =============================================================================
# Phase 2: Feature Engineering and Selection (v2.0 - CORRECTED NAMES)
# =============================================================================
print("[REDACTED_BY_SCRIPT]")
renovation_cols = ['[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]']
df_merged['[REDACTED_BY_SCRIPT]'] = df_merged[renovation_cols].mean(axis=1)

# --- Define Feature Lists with CORRECTED Column Names ---
# The two problematic column names have been manually truncated to match how pandas reads them.
GWR_FEATURES_PROPERTY = [
    '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]',  # <<< CORRECTED
    '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]', # <<< CORRECTED
    '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]'
]
GWR_FEATURES_GEMINI = ["[REDACTED_BY_SCRIPT]", "[REDACTED_BY_SCRIPT]", "[REDACTED_BY_SCRIPT]"]
GWR_FEATURES_ALL = GWR_FEATURES_PROPERTY + GWR_FEATURES_GEMINI

print(f"[REDACTED_BY_SCRIPT]")

# --- Prepare main modeling variables ---
df_clean = df_merged.dropna(subset=[TARGET_VARIABLE, 'latitude', 'longitude']).copy()
print(f"[REDACTED_BY_SCRIPT]")

# =============================================================================
# Phase 3: Target Variable Transformation
# =============================================================================
print("[REDACTED_BY_SCRIPT]")
y_original = df_clean[TARGET_VARIABLE].values
target_transformer = PowerTransformer(method='yeo-johnson', standardize=True)
y_transformed = target_transformer.fit_transform(y_original.reshape(-1, 1)).flatten()
y = y_transformed.reshape(-1, 1)
X = df_clean[GWR_FEATURES_ALL]
coords = df_clean[['longitude', 'latitude']].values
print("[REDACTED_BY_SCRIPT]")

# =============================================================================
# Phase 4: Advanced Preprocessing Pipeline with Multiple Strategies
# =============================================================================
print("[REDACTED_BY_SCRIPT]")
numerical_property_transformer = Pipeline(steps=[
    ('imputer', KNNImputer(n_neighbors=5)),
    ('transformer', PowerTransformer(method='yeo-johnson')),
    ('scaler', StandardScaler())])
numerical_gemini_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value=0.0)),
    ('transformer', PowerTransformer(method='yeo-johnson')),
    ('scaler', StandardScaler())])

preprocessor = ColumnTransformer(
    transformers=[
        ('prop', numerical_property_transformer, GWR_FEATURES_PROPERTY),
        ('gemini', numerical_gemini_transformer, GWR_FEATURES_GEMINI)
    ],
    remainder='passthrough')

X_processed = preprocessor.fit_transform(X)
print(f"[REDACTED_BY_SCRIPT]")

# =============================================================================
# Phase 5: Multiscale GWR (MGWR) Bandwidth Selection & Model Fitting
# =============================================================================
print("[REDACTED_BY_SCRIPT]")
print("[REDACTED_BY_SCRIPT]")

# The Sel_BW class is used for MGWR, but we specify multi=True to trigger the
# more complex, iterative back-fitting algorithm required for MGWR.
selector = Sel_BW(coords, y, X_processed, multi=True, spherical=True)
# Remove the problematic multi_scale_kwargs parameter
selector.search()

print(f"[REDACTED_BY_SCRIPT]")

# Initialize and fit the MGWR model using the selector object which now contains
# the optimal bandwidth for each variable.
model = MGWR(coords, y, X_processed, selector=selector, spherical=True)
results = model.fit()

print("[REDACTED_BY_SCRIPT]")
print("[REDACTED_BY_SCRIPT]")
# The MGWR summary is more detailed and includes per-variable information.
print(results.summary())


# =============================================================================
# Phase 6: Extracting, Analyzing, and Saving MGWR Results
# =============================================================================
print("[REDACTED_BY_SCRIPT]")

# Use get_feature_names_out() for robust column naming from the preprocessor
final_feature_names = preprocessor.get_feature_names_out()

# Create the list of column names for the coefficients, including the Intercept
coeff_column_names = ['Intercept'] + list(final_feature_names)

# Create the DataFrame with the local coefficients
local_coeffs = pd.DataFrame(results.params, columns=[f'mgwr_coeff_{name}' for name in coeff_column_names])

# Add the local R-squared values as a measure of local model performance
local_coeffs['mgwr_local_R2'] = results.localR2
print("[REDACTED_BY_SCRIPT]")


# --- CRITICAL NEW INSIGHT: Display Per-Variable Bandwidths ---
print("[REDACTED_BY_SCRIPT]")
print("[REDACTED_BY_SCRIPT]")
bandwidth_results = pd.DataFrame({
    'Feature': coeff_column_names,
    '[REDACTED_BY_SCRIPT]': results.bws
})
print(bandwidth_results.to_string())


# --- Merge new MGWR features back into the original data ---
df_clean.reset_index(drop=True, inplace=True)
local_coeffs.reset_index(drop=True, inplace=True)
df_final_features = pd.concat([df_clean, local_coeffs], axis=1)

# Add original and transformed target values for reference
df_final_features['target_original'] = y_original
df_final_features['target_transformed'] = y_transformed


# --- Save final results and transformer ---
output_file_path = r"[REDACTED_BY_SCRIPT]"
transformer_path = r"[REDACTED_BY_SCRIPT]"
try:
    df_final_features.to_csv(output_file_path, index=False)
    print(f"[REDACTED_BY_SCRIPT]")
    joblib.dump(target_transformer, transformer_path)
    print(f"[REDACTED_BY_SCRIPT]")
except Exception as e:
    print(f"[REDACTED_BY_SCRIPT]")

# =============================================================================
# Phase 7: Usage Guide
# =============================================================================
print("[REDACTED_BY_SCRIPT]")
print("[REDACTED_BY_SCRIPT]")
print("[REDACTED_BY_SCRIPT]'target_transformer_v7_final.pkl')")
print("[REDACTED_BY_SCRIPT]")
print("[REDACTED_BY_SCRIPT]")