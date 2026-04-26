#!/bin/bash

# ==============================================================================
# run_gwr_pipeline_v2.sh - v2.0
#
# A robust, two-stage pipeline that uses Geographically Weighted Regression
# as a powerful feature engineering step (GWR-FE).
#
# STAGE 1: Performs advanced, multi-tiered imputation, creates latent
#          missingness features, and uses LassoCV for robust feature selection
#          to create a parsimonious feature set for GWR.
#
# STAGE 2: Trains a single, robust GWR model on the selected features and
#          harvests its outputs (spatially varying intercept, coefficients,
#          and local R-squared) as new, powerful features for a final
#          downstream model.
# ==============================================================================

# --- Strict Mode & Error Handling ---
set -e
set -o pipefail
set -x

# --- Configuration ---
GCS_BUCKET="srgan-bucket-ace-botany-453819-t4" # <-- IMPORTANT: Use your bucket name
INPUT_CSV_GCS_PATH="gs://${GCS_BUCKET}/features/final_master_dataset/master_feature_set.csv"
TIMESTAMP=$(date +"%Y%m%d-%H%M%S")
RUN_OUTPUT_DIR="gwr_fe_runs/${TIMESTAMP}"
LOG_FILE_GCS_PATH="${RUN_OUTPUT_DIR}/logs/gwr_pipeline_${TIMESTAMP}.log"

# Local workspace
WORKDIR="${HOME}/gwr_pipeline_work_v2"
LOG_FILE="${WORKDIR}/run_gwr_pipeline.log"

# --- Main Execution ---
mkdir -p "${WORKDIR}"
cd "${WORKDIR}"

# Redirect all output to a log file AND the console
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "--- GWR Feature Engineering Pipeline (v2) Started: $(date) ---"

# --- Phase 0: Environment Setup (One-Time) ---
echo "--- Ensuring a clean Python 3.11 virtual environment... ---"
if ! command -v python3.11 &> /dev/null; then
    echo "--- Installing Python 3.11... ---"
    sudo apt-get update -y
    sudo apt-get install -y python3.11 python3.11-venv git
fi

VENV_PATH="${WORKDIR}/gwr_env"
if [ ! -d "${VENV_PATH}" ]; then
    echo "--- Creating Python 3.11 virtual environment at ${VENV_PATH}... ---"
    python3.11 -m venv "${VENV_PATH}"
fi

echo "--- Activating virtual environment... ---"
source "${VENV_PATH}/bin/activate"
echo "--- Running with Python: $(which python) ---"

echo "--- Installing/updating Python dependencies... ---"
cat > requirements.txt << EOL
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.4.2
google-cloud-storage==2.16.0
mgwr==2.2.1
libpysal==4.12.0
pyarrow==16.1.0
scipy==1.13.1
EOL

pip install --upgrade pip
pip install -r requirements.txt
echo "--- Environment setup complete. ---"

# --- Generate Python Script for Stage 1: Preparation & Selection ---
echo "--- Generating Stage 1 script: 01_prepare_and_select.py ---"
cat > 01_prepare_and_select.py << 'EOL'
# 01_prepare_and_select.py
import argparse
import logging
import pandas as pd
import numpy as np
import gc
from pathlib import Path
from google.cloud import storage
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.linear_model import BayesianRidge, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
RANDOM_STATE = 42
N_MISSINGNESS_COMPONENTS = 30 # Number of PCA components for latent missingness

# --- GCS Helper Functions ---
def download_gcs_file(bucket_name, source_blob_name, dest_path):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    logging.info(f"Downloading gs://{bucket_name}/{source_blob_name} to {dest_path}...")
    blob.download_to_filename(str(dest_path))

def upload_to_gcs(bucket_name, source_path, dest_blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(dest_blob_name)
    logging.info(f"Uploading {source_path} to gs://{bucket_name}/{dest_blob_name}...")
    blob.upload_from_filename(str(source_path))

# --- Main Logic ---
def main():
    parser = argparse.ArgumentParser(description="Impute data and select features for GWR.")
    parser.add_argument("--gcs_bucket", required=True)
    parser.add_argument("--input_gcs_path", required=True)
    parser.add_argument("--imputed_data_gcs_path", required=True)
    parser.add_argument("--selected_features_gcs_path", required=True)
    args = parser.parse_args()

    local_work_dir = Path("/tmp/gwr_stage1")
    local_csv_path = local_work_dir / "master_feature_set.csv"
    download_gcs_file(args.gcs_bucket, args.input_gcs_path.split(f"gs://{args.gcs_bucket}/")[1], local_csv_path)
    
    logging.info("Loading full dataset...")
    df = pd.read_csv(local_csv_path, low_memory=False)

    roles = {'target': "most_recent_sale_price", 'coords': ['pcd_latitude', 'pcd_longitude'], 'ids': ['property_id']}
    non_feature_cols = {roles['target']} | set(roles['coords']) | set(roles['ids'])
    numeric_features = [c for c in df.columns if c not in non_feature_cols and pd.api.types.is_numeric_dtype(df[c])]
    features_df = df[numeric_features].copy()
    
    logging.info("Replacing placeholder values with np.nan...")
    features_df.replace([-1, -1.0, 0, 0.0], np.nan, inplace=True)

    # --- Advanced Multi-Tiered Imputation ---
    logging.info("--- Starting Advanced Imputation ---")
    nan_ratios = features_df.isna().sum() / len(features_df)

    # Tier 1: Drop >95% NaNs
    cols_to_drop = nan_ratios[nan_ratios > 0.95].index
    if not cols_to_drop.empty:
        features_df.drop(columns=cols_to_drop, inplace=True)
        logging.info(f"Tier 1: Dropped {len(cols_to_drop)} columns with >95% NaNs.")

    # Tier 2: Handle high and medium missingness with indicators
    nan_ratios = features_df.isna().sum() / len(features_df)
    indicator_cols = []

    # Tier 2a: 75-95% NaNs -> Create indicator, then DROP original
    cols_high_missing = nan_ratios[(nan_ratios > 0.75) & (nan_ratios <= 0.95)].index
    if not cols_high_missing.empty:
        logging.info(f"Tier 2a: Creating indicators and dropping {len(cols_high_missing)} original columns (75-95% missing).")
        for col in cols_high_missing:
            indicator_name = f"{col}_was_missing"
            features_df[indicator_name] = features_df[col].isna().astype(int)
            indicator_cols.append(indicator_name)
        features_df.drop(columns=cols_high_missing, inplace=True)

    # Tier 2b: 50-75% NaNs -> Create indicator, then simple impute original
    cols_medium_missing = nan_ratios[(nan_ratios > 0.50) & (nan_ratios <= 0.75)].index
    if not cols_medium_missing.empty:
        logging.info(f"Tier 2b: Creating indicators and imputing {len(cols_medium_missing)} original columns (50-75% missing).")
        for col in cols_medium_missing:
            indicator_name = f"{col}_was_missing"
            features_df[indicator_name] = features_df[col].isna().astype(int)
            indicator_cols.append(indicator_name)
        simple_imputer = SimpleImputer(strategy='median')
        features_df[cols_medium_missing] = simple_imputer.fit_transform(features_df[cols_medium_missing])

    # Tier 2c: Create Latent Missingness Features via PCA
    if indicator_cols:
        logging.info(f"Compressing {len(indicator_cols)} missingness indicators into {N_MISSINGNESS_COMPONENTS} latent features using PCA...")
        indicator_df = features_df[indicator_cols]
        pca = PCA(n_components=N_MISSINGNESS_COMPONENTS, random_state=RANDOM_STATE)
        missingness_pca_features = pca.fit_transform(indicator_df)
        
        pca_cols = [f'missingness_pca_{i+1}' for i in range(N_MISSINGNESS_COMPONENTS)]
        missingness_pca_df = pd.DataFrame(missingness_pca_features, columns=pca_cols, index=features_df.index)
        
        features_df.drop(columns=indicator_cols, inplace=True)
        features_df = pd.concat([features_df, missingness_pca_df], axis=1)
        logging.info("Latent missingness features created and integrated.")

    # Tier 3: Iterative Imputation for remaining NaNs (<50%)
    if features_df.isna().sum().sum() > 0:
        logging.info("Tier 3: Applying IterativeImputer. This may take time...")
        imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=RANDOM_STATE, verbose=2)
        imputed_array = imputer.fit_transform(features_df)
        features_df = pd.DataFrame(imputed_array, columns=features_df.columns, index=features_df.index)
    
    logging.info("--- Imputation Complete ---")
    
    # --- Feature Selection using LassoCV ---
    logging.info("--- Starting Feature Selection with LassoCV ---")
    # Align target variable with feature dataframe, dropping rows with missing target
    y = df[roles['target']].copy()
    aligned_df = pd.concat([y, features_df], axis=1).dropna(subset=[roles['target']])
    y_aligned = aligned_df[roles['target']]
    X_aligned = aligned_df.drop(columns=[roles['target']])

    logging.info("Scaling data before applying Lasso.")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_aligned)
    
    logging.info("Fitting LassoCV to find optimal features for GWR...")
    lasso = LassoCV(cv=5, random_state=RANDOM_STATE, n_jobs=-1, verbose=2).fit(X_scaled, y_aligned)
    
    selected_mask = lasso.coef_ != 0
    selected_features = X_aligned.columns[selected_mask].tolist()
    logging.info(f"Lasso selected {len(selected_features)} features for the GWR model.")

    local_features_path = local_work_dir / "selected_gwr_features.txt"
    with open(local_features_path, 'w') as f:
        for feature in selected_features:
            f.write(f"{feature}\n")
    upload_to_gcs(args.gcs_bucket, str(local_features_path), args.selected_features_gcs_path)

    # --- Save the fully imputed master dataset ---
    logging.info("Constructing and saving the final imputed master dataset...")
    # We need to save the scaled data for the next step
    X_scaled_df = pd.DataFrame(X_scaled, columns=X_aligned.columns, index=X_aligned.index)
    final_df_to_save = pd.concat([df.loc[X_aligned.index, roles['ids'] + roles['coords'] + [roles['target']]], X_scaled_df], axis=1)

    imputed_data_path = local_work_dir / "imputed_master_data_scaled.parquet"
    final_df_to_save.to_parquet(imputed_data_path, index=False)
    upload_to_gcs(args.gcs_bucket, str(imputed_data_path), args.imputed_data_gcs_path)

    logging.info("--- Stage 1 Complete. ---")

if __name__ == "__main__":
    main()
EOL


# --- Generate Python Script for Stage 2: GWR Feature Engineering ---
echo "--- Generating Stage 2 script: 02_run_gwr_feature_engineering.py ---"
cat > 02_run_gwr_feature_engineering.py << 'EOL'
# 02_run_gwr_feature_engineering.py
import argparse
import logging
import pandas as pd
import numpy as np 
from pathlib import Path
from google.cloud import storage
from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- GCS Helper Functions ---
def download_gcs_file(bucket_name, source_blob_name, dest_path):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    logging.info(f"Downloading gs://{bucket_name}/{source_blob_name} to {dest_path}...")
    blob.download_to_filename(str(dest_path))

def upload_to_gcs(bucket_name, source_path, dest_blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(dest_blob_name)
    logging.info(f"Uploading {source_path} to gs://{bucket_name}/{dest_blob_name}...")
    blob.upload_from_filename(str(source_path))

# --- Main Logic ---
def main():
    parser = argparse.ArgumentParser(description="Run GWR as a feature engineering step.")
    parser.add_argument("--gcs_bucket", required=True)
    parser.add_argument("--imputed_data_gcs_path", required=True)
    parser.add_argument("--selected_features_gcs_path", required=True)
    parser.add_argument("--gwr_features_gcs_path", required=True)
    parser.add_argument("--gwr_summary_gcs_path", required=True)
    args = parser.parse_args()

    local_work_dir = Path("/tmp/gwr_stage2")
    
    # Download the list of selected features
    local_features_path = local_work_dir / "selected_gwr_features.txt"
    download_gcs_file(args.gcs_bucket, args.selected_features_gcs_path, local_features_path)
    with open(local_features_path, 'r') as f:
        selected_features = [line.strip() for line in f]
    logging.info(f"Loaded {len(selected_features)} features selected by LassoCV.")

    # Download the fully imputed and scaled master data
    local_parquet_path = local_work_dir / "imputed_master_data_scaled.parquet"
    download_gcs_file(args.gcs_bucket, args.imputed_data_gcs_path, local_parquet_path)
    df = pd.read_parquet(local_parquet_path)
    
    # Prepare model inputs
    y = df['most_recent_sale_price'].values.reshape(-1, 1)
    coords = df[['pcd_longitude', 'pcd_latitude']].values
    # Data is already scaled from Stage 1
    X = df[selected_features].values 
    ids_df = df[['property_id']].copy()
    
    logging.info(f"Data prepared for GWR with {X.shape[0]} samples and {X.shape[1]} features.")

    # Wrap model fitting in a try...except block
    try:
        # Bandwidth Selection
        logging.info("Starting bandwidth selection for GWR... This is the most compute-intensive step.")
        selector = Sel_BW(coords, y, X)
        bandwidth = selector.search(verbose=True)
        logging.info(f"Optimal bandwidth found: {bandwidth}")

        # Fit GWR Model
        logging.info("Fitting GWR model with the optimal bandwidth...")
        model = GWR(coords, y, X, bandwidth)
        results = model.fit()
        logging.info(f"Model Diagnostics: AICc={results.aicc:.2f}, R2={results.R2:.4f}")

        # --- HARVEST GWR OUTPUTS AS NEW FEATURES ---
        logging.info("Harvesting GWR outputs as new spatial features...")
        
        # 1. Spatially Varying Coefficients
        gwr_coeff_names = [f'gwr_coeff_{feat}' for feat in selected_features]
        gwr_params_df = pd.DataFrame(results.params[:, 1:], columns=gwr_coeff_names)

        # 2. Spatially Varying Intercept
        gwr_intercept_df = pd.DataFrame(results.params[:, 0], columns=['gwr_intercept'])

        # 3. Local R-squared
        gwr_local_r2_df = pd.DataFrame(results.localR2, columns=['gwr_local_r2'])

        # Combine all new features with property IDs
        gwr_features_df = pd.concat([
            ids_df.reset_index(drop=True), 
            gwr_intercept_df,
            gwr_params_df,
            gwr_local_r2_df
        ], axis=1)

        # Save and upload the new features
        local_gwr_features_path = local_work_dir / "gwr_generated_features.parquet"
        gwr_features_df.to_parquet(local_gwr_features_path, index=False)
        upload_to_gcs(args.gcs_bucket, str(local_gwr_features_path), args.gwr_features_gcs_path)
        logging.info("Successfully saved GWR-generated features to GCS.")

        # Save and upload the model summary
        local_summary_path = local_work_dir / "gwr_summary.txt"
        summary_text = results.summary()
        with open(local_summary_path, 'w') as f:
            f.write(summary_text)
        upload_to_gcs(args.gcs_bucket, str(local_summary_path), args.gwr_summary_gcs_path)

    except Exception as e:
        logging.error(f"A fatal error occurred during GWR processing: {e}", exc_info=True)
        logging.error("GWR Feature Engineering failed. Check logs for details.")
        exit(1) # Exit with an error code

    logging.info("--- GWR Feature Engineering (Stage 2) complete. ---")

if __name__ == "__main__":
    main()
EOL


# --- Define GCS paths for the new workflow ---
IMPUTED_DATA_GCS_PATH="${RUN_OUTPUT_DIR}/imputed_data/imputed_master_data_scaled.parquet"
SELECTED_FEATURES_GCS_PATH="${RUN_OUTPUT_DIR}/selected_features/gwr_features.txt"
GWR_FEATURES_GCS_PATH="${RUN_OUTPUT_DIR}/gwr_outputs/gwr_generated_features.parquet"
GWR_SUMMARY_GCS_PATH="${RUN_OUTPUT_DIR}/gwr_outputs/gwr_summary.txt"

# --- Execute Stage 1: Data Preparation & Feature Selection ---
echo "--- EXECUTING STAGE 1: DATA PREPARATION AND SELECTION ---"
python 01_prepare_and_select.py \
    --gcs_bucket="${GCS_BUCKET}" \
    --input_gcs_path="${INPUT_CSV_GCS_PATH}" \
    --imputed_data_gcs_path="${IMPUTED_DATA_GCS_PATH}" \
    --selected_features_gcs_path="${SELECTED_FEATURES_GCS_PATH}"

if [ $? -ne 0 ]; then
    echo "ERROR: Stage 1 (Data Prep & Selection) failed. Uploading log and exiting."
    gsutil -m cp "${LOG_FILE}" "gs://${GCS_BUCKET}/${LOG_FILE_GCS_PATH}_FAILED_STAGE1.log"
    exit 1
fi

# --- Execute Stage 2: GWR Feature Engineering ---
echo "--- EXECUTING STAGE 2: GWR FEATURE ENGINEERING ---"
python 02_run_gwr_feature_engineering.py \
    --gcs_bucket="${GCS_BUCKET}" \
    --imputed_data_gcs_path="${IMPUTED_DATA_GCS_PATH}" \
    --selected_features_gcs_path="${SELECTED_FEATURES_GCS_PATH}" \
    --gwr_features_gcs_path="${GWR_FEATURES_GCS_PATH}" \
    --gwr_summary_gcs_path="${GWR_SUMMARY_GCS_PATH}"

if [ $? -ne 0 ]; then
    echo "ERROR: Stage 2 (GWR Feature Engineering) failed. Uploading log and exiting."
    gsutil -m cp "${LOG_FILE}" "gs://${GCS_BUCKET}/${LOG_FILE_GCS_PATH}_FAILED_STAGE2.log"
    exit 1
fi

# --- Finalization ---
echo "--- GWR Feature Engineering Pipeline Finished Successfully ---"
echo "Final artifacts available in: gs://${GCS_BUCKET}/${RUN_OUTPUT_DIR}/"
echo "--- Uploading final execution log to GCS... ---"
gsutil -m cp "${LOG_FILE}" "gs://${GCS_BUCKET}/${LOG_FILE_GCS_PATH}.log"

echo "--- Done: $(date) ---"