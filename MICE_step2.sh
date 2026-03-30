#!/bin/bash

# ==============================================================================
# MICE_step2.sh - LightGBM Imputation
# DESCRIPTION:
# This script performs the second stage of imputation using a MICE (Multivariate
# Imputation by Chained Equations) strategy with LightGBM. It targets any
# remaining null values in the master dataset produced by the deep learning
# imputation pipeline.
#
# ARCHITECTURE:
# 1. Downloads the master feature set from GCS.
# 2. Generates a Python script to handle the imputation logic.
# 3. The Python script iterates through each column with missing data:
#    a. It trains a LightGBM model on the non-missing data.
#    b. It predicts and fills the missing values for that column.
# 4. This process is repeated for a set number of cycles to allow imputed
#    values to stabilize.
# 5. The final, fully-imputed dataset is uploaded back to GCS.
# ==============================================================================

# --- Strict Mode & Error Handling ---
set -e
set -o pipefail
set -x

# --- Configuration ---
GCS_BUCKET="srgan-bucket-ace-botany-453819-t4"
TIMESTAMP=$(date +"%Y%m%d-%H%M%S")

# === ACTION REQUIRED: Verify these GCS paths ===
# Input should be the output directory from the previous MICE script
INPUT_GCS_DIR="features/geospatial_pipeline_v16_full_with_pp"
INPUT_FILENAME="final_geospatial_enriched_dataset.parquet"

# Output directory for this LightGBM imputation step
OUTPUT_GCS_DIR="imputation_pipeline/output_lgbm_legacy"

# Local workspace and logging
WORKDIR="${HOME}/mice_step2_work"
LOG_FILE="${WORKDIR}/run_lgbm_imputation.log"
LOG_FILE_GCS_PATH="${OUTPUT_GCS_DIR}/logs/pipeline_run.log"

# --- MICE Configuration ---
MICE_ITERATIONS=5 # Number of times to cycle through all columns

# --- NEW: LightGBM Performance Tuning ---
LGBM_N_ESTIMATORS=50      # Number of trees per model (default: 100). Lower is faster.
LGBM_NUM_LEAVES=20        # Max leaves per tree (default: 31). Lower is faster.
LGBM_MAX_FEATURES=500     # Use a random subset of features for each model. Speeds up high-dimensional data. Set to -1 to disable.

# --- Main Execution ---
mkdir -p "${WORKDIR}"
cd "${WORKDIR}"

exec > >(tee -a "${LOG_FILE}") 2>&1

echo "--- LightGBM MICE Imputation Pipeline Started: $(date) ---"

# --- Environment Setup ---
echo "--- Setting up Python environment... ---"
sudo apt-get update -y && sudo apt-get install -y python3-pip git
python3 -m pip install --user --upgrade pip
export PATH="/home/jupyter/.local/bin:${PATH}"
echo "--- Pip version: $(pip --version) ---"

echo "--- Installing Python dependencies... ---"
cat > requirements.txt << EOL
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.4.2
google-cloud-storage==2.16.0
pyarrow==15.0.0
lightgbm==4.3.0
nvidia-cudnn-cu11==8.6.0.163
tqdm==4.66.4
EOL

python3 -m pip install --user --force-reinstall --no-cache-dir -r requirements.txt
python3 -m pip show pandas scikit-learn google-cloud-storage lightgbm

# Install NVIDIA CUDA toolkit
echo "--- Installing NVIDIA CUDA toolkit... ---"
sudo apt-get install -y nvidia-cuda-toolkit

# --- Python Script Generation ---
echo "--- Generating Python Script: impute_mice_lightgbm.py ---"
cat > impute_mice_lightgbm.py << 'EOL'
import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb
from google.cloud import storage
from tqdm import tqdm
import re

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler()])

# --- GCS Helper Functions ---
def download_gcs_file(bucket_name: str, source_blob_name: str, destination_file_path: Path):
    """Downloads a file from GCS."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    
    logging.info(f"Downloading gs://{bucket_name}/{source_blob_name} to {destination_file_path}...")
    destination_file_path.parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(str(destination_file_path))
    logging.info("Download complete.")

def upload_to_gcs(bucket_name: str, source_file_path: str, destination_blob_name: str):
    """Uploads a file to GCS."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    logging.info(f"Uploading {source_file_path} to gs://{bucket_name}/{destination_blob_name}...")
    blob.upload_from_filename(source_file_path)
    logging.info("Upload complete.")

def make_unique_columns(df_columns):
    """Generates a list of unique column names by appending '_<count>' to duplicates."""
    seen = {}
    new_columns = []
    for col in df_columns:
        if col not in seen:
            seen[col] = 1
            new_columns.append(col)
        else:
            count = seen[col]
            new_name = f"{col}_{count}"
            while new_name in seen:
                count += 1
                new_name = f"{col}_{count}"
            new_columns.append(new_name)
            seen[col] = count + 1
    return new_columns

# --- Main Imputation Logic ---
def main():
    parser = argparse.ArgumentParser(description="Run LightGBM MICE Imputation.")
    parser.add_argument("--gcs_bucket", required=True)
    parser.add_argument("--input_gcs_path", required=True)
    parser.add_argument("--output_gcs_path", required=True)
    parser.add_argument("--iterations", type=int, required=True)
    # --- NEW: Add arguments for performance tuning ---
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--num_leaves", type=int, default=31)
    parser.add_argument("--max_features", type=int, default=-1)
    args = parser.parse_args()

    local_work_dir = Path("/tmp/mice_lgbm_data")
    local_input_path = local_work_dir / "input_data.parquet"
    local_output_path = local_work_dir / "final_imputed_data.parquet"

    # Download the dataset
    download_gcs_file(args.gcs_bucket, args.input_gcs_path, local_input_path)

    # Load data
    df = pd.read_parquet(local_input_path)
    logging.info(f"Successfully loaded data. Shape: {df.shape}")

    # --- FIX: Sanitize column names for LightGBM compatibility FIRST ---
    logging.info("Sanitizing column names for LightGBM...")
    original_columns = df.columns.tolist()
    # Replace any character that is not a letter, number, or underscore with an underscore
    sanitized_columns = [re.sub(r'[^A-Za-z0-9_]+', '_', col) for col in original_columns]
    df.columns = sanitized_columns
    rename_count = sum(1 for orig, new in zip(original_columns, sanitized_columns) if orig != new)
    if rename_count > 0:
        logging.info(f"Renamed {rename_count} columns to be LightGBM-compatible.")

    # --- FIX: Run de-duplication AFTER sanitizing to catch new duplicates ---
    logging.info("Checking for and resolving duplicate column names post-sanitization...")
    original_columns_count = len(df.columns)
    df.columns = make_unique_columns(df.columns)
    if len(set(df.columns)) != original_columns_count:
        logging.warning("Duplicate column names were found and resolved after sanitization.")
    else:
        logging.info("No duplicate column names found after sanitization.")

    # Identify columns with missing values
    initial_cols_with_missing = df.columns[df.isnull().any()].tolist()
    
    # Exclude non-feature columns (like IDs) and all temporally-imputed 'pp_' columns from the imputation target list.
    # This is critical because 'pp_' columns have their own temporal imputation logic and should not be altered here.
    # Note: Column names have been sanitized, so we check for '_pp_' or a 'pp_' prefix.
    cols_with_missing = [
        col for col in initial_cols_with_missing
        if col not in {'property_id', 'property_address'} and not ('_pp_' in col or col.startswith('pp_'))
    ]

    num_excluded = len(initial_cols_with_missing) - len(cols_with_missing)
    logging.info(f"Identified {len(initial_cols_with_missing)} total columns with missing values. Excluding {num_excluded} (IDs, 'pp_' features, etc.).")

    if not cols_with_missing:
        logging.info("No missing values found in non-'pp_' columns. No MICE imputation needed. Exiting.")
        # Upload the original file as the output
        upload_to_gcs(args.gcs_bucket, str(local_input_path), args.output_gcs_path)
        return

    logging.info(f"Found {len(cols_with_missing)} columns with missing values to impute.")

    # MICE procedure
    for i in range(args.iterations):
        logging.info(f"\n--- MICE Iteration {i + 1}/{args.iterations} ---")
        
        for col_to_impute in tqdm(cols_with_missing, desc=f"Imputing columns (iteration {i+1})"):
            
            # Define features (all columns except the target)
            all_features = [col for col in df.columns if col != col_to_impute]
            
            # --- MODIFICATION: Exclude all price paid ('pp_') historical features from the predictor set ---
            # This prevents data leakage from temporally imputed columns into other features.
            # We check for '_pp_' (e.g., in compass_mean_pp_...) and 'pp_' prefixes (raw temporal features).
            non_pp_features = [col for col in all_features if not ('_pp_' in col or col.startswith('pp_'))]
            
            # --- SPEEDUP: Use a random subset of features from the filtered list ---
            if args.max_features > 0 and len(non_pp_features) > args.max_features:
                features = np.random.choice(non_pp_features, args.max_features, replace=False).tolist()
            else:
                features = non_pp_features
            
            # --- FIX: Use .copy() to prevent SettingWithCopyWarning ---
            # Split data into training set (where target is not null) and prediction set (where target is null)
            train_df = df[df[col_to_impute].notnull()].copy()
            predict_df = df[df[col_to_impute].isnull()].copy()

            if predict_df.empty:
                logging.info(f"Column '{col_to_impute}' has no missing values to impute. Skipping.")
                continue

            # Correctly handle categorical features on the copied dataframes to avoid SettingWithCopyWarning
            # and ensure LightGBM receives compatible dtypes (no raw 'object' strings).
            for col in features:
                if train_df[col].dtype.name == 'object':
                    train_df[col] = train_df[col].astype('category')
                    predict_df[col] = predict_df[col].astype('category')

            X_train = train_df[features]
            y_train = train_df[col_to_impute]
            X_predict = predict_df[features]


            # --- ROBUSTNESS: Dynamically select device based on feature cardinality ---
            GPU_MAX_BINS = 255
            device_type = 'gpu'
            lgbm_params = {
                'objective': 'regression_l1',
                'n_estimators': args.n_estimators,
                'num_leaves': args.num_leaves,
                'learning_rate': 0.1,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'verbose': -1
            }

            # FIX: Check ALL features for high cardinality, as LGBM can treat integers as categoricals.
            for feature_col in X_train.columns:
                if X_train[feature_col].nunique() >= GPU_MAX_BINS:
                    logging.warning(
                        f"High cardinality feature '{feature_col}' (dtype: {X_train[feature_col].dtype}, "
                        f"unique values: {X_train[feature_col].nunique()}) detected. "
                        f"Switching to CPU for imputing '{col_to_impute}' to avoid GPU bin limit."
                    )
                    device_type = 'cpu'
                    break # Found a problematic column, decision is made.
            
            if device_type == 'gpu':
                lgbm_params.update({
                    'device': 'gpu',
                    'gpu_platform_id': 0,
                    'gpu_device_id': 0
                })
            
            # Train LightGBM model
            lgbm = lgb.LGBMRegressor(**lgbm_params)
            lgbm.fit(X_train, y_train)

            # Predict missing values
            predicted_values = lgbm.predict(X_predict)

            # Fill in the missing values in the original dataframe
            df.loc[df[col_to_impute].isnull(), col_to_impute] = predicted_values

    logging.info("\n--- MICE Imputation Complete ---")
    final_missing_count = df.isnull().sum().sum()
    if final_missing_count == 0:
        logging.info("Successfully imputed all missing values.")
    else:
        logging.warning(f"Imputation finished, but {final_missing_count} missing values remain.")

    # Save and upload the final dataset
    df.to_parquet(local_output_path, index=False)
    upload_to_gcs(args.gcs_bucket, str(local_output_path), args.output_gcs_path)
    logging.info("--- Pipeline Finished Successfully ---")

if __name__ == "__main__":
    main()
EOL

# --- Run the Main Python Script ---
echo "--- Starting the LightGBM MICE Imputation Script ---"

python3 impute_mice_lightgbm.py \
    --gcs_bucket="${GCS_BUCKET}" \
    --input_gcs_path="${INPUT_GCS_DIR}/${INPUT_FILENAME}" \
    --output_gcs_path="${OUTPUT_GCS_DIR}/final_fully_imputed_dataset.parquet" \
    --iterations=${MICE_ITERATIONS} \
    --n_estimators=${LGBM_N_ESTIMATORS} \
    --num_leaves=${LGBM_NUM_LEAVES} \
    --max_features=${LGBM_MAX_FEATURES}

if [ $? -ne 0 ]; then
    echo "ERROR: LightGBM MICE script failed. Exiting pipeline."
    gsutil -m cp "${LOG_FILE}" "gs://${GCS_BUCKET}/${LOG_FILE_GCS_PATH}_FAILED"
    exit 1
fi

# --- Finalization: Upload Logs ---
echo "--- Uploading execution log to GCS... ---"
gsutil -m cp "${LOG_FILE}" "gs://${GCS_BUCKET}/${LOG_FILE_GCS_PATH}"

echo "--- Pipeline Finished: $(date) ---"