#!/bin/bash

# ==============================================================================
# run_avm_bias_analysis.sh - AVM Bias Prediction and Feature Generation Pipeline
#
# This script orchestrates a sophisticated machine learning pipeline to model
# and predict the systematic bias of several Automated Valuation Models (AVMs).
#
# The pipeline performs the following steps:
# 1.  Starts with a clean, imputed master dataset from a previous GWR pipeline.
# 2.  Identifies a "ground truth" subset of properties with recent sale prices.
# 3.  For each AVM, calculates the actual log-transformed bias against the sale price.
# 4.  Trains TWO types of bias prediction models (LightGBM) for each AVM:
#     - "Pure" Models: Predict bias using only fundamental property features,
#       simulating a model built in isolation.
#     - "Competitive" Models: Predict bias using fundamental features PLUS the
#       raw feature data from competing AVMs, simulating a more informed model.
# 5.  Uses these six trained models to generate bias predictions as new features
#     for the ENTIRE dataset.
# 6.  Saves the final, enriched dataset (with 6 new bias features) and all
#     trained models to Google Cloud Storage.
#
# This approach adheres to critical "lessons learned" regarding environment
# setup (PATH variable, unambiguous commands) to ensure reliable execution on GCP.
# ==============================================================================

# --- Strict Mode & Error Handling ---
set -e  # Exit immediately if a command exits with a non-zero status.
set -o pipefail # The return value of a pipeline is the status of the last command to exit with a non-zero status.
set -x # Print commands and their arguments as they are executed.

# --- Configuration ---
GCS_BUCKET="srgan-bucket-ace-botany-453819-t4"

# INPUT: The clean, imputed, and normalized dataset from the GWR pipeline.
INPUT_PARQUET_GCS_PATH="gs://${GCS_BUCKET}/models/gwr_outputs/imputed_normalized_master_data.parquet"

# OUTPUTS:
OUTPUT_GCS_DIR="models/avm_bias_outputs"
TIMESTAMP=$(date +"%Y%m%d-%H%M%S")
LOG_FILE_GCS_PATH="outputs/logs/avm_bias_analysis_${TIMESTAMP}.log"

# Local workspace on the VM
WORKDIR="${HOME}/avm_bias_work"
LOG_FILE="${WORKDIR}/run_pipeline.log"

# --- Main Execution ---
mkdir -p "${WORKDIR}"
cd "${WORKDIR}"

# Redirect all output to a log file AND the console
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "--- AVM Bias Analysis Pipeline Started: $(date) ---"

# --- Environment Setup (THE "GOLDEN" BLOCK) ---
echo "--- Setting up Python environment... ---"
sudo apt-get update -y && sudo apt-get install -y python3-pip git

echo "--- Upgrading pip using 'python3 -m pip' to be unambiguous... ---"
python3 -m pip install --user --upgrade pip

# CRITICAL FIX: The VM's default shell PATH does not include the user's local
# binary directory. This command adds it, ensuring that the newly upgraded pip
# and any subsequently installed packages are found and used. Without this,
# the script would fall back to the old system pip, causing dependency errors.
echo "--- Exporting new PATH to use upgraded pip and packages... ---"
export PATH="${HOME}/.local/bin:${PATH}"

echo "--- Verifying Environment ---"
echo "--- Current PATH: ${PATH} ---"
echo "--- Python3 version: $(python3 --version) ---"
echo "--- Pip version: $(pip --version) ---"

echo "--- Installing Python dependencies from requirements.txt... ---"
# We pin versions based on previous successful scripts for maximum stability.
# PyArrow is added for Parquet file support.
cat > requirements.txt << EOL
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.4.2
lightgbm==4.3.0
joblib==1.4.2
google-cloud-storage==2.16.0
pyarrow==16.1.0
EOL

python3 -m pip install --user --force-reinstall --no-cache-dir -r requirements.txt

echo "--- Verifying key installations... ---"
python3 -m pip show pandas scikit-learn lightgbm google-cloud-storage pyarrow

# --- Python Script Generation ---
echo "--- Generating the Python script for AVM Bias Modeling... ---"
cat > generate_bias_features.py << 'EOL'
import argparse
import logging
import re
from collections import defaultdict

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from google.cloud import storage
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# --- GCS Helper Functions ---
def download_gcs_file(bucket_name, source_blob_name, destination_file_path):
    """Downloads a file from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    logging.info(f"Downloading gs://{bucket_name}/{source_blob_name} to {destination_file_path}...")
    blob.download_to_filename(destination_file_path)
    logging.info("Download complete.")

def upload_to_gcs(bucket_name, source_file_path, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    logging.info(f"Uploading {source_file_path} to gs://{bucket_name}/{destination_blob_name}...")
    blob.upload_from_filename(source_file_path)
    logging.info("Upload complete.")

# --- Core Logic Functions ---
def define_feature_sets(columns):
    """
    Identifies and categorizes columns into base, pure, and competitive sets.
    This is the methodological core of the feature generation.
    """
    logging.info("Defining feature sets for 'pure' and 'competitive' models...")
    
    # NOTE: The user's GWR script cleaned columns, removing special chars.
    # We must match that cleaning to find our columns.
    # e.g., 'last_sold_date_year_(YYYY)_hm' -> 'last_sold_date_year_YYYY_hm'
    
    # Define AVM prefixes and key columns using the cleaned format
    avm_data = {
        'homipi': {
            'prefix': 'hm',
            'estimate_col': 'homipi_price_estimate_gbp_hm'
        },
        'mouseprice': {
            'prefix': 'mp',
            'estimate_col': 'mouseprice_estimated_value_gbp_numeric_string_mp'
        },
        'bnl': {
            'prefix': 'bnl',
            'estimate_col': 'bricksandlogic_estimated_price_gbp_numeric_extracted_or_original_text_bnl'
        }
    }
    
    # Identify all columns associated with each AVM by its prefix
    avm_raw_cols = defaultdict(list)
    for avm_name, info in avm_data.items():
        # A regex to find columns ending with '_hm', '_mp', etc.
        pattern = f'.*_{info["prefix"]}$'
        for col in columns:
            if re.match(pattern, col) and col != info['estimate_col']:
                 avm_raw_cols[avm_name].append(col)

    # Base features are columns that DO NOT belong to any AVM
    all_avm_cols = set()
    for cols in avm_raw_cols.values():
        all_avm_cols.update(cols)
    for info in avm_data.values():
        all_avm_cols.add(info['estimate_col'])
        
    base_features = [c for c in columns if c not in all_avm_cols]
    
    # Now, build the specific feature sets for each model
    feature_sets = {}
    avm_names = list(avm_data.keys())
    
    for i in range(len(avm_names)):
        current_avm_name = avm_names[i]
        other_avm_names = [name for name in avm_names if name != current_avm_name]
        
        # PURE model: uses only base features
        feature_sets[f'{current_avm_name}_pure'] = base_features
        
        # COMPETITIVE model: uses base features + raw data from other AVMs
        competitive_set = base_features.copy()
        for other_name in other_avm_names:
            competitive_set.extend(avm_raw_cols[other_name])
        feature_sets[f'{current_avm_name}_competitive'] = competitive_set

    for name, features in feature_sets.items():
        logging.info(f"  - Set '{name}' created with {len(features)} features.")
        
    return feature_sets, avm_data

def prepare_ground_truth(df, avm_data):
    """
    Filters for valid sales and calculates the log-transformed bias targets.
    """
    logging.info("Preparing ground truth data...")
    
    # Use log1p for robustness against zeros
    sale_price_col = 'most_recent_sale_price'
    gt_df = df[df[sale_price_col] > 0].copy()
    gt_df['log_sale_price'] = np.log1p(gt_df[sale_price_col])

    for avm_name, info in avm_data.items():
        estimate_col = info['estimate_col']
        # Ensure AVM estimate is positive before log transform
        gt_df = gt_df[gt_df[estimate_col] > 0]
        
        target_col = f'Target_Bias_Log_{avm_name}'
        gt_df[target_col] = np.log1p(gt_df[estimate_col]) - gt_df['log_sale_price']

    # Drop rows where any bias could not be calculated
    bias_cols = [f'Target_Bias_Log_{name}' for name in avm_data.keys()]
    gt_df.dropna(subset=bias_cols, inplace=True)
    
    logging.info(f"Ground truth data prepared. Shape: {gt_df.shape}")
    return gt_df

def main():
    parser = argparse.ArgumentParser(description="Train AVM bias models and generate features.")
    parser.add_argument("--gcs_bucket", required=True)
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    # --- 1. Load Data ---
    local_path = "input_data.parquet"
    download_gcs_file(args.gcs_bucket, args.input_path, local_path)
    df = pd.read_parquet(local_path)
    logging.info(f"Successfully loaded data with shape: {df.shape}")

    # --- 2. Define Feature and Target Sets ---
    # The GWR script cleans column names. We must get a fresh list of columns from the dataframe.
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    # Exclude identifiers and coordinates from feature list
    ids_and_coords = ['property_id', 'latitude', 'longitude', 'most_recent_sale_price']
    all_potential_features = [c for c in numeric_cols if c not in ids_and_coords]
    
    feature_sets, avm_data = define_feature_sets(all_potential_features)
    ground_truth_df = prepare_ground_truth(df, avm_data)

    if ground_truth_df.empty:
        logging.error("Ground truth dataset is empty after filtering. Cannot proceed with training.")
        raise ValueError("No valid ground truth data available.")

    # --- 3. Create Geographical Folds for CV ---
    logging.info("Creating geographical clusters for cross-validation...")
    coords = ground_truth_df[['latitude', 'longitude']].dropna()
    kmeans = KMeans(n_clusters=20, random_state=42, n_init=10) # 20 clusters for faster CV
    ground_truth_df['geo_cluster'] = kmeans.fit_predict(coords)

    # --- 4. Train Models & Generate Out-of-Fold Predictions ---
    logging.info("--- Starting Model Training & Out-of-Fold Prediction Phase ---")
    final_df = df.copy()
    group_kfold = GroupKFold(n_splits=5)

    for model_type_name, features in feature_sets.items():
        avm_name = model_type_name.split('_')[0]
        logging.info(f"\n--- Processing model: {model_type_name} ---")

        target_col = f'Target_Bias_Log_{avm_name}'
        X = ground_truth_df[features]
        y = ground_truth_df[target_col]
        groups = ground_truth_df['geo_cluster']
        
        # Initialize a new column for predictions in the final dataframe
        predicted_bias_col = f'predicted_log_bias_{model_type_name}'
        # Use a temporary holder for out-of-fold predictions
        oof_preds = pd.Series(np.nan, index=ground_truth_df.index)

        model_proto = lgb.LGBMRegressor(
            objective='regression_l1',
            metric='mae',
            n_estimators=1000,
            learning_rate=0.05,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            lambda_l1=0.1,
            num_leaves=31,
            verbose=-1,
            n_jobs=-1,
            seed=42,
        )

        # Generate out-of-fold predictions for the ground truth data
        logging.info(f"  - Generating out-of-fold predictions for ground truth data...")
        for train_idx, val_idx in group_kfold.split(X, y, groups):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train = y.iloc[train_idx]
            
            # Clone the model to ensure it's fresh for each fold
            fold_model = lgb.LGBMRegressor(**model_proto.get_params())
            fold_model.fit(X_train, y_train, 
                           eval_set=[(X_val, y.iloc[val_idx])], 
                           callbacks=[lgb.early_stopping(50, verbose=False)])
            
            oof_preds.iloc[val_idx] = fold_model.predict(X_val)

        # Add the out-of-fold predictions to the final dataframe
        final_df.loc[oof_preds.index, predicted_bias_col] = oof_preds
        logging.info(f"  - Out-of-fold predictions generated for {len(oof_preds.dropna())} properties.")

        # Train the final model on ALL ground truth data
        logging.info(f"  - Training final model on all ground truth data...")
        final_model = lgb.LGBMRegressor(**model_proto.get_params())
        final_model.fit(X, y) # No early stopping, use all data

        # Save final model and upload to GCS
        model_filename = f"{model_type_name}_predictor.pkl"
        joblib.dump(final_model, model_filename)
        logging.info(f"  - Final model trained and saved locally.")
        upload_to_gcs(args.gcs_bucket, model_filename, f"{args.output_dir}/models/{model_filename}")

    # --- 5. Generate Features on Full Dataset ---
    logging.info("\n--- Starting Feature Generation Phase ---")
    final_df = df.copy()

    for model_type_name, features in feature_sets.items():
        logging.info(f"Predicting with model: {model_type_name}")
        model_filename = f"{model_type_name}_predictor.pkl"
        
        # Download the model we just uploaded
        download_gcs_file(args.gcs_bucket, f"{args.output_dir}/models/{model_filename}", model_filename)
        model = joblib.load(model_filename)
        
        # Predict on the full dataframe's features
        X_full = final_df[features]
        
        predicted_bias_col = f'predicted_log_bias_{model_type_name}'
        final_df[predicted_bias_col] = model.predict(X_full)
        logging.info(f"  - Created feature: '{predicted_bias_col}'")

    # --- 6. Save Final Enriched Dataset ---
    output_filename = "final_dataset_with_bias_features.parquet"
    final_df.to_parquet(output_filename, index=False)
    logging.info(f"Final enriched data saved locally. Shape: {final_df.shape}")
    upload_to_gcs(args.gcs_bucket, output_filename, f"{args.output_dir}/{output_filename}")
    
    logging.info("--- Pipeline Finished Successfully! ---")

if __name__ == "__main__":
    main()
EOL

# --- Run the Python Script ---
echo "--- Starting AVM Bias Analysis Python Script ---"
# Note: The input path needs to be just the blob name, not the full gs:// path
INPUT_BLOB_NAME=$(echo "${INPUT_PARQUET_GCS_PATH}" | sed "s|gs://${GCS_BUCKET}/||")

python3 generate_bias_features.py \
    --gcs_bucket="${GCS_BUCKET}" \
    --input_path="${INPUT_BLOB_NAME}" \
    --output_dir="${OUTPUT_GCS_DIR}"

if [ $? -ne 0 ]; then
    echo "ERROR: AVM Bias Analysis script failed. Exiting pipeline."
    gsutil -m cp "${LOG_FILE}" "gs://${GCS_BUCKET}/outputs/logs/avm_bias_analysis_${TIMESTAMP}_FAILED.log"
    exit 1
fi

echo "--- AVM Bias Analysis Finished Successfully ---"

# --- Finalization: Upload Logs ---
echo "--- Uploading execution log to GCS... ---"
gsutil -m cp "${LOG_FILE}" "gs://${GCS_BUCKET}/${LOG_FILE_GCS_PATH}"

echo "--- Pipeline Finished: $(date) ---"