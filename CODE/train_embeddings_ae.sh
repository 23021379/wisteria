#!/bin/bash

# ==============================================================================
# run_modular_pipeline.sh - Orchestrates a Modular Autoencoder Training Pipeline
#
# This script executes a two-phase process based on a modular, specialized
# autoencoder architecture.
#
# 1. Preprocesses raw CSV data, splitting it into multiple NumPy binary files,
#    one for each semantic group of embeddings (e.g., persona, kitchen, etc.).
# 2. Iteratively trains a specialized autoencoder for each data group using
#    Optuna to find the best hyperparameters for that specific group.
#
# This approach improves training efficiency... by splitting the wide 2D
# feature matrix into multiple, smaller 2D matrices (one per semantic group),
# allowing each autoencoder to become an expert...
# ==============================================================================

# --- Strict Mode & Error Handling ---
set -e  # Exit immediately if a command exits with a non-zero status.
set -o pipefail # The return value of a pipeline is the status of the last command to exit with a non-zero status.
set -x # Print commands and their arguments as they are executed.

# --- Configuration ---
GCS_BUCKET="srgan-bucket-ace-botany-453819-t4"
INPUT_CSV_GCS_PATH="gs://${GCS_BUCKET}/house data scrape/property_features_qualitative_embeddings_v4.csv"

# Phase 1 Output / Phase 2 Input (Now a directory)
PREPROCESSED_DATA_GCS_DIR="preprocessed_modular_data"

# Phase 2 Outputs (Now base directories)
MODULAR_MODEL_OUTPUT_GCS_DIR="models/modular_autoencoders"
MODULAR_STUDY_DB_GCS_DIR="outputs/modular_optuna_studies"
TIMESTAMP=$(date +"%Y%m%d-%H%M%S")
LOG_FILE_GCS_PATH="outputs/logs/modular_autoencoder_training_${TIMESTAMP}.log"

# Local workspace
WORKDIR="${HOME}/autoencoder_work_modular"
LOG_FILE="${WORKDIR}/run_pipeline.log"

# --- Main Execution ---
mkdir -p "${WORKDIR}"
cd "${WORKDIR}"

# Redirect all output to a log file AND the console
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "--- Modular Autoencoder Pipeline Started: $(date) ---"

# --- Environment Setup (CRITICAL BLOCK) ---
echo "--- Setting up Python environment... ---"
sudo apt-get update -y && sudo apt-get install -y python3-pip git

echo "--- Upgrading pip... ---"
python3 -m pip install --user --upgrade pip

# The VM's default PATH may not include the user's local bin. We add it
# manually to ensure the new pip and installed packages are found.
echo "--- Exporting new PATH to use upgraded pip... ---"
export PATH="/home/jupyter/.local/bin:${PATH}"
echo "--- Current PATH: ${PATH} ---"
echo "--- Pip version: $(pip --version) ---"

echo "--- Installing Python dependencies from requirements.txt... ---"
cat > requirements.txt << EOL
--extra-index-url https://download.pytorch.org/whl/cu121
torch==2.2.2
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.4.2
optuna==3.6.1
google-cloud-storage==2.16.0
EOL

python3 -m pip install --user --force-reinstall --no-cache-dir -r requirements.txt

echo "--- Verifying key installations... ---"
python3 -m pip show torch optuna google-cloud-storage

# --- Phase 1: Data Preprocessing (Modular) ---
echo "--- Starting Phase 1: Modular Data Preprocessing ---"
# Create the modular preprocessing script locally
cat > preprocess_data.py << 'EOL'
# preprocess_data.py
import argparse
import logging
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from google.cloud import storage

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

def upload_to_gcs(bucket_name: str, source_file_path: str, destination_blob_name: str):
    """Uploads a file to the bucket."""
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        logging.info(f"Uploading {source_file_path} to gs://{bucket_name}/{destination_blob_name}...")
        blob.upload_from_filename(source_file_path)
        logging.info("Upload complete.")
    except Exception as e:
        logging.error(f"Failed to upload {source_file_path} to GCS. Error: {e}")
        raise

def get_column_groups(columns: list[str]) -> dict[str, list[str]]:
    """
    [CORRECTED & REFINED] Parses column names and groups them by semantic concept.
    - Creates a DEDICATED group for EACH persona's justification embedding.
    - Explicitly IGNORES persona rating embeddings.
    - Groups property-wide, primary rooms, and other rooms as before.
    """
    final_groups = defaultdict(list)
    logging.info("Identifying semantic groups from column names...")

    for col in columns:
        # Step 1: We only care about columns that are part of an embedding vector.
        # This correctly and intentionally filters out the single-value '_Score' and '_rating' columns.
        if not re.search(r'_\d+$', col):
            continue

        # Step 2: Create a dedicated group for EACH persona's justification embedding.
        # The regex captures the full persona name as the group key.
        # e.g., captures "persona_Persona_YoungFamily_justification" from the full column name.
        match = re.match(r"^(persona_Persona_[^_]+_justification)", col)
        if match:
            group_name = match.group(1)
            final_groups[group_name].append(col)
            continue # Matched, so move to the next column.

        # Step 3: Handle the remaining, non-persona embedding groups as before.
        # Group for Property-Wide Aggregates
        if col.startswith("property_wide_"):
            final_groups["property_wide"].append(col)
            continue

        # Group for Canonical Room Features (e.g., primary_MainKitchen)
        match = re.match(r"^(primary_[^_]+|other_[^_]+)", col)
        if match:
            group_name = match.group(1)
            final_groups[group_name].append(col)
            continue

    # Sort columns within each group to ensure a consistent order
    for group_name, cols in final_groups.items():
        cols.sort(key=lambda c: int(c.split('_')[-1]))
        cols.sort(key=lambda c: '_'.join(c.split('_')[:-1]))

    logging.info(f"Created {len(final_groups)} groups: {sorted(list(final_groups.keys()))}")
    return final_groups

def main():
    """
    Main function to run the data preprocessing.
    - Loads a large CSV in chunks.
    - Groups columns by semantic concepts.
    - For each group, concatenates data, imputes NaNs, and saves as a .npy file.
    - Uploads all resulting .npy files to a GCS directory.
    """
    parser = argparse.ArgumentParser(description="Preprocess embedding CSV data into modular .npy files.")
    parser.add_argument("--input_csv_path", type=str, required=True, help="Full GCS path to the input CSV file.")
    parser.add_argument("--output_gcs_dir", type=str, required=True, help="GCS directory to save the output .npy files.")
    parser.add_argument("--gcs_bucket", type=str, required=True, help="GCS bucket name.")
    args = parser.parse_args()

    # --- Setup Paths ---
    local_temp_dir = Path("/tmp/data_preprocessing")
    local_temp_dir.mkdir(parents=True, exist_ok=True)
    local_output_dir = local_temp_dir / "output"
    local_output_dir.mkdir(parents=True, exist_ok=True)

    input_gcs_path = Path(args.input_csv_path)
    bucket_name = args.gcs_bucket
    gcs_csv_blob_name = "/".join(input_gcs_path.parts[2:])
    local_csv_path = local_temp_dir / input_gcs_path.name
    gcs_output_dir_blob_name = args.output_gcs_dir

    # --- Download Data ---
    logging.info(f"Downloading {args.input_csv_path} to {local_csv_path}...")
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(gcs_csv_blob_name)
    blob.download_to_filename(local_csv_path)
    logging.info("Download complete.")

    start_time = time.time()
    
    # --- Identify Column Groups ---
    df_header = pd.read_csv(local_csv_path, nrows=0)
    column_groups = get_column_groups(df_header.columns.tolist())

    # --- Process in Chunks to Manage Memory ---
    chunk_size = 250
    processed_chunks_by_group = {group_name: [] for group_name in column_groups}

    with pd.read_csv(local_csv_path, chunksize=chunk_size, header=0, low_memory=False) as reader:
        for i, chunk in enumerate(reader):
            logging.info(f"Processing chunk {i+1}...")
            
            # Pre-impute NaNs for the whole chunk
            chunk.fillna(0.0, inplace=True)
            
            for group_name, cols in column_groups.items():
                # Select the columns for the current group and convert to NumPy
                group_data = chunk[cols].values.astype(np.float32)
                processed_chunks_by_group[group_name].append(group_data)

    # --- Concatenate, Save, and Upload Each Group ---
    for group_name, chunks_list in processed_chunks_by_group.items():
        if not chunks_list:
            logging.warning(f"Skipping empty group: {group_name}")
            continue
            
        logging.info(f"Finalizing data for group: {group_name}")
        final_data = np.concatenate(chunks_list, axis=0)
        logging.info(f"  - Final shape: {final_data.shape}")

        local_npy_path = local_output_dir / f"{group_name}.npy"
        gcs_npy_blob_name = f"{gcs_output_dir_blob_name}/{group_name}.npy"
        
        logging.info(f"  - Saving to local file: {local_npy_path}")
        np.save(local_npy_path, final_data)

        upload_to_gcs(bucket_name, str(local_npy_path), gcs_npy_blob_name)

    end_time = time.time()
    logging.info(f"Modular preprocessing finished successfully in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
EOL

python3 preprocess_data.py \
    --input_csv_path="${INPUT_CSV_GCS_PATH}" \
    --output_gcs_dir="${PREPROCESSED_DATA_GCS_DIR}" \
    --gcs_bucket="${GCS_BUCKET}"

echo "--- Phase 1 Finished Successfully ---"

# --- Phase 2: Modular Model Training with Optuna ---
echo "--- Starting Phase 2: Modular Hyperparameter Search & Training ---"

# Create the modular training script locally
cat > train_autoencoder.py << 'EOL'
# train_autoencoder.py
import argparse
import logging
import time
from pathlib import Path

import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from google.cloud import storage
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# --- GCS Helper Functions ---
def download_from_gcs(bucket_name: str, source_blob_name: str, destination_file_path: str):
    """Downloads a file from the bucket."""
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        logging.info(f"Downloading gs://{bucket_name}/{source_blob_name} to {destination_file_path}...")
        Path(destination_file_path).parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(destination_file_path)
        logging.info("Download complete.")
    except Exception as e:
        logging.error(f"Failed to download from GCS. Error: {e}")
        raise

def upload_to_gcs(bucket_name: str, source_file_path: str, destination_blob_name: str):
    """Uploads a file to the bucket."""
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        logging.info(f"Uploading {source_file_path} to gs://{bucket_name}/{destination_blob_name}...")
        blob.upload_from_filename(source_file_path)
        logging.info("Upload complete.")
    except Exception as e:
        logging.error(f"Failed to upload to GCS. Error: {e}")
        raise

# --- Model Definition ---
class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, num_layers: int):
        super().__init__()
        encoder_layers = []
        current_dim = input_dim
        for i in range(num_layers):
            next_dim = latent_dim if i == num_layers - 1 else max(latent_dim, (current_dim + latent_dim) // 2)
            encoder_layers.append(nn.Linear(current_dim, next_dim))
            encoder_layers.append(nn.ReLU())
            current_dim = next_dim
        self.encoder = nn.Sequential(*encoder_layers)
        decoder_layers = []
        for i in range(num_layers):
            next_dim = input_dim if i == num_layers - 1 else max(current_dim, (current_dim + input_dim) // 2)
            decoder_layers.append(nn.Linear(current_dim, next_dim))
            if i < num_layers - 1:
                decoder_layers.append(nn.ReLU())
            current_dim = next_dim
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# --- Optuna Objective Function ---
def objective(trial, data, epochs, device, group_name):
    # Suggest hyperparameters, dynamically adjusting search space based on group type
    if 'persona' in group_name:
        latent_dim = trial.suggest_categorical("latent_dim", [16, 32, 48])
    elif 'property_wide' in group_name:
        latent_dim = trial.suggest_categorical("latent_dim", [8, 16, 24])
    else:  # Canonical rooms
        latent_dim = trial.suggest_categorical("latent_dim", [8, 12, 16])
    
    num_layers = trial.suggest_categorical("num_layers", [1, 2])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])

    # --- Data Setup ---
    # Input data is already the correct shape: (num_properties, features_for_this_group)
    X_train, X_val = train_test_split(data, test_size=0.2, random_state=42)
    train_dataset = TensorDataset(torch.from_numpy(X_train).float())
    val_dataset = TensorDataset(torch.from_numpy(X_val).float())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=2, pin_memory=True)

    # --- Model, Optimizer, Loss ---
    model = Autoencoder(input_dim=data.shape[1], latent_dim=latent_dim, num_layers=num_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # --- Training & Validation Loop ---
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            inputs = batch[0].to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
    
    # --- Final Validation ---
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch[0].to(device, non_blocking=True)
            outputs = model(inputs)
            val_loss = criterion(outputs, inputs)
            total_val_loss += val_loss.item()
    
    avg_val_loss = total_val_loss / len(val_loader)
    logging.info(f"Trial {trial.number} for '{group_name}': Avg Val Loss: {avg_val_loss:.6f} with params {trial.params}")
    return avg_val_loss

def main():
    parser = argparse.ArgumentParser(description="Train a modular autoencoder with Optuna for a specific data group.")
    parser.add_argument("--input_npy_gcs_path", type=str, required=True, help="Full GCS path to the specific preprocessed .npy file.")
    parser.add_argument("--gcs_bucket", type=str, required=True, help="GCS bucket name for storing artifacts.")
    parser.add_argument("--group_name", type=str, required=True, help="Name of the semantic group (e.g., 'primary_MainKitchen').")
    parser.add_argument("--model_output_dir", type=str, required=True, help="Base GCS directory for final trained model params.")
    parser.add_argument("--study_db_dir", type=str, required=True, help="Base GCS directory for Optuna study databases.")
    parser.add_argument("--n_trials", type=int, default=25, help="Number of Optuna trials to run.")
    parser.add_argument("--epochs_per_trial", type=int, default=20, help="Number of epochs to train each model during a trial.")
    args = parser.parse_args()

    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device} for group '{args.group_name}'")

    local_temp_dir = Path(f"/tmp/training_{args.group_name}")
    local_temp_dir.mkdir(parents=True, exist_ok=True)
    
    # --- Download Data ---
    gcs_npy_blob_name = "/".join(Path(args.input_npy_gcs_path).parts[2:])
    local_npy_path = local_temp_dir / Path(args.input_npy_gcs_path).name
    download_from_gcs(args.gcs_bucket, gcs_npy_blob_name, str(local_npy_path))
    data = np.load(local_npy_path)
    logging.info(f"Loaded data for group '{args.group_name}' with shape: {data.shape}")

    # --- Run Optuna Study ---
    local_study_path = str(local_temp_dir / "optuna_study.db")
    study = optuna.create_study(
        direction="minimize",
        storage=f"sqlite:///{local_study_path}",
        study_name=f"ae-hyperparam-{args.group_name}",
        load_if_exists=True
    )
    study.optimize(lambda trial: objective(trial, data, args.epochs_per_trial, device, args.group_name), n_trials=args.n_trials)

    # --- Save Best Model Parameters ---
    best_params = study.best_trial.params
    logging.info(f"Optimization finished for group '{args.group_name}'.")
    logging.info(f"Best validation loss: {study.best_value:.6f}")
    logging.info(f"Best hyperparameters: {best_params}")

    # --- Retrain the best model on the full dataset and save its weights ---
    logging.info("Retraining final model with best hyperparameters on the full dataset...")
    final_model = Autoencoder(
        input_dim=data.shape[1],
        latent_dim=best_params['latent_dim'],
        num_layers=best_params['num_layers']
    ).to(device)

    final_optimizer = optim.Adam(final_model.parameters(), lr=best_params['learning_rate'])
    criterion = nn.MSELoss()
    final_dataset = TensorDataset(torch.from_numpy(data).float())
    # Use the best batch size found by Optuna
    final_loader = DataLoader(final_dataset, batch_size=best_params['batch_size'], shuffle=True)

    # Train for a fixed number of epochs (or the same as in trials)
    final_model.train()
    for epoch in range(args.epochs_per_trial): # Reusing epochs_per_trial for final training
        for batch in final_loader:
            inputs = batch[0].to(device)
            final_optimizer.zero_grad()
            outputs = final_model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            final_optimizer.step()
        logging.info(f"Final training epoch {epoch+1}/{args.epochs_per_trial} completed.")

    # --- Save the Final Model's State Dictionary ---
    local_model_weights_path = local_temp_dir / "best_model_weights.pth"
    torch.save(final_model.state_dict(), local_model_weights_path)
    logging.info(f"Best model weights saved locally to {local_model_weights_path}")

    # --- Generate and Save Encodings ---
    logging.info("Generating encodings from the trained model...")
    final_model.eval()
    all_encodings = []

    with torch.no_grad():
        for batch in final_loader:
            inputs = batch[0].to(device)
            # Get only the encoder output (latent representation)
            encoded = final_model.encoder(inputs)
            all_encodings.append(encoded.cpu().numpy())

    # Concatenate all batch encodings
    final_encodings = np.concatenate(all_encodings, axis=0)
    logging.info(f"Generated encodings shape: {final_encodings.shape}")

    # Save encodings locally
    local_encodings_path = local_temp_dir / "encodings.npy"
    np.save(local_encodings_path, final_encodings)
    logging.info(f"Encodings saved locally to {local_encodings_path}")

    # Upload encodings to GCS
    encodings_blob_name = f"{args.model_output_dir}/{args.group_name}/encodings.npy"
    upload_to_gcs(args.gcs_bucket, str(local_encodings_path), encodings_blob_name)

    # --- Upload Artifacts (including the trained model) ---
    # Also save the best_params dictionary for reference
    local_model_params_path = local_temp_dir / "best_model_params.json"
    import json
    with open(local_model_params_path, 'w') as f:
        json.dump(best_params, f)

    model_weights_blob_name = f"{args.model_output_dir}/{args.group_name}/best_model_weights.pth"
    model_params_blob_name = f"{args.model_output_dir}/{args.group_name}/best_model_params.json"
    study_db_blob_name = f"{args.study_db_dir}/{args.group_name}/optuna_study.db"

    upload_to_gcs(args.gcs_bucket, str(local_model_weights_path), model_weights_blob_name)
    upload_to_gcs(args.gcs_bucket, str(local_model_params_path), model_params_blob_name)
    upload_to_gcs(args.gcs_bucket, local_study_path, study_db_blob_name)

if __name__ == "__main__":
    main()
EOL

# --- Training Loop for All Modules ---
PREPROCESSED_DATA_GCS_FULL_DIR="gs://${GCS_BUCKET}/${PREPROCESSED_DATA_GCS_DIR}"
NPY_FILES=$(gsutil ls "${PREPROCESSED_DATA_GCS_FULL_DIR}/*.npy")

if [ -z "$NPY_FILES" ]; then
    echo "ERROR: No .npy files found in ${PREPROCESSED_DATA_GCS_FULL_DIR}. Preprocessing may have failed."
    exit 1
fi

echo "--- Found $(echo "$NPY_FILES" | wc -w) data groups to train on. Starting loop. ---"

# Loop through each data file and train a dedicated autoencoder
gsutil ls "${PREPROCESSED_DATA_GCS_FULL_DIR}/*.npy" | while read -r GCS_FILE_PATH; do
    # Exit if GCS_FILE_PATH is empty (can happen if ls returns nothing)
    if [ -z "$GCS_FILE_PATH" ]; then
        continue
    fi
    
    # Extract the group name from the file path, e.g., "primary_MainKitchen"
    FILENAME=$(basename "${GCS_FILE_PATH}")
    GROUP_NAME="${FILENAME%.npy}"

    echo
    echo "=========================================================="
    echo "--- Training Autoencoder for Group: ${GROUP_NAME} ---"
    echo "=========================================================="
    
    python3 train_autoencoder.py \
        --input_npy_gcs_path="${GCS_FILE_PATH}" \
        --gcs_bucket="${GCS_BUCKET}" \
        --group_name="${GROUP_NAME}" \
        --model_output_dir="${MODULAR_MODEL_OUTPUT_GCS_DIR}" \
        --study_db_dir="${MODULAR_STUDY_DB_GCS_DIR}" \
        --n_trials=25 \
        --epochs_per_trial=20

    if [ $? -ne 0 ]; then
        echo "ERROR: Training failed for group ${GROUP_NAME}. Exiting pipeline."
        # Upload logs before exiting to help debug
        gsutil cp "${LOG_FILE}" "gs://${GCS_BUCKET}/${LOG_FILE_GCS_PATH}_FAILED"
        exit 1
    fi
    echo "--- Finished training for group: ${GROUP_NAME} ---"
done

echo "--- Phase 2 Finished Successfully for all groups ---"

# --- Finalization: Upload Logs ---
echo "--- Uploading execution log to GCS... ---"
gsutil cp "${LOG_FILE}" "gs://${GCS_BUCKET}/${LOG_FILE_GCS_PATH}"

echo "--- Modular Autoencoder Pipeline Finished: $(date) ---"