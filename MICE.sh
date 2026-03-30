#!/bin/bash

# ==============================================================================
# run_chained_imputation.sh - V1.0 - Definitive Pipeline
# DESCRIPTION:
# This script orchestrates the "Chained Deep Learning Imputation" pipeline.
# It is built upon the collective knowledge from previous project handoffs,
# prioritizing environment stability, data integrity, and modular execution.
#
# ARCHITECTURE:
# 1. Environment Setup: Creates an isolated Python virtual environment to prevent
#    any system-level package conflicts.
# 2. Python Script Generation: Writes a self-contained, powerful Python script
#    to the VM's local disk.
# 3. Chained Imputation Execution: The Python script executes the core logic:
#    a. It loads data chunks in a user-defined, logical order.
#    b. For each chunk, it trains a specialized deep learning model (DAE for
#       numerical data, VAE for abstract embeddings) using all previously
#       imputed data as context.
#    c. This process iteratively builds a complete, high-quality dataset.
# 4. Artifact Upload: The final, imputed dataset and execution logs are
#    uploaded to Google Cloud Storage.
# ==============================================================================

# --- Strict Mode & Error Handling ---
#!/bin/bash

# ==============================================================================
# run_chained_imputation.sh - V2.0 - User-Corrected & Hardened
# DESCRIPTION:
# This script incorporates critical user feedback to create a more efficient and
# accurate imputation pipeline.
#
# KEY REVISIONS BASED ON USER FEEDBACK:
# 1. ADDED "Contextual Features": The output from the geospatial pipeline is now
#    a core input, used to improve all subsequent imputations.
# 2. REMOVED Redundant Data: The 'gwr_features' file has been removed.
# 3. OPTIMIZED Large File Handling: The massive 4GB embedding file is now handled
#    with a fast, simple imputation, avoiding a huge computational bottleneck.
# 4. DEFERRED Bayesian Optimization: The focus is on robust, timely imputation,
#    not exhaustive hyperparameter tuning.
# ==============================================================================

# ==============================================================================
# run_chained_imputation.sh - V3.0 - PLACEHOLDER-AWARE & DEFINITIVE
# DESCRIPTION:
# This is the final, hardened version of the imputation pipeline, incorporating
# the critical user insight that missing values are encoded as placeholders
# (e.g., -1, 0) from previous processes.
#
# KEY REVISIONS (V3.0):
# 1. ADDED Placeholder-to-NaN Conversion: A new, crucial pre-processing step
#    has been added. After loading each data chunk, it intelligently converts
#    known placeholder values back to standard np.nan based on user-defined
#    column name patterns. This "un-does" the previous constant imputation,
#    allowing our superior deep learning models to work correctly.
# 2. RETAINED all logic from V2, including the optimized handling of large
#    files and the inclusion of geospatial context features.
# ==============================================================================

# ==============================================================================
# run_chained_imputation.sh - V5.0 - GLOBAL PLACEHOLDER REPLACEMENT
# DESCRIPTION:
# This script incorporates the user's final, critical correction regarding
# placeholder values. It removes the flawed pattern-matching logic and implements
# a direct, global replacement of all known sentinel values (-1, 0, etc.)
# across all features before imputation.
#
# KEY REVISIONS (V5.0):
# 1. REMOVED Flawed Pattern Matching: The complex `PLACEHOLDER_CONFIG` dictionary
#    and its associated logic have been completely removed.
# 2. IMPLEMENTED Global Placeholder Replacement: A new, simpler function now
#    replaces a list of known placeholder values with `np.nan` across ALL
#    columns of each data chunk, as per the user's directive. This ensures all
#    missing data is correctly identified before the imputation models run.
# ==============================================================================

# ==============================================================================
# run_chained_imputation.sh - V7.0 - DEFENSIVE DOWNLOADING
# DESCRIPTION:
# This script addresses a `pyarrow.lib.ArrowInvalid` error caused by attempting
# to read a 0-byte Parquet file. This indicates the download from GCS failed
# silently, creating an empty local file.
#
# KEY REVISIONS (V7.0):
# 1. HARDENED Download Function: The `download_gcs_file` Python function has
#    been rewritten to be more defensive. It now explicitly verifies that the
#    source blob exists in GCS before starting the download and verifies that
#    the downloaded file is not empty afterward. This will prevent the pipeline
#    from proceeding with corrupted inputs and provide a much clearer error
#    if a download fails.
# ==============================================================================

# --- Strict Mode & Error Handling ---
set -e
set -o pipefail
set -x

# --- Configuration ---
GCS_BUCKET="srgan-bucket-ace-botany-453819-t4"
TIMESTAMP=$(date +"%Y%m%d-%H%M%S")

# === ACTION REQUIRED: Verify these input paths ===
INPUT_DATA_GCS_DIR="house data scrape"
# --- FIX: This now points to the full GCS object path, not just the directory ---
GEOSPATIAL_FEATURES_GCS_PATH="/contextual_features/contextual_features.parquet"
MASTER_DATA_GCS_PATH="features/final_master_dataset/master_feature_set.csv" # gs://srgan-bucket-ace-botany-453819-t4/features/final_master_dataset/master_feature_set.csv
# --- NEW: Path to the processed embeddings from the autoencoder pipeline ---
EMBEDDINGS_GCS_DIR="models/modular_autoencoders"

# --- Intermediate & Output Paths ---
HYBRID_PIPELINE_DIR="imputation_pipeline/hybrid_${TIMESTAMP}"
STAGE1_DAE_OUTPUT_GCS_PATH="${HYBRID_PIPELINE_DIR}/stage1_imputed_embeddings.parquet"
STAGE2_MERGED_INPUT_GCS_PATH="${HYBRID_PIPELINE_DIR}/stage2_merged_for_mice.parquet"
FINAL_OUTPUT_GCS_PATH="${HYBRID_PIPELINE_DIR}/final_fully_imputed_dataset.parquet"

# --- Parameters ---
DAE_EPOCHS=150
DAE_BATCH_SIZE=256
DAE_PATIENCE=20
DAE_MIN_DELTA=0.00001
MICE_ITERATIONS=5
LGBM_N_ESTIMATORS=50
LGBM_NUM_LEAVES=20
LGBM_MAX_FEATURES=500

# --- Additional Variables ---
WORKDIR="/tmp/imputation_work_${TIMESTAMP}"
STAGING_DIR="${WORKDIR}/staging"
LOG_FILE="${WORKDIR}/pipeline.log"
LOG_FILE_GCS_PATH="logs/imputation_pipeline_${TIMESTAMP}.log"
OUTPUT_GCS_DIR="imputation_pipeline/hybrid_${TIMESTAMP}"
IMPUTATION_EPOCHS=${DAE_EPOCHS}
IMPUTATION_BATCH_SIZE=${DAE_BATCH_SIZE}
EARLY_STOPPING_PATIENCE=${DAE_PATIENCE}
EARLY_STOPPING_MIN_DELTA=${DAE_MIN_DELTA}
OPTUNA_TRIALS=10

# --- Pipeline Execution ---
echo "--- Cleaning up previous workspace at ${WORKDIR} ---"
rm -rf "${WORKDIR}"
mkdir -p "${WORKDIR}"
cd "${WORKDIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "--- Chained Deep Learning Imputation Pipeline V2 Started: $(date) ---"

echo "--- Starting Data Staging Phase ---"
echo "--- Staging directory for this run will be: ${STAGING_DIR} ---"

echo "--- Data Staging Complete. All files copied. ---"

# --- Environment Setup ---
echo "--- Setting up Python virtual environment... ---"
VENV_PATH="${WORKDIR}/imputation_env"
python3 -m venv "${VENV_PATH}"
source "${VENV_PATH}/bin/activate"
pip install --upgrade pip

echo "--- Installing Python dependencies... ---"
cat > requirements.txt << EOL
--extra-index-url https://download.pytorch.org/whl/cu121
torch==2.2.2
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.4.2
google-cloud-storage==2.16.0
pyarrow==16.1.0
tqdm==4.66.4
optuna==3.6.1
EOL
pip install --force-reinstall --no-cache-dir -r requirements.txt
echo "--- GPU Check... ---"
nvidia-smi

# Fix PyTorch installation issues
echo "--- Fixing PyTorch installation... ---"
source "${VENV_PATH}/bin/activate"
pip uninstall -y torch
pip install torch==2.2.2+cu121 --no-cache-dir -f https://download.pytorch.org/whl/cu121/torch_stable.html

# Verify torch installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"


# ==============================================================================
# --- PYTHON SCRIPT GENERATION (V2) ---
# ==============================================================================
echo "--- Generating Python Script: impute_chained_deep_learning_v2.py ---"
cat > impute_chained_deep_learning_v2.py << 'EOL'
import argparse, logging, pandas as pd, numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
from google.cloud import storage
import time, gc
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error
import copy
from sklearn.model_selection import train_test_split
import optuna

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler()])

PLACEHOLDER_VALUES = [-1.0, -1, 0.0, 0]
CHUNK_DEFINITIONS = [
    { "name": "original_master", "filename": "master_feature_set.csv", "model_type": "DAE", "key_col": "property_id" },
    { "name": "core_property", "filename": "property_features_quantitative_v4.csv", "model_type": "DAE", "key_col": "property_id" },
    { "name": "geospatial_context", "filename": "contextual_features.parquet", "model_type": "CLEAN", "key_col": "property_id" },
    { "name": "global_context_1", "filename": "property_data_subset1.csv", "model_type": "DAE", "key_col": "property_address" },
    { "name": "global_context_2", "filename": "property_data_subset2.csv", "model_type": "DAE", "key_col": "property_address" },
    { "name": "global_context_3", "filename": "property_data_subset3.csv", "model_type": "DAE", "key_col": "property_address" },
    { "name": "global_context_4", "filename": "property_data_subset4.csv", "model_type": "DAE", "key_col": "property_address" },
    { "name": "global_context_5", "filename": "property_data_subset5.csv", "model_type": "DAE", "key_col": "property_address" },
]

def download_gcs_directory(bucket, source_directory, destination_dir):
    """Downloads all files from a GCS directory."""
    logging.info(f"Downloading directory gs://{bucket.name}/{source_directory} to {destination_dir}...")
    storage_client = storage.Client()
    blobs = storage_client.list_blobs(bucket.name, prefix=source_directory)
    for blob in blobs:
        if not blob.name.endswith('/'):
            destination_file_path = destination_dir / Path(blob.name).relative_to(source_directory)
            destination_file_path.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(str(destination_file_path))
    logging.info("Directory download complete.")

def download_gcs_file(bucket, source_blob_name, dest_path):
    """Defensively downloads a file from GCS, verifying existence and non-zero size."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    blob = bucket.blob(source_blob_name)

    logging.info(f"Verifying existence of gs://{bucket.name}/{source_blob_name}...")
    if not blob.exists():
        logging.error(f"FATAL: GCS object not found at gs://{bucket.name}/{source_blob_name}")
        raise FileNotFoundError(f"Could not find GCS object: {source_blob_name}")

    logging.info(f"Object exists. Downloading to {dest_path}...")
    blob.download_to_filename(dest_path)

    # CRITICAL CHECK: Verify that the downloaded file is not empty.
    if dest_path.stat().st_size == 0:
        logging.error(f"FATAL: Download of {source_blob_name} resulted in a 0-byte file.")
        raise IOError(f"Download failed for {source_blob_name}, created empty file.")

    logging.info(f"Download successful for {dest_path} ({dest_path.stat().st_size} bytes).")

def load_embeddings_as_df(local_data_dir: Path) -> pd.DataFrame:
    """Loads all encodings.npy files from subdirectories into a single DataFrame."""
    all_embeddings_dfs = []
    group_dirs = sorted([d for d in local_data_dir.iterdir() if d.is_dir()]) # Sort for consistent order
    logging.info(f"Found {len(group_dirs)} embedding group directories.")

    for group_dir in group_dirs:
        encodings_file = group_dir / "encodings.npy"
        if encodings_file.exists():
            logging.info(f"Loading embeddings from: {encodings_file}")
            group_name = group_dir.name
            embeddings_array = np.load(encodings_file)
            
            # Create column names like 'embedding_property_wide_0', 'embedding_property_wide_1', ...
            column_names = [f"emb_{group_name}_{i}" for i in range(embeddings_array.shape[1])]
            
            group_df = pd.DataFrame(embeddings_array, columns=column_names)
            all_embeddings_dfs.append(group_df)
        else:
            logging.warning(f"No 'encodings.npy' file found in {group_dir}. Skipping.")
            
    if not all_embeddings_dfs:
        raise ValueError("No embedding data was loaded. Check the embeddings directory and files.")
        
    # Concatenate all dataframes horizontally
    embeddings_df = pd.concat(all_embeddings_dfs, axis=1)
    logging.info(f"Successfully loaded and combined all embeddings. Final shape: {embeddings_df.shape}")
    return embeddings_df
    
def upload_gcs_file(bucket, source_path, dest_blob_name):
    blob = bucket.blob(dest_blob_name)
    logging.info(f"Uploading {source_path} to gs://{bucket.name}/{dest_blob_name}...")
    blob.upload_from_filename(source_path)

class DAE(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, dropout_rate):
        super(DAE, self).__init__()
        
        encoder_layers = []
        # Dynamically build encoder layers
        current_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(current_dim, h_dim))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.Dropout(dropout_rate))
            current_dim = h_dim
        self.encoder = nn.Sequential(*encoder_layers)
        
        decoder_layers = []
        # Dynamically build decoder layers (in reverse)
        for h_dim in reversed(hidden_dims[:-1]):
            decoder_layers.append(nn.Linear(current_dim, h_dim))
            decoder_layers.append(nn.ReLU())
            current_dim = h_dim
        decoder_layers.append(nn.Linear(current_dim, output_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        return self.decoder(self.encoder(x))


from sklearn.model_selection import train_test_split

# --- FIX: Helper function for safe scaling ---
def safe_transform(df_partial, scaler):
    scaffold = pd.DataFrame(0.0, index=df_partial.index, columns=scaler.feature_names_in_)
    scaffold[df_partial.columns] = df_partial
    transformed_scaffold = scaler.transform(scaffold)
    return pd.DataFrame(transformed_scaffold, index=df_partial.index, columns=scaler.feature_names_in_)[df_partial.columns]

# Add the missing safe_inverse_transform function
def safe_inverse_transform(array, columns, scaler):
    scaffold = pd.DataFrame(0.0, index=range(array.shape[0]), columns=scaler.feature_names_in_)
    scaffold[columns] = array
    inverse_transformed = scaler.inverse_transform(scaffold)
    return pd.DataFrame(inverse_transformed, columns=scaler.feature_names_in_)[columns].values

def run_diagnostic_accuracy_test(full_df, context_cols, target_cols, global_scaler, device, epochs, batch_size, patience, min_delta, n_trials):
    """
    Runs a controlled experiment with Bayesian Optimization to find the best DAE hyperparameters.
    """
    logging.info(f"\n{'='*80}\n--- Running Diagnostic DAE Accuracy Test with Bayesian Optimization ---\n{'='*80}")
    
    complete_subset_df = full_df[context_cols + target_cols].dropna()
    logging.info(f"Found {len(complete_subset_df)} rows with no missing values for the test.")

    if len(complete_subset_df) < 200: # Increased threshold for reliable optimization
        logging.warning("Insufficient complete data (< 200 rows) to run reliable optimization. Skipping."); return None

    # 1. Split data: Test set is held out. Train is split again for validation inside objective.
    X = complete_subset_df[context_cols]
    Y = complete_subset_df[target_cols]
    X_train_full, X_test, Y_train_full, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    logging.info(f"Split data into {len(X_train_full)} training rows and {len(X_test)} testing rows.")

    # 2. Scale all data subsets using the global scaler
    X_train_full_scaled = safe_transform(X_train_full, global_scaler)
    Y_train_full_scaled = safe_transform(Y_train_full, global_scaler)
    X_test_scaled = safe_transform(X_test, global_scaler)
    Y_test_scaled = safe_transform(Y_test, global_scaler)
    
    input_dim, output_dim = len(context_cols), len(target_cols)

    # 3. Define the objective function for Optuna
    def objective(trial):
        # Suggest hyperparameters
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        dropout_rate = trial.suggest_float("dropout", 0.1, 0.5)
        n_layers = trial.suggest_int("n_layers", 2, 4)
        hidden_dims = []
        last_dim = 512
        for i in range(n_layers):
            # Suggest dimension for this layer, making it progressively smaller
            h_dim = trial.suggest_int(f"h_dim_{i}", 64, last_dim, log=True)
            hidden_dims.append(h_dim)
            last_dim = h_dim

        # Split training data for this trial
        X_train, X_val, Y_train, Y_val = train_test_split(X_train_full_scaled, Y_train_full_scaled, test_size=0.25, random_state=42)
        
        train_dataset = TensorDataset(torch.from_numpy(X_train.values).float(), torch.from_numpy(Y_train.values).float())
        val_dataset = TensorDataset(torch.from_numpy(X_val.values).float(), torch.from_numpy(Y_val.values).float())
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        model = DAE(input_dim, output_dim, hidden_dims, dropout_rate).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_function = nn.MSELoss()

        # Training loop with early stopping
        best_val_loss, patience_counter = float('inf'), 0
        for epoch in range(epochs):
            model.train()
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                y_pred = model(x_batch)
                loss = loss_function(y_pred, y_batch)
                loss.backward()
                optimizer.step()
            
            model.eval()
            current_val_loss = 0.0
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                    y_pred = model(x_batch)
                    loss = loss_function(y_pred, y_batch)
                    current_val_loss += loss.item()
            avg_val_loss = current_val_loss / len(val_loader)
            
            if avg_val_loss < best_val_loss - min_delta:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= patience: break
        
        return best_val_loss

    # 4. Run the Optuna study
    logging.info(f"Starting Optuna study with {n_trials} trials...")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    best_params = study.best_params
    logging.info(f"Optuna study complete. Best params: {best_params}")

    # 5. Train the final model on the full training data with the best hyperparameters
    logging.info("Training final diagnostic model with best hyperparameters...")
    best_hidden_dims = [best_params[f"h_dim_{i}"] for i in range(best_params["n_layers"])]
    final_model = DAE(input_dim, output_dim, best_hidden_dims, best_params["dropout"]).to(device)
    
    full_train_dataset = TensorDataset(torch.from_numpy(X_train_full_scaled.values).float(), torch.from_numpy(Y_train_full_scaled.values).float())
    final_model = train_predictive_dae(final_model, full_train_dataset, epochs, device, batch_size, patience, min_delta)

    # 6. Evaluate the final model on the unseen test set
    final_model.eval()
    with torch.no_grad():
        X_test_tensor = torch.from_numpy(X_test_scaled.values).float().to(device)
        Y_pred_scaled = final_model(X_test_tensor)
        
        scaled_rmse = np.sqrt(mean_squared_error(Y_test_scaled.values, Y_pred_scaled.cpu().numpy()))
        scaled_mae = mean_absolute_error(Y_test_scaled.values, Y_pred_scaled.cpu().numpy())
        
        Y_pred_unscaled = safe_inverse_transform(Y_pred_scaled.cpu().numpy(), target_cols, global_scaler)
        
    rmse = np.sqrt(mean_squared_error(Y_test.values, Y_pred_unscaled))
    mae = mean_absolute_error(Y_test.values, Y_pred_unscaled)

    logging.info(f"--- Final Diagnostic Test Results (Post-Optimization) ---")
    logging.info(f"Scaled Metrics (normalized space) | RMSE: {scaled_rmse:.4f} | MAE: {scaled_mae:.4f}")
    logging.info(f"Unscaled Metrics (original space) | RMSE: {rmse:.4f} | MAE: {mae:.4f}")
    logging.info(f"\n--- Finished Diagnostic Test ---\n{'='*80}\n")
    
    return best_params

def replace_placeholders(df, values_to_replace):
    logging.info(f"Globally replacing {values_to_replace} with NaN...")
    original_nan_count = df.isna().sum().sum()
    df.replace(values_to_replace, np.nan, inplace=True)
    new_nan_count = df.isna().sum().sum()
    logging.info(f"Created {new_nan_count - original_nan_count} new NaNs from placeholders.")
    # Drop any columns that are now 100% empty after replacement
    df.dropna(axis=1, how='all', inplace=True)
    logging.info(f"Dataframe shape after empty column removal: {df.shape}")
    return df

def train_dae_model(model, full_train_dataset, epochs, device, batch_size, patience, min_delta, lr=1e-4, noise_level=0.1):
    """
    Trains a DAE model with early stopping, clear logging, and configurable learning rate/noise.
    """
    train_subset, val_subset = train_test_split(full_train_dataset, test_size=0.2, random_state=42)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size * 2, num_workers=2, pin_memory=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_function = nn.MSELoss()
    
    best_val_loss, patience_counter = float('inf'), 0
    best_model_state = None

    pbar = tqdm(range(epochs), desc=f"Training DAE")
    for epoch in pbar:
        model.train()
        for data_batch, in train_loader:
            inputs = data_batch.to(device)
            optimizer.zero_grad()
            noise = torch.randn_like(inputs) * noise_level
            outputs = model(inputs + noise)
            loss = loss_function(outputs, inputs)
            loss.backward()
            optimizer.step()
            
        model.eval()
        current_val_loss = 0.0
        with torch.no_grad():
            for data_batch, in val_loader:
                inputs = data_batch.to(device)
                outputs = model(inputs)
                loss = loss_function(outputs, inputs)
                current_val_loss += loss.item()
        
        avg_val_loss = current_val_loss / len(val_loader)
        pbar.set_postfix({"Val Loss": f"{avg_val_loss:.6f}"})
        
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logging.info(f"Early stopping at epoch {epoch+1}. Best validation loss: {best_val_loss:.6f}")
            break
    
    if best_model_state:
        logging.info("Restoring best model weights from training.")
        model.load_state_dict(best_model_state)
    else:
        logging.warning("Training did not improve. Using final model weights.")
        
    return model

def train_predictive_dae(model, train_dataset, epochs, device, batch_size, patience, min_delta):
    """
    Trains a DAE model for predictive tasks (X→Y mapping) with early stopping.
    
    This differs from the regular train_dae_model because it handles separate input/output pairs
    rather than reconstructing the same data.
    """
    # Create a training and validation split
    train_subset, val_subset = train_test_split(train_dataset, test_size=0.2, random_state=42)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_function = nn.MSELoss()
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    logging.info(f"Starting DAE training for up to {epochs} epochs with patience of {patience}.")
    
    for epoch in tqdm(range(epochs), desc="Training DAE"):
        # --- Training Phase ---
        model.train()
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = loss_function(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            
        # --- Validation Phase ---
        model.eval()
        current_val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                y_pred = model(x_batch)
                loss = loss_function(y_pred, y_batch)
                current_val_loss += loss.item()
        
        avg_val_loss = current_val_loss / len(val_loader)
        
        # --- Early Stopping Logic ---
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
            logging.info(f"Epoch {epoch+1}: New best validation loss: {best_val_loss:.6f}")
        else:
            patience_counter += 1
            logging.info(f"Epoch {epoch+1}: No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            logging.info(f"Early stopping triggered at epoch {epoch+1}. Best validation loss: {best_val_loss:.6f}")
            break
    
    # Restore the best model weights
    if best_model_state:
        model.load_state_dict(best_model_state)
        
    return model

def impute_with_dae(model, data_to_impute_df, scaler, device, batch_size):
    """Imputes missing values in a dataframe using a DAE model and a global scaler."""
    model.eval()
    
    imputed_df = data_to_impute_df.copy()
    missing_mask = imputed_df.isnull()
    
    # --- FIX: Use scaffolding for safe scaling ---
    scaffold = pd.DataFrame(0.0, index=imputed_df.index, columns=scaler.feature_names_in_)
    scaffold[imputed_df.columns] = imputed_df.fillna(0.0) # Fill NaNs before copying
    scaled_data = scaler.transform(scaffold)
    
    dataset = TensorDataset(torch.from_numpy(scaled_data).float())
    loader = DataLoader(dataset, batch_size=batch_size)
    
    reconstructed_full = []
    with torch.no_grad():
        for batch_data, in loader:
            inputs = batch_data.to(device)
            reconstructed = model(inputs)
            reconstructed_full.append(reconstructed.cpu().numpy())
            
    reconstructed_np_scaled = np.vstack(reconstructed_full)
    
    # --- FIX: Safe inverse transform ---
    reconstructed_np_unscaled = scaler.inverse_transform(reconstructed_np_scaled)
    
    unscaled_df = pd.DataFrame(reconstructed_np_unscaled, index=imputed_df.index, columns=scaler.feature_names_in_)
    final_imputed_df = data_to_impute_df.copy()
    
    # Only fill the originally missing values from the relevant columns
    final_imputed_df[missing_mask] = unscaled_df[data_to_impute_df.columns][missing_mask]
    
    return final_imputed_df  # Return the DataFrame, not the values

def make_unique_columns(df_columns):
    """
    Generates a list of unique column names by appending '_<count>' to duplicates.
    This is a robust method that handles multiple rounds of duplication.
    """
    seen = {}
    new_columns = []
    for col in df_columns:
        if col not in seen:
            seen[col] = 1
            new_columns.append(col)
        else:
            count = seen[col]
            new_name = f"{col}_{count}"
            # Handle cases where the new name itself might already exist
            while new_name in seen:
                count += 1
                new_name = f"{col}_{count}"
            new_columns.append(new_name)
            seen[col] = count + 1
    return new_columns

class VAE(nn.Module):
    def __init__(self, input_dim):
        super(VAE, self).__init__()
        self.encoder_fc1 = nn.Linear(input_dim, 512)
        self.encoder_fc2_mu = nn.Linear(512, 256)
        self.encoder_fc2_logvar = nn.Linear(512, 256)
        self.decoder = nn.Sequential(nn.Linear(256, 512), nn.ReLU(), nn.Linear(512, input_dim))
    def encode(self, x):
        h1 = nn.ReLU()(self.encoder_fc1(x))
        return self.encoder_fc2_mu(h1), self.encoder_fc2_logvar(h1)
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

def evaluate_imputation_accuracy(original_df, imputed_df, mask):
    """
    Calculates RMSE and MAE for imputed values against original values.
    
    Args:
        original_df (pd.DataFrame): The dataframe before values were removed.
        imputed_df (pd.DataFrame): The dataframe after imputation.
        mask (np.ndarray): A boolean numpy array where True indicates a value was artificially removed.
    """
    # Use the boolean mask directly to index the DataFrame's values.
    original_values = original_df.values[mask]
    imputed_values = imputed_df.values[mask]

    # Crucially, check for and handle any NaNs that may have slipped through
    # before calculating metrics.
    valid_indices = ~np.isnan(original_values) & ~np.isnan(imputed_values)
    if not np.any(valid_indices):
        logging.warning("No valid (non-NaN) values found to compare for accuracy. Skipping.")
        return np.nan, np.nan

    original_values = original_values[valid_indices]
    imputed_values = imputed_values[valid_indices]
    
    rmse = np.sqrt(mean_squared_error(original_values, imputed_values))
    mae = mean_absolute_error(original_values, imputed_values)
    
    logging.info(f"Imputation Accuracy | RMSE: {rmse:.4f} | MAE: {mae:.4f}")
    return rmse, mae

# ==============================================================================
# Core Training and Imputation Logic
# ==============================================================================
def train_model(model, model_type, train_loader, epochs, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) # Lower learning rate for stability
    if model_type == "VAE":
        def loss_function(recon_x, x, mu, logvar):
            MSE = nn.functional.mse_loss(recon_x, x, reduction='sum')
            KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            return MSE + KLD
    else: # DAE
        loss_function = nn.MSELoss()

    model.train()
    for epoch in tqdm(range(epochs), desc=f"Training {model_type}"):
        for data_batch, in train_loader:
            inputs = data_batch.to(device)
            optimizer.zero_grad()
            if model_type == "DAE":
                noise = torch.randn_like(inputs) * 0.1
                outputs = model(inputs + noise)
                loss = loss_function(outputs, inputs)
            elif model_type == "VAE":
                outputs, mu, logvar = model(inputs)
                loss = loss_function(outputs, inputs, mu, logvar)
            loss.backward()
            optimizer.step()
    return model

def impute_chunk(model, model_type, data_to_impute, scaler, device, batch_size):
    model.eval()
    imputed_data = data_to_impute.copy()
    missing_mask = np.isnan(imputed_data)
    if not np.any(missing_mask):
        logging.info("No missing values to impute in this chunk.")
        return imputed_data

    # Use a simple but effective initial fill
    imputed_data[missing_mask] = 0.0
    scaled_data = scaler.transform(imputed_data)
    dataset = TensorDataset(torch.from_numpy(scaled_data).float())
    loader = DataLoader(dataset, batch_size=batch_size)
    
    reconstructed_full = []
    with torch.no_grad():
        for batch_data, in loader:
            inputs = batch_data.to(device)
            if model_type == "VAE":
                reconstructed, _, _ = model(inputs)
            else: # DAE
                reconstructed = model(inputs)
            reconstructed_full.append(reconstructed.cpu().numpy())
    
    reconstructed_np_scaled = np.vstack(reconstructed_full)
    reconstructed_np_unscaled = scaler.inverse_transform(reconstructed_np_scaled)
    
    # Only fill in the values that were originally missing
    final_imputed_data = data_to_impute.copy()
    final_imputed_data[missing_mask] = reconstructed_np_unscaled[missing_mask]
        
    return final_imputed_data

# ==============================================================================
# Main Orchestration Function
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Run DAE Imputation for Embeddings ONLY.")
    parser.add_argument("--gcs_bucket", required=True)
    parser.add_argument("--embeddings_gcs_dir", required=True)
    parser.add_argument("--output_gcs_path", required=True) # Changed to a single output file path
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--patience", type=int, required=True)
    parser.add_argument("--min_delta", type=float, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local_work_dir = Path("/tmp/embedding_imputation_data")
    local_embeddings_dir = local_work_dir / "embeddings"
    local_output_path = local_work_dir / "imputed_embeddings.parquet"
    local_embeddings_dir.mkdir(parents=True, exist_ok=True)
    storage_client = storage.Client()
    bucket = storage_client.bucket(args.gcs_bucket)

    # === STAGE 1: Load Embeddings ===
    logging.info(f"\n--- STAGE 1: Loading and Processing Embeddings ---\n")
    download_gcs_directory(bucket, args.embeddings_gcs_dir, local_embeddings_dir)
    embeddings_df = load_embeddings_as_df(local_embeddings_dir)
    diagnostic_df = embeddings_df # Use a consistent name for the rest of the logic

    if diagnostic_df.isnull().sum().sum() == 0:
        logging.info("No missing values found in embeddings. Saving original file.")
        diagnostic_df.to_parquet(local_output_path, index=False)
        upload_gcs_file(bucket, str(local_output_path), args.output_gcs_path)
        logging.info("--- Pipeline Finished Successfully ---")
        return

    # === STAGE 2: Global DAE Training on Embeddings ===
    logging.info(f"\n--- STAGE 2: Global DAE Training for Embeddings ---\n")
    global_scaler = MinMaxScaler().fit(diagnostic_df.fillna(0))
    
    complete_rows_mask = diagnostic_df.notna().all(axis=1)
    train_df = diagnostic_df[complete_rows_mask]
    
    if len(train_df) < 100:
        raise ValueError(f"Insufficient complete embedding vectors ({len(train_df)}) to train a reliable DAE.")
    
    logging.info(f"Found {len(train_df)} fully complete embedding vectors for training.")
    train_data_scaled = global_scaler.transform(train_df)
    train_dataset = TensorDataset(torch.from_numpy(train_data_scaled).float())
    
    input_dim = diagnostic_df.shape[1]
    global_dae = DAE(input_dim=input_dim, output_dim=input_dim, hidden_dims=[512, 256, 128], dropout_rate=0.2).to(device)
    
    global_dae = train_dae_model(global_dae, train_dataset, args.epochs, device, args.batch_size, args.patience, args.min_delta)
    
    # === STAGE 3: Global Imputation of Embeddings ===
    logging.info(f"\n--- STAGE 3: Performing Global Imputation on Embeddings ---\n")
    imputed_embeddings = impute_with_dae(global_dae, diagnostic_df, global_scaler, device, args.batch_size)

    # Save and upload
    imputed_embeddings.to_parquet(local_output_path, index=False)
    upload_gcs_file(bucket, str(local_output_path), args.output_gcs_path)
    logging.info("--- Embedding Imputation Finished Successfully ---")

if __name__ == "__main__":
    main()
EOL

# --- Run the Main Python Script ---
echo "--- Starting the Chained Imputation Orchestrator Script V2 ---"

python3 impute_chained_deep_learning_v2.py \
    --gcs_bucket="${GCS_BUCKET}" \
    --input_gcs_dir="${INPUT_DATA_GCS_DIR}" \
    --geospatial_gcs_path="${GEOSPATIAL_FEATURES_GCS_PATH}" \
    --original_master_gcs_path="${MASTER_DATA_GCS_PATH}" \
    --embeddings_gcs_dir="${EMBEDDINGS_GCS_DIR}" \
    --output_gcs_dir="${OUTPUT_GCS_DIR}" \
    --epochs=${IMPUTATION_EPOCHS} \
    --batch_size=${IMPUTATION_BATCH_SIZE} \
    --patience=${EARLY_STOPPING_PATIENCE} \
    --min_delta=${EARLY_STOPPING_MIN_DELTA} \
    --optuna_trials=${OPTUNA_TRIALS}

if [ $? -ne 0 ]; then
    echo "ERROR: Chained imputation script failed. Exiting pipeline."
    gsutil -m cp "${LOG_FILE}" "gs://${GCS_BUCKET}/${LOG_FILE_GCS_PATH}_FAILED"
    exit 1
fi


# ==============================================================================
# --- STAGE 2: Merge and MICE Imputation for Tabular Data ---
# ==============================================================================
echo "--- STAGE 2: Preparing data for MICE Imputation ---"

# This small python script handles the merge logic cleanly.
cat > merge_for_mice.py << 'EOL'
import pandas as pd
from google.cloud import storage
import argparse
from pathlib import Path

def download_gcs(bucket_name, gcs_path, local_path):
    storage.Client().bucket(bucket_name).blob(gcs_path).download_to_filename(local_path)

def upload_gcs(bucket_name, local_path, gcs_path):
    storage.Client().bucket(bucket_name).blob(gcs_path).upload_from_filename(local_path)

parser = argparse.ArgumentParser()
parser.add_argument("--bucket")
parser.add_argument("--master_path")
parser.add_argument("--embeddings_path")
parser.add_argument("--output_path")
args = parser.parse_args()

local_dir = Path("/tmp/merge_work")
local_dir.mkdir(exist_ok=True)
local_master = local_dir / "master.csv"
local_embeddings = local_dir / "embeddings.parquet"
local_output = local_dir / "merged.parquet"

print("Downloading data...")
download_gcs(args.bucket, args.master_path, local_master)
download_gcs(args.bucket, args.embeddings_path, local_embeddings)

df_master = pd.read_csv(local_master, low_memory=False)
df_embeddings = pd.read_parquet(local_embeddings)

if len(df_master) != len(df_embeddings):
    raise ValueError("Row count mismatch between master data and embeddings!")

# Combine the master dataset (without its own embeddings) and the newly imputed embeddings
emb_cols_in_master = [c for c in df_master.columns if c.startswith('emb_')]
df_master.drop(columns=emb_cols_in_master, inplace=True, errors='ignore')
df_merged = pd.concat([df_master, df_embeddings], axis=1)

print(f"Merged data shape: {df_merged.shape}")
df_merged.to_parquet(local_output, index=False)
upload_gcs(args.bucket, str(local_output), args.output_path)
print("Merge complete and uploaded.")
EOL

python3 merge_for_mice.py \
    --bucket="${GCS_BUCKET}" \
    --master_path="${MASTER_DATA_GCS_PATH}" \
    --embeddings_path="${STAGE1_DAE_OUTPUT_GCS_PATH}" \
    --output_path="${STAGE2_MERGED_INPUT_GCS_PATH}"

echo "--- Data merged. Starting MICE imputation. ---"


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
    parser.add_argument("--geospatial_gcs_path", required=True)
    parser.add_argument("--output_gcs_path", required=True)
    parser.add_argument("--iterations", type=int, required=True)
    # --- NEW: Add arguments for performance tuning ---
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--num_leaves", type=int, default=31)
    parser.add_argument("--max_features", type=int, default=-1)
    args = parser.parse_args()

    local_work_dir = Path("/tmp/mice_lgbm_data")
    local_input_path = local_work_dir / "input_data.parquet"
    local_geospatial_path = local_work_dir / "geospatial_data.parquet"
    local_output_path = local_work_dir / "final_imputed_data.parquet"

    # Download the datasets
    download_gcs_file(args.gcs_bucket, args.input_gcs_path, local_input_path)
    download_gcs_file(args.gcs_bucket, args.geospatial_gcs_path, local_geospatial_path)

    # Load and merge data
    df = pd.read_parquet(local_input_path)
    logging.info(f"Successfully loaded input data. Shape: {df.shape}")
    
    df_geospatial = pd.read_parquet(local_geospatial_path)
    logging.info(f"Successfully loaded geospatial data. Shape: {df_geospatial.shape}")
    
    # Merge geospatial features with main dataset on property_id
    if 'property_id' in df.columns and 'property_id' in df_geospatial.columns:
        df = df.merge(df_geospatial, on='property_id', how='left', suffixes=('', '_geo'))
        logging.info(f"Merged datasets on property_id. Final shape: {df.shape}")
    else:
        logging.warning("property_id column not found in one or both datasets. Concatenating instead.")
        df = pd.concat([df, df_geospatial], axis=1)
        logging.info(f"Concatenated datasets. Final shape: {df.shape}")

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
    cols_with_missing = df.columns[df.isnull().any()].tolist()
    
    # Exclude the primary key from imputation
    if 'property_id' in cols_with_missing:
        cols_with_missing.remove('property_id')
    if 'property_address' in cols_with_missing:
        cols_with_missing.remove('property_address')

    if not cols_with_missing:
        logging.info("No missing values found. No imputation needed. Exiting.")
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
            
            # --- SPEEDUP: Use a random subset of features if configured ---
            if args.max_features > 0 and len(all_features) > args.max_features:
                features = np.random.choice(all_features, args.max_features, replace=False).tolist()
            else:
                features = all_features
            
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

python3 impute_mice_lightgbm.py \
    --gcs_bucket="${GCS_BUCKET}" \
    --input_gcs_path="${STAGE2_MERGED_INPUT_GCS_PATH}" \
    --geospatial_gcs_path="${GEOSPATIAL_FEATURES_GCS_PATH}" \
    --output_gcs_path="${FINAL_OUTPUT_GCS_PATH}" \
    --iterations=${MICE_ITERATIONS} \
    --n_estimators=${LGBM_N_ESTIMATORS} \
    --num_leaves=${LGBM_NUM_LEAVES} \
    --max_features=${LGBM_MAX_FEATURES}

echo "--- HYBRID IMPUTATION PIPELINE COMPLETE ---"
echo "Final fully imputed dataset available at: gs://${GCS_BUCKET}/${FINAL_OUTPUT_GCS_PATH}"

echo "--- Pipeline Finished: $(date) ---"