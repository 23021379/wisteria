#!/bin/bash
#
# run_model_training_v5.sh
#
# V5 REVISION: Using row index matching between raw and processed files
#

# -- Strict Mode & Configuration --
set -e
set -o pipefail
set -x

# This trap ensures that the VM will automatically stop itself when the script exits,
# either by completing successfully or by crashing due to an error. This is a
# crucial cost-saving measure to prevent the VM from running indefinitely.
# It uses the instance metadata server to reliably find its own name and zone.

trap "echo '--- SCRIPT FINISHED OR CRASHED: INITIATING AUTO-SHUTDOWN ---'; gcloud compute instances stop $(hostname) --zone=$(curl -s -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/zone | awk -F/ '{print $NF}')" EXIT



# -- GCP & Project Configuration --
GCS_BUCKET="srgan-bucket-ace-botany-453819-t4"
TIMESTAMP=$(date +"%Y%m%d-%H%M%S")
OUTPUT_GCS_DIR="gs://${GCS_BUCKET}/model_training/run_${TIMESTAMP}"
# Use the specified pre-trained forecast artifacts instead of generating new ones.
FORECAST_ARTIFACTS_GCS_DIR="gs://srgan-bucket-ace-botany-453819-t4/model_training/run_20250805-110318/forecast_artifacts"
N_TRIALS=50 # Number of Bayesian Optimization trials for the main model
N_TRIALS_AE=25 # Number of trials for offline Autoencoder tuning
MASTER_DATA_GCS_PATH="gs://${GCS_BUCKET}/imputation_pipeline/output_lgbm_legacy/final_fully_imputed_dataset.parquet"
LATEST_FEATURE_SORTING_RUN_DIR="gs://${GCS_BUCKET}/feature_sorting/run_legacy" 
FEATURE_SETS_GCS_PATH="${LATEST_FEATURE_SORTING_RUN_DIR}/feature_sets.json"
RIGHTMOVE_GCS_PATH="gs://${GCS_BUCKET}/house data scrape/Rightmove.csv"

# --- UPDATED: Use the unprocessed file as key map ---
KEY_MAP_GCS_PATH="gs://${GCS_BUCKET}/house data scrape/merged_property_data_with_coords.csv"

# -- GCP Permission Check --
echo "Verifying GCS permissions for bucket gs://${GCS_BUCKET}/..."
# 1. Check for bucket existence and list permissions (read access)
if ! gsutil -q ls -b "gs://${GCS_BUCKET}/"; then
    echo "ERROR: Bucket gs://${GCS_BUCKET}/ does not exist or you don't have permissions to list it (storage.objects.list)." >&2
    exit 1
fi
# 2. Check for write permissions by creating and deleting a test object.
if ! (echo "permission test" | gsutil cp - "gs://${GCS_BUCKET}/.wisteria_permission_test" > /dev/null 2>&1 && gsutil rm "gs://${GCS_BUCKET}/.wisteria_permission_test" > /dev/null 2>&1); then
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" >&2
    echo "!!! ERROR: GCS UPLOAD PERMISSION DENIED.                                   !!!" >&2
    echo "!!! The service account for this VM does not have write permissions        !!!" >&2
    echo "!!! (storage.objects.create/delete) on the bucket 'gs://${GCS_BUCKET}/'.   !!!" >&2
    echo "!!!                                                                        !!!" >&2
    echo "!!! TO FIX THIS:                                                           !!!" >&2
    echo "!!! 1. Go to the Google Cloud Console -> VM Instances.                     !!!" >&2
    echo "!!! 2. STOP the instance running this script.                              !!!" >&2
    echo "!!! 3. EDIT the instance. In the 'API and identity management' section,    !!!" >&2
    echo "!!!    ensure the access scope is 'Allow full access to all Cloud APIs'.   !!!" >&2
    echo "!!! 4. Alternatively, ensure the attached Service Account has the          !!!" >&2
    echo "!!!    'Storage Object Admin' IAM role for this project or bucket.         !!!" >&2
    echo "!!! 5. START the instance and re-run the script.                         !!!" >&2
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" >&2
    exit 1
fi
echo "GCS read/write permissions verified successfully."


# -- Local Configuration --
PROJECT_DIR="./model_training_project_v5"
VENV_DIR="${PROJECT_DIR}/venv_mt"
OUTPUT_DIR="${PROJECT_DIR}/output"
DATA_DIR="${PROJECT_DIR}/data"
MASTER_DATA_LOCAL_PATH="${DATA_DIR}/master_dataset.parquet"
FEATURE_SETS_LOCAL_PATH="${DATA_DIR}/feature_sets.json"
RIGHTMOVE_DATA_LOCAL_PATH="${DATA_DIR}/Rightmove.csv"
SCRIPT_PATH="${PROJECT_DIR}/02_train_multi_head_model_v5.py"
KEY_MAP_LOCAL_PATH="${DATA_DIR}/key_map.csv"

# --- Local Project Setup ---
PROJECT_DIR_NAME="model_training_project_v5" # Just the name
mkdir -p "./${PROJECT_DIR_NAME}"
cd "./${PROJECT_DIR_NAME}"

# --- Environment Setup (Define paths relative to the NEW current directory) ---
VENV_DIR="./venv_mt" # Simple relative path

# --- System & Driver Setup (MUST be done before venv creation) ---
echo "Updating system packages and installing L4 GPU dependencies..."
sudo apt-get update
# Install a modern NVIDIA driver compatible with the L4 GPU's Ada Lovelace architecture
sudo apt-get install -y libgomp1 nvidia-driver-535


# --- Python Environment & Package Installation ---
# Force remove the old venv to ensure a clean installation
echo "Removing old virtual environment to ensure a clean slate..."
rm -rf "${VENV_DIR}"

if [ ! -d "${VENV_DIR}" ]; then
    echo "Creating Python virtual environment..."
    # python3 -m venv needs the path relative to here
    python3 -m venv "${VENV_DIR}"
fi
# source also needs the path relative to here
source "${VENV_DIR}/bin/activate"

echo "Installing required Python packages for deep learning..."
pip install --upgrade pip
# IMPORTANT: Force re-install PyTorch compiled for CUDA 12.1 to overwrite any old, cached versions.
pip install --force-reinstall --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install pandas pyarrow gcsfs google-cloud-storage scikit-learn lightgbm fuzzywuzzy optuna matplotlib seaborn python-Levenshtein tqdm shap


# # --- Python FORECASTING Worker Script Generation (NEW) ---    !TEMPORAL MODELS HAVE BEEN TRAINED, IT IS QUICKER TO JUST USE THEIR ARTIFACTS FROM NOW ON.!
# cat > "${FORECAST_SCRIPT_PATH}" <<'EOF'
# import os
# import gc
# import re
# import json
# import joblib
# import pandas as pd
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# from sklearn.preprocessing import QuantileTransformer
# from sklearn.model_selection import train_test_split

# # --- Configuration ---
# MASTER_DATA_PATH = os.environ.get("MASTER_DATA_LOCAL_PATH")
# FEATURE_SETS_PATH = os.environ.get("FEATURE_SETS_LOCAL_PATH")
# OUTPUT_DIR = os.environ.get("OUTPUT_DIR")
# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# INPUT_WINDOW = 24  # Use 24 months of history to predict the next month
# EPOCHS = 100
# BATCH_SIZE = 512

# # --- Model Definitions ---
# class Chomp1d(nn.Module):
#     def __init__(self, chomp_size): super(Chomp1d, self).__init__(); self.chomp_size = chomp_size
#     def forward(self, x): return x[:, :, :-self.chomp_size].contiguous()

# class TemporalBlock(nn.Module):
#     def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
#         super(TemporalBlock, self).__init__()
#         self.conv1 = nn.utils.parametrizations.weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
#         self.chomp1 = Chomp1d(padding)
#         self.relu1 = nn.ReLU()
#         self.dropout1 = nn.Dropout(dropout)
#         self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1)
#         self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
#         self.relu = nn.ReLU()
#         self.init_weights()
#     def init_weights(self):
#         self.conv1.weight.data.normal_(0, 0.01)
#         if self.downsample is not None: self.downsample.weight.data.normal_(0, 0.01)
#     def forward(self, x):
#         out = self.net(x)
#         res = x if self.downsample is None else self.downsample(x)
#         return self.relu(out + res)

# class ForecastTCN(nn.Module):
#     def __init__(self, input_feature_dim, output_dim, num_channels, kernel_size=3, dropout=0.3):
#         super(ForecastTCN, self).__init__()
#         layers = []
#         num_levels = len(num_channels)
#         for i in range(num_levels):
#             dilation_size = 2**i
#             in_channels = input_feature_dim if i == 0 else num_channels[i-1]
#             out_channels = num_channels[i]
#             layers.append(TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, padding=(kernel_size-1) * dilation_size, dropout=dropout))
#         self.network = nn.Sequential(*layers)
#         self.final_conv = nn.Conv1d(num_channels[-1], output_dim, 1)

#     def forward(self, x):
#         # x shape: (batch_size, n_features, n_timesteps)
#         y = self.network(x)
#         # We want the prediction at the last time step
#         y = self.final_conv(y[:, :, -1:])
#         return y.squeeze(-1)

# # --- Data Preparation Functions ---
# def parse_and_prepare_data_for_prop_type(df, cols):
#     if not cols: return None, 0, 0
#     parsed_data = {}
#     # Universal pattern for both raw and compass pp history
#     pattern = re.compile(r".*_pp_(\d{4})_(\d{2})_([DSTF])_.*")
    
#     feature_stems = set()
#     for col in cols:
#         match = pattern.match(col)
#         if match:
#             year, month, _ = match.groups()
#             timestep = (int(year), int(month))
#             feature_stem = col.replace(f"_{year}_{month}", "")
#             feature_stems.add(feature_stem)
#             if timestep not in parsed_data: parsed_data[timestep] = {}
#             parsed_data[timestep][feature_stem] = col

#     timesteps = sorted(parsed_data.keys())
#     canonical_features = sorted(list(feature_stems))
#     n_timesteps = len(timesteps)
#     n_features_per_timestep = len(canonical_features)
    
#     if n_timesteps == 0 or n_features_per_timestep == 0: return None, 0, 0
    
#     tcn_array = np.zeros((len(df), n_timesteps, n_features_per_timestep))
#     for j, ts in enumerate(timesteps):
#         for i, stem in enumerate(canonical_features):
#             col_name = parsed_data.get(ts, {}).get(stem)
#             if col_name and col_name in df.columns:
#                 tcn_array[:, j, i] = df[col_name].fillna(0).values

#     return tcn_array, n_timesteps, n_features_per_timestep

# # --- NEW: Memory-Efficient Dataset Class ---
# class SlidingWindowDataset(Dataset):
#     def __init__(self, data_array, input_window):
#         # Keep data as a NumPy array to conserve memory.
#         self.data_array = data_array 
#         self.input_window = input_window
#         # Get shape from the numpy array directly
#         self.n_samples, self.n_timesteps, self.n_features = self.data_array.shape
#         self.num_windows_per_sample = self.n_timesteps - self.input_window

#     def __len__(self):
#         return self.n_samples * self.num_windows_per_sample

#     def __getitem__(self, idx):
#         sample_idx = idx // self.num_windows_per_sample
#         start_idx = idx % self.num_windows_per_sample
        
#         # Slicing is done on the NumPy array, which is very fast.
#         X_np = self.data_array[sample_idx, start_idx:start_idx + self.input_window, :]
#         y_np = self.data_array[sample_idx, start_idx + self.input_window, :]
        
#         # Convert ONLY the small slice to a tensor on-the-fly.
#         X = torch.tensor(X_np, dtype=torch.float32)
#         y = torch.tensor(y_np, dtype=torch.float32)
        
#         # Permute X for TCN: (Batch, Features, Time)
#         return X.permute(1, 0), y

# def main():
#     print("--- Starting Temporal Forecast Model Training ---")
#     os.makedirs(OUTPUT_DIR, exist_ok=True)
#     df = pd.read_parquet(MASTER_DATA_PATH)
#     with open(FEATURE_SETS_PATH, 'r') as f: feature_sets = json.load(f)
    
#     PROP_TYPES = ['D', 'S', 'T', 'F']
#     scalers = {}
    
#     for p_type in PROP_TYPES:
#         print(f"\n{'='*20} Processing Property Type: {p_type} {'='*20}")
#         raw_cols = [c for c in df.columns if re.match(fr".*_pp_(\d{{4}})_(\d{{2}})_{p_type}_.*", c)]
#         compass_cols = [c for c in df.columns if re.match(fr".*compass_.*_pp_(\d{{4}})_(\d{{2}})_{p_type}_.*", c)]
#         spatio_temporal_cols = feature_sets.get(f'head_F_spatio_temporal_{p_type}', [])
        
#         all_cols_for_type = sorted(list(set(raw_cols + compass_cols + spatio_temporal_cols)))
        
#         if not all_cols_for_type:
#             print(f"  - No historical price data found for type '{p_type}'. Skipping.")
#             continue
            
#         data_array, n_timesteps, n_features = parse_and_prepare_data_for_prop_type(df, all_cols_for_type)
#         if data_array is None or n_features == 0 or n_timesteps <= INPUT_WINDOW:
#             print(f"  - Not enough data or timesteps to train for type '{p_type}'. Skipping.")
#             continue
        
#         n_samples = data_array.shape[0]
#         data_flat = data_array.reshape(n_samples * n_timesteps, n_features)
        
#         scaler = QuantileTransformer(output_distribution='normal', n_quantiles=max(min(n_samples // 10, 100), 10), random_state=42)
#         data_scaled_flat = scaler.fit_transform(data_flat)
#         scalers[p_type] = scaler
        
#         data_scaled_array = data_scaled_flat.reshape(n_samples, n_timesteps, n_features)
        
#         # Create the full dataset object
#         full_dataset = SlidingWindowDataset(data_scaled_array, INPUT_WINDOW)
#         print(f"  - Created on-the-fly dataset with {len(full_dataset)} total sequences for type '{p_type}'.")

#         # Split the dataset using indices, not by creating new data arrays
#         train_size = int(0.85 * len(full_dataset))
#         val_size = len(full_dataset) - train_size
#         train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

#         train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
#         val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE * 2)

#         model = ForecastTCN(
#             input_feature_dim=n_features,
#             output_dim=n_features,
#             num_channels=[64, 128],
#             kernel_size=3,
#             dropout=0.3
#         ).to(DEVICE)

#         optimizer = optim.AdamW(model.parameters(), lr=1e-3)
#         scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=5, verbose=False)
#         criterion = nn.MSELoss()

#         print(f"  - Starting model training for type '{p_type}'...")
#         for epoch in range(EPOCHS):
#             model.train()
#             train_loss = 0
#             for x_batch, y_batch in train_loader:
#                 x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
#                 optimizer.zero_grad()
#                 preds = model(x_batch)
#                 loss = criterion(preds, y_batch)
#                 loss.backward()
#                 optimizer.step()
#                 train_loss += loss.item()
            
#             model.eval()
#             val_loss = 0
#             with torch.no_grad():
#                 for x_batch, y_batch in val_loader:
#                     x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
#                     preds = model(x_batch)
#                     val_loss += criterion(preds, y_batch).item()
            
#             avg_train_loss = train_loss / len(train_loader)
#             avg_val_loss = val_loss / len(val_loader)
#             scheduler.step(avg_val_loss)
            
#             if (epoch + 1) % 20 == 0:
#                 print(f"    Epoch {epoch+1}/{EPOCHS} -> Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        
#         print(f"  - Training complete for type '{p_type}'. Saving model.")
#         torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f"forecast_model_{p_type}.pt"))
#         gc.collect()

#     print("\nSaving scalers artifact.")
#     joblib.dump(scalers, os.path.join(OUTPUT_DIR, "forecast_scalers.joblib"))
#     print("All forecast models and scalers saved successfully.")

# if __name__ == "__main__":
#     main()
# EOF


# --- Python Worker Script Generation (V5) ---
# Update the Python worker script to handle missing MSOA column
cat > "${SCRIPT_PATH}" <<'EOF'
import os
import gc
import json
import re
import ast
from datetime import datetime
import warnings
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, r2_score
import lightgbm as lgb
import torch.nn.functional as F
import hashlib
from fuzzywuzzy import fuzz, process
import optuna
import joblib
from joblib import Parallel, delayed
from tqdm import tqdm

# --- Configuration ---
MASTER_DATA_PATH = os.environ.get("MASTER_DATA_LOCAL_PATH")
FEATURE_SETS_PATH = os.environ.get("FEATURE_SETS_LOCAL_PATH")
FORECAST_ARTIFACTS_GCS_DIR = os.environ.get("FORECAST_ARTIFACTS_GCS_DIR")
RIGHTMOVE_DATA_PATH = os.environ.get("RIGHTMOVE_DATA_LOCAL_PATH")
KEY_MAP_PATH = os.environ.get("KEY_MAP_LOCAL_PATH")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR")
N_TRIALS_MAIN = int(os.environ.get("N_TRIALS", 50)) # For the main 'mid' stratum
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_FOLDS_OPTUNA = 3 # Use fewer folds for fast hyperparameter search.
NUM_FOLDS_FINAL = 10 # Use more folds for robust final model training.
BATCH_SIZE = 256 # This will be optimized
EPOCHS = 150 # Reduced for faster trials, can be increased later

# --- Rightmove Parsing Functions ---
def parse_sales_history(sales_str):
    if pd.isna(sales_str) or sales_str == '' or sales_str == '[]': return pd.DataFrame()
    try:
        sales_list = ast.literal_eval(str(sales_str)); sales_data = []
        if not isinstance(sales_list, list): return pd.DataFrame()
        for item in sales_list:
            item = str(item).strip()
            if '\n' not in item: continue
            lines = item.split('\n')
            if len(lines) >= 2:
                date_str = lines[0].strip()
                try:
                    sale_date = datetime.strptime(date_str, '%d %b %Y')
                    price_match = re.search(r'£([\d,]+)', item)
                    price = int(price_match.group(1).replace(',', '')) if price_match else 0
                    if price > 0: sales_data.append({'sale_date': sale_date, 'sale_price': price})
                except ValueError: continue
        return pd.DataFrame(sales_data)
    except (ValueError, SyntaxError): return pd.DataFrame()

def process_rightmove_data(raw_rightmove_df):
    print("Processing Rightmove sales data to extract features...")
    processed_data = []; sales_history_col = None
    if 'sales_history' in raw_rightmove_df.columns: sales_history_col = 'sales_history'
    elif raw_rightmove_df.shape[1] > 2: sales_history_col = raw_rightmove_df.columns[2]
    else: raise ValueError("Sales history column not found in Rightmove data.")
    for idx, row in raw_rightmove_df.iterrows():
        sales_df = parse_sales_history(row[sales_history_col])
        if not sales_df.empty:
            sales_df = sales_df.sort_values('sale_date', ascending=False).reset_index(drop=True)
            most_recent = sales_df.iloc[0]
            features = {'rightmove_row_id': idx, 'most_recent_sale_price': most_recent['sale_price'], 'most_recent_sale_year': most_recent['sale_date'].year, 'most_recent_sale_month': most_recent['sale_date'].month, 'total_sales_count': len(sales_df)}
            if len(sales_df) > 1:
                previous_sale = sales_df.iloc[1]
                features['days_since_last_sale'] = (most_recent['sale_date'] - previous_sale['sale_date']).days
                features['price_change_since_last'] = most_recent['sale_price'] - previous_sale['sale_price']
            processed_data.append(features)
    print(f"Successfully processed {len(processed_data)} properties with valid sales data.")
    return pd.DataFrame(processed_data)


def generate_forecast_features(df, forecast_models, forecast_scalers, feature_sets, max_horizon=36, input_window=24):
    """
    [LEAKAGE-PROOF & PARALLELIZED] Generates DENSE monthly forecasts for each property
    using ONLY the data available at the time of its sale.
    """
    print(f"  - Generating dense, point-in-time correct monthly forecasts up to {max_horizon} months...")
    
    # Initialize a list to hold all new feature columns, one for each month
    all_forecasts_dfs = []
    
    property_type_col = 'num__property_main_type_encoded__1Flat_hm'
    if property_type_col not in df.columns:
        print(f"FATAL ERROR: The required property type column ('{property_type_col}') was not found.")
        exit(1)

    # --- Pre-computation for all workers ---
    df['property_type_char'] = np.where(df[property_type_col] == 1, 'F', 'D')
    p_types_np = df['property_type_char'].to_numpy()
    sale_years_np = df['most_recent_sale_year'].to_numpy()
    sale_months_np = df['most_recent_sale_month'].to_numpy()
    fallback_count = pd.isna(sale_years_np).sum()

    all_cols_by_type = {}
    for p_type in ['D', 'S', 'T', 'F']:
        raw_cols = [c for c in df.columns if re.match(fr".*_pp_(\d{{4}})_(\d{{2}})_{p_type}_.*", c)]
        compass_cols = [c for c in df.columns if re.match(fr".*compass_.*_pp_(\d{{4}})_(\d{{2}})_{p_type}_.*", c)]
        spatio_temporal_cols = feature_sets.get(f'head_F_spatio_temporal_{p_type}', [])
        all_cols_by_type[p_type] = sorted(list(set(raw_cols + compass_cols + spatio_temporal_cols)))

    # --- Parallel Execution using joblib ---
    tasks = []
    for i in range(len(df)):
        p_type = p_types_np[i]
        model_state_dict = forecast_models[p_type].state_dict() if p_type in forecast_models else None
        scaler = forecast_scalers.get(p_type)
        
        tasks.append(delayed(_forecast_worker)(
            i, df.iloc[i], p_type, sale_years_np[i], sale_months_np[i],
            model_state_dict, scaler, all_cols_by_type,
            max_horizon, input_window, DEVICE
        ))

    results = Parallel(n_jobs=-1, backend="loky")(tqdm(tasks, total=len(df), desc="Generating Forecast Features"))

    # --- Assemble results ---
    # Create a temporary dictionary to hold the full results matrix
    results_matrix = {f'forecast_price_{h}m': np.zeros(len(df)) for h in range(1, max_horizon + 1)}
    
    for i, horizon_predictions in results:
        if horizon_predictions:
            for h, value in horizon_predictions.items():
                col_name = f'forecast_price_{h}m'
                if col_name in results_matrix:
                    results_matrix[col_name][i] = value

    # Convert the matrix to a DataFrame and add it to the main df
    forecasts_df = pd.DataFrame(results_matrix, index=df.index)
    df = pd.concat([df, forecasts_df], axis=1)
    new_feature_names = forecasts_df.columns.tolist()
        
    print(f"  - Generated {len(new_feature_names)} new dense monthly forecast features.")
    if fallback_count > 0:
        print(f"  - NOTE: Used fallback date (August 2024) for {fallback_count} properties with missing sale dates.")
        
    return df, new_feature_names

def create_head_params(params_dict, feature_configs):
    """Creates the head_params dictionary from a given parameter set (from Optuna trial or best_params)."""
    head_params = {}
    
    # Define head categories consistently
    LARGE_HEADS = ['head_A_dna', 'head_C_census', 'head_compass_raw', 'head_E_temporal']
    # Add any spatio-temporal heads that are dynamically found
    LARGE_HEADS.extend([h for h in feature_configs if 'spatio_temporal' in h])
    
    VISUAL_HEADS = ['head_B_aesthetic', 'head_G_gemini_quantitative']
    
    # Dynamically find all other specialist heads (like AVM, price history, etc.)
    all_categorized_heads = set(LARGE_HEADS + VISUAL_HEADS)
    SMALL_HEADS = [h for h in feature_configs if h != 'head_base' and h not in all_categorized_heads]

    for head_name in list(feature_configs.keys()):
        if head_name in LARGE_HEADS:
            head_params[head_name] = {'hidden_dim': params_dict['large_head_hidden_dim'], 'dropout_rate': params_dict['large_head_dropout']}
        elif head_name in VISUAL_HEADS:
            head_params[head_name] = {'hidden_dim': params_dict['small_head_hidden_dim'], 'dropout_rate': params_dict['visual_head_dropout']}
        elif head_name in SMALL_HEADS:
            head_params[head_name] = {'hidden_dim': params_dict['small_head_hidden_dim'], 'dropout_rate': params_dict['small_head_dropout']}
        elif head_name == 'head_base':
            # Base head is treated as a large head
            head_params[head_name] = {'hidden_dim': params_dict['large_head_hidden_dim'], 'dropout_rate': params_dict['large_head_dropout']}
            
    return head_params

def mean_absolute_percentage_error(y_true, y_pred):
    """Calculates MAPE, ensuring robustness against division by zero."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Avoid division by zero for properties with a price of 0, though our filtering should prevent this.
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def parse_and_prepare_data_for_prop_type(df, cols):
    if not cols: return None, 0, 0
    parsed_data = {}
    pattern = re.compile(r".*_pp_(\d{4})_(\d{2})_([DSTF])_.*")
    
    feature_stems = set()
    for col in cols:
        match = pattern.match(col)
        if match:
            year, month, _ = match.groups()
            timestep = (int(year), int(month))
            feature_stem = col.replace(f"_{year}_{month}", "")
            feature_stems.add(feature_stem)
            if timestep not in parsed_data: parsed_data[timestep] = {}
            parsed_data[timestep][feature_stem] = col

    timesteps = sorted(parsed_data.keys())
    canonical_features = sorted(list(feature_stems))
    n_timesteps = len(timesteps)
    n_features_per_timestep = len(canonical_features)
    
    if n_timesteps == 0 or n_features_per_timestep == 0: return None, 0, 0
    
    tcn_array = np.zeros((len(df), n_timesteps, n_features_per_timestep))
    for j, ts in enumerate(timesteps):
        for i, stem in enumerate(canonical_features):
            col_name = parsed_data.get(ts, {}).get(stem)
            if col_name and col_name in df.columns:
                tcn_array[:, j, i] = df[col_name].fillna(0).values

    return tcn_array, n_timesteps, n_features_per_timestep


# --- PyTorch & Model Classes ---
class ForecastTCN(nn.Module):
    def __init__(self, input_feature_dim, output_dim, num_channels, kernel_size=3, dropout=0.3):
        super(ForecastTCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = input_feature_dim if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers.append(TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, padding=(kernel_size-1) * dilation_size, dropout=dropout))
        self.network = nn.Sequential(*layers)
        self.final_conv = nn.Conv1d(num_channels[-1], output_dim, 1)

    def forward(self, x):
        y = self.network(x)
        y = self.final_conv(y[:, :, -1:])
        return y.squeeze(-1)

class WisteriaDataset(Dataset):
    def __init__(self, data_dict, y_price, y_uncertainty):
        # FIX: Make a local copy to prevent mutating the original dictionary passed in.
        local_data_dict = data_dict.copy()
        msoa_df = local_data_dict.pop('msoa_id')
        self.msoa_id_tensor = torch.tensor(msoa_df.values, dtype=torch.long).squeeze()
        self.feature_tensors = {k: torch.tensor(v.values, dtype=torch.float32) for k, v in local_data_dict.items()}
        self.y_price = torch.tensor(y_price.values, dtype=torch.float32).unsqueeze(1)
        self.y_uncertainty = torch.tensor(y_uncertainty.values, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.y_price)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.feature_tensors.items()}
        item['msoa_id'] = self.msoa_id_tensor[idx]
        item['target_price'] = self.y_price[idx]
        item['target_uncertainty'] = self.y_uncertainty[idx]
        return item

class Chomp1d(nn.Module):
    def __init__(self, chomp_size): super(Chomp1d, self).__init__(); self.chomp_size = chomp_size
    def forward(self, x): return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2): super(TemporalBlock, self).__init__(); self.conv1 = nn.utils.parametrizations.weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)); self.chomp1 = Chomp1d(padding); self.relu1 = nn.ReLU(); self.dropout1 = nn.Dropout(dropout); self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1); self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None; self.relu = nn.ReLU(); self.init_weights()
    def init_weights(self): self.conv1.weight.data.normal_(0, 0.01); _ = self.downsample.weight.data.normal_(0, 0.01) if self.downsample is not None else None
    def forward(self, x): out = self.net(x); res = x if self.downsample is None else self.downsample(x); return self.relu(out + res)

class TCNHead(nn.Module):
    def __init__(self, input_dim, n_features_per_timestep, n_timesteps, num_channels, kernel_size=2, dropout=0.2):
        super(TCNHead, self).__init__()
        # Ensure the input dimension is correct
        assert input_dim == n_features_per_timestep * n_timesteps, \
            f"Input dimension ({input_dim}) must equal n_features_per_timestep ({n_features_per_timestep}) * n_timesteps ({n_timesteps})"
        
        self.n_features_per_timestep = n_features_per_timestep
        self.n_timesteps = n_timesteps
        
        layers = []
        num_levels = len(num_channels)
        # The first TemporalBlock input should be the number of features per timestep
        input_channels = n_features_per_timestep
        
        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = input_channels if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers.append(TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, padding=(kernel_size-1) * dilation_size, dropout=dropout))
        
        self.network = nn.Sequential(*layers)
        self.output_dim = num_channels[-1]

    def forward(self, x):
        # x is expected to have shape (batch_size, n_timesteps * n_features_per_timestep)
        batch_size = x.shape[0]
        
        # Reshape to (batch_size, n_timesteps, n_features_per_timestep)
        x_reshaped = x.view(batch_size, self.n_timesteps, self.n_features_per_timestep)
        
        # Permute to (batch_size, n_features_per_timestep, n_timesteps) for TCN input
        x_permuted = x_reshaped.permute(0, 2, 1)
        
        # The output of the TCN network will be (batch_size, num_output_channels, n_timesteps)
        output = self.network(x_permuted)
        
        # Return the output of the last timestep
        return output[:, :, -1]


class SimpleHead(nn.Module):
    """A simple, standardized MLP head for all feature groups."""
    def __init__(self, input_dim, output_dim, hidden_dim=256, dropout=0.3):
        super(SimpleHead, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        self.output_dim = output_dim

    def forward(self, x):
        return self.network(x)


class EnhancedTCNHead(nn.Module):
    def __init__(self, input_dim, n_features_per_timestep, n_timesteps, num_channels, kernel_size=2, dropout=0.2):
        super(EnhancedTCNHead, self).__init__()
        assert input_dim == n_features_per_timestep * n_timesteps, \
            f"Input dimension ({input_dim}) must equal n_features_per_timestep ({n_features_per_timestep}) * n_timesteps ({n_timesteps})"
        
        self.n_features_per_timestep = n_features_per_timestep
        self.n_timesteps = n_timesteps
        
        # --- TCN Backbone ---
        layers = []
        num_levels = len(num_channels)
        input_channels = n_features_per_timestep
        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = input_channels if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers.append(TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, padding=(kernel_size-1) * dilation_size, dropout=dropout))
        self.tcn_backbone = nn.Sequential(*layers)
        
        # --- Attention Mechanism ---
        tcn_output_channels = num_channels[-1]
        self.attention = nn.Sequential(
            nn.Linear(tcn_output_channels, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        self.output_dim = tcn_output_channels

    def forward(self, x):
        # x: (batch_size, n_timesteps * n_features_per_timestep)
        batch_size = x.shape[0]
        
        # Reshape and Permute for TCN
        x_reshaped = x.view(batch_size, self.n_timesteps, self.n_features_per_timestep)
        x_permuted = x_reshaped.permute(0, 2, 1) # -> (batch_size, features, timesteps)
        
        # Get TCN feature maps
        tcn_out = self.tcn_backbone(x_permuted) # -> (batch_size, tcn_channels, timesteps)
        tcn_out_permuted = tcn_out.permute(0, 2, 1) # -> (batch_size, timesteps, tcn_channels)
        
        # Calculate attention weights
        # attn_weights shape: (batch_size, timesteps, 1)
        attn_weights = self.attention(tcn_out_permuted)
        attn_weights = torch.softmax(attn_weights, dim=1)
        
        # Apply attention weights
        # context_vector shape: (batch_size, 1, tcn_channels)
        context_vector = torch.bmm(attn_weights.transpose(1, 2), tcn_out_permuted)
        
        # Squeeze to final output shape: (batch_size, tcn_channels)
        final_output = context_vector.squeeze(1)
        
        return final_output


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.projection = nn.Linear(input_dim, embed_dim)
        self.dropout = nn.Dropout(p=dropout)
        
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-np.log(10000.0) / embed_dim))
        self.register_buffer('div_term', div_term)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projection(x)
        pe = torch.zeros_like(x)
        pe[:, 0::2] = torch.sin(x[:, 0::2] * self.div_term)
        pe[:, 1::2] = torch.cos(x[:, 1::2] * self.div_term)
        return self.dropout(x + pe)

class SpatialFusion(nn.Module):
    def __init__(self, msoa_embed_dim, pos_encode_dim, spatial_head_dim, fusion_output_dim, num_heads=4, dropout=0.2):
        super().__init__()
        self.fusion_output_dim = fusion_output_dim
        self.msoa_proj = nn.Linear(msoa_embed_dim, fusion_output_dim)
        self.pos_proj = nn.Linear(pos_encode_dim, fusion_output_dim)
        self.spatial_head_proj = nn.Linear(spatial_head_dim, fusion_output_dim)
        
        self.attention = nn.MultiheadAttention(embed_dim=fusion_output_dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(fusion_output_dim)
        
    def forward(self, msoa_embed, pos_encoding, spatial_head_output):
        msoa_p = self.msoa_proj(msoa_embed).unsqueeze(1)
        pos_p = self.pos_proj(pos_encoding).unsqueeze(1)
        spatial_p = self.spatial_head_proj(spatial_head_output).unsqueeze(1)
        
        stacked_inputs = torch.cat([msoa_p, pos_p, spatial_p], dim=1)
        attn_output, _ = self.attention(stacked_inputs, stacked_inputs, stacked_inputs)
        fused_vector = self.norm(attn_output.mean(dim=1))
        return fused_vector


class FusionModel(nn.Module):
    def __init__(self, feature_configs, msoa_cardinality, head_params, fusion_embed_dim, msoa_embedding_dim=16, fusion_dropout_rate=0.5, use_enhanced_tcn=True, fusion_input_cap=2048, cross_attention_dropout_rate=0.2, attention_dropout_rate=0.1, tcn_dropout=0.3, spatio_temporal_attention_dropout=0.25, stratum_name='monolith'):
        super().__init__()
        
        self.specialist_heads = nn.ModuleDict()
        self.base_head = None
        tcn_class = EnhancedTCNHead if use_enhanced_tcn else TCNHead
        common_output_dim = fusion_embed_dim

        # --- Set Model Capacity ---
        # With the removal of stratification, we now use a single, high-capacity architecture.
        print("  - Building a HIGH-CAPACITY monolithic model.")
        fusion_mlp_dims = [1024, 512, 256]

        # Define the hierarchical structure
        self.HEAD_GROUPS = {
            'spatio_temporal': [f'head_F_spatio_temporal_{p_type}' for p_type in ['D', 'S', 'T', 'F']],
            'static_context': ['head_C_census', 'head_compass_raw'],
            'visual': ['head_B_aesthetic', 'head_G_gemini_quantitative']
        }
        # Dynamically determine individual heads as any specialist head not in a defined group.
        all_grouped_heads = set(h for group_list in self.HEAD_GROUPS.values() for h in group_list)
        self.INDIVIDUAL_HEADS = [
            h_name for h_name in feature_configs.keys()
            if h_name not in all_grouped_heads and h_name != 'head_base'
        ]

        # --- 1. Create all head modules ---
        for head_name, config in feature_configs.items():
            # This loop now robustly creates a head for every entry in feature_configs.
            # It distinguishes between TCN and MLP heads based on the config content.
            if 'input_dim' not in config: continue # Skip if config is malformed
            input_dim = config['input_dim']
            
            # --- Generalized Head Creation ---
            if 'n_features_per_timestep' in config and 'n_timesteps' in config:
                print(f" - Creating TCN head for '{head_name}'...")
                head_module = tcn_class(
                    input_dim=input_dim,
                    n_features_per_timestep=config['n_features_per_timestep'],
                    n_timesteps=config['n_timesteps'],
                    num_channels=[64, common_output_dim],
                    kernel_size=3,
                    dropout=tcn_dropout
                )
            elif config.get('is_tabnet', False):
                print(f" - Creating TabNet head for '{head_name}'...")
                # TabNet parameters can be tuned, but we'll use robust defaults for now.
                head_module = TabNetHead(
                    input_dim=input_dim,
                    output_dim=common_output_dim,
                    n_d=32, n_a=32, n_steps=4, gamma=1.5
                )
            else: # Standard MLP Head
                h_params = head_params[head_name]
                dropout, hidden_dim = h_params['dropout_rate'], h_params['hidden_dim']
                print(f" - Creating Simple MLP head for '{head_name}'...")
                head_module = SimpleHead(input_dim, common_output_dim, hidden_dim=hidden_dim, dropout=dropout)
            
            if head_name == 'head_base':
                self.base_head = head_module
            else:
                self.specialist_heads[head_name] = head_module

        # --- 1.5 Create Spatial Fusion Components ---
        pos_encoding_dim = 32
        self.positional_encoder = SinusoidalPositionalEncoding(input_dim=2, embed_dim=pos_encoding_dim)
        self.spatial_fusion_block = SpatialFusion(
            msoa_embed_dim=msoa_embedding_dim,
            pos_encode_dim=pos_encoding_dim,
            spatial_head_dim=common_output_dim,
            fusion_output_dim=64
        )

        # --- 2. Create Intra-Group Attention Layers (Stage 1) with targeted regularization ---
        self.group_attentions = nn.ModuleDict()
        for group_name in self.HEAD_GROUPS:
            # Apply stronger dropout to the spatio-temporal fusion block
            dropout = spatio_temporal_attention_dropout if group_name == 'spatio_temporal' else attention_dropout_rate
            self.group_attentions[group_name] = nn.MultiheadAttention(
                embed_dim=common_output_dim, num_heads=4, batch_first=True, dropout=dropout
            )

        # --- NEW: Visual Cross-Attention Enhancement ---
        self.visual_cross_attention = nn.MultiheadAttention(embed_dim=common_output_dim, num_heads=4, batch_first=True, dropout=cross_attention_dropout_rate)

        # --- 3. Create Inter-Group Attention Layer (Stage 2) ---
        self.final_attention = nn.MultiheadAttention(embed_dim=common_output_dim, num_heads=8, batch_first=True, dropout=attention_dropout_rate)

        # --- 4. Create Final Fusion Components (with DYNAMIC dimension calculation) ---
        self.msoa_embedding = nn.Embedding(msoa_cardinality, msoa_embedding_dim)
        
        # Dynamically calculate the number of concepts based on what's actually present in this run
        present_groups = [g for g, h_list in self.HEAD_GROUPS.items() if any(h in feature_configs for h in h_list)]
        present_individuals = [h for h in self.INDIVIDUAL_HEADS if h in feature_configs]
        num_top_level_concepts = len(present_groups) + len(present_individuals)
        print(f"  - Dynamically detected {num_top_level_concepts} base specialist concepts for fusion.")

        # Account for the extra concept vector generated by the visual cross-attention mechanism
        if 'visual' in present_groups and (len(present_groups) > 1 or present_individuals):
            print("  - Adding 1 extra concept for visual cross-attention context vector.")
            num_top_level_concepts += 1
        
        final_specialist_dim = num_top_level_concepts * common_output_dim

        self.gating_network = nn.Sequential(nn.Linear(final_specialist_dim, 1), nn.Sigmoid())
        
        fusion_input_dim = final_specialist_dim + common_output_dim + self.spatial_fusion_block.fusion_output_dim
        print(f"--- Hierarchical Fusion: Final MLP input dimension: {fusion_input_dim} ---")
        
        # Dynamically build the final MLP based on the stratum's capacity requirements
        fusion_layers = []
        current_dim = fusion_input_dim
        for h_dim in fusion_mlp_dims:
            fusion_layers.append(nn.Linear(current_dim, h_dim))
            fusion_layers.append(nn.ReLU())
            fusion_layers.append(nn.Dropout(fusion_dropout_rate))
            current_dim = h_dim
        self.fusion_mlp = nn.Sequential(*fusion_layers)

        self.evidential_head = nn.Linear(current_dim, 4) # Outputs: gamma, nu, alpha, beta
        self.mae_head = nn.Linear(current_dim, 1) # Direct prediction head for MAE loss
        
        # --- NEW: Head-Specific Regularization ---
        # Apply a very strong dropout ONLY to the AVM head's output embedding.
        self.avm_head_regularizer = nn.Dropout(0.75)

    def forward(self, x, return_intermediates=False):
        # --- Stage 0: Process all heads to get initial embeddings ---
        head_outputs = {name: module(x[name]) for name, module in self.specialist_heads.items() if name in x}
        
        # --- NEW: Apply targeted regularization to the AVM head's output ---
        if 'head_AVM' in head_outputs:
            head_outputs['head_AVM'] = self.avm_head_regularizer(head_outputs['head_AVM'])

        # --- Stage 0.5: Dedicated Spatial Fusion ---
        msoa_embed = self.msoa_embedding(x['msoa_id'])
        pos_encoding = self.positional_encoder(x['lat_lon'])
        spatial_engineered_output = head_outputs['head_compass_raw']
        hyperlocal_context_vector = self.spatial_fusion_block(msoa_embed, pos_encoding, spatial_engineered_output)

        # --- Stage 1: Intra-Group Attention ---
        group_vectors = {} # Changed to dict for easier access
        for group_name, head_names in self.HEAD_GROUPS.items():
            group_head_outputs = [head_outputs[h_name] for h_name in head_names if h_name in head_outputs]
            if not group_head_outputs: continue
            stacked_group = torch.stack(group_head_outputs, dim=1)
            attn_output, _ = self.group_attentions[group_name](stacked_group, stacked_group, stacked_group)
            group_vectors[group_name] = attn_output.mean(dim=1)

        # --- Stage 1.5: Visual Cross-Attention Enhancement ---
        other_specialist_concepts = []
        for group_name, vec in group_vectors.items():
            if group_name != 'visual':
                other_specialist_concepts.append(vec)
        other_specialist_concepts += [head_outputs[h_name] for h_name in self.INDIVIDUAL_HEADS if h_name in head_outputs]

        top_level_concepts = other_specialist_concepts[:] # Start with a copy of non-visual concepts

        if 'visual' in group_vectors and other_specialist_concepts:
            visual_vector = group_vectors['visual']
            top_level_concepts.append(visual_vector) # Add the original visual vector to the final attention mix

            # Use the visual vector to query the other specialist concepts
            other_specialists_kv = torch.stack(other_specialist_concepts, dim=1)
            visual_query = visual_vector.unsqueeze(1)
            
            # Get the new, visually-informed context vector
            visual_context_vector, _ = self.visual_cross_attention(
                query=visual_query, key=other_specialists_kv, value=other_specialists_kv
            )
            
            # Add this new "view" as another top-level concept to boost the visual signal
            top_level_concepts.append(visual_context_vector.squeeze(1))
        elif 'visual' in group_vectors:
            # Fallback if there are no other specialist heads, just add the visual vector
            top_level_concepts.append(group_vectors['visual'])

        # --- Stage 2: Inter-Group Attention ---
        stacked_top_level = torch.stack(top_level_concepts, dim=1)
        final_attn_output, _ = self.final_attention(stacked_top_level, stacked_top_level, stacked_top_level)
        final_specialist_representation = final_attn_output.reshape(final_attn_output.size(0), -1)

        # --- Stage 3: Final Gated Fusion ---
        base_output = self.base_head(x['head_base'])
        gate_weight = self.gating_network(final_specialist_representation)
        gated_base_output = base_output * gate_weight
        
        final_fused = torch.cat([final_specialist_representation, gated_base_output, hyperlocal_context_vector], dim=1)
        
        final_representation = self.fusion_mlp(final_fused)
        
        # --- Stage 4: Evidential Output Layer ---
        evidential_output = self.evidential_head(final_representation)
        gamma, nu, alpha, beta = torch.split(evidential_output, 1, dim=-1)
        
        # Apply activations to ensure valid NIG parameters
        nu = F.softplus(nu) + 1e-6
        alpha = F.softplus(alpha) + 1.0 # alpha > 1
        beta = F.softplus(beta) + 1e-6
        
        final_evidential_params = torch.cat([gamma, nu, alpha, beta], dim=-1)

        mae_pred = self.mae_head(final_representation)

        if not return_intermediates:
            return final_evidential_params, mae_pred
        else:
            intermediates = {
                'head_base_embedding': base_output,
                'hyperlocal_context_embedding': hyperlocal_context_vector
            }
            # Add all specialist head outputs
            for name, output in head_outputs.items():
                intermediates[f'head_output_{name}'] = output
            
            return (final_evidential_params, mae_pred), intermediates

class AsymmetricEvidentialLoss(nn.Module):
    def __init__(self, coeff=1.0, undervalue_penalty=1.5):
        super(AsymmetricEvidentialLoss, self).__init__()
        self.coeff = coeff
        self.undervalue_penalty = undervalue_penalty

    def forward(self, pred, target):
        gamma, nu, alpha, beta = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        
        # Negative Log-Likelihood (same as before)
        two_beta_lambda = 2 * beta * (1 + nu)
        nll = 0.5 * torch.log(np.pi / nu) \
            - alpha * torch.log(two_beta_lambda) \
            + (alpha + 0.5) * torch.log(nu * (target - gamma)**2 + two_beta_lambda) \
            + torch.lgamma(alpha) \
            - torch.lgamma(alpha + 0.5)
        
        # Asymmetric Error calculation
        error = target - gamma
        # Apply penalty for undervaluation (error > 0 in log space)
        loss_weights = torch.ones_like(error)
        loss_weights[error > 0] = self.undervalue_penalty
        
        # Weighted Regularizer
        reg = torch.abs(error) * (2 * nu + alpha)
        
        # Apply the asymmetry to both the NLL and the regularizer
        loss = loss_weights * (nll + self.coeff * reg)
        return loss.mean()

def calculate_evidential_uncertainty(nig_params):
    """Calculates aleatoric and epistemic uncertainty from NIG parameters."""
    gamma, nu, alpha, beta = nig_params[:, 0], nig_params[:, 1], nig_params[:, 2], nig_params[:, 3]
    
    # Ensure alpha > 1 for variance calculation
    alpha_safe = torch.clamp(alpha, min=1.0001)
    
    aleatoric = beta / (alpha_safe - 1)
    epistemic = beta / (nu * (alpha_safe - 1))
    
    return aleatoric, epistemic


# --- New TabNet Implementation ---
# Ghost Batch Normalization
class GBN(nn.Module):
    def __init__(self, input_dim, virtual_batch_size=128, momentum=0.01):
        super(GBN, self).__init__()
        self.input_dim = input_dim
        self.virtual_batch_size = virtual_batch_size
        self.bn = nn.BatchNorm1d(self.input_dim, momentum=momentum)

    def forward(self, x):
        chunks = x.chunk(int(np.ceil(x.shape[0] / self.virtual_batch_size)), 0)
        res = [self.bn(chunk) for chunk in chunks]
        return torch.cat(res, dim=0)

def sparsemax(z, dim=-1):
    # Sort z in descending order
    z_sorted, _ = torch.sort(z, dim=dim, descending=True)
    # Calculate the cumulative sum of sorted z
    z_cumsum = torch.cumsum(z_sorted, dim=dim)
    # Create a range tensor [1, 2, ..., N]
    k = torch.arange(1, z.size(dim) + 1, device=z.device).view(1, -1)
    # Find the largest k such that 1 + k * z_k > sum(z_i for i=1 to k)
    z_check = 1 + k * z_sorted > z_cumsum
    # Get the k_max (the number of non-zero probabilities)
    k_max = torch.sum(z_check, dim=dim, keepdim=True)
    # Get the threshold tau
    k_max_values = k_max.long() - 1
    tau = (torch.gather(z_cumsum, dim, k_max_values) - 1) / k_max.float()
    # Apply the sparsemax transformation
    return torch.max(torch.zeros_like(z), z - tau)

# Attentive Transformer for feature selection
class AttentiveTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, momentum=0.02, virtual_batch_size=128):
        super(AttentiveTransformer, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim, bias=False)
        self.bn = GBN(output_dim, virtual_batch_size=virtual_batch_size, momentum=momentum)

    def forward(self, a, priors):
        a = self.fc(a)
        a = self.bn(a)
        mask = a * priors
        return sparsemax(mask, dim=-1)

# Feature Transformer for processing selected features
class FeatureTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, n_shared, n_independent, momentum=0.02, virtual_batch_size=128):
        super(FeatureTransformer, self).__init__()
        self.n_independent = n_independent
        self.shared_layers = nn.ModuleList()
        self.independent_layers = nn.ModuleList()

        if n_shared > 0:
            shared_galu = nn.Linear(input_dim, 2 * output_dim, bias=False)
            self.shared_layers.append(shared_galu)
            self.shared_layers.append(GBN(2 * output_dim, virtual_batch_size, momentum))

        for _ in range(self.n_independent):
            ind_galu = nn.Linear(input_dim, 2 * output_dim, bias=False)
            self.independent_layers.append(nn.ModuleList([ind_galu, GBN(2 * output_dim, virtual_batch_size, momentum)]))

    def forward(self, x):
        shared_output = x
        if self.shared_layers:
            shared_output = self.shared_layers[0](shared_output)
            shared_output = self.shared_layers[1](shared_output)
            shared_output = F.glu(shared_output)
        
        outputs = []
        for i in range(self.n_independent):
            ind_out = self.independent_layers[i][0](x)
            ind_out = self.independent_layers[i][1](ind_out)
            ind_out = F.glu(ind_out)
            outputs.append(ind_out)
        
        # Aggregate shared and independent outputs
        return torch.stack([shared_output] + outputs, dim=1)

# Main TabNet Head Module
class TabNetHead(nn.Module):
    def __init__(self, input_dim, output_dim, n_d=32, n_a=32, n_steps=5, gamma=1.3, n_independent=2, n_shared=2, dropout=0.2):
        super(TabNetHead, self).__init__()
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.dropout = dropout

        self.initial_bn = nn.BatchNorm1d(input_dim, momentum=0.01)
        self.feature_transformer = FeatureTransformer(input_dim, n_d + n_a, n_shared, n_independent)
        self.attentive_transformers = nn.ModuleList()
        for _ in range(n_steps):
            self.attentive_transformers.append(AttentiveTransformer(n_a, input_dim))
        self.final_map = nn.Linear(n_d, output_dim)

    def forward(self, x):
        x = self.initial_bn(x)
        priors = torch.ones(x.shape).to(x.device)
        total_output = 0.
        
        # Split initial features into decision and attention parts
        x_transformed = self.feature_transformer(x)
        x_a = x_transformed[:, :, self.n_d:]
        
        for step in range(self.n_steps):
            mask = self.attentive_transformers[step](x_a.sum(1), priors)
            priors = priors * (self.gamma - mask)
            
            # Apply mask to original features and transform
            masked_x = mask * x
            x_step_transformed = self.feature_transformer(masked_x)
            
            # Get decision part for this step
            d_step = F.relu(x_step_transformed[:, :, :self.n_d].sum(1))
            total_output += d_step
            
            # Update attention part for next step
            x_a = x_step_transformed[:, :, self.n_d:]
            
        # Final projection to common output dim
        return self.final_map(total_output)


class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim_1, hidden_dim_2):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1), nn.ReLU(),
            nn.Linear(hidden_dim_1, hidden_dim_2), nn.ReLU(),
            nn.Linear(hidden_dim_2, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim_2), nn.ReLU(),
            nn.Linear(hidden_dim_2, hidden_dim_1), nn.ReLU(),
            nn.Linear(hidden_dim_1, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def encode(self, x):
        return self.encoder(x)


def train_one_epoch(model, loader, optimizer, loss_fn):
    model.train(); total_loss = 0
    for batch in loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        optimizer.zero_grad()
        evidential_preds, mae_pred = model(batch)
        evidential_loss = loss_fn(evidential_preds, batch['target_price'].squeeze(-1))
        mae_loss = F.l1_loss(mae_pred, batch['target_price'])
        
        # Give equal weight to both objectives, can be tuned
        loss = 0.5 * evidential_loss + 0.5 * mae_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, return_intermediates=False):
    model.eval()
    all_preds, all_intermediates = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            if return_intermediates:
                # model() now returns ((evidential_params, mae_pred), intermediates)
                (evidential_params, _), intermediates = model(batch, return_intermediates=True)
                all_preds.append(evidential_params.cpu())
                # Detach and move intermediate tensors to CPU
                cpu_intermediates = {k: v.cpu() for k, v in intermediates.items()}
                all_intermediates.append(cpu_intermediates)
            else:
                # model() now returns (evidential_params, mae_pred)
                evidential_params, _ = model(batch, return_intermediates=False)
                all_preds.append(evidential_params.cpu())

    if return_intermediates:
        # Stitch together the dictionaries of intermediates from each batch
        if not all_intermediates: return torch.cat(all_preds), {}
        keys = all_intermediates[0].keys()
        stitched_intermediates = {k: torch.cat([d[k] for d in all_intermediates], dim=0) for k in keys}
        return torch.cat(all_preds), stitched_intermediates
    else:
        return torch.cat(all_preds)

def find_msoa_column(df):
    """Finds the best MSOA column in the dataframe, prioritizing official codes."""
    all_cols = df.columns.str.lower()
    
    # Priority 1: Official MSOA codes (e.g., MSOA11CD, MSOA21CD)
    official_codes = [c for c in df.columns if c.upper() in ['MSOA11CD', 'MSOA21CD']]
    if official_codes:
        print(f"Found official MSOA code column: {official_codes[0]}")
        return official_codes[0]

    # Priority 2: Columns containing 'msoa' and 'cd' or 'code'
    code_candidates = [c for c in df.columns if 'msoa' in c.lower() and ('cd' in c.lower() or 'code' in c.lower())]
    if code_candidates:
        print(f"Found high-priority MSOA code candidate: {code_candidates[0]}")
        return code_candidates[0]

    # Priority 3: Fallback to any column with 'msoa'
    msoa_candidates = [col for col in df.columns if 'msoa' in col.lower()]
    if msoa_candidates:
        print(f"Found MSOA candidates (fallback): {msoa_candidates}")
        print(f"  - WARNING: Using fallback MSOA column: {msoa_candidates[0]}. Please verify this is the correct categorical MSOA identifier.")
        return msoa_candidates[0]
    
    # Final Fallback: Create a dummy column
    print("WARNING: No MSOA column found. Creating dummy MSOA IDs.")
    df['dummy_msoa'] = pd.factorize(df.index % 100)[0]
    return 'dummy_msoa'

def prepare_tcn_data(df, tcn_cols, fit_scaler=True, scaler_obj=None):
    """
    Parses column names to prepare data for a TCN head.
    Can either fit a new scaler or apply an existing one.
    """
    if df.empty:
        return pd.DataFrame(), {'n_timesteps': 1, 'n_features_per_timestep': 1}
    if not tcn_cols:
        return pd.DataFrame(np.zeros((len(df), 1)), index=df.index), {'n_timesteps': 1, 'n_features_per_timestep': 1}

    parsed_cols = {}
    spatio_pattern = re.compile(r"^(.*)_(\d{4})_(.*)$")
    temporal_pattern = re.compile(r"^(\d{4})_(.*)$")
    hpi_pattern = re.compile(r"^(HPI_Adjusted_Median_Price|Sale_Count)_(D|S|T|F)$")

    for col in tcn_cols:
        year, feature_stem = None, None
        match_spatio = spatio_pattern.match(col)
        match_temporal = temporal_pattern.match(col)
        match_hpi = hpi_pattern.match(col)

        if match_spatio:
            prefix, year, suffix = match_spatio.groups(); feature_stem = f"{prefix}_{suffix}"
        elif match_temporal:
            year, feature_stem = match_temporal.groups()
        elif match_hpi:
            year, _ = "all", match_hpi.groups(); feature_stem = col
        
        if feature_stem:
            if feature_stem not in parsed_cols: parsed_cols[feature_stem] = {}
            if year != "all": parsed_cols[feature_stem][year] = col
            else: parsed_cols[feature_stem]['all_years_col'] = col

    feature_stems = sorted(list(parsed_cols.keys()))
    all_years = sorted(list(set(y for f in parsed_cols.values() for y in f.keys() if y != 'all_years_col')))
    n_timesteps = len(all_years) if all_years else 1
    n_features = len(feature_stems) if feature_stems else 1

    tcn_array = np.zeros((len(df), n_timesteps, n_features))
    for i, stem in enumerate(feature_stems):
        if 'all_years_col' in parsed_cols[stem]:
            col_name = parsed_cols[stem]['all_years_col']
            if col_name in df.columns:
                tcn_array[:, :, i] = df[col_name].fillna(0).values[:, np.newaxis]
        else:
            for j, year in enumerate(all_years):
                col_name = parsed_cols[stem].get(year)
                if col_name and col_name in df.columns:
                    tcn_array[:, j, i] = df[col_name].fillna(0).values

    tcn_flat = tcn_array.reshape(len(df), -1)
    
    if fit_scaler:
        scaler = QuantileTransformer(output_distribution='normal', n_quantiles=max(min(len(df) // 10, 100), 10), random_state=42)
        tcn_scaled = scaler.fit_transform(tcn_flat)
        config = {'n_timesteps': n_timesteps, 'n_features_per_timestep': n_features, 'scaler': scaler}
    else:
        if scaler_obj is None: raise ValueError("scaler_obj must be provided when fit_scaler is False.")
        tcn_scaled = scaler_obj.transform(tcn_flat)
        config = {'n_timesteps': n_timesteps, 'n_features_per_timestep': n_features}

    return pd.DataFrame(tcn_scaled, index=df.index), config



def prepare_pp_history_for_tcn(df, pp_cols, fit_scaler=True, scaler_obj=None):
    """
    Parses Price Paid history columns (pp_YYYY_MM_TYPE...) and prepares them for a TCN.
    """
    if df.empty:
        return pd.DataFrame(), {'n_timesteps': 1, 'n_features_per_timestep': 1}
    if not pp_cols:
        return pd.DataFrame(np.zeros((len(df), 1)), index=df.index), {'n_timesteps': 1, 'n_features_per_timestep': 1}

    parsed_data = {}
    pattern = re.compile(r"^pp_(\d{4})_(\d{2})_([DSTF])_avg_price$")
    
    for col in pp_cols:
        match = pattern.match(col)
        if match:
            year, month, prop_type = match.groups()
            timestep = (int(year), int(month))
            if timestep not in parsed_data:
                parsed_data[timestep] = {}
            parsed_data[timestep][prop_type] = col

    timesteps = sorted(parsed_data.keys())
    prop_types = ['D', 'S', 'T', 'F']
    n_timesteps = len(timesteps)
    n_features_per_timestep = len(prop_types)

    tcn_array = np.zeros((len(df), n_timesteps, n_features_per_timestep))

    for j, ts in enumerate(timesteps):
        for i, p_type in enumerate(prop_types):
            col_name = parsed_data.get(ts, {}).get(p_type)
            if col_name and col_name in df.columns:
                tcn_array[:, j, i] = df[col_name].fillna(-1.0).values

    tcn_flat = tcn_array.reshape(len(df), -1)
    
    if fit_scaler:
        scaler = QuantileTransformer(output_distribution='normal', n_quantiles=max(min(len(df) // 10, 100), 10), random_state=42)
        tcn_scaled = scaler.fit_transform(tcn_flat)
        config = {'n_timesteps': n_timesteps, 'n_features_per_timestep': n_features_per_timestep, 'scaler': scaler}
    else:
        if scaler_obj is None: raise ValueError("scaler_obj must be provided when fit_scaler is False.")
        tcn_scaled = scaler_obj.transform(tcn_flat)
        config = {'n_timesteps': n_timesteps, 'n_features_per_timestep': n_features_per_timestep}

    return pd.DataFrame(tcn_scaled, index=df.index), config


def prepare_compass_pp_for_tcn(df, compass_pp_cols, fit_scaler=True, scaler_obj=None):
    """
    Parses Compass Price Paid history columns (compass_STAT_pp_YYYY_MM_TYPE_..._nK) for TCN.
    """
    if df.empty:
        return pd.DataFrame(), {'n_timesteps': 1, 'n_features_per_timestep': 1}
    if not compass_pp_cols:
        return pd.DataFrame(np.zeros((len(df), 1)), index=df.index), {'n_timesteps': 1, 'n_features_per_timestep': 1}

    parsed_data = {}
    feature_stems = set()
    pattern = re.compile(r"^compass_(mean|std)_pp_(\d{4})_(\d{2})_([DSTF])_avg_price_n(\d+)$")

    for col in compass_pp_cols:
        match = pattern.match(col)
        if match:
            stat, year, month, prop_type, n_neighbors = match.groups()
            timestep = (int(year), int(month))
            feature_stem = f"{stat}_{prop_type}_n{n_neighbors}"
            feature_stems.add(feature_stem)
            
            if timestep not in parsed_data:
                parsed_data[timestep] = {}
            parsed_data[timestep][feature_stem] = col

    timesteps = sorted(parsed_data.keys())
    canonical_features = sorted(list(feature_stems))
    n_timesteps = len(timesteps)
    n_features_per_timestep = len(canonical_features)

    tcn_array = np.zeros((len(df), n_timesteps, n_features_per_timestep))

    for j, ts in enumerate(timesteps):
        for i, stem in enumerate(canonical_features):
            col_name = parsed_data.get(ts, {}).get(stem)
            if col_name and col_name in df.columns:
                tcn_array[:, j, i] = df[col_name].fillna(-1.0).values

    tcn_flat = tcn_array.reshape(len(df), -1)

    if fit_scaler:
        scaler = QuantileTransformer(output_distribution='normal', n_quantiles=max(min(len(df) // 10, 100), 10), random_state=42)
        tcn_scaled = scaler.fit_transform(tcn_flat)
        config = {'n_timesteps': n_timesteps, 'n_features_per_timestep': n_features_per_timestep, 'scaler': scaler}
    else:
        if scaler_obj is None: raise ValueError("scaler_obj must be provided when fit_scaler is False.")
        tcn_scaled = scaler_obj.transform(tcn_flat)
        config = {'n_timesteps': n_timesteps, 'n_features_per_timestep': n_features_per_timestep}

    return pd.DataFrame(tcn_scaled, index=df.index), config

def preprocess_fold_data(df_train, df_val, y_train_log, feature_sets, universal_cols_present, LAT_LON_COLS, N_TOP_FEATURES_PER_HEAD, AE_BEST_PARAMS, return_fitters=False):
    """
    Preprocesses training and validation data for a single fold to prevent data leakage.
    All fitting (scalers, selectors, encoders) is done ONLY on df_train.
    If return_fitters is True, it returns the fitted objects instead of the transformed validation data.
    """
    train_data_for_model, val_data_for_model = {}, {}
    feature_configs = {}

    # --- Process data that doesn't need fitting (or is simple) ---
    train_data_for_model['lat_lon'] = df_train[LAT_LON_COLS].copy().apply(pd.to_numeric, errors='coerce').fillna(0)
    if not df_val.empty:
        val_data_for_model['lat_lon'] = df_val[LAT_LON_COLS].copy().apply(pd.to_numeric, errors='coerce').fillna(0)

    # --- Process `head_base` ---
    base_df_train_raw = df_train[universal_cols_present].copy().apply(pd.to_numeric, errors='coerce').fillna(0)
    scaler_base = QuantileTransformer(output_distribution='normal', n_quantiles=max(min(len(df_train) // 10, 100), 10), random_state=42)
    train_data_for_model['head_base'] = pd.DataFrame(scaler_base.fit_transform(base_df_train_raw), columns=base_df_train_raw.columns, index=df_train.index)
    if not df_val.empty:
        base_df_val_raw = df_val[universal_cols_present].copy().apply(pd.to_numeric, errors='coerce').fillna(0)
        val_data_for_model['head_base'] = pd.DataFrame(scaler_base.transform(base_df_val_raw), columns=base_df_val_raw.columns, index=df_val.index)
    feature_configs['head_base'] = {'input_dim': len(universal_cols_present)}
    if return_fitters:
        feature_configs['head_base']['scaler'] = scaler_base
        feature_configs['head_base']['selected_features'] = universal_cols_present

    # --- Process Specialist Heads ---
    for head_name, cols in feature_sets.items():
        if head_name in ['unassigned_features', 'head_base', 'head_atlas', 'head_compass', 'head_microscope'] or not cols: continue

        available_cols = [c for c in cols if c in df_train.columns and c not in universal_cols_present]
        if not available_cols: continue

        # --- ARCHITECTURAL ROUTING: TCN, TabNet, or Supervised MLP ---
        if 'spatio_temporal' in head_name or 'head_E_temporal' in head_name:
            print(f"    - Preparing standard TCN data for '{head_name}'...")
            train_tcn_data, tcn_config = prepare_tcn_data(df_train, available_cols, fit_scaler=True)
            train_data_for_model[head_name] = train_tcn_data
            if not df_val.empty:
                val_tcn_data, _ = prepare_tcn_data(df_val, available_cols, fit_scaler=False, scaler_obj=tcn_config.get('scaler'))
                val_data_for_model[head_name] = val_tcn_data
            feature_configs[head_name] = {**tcn_config, 'input_dim': train_tcn_data.shape[1]}
            if return_fitters:
                feature_configs[head_name]['raw_features'] = available_cols
        elif head_name.startswith('head_H_'):
            print(f"    - Preparing TCN data for Raw Price Paid History head: {head_name}...")
            train_tcn_data, tcn_config = prepare_pp_history_for_tcn(df_train, available_cols, fit_scaler=True)
            if not df_val.empty:
                val_tcn_data, _ = prepare_pp_history_for_tcn(df_val, available_cols, fit_scaler=False, scaler_obj=tcn_config.get('scaler'))
                val_data_for_model[head_name] = val_tcn_data
            train_data_for_model[head_name] = train_tcn_data
            feature_configs[head_name] = {**tcn_config, 'input_dim': train_tcn_data.shape[1]}
            if return_fitters:
                feature_configs[head_name]['raw_features'] = available_cols
        elif head_name.startswith('head_I_'):
            print(f"    - Preparing TCN data for Compass Price Paid History head: {head_name}...")
            train_tcn_data, tcn_config = prepare_compass_pp_for_tcn(df_train, available_cols, fit_scaler=True)
            if not df_val.empty:
                val_tcn_data, _ = prepare_compass_pp_for_tcn(df_val, available_cols, fit_scaler=False, scaler_obj=tcn_config.get('scaler'))
                val_data_for_model[head_name] = val_tcn_data
            train_data_for_model[head_name] = train_tcn_data
            feature_configs[head_name] = {**tcn_config, 'input_dim': train_tcn_data.shape[1]}
            if return_fitters:
                feature_configs[head_name]['raw_features'] = available_cols
        elif head_name == 'head_compass_raw':
            print(f"    - [TabNet] Preparing raw heterogeneous data for '{head_name}'...")
            head_df_train_raw = df_train[available_cols].copy().apply(pd.to_numeric, errors='coerce').fillna(0).replace([np.inf, -np.inf], 0)
            scaler = QuantileTransformer(output_distribution='normal', n_quantiles=max(min(len(df_train) // 10, 100), 10), random_state=42)
            train_data_for_model[head_name] = pd.DataFrame(scaler.fit_transform(head_df_train_raw), columns=available_cols, index=df_train.index)
            if not df_val.empty:
                head_df_val_raw = df_val[available_cols].copy().apply(pd.to_numeric, errors='coerce').fillna(0).replace([np.inf, -np.inf], 0)
                val_data_for_model[head_name] = pd.DataFrame(scaler.transform(head_df_val_raw), columns=available_cols, index=df_val.index)
            
            feature_configs[head_name] = {'input_dim': len(available_cols), 'is_tabnet': True} # Flag for model constructor
            if return_fitters:
                feature_configs[head_name]['scaler'] = scaler
                feature_configs[head_name]['selected_features'] = available_cols

        else: # Standard MLP Head with Supervised Feature Selection
            head_df_train_raw = df_train[available_cols].copy().apply(pd.to_numeric, errors='coerce').fillna(0).replace([np.inf, -np.inf], 0)
            
            final_head_train_df = head_df_train_raw
            selected_cols = available_cols

            N_TOP = N_TOP_FEATURES_PER_HEAD.get(head_name, N_TOP_FEATURES_PER_HEAD["DEFAULT"])
            if len(available_cols) > N_TOP:
                print(f"    - [MLP] Applying Supervised Pre-selection for '{head_name}': {len(available_cols)} -> {N_TOP} features")
                lgbm_selector = lgb.LGBMRegressor(random_state=42, n_estimators=100, n_jobs=-1, verbose=-1)
                lgbm_selector.fit(head_df_train_raw, y_train_log)
                importances = pd.DataFrame({'feature': head_df_train_raw.columns, 'importance': lgbm_selector.feature_importances_}).sort_values('importance', ascending=False)
                selected_cols = importances.head(N_TOP)['feature'].tolist()
                final_head_train_df = head_df_train_raw[selected_cols]
            else:
                print(f"    - [MLP] Using all {len(available_cols)} features for '{head_name}' (below threshold).")

            scaler = QuantileTransformer(output_distribution='normal', n_quantiles=max(min(len(df_train) // 10, 100), 10), random_state=42)
            train_data_for_model[head_name] = pd.DataFrame(scaler.fit_transform(final_head_train_df), columns=selected_cols, index=df_train.index)
            
            if not df_val.empty:
                head_df_val_raw = df_val[selected_cols].copy().apply(pd.to_numeric, errors='coerce').fillna(0).replace([np.inf, -np.inf], 0)
                val_data_for_model[head_name] = pd.DataFrame(scaler.transform(head_df_val_raw), columns=selected_cols, index=df_val.index)
            
            feature_configs[head_name] = {'input_dim': len(selected_cols)}
            if return_fitters:
                feature_configs[head_name]['scaler'] = scaler
                feature_configs[head_name]['selected_features'] = selected_cols

    if return_fitters:
        return train_data_for_model, None, feature_configs

    return train_data_for_model, val_data_for_model, feature_configs


def validate_merge_quality(df, sample_size=10):
    """Validate that merged properties actually match by comparing original addresses."""
    print(f"\n--- MERGE VALIDATION (Sample of {sample_size}) ---")
    address_col_rm = [col for col in df.columns if 'address' in col.lower()][0] # Find the rightmove address col
    
    def normalize_for_comparison(s):
        """Canonicalizes an address string by lowercasing and removing non-alphanumerics."""
        return re.sub(r'[^a-z0-9]', '', str(s).lower())

    sample_df = df.sample(min(sample_size, len(df)))

    for i, (_, row) in enumerate(sample_df.iterrows()):
        feature_addr = row.get('property_id', 'N/A')
        rightmove_addr = row.get(address_col_rm, 'N/A')
        merge_key = row.get('final_merge_key', 'N/A')
        price = row.get('most_recent_sale_price', 0)

        # Use fuzzy matching on the CANONICALIZED addresses
        similarity_score = fuzz.token_set_ratio(
            normalize_for_comparison(feature_addr),
            normalize_for_comparison(rightmove_addr)
        )

        print(f"--- Record {i+1} / Price: £{price:,.0f} ---")
        print(f"  Merge Key   : '{merge_key}'")
        print(f"  Similarity  : {similarity_score}%")
        print(f"  Features Addr: {str(feature_addr)[:80]}")
        print(f"  Rightmove Addr: {str(rightmove_addr)[:80]}")
        if similarity_score < 70:
            print("  [!! WARNING: LOW SIMILARITY. THIS IS LIKELY A BAD MATCH !!]")
        print()




def objective(trial, data_train, y_train, data_val, y_val, feature_configs, msoa_cardinality, y_price_std, y_price_mean, df_val):
    """
    The objective function for Optuna for the single, monolithic model.
    """
    # --- 1. Define Hyperparameter Search Space ---
    print(f"\n--- Starting Trial {trial.number} ---")
    
    trial_params = {
        # Core training params
        'learning_rate': trial.suggest_float('learning_rate', 3e-4, 1e-3, log=True), # Lowered max LR
        'weight_decay': trial.suggest_float('weight_decay', 1e-3, 1e-1, log=True), # Increased regularization
        'batch_size': trial.suggest_categorical('batch_size', [512, 1024]), # Larger batches for stability

        # Loss function params
        'evidential_reg_coeff': trial.suggest_float('evidential_reg_coeff', 0.05, 0.2), # Tighter range
        'undervalue_penalty': trial.suggest_float('undervalue_penalty', 1.1, 1.5), # Reduced penalty range

        # Embedding and Fusion dimensions (Capacity Reduction)
        'msoa_embedding_dim': trial.suggest_categorical('msoa_embedding_dim', [8, 16]), # Reduced capacity
        'fusion_embed_dim': trial.suggest_categorical('fusion_embed_dim', [64, 128]), # Reduced capacity
        
        # Head dimensions (Capacity Reduction)
        'large_head_hidden_dim': trial.suggest_categorical('large_head_hidden_dim', [128, 256]), # Reduced capacity
        'small_head_hidden_dim': trial.suggest_categorical('small_head_hidden_dim', [64, 128]), # Reduced capacity

        # Dropout rates (Regularization Increase)
        'fusion_dropout_rate': trial.suggest_float('fusion_dropout_rate', 0.25, 0.5), # Tighter, lower range
        'large_head_dropout': trial.suggest_float('large_head_dropout', 0.2, 0.4), # Tighter, lower range
        'small_head_dropout': trial.suggest_float('small_head_dropout', 0.2, 0.4), # Tighter, lower range
        'visual_head_dropout': trial.suggest_float('visual_head_dropout', 0.3, 0.5), # Tighter, lower range
        'tcn_dropout': trial.suggest_float('tcn_dropout', 0.2, 0.4), # Tighter, lower range

        # Attention dropout rates (Regularization Increase)
        'cross_attention_dropout': trial.suggest_float('cross_attention_dropout', 0.1, 0.25), # Tighter range
        'attention_dropout': trial.suggest_float('attention_dropout', 0.1, 0.2), # Tighter range
        'spatio_temporal_attention_dropout': trial.suggest_float('spatio_temporal_attention_dropout', 0.2, 0.4), # Tighter range
    }

    head_params = create_head_params(trial_params, feature_configs)

    train_dataset = WisteriaDataset(data_train, y_train, y_train)
    val_dataset = WisteriaDataset(data_val, y_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=trial_params['batch_size'], shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=trial_params['batch_size'] * 2, shuffle=False, num_workers=2, pin_memory=True)

    model = FusionModel(
        feature_configs=feature_configs, msoa_cardinality=msoa_cardinality, head_params=head_params,
        fusion_embed_dim=trial_params['fusion_embed_dim'], msoa_embedding_dim=trial_params['msoa_embedding_dim'],
        fusion_dropout_rate=trial_params['fusion_dropout_rate'], use_enhanced_tcn=True, fusion_input_cap=2048,
        cross_attention_dropout_rate=trial_params['cross_attention_dropout'],
        attention_dropout_rate=trial_params['attention_dropout'],
        stratum_name='monolith' # Use a fixed name as it's no longer stratified
    ).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=trial_params['learning_rate'], weight_decay=trial_params['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=5)
    loss_fn = AsymmetricEvidentialLoss(
        coeff=trial_params['evidential_reg_coeff'],
        undervalue_penalty=trial_params['undervalue_penalty']
    )

    best_mape = float('inf')
    patience_counter = 0
    patience = 15

    for epoch in range(EPOCHS):
        _ = train_one_epoch(model, train_loader, optimizer, loss_fn)
        
        evidential_preds = evaluate(model, val_loader)
        point_preds_scaled = evidential_preds.numpy()[:, 0]
        point_preds_unscaled = np.expm1(point_preds_scaled * y_price_std + y_price_mean)
        true_values = df_val['most_recent_sale_price']
        val_mape = mean_absolute_percentage_error(true_values, point_preds_unscaled)
        
        scheduler.step(val_mape)

        if val_mape < best_mape:
            best_mape = val_mape
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(f"  Trial {trial.number}, Epoch {epoch+1}, Val MAPE: {val_mape:.2f}% (Best: {best_mape:.2f}%)")

        if patience_counter >= patience:
            print(f"  - Early stopping trial {trial.number} after {epoch+1} epochs.")
            break

        trial.report(val_mape, epoch)
        if trial.should_prune():
            del model, optimizer, scheduler; gc.collect(); torch.cuda.empty_cache()
            raise optuna.exceptions.TrialPruned()

    del model, optimizer, scheduler; gc.collect(); torch.cuda.empty_cache()
    return best_mape


def find_best_matches_fuzzy(keys1, keys2, threshold=85):
    """Finds the best fuzzy match for each key in keys1 from keys2."""
    print(f"  - Attempting to find fuzzy matches for {len(keys1)} keys...")
    matches = []
    keys2_list = list(keys2) # Convert set to list for fuzzywuzzy
    
    for key1 in list(keys1):
        # extractOne returns (match, score)
        best_match, score = process.extractOne(key1, keys2_list)
        if score >= threshold:
            matches.append((key1, best_match, score))
    
    print(f"  - Found {len(matches)} potential fuzzy matches with score >= {threshold}")
    return matches


def mitigate_avm_leakage(df, label_col, correlation_threshold=0.995, noise_level=0.05):
    """
    Detects leakage by checking for near-perfect correlation between scaled AVM features and a scaled label.

    Args:
        df (pd.DataFrame): The dataframe, which will be modified.
        label_col (str): The raw, unscaled target variable column (e.g., 'most_recent_sale_price').
        correlation_threshold (float): The Pearson correlation coefficient above which a feature is flagged.
        noise_level (float): The standard deviation of the noise to add, as a fraction of the feature's std dev.

    Returns:
        pd.DataFrame: The dataframe with noise added to leaky columns.
    """
    print("\n--- STAGE 4.1: Mitigating AVM Target Leakage (Correlation Check) ---")
    avm_keywords = ['estimate', 'valuation', 'avm', 'homipi', 'zoopla', 'bnl', 'bricks', 'chimnie']
    mitigated_cols = []
    df_copy = df.copy()

    if label_col not in df_copy.columns:
        print(f"  - FATAL: Label column '{label_col}' not found. Cannot perform leakage check.")
        return df

    # 1. Scale the raw label to create a comparable series
    label_raw = pd.to_numeric(df_copy[label_col], errors='coerce').dropna()
    scaler = StandardScaler()
    label_scaled = pd.Series(scaler.fit_transform(label_raw.values.reshape(-1, 1)).flatten(), index=label_raw.index)

    # 2. Iterate through columns to find potential AVM features
    for col in df_copy.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in avm_keywords):
            print(f"  - Analyzing potential AVM feature: '{col}'")
            if not pd.api.types.is_numeric_dtype(df_copy[col]):
                print(f"    - Skipping non-numeric column.")
                continue

            # 3. Calculate correlation between the scaled feature and the scaled label
            #    We align them by index to handle any potential NaNs correctly.
            temp_df = pd.concat([df_copy[col], label_scaled], axis=1, keys=['feature', 'label']).dropna()
            if len(temp_df) < 2: continue # Not enough data to calculate correlation
            
            correlation = temp_df['feature'].corr(temp_df['label'])

            if pd.isna(correlation): continue

            # 4. Check if correlation exceeds the threshold
            if abs(correlation) > correlation_threshold:
                print(f"  [!! LEAKAGE DETECTED !!] Column '{col}' has a correlation of {correlation:.4f} with the scaled label.")
                col_std = df_copy[col].std()
                if pd.notna(col_std) and col_std > 0:
                    noise = np.random.normal(0, noise_level * col_std, size=len(df_copy))
                    df_copy.loc[df_copy[col].notna(), col] += noise
                    print(f"  - Added noise (std dev: {noise_level * col_std:.4f}) to column '{col}'.")
                    mitigated_cols.append(col)
                else:
                    print(f"  - Could not add noise to column '{col}' due to zero or NaN standard deviation.")
            else:
                print(f"  - Column '{col}' passed leakage check (Correlation: {correlation:.4f}).")

    if not mitigated_cols:
        print("  - No significant correlation leakage detected.")

    return df_copy

def engineer_temporal_summary_features(df, feature_sets):
    """
    Performs point-in-time correct longitudinal compression.
    For each property and temporal feature, it calculates trend, mean, and std dev
    using only data available up to the property's sale year.
    """
    print("\n--- STAGE 4.1.5: Point-in-Time Temporal Summary Feature Engineering ---")
    
    if 'most_recent_sale_year' not in df.columns:
        print("  - WARNING: `most_recent_sale_year` not found. Skipping temporal summary feature engineering.")
        return df, []
        
    temporal_cols = []
    for head_name, cols in feature_sets.items():
        if 'temporal' in head_name:
            temporal_cols.extend(cols)
    temporal_cols = sorted(list(set([c for c in temporal_cols if c in df.columns])))
    
    if not temporal_cols:
        print("  - No temporal columns found to engineer. Skipping.")
        return df, []

    feature_stems = {}
    pattern = re.compile(r"(\d{4})_(.+)") 
    spatio_pattern = re.compile(r"^(.*?)_(\d{4})_(.*)$")
    for col in temporal_cols:
        year, stem = None, None
        match = pattern.match(col)
        if match:
            year, stem = match.groups()
        else:
            spatio_match = spatio_pattern.match(col)
            if spatio_match:
                prefix, year, suffix = spatio_match.groups()
                stem = f"{prefix}_{suffix}"
        if year and stem:
            if stem not in feature_stems: feature_stems[stem] = []
            feature_stems[stem].append({'year': int(year), 'col': col})

    new_feature_names = set()
    all_new_features = []

    print(f"  - Found {len(feature_stems)} unique temporal feature stems to process.")
    for i, (stem, year_cols) in enumerate(feature_stems.items()):
        if (i+1) % 50 == 0: print(f"    - Processing stem {i+1}/{len(feature_stems)}: {stem}")
        
        # Create a df of this feature's history: columns are years, rows are properties
        history_df = pd.DataFrame({info['year']: df[info['col']] for info in year_cols}).sort_index(axis=1)
        
        # Prepare arrays for calculations
        trends = np.zeros(len(df))
        means = np.zeros(len(df))
        stds = np.zeros(len(df))
        
        # Iterate through each property (row) to perform a point-in-time correct calculation
        for j, (idx, row) in enumerate(df.iterrows()):
            sale_year = row['most_recent_sale_year']
            
            # Select only columns (years) up to the sale year
            available_years = [y for y in history_df.columns if y <= sale_year]
            if len(available_years) < 2: continue # Not enough data for stats

            # Get the historical values for this specific property
            property_history = history_df.loc[idx, available_years].values.flatten()
            years_for_fit = np.array(available_years).flatten()
            
            # Remove NaNs for robust calculations
            finite_mask = np.isfinite(property_history)
            property_history = property_history[finite_mask]
            years_for_fit = years_for_fit[finite_mask]
            
            if len(property_history) < 2: continue

            # Calculate stats using only the available past data
            with warnings.catch_warnings():
                # The simplefilter for the removed np.RankWarning was here.
                trends[j] = np.polyfit(years_for_fit, property_history, 1)[0]
            means[j] = np.mean(property_history)
            stds[j] = np.std(property_history)

        # Add to our new features dataframe
        trend_col_name = f"ts_summary_trend_{stem}"
        mean_col_name = f"ts_summary_mean_{stem}"
        std_col_name = f"ts_summary_std_{stem}"
        
        df[trend_col_name] = trends
        df[mean_col_name] = means
        df[std_col_name] = stds
        
        new_feature_names.update([trend_col_name, mean_col_name, std_col_name])

    print(f"  - Successfully created {len(new_feature_names)} new point-in-time correct temporal summary features.")
    
    return df, list(new_feature_names)

def generate_final_report(eval_df, holdout_results_df, baseline_mape, output_dir):
    """
    Generates a comprehensive report with text and plots to prove model effectiveness.
    """
    import matplotlib
    matplotlib.use('Agg') # Use non-interactive backend for server-side execution
    import matplotlib.pyplot as plt
    import seaborn as sns
    print("\n--- GENERATING FINAL PERFORMANCE & ANALYSIS REPORT ---")
    report_path = os.path.join(output_dir, "final_performance_report.txt")

    # --- Calculate Key Metrics ---
    holdout_mape = mean_absolute_percentage_error(holdout_results_df['most_recent_sale_price'], holdout_results_df['predicted_price'])
    oof_mape = mean_absolute_percentage_error(eval_df['most_recent_sale_price'], eval_df['final_predicted_price'])
    holdout_mae = holdout_results_df['absolute_error'].mean() # Keep MAE for context
    
    with open(report_path, "w") as f:
        f.write("=====================================================\n")
        f.write("=== Wisteria Valuation Model Performance Report ===\n")
        f.write("=====================================================\n\n")

        # --- 1. Executive Summary & Benchmark Comparison (MAPE-centric) ---
        f.write("--- 1. Executive Summary ---\n")
        f.write(f"Final Validated Holdout Set MAPE: {holdout_mape:.2f}%\n")
        f.write(f"LightGBM Baseline MAPE:           {baseline_mape:.2f}%\n")
        performance_delta = (holdout_mape - baseline_mape)
        f.write(f"Performance vs. Baseline:         {performance_delta:+.2f} percentage points\n")
        f.write(f"(Contextual Holdout MAE:          £{holdout_mae:,.2f})\n")
        f.write("Note: Lower MAPE is better. The DL model provides uncertainty & embeddings.\n\n")

        # --- 2. Generalization Check: OOF vs. Holdout (MAPE-centric) ---
        f.write("--- 2. Generalization & Overfitting Check ---\n")
        f.write(f"Out-of-Fold (OOF) MAPE on Training Set: {oof_mape:.2f}%\n")
        f.write(f"True Holdout Set MAPE:                  {holdout_mape:.2f}%\n")
        generalization_gap = (holdout_mape - oof_mape) / oof_mape if oof_mape != 0 else 0
        f.write(f"Generalization Gap:                     {generalization_gap:+.2%}\n")
        f.write("Note: A small gap (< 25%) indicates the model is generalizing well and not overfitting.\n\n")

        # --- 3. Business-Focused Accuracy Tiers (Holdout Set) ---
        f.write("--- 3. Business Accuracy Tiers (Holdout Set Performance) ---\n")
        for tier in [0.05, 0.10, 0.15, 0.25]:
            within_tier_count = (holdout_results_df['absolute_error'] / holdout_results_df['most_recent_sale_price'] <= tier).sum()
            percentage = (within_tier_count / len(holdout_results_df)) * 100
            f.write(f"Percentage of valuations within {tier:.0%} of sale price: {percentage:.1f}%\n")
        f.write("\n")

    # --- 4. Visualizations ---
    # Error Distribution Plot
    plt.figure(figsize=(10, 6))
    sns.histplot(eval_df['final_predicted_price'] - eval_df['most_recent_sale_price'], bins=50, kde=True)
    plt.title('Distribution of Prediction Errors (OOF Set)', fontsize=16)
    plt.xlabel('Prediction Error (Predicted - Actual)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "error_distribution.png"))
    plt.close()

    # Uncertainty Correlation Plot
    plt.figure(figsize=(10, 6))
    # With the monolithic model, we plot uncertainty across the entire OOF set.
    if not eval_df.empty and 'epistemic_uncertainty' in eval_df.columns:
        plot_df = eval_df.sample(min(2000, len(eval_df))) # Sample for clarity
        plot_df['abs_error'] = (plot_df['final_predicted_price'] - plot_df['most_recent_sale_price']).abs()
        sns.regplot(data=plot_df, x='epistemic_uncertainty', y='abs_error', scatter_kws={'alpha':0.2}, line_kws={'color':'red'})
    
    plt.title('Model Error vs. Epistemic Uncertainty (OOF Set)', fontsize=16)
    plt.xlabel('Epistemic Uncertainty (Model "Confusion")')
    plt.ylabel('Absolute Prediction Error')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "uncertainty_correlation.png"))
    plt.close()

    print("  - Report and plots saved to output directory.")



def _forecast_worker(i, df_row, p_type, sale_year, sale_month, model_state_dict, scaler, all_cols_by_type, max_horizon, input_window, DEVICE):
    """
    Worker function to generate forecast features for a single property (row).
    """
    try:
        if model_state_dict is None:
            return i, None

        # Reconstruct the model inside the worker process
        n_features = scaler.n_features_in_
        forecast_model = ForecastTCN(
            input_feature_dim=n_features,
            output_dim=n_features,
            num_channels=[64, 128]
        ).to(DEVICE)
        forecast_model.load_state_dict(model_state_dict)
        forecast_model.eval()

        if pd.isna(sale_year) or pd.isna(sale_month):
            sale_year, sale_month = 2024, 8

        all_cols_for_type = all_cols_by_type[p_type]
        
        point_in_time_cols = []
        pp_pattern = re.compile(r".*_(\d{4})_(\d{2})_.*")
        st_pattern = re.compile(r".*_(\d{4})_.*")
        for col in all_cols_for_type:
            pp_match, st_match = pp_pattern.match(col), st_pattern.match(col)
            col_year, col_month = 0, 0
            if pp_match: col_year, col_month = int(pp_match.group(1)), int(pp_match.group(2))
            elif st_match: col_year, col_month = int(st_match.group(1)), 1
            
            if col_year < sale_year or (col_year == sale_year and col_month < sale_month):
                point_in_time_cols.append(col)
        
        single_row_df = pd.DataFrame([df_row])
        data_array, n_timesteps, n_features = parse_and_prepare_data_for_prop_type(single_row_df, point_in_time_cols)
        
        if data_array is None or n_timesteps < input_window:
            return i, None

        data_flat = data_array.reshape(n_timesteps, n_features)
        data_scaled_flat = scaler.transform(data_flat)
        data_scaled_array = data_scaled_flat.reshape(1, n_timesteps, n_features)

        current_sequence = data_scaled_array[:, -input_window:, :]
        current_sequence_tensor = torch.tensor(current_sequence, dtype=torch.float32).permute(0, 2, 1).to(DEVICE)
        
        horizon_predictions = {}
        with torch.no_grad():
            # The loop now runs up to the maximum desired horizon
            for step in range(max_horizon):
                next_step_pred = forecast_model(current_sequence_tensor)
                # The sequence is updated with the new prediction for the next step
                current_sequence_tensor = torch.cat([current_sequence_tensor[:, :, 1:], next_step_pred.unsqueeze(2)], dim=2)
                
                # We now store the prediction for EVERY month
                final_forecast_scaled = next_step_pred.cpu().numpy()
                final_forecast_unscaled = scaler.inverse_transform(final_forecast_scaled)
                horizon_predictions[step + 1] = final_forecast_unscaled[0, 0]
        
        return i, horizon_predictions
    except Exception as e:
        # Return gracefully on error to avoid crashing the whole pool
        print(f"Warning: Error processing row {i}: {e}")
        return i, None


def train_final_model(df_main, feature_sets, msoa_col_name, msoa_cardinality, y_price_mean, y_price_std, N_TOP_FEATURES_PER_HEAD, AE_BEST_PARAMS, universal_cols_present, LAT_LON_COLS, n_trials):
    """
    This function encapsulates Optuna optimization, K-Fold cross-validation, and artifact generation for the monolithic model.
    """
    print(f"\n{'='*20} TRAINING FINAL MONOLITHIC DEEP LEARNING MODEL {'='*20}")
    
    if len(df_main) < (NUM_FOLDS_FINAL * 2):
        print(f"  - FATAL: Dataset is too small ({len(df_main)} samples) for training. Aborting.")
        return None, None

    y_price_log = np.log1p(df_main['most_recent_sale_price'])
    y_price_for_training = pd.Series((y_price_log - y_price_mean) / y_price_std, index=df_main.index)

    print(f"\n--- STAGE 6: Starting Bayesian Hyperparameter Optimization ---")
    train_idx, val_idx = train_test_split(df_main.index, test_size=0.2, random_state=42)
    df_train_optuna, df_val_optuna = df_main.loc[train_idx], df_main.loc[val_idx]
    y_train_optuna = y_price_for_training.loc[train_idx]
    y_val_optuna = y_price_for_training.loc[val_idx]
    y_train_log_optuna = y_price_log.loc[train_idx]

    data_train_optuna, data_val_optuna, feature_configs = preprocess_fold_data(
        df_train_optuna, df_val_optuna, y_train_log_optuna, feature_sets, universal_cols_present, LAT_LON_COLS,
        N_TOP_FEATURES_PER_HEAD, AE_BEST_PARAMS
    )
    data_train_optuna['msoa_id'] = df_train_optuna[[msoa_col_name]]
    data_val_optuna['msoa_id'] = df_val_optuna[[msoa_col_name]]

    study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner())
    study.optimize(lambda trial: objective(
        trial, data_train_optuna, y_train_optuna, data_val_optuna, y_val_optuna, 
        feature_configs, msoa_cardinality, y_price_std, y_price_mean, df_val_optuna
    ), n_trials=n_trials)

    best_params = study.best_trial.params
    with open(os.path.join(OUTPUT_DIR, "best_params.json"), 'w') as f: json.dump(best_params, f)

    print(f"\n--- STAGE 7: Training Final Model using Best Hyperparameters & K-Fold CV ---")
    kf = KFold(n_splits=NUM_FOLDS_FINAL, shuffle=True, random_state=42)
    oof_evidential_preds = np.zeros((len(df_main), 4))
    oof_final_gamma = np.zeros(len(df_main))

    for fold, (train_idx, val_idx) in enumerate(kf.split(df_main)):
        print(f"\n========== FINAL TRAINING: FOLD {fold+1}/{NUM_FOLDS_FINAL} ==========")
        df_train_fold, df_val_fold = df_main.iloc[train_idx], df_main.iloc[val_idx]
        y_train_fold, y_val_fold = y_price_for_training.iloc[train_idx], y_price_for_training.iloc[val_idx]
        y_train_log_fold = y_price_log.iloc[train_idx]
        
        data_train_fold, data_val_fold, fold_feature_configs = preprocess_fold_data(
            df_train_fold, df_val_fold, y_train_log_fold, feature_sets, universal_cols_present, LAT_LON_COLS,
            N_TOP_FEATURES_PER_HEAD, AE_BEST_PARAMS
        )
        best_head_params = create_head_params(best_params, fold_feature_configs)
        data_train_fold['msoa_id'] = df_train_fold[[msoa_col_name]]
        data_val_fold['msoa_id'] = df_val_fold[[msoa_col_name]]

        train_dataset = WisteriaDataset(data_train_fold, y_train_fold, y_train_fold)
        val_dataset = WisteriaDataset(data_val_fold, y_val_fold, y_val_fold)
        train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=best_params['batch_size'] * 2, shuffle=False, num_workers=2, pin_memory=True)
        
        model = FusionModel(
            feature_configs=fold_feature_configs, msoa_cardinality=msoa_cardinality, head_params=best_head_params,
            fusion_embed_dim=best_params['fusion_embed_dim'], msoa_embedding_dim=best_params['msoa_embedding_dim'],
            fusion_dropout_rate=best_params['fusion_dropout_rate'], use_enhanced_tcn=True,
            cross_attention_dropout_rate=best_params.get('cross_attention_dropout', 0.2),
            attention_dropout_rate=best_params.get('attention_dropout', 0.1),
            tcn_dropout=best_params.get('tcn_dropout', 0.45),
            spatio_temporal_attention_dropout=best_params.get('spatio_temporal_attention_dropout', 0.35)
        ).to(DEVICE)
        
        optimizer = optim.AdamW(model.parameters(), lr=best_params['learning_rate'], weight_decay=best_params['weight_decay'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=10, verbose=False)
        loss_fn = AsymmetricEvidentialLoss(coeff=best_params['evidential_reg_coeff'], undervalue_penalty=best_params['undervalue_penalty'])
        
        best_val_mape, patience_counter, best_model_state = float('inf'), 0, None
        patience = 25 # Longer patience for final model training

        for epoch in range(EPOCHS + 100):
            _ = train_one_epoch(model, train_loader, optimizer, loss_fn)
            evidential_preds_val = evaluate(model, val_loader)
            point_preds_scaled = evidential_preds_val.numpy()[:, 0]
            point_preds_unscaled = np.expm1(point_preds_scaled * y_price_std + y_price_mean)
            val_mape = mean_absolute_percentage_error(df_val_fold['most_recent_sale_price'], point_preds_unscaled)
            scheduler.step(val_mape)
            
            if val_mape < best_val_mape:
                best_val_mape = val_mape; patience_counter = 0; best_model_state = model.state_dict()
            else:
                patience_counter += 1
            if (epoch + 1) % 10 == 0: print(f"      Epoch {epoch+1}, Val MAPE: {val_mape:.2f}% (Best: {best_val_mape:.2f}%)")
            if patience_counter >= patience: break
        
        model.load_state_dict(best_model_state)
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f"model_fold_{fold}.pt"))
        evidential_preds_val = evaluate(model, val_loader, return_intermediates=False)
        oof_evidential_preds[val_idx] = evidential_preds_val.numpy()
        
        del model, optimizer, scheduler, data_train_fold, data_val_fold; gc.collect(); torch.cuda.empty_cache()

    oof_df = pd.DataFrame(oof_evidential_preds, columns=['pred_gamma', 'pred_nu', 'pred_alpha', 'pred_beta'], index=df_main.index)
    oof_df['final_gamma'] = oof_df['pred_gamma'] # No residual model for simplicity now
    aleatoric_oof, epistemic_oof = calculate_evidential_uncertainty(torch.tensor(oof_evidential_preds))
    oof_df['aleatoric_uncertainty'] = aleatoric_oof.numpy()
    oof_df['epistemic_uncertainty'] = epistemic_oof.numpy()
    oof_df = df_main.join(oof_df)

    print(f"  - Saving canonical preprocessing artifacts for inference...")
    _, _, final_fitter_configs = preprocess_fold_data(
        df_main, pd.DataFrame(), y_price_log, feature_sets, universal_cols_present, LAT_LON_COLS,
        N_TOP_FEATURES_PER_HEAD, AE_BEST_PARAMS, return_fitters=True
    )
    joblib.dump(final_fitter_configs, os.path.join(OUTPUT_DIR, "preprocessing_artifacts.joblib"))

    return oof_df, best_params


def generate_active_learning_report(holdout_results_df, output_dir, top_percent=5.0):
    """
    Analyzes holdout results to identify properties the model was most uncertain about.
    Generates a CSV report to guide future data acquisition.
    """
    print("\n--- Generating Active Learning Report ---")
    if 'epistemic_uncertainty' not in holdout_results_df.columns:
        print("  - WARNING: 'epistemic_uncertainty' not found in holdout results. Skipping Active Learning report.")
        return

    # Sort by the model's confusion score
    sorted_df = holdout_results_df.sort_values('epistemic_uncertainty', ascending=False)
    
    # Calculate how many properties to select for the report
    num_to_select = int(len(sorted_df) * (top_percent / 100.0))
    # Ensure at least one property is selected if the dataset is small
    if num_to_select == 0 and len(sorted_df) > 0:
        num_to_select = 1
    
    active_learning_targets = sorted_df.head(num_to_select)
    
    # Define the columns for the data acquisition team's report
    report_cols = [
        'property_id', 
        'epistemic_uncertainty',
        'predicted_price',
        'most_recent_sale_price',
        'absolute_error'
    ]
    
    # Save the report to a CSV file
    report_path = os.path.join(output_dir, "active_learning_targets.csv")
    active_learning_targets[report_cols].to_csv(report_path, index=False, float_format='%.4f')
    
    print(f"  - Identified the top {top_percent}% ({len(active_learning_targets)}) most uncertain properties.")
    print(f"  - Actionable report saved to: {report_path}")


def generate_shap_reports_for_holdout(df_main, df_holdout, best_params, canonical_artifacts, msoa_map, msoa_col_name, holdout_results_df, output_dir):
    """
    Efficiently generates structured SHAP explanations for every property in the holdout set.
    Produces a single CSV with head-level aggregations and top feature contributions.
    """
    import shap
    print("\n--- Generating Structured SHAP Explanations for Holdout Set ---")
    if df_holdout.empty:
        print("  - Holdout set is empty. Skipping SHAP report generation.")
        return

    # The msoa_map artifact now contains everything needed.
    all_shap_reports = []

    print(f"\n--- Preparing SHAP Explainer for Monolithic Model ---")
    background_data = df_main.sample(min(100, len(df_main)), random_state=42)
    
    # The function now relies only on its inputs, not on external files or global state.
    feature_configs = canonical_artifacts
    best_head_params = create_head_params(best_params, feature_configs)

    model_path = os.path.join(output_dir, "model_fold_0.pt")
    if not os.path.exists(model_path):
        print("  - WARNING: model_fold_0.pt not found. Cannot generate SHAP reports.")
        return
        
    model = FusionModel(
        feature_configs=feature_configs, msoa_cardinality=msoa_map['unknown_id'] + 1, head_params=best_head_params,
        fusion_embed_dim=best_params['fusion_embed_dim'], msoa_embedding_dim=best_params['msoa_embedding_dim'],
        fusion_dropout_rate=best_params['fusion_dropout_rate'],
        tcn_dropout=best_params.get('tcn_dropout', 0.45),
        spatio_temporal_attention_dropout=best_params.get('spatio_temporal_attention_dropout', 0.35)
    ).to(DEVICE)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Preprocess the background data using the safe inference function
    bg_data_processed = preprocess_for_inference(background_data, feature_configs)
    bg_data_processed['msoa_id'] = background_data[msoa_col_name].map(msoa_map['codes']).fillna(msoa_map['unknown_id']).astype(int)

    background_tensors = {k: torch.tensor(v.values, dtype=torch.float32).to(DEVICE) for k, v in bg_data_processed.items()}
    ordered_keys = sorted(list(k for k in background_tensors.keys() if k in feature_configs or k == 'lat_lon'))
    background_tensor_list = [background_tensors[k] for k in ordered_keys]

    def model_shap_wrapper(*tensors):
        input_dict = dict(zip(ordered_keys, tensors))
        input_dict['msoa_id'] = background_tensors['msoa_id']
        evidential_params, _ = model(input_dict)
        return evidential_params[:, 0]

    print("  - Initializing DeepExplainer...")
    explainer = shap.DeepExplainer(model_shap_wrapper, background_tensor_list)
    print("  - Explainer initialized.")

    feature_to_head_map = {feat: head for head, conf in feature_configs.items() if 'selected_features' in conf for feat in conf['selected_features']}

    print(f"  - Generating structured explanations for {len(df_holdout)} properties...")
    for idx, property_row in tqdm(df_holdout.iterrows(), total=len(df_holdout)):
        property_df = pd.DataFrame([property_row])
        
        # Preprocess the single property to explain using the safe inference function
        property_data_processed = preprocess_for_inference(property_df, feature_configs)
        
        explain_tensors = {k: torch.tensor(v.values, dtype=torch.float32).to(DEVICE) for k, v in property_data_processed.items()}
        explain_tensor_list = [explain_tensors[k] for k in ordered_keys]
        
        shap_values_list = explainer.shap_values(explain_tensor_list)
        
        full_feature_names = [col for key in ordered_keys for col in bg_data_processed[key].columns if key in feature_configs]
        flat_shap_values = np.concatenate([sv for sv in shap_values_list if sv.ndim > 1]) # Ensure shap value is not empty
        
        if len(full_feature_names) != len(flat_shap_values): continue # Skip if mismatched

        shap_df = pd.DataFrame({'feature': full_feature_names, 'shap_value': flat_shap_values})
        shap_df['head'] = shap_df['feature'].map(feature_to_head_map)
        
        head_level_shap = shap_df.groupby('head')['shap_value'].sum().to_dict()
        shap_df_sorted = shap_df.sort_values(by='shap_value', key=abs, ascending=False)
        top_5_contributors = shap_df_sorted.head(5)

        property_info = holdout_results_df.loc[idx]
        report = {
            'holdout_index': idx, 'property_id': property_info['property_id'],
            'predicted_price': property_info['predicted_price'], 'actual_price': property_info['most_recent_sale_price'],
            'absolute_error': property_info['absolute_error'], 'epistemic_uncertainty': property_info['epistemic_uncertainty'],
            'shap_base_value': explainer.expected_value,
            **{f'shap_head_{k}': v for k, v in head_level_shap.items()}
        }
        for i, (_, row) in enumerate(top_5_contributors.iterrows()):
            report[f'contrib_feat_{i+1}'] = row['feature']
            report[f'contrib_feat_{i+1}_shap'] = row['shap_value']
            
        all_shap_reports.append(report)

    if all_shap_reports:
        final_shap_df = pd.DataFrame(all_shap_reports)
        report_path = os.path.join(output_dir, "shap_structured_report.csv")
        final_shap_df.to_csv(report_path, index=False, float_format='%.6f')
        print(f"\nSuccessfully generated and saved structured SHAP report to {report_path}")


def load_and_merge_ae_features(df, encodings_dir):
    """
    Loads autoencoder features from .npy files and merges them into the main dataframe.
    ASSUMES a 1-to-1 row order correspondence.
    """
    print("\n--- STAGE 1.2: Loading and Merging External Autoencoder (Head G) Features ---")
    if not encodings_dir or not os.path.exists(encodings_dir):
        print(f"  - WARNING: Encodings directory not found at '{encodings_dir}'. Head G features will be missing.")
        return df

    all_ae_dfs = []
    for filename in sorted(os.listdir(encodings_dir)):
        if filename.endswith(".npy"):
            try:
                group_name = filename.replace("_encodings.npy", "")
                file_path = os.path.join(encodings_dir, filename)
                data = np.load(file_path)

                # Critical assumption check
                if len(data) != len(df):
                    print(f"  - FATAL ERROR: Row count mismatch for '{filename}'. Parquet has {len(df)} rows, but .npy has {len(data)}. Cannot merge.")
                    # In a real scenario, you might want to handle this more gracefully, but for now, we exit.
                    exit(1)

                latent_dim = data.shape[1]
                feature_names = [f"ae_{group_name}_{i}" for i in range(latent_dim)]
                ae_df = pd.DataFrame(data, columns=feature_names, index=df.index)
                all_ae_dfs.append(ae_df)
                print(f"  - Loaded {latent_dim} features for group '{group_name}'.")
            except Exception as e:
                print(f"  - ERROR: Could not process file {filename}. Error: {e}")
    
    if all_ae_dfs:
        df = pd.concat([df] + all_ae_dfs, axis=1)
        print(f"  - Successfully merged {len(all_ae_dfs)} AE feature sets. New dataframe shape: {df.shape}")
    
    return df

def preprocess_for_inference(df, artifacts):
    """
    Applies pre-fitted transformations to new data using saved artifacts.
    This is a critical function for ensuring consistency between training and inference.
    """
    data_for_model = {}
    
    for head_name, config in artifacts.items():
        # --- ARCHITECTURAL ROUTING FOR INFERENCE ---
        # Route 1: MLP or TabNet heads (identified by 'selected_features')
        if 'selected_features' in config:
            selected_features = config['selected_features']
            available_cols = [c for c in selected_features if c in df.columns]
            
            if not available_cols:
                print(f"  - WARNING: No available columns for head '{head_name}' during inference. Creating zero-filled placeholder.")
                data_for_model[head_name] = pd.DataFrame(np.zeros((len(df), len(selected_features))), columns=selected_features, index=df.index)
                continue
                
            head_df_raw = df[available_cols].copy().apply(pd.to_numeric, errors='coerce').fillna(0).replace([np.inf, -np.inf], 0)
            
            if len(available_cols) < len(selected_features):
                missing_cols = set(selected_features) - set(available_cols)
                for col in missing_cols:
                    head_df_raw[col] = 0
                head_df_raw = head_df_raw[selected_features]

            scaler = config['scaler']
            data_for_model[head_name] = pd.DataFrame(scaler.transform(head_df_raw), columns=selected_features, index=head_df_raw.index)

        # Route 2: TCN heads (identified by 'raw_features' and 'scaler')
        elif 'raw_features' in config and 'scaler' in config:
            raw_cols = config['raw_features']
            available_cols = [c for c in raw_cols if c in df.columns]
            
            if not available_cols:
                print(f"  - WARNING: No available columns for TCN head '{head_name}' during inference. Creating zero-filled placeholder.")
                # For TCN, the shape is (n_samples, n_timesteps * n_features_per_timestep)
                n_timesteps = config.get('n_timesteps', 1)
                n_features_per_timestep = config.get('n_features_per_timestep', 1)
                placeholder_shape = (len(df), n_timesteps * n_features_per_timestep)
                data_for_model[head_name] = pd.DataFrame(np.zeros(placeholder_shape), index=df.index)
                continue

            if head_name.startswith('head_H_'):
                prep_func = prepare_pp_history_for_tcn
            elif head_name.startswith('head_I_'):
                prep_func = prepare_compass_pp_for_tcn
            else:
                prep_func = prepare_tcn_data

            processed_data, _ = prep_func(df, available_cols, fit_scaler=False, scaler_obj=config['scaler'])
            data_for_model[head_name] = processed_data

    # Universal columns (not part of a complex head in artifacts)
    data_for_model['lat_lon'] = df[['latitude', 'longitude']].copy().apply(pd.to_numeric, errors='coerce').fillna(0)
    
    return data_for_model


def predict_on_holdout(df_to_predict, best_params, canonical_artifacts, msoa_map, price_stats, output_dir):
    """
    Generates predictions for a holdout set using the final monolithic model.
    It averages predictions from all K-Fold models.
    """
    if df_to_predict.empty:
        return pd.DataFrame()

    print("  - Preparing holdout data using canonical artifacts for prediction...")
    # Use the dedicated, safe inference preprocessing function
    data_for_model = preprocess_for_inference(df_to_predict, canonical_artifacts)
    
    msoa_col_name = msoa_map['column_name']
    data_for_model['msoa_id'] = df_to_predict[msoa_col_name].map(msoa_map['codes']).fillna(msoa_map['unknown_id']).astype(int)

    y_price_log_holdout = np.log1p(df_to_predict['most_recent_sale_price'])
    y_price_scaled_holdout = pd.Series((y_price_log_holdout - price_stats['y_price_mean']) / price_stats['y_price_std'], index=df_to_predict.index)
    dataset = WisteriaDataset(data_for_model, y_price_scaled_holdout, y_price_scaled_holdout)
    loader = DataLoader(dataset, batch_size=len(df_to_predict), shuffle=False)

    head_params = create_head_params(best_params, canonical_artifacts)

    all_fold_preds = []
    for fold in range(NUM_FOLDS_FINAL):
        model_path = os.path.join(output_dir, f"model_fold_{fold}.pt")
        if not os.path.exists(model_path): continue
        
        model = FusionModel(feature_configs=canonical_artifacts, msoa_cardinality=msoa_map['unknown_id'] + 1, head_params=head_params, fusion_embed_dim=best_params['fusion_embed_dim'], msoa_embedding_dim=best_params['msoa_embedding_dim'], fusion_dropout_rate=best_params['fusion_dropout_rate'], stratum_name='monolith').to(DEVICE)
        model.load_state_dict(torch.load(model_path))
        preds = evaluate(model, loader, return_intermediates=False)
        all_fold_preds.append(preds)
        del model; gc.collect(); torch.cuda.empty_cache()

    if not all_fold_preds: 
        print("  - WARNING: No fold models found for holdout prediction.")
        return pd.DataFrame()

    avg_preds = torch.mean(torch.stack(all_fold_preds), dim=0)

    final_gamma = avg_preds[:, 0].numpy()
    predicted_price = np.expm1(final_gamma * price_stats['y_price_std'] + price_stats['y_price_mean'])

    aleatoric, epistemic = calculate_evidential_uncertainty(avg_preds)

    results_df = df_to_predict[['property_id', 'most_recent_sale_price']].copy()
    results_df['predicted_price'] = predicted_price
    results_df['aleatoric_uncertainty'] = aleatoric.numpy()
    results_df['epistemic_uncertainty'] = epistemic.numpy()

    return results_df


def main():
    # --- Headless-Safe Matplotlib Configuration ---
    # This MUST be done before importing pyplot, seaborn, or running shap plots.
    import matplotlib
    matplotlib.use('Agg')
    
    warnings.filterwarnings('ignore')
    print(f"Using device: {DEVICE}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Configuration for Per-Head Feature Selection ---
    N_TOP_FEATURES_PER_HEAD = {
        "head_A_dna": 150, # Note: This will be ignored due to TabNet using all features
        "head_B_aesthetic": 75,
        "head_C_census": 150,
        "head_G_gemini_quantitative": 100,
        "head_AVM": 20, # AVM head is small, keep top 20 raw
        "head_compass_raw": 500, # Max features for the spatial TabNet head after pre-selection
        "DEFAULT": 125  # Default for any other heads
    }
    DEFAULT_AE_LATENT_DIM = 48 # Fallback for any heads not specifically tuned

    # --- Read Tuned Dimensions from Environment ---
    print("\n--- Using Static Autoencoder Hyperparameters ---")
    AE_BEST_PARAMS = {
        'head_B_aesthetic': {
            'latent_dim': 120, 'hidden_dim_1': 512, 'hidden_dim_2': 256, 'lr': 0.001299
        },
        'head_G_gemini_quantitative': {
            'latent_dim': 128, 'hidden_dim_1': 512, 'hidden_dim_2': 256, 'lr': 0.001292
        }
    }
    print(f"  - Using AE Best Params: {AE_BEST_PARAMS}")


    # --- STAGE 1: Load Data Sources (Optimized) ---
    print("--- STAGE 1: Loading Data Sources ---")
    try:
        # --- Pre-determine required columns to minimize memory usage ---
        print("  - Pre-parsing feature sets to determine required columns...")
        with open(FEATURE_SETS_PATH, 'r') as f:
            feature_sets = json.load(f)

        required_columns = set()
        for head, cols in feature_sets.items():
            required_columns.update(cols)
        
        # Add essential non-feature columns needed for processing later in the script
        required_columns.add('property_id')
        required_columns.add('latitude')
        required_columns.add('longitude')
        required_columns.add('num__property_main_type_encoded__1Flat_hm')
        # Add any other MSOA or date columns if they are not in feature_sets but are needed
        # For example, if 'MSOA11CD' is the column name:
        # required_columns.add('MSOA11CD') 
        # required_columns.add('most_recent_sale_year') # This comes from the merge, not needed here.

        print(f"  - Identified {len(required_columns)} unique columns required for the model.")
        
        # Load datasets with targeted column selection
        print("  - Loading only required columns from the master parquet file...")
        # We must now load ALL columns initially, as the AE features need to be merged by index.
        # The `required_columns` logic was causing the original `gsutil` crash.
        df_features = pd.read_parquet(MASTER_DATA_PATH)

        # NEW STEP: Load and merge the external Head G features
        AE_ENCODINGS_PATH = os.environ.get("AE_ENCODINGS_LOCAL_DIR")
        df_features = load_and_merge_ae_features(df_features, AE_ENCODINGS_PATH)
        
        # --- Load Locally Available Forecast Artifacts ---
        print("  - Loading pre-downloaded forecast model artifacts...")
        forecast_scalers = joblib.load("./forecast_artifacts/forecast_scalers.joblib")
        forecast_models = {}
        
        for p_type in ['D', 'S', 'T', 'F']:
            model_path = f"./forecast_artifacts/forecast_model_{p_type}.pt"
            if os.path.exists(model_path) and p_type in forecast_scalers:
                print(f"  - Loading forecast model for type '{p_type}'...")
                n_features_forecast = forecast_scalers[p_type].n_features_in_
                model = ForecastTCN(
                    input_feature_dim=n_features_forecast,
                    output_dim=n_features_forecast,
                    num_channels=[64, 128]
                ).to(DEVICE)
                model.load_state_dict(torch.load(model_path))
                forecast_models[p_type] = model
        
        print(f"  - Loaded {len(forecast_models)} forecast models and {len(forecast_scalers)} scalers successfully.")
        df_rightmove_raw = pd.read_csv(RIGHTMOVE_DATA_PATH, on_bad_lines='skip')
        
        print(f"  - Loaded features dataframe shape: {df_features.shape}")
        print(f"  - Loaded Rightmove dataframe shape: {df_rightmove_raw.shape}")
        print(f"  - Features df has 'property_id' column: {'property_id' in df_features.columns}")
        
    except Exception as e:
        raise IOError(f"FATAL: Could not load data files. Error: {e}")

    # --- STAGE 1.5: Refactor Feature Sets for New AVM Head ---
    print("\n--- STAGE 1.5: Re-allocating AVM features to dedicated head ---")
    AVM_FEATURES = [
        'num__bricksandlogic_estimated_price_gbp__numeric_extracted_or_original_text__bnl',
        'num__homipi_estimated_price_range_lower_bound_gbp_hm',
        'num__homipi_estimated_price_range_upper_bound_gbp_hm',
        'num__mouseprice_estimated_value_gbp__numeric_string__mp',
        'num__last_sold_price_gbp_hm',
        'num__StreetScan_past_sales_avg_price_gbp_for_All_Properties_ss',
        'num__StreetScan_past_sales_avg_price_gbp_for_Detached_ss',
        'num__StreetScan_past_sales_avg_price_gbp_for_Flats_ss',
        'num__StreetScan_past_sales_avg_price_gbp_for_New_Builds_ss',
        'num__StreetScan_past_sales_avg_price_gbp_for_Semi_Detached_ss',
        'num__StreetScan_past_sales_avg_price_gbp_for_Terraced_ss',
        'num__INT_avg_estimated_price',
        'num__INT_std_dev_estimated_price',
        'num__INT_bnl_price_per_sqft',
        'num__INT_mp_price_per_sqm',
        'num__INT_hm_price_range_spread_gbp',
        'num__INT_hm_price_vs_last_sold_ratio',
        'num__INT_hm_price_vs_ss_area_avg_price_ratio',
        'num__homipi_value_change_percentage_hm',
        'cat__mouseprice_estimated_rental_value_text__string_or_empty__mp_infrequent_sklearn',
        'cat__mouseprice_estimated_rental_value_text__string_or_empty__mp_nan'
    ]
    
    # 1. Atomically create the AVM head.
    feature_sets['head_AVM'] = AVM_FEATURES
    print(f"  - Created 'head_AVM' with {len(AVM_FEATURES)} features.")

    # 2. Definitively remove these features from all other heads to prevent duplication.
    avm_feature_set = set(AVM_FEATURES)
    for head_name in list(feature_sets.keys()):
        if head_name == 'head_AVM':
            continue # Skip the head we just created
        
        original_count = len(feature_sets[head_name])
        # This line is critical: it rebuilds the list, excluding any AVM features.
        feature_sets[head_name] = [col for col in feature_sets[head_name] if col not in avm_feature_set]
        removed_count = original_count - len(feature_sets[head_name])
        
        if removed_count > 0:
            print(f"  - Removed {removed_count} AVM features from '{head_name}'.")

    # --- STAGE 2: Direct Address Normalization (No Key Map Needed) ---
    print("\n--- STAGE 2: Direct Address Processing ---")
    
    def normalize_address_key_v4(address_str):
        """A robust address key combining postcode and street for UK properties."""
        if pd.isna(address_str) or not isinstance(address_str, str):
            return None
        
        # 1. Pre-clean the string
        address = address_str.lower().strip()
        address = re.sub(r'^(.*?)https://www\.rightmove\.co\.uk', r'\1', address) # Remove Rightmove URL prefix
        address = re.sub(r'[^\w\s]', '', address) # Remove punctuation but keep spaces

        # 2. Extract Postcode (most unique part)
        postcode_match = re.search(r'([a-z]{1,2}\d[a-z\d]?\s*\d[a-z]{2})', address)
        postcode = postcode_match.group(1).replace(" ", "") if postcode_match else ''
        if not postcode:
            return None # If there's no postcode, the address is too ambiguous to be a reliable key

        # 3. Extract Street Name (remove postcode to find street)
        street_part = address.replace(postcode_match.group(1), '').strip()
        street_key = re.sub(r'\s+', '', street_part)[:5] # Take first 5 non-space chars of the street

        return f"{postcode}_{street_key}"
    
    # Process features addresses (using existing property_id)
    print("  - Normalizing feature addresses using V4 key (postcode_street)...")
    df_features['normalized_address_key'] = df_features['property_id'].apply(normalize_address_key_v4)
    
    # Remove rows with null addresses
    df_features = df_features[df_features['normalized_address_key'].notna()].copy()
    
    print(f"  - Features with valid addresses: {len(df_features)}")
    print("  - Sample normalized feature keys:")
    sample_feature_keys = df_features['normalized_address_key'].head(10).tolist()
    print("   ", sample_feature_keys)

    # Process Rightmove data
    df_rightmove_raw.columns = df_rightmove_raw.columns.str.lower()
    address_col_rm = 'address' if 'address' in df_rightmove_raw.columns else df_rightmove_raw.columns[0]

    print("  - Normalizing Rightmove addresses using V4 key (postcode_street)...")
    df_rightmove_raw['normalized_address_key'] = df_rightmove_raw[address_col_rm].apply(normalize_address_key_v4)
    
    # Remove rows with null addresses
    df_rightmove_raw = df_rightmove_raw[df_rightmove_raw['normalized_address_key'].notna()].copy()
    
    print(f"  - Rightmove with valid addresses: {len(df_rightmove_raw)}")
    print("  - Sample normalized Rightmove keys:")
    sample_rm_keys = df_rightmove_raw['normalized_address_key'].head(10).tolist()
    print("   ", sample_rm_keys)

    # Process sales data
    df_rightmove_processed = process_rightmove_data(df_rightmove_raw)

    # Merge with address keys AND original address for validation
    address_col_rm = 'address' if 'address' in df_rightmove_raw.columns else df_rightmove_raw.columns[0]
    df_rightmove_with_address_key = pd.merge(
        df_rightmove_raw[['normalized_address_key', address_col_rm]],
        df_rightmove_processed,
        left_index=True,
        right_on='rightmove_row_id',
        how='inner'
    )
    df_rightmove_with_address_key.drop(columns=['rightmove_row_id'], inplace=True, errors='ignore')

    print(f"  - Rightmove processed: {df_rightmove_with_address_key.shape}")

    # --- STAGE 3: Enhanced Matching Strategy ---
    print("\n--- STAGE 3: Enhanced Address Matching ---")
    
    # Check initial overlap
    features_keys = set(df_features['normalized_address_key'])
    rightmove_keys = set(df_rightmove_with_address_key['normalized_address_key'])
    initial_overlap = len(features_keys & rightmove_keys)
    
    print(f"  - Initial exact overlap: {initial_overlap} properties")
    
    if initial_overlap == 0:
        print("  - No exact matches found. Fuzzy/partial matching will be attempted in STAGE 4.")
    else:
        print(f"  - Exact key overlap of {initial_overlap} found. Merge will proceed in STAGE 4.")

    
    
    print("\n--- STAGE 4: Definitive Merge and Validation ---")
    # Handle duplicates before merging
    df_features.drop_duplicates(subset=['normalized_address_key'], keep='first', inplace=True)
    df_rightmove_with_address_key.drop_duplicates(subset=['normalized_address_key'], keep='first', inplace=True)

    # Check for overlap to determine merge strategy
    features_keys = set(df_features['normalized_address_key'].dropna())
    rightmove_keys = set(df_rightmove_with_address_key['normalized_address_key'].dropna())
    overlap = len(features_keys & rightmove_keys)
    merge_on_key = 'final_merge_key'

    if overlap > 0:
        print(f"  - Using {overlap} exact matches for merge.")
        df_features[merge_on_key] = df_features['normalized_address_key']
        df_rightmove_with_address_key[merge_on_key] = df_rightmove_with_address_key['normalized_address_key']
    else:
        print("  - No exact matches found. Attempting fuzzy matching fallback...")
        fuzzy_matches = find_best_matches_fuzzy(features_keys, rightmove_keys, threshold=85)
        
        if fuzzy_matches:
            fuzzy_mapping = {fkey: rkey for fkey, rkey, score in fuzzy_matches}
            df_features[merge_on_key] = df_features['normalized_address_key'].map(fuzzy_mapping)
            df_rightmove_with_address_key[merge_on_key] = df_rightmove_with_address_key['normalized_address_key']
            print(f"  - Created {len(fuzzy_mapping)} mappings from fuzzy matching.")
        else:
            print("  - WARNING: Fuzzy matching failed. Falling back to exact keys (will likely result in an empty merge).")
            df_features[merge_on_key] = df_features['normalized_address_key']
            df_rightmove_with_address_key[merge_on_key] = df_rightmove_with_address_key['normalized_address_key']

    # Define all columns to bring in from the Rightmove dataset
    if address_col_rm not in df_rightmove_with_address_key.columns:
        raise ValueError(f"FATAL: Rightmove address column '{address_col_rm}' missing before final merge.")
    
    cols_to_merge_from_rightmove = [
        merge_on_key, 'most_recent_sale_price', 'most_recent_sale_year',
        'most_recent_sale_month', 'total_sales_count', 'days_since_last_sale',
        'price_change_since_last', address_col_rm
    ]

    # Prevent column name collisions that cause KeyErrors downstream
    cols_to_drop_from_features = [c for c in cols_to_merge_from_rightmove if c in df_features.columns and c != merge_on_key]
    if cols_to_drop_from_features:
        print(f"  - Dropping conflicting columns from features df before merge: {cols_to_drop_from_features}")
        df_features.drop(columns=cols_to_drop_from_features, inplace=True, errors='ignore')

    # Perform the single, definitive merge
    df = pd.merge(
        df_features.dropna(subset=[merge_on_key]),
        df_rightmove_with_address_key,
        on=merge_on_key,
        how='inner'
    )

    print(f"  - Final merged dataset shape: {df.shape}")
    if len(df) == 0:
        raise ValueError("FATAL: Merge resulted in an empty dataset. Check normalization and key matching.")

    # --- DIAGNOSTICS: Report on Merge Failures ---
    print("\n--- DIAGNOSTICS: Analyzing Merge Failures ---")
    original_feature_keys = set(df_features[merge_on_key].dropna())
    merged_keys = set(df[merge_on_key])
    failed_keys = original_feature_keys - merged_keys
    
    print(f"  - Total properties in source features (pre-merge): {len(original_feature_keys)}")
    print(f"  - Total properties successfully merged: {len(merged_keys)}")
    print(f"  - Total properties that failed to merge: {len(failed_keys)}")
    
    if failed_keys:
        failed_merges_df = df_features[df_features[merge_on_key].isin(failed_keys)]
        print("\n  --- Sample of Properties That FAILED to Merge ---")
        sample_size = min(10, len(failed_merges_df))
        sample_failures = failed_merges_df.head(sample_size)
        for index, row in sample_failures.iterrows():
            print(f"  - Original Address: '{row.get('property_id', 'N/A')[:70]}...'")
            print(f"    Normalized Key: '{row.get(merge_on_key, 'N/A')}'")
        print("\n  - DEBUGGING HINT: The 'Normalized Key' above could not be found in the Rightmove data.")

    # --- Final Validation ---
    validate_merge_quality(df)
    
    # --- NEW: Data Quality Filter ---
    print("\n--- Filtering for High-Confidence Merges ---")
    address_col_rm = [col for col in df.columns if 'address' in col.lower()][0]
    
    def normalize_for_comparison(s):
        """Canonicalizes an address string by lowercasing and removing non-alphanumerics."""
        return re.sub(r'[^a-z0-9]', '', str(s).lower())

    # Initialize tqdm for pandas apply
    tqdm.pandas(desc="Calculating Merge Similarities")

    # Calculate similarity score on the CANONICALIZED addresses to ensure a fair comparison.
    df['merge_similarity_score'] = df.progress_apply(
        lambda row: fuzz.token_set_ratio(
            normalize_for_comparison(row.get('property_id', '')),
            normalize_for_comparison(row.get(address_col_rm, ''))
        ),
        axis=1
    )
    
    initial_rows = len(df)
    similarity_threshold = 85 # Only keep confident matches

    # Memory-efficient filtering: identify indices to drop and remove them in-place.
    indices_to_drop = df[df['merge_similarity_score'] < similarity_threshold].index
    df.drop(indices_to_drop, inplace=True)
    
    print(f" - Kept {len(df)} rows out of {initial_rows} after applying similarity threshold of {similarity_threshold}%.")
    if len(df) < 5000:
        print(" - WARNING: Dataset size is now very small. Consider lowering the similarity threshold if performance degrades.")

    # --- NEW: Temporal Leakage Removal for Price Paid Features ---
    print("\n--- Removing Temporal Price-Paid Leakage (Features >= 2024) ---")
    LEAKAGE_YEAR_THRESHOLD = 2024
    leaky_pp_cols = set()

     # Regex to find columns like pp_YYYY_... or compass_..._pp_YYYY_...
    pp_leakage_pattern = r"(?:pp_|compass_.*_pp_)(\d{4})_"
    
    # Use fast, vectorized string operations instead of a slow Python loop.
    # 1. Create a Series of column names
    cols_series = pd.Series(df.columns)
    # 2. Extract the year from all columns that match the pattern
    years = cols_series.str.extract(pp_leakage_pattern).squeeze().astype(float)
    # 3. Find the columns where the year meets the leakage threshold
    leaky_cols_mask = (years >= LEAKAGE_YEAR_THRESHOLD)
    leaky_pp_cols = set(cols_series[leaky_cols_mask].tolist())

    if leaky_pp_cols:
        print(f"  - Found {len(leaky_pp_cols)} price-paid features from {LEAKAGE_YEAR_THRESHOLD} onwards to remove.")
        
        # Remove from the main DataFrame
        df.drop(columns=list(leaky_pp_cols), inplace=True)
        print(f"  - Dropped leaky columns from the main DataFrame. New shape: {df.shape}")
        
        # Remove from the feature_sets dictionary to prevent errors downstream
        print("  - Sanitizing feature_sets dictionary...")
        for head_name in list(feature_sets.keys()):
            original_count = len(feature_sets[head_name])
            feature_sets[head_name] = [c for c in feature_sets[head_name] if c not in leaky_pp_cols]
            removed_count = original_count - len(feature_sets[head_name])
            if removed_count > 0:
                print(f"    - Removed {removed_count} leaky features from '{head_name}'.")
    else:
        print(f"  - No temporal price-paid leakage found for years >= {LEAKAGE_YEAR_THRESHOLD}.")
    
    # --- NEW: Global Correlation-Based Leakage Screener ---
    print("\n--- STAGE 4.1: Proactive Leakage Detection via Correlation Analysis ---")
    TARGET_VARIABLE = 'most_recent_sale_price'
    LEAKAGE_THRESHOLD = 0.999 # Use a very high threshold to only catch definitive leaks

    confirmed_leaks = []
    # We check all numeric columns except the target itself
    numeric_cols = df.select_dtypes(include=np.number).columns.drop(TARGET_VARIABLE, errors='ignore')

    # Use a single, vectorized operation to calculate all correlations at once.
    print("  - Calculating all feature correlations with the target variable...")
    correlations = df[numeric_cols].corrwith(df[TARGET_VARIABLE])
    
    # Find leaks by filtering the resulting series
    leaky_corrs = correlations[correlations.abs() > LEAKAGE_THRESHOLD]
    confirmed_leaks = leaky_corrs.index.tolist()

    if confirmed_leaks:
        for col in confirmed_leaks:
            print(f"  [!!! LEAKAGE DETECTED !!!] Column '{col}' has a correlation of {leaky_corrs[col]:.6f} with the target.")

    if confirmed_leaks:
        print("\n  - Removing confirmed leaks from all feature sets...")
        leaks_to_remove = set(confirmed_leaks)
        feature_sets_clean = {}
        for head_name, cols in feature_sets.items():
            original_count = len(cols)
            cleaned_cols = [col for col in cols if col not in leaks_to_remove]
            feature_sets_clean[head_name] = cleaned_cols
            removed_count = original_count - len(cleaned_cols)
            if removed_count > 0:
                print(f"    - Removed {removed_count} leak(s) from '{head_name}'.")
        
        # Overwrite the original feature_sets with the sanitized version
        feature_sets = feature_sets_clean
        
        # Also drop the columns from the main dataframe to be safe
        df.drop(columns=list(leaks_to_remove), inplace=True)
        print(f"  - Dropped {len(leaks_to_remove)} leaky columns from the main DataFrame.")
    else:
        print(f"  - No definitive leaks found with correlation > {LEAKAGE_THRESHOLD}.")
    
    
    
     # --- NEW: Generate Forecast Features BEFORE other processing ---
    # We now specify a maximum horizon for dense monthly forecasting.
    df, new_forecast_feature_names = generate_forecast_features(df, forecast_models, forecast_scalers, feature_sets, max_horizon=36)
    if 'head_AVM' in feature_sets:
        feature_sets['head_AVM'].extend(new_forecast_feature_names)
        print(f"  - Added {len(new_forecast_feature_names)} forecast features to 'head_AVM'.")
    else:
        feature_sets['head_AVM'] = new_forecast_feature_names

    # --- AVM LEAKAGE MITIGATION (Now a Secondary Check) ---
    df = mitigate_avm_leakage(df, label_col='most_recent_sale_price', correlation_threshold=0.995)

    # --- NEW: Sanitize feature sets to remove head_F now that it's used in forecasting ---
    print("  - Sanitizing feature sets to remove head_F_* groups to prevent leakage...")
    keys_to_remove = [k for k in feature_sets if k.startswith('head_F_')]
    for k in keys_to_remove:
        del feature_sets[k]
        print(f"    - Removed feature group '{k}' from main model inputs.")

    # --- NEW STAGE 4.1.5: Temporal Summary Feature Engineering (Longitudinal Compression) ---
    df, new_temporal_summary_names = engineer_temporal_summary_features(df, feature_sets)
    if 'head_C_census' in feature_sets:
        feature_sets['head_C_census'].extend(new_temporal_summary_names)
        print(f"  - Added {len(new_temporal_summary_names)} summary features to 'head_C_census'.")
    else:
        # Fallback in case the head doesn't exist for some reason
        feature_sets['head_C_census'] = new_temporal_summary_names

    
    # --- NEW STAGE 4.2: Autoencoder Feature Engineering (BEFORE Feature Selection) ---
    print("\n--- STAGE 4.2: Consolidating Raw Spatial Features for TabNet Head ---")
    
    spatial_prefixes = ['compass_', 'atlas_', 'microscope_']
    raw_static_spatial_cols = []

    for prefix in spatial_prefixes:
        # Find all static (non-temporal) columns for this prefix
        static_cols = [c for c in df.columns if c.startswith(prefix) and not re.search(r'_\d{4}_', c)]
        raw_static_spatial_cols.extend(static_cols)
        print(f"  - Found {len(static_cols)} raw static features for prefix '{prefix}'.")

    # Create the new head in the feature_sets dictionary
    feature_sets['head_compass_raw'] = raw_static_spatial_cols
    print(f"  - Created new 'head_compass_raw' with a total of {len(raw_static_spatial_cols)} features for TabNet processing.")
    
    
    print(f"  - DataFrame now has {df.shape[1]} total columns before entering head-specific processing.")

    # --- STAGE 5: Filter, Validate, and Prepare Data for Model ---
    print("\n--- STAGE 5: Filtering, Validating and Preparing Data for Sub-Head Architecture ---")
    min_price, max_price = 10000, df['most_recent_sale_price'].quantile(0.999)
    df = df[(df['most_recent_sale_price'] >= min_price) & (df['most_recent_sale_price'] <= max_price)].copy().reset_index(drop=True)
    print(f"  - Dataset after outlier removal (min: £{min_price:,.0f}, max: £{max_price:,.0f}): {len(df)} properties")

    # --- NEW: Create a true holdout set ---
    n_holdout = 500 # Increased from 100 for a more robust final validation

    # Safeguard: If the dataset is smaller than the holdout set, reduce the holdout size.
    if len(df) <= n_holdout:
        print(f"  - WARNING: Dataset size ({len(df)}) is smaller than n_holdout ({n_holdout}).")
        n_holdout = int(len(df) * 0.1) # Use 10% for holdout instead
        print(f"  - Reducing holdout size to {n_holdout} to prevent crash.")

    df_main, df_holdout = train_test_split(df, test_size=n_holdout, random_state=42, shuffle=True)
    df_main = df_main.reset_index(drop=True)
    df_holdout = df_holdout.reset_index(drop=True)
    print(f"  - Created main training set with {len(df_main)} properties.")
    print(f"  - Created true holdout set with {len(df_holdout)} properties for final validation.")

    y_price_log = np.log1p(df_main['most_recent_sale_price'])
    y_price_mean, y_price_std = y_price_log.mean(), y_price_log.std()
    # --- Global, Non-Leaky Preprocessing ---
    msoa_col = find_msoa_column(df_main)
    msoa_codes, msoa_uniques = pd.factorize(df_main[msoa_col].fillna('Unknown'))
    df_main[msoa_col] = msoa_codes
    msoa_cardinality = len(msoa_uniques) + 1
    print(f"\n--- MSOA cardinality set to: {msoa_cardinality} ---")

    LAT_LON_COLS = ['latitude', 'longitude']
    UNIVERSAL_PREDICTORS = [
        'num__floor_area_sqm_from_homipi__numeric_or_empty__hm', 'num__number_of_bedrooms_from_homipi__numeric_or_empty__hm', 'num__number_of_reception_rooms_from_homipi__numeric_or_empty__hm',
        'num__number_of_storeys_from_homipi__numeric_or_empty__hm', 'num__property_construction_year_from_homipi__YYYY_or_2025NewBuildOrError__hm', 'num__property_main_type_encoded__1Flat_hm',
        'num__property_sub_type_code_from_homipi__4Detached_hm', 'num__property_tenure_code_from_homipi__2Freehold_hm', 'primary_MainGarden_area_sqm', 'primary_MainGarage_area_sqm',
        'primary_MainDrivewayParking_area_sqm', 'other_OtherBathroomsWCs_count', 'num_rooms_identified_in_step5', 'atlas_cluster_id',
        'num__StreetScan_average_household_income_gbp_ss', 'num__StreetScan_deprivation_rank_for_Overall_Multiple_Deprivation_ss', 'num__StreetScan_deprivation_rank_for_Crime_ss',
        'num__chimnie_local_area_safety_index_score_ch', 'num__primary_or_first_type_school_1_distance__numeric_extracted_or_original_text__bnl', 'num__primary_or_first_type_school_1_ofsted_rating_encoded__numeric_extracted_or_original_text__bnl',
        'num__train_station_1_distance__numeric_extracted_or_original_text__bnl', 'num__homipi_nearest_hospital_distance_or_number_1_hm', 'LSOA_Property_Density', 'ah4gpas', 'ah4no2',
        'ah4pm10', 'compass_mean_LSOA_Price_Growth_5Year_n100', 'num__last_sold_date_year__YYYY__hm', 'HPI_Adjusted_Median_Price_D',
        'HPI_Adjusted_Median_Price_F', 'LSOA_MedPrice_Recent', 'LSOA_Price_Growth_5Year', 'LSOA_Annual_Transaction_Rate', 'Sale_Count_D', 'num__current_epc_value_extracted_by_initial_homipi_script_hm_numeric',
        'num__potential_epc_value_extracted_by_initial_homipi_script_hm_numeric', 'primary_MainKitchen_renovation_score', 'primary_MainBathroom_renovation_score', 'primary_MainExteriorFront_renovation_score',
        'primary_MainLivingArea_renovation_score', 'primary_MainKitchen_num_features', 'primary_MainBathroom_num_flaws', 'Composite_CoreLiving_Strength', 'Opportunity_Good_Bones_Score',
        'Risk_Cosmetic_Burden_X_Age', 'Thesis_FutureProofingScore', 'avg_persona_rating_overall'
    ]
    universal_cols_present = [col for col in UNIVERSAL_PREDICTORS if col in df_main.columns]

    # --- STRATEGY 1 - ESTABLISH A SIMPLE BASELINE ---
    print("\n--- STRATEGY 1: Training a LightGBM Baseline Model ---")
    all_features = [col for col in df_main.columns if col not in ['most_recent_sale_price', 'property_id', 'normalized_address_key', 'final_merge_key', 'address'] and pd.api.types.is_numeric_dtype(df_main[col])]
    lgbm_selector = lgb.LGBMRegressor(random_state=42, n_estimators=200, n_jobs=-1, verbose=-1)
    lgbm_selector.fit(df_main[all_features].fillna(0), y_price_log)
    importances = pd.DataFrame({'feature': all_features, 'importance': lgbm_selector.feature_importances_}).sort_values('importance', ascending=False)
    N_TOP_GLOBAL_FEATURES = 750
    top_global_features = importances.head(N_TOP_GLOBAL_FEATURES)['feature'].tolist()
    X_baseline = df_main[top_global_features].fillna(0)
    y_baseline = df_main['most_recent_sale_price']
    baseline_kf = KFold(n_splits=NUM_FOLDS_FINAL, shuffle=True, random_state=42)
    oof_preds_baseline = np.zeros(len(df_main))
    for fold, (train_idx, val_idx) in enumerate(baseline_kf.split(X_baseline)):
        X_train, X_val = X_baseline.iloc[train_idx], X_baseline.iloc[val_idx]
        y_train_log, y_val_log = y_price_log.iloc[train_idx], y_price_log.iloc[val_idx]
        lgbm_baseline = lgb.LGBMRegressor(random_state=42, n_estimators=2000, learning_rate=0.02, num_leaves=31, n_jobs=-1, colsample_bytree=0.7, subsample=0.7, reg_alpha=0.1, reg_lambda=0.1)
        lgbm_baseline.fit(X_train, y_train_log, eval_set=[(X_val, y_val_log)], eval_metric='mae', callbacks=[lgb.early_stopping(100, verbose=False)])
        oof_preds_baseline[val_idx] = np.expm1(lgbm_baseline.predict(X_val))
    baseline_mape = mean_absolute_percentage_error(y_baseline, oof_preds_baseline)
    print(f"\n--- LIGHTGBM BASELINE MODEL OOF MAPE: {baseline_mape:.2f}% ---\n")

    # --- STAGES 6 & 7: Train Final Monolithic Deep Learning Model ---
    eval_df, best_params = train_final_model(
        df_main=df_main, feature_sets=feature_sets.copy(),
        msoa_col_name=msoa_col, msoa_cardinality=msoa_cardinality,
        y_price_mean=y_price_mean, y_price_std=y_price_std,
        N_TOP_FEATURES_PER_HEAD=N_TOP_FEATURES_PER_HEAD, AE_BEST_PARAMS=AE_BEST_PARAMS,
        universal_cols_present=universal_cols_present, LAT_LON_COLS=LAT_LON_COLS,
        n_trials=N_TRIALS_MAIN
    )

    # --- STAGE 8: Save Final Training Artifacts ---
    print("\n--- STAGE 8: Evaluating Final OOF Performance & Saving Artifacts ---")
    if eval_df is None:
        raise ValueError("FATAL: Model training failed.")

    eval_df['final_predicted_price'] = np.expm1(eval_df['final_gamma'] * y_price_std + y_price_mean)
    final_mae = mean_absolute_error(eval_df['most_recent_sale_price'], eval_df['final_predicted_price'])
    final_mape = mean_absolute_percentage_error(eval_df['most_recent_sale_price'], eval_df['final_predicted_price'])
    final_r2 = r2_score(eval_df['most_recent_sale_price'], eval_df['final_predicted_price'])
    print(f"\n--- OOF PERFORMANCE (MONOLITHIC DEEP LEARNING MODEL) ---")
    print(f"  - Final MAPE: {final_mape:.2f}%, MAE: £{final_mae:,.2f}, R²: {final_r2:.4f}")
    eval_df.to_csv(os.path.join(OUTPUT_DIR, "oof_predictions.csv"), index=False)
    print("OOF predictions and preprocessing artifacts saved successfully.")

    # --- STAGE 10: Final Holdout Set Evaluation ---
    print("\n--- STAGE 10: Final Evaluation on True Holdout Set ---")
    if len(df_holdout) > 0:
        price_stats = {'y_price_mean': y_price_mean, 'y_price_std': y_price_std}
        msoa_map = {
            'codes': dict(zip(msoa_uniques, range(len(msoa_uniques)))), 
            'unknown_id': len(msoa_uniques),
            'column_name': msoa_col
        }
        canonical_artifacts = joblib.load(os.path.join(OUTPUT_DIR, "preprocessing_artifacts.joblib"))

        holdout_results = predict_on_holdout(
            df_holdout, 
            best_params, 
            canonical_artifacts,
            msoa_map,
            price_stats,
            OUTPUT_DIR
        )
        
        holdout_results['absolute_error'] = (holdout_results['predicted_price'] - holdout_results['most_recent_sale_price']).abs()
        print("\n--- HOLDOUT SET FINAL RESULTS ---")
        final_mae_holdout = holdout_results['absolute_error'].mean()
        final_mape_holdout = mean_absolute_percentage_error(holdout_results['most_recent_sale_price'], holdout_results['predicted_price'])
        print(f"  - Holdout MAPE: {final_mape_holdout:.2f}%")
        print(f"  - Holdout MAE:  £{final_mae_holdout:,.2f}")
        holdout_results.to_csv(os.path.join(OUTPUT_DIR, "holdout_evaluation_results.csv"), index=False)

        # Generate Reports
        generate_final_report(eval_df, holdout_results, baseline_mape, OUTPUT_DIR)
        generate_active_learning_report(holdout_results, OUTPUT_DIR)

        # --- STAGE 11: Save Final Universal Artifacts for Inference ---
        print("\n--- STAGE 11: Saving Final Universal Artifacts for Inference Pipeline ---")
        with open(os.path.join(OUTPUT_DIR, "price_stats.json"), 'w') as f: json.dump(price_stats, f)
        with open(os.path.join(OUTPUT_DIR, "msoa_map.json"), 'w') as f: json.dump(msoa_map, f)
        print("  - Saved price_stats.json and msoa_map.json.")

        # --- STAGE 12: Generate Batch SHAP Explanations ---
        print("\n--- STAGE 12: Generating SHAP reports for monolithic model ---")
        generate_shap_reports_for_holdout(
            df_main=df_main,
            df_holdout=df_holdout,
            best_params=best_params,
            canonical_artifacts=canonical_artifacts,
            msoa_map=msoa_map,
            msoa_col_name=msoa_col,
            holdout_results_df=holdout_results,
            output_dir=OUTPUT_DIR
        )

    print("\nPipeline finished successfully.")

if __name__ == "__main__":
    main()
EOF

echo "Downloading master dataset from GCS with retries..."
# --- ROBUST DOWNLOAD LOGIC ---
# 1. Ensure a clean slate by removing any potentially corrupted partial file.
rm -f "${MASTER_DATA_LOCAL_PATH}"

# 2. Set gsutil to retry and disable slicing for maximum integrity on large files.
#    Disabling slicing can be slower but avoids potential issues with component reassembly.
gsutil \
  -o "GSUtil:max_retries=10" \
  -o "GSUtil:parallel_thread_count=1" \
  -o "GSUtil:SLICED_DOWNLOAD_THRESHOLD=-1" \
  cp "${MASTER_DATA_GCS_PATH}" "${MASTER_DATA_LOCAL_PATH}"

# 3. Add a crucial check to ensure the file was downloaded successfully.
if [ ! -f "${MASTER_DATA_LOCAL_PATH}" ]; then
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" >&2
    echo "!!! FATAL: MASTER DATASET DOWNLOAD FAILED.                                 !!!" >&2
    echo "!!! This was likely due to a Hash Mismatch, indicating data corruption     !!!" >&2
    echo "!!! during transfer. The corrupted local file has been deleted.            !!!" >&2
    echo "!!!                                                                        !!!" >&2
    echo "!!! TO FIX THIS:                                                           !!!" >&2
    echo "!!! 1. Check the VM's network connection and available disk space.         !!!" >&2
    echo "!!! 2. Re-run the script. The robust download logic will try again.        !!!" >&2
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" >&2
    exit 1
fi
echo "Master dataset downloaded and verified successfully."


echo "Downloading feature set definitions from GCS..."
gsutil cp "${FEATURE_SETS_GCS_PATH}" "${FEATURE_SETS_LOCAL_PATH}"

echo "Downloading Rightmove dataset from GCS..."
gsutil cp "${RIGHTMOVE_GCS_PATH}" "${RIGHTMOVE_DATA_LOCAL_PATH}"

echo "Downloading Key Mapping dataset from GCS..."
gsutil cp "${KEY_MAP_GCS_PATH}" "${KEY_MAP_LOCAL_PATH}"

# --- NEW: Offline Preprocessing Tuning Stage ---
echo "--- STAGE 0: Running Offline Preprocessing Tuning ---"
export MASTER_DATA_LOCAL_PATH
export FEATURE_SETS_LOCAL_PATH
export N_TRIALS_AE


# --- STAGE A: Download Pre-Trained Temporal Forecasting Models ---
echo "--- STAGE A: Downloading Pre-Trained Temporal Forecasting Models ---"
# The main script expects these artifacts in a local directory named 'forecast_artifacts'.
mkdir -p "./forecast_artifacts"
echo "Downloading specific forecast artifacts from ${FORECAST_ARTIFACTS_GCS_DIR}..."
# Download only the necessary model and scaler files, not the entire directory.
gsutil cp "${FORECAST_ARTIFACTS_GCS_DIR}/forecast_model_D.pt" "./forecast_artifacts/"
gsutil cp "${FORECAST_ARTIFACTS_GCS_DIR}/forecast_model_F.pt" "./forecast_artifacts/"
gsutil cp "${FORECAST_ARTIFACTS_GCS_DIR}/forecast_model_S.pt" "./forecast_artifacts/"
gsutil cp "${FORECAST_ARTIFACTS_GCS_DIR}/forecast_model_T.pt" "./forecast_artifacts/"
gsutil cp "${FORECAST_ARTIFACTS_GCS_DIR}/forecast_scalers.joblib" "./forecast_artifacts/"
echo "Pre-trained forecast artifacts downloaded successfully."
# --- End of New Stage ---


# --- NEW STAGE: Download Autoencoder Encodings for Head G ---
echo "--- STAGE B1: Downloading Autoencoder Encodings for Head G ---"
MODULAR_AE_OUTPUT_BASE_GCS="gs://${GCS_BUCKET}/models/modular_autoencoders"
AE_ENCODINGS_LOCAL_DIR="${DATA_DIR}/ae_encodings_head_g"
mkdir -p "${AE_ENCODINGS_LOCAL_DIR}"

ENCODING_FILES=$(gsutil ls "${MODULAR_AE_OUTPUT_BASE_GCS}/*/encodings.npy")
if [ -z "$ENCODING_FILES" ]; then
    echo "WARNING: No 'encodings.npy' files found for Head G. It will be missing from the model."
else
    for gcs_path in $ENCODING_FILES; do
        group_name=$(basename "$(dirname "$gcs_path")")
        echo "  - Downloading Head G encoding for group: ${group_name}"
        gsutil -q cp "${gcs_path}" "${AE_ENCODINGS_LOCAL_DIR}/${group_name}_encodings.npy"
    done
fi
export AE_ENCODINGS_LOCAL_DIR # Export path for the Python script
# --- END NEW STAGE ---

echo "--- STAGE B2: Running Main Valuation Model Training ---"
export RIGHTMOVE_DATA_LOCAL_PATH
export KEY_MAP_LOCAL_PATH
export N_TRIALS
export FORECAST_ARTIFACTS_GCS_DIR # Pass GCS path to the main script
export OUTPUT_DIR # Export the output directory path to the python script

LOG_FILE="${OUTPUT_DIR}/training_run.log"
python3 -u "${SCRIPT_PATH}" | tee -a "${LOG_FILE}"


echo "Uploading all training artifacts to GCS..."
gsutil -m cp -r "${OUTPUT_DIR}/*" "${OUTPUT_GCS_DIR}/"

echo "all done! The model training artifacts are available at ${OUTPUT_GCS_DIR}."