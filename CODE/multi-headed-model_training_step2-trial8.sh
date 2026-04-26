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
# DEFERRED: Path variables will be defined AFTER changing into the project directory.

# --- Local Project Setup ---
PROJECT_DIR_NAME="model_training_project_v5" # Just the name
mkdir -p "./${PROJECT_DIR_NAME}"
cd "./${PROJECT_DIR_NAME}"

# --- Environment Setup & Path Definitions (Now relative to the correct PWD) ---
VENV_DIR="./venv_mt"
OUTPUT_DIR="./output"
DATA_DIR="./data"
mkdir -p "${OUTPUT_DIR}" "${DATA_DIR}" # Ensure subdirectories exist

# Define paths for files that will be created or used inside the project directory
MASTER_DATA_LOCAL_PATH="${DATA_DIR}/master_dataset.parquet"
FEATURE_SETS_LOCAL_PATH="${DATA_DIR}/feature_sets.json"
RIGHTMOVE_DATA_LOCAL_PATH="${DATA_DIR}/Rightmove.csv"
SCRIPT_PATH="./02_train_multi_head_model_v5.py" # Simple, robust path
KEY_MAP_LOCAL_PATH="${DATA_DIR}/key_map.csv"

# --- System & Driver Setup (MUST be done before venv creation) ---
echo "Updating system packages and installing L4 GPU dependencies..."
sudo apt-get update
# Install a modern NVIDIA driver compatible with the L4 GPU's Ada Lovelace architecture
# ARCHITECTURAL FIX: Added python3-venv, which is required to create virtual environments.
sudo apt-get install -y libgomp1 nvidia-driver-535 python3-venv


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
pip install pandas pyarrow gcsfs google-cloud-storage scikit-learn lightgbm fuzzywuzzy optuna matplotlib seaborn python-Levenshtein tqdm shap tensorflow

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
def normalize_address_key_v4(address_str):
    """A robust address key combining postcode and street for UK properties."""
    if pd.isna(address_str) or not isinstance(address_str, str):
        return None
    
    # 1. Pre-clean the string
    address = str(address_str).lower().strip()
    address = re.sub(r'^(.*?)https://www\.rightmove\.co\.uk', r'\1', address) # Remove Rightmove URL prefix
    address = re.sub(r'[^\w\s]', '', address) # Remove punctuation but keep spaces

    # 2. Extract Postcode (most unique part)
    postcode_match = re.search(r'([a-z]{1,2}\d[a-z\d]?\s*\d[a-z]{2})', address)
    postcode = postcode_match.group(1).replace(" ", "") if postcode_match else ''
    if not postcode:
        return None # If there's no postcode, the address is too ambiguous to be a reliable key

    # 3. Extract Street Name (remove postcode to find street)
    street_part = address.replace(postcode_match.group(1), '').strip() if postcode_match else address
    street_key = re.sub(r'\s+', '', street_part)[:5] # Take first 5 non-space chars of the street

    return f"{postcode}_{street_key}"

def normalize_for_comparison(s):
    """Canonicalizes an address string by lowercasing and removing non-alphanumerics."""
    return re.sub(r'[^a-z0-9]', '', str(s).lower())

def parse_sales_history(sales_str):
    if pd.isna(sales_str) or sales_str == '' or sales_str == '[]':
        return pd.DataFrame(columns=['sale_date', 'sale_price'])
    try:
        sales_list = ast.literal_eval(str(sales_str))
        sales_data = []
        if not isinstance(sales_list, list):
            return pd.DataFrame(columns=['sale_date', 'sale_price'])
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
                    if price > 0:
                        sales_data.append({'sale_date': sale_date, 'sale_price': price})
                except ValueError:
                    continue
        # Ensure the returned DataFrame has the correct schema, even if empty.
        return pd.DataFrame(sales_data, columns=['sale_date', 'sale_price'])
    except (ValueError, SyntaxError):
        return pd.DataFrame(columns=['sale_date', 'sale_price'])

def process_rightmove_data(raw_rightmove_df):
    print("Processing Rightmove sales data to extract features...")
    processed_data = []
    
    # --- ARCHITECTURAL MANDATE: Enforce Positional Contract for Headerless Data ---
    # This file is headerless. We rely on a fixed column order: 0=Address, 1=URL, 2=Sales History.
    if raw_rightmove_df.shape[1] < 3:
        raise ValueError(
            f"FATAL: Rightmove data has fewer than 3 columns ({raw_rightmove_df.shape[1]}). "
            "The required positional contract (Address, URL, Sales History) is violated. "
            "Check the upstream scraping process."
        )
    
    address_col = 0
    sales_history_col = 2
    
    print(f"  - Using positional contract: address column index '{address_col}'")
    print(f"  - Using positional contract: sales history column index '{sales_history_col}'")

    for idx, row in raw_rightmove_df.iterrows():
        # --- NEW: Data Sanitization for list-like strings ---
        # This handles the case where the data is `['address text', 'url']`
        try:
            # Safely evaluate the string as a Python literal
            addr_val = ast.literal_eval(str(row[address_col]))
            # If it's a list and not empty, take the first element.
            clean_address = addr_val[0] if isinstance(addr_val, list) and addr_val else ''
        except (ValueError, SyntaxError):
            # If it's not a list-like string, use it as is.
            clean_address = str(row[address_col])

        sales_df = parse_sales_history(row[sales_history_col])
        if not sales_df.empty:
            sales_df = sales_df.sort_values('sale_date', ascending=False).reset_index(drop=True)
            most_recent = sales_df.iloc[0]
            features = {
                'rightmove_row_id': idx,
                'rightmove_address_text': clean_address, # Use the cleaned address
                'most_recent_sale_price': most_recent['sale_price'],
                'most_recent_sale_year': most_recent['sale_date'].year,
                'most_recent_sale_month': most_recent['sale_date'].month,
                'total_sales_count': len(sales_df)
            }
            if len(sales_df) > 1:
                previous_sale = sales_df.iloc[1]
                features['days_since_last_sale'] = (most_recent['sale_date'] - previous_sale['sale_date']).days
                features['price_change_since_last'] = most_recent['sale_price'] - previous_sale['sale_price']
            processed_data.append(features)
    print(f"Successfully processed {len(processed_data)} properties with valid sales data.")
    return pd.DataFrame(processed_data)


def generate_forecast_features(df, forecast_models, forecast_scalers, feature_sets, max_horizon=36, input_window=24):
    """
    [ARCHITECTURALLY CORRECTED] Generates DENSE monthly forecasts for each property
    using ONLY the data available at the time of its sale.
    """
    print(f"  - Generating dense, point-in-time correct monthly forecasts up to {max_horizon} months...")
    
    df_copy = df.copy() # Operate on a copy to prevent side-effects.

    # Define all property type columns for accurate mapping.
    prop_type_cols = {
        'F': 'num__property_main_type_encoded__1Flat_hm',
        'D': 'num__property_sub_type_code_from_homipi__4Detached_hm',
        'S': 'num__3SemiDetached_hm',
        'T': 'num__2Terraced_hm'
    }


    # Verify all necessary columns exist.
    for p_type, col_name in prop_type_cols.items():
        if col_name not in df_copy.columns:
            print(f"WARNING: Property type column '{col_name}' for type '{p_type}' not found. These properties may be misclassified.")
            # Create a dummy column of zeros to prevent a crash.
            df_copy[col_name] = 0

    # --- Pre-computation for all workers ---
    # CORRECTED: Use np.select for accurate multi-class assignment.
    conditions = [
        df_copy[prop_type_cols['F']] == 1,
        df_copy[prop_type_cols['S']] == 1,
        df_copy[prop_type_cols['T']] == 1,
    ]
    choices = ['F', 'S', 'T']
    df_copy['property_type_char'] = np.select(conditions, choices, default='D') # Default to Detached if no other type matches.

    p_types_np = df_copy['property_type_char'].to_numpy()
    sale_years_np = df_copy['most_recent_sale_year'].to_numpy()
    sale_months_np = df_copy['most_recent_sale_month'].to_numpy()
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
    forecasts_df = pd.DataFrame(results_matrix, index=df_copy.index)
    df_copy = pd.concat([df_copy, forecasts_df], axis=1)
    new_feature_names = forecasts_df.columns.tolist()
        
    print(f"  - Generated {len(new_feature_names)} new dense monthly forecast features.")
    if fallback_count > 0:
        print(f"  - NOTE: Used fallback date (August 2024) for {fallback_count} properties with missing sale dates.")
        
    return df_copy, new_feature_names

def create_head_params(params_dict, feature_configs):
    """Creates the head_params dictionary from a given parameter set (from Optuna trial or best_params)."""
    head_params = {}
    
    # --- GRANULAR HEAD CATEGORIZATION FOR TRIAL 8.5 ---
    # This list is our hypothesis about which heads contain the most dense signal.
    HIGH_SIGNAL_DENSITY_HEADS = [
        'head_B_aesthetic',          # Proven high-signal visual features
        'head_G_gemini_quantitative',# Proven high-signal quantitative features
        'head_A_dna_engineered'      # Contains high-level engineered scores
    ]
    
    # Define all other head types for clarity
    TABNET_HEADS = ['head_A_dna_raw', 'head_C_census', 'head_compass_raw']
    TCN_HEADS = [h for h in feature_configs if 'temporal' in h or h == 'head_E_temporal']
    
    # All other heads not in the above lists are considered "low signal density" MLPs
    all_categorized = set(HIGH_SIGNAL_DENSITY_HEADS + TABNET_HEADS + TCN_HEADS)
    LOW_SIGNAL_DENSITY_MLP_HEADS = [
        h for h in feature_configs 
        if h not in all_categorized and h not in ['lat_lon', 'head_base']
    ]

    for head_name in list(feature_configs.keys()):
        if head_name in LOW_SIGNAL_DENSITY_MLP_HEADS:
            head_params[head_name] = {
                'hidden_dim': params_dict['mlp_head_hidden_dim'],
                'dropout_rate': params_dict['mlp_head_dropout'],
                'output_dim': params_dict['low_signal_output_dim']
            }
        # ARCHITECTURAL MANDATE: Treat head_base as a privileged, high-capacity head.
        elif head_name == 'head_base':
            head_params[head_name] = {
                'hidden_dim': params_dict['base_head_hidden_dim'], # Use dedicated capacity param
                'dropout_rate': params_dict['mlp_head_dropout'],
                'output_dim': params_dict['high_signal_output_dim'] # Remove bottleneck
            }
        elif head_name in HIGH_SIGNAL_DENSITY_HEADS:
            head_params[head_name] = {
                'hidden_dim': params_dict['mlp_head_hidden_dim'],
                'dropout_rate': params_dict['visual_head_dropout'], # Use specific dropout for visual/high-signal
                'output_dim': params_dict['high_signal_output_dim']
            }
        elif head_name in TABNET_HEADS or head_name in TCN_HEADS:
            # For TabNet/TCN, their "native output" is the common fusion dimension,
            # as they are more complex and project directly.
            head_params[head_name] = {
                'output_dim': params_dict['fusion_embed_dim']
            }
        elif head_name in ['lat_lon']:
            continue
        else:
            # This check is now even more important.
            raise ValueError(f"Uncategorized head '{head_name}' in create_head_params.")
            
    return head_params

def mean_absolute_percentage_error(y_true, y_pred):
    """Calculates MAPE, ensuring robustness against division by zero and empty inputs."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Avoid division by zero for properties with a price of 0.
    mask = y_true != 0
    
    # FAIL-SAFE: If no valid (non-zero) true values exist, the error is 0.
    if not np.any(mask):
        return 0.0
        
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def parse_and_prepare_data_for_prop_type(df, cols):
    if not cols: return None, 0, 0
    parsed_data = {}
    # This pattern is now more general to capture various temporal features, not just price-paid.
    pattern = re.compile(r"(.*)_(\d{4})_(\d{2})_(.*)")
    
    feature_stems = set()
    for col in cols:
        match = pattern.match(col)
        if match:
            prefix, year, month, suffix = match.groups()
            timestep = (int(year), int(month))
            # ROBUST: Create a canonical stem by replacing the date part, not using simple string replacement.
            feature_stem = f"{prefix}_TIMESTAMP_{suffix}"
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
        local_data_dict = data_dict.copy()
        
        # ARCHITECTURALLY ROBUST: Explicitly handle all known non-feature keys.
        # This prevents metadata from being incorrectly treated as a feature tensor.
        msoa_df = local_data_dict.pop('msoa_id', None)
        if msoa_df is None:
            raise ValueError("'msoa_id' not found in data_dict for WisteriaDataset.")
        
        # Handle other potential non-feature keys if they exist.
        local_data_dict.pop('lat_lon', None)
        
        # Squeeze to handle both (N, 1) and (N,) shapes safely.
        self.msoa_id_tensor = torch.tensor(msoa_df.values, dtype=torch.long).squeeze()
        
        # What remains in local_data_dict is guaranteed to be feature data.
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
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.output_dim = output_dim

    def forward(self, x):
        x = self.fc1(x)
        # FAIL-SAFE: BatchNorm1d crashes on batch size 1. Skip it in that case.
        if x.shape[0] > 1:
            x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x



class HierarchicalAttention(nn.Module):
    """
    Learns to compute a context-aware weighted sum of head embeddings.
    """
    def __init__(self, embedding_dim, dropout=0.1):
        super(HierarchicalAttention, self).__init__()
        self.attention_network = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, head_outputs_tensor):
        """
        Args:
            head_outputs_tensor (torch.Tensor): (batch_size, num_heads, embedding_dim)
        Returns:
            torch.Tensor: context_vector (batch_size, embedding_dim)
            torch.Tensor: attention_weights (batch_size, num_heads, 1)
        """
        attention_scores = self.attention_network(head_outputs_tensor)
        attention_weights = self.softmax(attention_scores)
        weighted_heads = head_outputs_tensor * attention_weights
        context_vector = torch.sum(weighted_heads, dim=1)
        return context_vector, attention_weights


class EnhancedTCNHead(nn.Module):
    def __init__(self, input_dim, n_features_per_timestep, n_timesteps, num_channels, attention_hidden_dim, kernel_size=2, dropout=0.2):
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
            nn.Linear(tcn_output_channels, attention_hidden_dim),
            nn.Tanh(),
            nn.Linear(attention_hidden_dim, 1)
        )
        # Add LayerNorm for stability, a standard practice in attention architectures.
        self.norm = nn.LayerNorm(tcn_output_channels)
        
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
        
        # Apply LayerNorm for stability
        return self.norm(final_output)


class SpatialFusion(nn.Module):
    """
    [ARCHITECTURAL WARNING - ORPHANED MODULE]
    This module is defined but is NOT USED by the current `FusionModel` implementation,
    which uses simple concatenation instead. Its existence represents an architectural
    inconsistency that must be resolved if hierarchical attention is a project goal.
    """
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
    def __init__(self, feature_configs, msoa_cardinality, head_params, fusion_embed_dim, msoa_embedding_dim=16, fusion_dropout_rate=0.5, use_enhanced_tcn=True, attention_dropout_rate=0.1, tcn_dropout=0.3, stratum_name='monolith', mode='evidential', tabnet_n_steps=4, tabnet_gamma=1.5, fusion_mlp_depth='deep', attention_hidden_dim=64):
        super().__init__()
        self.mode = mode
        self.specialist_heads = nn.ModuleDict()
        self.projection_layers = nn.ModuleDict()

        # --- Store head names for two-level fusion ---
        self.temporal_head_names = []
        self.static_head_names = []

        tcn_class = EnhancedTCNHead if use_enhanced_tcn else TCNHead
        print(f"--- Building Trial 8.5 FusionModel ({stratum_name}, {self.mode} mode) ---")
        if fusion_mlp_depth == 'deep':
            fusion_mlp_dims = [1024, 512, 256]
            print("  - Using DEEP final fusion MLP:")
        else: # shallow
            fusion_mlp_dims = [512, 256]
            print("  - Using SHALLOW final fusion MLP:")

        for head_name, config in feature_configs.items():
            if 'input_dim' not in config or config['input_dim'] == 0:
                print(f"  - WARNING: Skipping head '{head_name}' due to zero input dimensions.")
                continue
            
            input_dim = config['input_dim']
            h_params = head_params.get(head_name)
            if not h_params:
                raise ValueError(f"FATAL: No head_params found for '{head_name}'. Configuration is incomplete.")
            
            # --- Differentiated Capacity: Use native output_dim from head_params ---
            native_output_dim = h_params['output_dim']
            
            if 'n_features_per_timestep' in config and 'n_timesteps' in config:
                print(f"  - Creating TCN head '{head_name}' -> native dim {native_output_dim}")
                head_module = tcn_class(
                    input_dim=input_dim,
                    n_features_per_timestep=config['n_features_per_timestep'],
                    n_timesteps=config['n_timesteps'],
                    num_channels=[64, native_output_dim], # Output of TCN is the native dim
                    attention_hidden_dim=attention_hidden_dim,
                    kernel_size=3, dropout=tcn_dropout
                )
            elif config.get('is_tabnet', False):
                print(f"  - Creating TabNet head '{head_name}' -> native dim {native_output_dim}")
                head_module = TabNetHead(
                    input_dim=input_dim, output_dim=native_output_dim,
                    n_d=32, n_a=32, n_steps=tabnet_n_steps, gamma=tabnet_gamma
                )
            else: # Standard MLP Head
                # ARCHITECTURAL MANDATE: Use the correct hidden_dim based on head type
                hidden_dim = h_params['hidden_dim']
                dropout = h_params['dropout_rate']
                print(f"  - Creating Simple MLP head '{head_name}' -> native dim {native_output_dim}")
                head_module = SimpleHead(input_dim, native_output_dim, hidden_dim=hidden_dim, dropout=dropout)

            # ARCHITECTURAL MANDATE: Unify all heads into a single council.
            self.specialist_heads[head_name] = head_module
            
            # --- Create projection layer if native dim differs from fusion dim ---
            if native_output_dim != fusion_embed_dim:
                self.projection_layers[head_name] = nn.Linear(native_output_dim, fusion_embed_dim)
                print(f"    - Added projection layer: {native_output_dim} -> {fusion_embed_dim}")
            
            # --- Categorize for two-level fusion ---
            if 'temporal' in head_name or 'head_E_temporal' in head_name:
                self.temporal_head_names.append(head_name)
            else:
                self.static_head_names.append(head_name)
        
        # --- Two-Level Attention Modules ---
        self.temporal_attention = HierarchicalAttention(embedding_dim=fusion_embed_dim, dropout=attention_dropout_rate)
        self.global_attention = HierarchicalAttention(embedding_dim=fusion_embed_dim, dropout=attention_dropout_rate)
        
        # --- Final Fusion MLP ---
        # ARCHITECTURALLY CORRECTED: The input is the MSOA embedding plus the single,
        # authoritative context vector from the global attention mechanism.
        fusion_mlp_input_dim = 0
        fusion_mlp_input_dim += msoa_embedding_dim
        if self.specialist_heads: # Check if any heads were created at all
             fusion_mlp_input_dim += fusion_embed_dim # The output of global attention

        self.msoa_embedding = nn.Embedding(msoa_cardinality, msoa_embedding_dim)
        print(f"--- Hierarchical Fusion: Final MLP input dimension: {fusion_mlp_input_dim} ---")
        
        fusion_layers = []
        current_dim = fusion_mlp_input_dim
        for h_dim in fusion_mlp_dims:
            fusion_layers.append(nn.Linear(current_dim, h_dim))
            fusion_layers.append(nn.ReLU())
            fusion_layers.append(nn.Dropout(fusion_dropout_rate))
            current_dim = h_dim
        self.fusion_mlp = nn.Sequential(*fusion_layers)

        self.evidential_head = nn.Linear(current_dim, 4)
        self.mae_head = nn.Linear(current_dim, 1)

    def forward(self, x):
        # --- Process all specialist heads (including head_base) and apply projections ---
        projected_outputs = {}
        for name, module in self.specialist_heads.items():
            if name in x:
                native_output = module(x[name])
                if name in self.projection_layers:
                    projected_outputs[name] = self.projection_layers[name](native_output)
                else:
                    projected_outputs[name] = native_output

        # --- LEVEL 1: Temporal Fusion ---
        temporal_context_vector = None
        if self.temporal_head_names:
            temporal_outputs = [projected_outputs[name] for name in self.temporal_head_names if name in projected_outputs]
            if temporal_outputs:
                temporal_tensor = torch.stack(temporal_outputs, dim=1)
                temporal_context_vector, _ = self.temporal_attention(temporal_tensor)

        # --- LEVEL 2: Global Fusion (now includes head_base) ---
        global_attention_inputs = []
        # Add static head outputs
        static_outputs = [projected_outputs[name] for name in self.static_head_names if name in projected_outputs]
        global_attention_inputs.extend(static_outputs)
        # Add the fused temporal vector as a single "super-expert"
        if temporal_context_vector is not None:
            global_attention_inputs.append(temporal_context_vector)
        
        global_context_vector = None
        if global_attention_inputs:
            global_tensor = torch.stack(global_attention_inputs, dim=1)
            global_context_vector, _ = self.global_attention(global_tensor)

        # --- Final Concatenation and MLP ---
        msoa_embed = self.msoa_embedding(x['msoa_id'])

        final_concat_list = []
        final_concat_list.append(msoa_embed)
        if global_context_vector is not None:
            final_concat_list.append(global_context_vector)

        if len(final_concat_list) < 2: # Should contain at least msoa and global context
             raise RuntimeError("FATAL: Not enough heads produced output for the final fusion MLP.")

        final_fused = torch.cat(final_concat_list, dim=1)
        final_representation = self.fusion_mlp(final_fused)
        
        mae_pred = self.mae_head(final_representation)

        if self.mode == 'point_prediction':
            return mae_pred

        evidential_output = self.evidential_head(final_representation)
        gamma, nu, alpha, beta = torch.split(evidential_output, 1, dim=-1)
        
        nu = F.softplus(nu) + 1e-6; alpha = F.softplus(alpha) + 1.0; beta = F.softplus(beta) + 1e-6
        final_evidential_params = torch.cat([gamma, nu, alpha, beta], dim=-1)
        return final_evidential_params, mae_pred

class ShapModelWrapper(nn.Module):
    """
    An nn.Module wrapper to make the FusionModel compatible with SHAP's DeepExplainer.
    SHAP's explainer passes a list of tensors, but our model expects a dictionary.
    This module acts as an adapter to perform that translation.
    """
    def __init__(self, model, ordered_keys):
        super().__init__()
        self.model = model
        self.ordered_keys = ordered_keys

    def forward(self, *tensors):
        # Reconstruct the dictionary from the ordered keys and the list of tensors
        input_dict = dict(zip(self.ordered_keys, tensors))
        
        # Ensure msoa_id is the correct type (long) and shape for the embedding layer
        if 'msoa_id' in input_dict:
            # The embedding layer expects a 1D tensor of indices.
            # SHAP provides a 2D tensor of shape [batch_size, 1].
            # We must cast to long and then squeeze to remove the singleton dimension.
            input_dict['msoa_id'] = input_dict['msoa_id'].long().squeeze(-1)
            
        # Call the actual model with the correctly formatted input
        evidential_params, _ = self.model(input_dict)
        
        # SHAP explains a single output; we choose gamma (the point prediction).
        # CRITICAL: We must return a 2D tensor of shape [batch_size, 1], not a 1D tensor.
        # Slicing with [:, 0:1] preserves the dimension, whereas [:, 0] would collapse it.
        return evidential_params[:, 0:1]


class AsymmetricEvidentialLoss(nn.Module):
    def __init__(self, coeff=1.0, undervalue_penalty=1.5, epsilon=1e-7):
        super(AsymmetricEvidentialLoss, self).__init__()
        self.coeff = coeff
        self.undervalue_penalty = undervalue_penalty
        self.epsilon = epsilon # Defensive epsilon for numerical stability

    def forward(self, pred, target):
        gamma, nu, alpha, beta = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        
        # Negative Log-Likelihood with defensive epsilon for stability
        two_beta_lambda = 2 * beta * (1 + nu)
        nll = 0.5 * torch.log(np.pi / (nu + self.epsilon)) \
            - alpha * torch.log(two_beta_lambda + self.epsilon) \
            + (alpha + 0.5) * torch.log(nu * (target - gamma)**2 + two_beta_lambda + self.epsilon) \
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

def train_autoencoder_for_head(df_raw, cols, head_name, ae_params, device):
    """
    Trains a dedicated Autoencoder for a specific feature head.
    """
    print(f"\n--- Training Autoencoder for '{head_name}' ---")
    
    # 1. Data Preparation
    available_cols = [c for c in cols if c in df_raw.columns]
    if not available_cols:
        print(f"  - No columns found for '{head_name}'. Skipping AE training.")
        return None
        
    head_df = df_raw[available_cols].copy().apply(pd.to_numeric, errors='coerce').fillna(0)
    
    # Use a simple scaler internally for AE training stability
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(head_df)
    
    # 2. Dataset and DataLoader
    tensor_dataset = torch.utils.data.TensorDataset(torch.tensor(data_scaled, dtype=torch.float32))
    loader = torch.utils.data.DataLoader(tensor_dataset, batch_size=512, shuffle=True)
    
    # 3. Model Initialization
    model = Autoencoder(
        input_dim=len(available_cols),
        latent_dim=ae_params['latent_dim'],
        hidden_dim_1=ae_params['hidden_dim_1'],
        hidden_dim_2=ae_params['hidden_dim_2']
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=ae_params.get('lr', 0.001))
    criterion = nn.MSELoss()
    
    # 4. Training Loop
    print(f"  - Training for 50 epochs with latent_dim={ae_params['latent_dim']}...")
    model.train()
    for epoch in range(50): # A fixed, short training schedule is sufficient for preprocessing
        for data in loader:
            inputs = data[0].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
    
    print(f"  - AE for '{head_name}' trained successfully.")
    
    # Return a dictionary containing the trained model and the scaler needed for inference
    return {'model': model.cpu(), 'scaler': scaler, 'latent_dim': ae_params['latent_dim']}


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
        
        self.initial_bn = nn.BatchNorm1d(input_dim, momentum=0.01)
        self.feature_transformer = FeatureTransformer(input_dim, n_d + n_a, n_shared, n_independent)
        self.attentive_transformers = nn.ModuleList()
        for _ in range(n_steps):
            self.attentive_transformers.append(AttentiveTransformer(n_a, input_dim))
        
        # CORRECTED: Initialize dropout layer
        self.dropout_layer = nn.Dropout(dropout)
        self.final_map = nn.Linear(n_d, output_dim)

    def forward(self, x):
        # FAIL-SAFE: BatchNorm1d crashes on batch size 1. Skip it in that case.
        if x.shape[0] > 1:
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
            
            # CORRECTED: Apply dropout to regularize the decision step contribution
            d_step = self.dropout_layer(d_step)
            
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


def train_one_epoch(model, loader, optimizer, loss_fn, mode='evidential', loss_weights={'evidential': 0.5, 'mae': 0.5}):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        target = batch['target_price']
        optimizer.zero_grad()
        
        if mode == 'point_prediction':
            mae_pred = model(batch)
            loss = loss_fn(mae_pred, target)
        else: # 'evidential' mode
            evidential_preds, mae_pred = model(batch)
            evidential_loss = loss_fn(evidential_preds, target.squeeze(-1))
            mae_loss = F.l1_loss(mae_pred, target)
            loss = (loss_weights['evidential'] * evidential_loss) + (loss_weights['mae'] * mae_loss)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, mode='evidential'):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            if mode == 'point_prediction':
                preds = model(batch)
                all_preds.append(preds.cpu())
            else: # 'evidential' mode
                evidential_params, _ = model(batch)
                all_preds.append(evidential_params.cpu())

    if not all_preds:
        # Return empty tensor with correct final dimension based on mode
        return torch.empty(0, 1) if mode == 'point_prediction' else torch.empty(0, 4)

    return torch.cat(all_preds)

def find_msoa_column(df):
    """
    [ARCHITECTURALLY HARDENED v2] Finds the MSOA column with extreme prejudice, prioritizing official codes.
    This function is hardened against selecting ambiguous or incorrect columns.
    """
    # --- PRIORITY 1: EXACT MATCH for official ONS codes (case-insensitive) ---
    official_codes = ['MSOA11CD', 'MSOA21CD']
    for col in df.columns:
        if col.upper() in official_codes:
            print(f"Found official MSOA code column by exact match: '{col}'")
            return col

    # --- PRIORITY 2: SUBSTRING MATCH for official codes ---
    # Catches variations like 'property_MSOA11CD'
    for code in official_codes:
        candidates = [c for c in df.columns if code in c.upper()]
        if candidates:
            print(f"Found official MSOA code column by substring match: '{candidates}'")
            return candidates

    # --- PRIORITY 3 (LAST RESORT): Fallback to a generic name ---
    # This is now the source of the fallback warning. It is much stricter than before.
    fallback_candidates = [c for c in df.columns if 'msoa' in c.lower() and 'code' in c.lower()]
    if fallback_candidates:
        print(f"  - WARNING: No official ONS MSOA code column found. Falling back to candidate: '{fallback_candidates}'.")
        print(f"  - Please verify this is the correct categorical MSOA identifier and not a metric.")
        return fallback_candidates[0]

    # --- FINAL: Fail Fast ---
    # If no high-confidence column is found, abort. Do not guess.
    raise ValueError(
        "FATAL: No MSOA column found. Could not identify a column with 'MSOA11CD', 'MSOA21CD', or containing 'msoa_code'."
    )

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
                tcn_array[:, j, i] = df[col_name].fillna(0).values

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
                tcn_array[:, j, i] = df[col_name].fillna(0).values

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


def fit_and_create_canonical_artifacts(df, feature_sets, n_components_map, universal_cols_present, head_config, trained_autoencoders):
    """
    [AUTHORITATIVE REFACTORED] Creates a single, comprehensive artifact file containing all
    pre-fitted objects (Autoencoders, Scalers) for ALL head types using a unified loop.
    """
    print("\n--- STAGE 4.5: Fitting/Assembling Unified Global Preprocessors ---")
    canonical_artifacts = {}

    # --- A SINGLE, UNIFIED LOOP TO PROCESS ALL HEAD TYPES ---
    all_head_names = list(feature_sets.keys()) + ['head_base']
    
    for head_name in all_head_names:
        if head_name == 'head_base':
            cols = universal_cols_present
            head_type = 'mlp' # The base head is a simple MLP
        else:
            cols = feature_sets.get(head_name, [])
            head_type = head_config.get(head_name, {}).get('type')

        available_cols = [c for c in cols if c in df.columns]
        if not available_cols:
            print(f"  - WARNING: Skipping head '{head_name}' due to zero available columns.")
            continue

        # --- ROUTE 1: MLP Heads Requiring a Pre-Trained Autoencoder ---
        if head_type == 'mlp' and head_name in trained_autoencoders:
            ae_artifact = trained_autoencoders[head_name]
            print(f"  - [MLP/AE] Storing pre-trained Autoencoder for '{head_name}': {len(available_cols)} -> {ae_artifact['latent_dim']} latent dims.")
            canonical_artifacts[head_name] = {
                'type': 'autoencoder',
                'autoencoder': ae_artifact['model'],
                'scaler': ae_artifact['scaler'],
                'original_cols': available_cols,
                'input_dim': ae_artifact['latent_dim']
            }
        
        # --- ROUTE 2: PRE-EMBEDDED Heads (NO SCALER) ---
        # These features are already perfected embeddings and must not be transformed.
        elif head_type == 'pre_embedded':
            print(f"  - [PRE-EMBEDDED] Registering '{head_name}' without transformation.")
            canonical_artifacts[head_name] = {
                'type': 'pre_embedded',
                'scaler': None, # EXPLICITLY NO SCALER
                'raw_cols': available_cols,
                'input_dim': len(available_cols)
            }

        # --- ROUTE 3: All Other Heads (TCN, TabNet, and Base MLP) ---
        # These heads require a scaler to be fit on their raw features.
        elif head_type in ['tcn', 'tabnet', 'mlp']:
            print(f"  - [{head_type.upper()}] Fitting Scaler for '{head_name}'...")
            
            # The TCN head has a special requirement to filter for parseable columns first.
            if head_type == 'tcn':
                parsed_cols = {}
                parsed_and_valid_cols = []
                spatio_pattern = re.compile(r"^(.*)_(\d{4})_(.*)$")
                temporal_pattern = re.compile(r"^(\d{4})_(.*)$")
                for col in available_cols:
                    year, feature_stem = None, None
                    match_spatio = spatio_pattern.match(col)
                    match_temporal = temporal_pattern.match(col)
                    if match_spatio:
                        prefix, year, suffix = match_spatio.groups(); feature_stem = f"{prefix}_{suffix}"
                    elif match_temporal:
                        year, feature_stem = match_temporal.groups()
                    if feature_stem:
                        if feature_stem not in parsed_cols: parsed_cols[feature_stem] = {}
                        parsed_cols[feature_stem][year] = col
                        parsed_and_valid_cols.append(col)
                
                unparsed_count = len(available_cols) - len(parsed_and_valid_cols)
                if unparsed_count > 0:
                    print(f"    - WARNING: Dropped {unparsed_count} unparseable columns from '{head_name}'.")
                
                # Use the filtered list for fitting.
                cols_for_fitting = parsed_and_valid_cols
            else:
                # All other heads in this route use all their available columns.
                cols_for_fitting = available_cols

            head_df_for_fitting = df[cols_for_fitting].copy().apply(pd.to_numeric, errors='coerce').fillna(0)
            scaler = QuantileTransformer(output_distribution='normal', n_quantiles=max(min(len(df) // 10, 100), 10), random_state=42).fit(head_df_for_fitting)
            
            artifact = {
                'type': head_type, 
                'scaler': scaler, 
                'raw_cols': cols_for_fitting,
                'input_dim': len(cols_for_fitting)
            }

            if head_type == 'tcn':
                feature_stems = sorted(list(parsed_cols.keys()))
                all_years = sorted(list(set(y for f in parsed_cols.values() for y in f.keys())))
                n_timesteps = len(all_years) if all_years else 1
                n_features = len(feature_stems) if feature_stems else 1
                artifact['n_timesteps'] = n_timesteps
                artifact['n_features_per_timestep'] = n_features
                print(f"    - Determined TCN structure: {n_timesteps} timesteps, {n_features} features/step.")
            
            canonical_artifacts[head_name] = artifact
        
        else:
                print(f"  - NOTE: Head '{head_name}' with type '{head_type}' has no defined processing path and will be skipped.")

    return canonical_artifacts

def transform_with_canonical_artifacts(df, artifacts):
    """
    [ARCHITECTURALLY ROBUST] Transforms a raw dataframe into a fully preprocessed one
    suitable for model input, using the single canonical artifact file.
    """
    df_transformed = df.copy()
    feature_configs = {}

    for head_name, config in artifacts.items():
        # This logic applies to all head types that use a scaler on raw columns.
        if 'original_cols' in config or 'raw_cols' in config:
            expected_cols = config.get('original_cols') or config.get('raw_cols')
            
            # FAIL-SAFE: Handle missing columns gracefully.
            available_cols = [col for col in expected_cols if col in df.columns]
            missing_cols = set(expected_cols) - set(available_cols)
            
            head_df_raw = df[available_cols].copy().apply(pd.to_numeric, errors='coerce').fillna(0)
            
            # Add back any missing columns with a neutral fill value of 0.
            if missing_cols:
                for col in missing_cols:
                    head_df_raw[col] = 0
            
            # Ensure column order matches what the scaler was fitted on.
            head_df_raw = head_df_raw[expected_cols]

            if config['type'] == 'pca':
                head_df_scaled = config['scaler'].transform(head_df_raw)
                head_pca_result = config['pca'].transform(head_df_scaled)
                
                pca_cols = [f"pca_{head_name}_{i}" for i in range(head_pca_result.shape[1])]
                df_transformed = pd.concat([df_transformed, pd.DataFrame(head_pca_result, columns=pca_cols, index=df.index)], axis=1)
                feature_configs[head_name] = {'input_dim': head_pca_result.shape[1], 'pca_cols': pca_cols}

            elif config['type'] == 'autoencoder':
                head_df_scaled = config['scaler'].transform(head_df_raw)
                with torch.no_grad():
                    latent_vector = config['autoencoder'].encode(torch.tensor(head_df_scaled, dtype=torch.float32))
                
                ae_cols = [f"ae_{head_name}_{i}" for i in range(latent_vector.shape[1])]
                df_transformed = pd.concat([df_transformed, pd.DataFrame(latent_vector.numpy(), columns=ae_cols, index=df.index)], axis=1)
                # The 'raw_cols' key here now points to the NEWLY created AE feature columns
                feature_configs[head_name] = {'input_dim': latent_vector.shape[1], 'raw_cols': ae_cols}

            elif config['type'] in ['tcn', 'tabnet', 'mlp', 'pre_embedded']:
                # For pre-embedded heads, the scaler is None, so we just use the raw data.
                if config['scaler'] is not None:
                    head_df_scaled = config['scaler'].transform(head_df_raw)
                    df_transformed[expected_cols] = head_df_scaled
                else: # This path is for pre_embedded
                    df_transformed[expected_cols] = head_df_raw
                
                # ARCHITECTURALLY CRITICAL: Propagate all necessary metadata for model construction.
                head_feature_config = {'input_dim': len(expected_cols), 'raw_cols': expected_cols}
                if config['type'] == 'tabnet':
                    head_feature_config['is_tabnet'] = True
                if config['type'] == 'tcn':
                    head_feature_config['n_timesteps'] = config['n_timesteps']
                    head_feature_config['n_features_per_timestep'] = config['n_features_per_timestep']
                
                feature_configs[head_name] = head_feature_config

    # Add pass-through configs for model constructor
    feature_configs['lat_lon'] = {}

    return df_transformed, feature_configs





def validate_merge_quality(df, sample_size=10):
    """Validate that merged properties actually match by comparing original addresses."""
    print(f"\n--- MERGE VALIDATION (Sample of {sample_size}) ---")
    
    # The merge operation now provides an unambiguous column name.
    address_col_rm = 'rightmove_address_text'
    if address_col_rm not in df.columns:
        raise ValueError(f"FATAL: The canonical address column '{address_col_rm}' was not found after the merge.")

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




def objective(trial, data_train, y_train, data_val, y_val, feature_configs, msoa_cardinality, y_price_std, y_price_mean, df_val_raw, mode='evidential', base_params=None):
    print(f"\n--- Starting Trial {trial.number} for Optuna Phase: {mode.upper()} ---")
    
    if mode == 'point_prediction':
        trial_params = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 5e-4, log=True),
            'weight_decay': trial.suggest_float('weight_decay', 1e-4, 1e-2, log=True),
            'fusion_dropout_rate': trial.suggest_float('fusion_dropout_rate', 0.3, 0.5),
            'mlp_head_dropout': trial.suggest_float('mlp_head_dropout', 0.15, 0.35), # Consolidated MLP dropout
            'visual_head_dropout': trial.suggest_float('visual_head_dropout', 0.3, 0.5), # Keep separate for visual heads
            'tcn_dropout': trial.suggest_float('tcn_dropout', 0.2, 0.4),
            'batch_size': trial.suggest_categorical('batch_size', [128, 256, 512]),
            'fusion_mlp_depth': trial.suggest_categorical('fusion_mlp_depth', ['shallow', 'deep']),

            # --- REFINED CAPACITY PARAMETERS FOR TRIAL 8.5 ---
            # 1. Widen the search space for critical embedding dimensions
            'fusion_embed_dim': trial.suggest_categorical('fusion_embed_dim', [128, 192, 256]),
            'msoa_embedding_dim': trial.suggest_categorical('msoa_embedding_dim', [16, 24, 32]),

            # 2. ARCHITECTURAL MANDATE: Re-introduce granular capacity control.
            'base_head_hidden_dim': trial.suggest_categorical('base_head_hidden_dim', [256, 384, 512]),
            'mlp_head_hidden_dim': trial.suggest_categorical('mlp_head_hidden_dim', [128, 256]),

            # 3. Granular control over NATIVE output dimensions
            'high_signal_output_dim': trial.suggest_categorical('high_signal_output_dim', [96, 128, 160]),
            'low_signal_output_dim': trial.suggest_categorical('low_signal_output_dim', [32, 48, 64]),
            
            # 4. Widen the TabNet search space again
            'tabnet_n_steps': trial.suggest_int('tabnet_n_steps', 5, 10),
            'tabnet_gamma': trial.suggest_float('tabnet_gamma', 1.2, 2.0),
            'attention_hidden_dim': trial.suggest_categorical('attention_hidden_dim', [32, 64]),
        }
        loss_fn = F.l1_loss
    else: # 'evidential' mode
        if base_params is None:
            raise ValueError("base_params must be provided for 'evidential' mode tuning.")
        trial_params = base_params.copy()
        trial_params.update({
            'evidential_reg_coeff': trial.suggest_float('evidential_reg_coeff', 0.05, 0.25),
            'undervalue_penalty': trial.suggest_float('undervalue_penalty', 1.2, 2.0),
            'evidential_weight': trial.suggest_float('evidential_weight', 0.4, 0.6),
        })
        loss_fn = AsymmetricEvidentialLoss(
            coeff=trial_params['evidential_reg_coeff'],
            undervalue_penalty=trial_params['undervalue_penalty']
        )

    head_params = create_head_params(trial_params, feature_configs)
    train_dataset = WisteriaDataset(data_train, y_train, y_train)
    val_dataset = WisteriaDataset(data_val, y_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=trial_params['batch_size'], shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=trial_params['batch_size'] * 2, shuffle=False, num_workers=2, pin_memory=True)

    model = FusionModel(
        feature_configs=feature_configs, msoa_cardinality=msoa_cardinality, head_params=head_params,
        fusion_embed_dim=trial_params['fusion_embed_dim'], msoa_embedding_dim=trial_params['msoa_embedding_dim'],
        fusion_dropout_rate=trial_params['fusion_dropout_rate'], use_enhanced_tcn=True,
        tcn_dropout=trial_params['tcn_dropout'], stratum_name='monolith', mode=mode,
        tabnet_n_steps=trial_params.get('tabnet_n_steps', 4),
        tabnet_gamma=trial_params.get('tabnet_gamma', 1.5),
        fusion_mlp_depth=trial_params.get('fusion_mlp_depth', 'deep'),
        attention_hidden_dim=trial_params.get('attention_hidden_dim', 64)
    ).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=trial_params['learning_rate'], weight_decay=trial_params['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=5)
    
    best_mae = float('inf')
    patience_counter = 0
    patience = 15
    loss_weights = {
        'evidential': trial_params.get('evidential_weight', 0.5),
        'mae': 1.0 - trial_params.get('evidential_weight', 0.5)
    }

    for epoch in range(EPOCHS):
        _ = train_one_epoch(model, train_loader, optimizer, loss_fn, mode=mode, loss_weights=loss_weights)
        
        preds_val = evaluate(model, val_loader, mode=mode)
        point_preds_scaled = preds_val.numpy()[:, 0]
        point_preds_unscaled = np.expm1(point_preds_scaled * y_price_std + y_price_mean)
        true_values = df_val_raw['most_recent_sale_price'].values
        val_mae = mean_absolute_error(true_values, point_preds_unscaled)
        
        scheduler.step(val_mae)

        if val_mae < best_mae:
            best_mae = val_mae
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(f"  Trial {trial.number}, Epoch {epoch+1}, Val MAE: £{val_mae:,.2f} (Best: £{best_mae:,.2f})")

        if patience_counter >= patience:
            print(f"  - Early stopping trial {trial.number} after {epoch+1} epochs.")
            break

        trial.report(val_mae, epoch)
        if trial.should_prune():
            del model, optimizer, scheduler; gc.collect(); torch.cuda.empty_cache()
            raise optuna.exceptions.TrialPruned()

    del model, optimizer, scheduler; gc.collect(); torch.cuda.empty_cache()
    return best_mae


def find_best_matches_fuzzy(keys1, keys2, threshold=85):
    """
    Finds the best fuzzy match for each key in keys1 from keys2.
    
    ARCHITECTURAL NOTE: This is a high-complexity O(N*M) operation and can be a
    significant performance bottleneck for large key sets.
    """
    print(f"  - Attempting to find fuzzy matches for {len(keys1)} keys...")
    
    # FAIL-SAFE: If the target key set is empty, no matches can be found.
    if not keys2:
        print("  - WARNING: The target key set (keys2) is empty. No fuzzy matches possible.")
        return []
        
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
    [VECTORIZED] Detects leakage by checking for near-perfect correlation.
    """
    print("\n--- STAGE 4.1: Mitigating AVM Target Leakage (Correlation Check) ---")
    avm_keywords = ['estimate', 'valuation', 'avm', 'homipi', 'zoopla', 'bnl', 'bricks', 'chimnie']
    df_copy = df.copy()

    if label_col not in df_copy.columns:
        print(f"  - FATAL: Label column '{label_col}' not found. Cannot perform leakage check.")
        return df

    # 1. Identify all potential AVM columns at once.
    candidate_cols = [
        col for col in df_copy.select_dtypes(include=np.number).columns
        if any(keyword in col.lower() for keyword in avm_keywords)
    ]
    if not candidate_cols:
        print("  - No potential AVM feature columns found to check.")
        return df_copy

    # 2. Calculate correlation for all candidates in a single vectorized operation.
    correlations = df_copy[candidate_cols].corrwith(df_copy[label_col])
    
    # 3. Identify leaky columns based on the threshold.
    leaky_cols = correlations[correlations.abs() > correlation_threshold].index.tolist()

    if not leaky_cols:
        print("  - No significant correlation leakage detected.")
        return df_copy

    print("  [!! LEAKAGE DETECTED !!] The following columns will be mitigated:")
    for col in leaky_cols:
        print(f"    - '{col}' (Correlation: {correlations[col]:.4f})")
        col_std = df_copy[col].std()
        if pd.notna(col_std) and col_std > 0:
            noise = np.random.normal(0, noise_level * col_std, size=len(df_copy))
            df_copy.loc[df_copy[col].notna(), col] += noise
            print(f"      - Added noise (std dev: {noise_level * col_std:.4f}).")
        else:
            print(f"      - Could not add noise due to zero or NaN standard deviation.")
            
    return df_copy

def _temporal_worker(df_row, history_df_dict):
    """Worker function for parallel temporal feature engineering."""
    sale_year = df_row['most_recent_sale_year']
    new_features = {}
    for stem, history_df in history_df_dict.items():
        available_years = [y for y in history_df.columns if y <= sale_year]
        if len(available_years) < 2: continue

        property_history = history_df.loc[df_row.name, available_years].values.flatten()
        years_for_fit = np.array(available_years).flatten()
        
        finite_mask = np.isfinite(property_history)
        property_history = property_history[finite_mask]
        years_for_fit = years_for_fit[finite_mask]
        
        if len(property_history) < 2: continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", np.RankWarning)
            new_features[f"ts_summary_trend_{stem}"] = np.polyfit(years_for_fit, property_history, 1)[0]
        new_features[f"ts_summary_mean_{stem}"] = np.mean(property_history)
        new_features[f"ts_summary_std_{stem}"] = np.std(property_history)
    return new_features

def engineer_temporal_summary_features(df, feature_sets):
    """
    [VECTORIZED] Performs point-in-time correct longitudinal compression.
    For each temporal feature stem, this function calculates trend, mean, and std dev
    using optimized array operations, avoiding slow row-wise loops.
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

    # --- 1. Parse all temporal columns to group them by feature stem ---
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
            # Store year as integer for correct sorting and comparison
            feature_stems[stem].append({'year': int(year), 'col': col})

    df_copy = df.copy()
    new_feature_names = []
    sale_years = df_copy['most_recent_sale_year'].to_numpy()

    print(f"  - Found {len(feature_stems)} unique temporal feature stems to process.")
    # --- 2. Iterate through FEATURE STEMS (efficient) instead of rows (inefficient) ---
    for stem, year_cols in tqdm(feature_stems.items(), desc="Engineering Temporal Features"):
        
        # Create a dataframe for just this feature's history
        history_df = pd.DataFrame({info['year']: df_copy[info['col']] for info in year_cols}).sort_index(axis=1)
        history_years = history_df.columns.to_numpy()
        history_values = history_df.to_numpy()

        # --- 3. Create a point-in-time correct MASK ---
        # This mask is the key: for each property, it's True only for years <= its sale year.
        mask = history_years <= sale_years[:, np.newaxis]
        history_values[~mask] = np.nan # Invalidate future data

        # --- 4. Perform vectorized calculations ---
        with warnings.catch_warnings():
            # The specific np.RankWarning is deprecated. The context manager will
            # handle any warnings generated by polyfit without needing to be specific.
            warnings.simplefilter("ignore")
            # Calculate mean and std dev, ignoring NaNs (future data)
            means = np.nanmean(history_values, axis=1)
            stds = np.nanstd(history_values, axis=1)
            
            # Trend is more complex, requires a helper function to apply efficiently
            def calculate_trend(row_values, years):
                finite_mask = np.isfinite(row_values)
                if finite_mask.sum() < 2:
                    return 0.0
                return np.polyfit(years[finite_mask], row_values[finite_mask], 1)[0]
            
            trends = np.apply_along_axis(calculate_trend, 1, history_values, years=history_years)

        # --- 5. Add new features to the dataframe ---
        trend_col_name = f"ts_summary_trend_{stem}"
        mean_col_name = f"ts_summary_mean_{stem}"
        std_col_name = f"ts_summary_std_{stem}"
        
        df_copy[trend_col_name] = trends
        df_copy[mean_col_name] = means
        df_copy[std_col_name] = stds
        new_feature_names.extend([trend_col_name, mean_col_name, std_col_name])

    print(f"  - Successfully created {len(new_feature_names)} new point-in-time correct temporal summary features.")
    
    return df_copy, new_feature_names

def generate_final_report(eval_df, holdout_results_df, baseline_mae, output_dir):
    """
    [ROBUST] Generates a comprehensive, MAE-centric report with text and plots.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    print("\n--- GENERATING FINAL PERFORMANCE & ANALYSIS REPORT ---")
    report_path = os.path.join(output_dir, "final_performance_report.txt")

    if holdout_results_df.empty or not all(c in holdout_results_df.columns for c in ['most_recent_sale_price', 'predicted_price', 'absolute_error']):
        print("  - WARNING: Holdout results are empty or missing required columns. Skipping final report generation.")
        with open(report_path, "w") as f:
            f.write("Holdout results were empty or malformed. No report could be generated.\n")
        return

    # --- Calculate Key Metrics (MAE-centric) ---
    holdout_mae = holdout_results_df['absolute_error'].mean()
    oof_mae = mean_absolute_error(eval_df['most_recent_sale_price'], eval_df['final_predicted_price'])
    
    with open(report_path, "w") as f:
        f.write("=====================================================\n")
        f.write("=== Wisteria Valuation Model Performance Report ===\n")
        f.write("=====================================================\n\n")

        f.write("--- 1. Executive Summary (MAE-Centric) ---\n")
        f.write(f"Final Validated Holdout Set MAE: £{holdout_mae:,.2f}\n")
        f.write(f"LightGBM Baseline MAE:           £{baseline_mae:,.2f}\n")
        performance_delta = (baseline_mae - holdout_mae)
        f.write(f"Performance vs. Baseline:        £{performance_delta:+,.2f} improvement\n")
        f.write("Note: MAE (Mean Absolute Error) is the average absolute difference in pounds between prediction and sale price. Lower is better.\n\n")

        f.write("--- 2. Generalization & Overfitting Check ---\n")
        f.write(f"Out-of-Fold (OOF) MAE on Training Set: £{oof_mae:,.2f}\n")
        f.write(f"True Holdout Set MAE:                  £{holdout_mae:,.2f}\n")
        generalization_gap = (holdout_mae - oof_mae) / (oof_mae + 1e-9)
        f.write(f"Generalization Gap:                    {generalization_gap:+.2%}\n")
        f.write("Note: A small positive gap (<15%) indicates the model is generalizing well and not overfitting.\n\n")

        f.write("--- 3. Business Accuracy Tiers (Holdout Set Performance) ---\n")
        for tier in [0.05, 0.10, 0.15, 0.25]:
            within_tier_count = (holdout_results_df['absolute_error'] / holdout_results_df['most_recent_sale_price'] <= tier).sum()
            percentage = (within_tier_count / len(holdout_results_df)) * 100
            f.write(f"Percentage of valuations within {tier:.0%} of sale price: {percentage:.1f}%\n")
        f.write("\n")

    # --- 4. Visualizations (Unchanged but still relevant) ---
    plt.figure(figsize=(10, 6))
    sns.histplot(eval_df['final_predicted_price'] - eval_df['most_recent_sale_price'], bins=50, kde=True)
    plt.title('Distribution of Prediction Errors (OOF Set)', fontsize=16)
    plt.xlabel('Prediction Error (Predicted - Actual) in Pounds')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "error_distribution.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    if not eval_df.empty and 'epistemic_uncertainty' in eval_df.columns:
        plot_df = eval_df.sample(min(2000, len(eval_df)))
        plot_df['abs_error'] = (plot_df['final_predicted_price'] - plot_df['most_recent_sale_price']).abs()
        sns.regplot(data=plot_df, x='epistemic_uncertainty', y='abs_error', scatter_kws={'alpha':0.2}, line_kws={'color':'red'})
    
    plt.title('Model Error vs. Epistemic Uncertainty (OOF Set)', fontsize=16)
    plt.xlabel('Epistemic Uncertainty (Model "Confusion")')
    plt.ylabel('Absolute Prediction Error (Pounds)')
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
        # Define all patterns to check against
        patterns = [
            re.compile(r".*_(\d{4})_(\d{2})_.*"), # Monthly data
            re.compile(r".*_(\d{4})_.*")         # Yearly data
        ]
        
        for col in all_cols_for_type:
            col_year, col_month = 0, 1 # Default to Jan for year-only patterns
            
            # Robustly check all patterns for a match
            for pattern in patterns:
                match = pattern.match(col)
                if match:
                    groups = match.groups()
                    col_year = int(groups[0])
                    if len(groups) > 1:
                        col_month = int(groups[1])
                    break # Use the first pattern that matches
            
            if col_year > 0 and (col_year < sale_year or (col_year == sale_year and col_month < sale_month)):
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


def train_final_model(df_main_transformed, y_main_raw, feature_configs, price_stats, msoa_config, universal_cols_present, LAT_LON_COLS, n_trials):
    print(f"\n{'='*20} TRAINING FINAL Concatenated-Experts MODEL {'='*20}")
    
    y_price_mean, y_price_std = price_stats['y_price_mean'], price_stats['y_price_std']
    msoa_col_name, msoa_cardinality = msoa_config['column_name'], msoa_config['cardinality']
    y_price_log = np.log1p(y_main_raw)
    y_price_for_training = pd.Series((y_price_log - y_price_mean) / y_price_std, index=y_main_raw.index)

    # --- Data Preparation for Optuna (done once) ---
    train_idx_opt, val_idx_opt = train_test_split(df_main_transformed.index, test_size=0.2, random_state=42)
    df_train_opt, df_val_opt = df_main_transformed.loc[train_idx_opt], df_main_transformed.loc[val_idx_opt]
    y_train_opt, y_val_opt = y_price_for_training.loc[train_idx_opt], y_price_for_training.loc[val_idx_opt]
    data_train_optuna, data_val_optuna = {}, {}
    for head_name, config in feature_configs.items():
        cols = config.get('pca_cols') or config.get('raw_cols')
        if cols:
            data_train_optuna[head_name], data_val_optuna[head_name] = df_train_opt[cols], df_val_opt[cols]
    data_train_optuna['msoa_id'], data_val_optuna['msoa_id'] = df_train_opt[[msoa_col_name]], df_val_opt[[msoa_col_name]]

    # --- PHASE 1: Tune Core Model Parameters for Point Prediction ---
    print(f"\n--- STAGE 6A: HYPERPARAMETER TUNING (PHASE 1 - CORE MODEL) ---")
    study_phase1 = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner())
    study_phase1.optimize(lambda trial: objective(
        trial, data_train_optuna, y_train_opt, data_val_optuna, y_val_opt,
        feature_configs, msoa_cardinality, y_price_std, y_price_mean,
        y_main_raw.loc[val_idx_opt].to_frame(), mode='point_prediction'
    ), n_trials=n_trials)
    best_params_phase1 = study_phase1.best_params
    print(f"  - Phase 1 Optuna study complete. Best MAE: £{study_phase1.best_value:,.2f}")
    print(f"  - Best core parameters: {best_params_phase1}")

    # --- PHASE 2: Tune Evidential Loss Parameters ---
    print(f"\n--- STAGE 6B: HYPERPARAMETER TUNING (PHASE 2 - EVIDENTIAL LOSS) ---")
    # Pass the results of phase 1 as the base for phase 2
    study_phase2 = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner())
    study_phase2.optimize(lambda trial: objective(
        trial, data_train_optuna, y_train_opt, data_val_optuna, y_val_opt,
        feature_configs, msoa_cardinality, y_price_std, y_price_mean,
        y_main_raw.loc[val_idx_opt].to_frame(), mode='evidential', base_params=best_params_phase1
    ), n_trials=max(15, n_trials // 2))
    best_params_phase2 = study_phase2.best_params
    print(f"  - Phase 2 Optuna study complete. Best MAE with Evidential Loss: £{study_phase2.best_value:,.2f}")
    print(f"  - Best evidential parameters: {best_params_phase2}")

    # --- AUTOMATED & COMPLETE MERGE of Hyperparameters ---
    # The results from the Phase 1 study are the source of truth for the core architecture.
    # The results from Phase 2 are the source of truth for the loss function.
    # We merge them to create the single, definitive configuration.
    best_params = best_params_phase1.copy()
    best_params.update(best_params_phase2)
    
    print("\n--- Final Combined Hyperparameters for K-Fold Training ---")
    print(json.dumps(best_params, indent=2))

    print(f"\n--- STAGE 7: Training Final Model using Best Hyperparameters & K-Fold CV ---")
    kf = KFold(n_splits=NUM_FOLDS_FINAL, shuffle=True, random_state=42)
    oof_evidential_preds = np.zeros((len(df_main_transformed), 4))
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(df_main_transformed)):
        print(f"\n========== FINAL TRAINING: FOLD {fold+1}/{NUM_FOLDS_FINAL} ==========")
        df_train_fold, df_val_fold = df_main_transformed.iloc[train_idx], df_main_transformed.iloc[val_idx]
        y_train_fold, y_val_fold = y_price_for_training.iloc[train_idx], y_price_for_training.iloc[val_idx]
        y_val_raw_fold = y_main_raw.iloc[val_idx]
        
        data_train_fold, data_val_fold = {}, {}
        for head_name, config in feature_configs.items():
            cols = config.get('pca_cols') or config.get('raw_cols')
            if cols:
                data_train_fold[head_name], data_val_fold[head_name] = df_train_fold[cols], df_val_fold[cols]
        data_train_fold['lat_lon'], data_val_fold['lat_lon'] = df_train_fold[LAT_LON_COLS], df_val_fold[LAT_LON_COLS]
        data_train_fold['msoa_id'], data_val_fold['msoa_id'] = df_train_fold[[msoa_col_name]], df_val_fold[[msoa_col_name]]
        
        best_head_params = create_head_params(best_params, feature_configs)
        train_dataset = WisteriaDataset(data_train_fold, y_train_fold, y_train_fold)
        val_dataset = WisteriaDataset(data_val_fold, y_val_fold, y_val_fold)
        train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=best_params['batch_size'] * 2, shuffle=False, num_workers=2, pin_memory=True)
        
        model = FusionModel(
            feature_configs=feature_configs, msoa_cardinality=msoa_cardinality, head_params=best_head_params,
            fusion_embed_dim=best_params['fusion_embed_dim'], msoa_embedding_dim=best_params['msoa_embedding_dim'],
            fusion_dropout_rate=best_params['fusion_dropout_rate'], use_enhanced_tcn=True,
            tcn_dropout=best_params.get('tcn_dropout', 0.3), stratum_name='monolith', mode='evidential',
            tabnet_n_steps=best_params.get('tabnet_n_steps', 4),
            tabnet_gamma=best_params.get('tabnet_gamma', 1.5),
            fusion_mlp_depth=best_params.get('fusion_mlp_depth', 'deep')
        ).to(DEVICE)
        
        optimizer = optim.AdamW(model.parameters(), lr=best_params['learning_rate'], weight_decay=best_params['weight_decay'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=10, verbose=False)
        loss_fn = AsymmetricEvidentialLoss(coeff=best_params['evidential_reg_coeff'], undervalue_penalty=best_params['undervalue_penalty'])
        
        best_val_mae, patience_counter, best_model_state = float('inf'), 0, None
        patience = 25
        loss_weights = {
            'evidential': best_params.get('evidential_weight', 0.5),
            'mae': 1.0 - best_params.get('evidential_weight', 0.5)
        }

        for epoch in range(EPOCHS + 100):
            _ = train_one_epoch(model, train_loader, optimizer, loss_fn, mode='evidential', loss_weights=loss_weights)
            evidential_preds_val = evaluate(model, val_loader, mode='evidential')
            point_preds_scaled = evidential_preds_val.numpy()[:, 0]
            point_preds_unscaled = np.expm1(point_preds_scaled * y_price_std + y_price_mean)
            val_mae = mean_absolute_error(y_val_raw_fold.values, point_preds_unscaled)
            scheduler.step(val_mae)
            
            if val_mae < best_val_mae:
                best_val_mae = val_mae; patience_counter = 0; best_model_state = model.state_dict()
            else:
                patience_counter += 1
            if (epoch + 1) % 10 == 0: print(f"      Epoch {epoch+1}, Val MAE: £{val_mae:,.2f} (Best: £{best_val_mae:,.2f})")
            if patience_counter >= patience: break
        
        model.load_state_dict(best_model_state)
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f"model_fold_{fold}.pt"))
        evidential_preds_val = evaluate(model, val_loader, mode='evidential')
        oof_evidential_preds[val_idx] = evidential_preds_val.numpy()
        
        del model, optimizer, scheduler, data_train_fold, data_val_fold; gc.collect(); torch.cuda.empty_cache()

    oof_df = pd.DataFrame(oof_evidential_preds, columns=['pred_gamma', 'pred_nu', 'pred_alpha', 'pred_beta'], index=df_main_transformed.index)
    oof_df['final_gamma'] = oof_df['pred_gamma']
    aleatoric_oof, epistemic_oof = calculate_evidential_uncertainty(torch.tensor(oof_evidential_preds))
    oof_df['aleatoric_uncertainty'] = aleatoric_oof.numpy()
    oof_df['epistemic_uncertainty'] = epistemic_oof.numpy()
    oof_df = df_main_transformed.join(oof_df)

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
    ideal_report_cols = [
        'property_id', 
        'epistemic_uncertainty',
        'predicted_price',
        'most_recent_sale_price',
        'absolute_error'
    ]
    
    # FAIL-SAFE: Only select columns that actually exist in the dataframe.
    report_cols = [col for col in ideal_report_cols if col in active_learning_targets.columns]
    
    # Save the report to a CSV file
    report_path = os.path.join(output_dir, "active_learning_targets.csv")
    active_learning_targets[report_cols].to_csv(report_path, index=False, float_format='%.4f')
    
    print(f"  - Identified the top {top_percent}% ({len(active_learning_targets)}) most uncertain properties.")
    print(f"  - Actionable report saved to: {report_path}")


def generate_shap_reports_for_holdout(df_main_raw, df_holdout_raw, best_params, canonical_artifacts, msoa_map, msoa_col_name, holdout_results_df, output_dir):
    """
    [ARCHITECTURAL CORRECTED] Generates structured SHAP explanations.
    """
    import shap
    print("\n--- Generating Structured SHAP Explanations for Holdout Set ---")
    if df_holdout_raw.empty:
        print("  - Holdout set is empty. Skipping SHAP report generation.")
        return

    all_shap_reports = []
    print(f"\n--- Preparing SHAP Explainer for Monolithic Model ---")
    
    background_data = df_main_raw.sample(min(100, len(df_main_raw)), random_state=42)
    
    # Use the authoritative transformation function to get both the transformed data and the correct model config.
    bg_data_transformed, feature_configs = transform_with_canonical_artifacts(background_data, canonical_artifacts)
    best_head_params = create_head_params(best_params, feature_configs)

    model_path = os.path.join(output_dir, "model_fold_0.pt")
    if not os.path.exists(model_path):
        print(f"  - WARNING: {model_path} not found. Cannot generate SHAP reports.")
        return
        
    model = FusionModel(
        feature_configs=feature_configs, msoa_cardinality=msoa_map['unknown_id'] + 1, head_params=best_head_params,
        fusion_embed_dim=best_params['fusion_embed_dim'], msoa_embedding_dim=best_params['msoa_embedding_dim'],
        fusion_dropout_rate=best_params['fusion_dropout_rate'], use_enhanced_tcn=True,
        tcn_dropout=best_params.get('tcn_dropout', 0.3),
        tabnet_n_steps=best_params.get('tabnet_n_steps', 4), tabnet_gamma=best_params.get('tabnet_gamma', 1.5),
        fusion_mlp_depth=best_params.get('fusion_mlp_depth', 'deep')
    ).to(DEVICE)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # --- ARCHITECTURALLY CORRECTED: Rebuild data dictionary from the single transformed dataframe ---
    data_for_model_bg = {}
    for head_name, config in feature_configs.items():
        cols = config.get('pca_cols') or config.get('raw_cols')
        if cols:
            data_for_model_bg[head_name] = bg_data_transformed[cols]
    
    data_for_model_bg['msoa_id'] = pd.DataFrame(
        background_data[msoa_col_name].map(msoa_map['codes']).fillna(msoa_map['unknown_id']).astype(int)
    )

    # Define a canonical order for ALL model inputs
    ordered_keys = sorted(list(data_for_model_bg.keys()))
    background_tensor_list = [
        torch.tensor(data_for_model_bg[k].values, dtype=torch.float32).to(DEVICE) for k in ordered_keys
    ]

    # Use the architecturally sound wrapper
    wrapped_model = ShapModelWrapper(model, ordered_keys).to(DEVICE)

    print("  - Initializing DeepExplainer...")
    explainer = shap.DeepExplainer(wrapped_model, background_tensor_list)
    print("  - Explainer initialized.")

    feature_to_head_map = {}
    for head, conf in feature_configs.items():
        feature_list = conf.get('pca_cols') or conf.get('raw_cols') or conf.get('original_cols')
        if feature_list:
            for feat in feature_list:
                feature_to_head_map[feat] = head

    print(f"  - Generating structured explanations for {len(df_holdout_raw)} properties...")
    for idx, property_row in tqdm(df_holdout_raw.iterrows(), total=len(df_holdout_raw)):
        property_df = pd.DataFrame([property_row])
        
        # Use the single, authoritative transformation function
        property_df_transformed, _ = transform_with_canonical_artifacts(property_df, canonical_artifacts)

        # Rebuild the data dictionary from the transformed dataframe
        data_for_model_prop = {}
        for head_name, config in feature_configs.items():
            cols = config.get('pca_cols') or config.get('raw_cols')
            if cols:
                data_for_model_prop[head_name] = property_df_transformed[cols]
        
        data_for_model_prop['msoa_id'] = pd.DataFrame(
            property_df[msoa_col_name].map(msoa_map['codes']).fillna(msoa_map['unknown_id']).astype(int)
        )
        
        # Ensure the tensor list for explanation has the same canonical order
        explain_tensor_list = [
            torch.tensor(data_for_model_prop[k].values, dtype=torch.float32).to(DEVICE) for k in ordered_keys
        ]
        
        shap_values_list = explainer.shap_values(explain_tensor_list)
        
        # Reconstruct full feature names from the ordered, processed data
        full_feature_names = [col for key in ordered_keys for col in data_for_model_prop[key].columns]
        flat_shap_values = np.concatenate([sv.flatten() for sv in shap_values_list if isinstance(sv, np.ndarray)])
        
        if len(full_feature_names) != len(flat_shap_values): continue

        shap_df = pd.DataFrame({'feature': full_feature_names, 'shap_value': flat_shap_values})
        shap_df['head'] = shap_df['feature'].map(lambda f: feature_to_head_map.get(f, 'msoa_id' if 'msoa' in f else 'unknown'))
        
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
    [ARCHITECTURALLY CORRECTED] Loads autoencoder features from .npy files and merges them
    into the main dataframe using an index-based join, assuming identical row ordering.
    """
    print("\n--- STAGE 1.2: Loading and Merging External Autoencoder (Head G) Features ---")
    if not encodings_dir or not os.path.exists(encodings_dir):
        print(f"  - WARNING: Encodings directory not found at '{encodings_dir}'. Head G features will be missing.")
        return df, []

    merged_df = df.copy()
    num_merged_sets = 0
    all_new_ae_cols = []
    
    # The bash script downloads .npy files, so we must look for those.
    for filename in sorted(os.listdir(encodings_dir)):
        if filename.endswith(".npy"):
            try:
                group_name = filename.replace("_encodings.npy", "")
                file_path = os.path.join(encodings_dir, filename)
                
                # Load the NumPy array
                ae_encodings = np.load(file_path)
                
                # ARCHITECTURAL CONTRACT: The number of rows in the .npy file MUST match the main dataframe.
                if ae_encodings.shape[0] != len(df):
                    print(f"  - FATAL ERROR: Row count mismatch for '{filename}'.")
                    print(f"    Main dataframe has {len(df)} rows, but encoding file has {ae_encodings.shape[0]} rows.")
                    print("    This indicates the AE features were generated from a different base dataset.")
                    exit(1)
                
                # Create column names for the new features
                ae_cols = [f"ae_feat_{group_name}_{i}" for i in range(ae_encodings.shape[1])]
                all_new_ae_cols.extend(ae_cols)
                
                # Convert to a DataFrame, crucially setting the index to match the main dataframe for a safe merge.
                ae_df = pd.DataFrame(ae_encodings, columns=ae_cols, index=df.index)

                # Merge on index. This is safer than concat and respects the row alignment.
                merged_df = merged_df.join(ae_df)
                
                num_merged_sets += 1
                print(f"  - Successfully merged {ae_df.shape[1]} features for group '{group_name}'.")

            except Exception as e:
                print(f"  - ERROR: Could not process file {filename}. Error: {e}")

    if num_merged_sets > 0:
        print(f"  - Successfully merged {num_merged_sets} AE feature sets. New dataframe shape: {merged_df.shape}")
    
    return merged_df, all_new_ae_cols


def predict_on_holdout(df_holdout_raw, best_params, canonical_artifacts, msoa_map, price_stats, output_dir):
    """
    [CORRECTED] Generates predictions on raw data using the unified canonical artifacts.
    """
    if df_holdout_raw.empty:
        return pd.DataFrame()

    print("  - Applying canonical transformations to holdout data...")
    df_holdout_transformed, feature_configs = transform_with_canonical_artifacts(df_holdout_raw, canonical_artifacts)

    # --- Data Selection ---
    data_for_model = {}
    for head_name, config in feature_configs.items():
        cols = config.get('pca_cols') or config.get('raw_cols')
        if cols:
            data_for_model[head_name] = df_holdout_transformed[cols]
    
    data_for_model['lat_lon'] = df_holdout_transformed[['latitude', 'longitude']]
    msoa_col_name = msoa_map['column_name']
    
    # ARCHITECTURAL MANDATE: Source ALL inputs from the single, transformed dataframe to guarantee alignment.
    # The MSOA column in this dataframe is already integer-mapped.
    data_for_model['msoa_id'] = df_holdout_transformed[msoa_col_name]

    y_price_log_holdout = np.log1p(df_holdout_raw['most_recent_sale_price'])
    y_price_scaled_holdout = pd.Series((y_price_log_holdout - price_stats['y_price_mean']) / price_stats['y_price_std'], index=df_holdout_raw.index)
    
    dataset = WisteriaDataset(data_for_model, y_price_scaled_holdout, y_price_scaled_holdout)
    loader = DataLoader(dataset, batch_size=2048, shuffle=False)

    head_params = create_head_params(best_params, feature_configs)
    
    all_fold_preds = []
    for fold in range(NUM_FOLDS_FINAL):
        model_path = os.path.join(output_dir, f"model_fold_{fold}.pt")
        if not os.path.exists(model_path): continue
        
        model = FusionModel(
            feature_configs=feature_configs,
            msoa_cardinality=msoa_map['unknown_id'] + 1, 
            head_params=head_params, 
            fusion_embed_dim=best_params['fusion_embed_dim'], 
            msoa_embedding_dim=best_params['msoa_embedding_dim'], 
            fusion_dropout_rate=best_params['fusion_dropout_rate'],
            use_enhanced_tcn=True,
            tcn_dropout=best_params.get('tcn_dropout', 0.3),
            stratum_name='monolith',
            mode='evidential',
            tabnet_n_steps=best_params.get('tabnet_n_steps', 4),
            tabnet_gamma=best_params.get('tabnet_gamma', 1.5),
            fusion_mlp_depth=best_params.get('fusion_mlp_depth', 'deep')
        ).to(DEVICE)
        model.load_state_dict(torch.load(model_path))
        preds = evaluate(model, loader, mode='evidential') # EXPLICIT: Call evaluate in evidential mode.
        all_fold_preds.append(preds)
        del model; gc.collect(); torch.cuda.empty_cache()

    if not all_fold_preds: 
        print("  - WARNING: No fold models found for holdout prediction.")
        return pd.DataFrame()

    avg_preds = torch.mean(torch.stack(all_fold_preds), dim=0)

    final_gamma = avg_preds[:, 0].numpy()
    predicted_price = np.expm1(final_gamma * price_stats['y_price_std'] + price_stats['y_price_mean'])

    aleatoric, epistemic = calculate_evidential_uncertainty(avg_preds)

    results_df = df_holdout_raw[['property_id', 'most_recent_sale_price']].copy()
    results_df['predicted_price'] = predicted_price
    results_df['aleatoric_uncertainty'] = aleatoric.numpy()
    results_df['epistemic_uncertainty'] = epistemic.numpy()

    return results_df



def select_features_with_lgbm(df_raw, target_series, candidate_cols, n_top_features, head_name):
    """
    Uses a LightGBM model to perform feature selection on a given set of columns.
    """
    print(f"\n--- Running LGBM Feature Selection for '{head_name}' ---")
    
    available_candidates = [col for col in candidate_cols if col in df_raw.columns]
    X = df_raw[available_candidates].copy()
    y = target_series.copy()
    print(f"  - Evaluating {len(available_candidates)} candidate features.")

    X.fillna(0, inplace=True)

    lgbm_selector = lgb.LGBMRegressor(random_state=42, n_jobs=-1)

    print("  - Fitting selector model...")
    lgbm_selector.fit(X, np.log1p(y))

    importances_df = pd.DataFrame({
        'feature': X.columns,
        'importance': lgbm_selector.feature_importances_
    }).sort_values('importance', ascending=False)

    n_to_select = min(n_top_features, len(available_candidates))
    selected_features = importances_df.head(n_to_select)['feature'].tolist()
    
    print(f"  - Selected the top {len(selected_features)} features.")
    
    del X, y, lgbm_selector, importances_df
    gc.collect()

    return selected_features

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
        # Tuned parameters from previous studies
        'head_B_aesthetic': {
            'latent_dim': 120, 'hidden_dim_1': 512, 'hidden_dim_2': 256, 'lr': 0.001299
        },
        # NEW: Sensible defaults for other MLP heads to prevent pipeline failure.
        # These can be properly tuned in a dedicated offline study later.
        'head_A_dna_microscope': {
            'latent_dim': 16, 'hidden_dim_1': 64, 'hidden_dim_2': 32, 'lr': 0.001
        },
        'head_A_dna_engineered': {
            'latent_dim': 64, 'hidden_dim_1': 256, 'hidden_dim_2': 128, 'lr': 0.001
        },
        'head_AVM': {
            'latent_dim': 32, 'hidden_dim_1': 128, 'hidden_dim_2': 64, 'lr': 0.001
        }
    }
    print(f"  - Using AE Best Params: {AE_BEST_PARAMS}")


    # --- Static Feature List Definitions ---
    # These lists define core feature sets and must be available globally within main().
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

    # --- STAGE 1: Load Data Sources (Optimized) ---
    print("--- STAGE 1: Loading Data Sources ---")
    try:
        # --- Pre-determine required columns to minimize memory usage ---
        print("  - Pre-parsing feature sets to determine required columns...")
        with open(FEATURE_SETS_PATH, 'r') as f:
            feature_sets = json.load(f)

        # --- NEW: Definitive Target Variable Sanitization ---
        # Architect's Mandate: The target variable must NEVER be a feature.
        # This prevents data leakage and brittle column-dropping logic downstream.
        print("\n--- Sanitizing feature sets to remove target variable ---")
        TARGET_COLUMNS = ['most_recent_sale_price', 'sale_price'] # Add any other aliases
        target_set = set(TARGET_COLUMNS)
        for head_name in list(feature_sets.keys()):
            original_count = len(feature_sets[head_name])
            # This list comprehension rebuilds the feature list, excluding any target columns.
            feature_sets[head_name] = [col for col in feature_sets[head_name] if col not in target_set]
            removed_count = original_count - len(feature_sets[head_name])
            if removed_count > 0:
                print(f"  - Removed {removed_count} target variable column(s) from '{head_name}'.")

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
        df_features, head_g_cols = load_and_merge_ae_features(df_features, AE_ENCODINGS_PATH)
        
        # ARCHITECTURAL MANDATE: Dynamically register the newly loaded Head G columns.
        if head_g_cols:
            print(f"  - Registering {len(head_g_cols)} new features into 'head_G_gemini_quantitative'.")
            if 'head_G_gemini_quantitative' not in feature_sets:
                feature_sets['head_G_gemini_quantitative'] = []
            feature_sets['head_G_gemini_quantitative'].extend(head_g_cols)
        
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
        df_rightmove_raw = pd.read_csv(RIGHTMOVE_DATA_PATH, header=None, on_bad_lines='skip')
        
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

    # --- STAGE 1.6: Refactor `head_A_dna` into sub-heads ---
    print("\n--- STAGE 1.6: Refactoring 'head_A_dna' into specialized sub-heads ---")
    if 'head_A_dna' in feature_sets:
        original_dna_cols = feature_sets.pop('head_A_dna')
        
        dna_raw_cols = []
        dna_microscope_cols = []
        dna_engineered_cols = []
        
        engineered_prefixes = ('BP_', 'Inter_', 'MODE', 'Mismatch_', 'Opportunity_', 'Persona_', 'Ratio_', 'Risk_', 'Thesis_', 'Tradeoff_', 'missingindicator_')
        engineered_uniques = {'Composite_CoreLiving_Strength', 'latitude', 'longitude'}
        identifiers = {'property_id', 'most_recent_sale_price'}

        for col in original_dna_cols:
            if col in identifiers:
                continue
            elif col.startswith(('cat__', 'num__')):
                dna_raw_cols.append(col)
            elif col.startswith('microscope_emb_'):
                dna_microscope_cols.append(col)
            elif col.startswith(engineered_prefixes) or col in engineered_uniques:
                dna_engineered_cols.append(col)
            # Note: Any unclassified columns from the original head are implicitly dropped, enforcing architectural purity.

        feature_sets['head_A_dna_raw'] = dna_raw_cols
        feature_sets['head_A_dna_microscope'] = dna_microscope_cols
        feature_sets['head_A_dna_engineered'] = dna_engineered_cols

        print(f"  - Created 'head_A_dna_raw' (for TabNet) with {len(dna_raw_cols)} features.")
        print(f"  - Created 'head_A_dna_microscope' (for MLP) with {len(dna_microscope_cols)} features.")
        print(f"  - Created 'head_A_dna_engineered' (for MLP) with {len(dna_engineered_cols)} features.")
    else:
        print("  - WARNING: 'head_A_dna' not found in feature_sets. Skipping refactoring.")

    # --- STAGE 2: Direct Address Normalization (No Key Map Needed) ---
    print("\n--- STAGE 2: Direct Address Processing ---")

    # Process features addresses (using existing property_id)
    print("  - Normalizing feature addresses using V4 key (postcode_street)...")
    df_features['normalized_address_key'] = df_features['property_id'].apply(normalize_address_key_v4)
    
    # Remove rows with null addresses
    df_features = df_features[df_features['normalized_address_key'].notna()].copy()
    
    print(f"  - Features with valid addresses: {len(df_features)}")
    print("  - Sample normalized feature keys:")
    sample_feature_keys = df_features['normalized_address_key'].head(10).tolist()
    print("   ", sample_feature_keys)

    # Process sales data using the newly refactored, single-responsibility function
    df_rightmove_processed = process_rightmove_data(df_rightmove_raw)

    # Now, normalize the addresses in the processed dataframe
    print("  - Normalizing Rightmove addresses using V4 key (postcode_street)...")
    df_rightmove_processed['normalized_address_key'] = df_rightmove_processed['rightmove_address_text'].apply(normalize_address_key_v4)
    
    # Remove rows with null addresses that may have been generated
    df_rightmove_with_address_key = df_rightmove_processed[df_rightmove_processed['normalized_address_key'].notna()].copy()
    
    print(f"  - Rightmove with valid addresses: {len(df_rightmove_with_address_key)}")
    print("  - Sample normalized Rightmove keys:")
    sample_rm_keys = df_rightmove_with_address_key['normalized_address_key'].head(10).tolist()
    print("   ", sample_rm_keys)
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
    # Surgically select only the columns needed from the right-hand dataframe.
    cols_to_merge = [
        merge_on_key, 
        'most_recent_sale_price', 
        'most_recent_sale_year',
        'most_recent_sale_month', 
        'total_sales_count', 
        'days_since_last_sale',
        'price_change_since_last', 
        'rightmove_address_text'
    ]
    
    # Architect's Mandate: Enforce a single source of truth for the target variable.
    # Unconditionally remove any stale version of the target from the features dataframe
    # to prevent merge conflicts (_x/_y columns) that cause downstream KeyErrors.
    if 'most_recent_sale_price' in df_features.columns:
        print("  - Dropping stale 'most_recent_sale_price' from features dataframe to prevent merge conflict.")
        df_features.drop(columns=['most_recent_sale_price'], inplace=True, errors='ignore')

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

    # --- ARCHITECTURAL MANDATE: Fail-Fast on Missing Critical Feature Sets ---
    print("\n--- Validating Presence of Critical Feature Sets ---")
    CRITICAL_HEADS = [
        'head_B_aesthetic',          # Proven high-signal visual features
        'head_G_gemini_quantitative',# Proven high-signal quantitative features
        'head_A_dna_engineered'      # Contains high-level engineered scores
    ]
    
    missing_critical_heads = []
    for head in CRITICAL_HEADS:
        if head not in feature_sets or not feature_sets[head]:
            missing_critical_heads.append(f"- '{head}' is missing or empty in feature_sets.json.")
            continue # No need to check columns if the set is empty
        
        # Check if ANY of the columns for the head exist in the dataframe
        if not any(col in df.columns for col in feature_sets[head]):
             missing_critical_heads.append(f"- '{head}' is defined, but NONE of its columns were found in the final merged dataframe.")

    if missing_critical_heads:
        error_message = "FATAL: Critical feature sets are missing, which would lead to silent model degradation.\n"
        error_message += "\n".join(missing_critical_heads)
        error_message += "\nCheck the upstream data generation process and the feature_sets.json file."
        raise ValueError(error_message)
    
    print("  - All critical feature sets are present. Proceeding.")
    # --- END OF NEW VALIDATION GATE ---
    
    # --- NEW: Data Quality Filter ---
    print("\n--- Filtering for High-Confidence Merges ---")
    # Use the same canonical column name as the validation function.
    address_col_rm = 'rightmove_address_text'
    if address_col_rm not in df.columns:
        raise ValueError(f"FATAL: Canonical column '{address_col_rm}' not found before filtering. Check merge logic.")

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
    print("\n--- STAGE 5: Filtering, Validating and Splitting Data ---")
    min_price, max_price = 10000, df['most_recent_sale_price'].quantile(0.999)
    df = df[(df['most_recent_sale_price'] >= min_price) & (df['most_recent_sale_price'] <= max_price)].copy().reset_index(drop=True)

    n_holdout = 0.15
    df_main_raw, df_holdout_raw = train_test_split(df, test_size=n_holdout, random_state=42, shuffle=True)
    print(f"  - Split data into {len(df_main_raw)} training rows and {len(df_holdout_raw)} holdout rows.")

    # --- NEW STAGE: Offline Autoencoder Training (on training data ONLY) ---
    print("\n--- STAGE 4.3: Offline Autoencoder Pre-Training for MLP Heads ---")
    HEAD_CONFIG = {
        'head_A_dna_raw': {'type': 'tabnet'},
        'head_A_dna_microscope': {'type': 'mlp'},
        'head_A_dna_engineered': {'type': 'mlp'},
        'head_B_aesthetic': {'type': 'mlp'},
        'head_C_census': {'type': 'tabnet'},
        'head_G_gemini_quantitative': {'type': 'pre_embedded'},
        'head_AVM': {'type': 'mlp'},
        'head_compass_raw': {'type': 'tabnet'},
        'head_E_temporal': {'type': 'tcn'},
    }
    trained_autoencoders = {}
    for head_name, config in HEAD_CONFIG.items():
        if config['type'] == 'mlp' and head_name in feature_sets:
            if head_name in AE_BEST_PARAMS:
                ae_artifact = train_autoencoder_for_head(
                    df_raw=df_main_raw, # CRITICAL: Use training data only
                    cols=feature_sets[head_name],
                    head_name=head_name,
                    ae_params=AE_BEST_PARAMS[head_name],
                    device=DEVICE
                )
                if ae_artifact:
                    trained_autoencoders[head_name] = ae_artifact
            else:
                print(f"  - NOTE: No AE parameters found for '{head_name}'. It will use PCA fallback.")

    # --- ARCHITECTURALLY CORRECTED STAGE: High-Dimensionality Feature Curation ---
    # This stage runs AFTER the train/holdout split to PREVENT DATA LEAKAGE.
    print("\n--- Pre-filtering TabNet features with LightGBM (on training data ONLY) ---")
    TABNET_HEADS_TO_CURATE = {
        'head_A_dna_raw': 500,
        'head_C_census': 250,
        'head_compass_raw': 750
    }
    for head_name, n_features in TABNET_HEADS_TO_CURATE.items():
        if head_name in feature_sets and feature_sets[head_name]:
            curated_features = select_features_with_lgbm(
                df_raw=df_main_raw, # CRITICAL: Use training data only
                target_series=df_main_raw['most_recent_sale_price'],
                candidate_cols=feature_sets[head_name],
                n_top_features=n_features,
                head_name=head_name
            )
            original_count = len(feature_sets[head_name])
            feature_sets[head_name] = curated_features
            print(f"  - Updated '{head_name}': {original_count} features -> {len(curated_features)} features.")
        else:
            print(f"  - Skipping feature selection for '{head_name}' (not found or empty).")

    # --- UNIFIED GLOBAL PREPROCESSING (LEAKAGE-PROOF) ---
    HEAD_CONFIG = {
        'head_A_dna_raw': {'type': 'tabnet'},
        'head_A_dna_microscope': {'type': 'mlp'},
        'head_A_dna_engineered': {'type': 'mlp'},
        'head_B_aesthetic': {'type': 'mlp'},
        'head_C_census': {'type': 'tabnet'},
        'head_G_gemini_quantitative': {'type': 'pre_embedded'},
        'head_AVM': {'type': 'mlp'},
        'head_compass_raw': {'type': 'tabnet'},
        'head_E_temporal': {'type': 'tcn'},
    }
    N_COMPONENTS_PER_HEAD = {
        "head_A_dna_microscope": 16,
        "head_A_dna_engineered": 50,
        "head_B_aesthetic": 40,
        "head_C_census": 100,
        "head_G_gemini_quantitative": 50,
        "head_AVM": 15,
        "DEFAULT": 50
    }
    
    # 1. Fit all preprocessors ONLY on the training data and create the canonical artifact file.
    # This is the single most important change to prevent data leakage.
    print("\n--- Fitting canonical preprocessors ON TRAINING DATA ONLY ---")
    # CORRECTED: Base this on the training data columns to prevent any schema leakage.
    universal_cols_present = [col for col in df_main_raw.columns if col in UNIVERSAL_PREDICTORS]
    canonical_artifacts = fit_and_create_canonical_artifacts(df_main_raw, feature_sets, N_COMPONENTS_PER_HEAD, universal_cols_present, HEAD_CONFIG, trained_autoencoders)
    joblib.dump(canonical_artifacts, os.path.join(OUTPUT_DIR, "canonical_artifacts.joblib"))
    print("  - Canonical artifacts created and saved.")
    
    # 2. Transform BOTH the training and holdout datasets using these artifacts
    print("  - Applying canonical transformations to train and holdout sets...")
    df_main_transformed, final_feature_configs = transform_with_canonical_artifacts(df_main_raw, canonical_artifacts)
    # ADDED: Create the transformed holdout set. This is the critical missing step.
    df_holdout_transformed, _ = transform_with_canonical_artifacts(df_holdout_raw, canonical_artifacts)

    # 3. Prepare target variable and MSOA encodings from the raw training data
    y_price_log = np.log1p(df_main_raw['most_recent_sale_price'])
    y_price_mean, y_price_std = y_price_log.mean(), y_price_log.std()
    
    msoa_col = find_msoa_column(df_main_raw)

    if msoa_col:
        print(f"  - MSOA identifier column found: '{msoa_col}'")
        msoa_codes, msoa_uniques = pd.factorize(df_main_raw[msoa_col].fillna('Unknown'))
        msoa_cardinality = len(msoa_uniques) + 1
        print(f"\n--- MSOA cardinality set to: {msoa_cardinality} ---")

        # --- DEFENSIVE GATE: MSOA VALIDATION ---
        MINIMUM_PLAUSIBLE_MSOA_CARDINALITY = 1000
        if msoa_cardinality < MINIMUM_PLAUSIBLE_MSOA_CARDINALITY:
            # --- GRACEFUL DEGRADATION: MSOA feature is disabled ---
            print("\n  - WARNING: No suitable high-cardinality MSOA column found. Proceeding without MSOA feature.")
            print("  - A dummy MSOA column will be created for model compatibility.")
        print("  - MSOA cardinality passed validation.")

        # Add the mapped MSOA column to the transformed dataframe for the model
        df_main_transformed[msoa_col] = msoa_codes
        
        # Apply the exact same MSOA mapping to the holdout set
        msoa_map_for_holdout = dict(zip(msoa_uniques, range(len(msoa_uniques))))
        unknown_msoa_id = len(msoa_uniques)
        df_holdout_transformed[msoa_col] = df_holdout_raw[msoa_col].map(msoa_map_for_holdout).fillna(unknown_msoa_id).astype(int)

    else:
        # --- GRACEFUL DEGRADATION: MSOA feature is disabled ---
        print("\n  - WARNING: No suitable high-cardinality MSOA column found. Proceeding without MSOA feature.")
        print("  - A dummy MSOA column will be created for model compatibility.")
        msoa_col = 'msoa_id_dummy' # Use a new, unambiguous name
        msoa_cardinality = 1 # Cardinality is 1 (only the 'unknown' category)
        msoa_uniques = ['unknown'] # Define the dummy unique values
    
        # Add the dummy column (all zeros) to both dataframes
        df_main_transformed[msoa_col] = 0
        df_holdout_transformed[msoa_col] = 0

    # --- STRATEGY 1 - ESTABLISH A SIMPLE BASELINE ---
    print("\n--- STRATEGY 1: Training a LightGBM Baseline Model ---")
    lgbm_selector = lgb.LGBMRegressor(random_state=42)
    all_features = [col for col in df_main_raw.columns if col not in ['most_recent_sale_price', 'property_id', 'normalized_address_key', 'final_merge_key', 'address'] and pd.api.types.is_numeric_dtype(df_main_raw[col])]
    lgbm_selector.fit(df_main_raw[all_features].fillna(0), y_price_log)

    importances = pd.DataFrame({'feature': all_features, 'importance': lgbm_selector.feature_importances_}).sort_values('importance', ascending=False)
    N_TOP_GLOBAL_FEATURES = 750
    top_global_features = importances.head(N_TOP_GLOBAL_FEATURES)['feature'].tolist()
    X_baseline = df_main_raw[top_global_features].fillna(0)
    y_baseline = df_main_raw['most_recent_sale_price']
    baseline_kf = KFold(n_splits=NUM_FOLDS_FINAL, shuffle=True, random_state=42)
    oof_preds_baseline = np.zeros(len(df_main_raw))
    for fold, (train_idx, val_idx) in enumerate(baseline_kf.split(X_baseline)):
        X_train, X_val = X_baseline.iloc[train_idx], X_baseline.iloc[val_idx]
        y_train_log, y_val_log = y_price_log.iloc[train_idx], y_price_log.iloc[val_idx]
        lgbm_baseline = lgb.LGBMRegressor(random_state=42, n_estimators=2000, learning_rate=0.02, num_leaves=31, n_jobs=-1, colsample_bytree=0.7, subsample=0.7, reg_alpha=0.1, reg_lambda=0.1)
        lgbm_baseline.fit(X_train, y_train_log, eval_set=[(X_val, y_val_log)], eval_metric='mae', callbacks=[lgb.early_stopping(100, verbose=False)])
        oof_preds_baseline[val_idx] = np.expm1(lgbm_baseline.predict(X_val))
    baseline_mae = mean_absolute_error(y_baseline, oof_preds_baseline)
    print(f"\n--- LIGHTGBM BASELINE MODEL OOF MAE: £{baseline_mae:,.2f} ---\n")

    # --- STAGES 6 & 7: Train Final Monolithic Deep Learning Model ---
    # Create config dictionaries to enforce a clean contract with the training function.
    price_stats = {'y_price_mean': y_price_mean, 'y_price_std': y_price_std}
    msoa_config = {'column_name': msoa_col, 'cardinality': msoa_cardinality}

    eval_df, best_params = train_final_model(
        df_main_transformed=df_main_transformed.reset_index(drop=True),
        y_main_raw=df_main_raw['most_recent_sale_price'].reset_index(drop=True),
        feature_configs=final_feature_configs,
        price_stats=price_stats,
        msoa_config=msoa_config,
        universal_cols_present=universal_cols_present,
        LAT_LON_COLS=LAT_LON_COLS,
        n_trials=N_TRIALS_MAIN
    )

    # --- STAGE 8: Save Final Training Artifacts ---
    print("\n--- STAGE 8: Evaluating Final OOF Performance & Saving Artifacts ---")
    if eval_df is None:
        raise ValueError("FATAL: Model training failed.")

    eval_df['final_predicted_price'] = np.expm1(eval_df['final_gamma'] * y_price_std + y_price_mean)
    final_mae = mean_absolute_error(eval_df['most_recent_sale_price'], eval_df['final_predicted_price'])
    final_r2 = r2_score(eval_df['most_recent_sale_price'], eval_df['final_predicted_price'])
    print(f"\n--- OOF PERFORMANCE (MONOLITHIC DEEP LEARNING MODEL) ---")
    print(f"  - Final MAE: £{final_mae:,.2f}, R²: {final_r2:.4f}")
    eval_df.to_csv(os.path.join(OUTPUT_DIR, "oof_predictions.csv"), index=False)
    print("OOF predictions and preprocessing artifacts saved successfully.")

    # --- STAGE 10: Final Holdout Set Evaluation ---
    print("\n--- STAGE 10: Final Evaluation on True Holdout Set ---")
    if not df_holdout_raw.empty:
        price_stats = {'y_price_mean': y_price_mean, 'y_price_std': y_price_std}
        msoa_map = {
            'codes': dict(zip(msoa_uniques, range(len(msoa_uniques)))), 
            'unknown_id': len(msoa_uniques),
            'column_name': msoa_col
        }


        holdout_results = predict_on_holdout(
            df_holdout_raw, # Pass the RAW holdout data to simulate production
            best_params, 
            canonical_artifacts, # Pass the single, unified artifact file
            msoa_map,
            price_stats,
            OUTPUT_DIR
        )
        
        holdout_results['absolute_error'] = (holdout_results['predicted_price'] - holdout_results['most_recent_sale_price']).abs()
        print("\n--- HOLDOUT SET FINAL RESULTS ---")
        final_mae_holdout = holdout_results['absolute_error'].mean()
        print(f"  - Holdout MAE:  £{final_mae_holdout:,.2f}")
        holdout_results.to_csv(os.path.join(OUTPUT_DIR, "holdout_evaluation_results.csv"), index=False)

        # Generate Reports
        generate_final_report(eval_df, holdout_results, baseline_mae, OUTPUT_DIR)
        generate_active_learning_report(holdout_results, OUTPUT_DIR)

        # --- STAGE 11: Save Final Universal Artifacts for Inference ---
        print("\n--- STAGE 11: Saving Final Universal Artifacts for Inference Pipeline ---")
        with open(os.path.join(OUTPUT_DIR, "price_stats.json"), 'w') as f: json.dump(price_stats, f)
        with open(os.path.join(OUTPUT_DIR, "msoa_map.json"), 'w') as f: json.dump(msoa_map, f)
        print("  - Saved price_stats.json and msoa_map.json.")

        # --- STAGE 12: Generate Batch SHAP Explanations ---
        print("\n--- STAGE 12: Generating SHAP reports for monolithic model ---")
        # CORRECTED: Use the correct transformed dataframe variables.
        generate_shap_reports_for_holdout(
            df_main_raw=df_main_raw,
            df_holdout_raw=df_holdout_raw,
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