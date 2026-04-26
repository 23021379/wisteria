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

#trap "echo '--- SCRIPT FINISHED OR CRASHED: INITIATING AUTO-SHUTDOWN ---'; gcloud compute instances stop $(hostname) --zone=$(curl -s -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/zone | awk -F/ '{print $NF}')" EXIT



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
# [REMOVED] All local path variables are now defined AFTER the cd command
# to ensure their paths are relative to the correct working directory.

# --- Local Project Setup ---
PROJECT_DIR_NAME="model_training_project_v5" # Just the name
mkdir -p "./${PROJECT_DIR_NAME}"
cd "./${PROJECT_DIR_NAME}"

# --- Environment Setup & Local Paths (Define paths relative to the NEW current directory) ---
VENV_DIR="./venv_mt"
SCRIPT_PATH="./02_train_multi_head_model_v5.py"
OUTPUT_DIR="./output"
DATA_DIR="./data"
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${DATA_DIR}"
MASTER_DATA_LOCAL_PATH="${DATA_DIR}/master_dataset.parquet"
FEATURE_SETS_LOCAL_PATH="${DATA_DIR}/feature_sets.json"
RIGHTMOVE_DATA_LOCAL_PATH="${DATA_DIR}/Rightmove.csv"
KEY_MAP_LOCAL_PATH="${DATA_DIR}/key_map.csv"

# --- System & Driver Setup (MUST be done before venv creation) ---
echo "Updating system packages"
sudo apt-get update
sudo apt-get install -y libgomp1 cmake


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
pip install pandas pyarrow gcsfs google-cloud-storage scikit-learn "lightgbm" fuzzywuzzy optuna matplotlib seaborn python-Levenshtein tqdm shap tensorflow

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
DEVICE = 'cpu'
NUM_FOLDS_OPTUNA = 3 # Use fewer folds for fast hyperparameter search.
NUM_FOLDS_FINAL = 10 # Use more folds for robust final model training.
BATCH_SIZE = 256 # This will be optimized
EPOCHS = 150 # Reduced for faster trials, can be increased later

# --- PyTorch & Model Classes ---
class Chomp1d(nn.Module):
    def __init__(self, chomp_size): super(Chomp1d, self).__init__(); self.chomp_size = chomp_size
    def forward(self, x): return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2): super(TemporalBlock, self).__init__(); self.conv1 = nn.utils.parametrizations.weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)); self.chomp1 = Chomp1d(padding); self.relu1 = nn.ReLU(); self.dropout1 = nn.Dropout(dropout); self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1); self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None; self.relu = nn.ReLU(); self.init_weights()
    def init_weights(self): self.conv1.weight.data.normal_(0, 0.01); _ = self.downsample.weight.data.normal_(0, 0.01) if self.downsample is not None else None
    def forward(self, x): out = self.net(x); res = x if self.downsample is None else self.downsample(x); return self.relu(out + res)

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
        'S': 'num__property_sub_type_code_from_homipi__5Semi_Detached_hm',
        'T': 'num__property_sub_type_code_from_homipi__6Terraced_hm'
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

def load_and_merge_ae_features(df, encodings_dir):
    """
    [ARCHITECTURALLY CORRECTED] Loads autoencoder features from .npy files and merges them
    into the main dataframe using an index-based join, assuming identical row ordering.
    Returns the merged dataframe and a list of the new column names.
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

    

# --- Rightmove Parsing Functions ---
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
    sales_history_col = None
    
    # --- Defensive Column Finding Logic ---
    # Priority 1: Exact match (columns are lowercased in main()).
    if 'sales_history' in raw_rightmove_df.columns:
        sales_history_col = 'sales_history'
        print(f"  - Found sales history column by exact match: '{sales_history_col}'")
    
    # Priority 2: Substring match.
    if not sales_history_col:
        candidates = [col for col in raw_rightmove_df.columns if 'sales_history' in col]
        if candidates:
            sales_history_col = candidates[0]
            print(f"  - Found sales history column by substring match: '{sales_history_col}'")

    # Priority 3: Positional fallback (last resort with a warning).
    if not sales_history_col and raw_rightmove_df.shape[1] > 2:
        sales_history_col = raw_rightmove_df.columns[2]
        print(f"  - WARNING: Could not find sales history by name. Falling back to 3rd column: '{sales_history_col}'")

    # Final Check: If still not found, raise an informative error.
    if not sales_history_col:
        raise ValueError(
            "Could not find a 'sales_history' column. Processing cannot continue.\n"
            f"Available columns are: {raw_rightmove_df.columns.tolist()}"
        )
        
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


def mean_absolute_percentage_error(y_true, y_pred):
    """Calculates MAPE, ensuring robustness against division by zero and empty inputs."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Avoid division by zero for properties with a price of 0.
    mask = y_true != 0
    
    # FAIL-SAFE: If no valid (non-zero) true values exist, the error is 0.
    if not np.any(mask):
        return 0.0
        
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


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
    
    # Final Fallback: Fail fast if no MSOA column is found.
    # The calling function is responsible for deciding how to handle this.
    raise ValueError(
        "FATAL: No MSOA column found. Could not identify a column with 'MSOA11CD', 'MSOA21CD', or 'msoa'."
    )


def fit_and_create_canonical_artifacts(df, feature_sets, universal_cols_present):
    """
    [ARCHITECTURALLY SIMPLIFIED] Creates a single artifact file containing
    only fitted scalers for ALL feature heads, suitable for tree-based models.
    """
    print("\n--- STAGE 4.5: Fitting Unified Scalers for Tree-Based Models ---")
    canonical_artifacts = {}

    # Create a combined dictionary of all heads for simple iteration
    all_heads = feature_sets.copy()
    all_heads['head_base'] = universal_cols_present

    for head_name, cols in all_heads.items():
        available_cols = [c for c in cols if c in df.columns]
        if not available_cols:
            print(f"  - WARNING: No columns found for '{head_name}'. Skipping scaler fitting.")
            continue

        print(f"  - Fitting scaler for '{head_name}' with {len(available_cols)} features...")
        head_df_raw = df[available_cols].copy().apply(pd.to_numeric, errors='coerce').fillna(0)
        
        # Use a robust scaler suitable for tree models
        scaler = QuantileTransformer(output_distribution='normal', n_quantiles=max(min(len(df) // 10, 100), 10), random_state=42)
        scaler.fit(head_df_raw)
        
        canonical_artifacts[head_name] = {
            'type': 'scaler',
            'scaler': scaler,
            'original_cols': available_cols
        }
    
    return canonical_artifacts

def transform_with_canonical_artifacts(df, artifacts):
    """
    [ARCHITECTURALLY SIMPLIFIED] Transforms a raw dataframe into a fully scaled one
    using the canonical scaler artifacts.
    """
    df_transformed = df.copy()
    
    for head_name, config in artifacts.items():
        if config.get('type') == 'scaler':
            expected_cols = config['original_cols']
            available_cols = [col for col in expected_cols if col in df.columns]
            
            if not available_cols:
                continue

            head_df_raw = df[available_cols].copy().apply(pd.to_numeric, errors='coerce').fillna(0)
            
            # Handle missing columns in the dataframe being transformed (e.g., holdout set)
            missing_cols = set(expected_cols) - set(available_cols)
            if missing_cols:
                for col in missing_cols:
                    head_df_raw[col] = 0
            head_df_raw = head_df_raw[expected_cols] # Ensure column order

            scaled_data = config['scaler'].transform(head_df_raw)
            df_transformed[expected_cols] = scaled_data
    
    return df_transformed





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

    # Clean up the temporary helper column to prevent dtype contamination downstream.
    df_copy.drop(columns=['property_type_char'], inplace=True, errors='ignore')

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
    print("  - Report and plots saved to output directory.")

def create_expert_interaction_features(df_expert_preds):
    """ Creates interaction features from the most powerful expert predictions. """
    interactions_df = pd.DataFrame(index=df_expert_preds.index)
    
    # Define our top experts based on SHAP analysis
    dna_raw = 'oof_pred_head_A_dna_raw'
    aesthetic = 'oof_pred_head_B_aesthetic'
    census = 'oof_pred_head_C_census'
    
    # Check if columns exist before creating interactions to prevent KeyErrors
    if dna_raw in df_expert_preds and census in df_expert_preds:
        interactions_df['dna_minus_census'] = df_expert_preds[dna_raw] - df_expert_preds[census]
    
    if dna_raw in df_expert_preds and aesthetic in df_expert_preds:
        interactions_df['dna_x_aesthetic'] = df_expert_preds[dna_raw] * df_expert_preds[aesthetic]

    return interactions_df

def train_final_model(df_train_raw, df_tune_raw, feature_sets, specialist_config, universal_cols_present):
    """
    [ARCHITECTURALLY REBUILT] Implements a Tune-then-Train workflow.
    """
    # --- PHASE 1: Generate Specialist OOF Predictions for BOTH Train and Tune sets ---
    print("\n--- PHASE 1: Generating Specialist OOF Predictions ---")
    df_main_raw = pd.concat([df_train_raw, df_tune_raw])
    y_main_raw = df_main_raw['most_recent_sale_price']
    y_main_log = np.log1p(y_main_raw)
    
    oof_specialist_preds_df = pd.DataFrame(index=df_main_raw.index)
    kf_specialist = KFold(n_splits=NUM_FOLDS_FINAL, shuffle=True, random_state=42)
    
    specialist_heads_to_train = list(specialist_config.keys())
    for head_name in specialist_heads_to_train:
        if head_name not in feature_sets or not feature_sets[head_name]: continue
        print(f"  - Training specialist for '{head_name}'...")
        X_specialist = df_main_raw[feature_sets[head_name]].fillna(0)
        oof_preds_for_head = np.zeros(len(df_main_raw))
        for fold, (train_idx, val_idx) in enumerate(kf_specialist.split(X_specialist)):
            model = lgb.LGBMRegressor(**specialist_config[head_name])
            model.fit(X_specialist.iloc[train_idx], y_main_log.iloc[train_idx], eval_set=[(X_specialist.iloc[val_idx], y_main_log.iloc[val_idx])], eval_metric='mae', callbacks=[lgb.early_stopping(50, verbose=False)])
            oof_preds_for_head[val_idx] = model.predict(X_specialist.iloc[val_idx])
        oof_specialist_preds_df[f'oof_pred_{head_name}'] = oof_preds_for_head

    # --- PHASE 2: Create Augmented Datasets ---
    print("\n--- PHASE 2: Assembling Augmented Datasets for Tuning and Training ---")
    all_original_features = [col for col in df_main_raw.columns if col not in ['most_recent_sale_price', 'property_id', 'normalized_address_key', 'final_merge_key', 'rightmove_address_text']]
    interaction_features_df = create_expert_interaction_features(oof_specialist_preds_df)
    X_fusion_main_aug = pd.concat([df_main_raw[all_original_features].fillna(0), oof_specialist_preds_df, interaction_features_df], axis=1)
    
    # Split the augmented data back into train and tune sets
    X_fusion_train_aug = X_fusion_main_aug.loc[df_train_raw.index]
    X_fusion_tune_aug = X_fusion_main_aug.loc[df_tune_raw.index]
    y_train_log = y_main_log.loc[df_train_raw.index]
    y_tune_log = y_main_log.loc[df_tune_raw.index]

    # --- PHASE 3: Run a SINGLE Optuna Study ---
    print("\n--- PHASE 3: Running Single Optuna Study on Tuning Set ---")
    def objective_single(trial):
        params = {
            'n_estimators': 2000, 'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 50), 'max_depth': trial.suggest_int('max_depth', 5, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 1.0, log=True), 'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 1.0, log=True),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.8), 'subsample': trial.suggest_float('subsample', 0.5, 0.8),
            'random_state': 42, 'n_jobs': -1
        }
        model = lgb.LGBMRegressor(**params)
        model.fit(X_fusion_train_aug, y_train_log, eval_set=[(X_fusion_tune_aug, y_tune_log)], eval_metric='mae', callbacks=[lgb.early_stopping(50, verbose=False)])
        preds_log = model.predict(X_fusion_tune_aug)
        mae = mean_absolute_error(np.expm1(y_tune_log), np.expm1(preds_log))
        return mae

    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective_single, n_trials=20)
    best_fusion_params = study.best_params
    print(f"  - Optuna search complete. Best MAE on tuning set: £{study.best_value:,.2f}")
    print(f"  - Best overall fusion params: {best_fusion_params}")

    # --- PHASE 4: Train Final K-Fold Model with Best Parameters ---
    print("\n--- PHASE 4: Training Final K-Fold Model on Combined Data ---")
    kf_final = KFold(n_splits=NUM_FOLDS_FINAL, shuffle=True, random_state=42)
    oof_fusion_preds = np.zeros(len(X_fusion_main_aug))
    
    for fold, (train_idx, val_idx) in enumerate(kf_final.split(X_fusion_main_aug)):
        print(f"  - Training Fold {fold+1}/{NUM_FOLDS_FINAL}...")
        X_train, X_val = X_fusion_main_aug.iloc[train_idx], X_fusion_main_aug.iloc[val_idx]
        y_train, y_val = y_main_log.iloc[train_idx], y_main_log.iloc[val_idx]
        model = lgb.LGBMRegressor(n_estimators=2500, **best_fusion_params, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='mae', callbacks=[lgb.early_stopping(50, verbose=False)])
        oof_fusion_preds[val_idx] = model.predict(X_val)

    # --- PHASE 5: Final Artifact Preparation ---
    print("\n--- PHASE 5: Finalizing Models for Inference ---")
    final_fusion_model = lgb.LGBMRegressor(n_estimators=2500, **best_fusion_params, random_state=42, n_jobs=-1)
    final_fusion_model.fit(X_fusion_main_aug, y_main_log)
    trained_models = {'fusion_model': final_fusion_model}
    
    for head_name in specialist_heads_to_train:
        if head_name not in feature_sets or not feature_sets[head_name]: continue
        print(f"  - Retraining final specialist for '{head_name}' on all data...")
        X_specialist_full = df_main_raw[feature_sets[head_name]].fillna(0)
        final_specialist_model = lgb.LGBMRegressor(**specialist_config[head_name])
        final_specialist_model.fit(X_specialist_full, y_main_log)
        trained_models[f'specialist_{head_name}'] = final_specialist_model

    oof_df = df_main_raw[['property_id', 'most_recent_sale_price']].copy()
    oof_df['final_predicted_price'] = np.expm1(oof_fusion_preds)
    
    return trained_models, oof_df, X_fusion_main_aug, X_fusion_main_aug.columns.tolist()


def generate_shap_reports_for_holdout(df_main_raw, df_holdout_raw, trained_models, feature_sets, universal_cols_present, holdout_results_df, output_dir, canonical_feature_set):
    """
    [ARCHITECTURALLY REBUILT] Generates SHAP explanations for the two-stage ensemble.
    This version correctly handles the additive nature of the primary and residual models.
    """
    import shap
    print("\n--- Generating SHAP Explanations for Two-Stage Ensemble Model ---")
    if df_holdout_raw.empty:
        print("  - Holdout set is empty. Skipping SHAP report generation.")
        return
    if 'residual_model' not in trained_models:
        print("  - WARNING: Residual model not found in artifact. Generating SHAP for primary model only.")
        # Fallback to old behavior if residual model is missing
        # (This part is simplified as the main path assumes its presence)
        return 

    # --- 1. Prepare the augmented feature set for the explainer ---
    specialist_heads = [k.replace('specialist_', '') for k in trained_models if k.startswith('specialist_')]
    
    def get_augmented_features(df):
        specialist_preds_df = pd.DataFrame(index=df.index)
        for head_name in specialist_heads:
            model = trained_models[f'specialist_{head_name}']
            head_cols = feature_sets[head_name]
            X_specialist = df[head_cols].fillna(0)
            specialist_preds_df[f'oof_pred_{head_name}'] = model.predict(X_specialist)
        
        all_original_features = [
            col for col in df.columns if col not in ['most_recent_sale_price', 'property_id', 'normalized_address_key', 'final_merge_key', 'rightmove_address_text']
        ]
        interaction_features_df = create_expert_interaction_features(specialist_preds_df)
        return pd.concat([df[all_original_features].fillna(0), specialist_preds_df, interaction_features_df], axis=1)

    print("  - Preparing background data for SHAP explainers...")
    background_data = df_main_raw.sample(min(200, len(df_main_raw)), random_state=42)
    X_background_augmented = get_augmented_features(background_data).reindex(columns=canonical_feature_set, fill_value=0)
    
    print("  - Preparing holdout data for SHAP explanation...")
    X_holdout_augmented = get_augmented_features(df_holdout_raw).reindex(columns=canonical_feature_set, fill_value=0)

    # --- 2. Sanitize feature sets and run Explainers for BOTH models ---
    fusion_model = trained_models['fusion_model']
    residual_model = trained_models['residual_model']

    # ARCHITECTURAL MANDATE: Dynamically sanitize feature sets for Model 2 explanation.
    print("  - Sanitizing feature sets for Model 2 SHAP explainer...")
    leaky_prefixes = ('oof_pred_', 'dna_minus_', 'dna_x_')
    cols_to_drop_for_m2 = [c for c in X_background_augmented.columns if c.startswith(leaky_prefixes)]
    X_background_sanitized = X_background_augmented.drop(columns=cols_to_drop_for_m2)
    X_holdout_sanitized = X_holdout_augmented.drop(columns=cols_to_drop_for_m2)
    print(f"    - Removed {len(cols_to_drop_for_m2)} leaky features for Model 2 explanation.")

    print("  - Initializing SHAP TreeExplainer for Model 1 (Fusion) on full feature set...")
    explainer_m1 = shap.TreeExplainer(fusion_model, X_background_augmented)
    print("  - Initializing SHAP TreeExplainer for Model 2 (Residual) on SANITIZED feature set...")
    explainer_m2 = shap.TreeExplainer(residual_model, X_background_sanitized)

    print(f"  - Calculating SHAP values for {len(X_holdout_augmented)} holdout properties (both models)...")
    shap_values_m1 = explainer_m1.shap_values(X_holdout_augmented)

    # ARCHITECTURAL MANDATE: Load and enforce the canonical feature set.
    residual_model_features = trained_models['residual_model_features']
    X_background_sanitized_for_m2 = X_background_augmented[residual_model_features]
    X_holdout_sanitized_for_m2 = X_holdout_augmented[residual_model_features]
    
    print("  - Calculating SHAP values for Model 2...")
    # This call will now succeed because the dataframe schema is guaranteed to be correct.
    shap_values_m2 = explainer_m2.shap_values(X_holdout_sanitized_for_m2)

    # --- 3. Combine SHAP values from models with different feature sets ---
    # Create DataFrames to handle column alignment
    shap_df_m1 = pd.DataFrame(shap_values_m1, columns=X_holdout_augmented.columns, index=X_holdout_augmented.index)
    shap_df_m2 = pd.DataFrame(shap_values_m2, columns=X_holdout_sanitized.columns, index=X_holdout_sanitized.index)

    # Reindex M2's SHAP values to the full feature space, filling with 0 for the features it didn't see.
    shap_df_m2_reindexed = shap_df_m2.reindex(columns=shap_df_m1.columns, fill_value=0)

    # ARCHITECTURAL MANDATE: Combine SHAP values via element-wise addition.
    shap_df_combined = shap_df_m1 + shap_df_m2_reindexed

    # --- 4. Format and save the report ---
    shap_df = shap_df_combined
    
    report_df = holdout_results_df.copy()
    # The combined base value is the sum of the individual model base values.
    report_df['shap_base_value'] = explainer_m1.expected_value + explainer_m2.expected_value
    
    top_contributors = []
    for i in tqdm(range(len(shap_df)), desc="Aggregating SHAP values"):
        top_series = shap_df.iloc[i].abs().nlargest(5)
        contribs = {}
        for j, (feature, value) in enumerate(top_series.items()):
            contribs[f'contrib_feat_{j+1}'] = feature
            contribs[f'contrib_feat_{j+1}_shap'] = shap_df.iloc[i][feature]
        top_contributors.append(contribs)
    
    top_contributors_df = pd.DataFrame(top_contributors, index=shap_df.index)
    final_report_df = pd.concat([report_df, top_contributors_df], axis=1)
    
    report_path = os.path.join(output_dir, "shap_structured_report.csv")
    final_report_df.to_csv(report_path, index=False, float_format='%.6f')
    print(f"\nSuccessfully generated and saved combined structured SHAP report to {report_path}")



def predict_on_holdout(df_holdout_raw, trained_models, feature_sets, universal_cols_present, canonical_feature_set):
    """
    [ARCHITECTURALLY REBUILT] Generates predictions on the holdout set using the
    two-stage ensemble with residual correction.
    """
    print("\n--- STAGE 10: Final Evaluation on True Holdout Set ---")
    if df_holdout_raw.empty:
        return pd.DataFrame()

    # --- STAGE 1: Generate predictions from specialist models ---
    print("  - Generating predictions from specialist models...")
    holdout_specialist_preds_df = pd.DataFrame(index=df_holdout_raw.index)
    specialist_heads = [k.replace('specialist_', '') for k in trained_models if k.startswith('specialist_')]
    
    for head_name in specialist_heads:
        model = trained_models[f'specialist_{head_name}']
        head_cols = feature_sets[head_name]
        X_holdout_specialist = df_holdout_raw[head_cols].fillna(0)
        
        preds = model.predict(X_holdout_specialist)
        holdout_specialist_preds_df[f'oof_pred_{head_name}'] = preds

    # --- STAGE 2: Prepare Augmented Feature Set for Both Models ---
    print("  - Engineering expert interaction features for holdout set...")
    all_original_features = [
        col for col in df_holdout_raw.columns if col not in ['most_recent_sale_price', 'property_id', 'normalized_address_key', 'final_merge_key', 'rightmove_address_text']
    ]
    interaction_features_holdout_df = create_expert_interaction_features(holdout_specialist_preds_df)
    X_fusion_holdout = pd.concat([
        df_holdout_raw[all_original_features].fillna(0),
        holdout_specialist_preds_df,
        interaction_features_holdout_df
    ], axis=1)

    # ARCHITECTURAL MANDATE: Enforce the canonical schema from training to prevent drift.
    X_fusion_holdout = X_fusion_holdout.reindex(columns=canonical_feature_set, fill_value=0)

    # --- STAGE 3: Generate Two-Stage Prediction ---
    print("  - Generating two-stage prediction (Primary + Residual)...")
    fusion_model = trained_models['fusion_model']
    
    # Model 1: Predict in log-space and convert to real-space using the full augmented feature set
    m1_preds_log = fusion_model.predict(X_fusion_holdout)
    m1_preds_real = np.expm1(m1_preds_log)
    
    final_preds = m1_preds_real
    
    # Model 2: If available, predict residual on SANITIZED data and add to primary prediction
    if 'residual_model' in trained_models:
        residual_model = trained_models['residual_model']
        
        # ARCHITECTURAL MANDATE: Dynamically derive the sanitized feature set from the
        # canonical dataframe to guarantee consistency and prevent latent KeyErrors.
        leaky_prefixes = ('oof_pred_', 'dna_minus_', 'dna_x_')
        cols_to_drop_for_m2 = [c for c in X_fusion_holdout.columns if c.startswith(leaky_prefixes)]
        X_holdout_sanitized_for_m2 = X_fusion_holdout.drop(columns=cols_to_drop_for_m2)

        # The model is now guaranteed to receive a dataframe with the correct schema,
        # as this logic perfectly mirrors the training process.
        m2_preds_residual = residual_model.predict(X_holdout_sanitized_for_m2)
        final_preds += m2_preds_residual
        print(f"  - Residual correction applied after removing {len(cols_to_drop_for_m2)} leaky features.")
    else:
        print("  - WARNING: Residual model not found. Returning primary prediction only.")

    results_df = df_holdout_raw[['property_id', 'most_recent_sale_price']].copy()
    results_df['predicted_price'] = final_preds
    
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
    # [SURGICALLY REMOVED] Obsolete print statement referencing the deleted 'DEVICE' variable.
    os.makedirs(OUTPUT_DIR, exist_ok=True)


    # --- Read Tuned Dimensions from Environment ---
    print("\n--- Using Static Autoencoder Hyperparameters ---")
    # [SURGICALLY REMOVED] AE_BEST_PARAMS is obsolete in the new LGBM architecture.
    print(f"  - Using AE Best Params: {{}}")


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
        # We must now load ALL columns initially, as the AE features need to be merged by index.
        # The `required_columns` logic was causing the original `gsutil` crash.
        df_features = pd.read_parquet(MASTER_DATA_PATH)

        # NEW STEP: Load and merge the external Head G features
        AE_ENCODINGS_PATH = os.environ.get("AE_ENCODINGS_LOCAL_DIR")
        df_features, head_g_new_cols = load_and_merge_ae_features(df_features, AE_ENCODINGS_PATH)
        
        # CRITICAL: Register these new columns as head_G features
        if head_g_new_cols:
            feature_sets['head_G_gemini_quantitative'] = head_g_new_cols
            print(f"  - Registered {len(head_g_new_cols)} features for 'head_G_gemini_quantitative'.")
        
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
                state_dict = torch.load(model_path, map_location=torch.device('cpu'))
                model.load_state_dict(state_dict)
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
    # Define all columns to bring in from the Rightmove dataset
    if address_col_rm not in df_rightmove_with_address_key.columns:
        raise ValueError(f"FATAL: Rightmove address column '{address_col_rm}' missing before final merge.")
    
    # ARCHITECTURALLY ROBUST: Explicitly rename the address column to prevent merge conflicts.
    df_rightmove_with_address_key.rename(columns={address_col_rm: 'rightmove_address_text'}, inplace=True)
    
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
    
    # --- NEW: Data Quality Filter ---
    print("\n--- Filtering for High-Confidence Merges ---")
    address_col_rm = [col for col in df.columns if 'address' in col.lower()][0]
    
    def normalize_for_comparison(s):
        """Canonicalizes an address string by lowercasing and removing non-alphanumerics."""
        raw_str = str(s)
        # Proactively handle stringified lists from upstream data issues.
        if raw_str.startswith('[') and raw_str.endswith(']'):
            try:
                # Use ast.literal_eval for safe evaluation of the string as a Python literal.
                s_list = ast.literal_eval(raw_str)
                if isinstance(s_list, list) and s_list:
                    raw_str = str(s_list[0]) # Use the first element of the list
            except (ValueError, SyntaxError):
                pass # If parsing fails, fall back to using the raw string.
        return re.sub(r'[^a-z0-9]', '', raw_str.lower())

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

    # --- TERMINAL DATA SANITIZATION GATE ---
    # ARCHITECTURAL MANDATE: Define and isolate all non-feature columns to prevent leakage.
    IDENTIFIER_COLS = [
        'property_id',
        'normalized_address_key',
        'final_merge_key',
        'rightmove_address_text'
    ]
    # Retain these for reporting, but they must never enter a model.
    present_identifiers = [col for col in IDENTIFIER_COLS if col in df.columns]
    df_identifiers = df[present_identifiers]
    df_features_only = df.drop(columns=present_identifiers, errors='ignore')

    print("\n--- Enforcing Universal Numeric Contract for Modeling ---")
    # Convert all remaining object columns to integer codes
    for col in df_features_only.select_dtypes(include=['object', 'category']).columns:
        print(f"  - Factorizing high-cardinality categorical column: {col}")
        # pd.factorize returns codes and unique values. We only need the codes.
        df_features_only[col], _ = pd.factorize(df_features_only[col])

    # At this point, df_features_only is guaranteed to be 100% numeric.
    # Now, we can re-join the identifiers for splitting and reporting.
    df = pd.concat([df_identifiers, df_features_only], axis=1)

    # --- STAGE 5: Filter, Validate, and Prepare Data for Model ---
    print("\n--- STAGE 5: Filtering, Validating and Splitting Data ---")
    min_price, max_price = 10000, df['most_recent_sale_price'].quantile(0.999)
    df = df[(df['most_recent_sale_price'] >= min_price) & (df['most_recent_sale_price'] <= max_price)].copy().reset_index(drop=True)

    n_holdout = 0.05
    # Split into three sets: train (for model weights), tune (for HPs), and holdout (for final eval)
    df_main_raw, df_holdout_raw = train_test_split(df, test_size=n_holdout, random_state=42)
    df_train_raw, df_tune_raw = train_test_split(df_main_raw, test_size=0.2, random_state=42)
    
    print(f"  - Split data into {len(df_train_raw)} training rows, {len(df_tune_raw)} tuning rows, and {len(df_holdout_raw)} holdout rows.")

    # --- NEW: Configuration for the Council of LGBM Experts ---
    SPECIALIST_LGBM_CONFIG = {
        'head_A_dna_raw': {'n_estimators': 1000, 'num_leaves': 31, 'learning_rate': 0.05, 'random_state': 42, 'n_jobs': -1, 'colsample_bytree': 0.8},
        'head_C_census': {'n_estimators': 1000, 'num_leaves': 31, 'learning_rate': 0.05, 'random_state': 42, 'n_jobs': -1, 'colsample_bytree': 0.8},
        'head_compass_raw': {'n_estimators': 1500, 'num_leaves': 61, 'learning_rate': 0.03, 'random_state': 42, 'n_jobs': -1, 'colsample_bytree': 0.7},
        'head_B_aesthetic': {'n_estimators': 1000, 'num_leaves': 31, 'learning_rate': 0.05, 'random_state': 42, 'n_jobs': -1, 'colsample_bytree': 0.8},
        'head_A_dna_engineered': {'n_estimators': 1200, 'num_leaves': 41, 'learning_rate': 0.04, 'random_state': 42, 'n_jobs': -1, 'colsample_bytree': 0.7},
        'head_G_gemini_quantitative': {'n_estimators': 1200, 'num_leaves': 41, 'learning_rate': 0.04, 'random_state': 42, 'n_jobs': -1, 'colsample_bytree': 0.7},
    }
    print(f"Defined {len(SPECIALIST_LGBM_CONFIG)} specialist heads for the ensemble.")


    # The new LGBM architecture operates on raw data; pre-transformation is not required.
    universal_cols_present = [col for col in df_main_raw.columns if col in UNIVERSAL_PREDICTORS]
    y_price_log = np.log1p(df_main_raw['most_recent_sale_price'])

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

    # --- STAGES 6 & 7: Train Final "Council of Experts" Ensemble ---
    trained_models, eval_df, X_fusion_main_aug, canonical_feature_set = train_final_model(
        df_train_raw=df_train_raw,
        df_tune_raw=df_tune_raw,
        feature_sets=feature_sets,
        specialist_config=SPECIALIST_LGBM_CONFIG,
        universal_cols_present=universal_cols_present
    )

    # --- STAGE 8: Evaluating Primary Model OOF Performance ---
    print("\n--- STAGE 8: Evaluating Final OOF Performance (Model 1) ---")
    if eval_df is None:
        raise ValueError("FATAL: Model training failed.")

    final_mae_m1 = mean_absolute_error(eval_df['most_recent_sale_price'], eval_df['final_predicted_price'])
    final_r2_m1 = r2_score(eval_df['most_recent_sale_price'], eval_df['final_predicted_price'])
    print(f"\n--- [Model 1] OOF PERFORMANCE (COUNCIL OF LGBM EXPERTS) ---")
    print(f"  - [Model 1] Final MAE: £{final_mae_m1:,.2f}, R²: {final_r2_m1:.4f}")

    # --- ARCHITECTURAL MANDATE: RESIDUAL FITTING STAGE (MODEL 2) ---
    print("\n--- STAGE 9: Training Residual Correction Model (Model 2) ---")
    
    # 1. CRITICAL: Calculate residuals in REAL-SPACE using OOF predictions to prevent target leakage.
    eval_df['residual'] = eval_df['most_recent_sale_price'] - eval_df['final_predicted_price']

    # 2. Use the pre-computed augmented feature set from the previous stage.
    # The entire redundant "Re-generating..." block has been surgically excised.
    y_residual = eval_df['residual'].loc[X_fusion_main_aug.index]

    # ARCHITECTURAL MANDATE: Sanitize feature set for Model 2 to prevent leakage.
    print("  - Sanitizing feature set for Model 2 to remove OOF-derived features...")
    leaky_prefixes = ('oof_pred_', 'dna_minus_', 'dna_x_')
    cols_to_drop_for_m2 = [c for c in X_fusion_main_aug.columns if c.startswith(leaky_prefixes)]
    X_residual_sanitized = X_fusion_main_aug.drop(columns=cols_to_drop_for_m2)
    print(f"    - Removed {len(cols_to_drop_for_m2)} leaky features for Model 2 training.")

    # 3. Train Model 2 on SANITIZED data with robust K-Fold CV.
    print("  - Training K-Fold residual model to get OOF-based combined metric...")
    # ARCHITECTURAL MANDATE: Use a weaker, more regularized model for the residual task
    # to prevent overfitting the noise from Model 1's errors.
    residual_model_params = {
        'n_estimators': 750,
        'learning_rate': 0.05,
        'num_leaves': 15,
        'reg_alpha': 0.5,
        'reg_lambda': 0.5,
        'colsample_bytree': 0.7,
        'random_state': 1337, 
        'n_jobs': -1
    }
    kf_residual = KFold(n_splits=NUM_FOLDS_FINAL, shuffle=True, random_state=1337)
    oof_residual_preds = np.zeros(len(X_residual_sanitized))

    for fold, (train_idx, val_idx) in enumerate(kf_residual.split(X_residual_sanitized)):
        X_train, X_val = X_residual_sanitized.iloc[train_idx], X_residual_sanitized.iloc[val_idx]
        y_train, y_val = y_residual.iloc[train_idx], y_residual.iloc[val_idx]
        res_model = lgb.LGBMRegressor(**residual_model_params)
        res_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='mae', callbacks=[lgb.early_stopping(50, verbose=False)])
        oof_residual_preds[val_idx] = res_model.predict(X_val)

    # 4. Finalize and store Model 2 in the master artifact.
    print("  - Training final residual model on all SANITIZED data...")
    final_residual_model = lgb.LGBMRegressor(**residual_model_params)
    final_residual_model.fit(X_residual_sanitized, y_residual)
    trained_models['residual_model'] = final_residual_model
    
    # Save the master model artifact AFTER adding the residual model.
    # The obsolete logic for saving residual_model_features has been removed.
    joblib.dump(trained_models, os.path.join(OUTPUT_DIR, "council_of_experts_models.joblib"))

    # 5. Report final combined performance.
    combined_oof_preds = eval_df['final_predicted_price'] + oof_residual_preds
    combined_mae = mean_absolute_error(eval_df['most_recent_sale_price'], combined_oof_preds)
    combined_r2 = r2_score(eval_df['most_recent_sale_price'], combined_oof_preds)
    
    print(f"\n--- [Model 1 + Model 2] COMBINED OOF PERFORMANCE ---")
    print(f"  - Final Combined MAE: £{combined_mae:,.2f}, R²: {combined_r2:.4f}")
    
    eval_df['residual_prediction'] = oof_residual_preds
    eval_df['final_combined_prediction'] = combined_oof_preds
    eval_df.to_csv(os.path.join(OUTPUT_DIR, "oof_predictions.csv"), index=False)
    print("OOF predictions and master model artifact saved successfully.")

    # --- STAGE 10: Final Holdout Set Evaluation ---
    print("\n--- STAGE 10: Final Evaluation on True Holdout Set ---")
    if not df_holdout_raw.empty:
        # Load the master model artifact
        trained_models = joblib.load(os.path.join(OUTPUT_DIR, "council_of_experts_models.joblib"))
        
        holdout_results = predict_on_holdout(
            df_holdout_raw,
            trained_models,
            feature_sets,
            universal_cols_present,
            canonical_feature_set
        )
        
        holdout_results['absolute_error'] = (holdout_results['predicted_price'] - holdout_results['most_recent_sale_price']).abs()
        print("\n--- HOLDOUT SET FINAL RESULTS ---")
        final_mae_holdout = holdout_results['absolute_error'].mean()
        print(f"  - Holdout MAE:  £{final_mae_holdout:,.2f}")
        holdout_results.to_csv(os.path.join(OUTPUT_DIR, "holdout_evaluation_results.csv"), index=False)

        # Generate Reports
        generate_final_report(eval_df, holdout_results, baseline_mae, OUTPUT_DIR)

        # --- STAGE 11: Save Final Universal Artifacts for Inference ---
        # Note: The main artifact is now council_of_experts_models.joblib
        print("\n--- STAGE 11: Finalizing Artifacts for Inference Pipeline ---")

        # --- STAGE 12: Generate Batch SHAP Explanations ---
        print("\n--- STAGE 12: Generating SHAP reports for 'Council of Experts' model ---")
        generate_shap_reports_for_holdout(
            df_main_raw=df_main_raw,
            df_holdout_raw=df_holdout_raw,
            trained_models=trained_models,
            feature_sets=feature_sets,
            universal_cols_present=universal_cols_present,
            holdout_results_df=holdout_results,
            output_dir=OUTPUT_DIR,
            canonical_feature_set=canonical_feature_set
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

LOG_FILE="./output/training_run.log" # Path relative to the new CWD
# Execute the script, saving the full log
python3 -u "${SCRIPT_PATH}" 2>&1 | tee -a "${LOG_FILE}"

# ARCHITECTURAL MANDATE: Convert warnings into a quantitative health metric.
echo "\n--- Post-Hoc Analysis: Quantifying Benign Warnings ---" | tee -a "${LOG_FILE}"
WARNING_COUNT=$(grep -c "No further splits with positive gain" "${LOG_FILE}")
echo "Total 'No Positive Gain' Warnings: ${WARNING_COUNT}" | tee -a "${LOG_FILE}"
echo "NOTE: This is a health metric. A high number is expected and indicates Model 2 is correctly resisting overfitting noise." | tee -a "${LOG_FILE}"


echo "Uploading all training artifacts to GCS..."
gsutil -m cp -r "${OUTPUT_DIR}/*" "${OUTPUT_GCS_DIR}/"

echo "all done! The model training artifacts are available at ${OUTPUT_GCS_DIR}."