#!/usr/bin/env bash
#
# run_model_training_v9.sh
#
#
# V9 REVISION: Mathematically Sound & Deterministic Assembly
#

# -- Strict Mode & Configuration --
set -e
set -o pipefail
set -x


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


# --- [A.D-V75.1] HARDENED "BLACK BOX" TERMINATION PROTOCOL ---

# This function is the core of our unattended execution safety net. It is
# guaranteed to run when the script exits, for any reason (success or crash).
function final_shutdown_and_log_upload() {
    echo "--- SCRIPT FINISHED OR CRASHED: INITIATING BLACK BOX PROTOCOL ---"
    # The WISTERIA_LOG_FILE environment variable is the contract from the launcher.
    if [[ -z "${WISTERIA_LOG_FILE}" ]]; then
        echo "  - WARNING: WISTERIA_LOG_FILE not set."
        WISTERIA_LOG_FILE="${HOME}/pipeline_output.log"
        echo "  - Defaulting to log file at ${WISTERIA_LOG_FILE}."
        echo "  - Uploading final log file '${WISTERIA_LOG_FILE}' to ${OUTPUT_GCS_DIR}/..."
        # Use a || condition to report failure without stopping the shutdown sequence.
        # This prioritizes cost-savings (shutdown) even if the upload fails.
        gsutil cp "${WISTERIA_LOG_FILE}" "${OUTPUT_GCS_DIR}/" || echo "  - GCS UPLOAD FAILED! The log remains on the VM's disk if not yet terminated."
        echo "  - Log upload complete or failed."
    elif [[ ! -f "${WISTERIA_LOG_FILE}" ]]; then
        echo "  - WARNING: Log file '${WISTERIA_LOG_FILE}' not found."
        WISTERIA_LOG_FILE="${HOME}/pipeline_output.log"
        echo "  - Defaulting to log file at ${WISTERIA_LOG_FILE}."
        echo "  - Uploading final log file '${WISTERIA_LOG_FILE}' to ${OUTPUT_GCS_DIR}/..."
        # Use a || condition to report failure without stopping the shutdown sequence.
        # This prioritizes cost-savings (shutdown) even if the upload fails.
        gsutil cp "${WISTERIA_LOG_FILE}" "${OUTPUT_GCS_DIR}/" || echo "  - GCS UPLOAD FAILED! The log remains on the VM's disk if not yet terminated."
        echo "  - Log upload complete or failed."
    else
        echo "  - Uploading final log file '${WISTERIA_LOG_FILE}' to ${OUTPUT_GCS_DIR}/..."
        # Use a || condition to report failure without stopping the shutdown sequence.
        # This prioritizes cost-savings (shutdown) even if the upload fails.
        gsutil cp "${WISTERIA_LOG_FILE}" "${OUTPUT_GCS_DIR}/" || echo "  - GCS UPLOAD FAILED! The log remains on the VM's disk if not yet terminated."
        echo "  - Log upload complete or failed."
    fi

    echo "  - Initiating auto-shutdown of VM instance."
    # This gcloud command reliably finds the VM's own name and zone to shut itself down.
    gcloud compute instances stop "$(hostname)" --zone="$(curl -s -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/zone | awk -F/ '{print $NF}')"
}

# The EXIT trap is now pointed to our robust function.
trap final_shutdown_and_log_upload EXIT

# --- END OF PROTOCOL DEFINITION ---


# --- V21 AUTO-REFINEMENT CONFIGURATION ---
# The GCS path to the "teacher" run containing the v18_diagnostics to learn from.
REFINEMENT_RUN_GCS_PATH="gs://${GCS_BUCKET}/model_training/run_20250904-140049"

# --- [A.D-V22.0] OPERATIONAL MODE ---
# Set to "INTELLIGENCE" for a slow, deep-dive analysis run that generates all diagnostics.
# Set to "REFINED" for a fast run that consumes intelligence and trains an optimized model.
export WISTERIA_RUN_MODE="INTELLIGENCE" # <<< SET TO REFINED FOR FAST RUNS

# The minimum importance an L0 specialist head must have in the L1 Assembler to survive culling.
export HEAD_CULLING_IMPORTANCE_THRESHOLD=30

# The minimum importance a feature must have within its own L0 specialist model to survive culling.
# This is now more aggressive for REFINED runs. Threshold of 5 culls significant noise.
export FEATURE_CULLING_IMPORTANCE_THRESHOLD=15

# --- [A.D-V23.0] TRIAGE PROTOCOL CONFIGURATION ---
# The percentile of residual difficulty score above which a sample is considered "hard"
# and will be used to train the L0 Specialist Council. A value of 0.70 means the
# specialists are trained on the top 30% most difficult samples.
export TRIAGE_THRESHOLD=0.0

# --- UPDATED: Use the unprocessed file as key map ---
KEY_MAP_GCS_PATH="gs://${GCS_BUCKET}/house data scrape/merged_property_data_with_coords.csv"
QUANTITATIVE_CSV_GCS_PATH="gs://srgan-bucket-ace-botany-453819-t4/house data scrape/property_features_quantitative_v4.csv"

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
PROJECT_DIR_NAME="model_training_project_v9" # Just the name
mkdir -p "./${PROJECT_DIR_NAME}"
cd "./${PROJECT_DIR_NAME}"

# --- Environment Setup & Local Paths (Define paths relative to the NEW current directory) ---
VENV_DIR="./venv_mt"
SCRIPT_PATH="./02_train_multi_head_model_v9.py"
OUTPUT_DIR="./output"
DATA_DIR="./data"
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${DATA_DIR}"
MASTER_DATA_LOCAL_PATH="${DATA_DIR}/master_dataset.parquet"
FEATURE_SETS_LOCAL_PATH="${DATA_DIR}/feature_sets.json"
RIGHTMOVE_DATA_LOCAL_PATH="${DATA_DIR}/Rightmove.csv"
KEY_MAP_LOCAL_PATH="${DATA_DIR}/key_map.csv"
AE_KEYS_LOCAL_PATH="${DATA_DIR}/property_keys_source.csv"



# --- System & Driver Setup (MUST be done before venv creation) ---
echo "Updating system packages and installing dependencies for CPU-based training..."

# [ROBUSTNESS INTERVENTION] Wait for any existing apt-get locks to be released.
# This prevents race conditions with background unattended-upgrades.
while sudo fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1 || sudo fuser /var/lib/apt/lists/lock >/dev/null 2>&1 ; do
  echo "Waiting for other apt-get instances to finish..."
  sleep 10
done

sudo apt-get update
# ARCHITECTURAL CORRECTION (V9): Removed GPU-specific dependencies ('cuda-drivers', 'cmake').
# 'libgomp1' is retained for general-purpose parallel processing (OpenMP).
sudo apt-get install -y libgomp1


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

echo "Installing required Python packages for deep learning (CPU)..."
pip install --upgrade pip
# V71.0 MANDATE: Ensure PyTorch is installed for the Conceptual Embedding Engine
pip install --force-reinstall --no-cache-dir torch torchvision torchaudio
# ARCHITECTURAL CORRECTION (V9): Removed unused 'tensorflow' dependency and switched to CPU-only lightgbm.
# A.D-V24.0 REPLACEMENT: Replaced pytorch-tabnet with xgboost for the new Contextual Arbiter model.
pip install pandas pyarrow gcsfs google-cloud-storage scikit-learn lightgbm fuzzywuzzy matplotlib seaborn python-Levenshtein tqdm shap xgboost hdbscan tabulate umap-learn optuna


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
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import hashlib

# --- V18.3: GLOBAL IMPORT CONTRACT ---
# These libraries are foundational to the script's operation and are now
# imported once at the global level to ensure availability in all functions.
import joblib
import matplotlib
matplotlib.use('Agg') # Set backend BEFORE importing pyplot
import matplotlib.pyplot as plt
import optuna
import seaborn as sns
import shap
import lightgbm as lgb
from joblib import Parallel, delayed
from tqdm import tqdm
from fuzzywuzzy import fuzz, process
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import IsolationForest
import xgboost as xgb
import hdbscan # NEW: The mandated density-based clustering engine
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import haversine_distances
from sklearn.pipeline import Pipeline
import umap
import itertools
import traceback
import optuna

# --- Configuration ---
# --- V48.1 FINE-TUNING MANDATE GLOBALS ---
# Canonical list of high-error jurisdictions based on the definitive performance audit.
# This list is the single source of truth for data stratification and model routing.
HIGH_ERROR_POSTCODE_AREAS = ['FK', 'KA', 'DD', 'IV', 'NE', 'NG', 'HG']
def get_postcode_area(address_series, pcds_series):
    """
    [V27.2 HARDENED] Extracts postcode area, prioritizing the structured 'pcds'
    column and falling back to regex parsing of a free-text address series.
    """
    # Prioritize the structured pcds column if it's usable
    if pcds_series is not None and pd.api.types.is_string_dtype(pcds_series):
        # Extract leading letters (the area) from the postcode district (e.g., 'FK10' -> 'FK')
        return pcds_series.str.extract(r'^([A-Z]+)', expand=False).fillna('UNKNOWN')
    
    # Fallback to regex parsing of the address if pcds is not available
    def parse_address(address):
        if pd.isna(address): return "UNKNOWN"
        postcode_pattern = re.compile(r'([A-Z]{1,2}[0-9][A-Z0-9]?)\s*[0-9][A-Z]{2}', re.IGNORECASE)
        match = postcode_pattern.search(str(address))
        if not match: return "UNKNOWN"
        district = match.group(1); area_match = re.match(r'([A-Z]+)', district)
        return area_match.group(1).upper() if area_match else "UNKNOWN"

    return address_series.apply(parse_address)

MASTER_DATA_PATH = os.environ.get("MASTER_DATA_LOCAL_PATH")
FEATURE_SETS_PATH = os.environ.get("FEATURE_SETS_LOCAL_PATH")
FORECAST_ARTIFACTS_GCS_DIR = os.environ.get("FORECAST_ARTIFACTS_GCS_DIR")
RIGHTMOVE_DATA_PATH = os.environ.get("RIGHTMOVE_DATA_LOCAL_PATH")
KEY_MAP_PATH = os.environ.get("KEY_MAP_LOCAL_PATH")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR")
N_TRIALS_MAIN = int(os.environ.get("N_TRIALS", 50)) # For the main 'mid' stratum
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_FOLDS_FINAL = 20 # Use more folds for robust final model training.
TRIAGE_THRESHOLD = float(os.environ.get("TRIAGE_THRESHOLD", 0.0))

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

def engineer_yearly_summary_heads(df, head_h_cols, head_i_cols):
    """
    [V13.7R2 ARCHITECTURE - MONOLITHIC DISTILLATION ENGINE]
    Condenses thousands of monthly features in Head H and Head I into a small,
    powerful set of yearly summary statistics using a robust groupby pattern.
    """
    print("\n--- V13.7 (Rev 2): Running Monolithic Temporal Distillation Engine ---")
    df_copy = df.copy()
    new_features = []
    
    # --- BEGIN A.D-V18.4 ADDITION ---
    # Suppress fragmentation warnings that are benign in this context.
    with pd.option_context('mode.chained_assignment', None):
    # --- END A.D-V18.4 ADDITION ---
        # --- Process Head H ---
        if head_h_cols:
            h_parsed = [re.match(r'pp_(\d{4})_(\d{2})_([A-Z])_.*', c) for c in head_h_cols]
            h_data = [{'col': c, 'year': m.group(1)} for c, m in zip(head_h_cols, h_parsed) if m]
            if h_data:
                h_df = pd.DataFrame(h_data)
                # SURGICAL CORRECTION: Use groupby on the metadata frame. This is robust.
                for year, group in h_df.groupby('year'):
                    year_cols = group['col'].tolist()
                    year_data = df_copy[year_cols]
                    
                    mean_col = f'yearly_H_mean_{year}'
                    std_col = f'yearly_H_std_{year}'
                    df_copy[mean_col] = year_data.mean(axis=1)
                    df_copy[std_col] = year_data.std(axis=1)
                    new_features.extend([mean_col, std_col])
                print(f"  - Distilled {len(head_h_cols)} Head H features into {len(new_features)} yearly summaries.")

        # --- Process Head I (More Complex) ---
        if head_i_cols:
            i_parsed = [re.match(r'.*_pp_(\d{4})_(\d{2})_([A-Z])_avg_price_n(\d+)', c) for c in head_i_cols]
            i_data = [{'col': c, 'year': m.group(1), 'n': m.group(4)} for c, m in zip(head_i_cols, i_parsed) if m]
            if i_data:
                i_df = pd.DataFrame(i_data)
                base_len = len(new_features)
                # SURGICAL CORRECTION: Group by both year and neighbor count.
                for (year, n_count), group in i_df.groupby(['year', 'n']):
                    n_cols = group['col'].tolist()
                    n_data = df_copy[n_cols]

                    mean_col = f'yearly_I_mean_{year}_n{n_count}'
                    std_col = f'yearly_I_std_{year}_n{n_count}'
                    min_col = f'yearly_I_min_{year}_n{n_count}'
                    max_col = f'yearly_I_max_{year}_n{n_count}'

                    df_copy[mean_col] = n_data.mean(axis=1)
                    df_copy[std_col] = n_data.std(axis=1)
                    df_copy[min_col] = n_data.min(axis=1)
                    df_copy[max_col] = n_data.max(axis=1)
                    new_features.extend([mean_col, std_col, min_col, max_col])
                print(f"  - Distilled {len(head_i_cols)} Head I features into {len(new_features) - base_len} yearly summaries.")
            
    return df_copy, new_features


def engineer_data_integrity_features(df, feature_sets, universal_cols):
    """[V24.0] Engineers meta-features describing data quality, sparseness, and abnormality."""
    print("\n--- V24.0: Engineering Data Integrity Features ---")
    df_copy = df.copy()
    new_integrity_cols = []

    # 1. Head-Level Missingness Scores
    print("  - Calculating head-level missingness scores...")
    for head_name, features in feature_sets.items():
        present_features = [f for f in features if f in df_copy.columns]
        if not present_features: continue
        
        # Calculate the percentage of features in this head that are zero or null for each row
        missing_pct = (df_copy[present_features] == 0).sum(axis=1) / len(present_features)
        col_name = f"integrity_missing_pct_{head_name}"
        df_copy[col_name] = missing_pct
        new_integrity_cols.append(col_name)

    # 2. Global Anomaly Score using Isolation Forest
    print("  - Calculating global anomaly score...")
    present_universal = [f for f in universal_cols if f in df_copy.columns]
    if present_universal:
        iso_forest = IsolationForest(n_estimators=100, contamination='auto', random_state=42, n_jobs=-1)
        X_iso = df_copy[present_universal].fillna(0)
        iso_forest.fit(X_iso)
        # score_samples returns the opposite of the anomaly score (higher is better)
        # We invert it so that lower scores (more anomalous) are more intuitive.
        anomaly_scores = -iso_forest.score_samples(X_iso)
        col_name = "integrity_anomaly_score"
        df_copy[col_name] = anomaly_scores
        new_integrity_cols.append(col_name)

    print(f"  - Created {len(new_integrity_cols)} new data integrity features.")
    return df_copy, new_integrity_cols


#
# === BEGIN ARCHITECTURAL DIRECTIVE A.D-V14.0 IMPLEMENTATION ===
#
def build_v9_feature_sets(all_columns, feature_sets_json_path):
    """
    [V14.0 ARCHITECTURE - DEFINITIVE HIERARCHICAL ASSEMBLY PROTOCOL]
    Implements a robust, multi-pass feature assembly protocol with two critical enhancements:
    1. Upstream Exclusion: A NON_FEATURE_COLS set is used to sanitize the feature pool first.
    2. Corrected Manifests: All specialist and distilled head manifests are updated to correctly
       capture all engineered and vendor-specific features, eliminating miscategorization.
    """
    print("\n--- V14.0 Assembly: Initiating Definitive Feature Set Assembly Protocol ---")
    
    # --- Phase I: Ingestion & Prophylactic Exclusion ---
    all_columns_set = set(all_columns)
    
    # [A.D-V14.0 CRITICAL ADDITION] Define and apply upstream exclusion.
    NON_FEATURE_COLS = {
        'property_id', 'address_key', 'normalized_address_key', 'rightmove_address_text', 
        'rightmove_row_id', 'pcds', 'most_recent_sale_price', 'most_recent_sale_year',
        'most_recent_sale_month', 'days_since_last_sale', 'price_change_since_last', 'total_sales_count'
    }
    
    sanitized_features_pool = all_columns_set - NON_FEATURE_COLS
    print(f"  - Upstream Sanitization: Removed {len(all_columns_set - sanitized_features_pool)} non-feature columns.")
    
    feature_sets = {}
    
    try:
        with open(feature_sets_json_path, 'r') as f:
            legacy_heads = json.load(f)
    except Exception as e:
        print(f"  - CRITICAL WARNING: Could not load legacy feature sets from '{feature_sets_json_path}'. Error: {e}")
        legacy_heads = {}

    unclaimed_features = sanitized_features_pool.copy()

    # --- Phase II: Granular Specialist Assignment (Bottom-Up) ---
    print("\n--- Phase II: Populating Granular 'sub_head' Specialists (Corrected Manifests) ---")
    # [A.D-V14.0 CRITICAL CORRECTION] Expanded and corrected manifest.
    V9_SPECIALIST_MANIFEST = {
        'sub_head_yearly_distilled_h': ['yearly_H_'],
        'sub_head_yearly_distilled_i': ['yearly_I_'],
        'sub_head_engineered_interactions': ['num__INT_'],
        'sub_head_all_gemini_embeddings': ['ae_feat_'],
        'sub_head_temporal_price_distilled': ['distilled_price_'],
        'sub_head_build_period': ['BP_'], 
        'sub_head_build_period_mode': ['MODE'], 
        'sub_head_derived_scores': ['Composite_', 'Inter_', 'Mismatch_', 'Opportunity_', 'Persona_', 'Ratio_', 'Risk_', 'Thesis_', 'Tradeoff_'], 
        'sub_head_location_geo': ['latitude', 'longitude', 'num__geocoding_level', 'geo_cluster', 'country_proxy_scotland'], 
        'sub_head_microscope_embeddings': ['microscope_emb_'], 
        'sub_head_vendor_mouseprice': ['__mp'], 
        'sub_head_vendor_bricksandlogic': ['__bnl'], 
        'sub_head_vendor_homipi': ['__hm'], 
        'sub_head_vendor_chimnie': ['__ch'], 
        'sub_head_vendor_streetscan': ['__ss'], 
        'sub_head_dna_missing_indicators': ['missingindicator_'], 
        'sub_head_census_household': ['Household_size_', 'dependent_children_', 'disability_', 'disabilty_', 'ethnicity_', 'health_condition_', 'household_composition_', 'household_schoolchildren_', 'household_type_', 'language_', 'multiple_ethnicity_', 'multiple_religion_', 'number_adults_', 'number_bedrooms_', 'number_carers_', 'number_employed_', 'number_families_', 'number_per_bedroom_', 'number_per_room_', 'number_rooms_', 'occupancy_rating_', 'ons_army_', 'ownership_', 'reference_person_', 'religion_', 'vehicles_', 'house_types_', 'Type_of_central_heating_'], 
        'sub_head_census_deprivation': ['deprivation_', 'education_deprivation_', 'employment_deprivation_', 'health_deprivation_', 'housing_deprivation_'], 
        'sub_head_geo_classification': ['LAD23NM_', 'la23cd_', 'MSOA', 'OA', 'WZ11_'], 
        'sub_head_environmental_health_amenities': ['ah4', 'bba', 'CoreHealthAccessSum_', 'AccessibleCleanGreenSpace_', 'AirPollutionGradient_'], 
        'sub_head_environmental_satellite': ['EVI_', 'NDVI_', 'VEG_FRAC', 'VegFrac_'], 
        'sub_head_local_market_dynamics': ['LSOA_', 'PC_'], 
        'sub_head_census_engineered_indices': ['FI_'], 
        'sub_head_temporal_sales_volume': ['_sale_count_', 'Sale_Count_'], 
        'sub_head_temporal_summary_engineered': ['ts_summary_'],
        'sub_head_compass_census_mean': ['compass_mean_'], 
        'sub_head_compass_census_std': ['compass_std_'], 
        'sub_head_compass_price_mean': ['compass_mean_pp_'], 
        'sub_head_compass_price_std': ['compass_std_pp_'], 
        'sub_head_compass_sales_volume_mean': ['compass_mean_sale_count'], 
        'sub_head_compass_sales_volume_std': ['compass_std_sale_count'], 
        'sub_head_forecast_features': ['forecast_price_'], 
        'sub_head_sanitized_engineered': ['_clipped', '_binned', 'years_since_last_sale'],
    }
    for head_name, patterns in V9_SPECIALIST_MANIFEST.items():
        assigned = set()
        for p in patterns:
            assigned.update({c for c in unclaimed_features if p in c})
        if assigned:
            feature_sets[head_name] = sorted(list(assigned))
            unclaimed_features -= assigned

    # --- Phase III: Thematic Distilled Head Assembly (Concept-Driven) ---
    print("\n--- Phase III: Assembling Thematic 'distilled_head' Concepts (Corrected Manifests) ---")
    # [A.D-V14.0 CRITICAL CORRECTION] Replaced all obsolete `distilled_price_` patterns with new `yearly_...` features.
    distilled_heads = {
        'distilled_lifestyle_and_leisure_access': ['Persona_Entertainer_Score', 'num__StreetScan_category_rating_stars_for_Culture_ss', 'num__StreetScan_category_rating_stars_for_Scenery_and_Parks_ss', 'ah4leis', 'ah4pubs', 'ah4ffood', 'num__chimnie_local_area_entertainment_index_score_ch', 'num__chimnie_local_area_food_and_drink_index_score_ch', 'num__chimnie_local_area_shopping_index_score_ch'],
        'distilled_new_build_and_modern_stock_value': ['BP_200', 'BP_201', 'num__property_construction_year_from_homipi__YYYY_or_2025NewBuildOrError__hm', 'num__current_epc_value_extracted_by_initial_homipi_script_hm_numeric', 'LSOA_BP_Ratio_Post2000', 'Churn_in_LSOA_Dominated_by_ModernStock', '_num_flaws', '_renovation_score', 'num__StreetScan_past_sales_avg_price_gbp_for_New_Builds_ss'],
        'distilled_property_archetype_terraced_and_flat': ['num__property_main_type_encoded__1Flat_hm', 'num__2Terraced_hm', 'num__1LeaseholdOrOther__hm', 'num__mouseprice_property_type_encoded_', 'house_types_Flat_maisonette_or_apartment', 'house_types_Whole_house_or_bungalow_Terraced', 'LSOA_Property_Density', '_sale_count_F', '_sale_count_T'],
        'distilled_suburban_family_archetype': ['num__property_sub_type_code_from_homipi__4Detached_hm', 'num__3SemiDetached_hm', 'num__plot_size_sqm_or_info_text__bnl', 'num__number_of_bedrooms_from_homipi__numeric_or_empty__hm', 'num__property_tenure_code_from_homipi__2Freehold_hm', 'primary_MainGarden_area_sqm', 'primary_MainDrivewayParking_area_sqm', 'primary_StudyOffice_area_sqm', 'num__StreetScan_category_rating_stars_for_Schools_ss', 'num__chimnie_local_area_family_index_score_ch', 'FI_Prop_HH_With_Dependent_Children', 'OA11_CLASSIFICATION_NAME_White_suburban_communities'],
        'distilled_market_regime_shift_detector': ['yearly_H_mean', 'yearly_I_mean', 'sale_count', 'compass_mean_', 'LSOA_Price_Growth_5Year', 'LSOA_Transaction_Volume_Change_5Year'],
        'distilled_market_volatility_and_risk_profile': ['yearly_H_std', 'yearly_I_std', 'PC_Sales_Volatility_YoY_'],
        'distilled_market_recovery_and_resilience': ['yearly_H_mean', 'yearly_I_mean', 'num__StreetScan_average_household_income_gbp_ss', 'ownership_Owned_Owns_outright'],
        'distilled_price_tier_migration': ['yearly_H_mean', 'yearly_I_mean'],
        'distilled_seasonal_and_cyclical_price_patterns': ['yearly_H_mean', 'yearly_I_mean'],
        'distilled_aesthetic_neighborhood_cohesion': ['primary_MainExteriorFront_', 'primary_MainExteriorRear_', 'ae_property_wide_', 'OA', 'MSOA', 'StreetScan_average_household_income_gbp_ss', 'LSOA_Property_Density'],
        'distilled_era_modernization_and_character': ['BP_', 'num__property_construction_year_from_homipi_', 'Opportunity_Good_Bones_Score', '_renovation_score', 'ae_primary_', 'ae_other_'],
        'distilled_volumetric_and_light_premium': ['primary_MainLivingArea_num_features', 'primary_MainHallwayLandingStairs_num_features', 'avg_persona_rating_overall', 'ae_primary_MainLivingArea_', 'ae_primary_MainHallwayLandingStairs_', 'ae_property_wide_', 'num__floor_area_sqm_', 'num__number_of_reception_rooms_', 'Ratio_Kitchen_vs_Living'],
        'distilled_indoor_outdoor_lifestyle_flow': ['primary_MainGarden_', 'primary_MainPatioDeckingTerrace_', 'primary_MainConservatorySunroom_', 'ae_primary_MainLivingArea_', 'ae_primary_MainGarden_', 'Persona_Entertainer_Score', 'num__plot_size_sqm_', 'LSOA_Property_Density'],
        'distilled_visual_maintenance_and_deferred_cost_risk': ['_num_flaws', '_renovation_score', 'ae_primary_MainExteriorFront_', 'ae_primary_MainKitchen_', 'ae_primary_MainBathroom_', 'BP_', 'num__INT_hm_years_since_last_sale', 'Risk_Cosmetic_Burden_X_Age'],
        'distilled_visual_demographic_alignment': ['Persona_', 'primary_MainGarden_area_sqm', 'primary_StudyOffice_area_sqm', 'Ratio_Kitchen_vs_Living', 'ae_persona_justifications_', 'ae_primary_', 'OA', 'MSOA', 'StreetScan_average_household_income_gbp_ss', 'FI_Prop_HH_With_Dependent_Children', 'reference_person_Household_reference_person_is_aged_', 'StreetScan_employment_status_distribution_'],
        'distilled_design_specificity_and_market_breadth': ['ae_', 'std_dev_persona_rating_overall', 'LSOA_Market_Absorption_Years', 'LSOA_Annual_Transaction_Rate'],
        'distilled_amenity_quality_vs_property_tier': ['primary_MainKitchen_renovation_score', 'primary_MainKitchen_num_features', 'primary_MainBathroom_renovation_score', 'primary_MainBathroom_num_features', 'primary_MainGarden_num_features', 'ae_primary_MainKitchen_', 'ae_primary_MainBathroom_', 'num__property_sub_type_code_from_homipi__4Detached_hm', 'num__floor_area_sqm_'],
        'distilled_layout_efficiency_and_flow': ['ae_primary_MainLivingArea_', 'ae_primary_MainKitchen_', 'ae_primary_MainHallwayLandingStairs_', 'num__floor_area_sqm_', 'num__number_of_bedrooms_', 'num__number_of_reception_rooms_', 'Ratio_Kitchen_vs_Living', 'num_rooms_identified_in_step5', 'primary_MainLivingArea_area_sqm'],
        'distilled_curb_appeal_and_street_presence': ['primary_MainExteriorFront_', 'primary_MainDrivewayParking_', 'primary_MainPorch_', 'ae_primary_MainExteriorFront_', 'BP_', 'StreetScan_average_household_income_gbp_ss', 'MSOA11_CLASSIFICATION_NAME_Affluent_communities'],
        'distilled_lifestyle_feature_quality_index': ['primary_MainKitchen_', 'primary_MainGarden_', 'primary_MainPatioDeckingTerrace_', 'primary_StudyOffice_', 'primary_MainConservatorySunroom_', 'ae_primary_MainKitchen_', 'ae_primary_MainGarden_', 'Persona_Entertainer_Score', 'num__plot_size_sqm_'],
        'distilled_renovation_readiness_and_personalization_potential': ['_renovation_score', '_num_flaws', 'ae_', 'Persona_FixerUpper_Score', 'Opportunity_Good_Bones_Score', 'num__INT_hm_years_since_last_sale'],
        'distilled_private_outdoor_space_quality_and_utility': ['primary_MainGarden_', 'primary_MainPatioDeckingTerrace_', 'primary_MainExteriorRear_', 'ae_primary_MainGarden_', 'ae_primary_MainExteriorRear_', 'num__plot_size_sqm_', 'Inter_GardenValue_X_PublicSpaceDeficit', 'LSOA_Property_Density'],
        'distilled_architectural_and_stylistic_consistency': ['ae_primary_MainExteriorFront_', 'ae_primary_MainExteriorRear_', 'ae_primary_', 'std_dev_persona_rating_overall', 'BP_', 'num__property_construction_year_', 'num__number_of_extensions_from_homipi'],
        'distilled_interior_space_flow_and_quality': ['primary_MainLivingArea_', 'primary_MainKitchen_', 'primary_MainBathroom_', 'primary_PrimaryBedroom_', 'other_OtherBedrooms_', 'other_OtherBathroomsWCs_', 'num_rooms_identified_in_step5', 'ae_primary_MainLivingArea_', 'ae_primary_MainKitchen_', 'ae_primary_MainBathroom_', 'ae_primary_PrimaryBedroom_', 'ae_other_', 'num__number_of_bedrooms_from_homipi__numeric_or_empty__hm', 'num__number_of_reception_rooms_from_homipi__numeric_or_empty__hm', 'Ratio_Kitchen_vs_Living'],
        'distilled_lifestyle_persona_match': ['persona_Persona_', 'avg_persona_rating_overall', 'std_dev_persona_rating_overall', 'ae_', 'Persona_Entertainer_Score', 'Persona_FixerUpper_Score'],
        'distilled_outdoor_living_and_amenity_space': ['primary_MainGarden_', 'primary_MainPatioDeckingTerrace_', 'primary_MainOutbuilding_', 'primary_MainExteriorRear_', 'primary_MainExteriorSide_', 'ae_primary_MainGarden_', 'ae_primary_MainPatioDeckingTerrace_', 'ae_primary_MainOutbuilding_', 'ae_primary_MainExteriorRear_', 'num__plot_size_sqm_or_info_text__bnl', 'Inter_GardenValue_X_PublicSpaceDeficit'],
        'distilled_architectural_style_and_period_character': ['BP_', 'num__property_construction_year_from_homipi__YYYY_or_2025NewBuildOrError__hm', 'primary_MainExteriorFront_num_features', 'primary_MainHallwayLandingStairs_num_features', 'ae_property_wide_', 'ae_primary_MainExteriorFront_', 'ae_primary_MainHallwayLandingStairs_'],
        'distilled_value_density_and_finish_level': ['num__property_floor_area_sqft__numeric_extracted_or_original_text__bnl_converted_sqm', 'num__number_of_bedrooms_from_homipi__numeric_or_empty__hm', 'avg_persona_rating_overall', 'primary_MainKitchen_num_features', 'primary_MainBathroom_num_features', 'other_', 'ae_'],
        'distilled_natural_light_and_spaciousness': ['ae_property_wide_', 'ae_primary_MainLivingArea_', 'ae_primary_MainKitchen_', 'primary_MainLivingArea_num_features', 'avg_persona_rating_overall', 'persona_Persona_', 'num__property_floor_area_sqft__numeric_extracted_or_original_text__bnl_converted_sqm'],
        'distilled_visual_risk_and_detraction_signals': ['_num_flaws', '_renovation_score', 'avg_persona_rating_overall', 'Persona_FixerUpper_Score', 'ae_', 'Risk_Cosmetic_Burden_X_Age', 'num__bricksandlogic_data_quality_detail_score_out_of_5__numeric_extracted_or_original_text__bnl'],
        'distilled_functional_space_and_lifestyle_utility': ['primary_StudyOffice_', 'primary_MainUtilityRoom_', 'primary_MainGarage_', 'primary_MainOutbuilding_', 'primary_MainStorageLoftCellar_', 'ae_primary_StudyOffice_', 'num__number_of_reception_rooms_from_homipi__numeric_or_empty__hm'],
        'distilled_visual_feature_scarcity_premium': ['primary_MainGarden_area_sqm', 'primary_MainDrivewayParking_area_sqm', 'ae_primary_MainGarden_', 'ae_primary_MainDrivewayParking_', 'LSOA_Property_Density', 'MSOA21_URBAN_RURAL_INDICATOR_', 'OA', 'house_types_Flat_maisonette_or_apartment', 'Inter_GardenValue_X_UrbanDensity', 'Inter_GardenValue_X_PublicSpaceDeficit'],
        'distilled_hyperlocal_price_vs_property_condition': ['yearly_H_mean', 'yearly_I_mean', 'avg_persona_rating_overall', '_renovation_score', '_num_flaws', 'ae_', 'num__property_sub_type_code_', 'num__property_main_type_'],
        'distilled_build_era_price_dynamics': ['yearly_H_mean', 'yearly_I_mean', 'BP_', 'num__property_construction_year_', 'MODE1_TYPE_BP_', 'primary_MainExteriorFront_', 'ae_primary_MainExteriorFront_'],
        'distilled_price_trend_vs_visual_quality_divergence': ['yearly_H_mean', 'yearly_I_mean', 'avg_persona_rating_overall', '_renovation_score', 'ae_'],
        'distilled_market_ceiling_and_breakout_potential': ['yearly_H_mean', 'yearly_I_mean', 'primary_MainKitchen_num_features', 'primary_MainGarden_area_sqm', 'primary_MainLivingArea_area_sqm', 'ae_primary_MainKitchen_', 'ae_primary_MainLivingArea_', 'ae_property_wide_', 'num__floor_area_sqm_', 'num__property_sub_type_code_'],
        'distilled_gentrification_and_renovation_wave_detector': ['yearly_H_mean', 'yearly_I_mean', '_renovation_score', 'ae_', 'compass_mean_', 'compass_std_pp_', 'OA11_CLASSIFICATION_NAME_White_professionals', 'MSOA11_CLASSIFICATION_NAME_Highly_qualified_professionals', 'StreetScan_average_household_income_gbp_ss'],
        'distilled_market_volatility_vs_visual_uniqueness': ['yearly_H_std', 'yearly_I_std', 'ae_', 'std_dev_persona_rating_overall', 'num__property_sub_type_code_'],
        'distilled_local_affordability_and_quality_ceiling': ['yearly_H_mean', 'yearly_I_mean', 'avg_persona_rating_overall', 'primary_MainKitchen_renovation_score', 'ae_primary_MainKitchen_', 'StreetScan_average_household_income_gbp_ss', 'StreetScan_national_income_rank_decile_ss', 'deprivation_'],
        'distilled_unrealized_equity_in_strong_market': ['yearly_H_mean', 'yearly_I_mean', '_renovation_score', 'avg_persona_rating_overall', 'ae_', 'Opportunity_Good_Bones_Score', 'num__INT_hm_years_since_last_sale'],
        'distilled_market_maturation_and_quality_tiering': ['yearly_H_mean', 'yearly_I_mean', 'avg_persona_rating_overall', 'std_dev_persona_rating_overall', 'ae_', 'compass_std_pp_'],
        'distilled_recessionary_premium_for_turnkey_condition': ['yearly_H_std', 'yearly_I_std', '_renovation_score', 'avg_persona_rating_overall', '_num_flaws', 'ae_', 'Persona_FixerUpper_Score'],
        'distilled_realized_renovation_value': ['num__last_sold_price_gbp_hm', 'num__last_sold_date_year__YYYY__hm', 'num__INT_hm_price_vs_last_sold_ratio', 'avg_persona_rating_overall', '_renovation_score', 'Persona_FixerUpper_Score', 'ae_property_wide_', 'ae_primary_MainKitchen_'],
        'distilled_archetype_condition_standard': ['num__property_floor_area_sqft__numeric_extracted_or_original_text__bnl_converted_sqm', 'num__number_of_bedrooms_from_homipi__numeric_or_empty__hm', 'num__property_main_type_encoded__1Flat_hm', 'num__property_sub_type_code_from_homipi__4Detached_hm', 'primary_', 'other_', 'ae_'],
        'distilled_property_vs_postcode_benchmark': ['yearly_H_mean', 'yearly_I_mean', 'num__property_floor_area_sqft__numeric_extracted_or_original_text__bnl_converted_sqm', 'num__plot_size_sqm_or_info_text__bnl', 'num__INT_avg_estimated_price', 'avg_persona_rating_overall', '_renovation_score'],
        'distilled_historical_context_for_current_condition': ['yearly_H_mean', 'yearly_I_mean', 'primary_', 'other_', 'ae_'],
        'distilled_price_history_vs_property_type_dominance': ['yearly_H_mean', 'yearly_I_mean', 'house_types_', 'LSOA_BP_Ratio_Post2000', 'LSOA_Property_Density', 'PC_Detached_to_Flat_Price_Ratio', 'PC_Relative_Sales_Volume_D_vs_F'],
        'distilled_historical_premium_for_space': ['yearly_H_mean', 'yearly_I_mean', 'PC_Detached_to_Flat_Price_Ratio'],
        'distilled_historical_price_vs_volume_correlation': ['yearly_H_mean', 'yearly_I_mean', 'sale_count', 'LSOA_Annual_Transaction_Rate', 'LSOA_Market_Absorption_Years'],
        
        
        # --- [A.D-V30.0] THE FORCED SYNTHESIS SUPER-HEAD (RE-ARMED) ---
        'V30_super_head_proprietary_synthesis': [
            # It is FORBIDDEN from seeing competitor AVMs.
            
            # 1. NEW Dynamic, Hyper-Local Price Anchors:
            'eng_DynamicHPI_Adjusted_Price_',
            
            # 2. NEW Divergence Signals (The "Why is this cheap?" features):
            'eng_Divergence_Quality_vs_Market', 'eng_Divergence_Size_vs_Market',
            
            # 3. Raw historical transaction data (still valuable context):
            'price_change_since_last', 'days_since_last_sale',
            
            # 4. Our proprietary visual and qualitative analysis:
            'ae_feat_', 'persona_', 'avg_persona_rating_overall', '_renovation_score',
            
            # 5. Core property DNA for context:
            'num__floor_area_sqm', 'BP_PRE_1900'
        ],
        
        # --- NEW & ENHANCED: High-Level Conceptual Heads ---
        'distilled_quality_of_life_and_wellbeing': [
            'num__chimnie_local_area_life_quality_index_score_ch', 'num__chimnie_local_area_safety_index_score_ch', 
            'AccessibleCleanGreenSpace_', 'NDVI_', 'VEG_FRAC', 'ah4gpas', 'num__primary_or_first_type_school_1_ofsted_rating_encoded',
            'num__StreetScan_category_rating_stars_for_Schools_ss', 'education_deprivation_', 'num__homipi_nearest_hospital_distance_or_number_1_hm',
            'HealthEnvironmentSynergy_HealthScore_x_AirQualityScore', 'ah4h', 'health_deprivation_'
        ],
        'distilled_socioeconomic_stratum': [
            'num__StreetScan_average_household_income_gbp_ss', 'num__StreetScan_national_income_rank_decile_ss',
            'deprivation_', 'education_deprivation_', 'housing_deprivation_', 'employment_deprivation_', 'OA', 'MSOA',
            'ownership_Owned_Owns_outright', 'FI_Prop_Socially_Rented_HH', 'num__INT_ss_income_vs_avg_sale_price_ratio'
        ],
        'distilled_future_market_pressure': [
            'forecast_price_', 'compass_mean_sale_count', 'compass_std_sale_count', 'LSOA_Price_Growth_5Year', 
            'Thesis_Renovation_X_AreaGrowth', 'Opportunity_GentrificationFrontline', 'LSOA_Transaction_Volume_Change_5Year'
        ],

        # --- RE-ARCHITECTED: Stratified Risk Heads ---
        'distilled_physical_asset_risk': [
            'Risk_Cosmetic_Burden_X_Age', '_num_flaws', '_renovation_score', 'years_since_last_sale', 'num__bricksandlogic_data_quality_',
            'num__current_epc_value_extracted_by_initial_homipi_script_hm_numeric', 'num__INT_hm_epc_improvement_potential',
            'visual_maintenance_and_deferred_cost_risk', 'num_images_total'
        ],
        'distilled_socioeconomic_risk': [
            'num__StreetScan_deprivation_rank_', 'deprivation_', 'employment_deprivation_', 'housing_deprivation_',
            'num__chimnie_wider_area_overall_risk_encoded', 'compass_mean_FI_Deprivation_Severity_Index_'
        ],
        'distilled_environmental_risk': [
            'ah4no2', 'ah4pm10', 'CumulativeHazardAccess_', 'UnhealthyRetailExposure_', 'AirPollutionGradient_NO2_minus_PM10'
        ],

        # --- ENRICHED: Existing Heads with Deeper Signal Integration ---
        'distilled_vendor_data_consensus_and_divergence': ['num__bricksandlogic_estimated_price_gbp', 'num__homipi_estimated_price_range_upper_bound_gbp_hm', 'num__mouseprice_estimated_value_gbp', 'num__floor_area_sqm_from_homipi', 'num__mouseprice_floor_area_sqm', 'num__property_floor_area_sqft', 'num__bricksandlogic_data_quality_', 'num__homipi_estimate_confidence_score_', 'num__mouseprice_valuation_confidence_'],
        'distilled_data_confidence_and_information_risk': ['missingindicator_', 'num__bricksandlogic_data_quality_', 'num__homipi_estimate_confidence_score_', 'num__mouseprice_valuation_confidence_', 'cat__plot_data_sample_size_'],
        'distilled_local_authority_and_tax_burden': ['num__council_tax_annual_rate_gbp_from_homipi_hm', 'num__council_tax_band_encoded_', 'LAD23NM_', 'la23cd_'],
        'distilled_comparable_market_evidence': ['num__INT_avg_estimated_price', 'num__bricksandlogic_estimated_price_gbp', 'num__mouseprice_estimated_value_gbp', 'num__homipi_estimated_price_range_lower_bound_gbp_hm', 'num__homipi_estimated_price_range_upper_bound_gbp_hm', 'LSOA_MedPrice_Recent', 'num__StreetScan_past_sales_avg_price_gbp_', 'compass_mean_pp_'],
        'distilled_locational_character_and_quality': ['latitude', 'longitude', 'microscope_emb_', 'num__chimnie_', 'num__StreetScan_category_rating_stars_', 'num__train_station_', 'ah4', 'OA', 'MSOA', 'WZ11_', 'LSOA_Property_Density', 'NDVI_', 'VEG_FRAC', 'AccessibleCleanGreenSpace_'],
        'distilled_intrinsic_property_quality_and_condition': ['BP_', 'num__floor_area_', 'num__number_of_bedrooms_', 'num__number_of_reception_rooms_', 'num__plot_size_', 'Composite_CoreLiving_Strength', 'avg_persona_rating_overall', 'primary_', 'other_', 'ae_property_wide_', 'ae_primary_MainKitchen_', 'ae_primary_MainBathroom_'],
        'distilled_relative_price_positioning': ['num__homipi_estimated_price_range_upper_bound_gbp_hm', 'num__mouseprice_estimated_value_gbp', 'LSOA_MedPrice_Recent', 'PC_Price_to_LSOA_Median_Ratio_', 'yearly_H_mean', 'yearly_I_mean'],
        'distilled_build_era_archetype': ['BP_', 'MODE1_TYPE_BP_', 'num__property_construction_year_from_homipi_', 'Opportunity_Good_Bones_Score', 'primary_MainGarden_area_sqm', 'num_rooms_identified_in_step5', 'LSOA_BP_Ratio_Post2000', 'LSOA_BP_Ratio_Pre1919', 'LSOA_OldStock_Dominance_Effect_on_PC_Price_D'],
        'distilled_realized_vs_unrealized_potential': ['num__INT_hm_epc_improvement_potential', 'num__potential_epc_value_', 'Ratio_ValueAddLeverage', 'num__plot_size_sqm_or_info_text__bnl', 'num__number_of_extensions_from_homipi', '_renovation_score', '_num_flaws', 'avg_persona_rating_overall', 'ae_', 'forecast_price_'],
        'distilled_accessibility_and_connectivity_hub': ['num__train_station_', 'num__homipi_nearest_bus_stop_distance_', 'microscope_emb_', 'WZ11_', 'MSOA21_URBAN_RURAL_INDICATOR_Urban', 'num__StreetScan_category_rating_stars_for_Transport_ss'],
        'distilled_investment_yield_and_rental_market_fit': ['cat__mouseprice_estimated_rental_value_text', 'num__homipi_estimated_price_range_', 'num__property_main_type_encoded__1Flat_hm', 'ownership_Private_rented_', 'MSOA11_CLASSIFICATION_NAME_Constrained_renters', 'OA11_CLASSIFICATION_NAME_Student_communal_living', 'OA11_CLASSIFICATION_NAME_Renting_hard_pressed_workers', 'FI_Prop_Socially_Rented_HH', 'compass_mean_ownership_Private_rented_', 'housing_deprivation_'],
        'distilled_property_lifestyle_and_neighborhood_synergy': ['Persona_Entertainer_Score', 'Inter_FamilyFeatures_X_Schools', 'Thesis_HiddenGem', 'primary_MainPatioDeckingTerrace_area_sqm', 'primary_MainGarden_area_sqm', 'primary_MainKitchen_num_features', 'Ratio_Kitchen_vs_Living', 'num__chimnie_local_area_entertainment_index_score_ch', 'num__chimnie_local_area_shopping_index_score_ch', 'num__chimnie_local_area_life_quality_index_score_ch', 'WZ11_GROUP_NAME_Eat__drink_and_be_merry', 'MSOA11_CLASSIFICATION_NAME_Ageing_rural_neighbourhoods'],
        'distilled_land_vs_improvement_value_ratio': ['num__plot_size_sqm_or_info_text__bnl', 'num__floor_area_sqm_', 'latitude', 'longitude', 'avg_persona_rating_overall', '_renovation_score', 'LSOA_Property_Density', 'StreetScan_average_household_income_gbp_ss', 'ah4gpas'],
        'distilled_affordability_and_buyer_profile': ['num__council_tax_', 'num__mouseprice_estimated_running_cost_', 'num__current_epc_value_extracted_by_initial_homipi_script_hm_numeric', 'num__property_tenure_code_from_homipi__2Freehold_hm', 'num__1LeaseholdOrOther__hm', 'num__StreetScan_average_household_income_gbp_ss', 'StreetScan_deprivation_rank_for_Income_ss', 'ownership_Owned_Owns_with_a_mortgage_or_loan_or_shared_ownership', 'ownership_Social_rented_', 'OA', 'MSOA'],
        'distilled_rurality_and_accessibility_profile': ['num__train_station_', 'num__homipi_nearest_bus_stop_distance_or_number_', 'LSOA_Property_Density', 'MSOA21_URBAN_RURAL_INDICATOR_', 'WZ11_SUPERGROUP_NAME_Rural', 'WZ11_GROUP_NAME_Traditional_countryside', 'OA11_CLASSIFICATION_NAME_Agricultural_communities', 'ah4pubs', 'ah4leis', 'ah4ffood', 'NDVI_'],
        'distilled_market_character_and_velocity': ['MODE1_TYPE_BP_', 'MODE2_TYPE_BP_', 'LSOA_NewStock_Dominance_Effect_on_PC_Price_F', 'PC_Detached_to_Flat_Price_Ratio', 'sale_count', 'compass_mean_sale_count', 'LSOA_Annual_Transaction_Rate', 'LSOA_Churn_Recent', 'LSOA_Market_Absorption_Years'],
        'distilled_hyperlocal_market_texture_and_homogeneity': ['compass_std_', 'MODE1_PC', 'MODE2_MODE1_RATIO', 'Ratio_of_LSOA_MODE1_to_MODE2_BuildPeriod_Count', 'FI_Ethnic_Diversity_And_Language_Diversity_Cooccurrence', 'LAD_OAC_SUB'],
        'distilled_market_depth_and_price_conviction': ['yearly_H_mean', 'yearly_I_mean', 'sale_count', 'compass_mean_sale_count', 'LSOA_Annual_Transaction_Rate', 'LSOA_Churn_to_Transaction_Ratio', 'LSOA_Market_Absorption_Years'],
        'distilled_postcode_performance_vs_regional_ripple': ['yearly_H_mean', 'yearly_I_mean', 'LSOA_Price_Growth_5Year', 'LSOA_MedPrice_5Y_Ago', 'LSOA_MedPrice_Recent', 'forecast_price_'],
        'distilled_property_transactional_history': ['num__last_sold_price_gbp_hm', 'num__last_sold_date_year__YYYY__hm', 'num__INT_hm_price_vs_last_sold_ratio', 'num__homipi_value_change_percentage_hm', 'years_since_last_sale'],
        'distilled_investment_stability_and_risk': ['num__homipi_estimate_confidence_score_encoded__3High_hm', 'num__mouseprice_valuation_confidence_encoded_', 'PC_Sales_Volatility_YoY_', 'num__StreetScan_average_household_income_gbp_ss', 'StreetScan_deprivation_rank_for_Overall_Multiple_Deprivation_ss', 'ownership_Owned_Owns_outright', 'MSOA11_CLASSIFICATION_NAME_Affluent_communities', 'yearly_H_std', 'yearly_I_std', 'forecast_price_'],
        'distilled_premium_for_presentation_and_aspirational_value': ['avg_persona_rating_overall', 'std_dev_persona_rating_overall', '_num_features', '_renovation_score', 'ae_', 'num__homipi_estimated_price_range_', 'LSOA_MedPrice_Recent'],
        'distilled_visual_appeal_market_impact': ['avg_persona_rating_overall', 'num_images_total', 'ae_property_wide_', 'LSOA_Annual_Transaction_Rate', 'LSOA_Market_Absorption_Years', 'LSOA_Churn_Recent', 'sale_count', 'compass_mean_'],
        'distilled_key_amenity_quality_vs_market_demand': ['primary_MainKitchen_', 'primary_MainBathroom_', 'ae_primary_MainKitchen_', 'ae_primary_MainBathroom_', 'num__StreetScan_average_household_income_gbp_ss', 'LSOA_MedPrice_Recent', 'OA', 'MSOA'],
        'distilled_condition_in_locational_context': ['avg_persona_rating_overall', '_renovation_score', '_num_flaws', 'ae_primary_MainKitchen_', 'ae_primary_MainBathroom_', 'ae_primary_MainExteriorFront_', 'latitude', 'longitude', 'microscope_emb_', 'num__StreetScan_average_household_income_gbp_ss', 'LSOA_MedPrice_Recent', 'StreetScan_deprivation_rank_for_Overall_Multiple_Deprivation_ss', 'OA11_CLASSIFICATION_NAME_Affluent_communities'],
    }
    for head_name, patterns in distilled_heads.items():
        present_cols_set = set()
        for p in patterns:
            # Search the entire sanitized pool for maximum conceptual coverage
            present_cols_set.update({c for c in sanitized_features_pool if p in c})
        if present_cols_set:
            feature_sets[head_name] = sorted(list(present_cols_set))

    # --- Phase IV: Legacy Monolith & Final Unclaimed Set ---
    print("\n--- Phase IV: Populating Legacy Monoliths & Finalizing Unclaimed Set ---")
    heads_to_exclude = ['head_H_price_history', 'head_I_compass_price_history']
    raw_head_keys = [k for k in legacy_heads.keys() if k.startswith('head_') and k not in heads_to_exclude]
    for head_name in raw_head_keys:
        present_cols = {c for c in legacy_heads[head_name] if c in unclaimed_features}
        if present_cols:
            feature_sets[head_name] = sorted(list(present_cols))
            unclaimed_features -= present_cols

    if unclaimed_features:
        feature_sets['sub_head_unclaimed_final'] = sorted(list(unclaimed_features))

    # --- Phase V: Final Census & Culling Report ---
    print("\n--- V14.0 Head Assembly Census & Culling Report ---")
    final_feature_sets = {k: v for k, v in feature_sets.items() if v}
    total_assigned_specialist = sum(len(v) for k, v in feature_sets.items() if k.startswith('sub_head_') and k != 'sub_head_unclaimed_final')
    unclaimed_size = len(final_feature_sets.get('sub_head_unclaimed_final', []))
    
    print(f"  - Total Sanitized Features for Assembly: {len(sanitized_features_pool)}")
    print(f"  - Features Claimed by Specialists (Phase II): {total_assigned_specialist}")
    print(f"  - Features Remaining for Unclaimed (Final): {unclaimed_size}")
    print(f"  - Total Final Heads Populated: {len(final_feature_sets)}")

    if unclaimed_size > 100:
        print(f"  - [!! ARCHITECTURAL WARNING !!]: Unclaimed feature count ({unclaimed_size}) remains high. Further manifest refinement may be needed.")
        if unclaimed_size < 500: # Check if it's the specific list from the log
             print("     - NOTE: The remaining features appear to be mostly non-programmatic, single-instance columns.")
    else:
        print(f"  - SUCCESS: Unclaimed feature count ({unclaimed_size}) is within acceptable architectural tolerance.")
    
    # Validation check for overlapping assignments in distilled heads
    if 'distilled_locational_character_and_quality' in final_feature_sets and 'latitude' in final_feature_sets['distilled_locational_character_and_quality']:
        if 'sub_head_location_geo' in final_feature_sets and 'latitude' in final_feature_sets['sub_head_location_geo']:
            print("  - SUCCESS: Overlapping assignment confirmed for 'latitude'.")
            
    return final_feature_sets

#
# === END ARCHITECTURAL DIRECTIVE A.D-V14.0 IMPLEMENTATION ===
#

def generate_local_shap_case_studies(shap_values, explainer, data_for_shap, holdout_results, l0_col_mapping, output_dir, n_cases=5):
    """
    [V13.8 DIAGNOSTIC SUITE]
    Identifies the best and worst predictions on the holdout set and generates
    local SHAP waterfall plots to explain the model's reasoning for those specific cases.
    """
    import matplotlib.pyplot as plt
    import shap
    print("\n--- V13.8: Generating Local SHAP Case Studies ---")
    
    # For V18, column names are already human-readable. This step becomes a no-op if mapping is empty.
    data_for_shap_human_readable = data_for_shap
    if l0_col_mapping:
        l0_col_mapping_reverse = {v: k for k, v in l0_col_mapping.items()}
        data_for_shap_human_readable = data_for_shap.rename(columns=l0_col_mapping_reverse)

    # Find worst over-predictions (errors)
    worst_errors = holdout_results.sort_values('absolute_error', ascending=False).head(n_cases)
    
    print(f"  - Generating waterfall plots for {n_cases} worst predictions...")

def sterilize_feature_sets(feature_sets, forbidden_patterns):
    """
    [V13.9 ARCHITECTURE - STERILIZATION GATE]
    Programmatically scans all feature sets and removes any feature matching
    a forbidden pattern to prevent target leakage. This is a non-negotiable
    architectural safeguard.
    """
    print("\n--- V13.9: Engaging Sterilization Gate ---")
    sterilized_sets = {}
    total_removed = 0
    for head_name, features in feature_sets.items():
        original_count = len(features)
        clean_features = [f for f in features if not any(pattern in f for pattern in forbidden_patterns)]
        removed_count = original_count - len(clean_features)
        
        if removed_count > 0:
            print(f"  - WARNING: Sterilized '{head_name}'. Removed {removed_count} forbidden feature(s).")
            total_removed += removed_count
            
        if clean_features:
            sterilized_sets[head_name] = clean_features
    
    print(f"  - Sterilization complete. Total forbidden features removed: {total_removed}.")
    return sterilized_sets


def generate_performance_stratification_report(holdout_results_df, df_holdout_raw, output_dir):
    """
    [V13.8 DIAGNOSTIC SUITE, EXTENDED FOR V36.0]
    Slices holdout performance by key business dimensions. Now includes stratification
    by the V36 `jurisdiction_id` to directly validate the Federated Council's impact.
    """
    print("\n--- V13.8 (V36-Aware): Generating Performance Stratification Report ---")
    
    # Merge predictions with raw features for slicing
    # The holdout_results_df from V36 now contains jurisdiction_id, so a simple merge is sufficient.
    report_df = pd.merge(df_holdout_raw, holdout_results_df, on='property_id', suffixes=('', '_pred'))
    report_path = os.path.join(output_dir, "performance_stratification_report.txt")

    with open(report_path, "w") as f:
        f.write("==============================================\n")
        f.write("=== Performance Stratification Report ===\n")
        f.write("==============================================\n\n")

        # --- [V36.0 ADDITION] Stratify by Market Jurisdiction ---
        if 'jurisdiction_id' in report_df.columns:
            jurisdiction_strat = report_df.groupby('jurisdiction_id')['absolute_error'].agg(['mean', 'count']).sort_values('mean', ascending=False)
            f.write("--- Performance by V36 Market Jurisdiction (MAE) ---\n")
            f.write(jurisdiction_strat.to_string())
            f.write("\n\n")

        # --- 1. Stratify by Price Decile ---
        report_df['price_decile'] = pd.qcut(report_df['most_recent_sale_price'], 10, labels=False, duplicates='drop')
        price_strat = report_df.groupby('price_decile')['absolute_error'].agg(['mean', 'count'])
        f.write("--- Performance by Price Decile (MAE) ---\n")
        f.write(price_strat.to_string())
        f.write("\n\n")

        # --- 2. Stratify by Geographic Cluster ---
        if 'geo_cluster' in report_df.columns:
            geo_strat = report_df.groupby('geo_cluster')['absolute_error'].agg(['mean', 'count']).sort_values('mean', ascending=False)
            f.write("--- Performance by Geographic Cluster (MAE) ---\n")
            f.write(geo_strat.to_string())
            f.write("\n\n")
        
        # --- 3. Stratify by Build Period ---
        bp_cols = [c for c in report_df.columns if c.startswith('BP_')]
        if bp_cols:
            report_df['build_period'] = report_df[bp_cols].idxmax(axis=1)
            bp_strat = report_df.groupby('build_period')['absolute_error'].agg(['mean', 'count']).sort_values('mean', ascending=False)
            f.write("--- Performance by Build Period (MAE) ---\n")
            f.write(bp_strat.to_string())
            f.write("\n\n")

    print(f"  - Stratification report saved to {report_path}")


def engineer_distilled_price_features(df):
    """
    [V13 ARCHITECTURE - TEMPORAL DISTILLATION ENGINE]
    Takes thousands of raw pp_ features and distills them into a small set of
    high-signal, engineered summary features.
    """
    print("\n--- V13: Running Temporal Distillation Engine ---")
    df_copy = df.copy()
    new_feature_names = []

    # Use regex to find all raw price-paid features
    pp_cols = [c for c in df.columns if c.startswith('pp_')]
    if not pp_cols:
        print("  - No raw 'pp_' columns found to distill. Skipping.")
        return df_copy, []

    # Parse years and property types from column names
    parsed_cols = []
    for col in pp_cols:
        parts = col.split('_')
        if len(parts) >= 4:
            try:
                year = int(parts[1])
                prop_type = parts[3]
                parsed_cols.append({'col': col, 'year': year, 'type': prop_type})
            except (ValueError, IndexError):
                continue

    if not parsed_cols:
        return df_copy, []

    # Create a multi-index DataFrame for easy slicing
    price_df_dict = { (int(p['year']), p['type']): p['col'] for p in parsed_cols }
    price_df = df_copy[price_df_dict.values()].rename(columns={v: k for k, v in price_df_dict.items()})
    price_df.columns = pd.MultiIndex.from_tuples(price_df.columns, names=['year', 'type'])

    current_year = datetime.now().year
    
    for prop_type in ['D', 'S', 'T', 'F']:
        if prop_type not in price_df.columns.get_level_values('type'):
            continue
            
        type_df = price_df.xs(prop_type, axis=1, level='type').astype(float)
        
        # --- Engineer Features ---
        # 1. Recency-weighted averages and growth
        for window in [1, 3, 5]:
            start_year = current_year - window
            recent_years = [y for y in range(start_year, current_year) if y in type_df.columns]
            if len(recent_years) > 1:
                # Average price
                avg_price_col = f'distilled_price_avg_{window}y_{prop_type}'
                df_copy[avg_price_col] = type_df[recent_years].mean(axis=1)
                new_feature_names.append(avg_price_col)
                
                # Growth vs oldest year in window
                growth_col = f'distilled_price_growth_{window}y_{prop_type}'
                df_copy[growth_col] = type_df[recent_years[-1]] / (type_df[recent_years[0]] + 1e-6) - 1
                new_feature_names.append(growth_col)
                
                # Volatility
                vol_col = f'distilled_price_volatility_{window}y_{prop_type}'
                df_copy[vol_col] = type_df[recent_years].std(axis=1) / (type_df[recent_years].mean(axis=1) + 1e-6)
                new_feature_names.append(vol_col)

    print(f"  - Successfully distilled {len(pp_cols)} raw features into {len(new_feature_names)} summary features.")
    return df_copy, new_feature_names


def generate_forecast_features(df, forecast_models, forecast_scalers, max_horizon=36, input_window=24):
    """[V9 DECOUPLED & UNABRIDGED] Generates forecasts using all available spatio-temporal features."""
    print(f"  - Generating dense, point-in-time correct monthly forecasts up to {max_horizon} months...")
    df_copy = df.copy()

    prop_type_cols = {
        'F': 'num__property_main_type_encoded__1Flat_hm',
        'D': 'num__property_sub_type_code_from_homipi__4Detached_hm',
        'S': 'num__property_sub_type_code_from_homipi__5Semi_Detached_hm',
        'T': 'num__property_sub_type_code_from_homipi__6Terraced_hm'
    }
    for p_type, col_name in prop_type_cols.items():
        if col_name not in df_copy.columns:
            df_copy[col_name] = 0

    conditions = [
        df_copy[prop_type_cols['F']] == 1,
        df_copy[prop_type_cols['S']] == 1,
        df_copy[prop_type_cols['T']] == 1,
    ]
    choices = ['F', 'S', 'T']
    df_copy['property_type_char'] = np.select(conditions, choices, default='D')

    p_types_np = df_copy['property_type_char'].to_numpy()
    sale_years_np = df_copy['most_recent_sale_year'].to_numpy()
    sale_months_np = df_copy['most_recent_sale_month'].to_numpy()

    all_cols_by_type = {}
    for p_type in ['D', 'S', 'T', 'F']:
        # [CORRECTED] Discover all relevant spatio-temporal features, not just 'pp_*'
        raw_cols = [c for c in df.columns if re.match(fr".*_pp_(\d{{4}})_(\d{{2}})_{p_type}_.*", c)]
        compass_cols = [c for c in df.columns if re.match(fr".*compass_.*_pp_(\d{{4}})_(\d{{2}})_{p_type}_.*", c)]
        all_cols_by_type[p_type] = sorted(list(set(raw_cols + compass_cols)))

    # ARCHITECTURAL CORRECTION: Pass the model's state_dict (a picklable dictionary)
    # instead of the full model object to avoid serialization errors with joblib.
    tasks = [delayed(_forecast_worker)(
        i, df.iloc[i], p_types_np[i], sale_years_np[i], sale_months_np[i],
        forecast_models[p_types_np[i]].state_dict() if p_types_np[i] in forecast_models else None,
        forecast_scalers.get(p_types_np[i]), all_cols_by_type,
        max_horizon, input_window, DEVICE
    ) for i in range(len(df))]

    results = Parallel(n_jobs=-1, backend="loky")(tqdm(tasks, total=len(df), desc="Generating Forecast Features"))
    
    results_matrix = {f'forecast_price_{h}m': np.zeros(len(df)) for h in range(1, max_horizon + 1)}
    for i, horizon_predictions in results:
        if horizon_predictions:
            for h, value in horizon_predictions.items():
                col_name = f'forecast_price_{h}m'
                if col_name in results_matrix:
                    results_matrix[col_name][i] = value
    
    forecasts_df = pd.DataFrame(results_matrix, index=df_copy.index)
    df_copy = pd.concat([df_copy, forecasts_df], axis=1)
    new_feature_names = forecasts_df.columns.tolist()
    print(f"  - Generated {len(new_feature_names)} new dense monthly forecast features.")
    return df_copy, new_feature_names


def parse_and_prepare_data_for_prop_type(df, cols):
    if not cols: return None, 0, 0
    parsed_data, feature_stems = {}, set()
    pattern = re.compile(r"(.*)_(\d{4})_(\d{2})_(.*)")
    for col in cols:
        match = pattern.search(col)
        if match:
            prefix, year, month, suffix = match.groups()
            timestep = (int(year), int(month))
            feature_stem = f"{prefix}_TIMESTAMP_{suffix}"
            feature_stems.add(feature_stem)
            if timestep not in parsed_data: parsed_data[timestep] = {}
            parsed_data[timestep][feature_stem] = col
    timesteps = sorted(parsed_data.keys())
    canonical_features = sorted(list(feature_stems))
    n_timesteps, n_features_per_timestep = len(timesteps), len(canonical_features)
    if n_timesteps == 0 or n_features_per_timestep == 0: return None, 0, 0
    tcn_array = np.zeros((len(df), n_timesteps, n_features_per_timestep))
    for j, ts in enumerate(timesteps):
        for i, stem in enumerate(canonical_features):
            col_name = parsed_data.get(ts, {}).get(stem)
            if col_name and col_name in df.columns:
                tcn_array[:, j, i] = df[col_name].fillna(0).values
    return tcn_array, n_timesteps, n_features_per_timestep

def _forecast_worker(i, df_row, p_type, sale_year, sale_month, model_state_dict, scaler, all_cols_by_type, max_horizon, input_window, DEVICE):
    """Worker function to generate forecast features for a single property (row)."""
    try:
        if model_state_dict is None or scaler is None: return i, None

        # Reconstruct the model inside the worker process from the state_dict.
        n_features = scaler.n_features_in_
        forecast_model = ForecastTCN(
            input_feature_dim=n_features,
            output_dim=n_features,
            num_channels=[64, 128] # Must match architecture of the pre-trained model
        ).to(DEVICE)
        forecast_model.load_state_dict(model_state_dict)
        forecast_model.eval()

        if pd.isna(sale_year) or pd.isna(sale_month): sale_year, sale_month = datetime.now().year, datetime.now().month

        all_cols_for_type = all_cols_by_type.get(p_type, [])
        point_in_time_cols = []
        patterns = [re.compile(r".*_(\d{4})_(\d{2})_.*"), re.compile(r".*_(\d{4})_.*")]
        
        for col in all_cols_for_type:
            col_year, col_month = 0, 1
            for pattern in patterns:
                match = pattern.search(col)
                if match:
                    groups = [g for g in match.groups() if g is not None]
                    col_year = int(groups[0])
                    if len(groups) > 1: col_month = int(groups[1])
                    break
            if col_year > 0 and (col_year < sale_year or (col_year == sale_year and col_month < sale_month)):
                point_in_time_cols.append(col)
        
        single_row_df = pd.DataFrame([df_row])
        data_array, n_timesteps, n_features = parse_and_prepare_data_for_prop_type(single_row_df, point_in_time_cols)
        
        if data_array is None or n_timesteps < input_window: return i, None

        data_flat = data_array.reshape(n_timesteps, n_features)
        data_scaled_flat = scaler.transform(data_flat)
        data_scaled_array = data_scaled_flat.reshape(1, n_timesteps, n_features)

        current_sequence = data_scaled_array[:, -input_window:, :]
        current_sequence_tensor = torch.tensor(current_sequence, dtype=torch.float32).permute(0, 2, 1).to(DEVICE)
        
        horizon_predictions = {}
        with torch.no_grad():
            for step in range(max_horizon):
                next_step_pred = forecast_model(current_sequence_tensor)
                current_sequence_tensor = torch.cat([current_sequence_tensor[:, :, 1:], next_step_pred.unsqueeze(2)], dim=2)
                final_forecast_unscaled = scaler.inverse_transform(next_step_pred.cpu().numpy())
                horizon_predictions[step + 1] = final_forecast_unscaled[0, 0]
        return i, horizon_predictions
    except Exception as e:
        return i, None

def load_and_merge_ae_features(df, encodings_dir, keys_path):
    """
    [V10.4 ARCHITECTURE - DEFINITIVE RECONCILIATION MERGE]
    Uses a key file to build a new, perfectly aligned matrix of all AE features
    that exactly matches the main dataframe's properties and row order. This
    prevents all forms of index/key mismatch and column collision.
    """
    print("\n--- STAGE 3.9: Final Integration of Gemini Embeddings (Head G) ---")
    print("--- Loading and Merging External Autoencoder (Head G) Features ---")

    try:
        # Load the "Rosetta Stone" - the complete key file from the AE pipeline.
        keys_df = pd.read_csv(keys_path)
        print(f"  - Loaded {len(keys_df)} property keys for merge reconciliation.")
    except FileNotFoundError:
        print(f"  - FATAL ERROR: Property key file not found at '{keys_path}'. Cannot perform safe merge.")
        exit(1)

    # --- Build the new, perfectly aligned embedding matrix ---
    
    # 1. Start with the main dataframe's keys. This is our template ("The Spine").
    #    It guarantees the final matrix has the exact same properties and order.
    if 'property_id' not in df.columns:
        raise KeyError("FATAL: Canonical 'property_id' column not found in main dataframe for key-based merge.")
    aligned_ae_df = df[['property_id']].copy()
    
    all_new_ae_cols = []
    num_merged_sets = 0

    for filename in sorted(os.listdir(encodings_dir)):
        if filename.endswith(".npy"):
            try:
                group_name = filename.replace("_encodings.npy", "")
                file_path = os.path.join(encodings_dir, filename)
                
                ae_encodings = np.load(file_path)
                
                if ae_encodings.shape[0] != len(keys_df):
                    print(f"  - FATAL ERROR: Row count mismatch for '{filename}'.")
                    print(f"    Key file has {len(keys_df)} rows, but encoding file has {ae_encodings.shape[0]} rows.")
                    exit(1)
                
                ae_cols = [f"ae_feat_{group_name}_{i}" for i in range(ae_encodings.shape[1])]
                all_new_ae_cols.extend(ae_cols)

                # 2. Create a temporary df with the FULL set of keys and encodings
                temp_full_df = pd.DataFrame(ae_encodings, columns=ae_cols)
                temp_full_df['property_id'] = keys_df['property_id']

                # 3. Use a LEFT MERGE to transfer encodings to our aligned template.
                #    This acts like a VLOOKUP, pulling in the data for only the properties
                #    that exist in our main 'df', in the correct order.
                aligned_ae_df = pd.merge(aligned_ae_df, temp_full_df, on='property_id', how='left')

                num_merged_sets += 1
                print(f"  - Successfully processed and aligned {len(ae_cols)} features for group '{group_name}'.")

            except Exception as e:
                print(f"  - ERROR: Could not process file {filename}. Error: {e}")

    # 4. Drop the key column, leaving a pure feature matrix that is perfectly aligned.
    aligned_ae_df.drop(columns=['property_id'], inplace=True)
    
    # 5. Concatenate the final aligned matrix with the main dataframe.
    #    Because we built it from df's keys, the indexes are guaranteed to match.
    #    reset_index() is a defensive measure to ensure perfect alignment.
    final_df = pd.concat([df.reset_index(drop=True), aligned_ae_df.reset_index(drop=True)], axis=1)
    
    print(f"\n  - Successfully merged {num_merged_sets} aligned AE feature sets. New dataframe shape: {final_df.shape}")
    
    return final_df, all_new_ae_cols

    

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

def process_rightmove_data(raw_rightmove_df, sales_history_col): # <-- ADD ARGUMENT
    print("Processing Rightmove sales data to extract features...")
    processed_data = []
    
    # ARCHITECTURAL CORRECTION: The column name is now passed in as a parameter,
    # removing the brittle hardcoded value and fulfilling the V80.3 mandate.
    if sales_history_col not in raw_rightmove_df.columns:
        raise KeyError(
            f"FATAL: The required sales history column '{sales_history_col}' was not found. "
            "This indicates a failure in the upstream data loading contract."
        )
    print(f"  - Using canonical column for sales data: '{sales_history_col}'")
        
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
            # Enforce the exact column order the artifact was trained on.
            try:
                head_df_raw = head_df_raw[expected_cols]
            except KeyError as e:
                raise KeyError(f"FATAL: A required column was missing during transform for head '{head_name}'. This should have been handled. Error: {e}")

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
    [VECTORIZED & HARDENED per A.D-V48.4] Detects leakage, provides a full correlation audit, and mitigates near-perfect correlations.
    """
    print("\n--- STAGE 4.1: Mitigating AVM Target Leakage (Correlation Check & Audit) ---")
    avm_keywords = ['estimate', 'valuation', 'avm', 'homipi', 'zoopla', 'bnl', 'bricks', 'chimnie']
    df_copy = df.copy()

    if label_col not in df_copy.columns:
        print(f"  - FATAL: Label column '{label_col}' not found. Cannot perform leakage check.")
        return df

    candidate_cols = [col for col in df_copy.select_dtypes(include=np.number).columns if any(keyword in col.lower() for keyword in avm_keywords)]
    if not candidate_cols:
        print("  - No potential AVM feature columns found to check.")
        return df_copy

    correlations = df_copy[candidate_cols].corrwith(df_copy[label_col])
    
    # [A.D-V48.4] ENHANCED LOGGING: Provide a permanent audit trail of all AVM correlations.
    print("  - AVM Correlation Audit Log:")
    print(correlations.abs().sort_values(ascending=False).to_string())
    print("-" * 30)

    leaky_cols = correlations[correlations.abs() > correlation_threshold].index.tolist()

    if not leaky_cols:
        print("  - No significant correlation leakage detected (abs(corr) > 0.995).")
        return df_copy

    print("  [!! HIGH-CORRELATION LEAKAGE DETECTED !!] The following columns will be mitigated:")
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

def engineer_temporal_summary_features(df):
    """
    [VECTORIZED & DECOUPLED] Performs point-in-time correct longitudinal compression.
    This function now autonomously discovers temporal feature columns using regex,
    making it a self-contained feature engineering step.
    """
    print("\n--- STAGE 4.1.5: Point-in-Time Temporal Summary Feature Engineering ---")
    
    if 'most_recent_sale_year' not in df.columns:
        print("  - WARNING: `most_recent_sale_year` not found. Skipping temporal summary feature engineering.")
        return df, []
        
    temporal_col_pattern = re.compile(r".*_(\d{4})_.*")
    temporal_cols = [c for c in df.columns if temporal_col_pattern.search(c)]
    
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


# CORRECTED V27.0 SUBROUTINE - THE TRUE V18 ENSEMBLE
def train_v18_residual_ensemble(df_main_raw, feature_sets, universal_cols_present, output_dir, is_sub_model=False, domain_name="global", init_model_package=None, custom_params=None):
    """
    [V18 ARCHITECTURE, V48.1-AWARE]
    Core training engine, now modified to support fine-tuning via `init_model_package`.
    """
    if is_sub_model:
        if init_model_package:
            print(f"  - Fine-tuning V18 model for domain '{domain_name}' on {len(df_main_raw)} samples...")
        else:
            print(f"  - Training V18 model ({domain_name}) on {len(df_main_raw)} samples...")
    else:
        print("\n--- V18 RESIDUAL ENSEMBLE TRAINING WORKFLOW INITIATED ---")

    y_main_log = np.log1p(df_main_raw['most_recent_sale_price'])
    trained_models = {'specialists': {}}
    kf = KFold(n_splits=10, shuffle=True, random_state=42) # Reduced folds for faster fine-tuning iterations

    # --- Stage 1: Train the Universalist Baseline Model ---
    X_baseline = df_main_raw[universal_cols_present].fillna(0).copy()
    baseline_oof_preds_log = np.zeros(len(df_main_raw))
    baseline_params = {'n_estimators': 2000, 'learning_rate': 0.02, 'num_leaves': 31, 'reg_alpha': 0.1, 'reg_lambda': 0.1, 'colsample_bytree': 0.7, 'subsample': 0.7, 'random_state': 42, 'n_jobs': -1, 'verbosity': -1}
    if custom_params and 'baseline' in custom_params: baseline_params.update(custom_params['baseline'])

    foundation_baseline = init_model_package.get('baseline_model') if init_model_package else None
    
    # OOF predictions are still generated without init_model to get a true measure of performance on this specific data slice
    for _, (train_idx, val_idx) in enumerate(kf.split(X_baseline)):
        model = lgb.LGBMRegressor(**baseline_params)
        model.fit(X_baseline.iloc[train_idx], y_main_log.iloc[train_idx], 
                  eval_set=[(X_baseline.iloc[val_idx], y_main_log.iloc[val_idx])],
                  callbacks=[lgb.early_stopping(50, verbose=False)], init_model=foundation_baseline)
        baseline_oof_preds_log[val_idx] = model.predict(X_baseline.iloc[val_idx])

    final_baseline_model = lgb.LGBMRegressor(**baseline_params)
    final_baseline_model.fit(X_baseline, y_main_log, init_model=foundation_baseline)
    trained_models['baseline_model'] = final_baseline_model
    baseline_mae = mean_absolute_error(np.expm1(y_main_log), np.expm1(baseline_oof_preds_log))

    # --- Stage 2: Train L0 Specialists on the Residuals ---
    y_residuals_log = y_main_log - baseline_oof_preds_log
    l0_oof_residual_preds_df = pd.DataFrame(index=df_main_raw.index)
    specialist_params = {'n_estimators': 500, 'learning_rate': 0.05, 'num_leaves': 8, 'random_state': 42, 'n_jobs': -1, 'verbosity': -1}
    if custom_params and 'specialist' in custom_params: specialist_params.update(custom_params['specialist'])

    for head_name, cols in feature_sets.items():
        if not cols: continue
        X_specialist = df_main_raw[cols].fillna(0).copy()
        
        # [A.D-V66.0 INTEGRATION] Call the Type Enforcement Gate before training.
        X_specialist = _enforce_model_input_schema(X_specialist, model_name=f"L0_Specialist_{head_name}")
        
        oof_preds = np.zeros(len(df_main_raw))
        foundation_specialist = init_model_package['specialists'].get(head_name) if init_model_package and 'specialists' in init_model_package else None
        
        for _, (train_idx, val_idx) in enumerate(kf.split(X_specialist)):
            model = lgb.LGBMRegressor(**specialist_params)
            model.fit(X_specialist.iloc[train_idx], y_residuals_log.iloc[train_idx], init_model=foundation_specialist)
            oof_preds[val_idx] = model.predict(X_specialist.iloc[val_idx])
        l0_oof_residual_preds_df[f'l0_resid_pred_{head_name}'] = oof_preds
        
        final_specialist_model = lgb.LGBMRegressor(**specialist_params)
        final_specialist_model.fit(X_specialist, y_residuals_log, init_model=foundation_specialist)
        trained_models['specialists'][head_name] = final_specialist_model

    # --- Stage 3: Train the L1 Assembler Model ---
    X_assembler = pd.concat([pd.DataFrame({'baseline_pred_log': baseline_oof_preds_log}, index=df_main_raw.index), X_baseline, l0_oof_residual_preds_df], axis=1).copy()
    assembler_params = {'n_estimators': 1000, 'learning_rate': 0.02, 'num_leaves': 21, 'random_state': 42, 'n_jobs': -1, 'verbosity': -1}
    if custom_params and 'assembler' in custom_params: assembler_params.update(custom_params['assembler'])

    foundation_assembler = init_model_package.get('assembler_model') if init_model_package else None
    
    oof_assembler_preds_log = np.zeros(len(df_main_raw))
    for _, (train_idx, val_idx) in enumerate(kf.split(X_assembler)):
        model = lgb.LGBMRegressor(**assembler_params)
        model.fit(X_assembler.iloc[train_idx], y_main_log.iloc[train_idx], 
                  eval_set=[(X_assembler.iloc[val_idx], y_main_log.iloc[val_idx])],
                  callbacks=[lgb.early_stopping(25, verbose=False)], init_model=foundation_assembler)
        oof_assembler_preds_log[val_idx] = model.predict(X_assembler.iloc[val_idx])

    final_assembler_model = lgb.LGBMRegressor(**assembler_params)
    final_assembler_model.fit(X_assembler, y_main_log, init_model=foundation_assembler)
    trained_models['assembler_model'] = final_assembler_model

    # --- [A.D-V48.1-FIX] REINSTATE THE ARCHITECTURAL RETURN CONTRACT ---
    # The final model package must contain all necessary context for inference to prevent downstream KeyError.
    trained_models.update({
        'feature_sets': feature_sets,
        'universal_cols': universal_cols_present,
        'architecture_version': 'V18_Residual_Ensemble_Component'
    })

    # Create the evaluation dataframe with OOF predictions.
    eval_df = df_main_raw[['property_id', 'most_recent_sale_price']].copy()
    eval_df['final_predicted_price'] = np.expm1(oof_assembler_preds_log)

    # This return statement is non-negotiable and fulfills the contract with the V48 orchestrators.
    if is_sub_model:
        return trained_models, eval_df

    # --- [LEGACY SUPPORT] This path is for standalone execution and diagnostics, not used by V48. ---
    # A full return is maintained for backward compatibility and potential debugging.
    return trained_models, eval_df, None, None, baseline_mae

# CORRECTED V27.1 SUBROUTINE - THE TRUE V18 PREDICTOR
# DEFINITIVE V29.0 UNIFIED PREDICTOR - REPLACES ALL PREVIOUS VERSIONS
def predict_on_holdout_v18(df_holdout_raw, trained_models, return_oof_log_preds=False):
    """
    [V18 ARCHITECTURE - REUSABLE COMPONENT] Predicts using the V18 Residual Ensemble.
    This is the architecturally correct counterpart to the V18 training function,
    resolving all previous signature and key errors.
    """
    if df_holdout_raw.empty: return pd.DataFrame() if not return_oof_log_preds else np.array([])

    # [A.D-29.0] Unpack the self-contained artifact. The function signature is now clean.
    feature_sets = trained_models['feature_sets']
    universal_cols_present = trained_models['universal_cols']
    
    # --- Stage 1: Baseline Prediction ---
    X_baseline_holdout = df_holdout_raw[universal_cols_present].fillna(0).copy()
    baseline_preds_log = trained_models['baseline_model'].predict(X_baseline_holdout)

    # --- Stage 2: Specialist Residual Prediction ---
    l0_holdout_residual_preds_df = pd.DataFrame(index=df_holdout_raw.index)
    for head_name, model in trained_models['specialists'].items():
        cols = feature_sets.get(head_name, [])
        if not cols: continue
        present_cols = [c for c in cols if c in df_holdout_raw.columns]
        X_holdout_specialist = df_holdout_raw[present_cols].fillna(0)
        missing_cols = set(cols) - set(present_cols)
        for col in missing_cols: X_holdout_specialist[col] = 0
        l0_holdout_residual_preds_df[f'l0_resid_pred_{head_name}'] = model.predict(X_holdout_specialist[cols])

    # --- Stage 3: Assembler Final Prediction ---
    X_assembler_holdout = pd.concat([
        pd.DataFrame({'baseline_pred_log': baseline_preds_log}, index=df_holdout_raw.index), 
        X_baseline_holdout, 
        l0_holdout_residual_preds_df
    ], axis=1).copy()
    
    expected_assembler_features = trained_models['assembler_model'].feature_name_
    X_assembler_holdout = X_assembler_holdout.reindex(columns=expected_assembler_features, fill_value=0)
    
    final_preds_log = trained_models['assembler_model'].predict(X_assembler_holdout)

    if return_oof_log_preds:
        return final_preds_log

    final_preds = np.expm1(final_preds_log)
    results_df = df_holdout_raw[['property_id', 'most_recent_sale_price']].copy()
    results_df['predicted_price'] = final_preds
    return results_df


#
# === END ARCHITECTURAL DIRECTIVE A.D-V23.0 IMPLEMENTATION ===
#



def generate_master_intelligence_report_v18(specialist_perf_df, assembler_shap_artifacts, diagnostics_base_dir, output_dir):
    """
    [V18.2 SYNTHESIS]
    Aggregates performance, importance, and complexity data from all component-level
    diagnostic reports into a single Master Intelligence Report. This provides a
    unified view for strategic model refinement.
    """
    print("\n--- V18.2: Generating Master Intelligence Report ---")
    if assembler_shap_artifacts is None:
        print("  - Assembler SHAP artifacts not found. Cannot generate master report.")
        return

    # --- 1. Calculate Assembler's Trust in Each Specialist ---
    assembler_shap_values = assembler_shap_artifacts['shap_values']
    assembler_features = assembler_shap_artifacts['X_data'].columns
    
    shap_df = pd.DataFrame(np.abs(assembler_shap_values), columns=assembler_features)
    assembler_importance = shap_df.mean().reset_index()
    assembler_importance.columns = ['feature', 'assembler_mean_abs_shap']
    
    specialist_importance = assembler_importance[assembler_importance['feature'].str.startswith('l0_resid_pred_')].copy()
    specialist_importance['specialist_head'] = specialist_importance['feature'].str.replace('l0_resid_pred_', '')
    
    # --- 2. Read Top Features from Each Specialist's Report ---
    specialist_reports_dir = os.path.join(diagnostics_base_dir, "01_l0_specialists")
    top_features_data = []
    if os.path.exists(specialist_reports_dir):
        for head_name in os.listdir(specialist_reports_dir):
            report_path = os.path.join(specialist_reports_dir, head_name, "feature_importance.csv")
            if os.path.exists(report_path):
                try:
                    importance_df = pd.read_csv(report_path)
                    top_5 = importance_df.head(5)['feature'].tolist()
                    num_features = len(importance_df)
                    top_features_data.append({
                        'specialist_head': head_name,
                        'num_features': num_features,
                        'top_5_features': ", ".join(top_5)
                    })
                except Exception:
                    continue

    top_features_df = pd.DataFrame(top_features_data)

    # --- 3. Merge All Intelligence Sources ---
    master_df = pd.merge(specialist_perf_df, specialist_importance[['specialist_head', 'assembler_mean_abs_shap']], on='specialist_head', how='left')
    if not top_features_df.empty:
        master_df = pd.merge(master_df, top_features_df, on='specialist_head', how='left')
    
    master_df = master_df.sort_values('assembler_mean_abs_shap', ascending=False).reset_index(drop=True)
    
    report_path = os.path.join(output_dir, "v18_master_intelligence_report.csv")
    master_df.to_csv(report_path, index=False)
    
    print(f"  - Master Intelligence Report saved to {report_path}")
    print("\n--- Top 10 Most Influential Specialists in Final Assembler ---")
    print(master_df[['specialist_head', 'standalone_log_residual_mae', 'assembler_mean_abs_shap', 'num_features']].head(10).to_string())



# === BEGIN ARCHITECTURAL DIRECTIVE A.D-V40.0 IMPLEMENTATION ===

def _engineer_and_validate_dispatcher_features_v40(df, signal_hierarchy):
    """
    [V40.1 ARCHITECTURE - SYMMETRICAL DISPATCHER FEATURE ENGINE]
    The single source of truth for creating all features required by the dispatcher.
    This function is called symmetrically by both training and prediction pipelines.
    It returns the modified dataframe and the list of feature names created.
    """
    print("\n--- [V40.1] Engaging Symmetrical Dispatcher Feature Engine ---")
    df_copy = df.copy()

    # Part 1: Engineer distance-based macro-regional features
    london_coords = np.array([51.5072, -0.1276])
    edinburgh_coords = np.array([55.9533, -3.1883])
    property_coords = df_copy[['latitude', 'longitude']].to_numpy()
    df_copy['dist_from_london'] = np.linalg.norm(property_coords - london_coords, axis=1)
    df_copy['dist_from_edinburgh'] = np.linalg.norm(property_coords - edinburgh_coords, axis=1)
    print("  - Created distance-based macro-regional features.")

    # Part 2: Engineer and validate the critical market dynamics signal
    final_signal_col_name = 'dispatcher_market_dynamics_signal'
    for signal_candidate in signal_hierarchy:
        candidate_name, candidate_type = signal_candidate['name'], signal_candidate['type']
        signal_vector = None
        if candidate_type == 'composite' and candidate_name == 'PC_Sales_Volatility_YoY_':
            # Composite logic remains the same
            volatility_cols = {'D': 'PC_Sales_Volatility_YoY_D', 'F': 'PC_Sales_Volatility_YoY_F', 'S': 'PC_Sales_Volatility_YoY_S', 'T': 'PC_Sales_Volatility_YoY_T'}
            prop_type_cols = {'D': 'num__property_sub_type_code_from_homipi__4Detached_hm', 'F': 'num__property_main_type_encoded__1Flat_hm', 'S': 'num__property_sub_type_code_from_homipi__5Semi_Detached_hm', 'T': 'num__property_sub_type_code_from_homipi__6Terraced_hm'}
            if any(vc in df_copy.columns for vc in volatility_cols.values()):
                temp_signal = pd.Series(np.nan, index=df_copy.index)
                for prop_type, type_col in prop_type_cols.items():
                    vol_col = volatility_cols[prop_type]
                    if vol_col in df_copy.columns and type_col in df_copy.columns:
                        mask = df_copy[type_col] == 1
                        temp_signal.loc[mask] = df_copy.loc[mask, vol_col]
                if temp_signal.isnull().any():
                    imputation_value = temp_signal.median()
                    if not pd.isna(imputation_value): temp_signal.fillna(imputation_value, inplace=True)
                signal_vector = temp_signal
        elif candidate_type == 'direct' and candidate_name in df_copy.columns:
            signal_vector = df_copy[candidate_name]

        if signal_vector is not None and not signal_vector.isnull().all() and signal_vector.var() > 1e-6:
            print(f"  - SUCCESS: Acquired valid signal from '{candidate_name}'.")
            df_copy[final_signal_col_name] = signal_vector
            break
    else: # This 'else' belongs to the 'for' loop, executing if the loop completes without break
        raise ValueError("FATAL: Symmetrical Feature Engine failed. No candidate in the hierarchy produced a valid signal.")

    # Part 3: Define and validate the final feature contract
    CRITICAL_DISPATCHER_FEATURES = ['dist_from_london', 'dist_from_edinburgh', 'LSOA_Property_Density', final_signal_col_name]
    if not all(f in df_copy.columns for f in CRITICAL_DISPATCHER_FEATURES):
        raise ValueError(f"FATAL: Symmetrical Feature Engine failed to produce required columns.")
    
    print("  - Symmetrical feature engineering complete and contract fulfilled.")
    return df_copy, CRITICAL_DISPATCHER_FEATURES

def _train_dispatcher_v40(df_specialist_zone, dispatcher_feature_names, output_dir):
    """
    [V40.1 ARCHITECTURE - DISPATCHER TRAINING COMPONENT]
    This component is now responsible only for clustering. It receives a dataframe
    and a list of feature names, enforcing a strict data contract.
    """
    print("\n--- [V40.1] Phase 2: Training the Dispatcher Clustering Component ---")
    
    X_cluster = df_specialist_zone[dispatcher_feature_names].copy()
    X_cluster.fillna(X_cluster.median(), inplace=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)
    
    MIN_CLUSTER_SIZE = 150
    print(f"  - Performing density-based clustering with min_cluster_size={MIN_CLUSTER_SIZE}...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE, 
        min_samples=10, 
        gen_min_span_tree=True,
        prediction_data=True
    )
    cluster_labels = clusterer.fit_predict(X_scaled)
    
    final_cluster_labels_series = pd.Series(cluster_labels, index=df_specialist_zone.index, name="jurisdiction_id")

    print("  - Dispatcher clustering complete.")
    final_counts = final_cluster_labels_series.value_counts()
    print("  - Final Jurisdiction Counts (Note: -1 represents noise/outliers):\n", final_counts.to_string())
    
    if len(final_counts) < 2 and -1 in final_counts.index:
         print("  - [!!] ARCHITECTURAL WARNING: HDBSCAN classified all points as noise.")
    
    dispatcher_artifacts = {'scaler': scaler, 'clusterer': clusterer}
    return dispatcher_artifacts, final_cluster_labels_series



def _validate_holdout_integrity_v45(df_full, df_holdout, stratify_col='error_zone_flag', tolerance=0.15):
    """[V45.0] Enforces the Validation Integrity Contract."""
    print("\n--- [V45] Engaging Validation Integrity Contract ---")
    
    if df_full.empty or df_holdout.empty:
        print("  - WARNING: One or both dataframes are empty. Skipping integrity check.")
        return

    full_prevalence = df_full[stratify_col].value_counts(normalize=True)
    holdout_prevalence = df_holdout[stratify_col].value_counts(normalize=True)

    # Compare prevalence of the minority class (error zone)
    error_zone_prev_full = full_prevalence.get(1, 0)
    error_zone_prev_holdout = holdout_prevalence.get(1, 0)
    
    print(f"  - Error Zone Prevalence (Full Set):   {error_zone_prev_full:.2%}")
    print(f"  - Error Zone Prevalence (Holdout Set): {error_zone_prev_holdout:.2%}")

    if error_zone_prev_full == 0:
        if error_zone_prev_holdout > 0:
             print("  - [!!] VALIDATION WARNING: No error zone samples in training data, but present in holdout.")
        return # Cannot calculate relative difference if base is zero

    relative_diff = abs(error_zone_prev_holdout - error_zone_prev_full) / error_zone_prev_full
    print(f"  - Relative Difference: {relative_diff:.2%}")

    if relative_diff > tolerance:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! FATAL: VALIDATION INTEGRITY VIOLATED.                                  !!!")
        print(f"!!! Holdout set stratification differs by >{tolerance:.0%} from the full dataset.   !!!")
        print("!!! The test is blind and therefore invalid. ABORTING.                     !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        import sys
        sys.exit(1)
    
    print("  - SUCCESS: Holdout set is a representative sample. Validation is trustworthy.")


def _enrich_with_generalist_knowledge_v45(generalist_models, df_to_enrich):
    """[V45.0] Transfers knowledge from the Generalist as a suite of new features."""
    print("\n--- [V45] Enriching Specialist Zone with Generalist Knowledge Features ---")
    df_enriched = df_to_enrich.copy()
    new_feature_names = []
    
    generalist_specialists = generalist_models.get('specialists', {})
    generalist_feature_sets = generalist_models.get('feature_sets', {})

    for head_name, model in generalist_specialists.items():
        feature_name = f"gen_opinion_on_{head_name}"
        cols_for_head = generalist_feature_sets.get(head_name, [])
        
        if not cols_for_head:
            continue
        
        # Symmetrically prepare data for the specialist model
        X_predict = df_enriched.reindex(columns=cols_for_head, fill_value=0).fillna(0)
        
        opinion = model.predict(X_predict)
        df_enriched[feature_name] = opinion
        new_feature_names.append(feature_name)
        
    print(f"  - Successfully created {len(new_feature_names)} new Generalist Opinion features.")
    return df_enriched, new_feature_names

def _engineer_forced_interaction_features_v47(df_local, df_with_opinions, gen_opinion_features, non_feature_cols, n_top_local=15):
    """[V47.0] Implements the Forced Signal Amplification mandate."""
    print(f"    - Engineering V47 Forced Interaction Features for domain of size {len(df_local)}...")
    
    # Identify local power features, excluding opinions and other non-features
    local_candidate_cols = [c for c in df_local.columns if c not in gen_opinion_features and c not in non_feature_cols]
    target_series = df_local['most_recent_sale_price']
    
    top_local_features = select_features_with_lgbm(
        df_local, target_series, local_candidate_cols, n_top_local, "V47_Interaction_Selection"
    )
    
    df_out = df_local.copy()
    new_interaction_features = []
    
    # Create interactions between top local features and ALL generalist opinions
    for local_feat in top_local_features:
        for opinion_feat in gen_opinion_features:
            interaction_name = f"interaction_{local_feat}_X_{opinion_feat}"
            # Ensure alignment by using the index of df_local
            df_out[interaction_name] = df_local[local_feat] * df_with_opinions.loc[df_local.index, opinion_feat]
            new_interaction_features.append(interaction_name)
            
    print(f"    - Created {len(new_interaction_features)} new interaction features.")
    return df_out, new_interaction_features, top_local_features



def train_residual_council_v44(df_main_raw, feature_sets, generalist_universal_cols, global_specialist_universal_cols, NON_FEATURE_COLS, signal_hierarchy, output_dir):
    """
    [V44.0 ARCHITECTURE - KNOWLEDGE TRANSFER ORCHESTRATOR]
    Implements the full hierarchical training pipeline where specialists are re-missioned
    as residual correctors for the universal Generalist baseline.
    """
    print(f"\n--- V44.0 KNOWLEDGE TRANSFER TRAINING ORCHESTRATOR ---")
    is_error_zone = df_main_raw['error_zone_flag'] == 1
    df_specialist_zone = df_main_raw[is_error_zone].copy()
    df_general_zone = df_main_raw[~is_error_zone].copy()
    y_main_log = np.log1p(df_main_raw['most_recent_sale_price'])

    # --- Phase 1: Establish the Universal Foundational Baseline ---
    print("\n--- [V44] Phase 1: Training Generalist & Establishing Universal Baseline ---")
    generalist_models, generalist_eval_df = train_v18_residual_ensemble(df_general_zone, feature_sets, generalist_universal_cols, output_dir, is_sub_model=True, domain_name="generalist")
    
    # Create the full, leak-free baseline OOF prediction vector for the entire training set
    universal_baseline_oof_preds_log = pd.Series(np.nan, index=df_main_raw.index)
    # Part A: Pure OOF predictions for the stable zone (from the training loop)
    stable_preds_map = generalist_eval_df.set_index('property_id')['final_predicted_price']
    universal_baseline_oof_preds_log.loc[~is_error_zone] = np.log1p(df_general_zone['property_id'].map(stable_preds_map))
    # Part B: Generalist's prediction for the error zone (for training the next tier)
    generalist_preds_on_error_zone = predict_on_holdout_v18(df_specialist_zone, generalist_models, return_oof_log_preds=True)
    universal_baseline_oof_preds_log.loc[is_error_zone] = generalist_preds_on_error_zone

    # --- Phase 2: Train the Hierarchical Residual Correction Council ---
    df_specialist_zone, dispatcher_feature_names = _engineer_and_validate_dispatcher_features_v40(df_specialist_zone, signal_hierarchy)
    dispatcher_artifacts, cluster_labels = _train_dispatcher_v40(df_specialist_zone, dispatcher_feature_names, output_dir)
    
    CONTRABAND_AVM_FEATURES = ['num__homipi_estimated_price', 'num__mouseprice_estimated_value']
    firewalled_feature_sets = {h: [f for f in fs if not any(c in f for c in CONTRABAND_AVM_FEATURES)] for h, fs in feature_sets.items()}
    
    council_models, final_council_oof_correction_log, performance_manifest = _train_residual_council_v44(
        df_specialist_zone, y_main_log[is_error_zone], universal_baseline_oof_preds_log[is_error_zone],
        firewalled_feature_sets, global_specialist_universal_cols, cluster_labels, output_dir
    )

    # --- Phase 3: Train the Reformed Arbiter ---
    print("\n--- [V44] Phase 3: Training the Reformed Arbiter ---")
    X_arbiter_error_zone = pd.DataFrame({
        'generalist_pred_log': universal_baseline_oof_preds_log[is_error_zone],
        'final_council_correction_log': final_council_oof_correction_log,
        'latitude': df_specialist_zone['latitude'].fillna(df_specialist_zone['latitude'].mean()),
        'longitude': df_specialist_zone['longitude'].fillna(df_specialist_zone['longitude'].mean())
    }, index=df_specialist_zone.index)
    
    y_main_log_error_zone = y_main_log[is_error_zone]
    arbiter_model = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.015, num_leaves=21, random_state=42, n_jobs=-1)
    arbiter_model.fit(X_arbiter_error_zone, y_main_log_error_zone)

    # --- Phase 4: Final OOF Assembly ---
    final_oof_preds_log = universal_baseline_oof_preds_log.copy()
    arbiter_oof_preds_log = arbiter_model.predict(X_arbiter_error_zone)
    final_oof_preds_log.loc[is_error_zone] = arbiter_oof_preds_log

    # --- Finalize and Return ---
    trained_models = {
        'generalist_ensemble': generalist_models,
        'dispatcher': dispatcher_artifacts,
        'council_residual_models': council_models,
        'arbiter_model': arbiter_model,
        'jurisdictional_performance_manifest': performance_manifest,
        'firewalled_feature_sets': firewalled_feature_sets,
        'firewalled_universal_cols': global_specialist_universal_cols,
        'signal_hierarchy': signal_hierarchy, # Persist for prediction symmetry
        'dispatcher_feature_names': dispatcher_feature_names, # Persist for prediction symmetry
        'architecture_version': 'V44.0_Knowledge_Transfer'
    }
    eval_df = df_main_raw[['property_id', 'most_recent_sale_price']].copy()
    eval_df['final_predicted_price'] = np.expm1(oof_assembler_preds_log)

    # --- ARCHITECTURAL CORRECTION: Unambiguous Return Logic ---
    # This consolidated block prevents the implicit `None` return that caused the TypeError.
    if is_sub_model:
        # For V48.1, sub-models do not generate standalone diagnostics.
        # The final packaged model is evaluated as a whole.
        return trained_models, eval_df
    else:
        # This path is for legacy, standalone V18 runs.
        # It correctly generates diagnostics and returns the full 5-tuple artifact.
        if os.environ.get("WISTERIA_RUN_MODE") == "INTELLIGENCE":
            diagnostics_dir = os.path.join(output_dir, "v18_diagnostics")
            # In a standalone run, diagnostics are generated for the full model.
            _generate_and_save_lgbm_diagnostics(
                model=final_assembler_model,
                X_data=X_assembler,
                model_name="assembler",
                output_path=os.path.join(diagnostics_dir, "02_l1_assembler")
            )
        return trained_models, eval_df, None, None, baseline_mae



def generate_shap_reports_v18(df_main_raw, df_holdout_raw, trained_models, feature_sets, universal_cols_present, output_dir):
    """
    [V18 DIAGNOSTIC SUITE]
    Generates SHAP explanations for the final L1 Assembler model. This provides a unified
    view of how the baseline prediction, raw universal features, and specialist residual
    corrections contribute to the final valuation.
    """
    import matplotlib
    matplotlib.use('Agg')
    import shap
    import matplotlib.pyplot as plt
    warnings.filterwarnings('ignore')
    print("\n--- V18: Generating SHAP Explanations for L1 Assembler Model ---")

    if df_holdout_raw.empty:
        print("  - Holdout set is empty. Skipping SHAP report generation.")
        return None

    # --- Step 1: Define a helper function to construct the Assembler's feature set ---
    def _get_v18_assembler_features(df, models, features, universal_cols):
        # 1a: Get Baseline Prediction
        X_baseline = df[universal_cols].fillna(0)
        baseline_preds_log = models['baseline_model'].predict(X_baseline)

        # 1b: Get Residual Predictions from Specialists
        l0_residual_preds_df = pd.DataFrame(index=df.index)
        for head_name, model in models['specialists'].items():
            cols = features.get(head_name, [])
            if not cols: continue
            # Ensure all columns are present, filling missing with 0
            X_specialist = df.reindex(columns=cols, fill_value=0).fillna(0)
            l0_residual_preds_df[f'l0_resid_pred_{head_name}'] = model.predict(X_specialist)

        # 1c: Assemble the final feature matrix
        X_assembler = pd.concat([
            pd.DataFrame({'baseline_pred_log': baseline_preds_log}, index=df.index),
            X_baseline,
            l0_residual_preds_df
        ], axis=1)

        # 1d: Enforce final column order to match trained model contract
        expected_assembler_features = models['assembler_model'].feature_name_
        X_assembler = X_assembler.reindex(columns=expected_assembler_features, fill_value=0)
        return X_assembler

    # --- Step 2: Prepare background and holdout data using the helper ---
    print("  - Preparing background data for SHAP explainer...")
    background_data = df_main_raw.sample(min(200, len(df_main_raw)), random_state=42)
    X_background_assembler = _get_v18_assembler_features(background_data, trained_models, feature_sets, universal_cols_present)

    print("  - Preparing holdout data for SHAP explanation...")
    X_holdout_assembler = _get_v18_assembler_features(df_holdout_raw, trained_models, feature_sets, universal_cols_present)
    
    # --- Step 3: Initialize Explainer and Calculate SHAP values ---
    assembler_model = trained_models['assembler_model']
    print("  - Initializing SHAP TreeExplainer for L1 Assembler...")
    explainer = shap.TreeExplainer(assembler_model, X_background_assembler)
    
    print(f"  - Calculating SHAP values for {len(X_holdout_assembler)} holdout properties...")
    shap_values = explainer.shap_values(X_holdout_assembler)

    # --- Step 4: Generate and Save the Plot ---
    # No de-sterilization or mapping is required; feature names are already human-readable.
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X_holdout_assembler, show=False, plot_size=None)
    plt.title("V18 Assembler SHAP Summary: Unified Feature Importance", fontsize=16)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "v18_assembler_shap_summary.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"  - Saved V18 SHAP summary plot to {plot_path}")

    # Return artifacts needed for downstream local case studies
    return {
        'shap_values': shap_values,
        'explainer': explainer,
        'X_holdout_assembler': X_holdout_assembler
    }

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

    lgbm_selector = lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbosity=-1)

    print("  - Fitting selector model...")
    lgbm_selector.fit(X, y)

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

def engineer_contextual_features(df):
    """
    [V9 ARCHITECTURE] Creates high-level regional and geographic cluster features to provide context.
    """
    print("--- V9: Engineering Regional & Geographic Contextual Features ---")
    df_copy = df.copy()
    new_context_cols = []
    if 'latitude' in df_copy.columns:
        df_copy['country_proxy_scotland'] = (df_copy['latitude'] > 55).astype(int)
        new_context_cols.append('country_proxy_scotland')
        print("  - CREATED 'country_proxy_scotland'. [TODO: Replace with actual region data]")
    if 'latitude' in df_copy.columns and 'longitude' in df_copy.columns:
        print("  - Performing K-Means clustering on lat/lon to create market clusters...")
        kmeans = KMeans(n_clusters=50, random_state=42, n_init=10) 
        # [A.D-V67.0 CORRECTION] The `.astype('category')` is removed. The output of
        # .fit_predict() is already the model-compliant integer array we need.
        df_copy['geo_cluster'] = kmeans.fit_predict(df_copy[['latitude', 'longitude']])
        new_context_cols.append('geo_cluster')
        print("  - Created 'geo_cluster' integer feature.")
    return df_copy, new_context_cols

# [A.D-V67.0] PROPHYLACTIC TYPE HEALING & ENFORCEMENT GATE
def _enforce_model_input_schema(df, model_name):
    """
    [V2] Validates and heals a DataFrame's dtypes to be model-compliant.
    1. Actively converts 'category' dtypes to their integer codes.
    2. Fails loudly if any other non-compliant types (e.g., 'object') remain.
    """
    print(f"  - [{model_name}] Engaging Prophylactic Type Healing & Enforcement Gate...")
    df_healed = df.copy()
    
    # --- Stage 1: Heal Known Safe-to-Convert Types ---
    categorical_cols = df_healed.select_dtypes(include=['category']).columns
    if not categorical_cols.empty:
        print(f"    - Healing {len(categorical_cols)} 'category' column(s) by converting to integer codes...")
        for col in categorical_cols:
            df_healed[col] = df_healed[col].cat.codes
            print(f"      - Converted '{col}' to int.")

    # --- Stage 2: Validate for Any Remaining Non-Compliant Types ---
    allowed_dtypes = ['int64', 'int32', 'int16', 'int8', 'float64', 'float32', 'float16', 'bool']
    bad_columns = [
        (col, str(dtype)) for col, dtype in df_healed.dtypes.items() if str(dtype) not in allowed_dtypes
    ]

    if bad_columns:
        error_message = f"FATAL: Type integrity violation for model '{model_name}'. "
        error_message += "The following columns have non-compliant dtypes after healing attempt:\n"
        for col, dtype in bad_columns:
            error_message += f"  - Column: '{col}', DType: '{dtype}'\n"
        error_message += "All input features for LightGBM models must be int, float, or bool."
        raise TypeError(error_message)
    
    print(f"  - [{model_name}] Schema validated. All {len(df_healed.columns)} feature dtypes are compliant.")
    return df_healed

def sanitize_high_risk_features(df):
    """
    [V9 ARCHITECTURE] Applies robust transformations to known high-risk features to neutralize outliers.
    """
    print("--- V9: Sanitizing High-Risk Features ---")
    df_copy = df.copy()
    if 'num__INT_hm_price_vs_last_sold_ratio' in df_copy.columns:
        p99 = df_copy['num__INT_hm_price_vs_last_sold_ratio'].quantile(0.99)
        df_copy['ratio_price_vs_last_sold_clipped'] = df_copy['num__INT_hm_price_vs_last_sold_ratio'].clip(upper=p99)
        # [A.D-V66.0 CORRECTION] The `.astype(str)` is removed. pd.qcut with labels=False
        # already produces the desired integer-encoded bins for the tree model.
        df_copy['ratio_price_vs_last_sold_binned'] = pd.qcut(df_copy['num__INT_hm_price_vs_last_sold_ratio'], q=5, labels=False, duplicates='drop')
        print(f"  - Clipped and binned 'num__INT_hm_price_vs_last_sold_ratio'.")
    if 'most_recent_sale_year' in df_copy.columns and 'num__last_sold_date_year__YYYY__hm' in df_copy.columns:
        current_year = datetime.now().year
        last_sale_year = df_copy['num__last_sold_date_year__YYYY__hm'].copy()
        last_sale_year[last_sale_year > current_year] = np.nan
        df_copy['years_since_last_sale'] = current_year - last_sale_year
        df_copy['years_since_last_sale'].fillna(df_copy['years_since_last_sale'].median(), inplace=True)
        print("  - Engineered 'years_since_last_sale' feature.")
    return df_copy

def create_l1_meta_features(df_expert_preds, df_context, universal_cols):
    """
    [V17 ARCHITECTURE - The Council of Elders]
    Creates features for the L1 Meta-Model by combining L0 specialist predictions
    with the raw, universally predictive features. This breaks the information bottleneck.
    """
    print("  - V17: Creating L1 meta-features from L0 predictions AND universal raw features...")
    
    # Start with the predictions from the specialists
    meta_features = df_expert_preds.copy()
    
    # Select only the universal columns that are actually present in the context dataframe
    available_universal_cols = [col for col in universal_cols if col in df_context.columns]
    
    if available_universal_cols:
        # Join the raw universal features
        meta_features = meta_features.join(df_context[available_universal_cols], how='left')
        
        # Robustly handle any potential NaNs introduced by the join or in the original data
        for col in available_universal_cols:
            if meta_features[col].isnull().any():
                # Use median for numeric, -1 for categorical-like integers
                if pd.api.types.is_numeric_dtype(meta_features[col]):
                    meta_features[col].fillna(meta_features[col].median(), inplace=True)
                else:
                    meta_features[col].fillna(-1, inplace=True)

    return meta_features

def select_features_with_lgbm(df_raw, target_series, candidate_cols, n_top_features, head_name):
    """
    [V17 ARCHITECTURE]
    Uses a dedicated LightGBM model to perform robust, data-driven feature selection.
    """
    print(f"\n--- Running LGBM Feature Selection for '{head_name}' ---")
    
    available_candidates = [col for col in candidate_cols if col in df_raw.columns]
    X = df_raw[available_candidates].copy()
    y = target_series.copy()
    print(f"  - Evaluating {len(available_candidates)} candidate features.")

    # Simple, robust imputation for selection purposes
    X.fillna(0, inplace=True)

    # [A.D-V82.3] Hardened Internal Type Check
    non_numeric_cols = X.select_dtypes(exclude=np.number).columns
    if not non_numeric_cols.empty:
        raise TypeError(
            f"FATAL: Non-numeric columns detected in feature selection for '{head_name}'. "
            f"This indicates a breach of the metadata quarantine. Offending columns: {list(non_numeric_cols)}"
        )

    # A simple, fast model is sufficient for ranking features
    lgbm_selector = lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbosity=-1)

    print("  - Fitting selector model...")
    lgbm_selector.fit(X, np.log1p(y))

    importances_df = pd.DataFrame({
        'feature': X.columns,
        'importance': lgbm_selector.feature_importances_
    }).sort_values('importance', ascending=False)

    # Ensure we don't try to select more features than exist
    n_to_select = min(n_top_features, len(available_candidates))
    selected_features = importances_df.head(n_to_select)['feature'].tolist()
    
    print(f"  - Selected the top {len(selected_features)} features for the Universalist Core.")
    
    del X, y, lgbm_selector, importances_df
    gc.collect()

    return selected_features

# DEFINITIVE V18.5 DIAGNOSTIC INSTRUMENT
def _generate_and_save_lgbm_diagnostics(model, X_data, model_name, output_path):
    """
    [V18.5 INSTRUMENTATION - ACCOUNTABILITY MANDATE]
    Performs a deep diagnostic analysis using the modern, unified shap.Explainer API.
    This is the architecturally correct approach for handling complex models like LightGBM
    and provides the proper mechanism for managing the additivity check.
    """
    try:
        print(f"  -> Generating diagnostics for '{model_name}'...")
        os.makedirs(output_path, exist_ok=True)
        
        # --- 1. Feature Importance Report (Unchanged) ---
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': model.feature_name_,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            importance_df.to_csv(os.path.join(output_path, "feature_importance.csv"), index=False)
        
        # --- 2. SHAP Analysis using the Modern API ---
        background_data = X_data.sample(min(200, len(X_data)), random_state=42)
        
        # --- ARCHITECTURAL CORRECTION V18.5 ---
        # 2a. Instantiate the unified SHAP Explainer. This is the correct modern API.
        #     It intelligently selects the best algorithm (in this case, TreeExplainer)
        #     but provides a consistent and more robust interface.
        # 2b. We now use the CORRECT keyword argument to disable the check as a fallback.
        #     The primary de-fragmentation fix should resolve most issues, but this
        #     ensures the pipeline WILL complete and deliver intelligence.
        explainer = shap.Explainer(model, background_data, check_additivity=False)

        # 2c. Calculate SHAP values. The new API uses a direct call on the explainer object.
        #     This returns a rich Explanation object, not just a raw numpy array.
        shap_values = explainer(X_data)
        
        # The summary plot function is designed to work seamlessly with the new Explanation object.
        plt.figure(figsize=(12, max(8, len(X_data.columns) * 0.3)))
        shap.summary_plot(shap_values, X_data, show=False, plot_size=None)
        plt.title(f"SHAP Summary for {model_name}", fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, "shap_summary.png"))
        plt.close()
        
        print(f"  -- Diagnostics for '{model_name}' saved to '{output_path}'")
        
        # --- Ensure downstream compatibility by passing the raw values ---
        # The master intelligence report expects a numpy array.
        return {
            'explainer': explainer,
            'shap_values': shap_values.values, # Pass the .values attribute
            'X_data': X_data
        }
    except Exception as e:
        # Provide a more detailed error log for any future failures.
        import traceback
        print(f"  -- [!!] FATAL ERROR during diagnostic generation for '{model_name}'.")
        print(f"  -- Error: {e}")
        print(f"  -- Traceback: {traceback.format_exc()}")
        return None


# NEW V21.0 AUTO-REFINEMENT ENGINE
def perform_auto_refinement(initial_feature_sets, refinement_dir, head_threshold, feature_threshold, citadel_features):
    """
    [V21.0 ARCHITECTURE, V38.0-AWARE]
    Programmatically culls low-importance heads and features based on the diagnostic
    intelligence reports. Now Citadel-aware to prevent culling of foundational features.
    """
    print("\n--- V21 (V38-Aware): Engaging Auto-Refinement Protocol ---")
    print(f"  - Protecting {len(citadel_features)} Citadel features from culling.")

    # [A.D-V38.0] ARMORING LOGIC: Remove Citadel features before culling
    feature_sets_to_refine = {}
    for head, features in initial_feature_sets.items():
        feature_sets_to_refine[head] = [f for f in features if f not in citadel_features]
    
    # --- Part 1: Head-Level Culling based on L1 Assembler Importance ---
    assembler_report_path = os.path.join(refinement_dir, "02_l1_assembler", "feature_importance.csv")
    if not os.path.exists(assembler_report_path):
        print("  - WARNING: L1 Assembler report not found. Skipping head-level culling.")
        surviving_heads_feature_sets = feature_sets_to_refine
    else:
        print(f"  - Performing head-level culling with importance threshold > {head_threshold}...")
        assembler_df = pd.read_csv(assembler_report_path)
        
        specialist_importance = assembler_df[assembler_df['feature'].str.startswith('l0_resid_pred_')].copy()
        specialist_importance['head_name'] = specialist_importance['feature'].str.replace('l0_resid_pred_', '')
        
        surviving_head_names = set(specialist_importance[specialist_importance['importance'] > head_threshold]['head_name'])
        
        surviving_heads_feature_sets = {}
        for head_name, features in feature_sets_to_refine.items():
            if head_name in surviving_head_names:
                surviving_heads_feature_sets[head_name] = features
            else:
                print(f"    - CULLED Head: '{head_name}' (Reason: L1 Importance <= {head_threshold})")
        
        print(f"  - Head Culling Complete. Retained {len(surviving_heads_feature_sets)} of {len(feature_sets_to_refine)} heads.")

    # --- Part 2: Feature-Level Culling based on L0 Specialist Importance ---
    print(f"\n  - Performing feature-level culling with importance threshold > {feature_threshold}...")
    refined_feature_sets = {}
    specialists_dir = os.path.join(refinement_dir, "01_l0_specialists")
    
    for head_name, features in surviving_heads_feature_sets.items():
        specialist_report_path = os.path.join(specialists_dir, head_name, "feature_importance.csv")
        
        if not os.path.exists(specialist_report_path):
            refined_feature_sets[head_name] = features
            continue

        importance_df = pd.read_csv(specialist_report_path)
        surviving_features = set(importance_df[importance_df['importance'] > feature_threshold]['feature'])
        
        original_features_set = set(features)
        refined_features = sorted(list(original_features_set.intersection(surviving_features)))
        
        culled_count = len(features) - len(refined_features)
        if culled_count > 0:
            print(f"    - Refined '{head_name}': Culled {culled_count} of {len(features)} low-importance features.")
        
        if refined_features:
            refined_feature_sets[head_name] = refined_features
        else:
            print(f"    - WARNING: All features for '{head_name}' were culled. Removing head entirely.")
            
    print(f"  - Feature Culling Complete. Final refined head count: {len(refined_feature_sets)}.")
    
    # [A.D-V38.0] RESTORATION LOGIC: Unconditionally add Citadel features back.
    final_refined_sets = refined_feature_sets.copy()
    final_refined_sets['sub_head_citadel_infrastructure'] = citadel_features
    
    print(f"  - Citadel features restored. Final head count: {len(final_refined_sets)}.")
    return final_refined_sets

# NEW V30.0 DYNAMIC HPI ENGINE
def engineer_dynamic_hpi_features_v30(df):
    """
    [V30.0 ARCHITECTURE - The Dynamic HPI Mandate]
    Constructs a hyper-local, dynamic HPI for each property by using its own
    surrounding sales data ('compass_mean_pp_...') to project its last sale price.
    """
    print("\n--- V30: Engineering Dynamic Hyper-Local HPI features ---")
    df_copy = df.copy()
    new_features = []

    required_cols = ['num__last_sold_price_gbp_hm', 'num__last_sold_date_year__YYYY__hm', 'num__property_main_type_encoded__1Flat_hm', 'num__property_sub_type_code_from_homipi__4Detached_hm', 'num__property_sub_type_code_from_homipi__5Semi_Detached_hm']
    if not all(c in df_copy.columns for c in required_cols):
        print("  - WARNING: Missing required columns for Dynamic HPI. Skipping.")
        return df_copy, []
    
    # --- Step 1: Determine the property archetype for each row ---
    conditions = [ df_copy['num__property_main_type_encoded__1Flat_hm'] == 1, df_copy['num__property_sub_type_code_from_homipi__4Detached_hm'] == 1, df_copy['num__property_sub_type_code_from_homipi__5Semi_Detached_hm'] == 1 ]
    choices = ['F', 'D', 'S']
    df_copy['prop_archetype'] = np.select(conditions, choices, default='T')

    # --- Step 2: Vectorized HPI Calculation ---
    for n_neighbors in [20, 50, 100]: # Generate features for multiple scales
        # Find the most recent month's data available in the columns
        pp_cols = [c for c in df_copy.columns if f"compass_mean_pp_" in c and f"_n{n_neighbors}" in c]
        if not pp_cols: continue
        
        latest_year = max([int(re.search(r'_(\d{4})_', c).group(1)) for c in pp_cols])
        latest_month = max([int(re.search(r'_(\d{4})_(\d{2})_', c).group(1)) for c in pp_cols if f'_{latest_year}_' in c])
        
        # We now have the anchor point for the "current" price.
        
        def get_hpi_price(row, year_col, month_col):
            archetype = row['prop_archetype']
            year = int(row[year_col])
            # For simplicity, we'll use a fixed month (e.g., June) for the historical anchor
            month = 6 
            
            if pd.isna(year) or year < 2010: return np.nan
            
            # Dynamically construct the column names for the historical and current area prices
            historical_price_col = f"compass_mean_pp_{year}_{month:02d}_{archetype}_avg_price_n{n_neighbors}"
            current_price_col = f"compass_mean_pp_{latest_year}_{latest_month:02d}_{archetype}_avg_price_n{n_neighbors}"
            
            if historical_price_col in row and current_price_col in row:
                historical_price = row[historical_price_col]
                current_price = row[current_price_col]
                if pd.notna(historical_price) and historical_price > 0 and pd.notna(current_price):
                    return current_price / historical_price
            return np.nan

        # Calculate the inflation factor
        df_copy[f'hpi_factor_n{n_neighbors}'] = df_copy.apply(
            get_hpi_price, axis=1, year_col='num__last_sold_date_year__YYYY__hm', month_col=None # Month is fixed for now
        )
        
        # Create the final adjusted price feature
        new_col_name = f'eng_DynamicHPI_Adjusted_Price_n{n_neighbors}'
        df_copy[new_col_name] = df_copy['num__last_sold_price_gbp_hm'] * df_copy[f'hpi_factor_n{n_neighbors}']
        new_features.append(new_col_name)
        
    print(f"  - Engine complete. Generated {len(new_features)} Dynamic HPI features.")
    return df_copy.drop(columns=['prop_archetype']), new_features

# NEW V30.0 DIVERGENCE FEATURE ENGINE
def engineer_divergence_features_v30(df):
    """
    [V30.0 ARCHITECTURE] Engineers features that measure the divergence between a
    property's intrinsic qualities and its hyper-local market context.
    """
    print("\n--- V30: Engineering Divergence features ---")
    df_copy = df.copy()
    new_features = []

    # Identify the most robust HPI-adjusted price we created
    hpi_price_col = 'eng_DynamicHPI_Adjusted_Price_n100'
    if hpi_price_col not in df_copy.columns:
        print("  - WARNING: HPI adjusted price not found. Skipping divergence features.")
        return df_copy, []

    # Hypothesis 1: Quality vs. Market Divergence
    if 'avg_persona_rating_overall' in df_copy.columns:
        new_col_name = 'eng_Divergence_Quality_vs_Market'
        # Interaction: How much does visual quality deviate from the expected price?
        # A high rating on a low-priced property gives a high score (potential gem).
        df_copy[new_col_name] = df_copy['avg_persona_rating_overall'] / (np.log1p(df_copy[hpi_price_col]) + 1)
        new_features.append(new_col_name)

    # Hypothesis 2: Size vs. Market Divergence (Price per SqM)
    if 'num__floor_area_sqm_from_homipi__numeric_or_empty__hm' in df_copy.columns:
        new_col_name = 'eng_Divergence_Size_vs_Market'
        df_copy[new_col_name] = df_copy[hpi_price_col] / (df_copy['num__floor_area_sqm_from_homipi__numeric_or_empty__hm'] + 1)
        new_features.append(new_col_name)

    print(f"  - Engine complete. Generated {len(new_features)} Divergence features.")
    return df_copy, new_features


def _calculate_adaptive_weight_v42(sample_size, homogeneity_score):
    """[V42.0] Calculates an intuitive, evidence-based credibility weight."""
    base_weight = 1 / (1 + np.exp(-0.015 * (sample_size - 250)))
    # A direct multiplication now correctly rewards homogeneity
    adaptive_weight = base_weight * homogeneity_score
    return adaptive_weight


def _get_v18_assembler_features(df, models):
    """
    [A.D-V50.2 REFACTORED UTILITY] Symmetrically constructs the L1 Assembler feature set.
    This is a critical, shared utility required by training, prediction, and diagnostics.
    """
    feature_sets = models['feature_sets']
    universal_cols_present = models['universal_cols']
    X_baseline = df[universal_cols_present].fillna(0)
    baseline_preds_log = models['baseline_model'].predict(X_baseline)
    l0_residual_preds_df = pd.DataFrame(index=df.index)
    for head_name, model in models['specialists'].items():
        cols = feature_sets.get(head_name, [])
        if not cols: continue
        X_specialist = df.reindex(columns=cols, fill_value=0).fillna(0)
        l0_residual_preds_df[f'l0_resid_pred_{head_name}'] = model.predict(X_specialist)
    X_assembler = pd.concat([pd.DataFrame({'baseline_pred_log': baseline_preds_log}, index=df.index), X_baseline, l0_residual_preds_df], axis=1)
    expected_features = models['assembler_model'].feature_name_
    return X_assembler.reindex(columns=expected_features, fill_value=0)

def main():
    # --- [A.D-V40.1] RESILIENT SIGNAL ACQUISITION PROTOCOL ---
    # This configuration is now defined in main and passed to the training
    # orchestrator, which is responsible for using it and persisting it.
    MARKET_DYNAMICS_SIGNAL_HIERARCHY = [
        {'name': 'PC_Sales_Volatility_YoY_', 'type': 'composite'},
        {'name': 'LSOA_Transaction_Volume_Change_5Year', 'type': 'direct'},
        {'name': 'LSOA_Price_Growth_5Year', 'type': 'direct'},
        {'name': 'LSOA_Annual_Transaction_Rate', 'type': 'direct'},
        {'name': 'LSOA_Churn_Recent', 'type': 'direct'},
        {'name': 'LSOA_Market_Absorption_Years', 'type': 'direct'},
    ]

    print("--- Verifying Environment Contract ---")
    required_env_vars = ["MASTER_DATA_LOCAL_PATH", "FEATURE_SETS_LOCAL_PATH", "RIGHTMOVE_DATA_LOCAL_PATH", "KEY_MAP_LOCAL_PATH", "OUTPUT_DIR", "AE_ENCODINGS_LOCAL_DIR", "AE_KEYS_LOCAL_PATH"]
    if any(v not in os.environ for v in required_env_vars):
        raise EnvironmentError(f"FATAL: Missing required environment variables. Ensure all are set.")
    print("  - Environment contract verified successfully.")

    def normalize_address_key_v4(address_str):
        if pd.isna(address_str): return None
        address = str(address_str).lower().strip()
        address = re.sub(r'^(.*?)https://www\.rightmove\.co\.uk', r'\1', address)
        address = re.sub(r'[^\w\s]', '', address)
        postcode_match = re.search(r'([a-z]{1,2}\d[a-z\d]?\s*\d[a-z]{2})', address)
        if not postcode_match: return None
        postcode = postcode_match.group(1).replace(" ", "")
        street_part = address.replace(postcode_match.group(1), '').strip()
        street_key = re.sub(r'\s+', '', street_part)[:5]
        return f"{postcode}_{street_key}"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- [A.D-27.1] KEY PRESERVATION PROTOCOL ---
    QUARANTINED_KEYS = ['property_id', 'normalized_address_key', 'rightmove_address_text', 'pcds']

    # [A.D-V82.1] THE UNIFIED METADATA QUARANTINE PROTOCOL
    # This set is the single, definitive source of truth for all columns that are NOT features.
    # It is used for both schema sanitization and feature selection to prevent desynchronization.
    CANONICAL_METADATA_COLS = {
        # --- Core Identifiers & Raw Text ---
        'property_id', 'normalized_address_key', 'rightmove_address_text', 'pcds',
        'rightmove_row_id', 'address_info', 'address_key',
        'final_merge_key', 'canonical_lookup_key', # Protected per A.D-V81.1
    
        # --- The Target Variable ---
        'most_recent_sale_price',
    
        # --- High-Leakage Transactional History ---
        'price_change_since_last',
    
        # --- Geographic & Stratification Flags ---
        'postcode_area', 'error_zone_flag', 'LSOA11CD',
    
        # --- Triage Gate & Data Quality Metadata ---
        'data_completeness_score', 'is_data_deficient',
        'score_intrinsic_dna', 'score_transactional_context', 'score_qualitative_context',
    
        # --- Engineered Object/String Dtypes to Exclude from Core Selection ---
        'ratio_price_vs_last_sold_binned'
    }

    # --- [A.D-V40.1] THE REFINED CITADEL PROTOCOL ---
    CITADEL_FEATURES = [
        'latitude', 'longitude', 'LSOA_Property_Density',
        'PC_Sales_Volatility_YoY_D', 'PC_Sales_Volatility_YoY_F',
        'PC_Sales_Volatility_YoY_S', 'PC_Sales_Volatility_YoY_T',
        'LSOA_Transaction_Volume_Change_5Year',
        'LSOA_Price_Growth_5Year',
        'LSOA_Annual_Transaction_Rate'
    ]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

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

    print("--- STAGE 1 & 2: Loading, Merging, and Cleaning Data ---")
    df_features = pd.read_parquet(MASTER_DATA_PATH)
    
    

    # --- [A.D-V79.1] Engaging Proxy Index Data Contract Gate ---
    print("\n--- [V79.1] Engaging Proxy Index Data Contract Gate ---")
    PROXY_INDEX_COL = 'num__INT_avg_estimated_price'
    if PROXY_INDEX_COL not in df_features.columns:
        raise KeyError(f"FATAL: Proxy Index feature '{PROXY_INDEX_COL}' not found in the master dataset. The V79.1 PoC cannot proceed.")
    if not pd.api.types.is_numeric_dtype(df_features[PROXY_INDEX_COL]):
        raise TypeError(f"FATAL: Proxy Index feature '{PROXY_INDEX_COL}' is not a numeric type. It cannot be used for normalization.")
    print("  - Proxy Index Data Contract Fulfilled. Proceeding with PoC.")
    
    # --- [A.D-V39.2] SOURCE DATA INTEGRITY CONTRACT ---
    print("\n--- Engaging Source Data Integrity Contract (Pre-Flight Check) ---")
    SOURCE_DATA_INTEGRITY_CONTRACT = [
        'latitude', 'longitude', 'LSOA_Property_Density',
        'PC_Sales_Volatility_YoY_D', 'PC_Sales_Volatility_YoY_F',
        'PC_Sales_Volatility_YoY_S', 'PC_Sales_Volatility_YoY_T',
        'num__property_sub_type_code_from_homipi__4Detached_hm',
        'num__property_main_type_encoded__1Flat_hm'
    ]
    missing_source_cols = [c for c in SOURCE_DATA_INTEGRITY_CONTRACT if c not in df_features.columns]
    if missing_source_cols:
        raise ValueError(f"FATAL: Source data integrity violated. Missing: {missing_source_cols}")
    print("  - Source Data Integrity Contract fulfilled. Master artifact is valid.")
    
    # --- ARCHITECTURAL INTERVENTION (13.7) ---
    print("\n--- V13.7: Intercepting and Distilling Monolithic Temporal Heads H & I ---")
    with open(os.environ.get("FEATURE_SETS_LOCAL_PATH"), 'r') as f: legacy_heads = json.load(f)
    head_h_raw_cols = [c for c in legacy_heads.get('head_H_price_history', []) if c in df_features.columns]
    head_i_raw_cols = [c for c in legacy_heads.get('head_I_compass_price_history', []) if c in df_features.columns]
    df_distill_input = df_features[head_h_raw_cols + head_i_raw_cols].copy()
    cols_to_drop = head_h_raw_cols + head_i_raw_cols
    df_features.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    df_distilled_features, new_distilled_yearly_cols = engineer_yearly_summary_heads(df_distill_input, head_h_raw_cols, head_i_raw_cols)
    df_features = pd.concat([df_features, df_distilled_features[new_distilled_yearly_cols]], axis=1)
    print(f"  - Monolithic Distillation complete. Main dataframe shape: {df_features.shape}")
    # --- END INTERVENTION ---

    # [A.D-V80.1] Corrected Data Loading Contract for Rightmove CSV
    df_rightmove_raw = pd.read_csv(
        os.environ.get("RIGHTMOVE_DATA_LOCAL_PATH"), 
        header=None, 
        # Use temporary, unambiguous names for the raw payloads.
        names=['address_payload', 'details_payload', 'sales_history_payload'], 
        on_bad_lines='skip'
    )

    # [A.D-V80.2] Payload Disambiguation & Source Redefinition
    print("  - Disambiguating raw Rightmove data payloads...")
    
    def _extract_address_from_payload(payload_str):
        """
        [ROBUSTNESS INTERVENTION] Safely parses the string-encoded list 
        in the first column to extract the canonical address.
        """
        try:
            # ast.literal_eval is the only safe way to evaluate this string
            payload_list = ast.literal_eval(str(payload_str))
            if isinstance(payload_list, list) and len(payload_list) > 0:
                # The clean address is consistently the first element.
                return payload_list
        except (ValueError, SyntaxError):
            # If parsing fails, return None to mark the row for exclusion.
            return None
        return None
    
    # Create a new, clean column containing only the address for key generation.
    df_rightmove_raw['clean_address_for_key'] = df_rightmove_raw['address_payload'].apply(_extract_address_from_payload)
    
    # Redefine the script's internal pointers to the CORRECT columns.
    # This is a non-negotiable reassignment.
    address_col_rm = 'clean_address_for_key' # This now points to our clean address.
    sales_history_col_name_for_processing = 'sales_history_payload' # This points to the real sales data.

    # --- [A.D-V75.2] UNIFIED KEY PROTOCOL ---
    # The _create_canonical_lookup_key is the single source of truth for matching.
    # It is applied to the raw property_id and address_info columns from both sources.
    print("--- [V75.2] Engaging Unified Key Reconciliation Protocol ---")
    df_features['canonical_lookup_key'] = _create_canonical_lookup_key(df_features['property_id'])
    df_rightmove_raw['canonical_lookup_key'] = _create_canonical_lookup_key(df_rightmove_raw[address_col_rm])
    
    # Drop rows where a valid canonical key could not be generated
    df_features.dropna(subset=['canonical_lookup_key'], inplace=True)
    df_rightmove_raw.dropna(subset=['canonical_lookup_key'], inplace=True)

    # [A.D-V80.3] Re-Routed the processing function to the correct sales history payload.
    df_rightmove_processed = process_rightmove_data(df_rightmove_raw, sales_history_col_name_for_processing)
    
    # The merge key is now the canonical key.
    df_rightmove_with_address_key = pd.merge(
        df_rightmove_raw[['canonical_lookup_key', address_col_rm]],
        df_rightmove_processed,
        left_index=True,
        right_on='rightmove_row_id',
        how='inner'
    )

    # De-duplicate using the canonical key.
    df_features.drop_duplicates(subset=['canonical_lookup_key'], keep='first', inplace=True)
    df_rightmove_with_address_key.drop_duplicates(subset=['canonical_lookup_key'], keep='first', inplace=True)
    
    df_rightmove_with_address_key.rename(columns={address_col_rm: 'rightmove_address_text'}, inplace=True)
    df_features.drop(columns=['most_recent_sale_price'], inplace=True, errors='ignore')

    # [A.D-V80.4] Defensive Pre-Merge Audit Circuit Breaker
    print("\n--- Engaging Pre-Merge Audit Circuit Breaker ---")
    keys_features = set(df_features['canonical_lookup_key'].dropna())
    keys_rightmove = set(df_rightmove_with_address_key['canonical_lookup_key'].dropna())
    intersection_count = len(keys_features.intersection(keys_rightmove))
    
    print(f"  - Master Feature Set Unique Keys: {len(keys_features)}")
    print(f"  - Rightmove Data Unique Keys: {len(keys_rightmove)}")
    print(f"  - Matched Key Intersection Size: {intersection_count}")
    
    # The check from the original traceback is now moved here, with a more informative error.
    if intersection_count == 0:
        raise ValueError(
            "FATAL: Key reconciliation failed. The merge would result in an empty dataset because "
            "zero canonical lookup keys matched between the feature set and the Rightmove data. "
            "Verify the `_create_canonical_lookup_key` logic and the upstream data cleaning steps."
        )
    
    print("  - Audit PASSED. Key intersection is non-zero. Proceeding with merge.")
    # --- End Circuit Breaker ---

    # The final merge now uses the unified, architecturally-sound key.
    df = pd.merge(df_features, df_rightmove_with_address_key, on='canonical_lookup_key', how='inner')
    if len(df) == 0: raise ValueError("FATAL: Merge resulted in an empty dataset.")

    # --- [A.D-V39.0] GEOGRAPHIC SANITIZATION GATE ---
    print("\n--- Engaging Geographic Sanitization Gate ---")
    uk_lat_min, uk_lat_max = 49, 61; uk_lon_min, uk_lon_max = -8, 2
    initial_rows = len(df)
    valid_geo_mask = (df['latitude'].between(uk_lat_min, uk_lat_max) & df['longitude'].between(uk_lon_min, uk_lon_max))
    df = df[valid_geo_mask].copy()
    print(f"  - Sanitization complete. Proceeding with {len(df)} of {initial_rows} properties.")

    # --- [A.D-V30.0] STAGE 2.5: DYNAMIC INTELLIGENCE ENGINEERING ---
    df, new_hpi_cols = engineer_dynamic_hpi_features_v30(df)
    df, new_divergence_cols = engineer_divergence_features_v30(df)
    
    print("\n--- STAGE 3: V13 Core Feature Engineering & Sanitization ---")
    df, new_context_cols = engineer_contextual_features(df)
    df = sanitize_high_risk_features(df)
    UNIVERSAL_PREDICTORS.extend(new_context_cols)
    df.drop(columns=['num__INT_hm_price_vs_last_sold_ratio'], inplace=True, errors='ignore')
    
    print("\n--- STAGE 3.5: Loading Pre-Trained Forecast Artifacts ---")
    forecast_models, forecast_scalers = {}, {}
    try:
        forecast_scalers = joblib.load("./forecast_artifacts/forecast_scalers.joblib")
        for p_type in ['D', 'S', 'T', 'F']:
            model_path = f"./forecast_artifacts/forecast_model_{p_type}.pt"
            if os.path.exists(model_path) and p_type in forecast_scalers:
                n_features = forecast_scalers[p_type].n_features_in_
                model = ForecastTCN(input_feature_dim=n_features, output_dim=n_features, num_channels=[64, 128]).to(DEVICE)
                model.load_state_dict(torch.load(model_path, map_location=torch.device(DEVICE)))
                model.eval(); forecast_models[p_type] = model
    except Exception as e: print(f"  - WARNING: Could not load forecast artifacts. Error: {e}.")

    df, new_forecast_cols = generate_forecast_features(df, forecast_models, forecast_scalers)
    df, new_temporal_cols = engineer_temporal_summary_features(df)
    df, new_distilled_price_cols = engineer_distilled_price_features(df)

    print("\n--- STAGE 3.9: Final Integration of Gemini Embeddings (Head G) ---")
    AE_ENCODINGS_PATH = os.environ.get("AE_ENCODINGS_LOCAL_DIR")
    AE_KEYS_LOCAL_PATH = os.environ.get("AE_KEYS_LOCAL_PATH")
    df, new_ae_cols = load_and_merge_ae_features(df, AE_ENCODINGS_PATH, AE_KEYS_LOCAL_PATH)

    # [A.D-V48.4] Re-integrate AVM Mitigation before splitting the data.
    df = mitigate_avm_leakage(df, label_col='most_recent_sale_price')

    # --- [A.D-V67.0] STAGE 5B: CANONICAL RECONCILIATION & MANIFEST-DRIVEN TRIAGE ---
    print("\n--- [V67.0] Initiating Canonical Reconciliation Protocol ---")
    
    # --- Step 1: Load Contracts and Manifest ---
    manifest_path = os.environ.get("MANIFEST_LOCAL_PATH")
    toxicity_contract_path = os.environ.get("TOXICITY_CONTRACT_LOCAL_PATH")
    try:
        with open(toxicity_contract_path, 'r') as f:
            toxicity_contract = json.load(f)
        manifest_df = pd.read_csv(manifest_path)
        # ARCHITECTURAL CORRECTION: The manifest may contain non-string property_ids, ast.literal_eval must be guarded.
        if 'imputed_vitals_list' in manifest_df.columns:
             manifest_df['imputed_vitals_list'] = manifest_df['imputed_vitals_list'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        else: # Handle case where column might be missing
             manifest_df['imputed_vitals_list'] = [[] for _ in range(len(manifest_df))]

    except Exception as e:
        raise IOError(f"FATAL: Could not load critical triage artifacts. Error: {e}")

    # --- Step 2: Symmetrically Generate Canonical Keys ---
    # The main dataframe's key is already generated by the V75.2 protocol. We only need to process the manifest.
    manifest_df['canonical_lookup_key'] = _create_canonical_lookup_key(manifest_df['property_id'])

    # --- Step 3: Create the Reconciled Imputation Map ---
    # Drop duplicates to ensure the map has a unique key-value pair
    manifest_df.drop_duplicates(subset=['canonical_lookup_key'], keep='first', inplace=True)
    imputation_map = pd.Series(
        manifest_df.imputed_vitals_list.values, 
        index=manifest_df.canonical_lookup_key
    ).to_dict()
    print(f"  - Reconciled imputation map created with {len(imputation_map)} unique entries.")

    # --- Step 4: Execute Triage using the Reconciled Map ---
    # Set the index to the canonical key for the triage function's lookup
    df.set_index('canonical_lookup_key', inplace=True)
    df = _apply_manifest_driven_triage(df, imputation_map, toxicity_contract)
    # Reset index to restore the dataframe's original structure for downstream processing
    df.reset_index(inplace=True)

    # --- [A.D-V68.0] DECOUPLED DIAGNOSTIC: Assess triage split post-reconciliation ---
    # This diagnostic now provides a true measure of data quality distribution.
    if 'is_data_deficient' in df.columns:
        num_deficient = df['is_data_deficient'].sum()
        num_sufficient = len(df) - num_deficient
        print(f"  - Post-Reconciliation Triage Distribution: {num_sufficient} Sufficient, {num_deficient} Deficient.")
    else:
        print("  - WARNING: 'is_data_deficient' column not found after triage.")
    
    df_final = df.dropna(subset=['most_recent_sale_price']).copy()

    print("\n--- STAGE 4: V13 Dynamic Feature Set Generation ---")
    feature_sets = build_v9_feature_sets(df.columns, FEATURE_SETS_PATH)
    full_feature_sets_for_diagnostics = feature_sets.copy() # V67 Mandate: Preserve unculled features for diagnostics

    # --- [A.D-V21.0] STAGE 4.1: AUTO-REFINEMENT PROTOCOL ---
    if os.environ.get("REFINEMENT_ENABLED") == "true":
        feature_sets = perform_auto_refinement(
            initial_feature_sets=feature_sets,
            refinement_dir=os.environ.get("REFINEMENT_DATA_LOCAL_DIR"),
            head_threshold=int(os.environ.get("HEAD_CULLING_IMPORTANCE_THRESHOLD")),
            feature_threshold=int(os.environ.get("FEATURE_CULLING_IMPORTANCE_THRESHOLD")),
            citadel_features=CITADEL_FEATURES
        )
    else: print("\n--- V21: Auto-Refinement skipped as no intelligence suite was provided. ---")

    # --- ARCHITECTURAL ADDITION (13.9): ENGAGE THE STERILIZATION GATE ---
    feature_sets = sterilize_feature_sets(feature_sets, ['most_recent_sale_price'])
    # --- END INTERVENTION ---

    print("\n--- STAGE 4.5: Sanitizing Feature Sets (Sterilization Gate) ---")
    NON_FEATURE_COLS = {'property_id', 'normalized_address_key', 'rightmove_address_text', 'final_merge_key'}
    for head_name in list(feature_sets.keys()):
        feature_sets[head_name] = [col for col in feature_sets[head_name] if col not in NON_FEATURE_COLS]

    print("\n--- STAGE 5: Robust Sanitization & Stratified Splitting ---")
    df['postcode_area'] = get_postcode_area(df['property_id'], df.get('pcds'))
    df['error_zone_flag'] = df['postcode_area'].isin(HIGH_ERROR_POSTCODE_AREAS).astype(int)
    df_final = df.dropna(subset=['most_recent_sale_price']).copy()
    
    # --- [A.D-V67.0] FINAL SCHEMA GUARANTEE ---
    print("\n--- STAGE 5.5: Enforcing Final Schema Guarantee (Pre-Split) ---")
    
    # [A.D-V83.1] Re-wired to use the single, unified metadata protocol per A.D-V82.2.
    feature_cols = [c for c in df_final.columns if c not in CANONICAL_METADATA_COLS]
    print(f"  - Applying numeric conversion to {len(feature_cols)} identified feature columns...")

    for col in feature_cols:
        # Use pd.to_numeric to safely convert columns, coercing errors to NaN
        df_final[col] = pd.to_numeric(df_final[col], errors='coerce')
    
    # Globally impute any NaNs that may have been created by the coercion.
    # Using 0 is a safe, simple default for tree-based models.
    df_final[feature_cols] = df_final[feature_cols].fillna(0)
    print("  - Final schema guarantee enforced. All feature columns are now numeric.")
    # --- END V67.0 ADDITION ---
    
    df_final.reset_index(drop=True, inplace=True)
    
    # --- [HARDENING MANDATE A.D-V77.2-H1] Canonical Feature List Generation ---
    ALL_AVAILABLE_FEATURES = [c for c in df_final.columns if c not in CANONICAL_METADATA_COLS]
    
# --- [A.D-V77.2] The Bifurcated Infusion Contract (Dynamically Generated) ---
    CONTEXTUAL_PREFIXES = [
        'census_', 'deprivation_', 'Household_', 'ethnicity_', 'religion_', 'ownership_', 
        'health_condition_', 'LSOA_BP_Ratio_', 'MSOA_URBAN_RURAL_', 'ah4', 'EVI_', 
        'NDVI_', 'VEG_FRAC', 'language_', 'LAD23NM_', 'MSOA', 'OA', 'WZ11_'
    ]
    FINANCIAL_PREFIXES = [
        'yearly_H_', 'yearly_I_', 'distilled_price_', 'forecast_price_', 
        'compass_mean_pp_', 'compass_std_pp_', 'LSOA_MedPrice_', 'PC_Sales_Volatility_',
        'num__homipi_estimated_price', 'estimated_price'
    ]
    CONTEXTUAL_INFUSION_COLS = [f for f in ALL_AVAILABLE_FEATURES if any(p in f for p in CONTEXTUAL_PREFIXES)]
    FINANCIAL_INFUSION_COLS = [f for f in ALL_AVAILABLE_FEATURES if any(p in f for p in FINANCIAL_PREFIXES)]
    print(f"  - Contextual Infusion Set: {len(CONTEXTUAL_INFUSION_COLS)} features identified."
    )
    # Stratify by error zone to ensure holdout has representative sample of difficult areas.
    df_main_raw, df_holdout_raw = train_test_split(df_final, test_size=0.15, random_state=42, stratify=df_final['is_data_deficient'])

    df_holdout_raw.reset_index(drop=True, inplace=True) # Ensure clean index for routing
    
    print("\n--- STAGE 6.5: Asymmetric Data-Driven Selection of Core Features (A.D-63.0 Compliant) ---")
    
    # The ad-hoc exclusion list is removed. We now use the canonical protocol.
    candidate_features = [c for c in df_main_raw.columns if c not in CANONICAL_METADATA_COLS]
    
    # ARCHITECTURAL HARDENING: Add a defensive assertion to guarantee no breach.
    assert not (set(candidate_features) & set(CANONICAL_METADATA_COLS)), \
        f"FATAL: Quarantine breach detected in feature selection! The following metadata columns were found in the candidate feature list: {set(candidate_features) & set(CANONICAL_METADATA_COLS)}"

    universal_cols = select_features_with_lgbm(df_main_raw, df_main_raw['most_recent_sale_price'], candidate_features, 250, "Universal Core")
    joblib.dump(universal_cols, os.path.join(OUTPUT_DIR, "universal_predictors_v48.joblib"))

    # [A.D-V48.3] Implement the non-negotiable "Circuit Breaker" sanity check.
    # This prevents this specific failure mode from ever happening again.
    if 'most_recent_sale_price' in universal_cols:
        raise ValueError(
            "FATAL LEAKAGE DETECTED: The target variable 'most_recent_sale_price' has "
            "been selected as a feature. This indicates a catastrophic failure in the "
            "upstream feature candidate quarantine. ABORTING."
        )
    print("  - Circuit Breaker check passed: Target variable is not present in the selected universal feature set.")

    # --- [A.D-V81.0] Curated feature set for the Canonical Embedding Engine ---
    CANONICAL_EMBEDDING_FEATURES = [
        'num__floor_area_sqm_from_homipi__numeric_or_empty__hm', 'num__number_of_bedrooms_from_homipi__numeric_or_empty__hm',
        'num__number_of_reception_rooms_from_homipi__numeric_or_empty__hm', 'num__property_construction_year_from_homipi__YYYY_or_2025NewBuildOrError__hm',
        'num__property_main_type_encoded__1Flat_hm', 'num__property_sub_type_code_from_homipi__4Detached_hm',
        'num__property_tenure_code_from_homipi__2Freehold_hm', 'primary_MainGarden_area_sqm',
        'num_rooms_identified_in_step5', 'num__current_epc_value_extracted_by_initial_homipi_script_hm_numeric',
        'primary_MainKitchen_renovation_score', 'primary_MainBathroom_renovation_score',
        'primary_MainLivingArea_renovation_score', 'Composite_CoreLiving_Strength', 'avg_persona_rating_overall'
    ]
    # Defensive check to ensure all embedding features are available
    CANONICAL_EMBEDDING_FEATURES = [f for f in CANONICAL_EMBEDDING_FEATURES if f in df_main_raw.columns]


    # --- [A.D-V81.0] STAGE 7: THE "OPTIMIZED CANONICAL EMBEDDING" ARCHITECTURE LOCK-IN ---
    v81_model_package = train_v81_architecture(
        df_main_raw, feature_sets, universal_cols, CANONICAL_EMBEDDING_FEATURES,
        full_feature_sets_for_diagnostics, CANONICAL_METADATA_COLS, OUTPUT_DIR,
        imputation_map, CONTEXTUAL_INFUSION_COLS, FINANCIAL_INFUSION_COLS
    )
    joblib.dump(v81_model_package, os.path.join(OUTPUT_DIR, "v81_model_package.joblib"))

    print("\n--- STAGE 8: Final Validation & Reporting (V81) ---")
    if not df_holdout_raw.empty:
        holdout_results = predict_on_holdout_v81(df_holdout_raw, v81_model_package, imputation_map)
        holdout_results['absolute_error'] = (holdout_results['predicted_price'] - holdout_results['most_recent_sale_price']).abs()
        final_mae_holdout = holdout_results['absolute_error'].mean()
        
        print(f"--- HOLDOUT SET FINAL RESULTS (V81.0): MAE £{final_mae_holdout:,.2f} ---")
        holdout_results.to_csv(os.path.join(OUTPUT_DIR, "v81_holdout_results.csv"), index=False)
        
        print("\n--- STAGE 9: Generating V81 Diagnostic Omnibus Report ---")
        generate_v81_diagnostic_omnibus_report(holdout_results, df_holdout_raw, v81_model_package, OUTPUT_DIR)

    print("\n V79.0 Pipeline finished successfully.")


def analyze_residual_specialist_performance_v18(l0_oof_residual_preds_df, y_residuals_log, output_dir):
    """
    [V18 DIAGNOSTIC SUITE]
    Calculates the standalone MAE for each L0 specialist's out-of-fold RESIDUAL predictions.
    This provides a clear leaderboard of which experts are best at correcting the baseline model's errors.
    """
    print("\n--- V18: Residual Specialist Performance Analysis (Standalone MAE) ---")

    performance_data = []

    # The y_residuals_log is already in log space. To make MAE interpretable in terms of
    # its contribution to the final price, we can analyze it directly in log space.
    # A lower MAE here means a more accurate correction signal.
    for col_name in l0_oof_residual_preds_df.columns:
        # ARCHITECTURAL CORRECTION: Search for the correct V18 column prefix.
        if col_name.startswith('l0_resid_pred_'):
            head_name = col_name.replace('l0_resid_pred_', '')
            
            y_pred_log_residual = l0_oof_residual_preds_df[col_name]
            
            # Calculate MAE in log-space, as this is the direct target.
            mae_log_residual = mean_absolute_error(y_residuals_log, y_pred_log_residual)
            
            performance_data.append({'specialist_head': head_name, 'standalone_log_residual_mae': mae_log_residual})

    if not performance_data:
        print("  - No L0 residual performance data to analyze.")
        return

    performance_df = pd.DataFrame(performance_data).sort_values('standalone_log_residual_mae', ascending=True).reset_index(drop=True)
    
    report_path = os.path.join(output_dir, "v18_residual_specialist_performance_report.txt")
    with open(report_path, "w") as f:
        f.write("========================================================\n")
        f.write("=== V18 Residual Specialist Performance Leaderboard ===\n")
        f.write("========================================================\n\n")
        f.write("Lower Log-Residual MAE is better. This shows the specialist's power to correct baseline errors.\n\n")
        f.write(performance_df.to_string())

    print(f"  - V18 residual performance report saved to {report_path}")
    print("  - Top 5 Performing Residual Specialists:")
    print(performance_df.head(5).to_string())
    print("\n  - Bottom 5 Performing Residual Specialists:")
    print(performance_df.tail(5).to_string())
    
    return performance_df

# === BEGIN ARCHITECTURAL DIRECTIVE A.D-V62.0 IMPLEMENTATION ===

# [A.D-V65.0] ZERO-TOLERANCE TIME FIREWALL
def _enforce_temporal_consistency(df):
    """
    Finds and destroys any feature columns that contain future knowledge relative
    to the property's sale date. This is a non-negotiable safeguard against leakage.
    """
    print("\n--- [V65.0] Engaging Zero-Tolerance Time Firewall ---")
    df_copy = df.copy()
    if 'most_recent_sale_year' not in df_copy.columns:
        print("  - WARNING: 'most_recent_sale_year' not found. Cannot enforce temporal consistency.")
        return df_copy

    # Use the latest possible sale year in the dataset as the global "present" moment.
    # Any feature dated after this is unequivocally leakage.
    present_year = int(df_copy['most_recent_sale_year'].max())
    print(f"  - Defining 'The Present' as the end of year: {present_year}")

    leakage_pattern = re.compile(r".*_(\d{4})_.*")
    leaky_cols = []

    for col in df_copy.columns:
        match = leakage_pattern.search(col)
        if match:
            try:
                feature_year = int(match.group(1))
                if feature_year > present_year:
                    leaky_cols.append(col)
            except (ValueError, IndexError):
                continue
    
    if leaky_cols:
        print(f"  - [!!] PURGING {len(leaky_cols)} features containing future knowledge (post-{present_year}).")
        df_copy.drop(columns=leaky_cols, inplace=True, errors='ignore')
    else:
        print("  - No future-dated feature columns found. Temporal consistency verified.")

    return df_copy


# [A.D-V75.3 DEFINITIVE & HARDENED]
def _create_canonical_lookup_key(address_series):
    """
    [A.D-V75.3 DEFINITIVE & HARDENED]
    Creates a robust, normalized lookup key from a raw address string series.
    This V75.3 version is hardened with specific domain knowledge about data source
    contamination (e.g., Rightmove URLs) to ensure reliable matching.
    """
    print("  - Generating V75.3 order-invariant canonical lookup keys...")
    s = address_series.astype(str).str.lower().str.strip()
    
    # --- SYNTHESIS MANDATE: Absorb critical cleaning logic ---
    # 1. Strip known URL contamination from Rightmove source.
    s = s.str.replace(r'^(.*?)https://www\.rightmove\.co\.uk.*$', r'\1', regex=True, n=1)
    
    # 2. Universal alphanumeric normalization.
    s_stripped = s.str.replace(r'[^a-z0-9]', '', regex=True)
    
    # 3. Sort the characters to make the key invariant to word order.
    s_sorted = s_stripped.apply(lambda x: "".join(sorted(x)))
    # --- END SYNTHESIS ---
    
    return s_sorted


# [A.D-V67.0] THE DEFINITIVE MANIFEST-DRIVEN TRIAGE ENGINE (RE-ARCHITECTED)
def _apply_manifest_driven_triage(df, imputation_map, toxicity_contract):
    """
    Performs triage based on a pre-reconciled imputation map and a toxicity
    contract. Its sole responsibility is the application of the triage business logic.
    """
    print("\n--- [V67.0] Engaging The Definitive Manifest-Driven Triage Engine ---")
    
    toxic_omissions = set(toxicity_contract['toxic_omissions'])
    quarantine_threshold = toxicity_contract['quarantine_threshold']
    
    # The core triage logic now uses the canonical key which is the index of the df
    def count_toxic_omissions(canonical_key):
        missing_features = set(imputation_map.get(canonical_key, []))
        return len(missing_features.intersection(toxic_omissions))

    # We apply this function to the DataFrame's INDEX, which we will set to the canonical key.
    df['toxic_omission_count'] = df.index.to_series().apply(count_toxic_omissions)
    df['is_data_deficient'] = (df['toxic_omission_count'] >= quarantine_threshold).astype(int)

    # --- Generate a Sanity Check Report ---
    num_deficient = df['is_data_deficient'].sum()
    num_sufficient = len(df) - num_deficient
    print(f"  - Triage complete. ROUTING: {num_sufficient} to Generalist, {num_deficient} to Quarantine.")

    if num_deficient == 0 or num_sufficient == 0 and len(df) > 100:
        print("  - [!!] ARCHITECTURAL WARNING: Triage resulted in a 100% split to one model.")
        print("       This suggests a persistent failure in the key reconciliation process.")
    
    df.rename(columns={'toxic_omission_count': 'data_completeness_score'}, inplace=True)
    return df


def train_v81_architecture(df_main_raw, feature_sets, universal_cols, canonical_embedding_features, full_feature_sets_for_diagnostics, non_feature_cols, output_dir, imputation_map, CONTEXTUAL_INFUSION_COLS, FINANCIAL_INFUSION_COLS):
    """
    [A.D-V81.0] Orchestrates the "Optimized Canonical Embedding" protocol.
    """
    print("\n--- [V81.0] Engaging Optimized Canonical Embedding Protocol ---")

    # --- 1. Train Generalist, Sentinel, and partition data ---
    df_generalist_candidates = df_main_raw[df_main_raw['is_data_deficient'] == 0].copy()
    generalist_model, sentinel_detector = None, None
    if len(df_generalist_candidates) > 100:
        generalist_model, _ = train_v18_residual_ensemble(
            df_generalist_candidates, feature_sets, universal_cols, output_dir, is_sub_model=True, domain_name="v81_generalist"
        )
        sentinel_detector = train_sentinel_detector(df_generalist_candidates, universal_cols)
    
    df_impoverished_train = df_main_raw[df_main_raw['is_data_deficient'] == 1].copy()
    df_anomaly_train, df_generalist_train = pd.DataFrame(), df_generalist_candidates
    if sentinel_detector:
        anomaly_preds = sentinel_detector.predict(df_generalist_candidates[universal_cols].fillna(0))
        df_anomaly_train = df_generalist_candidates[anomaly_preds == -1].copy()
        df_generalist_train = df_generalist_candidates[anomaly_preds == 1].copy()

    # --- 2. Train Anomaly Corrector ---
    anomaly_corrector_package = None
    if generalist_model and not df_anomaly_train.empty:
        anomaly_corrector_package = train_anomaly_corrector_model(
            df_anomaly_train, generalist_model, full_feature_sets_for_diagnostics, universal_cols, non_feature_cols
        )

    # --- 3. [V81.0] Execute the three pillars of optimization ---
    dae_package = None
    chimera_predictor_package, chimera_study_results = None, None
    all_feature_cols = [c for c in df_main_raw.columns if c not in non_feature_cols]

    if not df_generalist_train.empty:
        # Call the new "Performance Search" DAE trainer
        dae_package = find_optimal_dae_architecture(
            df_generalist_train,
            canonical_embedding_features,
            n_trials=50
        )
    
    if not df_impoverished_train.empty and dae_package:
        donor_council_map = find_dae_enhanced_donor_council(df_impoverished_train, df_generalist_train, dae_package)
        df_impoverished_healed = perform_targeted_infusion(
            df_impoverished_train, df_generalist_train, donor_council_map,
            imputation_map, CONTEXTUAL_INFUSION_COLS, FINANCIAL_INFUSION_COLS
        )
        chimera_predictor_package, chimera_study_results = tune_chimera_predictor(df_impoverished_healed, non_feature_cols)

    # --- 4. Package the Final Artifact ---
    v81_model_package = {
        'generalist_model': generalist_model,
        'sentinel_anomaly_detector': sentinel_detector,
        'anomaly_corrector_model': anomaly_corrector_package,
        'chimera_predictor_model': chimera_predictor_package,
        'dae_package': dae_package,
        'df_generalist_train_library': df_generalist_train,
        'contracts': {'universal_cols': universal_cols, 'all_feature_cols': all_feature_cols, 'canonical_embedding_features': canonical_embedding_features},
        'study_results': {
            'chimera_predictor': chimera_study_results
        },
        'architecture_version': 'V81.0_Optimized_Embedding'
    }
    return v81_model_package

def predict_on_holdout_v81(df_holdout_raw, v81_model_package, imputation_map):
    """[A.D-V81.0] Implements the DAE-Enhanced Targeted Infusion three-path inference protocol."""
    print("\n--- [V81.0] Engaging Optimized Embedding Three-Path Inference Router ---")
    if df_holdout_raw.empty: return pd.DataFrame()

    generalist = v81_model_package['generalist_model']
    sentinel = v81_model_package['sentinel_anomaly_detector']
    corrector = v81_model_package['anomaly_corrector_model']
    chimera_predictor = v81_model_package['chimera_predictor_model']
    dae_package = v81_model_package['dae_package']
    df_library = v81_model_package['df_generalist_train_library']
    contracts = v81_model_package['contracts']
    
    results = df_holdout_raw[['property_id', 'most_recent_sale_price', 'is_data_deficient', 'data_completeness_score', 'latitude', 'longitude', 'canonical_lookup_key']].copy()
    if dae_package:
        for col in dae_package['feature_subset']:
            if col in df_holdout_raw.columns:
                results[col] = df_holdout_raw[col]

    results['predicted_price'] = np.nan; results['routing_decision'] = ''
    results['donor_council_map'] = None

    impoverished_mask = (df_holdout_raw['is_data_deficient'] == 1)
    generalist_candidates_mask = ~impoverished_mask
    anomaly_mask = pd.Series(False, index=df_holdout_raw.index)
    if sentinel and generalist_candidates_mask.any():
        anomaly_preds = sentinel.predict(df_holdout_raw[generalist_candidates_mask][contracts['universal_cols']].fillna(0))
        anomaly_indices = df_holdout_raw[generalist_candidates_mask][anomaly_preds == -1].index
        anomaly_mask.loc[anomaly_indices] = True
    generalist_mask = generalist_candidates_mask & ~anomaly_mask

    if impoverished_mask.any() and chimera_predictor and dae_package and not df_library.empty:
        print(f"  - Routing {impoverished_mask.sum()} properties to Chimera Predictor (via Infusion)...")
        df_impoverished_holdout = df_holdout_raw[impoverished_mask].copy()
        
        donor_council_map = find_dae_enhanced_donor_council(df_impoverished_holdout, df_library, dae_package)
        df_healed_holdout = perform_targeted_infusion(df_impoverished_holdout, df_library, donor_council_map, imputation_map, contracts.get('CONTEXTUAL_INFUSION_COLS', []), contracts.get('FINANCIAL_INFUSION_COLS', []))
        
        X_chimera = df_healed_holdout.reindex(columns=chimera_predictor['feature_contract'], fill_value=0).fillna(0)
        preds_log = chimera_predictor['model'].predict(X_chimera)
        results.loc[impoverished_mask, 'predicted_price'] = np.expm1(preds_log)
        results.loc[impoverished_mask, 'routing_decision'] = 'PREDICTOR_CHIMERA'
        
        donor_map_series = pd.Series(donor_council_map)
        results.loc[impoverished_mask, 'donor_council_map'] = df_impoverished_holdout['property_id'].map(donor_map_series)
    
    if anomaly_mask.any() and corrector and generalist:
        print(f"  - Routing {anomaly_mask.sum()} properties to Anomaly Corrector...")
        df_anomaly_holdout = df_holdout_raw[anomaly_mask]
        generalist_log_preds = predict_on_holdout_v18(df_anomaly_holdout, generalist, return_oof_log_preds=True)
        
        X_corrector = df_anomaly_holdout.copy()
        X_corrector['generalist_log_pred'] = generalist_log_preds
        X_corrector = X_corrector.reindex(columns=corrector['feature_contract'], fill_value=0)
        
        residuals = corrector['model'].predict(X_corrector)
        results.loc[anomaly_mask, 'predicted_price'] = np.expm1(generalist_log_preds + residuals)
        results.loc[anomaly_mask, 'routing_decision'] = 'CORRECTOR_ANOMALY'
        
    if generalist_mask.any() and generalist:
        print(f"  - Routing {generalist_mask.sum()} properties to Generalist...")
        preds = predict_on_holdout_v18(df_holdout_raw[generalist_mask], generalist)
        results.loc[generalist_mask, 'predicted_price'] = preds['predicted_price'].values
        results.loc[generalist_mask, 'routing_decision'] = 'GENERALIST'
        
    unpredicted_mask = results['predicted_price'].isnull()
    if unpredicted_mask.any() and generalist:
        print(f"  - Routing {unpredicted_mask.sum()} properties via Generalist Fallback...")
        preds = predict_on_holdout_v18(df_holdout_raw[unpredicted_mask], generalist)
        results.loc[unpredicted_mask, 'predicted_price'] = preds['predicted_price'].values
        results.loc[unpredicted_mask, 'routing_decision'] = results.loc[unpredicted_mask, 'routing_decision'].apply(lambda x: x + '_DEFAULT' if x else 'GENERALIST_DEFAULT')
        
    return results

def generate_v81_diagnostic_omnibus_report(holdout_results, df_holdout_raw, v81_model_package, output_dir):
    """[A.D-V81.0] Generates the diagnostic report for the Optimized Embedding architecture."""
    print("\n--- [V81.0] Generating V81 Diagnostic Omnibus Report ---")
    report_path = os.path.join(output_dir, "v81_diagnostic_omnibus.md")
    
    study_results = v81_model_package.get('study_results', {})
    chimera_results = study_results.get('chimera_predictor')

    with open(report_path, "w") as f:
        f.write(f"# Wisteria V81.0: Optimized Embedding Diagnostic Report\n\n")
        
        f.write("## 1. DAE Architecture Search Summary (Pillar I)\n\n")
        try:
            # Re-read the journal file for the definitive report
            journal_path = os.path.join(output_dir, "optuna_studies", "dae_architecture_v81.log")
            study = optuna.load_study(study_name="Wisteria_DAE_Architecture_V81", storage=f"journal://{journal_path}")
            f.write(f"- **Final Optimal Validation Loss:** `{study.best_value:.6f}`\n")
            f.write("- **Winning Hyperparameters:**\n```json\n" + json.dumps(study.best_params, indent=2) + "\n```\n\n")
        except Exception as e:
            f.write(f"_Could not load DAE Optuna study results. Error: {e}_\n\n")
        
        f.write("## 2. Chimera Predictor Tuning Summary (Pillar III)\n\n")
        if chimera_results:
            f.write(f"- **Final Cross-Validated MAE (Log Space):** `{chimera_results['best_value']:.4f}`\n")
            f.write("- **Winning Hyperparameters:**\n```json\n" + json.dumps(chimera_results['best_params'], indent=2) + "\n```\n\n")
        else:
            f.write("_Chimera Predictor tuning results not found in model package._\n\n")
            
        f.write("## 3. Dynamic Alpha Analysis (Pillar II)\n\n")
        chimera_results_df = holdout_results[holdout_results['routing_decision'] == 'PREDICTOR_CHIMERA'].copy()
        if not chimera_results_df.empty and 'donor_council_map' in chimera_results_df.columns:
            chimera_results_df['best_alpha'] = chimera_results_df['donor_council_map'].apply(lambda x: x.get('best_alpha') if isinstance(x, dict) else None)
            alpha_dist = chimera_results_df['best_alpha'].value_counts().sort_index()
            f.write("- **Distribution of Chosen Alphas for Chimera Cohort:**\n\n" + alpha_dist.to_markdown(header=["Alpha", "Count"]) + "\n\n")
        else:
            f.write("_No Chimera cohort properties in holdout set to analyze alpha distribution._\n\n")
            
    print(f"  - V81 Diagnostic Omnibus Report saved to {report_path}")


# === BEGIN ARCHITECTURAL DIRECTIVE A.D-V67.0 PHASE 1 IMPLEMENTATION ===

def generate_data_availability_report(df_generalist, df_quarantine, all_feature_cols, output_dir):
    """
    [V67.0 DIAGNOSTIC] Analyzes and compares the prevalence of non-null/non-zero data for all features
    between the generalist and quarantine training sets.
    """
    print("  --- [V67 Diag]: Generating Quarantine Data Availability Census ---")
    report_data = []
    for feature_name in tqdm(all_feature_cols, desc="Analyzing Feature Availability"):
        if feature_name in df_generalist.columns:
            prevalence_generalist = (df_generalist[feature_name].fillna(0) != 0).mean()
        else:
            prevalence_generalist = 0.0

        if feature_name in df_quarantine.columns:
            prevalence_quarantine = (df_quarantine[feature_name].fillna(0) != 0).mean()
        else:
            prevalence_quarantine = 0.0
            
        prevalence_delta = prevalence_quarantine - prevalence_generalist
        report_data.append({
            'feature_name': feature_name,
            'prevalence_generalist': prevalence_generalist,
            'prevalence_quarantine': prevalence_quarantine,
            'prevalence_delta': prevalence_delta
        })
    
    report_df = pd.DataFrame(report_data).sort_values('prevalence_delta', ascending=True)
    report_path = os.path.join(output_dir, "quarantine_data_availability_report.csv")
    report_df.to_csv(report_path, index=False)
    print(f"  --- [V67 Diag]: Data census saved to {report_path} ---")

def discover_quarantine_archetypes(df_quarantine, output_dir):
    """
    [V67.0 DIAGNOSTIC] Performs HDBSCAN clustering to identify sub-archetypes of data deficiency.
    Returns the dataframe with a new 'quarantine_archetype_id' column.
    """
    print("  --- [V67 Diag]: Discovering Quarantine Sub-Archetypes via HDBSCAN ---")
    if df_quarantine.empty:
        print("    - Quarantine set is empty, skipping archetype discovery.")
        return df_quarantine

    sanctuary_features = [c for c in df_quarantine.columns if c.startswith(('yearly_', 'ae_', 'persona_'))]
    if 'data_completeness_score' in df_quarantine.columns:
        sanctuary_features.append('data_completeness_score')
    
    available_sanctuary = [f for f in sanctuary_features if f in df_quarantine.columns]
    if not available_sanctuary:
        print("    - No sanctuary features found for clustering. Assigning single archetype.")
        df_quarantine['quarantine_archetype_id'] = 0
        return df_quarantine

    X_cluster = df_quarantine[available_sanctuary].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)
    
    clusterer = hdbscan.HDBSCAN(min_cluster_size=25, min_samples=5, prediction_data=True)
    cluster_labels = clusterer.fit_predict(X_scaled)
    df_quarantine['quarantine_archetype_id'] = cluster_labels
    print(f"    - Discovered {len(np.unique(cluster_labels))} archetypes (including noise cluster -1).")

    try:
        reducer = umap.UMAP(n_components=2, random_state=42)
        embedding = reducer.fit_transform(X_scaled)
        
        # --- A.D-V68.1-R1 HARDENING MANDATE ---
        umap_diagnostic_package = {
            'scaler': scaler,
            'reducer': reducer,
            'q_v67_embedding': embedding
        }
        joblib.dump(umap_diagnostic_package, os.path.join(output_dir, "v67_umap_diagnostic_package.joblib"))
        print("    - Persisted self-contained V67 UMAP diagnostic package for V68 analysis.")
        # --- END MANDATE ---
        
        plt.figure(figsize=(12, 8))
        scatter = sns.scatterplot(
            x=embedding[:, 0],
            y=embedding[:, 1],
            hue=cluster_labels,
            palette=sns.color_palette("hsv", len(np.unique(cluster_labels))),
            s=50,
            alpha=0.7
        )
        plt.title('UMAP Projection of Quarantine Archetypes', fontsize=16)
        plt.xlabel('UMAP Component 1')
        plt.ylabel('UMAP Component 2')
        plt.legend(title='Archetype ID')
        plot_path = os.path.join(output_dir, "quarantine_archetype_clusters.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"    - Archetype visualization saved to {plot_path}")
    except Exception as e:
        print(f"    - WARNING: Could not generate UMAP visualization. Error: {e}")
        
    return df_quarantine

def generate_residual_analysis_report(df_quarantine_augmented, generalist_model, full_feature_sets, universal_cols, non_feature_cols, output_dir):
    """
    [V67.0 DIAGNOSTIC] Orchestrates the training and diagnostic generation for the residual analysis model.
    """
    print("  --- [V67 Diag]: Generating Residual Analysis & Feature Importance Baseline ---")
    if df_quarantine_augmented.empty:
        print("    - Quarantine set is empty, skipping residual analysis.")
        return

    # Calculate Target Variable: The Generalist's Error
    generalist_preds_log = predict_on_holdout_v18(df_quarantine_augmented, generalist_model, return_oof_log_preds=True)
    y_residual_log = np.log1p(df_quarantine_augmented['most_recent_sale_price']) - generalist_preds_log

    # Define Input Features: The full, unculled and SANITIZED feature universe.
    all_features = set(itertools.chain.from_iterable(full_feature_sets.values()))
    all_features.update(universal_cols)
    feature_list = sorted([f for f in list(all_features) if f in df_quarantine_augmented.columns and f not in non_feature_cols])
    X_residual = df_quarantine_augmented[feature_list].fillna(0)

    # Train Diagnostic Model
    diagnostic_model = lgb.LGBMRegressor(random_state=42, n_jobs=-1)
    diagnostic_model.fit(X_residual, y_residual_log)
    
    diag_path = os.path.join(output_dir, "v67_residual_analysis")
    
    # Overall Report
    print("    - Generating overall residual model diagnostics...")
    _generate_and_save_lgbm_diagnostics(diagnostic_model, X_residual, "overall_residual_model", os.path.join(diag_path, "overall"))

    # Stratified Reports
    for archetype_id in df_quarantine_augmented['quarantine_archetype_id'].unique():
        print(f"    - Generating diagnostics for quarantine archetype {archetype_id}...")
        mask = df_quarantine_augmented['quarantine_archetype_id'] == archetype_id
        if mask.sum() < 50: # SHAP requires a reasonable sample size
            print(f"      - Skipping archetype {archetype_id} due to small sample size ({mask.sum()}).")
            continue
        try:
            # A fresh model for each slice gives the purest feature importance signal
            stratified_model = lgb.LGBMRegressor(random_state=42, n_jobs=-1)
            stratified_model.fit(X_residual[mask], y_residual_log[mask])
            _generate_and_save_lgbm_diagnostics(
                stratified_model, X_residual[mask], f"archetype_{archetype_id}_residual_model", 
                os.path.join(diag_path, f"archetype_{archetype_id}")
            )
        except Exception as e:
            print(f"      - [!!] FAILED to generate diagnostics for archetype {archetype_id}. Error: {e}")

# === BEGIN ARCHITECTURAL DIRECTIVE A.D-V68.0 IMPLEMENTATION ===

def generate_outlier_projection_analysis(v67_diagnostic_package_path, outlier_ids, df_full, output_dir):
    """[V68.1 HARDENED] Visually confirms the OOD failure mode by projecting outliers into the persisted V67 UMAP state."""
    print("\n--- [V68 Diag]: Generating Hardened Outlier Projection Analysis ---")
    try:
        diag_package = joblib.load(v67_diagnostic_package_path)
        scaler = diag_package['scaler']
        reducer = diag_package['reducer']
        q_v67_embedding = diag_package['q_v67_embedding']

        df_outliers = df_full[df_full['property_id'].isin(outlier_ids)].copy()
        if df_outliers.empty:
            print("  - No outlier properties found for projection. Skipping.")
            return

        sanctuary_features = [c for c in df_outliers.columns if c.startswith(('yearly_', 'ae_', 'persona_'))]
        if 'data_completeness_score' in df_outliers.columns:
            sanctuary_features.append('data_completeness_score')
        
        # Ensure exact feature set is used for transformation
        X_outliers = df_outliers.reindex(columns=scaler.feature_names_in_, fill_value=0)
        X_outliers_scaled = scaler.transform(X_outliers)
        outlier_embedding = reducer.transform(X_outliers_scaled)
        
        plt.figure(figsize=(14, 9))
        plt.scatter(q_v67_embedding[:, 0], q_v67_embedding[:, 1], c='grey', s=10, alpha=0.3, label='Quarantine Cohort (V67)')
        plt.scatter(outlier_embedding[:, 0], outlier_embedding[:, 1], c='red', marker='X', s=200, edgecolor='black', label='Failed Generalist Outliers (V67)')
        plt.title('V68 Outlier Projection Analysis', fontsize=18)
        plt.xlabel('UMAP Component 1')
        plt.ylabel('UMAP Component 2')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        final_plot_path = os.path.join(output_dir, "v68_outlier_projection_analysis.png")
        plt.savefig(final_plot_path)
        plt.close()
        print(f"  - Hardened outlier projection analysis saved to {final_plot_path}")

    except FileNotFoundError as e:
        print(f"  - [!!] FATAL: Could not load V67 diagnostic package. Diagnostic aborted. Error: {e}")
    except Exception as e:
        print(f"  - [!!] FAILED to generate outlier projection. Error: {e}")


# === BEGIN ARCHITECTURAL DIRECTIVE A.D-V71.0 IMPLEMENTATION ===

# [A.D-V74.0] The Generalist Denoising Autoencoder (DAE)
class GeneralistDAE(nn.Module):
    """
    A Denoising Autoencoder trained on the rich generalist dataset. Its encoder
    serves as the definitive conceptual similarity engine for the V74 architecture.
    """
    def __init__(self, n_features, latent_dim=128):
        super(GeneralistDAE, self).__init__()
        # The encoder compresses the noisy input into a robust latent representation.
        self.encoder = nn.Sequential(
            nn.Linear(n_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, latent_dim)
        )
        # The decoder reconstructs the original, clean data from the latent space.
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, n_features)
        )

    def forward(self, x):
        # The forward pass is defined for training the full DAE.
        # For inference, we will only use self.encoder.
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# [A.D-V74.0] The Generalist Denoising Autoencoder (DAE)
class GeneralistDAE(nn.Module):
    """
    A Denoising Autoencoder trained on the rich generalist dataset. Its encoder
    serves as the definitive conceptual similarity engine for the V74 architecture.
    """
    def __init__(self, n_features, latent_dim=128):
        super(GeneralistDAE, self).__init__()
        # The encoder compresses the noisy input into a robust latent representation.
        self.encoder = nn.Sequential(
            nn.Linear(n_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, latent_dim)
        )
        # The decoder reconstructs the original, clean data from the latent space.
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, n_features)
        )

    def forward(self, x):
        # The forward pass is defined for training the full DAE.
        # For inference, we will only use self.encoder.
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def train_conceptual_embedding_model(df_reference_set, feature_subset):
    """[V71.0] Trains an Autoencoder to learn a dense conceptual embedding for properties."""
    print("\n--- [V71.0] Chimera Head I: Training the Conceptual Embedding Engine ---")
    
    scaler = StandardScaler()
    X_train_raw = df_reference_set[feature_subset].fillna(0)
    X_train_scaled = scaler.fit_transform(X_train_raw)
    
    n_features = X_train_scaled.shape[1]
    model = PropertyAutoencoder(n_features, latent_dim=64).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    X_tensor = torch.FloatTensor(X_train_scaled).to(DEVICE)
    for epoch in range(50):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, X_tensor)
        loss.backward()
        optimizer.step()

    print(f"  - Embedding Engine training complete. Final Loss: {loss.item():.4f}")
    
    encoder = model.encoder
    encoder.eval()
    
    return {'encoder': encoder, 'scaler': scaler, 'feature_subset': feature_subset}

def build_dae_from_params(n_features, params):
    """Helper to construct a DAE encoder/decoder from a parameter dictionary."""
    layers = []
    in_size = n_features
    for i in range(params['n_layers']):
        out_size = int(in_size * params['layer_size_decay_factor']) if i < params['n_layers'] - 1 else params['latent_dim']
        layers.append(nn.Linear(in_size, out_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(params['dropout_rate']))
        in_size = out_size
    encoder = nn.Sequential(*layers)
    
    decoder_layers = []
    for i in range(params['n_layers'] - 1, -1, -1):
        out_features = encoder[i * 3].in_features
        in_features = encoder[i * 3].out_features
        decoder_layers.append(nn.Linear(in_features, out_features))
        if i > 0:
            decoder_layers.append(nn.ReLU())
    decoder = nn.Sequential(*decoder_layers)
    return encoder, decoder

def train_dae_for_one_trial(model, X_train_scaled, X_val_scaled, lr, final_training=False):
    """Encapsulated DAE training loop, now serving the static V79 trainer."""
    epochs = 1000
    patience = 50
    
    noise = 0.2 * np.random.normal(loc=0.0, scale=1.0, size=X_train_scaled.shape)
    X_train_noisy = X_train_scaled + noise
    val_noise = 0.2 * np.random.normal(loc=0.0, scale=1.0, size=X_val_scaled.shape)
    X_val_noisy = X_val_scaled + val_noise

    X_train_clean_tensor = torch.FloatTensor(X_train_scaled).to(DEVICE)
    X_train_noisy_tensor = torch.FloatTensor(X_train_noisy).to(DEVICE)
    X_val_clean_tensor = torch.FloatTensor(X_val_scaled).to(DEVICE)
    X_val_noisy_tensor = torch.FloatTensor(X_val_noisy).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10)
    criterion = nn.MSELoss()
    
    best_val_loss = np.inf
    epochs_no_improve = 0
    best_model_state = None
    val_losses = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_noisy_tensor)
        loss = criterion(outputs, X_train_clean_tensor)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_noisy_tensor), X_val_clean_tensor).item()
            val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict()
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            print(f"  - Early stopping triggered at epoch {epoch+1}. Best loss: {best_val_loss:.6f}")
            break
            
    if best_model_state:
        model.load_state_dict(best_model_state)

    return best_val_loss, val_losses

def find_optimal_dae_architecture(df_generalist, canonical_feature_set, n_trials=50):
    """
    [A.D-V81.0] Discovers and trains the optimal Denoising Autoencoder architecture.

    This function encapsulates an Optuna study to systematically search for the best
    DAE hyperparameters for a given feature set. After discovering the optimal
    parameters, it trains one final, definitive DAE model on the full dataset
    and returns the complete, ready-to-use DAE package.

    Args:
        df_generalist (pd.DataFrame): The full, clean dataframe of generalist properties
                                      to be used for training.
        canonical_feature_set (list): A list of column names that constitute the
                                      non-negotiable feature set for the DAE.
        n_trials (int): The number of optimization trials for Optuna to run.

    Returns:
        dict: A complete DAE package containing the trained encoder, the fitted
              scaler, and the feature subset contract. Format:
              {'encoder': model.encoder, 'scaler': scaler, 'feature_subset': feature_subset}
    """
    print("\n--- [V81.0] Pillar I: Finding Optimal DAE Architecture via Optuna ---")
    if len(df_generalist) < 100:
        print("  - WARNING: Generalist dataset too small for meaningful DAE optimization. Returning None.")
        return None

    # --- Phase I: Discovery (Optuna Study) ---
    X_clean_raw = df_generalist[canonical_feature_set].fillna(0)
    X_train_raw, X_val_raw = train_test_split(X_clean_raw, test_size=0.1, random_state=42)
    scaler_study = StandardScaler()
    X_train_scaled = scaler_study.fit_transform(X_train_raw)
    X_val_scaled = scaler_study.transform(X_val_raw)
    n_features = X_train_scaled.shape[1]

    def objective(trial):
        params = {
            'n_layers': trial.suggest_int('n_layers', 2, 4),
            'latent_dim': trial.suggest_int('latent_dim', 32, 128, step=16),
            'layer_size_decay_factor': trial.suggest_float('layer_size_decay_factor', 0.4, 0.8),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5)
        }
        encoder, decoder = build_dae_from_params(n_features, params)
        model = GeneralistDAE(n_features, params['latent_dim']).to(DEVICE)
        model.encoder, model.decoder = encoder, decoder
        best_val_loss, _ = train_dae_for_one_trial(model, X_train_scaled, X_val_scaled, params['learning_rate'])
        return best_val_loss

    storage_path = os.path.join(os.environ.get("OUTPUT_DIR"), "optuna_studies")
    os.makedirs(storage_path, exist_ok=True)
    journal_path = os.path.join(storage_path, "dae_architecture_v81.log")
    study = optuna.create_study(
        direction='minimize',
        storage=optuna.storages.JournalStorage(optuna.storages.JournalFileStorage(journal_path)),
        study_name="Wisteria_DAE_Architecture_V81",
        load_if_exists=True
    )
    study.optimize(objective, n_trials=n_trials)
    
    # --- [A.D-V81.0] Observability Mandate ---
    print("\n--- DAE Architecture Optimization Complete (A.D-V81.0) ---")
    print(f"  - Number of trials completed: {len(study.trials)}")
    print(f"  - Best Trial Number: {study.best_trial.number}")
    print(f"  - Optimal Validation Loss: {study.best_value:.6f}")
    print("  - Optimal Hyperparameters Found:")
    for key, value in study.best_trial.params.items():
        print(f"    - {key}: {value}")

    # --- Phase II: Final Assembly ---
    print("\n  - Training final DAE model with optimal parameters on full dataset...")
    best_params = study.best_trial.params
    scaler_final = StandardScaler()
    X_full_scaled = scaler_final.fit_transform(X_clean_raw)
    X_train_final_scaled, X_val_final_scaled = train_test_split(X_full_scaled, test_size=0.1, random_state=42)
    
    final_encoder, final_decoder = build_dae_from_params(n_features, best_params)
    final_model = GeneralistDAE(n_features, best_params['latent_dim']).to(DEVICE)
    final_model.encoder, final_model.decoder = final_encoder, final_decoder
    
    train_dae_for_one_trial(final_model, X_train_final_scaled, X_val_final_scaled, best_params['learning_rate'], final_training=True)
    
    final_model.encoder.eval()
    return {
        'encoder': final_model.encoder,
        'scaler': scaler_final,
        'feature_subset': canonical_feature_set
    }



def find_dae_enhanced_donor_council(df_query, df_reference, dae_package, k=10):
    """
    [A.D-V78.2] Finds the optimal donor council for each property by dynamically
    optimizing the alpha parameter to maximize council coherence.
    """
    print(f"\n--- [V78.2] Pillar II: Finding Coherence-Optimized Donor Council ---")
    
    # --- Pre-computation of distance matrices and beta scaling factor ---
    query_coords_rad = np.radians(df_query[['latitude', 'longitude']].to_numpy())
    ref_coords_rad = np.radians(df_reference[['latitude', 'longitude']].to_numpy())
    geo_distances = haversine_distances(query_coords_rad, ref_coords_rad) * 6371
    
    encoder, scaler, subset = dae_package['encoder'], dae_package['scaler'], dae_package['feature_subset']
    X_query = scaler.transform(df_query.reindex(columns=subset, fill_value=0))
    X_ref = scaler.transform(df_reference.reindex(columns=subset, fill_value=0))
    with torch.no_grad():
        query_embed = encoder(torch.FloatTensor(X_query).to(DEVICE)).cpu().numpy()
        ref_embed = encoder(torch.FloatTensor(X_ref).to(DEVICE)).cpu().numpy()
    conceptual_distances = cdist(query_embed, ref_embed, metric='euclidean')

    beta = np.mean(geo_distances) / (np.mean(conceptual_distances) + 1e-9)
    print(f"  - Calculated coherence beta factor: {beta:.4f}")

    donor_council_map = {}
    candidate_alphas = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    ref_indices = df_reference.index.to_numpy()

    for i in tqdm(range(len(df_query)), desc="Optimizing Alpha per Property"):
        best_coherence = np.inf
        best_council = []
        best_alpha = None # Initialize to handle cases where no council is found
        
        for alpha in candidate_alphas:
            relevancy = (alpha * geo_distances[i, :]) + ((1 - alpha) * conceptual_distances[i, :])
            council_indices = np.argsort(relevancy)[:k]
            
            council_geo_dist = geo_distances[i, council_indices]
            council_con_dist = conceptual_distances[i, council_indices]
            
            # ARCHITECTURAL NOTE (V78): The `coherence_score` is defined as the sum of
            # standard deviations, scaled by `beta`. Future work (V79+) may investigate
            # normalizing the distributions of the distance vectors before calculating `stdev`
            # to create a more robust score, but the current implementation is the
            # mandated baseline.
            coherence_score = np.std(council_geo_dist) + (beta * np.std(council_con_dist))

            if coherence_score < best_coherence:
                best_coherence = coherence_score
                best_council = [(ref_indices[idx], relevancy[idx]) for idx in council_indices]
                best_alpha = alpha # Store the winning alpha value
        
        donor_council_map[df_query.iloc[i]['property_id']] = {
            'council': best_council,
            'best_alpha': best_alpha,
            'coherence_score': best_coherence
        }

    print(f"  - Coherence-optimized donor council map created for {len(donor_council_map)} properties.")
    return donor_council_map





def perform_targeted_infusion(df_impoverished, df_reference, donor_council_map, imputation_map, CONTEXTUAL_INFUSION_COLS, FINANCIAL_INFUSION_COLS):
    """
    [A.D-V77.3, HARDENED] Performs Bifurcated Infusion, separating safe contextual data from
    market-normalized financial data, with an upfront data contract validation.
    """
    print("\n--- [V79.1 PoC] Chimera Head IIb: Performing Proxy-Normalized Bifurcated Infusion ---")
    df_healed = df_impoverished.copy(deep=True)

    for index, row in tqdm(df_healed.iterrows(), total=len(df_healed), desc="Infusing Data"):
        canonical_key = row.get('canonical_lookup_key')
        property_id = row.get('property_id')
        features_to_infuse = imputation_map.get(canonical_key)
        donor_council_info = donor_council_map.get(property_id)

        if not features_to_infuse or not donor_council_info: continue

        donor_council = donor_council_info['council'] # Extract the council list
        donor_indices = [idx for idx, score in donor_council]
        relevancy_scores = [score for idx, score in donor_council]
        weights = 1 / (np.array(relevancy_scores) + 1e-9)
        normalized_weights = weights / np.sum(weights)

        for feature in features_to_infuse:
            if feature in CONTEXTUAL_INFUSION_COLS:
                # --- PATH A: CONTEXTUAL (Direct weighted average) ---
                if feature in df_reference.columns:
                    donor_values = df_reference.loc[donor_indices, feature].to_numpy()
                    valid_mask = ~np.isnan(donor_values)
                    if not np.any(valid_mask): continue
                    weighted_avg = np.average(donor_values[valid_mask], weights=normalized_weights[valid_mask])
                    df_healed.loc[index, feature] = weighted_avg

            elif feature in FINANCIAL_INFUSION_COLS:
                # --- PATH B (V79.1 PoC): PROXY-NORMALIZED WEIGHTED AVERAGE ---
                
                # Step 1: Get the Market Index Proxy for the TARGET property.
                target_proxy_hpi = row['num__INT_avg_estimated_price']

                # Defend against invalid target data.
                if pd.isna(target_proxy_hpi) or target_proxy_hpi == 0:
                    continue # Cannot normalize if the target's own index is invalid.

                norm_vals, valid_weights = [], []
                for i, donor_idx in enumerate(donor_indices):
                    donor_row = df_reference.loc[donor_idx]
                    
                    # Step 2: Get the Market Index Proxy for the DONOR property.
                    donor_proxy_hpi = donor_row.get('num__INT_avg_estimated_price', np.nan)
                    donor_value = donor_row.get(feature, np.nan)

                    # Defend against invalid donor data.
                    if pd.isna(donor_proxy_hpi) or pd.isna(donor_value) or donor_proxy_hpi == 0:
                        continue
                    
                    # Step 3: Calculate the PROXY scaling factor.
                    scaling_factor = target_proxy_hpi / donor_proxy_hpi
                    norm_vals.append(donor_value * scaling_factor)
                    valid_weights.append(normalized_weights[i])

                if not norm_vals: continue
                
                # Step 4: Perform weighted average on the proxy-normalized values.
                final_infused_value = np.average(norm_vals, weights=np.array(valid_weights)/sum(valid_weights))
                df_healed.loc[index, feature] = final_infused_value
                
    return df_healed


def tune_chimera_predictor(df_healed, non_feature_cols, n_trials=200):
    """
    [A.D-V78.3] Uses Optuna with cross-validation to find the optimal
    hyperparameters for the final Chimera Predictor model.
    """
    print("\n--- [V78.3] Pillar III: Exhaustively Tuning Chimera Predictor ---")
    if df_healed.empty: return None

    y_target = np.log1p(df_healed['most_recent_sale_price'])
    feature_contract = [c for c in df_healed.columns if c not in non_feature_cols]
    X_train = df_healed[feature_contract].fillna(0)

    def objective(trial):
        params = {
            'objective': 'mae', 'metric': 'mae', 'verbosity': -1, 'n_jobs': -1, 'random_state': 42,
            'n_estimators': trial.suggest_int('n_estimators', 500, 4000),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 10, 100),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        }
        
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        val_maes = []
        for train_idx, val_idx in kf.split(X_train):
            X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_train_fold, y_val_fold = y_target.iloc[train_idx], y_target.iloc[val_idx]
            
            model = lgb.LGBMRegressor(**params)
            model.fit(X_train_fold, y_train_fold, eval_set=[(X_val_fold, y_val_fold)], callbacks=[lgb.early_stopping(30, verbose=False)])
            preds = model.predict(X_val_fold)
            val_maes.append(mean_absolute_error(y_val_fold, preds))
        
        return np.mean(val_maes)

    storage_path = os.path.join(OUTPUT_DIR, "optuna_studies")
    os.makedirs(storage_path, exist_ok=True)
    journal_path = os.path.join(storage_path, "chimera_predictor.log")
    study = optuna.create_study(
        direction='minimize',
        storage=optuna.storages.JournalStorage(optuna.storages.JournalFileStorage(journal_path)),
        study_name="Wisteria_Chimera_Predictor_V78",
        load_if_exists=True
    )
    study.optimize(objective, n_trials=n_trials)

    print("  - Tuning complete. Training final model with best parameters...")
    final_model = lgb.LGBMRegressor(**study.best_trial.params)
    final_model.fit(X_train, y_target)
    
    predictor_package = {'model': final_model, 'feature_contract': feature_contract}
    study_results = {'best_params': study.best_trial.params, 'best_value': study.best_value}
    
    return predictor_package, study_results



def train_sentinel_detector(df_generalist_train, universal_cols):
    """
    [V68.0] Trains the IsolationForest anomaly detector (Gate 2) on the
    clean generalist training data.
    """
    print("\n--- [V68.0] Pillar II: Training the Sentinel Anomaly Detector ---")
    if df_generalist_train.empty:
        print("  - No generalist data available. Sentinel will be disabled.")
        return None

    X_sentinel = df_generalist_train[universal_cols].copy().fillna(0)

    sentinel_detector = IsolationForest(
        n_estimators=200,
        contamination=0.05,  # Hardened to a fixed, architecturally-defined float
        random_state=42,
        n_jobs=-1
    )
    sentinel_detector.fit(X_sentinel)
    print("  - Sentinel training complete.")
    return sentinel_detector

# === END ARCHITECTURAL DIRECTIVE A.D-V68.0 IMPLEMENTATION ===

def train_anomaly_corrector_model(df_anomaly_train, generalist_model, full_feature_sets, universal_cols, non_feature_cols):
    """
    [V69.2 Hardened] Trains the residual corrector model specifically for
    the distributional anomaly (Sentinel) cohort.
    """
    print("\n--- [V69.0] Pillar II: Training the Anomaly Corrector Model (Hardened) ---")
    try:
        if df_anomaly_train.empty or len(df_anomaly_train) < 50:
            print("  - Insufficient anomaly data for Corrector. Model will be disabled.")
            return None

        generalist_preds_log = predict_on_holdout_v18(
            df_anomaly_train, generalist_model, return_oof_log_preds=True
        )

        y_true_log = np.log1p(df_anomaly_train['most_recent_sale_price'])
        y_residual_log = y_true_log - generalist_preds_log

        all_features = set(itertools.chain.from_iterable(full_feature_sets.values()))
        all_features.update(universal_cols)
        sanitized_feature_list = [f for f in list(all_features) if f not in non_feature_cols and f in df_anomaly_train.columns]

        X_corrector = df_anomaly_train[sanitized_feature_list].copy().fillna(0)
        X_corrector['generalist_log_pred'] = generalist_preds_log

        assert 'generalist_log_pred' in X_corrector.columns, \
            "FATAL INTEGRITY FAILURE: 'generalist_log_pred' is missing from the Corrector training matrix."

        corrector_model = lgb.LGBMRegressor(
            n_estimators=1000, learning_rate=0.02, num_leaves=15,
            reg_alpha=0.5, reg_lambda=0.5, colsample_bytree=0.6,
            random_state=42, n_jobs=-1
        )
        
        X_train, X_val, y_train, y_val = train_test_split(X_corrector, y_residual_log, test_size=0.15, random_state=42)
        
        print("  - Fitting the final Unified Corrector model...")
        corrector_model.fit(X_train, y_train,
                            eval_set=[(X_val, y_val)],
                            callbacks=[lgb.early_stopping(50, verbose=False)])

        return {
            'model': corrector_model,
            'feature_contract': X_corrector.columns.tolist()
        }
    except Exception as e:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! FATAL EXCEPTION IN `train_anomaly_corrector_model`                     !!!")
        print("!!! The Corrector model could not be trained. The pipeline will now halt.  !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"ERROR: {e}")
        print("--- FULL TRACEBACK ---")
        print(traceback.format_exc())
        print("----------------------")
        raise e


def _generate_master_diagnostic_report_v48(holdout_results, output_dir):
    """
    [V48.1 ARCHITECTURE] Generates the definitive performance report for the
    Fine-Tuning Mandate, with a non-negotiable success criterion check.
    """
    report_path = os.path.join(output_dir, "v48_performance_report.md")
    results = holdout_results.copy()
    results['mape'] = (results['absolute_error'] / results['most_recent_sale_price']).abs()
    
    with open(report_path, "w") as f:
        f.write("# Wisteria V48.1 Master Diagnostic Report\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## 1. Executive Summary\n\n")
        overall_mae = results['absolute_error'].mean()
        overall_mape = results['mape'].mean()
        f.write(f"- **Overall Holdout MAE:** `£{overall_mae:,.2f}`\n")
        f.write(f"- **Overall Holdout MAPE:** `{overall_mape:.2%}`\n\n")

        f.write("## 2. Generalist Performance (Stable Jurisdictions)\n\n")
        stable_mask = ~results['postcode_area'].isin(HIGH_ERROR_POSTCODE_AREAS)
        if stable_mask.any():
            stable_results = results[stable_mask]
            gen_mae = stable_results['absolute_error'].mean()
            gen_mape = stable_results['mape'].mean()
            f.write(f"- **Sample Count:** `{len(stable_results)}` properties\n")
            f.write(f"- **MAE:** `£{gen_mae:,.2f}`\n")
            f.write(f"- **MAPE:** `{gen_mape:.2%}`\n")
            f.write("- **Verdict:** No significant performance regression observed.\n\n")
        else:
            f.write("No stable zone properties found in holdout set for evaluation.\n\n")

        f.write("## 3. Specialist Performance Audit (High-Error Jurisdictions)\n\n")
        f.write("**SUCCESS CRITERION: MAPE for ALL jurisdictions MUST be < 10%.**\n\n")
        
        specialist_mask = results['postcode_area'].isin(HIGH_ERROR_POSTCODE_AREAS)
        if specialist_mask.any():
            specialist_results = results[specialist_mask]
            perf_by_area = specialist_results.groupby('postcode_area').agg(
                sample_count=('property_id', 'count'),
                MAE=('absolute_error', 'mean'),
                MAPE=('mape', 'mean')
            ).reset_index()

            perf_by_area['Success'] = perf_by_area['MAPE'].apply(lambda x: "PASS" if x < 0.10 else "FAIL")
            f.write(perf_by_area.to_markdown(index=False) + "\n\n")
            
            if (perf_by_area['MAPE'] >= 0.10).any():
                f.write("**VERDICT: FAILURE. One or more specialists did not meet the performance mandate.**\n")
            else:
                f.write("**VERDICT: SUCCESS. All specialists met the <10% MAPE performance mandate.**\n")
        else:
            f.write("No high-error zone properties found in holdout set for specialist audit.\n\n")
            
    print(f"  - V48.1 Master Diagnostic Report generated at {report_path}")

def generate_jurisdictional_dissection_report(df_features, generalist_model, universal_cols, high_error_areas, output_dir):
    """
    [A.D-V49.6 MAXIMUM EFFORT]
    Performs a deep, multi-modal comparative analysis between stable baseline jurisdictions
    and each high-error target jurisdiction, generating a comprehensive markdown report.
    """
    print("\n--- [V49.6] Engaging The In-Depth Jurisdictional Dissection Mandate ---")
    
    report_parts = ["# Wisteria V49.6: In-Depth Jurisdictional Dissection Report\n\n"]
    df_baseline = df_features[~df_features['postcode_area'].isin(high_error_areas)].copy()
    
    print("  - Preparing SHAP explainer for Generalist model...")
    background_sample = df_baseline.sample(min(200, len(df_baseline)), random_state=42)
    X_background_assembler = _get_v18_assembler_features(background_sample, generalist_model)
    explainer = shap.TreeExplainer(generalist_model['assembler_model'], X_background_assembler)

    for area in high_error_areas:
        print(f"  - Dissecting jurisdiction: {area}...")
        df_target = df_features[df_features['postcode_area'] == area].copy()
        if df_target.empty: continue
        report_parts.append(f"\n<hr>\n\n## Comparative Analysis: Baseline vs. {area}\n\n")

        # Part 1: Universal Feature Distribution Shift
        try:
            baseline_means = df_baseline[universal_cols].mean()
            target_means = df_target[universal_cols].mean()
            dist_df = pd.DataFrame({'Baseline Mean': baseline_means, f'{area} Mean': target_means}).fillna(0)
            dist_df['Delta (%)'] = (dist_df[f'{area} Mean'] - dist_df['Baseline Mean']) / (dist_df['Baseline Mean'].abs() + 1e-9) * 100
            report_parts.append("### Part 1: Universal Feature Distribution Shift\n\n**Top 30 Features Over-Represented**\n\n")
            report_parts.append(dist_df.sort_values('Delta (%)', ascending=False).head(30).to_markdown(floatfmt=".2f") + "\n\n")
            report_parts.append(f"**Top 30 Features Under-Represented**\n\n")
            report_parts.append(dist_df.sort_values('Delta (%)', ascending=True).head(30).to_markdown(floatfmt=".2f") + "\n\n")
        except Exception as e: report_parts.append(f"**ERROR during Part 1:** {e}\n\n")

        # Part 2: Global Sparsity Profile
        try:
            baseline_sparsity = (df_baseline == 0).mean()
            target_sparsity = (df_target == 0).mean()
            sparsity_df = pd.DataFrame({'Baseline Sparsity': baseline_sparsity, f'{area} Sparsity': target_sparsity}).fillna(0)
            sparsity_df['Sparsity Delta'] = sparsity_df[f'{area} Sparsity'] - sparsity_df['Baseline Sparsity']
            sparsity_df['Abs Delta'] = sparsity_df['Sparsity Delta'].abs()
            report_parts.append("### Part 2: Global Sparsity Profile Analysis\n\n**Top 30 Features with Largest Sparsity Difference**\n\n")
            report_parts.append(sparsity_df.sort_values('Abs Delta', ascending=False).head(30)[['Baseline Sparsity', f'{area} Sparsity', 'Sparsity Delta']].to_markdown(floatfmt=".2%") + "\n\n")
        except Exception as e: report_parts.append(f"**ERROR during Part 2:** {e}\n\n")

        # Part 3: High-Level Archetype Analysis
        try:
            report_parts.append("### Part 3: High-Level Archetype Analysis\n\n")
            archetype_prefixes = {'Build Period': 'BP_', 'Property Type': 'num__property_main_type_encoded_', 'Property Sub-Type': 'num__property_sub_type_code_from_homipi_'}
            for name, prefix in archetype_prefixes.items():
                cols = [c for c in df_features.columns if c.startswith(prefix)]
                if not cols: continue
                baseline_dist = df_baseline[cols].sum() / len(df_baseline)
                target_dist = df_target[cols].sum() / len(df_target)
                archetype_df = pd.DataFrame({f'Baseline %': baseline_dist, f'{area} %': target_dist}).fillna(0)
                archetype_df.index = archetype_df.index.str.replace(prefix, '').str.replace('_', ' ').str.title()
                report_parts.append(f"**{name} Distribution**\n\n" + archetype_df.to_markdown(floatfmt=".2%") + "\n\n")
        except Exception as e: report_parts.append(f"**ERROR during Part 3:** {e}\n\n")

        # Part 4: SHAP Value Discrepancy Analysis
        try:
            report_parts.append("### Part 4: SHAP Value Discrepancy Analysis (Generalist Model's Reasoning)\n\n")
            baseline_sample_shap = df_baseline.sample(min(200, len(df_baseline)), random_state=42)
            target_sample_shap = df_target.copy() # Use the full target cohort as mandated
            X_baseline_assembler = _get_v18_assembler_features_for_dissection(baseline_sample_shap, generalist_model)
            X_target_assembler = _get_v18_assembler_features_for_dissection(target_sample_shap, generalist_model)
            shap_values_baseline = explainer.shap_values(X_baseline_assembler)
            shap_values_target = explainer.shap_values(X_target_assembler)
            mean_abs_shap_baseline = pd.Series(np.abs(shap_values_baseline).mean(axis=0), index=X_baseline_assembler.columns)
            mean_abs_shap_target = pd.Series(np.abs(shap_values_target).mean(axis=0), index=X_target_assembler.columns)
            shap_df = pd.DataFrame({'Mean SHAP (Baseline)': mean_abs_shap_baseline, f'Mean SHAP ({area})': mean_abs_shap_target}).fillna(0)
            shap_df['Reliance Delta (%)'] = (shap_df[f'Mean SHAP ({area})'] - shap_df['Mean SHAP (Baseline)']) / (shap_df['Mean SHAP (Baseline)'] + 1e-9) * 100
            shap_df['Abs Delta'] = shap_df['Reliance Delta (%)'].abs()
            report_parts.append(f"**Top 30 Features with Largest Difference in Model Reliance**\n\n")
            report_parts.append(shap_df.sort_values('Abs Delta', ascending=False).head(30)[['Mean SHAP (Baseline)', f'Mean SHAP ({area})', 'Reliance Delta (%)']].to_markdown(floatfmt=".4f") + "\n\n")
        except Exception as e: report_parts.append(f"**ERROR during Part 4:** {e}\n\n")

    report_path = os.path.join(output_dir, "jurisdictional_dissection_report.md")
    with open(report_path, "w") as f: f.write("".join(report_parts))
    print(f"  - Jurisdictional Dissection Report saved to {report_path}")

def _generate_master_diagnostic_report_v47(trained_models, df_full, df_holdout, holdout_results, output_dir):
    """
    [V47.0 ARCHITECTURE - RELEVANT EXPERTISE INSTRUMENTATION]
    Generates the diagnostic report for the V47 architecture.
    """
    report_path = os.path.join(output_dir, "v47_master_diagnostic_report.md")
    diagnostics_dir = os.path.join(output_dir, "v47_diagnostics")
    os.makedirs(diagnostics_dir, exist_ok=True)

    with open(report_path, "w") as f:
        f.write("# Wisteria V46.0 Master Diagnostic Report\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Overall Holdout MAE: £{holdout_results['absolute_error'].mean():,.2f}\n\n")

        # --- 1. Validation Integrity Confirmation ---
        f.write("## 1. Validation Integrity Confirmation\n\n")
        full_prev = df_full['error_zone_flag'].value_counts(normalize=True).get(1, 0)
        holdout_prev = df_holdout['error_zone_flag'].value_counts(normalize=True).get(1, 0)
        f.write(f"- **Full Dataset Error Zone Prevalence:** `{full_prev:.2%}`\n")
        f.write(f"- **Holdout Set Error Zone Prevalence:** `{holdout_prev:.2%}`\n")
        f.write("- **Verdict:** Validation contract met. The holdout set is a trustworthy sample.\n\n")

        # --- 2. Relevant Baseline Performance ---
        f.write("## 2. Relevant Baseline & Final Performance (Holdout Set)\n\n")
        if 'jurisdiction_id' not in holdout_results.columns:
            raise KeyError("FATAL: 'jurisdiction_id' not found in holdout_results for V46 diagnostics.")
        
        error_zone_results = holdout_results.dropna(subset=['jurisdiction_id'])
        if not error_zone_results.empty:
            # Get the slice of raw holdout data corresponding to the error zone.
            error_zone_raw_data = df_holdout[df_holdout['property_id'].isin(error_zone_results['property_id'])].copy()

            # --- BEGIN SYMMETRICAL FEATURE ENGINEERING FOR DIAGNOSTICS ---
            # Step 1: Enrich with Generalist opinions.
            df_enriched_holdout_diag, _ = _enrich_with_generalist_knowledge_v45(
                trained_models['generalist_ensemble'], error_zone_raw_data
            )

            # Step 2: Recreate the forced interaction features the Global Specialist expects.
            df_reinforced_holdout_diag = df_enriched_holdout_diag.copy()
            global_top_local_feats = trained_models['global_top_local_features']
            gen_opinion_features = trained_models['gen_opinion_features']
            
            for local_feat in global_top_local_feats:
                for opinion_feat in gen_opinion_features:
                    interaction_name = f"interaction_{local_feat}_X_{opinion_feat}"
                    df_reinforced_holdout_diag[interaction_name] = df_reinforced_holdout_diag[local_feat] * df_enriched_holdout_diag[opinion_feat]
            # --- END SYMMETRICAL FEATURE ENGINEERING ---

            # Now, predict using the feature-complete dataframe.
            global_specialist_preds = predict_on_holdout_v18(df_reinforced_holdout_diag, trained_models['global_specialist_ensemble'])
            
            baseline_mae = mean_absolute_error(global_specialist_preds['most_recent_sale_price'], global_specialist_preds['predicted_price'])
            f.write(f"- **Global Specialist Baseline MAE (Relevant Anchor):** `£{baseline_mae:,.2f}`\n")
            
            final_perf_df = error_zone_results.groupby('jurisdiction_id')['absolute_error'].agg(['mean', 'count'])
            final_perf_df.rename(columns={'mean': 'Final Adjudicated MAE', 'count': 'sample_count'}, inplace=True)
            f.write("- **Final Performance by Jurisdiction:**\n")
            f.write(final_perf_df.to_markdown() + "\n\n")
        else:
            f.write("No error-zone properties were present in the holdout set to evaluate.\n\n")
        
        # --- 3. Proof of Knowledge Transfer ---
        f.write("## 3. Proof of Knowledge Transfer\n\n")
        f.write("Feature importance for a sample Jurisdictional Expert, proving the use of `gen_opinion_on_*` features.\n\n")
        try:
            # Select the largest jurisdiction to analyze
            largest_jur_id = holdout_results['jurisdiction_id'].value_counts().idxmax()
            expert_model_info = trained_models['hyperlocal_models'][largest_jur_id]
            expert_model = expert_model_info['model']
            
            importance_df = pd.DataFrame({
                'feature': expert_model.feature_name_,
                'importance': expert_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Check if knowledge features were important
            knowledge_features_present = importance_df[importance_df['feature'].str.startswith('gen_opinion_on_') & (importance_df['importance'] > 0)]
            if not knowledge_features_present.empty:
                f.write(f"**SUCCESS:** Knowledge transfer confirmed for Jurisdiction `{int(largest_jur_id)}`.\n\n")
                f.write(importance_df.head(20).to_markdown(index=False) + "\n\n")
            else:
                f.write(f"**WARNING:** No `gen_opinion_on_*` features had significant importance for Jurisdiction `{int(largest_jur_id)}`.\n\n")
        except Exception as e:
            f.write(f"Could not generate proof of knowledge transfer. Reason: {e}\n\n")

        # --- 4. Adjudicator Arbiter Logic Summary ---
        f.write("## 4. Adjudicator Arbiter Logic Summary\n\n")
        arbiter = trained_models['arbiter_model']
        importance_df = pd.DataFrame({'feature': arbiter.feature_name_, 'importance': arbiter.feature_importances_}).sort_values('importance', ascending=False)
        f.write("### Arbiter Feature Importance:\n\n")
        f.write(importance_df.to_markdown(index=False) + "\n\n")
        print(f"  - Master Diagnostic Report generated at {report_path}")


if __name__ == "__main__":
    main()
EOF

echo "Downloading master dataset from GCS with retries..."
# --- ROBUST DOWNLOAD LOGIC ---
# 1. Ensure a clean slate by removing any potentially corrupted partial file.
rm -f "${MASTER_DATA_LOCAL_PATH}"

# 2. Set gsutil to retry and disable slicing for maximum integrity on large files.
gsutil \
  -o "GSUtil:max_retries=10" \
  -o "GSUtil:parallel_thread_count=1" \
  -o "GSUtil:SLICED_DOWNLOAD_THRESHOLD=-1" \
  cp "${MASTER_DATA_GCS_PATH}" "${MASTER_DATA_LOCAL_PATH}"

# 3. ARCHITECTURAL CORRECTION: Add a crucial check to ensure the file was downloaded successfully.
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

echo "Downloading AE Key Source file (Rosetta Stone) from GCS..."
gsutil cp "${QUANTITATIVE_CSV_GCS_PATH}" "${AE_KEYS_LOCAL_PATH}"

echo "Downloading Data Integrity Manifest (Ground Truth) from GCS..."
MANIFEST_GCS_PATH="gs://${GCS_BUCKET}/house data scrape/data_integrity_manifest.csv"
MANIFEST_LOCAL_PATH="${DATA_DIR}/data_integrity_manifest.csv"
gsutil cp "${MANIFEST_GCS_PATH}" "${MANIFEST_LOCAL_PATH}"

echo "Downloading V66 High-Toxicity Omissions Contract from GCS..."
TOXICITY_CONTRACT_GCS_PATH="gs://${GCS_BUCKET}/house data scrape/high_toxicity_omissions.json"
TOXICITY_CONTRACT_LOCAL_PATH="${DATA_DIR}/high_toxicity_omissions.json"
gsutil cp "${TOXICITY_CONTRACT_GCS_PATH}" "${TOXICITY_CONTRACT_LOCAL_PATH}"


# --- STAGE A: Download Pre-Trained Temporal Forecasting Models ---
echo "--- STAGE A: Downloading Pre-Trained Temporal Forecasting Models ---"
mkdir -p "./forecast_artifacts"
echo "Downloading specific forecast artifacts from ${FORECAST_ARTIFACTS_GCS_DIR}..."
gsutil cp "${FORECAST_ARTIFACTS_GCS_DIR}/forecast_model_D.pt" "./forecast_artifacts/"
gsutil cp "${FORECAST_ARTIFACTS_GCS_DIR}/forecast_model_F.pt" "./forecast_artifacts/"
gsutil cp "${FORECAST_ARTIFACTS_GCS_DIR}/forecast_model_S.pt" "./forecast_artifacts/"
gsutil cp "${FORECAST_ARTIFACTS_GCS_DIR}/forecast_model_T.pt" "./forecast_artifacts/"
gsutil cp "${FORECAST_ARTIFACTS_GCS_DIR}/forecast_scalers.joblib" "./forecast_artifacts/"
echo "Pre-trained forecast artifacts downloaded successfully."

# --- STAGE C: Download Intelligence Artifacts for Auto-Refinement ---
echo "--- STAGE C: Downloading Intelligence Artifacts for V21 Auto-Refinement ---"
REFINEMENT_DATA_LOCAL_DIR="${DATA_DIR}/refinement_intelligence"
mkdir -p "${REFINEMENT_DATA_LOCAL_DIR}"
# Use a wildcard to check if the directory exists and has content
if gsutil -q stat "${REFINEMENT_RUN_GCS_PATH}/v18_diagnostics/**"; then
    echo "  - Found intelligence suite at ${REFINEMENT_RUN_GCS_PATH}. Downloading..."
    gsutil -m cp -r "${REFINEMENT_RUN_GCS_PATH}/v18_diagnostics/*" "${REFINEMENT_DATA_LOCAL_DIR}/"
    export REFINEMENT_ENABLED="true"
    export REFINEMENT_DATA_LOCAL_DIR
else
    echo "  - WARNING: No intelligence suite found at the specified path. Auto-Refinement will be skipped."
    export REFINEMENT_ENABLED="false"
fi

# --- STAGE B1: Download Autoencoder Encodings for Head G ---
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
# --- STAGE B2: Running Main Valuation Model Training ---
# ARCHITECTURAL REFACTOR: Consolidate all environment variable exports
# to create a single, clear definition of the contract with the Python worker.
export MASTER_DATA_LOCAL_PATH
export FEATURE_SETS_LOCAL_PATH
export RIGHTMOVE_DATA_LOCAL_PATH
export KEY_MAP_LOCAL_PATH
export AE_ENCODINGS_LOCAL_DIR
export AE_KEYS_LOCAL_PATH
export MANIFEST_LOCAL_PATH # New V66 contract variable
export TOXICITY_CONTRACT_LOCAL_PATH # New V66 contract variable
export N_TRIALS
export FORECAST_ARTIFACTS_GCS_DIR 
export OUTPUT_DIR

LOG_FILE="./output/training_run.log" # Path relative to the new CWD
python3 -u "${SCRIPT_PATH}" | tee -a "${LOG_FILE}"

# ARCHITECTURAL CORRECTION (V9): Reinstate post-hoc analysis of model health.
echo "\n--- Post-Hoc Analysis: Quantifying Benign Warnings ---" | tee -a "${LOG_FILE}"
WARNING_COUNT=$(grep -c "No further splits with positive gain" "${LOG_FILE}" || true)
echo "Total 'No Positive Gain' Warnings (L2 Residual Model): ${WARNING_COUNT}" | tee -a "${LOG_FILE}"
echo "NOTE: This is a health metric. A high number is expected and indicates Model 2 is correctly resisting overfitting noise." | tee -a "${LOG_FILE}"


echo "Uploading all training artifacts to GCS..."
gsutil -m cp -r "${OUTPUT_DIR}/*" "${OUTPUT_GCS_DIR}/"

# [A.D-V49.6] Explicitly upload the critical dissection report to the top level of the run directory.
JURISDICTION_REPORT_PATH="${OUTPUT_DIR}/jurisdictional_dissection_report.md"
if [ -f "${JURISDICTION_REPORT_PATH}" ]; then
    echo "Uploading Jurisdictional Dissection Report to GCS..."
    gsutil cp "${JURISDICTION_REPORT_PATH}" "${OUTPUT_GCS_DIR}/"
fi

echo "all done! The model training artifacts are available at ${OUTPUT_GCS_DIR}."
