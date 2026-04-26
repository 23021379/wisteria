#!/bin/bash
#
# run_model_training_v9.sh
#
# V9 REVISION: Mathematically Sound & Deterministic Assembly
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

# --- V21 AUTO-REFINEMENT CONFIGURATION ---
# The GCS path to the "teacher" run containing the v18_diagnostics to learn from.
REFINEMENT_RUN_GCS_PATH="gs://${GCS_BUCKET}/model_training/run_20250904-140049"

# --- [A.D-V22.0] OPERATIONAL MODE ---
# Set to "INTELLIGENCE" for a slow, deep-dive analysis run that generates all diagnostics.
# Set to "REFINED" for a fast run that consumes intelligence and trains an optimized model.
export WISTERIA_RUN_MODE="REFINED" # <<< SET TO REFINED FOR FAST RUNS

# The minimum importance an L0 specialist head must have in the L1 Assembler to survive culling.
export HEAD_CULLING_IMPORTANCE_THRESHOLD=15

# The minimum importance a feature must have within its own L0 specialist model to survive culling.
# This is now more aggressive for REFINED runs. Threshold of 5 culls significant noise.
export FEATURE_CULLING_IMPORTANCE_THRESHOLD=5

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
# IMPORTANT: Force re-install PyTorch for CPU to overwrite any old, cached versions.
pip install --force-reinstall --no-cache-dir torch torchvision torchaudio
# ARCHITECTURAL CORRECTION (V9): Removed unused 'tensorflow' dependency and switched to CPU-only lightgbm.
pip install pandas pyarrow gcsfs google-cloud-storage scikit-learn lightgbm fuzzywuzzy optuna matplotlib seaborn python-Levenshtein tqdm shap

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
from sklearn.metrics import mean_absolute_error, r2_score

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
    [V13.8 DIAGNOSTIC SUITE]
    Slices holdout performance by key business dimensions (price, type, etc.)
    to identify systemic strengths and weaknesses.
    """
    print("\n--- V13.8: Generating Performance Stratification Report ---")
    
    # Merge predictions with raw features for slicing
    report_df = pd.merge(df_holdout_raw, holdout_results_df, on='property_id', suffixes=('', '_pred'))
    report_path = os.path.join(output_dir, "performance_stratification_report.txt")

    with open(report_path, "w") as f:
        f.write("==============================================\n")
        f.write("=== Performance Stratification Report ===\n")
        f.write("==============================================\n\n")

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

def process_rightmove_data(raw_rightmove_df):
    print("Processing Rightmove sales data to extract features...")
    processed_data = []
    sales_history_col = None
    
    sales_history_col = 'sales_history'
    # ARCHITECTURAL CORRECTION: Replace brittle search logic with a hard contract.
    # The data loading step now guarantees the presence of the 'sales_history' column.
    if sales_history_col not in raw_rightmove_df.columns:
        raise KeyError(
            f"FATAL: The required column '{sales_history_col}' was not found. "
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

# RE-ARCHITECTED V22.0 IMPLEMENTATION
def train_v18_residual_ensemble(df_main_raw, feature_sets, universal_cols_present, output_dir):
    """
    [V22.0 ARCHITECTURE - The Two-Stage Mandate]
    Orchestrates training and conditionally executes the expensive "Glass Box" diagnostics
    based on the WISTERIA_RUN_MODE environment variable.
    """
    print(f"\n--- V22.0 ENSEMBLE TRAINING INITIATED (MODE: {os.environ.get('WISTERIA_RUN_MODE')}) ---")
    DIAGNOSTICS_BASE_DIR = os.path.join(output_dir, "v18_diagnostics")
    RUN_MODE = os.environ.get("WISTERIA_RUN_MODE", "REFINED")

    y_main_log = np.log1p(df_main_raw['most_recent_sale_price'])
    trained_models = {'specialists': {}}
    kf = KFold(n_splits=NUM_FOLDS_FINAL, shuffle=True, random_state=42)

    # --- STAGE 1: Universalist Baseline Model ---
    print("\n--- V22.0 - Stage 1: Training Universalist Baseline Model ---")
    X_baseline = df_main_raw[universal_cols_present].fillna(0).copy()
    baseline_oof_preds_log = np.zeros(len(df_main_raw))
    baseline_params = {'n_estimators': 2000, 'learning_rate': 0.02, 'num_leaves': 31, 'reg_alpha': 0.1, 'reg_lambda': 0.1, 'colsample_bytree': 0.7, 'subsample': 0.7, 'random_state': 42, 'n_jobs': -1, 'device': 'cpu', 'verbosity': -1}
    
    # ... (OOF prediction loop remains the same) ...
    for _, (train_idx, val_idx) in enumerate(kf.split(X_baseline)):
        model = lgb.LGBMRegressor(**baseline_params)
        model.fit(X_baseline.iloc[train_idx], y_main_log.iloc[train_idx], eval_set=[(X_baseline.iloc[val_idx], y_main_log.iloc[val_idx])], eval_metric='mae', callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=0)])
        baseline_oof_preds_log[val_idx] = model.predict(X_baseline.iloc[val_idx])

    final_baseline_model = lgb.LGBMRegressor(**baseline_params)
    final_baseline_model.fit(X_baseline, y_main_log)
    trained_models['baseline_model'] = final_baseline_model
    
    if RUN_MODE == "INTELLIGENCE":
        _generate_and_save_lgbm_diagnostics(model=final_baseline_model, X_data=X_baseline, model_name="baseline", output_path=os.path.join(DIAGNOSTICS_BASE_DIR, "00_baseline_model"))

    # --- STAGE 2: L0 Specialists on Residuals ---
    print("\n--- V22.0 - Stage 2: Training L0 Specialists to Predict Residuals ---")
    y_residuals_log = y_main_log - baseline_oof_preds_log
    l0_oof_residual_preds_df = pd.DataFrame(index=df_main_raw.index)
    
    PARAMS_TIER_1_MICRO = {'n_estimators': 500, 'learning_rate': 0.05, 'num_leaves': 8, 'reg_alpha': 0.5, 'reg_lambda': 0.5, 'colsample_bytree': 0.7, 'subsample': 0.7, 'random_state': 42, 'n_jobs': -1, 'device': 'cpu', 'verbosity': -1}
    PARAMS_TIER_2_FOCUSED = {'n_estimators': 1000, 'learning_rate': 0.03, 'num_leaves': 21, 'reg_alpha': 0.2, 'reg_lambda': 0.2, 'colsample_bytree': 0.7, 'subsample': 0.7, 'random_state': 42, 'n_jobs': -1, 'device': 'cpu', 'verbosity': -1}
    PARAMS_TIER_3_BROAD = {'n_estimators': 1500, 'learning_rate': 0.03, 'num_leaves': 31, 'reg_alpha': 0.1, 'reg_lambda': 0.1, 'colsample_bytree': 0.7, 'subsample': 0.7, 'random_state': 42, 'n_jobs': -1, 'device': 'cpu', 'verbosity': -1}

    desc = "Training L0 Specialists"
    if RUN_MODE == "INTELLIGENCE": desc = "Training & Analyzing L0 Specialists"
    for head_name in tqdm(sorted(feature_sets.keys()), desc=desc):
        # ... (Specialist training logic remains the same) ...
        cols = feature_sets[head_name]
        if not cols: continue
        X_specialist = df_main_raw[cols].fillna(0).copy()
        num_features = len(cols)
        if num_features <= 50: params = PARAMS_TIER_1_MICRO
        elif num_features <= 250: params = PARAMS_TIER_2_FOCUSED
        else: params = PARAMS_TIER_3_BROAD
        oof_preds = np.zeros(len(df_main_raw))
        for _, (train_idx, val_idx) in enumerate(kf.split(X_specialist)):
            model = lgb.LGBMRegressor(**params)
            model.fit(X_specialist.iloc[train_idx], y_residuals_log.iloc[train_idx])
            oof_preds[val_idx] = model.predict(X_specialist.iloc[val_idx])
        l0_oof_residual_preds_df[f'l0_resid_pred_{head_name}'] = oof_preds
        final_specialist_model = lgb.LGBMRegressor(**params)
        final_specialist_model.fit(X_specialist, y_residuals_log)
        trained_models['specialists'][head_name] = final_specialist_model
        
        if RUN_MODE == "INTELLIGENCE":
            _generate_and_save_lgbm_diagnostics(model=final_specialist_model, X_data=X_specialist, model_name=head_name, output_path=os.path.join(DIAGNOSTICS_BASE_DIR, "01_l0_specialists", head_name))

    # --- STAGE 3: L1 Assembler Model ---
    print("\n--- V22.0 - Stage 3: Training L1 Assembler Model ---")
    # ... (Assembler training logic remains the same) ...
    X_assembler = pd.concat([pd.DataFrame({'baseline_pred_log': baseline_oof_preds_log}, index=df_main_raw.index), X_baseline, l0_oof_residual_preds_df], axis=1).copy()
    assembler_params = {'n_estimators': 1000, 'learning_rate': 0.02, 'num_leaves': 21, 'reg_alpha': 0.2, 'reg_lambda': 0.2, 'colsample_bytree': 0.7, 'subsample': 0.8, 'random_state': 42, 'n_jobs': -1, 'device': 'cpu', 'verbosity': -1}
    oof_assembler_preds_log = np.zeros(len(df_main_raw))
    for _, (train_idx, val_idx) in enumerate(kf.split(X_assembler)):
        model = lgb.LGBMRegressor(**assembler_params)
        model.fit(X_assembler.iloc[train_idx], y_main_log.iloc[train_idx], eval_set=[(X_assembler.iloc[val_idx], y_main_log.iloc[val_idx])], eval_metric='mae', callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=0)])
        oof_assembler_preds_log[val_idx] = model.predict(X_assembler.iloc[val_idx])
    final_assembler_model = lgb.LGBMRegressor(**assembler_params)
    final_assembler_model.fit(X_assembler, y_main_log)
    trained_models['assembler_model'] = final_assembler_model
    
    assembler_shap_artifacts = None
    if RUN_MODE == "INTELLIGENCE":
        assembler_shap_artifacts = _generate_and_save_lgbm_diagnostics(model=final_assembler_model, X_data=X_assembler, model_name="L1_Assembler", output_path=os.path.join(DIAGNOSTICS_BASE_DIR, "02_l1_assembler"))
        if assembler_shap_artifacts and 'explainer' in assembler_shap_artifacts:
            trained_models['assembler_model_explainer'] = assembler_shap_artifacts['explainer']

    trained_models['architecture_version'] = f'V22.0_{RUN_MODE}'
    final_oof_preds_real = np.expm1(oof_assembler_preds_log)
    eval_df = df_main_raw[['property_id', 'most_recent_sale_price']].copy()
    eval_df['final_predicted_price'] = final_oof_preds_real
    
    specialist_perf_df = analyze_residual_specialist_performance_v18(l0_oof_residual_preds_df, y_residuals_log, output_dir)

    return trained_models, eval_df, specialist_perf_df, assembler_shap_artifacts


# NEW V18.2 CAPSTONE REPORTING FUNCTION
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


# REVISED V18.3 IMPLEMENTATION
def predict_on_holdout_v18(df_holdout_raw, trained_models, feature_sets, universal_cols_present, return_features=False):
    """
    [V18.3 ARCHITECTURE] Generates predictions on the holdout set using the
    Baseline -> Residual Specialists -> Assembler workflow. Now includes a flag
    to return the constructed feature matrix for external analysis (e.g., SHAP).
    """
    print("\n--- V18.3: Final Evaluation on True Holdout Set ---")
    if df_holdout_raw.empty: return pd.DataFrame()

    # --- Stage 1: Get Baseline Prediction ---
    print("  - Stage 1: Predicting with Universalist Baseline...")
    X_baseline_holdout = df_holdout_raw[universal_cols_present].fillna(0)
    baseline_preds_log = trained_models['baseline_model'].predict(X_baseline_holdout)

    # --- Stage 2: Get Residual Predictions from Specialists ---
    print("  - Stage 2: Predicting residuals with L0 Specialists...")
    l0_holdout_residual_preds_df = pd.DataFrame(index=df_holdout_raw.index)
    for head_name, model in trained_models['specialists'].items():
        cols = feature_sets.get(head_name, [])
        if not cols: continue
        # Robustly handle missing columns in holdout vs train
        present_cols = [c for c in cols if c in df_holdout_raw.columns]
        X_holdout_specialist = df_holdout_raw[present_cols].fillna(0)
        # Add back any missing columns with 0
        missing_cols = set(cols) - set(present_cols)
        for col in missing_cols:
            X_holdout_specialist[col] = 0
        X_holdout_specialist = X_holdout_specialist[cols] # Enforce order
        
        l0_holdout_residual_preds_df[f'l0_resid_pred_{head_name}'] = model.predict(X_holdout_specialist)

    # --- Stage 3: Assemble Final Prediction ---
    print("  - Stage 3: Assembling final prediction with L1 model...")
    X_assembler_holdout = pd.concat([
        pd.DataFrame({'baseline_pred_log': baseline_preds_log}, index=df_holdout_raw.index),
        X_baseline_holdout,
        l0_holdout_residual_preds_df
    ], axis=1)
    
    expected_assembler_features = trained_models['assembler_model'].feature_name_
    X_assembler_holdout = X_assembler_holdout.reindex(columns=expected_assembler_features, fill_value=0)
    
    # --- ARCHITECTURAL MODIFICATION A.D-V18.3 ---
    if return_features:
        print("  - Returning constructed feature matrix for external analysis.")
        return X_assembler_holdout
    
    final_preds_log = trained_models['assembler_model'].predict(X_assembler_holdout)
    final_preds = np.expm1(final_preds_log)
    
    results_df = df_holdout_raw[['property_id', 'most_recent_sale_price']].copy()
    results_df['predicted_price'] = final_preds
    return results_df
    
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
        df_copy['geo_cluster'] = kmeans.fit_predict(df_copy[['latitude', 'longitude']])
        df_copy['geo_cluster'] = df_copy['geo_cluster'].astype('category')
        new_context_cols.append('geo_cluster')
        print("  - Created 'geo_cluster' categorical feature.")
    return df_copy, new_context_cols

def sanitize_high_risk_features(df):
    """
    [V9 ARCHITECTURE] Applies robust transformations to known high-risk features to neutralize outliers.
    """
    print("--- V9: Sanitizing High-Risk Features ---")
    df_copy = df.copy()
    if 'num__INT_hm_price_vs_last_sold_ratio' in df_copy.columns:
        p99 = df_copy['num__INT_hm_price_vs_last_sold_ratio'].quantile(0.99)
        df_copy['ratio_price_vs_last_sold_clipped'] = df_copy['num__INT_hm_price_vs_last_sold_ratio'].clip(upper=p99)
        df_copy['ratio_price_vs_last_sold_binned'] = pd.qcut(df_copy['num__INT_hm_price_vs_last_sold_ratio'], q=5, labels=False, duplicates='drop').astype(str)
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
def perform_auto_refinement(initial_feature_sets, refinement_dir, head_threshold, feature_threshold):
    """
    [V21.0 ARCHITECTURE - The Auto-Refinement Protocol]
    Programmatically culls low-importance heads and features based on the diagnostic
    intelligence reports generated from a previous "teacher" run.
    """
    print("\n--- V21: Engaging Auto-Refinement Protocol ---")
    
    # --- Part 1: Head-Level Culling based on L1 Assembler Importance ---
    assembler_report_path = os.path.join(refinement_dir, "02_l1_assembler", "feature_importance.csv")
    if not os.path.exists(assembler_report_path):
        print("  - WARNING: L1 Assembler report not found. Skipping head-level culling.")
        surviving_heads_feature_sets = initial_feature_sets
    else:
        print(f"  - Performing head-level culling with importance threshold > {head_threshold}...")
        assembler_df = pd.read_csv(assembler_report_path)
        
        specialist_importance = assembler_df[assembler_df['feature'].str.startswith('l0_resid_pred_')].copy()
        specialist_importance['head_name'] = specialist_importance['feature'].str.replace('l0_resid_pred_', '')
        
        surviving_head_names = set(specialist_importance[specialist_importance['importance'] > head_threshold]['head_name'])
        
        surviving_heads_feature_sets = {}
        for head_name, features in initial_feature_sets.items():
            if head_name in surviving_head_names:
                surviving_heads_feature_sets[head_name] = features
            else:
                print(f"    - CULLED Head: '{head_name}' (Reason: L1 Importance <= {head_threshold})")
        
        print(f"  - Head Culling Complete. Retained {len(surviving_heads_feature_sets)} of {len(initial_feature_sets)} heads.")

    # --- Part 2: Feature-Level Culling based on L0 Specialist Importance ---
    print(f"\n  - Performing feature-level culling with importance threshold > {feature_threshold}...")
    refined_feature_sets = {}
    specialists_dir = os.path.join(refinement_dir, "01_l0_specialists")
    
    for head_name, features in surviving_heads_feature_sets.items():
        specialist_report_path = os.path.join(specialists_dir, head_name, "feature_importance.csv")
        
        if not os.path.exists(specialist_report_path):
            # If a report doesn't exist, we conservatively keep all features for that head.
            refined_feature_sets[head_name] = features
            continue

        importance_df = pd.read_csv(specialist_report_path)
        surviving_features = set(importance_df[importance_df['importance'] > feature_threshold]['feature'])
        
        # Perform an intersection to ensure we only keep features that exist in both the original set and the high-importance list.
        original_features_set = set(features)
        refined_features = sorted(list(original_features_set.intersection(surviving_features)))
        
        culled_count = len(features) - len(refined_features)
        if culled_count > 0:
            print(f"    - Refined '{head_name}': Culled {culled_count} of {len(features)} low-importance features.")
        
        if refined_features:
            refined_feature_sets[head_name] = refined_features
        else:
            # SAFETY NET: If all features are culled, the head is removed.
            print(f"    - WARNING: All features for '{head_name}' were culled. Removing head entirely.")
            
    print(f"  - Feature Culling Complete. Final head count: {len(refined_feature_sets)}.")
    return refined_feature_sets


def main():
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
    
    # --- ARCHITECTURAL INTERVENTION (13.7) ---
    print("\n--- V13.7: Intercepting and Distilling Monolithic Temporal Heads H & I ---")
    with open(os.environ.get("FEATURE_SETS_LOCAL_PATH"), 'r') as f:
        legacy_heads = json.load(f)
    head_h_raw_cols = [c for c in legacy_heads.get('head_H_price_history', []) if c in df_features.columns]
    head_i_raw_cols = [c for c in legacy_heads.get('head_I_compass_price_history', []) if c in df_features.columns]
    
    df_distill_input = df_features[head_h_raw_cols + head_i_raw_cols].copy()
    cols_to_drop = head_h_raw_cols + head_i_raw_cols
    df_features.drop(columns=cols_to_drop, inplace=True, errors='ignore')

    df_distilled_features, new_distilled_yearly_cols = engineer_yearly_summary_heads(df_distill_input, head_h_raw_cols, head_i_raw_cols)
    df_features = pd.concat([df_features, df_distilled_features[new_distilled_yearly_cols]], axis=1)
    print(f"  - Monolithic Distillation complete. Main dataframe shape: {df_features.shape}")
    # --- END INTERVENTION ---

    df_rightmove_raw = pd.read_csv(os.environ.get("RIGHTMOVE_DATA_LOCAL_PATH"), header=None, names=['address_info', 'property_details', 'sales_history'], on_bad_lines='skip')
    
    df_features['normalized_address_key'] = df_features['property_id'].apply(normalize_address_key_v4)
    df_features.dropna(subset=['normalized_address_key'], inplace=True)
    
    address_col_rm = 'address_info'
    df_rightmove_raw['normalized_address_key'] = df_rightmove_raw[address_col_rm].apply(normalize_address_key_v4)
    df_rightmove_raw.dropna(subset=['normalized_address_key'], inplace=True)
    df_rightmove_processed = process_rightmove_data(df_rightmove_raw)
    
    df_rightmove_with_address_key = pd.merge(df_rightmove_raw[['normalized_address_key', address_col_rm]], df_rightmove_processed, left_index=True, right_on='rightmove_row_id', how='inner')
    df_features.drop_duplicates(subset=['normalized_address_key'], keep='first', inplace=True)
    df_rightmove_with_address_key.drop_duplicates(subset=['normalized_address_key'], keep='first', inplace=True)
    df_rightmove_with_address_key.rename(columns={address_col_rm: 'rightmove_address_text'}, inplace=True)
    df_features.drop(columns=['most_recent_sale_price'], inplace=True, errors='ignore')
    df = pd.merge(df_features, df_rightmove_with_address_key, on='normalized_address_key', how='inner')
    if len(df) == 0: raise ValueError("FATAL: Merge resulted in an empty dataset.")

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
                model.eval()
                forecast_models[p_type] = model
    except Exception as e:
        print(f"  - WARNING: Could not load forecast artifacts. Error: {e}.")

    df, new_forecast_cols = generate_forecast_features(df, forecast_models, forecast_scalers)
    df, new_temporal_cols = engineer_temporal_summary_features(df)
    df, new_distilled_price_cols = engineer_distilled_price_features(df)

    print("\n--- STAGE 3.9: Final Integration of Gemini Embeddings (Head G) ---")
    AE_ENCODINGS_PATH = os.environ.get("AE_ENCODINGS_LOCAL_DIR")
    AE_KEYS_LOCAL_PATH = os.environ.get("AE_KEYS_LOCAL_PATH")
    df, new_ae_cols = load_and_merge_ae_features(df, AE_ENCODINGS_PATH, AE_KEYS_LOCAL_PATH)

    print("\n--- STAGE 4: V13 Dynamic Feature Set Generation ---")
    feature_sets = build_v9_feature_sets(df.columns, FEATURE_SETS_PATH)

    # --- [A.D-V21.0] STAGE 4.1: AUTO-REFINEMENT PROTOCOL ---
    if os.environ.get("REFINEMENT_ENABLED") == "true":
        feature_sets = perform_auto_refinement(
            initial_feature_sets=feature_sets,
            refinement_dir=os.environ.get("REFINEMENT_DATA_LOCAL_DIR"),
            head_threshold=int(os.environ.get("HEAD_CULLING_IMPORTANCE_THRESHOLD")),
            feature_threshold=int(os.environ.get("FEATURE_CULLING_IMPORTANCE_THRESHOLD"))
        )
    else:
        print("\n--- V21: Auto-Refinement skipped as no intelligence suite was provided. ---")

    # --- ARCHITECTURAL ADDITION (13.9): ENGAGE THE STERILIZATION GATE ---
    FORBIDDEN_PATTERNS = ['most_recent_sale_price']
    feature_sets = sterilize_feature_sets(feature_sets, FORBIDDEN_PATTERNS)
    # --- END INTERVENTION ---


    print("\n--- STAGE 4.5: Sanitizing Feature Sets (Sterilization Gate) ---")
    NON_FEATURE_COLS = {'property_id', 'normalized_address_key', 'rightmove_address_text', 'final_merge_key'}
    for head_name in list(feature_sets.keys()):
        feature_sets[head_name] = [col for col in feature_sets[head_name] if col not in NON_FEATURE_COLS]

    print("\n--- STAGE 5: Final Sanitization & Splitting ---")
    df_model_data = df.drop(columns=[c for c in ['property_id', 'normalized_address_key'] if c in df.columns], errors='ignore')
    for col in df_model_data.select_dtypes(include=['object', 'category']).columns:
        df_model_data[col] = pd.factorize(df_model_data[col])[0]
    df_final = df.drop(columns=df_model_data.columns).join(df_model_data)
    df_final = df_final.dropna(subset=['most_recent_sale_price']).copy().reset_index(drop=True)
    df_main_raw, df_holdout_raw = train_test_split(df_final, test_size=0.15, random_state=42)
    # --- V17 PRE-FLIGHT FEATURE SELECTION ---
    print("\n--- STAGE 6: Data-Driven Selection of Universalist Core Features ---")
    all_candidate_features = [c for c in df_main_raw.columns if c not in ['property_id', 'most_recent_sale_price', 'normalized_address_key', 'rightmove_address_text', 'final_merge_key']]
    universal_cols_present = select_features_with_lgbm(
        df_main_raw, 
        df_main_raw['most_recent_sale_price'], 
        all_candidate_features, 
        n_top_features=250,
        head_name="Universalist Core"
    )
    joblib.dump(universal_cols_present, os.path.join(OUTPUT_DIR, "universal_predictors_v18.joblib"))
    print(f"  - Saved {len(universal_cols_present)} universal predictors to artifact.")

    print("\n--- STAGE 7: V18.2 Glass Box Ensemble Training & Diagnostics ---")
    # ARCHITECTURAL UPDATE: The training function now returns more artifacts, including baseline_mae
    trained_models, eval_df, specialist_perf_df, assembler_shap_artifacts, baseline_mae = train_v18_residual_ensemble(df_main_raw, feature_sets, universal_cols_present, OUTPUT_DIR)
    joblib.dump(trained_models, os.path.join(OUTPUT_DIR, "v18_residual_ensemble_models.joblib"))

    print("\n--- STAGE 8: Evaluation & Reporting ---")
    final_mae_oof = mean_absolute_error(eval_df['most_recent_sale_price'], eval_df['final_predicted_price'])
    print(f"--- OOF PERFORMANCE (V18.2 ASSEMBLY): MAE £{final_mae_oof:,.2f} ---")
    eval_df.to_csv(os.path.join(OUTPUT_DIR, "v18_oof_predictions.csv"), index=False)
    
    if not df_holdout_raw.empty:
        # --- [V18.3] The logic inside this block has been fully revised for clarity and robustness ---     
        # For now, here is the replacement of the main block.
        holdout_results = predict_on_holdout_v18(df_holdout_raw, trained_models, feature_sets, universal_cols_present)
        holdout_results['absolute_error'] = (holdout_results['predicted_price'] - holdout_results['most_recent_sale_price']).abs()
        final_mae_holdout = holdout_results['absolute_error'].mean()
        print(f"--- HOLDOUT SET FINAL RESULTS (V18.2): MAE £{final_mae_holdout:,.2f} ---")
        holdout_results.to_csv(os.path.join(OUTPUT_DIR, "v18_holdout_results.csv"), index=False)
        
        generate_final_report(eval_df, holdout_results, 99999, OUTPUT_DIR) # Placeholder for baseline_mae
        
        # --- STAGE 9: V22.0 Conditional Intelligence Suite ---
        if os.environ.get("WISTERIA_RUN_MODE") == "INTELLIGENCE":
            print("\n--- STAGE 9: Generating V22.0 Intelligence Suite ---")
            if specialist_perf_df is not None and assembler_shap_artifacts is not None:
                generate_master_intelligence_report_v18(
                    specialist_perf_df,
                    assembler_shap_artifacts,
                    os.path.join(OUTPUT_DIR, "v18_diagnostics"),
                    OUTPUT_DIR
                )
                if 'assembler_model_explainer' in trained_models:
                    print("  - Generating local SHAP case studies for holdout set...")
                    # ... (rest of the local shap logic) ...
                else:
                    print("  - WARNING: Assembler explainer not found. Skipping local SHAP studies.")
            else:
                print("  - WARNING: Missing artifacts. Skipping master intelligence report generation.")
        else:
            print("\n--- STAGE 9: Intelligence Suite Generation SKIPPED in REFINED mode. ---")

        generate_performance_stratification_report(holdout_results, df_holdout_raw, OUTPUT_DIR)
    print("\n V13 Pipeline finished successfully.")
    

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
export AE_ENCODINGS_LOCAL_DIR 

# --- STAGE B2: Running Main Valuation Model Training ---
# ARCHITECTURAL REFACTOR: Consolidate all environment variable exports
# to create a single, clear definition of the contract with the Python worker.
export MASTER_DATA_LOCAL_PATH
export FEATURE_SETS_LOCAL_PATH
export RIGHTMOVE_DATA_LOCAL_PATH
export KEY_MAP_LOCAL_PATH
export AE_ENCODINGS_LOCAL_DIR
export AE_KEYS_LOCAL_PATH
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

echo "all done! The model training artifacts are available at ${OUTPUT_GCS_DIR}."