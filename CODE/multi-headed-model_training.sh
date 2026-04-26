#!/bin/bash
#
# run_feature_sorter_v4.sh
#
# FINAL REVISION V4: Adds more sophisticated rules to Pass 2 to handle
# one-hot encoded features, embeddings, and other leftovers from the previous run.
# This should result in zero unassigned features.
#

# -- Strict Mode & Configuration --
set -e
set -o pipefail
set -x

# -- GCP & Project Configuration --
GCS_BUCKET="srgan-bucket-ace-botany-453819-t4"
TIMESTAMP=$(date +"%Y%m%d-%H%M%S")
OUTPUT_GCS_DIR="gs://${GCS_BUCKET}/feature_sorting/run_legacy" 
MASTER_DATA_GCS_PATH="gs://${GCS_BUCKET}/imputation_pipeline/output_lgbm_legacy/final_fully_imputed_dataset.parquet" #full dataset

# Define the GCS paths to the ORIGINAL, UN-AGGREGATED data files.
SOURCE_HEADER_PATHS=(
    "gs://${GCS_BUCKET}/house data scrape/subset1_processed_full.parquet"
    "gs://${GCS_BUCKET}/house data scrape/subset2_processed_full.parquet"
    "gs://${GCS_BUCKET}/house data scrape/subset3_processed_full.parquet"
    "gs://${GCS_BUCKET}/house data scrape/subset4_processed_full.parquet"
    "gs://${GCS_BUCKET}/house data scrape/subset5_processed_full.parquet"
    "gs://${GCS_BUCKET}/house data scrape/subset_pp_history_processed.parquet" # <-- NEW
    "gs://${GCS_BUCKET}/house data scrape/cleaned_property_data_gwr_with_coords_label.csv"
    "gs://${GCS_BUCKET}/house data scrape/property_features_quantitative_v4.csv"
    "gs://${GCS_BUCKET}/features/geospatial_pipeline_v16_full_with_pp/contextual_features/contextual_features.parquet" # <-- UPDATED CONTEXTUAL PATH
)

# "gs://${GCS_BUCKET}/features/geospatial_pipeline_property_aware_20250712-144826/contextual_features/contextual_features.parquet" is the ACM output.

# --- Local Configuration ---
PROJECT_DIR="./feature_sorter_project"
VENV_DIR="${PROJECT_DIR}/venv_fs"
OUTPUT_DIR="${PROJECT_DIR}/output"
DATA_DIR="${PROJECT_DIR}/data"
MASTER_DATA_LOCAL_PATH="${DATA_DIR}/master_dataset.parquet"
AE_ENCODINGS_LOCAL_DIR="${DATA_DIR}/ae_encodings_head_g" # <-- NEW: Local dir for Head G features
OUTPUT_JSON_PATH="${OUTPUT_DIR}/feature_sets.json"
SCRIPT_PATH="${PROJECT_DIR}/01_sort_features_v4.py"

# --- Create Local Directory Structure ---
mkdir -p "${PROJECT_DIR}" "${OUTPUT_DIR}" "${DATA_DIR}"
cd "${PROJECT_DIR}"

# --- Environment Setup ---
if [ ! -d "${VENV_DIR}" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv "${VENV_DIR}"
fi
source "${VENV_DIR}/bin/activate"

echo "Installing required Python packages..."
pip install --upgrade pip
pip install pandas pyarrow gcsfs google-cloud-storage

# --- Python Worker Script Generation (v4) ---
cat > "${SCRIPT_PATH}" <<'EOF'
import os
import re
import json
import warnings
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import gcsfs
from google.cloud import storage

# --- Configuration ---
MASTER_DATA_LOCAL_PATH = os.environ.get("MASTER_DATA_LOCAL_PATH")
OUTPUT_JSON_PATH = os.environ.get("OUTPUT_JSON_PATH")
SOURCE_HEADER_URIS = os.environ.get("SOURCE_HEADER_URIS").split(',')
AE_ENCODINGS_PATH = os.environ.get("AE_ENCODINGS_LOCAL_DIR")

# Maps source filenames to their designated model head
SOURCE_TO_HEAD_MAP = {
    "cleaned_property_data_gwr_with_coords_label.csv": "head_A_dna",
    # "property_features_quantitative_v4.csv" is INTENTIONALLY REMOVED.
    # It contains a mix of DNA and Aesthetic features that must be sorted by the
    # rule-based logic in Pass 2, not assigned wholesale to a single head here.
    "subset1_processed_full.parquet": "head_C_census",
    "subset2_processed_full.parquet": "head_C_census",
    "subset4_processed_full.parquet": "head_C_census",
    "subset5_processed_full.parquet": "head_C_census",
    # Price paid data will be handled by a specific rule in Pass 2
    "subset_pp_history_processed.parquet": "head_H_price_history",
}

def generate_head_g_features(encodings_dir):
    """
    Generates feature names for Head G by inspecting downloaded autoencoder .npy files.
    """
    print("\n--- Generating features for Head G from Autoencoder outputs... ---")
    head_g_features = []
    if not encodings_dir or not os.path.exists(encodings_dir):
        print(f"  - WARNING: Encodings directory not found at '{encodings_dir}'. Head G will be empty.")
        return head_g_features

    for filename in sorted(os.listdir(encodings_dir)):
        if filename.endswith(".npy"):
            try:
                group_name = filename.replace("_encodings.npy", "")
                file_path = os.path.join(encodings_dir, filename)
                data = np.load(file_path, mmap_mode='r')
                latent_dim = data.shape[1]
                print(f"  - Found group '{group_name}' with latent dimension {latent_dim}.")
                feature_names = [f"ae_{group_name}_{i}" for i in range(latent_dim)]
                head_g_features.extend(feature_names)
            except Exception as e:
                print(f"  - ERROR: Could not process file {filename}. Error: {e}")
    
    print(f"  - Generated a total of {len(head_g_features)} features for Head G.")
    return head_g_features

def get_source_headers_remotely(uri_list):
    """Reads headers/schemas of files directly from GCS without downloading them."""
    print("--- Reading headers REMOTELY from source files in GCS... ---")
    headers_dict = {}
    gcs = gcsfs.GCSFileSystem()

    for uri in uri_list:
        filename = os.path.basename(uri)
        try:
            print(f"  - Reading schema for: {filename}")
            if uri.endswith('.parquet'):
                with gcs.open(uri, 'rb') as f:
                    columns = pq.ParquetFile(f).schema.names
            elif uri.endswith('.csv'):
                storage_client = storage.Client()
                bucket_name, blob_name = uri.replace("gs://", "").split("/", 1)
                blob = storage_client.bucket(bucket_name).blob(blob_name)
                header_line = blob.download_as_bytes().splitlines()[0].decode('utf-8')
                columns = header_line.strip().split(',')
            else:
                continue
            
            sanitized_columns = [re.sub(r'[^A-Za-z0-9_]+', '_', col) for col in columns]
            headers_dict[filename] = sanitized_columns
            print(f"    - Success: Found {len(sanitized_columns)} columns.")
        except Exception as e:
            print(f"    - ERROR: Could not read header from {uri}. Error: {e}")
    return headers_dict

def organize_features_for_heads(master_column_list, source_headers):
    """Sorts a LIST OF STRINGS (column names) with refined rules for sub-heads."""
    print("\n--- Starting Feature Sorting Process on Column List (Sub-Head Strategy) ---")

    feature_sets = {
        "head_A_dna": [], "head_B_aesthetic": [], "head_C_census": [], "head_atlas": [],
        "head_compass": [], "head_microscope": [], "head_E_temporal": [], "head_G_gemini_quantitative": [],
        "head_H_price_history": [], "head_I_compass_price_history": [], "unassigned_features": []
    }
    property_types = ['D', 'S', 'T', 'F', 'O']
    for p_type in property_types:
        feature_sets[f"head_F_spatio_temporal_{p_type}"] = []

    all_master_columns = set(master_column_list)
    assigned_columns = set()

    # Pass 0: Pre-populate Head G from external autoencoder features
    head_g_features = generate_head_g_features(AE_ENCODINGS_PATH)
    feature_sets["head_G_gemini_quantitative"].extend(head_g_features)

    # Pass 1: Direct Matching from master dataset
    print("\n--- PASS 1: Assigning features based on direct source match... ---")
    for filename, columns in source_headers.items():
        if filename not in SOURCE_TO_HEAD_MAP: continue
        target_head = SOURCE_TO_HEAD_MAP[filename]
        found_cols = {col for col in columns if col in all_master_columns} - assigned_columns
        feature_sets[target_head].extend(list(found_cols))
        assigned_columns.update(found_cols)
        print(f"  -> Assigned {len(found_cols)} new features to '{target_head}' from '{filename}'.")

    # Pass 2: Rule-Based Matching for remaining master dataset columns
    print("\n--- PASS 2: Assigning leftover features with CORRECTED rules... ---")
    unassigned_pass1 = sorted(list(all_master_columns - assigned_columns))

    spatio_temporal_pattern = re.compile(r"^(compass|atlas|microscope)_.*?_(\d{4})_.*?_([DSTFO])_n\d+")
    temporal_pattern = re.compile(r"^\d{4}_|^(HPI_Adjusted|Sale_Count)_(D|S|T|F)$")
    property_dna_pattern = re.compile(r'avm|homipi|mouseprice|bnl|gemini_property_persona', re.IGNORECASE)
    aesthetic_quant_pattern = re.compile(r'^(persona_|primary_|other_|num_images_total|num_rooms_identified_in_step5|std_dev_persona_rating_overall|avg_persona_rating_overall)')
    gemini_thesis_pattern = re.compile(r'^(Composite_|Inter_|Mismatch_|Opportunity_|Ratio_|Risk_|Thesis_|Tradeoff_|Persona_Entertainer_Score|Persona_FixerUpper_Score|CoordProduct_|FI_|bba|y_)')
    build_period_pattern = re.compile(r'^BP_')

    for col in unassigned_pass1:
        if col.startswith('pp_'):
            feature_sets["head_H_price_history"].append(col); assigned_columns.add(col); continue
        
        st_match = spatio_temporal_pattern.match(col)
        if st_match:
            p_type = st_match.group(3)
            feature_sets[f"head_F_spatio_temporal_{p_type}"].append(col); assigned_columns.add(col); continue
        
        if aesthetic_quant_pattern.search(col):
            feature_sets["head_B_aesthetic"].append(col); assigned_columns.add(col); continue
        
        if temporal_pattern.search(col) or re.search(r'^(number_bedrooms|number_rooms|ownership)_.*_(\d)$', col) or col.startswith(('SFU_', 'UF_', 'Overall_Availability', 'Service_Quality_Score', 'Years_to_Full_UF')):
            feature_sets["head_E_temporal"].append(col); assigned_columns.add(col); continue
        
        if col.startswith('atlas_'):
            feature_sets["head_atlas"].append(col); assigned_columns.add(col); continue
        if col.startswith('compass_') and ('_pp_' in col):
            feature_sets["head_I_compass_price_history"].append(col); assigned_columns.add(col); continue
        if col.startswith('compass_'):
            feature_sets["head_compass"].append(col); assigned_columns.add(col); continue
        if col.startswith('microscope_') and not col.startswith('microscope_emb_'):
            feature_sets["head_microscope"].append(col); assigned_columns.add(col); continue
        
        if (col.startswith(('cat_', 'missingindicator_', 'MODE1_', 'MODE2_', 'num_', 'microscope_emb_', 'emb_')) or
            property_dna_pattern.search(col) or gemini_thesis_pattern.search(col) or
            build_period_pattern.search(col) or
            any(x in col.lower() for x in ['pcd_', 'property_id', 'property_address', 'atlas_cluster_id', 'latitude', 'longitude', 'most_recent_sale_price'])):
            feature_sets["head_A_dna"].append(col); assigned_columns.add(col); continue
        
        if col.startswith(('LAD23NM_', 'MSOA_', 'MSOA11_', 'OA11_', 'OA21_', 'WZ11_', 'Type_of_central_heating_in_household_', 'la23cd_', 'deprivation', 'disability', 'education_deprivation', 'employment_deprivation', 'ethnicity_', 'health_condition', 'health_deprivation', 'household_', 'housing_', 'language_', 'multiple_', 'number_', 'occupancy_', 'ons_army', 'reference_person', 'religion_', 'vehicles_', 'dependent_children', 'house_types', 'disabilty', 'AHAH', 'LSOA_', 'PC_', 'MSOA', 'EVI_', 'NDVI_', 'Dow_', 'LAD_OAC', 'NDVIRange', 'Postcode_Age', 'Weighted_LSOA', 'ALL_PROPERTIES', 'AccessibleCleanGreenSpace', 'AirPollutionGradient', 'BlueSpaceInCleanAir', 'Churn_in_LSOA', 'CoreHealthAccessSum', 'CumulativeHazardAccess', 'EffectiveHealthAccess', 'GamblingVsLeisure', 'GreenAirQuality', 'GreenDiversityInPollution', 'GreenHomogeneity', 'GreenVsBlue', 'GreenVsRetail', 'HealthEnvironmentSynergy', 'HealthVsAirRankDiff', 'HealthyGPVisit', 'NO2vsPM10', 'PollutionAndCareDeficit', 'PollutionDominance', 'SocialAmenityAccess', 'UnhealthyRetail', 'VEG_FRAC', 'VegFrac', 'ah4')):
            feature_sets["head_C_census"].append(col); assigned_columns.add(col); continue

    feature_sets["unassigned_features"] = sorted(list(all_master_columns - assigned_columns))
    final_feature_sets = {k: v for k, v in feature_sets.items() if v or k == "unassigned_features"}
    for p_type in property_types:
        key_to_check = f"head_F_spatio_temporal_{p_type}"
        if key_to_check in final_feature_sets and not final_feature_sets[key_to_check]:
            del final_feature_sets[key_to_check]
    return final_feature_sets

def main():
    """Main execution function."""
    warnings.filterwarnings('ignore')
    
    source_headers = get_source_headers_remotely(SOURCE_HEADER_URIS)
    
    print(f"\n--- Loading SCHEMA ONLY from master dataset: {MASTER_DATA_LOCAL_PATH} ---")
    try:
        parquet_file = pq.ParquetFile(MASTER_DATA_LOCAL_PATH)
        master_column_list = parquet_file.schema.names
        print(f"  - Schema loaded successfully. Found {len(master_column_list)} total columns.")
    except Exception as e:
        print(f"FATAL: Could not read master Parquet schema. Error: {e}"); exit(1)

    final_feature_sets = organize_features_for_heads(master_column_list, source_headers)
    
    print("\n\n=============== FINAL FEATURE SORTING REPORT ================")
    total_features = 0
    for head, features in final_feature_sets.items():
        # Deduplicate and sort lists for consistent output
        final_feature_sets[head] = sorted(list(set(features)))
        count = len(final_feature_sets[head])
        print(f"  - Head '{head}': {count} features")
        if head != "unassigned_features":
            total_features += count

    print(f"----------------------------------------------------------")
    print(f"  Total Assigned Features: {total_features}")
    
    unassigned_count = len(final_feature_sets["unassigned_features"])
    if unassigned_count > 0:
        print(f"\n--- WARNING: Found {unassigned_count} Unassigned Features ---")
        for feature in final_feature_sets["unassigned_features"]: print(f"  - {feature}")
    else:
        print("\n--- SUCCESS: All features were successfully assigned to a model head! ---")
    print("==========================================================")
    
    output_dir = os.path.dirname(OUTPUT_JSON_PATH)
    os.makedirs(output_dir, exist_ok=True)
    with open(OUTPUT_JSON_PATH, 'w') as f:
        json.dump(final_feature_sets, f, indent=4)
    print(f"\n--- Feature set definitions saved to: {OUTPUT_JSON_PATH} ---")

if __name__ == "__main__":
    main()
EOF

# --- Pipeline Execution ---
echo "Downloading master dataset from GCS (This is small and safe)..."
gsutil cp "${MASTER_DATA_GCS_PATH}" "${MASTER_DATA_LOCAL_PATH}"

# --- NEW STAGE: Download Autoencoder Encodings for Head G ---
echo "--- Downloading Autoencoder Encodings for Head G ---"
MODULAR_AE_OUTPUT_BASE_GCS="gs://${GCS_BUCKET}/models/modular_autoencoders"
mkdir -p "${AE_ENCODINGS_LOCAL_DIR}"

# Find all subdirectories containing an encodings.npy file and copy/rename it
ENCODING_FILES=$(gsutil ls "${MODULAR_AE_OUTPUT_BASE_GCS}/*/encodings.npy")
if [ -z "$ENCODING_FILES" ]; then
    echo "WARNING: No 'encodings.npy' files found in ${MODULAR_AE_OUTPUT_BASE_GCS}. Head G will be empty."
else
    for gcs_path in $ENCODING_FILES; do
        group_name=$(basename "$(dirname "$gcs_path")")
        echo "  - Downloading encoding for group: ${group_name}"
        gsutil -q cp "${gcs_path}" "${AE_ENCODINGS_LOCAL_DIR}/${group_name}_encodings.npy"
    done
fi
# --- END NEW STAGE ---

echo "Running Python worker script to sort features (V4 - Refined Rules)..."
# Export environment variables for the Python script
export MASTER_DATA_LOCAL_PATH
export OUTPUT_JSON_PATH
export SOURCE_HEADER_URIS=$(IFS=,; echo "${SOURCE_HEADER_PATHS[*]}")
export AE_ENCODINGS_LOCAL_DIR # <-- NEW: Export the path for the Python script

python3 "${SCRIPT_PATH}"

echo "Uploading feature set definition JSON to GCS..."
gsutil cp "${OUTPUT_JSON_PATH}" "${OUTPUT_GCS_DIR}/"

echo "Feature sorting finished. Results are in ${OUTPUT_GCS_DIR}/feature_sets.json"

# --- ADD THESE DIAGNOSTIC COMMANDS HERE ---
echo ""
echo "=== DIAGNOSTIC: Checking Key Mapping File Structure ==="
echo "First 5 lines of key mapping file:"
gsutil cat "gs://srgan-bucket-ace-botany-453819-t4/house data scrape/merged_property_data_with_coords.csv" | head -5

echo ""
echo "=== DIAGNOSTIC: Checking Feature Dataset Columns ==="
python3 -c "
import pandas as pd
df = pd.read_parquet('${MASTER_DATA_LOCAL_PATH}')
print('Total columns in features dataset:', len(df.columns))
print('')
print('Columns with ID potential:')
id_cols = [col for col in df.columns if any(x in col.lower() for x in ['id', 'index', 'property', 'address', 'postcode', 'lat', 'lng', 'coord', 'pcd'])]
for col in id_cols[:20]:
    print(f'  - {col}')
if len(id_cols) > 20:
    print(f'  ... and {len(id_cols) - 20} more')

print('')
print('Sample of first 10 columns:')
for col in df.columns[:10]:
    print(f'  - {col}')
"

echo ""
echo "=== DIAGNOSTIC: Checking Rightmove File Structure ==="
echo "First 3 lines of Rightmove file:"
gsutil cat "gs://srgan-bucket-ace-botany-453819-t4/house data scrape/Rightmove.csv" | head -3

echo "Diagnostics complete. Now uploading feature sets..."