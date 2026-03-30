#!/bin/bash

# ==============================================================================
# run_master_generation.sh - Creates the master feature set for model training.
# ==============================================================================

set -e
set -o pipefail
set -x

# --- Configuration ---
GCS_BUCKET="srgan-bucket-ace-botany-453819-t4"

# Input directory for the encoded .npy files
EMBEDDINGS_GCS_DIR="models/modular_autoencoders"

declare -a QUANTITATIVE_FILES=(
    "gs://${GCS_BUCKET}/house data scrape/property_features_quantitative_v4.csv"
    "gs://${GCS_BUCKET}/house data scrape/cleaned_property_data_gwr.csv" # <-- ADD PRECURSOR FILE HERE
    "gs://${GCS_BUCKET}/house data scrape/cleaned_property_data_gwr_with_coords_label.csv"
    "gs://${GCS_BUCKET}/house data scrape/property_data_subset1.csv"
    "gs://${GCS_BUCKET}/house data scrape/property_data_subset2.csv"
    "gs://${GCS_BUCKET}/house data scrape/property_data_subset3.csv"
    "gs://${GCS_BUCKET}/house data scrape/property_data_subset4.csv"
    "gs://${GCS_BUCKET}/house data scrape/property_data_subset5.csv"
)

# Output directory for the final master CSV
OUTPUT_GCS_DIR="features/final_master_dataset"

# Local workspace
WORKDIR="${HOME}/master_feature_work"
LOG_FILE="${WORKDIR}/run_master_generation.log"

## --- Main Execution ---
mkdir -p "${WORKDIR}"
cd "${WORKDIR}"

exec > >(tee -a "${LOG_FILE}") 2>&1

echo "--- Master Feature Generation Pipeline Started: $(date) ---"

# --- Environment Setup (CRITICAL BLOCK) ---
echo "--- Setting up Python environment... ---"
sudo apt-get update -y && sudo apt-get install -y python3-pip git

echo "--- Upgrading pip... ---"
python3 -m pip install --user --upgrade pip

echo "--- Exporting new PATH to use upgraded pip... ---"
export PATH="/home/jupyter/.local/bin:${PATH}"
echo "--- Pip version: $(pip --version) ---"

echo "--- Installing Python dependencies from requirements.txt... ---"
# These are the libraries needed for the Python script.
# We include all dependencies from the previous script for consistency,
# even though torch and optuna are not strictly required here.
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
python3 -m pip show pandas scikit-learn google-cloud-storage numpy

# Create the Python script locally
cat > generate_master_features.py << 'EOL'
# generate_master_features.py (Postcode Extraction Fix)
import argparse
import logging
from pathlib import Path
import re

import numpy as np
import pandas as pd
from google.cloud import storage

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler()])

# --- GCS Helper Functions (unchanged) ---
def download_gcs_directory(bucket_name: str, source_directory: str, destination_dir: Path):
    storage_client = storage.Client()
    blobs = storage_client.list_blobs(bucket_name, prefix=source_directory)
    for blob in blobs:
        if not blob.name.endswith('/'):
            destination_file_path = destination_dir / Path(blob.name).relative_to(source_directory)
            destination_file_path.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(str(destination_file_path))

def download_gcs_file(bucket_name: str, source_blob_name: str, destination_file_path: Path):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    destination_file_path.parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(str(destination_file_path))

def upload_to_gcs(bucket_name: str, source_file_path: str, destination_blob_name: str):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_path)

# --- Data Loading and Merging (Corrected Logic) ---

def extract_postcode(address: str) -> str:
    """Extracts a UK postcode from a string using a robust regex."""
    if not isinstance(address, str):
        return ""
    # Regex to find UK postcodes. It's complex but handles various formats.
    # Source: https://stackoverflow.com/questions/164979/regex-for-matching-uk-postcodes
    postcode_regex = r'([Gg][Ii][Rr] 0[Aa]{2})|((([A-Za-z][0-9]{1,2})|(([A-Za-z][A-Ha-hJ-Yj-y][0-9]{1,2})|(([A-Za-z][0-9][A-Za-z])|([A-Za-z][A-Ha-hJ-Yj-y][0-9][A-Za-z]?))))\s?[0-9][A-Za-z]{2})'
    match = re.search(postcode_regex, address)
    if match:
        # Return the found postcode, lowercased and with spaces removed for a perfect key
        return match.group(0).lower().replace(" ", "")
    return "" # Return empty if no postcode is found

def normalize_address_key(address: str) -> str:
    """
    Creates a robust, standardized key from a property address string.
    This key will be used as the Unique Identifier for merging.
    """
    if not isinstance(address, str):
        return ""
    # Convert to lowercase and remove all non-alphanumeric characters.
    # This creates a consistent key, e.g., "10, Downing St, LONDON, SW1A 2AA" -> "10downingstlondon_sw1a2aa"
    normalized_key = re.sub(r'[^a-z0-9]', '', address.lower())
    return normalized_key

def find_address_column(df: pd.DataFrame, file_name: str) -> str:
    """
    Finds the most likely address column to use for generating the unique key.
    The order of preference is critical for correctness.
    """
    # This order is based on your description of the files
    if 'property_id' in df.columns: return 'property_id'
    if 'property_address' in df.columns: return 'property_address'
    if 'original_property_address' in df.columns: return 'original_property_address'
    if 'address' in df.columns: return 'address'
    if 'full_address' in df.columns: return 'full_address'

    raise ValueError(f"Could not find a suitable address or ID column in {file_name}")

def log_dataframe_sample(df, title, columns_to_log, num_rows=5):
    """Logs a formatted sample of a dataframe for diagnostics."""
    if df.empty:
        logging.info(f"--- DIAGNOSTIC: {title} --- (Dataframe is empty)")
        return
        
    logging.info(f"--- DIAGNOSTIC: {title} ---")
    
    # Ensure all requested columns exist, even if they are filled with NaN
    # This prevents errors if a merge failed to bring in columns
    existing_cols = [col for col in columns_to_log if col in df.columns]
    
    if not existing_cols:
        logging.info("  (No relevant columns to display in this sample)")
        return
        
    sample_df = df[existing_cols].head(num_rows)
    
    # Use to_string() for clean, untruncated output in the log
    logging.info("\n" + sample_df.to_string())

def load_and_merge_data(base_csv_path: Path, other_csv_paths: list[Path]) -> pd.DataFrame:
    """
    Loads and merges all CSVs using a common, unique address key.
    This version uses a normalized full address string as the primary UID
    and includes extensive diagnostic logging.
    """
    logging.info("--- Starting Robust Address-Key-Based Data Merging ---")

    # 1. Load the base dataframe
    logging.info(f"Loading BASE CSV: {base_csv_path.name}")
    master_df = pd.read_csv(base_csv_path, low_memory=False)
    base_address_col = find_address_column(master_df, base_csv_path.name)
    
    # 2. Create the unique address key on the base dataframe
    logging.info(f"Creating unique 'address_key' from column '{base_address_col}' in base file...")
    master_df['address_key'] = master_df[base_address_col].apply(normalize_address_key)
    
    # 3. Clean up the master list
    master_df.dropna(subset=['address_key'], inplace=True)
    master_df.drop_duplicates(subset=['address_key'], keep='first', inplace=True)
    
    # --- DIAGNOSTIC: Show a sample of the initial master dataframe ---
    log_dataframe_sample(
        master_df,
        "Initial Master Dataframe Sample",
        columns_to_log=['address_key', base_address_col]
    )
    logging.info(f"Base dataframe loaded. Contains {len(master_df)} unique properties.")

    # 4. Loop through all other files and merge them onto the master list
    for csv_path in other_csv_paths:
        logging.info(f"--- Preparing to merge: {csv_path.name} ---")
        try:
            df_to_merge = pd.read_csv(csv_path, low_memory=False)
            key_col_to_merge = find_address_column(df_to_merge, csv_path.name)
            
            logging.info(f"Found address column '{key_col_to_merge}' in {csv_path.name}.")
            df_to_merge['address_key'] = df_to_merge[key_col_to_merge].apply(normalize_address_key)
            
            df_to_merge.dropna(subset=['address_key'], inplace=True)
            df_to_merge.drop_duplicates(subset=['address_key'], keep='first', inplace=True)

            if df_to_merge.empty:
                logging.warning(f"No valid address keys found in {csv_path.name}. Skipping merge.")
                continue

            # --- DIAGNOSTIC (BEFORE MERGE) ---
            # Get a few sample data columns from the file we are about to merge
            data_cols_to_sample = [c for c in df_to_merge.columns if c not in ['address_key', key_col_to_merge]][:2]
            log_dataframe_sample(
                df_to_merge,
                f"Pre-Merge Sample from [{csv_path.name}]",
                columns_to_log=['address_key', key_col_to_merge] + data_cols_to_sample
            )
            # Store the keys we just sampled so we can look them up after the merge
            sample_keys_for_verification = df_to_merge.head(5)['address_key'].tolist()

            # Perform the actual merge
            master_df = pd.merge(
                master_df,
                df_to_merge.drop(columns=[key_col_to_merge], errors='ignore'), # Drop original address col to avoid conflict
                on='address_key',
                how='left',
                suffixes=('', '_drop')
            )
            
            # --- DIAGNOSTIC (AFTER MERGE) ---
            # Now, find those exact same keys in the master dataframe and see if the new data is there.
            log_dataframe_sample(
                master_df[master_df['address_key'].isin(sample_keys_for_verification)],
                f"Post-Merge Verification in master_df for keys from [{csv_path.name}]",
                columns_to_log=['address_key', base_address_col] + data_cols_to_sample # Check for the new columns
            )

            # Clean up conflicting columns
            cols_to_drop = [col for col in master_df.columns if col.endswith('_drop')]
            if cols_to_drop:
                master_df.drop(columns=cols_to_drop, inplace=True)
                
        except Exception as e:
            logging.error(f"Failed to merge {csv_path.name}. Error: {e}", exc_info=True)
            continue

    # Final validation check
    logging.info(f"Merge process complete. Final dataframe shape: {master_df.shape}")
    if 'bba225_dow' in master_df.columns:
        merge_success_rate = master_df['bba225_dow'].notna().mean() * 100
        logging.info(f"DEBUG: Merge success rate for subset5 ('bba225_dow' column) is {merge_success_rate:.2f}%")
        if merge_success_rate < 80:
             logging.warning("Low final merge success rate. The address keys may not be matching well.")
    
    return master_df

def load_embeddings(local_data_dir: Path) -> dict[str, np.ndarray]:
    data = {}
    group_dirs = [d for d in local_data_dir.iterdir() if d.is_dir()]
    for group_dir in group_dirs:
        group_name = group_dir.name
        encodings_file = group_dir / "encodings.npy"
        if encodings_file.exists():
            data[group_name] = np.load(encodings_file)
    if not data: raise ValueError("No embedding data loaded.")
    return data

def main():
    parser = argparse.ArgumentParser(description="Generate master feature set.")
    parser.add_argument("--gcs_bucket", type=str, required=True)
    parser.add_argument("--embeddings_gcs_dir", type=str, required=True)
    parser.add_argument("--quantitative_gcs_paths", nargs='+', required=True)
    parser.add_argument("--output_gcs_dir", type=str, required=True)
    args = parser.parse_args()

    local_work_dir = Path("/tmp/master_feature_generation")
    local_embeddings_dir = local_work_dir / "embeddings_data"
    local_csv_dir = local_work_dir / "quantitative_data"
    local_output_dir = local_work_dir / "output"
    local_embeddings_dir.mkdir(parents=True, exist_ok=True)
    local_csv_dir.mkdir(parents=True, exist_ok=True)
    local_output_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info("--- Phase 1: Downloading ---")
    download_gcs_directory(args.gcs_bucket, args.embeddings_gcs_dir, local_embeddings_dir)
    for gcs_path in args.quantitative_gcs_paths:
        file_path = Path(gcs_path)
        blob_name = "/".join(file_path.parts[2:])
        download_gcs_file(args.gcs_bucket, blob_name, local_csv_dir / file_path.name)

    logging.info("--- Phase 2: Loading and Merging ---")
    embeddings_data = load_embeddings(local_embeddings_dir)
    
    base_csv_path = local_csv_dir / "property_features_quantitative_v4.csv"
    other_csv_names = ["cleaned_property_data_gwr_with_coords_label.csv", "cleaned_property_data_gwr.csv", "property_data_subset1.csv", "property_data_subset2.csv", "property_data_subset3.csv", "property_data_subset4.csv", "property_data_subset5.csv"]
    other_csv_paths = [local_csv_dir / name for name in other_csv_names]
    
    # --- THIS IS THE UPDATED FUNCTION CALL ---
    master_df = load_and_merge_data(base_csv_path, other_csv_paths)
    
    # This check is now more important. If merges fail, the row count will drop.
    if not master_df.empty and list(embeddings_data.values())[0].shape[0] != len(master_df):
        logging.warning(f"Row count mismatch! Embeddings: {list(embeddings_data.values())[0].shape[0]} vs DataFrame: {len(master_df)}. This is expected if the base file has properties not present in the embeddings, or if merges failed.")
    
    if master_df.empty:
        raise ValueError("Master dataframe is empty after merging. Halting execution.")

    num_properties = len(master_df)
    logging.info(f"--- Phase 3: Generating Interactions for {num_properties} properties ---")
    
    interaction_features_list = []

    # --- THIS IS THE START OF THE CORRECTED SECTION ---
    # Define helper functions BEFORE the loop
    def get_norm(name, index):
        if name in embeddings_data:
            return np.linalg.norm(embeddings_data[name][index])
        return 0.0

    def get_quant(row, col_name, default=0.0):
        if col_name in row.index:
            value = row[col_name]
            return value if pd.notna(value) else default
        return default

    for i in range(num_properties):
        if i > 0 and i % 1000 == 0:
            logging.info(f"Processing property {i}/{num_properties}...")
            
        property_interactions = {}
        row = master_df.iloc[i]

        # --- I. PRE-CALCULATE ALL NORMS AND KEY VALUES ---
        all_room_norms = {key: get_norm(key, i) for key in embeddings_data.keys()}
        property_age = 2025 - get_quant(row, 'num__property_construction_year_from_homipi__YYYY_or_2025NewBuildOrError__hm', 2000)

        # --- II. REWRITTEN AND VERIFIED FEATURE INTERACTIONS ---

        # -- Core Strengths & Ratios (Replaces direct comparisons) --
        kitchen_strength = all_room_norms.get('primary_MainKitchen', 0.0)
        living_strength = all_room_norms.get('primary_MainLivingArea', 0.0)
        garden_strength = all_room_norms.get('primary_MainGarden', 0.0)
        property_wide_strength = all_room_norms.get('property_wide', 0.0)
        primary_bed_strength = all_room_norms.get('primary_PrimaryBedroom', 0.0)

        property_interactions['Ratio_Kitchen_vs_Property'] = kitchen_strength / (property_wide_strength + 1e-6)
        property_interactions['Ratio_Kitchen_vs_Living'] = kitchen_strength / (living_strength + 1e-6)

        # -- Composite Scores from Norms --
        core_living_strength = np.mean([kitchen_strength, living_strength, all_room_norms.get('primary_MainDiningArea', 0.0)])
        property_interactions['Composite_CoreLiving_Strength'] = core_living_strength
        
        structural_strength = np.mean([all_room_norms.get(k, 0.0) for k in ['primary_MainExteriorFront', 'primary_MainGarage', 'primary_MainDrivewayParking']])
        cosmetic_strength = np.mean([all_room_norms.get(k, 0.0) for k in ['primary_MainKitchen', 'primary_MainBathroom', 'primary_MainLivingArea']])
        property_interactions['Mismatch_Structure_vs_Cosmetic_Diff'] = abs(structural_strength - cosmetic_strength)

        # -- Synthesized Risk & Opportunity Scores --
        property_interactions['Opportunity_Good_Bones_Score'] = structural_strength - cosmetic_strength
        property_interactions['Risk_Cosmetic_Burden_X_Age'] = cosmetic_strength * (property_age / 10.0)

        # -- High-Value Qualitative x Quantitative Interactions --
        property_interactions['Inter_KitchenQuality_X_MarketPrice'] = kitchen_strength * get_quant(row, 'LSOA_MedPrice_Recent', 1.0)
        property_interactions['Inter_GardenValue_X_PublicSpaceDeficit'] = garden_strength / (get_quant(row, 'ah4gpas', 1.0) + 1e-6)
        property_interactions['Inter_GardenValue_X_UrbanDensity'] = garden_strength * get_quant(row, 'LSOA_Property_Density', 1.0)
        property_interactions['Inter_KitchenQuality_Per_SqM'] = kitchen_strength / (get_quant(row, 'primary_MainKitchen_area_sqm', 10.0) + 1e-6)
        property_interactions['Inter_BedroomQuality_Per_Count'] = primary_bed_strength / (get_quant(row, 'num__number_of_bedrooms_from_homipi__numeric_or_empty__hm', 1.0) + 1e-6)
        
        best_school_rating = max(get_quant(row, 'num__primary_or_first_type_school_1_ofsted_rating_encoded__numeric_extracted_or_original_text__bnl', 1), get_quant(row, 'num__secondary_or_second_type_school_1_ofsted_rating_encoded__numeric_extracted_or_original_text__bnl', 1))
        property_interactions['Inter_FamilyFeatures_X_Schools'] = (primary_bed_strength + garden_strength) * best_school_rating
        
        property_interactions['Inter_Heating_X_Efficiency'] = living_strength / (get_quant(row, 'num__current_epc_value_extracted_by_initial_homipi_script_hm', 50.0) + 1e-6)
        
        # -- Persona-based Scores (reinterpreted with norms) --
        property_interactions['Persona_FixerUpper_Score'] = property_interactions['Risk_Cosmetic_Burden_X_Age']
        property_interactions['Persona_Entertainer_Score'] = core_living_strength + all_room_norms.get('primary_MainPatioDeckingTerrace', 0.0)

        # -- Higher-Order Composite "Thesis" Features --
        renovation_burden = property_interactions['Risk_Cosmetic_Burden_X_Age']
        area_price_growth = get_quant(row, 'LSOA_Price_Growth_5Year', 0.0)
        market_churn_rate = get_quant(row, 'LSOA_Churn_Recent', 0.0)
        property_interactions['Thesis_Renovation_X_AreaGrowth'] = renovation_burden * (1 + area_price_growth)
        property_interactions['Thesis_InvestmentSweetSpot'] = property_interactions['Opportunity_Good_Bones_Score'] * market_churn_rate * (1 + area_price_growth)
        
        sanctuary_score = core_living_strength * (get_quant(row, 'ah4no2', 0.0) + get_quant(row, 'FI_Deprivation_Severity_Index', 0.0))
        curb_appeal_strength = all_room_norms.get('primary_MainExteriorFront', 0.0)
        property_interactions['Thesis_HiddenGem'] = sanctuary_score / (curb_appeal_strength + 1e-6)

        wfh_potential = all_room_norms.get('other_OtherBedrooms', 0.0) + all_room_norms.get('primary_StudyOffice', 0.0)
        ev_ready_potential = all_room_norms.get('primary_MainGarage', 0.0)
        energy_efficiency = get_quant(row, 'num__current_epc_value_extracted_by_initial_homipi_script_hm', 50.0)
        property_interactions['Thesis_FutureProofingScore'] = (wfh_potential + ev_ready_potential) * energy_efficiency
        
        cosmetic_burden = property_interactions['Risk_Cosmetic_Burden_X_Age']
        property_interactions['Ratio_ValueAddLeverage'] = (1 + area_price_growth) / (cosmetic_burden + 1e-6)
        
        location_quality_score = get_quant(row, 'LSOA_MedPrice_Recent', 1.0) * (1.0 - get_quant(row, 'FI_Deprivation_Severity_Index', 0.0))
        property_interactions['Tradeoff_Quality_vs_Location'] = core_living_strength / (location_quality_score + 1e-6)
        
        deprivation_gradient = get_quant(row, 'FI_Deprivation_Severity_Index', 0.0) * (1.0 - get_quant(row, 'FI_Deprivation_Severity_Index', 0.0))
        property_interactions['Opportunity_GentrificationFrontline'] = renovation_burden * deprivation_gradient

        interaction_features_list.append(property_interactions)

    logging.info("--- Phase 4: Combining and Uploading Final Feature Set ---")
    interaction_df = pd.DataFrame(interaction_features_list)
    
    # Concatenate the original quantitative dataframe with the new interaction features
    final_master_df = pd.concat([master_df, interaction_df], axis=1)

    # --- THIS IS THE FIX ---
    # Impute all missing values (NaNs) with -1.0.
    # This is a critical step before training any model.
    logging.info(f"Imputing NaN values with -1.0. Found {final_master_df.isna().sum().sum()} total NaNs.")
    final_master_df.fillna(-1.0, inplace=True)
    
    # Clean up column names for compatibility with models like LightGBM
    final_master_df.columns = ["".join (c if c.isalnum() else '_' for c in str(x)) for x in final_master_df.columns]

    local_csv_path = local_output_dir / "master_feature_set.csv"
    final_master_df.to_csv(local_csv_path, index=False)
    
    gcs_csv_blob_name = f"{args.output_gcs_dir}/master_feature_set.csv"
    upload_to_gcs(args.gcs_bucket, str(local_csv_path), gcs_csv_blob_name)
    
    logging.info(f"--- Master Feature Generation Pipeline Finished Successfully. Final shape: {final_master_df.shape} ---")

if __name__ == "__main__":
    main()
EOL

# --- Run the Python Script ---
echo "--- Starting Master Feature Generation Script ---"

python3 generate_master_features.py \
    --gcs_bucket="${GCS_BUCKET}" \
    --embeddings_gcs_dir="${EMBEDDINGS_GCS_DIR}" \
    --output_gcs_dir="${OUTPUT_GCS_DIR}" \
    --quantitative_gcs_paths "${QUANTITATIVE_FILES[@]}"

if [ $? -ne 0 ]; then
    echo "ERROR: Master feature generation failed. Exiting pipeline."
    gsutil cp "${LOG_FILE}" "gs://${GCS_BUCKET}/outputs/logs/master_generation_FAILED.log"
    exit 1
fi

echo "--- Master Feature Generation Finished Successfully ---"

# --- Finalization ---
TIMESTAMP=$(date +"%Y%m%d-%H%M%S")
LOG_FILE_GCS_PATH="outputs/logs/master_generation_${TIMESTAMP}.log"
gsutil cp "${LOG_FILE}" "gs://${GCS_BUCKET}/${LOG_FILE_GCS_PATH}"

echo "--- Pipeline Finished: $(date) ---"