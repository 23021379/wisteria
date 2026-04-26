#!/bin/bash

# ==============================================================================
# run_surprise_feature_pipeline.sh - Generates "Surprise" Features
#
# This script orchestrates a pipeline to identify anomalous feature values in the
# master dataset. For a predefined list of 20 key features, it:
#
# 1. Trains a LightGBM model to predict the feature's value based on all
#    other contextual data, carefully excluding known data-leaking columns.
# 2. Calculates the standardized residual (actual - predicted) / stdev, which
#    represents the "surprise" score for that feature.
# 3. Appends these 20 new "surprise" features to the master dataset.
# 4. Saves the final, augmented dataset back to GCS for downstream modeling.
# ==============================================================================

# --- Strict Mode & Error Handling ---
set -e
set -o pipefail
set -x

# --- Configuration ---
GCS_BUCKET="srgan-bucket-ace-botany-453819-t4"
INPUT_GCS_PATH="gs://${GCS_BUCKET}/models/gwr_outputs/imputed_normalized_master_data.parquet"
OUTPUT_GCS_DIR="features/surprise_feature_dataset"
TIMESTAMP=$(date +"%Y%m%d-%H%M%S")
LOG_FILE_GCS_PATH="outputs/logs/surprise_feature_pipeline_${TIMESTAMP}.log"

# Local workspace
WORKDIR="${HOME}/surprise_feature_work"
LOG_FILE="${WORKDIR}/run_pipeline.log"

# --- Main Execution ---
mkdir -p "${WORKDIR}"
cd "${WORKDIR}"

# Redirect all output to a log file AND the console
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "--- Surprise Feature Generation Pipeline Started: $(date) ---"

# --- Environment Setup (CRITICAL BLOCK) ---
echo "--- Setting up Python environment... ---"
sudo apt-get update -y && sudo apt-get install -y python3-pip git

echo "--- Upgrading pip... ---"
python3 -m pip install --user --upgrade pip

# The VM's default PATH may not include the user's local bin.
echo "--- Exporting new PATH to use upgraded pip... ---"
export PATH="${HOME}/.local/bin:${PATH}"
echo "--- Current PATH: ${PATH} ---"
echo "--- Pip version: $(pip --version) ---"

echo "--- Installing Python dependencies for surprise feature modeling... ---"
cat > requirements.txt << EOL
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.4.2
google-cloud-storage==2.16.0
lightgbm==4.3.0
pyarrow==16.1.0
joblib==1.4.2
EOL

python3 -m pip install --user --force-reinstall --no-cache-dir -r requirements.txt

echo "--- Verifying key installations... ---"
python3 -m pip show pandas scikit-learn google-cloud-storage lightgbm

# --- Create the Python Worker Script ---
echo "--- Generating the Python script for surprise feature generation... ---"
cat > generate_surprise_features.py << 'EOL'
# generate_surprise_features.py
import argparse
import logging
import re
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from google.cloud import storage

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# --- GCS Helper Functions ---
def download_gcs_file(bucket_name: str, source_blob_name: str, destination_file_path: Path):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    destination_file_path.parent.mkdir(parents=True, exist_ok=True)
    logging.info(f"Downloading gs://{bucket_name}/{source_blob_name} to {destination_file_path}...")
    blob.download_to_filename(str(destination_file_path))
    logging.info("Download complete.")

def upload_to_gcs(bucket_name: str, source_file_path: str, destination_blob_name: str):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    logging.info(f"Uploading {source_file_path} to gs://{bucket_name}/{destination_blob_name}...")
    blob.upload_from_filename(source_file_path)
    logging.info("Upload complete.")

def get_leakage_map(df_columns):
    """
    Dynamically constructs the leakage map based on the DataFrame's columns.
    This prevents file I/O errors during script initialization.
    """
    LEAKAGE_MAP = {
        # --- I. Core Quantitative & Gemini-Derived Stats ---
        'num__floor_area_sqm_from_homipi__numeric_or_empty__hm': [
            'num__INT_hm_bedrooms_per_sqm', 'num__INT_mp_price_per_sqm', 'num__INT_bnl_price_per_sqft',
            'num__property_floor_area_sqft__numeric_extracted_or_original_text__bnl_converted_sqm',
            'num__INT_avg_floor_area_sqm', 'Ratio_Kitchen_vs_Property', 'Inter_KitchenQuality_Per_SqM'
        ],
        'num__number_of_bedrooms_from_homipi__numeric_or_empty__hm': [
            'other_OtherBedrooms_count', 'num__INT_hm_bedrooms_per_sqm', 'Inter_BedroomQuality_Per_Count',
            'Inter_FamilyFeatures_X_Schools'
        ] + [col for col in df_columns if 'number_bedrooms' in col],
        'num__property_construction_year_from_homipi__YYYY_or_2025NewBuildOrError__hm': [
            'num__INT_mp_built_year_range', 'num__mouseprice_built_year_start__YYYY_or_0_or_empty__mp',
            'num__mouseprice_built_year_end__YYYY_or_0_or_empty__mp', 'Postcode_Age', 'FI_Age_vs_OA_Modernity',
            'Risk_Cosmetic_Burden_X_Age'
        ] + [col for col in df_columns if col.startswith(('BP_', 'MODE_'))],
        'primary_MainKitchen_renovation_score': [
            'Mismatch_Structure_vs_Cosmetic_Diff', 'Risk_Cosmetic_Burden_X_Age', 'Inter_KitchenQuality_X_MarketPrice',
            'Inter_KitchenQuality_Per_SqM', 'Persona_FixerUpper_Score', 'Persona_Entertainer_Score',
            'Composite_CoreLiving_Strength'
        ],
        'primary_MainGarden_area_sqm': [
            'other_OtherGardens_count', 'other_OtherGardens_total_features', 'Inter_GardenValue_X_PublicSpaceDeficit',
            'Inter_GardenValue_X_UrbanDensity'
        ],
        'num__current_epc_value_extracted_by_initial_homipi_script_hm_numeric': [
            'num__potential_epc_value_extracted_by_initial_homipi_script_hm_numeric', 'num__INT_hm_epc_improvement_potential',
            'Inter_Heating_X_Efficiency', 'Thesis_FutureProofingScore'
        ],
        'num__property_tenure_code_from_homipi__2Freehold_hm': [
            'num__property_tenure_code_from_homipi__1LeaseholdOrOther__hm'
        ],
        # --- II. Key Engineered & Thesis Features ---
        'Opportunity_Good_Bones_Score': [
            'Mismatch_Structure_vs_Cosmetic_Diff', 'Ratio_ValueAddLeverage', 'Persona_FixerUpper_Score'
        ] + [col for col in df_columns if col.startswith(('Opportunity_', 'Risk_', 'Thesis_'))],
        'Thesis_InvestmentSweetSpot': [
            'Thesis_Renovation_X_AreaGrowth', 'Thesis_HiddenGem', 'Tradeoff_Quality_vs_Location'
        ] + [col for col in df_columns if col.startswith(('Opportunity_', 'Risk_'))],
        'Persona_FixerUpper_Score': [
            'Opportunity_Good_Bones_Score', 'Risk_Cosmetic_Burden_X_Age'
        ] + [col for col in df_columns if '_renovation_score' in col or '_num_flaws' in col],
        'num__INT_avg_estimated_price': [
            'num__INT_std_dev_estimated_price', 'num__INT_hm_price_vs_last_sold_ratio'
        ] + [col for col in df_columns if 'price' in col and ('homipi_' in col or 'mouseprice_' in col or 'bricksandlogic_' in col)],
        # --- III. Important Geo/Market Features ---
        'LSOA_MedPrice_Recent': [
            'LSOA_MedPrice_5Y_Ago', 'LSOA_Price_Growth_5Year', 'num__StreetScan_past_sales_avg_price_gbp_for_All_Properties_ss'
        ] + [col for col in df_columns if 'PC_Price_to_LSOA_Median_Ratio' in col or ('HPI_Adjusted_Median_Price' in col and col[-2] == '_')],
        'LSOA_Churn_Recent': [
            'LSOA_Transactions_Recent', 'LSOA_Transactions_5Y_Ago', 'LSOA_Churn_to_Transaction_Ratio',
            'PC_Sales_vs_LSOA_Churn_D', 'LSOA_Price_to_Churn_Ratio', 'Churn_in_LSOA_Dominated_by_ModernStock'
        ],
        'ah4gpas': [
            'AccessibleCleanGreenSpace_Gpas_x_InvNO2', 'GreenVsBlueAccessRelativity_GpasPct_minus_BluePct',
            'num__StreetScan_category_rating_stars_for_Scenery_and_Parks_ss', 'Inter_GardenValue_X_PublicSpaceDeficit'
        ],
        'num__StreetScan_deprivation_rank_for_Crime_ss': [
            'num__INT_safety_composite_chimnie_ss'
        ] + [col for col in df_columns if 'safety_index_score_ch' in col],
        'num__primary_or_first_type_school_1_ofsted_rating_encoded__numeric_extracted_or_original_text__bnl': [
            'num__INT_bnl_school_chimnie_family_composite', 'Inter_FamilyFeatures_X_Schools',
            'num__chimnie_local_area_family_index_score_ch', 'num__StreetScan_category_rating_stars_for_Schools_ss'
        ] + [col for col in df_columns if 'school' in col and 'ofsted_rating' in col],
        'num__homipi_nearest_rail_station_distance_or_number_1_hm': [
            'num__StreetScan_category_rating_stars_for_Transport_ss'
        ] + [col for col in df_columns if 'rail_station_distance' in col or 'train_station' in col],
        'FI_Prop_Owned_Outright_HH': [
            'ownership_Owned_Owns_outright', 'elderly_owned_outright_sum', 'FI_Elderly_Owned_Outright_Concentration'
        ],
        'FI_Overcrowding_Rate_Bedrooms': [
            'FI_Overcrowded_And_Children_Concentration_Index'
        ] + [col for col in df_columns if 'number_per_bedroom_' in col or 'occupancy_rating_' in col],
        'ah4pubs': [
            'UnhealthyRetailExposure_InvPubTime_Plus_InvFFTime', 'SocialAmenityAccess_InvPubTime_x_InvLeisureTime'
        ] + [col for col in df_columns if ('food_and_drink_index_score_ch' in col or 'entertainment_index_score' in col or 'lifestyle_index_score' in col)],
    }
    return LEAKAGE_MAP

def sanitize_col_name(col_name: str) -> str:
    """Cleans a column name to be model-friendly."""
    return re.sub(r'[^A-Za-z0-9_]+', '', col_name)

def main():
    parser = argparse.ArgumentParser(description="Generate 'surprise' features by modeling residuals.")
    parser.add_argument("--gcs_bucket", type=str, required=True)
    parser.add_argument("--input_gcs_path", type=str, required=True)
    parser.add_argument("--output_gcs_dir", type=str, required=True)
    args = parser.parse_args()

    local_work_dir = Path("/tmp/surprise_feature_generation")
    local_work_dir.mkdir(parents=True, exist_ok=True)
    
    # Correctly parse the GCS path to get the blob name
    input_blob_name = args.input_gcs_path.replace(f"gs://{args.gcs_bucket}/", "")
    local_parquet_path = local_work_dir / "input_data.parquet"
    
    # --- 1. Download & Load Data ---
    logging.info("--- Phase 1: Downloading and Loading Data ---")
    download_gcs_file(args.gcs_bucket, input_blob_name, local_parquet_path)
    df = pd.read_parquet(local_parquet_path)
    logging.info(f"Successfully loaded data with shape: {df.shape}")

    # --- 2. Verification and Setup ---
    logging.info("--- Phase 2: Verifying Feature Lists and Preparing for Modeling ---")
    
    # Dynamically generate the leakage map *after* data is loaded
    leakage_map = get_leakage_map(df.columns)
    
    # Identify non-feature columns to always exclude from predictors (X)
    id_cols = [c for c in df.columns if 'property_id' in c or 'address' in c]
    coord_cols = [c for c in df.columns if 'longitude' in c or 'latitude' in c]
    
    # Safely find the label column
    price_cols = [c for c in df.columns if 'price' in c.lower() and 'hm' in c.lower()]
    if not price_cols:
        logging.error("Could not find the homipi price column to use as the label. Exiting.")
        exit(1)
    label_col = price_cols[0]
    logging.info(f"Identified '{label_col}' as the primary price label to exclude from predictors.")
    
    base_exclusions = id_cols + coord_cols + [label_col]
    all_surprise_features = {}

    # --- 3. Iterative Modeling Loop ---
    logging.info("--- Phase 3: Starting Iterative Surprise Feature Generation ---")
    for target_y_col, leak_cols in leakage_map.items():
        start_time = time.time()
        logging.info(f"\n===== Processing Target: {target_y_col} =====")
        
        if target_y_col not in df.columns:
            logging.warning(f"Target column '{target_y_col}' not found in DataFrame. Skipping.")
            continue
        
        current_exclusions = base_exclusions + leak_cols + [target_y_col]
        valid_exclusions = [c for c in current_exclusions if c in df.columns]
        
        y = df[target_y_col]
        X = df.drop(columns=valid_exclusions).select_dtypes(include=np.number)

        logging.info(f"Predictor shape for this target: {X.shape}")

        model = lgb.LGBMRegressor(
            n_estimators=300, learning_rate=0.05, n_jobs=-1, random_state=42,
            colsample_bytree=0.8, subsample=0.8, reg_alpha=0.1, reg_lambda=0.1
        )
        model.fit(X, y)
        
        y_pred = model.predict(X)
        residuals = y - y_pred
        std_dev_residuals = residuals.std()
        
        surprise_score = residuals / (std_dev_residuals + 1e-9)
        
        surprise_col_name = f"surprise_for_{sanitize_col_name(target_y_col)}"
        all_surprise_features[surprise_col_name] = surprise_score
        
        end_time = time.time()
        logging.info(f"Finished processing '{target_y_col}' in {end_time - start_time:.2f} seconds.")
        logging.info(f"Surprise score stats: Mean={surprise_score.mean():.3f}, StdDev={surprise_score.std():.3f}, Min={surprise_score.min():.3f}, Max={surprise_score.max():.3f}")

    # --- 4. Final Consolidation and Upload ---
    logging.info("\n--- Phase 4: Consolidating and Uploading Final Dataset ---")
    
    if not all_surprise_features:
        logging.error("No surprise features were generated. Exiting.")
        exit(1)

    surprise_df = pd.DataFrame(all_surprise_features)
    final_df = pd.concat([df, surprise_df], axis=1)

    logging.info(f"Original shape: {df.shape}, New features: {surprise_df.shape[1]}, Final shape: {final_df.shape}")

    # Save and upload the final augmented dataset
    final_output_path = local_work_dir / "surprise_feature_master_dataset.parquet"
    final_df.to_parquet(final_output_path, index=False)
    
    output_blob_name = f"{args.output_gcs_dir}/surprise_feature_master_dataset.parquet"
    upload_to_gcs(args.gcs_bucket, str(final_output_path), output_blob_name)
    
    logging.info("--- Surprise Feature Generation Pipeline Finished Successfully ---")

if __name__ == "__main__":
    # A small hack is needed to dynamically build the LEAKAGE_MAP with column names
    # from the input file, which we don't know until it's downloaded.
    # We will download it here first to populate the map, then run main.
    parser = argparse.ArgumentParser()
    parser.add_argument("--gcs_bucket", required=True)
    parser.add_argument("--input_gcs_path", required=True)
    args, _ = parser.parse_known_args()
    
    local_path = Path("/tmp/header_check.parquet")
    blob_name = "/".join(Path(args.input_gcs_path).parts[2:])
    download_gcs_file(args.gcs_bucket, blob_name, local_path)
    
    # This re-initializes the global LEAKAGE_MAP with actual data
    # This is a bit unusual but necessary to handle the dynamic column lists.
    exec(Path(__file__).read_text(), globals(), locals())

    main()
EOL

# --- Run the Python Script ---
echo "--- Starting Surprise Feature Generation Script ---"
python3 generate_surprise_features.py \
    --gcs_bucket="${GCS_BUCKET}" \
    --input_gcs_path="${INPUT_GCS_PATH}" \
    --output_gcs_dir="${OUTPUT_GCS_DIR}"

if [ $? -ne 0 ]; then
    echo "ERROR: Surprise feature generation failed. Exiting pipeline."
    gsutil -m cp "${LOG_FILE}" "gs://${GCS_BUCKET}/outputs/logs/surprise_feature_pipeline_${TIMESTAMP}_FAILED.log"
    exit 1
fi

echo "--- Surprise Feature Generation Finished Successfully ---"

# --- Finalization: Upload Logs ---
echo "--- Uploading execution log to GCS... ---"
gsutil -m cp "${LOG_FILE}" "gs://${GCS_BUCKET}/${LOG_FILE_GCS_PATH}"

echo "--- Pipeline Finished: $(date) ---"