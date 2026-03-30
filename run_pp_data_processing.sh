#!/bin/bash

# ======================================================================================
# run_pp_data_processing.sh - V1
# DESCRIPTION:
# This script processes the raw HM Land Registry Price Paid (PP) dataset.
# It aggregates sales data by postcode, calculating the average monthly price for
# each primary property type (Detached, Semi-Detached, Terraced, Flat).
# The output is a Parquet file where each row represents a unique postcode and
# columns represent the historical price metric (e.g., pp_2024_01_D_avg_price).
# This creates a new historical feature subset for the main Wisteria pipeline.
#
# METHODOLOGY:
# 1. DATA PREPARATION: Downloads the raw PP data from GCS.
# 2. PROCESSING: A Python script loads, cleans, and transforms the data.
#    a. Assigns headers to the column-less CSV.
#    b. Parses dates and normalizes postcodes for accurate joining.
#    c. Filters for standard property types (D, S, T, F).
#    d. Groups by postcode, year, month, and property type to calculate mean price.
#    e. Pivots the data to create the wide format (one row per postcode).
#    f. Fills missing monthly sales data with -1.0 as requested.
# 3. OUTPUT: Saves the resulting dataframe as a compressed Parquet file and uploads
#    it back to a specified GCS location.
# ======================================================================================

# --- Strict Mode & Error Handling ---
set -e
set -o pipefail
set -x

# --- Configuration ---
GCS_BUCKET="srgan-bucket-ace-botany-453819-t4" 
RAW_PP_DATA_FILENAME="pp-complete (1).csv" 

# --- INPUTS ---
RAW_PP_DATA_GCS_PATH="gs://${GCS_BUCKET}/house data scrape/${RAW_PP_DATA_FILENAME}"
# --- NEW: Master postcode index to ensure row alignment ---
POSTCODE_INDEX_GCS_PATH="gs://${GCS_BUCKET}/house data scrape/postcode_index.csv"


# --- OUTPUT ---
# The output will be placed in the same directory as the other subsets for consistency.
OUTPUT_FILENAME="subset_pp_history_processed.parquet"
PROCESSED_DATA_GCS_PATH="gs://${GCS_BUCKET}/house data scrape/${OUTPUT_FILENAME}"
LOG_FILE_GCS_PATH="features/logs/pp_processing_run.log"

# --- LOCAL WORKSPACE ---
WORKDIR="${HOME}/pp_data_processing_work"
LOG_FILE="${WORKDIR}/run_pp_processing.log"

# --- Main Execution ---
echo "--- Cleaning up previous workspace at ${WORKDIR} ---"
rm -rf "${WORKDIR}"
mkdir -p "${WORKDIR}/data" "${WORKDIR}/artifacts"
cd "${WORKDIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "--- Price Paid Data Processing Pipeline V1 Started: $(date) ---"

# --- Environment Setup ---
echo "--- Setting up Python virtual environment... ---"
VENV_PATH="${WORKDIR}/pp_data_env"
python3 -m venv "${VENV_PATH}"
source "${VENV_PATH}/bin/activate"
pip install --upgrade pip
echo "--- Installing Python dependencies... ---"
# These are the standard dependencies used across the Wisteria project.
cat > requirements.txt << EOL
pandas==2.2.2
numpy==1.26.4
google-cloud-storage==2.16.0
pyarrow==16.1.0
tqdm==4.66.4
EOL
pip install --force-reinstall --no-cache-dir -r requirements.txt
echo "--- Dependency installation complete. ---"

# --- SCRIPT GENERATION ---
echo "--- Generating Python Script: 01_process_pp_data.py ---"
cat > 01_process_pp_data.py << 'EOL'
import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler()])

def normalize_postcode_key(postcode: str) -> str:
    """Normalizes a postcode to uppercase with no spaces for consistent joining."""
    if not isinstance(postcode, str): return ""
    return postcode.upper().replace(" ", "")

def main():
    parser = argparse.ArgumentParser(description="Process HM Land Registry Price Paid Data.")
    parser.add_argument("--input_path", type=Path, required=True, help="Path to the raw Price Paid CSV file.")
    parser.add_argument("--postcode_index_path", type=Path, required=True, help="Path to the master postcode_index.csv for alignment.")
    parser.add_argument("--output_path", type=Path, required=True, help="Path to save the processed Parquet file.")
    args = parser.parse_args()

    logging.info(f"Loading master postcode index from {args.postcode_index_path} for alignment...")
    index_df = pd.read_csv(args.postcode_index_path)
    index_df.rename(columns={'pcds': 'postcode'}, inplace=True)
    index_df['postcode_key'] = index_df['postcode'].apply(normalize_postcode_key)
    master_index = index_df[['postcode_key']].copy()
    del index_df

    # --- Memory Optimization: Process large CSV in chunks ---
    chunk_size = 5_000_000  # Process 5 million rows at a time
    aggregated_chunks = []
    valid_prop_types = ['D', 'S', 'T', 'F']

    logging.info(f"Starting chunked aggregation of {args.input_path} with chunk size {chunk_size}...")
    
    reader = pd.read_csv(
        args.input_path,
        header=None,
        on_bad_lines='skip',
        chunksize=chunk_size,
        usecols=[1, 2, 3, 4],
        names=['price', 'date_of_transfer', 'postcode', 'property_type'],
        dtype={'postcode': 'str', 'property_type': 'str'}
    )

    for chunk in tqdm(reader, desc="Processing raw data chunks"):
        # Clean and prepare the chunk
        chunk['price'] = pd.to_numeric(chunk['price'], errors='coerce')
        chunk['date_of_transfer'] = pd.to_datetime(chunk['date_of_transfer'], errors='coerce')
        chunk.dropna(inplace=True)
        chunk['year'] = chunk['date_of_transfer'].dt.year
        chunk['month'] = chunk['date_of_transfer'].dt.month
        chunk['postcode_key'] = chunk['postcode'].apply(normalize_postcode_key)

        chunk = chunk[chunk['property_type'].isin(valid_prop_types)]
        
        if chunk.empty:
            continue
        
        # Aggregate within the chunk and append to list
        chunk_agg = chunk.groupby(['postcode_key', 'year', 'month', 'property_type'], as_index=False)['price'].mean()
        aggregated_chunks.append(chunk_agg)
    
    logging.info("All chunks processed. Combining aggregated results...")
    if not aggregated_chunks:
        logging.warning("No data was aggregated. The output will only contain postcodes with no historical data.")
        pivoted_df = pd.DataFrame(columns=['postcode_key'])
    else:
        combined_agg = pd.concat(aggregated_chunks, ignore_index=True)
        del aggregated_chunks

        logging.info("Performing final aggregation to correctly average groups split across chunks...")
        aggregated = combined_agg.groupby(['postcode_key', 'year', 'month', 'property_type'], as_index=False)['price'].mean()
        del combined_agg

        logging.info("Pivoting data to wide format...")
        pivoted_df = aggregated.pivot_table(
            index='postcode_key',
            columns=['year', 'month', 'property_type'],
            values='price'
        )
        pivoted_df.columns = [f"pp_{year}_{month:02d}_{prop_type}_avg_price" for year, month, prop_type in pivoted_df.columns]
        pivoted_df = pivoted_df.reindex(sorted(pivoted_df.columns), axis=1)
        pivoted_df.reset_index(inplace=True)

    logging.info("Aligning processed data with master postcode index...")
    aligned_df = pd.merge(master_index, pivoted_df, on='postcode_key', how='left')
    del pivoted_df, master_index

    logging.info("Performing time-series imputation...")
    aligned_df.set_index('postcode_key', inplace=True)
    
    aligned_df.replace(-1.0, np.nan, inplace=True)
    aligned_df.interpolate(method='linear', axis=1, inplace=True, limit_direction='both')
    aligned_df.fillna(-1.0, inplace=True)
    
    logging.info("Imputation complete.")
    
    # The main pipeline expects a pure numerical matrix without the postcode key.
    # The row order is guaranteed by the earlier left merge with the master_index.
    # Saving with index=False drops the postcode_key index, which is the desired behavior.
    logging.info(f"Processing complete. Final aligned dataset has shape: {aligned_df.shape}")
    logging.info(f"Saving processed data to {args.output_path}...")
    aligned_df.to_parquet(args.output_path, index=False, compression='gzip')
    logging.info("File saved successfully.")

if __name__ == "__main__":
    main()
EOL

echo "--- Downloading raw Price Paid data and master index from GCS... ---"
gsutil -m cp "${RAW_PP_DATA_GCS_PATH}" "data/${RAW_PP_DATA_FILENAME}"
gsutil -m cp "${POSTCODE_INDEX_GCS_PATH}" "data/postcode_index.csv"
echo "--- Download complete. ---"

# --- Execute the Processing Script ---
echo "--- EXECUTING SCRIPT: 01_process_pp_data.py ---"
python3 01_process_pp_data.py \
    --input_path "data/${RAW_PP_DATA_FILENAME}" \
    --postcode_index_path "data/postcode_index.csv" \
    --output_path "artifacts/${OUTPUT_FILENAME}"
echo "--- SCRIPT COMPLETE. ---"

# --- Finalization: Upload Artifacts ---
echo "--- Uploading processed data to GCS... ---"
gsutil -m cp "artifacts/${OUTPUT_FILENAME}" "${PROCESSED_DATA_GCS_PATH}"
echo "--- Processed data upload complete. ---"

echo "--- Uploading execution log to GCS... ---"
gsutil -m cp "${LOG_FILE}" "gs://${GCS_BUCKET}/${LOG_FILE_GCS_PATH}"

echo "--- Price Paid Data Processing Pipeline V1 Finished Successfully: $(date) ---"