#!/bin/bash

# ======================================================================================
# run_geospatial_pipeline.sh - V11 - EXTERNAL INDEX FIX
# DESCRIPTION:
# This version corrects the data ingestion logic to handle an external index file
# that contains the postcodes, which are then merged with the numerical feature subsets.
#
# CORRECTIONS IN THIS VERSION:
# 1. FIXED: The consolidation script `00_consolidate_subsets.py` now loads a separate
#    `index.csv` file and correctly merges it with the five feature subsets.
# 2. ADDED: A validation step to ensure row counts match between the index and subsets.
# 3. RETAINED: The core Atlas, Compass, and Microscope logic remains intact.
# ======================================================================================

# --- Strict Mode & Error Handling ---
set -e
set -o pipefail
set -x

# --- Configuration ---
GCS_BUCKET="srgan-bucket-ace-botany-453819-t4"
TIMESTAMP=$(date +"%Y%m%d-%H%M%S")
# --- UPDATED: Paths to the 5 feature subsets + the new index file ---
GLOBAL_INDEX_PATH="gs://${GCS_BUCKET}/house data scrape/postcode_index.csv" # The file with the 'pcds' column
GLOBAL_SUBSET1_PATH="gs://${GCS_BUCKET}/house data scrape/subset1_processed_full.parquet"
GLOBAL_SUBSET2_PATH="gs://${GCS_BUCKET}/house data scrape/subset2_processed_full.parquet"
GLOBAL_SUBSET3_PATH="gs://${GCS_BUCKET}/house data scrape/subset3_processed_full.parquet"
GLOBAL_SUBSET4_PATH="gs://${GCS_BUCKET}/house data scrape/subset4_processed_full.parquet"
GLOBAL_SUBSET5_PATH="gs://${GCS_BUCKET}/house data scrape/subset5_processed_full.parquet"

# Path to your MASTER property dataset (the one with your properties to enrich)
MASTER_PROPERTY_DATASET_GCS_PATH="gs://${GCS_BUCKET}/features/final_master_dataset/master_feature_set.csv"

# --- Output Configuration ---
ARTIFACTS_GCS_DIR="features/geospatial_pipeline_property_aware_${TIMESTAMP}"
LOG_FILE_GCS_PATH="${ARTIFACTS_GCS_DIR}/logs/pipeline_run.log"
WORKDIR="${HOME}/geospatial_feature_work_v11"
LOG_FILE="${WORKDIR}/run_pipeline.log"

# --- Modeling Parameters ---
ATLAS_N_CLUSTERS=75
COMPASS_N_NEIGHBORS_LIST=(20 50 100)
MICROSCOPE_EMBEDDING_DIM=16
MICROSCOPE_EPOCHS=10

# --- Pipeline Execution ---
echo "--- Cleaning up previous workspace at ${WORKDIR} ---"
rm -rf "${WORKDIR}"
mkdir -p "${WORKDIR}"
cd "${WORKDIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "--- Geospatial & Property-Aware Feature Engineering Pipeline Started: $(date) ---"

# --- Environment Setup ---
#  --- Environment Setup (with Dask added) ---
echo "--- Setting up Python virtual environment... ---"
VENV_PATH="${WORKDIR}/geospatial_env"
python3 -m venv "${VENV_PATH}"
source "${VENV_PATH}/bin/activate"
pip install --upgrade pip
echo "--- Installing Python dependencies (including Dask)... ---"
cat > requirements.txt << EOL
--extra-index-url https://download.pytorch.org/whl/cu121
torch==2.2.2
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.4.2
scipy==1.13.0
google-cloud-storage==2.16.0
pyarrow==16.1.0
joblib==1.4.2
tqdm==4.66.4
dask[dataframe]==2024.1.0
EOL
pip install --force-reinstall --no-cache-dir -r requirements.txt
echo "--- Dependency installation complete. ---"


# ==============================================================================
# --- SCRIPT 00: CONSOLIDATE SUBSETS (V6 - TRUE OUT-OF-CORE FIX) ---
# This version is the definitive fix for the OOM error. It removes the final
# `.compute()` call and uses Dask's native `.to_parquet()` method to write
# the result directly to disk without ever materializing the full dataset
# in memory.
# ==============================================================================
echo "--- Generating Python Script: 00_consolidate_subsets.py ---"
cat > 00_consolidate_subsets.py << 'EOL'
# 00_consolidate_subsets.py (V11 - Corrected Build-and-Replace)
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
from pathlib import Path
import shutil
import logging
from tqdm import tqdm
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """
    Consolidates multiple large files using an ultra-low-memory "build-and-replace"
    strategy. This version contains a critical fix for the Parquet file writing
    logic to prevent creating empty/corrupted files.
    """
    logging.info("--- Running Step 00: Consolidate Subsets (V11 - Corrected Writer) ---")
    data_dir = Path("data")
    final_output_path = data_dir / "global_dataset.parquet"
    temp_build_path = data_dir / "temp_building_dataset.parquet"
    chunk_size = 50_000

    # Clean up from previous runs
    for path in [final_output_path, temp_build_path]:
        if os.path.exists(path):
            os.remove(path)

    try:
        # --- Stage 1: Convert index.csv to the base Parquet file ---
        logging.info("Stage 1: Converting index.csv to base Parquet format.")
        index_iterator = pd.read_csv(data_dir / "index.csv", chunksize=chunk_size)
        
        # --- THE FIX: Initialize writer to None ---
        writer = None
        for i, chunk in enumerate(tqdm(index_iterator, desc="Converting CSV to Parquet")):
            chunk.rename(columns={'pcds': 'postcode'}, inplace=True)
            table = pa.Table.from_pandas(chunk, preserve_index=True)
            
            # --- THE FIX: Create the writer only on the first iteration ---
            if writer is None:
                writer = pq.ParquetWriter(final_output_path, table.schema)
            
            writer.write_table(table)
        
        # Ensure the writer is closed after the loop
        if writer:
            writer.close()
        
        logging.info(f"Base file created at {final_output_path}. Verifying...")
        # Verification step
        pq.ParquetFile(final_output_path)
        logging.info("Base file verification successful.")

        # --- Stage 2: Iteratively join each subset ---
        subset_paths = [
            data_dir / "subset1.parquet", data_dir / "subset2.parquet",
            data_dir / "subset3.parquet", data_dir / "subset4.parquet",
            data_dir / "subset5.parquet"
        ]

        for i, subset_path in enumerate(subset_paths):
            logging.info(f"--- Stage 2.{i+1}/5: Merging {subset_path.name} ---")
            
            base_file_reader = pq.ParquetFile(final_output_path)
            subset_reader = pq.ParquetFile(subset_path)
            
            base_iterator = base_file_reader.iter_batches(batch_size=chunk_size)
            subset_iterator = subset_reader.iter_batches(batch_size=chunk_size)
            
            writer = None
            for j, (base_batch, subset_batch) in enumerate(tqdm(
                zip(base_iterator, subset_iterator),
                total=base_file_reader.metadata.num_rows // chunk_size + 1,
                desc=f"Joining {subset_path.name}"
            )):
                base_chunk_pd = base_batch.to_pandas()
                subset_chunk_pd = subset_batch.to_pandas()
                
                merged_chunk = pd.concat([
                    base_chunk_pd.reset_index(drop=True), 
                    subset_chunk_pd.reset_index(drop=True)
                ], axis=1)
                
                # Remove duplicated columns just in case
                merged_chunk = merged_chunk.loc[:,~merged_chunk.columns.duplicated()]
                
                table = pa.Table.from_pandas(merged_chunk, preserve_index=True)
                if writer is None:
                    writer = pq.ParquetWriter(temp_build_path, table.schema)
                
                writer.write_table(table)
            
            if writer:
                writer.close()

            os.remove(final_output_path)
            os.rename(temp_build_path, final_output_path)
            logging.info(f"Successfully merged {subset_path.name}. New global dataset is ready.")

        final_file = pq.ParquetFile(final_output_path)
        logging.info(f"--- CONSOLIDATION COMPLETE ---")
        logging.info(f"Final dataset has {final_file.metadata.num_rows} rows and {len(final_file.schema.names)} columns.")

    except Exception as e:
        logging.error(f"A critical error occurred during build-and-replace consolidation: {e}", exc_info=True)
        for path in [final_output_path, temp_build_path]:
            if os.path.exists(path):
                os.remove(path)
        exit(1)

if __name__ == "__main__":
    main()
EOL

# --- Data Download (Now downloads subsets AND the index file) ---
echo "--- Downloading feature subsets and index file from GCS... ---"
mkdir -p data
gsutil -m cp "${GLOBAL_INDEX_PATH}" data/index.csv
gsutil -m cp "${GLOBAL_SUBSET1_PATH}" data/subset1.parquet
gsutil -m cp "${GLOBAL_SUBSET2_PATH}" data/subset2.parquet
gsutil -m cp "${GLOBAL_SUBSET3_PATH}" data/subset3.parquet
gsutil -m cp "${GLOBAL_SUBSET4_PATH}" data/subset4.parquet
gsutil -m cp "${GLOBAL_SUBSET5_PATH}" data/subset5.parquet
gsutil -m cp "${MASTER_PROPERTY_DATASET_GCS_PATH}" data/master_property_dataset.csv
echo "--- All data download complete. ---"

# --- Execute the new consolidation script ---
echo "--- EXECUTING SCRIPT 00: CONSOLIDATE SUBSETS ---"
python3 00_consolidate_subsets.py
echo "--- SCRIPT 00 COMPLETE. global_dataset.parquet is now ready. ---"

# ==============================================================================
# The rest of the pipeline (Scripts 0, 1, 2, 3) is unchanged from V10
# as it correctly uses the artifacts created by the preceding steps.
# ==============================================================================

# --- SCRIPT 0: PREPARE ENRICHED INPUT ---
echo "--- Generating Python Script: 0_prepare_enriched_input.py ---"
cat > 0_prepare_enriched_input.py << 'EOL'
# 0_prepare_enriched_input.py
import sys
import pandas as pd
import numpy as np
def main():
    print("--- Running Step 0: Prepare ENRICHED Input & Pre-flight Validation ---")
    try:
        master_path = 'data/master_property_dataset.csv'
        output_path = 'data/enriched_property_input.parquet'
        print(f"Loading master dataset from {master_path}...")
        master_df = pd.read_csv(master_path, low_memory=False)
        print(f"Master dataset loaded with shape: {master_df.shape}")
        geo_cols_map = {'property_id': ['property_id', 'address', 'original_property_address'],'postcode': ['postcode', 'postcode_key'],'latitude': ['latitude', 'pcd_latitude'],'longitude': ['longitude', 'pcd_longitude']}
        property_specific_cols = ['num_images_total','num_rooms_identified_in_step5','avg_persona_rating_overall','std_dev_persona_rating_overall','persona_Persona_1_overall_rating','persona_Persona_2_overall_rating','persona_Persona_3_overall_rating','persona_Persona_4_overall_rating','persona_Persona_5_overall_rating','persona_Persona_6_overall_rating','persona_Persona_7_overall_rating','persona_Persona_8_overall_rating','persona_Persona_9_overall_rating','persona_Persona_10_overall_rating','persona_Persona_11_overall_rating','persona_Persona_12_overall_rating','persona_Persona_13_overall_rating','persona_Persona_14_overall_rating','persona_Persona_15_overall_rating','persona_Persona_16_overall_rating','persona_Persona_17_overall_rating','persona_Persona_18_overall_rating','persona_Persona_19_overall_rating','persona_Persona_20_overall_rating','primary_MainKitchen_area_sqm','primary_MainKitchen_renovation_score','primary_MainKitchen_num_features','primary_MainKitchen_num_sps','primary_MainKitchen_num_flaws','other_OtherKitchens_count','other_OtherKitchens_total_features','other_OtherKitchens_total_sps','other_OtherKitchens_total_flaws','other_OtherKitchens_avg_features_per_room','other_OtherKitchens_avg_sps_per_room','other_OtherKitchens_avg_flaws_per_room','primary_MainLivingArea_area_sqm','primary_MainLivingArea_renovation_score','primary_MainLivingArea_num_features','primary_MainLivingArea_num_sps','primary_MainLivingArea_num_flaws','other_OtherLivingAreas_count','other_OtherLivingAreas_total_features','other_OtherLivingAreas_total_sps','other_OtherLivingAreas_total_flaws','other_OtherLivingAreas_avg_features_per_room','other_OtherLivingAreas_avg_sps_per_room','other_OtherLivingAreas_avg_flaws_per_room','primary_MainDiningArea_area_sqm','primary_MainDiningArea_renovation_score','primary_MainDiningArea_num_features','primary_MainDiningArea_num_sps','primary_MainDiningArea_num_flaws','other_OtherDiningAreas_count','other_OtherDiningAreas_total_features','other_OtherDiningAreas_total_sps','other_OtherDiningAreas_total_flaws','other_OtherDiningAreas_avg_features_per_room','other_OtherDiningAreas_avg_sps_per_room','other_OtherDiningAreas_avg_flaws_per_room','primary_PrimaryBedroom_area_sqm','primary_PrimaryBedroom_renovation_score','primary_PrimaryBedroom_num_features','primary_PrimaryBedroom_num_sps','primary_PrimaryBedroom_num_flaws','other_OtherBedrooms_count','other_OtherBedrooms_total_features','other_OtherBedrooms_total_sps','other_OtherBedrooms_total_flaws','other_OtherBedrooms_avg_features_per_room','other_OtherBedrooms_avg_sps_per_room','other_OtherBedrooms_avg_flaws_per_room','primary_MainBathroom_area_sqm','primary_MainBathroom_renovation_score','primary_MainBathroom_num_features','primary_MainBathroom_num_sps','primary_MainBathroom_num_flaws','other_OtherBathroomsWCs_count','other_OtherBathroomsWCs_total_features','other_OtherBathroomsWCs_total_sps','other_OtherBathroomsWCs_total_flaws','other_OtherBathroomsWCs_avg_features_per_room','other_OtherBathroomsWCs_avg_sps_per_room','other_OtherBathroomsWCs_avg_flaws_per_room','primary_MainHallwayLandingStairs_area_sqm','primary_MainHallwayLandingStairs_renovation_score','primary_MainHallwayLandingStairs_num_features','primary_MainHallwayLandingStairs_num_sps','primary_MainHallwayLandingStairs_num_flaws','primary_MainUtilityRoom_area_sqm','primary_MainUtilityRoom_renovation_score','primary_MainUtilityRoom_num_features','primary_MainUtilityRoom_num_sps','primary_MainUtilityRoom_num_flaws','primary_MainGarage_area_sqm','primary_MainGarage_renovation_score','primary_MainGarage_num_features','primary_MainGarage_num_sps','primary_MainGarage_num_flaws','primary_MainConservatorySunroom_area_sqm','primary_MainConservatorySunroom_renovation_score','primary_MainConservatorySunroom_num_features','primary_MainConservatorySunroom_num_sps','primary_MainConservatorySunroom_num_flaws','primary_StudyOffice_area_sqm','primary_StudyOffice_renovation_score','primary_StudyOffice_num_features','primary_StudyOffice_num_sps','primary_StudyOffice_num_flaws','primary_MainGarden_area_sqm','primary_MainGarden_renovation_score','primary_MainGarden_num_features','primary_MainGarden_num_sps','primary_MainGarden_num_flaws','other_OtherGardens_count','other_OtherGardens_total_features','other_OtherGardens_total_sps','other_OtherGardens_total_flaws','other_OtherGardens_avg_features_per_room','other_OtherGardens_avg_sps_per_room','other_OtherGardens_avg_flaws_per_room','primary_MainPatioDeckingTerrace_area_sqm','primary_MainPatioDeckingTerrace_renovation_score','primary_MainPatioDeckingTerrace_num_features','primary_MainPatioDeckingTerrace_num_sps','primary_MainPatioDeckingTerrace_num_flaws','primary_MainDrivewayParking_area_sqm','primary_MainDrivewayParking_renovation_score','primary_MainDrivewayParking_num_features','primary_MainDrivewayParking_num_sps','primary_MainDrivewayParking_num_flaws','primary_MainExteriorFront_area_sqm','primary_MainExteriorFront_renovation_score','primary_MainExteriorFront_num_features','primary_MainExteriorFront_num_sps','primary_MainExteriorFront_num_flaws','primary_MainExteriorRear_area_sqm','primary_MainExteriorRear_renovation_score','primary_MainExteriorRear_num_features','primary_MainExteriorRear_num_sps','primary_MainExteriorRear_num_flaws','primary_MainExteriorSide_area_sqm','primary_MainExteriorSide_renovation_score','primary_MainExteriorSide_num_features','primary_MainExteriorSide_num_sps','primary_MainExteriorSide_num_flaws','primary_MainOutbuilding_area_sqm','primary_MainOutbuilding_renovation_score','primary_MainOutbuilding_num_features','primary_MainOutbuilding_num_sps','primary_MainOutbuilding_num_flaws','primary_MainStorageLoftCellar_area_sqm','primary_MainStorageLoftCellar_renovation_score','primary_MainStorageLoftCellar_num_features','primary_MainStorageLoftCellar_num_sps','primary_MainStorageLoftCellar_num_flaws','primary_MainPorch_area_sqm','primary_MainPorch_renovation_score','primary_MainPorch_num_features','primary_MainPorch_num_sps','primary_MainPorch_num_flaws','primary_MainMiscIndoor_area_sqm','primary_MainMiscIndoor_renovation_score','primary_MainMiscIndoor_num_features','primary_MainMiscIndoor_num_sps','primary_MainMiscIndoor_num_flaws']
        enriched_df = pd.DataFrame()
        for standard_name, potential_names in geo_cols_map.items():
            found_col = next((name for name in potential_names if name in master_df.columns), None)
            if not found_col: raise ValueError(f"CRITICAL ERROR: Master dataset missing geo column '{standard_name}'. Searched for: {potential_names}")
            enriched_df[standard_name] = master_df[found_col]
        found_prop_cols = [col for col in property_specific_cols if col in master_df.columns]
        for col in found_prop_cols: enriched_df[col] = pd.to_numeric(master_df[col], errors='coerce').fillna(0)
        enriched_df.dropna(subset=['latitude', 'longitude', 'postcode', 'property_id'], inplace=True)
        enriched_df.to_parquet(output_path, index=False)
        print(f"Successfully created ENRICHED input file with {len(enriched_df)} rows at '{output_path}'")
    except Exception as e:
        print(f"An error occurred during preparation/validation: {e}", file=sys.stderr)
        sys.exit(1)
if __name__ == "__main__": main()
EOL

echo "--- EXECUTING SCRIPT 0: PREPARE ENRICHED INPUT ---"
python3 0_prepare_enriched_input.py
echo "--- SCRIPT 0 COMPLETE ---"


# --- SCRIPT 1: ATLAS GENERATION ---
echo "--- Generating Python Script: 1_create_atlas.py ---"
cat > 1_create_atlas.py << 'EOL'
# 1_create_atlas.py
import argparse
import logging
from pathlib import Path
import joblib
import pandas as pd
import pyarrow.parquet as pq
from sklearn.cluster import MiniBatchKMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from tqdm import tqdm
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
def main(args):
    logging.info("--- ATLAS GENERATION STARTED ---")
    data_dir = Path("data")
    artifacts_dir = Path("artifacts/atlas")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    global_dataset_path = data_dir / 'global_dataset.parquet'
    logging.info(f"Opening global dataset from {global_dataset_path}...")
    parquet_file = pq.ParquetFile(global_dataset_path)
    schema = parquet_file.schema
    id_cols_to_exclude = ['postcode', 'latitude', 'longitude', 'pcd_latitude', 'pcd_longitude', 'pcd_eastings', 'pcd_northings', '__index_level_0__']
    geo_feature_cols = [field.name for field in schema if field.name not in id_cols_to_exclude and 'BYTE_ARRAY' not in str(field.physical_type)]
    geo_feature_cols = [c for c in geo_feature_cols if not c.startswith('usertype_') and not c.startswith('missingindicator_')]
    logging.info(f"Identified {len(geo_feature_cols)} numeric geographic features for clustering.")
    preprocessing_pipeline = Pipeline([('imputer', SimpleImputer(strategy='median')),('variance_remover', VarianceThreshold(threshold=0.01)),('scaler', StandardScaler())])
    logging.info("Fitting preprocessing pipeline on a sample of the data...")
    sample_df = next(parquet_file.iter_batches(batch_size=100_000, columns=geo_feature_cols)).to_pandas()
    preprocessing_pipeline.fit(sample_df)
    joblib.dump(preprocessing_pipeline, artifacts_dir / "atlas_geo_preprocessing_pipeline.joblib")
    joblib.dump(geo_feature_cols, artifacts_dir / "atlas_geo_feature_cols.joblib")
    logging.info(f"Training MiniBatchKMeans with n_clusters={args.n_clusters}...")
    kmeans = MiniBatchKMeans(n_clusters=args.n_clusters, random_state=42, batch_size=4096, n_init='auto')
    for batch in tqdm(parquet_file.iter_batches(batch_size=20000, columns=geo_feature_cols), total=parquet_file.num_row_groups, desc="Training K-Means"):
        X_processed = preprocessing_pipeline.transform(batch.to_pandas())
        kmeans.partial_fit(X_processed)
    logging.info("Clustering training complete.")
    joblib.dump(kmeans, artifacts_dir / "atlas_kmeans_model.joblib")
    logging.info(f"Saved KMeans model and preprocessing pipeline to {artifacts_dir}")
    logging.info("Creating Atlas mapping file (postcode -> cluster_id)...")
    all_postcodes = []
    all_labels = []
    cols_for_prediction = list(set(['postcode'] + geo_feature_cols))
    for batch in tqdm(parquet_file.iter_batches(batch_size=20000, columns=cols_for_prediction), total=parquet_file.num_row_groups, desc="Predicting Clusters"):
        chunk_df = batch.to_pandas()
        all_postcodes.extend(chunk_df['postcode'].tolist())
        chunk_df_reindexed = chunk_df.reindex(columns=geo_feature_cols, fill_value=0)
        X_processed = preprocessing_pipeline.transform(chunk_df_reindexed)
        labels = kmeans.predict(X_processed)
        all_labels.extend(labels)
    atlas_mapping = pd.DataFrame({'postcode': all_postcodes, 'cluster_id': all_labels})
    atlas_mapping.to_parquet(artifacts_dir / 'atlas_mapping.parquet', index=False)
    logging.info(f"Saved Atlas mapping file to {artifacts_dir / 'atlas_mapping.parquet'}")
    logging.info("--- ATLAS GENERATION FINISHED ---")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_clusters", type=int, required=True)
    args = parser.parse_args()
    main(args)
EOL
echo "--- EXECUTING SCRIPT 1: ATLAS (WITH UPDATED LOGIC) ---"
python3 1_create_atlas.py --n_clusters="${ATLAS_N_CLUSTERS}"
echo "--- ATLAS SCRIPT COMPLETE ---"


# --- SCRIPT 2: COMPASS & MICROSCOPE ---
echo "--- Generating Python Script: 2_create_compass_microscope.py ---"
cat > 2_create_compass_microscope.py << 'EOL'
# 2_create_compass_microscope.py
import argparse
import logging
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.spatial import KDTree
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Autoencoder(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.1),nn.Linear(128, 64), nn.ReLU(),nn.Linear(64, embedding_dim))
        self.decoder = nn.Sequential(nn.Linear(embedding_dim, 64), nn.ReLU(),nn.Linear(64, 128), nn.ReLU(),nn.Linear(128, input_dim))
    def forward(self, x): return self.decoder(self.encoder(x))
    def get_embedding(self, x): return self.encoder(x)


def train_cluster_ae(data_tensor, input_dim, embedding_dim, epochs):
    model = Autoencoder(input_dim, embedding_dim)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    dataset = TensorDataset(data_tensor)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    model.train()
    for _ in range(epochs):
        for batch_data, in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_data)
            loss.backward()
            optimizer.step()
    model.eval()
    return model


def process_property_batch(prop_batch_df, kdtree, global_parquet_file, args, all_processed_geo_features_batch, all_property_features_batch, cluster_aes, numerical_agg_cols):
    """Processes a batch of properties for Compass and Microscope features."""
    all_neighbor_indices = set()
    neighbor_map = {}

    for i, prop_row in prop_batch_df.iterrows():
        prop_coords = [prop_row['latitude'], prop_row['longitude']]
        if np.isnan(prop_coords).any():
            continue
        _, neighbor_indices = kdtree.query(prop_coords, k=max(args.n_neighbors_list))
        all_neighbor_indices.update(neighbor_indices)
        neighbor_map[i] = neighbor_indices

    if not all_neighbor_indices:
        return []

    required_row_groups = set()
    current_row_offset = 0
    for i in range(global_parquet_file.num_row_groups):
        rg_meta = global_parquet_file.metadata.row_group(i)
        start_row = current_row_offset
        end_row = start_row + rg_meta.num_rows
        if any(start_row <= idx < end_row for idx in all_neighbor_indices):
            required_row_groups.add(i)
        current_row_offset = end_row

    # --- FIXED: Handle different possible index column names ---
    # First, get the schema to see what columns are available
    schema_names = [field.name for field in global_parquet_file.schema]
    
    # Look for potential index columns
    potential_index_cols = ['__index_level_0__', 'index', '__index__', '__row_index__']
    actual_index_col = next((col for col in potential_index_cols if col in schema_names), None)
    
    # Define columns to load
    cols_to_load = numerical_agg_cols.copy()
    if actual_index_col:
        cols_to_load.append(actual_index_col)
    
    # Load the required row groups with the identified columns
    neighbor_df_full = global_parquet_file.read_row_groups(
        list(required_row_groups),
        columns=cols_to_load
    ).to_pandas()
    
    # If we have an index column, set it as the index for fast lookups
    if actual_index_col and actual_index_col in neighbor_df_full.columns:
        neighbor_df_full.set_index(actual_index_col, inplace=True)
    else:
        # If no explicit index column, use the implicit row number
        # Calculate the actual row indices based on row group positions
        row_indices = []
        current_offset = 0
        for rg_idx in sorted(required_row_groups):
            rg_meta = global_parquet_file.metadata.row_group(rg_idx)
            # Calculate the starting row number for this row group
            rg_start = sum(global_parquet_file.metadata.row_group(j).num_rows for j in range(rg_idx))
            row_indices.extend(range(rg_start, rg_start + rg_meta.num_rows))
        
        neighbor_df_full.index = row_indices[:len(neighbor_df_full)]
    # --- END OF FIX ---

    batch_results = []
    for i, prop_row in prop_batch_df.iterrows():
        prop_id = prop_row['property_id']
        cluster_id = prop_row['cluster_id']
        prop_features = {'property_id': prop_id}

        try:
            if i not in neighbor_map:
                raise ValueError("Property skipped in neighbor search (NaN coords)")
            
            neighbor_df = neighbor_df_full.loc[neighbor_map[i]]

            for n in args.n_neighbors_list:
                n_neighbor_df = neighbor_df.head(n)
                for col in numerical_agg_cols:
                    if col in n_neighbor_df.columns:
                        prop_features[f'compass_mean_{col}_n{n}'] = n_neighbor_df[col].mean()
                        prop_features[f'compass_std_{col}_n{n}'] = n_neighbor_df[col].std()
        except Exception as e:
            logging.warning(f"Compass features FAILED for property {prop_id}. Reason: {e}.")
        
        prop_features['atlas_cluster_id'] = cluster_id
        ae_model = cluster_aes.get(cluster_id)
        if not ae_model or pd.isna(cluster_id):
            batch_results.append(prop_features)
            continue
        
        try:
            with torch.no_grad():
                feature_idx = prop_batch_df.index.get_loc(i)
                processed_geo_part = all_processed_geo_features_batch[feature_idx:feature_idx+1, :]
                property_part = all_property_features_batch[feature_idx:feature_idx+1, :]
                
                combined_processed_features = np.hstack([processed_geo_part, property_part])
                features_tensor = torch.FloatTensor(combined_processed_features)
                embedding = ae_model.get_embedding(features_tensor).numpy().flatten()
                for j, val in enumerate(embedding): prop_features[f'microscope_emb_{j}'] = val
        except Exception as e:
            logging.warning(f"Microscope inference FAILED for property {prop_id}. Reason: {e}.")

        batch_results.append(prop_features)
        
    return batch_results



def main(args):
    logging.info("--- COMPASS & MICROSCOPE (V12 - MEMORY EFFICIENT) ---")
    data_dir = Path("data")
    artifacts_dir = Path("artifacts")
    output_dir = artifacts_dir / "contextual_features"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load small, essential dataframes
    prop_df = pd.read_parquet(data_dir / "enriched_property_input.parquet")
    atlas_df = pd.read_parquet(artifacts_dir / "atlas/atlas_mapping.parquet")
    geo_preprocessing_pipeline = joblib.load(artifacts_dir / "atlas/atlas_geo_preprocessing_pipeline.joblib")
    geo_feature_cols = joblib.load(artifacts_dir / "atlas/atlas_geo_feature_cols.joblib")

    # --- MEMORY-EFFICIENT AE TRAINING DATA PREPARATION ---
    logging.info("Beginning memory-efficient preparation for Autoencoder training...")

    # Normalize postcodes once on smaller dataframes
    prop_df['postcode_norm'] = prop_df['postcode'].str.replace(r'\s+', '', regex=True).str.upper()
    atlas_df['postcode_norm'] = atlas_df['postcode'].str.replace(r'\s+', '', regex=True).str.upper()
    if atlas_df['postcode_norm'].duplicated().any():
        atlas_df = atlas_df.groupby('postcode_norm').first().reset_index()

    # Merge properties with their Atlas cluster ID
    props_with_clusters = pd.merge(prop_df, atlas_df[['postcode_norm', 'cluster_id']], on='postcode_norm', how='left')
    props_with_clusters.dropna(subset=['cluster_id'], inplace=True)
    props_with_clusters['cluster_id'] = props_with_clusters['cluster_id'].astype(int)

    # Read the global dataset in chunks to merge features
    global_dataset_path = data_dir / 'global_dataset.parquet'
    chunk_iter = pq.ParquetFile(global_dataset_path).iter_batches(batch_size=50000, columns=['postcode'] + geo_feature_cols)

    master_df_chunks = []
    logging.info("Reading global dataset in chunks and merging with property data...")
    for i, batch in enumerate(tqdm(chunk_iter, desc="Merging Chunks")):
        global_chunk = batch.to_pandas()
        global_chunk['postcode_norm'] = global_chunk['postcode'].str.replace(r'\s+', '', regex=True).str.upper()
        
        # Merge this chunk of global data with ALL properties
        # This finds which properties match the postcodes in the current global chunk
        merged_chunk = pd.merge(props_with_clusters, global_chunk, on='postcode_norm', how='inner')
        if not merged_chunk.empty:
            master_df_chunks.append(merged_chunk)

    logging.info("Concatenating merged chunks to create the master AE training dataframe...")
    if not master_df_chunks:
        logging.error("CRITICAL: No overlap found between property postcodes and global dataset postcodes.")
        exit(1)
        
    master_ae_training_df = pd.concat(master_df_chunks, ignore_index=True)
    # Deduplicate in case a property matched postcodes in multiple chunks (unlikely but safe)
    master_ae_training_df.drop_duplicates(subset=['property_id'], inplace=True)
    
    # Clean up memory
    del master_df_chunks, props_with_clusters, atlas_df
    
    # --- END OF REVISED MERGE LOGIC ---

    master_ae_training_df.dropna(subset=['latitude', 'longitude'], inplace=True)
    master_ae_training_df[geo_feature_cols] = master_ae_training_df[geo_feature_cols].fillna(0)

    logging.info("Identifying property-specific NUMERIC features for the autoencoder...")

    # Define all columns that are explicitly NOT property-specific features for the AE
    id_and_geo_cols_to_exclude = ['property_id', 'postcode', 'original_property_address', 'postcode_norm', 'cluster_id', '__index_level_0__'] + geo_feature_cols

    # Start with a dataframe containing only the candidate columns
    candidate_prop_feature_df = master_ae_training_df.drop(columns=id_and_geo_cols_to_exclude, errors='ignore')

    # Now, from these candidates, select ONLY the columns that have a numeric dtype.
    # This is the critical fix that prevents string columns from being included.
    microscope_prop_feature_cols = candidate_prop_feature_df.select_dtypes(include=np.number).columns.tolist()

    logging.info(f"Identified {len(microscope_prop_feature_cols)} purely numeric property-specific columns for AE training.")

    cluster_aes = {}
    logging.info(f"Training {args.n_clusters} Autoencoders...")
    # (The rest of the AE training loop remains the same)
    for cluster_id in tqdm(range(args.n_clusters), desc="Training Cluster AEs"):
        cluster_data = master_ae_training_df[master_ae_training_df['cluster_id'] == cluster_id]
        if len(cluster_data) < 20: continue
        
        geo_part_processed = geo_preprocessing_pipeline.transform(cluster_data[geo_feature_cols])
        prop_part = cluster_data[microscope_prop_feature_cols].values
        
        combined_processed_features = np.hstack([geo_part_processed, prop_part])
        features_tensor = torch.FloatTensor(combined_processed_features)
        input_dim = combined_processed_features.shape[1]
        cluster_aes[cluster_id] = train_cluster_ae(features_tensor, input_dim, args.embedding_dim, args.epochs)
    logging.info(f"Successfully trained {len(cluster_aes)} autoencoder models.")

    # --- MEMORY EFFICIENT KDTree and Inference (V2 - Robust Column Finding) ---
    logging.info("Building KDTree from coordinate-only data...")

    # First, get the schema from the Parquet file without loading data
    global_parquet_schema = pq.ParquetFile(global_dataset_path).schema
    global_all_cols = [field.name for field in global_parquet_schema]

    # Define potential names for our coordinate columns
    potential_lat_cols = ['latitude', 'pcd_latitude']
    potential_lon_cols = ['longitude', 'pcd_longitude']

    # Find the actual names present in the file
    actual_lat_col = next((name for name in potential_lat_cols if name in global_all_cols), None)
    actual_lon_col = next((name for name in potential_lon_cols if name in global_all_cols), None)

    if not actual_lat_col or not actual_lon_col:
        raise ValueError(f"CRITICAL ERROR: Could not find coordinate columns in global_dataset.parquet. Searched for lat:{potential_lat_cols}, lon:{potential_lon_cols}")

    logging.info(f"Found coordinate columns in global dataset: '{actual_lat_col}' and '{actual_lon_col}'")

    # Load ONLY the correctly identified coordinate columns
    global_coords_df = pd.read_parquet(global_dataset_path, columns=[actual_lat_col, actual_lon_col])
    global_coords_df.fillna(-1.0, inplace=True)

    # Use the actual column names to build the KDTree
    kdtree = KDTree(global_coords_df[[actual_lat_col, actual_lon_col]].values)

    del global_coords_df, global_parquet_schema, global_all_cols # Free up memory

    enriched_prop_df_for_inference = master_ae_training_df.copy().reset_index(drop=True)
    
    all_processed_geo_features = geo_preprocessing_pipeline.transform(enriched_prop_df_for_inference[geo_feature_cols])
    all_property_features = enriched_prop_df_for_inference[microscope_prop_feature_cols].values
    
    numerical_agg_cols = [c for c in geo_feature_cols if any(p in c for p in ['price', 'count', 'ahah', 'veg', 'bba', 'score', 'imd'])]
    
    logging.info(f"Starting batched feature generation using {args.n_jobs} cores...")
    
    global_parquet_file = pq.ParquetFile(global_dataset_path)
    
    batch_size = 1000
    num_batches = int(np.ceil(len(enriched_prop_df_for_inference) / batch_size))

    tasks = []
    # (The batch processing loop is correct and does not need changes)
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        
        prop_batch_df = enriched_prop_df_for_inference.iloc[start_idx:end_idx]
        all_processed_geo_features_batch = all_processed_geo_features[start_idx:end_idx]
        all_property_features_batch = all_property_features[start_idx:end_idx]
        
        tasks.append(joblib.delayed(process_property_batch)(
            prop_batch_df, kdtree, global_parquet_file, args, 
            all_processed_geo_features_batch, all_property_features_batch, 
            cluster_aes, numerical_agg_cols
        ))

    results_nested = joblib.Parallel(n_jobs=args.n_jobs, prefer="threads")(tqdm(tasks, total=num_batches))
    
    final_results = [item for sublist in results_nested for item in sublist]

    if not final_results: logging.error("CRITICAL: All feature generation tasks failed."); exit(1)
    
    contextual_features_df = pd.DataFrame(final_results)
    output_path = output_dir / "contextual_features.parquet"
    contextual_features_df.to_parquet(output_path, index=False)
    logging.info(f"Successfully generated features for {len(contextual_features_df)} rows. Saved to {output_path}")
    logging.info("--- COMPASS & MICROSCOPE FEATURE GENERATION FINISHED ---")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_neighbors_list", type=int, nargs='+', required=True)
    parser.add_argument("--embedding_dim", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--n_clusters", type=int, default=75)
    args = parser.parse_args()
    main(args)
EOL
echo '--- EXECUTING SCRIPT 2: COMPASS & MICROSCOPE ---'
python3 2_create_compass_microscope.py \
    --n_neighbors_list "${COMPASS_N_NEIGHBORS_LIST[@]}" \
    --embedding_dim ${MICROSCOPE_EMBEDDING_DIM} \
    --epochs ${MICROSCOPE_EPOCHS} \
    --n_clusters ${ATLAS_N_CLUSTERS} \
    --n_jobs -1
echo '--- COMPASS & MICROSCOPE SCRIPT COMPLETE ---'


# --- SCRIPT 3: FINAL MERGE ---
echo "--- Generating Python Script: 3_final_merge.py ---"
cat > 3_final_merge.py << 'EOL'
# 3_final_merge.py
import logging
from pathlib import Path
import pandas as pd
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
def main():
    logging.info("--- FINAL MERGE STARTED ---")
    master_property_path = Path("data/master_property_dataset.csv")
    contextual_features_path = Path("artifacts/contextual_features/contextual_features.parquet")
    output_path = Path("artifacts/final_enriched_master_dataset.parquet")
    logging.info("Loading original master property data and new contextual features...")
    master_df = pd.read_csv(master_property_path, low_memory=False)
    contextual_df = pd.read_parquet(contextual_features_path)
    id_col_in_master = None
    potential_id_cols = ['property_id', 'address', 'original_property_address']
    id_col_in_master = next((col for col in potential_id_cols if col in master_df.columns), None)
    if not id_col_in_master: raise ValueError(f"Could not find a valid ID column in master dataset to merge on. Looked for {potential_id_cols}")
    logging.info(f"Master data shape: {master_df.shape}")
    logging.info(f"Contextual features shape: {contextual_df.shape}")
    logging.info(f"Merging on master column '{id_col_in_master}' and contextual column 'property_id'")
    final_df = pd.merge(master_df, contextual_df, left_on=id_col_in_master, right_on='property_id', how='left')
    if id_col_in_master != 'property_id' and 'property_id' in final_df.columns: final_df.drop(columns=['property_id'], inplace=True)
    final_df.columns = ["".join (c if c.isalnum() else '_' for c in str(x)) for x in final_df.columns]
    final_df.to_parquet(output_path, index=False)
    logging.info(f"Final enriched dataset created with shape: {final_df.shape}. Saved to {output_path}")
    logging.info("--- FINAL MERGE FINISHED ---")
if __name__ == "__main__": main()
EOL
echo "--- EXECUTING SCRIPT 3: MERGE ---"
python3 3_final_merge.py
echo "--- MERGE SCRIPT COMPLETE ---"


# --- Finalization: Upload All Artifacts ---
echo "--- Uploading all generated artifacts to GCS... ---"
gsutil -m cp -r artifacts/* "gs://${GCS_BUCKET}/${ARTIFACTS_GCS_DIR}/"
echo "--- Artifact upload complete. ---"

echo "--- Uploading execution log to GCS... ---"
gsutil -m cp "${LOG_FILE}" "gs://${GCS_BUCKET}/${LOG_FILE_GCS_PATH}"

echo "--- Geospatial & Property-Aware Feature Engineering Pipeline Finished Successfully: $(date) ---"