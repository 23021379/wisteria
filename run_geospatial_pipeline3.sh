#!/bin/bash

# ======================================================================================
# run_geospatial_pipeline.sh - V15 - CORRECTED SUBSET KEYING
# DESCRIPTION:
# This version fixes a fatal KeyError by correctly sourcing postcodes from the
# dedicated `postcode_index.csv` file instead of assuming they exist within the
# purely numerical subset files.
#
# METHODOLOGY:
# 1. DATA PREPARATION:
#    a. Loads the `postcode_index.csv` to get the master list of postcodes.
#    b. Loads the five numerical subsets and consolidates them.
#    c. CRITICAL VALIDATION: It verifies that the total row count of the five
#       subsets EXACTLY matches the row count of the index file before proceeding.
# 2. ATLAS: Trains clusters using the numerical subset data.
# 3. ATTRIBUTE JOIN: Merges clusters and canonical coordinates onto the master
#    property list.
# 4. SPATIAL JOIN & AGGREGATION: Builds a KDTree and generates all Compass and
#    Microscope features based on the validated, row-aligned data.
# ======================================================================================

# --- Strict Mode & Error Handling ---
set -e
set -o pipefail
set -x

# --- Configuration ---
GCS_BUCKET="srgan-bucket-ace-botany-453819-t4"

# --- INPUTS ---
MASTER_PROPERTY_DATASET_GCS_PATH="gs://${GCS_BUCKET}/features/final_master_dataset/master_feature_set.csv"
POSTCODE_LOOKUP_GCS_PATH="gs://${GCS_BUCKET}/house data scrape/PCD_OA21_LSOA21_MSOA21_LAD_AUG24_UK_LU.csv"
# --- NEW: Explicit path to the postcode index file ---
POSTCODE_INDEX_GCS_PATH="gs://${GCS_BUCKET}/house data scrape/postcode_index.csv"
GLOBAL_SUBSET1_PATH="gs://${GCS_BUCKET}/house data scrape/subset1_processed_full.parquet"
GLOBAL_SUBSET2_PATH="gs://${GCS_BUCKET}/house data scrape/subset2_processed_full.parquet"
GLOBAL_SUBSET3_PATH="gs://${GCS_BUCKET}/house data scrape/subset3_processed_full.parquet"
GLOBAL_SUBSET4_PATH="gs://${GCS_BUCKET}/house data scrape/subset4_processed_full.parquet"
GLOBAL_SUBSET5_PATH="gs://${GCS_BUCKET}/house data scrape/subset5_processed_full.parquet"
# --- NEW: Path to the processed price paid history subset ---
GLOBAL_SUBSET_PP_PATH="gs://${GCS_BUCKET}/house data scrape/subset_pp_history_processed.parquet"

# --- OUTPUT ---
ARTIFACTS_GCS_DIR="features/geospatial_pipeline_v16_full_with_pp" # Version bump
LOG_FILE_GCS_PATH="${ARTIFACTS_GCS_DIR}/logs/pipeline_run.log"

# --- LOCAL WORKSPACE ---
WORKDIR="${HOME}/geospatial_feature_work_v15"
LOG_FILE="${WORKDIR}/run_pipeline.log"

# --- MODELING PARAMETERS ---
ATLAS_N_CLUSTERS=75
COMPASS_N_NEIGHBORS_LIST=(20 50 100)
MICROSCOPE_EMBEDDING_DIM=16
MICROSCOPE_EPOCHS=10

# --- Main Execution ---
echo "--- Cleaning up previous workspace at ${WORKDIR} ---"
rm -rf "${WORKDIR}"
mkdir -p "${WORKDIR}/data" "${WORKDIR}/artifacts"
cd "${WORKDIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "--- Full Geospatial Enrichment Pipeline V15 Started: $(date) ---"

# --- Environment Setup ---
echo "--- Setting up Python virtual environment... ---"
VENV_PATH="${WORKDIR}/geospatial_env"
python3 -m venv "${VENV_PATH}"
source "${VENV_PATH}/bin/activate"
pip install --upgrade pip
echo "--- Installing Python dependencies... ---"
cat > requirements.txt << EOL
--extra-index-url https://download.pytorch.org/whl/cu121
torch==2.2.2
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.4.2
scipy==1.13.0
google-cloud-storage==2.16.0
pyarrow==16.1.0
tqdm==4.66.4
EOL
pip install --force-reinstall --no-cache-dir -r requirements.txt
echo "--- Dependency installation complete. ---"

# --- SCRIPT GENERATION ---
echo "--- Generating Python Script: 01_run_full_enrichment.py ---"
cat > 01_run_full_enrichment.py << 'EOL'
import argparse
import logging
import re
from pathlib import Path
import gc
import warnings
import pyarrow.parquet as pq
import pyarrow as pa
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.spatial import KDTree
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="Degrees of freedom <= 0 for slice", category=RuntimeWarning)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler()])

def extract_postcode_from_address(address: str) -> str:
    if not isinstance(address, str): return ""
    match = re.search(r'([A-Z]{1,2}[0-9][A-Z0-9]?\s?[0-9][A-Z]{2})$', address.upper())
    return match.group(0) if match else ""

def normalize_postcode_key(postcode: str) -> str:
    if not isinstance(postcode, str): return ""
    return postcode.upper().replace(" ", "")

class Autoencoder(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, embedding_dim))
        self.decoder = nn.Sequential(nn.Linear(embedding_dim, 64), nn.ReLU(), nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, input_dim))
    def forward(self, x): return self.decoder(self.encoder(x))
    def get_embedding(self, x): return self.encoder(x)

def train_autoencoder(data_tensor, input_dim, embedding_dim, epochs, device):
    model = Autoencoder(input_dim, embedding_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    dataset = TensorDataset(data_tensor)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    model.train()
    for _ in range(epochs):
        for batch_data, in dataloader:
            inputs = batch_data.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser(description="Full Geospatial Enrichment Pipeline")
    parser.add_argument("--master_properties_path", type=Path, required=True)
    parser.add_argument("--postcode_lookup_path", type=Path, required=True)
    parser.add_argument("--postcode_index_path", type=Path, required=True)
    parser.add_argument("--subset_pp_path", type=Path, required=True) # <-- NEW ARGUMENT FOR PRICE PAID DATA
    parser.add_argument("--subset_paths", type=Path, nargs='+', required=True)
    parser.add_argument("--n_neighbors_list", type=int, nargs='+', required=True)
    parser.add_argument("--n_clusters", type=int, required=True)
    parser.add_argument("--embedding_dim", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # ==========================================================================
    # STAGE 0: DATA LOADING AND VALIDATION
    # ==========================================================================
    logging.info(f"\n--- STAGE 0: Loading, Consolidating, and Validating Data ---")
    
    # 1. Load the index file to get the master list of postcodes and expected row count
    logging.info(f"Loading postcode index from: {args.postcode_index_path}")
    index_df = pd.read_csv(args.postcode_index_path)
    index_df.rename(columns={'pcds': 'postcode'}, inplace=True)
    subset_postcodes = index_df['postcode'].tolist()
    expected_rows = len(subset_postcodes)
    logging.info(f"Index file loaded. Expecting {expected_rows} rows in each subset.")
    
    subset_paths = args.subset_paths
    coord_cols = ['pcd_latitude', 'pcd_longitude']
    
    # 2. Pre-scan subsets to determine total feature dimension for memory-efficient pre-allocation
    all_feature_cols = []
    total_feature_count = 0
    logging.info("Pre-scanning subsets to determine total feature dimensions...")
    # NOTE: We include the pp_subset in this scan to ensure compass features are generated for it.
    for path in args.subset_paths:
        pq_file = pq.ParquetFile(path)
        arrow_schema = pq_file.schema_arrow
        subset_feature_cols = [
            field.name for field in arrow_schema
            if (pa.types.is_floating(field.type) or pa.types.is_integer(field.type)) and field.name not in coord_cols
        ]
        all_feature_cols.extend(subset_feature_cols)
        total_feature_count += len(subset_feature_cols)
    logging.info(f"Total features to be loaded: {total_feature_count}. Pre-allocating master feature array.")

    import shutil
    # Proactively check for sufficient disk space before creating the large file
    required_bytes = expected_rows * total_feature_count * np.dtype(np.float32).itemsize
    required_gb = required_bytes / (1024**3)
    
    # Check disk usage of the directory where the output will be saved
    _, _, free_bytes = shutil.disk_usage(args.output_dir)
    free_gb = free_bytes / (1024**3)

    logging.info(f"Required disk space for memmap file: {required_gb:.2f} GB. Available: {free_gb:.2f} GB.")
    if free_bytes < required_bytes * 1.1: # Add a 10% buffer
        raise MemoryError(
            f"FATAL: Insufficient disk space. Required: ~{required_gb:.2f} GB, "
            f"but only {free_gb:.2f} GB is available in {args.output_dir}. "
            "Please increase the VM's disk size."
        )

    # Create a memory-mapped file to store the large feature matrix on disk
    memmap_path = args.output_dir / "temp_feature_matrix.mmap"
    logging.info(f"Creating memory-mapped file for feature matrix at: {memmap_path}")
    all_subset_features = np.memmap(memmap_path, dtype=np.float32, mode='w+', shape=(expected_rows, total_feature_count))
    
    # The coordinate data is small, so it can remain in memory
    all_subset_coords = np.zeros((expected_rows, len(coord_cols)), dtype=np.float32)
    feature_cols = all_feature_cols
    
    # 3. Load subsets chunk-by-chunk directly into the memory-mapped array
    CHUNK_PROCESSING_BATCH_SIZE = 250_000
    current_col_offset = 0
    logging.info("Loading subsets via chunking directly into memory-mapped array...")
    for i, path in enumerate(tqdm(args.subset_paths, desc="Processing Subsets")):
        pq_file = pq.ParquetFile(path)
        
        if pq_file.metadata.num_rows != expected_rows:
            raise ValueError(f"FATAL: Row count mismatch in {path.name}! Index has {expected_rows}, but file metadata has {pq_file.metadata.num_rows}.")

        arrow_schema = pq_file.schema_arrow
        subset_feature_cols = [field.name for field in arrow_schema if (pa.types.is_floating(field.type) or pa.types.is_integer(field.type)) and field.name not in coord_cols]
        
        current_row = 0
        for batch in pq_file.iter_batches(batch_size=CHUNK_PROCESSING_BATCH_SIZE, columns=subset_feature_cols):
            batch_df = batch.to_pandas().fillna(0)
            num_rows_in_batch = len(batch_df)
            all_subset_features[current_row : current_row + num_rows_in_batch, current_col_offset:current_col_offset+len(subset_feature_cols)] = batch_df.values
            current_row += num_rows_in_batch
        
        all_subset_features.flush() # Ensure data is written to disk

        if i == 0:
            current_row_coords = 0
            for batch in pq_file.iter_batches(batch_size=CHUNK_PROCESSING_BATCH_SIZE, columns=coord_cols):
                batch_df = batch.to_pandas().fillna(0)
                num_rows_in_batch = len(batch_df)
                all_subset_coords[current_row_coords : current_row_coords + num_rows_in_batch, :] = batch_df.values
                current_row_coords += num_rows_in_batch
        
        current_col_offset += len(subset_feature_cols)
        del pq_file; gc.collect()
    
    logging.info(f"Consolidated subset data: features {all_subset_features.shape}, coordinates {all_subset_coords.shape}")
    
    if all_subset_features.shape[0] != expected_rows:
        raise ValueError("FATAL: Final feature matrix row count does not match index file.")
    logging.info("Validation successful: Final matrix has the correct number of rows.")

    try:
        # --- NEW: Identify column indices for different feature groups ---
        logging.info("Separating Price Paid (pp) features from other geospatial features for selective modeling.")
        pp_feature_indices = [i for i, col in enumerate(feature_cols) if col.startswith('pp_')]
        non_pp_feature_indices = [i for i, col in enumerate(feature_cols) if not col.startswith('pp_')]
        logging.info(f"Found {len(pp_feature_indices)} 'pp_' features and {len(non_pp_feature_indices)} other features.")

        # Create a memory-mapped view of the non-pp data for Atlas/Microscope to prevent data contamination.
        # This avoids creating a full copy in memory.
        atlas_microscope_features = all_subset_features[:, non_pp_feature_indices]

        # ==========================================================================
        # STAGE 1: ATLAS (GEOGRAPHIC CLUSTERING)
        # ==========================================================================
        logging.info(f"\n--- STAGE 1: ATLAS - Training Geographic Clusters (using non-pp features) ---")
        kmeans = MiniBatchKMeans(n_clusters=args.n_clusters, random_state=42, batch_size=4096, n_init='auto')
        # Fit the scaler ONLY on the non-pp data
        scaler = StandardScaler().fit(atlas_microscope_features[:100000])
        
        for i in tqdm(range(0, len(atlas_microscope_features), 100000), desc="Training K-Means"):
            batch_scaled = scaler.transform(atlas_microscope_features[i:i+100000])
            kmeans.partial_fit(batch_scaled)

        logging.info("Predicting Atlas cluster labels in batches...")
        batch_size = 100000
        all_subset_labels = np.zeros(len(all_subset_features), dtype=np.int32) 

        for i in tqdm(range(0, len(atlas_microscope_features), batch_size), desc="Predicting K-Means"):
            batch = atlas_microscope_features[i:i+batch_size]
            batch_scaled = scaler.transform(batch)
            all_subset_labels[i:i+batch_size] = kmeans.predict(batch_scaled)
        
        atlas_df = pd.DataFrame({'postcode': subset_postcodes, 'atlas_cluster_id': all_subset_labels})
        atlas_df['postcode_key'] = atlas_df['postcode'].apply(normalize_postcode_key)
        atlas_df = atlas_df.drop(columns=['postcode']).drop_duplicates(subset=['postcode_key'], keep='first')
        logging.info(f"Atlas mapping created with {len(atlas_df)} unique postcode mappings.")

        # ==========================================================================
        # STAGE 2: ATTRIBUTE JOIN (MASTER PROPERTIES + PP_DATA + GEO LOOKUP + ATLAS)
        # ==========================================================================
        logging.info(f"\n--- STAGE 2: ATTRIBUTE JOIN - Enriching Master Properties ---")
        master_df = pd.read_csv(args.master_properties_path, low_memory=False)
        
        # --- NEW: Load and merge the raw Price Paid history features FIRST ---
        logging.info(f"Loading raw Price Paid history from {args.subset_pp_path}")
        pp_df = pd.read_parquet(args.subset_pp_path)
        # The postcode index and the pp_subset are row-aligned.
        pp_df['postcode'] = subset_postcodes
        pp_df['postcode_key'] = pp_df['postcode'].apply(normalize_postcode_key)
        # Drop coordinate columns from pp_df to avoid conflicts
        pp_df.drop(columns=[c for c in ['pcd_latitude', 'pcd_longitude', 'postcode'] if c in pp_df.columns], inplace=True)
        
        master_df['postcode'] = master_df['property_id'].apply(extract_postcode_from_address)
        master_df['postcode_key'] = master_df['postcode'].apply(normalize_postcode_key)
        
        logging.info(f"Merging raw Price Paid features onto master dataset. Master shape: {master_df.shape}, PP shape: {pp_df.shape}")
        master_df = pd.merge(master_df, pp_df, on='postcode_key', how='left')
        logging.info(f"Merge complete. Master shape is now: {master_df.shape}")
        del pp_df; gc.collect()

        # --- IMPORTANT: Now that master_df is enriched, create the copy for the final merge ---
        original_master_df = master_df.copy()
        
        master_df.dropna(subset=['postcode_key'], inplace=True)
        id_cols_as_str = {'pcds': str, 'oa21cd': str, 'lsoa21cd': str, 'msoa21cd': str, 'ladcd': str}
        try:
            lookup_df = pd.read_csv(args.postcode_lookup_path, dtype=id_cols_as_str, encoding='utf-8')
        except UnicodeDecodeError:
            lookup_df = pd.read_csv(args.postcode_lookup_path, dtype=id_cols_as_str, encoding='latin-1')
        lookup_df['postcode_key'] = lookup_df['pcds'].apply(normalize_postcode_key)
        lookup_df.drop_duplicates(subset=['postcode_key'], keep='first', inplace=True)
        
        master_enriched_df = pd.merge(master_df, lookup_df, on='postcode_key', how='left')
        master_enriched_df = pd.merge(master_enriched_df, atlas_df, on='postcode_key', how='left')
        logging.info(f"Attribute joins complete. Shape: {master_enriched_df.shape}")
        del master_df, lookup_df, atlas_df; gc.collect()

        # ==========================================================================
        # STAGE 3: SPATIAL JOIN & AGGREGATION (COMPASS & MICROSCOPE DATA PREP)
        # ==========================================================================
        logging.info(f"\n--- STAGE 3: SPATIAL JOIN - Generating Compass & Microscope Features ---")

        # --- START: Integrate Advanced Compass Logic to Control Feature Explosion ---
        # Define the keywords to select only the most meaningful features for aggregation.
        # This is the logic from your older script that prevents creating 9000+ features.
        compass_keywords = ['price',                # Key market value signal (Subset 3, StreetScan)
                            'count',                # Market activity/liquidity (Subset 3)
                            'depriv',               # CRITICAL: Captures all deprivation metrics (ONS, StreetScan)
                            'income',               # CRITICAL: Direct economic signal (StreetScan)
                            'employ',               # CRITICAL: Detailed employment demographics (StreetScan)
                            'tenure',               # Strong demographic signal of stability/transience (Homipi, ONS)
                            'FI_Prop_',             # CRITICAL: Your engineered demographic proportions from ONS (Subset 2)
                            'score',                # Catches various pre-calculated indices (AHAH, Chimnie, etc.)
                            'ahah',                 # General health & environment access (Subset 4)
                            'LSOA_',                # Captures neighborhood-level market dynamics like churn/transactions (Subset 3)
                            '_rnk',                 # Includes all relative rank features (StreetScan, AHAH)
                            '_pct',                 # Includes all relative percentile features (AHAH)
                            "pp_"                   # New price paid features (Subset PP)
                            ]
        
        # Filter the full list of feature columns down to our selected source columns.
        compass_source_cols = [col for col in feature_cols if any(keyword in col.lower() for keyword in compass_keywords)]
        
        # Create a mapping from the filtered column names back to their original index in the numpy array
        compass_col_indices = {col: i for i, col in enumerate(feature_cols) if col in compass_source_cols}
        
        logging.info(f"Identified {len(feature_cols)} total subset features. Filtering down to {len(compass_source_cols)} source columns for Compass features based on keywords.")
        # --- END: Integrate Advanced Compass Logic ---

        kdtree = KDTree(all_subset_coords)
        max_k = max(args.n_neighbors_list)
        property_coords = master_enriched_df[['pcd_latitude', 'pcd_longitude']].fillna(0).values
        
        num_workers = max(1, min(4, os.cpu_count() // 2)) 
        logging.info(f"Querying KDTree with {num_workers} workers to find {max_k} nearest neighbors...")
        _, neighbor_indices = kdtree.query(property_coords, k=max_k, workers=num_workers)
        
        # CRITICAL: Select nearest neighbor features BUT ONLY from the non-pp columns for Microscope input.
        microscope_geo_features = atlas_microscope_features[neighbor_indices[:, 0]]
        
        chunk_size = 50000 
        compass_chunks = []
        logging.info(f"Aggregating Compass features in chunks of {chunk_size}...")
        for i in tqdm(range(0, len(master_enriched_df), chunk_size), desc="Aggregating Compass Features"):
            chunk_indices = neighbor_indices[i:i+chunk_size]
            chunk_compass_features = []
            for j in range(len(chunk_indices)):
                prop_compass = {}
                # Get all features for the neighbors of the current property
                current_neighbor_features = all_subset_features[chunk_indices[j]]
                
                for n in args.n_neighbors_list:
                    n_slice = current_neighbor_features[:n, :]
                    mean_f = np.nanmean(n_slice, axis=0)
                    std_f = np.nanstd(n_slice, axis=0)
                    
                    # --- MODIFIED LOOP: Iterate over the filtered list of columns ---
                    for col_name, original_idx in compass_col_indices.items():
                        prop_compass[f'compass_mean_{col_name}_n{n}'] = mean_f[original_idx]
                        prop_compass[f'compass_std_{col_name}_n{n}'] = std_f[original_idx]
                
                chunk_compass_features.append(prop_compass)
            
            chunk_df = pd.DataFrame(chunk_compass_features).fillna(0)
            compass_chunks.append(chunk_df)

        compass_df = pd.concat(compass_chunks, ignore_index=True)
        compass_df.index = master_enriched_df.index
        logging.info(f"Compass features generated. Shape: {compass_df.shape}")

        # ==========================================================================
        # STAGE 4: MICROSCOPE (AUTOENCODER TRAINING & INFERENCE)
        # ==========================================================================
        logging.info(f"\n--- STAGE 4: MICROSCOPE - Training Cluster Autoencoders ---")
        property_feature_cols = master_enriched_df.select_dtypes(include=np.number).columns.tolist()
        property_features_np = master_enriched_df[property_feature_cols].fillna(0).values
        
        microscope_training_data = np.hstack([property_features_np, microscope_geo_features])
        scaler_microscope = StandardScaler().fit(microscope_training_data)
        microscope_training_data_scaled = scaler_microscope.transform(microscope_training_data)
        
        trained_aes = {}
        for cluster_id in tqdm(master_enriched_df['atlas_cluster_id'].dropna().unique(), desc="Training AEs per cluster"):
            indices = master_enriched_df[master_enriched_df['atlas_cluster_id'] == cluster_id].index
            if len(indices) < 20: continue
            
            cluster_data_tensor = torch.FloatTensor(microscope_training_data_scaled[indices]).to(device)
            input_dim = cluster_data_tensor.shape[1]
            trained_aes[cluster_id] = train_autoencoder(cluster_data_tensor, input_dim, args.embedding_dim, args.epochs, device)

        logging.info(f"Microscope training complete. Trained {len(trained_aes)} autoencoders.")
        
        all_embeddings = np.zeros((len(master_enriched_df), args.embedding_dim))
        full_dataset_tensor = torch.FloatTensor(microscope_training_data_scaled).to(device)
        for cluster_id, model in tqdm(trained_aes.items(), desc="Generating Embeddings"):
            indices = master_enriched_df[master_enriched_df['atlas_cluster_id'] == cluster_id].index
            with torch.no_grad():
                embeddings = model.get_embedding(full_dataset_tensor[indices]).cpu().numpy()
                all_embeddings[indices] = embeddings

        microscope_cols = [f'microscope_emb_{i}' for i in range(args.embedding_dim)]
        microscope_df = pd.DataFrame(all_embeddings, columns=microscope_cols, index=master_enriched_df.index)
        logging.info(f"Microscope embeddings generated. Shape: {microscope_df.shape}")

        # ==========================================================================
        # STAGE 5: FINAL CONSOLIDATION & SAVE
        # ==========================================================================
        logging.info(f"\n--- STAGE 5: Consolidating all features and saving final dataset ---")

        # The compass_df and microscope_df are indexed correctly to align with master_enriched_df.
        # We need to merge them back into the original master dataset which contains all properties.
        # The 'original_master_df' already contains the raw 'pp_*' features from STAGE 2.
        
        # Create a temporary dataframe for merging, using the postcode_key
        geospatial_features_to_merge = master_enriched_df[['postcode_key']].copy()
        geospatial_features_to_merge = pd.concat([geospatial_features_to_merge, compass_df, microscope_df], axis=1)
        geospatial_features_to_merge.drop_duplicates(subset=['postcode_key'], inplace=True)
        
        logging.info(f"Merging Compass ({compass_df.shape}) and Microscope ({microscope_df.shape}) features back to master dataset.")
        final_df = pd.merge(original_master_df, geospatial_features_to_merge, on='postcode_key', how='left')
        
        # Post-merge cleanup and validation
        final_df.drop(columns=['postcode_key', 'postcode_x', 'postcode_y'], errors='ignore', inplace=True)
        if 'postcode' in final_df.columns:
            final_df.rename(columns={'postcode': 'pcds'}, inplace=True)
        logging.info(f"Final consolidated dataset created. Shape: {final_df.shape}")
        
        output_path = args.output_dir / "final_geospatial_enriched_dataset.parquet"
        logging.info(f"Saving final dataset to {output_path}")
        final_df.to_parquet(output_path, index=False, compression='gzip')
        logging.info("Final dataset saved successfully.")

    finally:
        # --- CLEANUP ---
        # Ensure the large memory-mapped file is deleted
        logging.info("Cleaning up temporary memory-mapped file...")
        # The 'all_subset_features' variable might not exist if an error happened before its creation
        if 'all_subset_features' in locals() and isinstance(all_subset_features, np.memmap):
            # The memmap object must be closed before the file can be deleted on some systems
            if all_subset_features.filename and os.path.exists(all_subset_features.filename):
                # The ._mmap attribute holds the file handle
                all_subset_features._mmap.close()
                os.remove(all_subset_features.filename)
                logging.info(f"Successfully removed {all_subset_features.filename}")

if __name__ == "__main__":
    main()
EOL

# --- Data Download ---
echo "--- Downloading all required datasets from GCS... ---"
gsutil -m cp "${MASTER_PROPERTY_DATASET_GCS_PATH}" data/master_property_dataset.csv
gsutil -m cp "${POSTCODE_LOOKUP_GCS_PATH}" data/postcode_lookup.csv
gsutil -m cp "${POSTCODE_INDEX_GCS_PATH}" data/postcode_index.csv
gsutil -m cp "${GLOBAL_SUBSET1_PATH}" data/subset1.parquet
gsutil -m cp "${GLOBAL_SUBSET2_PATH}" data/subset2.parquet
gsutil -m cp "${GLOBAL_SUBSET3_PATH}" data/subset3.parquet
gsutil -m cp "${GLOBAL_SUBSET4_PATH}" data/subset4.parquet
gsutil -m cp "${GLOBAL_SUBSET5_PATH}" data/subset5.parquet
gsutil -m cp "${GLOBAL_SUBSET_PP_PATH}" data/subset_pp.parquet
echo "--- All data downloads complete. ---"

# --- Execute the Full Enrichment Script ---
echo "--- EXECUTING SCRIPT: 01_run_full_enrichment.py ---"
python3 01_run_full_enrichment.py \
    --master_properties_path "data/master_property_dataset.csv" \
    --postcode_lookup_path "data/postcode_lookup.csv" \
    --postcode_index_path "data/postcode_index.csv" \
    --subset_pp_path "data/subset_pp.parquet" \
    --subset_paths "data/subset1.parquet" "data/subset2.parquet" "data/subset3.parquet" "data/subset4.parquet" "data/subset5.parquet" "data/subset_pp.parquet" \
    --n_neighbors_list ${COMPASS_N_NEIGHBORS_LIST[@]} \
    --n_clusters ${ATLAS_N_CLUSTERS} \
    --embedding_dim ${MICROSCOPE_EMBEDDING_DIM} \
    --epochs ${MICROSCOPE_EPOCHS} \
    --output_dir "artifacts" # Pass the base directory now
echo "--- SCRIPT COMPLETE. ---"


# --- Finalization: Upload All Artifacts ---
echo "--- Uploading all generated artifacts to GCS... ---"
gsutil -m cp -r artifacts/* "gs://${GCS_BUCKET}/${ARTIFACTS_GCS_DIR}/"
echo "--- Artifact upload complete. ---"

echo "--- Uploading execution log to GCS... ---"
gsutil -m cp "${LOG_FILE}" "gs://${GCS_BUCKET}/${LOG_FILE_GCS_PATH}"

echo "--- Full Geospatial Enrichment Pipeline V15 Finished Successfully: $(date) ---"