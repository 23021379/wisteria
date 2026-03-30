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

# --- OUTPUT ---
ARTIFACTS_GCS_DIR="features/geospatial_pipeline_v15_full" # No timestamp
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
    parser.add_argument("--postcode_index_path", type=Path, required=True) # <-- NEW ARGUMENT
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
    
    # 2. Load subsets, validate, and prepare for horizontal merge
    feature_arrays = []
    all_feature_cols = []
    all_subset_coords = None
    
    logging.info("Loading and validating subsets for horizontal merge...")
    for i, path in enumerate(tqdm(args.subset_paths, desc="Processing Subsets")):
        df = pd.read_parquet(path)
        
        # CRITICAL VALIDATION for this subset
        if len(df) != expected_rows:
            raise ValueError(f"FATAL: Row count mismatch in {path.name}! Index has {expected_rows}, but subset has {len(df)}.")
            
        # Extract features for this subset
        subset_feature_cols = [c for c in df.columns if c not in coord_cols and pd.api.types.is_numeric_dtype(df[c])]
        feature_arrays.append(df[subset_feature_cols].fillna(0).values)
        all_feature_cols.extend(subset_feature_cols)
        
        # Get coordinates from the first subset only
        if i == 0:
            all_subset_coords = df[coord_cols].fillna(0).values
            
        del df; gc.collect()

    # 3. Horizontally stack all feature arrays
    logging.info("Horizontally stacking feature arrays...")
    all_subset_features = np.hstack(feature_arrays)
    feature_cols = all_feature_cols # Use this for downstream code
    
    logging.info(f"Consolidated subset data: features {all_subset_features.shape}, coordinates {all_subset_coords.shape}")
    
    # Final validation
    if all_subset_features.shape[0] != expected_rows:
        raise ValueError("FATAL: Final feature matrix row count does not match index file.")
    logging.info("Validation successful: Final matrix has the correct number of rows.")

    # ==========================================================================
    # STAGE 1: ATLAS (GEOGRAPHIC CLUSTERING)
    # ==========================================================================
    logging.info(f"\n--- STAGE 1: ATLAS - Training Geographic Clusters ---")
    kmeans = MiniBatchKMeans(n_clusters=args.n_clusters, random_state=42, batch_size=4096, n_init='auto')
    scaler = StandardScaler().fit(all_subset_features[:100000])
    
    for i in tqdm(range(0, len(all_subset_features), 100000), desc="Training K-Means"):
        batch_scaled = scaler.transform(all_subset_features[i:i+100000])
        kmeans.partial_fit(batch_scaled)

    # --- MEMORY FIX: Predict K-Means labels in batches to avoid OOM error ---
    logging.info("Predicting Atlas cluster labels in batches...")
    batch_size = 100000  # Use the same batch size as training for consistency
    # Pre-allocate numpy array for efficiency instead of appending to a list
    all_subset_labels = np.zeros(len(all_subset_features), dtype=np.int32) 

    for i in tqdm(range(0, len(all_subset_features), batch_size), desc="Predicting K-Means"):
        batch = all_subset_features[i:i+batch_size]
        batch_scaled = scaler.transform(batch)
        all_subset_labels[i:i+batch_size] = kmeans.predict(batch_scaled)
    
    atlas_df = pd.DataFrame({'postcode': subset_postcodes, 'atlas_cluster_id': all_subset_labels})
    atlas_df['postcode_key'] = atlas_df['postcode'].apply(normalize_postcode_key)
    atlas_df = atlas_df.drop(columns=['postcode']).drop_duplicates(subset=['postcode_key'], keep='first')
    logging.info(f"Atlas mapping created with {len(atlas_df)} unique postcode mappings.")

    # ==========================================================================
    # STAGE 2: ATTRIBUTE JOIN (MASTER PROPERTIES + GEO LOOKUP + ATLAS CLUSTERS)
    # ==========================================================================
    logging.info(f"\n--- STAGE 2: ATTRIBUTE JOIN - Enriching Master Properties ---")
    master_df = pd.read_csv(args.master_properties_path, low_memory=False)
    
    # --- FIX: Create a copy of the original master dataframe for the final merge ---
    original_master_df = master_df.copy()
    
    master_df['postcode'] = master_df['property_id'].apply(extract_postcode_from_address)
    master_df['postcode_key'] = master_df['postcode'].apply(normalize_postcode_key)
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
                        '_pct'                  # Includes all relative percentile features (AHAH)
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
    
    microscope_geo_features = all_subset_features[neighbor_indices[:, 0]]
    
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
    # --- STAGE 5: FINAL MERGE & DUAL OUTPUT (REVISED) ---
    # ==========================================================================
    logging.info("\n--- STAGE 5: Generating and Saving Final Artifacts ---")

    # --- 5a. Create and save the standalone Contextual (ACM) Features file ---
    logging.info("Creating standalone contextual features (ACM) file...")
    contextual_df = pd.concat([
        master_enriched_df[['property_id', 'atlas_cluster_id']],
        compass_df,
        microscope_df
    ], axis=1)
    
    contextual_output_dir = args.output_dir / "contextual_features"
    contextual_output_dir.mkdir(parents=True, exist_ok=True)
    contextual_output_path = contextual_output_dir / "contextual_features.parquet"
    contextual_df.to_parquet(contextual_output_path, index=False)
    logging.info(f"Contextual features file saved to: {contextual_output_path}")

    # --- 5b. Create and save the final, fully merged master dataset ---
    logging.info("Creating final enriched master dataset...")
    # Merge the original master data with the new contextual features
    final_df = pd.merge(
        original_master_df,
        contextual_df,
        on='property_id',
        how='left'
    )
    final_df.columns = ["".join(c if c.isalnum() else '_' for c in str(x)) for x in final_df.columns]
    
    final_output_path = args.output_dir / "final_enriched_master_dataset.parquet"
    final_df.to_parquet(final_output_path, index=False)
    logging.info(f"Pipeline complete. Final dataset shape: {final_df.shape}")
    logging.info(f"Final enriched master dataset saved to: {final_output_path}")

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
echo "--- All data downloads complete. ---"

# --- Execute the Full Enrichment Script ---
echo "--- EXECUTING SCRIPT: 01_run_full_enrichment.py ---"
python3 01_run_full_enrichment.py \
    --master_properties_path "data/master_property_dataset.csv" \
    --postcode_lookup_path "data/postcode_lookup.csv" \
    --postcode_index_path "data/postcode_index.csv" \
    --subset_paths "data/subset1.parquet" "data/subset2.parquet" "data/subset3.parquet" "data/subset4.parquet" "data/subset5.parquet" \
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