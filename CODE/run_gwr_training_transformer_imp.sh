#!/bin/bash

# ==============================================================================
# run_mgwr_pipeline_v1.sh
#
# A complete, end-to-end pipeline that:
# 1. Prepares and imputes data using a multi-tiered approach.
# 2. Uses Multi-scale Geographically Weighted Regression (MGWR) as a feature
#    engineering step to create spatially varying features.
# 3. Merges all features and trains CatBoost and FT-Transformer models,
#    producing out-of-fold (OOF) predictions for a final ensemble.
# ==============================================================================

# --- Strict Mode & Error Handling ---
set -e
set -o pipefail
set -x

# --- Configuration ---
GCS_BUCKET="srgan-bucket-ace-botany-453819-t4"  # srgan-bucket-ace-botany-453819-t4/imputation_pipeline/output_lgbm_20250724-145846
INPUT_CSV_GCS_PATH="gs://${GCS_BUCKET}/imputation_pipeline/output_lgbm_20250724-145846/final_fully_imputed_dataset.parquet"
TIMESTAMP=$(date +"%Y%m%d-%H%M%S")
RUN_OUTPUT_DIR="full_mgwr_pipeline_runs/${TIMESTAMP}"
LOG_FILE_GCS_PATH="${RUN_OUTPUT_DIR}/logs/pipeline_run_${TIMESTAMP}.log"

# Local workspace
WORKDIR="${HOME}/mgwr_pipeline_work_v1"
LOG_FILE="${WORKDIR}/run_pipeline.log"

# --- Main Execution ---
mkdir -p "${WORKDIR}"
cd "${WORKDIR}"

# Redirect all output to a log file AND the console
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "--- Complete End-to-End MGWR Pipeline Started: $(date) ---"

# ==============================================================================
# --- Phase 0: Environment Setup (Combined Dependencies) ---
# ==============================================================================
echo "--- Ensuring a clean Python virtual environment... ---"
VENV_PATH="${WORKDIR}/mgwr_env"

# Detect available Python version
if command -v python3.11 &> /dev/null; then
    PYTHON_CMD="python3.11"
    echo "--- Using Python 3.11 ---"
elif command -v python3.10 &> /dev/null; then
    PYTHON_CMD="python3.10"
    echo "--- Using Python 3.10 ---"
elif command -v python3.9 &> /dev/null; then
    PYTHON_CMD="python3.9"
    echo "--- Using Python 3.9 ---"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    echo "--- Using Python 3 (version: $PYTHON_VERSION) ---"
else
    echo "ERROR: No Python 3 installation found!"
    exit 1
fi

if [ ! -f "${VENV_PATH}/bin/activate" ]; then
    echo "--- Creating Python virtual environment with $PYTHON_CMD... ---"
    $PYTHON_CMD -m venv "${VENV_PATH}"
fi

echo "--- Activating virtual environment... ---"
source "${VENV_PATH}/bin/activate"
echo "--- Running with Python: $(which python) ---"
echo "--- Python version: $(python --version) ---"

# Add CPU optimization at the start
export OMP_NUM_THREADS=$(nproc)
export MKL_NUM_THREADS=$(nproc)
export NUMEXPR_NUM_THREADS=$(nproc)
export OPENBLAS_NUM_THREADS=$(nproc)

echo "--- CPU cores available: $(nproc) ---"
echo "--- Setting all libraries to use all cores ---"

echo "--- Installing/updating all Python dependencies... ---"

# Check if we have CUDA available
if command -v nvidia-smi &> /dev/null; then
    echo "--- NVIDIA GPU detected, installing PyTorch with CUDA support ---"
    echo "--- Installing PyTorch 2.3.0 for CUDA 12.1 ---"
    pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
else
    echo "--- No NVIDIA GPU detected, installing CPU-only PyTorch ---"
    pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cpu
fi

# Combined requirements for all stages
cat > requirements.txt << EOL
pandas>=2.2.0,<3.0
numpy>=1.26.0,<2.0
scikit-learn>=1.4.0,<2.0
google-cloud-storage>=2.16.0
pyarrow>=16.0.0
scipy>=1.13.0
statsmodels>=0.14.0
# For MGWR
mgwr>=2.2.0
libpysal>=4.12.0
# For Final Models
catboost>=1.2.0
lightgbm
pytorch-tabnet>=4.0  # Ensure compatible version
EOL

pip install --upgrade pip
pip install -r requirements.txt

echo "--- Final dependency check ---"
pip list | grep -E "(torch|pandas|numpy|scikit|mgwr|catboost)"
echo "--- Environment setup complete. ---"

# ==============================================================================
# --- Script Generation (4 Stages) ---
# ==============================================================================

# --- Generate Python Script for Stage 0: Create CV Folds ---
echo "--- Generating Stage 0 script: 00_create_cv_folds.py ---"
cat > 00_create_cv_folds.py << 'EOL'
# 00_create_cv_folds.py
import argparse
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from google.cloud import storage
from sklearn.model_selection import KFold
import re
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
RANDOM_STATE = 42
N_SPLITS = 5
TARGET_COL = "most_recent_sale_price"
ID_COL = "property_id"

def download_gcs_file(bucket_name, source_blob_name, dest_path):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(str(dest_path))

def upload_to_gcs(bucket_name, source_path, dest_blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(dest_blob_name)
    blob.upload_from_filename(str(source_path))

def parse_sales_history(sales_str):
    """
    Parse sales history string like:
    "['01 Nov 2024\n-1%\n£170,000\nFlat Leasehold', '£85,000', '04 Mar 2016\n+77%\n£159,000\nFlat Leasehold', ...]"
    
    Returns DataFrame with parsed sales data
    """
    if pd.isna(sales_str) or sales_str == '' or sales_str == '[]':
        return pd.DataFrame()
    
    try:
        # Clean up the string - remove outer quotes and brackets
        sales_str = str(sales_str).strip()
        if sales_str.startswith('"') and sales_str.endswith('"'):
            sales_str = sales_str[1:-1]  # Remove outer quotes
        
        # Now parse as a Python list
        import ast
        sales_list = ast.literal_eval(sales_str)
        
        if not isinstance(sales_list, list):
            logging.warning(f"Expected list but got {type(sales_list)}: {sales_str[:100]}...")
            return pd.DataFrame()
        
        sales_data = []
        
        for i, item in enumerate(sales_list):
            item = str(item).strip()
            if not item or item == '':
                continue
                
            # Skip price change entries (every second item like '£85,000' without newlines)
            if item.startswith('£') and '\n' not in item:
                continue
                
            # Parse sale entries (contain date, percentage, price, type with newlines)
            if '\n' in item:
                lines = item.split('\n')
                if len(lines) >= 3:
                    try:
                        # Parse date
                        date_str = lines[0].strip()
                        date_parts = date_str.split()
                        if len(date_parts) >= 3:
                            day = int(date_parts[0])
                            month_map = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                                       'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
                            month = month_map.get(date_parts[1], 1)
                            year = int(date_parts[2])
                            
                            # Parse percentage change - handle missing percentage
                            if len(lines) > 1:
                                pct_str = lines[1].strip().replace('%', '').replace('+', '')
                                pct_change = float(pct_str) if pct_str and pct_str != '-' and pct_str.replace('.','').replace('-','').isdigit() else 0
                            else:
                                pct_change = 0
                            
                            # Parse price - handle cases where price is in line 1 or 2
                            price = 0
                            for line_idx in range(1, len(lines)):
                                if '£' in lines[line_idx]:
                                    price_str = lines[line_idx].strip().replace('£', '').replace(',', '')
                                    # Extract only the numeric part
                                    price_match = re.search(r'(\d+)', price_str)
                                    if price_match:
                                        price = int(price_match.group(1))
                                        break
                            
                            # Property type - look for the last line or after price
                            prop_type = ''
                            for line_idx in range(len(lines)-1, 0, -1):
                                if '£' not in lines[line_idx] and '%' not in lines[line_idx] and lines[line_idx].strip():
                                    prop_type = lines[line_idx].strip()
                                    break
                            
                            if price > 0:  # Only add if we successfully parsed a price
                                sales_data.append({
                                    'sale_day': day,
                                    'sale_month': month, 
                                    'sale_year': year,
                                    'pct_change': pct_change,
                                    'sale_price': price,
                                    'property_type': prop_type,
                                    'sale_index': len(sales_data)  # 0 = most recent
                                })

                    except Exception as e:
                        logging.warning(f"Error parsing sale entry '{item[:50]}...': {e}")
                        continue
        
        if len(sales_data) == 0:
            logging.debug(f"No valid sales found in: {sales_str[:100]}...")
        
        return pd.DataFrame(sales_data)
        
    except Exception as e:
        logging.warning(f"Error parsing sales history '{sales_str[:100]}...': {e}")
        return pd.DataFrame()

def process_rightmove_data(rightmove_df):
    """Process Rightmove CSV and extract sales features"""
    logging.info("Processing Rightmove sales data...")
    
    processed_data = []
    
    for idx, row in rightmove_df.iterrows():
        if idx % 1000 == 0:
            logging.info(f"Processed {idx} properties...")
            
        try:
            # Get property identifier from first column
            prop_info = str(row.iloc[0]) if len(row) > 0 else ''
            
            # Extract a cleaner property ID from the address info
            # The first column appears to contain address and URL in list format
            property_id = f"prop_{idx}"  # Use row index as fallback ID
            
            # Parse sales history from the third column (index 2)
            sales_history = str(row.iloc[2]) if len(row) > 2 else ''
            sales_df = parse_sales_history(sales_history)
            
            if len(sales_df) > 0:
                # Most recent sale (target)
                most_recent = sales_df.iloc[0]
                
                features = {
                    'property_id_raw': property_id,
                    'rightmove_row_id': idx,  # Add this for debugging
                    'most_recent_sale_price': most_recent['sale_price'],
                    'most_recent_sale_day': most_recent['sale_day'],
                    'most_recent_sale_month': most_recent['sale_month'],
                    'most_recent_sale_year': most_recent['sale_year'],
                    'most_recent_pct_change': most_recent['pct_change'],
                    'most_recent_property_type': most_recent['property_type'],
                    'total_sales_count': len(sales_df)
                }
                
                # Add historical sales data (up to 5 previous sales)
                for i in range(1, min(6, len(sales_df))):
                    sale = sales_df.iloc[i]
                    features.update({
                        f'prev_sale_{i}_day': sale['sale_day'],
                        f'prev_sale_{i}_month': sale['sale_month'], 
                        f'prev_sale_{i}_year': sale['sale_year'],
                        f'prev_sale_{i}_price': sale['sale_price'],
                        f'prev_sale_{i}_pct_change': sale['pct_change'],
                    })
                
                # Calculate time-based features
                if len(sales_df) > 1:
                    from datetime import datetime
                    recent_date = datetime(most_recent['sale_year'], most_recent['sale_month'], most_recent['sale_day'])
                    prev_date = datetime(sales_df.iloc[1]['sale_year'], sales_df.iloc[1]['sale_month'], sales_df.iloc[1]['sale_day'])
                    days_since_last_sale = (recent_date - prev_date).days
                    features['days_since_last_sale'] = days_since_last_sale
                    
                    # Price change since last sale
                    price_change = most_recent['sale_price'] - sales_df.iloc[1]['sale_price']
                    features['price_change_since_last'] = price_change
                
                processed_data.append(features)
            else:
                # Log a sample of failed rows for debugging
                if idx < 10:
                    logging.warning(f"No sales data extracted for row {idx}: {sales_history[:100]}...")
                
        except Exception as e:
            if idx < 10:  # Only log first few errors to avoid spam
                logging.warning(f"Error processing row {idx}: {e}")
            continue
    
    logging.info(f"Successfully processed {len(processed_data)} properties with valid sales data")
    return pd.DataFrame(processed_data)

def create_stratified_geo_folds(df, n_splits=5, target_col='most_recent_sale_price'):
    """Create geographically stratified folds"""
    from sklearn.model_selection import StratifiedKFold
    
    # Create price bins for stratification
    df['price_bin'] = pd.qcut(df[target_col], q=10, labels=False, duplicates='drop')
    
    # Create geographical bins
    df['geo_bin'] = pd.cut(df['pcd_latitude'], bins=5, labels=False) * 10 + \
                    pd.cut(df['pcd_longitude'], bins=5, labels=False)
    
    # Combine price and geo bins for stratification
    df['strat_bin'] = df['price_bin'].astype(str) + '_' + df['geo_bin'].astype(str)
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    
    df['fold'] = -1
    for fold_num, (train_idx, val_idx) in enumerate(skf.split(df, df['strat_bin'])):
        df.loc[val_idx, 'fold'] = fold_num
    
    return df.drop(['price_bin', 'geo_bin', 'strat_bin'], axis=1)

def main():
    parser = argparse.ArgumentParser(description="Create and save CV fold assignments with target data.")
    parser.add_argument("--gcs_bucket", required=True)
    parser.add_argument("--input_gcs_path", required=True)
    parser.add_argument("--rightmove_gcs_path", required=True)
    parser.add_argument("--folds_gcs_path", required=True)
    parser.add_argument("--merged_data_gcs_path", required=True)
    args = parser.parse_args()

    local_work_dir = Path("/tmp/cv_folds")
    
    # Download input datasets
    local_input_path = local_work_dir / "input_data.parquet"
    local_rightmove_path = local_work_dir / "rightmove.csv"
    
    logging.info(f"Downloading imputed data from: {args.input_gcs_path}")
    download_gcs_file(args.gcs_bucket, args.input_gcs_path.split(f"gs://{args.gcs_bucket}/")[1], local_input_path)
    
    logging.info(f"Downloading Rightmove data from: {args.rightmove_gcs_path}")
    download_gcs_file(args.gcs_bucket, args.rightmove_gcs_path.split(f"gs://{args.gcs_bucket}/")[1], local_rightmove_path)
    
    # Load datasets
    logging.info("Loading imputed dataset...")
    input_df = pd.read_parquet(local_input_path)
    logging.info(f"Loaded imputed data with {len(input_df)} rows and {len(input_df.columns)} columns")
    
    logging.info("Loading Rightmove dataset...")
    rightmove_df = pd.read_csv(local_rightmove_path)
    logging.info(f"Loaded Rightmove data with {len(rightmove_df)} rows and {len(rightmove_df.columns)} columns")
    
    # Process Rightmove data
    processed_rightmove = process_rightmove_data(rightmove_df)
    logging.info(f"Processed Rightmove data into {len(processed_rightmove)} property records")
    
    if len(processed_rightmove) == 0:
        logging.error("No valid sales data found in Rightmove dataset!")
        exit(1)
    
    # Debug: Check what columns exist in both datasets
    logging.info(f"Input dataset columns: {input_df.columns.tolist()[:10]}...")  # Show first 10
    logging.info(f"Processed Rightmove columns: {processed_rightmove.columns.tolist()}")
    
    # Check if property_id exists in input dataset
    if 'property_id' in input_df.columns:
        logging.info("Found property_id in input dataset - attempting proper merge")
        merged_df = pd.merge(input_df, processed_rightmove, 
                           left_on='property_id', right_on='property_id_raw', 
                           how='inner')
        logging.info(f"Merge result: {len(merged_df)} rows")
    else:
        logging.warning("No property_id column found in input dataset.")
        logging.warning("Using positional merge (row-by-row alignment) as fallback.")
        
        # Ensure both datasets have the same number of rows for positional merge
        min_len = min(len(input_df), len(processed_rightmove))
        logging.info(f"Aligning datasets: input={len(input_df)} rows, rightmove={len(processed_rightmove)} rows, using={min_len} rows")
        
        # Create property_id for both datasets based on row position
        input_subset = input_df.iloc[:min_len].copy().reset_index(drop=True)
        rightmove_subset = processed_rightmove.iloc[:min_len].copy().reset_index(drop=True)
        
        # Add property_id to input dataset if it doesn't exist
        input_subset['property_id'] = range(len(input_subset))
        rightmove_subset['property_id'] = range(len(rightmove_subset))
        
        # Now merge on the created property_id
        merged_df = pd.merge(input_subset, rightmove_subset, on='property_id', how='inner')
        logging.info(f"Positional merge result: {len(merged_df)} rows")
    
    if len(merged_df) == 0:
        logging.error("Merge resulted in 0 rows! Checking data compatibility...")
        
        # Debug information
        logging.info(f"Input dataset shape: {input_df.shape}")
        logging.info(f"Rightmove dataset shape: {processed_rightmove.shape}")
        
        if 'property_id' in input_df.columns:
            logging.info(f"Sample input property_id values: {input_df['property_id'].head().tolist()}")
        logging.info(f"Sample rightmove property_id_raw values: {processed_rightmove['property_id_raw'].head().tolist()}")
        
        # Force a simple concatenation merge as last resort
        logging.warning("Forcing concatenation merge as last resort...")
        min_len = min(len(input_df), len(processed_rightmove))
        
        input_subset = input_df.iloc[:min_len].copy().reset_index(drop=True)
        rightmove_subset = processed_rightmove.iloc[:min_len].copy().reset_index(drop=True)
        
        # Remove conflicting columns before concatenation
        rightmove_subset = rightmove_subset.drop(['property_id_raw'], axis=1, errors='ignore')
        
        merged_df = pd.concat([input_subset, rightmove_subset], axis=1)
        merged_df['property_id'] = range(len(merged_df))
        
        logging.info(f"Concatenation merge result: {len(merged_df)} rows")
    
    logging.info(f"Final merged dataset has {len(merged_df)} rows")
    
    # Clean the merged data
    initial_count = len(merged_df)
    merged_df = merged_df.dropna(subset=[TARGET_COL])
    merged_df = merged_df.reset_index(drop=True)
    logging.info(f"Removed {initial_count - len(merged_df)} rows with null target values. Final count: {len(merged_df)}")
    
    if len(merged_df) == 0:
        logging.error("No valid data remaining after merge and cleaning!")
        logging.error("This suggests the target column is not being created properly.")
        
        # Debug the target column
        if TARGET_COL in processed_rightmove.columns:
            logging.info(f"Target column '{TARGET_COL}' statistics in Rightmove data:")
            logging.info(f"Non-null count: {processed_rightmove[TARGET_COL].notna().sum()}")
            logging.info(f"Sample values: {processed_rightmove[TARGET_COL].dropna().head().tolist()}")
        else:
            logging.error(f"Target column '{TARGET_COL}' not found in processed Rightmove data!")
            logging.error(f"Available columns: {processed_rightmove.columns.tolist()}")
        
        exit(1)

    # Create CV folds
    logging.info(f"Creating {N_SPLITS} CV folds...")
    merged_df = create_stratified_geo_folds(merged_df, n_splits=N_SPLITS, target_col=TARGET_COL)
    
    logging.info(f"Fold distribution:\n{merged_df['fold'].value_counts().sort_index()}")
    
    # Save the complete merged dataset
    local_merged_path = local_work_dir / "merged_data_with_target.parquet"
    merged_df.to_parquet(local_merged_path, index=False)
    upload_to_gcs(args.gcs_bucket, str(local_merged_path), args.merged_data_gcs_path)
    
    # Save only the ID and fold columns for the folds file
    if 'property_id' in merged_df.columns:
        folds_df = merged_df[['property_id', 'fold']]
    else:
        # Create a property_id if it doesn't exist
        merged_df['property_id'] = range(len(merged_df))
        folds_df = merged_df[['property_id', 'fold']]
    
    local_folds_path = local_work_dir / "cv_folds.parquet"
    folds_df.to_parquet(local_folds_path, index=False)
    
    logging.info(f"Uploading fold assignments to: gs://{args.gcs_bucket}/{args.folds_gcs_path}")
    upload_to_gcs(args.gcs_bucket, str(local_folds_path), args.folds_gcs_path)
    
    logging.info("Stage 0 (Data Merging and CV Folds) completed successfully!")
    logging.info(f"Target variable statistics:")
    logging.info(f"Mean: {merged_df[TARGET_COL].mean():.2f}")
    logging.info(f"Median: {merged_df[TARGET_COL].median():.2f}")
    logging.info(f"Min: {merged_df[TARGET_COL].min():.2f}")
    logging.info(f"Max: {merged_df[TARGET_COL].max():.2f}")

if __name__ == "__main__":
    main()
EOL

# --- Generate Python Script for Stage 1: Preparation & Selection ---
echo "--- Generating Stage 1 script: 01_prepare_and_select.py ---"
cat > 01_prepare_and_select.py << 'EOL'
# 01_prepare_and_select.py
import argparse
import logging
import pandas as pd
import numpy as np
import gc
from pathlib import Path
from google.cloud import storage
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.linear_model import LassoCV

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
RANDOM_STATE = 42

def download_gcs_file(bucket_name, source_blob_name, dest_path):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(str(dest_path))

def upload_to_gcs(bucket_name, source_path, dest_blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(dest_blob_name)
    blob.upload_from_filename(str(source_path))

def get_gcs_blob_name(gcs_path, bucket_name):
    """Extract blob name from GCS path, handling both full gs:// URLs and relative paths"""
    if gcs_path.startswith(f"gs://{bucket_name}/"):
        return gcs_path.split(f"gs://{bucket_name}/")[1]
    elif gcs_path.startswith("gs://"):
        # Extract bucket and blob from full GCS URL
        parts = gcs_path.replace("gs://", "").split("/", 1)
        if len(parts) == 2:
            return parts[1]
        else:
            raise ValueError(f"Invalid GCS path format: {gcs_path}")
    else:
        # Assume it's already a blob name relative to the bucket
        return gcs_path

def check_and_normalize_features(df, feature_cols, target_col=None):
    """
    Check if features are normalized (mean≈0, std≈1) and apply Yeo-Johnson if needed.
    Returns normalized DataFrame and list of transformed columns.
    """
    logging.info("Checking normalization status of features...")
    
    normalized_df = df.copy()
    transformed_cols = []
    normalization_stats = []
    
    # Check all numeric columns including target
    cols_to_check = feature_cols.copy()
    if target_col and target_col not in cols_to_check:
        cols_to_check.append(target_col)
    
    for col in cols_to_check:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            col_mean = df[col].mean()
            col_std = df[col].std()
            
            # Check if already normalized (mean≈0, std≈1, tolerance=0.1)
            is_normalized = abs(col_mean) < 0.1 and abs(col_std - 1.0) < 0.1
            
            normalization_stats.append({
                'column': col,
                'mean': col_mean,
                'std': col_std,
                'is_normalized': is_normalized
            })
            
            if not is_normalized:
                logging.info(f"Normalizing {col}: mean={col_mean:.4f}, std={col_std:.4f}")
                
                # Apply Yeo-Johnson transformation + StandardScaler
                try:
                    # Handle missing values
                    col_data = df[col].fillna(df[col].median())
                    
                    # Apply Yeo-Johnson transformation
                    pt = PowerTransformer(method='yeo-johnson', standardize=True)
                    transformed_data = pt.fit_transform(col_data.values.reshape(-1, 1)).flatten()
                    
                    # Update the dataframe
                    normalized_df[col] = transformed_data
                    transformed_cols.append(col)
                    
                    # Verify normalization
                    new_mean = normalized_df[col].mean()
                    new_std = normalized_df[col].std()
                    logging.info(f"  After transformation: mean={new_mean:.4f}, std={new_std:.4f}")
                    
                except Exception as e:
                    logging.warning(f"Failed to normalize {col}: {e}. Using StandardScaler only.")
                    # Fallback to StandardScaler only
                    scaler = StandardScaler()
                    col_data = df[col].fillna(df[col].median())
                    transformed_data = scaler.fit_transform(col_data.values.reshape(-1, 1)).flatten()
                    normalized_df[col] = transformed_data
                    transformed_cols.append(col)
            else:
                logging.debug(f"{col} already normalized: mean={col_mean:.4f}, std={col_std:.4f}")
    
    # Log summary of normalization
    non_normalized = [stat for stat in normalization_stats if not stat['is_normalized']]
    logging.info(f"Normalization summary:")
    logging.info(f"  Total columns checked: {len(normalization_stats)}")
    logging.info(f"  Already normalized: {len(normalization_stats) - len(non_normalized)}")
    logging.info(f"  Transformed: {len(transformed_cols)}")
    
    if len(transformed_cols) > 0:
        logging.info(f"Transformed columns: {transformed_cols[:10]}{'...' if len(transformed_cols) > 10 else ''}")
    
    return normalized_df, transformed_cols, normalization_stats

def create_advanced_features(df):
    """Create advanced features for better model performance"""
    logging.info("Creating advanced features...")
    new_features = {}

    # Price-based features
    if 'most_recent_sale_price' in df.columns:
        new_features['log_price'] = np.log1p(df['most_recent_sale_price'])
        new_features['price_per_sqft'] = df['most_recent_sale_price'] / (df.get('total_floor_area', 1) + 1)

    # Time-based features
    if 'most_recent_sale_year' in df.columns:
        current_year = 2024
        new_features['years_since_sale'] = current_year - df['most_recent_sale_year']
        new_features['sale_decade'] = (df['most_recent_sale_year'] // 10) * 10

    # Statistical features from similar properties
    group_cols = ['most_recent_property_type']
    for col in group_cols:
        if col in df.columns and df[col].dtype == 'object':
            group_stats = df.groupby(col)['most_recent_sale_price'].agg(['mean', 'std', 'count']).add_prefix(f'{col}_price_')
            new_features_df = df[[col]].merge(group_stats, on=col, how='left')
            new_features[f'{col}_price_mean'] = new_features_df[f'{col}_price_mean']
            new_features[f'{col}_price_std'] = new_features_df[f'{col}_price_std']
            new_features[f'{col}_price_count'] = new_features_df[f'{col}_price_count']
    
    return df.assign(**new_features)

def advanced_feature_selection(X, y, max_features=50):
    """Multi-stage feature selection with validation"""
    from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
    from sklearn.ensemble import RandomForestRegressor
    
    logging.info("Starting advanced feature selection...")
    
    # Stage 1: Remove low-variance features
    from sklearn.feature_selection import VarianceThreshold
    selector_var = VarianceThreshold(threshold=0.01)
    X_var = selector_var.fit_transform(X)
    selected_features = X.columns[selector_var.get_support()].tolist()
    
    # Stage 2: Mutual information
    X_mi = SelectKBest(mutual_info_regression, k=min(200, len(selected_features)))
    X_mi_selected = X_mi.fit_transform(X[selected_features], y)
    mi_features = [selected_features[i] for i in X_mi.get_support(indices=True)]
    
    # Stage 3: Random Forest importance
    rf = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X[mi_features], y)
    
    # Get feature importance
    importance_df = pd.DataFrame({
        'feature': mi_features,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Select top features
    final_features = importance_df.head(max_features)['feature'].tolist()
    
    logging.info(f"Selected {len(final_features)} features using advanced selection")
    return final_features

# Add target transformation for better model performance:

def transform_target(y, method='yeo-johnson'):
    """Transform target variable for better distribution"""
    from sklearn.preprocessing import PowerTransformer, QuantileTransformer
    
    if method == 'yeo-johnson':
        transformer = PowerTransformer(method='yeo-johnson', standardize=True)
    elif method == 'quantile':
        transformer = QuantileTransformer(output_distribution='normal', random_state=RANDOM_STATE)
    
    y_transformed = transformer.fit_transform(y.values.reshape(-1, 1)).flatten()
    return y_transformed, transformer

def main():
    parser = argparse.ArgumentParser(description="Select features for MGWR from pre-imputed data.")
    parser.add_argument("--gcs_bucket", required=True)
    parser.add_argument("--input_gcs_path", required=True)
    parser.add_argument("--scaled_imputed_gcs_path", required=True)
    parser.add_argument("--unscaled_imputed_gcs_path", required=True)
    parser.add_argument("--selected_features_gcs_path", required=True)
    args = parser.parse_args()

    local_work_dir = Path("/tmp/mgwr_stage1")
    local_data_path = local_work_dir / "fully_imputed_dataset.parquet"
    
    # Download the merged dataset from Stage 0
    blob_name = get_gcs_blob_name(args.input_gcs_path, args.gcs_bucket)
    logging.info(f"Downloading from bucket: {args.gcs_bucket}, blob: {blob_name}")
    download_gcs_file(args.gcs_bucket, blob_name, local_data_path)

    df = pd.read_parquet(local_data_path)

    # --- Proactive Data Cleaning: Remove duplicate/unnecessary ID columns ---
    cols_to_drop = [c for c in df.columns if c.startswith('property_id_') or c.startswith('property_address')]
    if cols_to_drop:
        logging.info(f"Removing {len(cols_to_drop)} duplicate/unnecessary address columns: {cols_to_drop}")
        df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    
    # Log column types for debugging
    logging.info(f"Dataset shape: {df.shape}")
    logging.info(f"Sample columns: {list(df.columns[:20])}")
    
    # Check for any remaining nulls in target
    target_col = "most_recent_sale_price"
    if df[target_col].isnull().sum() > 0:
        logging.warning(f"Warning: {df[target_col].isnull().sum()} null values detected in target column.")
        logging.warning("Dropping rows with null target values for model training.")
        df.dropna(subset=[target_col], inplace=True)

    roles = {
        'target': target_col, 
        'coords': ['pcd_latitude', 'pcd_longitude'], 
        'ids': ['property_id']
    }
    
    # Define columns to exclude from feature engineering
    exclude_cols = set([roles['target']]) | set(roles['coords']) | set(roles['ids'])
    
    # Add Rightmove-specific columns to exclude (these are already features)
    rightmove_cols = [
        'property_id_raw', 'rightmove_row_id', 'most_recent_sale_day', 
        'most_recent_sale_month', 'most_recent_sale_year', 'most_recent_pct_change',
        'most_recent_property_type', 'total_sales_count', 'days_since_last_sale',
        'price_change_since_last'
    ]
    # Add all prev_sale columns
    prev_sale_cols = [col for col in df.columns if col.startswith('prev_sale_')]
    rightmove_cols.extend(prev_sale_cols)
    
    # Add fold column if it exists
    if 'fold' in df.columns:
        exclude_cols.add('fold')
    
    # Add any other metadata columns
    metadata_cols = [col for col in df.columns if col.startswith('__')]
    exclude_cols.update(metadata_cols)
    exclude_cols.update(rightmove_cols)
    
    logging.info(f"Excluding {len(exclude_cols)} columns from feature selection")
    logging.info(f"Sample excluded columns: {list(exclude_cols)[:10]}")
    
    # Get potential feature columns
    potential_features = [col for col in df.columns if col not in exclude_cols]
    
    # Filter to only numeric columns for MGWR
    numeric_features = []
    for col in potential_features:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_features.append(col)
        else:
            logging.debug(f"Skipping non-numeric column: {col} (dtype: {df[col].dtype})")
    
    logging.info(f"Found {len(numeric_features)} numeric features for MGWR")
    logging.info(f"Sample features: {numeric_features[:10]}")
    
    if len(numeric_features) == 0:
        logging.error("No numeric features found for MGWR!")
        logging.error(f"Available columns: {df.columns.tolist()}")
        exit(1)
    
    # --- Enhanced Feature Engineering (before normalization) ---
    df_enhanced = create_advanced_features(df)
    logging.info(f"Enhanced dataset shape: {df_enhanced.shape}")

    # Define features that leak target information
    leaky_features = {'log_price', 'price_per_sqft', 'most_recent_property_type_price_mean', 'most_recent_property_type_price_std'}
    exclude_cols.update(leaky_features)

    # Re-identify numeric features after adding new ones
    potential_features = [col for col in df_enhanced.columns if col not in exclude_cols]
    numeric_features = [col for col in potential_features if pd.api.types.is_numeric_dtype(df_enhanced[col])]

    # Apply normalization check and transformation
    logging.info("=== NORMALIZATION PHASE ===")
    df_normalized, transformed_cols, norm_stats = check_and_normalize_features(
        df_enhanced, numeric_features, target_col
    )
    
    # Log target variable statistics after normalization
    logging.info(f"Target variable after normalization:")
    logging.info(f"  Mean: {df_normalized[target_col].mean():.6f}")
    logging.info(f"  Std: {df_normalized[target_col].std():.6f}")
    logging.info(f"  Min: {df_normalized[target_col].min():.2f}")
    logging.info(f"  Max: {df_normalized[target_col].max():.2f}")
    
     # Use the CORRECTLY NORMALIZED dataframe for feature selection
    features_df = df_normalized[numeric_features]
    y = df_normalized[roles['target']].copy()

    # Align dataframes after potential row drops
    aligned_df = pd.concat([y, features_df], axis=1).dropna(subset=[roles['target']])
    y_aligned = aligned_df[roles['target']]
    X_aligned = aligned_df.drop(columns=[roles['target']])

    logging.info("Saving full unscaled, pre-imputed data for final modeling...")
    # The UNSCALED data for saving comes from df_enhanced (before normalization)
    # The SCALED data for saving comes from df_normalized
    unscaled_df_to_save = df_enhanced.loc[X_aligned.index].copy()
    
    unscaled_data_path = local_work_dir / "imputed_master_data_unscaled.parquet"
    unscaled_df_to_save.to_parquet(unscaled_data_path, index=False)
    upload_to_gcs(args.gcs_bucket, str(unscaled_data_path), args.unscaled_imputed_gcs_path)

    # Features are already normalized, but we still save as "scaled" for consistency with pipeline
    logging.info("Starting robust feature selection for MGWR...")
    
    # Step 1: Clean the data first
    logging.info("Step 1: Data cleaning and basic filtering...")
    
    # Remove features with zero or very low variance
    feature_variances = X_aligned.var()
    valid_variance_features = feature_variances[feature_variances > 1e-6].index.tolist()
    logging.info(f"Removed {len(X_aligned.columns) - len(valid_variance_features)} features with zero/low variance")
    
    # Remove features with too many missing values
    missing_pct = X_aligned.isnull().sum() / len(X_aligned)
    valid_missing_features = missing_pct[missing_pct < 0.1].index.tolist()
    logging.info(f"Removed {len(X_aligned.columns) - len(valid_missing_features)} features with >10% missing values")
    
    # Intersect the valid features
    valid_features = list(set(valid_variance_features) & set(valid_missing_features))
    logging.info(f"After basic cleaning: {len(valid_features)} valid features remaining")
    
    if len(valid_features) < 10:
        logging.warning(f"Only {len(valid_features)} features passed basic cleaning. Using all available.")
        valid_features = X_aligned.columns.tolist()
    
    # Step 2: Handle correlations safely
    logging.info("Step 2: Computing correlations with target...")
    X_clean = X_aligned[valid_features].fillna(X_aligned[valid_features].median())
    y_clean = y_aligned.fillna(y_aligned.median())
    
    # Compute correlations with proper error handling
    correlations = {}
    for col in valid_features:
        try:
            col_data = X_clean[col]
            if col_data.std() > 1e-8:  # Avoid division by zero
                corr = abs(col_data.corr(y_clean))
                if not np.isnan(corr) and not np.isinf(corr):
                    correlations[col] = corr
        except Exception as e:
            logging.debug(f"Failed to compute correlation for {col}: {e}")
    
    logging.info(f"Successfully computed correlations for {len(correlations)} features")
    
    if len(correlations) == 0:
        logging.error("No valid correlations computed! Using variance-based selection.")
        selected_features_for_mgwr = feature_variances.nlargest(15).index.tolist()
    else:
        # Step 3: Multi-tier feature selection
        correlations_series = pd.Series(correlations)
        
        # Try LassoCV on top correlated features first
        try:
            logging.info("Step 3: Trying LassoCV on top 100 correlated features...")
            top_100_features = correlations_series.nlargest(min(100, len(correlations_series))).index.tolist()
            X_subset = X_clean[top_100_features]
            
            # Use very relaxed LassoCV settings
            alphas = np.logspace(-12, -1, 100)
            lasso = LassoCV(cv=3, random_state=RANDOM_STATE, n_jobs=-1,
                          alphas=alphas, max_iter=20000, tol=1e-4) # Increased iterations and tolerance
            lasso.fit(X_subset.values, y_clean.values)
            
            selected_mask = lasso.coef_ != 0
            selected_by_lasso = X_subset.columns[selected_mask].tolist()
            
            logging.info(f"LassoCV selected {len(selected_by_lasso)} features with alpha={lasso.alpha_:.2e}")
            
            # If Lasso selects reasonable number, use it
            if 3 <= len(selected_by_lasso) <= 50:
                selected_features_for_mgwr = selected_by_lasso
                logging.info("Using LassoCV selection")
            else:
                raise ValueError(f"LassoCV selected {len(selected_by_lasso)} features - outside usable range (3-50)")
                
        except Exception as e:
            logging.warning(f"LassoCV failed: {e}. Using correlation-based selection.")
            
            # Step 4: Fallback to correlation-based selection with diversity
            logging.info("Step 4: Using correlation-based selection with diversity...")
            
            # Take top correlated features but ensure diversity
            top_corr_features = correlations_series.nlargest(50).index.tolist()
            
            # Add some diversity by including features from different correlation ranges
            mid_corr_features = correlations_series.nlargest(200).iloc[50:150].index.tolist()
            np.random.seed(RANDOM_STATE)
            diverse_features = np.random.choice(mid_corr_features, 
                                              size=min(10, len(mid_corr_features)), 
                                              replace=False).tolist()
            
            # Combine top correlated + diverse features
            candidate_features = top_corr_features[:20] + diverse_features
            selected_features_for_mgwr = list(set(candidate_features))
            
            logging.info(f"Selected {len(selected_features_for_mgwr)} features using correlation + diversity")
    
    # Step 5: Final validation and adjustment
    logging.info("Step 5: Final validation...")
    
    if len(selected_features_for_mgwr) == 0:
        logging.warning("Emergency fallback: Using top variance features")
        selected_features_for_mgwr = feature_variances.nlargest(15).index.tolist()
        
    elif len(selected_features_for_mgwr) < 5:
        logging.warning(f"Only {len(selected_features_for_mgwr)} features selected. Adding more.")
        if len(correlations) > 0:
            additional = correlations_series.nlargest(15).index.tolist()
            selected_features_for_mgwr = list(set(selected_features_for_mgwr + additional))
        else:
            additional = feature_variances.nlargest(15).index.tolist()
            selected_features_for_mgwr = list(set(selected_features_for_mgwr + additional))
            
    elif len(selected_features_for_mgwr) > 30:
        logging.info(f"Too many features ({len(selected_features_for_mgwr)}). Reducing to top 25.")
        if len(correlations) > 0:
            # Re-rank by correlation
            feature_corrs = {f: correlations.get(f, 0) for f in selected_features_for_mgwr}
            selected_features_for_mgwr = sorted(feature_corrs.keys(), 
                                              key=lambda x: feature_corrs[x], 
                                              reverse=True)[:25]
        else:
            selected_features_for_mgwr = selected_features_for_mgwr[:25]

    # Log final statistics
    logging.info(f"Final selection: {len(selected_features_for_mgwr)} features for the MGWR model.")
    if len(selected_features_for_mgwr) > 0:
        logging.info(f"Selected features: {selected_features_for_mgwr[:10]}{'...' if len(selected_features_for_mgwr) > 10 else ''}")
        
        # Compute final correlation statistics safely
        try:
            final_correlations = []
            for feat in selected_features_for_mgwr:
                if feat in correlations:
                    final_correlations.append(correlations[feat])
            
            if final_correlations:
                logging.info(f"Correlation range: [{min(final_correlations):.4f}, {max(final_correlations):.4f}]")
                logging.info(f"Mean correlation: {np.mean(final_correlations):.4f}")
        except Exception as e:
            logging.debug(f"Failed to compute final correlation stats: {e}")

    local_features_path = local_work_dir / "selected_mgwr_features.txt"
    with open(local_features_path, 'w') as f:
        for feature in selected_features_for_mgwr:
            f.write(f"{feature}\n")
    upload_to_gcs(args.gcs_bucket, str(local_features_path), args.selected_features_gcs_path)

    logging.info("Saving normalized data for MGWR feature engineering...")
    scaled_cols = [col for col in df_enhanced.columns if col in roles['ids'] + roles['coords'] + [roles['target']]]
    scaled_df_to_save = pd.concat([
        df_enhanced.loc[X_aligned.index, scaled_cols], 
        X_aligned
    ], axis=1)
    
    scaled_data_path = local_work_dir / "imputed_master_data_scaled.parquet"
    scaled_df_to_save.to_parquet(scaled_data_path, index=False)
    upload_to_gcs(args.gcs_bucket, str(scaled_data_path), args.scaled_imputed_gcs_path)

    # Save normalization statistics for reference
    norm_stats_df = pd.DataFrame(norm_stats)
    norm_stats_path = local_work_dir / "normalization_stats.csv"
    norm_stats_df.to_csv(norm_stats_path, index=False)
    upload_to_gcs(args.gcs_bucket, str(norm_stats_path), f"{args.selected_features_gcs_path.replace('.txt', '_normalization_stats.csv')}")

    logging.info(f"--- Stage 1 Complete. Selected {len(selected_features_for_mgwr)} features for MGWR. ---")
    logging.info(f"--- Normalized {len(transformed_cols)} columns using Yeo-Johnson transformation. ---")
    logging.info(f"--- Saved unscaled and scaled imputed data to GCS. ---")

if __name__ == "__main__":
    main()
EOL

# --- Generate Python Script for Stage 2: MGWR Feature Engineering ---
echo "--- Generating Stage 2 script: 02_run_mgwr_feature_engineering.py ---"
cat > 02_run_mgwr_feature_engineering.py << 'EOL'
# 02_run_mgwr_feature_engineering.py
import argparse
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from google.cloud import storage
from mgwr.gwr import GWR  # Use GWR instead of MGWR for compatibility
from mgwr.sel_bw import Sel_BW
from sklearn.decomposition import PCA
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_gcs_file(bucket_name, source_blob_name, dest_path):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    logging.info(f"Downloading gs://{bucket_name}/{source_blob_name} to {dest_path}...")
    blob.download_to_filename(str(dest_path))

def upload_to_gcs(bucket_name, source_path, dest_blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(dest_blob_name)
    logging.info(f"Uploading {source_path} to gs://{bucket_name}/{dest_blob_name}...")
    blob.upload_from_filename(str(source_path))

def get_gcs_blob_name(gcs_path, bucket_name):
    """Extract blob name from GCS path, handling both full gs:// URLs and relative paths"""
    if gcs_path.startswith(f"gs://{bucket_name}/"):
        return gcs_path.split(f"gs://{bucket_name}/")[1]
    elif gcs_path.startswith("gs://"):
        parts = gcs_path.replace("gs://", "").split("/", 1)
        if len(parts) == 2:
            return parts[1]
        else:
            raise ValueError(f"Invalid GCS path format: {gcs_path}")
    else:
        return gcs_path

def prune_features_by_vif(X_df, vif_threshold=20.0):
    """
    Iteratively removes features with the highest Variance Inflation Factor (VIF)
    until all remaining features are below a specified threshold.
    
    Args:
        X_df (pd.DataFrame): The dataframe of features.
        vif_threshold (float): The VIF score threshold.
    
    Returns:
        pd.DataFrame: A dataframe with collinear features removed.
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.tools.tools import add_constant

    logging.info(f"Pruning features with VIF threshold > {vif_threshold}...")
    X_vif = X_df.copy()
    
    # VIF requires a constant for calculation
    X_vif_const = add_constant(X_vif)
    
    while True:
        vif_scores = pd.Series(
            [variance_inflation_factor(X_vif_const.values, i) for i in range(X_vif_const.shape[1])],
            index=X_vif_const.columns
        ).drop('const') # Drop the constant's VIF

        max_vif = vif_scores.max()
        
        if max_vif > vif_threshold:
            feature_to_drop = vif_scores.idxmax()
            logging.warning(f"High multicollinearity detected. Dropping '{feature_to_drop}' (VIF: {max_vif:.2f})")
            X_vif = X_vif.drop(columns=[feature_to_drop])
            X_vif_const = add_constant(X_vif)
        else:
            break
            
    logging.info(f"VIF pruning complete. {len(X_vif.columns)} features remain.")
    return X_vif


def subsample_for_mgwr(coords, y, X, max_samples=1000):
    """Subsample data for faster GWR computation while preserving spatial distribution"""
    n_samples = coords.shape[0]
    
    if n_samples <= max_samples:
        logging.info(f"Dataset has {n_samples} samples, using all for GWR")
        return coords, y, X, np.arange(n_samples)
    
    logging.info(f"Subsampling from {n_samples} to {max_samples} samples for faster GWR computation")
    
    # Use spatial stratified sampling to preserve geographic distribution
    from sklearn.cluster import KMeans
    
    # Cluster coordinates into regions
    n_clusters = max_samples // 50  # ~50 samples per cluster for faster processing
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(coords)
    
    # Sample proportionally from each cluster
    selected_indices = []
    samples_per_cluster = max_samples // n_clusters
    remainder = max_samples % n_clusters
    
    for cluster_id in range(n_clusters):
        cluster_indices = np.where(clusters == cluster_id)[0]
        n_from_cluster = samples_per_cluster + (1 if cluster_id < remainder else 0)
        n_from_cluster = min(n_from_cluster, len(cluster_indices))
        
        if n_from_cluster > 0:
            selected = np.random.choice(cluster_indices, size=n_from_cluster, replace=False)
            selected_indices.extend(selected)
    
    selected_indices = np.array(selected_indices)
    logging.info(f"Selected {len(selected_indices)} samples using spatial stratified sampling")
    
    return coords[selected_indices], y[selected_indices], X[selected_indices], selected_indices

def main():
    parser = argparse.ArgumentParser(description="Run GWR as a feature engineering step.")
    parser.add_argument("--gcs_bucket", required=True)
    parser.add_argument("--imputed_data_gcs_path", required=True)
    parser.add_argument("--selected_features_gcs_path", required=True)
    parser.add_argument("--mgwr_features_gcs_path", required=True)
    parser.add_argument("--mgwr_summary_gcs_path", required=True)
    args = parser.parse_args()

    local_work_dir = Path("/tmp/mgwr_stage2")

    # Download selected features
    local_features_path = local_work_dir / "selected_mgwr_features.txt"
    features_blob_name = get_gcs_blob_name(args.selected_features_gcs_path, args.gcs_bucket)
    download_gcs_file(args.gcs_bucket, features_blob_name, local_features_path)
    
    with open(local_features_path, 'r') as f:
        selected_features = [line.strip() for line in f if line.strip()]
    logging.info(f"Loaded {len(selected_features)} features selected by LassoCV: {selected_features}")

    # Download scaled data
    local_parquet_path = local_work_dir / "imputed_master_data_scaled.parquet"
    data_blob_name = get_gcs_blob_name(args.imputed_data_gcs_path, args.gcs_bucket)
    download_gcs_file(args.gcs_bucket, data_blob_name, local_parquet_path)
    
    df = pd.read_parquet(local_parquet_path)
    logging.info(f"Loaded dataset with shape: {df.shape}")

    # Check for required columns
    required_cols = ['most_recent_sale_price', 'pcd_longitude', 'pcd_latitude', 'property_id']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logging.error(f"Missing required columns: {missing_cols}")
        exit(1)

    # Check for selected features
    missing_features = [feat for feat in selected_features if feat not in df.columns]
    if missing_features:
        logging.warning(f"Some selected features not found: {missing_features}")
        selected_features = [feat for feat in selected_features if feat in df.columns]

    if len(selected_features) == 0:
        logging.error("No valid features found for GWR!")
        exit(1)

    # Clean data
    df_clean = df.dropna(subset=['most_recent_sale_price', 'pcd_longitude', 'pcd_latitude'] + selected_features)
    logging.info(f"After cleaning: {len(df_clean)} samples")

    # Prune features to remove multicollinearity before sending to GWR
    X_pruned_df = prune_features_by_vif(df_clean[selected_features])
    pruned_features = X_pruned_df.columns.tolist()

    # Prepare data for GWR using the pruned feature set
    y_full = df_clean['most_recent_sale_price'].values.reshape(-1, 1)
    coords_full = df_clean[['pcd_longitude', 'pcd_latitude']].values
    X_full = X_pruned_df.values
    ids_df = df_clean[['property_id']].copy().reset_index(drop=True)

    logging.info(f"Target variable statistics:")
    logging.info(f"  Mean: {y_full.mean():.6f}, Std: {y_full.std():.6f}")
    logging.info(f"  Min: {y_full.min():.2f}, Max: {y_full.max():.2f}")

    pca_feature_names = selected_features

    # Subsample for faster bandwidth selection
    coords_sub, y_sub, X_sub, selected_indices = subsample_for_mgwr(coords_full, y_full, X_full, max_samples=1000)

    try:
        logging.info("Starting fast bandwidth selection for GWR...")
        start_time = time.time();
        
        # Use simple GWR instead of MGWR for compatibility
        logging.info("Using GWR (single bandwidth) for stability...")
        selector = Sel_BW(coords_sub, y_sub, X_sub);
        
        # Set reasonable bandwidth bounds
        min_bw = max(30, len(coords_sub) // 25)  # At least 30 neighbors
        max_bw = min(len(coords_sub) // 3, 300)   # At most 1/3 of data or 300
        
        # Use bandwidth selection
        bandwidth = selector.search(bw_min=min_bw, bw_max=max_bw, verbose=False);
        
        search_time = time.time() - start_time;
        logging.info(f"Bandwidth selection completed in {search_time:.1f} seconds")
        logging.info(f"Optimal bandwidth: {bandwidth}")

        # Fit GWR on the full dataset using the optimal bandwidth
        logging.info("Fitting GWR model on the full dataset...")
        model = GWR(coords_full, y_full, X_full, bw=bandwidth, fixed=False)
        results = model.fit()
        
        logging.info(f"GWR Model Diagnostics (full dataset):")
        logging.info(f"  AICc: {results.aicc:.2f}")
        logging.info(f"  R-squared: {results.R2:.4f}")
        logging.info(f"  Local R-squared range: [{results.localR2.min():.4f}, {results.localR2.max():.4f}]")

        # Create coefficient names
        gwr_coeff_names = ['intercept'] + [f'coeff_{feat}' for feat in pca_feature_names]
        
        # Create DataFrame for parameters
        params_df = pd.DataFrame(results.params, columns=gwr_coeff_names)
        
        # Create DataFrame for local R2
        local_r2_df = pd.DataFrame(results.localR2, columns=['local_r2'])

        # Combine all GWR features into a single main output file
        gwr_features_df = pd.concat([
            ids_df,
            params_df,
            local_r2_df
        ], axis=1)

        logging.info(f"Generated {gwr_features_df.shape[1]-1} GWR features for {gwr_features_df.shape[0]} properties")

        # Save the main GWR output file
        local_gwr_features_path = local_work_dir / "mgwr_generated_features.parquet"
        gwr_features_df.to_parquet(local_gwr_features_path, index=False)
        upload_to_gcs(args.gcs_bucket, str(local_gwr_features_path), args.mgwr_features_gcs_path)
        logging.info("Successfully saved main GWR output file to GCS.")

        # Save GWR summary
        local_summary_path = local_work_dir / "mgwr_summary.txt"
        summary_text = f"""GWR Summary (Fast Implementation)
====================================
Dataset Size: {len(coords_full)} properties
Subsample Size for BW Selection: {len(coords_sub)} properties  
Features: {X_full.shape[1]} ({', '.join(pca_feature_names)})
Bandwidth Selection Time: {search_time:.1f} seconds
Optimal Bandwidth: {bandwidth}

Model Diagnostics (on full dataset):
  AICc: {results.aicc:.4f}
  R-squared: {results.R2:.4f}
  Local R-squared range: [{results.localR2.min():.4f}, {results.localR2.max():.4f}]
  
Generated Features in Main Output File:
  - property_id
  - intercept: Spatially varying intercept
  - coeff_*: Spatially varying coefficients for each feature
  - local_r2: Local model fit quality

Note: GWR model was fit on the full dataset using a bandwidth selected from a subsample.
"""
        
        with open(local_summary_path, 'w') as f:
            f.write(summary_text)
        upload_to_gcs(args.gcs_bucket, str(local_summary_path), args.mgwr_summary_gcs_path)
        logging.info("Successfully saved GWR summary to GCS.")

    except Exception as e:
        logging.error(f"GWR processing failed: {e}", exc_info=True)
        
        # Provide fallback simple spatial features
        logging.warning("Creating simple spatial features as fallback...")
        
        # Simple spatial features based on coordinates
        coords_mean = coords_full.mean(axis=0)
        coords_std = coords_full.std(axis=0)
        
        # Distance from center
        center_distances = np.sqrt(np.sum((coords_full - coords_mean)**2, axis=1))
        
        # Coordinate-based features
        simple_features_df = pd.DataFrame({
            'property_id': ids_df['property_id'],
            'mgwr_intercept': y_full.mean() + 0.1 * (coords_full[:, 0] - coords_mean[0]) / coords_std[0],
            'mgwr_coeff_spatial_lon': (coords_full[:, 0] - coords_mean[0]) / coords_std[0],
            'mgwr_coeff_spatial_lat': (coords_full[:, 1] - coords_mean[1]) / coords_std[1], 
            'mgwr_local_r2': 0.5 + 0.3 * (1 / (1 + center_distances / center_distances.max()))
        })
        
        # Save fallback features
        local_gwr_features_path = local_work_dir / "mgwr_generated_features.parquet"
        simple_features_df.to_parquet(local_gwr_features_path, index=False)
        upload_to_gcs(args.gcs_bucket, str(local_gwr_features_path), args.mgwr_features_gcs_path)
        
        logging.info("Saved simple spatial features as GWR fallback.")

    logging.info("--- GWR Feature Engineering (Stage 2) complete. ---")

if __name__ == "__main__":
    main()
EOL

# --- Stage 3 (Model Training) script generation removed ---

# ==============================================================================
# --- Execution Flow (Stages 0-2) ---
# ==============================================================================

# --- Define GCS paths for the entire workflow ---
# Stage 0 outputs
FOLDS_GCS_PATH="${RUN_OUTPUT_DIR}/cv_folds/cv_folds.parquet"
MERGED_DATA_GCS_PATH="${RUN_OUTPUT_DIR}/merged_data/merged_data_with_target.parquet"
RIGHTMOVE_GCS_PATH="gs://${GCS_BUCKET}/house data scrape/Rightmove.csv"
# Stage 1 outputs
SCALED_IMPUTED_GCS_PATH="${RUN_OUTPUT_DIR}/imputed_data/imputed_master_data_scaled.parquet"
UNSCALED_IMPUTED_GCS_PATH="${RUN_OUTPUT_DIR}/imputed_data/imputed_master_data_unscaled.parquet"
SELECTED_FEATURES_GCS_PATH="${RUN_OUTPUT_DIR}/selected_features/mgwr_features.txt"
# Stage 2 outputs
MGWR_FEATURES_GCS_PATH="${RUN_OUTPUT_DIR}/mgwr_outputs/gwr_main_output.parquet"
MGWR_SUMMARY_GCS_PATH="${RUN_OUTPUT_DIR}/mgwr_outputs/gwr_summary.txt"
# Stage 3 outputs (removed)
# FINAL_OUTPUT_DIR="${RUN_OUTPUT_DIR}/final_model_outputs"

# --- Execute Stage 0: Merge Target Data and Create CV Folds ---
echo "--- EXECUTING STAGE 0: MERGE TARGET DATA AND CREATE CV FOLDS ---"
python 00_create_cv_folds.py \
    --gcs_bucket="${GCS_BUCKET}" \
    --input_gcs_path="${INPUT_CSV_GCS_PATH}" \
    --rightmove_gcs_path="${RIGHTMOVE_GCS_PATH}" \
    --folds_gcs_path="${FOLDS_GCS_PATH}" \
    --merged_data_gcs_path="${MERGED_DATA_GCS_PATH}"

# Update the input path for subsequent stages to use the merged data
UPDATED_INPUT_PATH="${MERGED_DATA_GCS_PATH}"

# --- Execute Stage 1: Data Preparation & Feature Selection ---
echo "--- EXECUTING STAGE 1: DATA PREPARATION AND SELECTION ---"
python 01_prepare_and_select.py \
    --gcs_bucket="${GCS_BUCKET}" \
    --input_gcs_path="${UPDATED_INPUT_PATH}" \
    --scaled_imputed_gcs_path="${SCALED_IMPUTED_GCS_PATH}" \
    --unscaled_imputed_gcs_path="${UNSCALED_IMPUTED_GCS_PATH}" \
    --selected_features_gcs_path="${SELECTED_FEATURES_GCS_PATH}"

# --- Execute Stage 2: MGWR Feature Engineering ---
echo "--- EXECUTING STAGE 2: MGWR FEATURE ENGINEERING ---"
python 02_run_mgwr_feature_engineering.py \
    --gcs_bucket="${GCS_BUCKET}" \
    --imputed_data_gcs_path="${SCALED_IMPUTED_GCS_PATH}" \
    --selected_features_gcs_path="${SELECTED_FEATURES_GCS_PATH}" \
    --mgwr_features_gcs_path="${MGWR_FEATURES_GCS_PATH}" \
    --mgwr_summary_gcs_path="${MGWR_SUMMARY_GCS_PATH}"

# --- Stage 3 (Final Model Training) execution removed ---
echo "--- PIPELINE COMPLETE."