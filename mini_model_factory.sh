#!/bin/bash
#
# run_comprehensive_model_factory.sh
#
# This script executes a state-of-the-art, multi-layered modeling strategy.
#

# -- Strict Mode & Configuration --
set -e
set -o pipefail
set -x

# -- GCP & Project Configuration --
GCS_BUCKET="srgan-bucket-ace-botany-453819-t4"
TIMESTAMP=$(date +"%Y%m%d-%H%M%S")
OUTPUT_GCS_BASE_DIR="gs://${GCS_BUCKET}/comprehensive_model_factory/run_${TIMESTAMP}"
MASTER_DATA_GCS_PATH="gs://${GCS_BUCKET}/imputation_pipeline/output_lgbm_20250718-200834/final_fully_imputed_dataset.parquet"
RIGHTMOVE_GCS_PATH="gs://${GCS_BUCKET}/house data scrape/Rightmove.csv"

# -- Local Configuration --
PROJECT_DIR="c:/Users/dell/Desktop/House Data Scrape/wisteria_comprehensive_models"
VENV_DIR="${PROJECT_DIR}/venv_cm"
OUTPUT_DIR="${PROJECT_DIR}/output"
DATA_DIR="${PROJECT_DIR}/data"
MASTER_DATA_LOCAL_PATH="${DATA_DIR}/master_dataset.parquet"
RIGHTMOVE_DATA_LOCAL_PATH="${DATA_DIR}/Rightmove.csv"

# --- NEW: Point to your new, standalone Python script ---
SCRIPT_PATH="${PROJECT_DIR}/train_model_suite_v2.py"

# --- Create Local Directory Structure ---
mkdir -p "${OUTPUT_DIR}" "${DATA_DIR}"
cd "${PROJECT_DIR}"

# --- Environment Setup ---
if [ ! -d "${VENV_DIR}" ]; then
    echo "Creating Python virtual environment..."
    # Try different Python commands that might work on Windows
    if command -v python3 &> /dev/null; then
        python3 -m venv "${VENV_DIR}"
    elif command -v py &> /dev/null; then
        py -m venv "${VENV_DIR}"
    elif command -v python &> /dev/null; then
        python -m venv "${VENV_DIR}"
    else
        echo "ERROR: No Python interpreter found. Please install Python."
        exit 1
    fi
fi

# Activate virtual environment (Windows style)
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
    source "${VENV_DIR}/Scripts/activate"
else
    source "${VENV_DIR}/bin/activate"
fi

# Install dependencies - use python3 or py if available
PYTHON_CMD=""
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v py &> /dev/null; then
    PYTHON_CMD="py"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "ERROR: No Python interpreter found after venv creation."
    exit 1
fi

$PYTHON_CMD -m pip install --upgrade pip
cat > requirements.txt <<EOF
pandas
pyarrow
gcsfs
google-cloud-storage
scikit-learn
lightgbm
xgboost
catboost
torch
pytorch-tabnet
joblib
EOF
$PYTHON_CMD -m pip install -r requirements.txt

# --- Python Worker Script Generation ---
# This is a large, self-contained script that performs all modeling tasks.
cat > "${SCRIPT_PATH}" <<'EOF'
# train_model_suite_v4.py
import os
import re
import gc
import joblib
import warnings
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error
from google.cloud import storage
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from pytorch_tabnet.tab_model import TabNetRegressor
from torch.utils.data import TensorDataset, DataLoader
import ast
from datetime import datetime

# --- Configuration ---
GCS_BUCKET_NAME = "srgan-bucket-ace-botany-453819-t4"
LOCAL_PROJECT_DIR = "c:/Users/dell/Desktop/House Data Scrape/wisteria_comprehensive_models"
LOCAL_DATA_DIR = os.path.join(LOCAL_PROJECT_DIR, "data")
LOCAL_OUTPUT_DIR = os.path.join(LOCAL_PROJECT_DIR, "output")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

SUBSET_GCS_PATHS = {
    "subset1_features": f"gs://{GCS_BUCKET_NAME}/house data scrape/subset1_processed_full.parquet",
    "subset2_features": f"gs://{GCS_BUCKET_NAME}/house data scrape/subset2_processed_full.parquet",
    "subset3_features": f"gs://{GCS_BUCKET_NAME}/house data scrape/subset3_processed_full.parquet",
    "subset4_features": f"gs://{GCS_BUCKET_NAME}/house data scrape/subset4_processed_full.parquet",
    "subset5_features": f"gs://{GCS_BUCKET_NAME}/house data scrape/subset5_processed_full.parquet",
    "gwr_features": f"gs://{GCS_BUCKET_NAME}/house data scrape/cleaned_property_data_gwr_with_coords_label.csv",
    "gemini_quant_features": f"gs://{GCS_BUCKET_NAME}/house data scrape/property_features_quantitative_v4.csv",
}

# --- Rightmove Parsing Functions (Unchanged) ---
def parse_sales_history(sales_str):
    if pd.isna(sales_str) or sales_str == '' or sales_str == '[]': return pd.DataFrame()
    try:
        sales_list = ast.literal_eval(str(sales_str))
        if not isinstance(sales_list, list): return pd.DataFrame()
        sales_data = []
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
                    if price > 0: sales_data.append({'sale_date': sale_date, 'sale_price': price})
                except ValueError: continue
        return pd.DataFrame(sales_data)
    except (ValueError, SyntaxError): return pd.DataFrame()

def process_rightmove_data(raw_rightmove_df):
    print("Processing Rightmove sales data to extract features...")
    processed_data = []
    if 'sales_history' not in raw_rightmove_df.columns:
        if raw_rightmove_df.shape[1] > 2:
            raw_rightmove_df.rename(columns={raw_rightmove_df.columns[2]: 'sales_history'}, inplace=True)
        else: raise ValueError("Sales history column not found in Rightmove data.")
    for idx, row in raw_rightmove_df.iterrows():
        sales_df = parse_sales_history(row['sales_history'])
        if not sales_df.empty:
            sales_df = sales_df.sort_values('sale_date', ascending=False).reset_index(drop=True)
            most_recent = sales_df.iloc[0]
            features = {
                'rightmove_row_id': idx,
                'most_recent_sale_price': most_recent['sale_price'],
                'most_recent_sale_year': most_recent['sale_date'].year,
                'most_recent_sale_month': most_recent['sale_date'].month,
                'total_sales_count': len(sales_df)
            }
            if len(sales_df) > 1:
                previous_sale = sales_df.iloc[1]
                features['days_since_last_sale'] = (most_recent['sale_date'] - previous_sale['sale_date']).days
                features['price_change_since_last'] = most_recent['sale_price'] - previous_sale['sale_price']
            processed_data.append(features)
    print(f"Successfully processed {len(processed_data)} properties with valid sales data.")
    return pd.DataFrame(processed_data)

# --- Utility Functions (Unchanged) ---
def get_feature_names_from_gcs(gcs_uri, target_column='most_recent_sale_price'):
    print(f"Fetching feature names from: {gcs_uri}")
    try:
        temp_local_path = os.path.join(LOCAL_DATA_DIR, 'temp_col_reader.tmp')
        storage_client = storage.Client()
        bucket_name, blob_name = gcs_uri.replace("gs://", "").split("/", 1)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.download_to_filename(temp_local_path)
        if gcs_uri.endswith('.csv'): df = pd.read_csv(temp_local_path, nrows=1)
        elif gcs_uri.endswith('.parquet'): df = pd.read_parquet(temp_local_path); df = df.head(1)
        else: print(f"  - WARNING: Unsupported file type: {gcs_uri}"); os.remove(temp_local_path); return []
        exclude_cols = ['property_id', 'pcd_latitude', 'pcd_longitude', target_column]
        features = [col for col in df.columns if col.lower() not in exclude_cols and 'unnamed' not in col.lower()]
        print(f"  - Found {len(features)} potential features.")
        os.remove(temp_local_path)
        return features
    except Exception as e:
        print(f"  - ERROR: Could not read columns from {gcs_uri}. Error: {e}")
        if os.path.exists(temp_local_path): os.remove(temp_local_path)
        return []

# --- Model Definitions & Management Functions (Unchanged) ---
class DAE(nn.Module):
    def __init__(self, input_dim): super(DAE, self).__init__(); self.encoder = nn.Sequential(nn.Linear(input_dim, 256), nn.ReLU(), nn.Linear(256, 128)); self.decoder = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, input_dim))
    def forward(self, x): encoded = self.encoder(x); decoded = self.decoder(encoded); return decoded, encoded
def manage_dae_model(X_df, subset_name, mode='train'):
    print(f"--- Managing DAE for subset: {subset_name} (mode: {mode}) ---")
    model_dir = os.path.join(LOCAL_OUTPUT_DIR, "dae_models"); os.makedirs(model_dir, exist_ok=True); model_path = os.path.join(model_dir, f"dae_{subset_name}.pt")
    if X_df.empty or X_df.shape[1] == 0: print("  - Skipping DAE, no features provided."); return None
    X = X_df.values
    if mode == 'train':
        model = DAE(X.shape[1]).to(DEVICE); criterion = nn.MSELoss(); optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5); dataset = TensorDataset(torch.tensor(X, dtype=torch.float32)); loader = DataLoader(dataset, batch_size=512, shuffle=True)
        for epoch in range(50):
            for batch_X_list in loader:
                batch_X = batch_X_list[0].to(DEVICE); noisy_batch_X = batch_X + torch.randn_like(batch_X) * 0.1; outputs, _ = model(noisy_batch_X); loss = criterion(outputs, batch_X); optimizer.zero_grad(); loss.backward(); optimizer.step()
        torch.save(model.state_dict(), model_path); print(f"  - DAE for {subset_name} trained and saved."); return None
    elif mode == 'predict':
        if not os.path.exists(model_path): print(f"  - WARNING: DAE model not found for {subset_name}. Skipping."); return None
        model = DAE(X.shape[1]).to(DEVICE); model.load_state_dict(torch.load(model_path)); model.eval()
        with torch.no_grad(): reconstructed, embeddings = model(torch.tensor(X, dtype=torch.float32).to(DEVICE))
        embed_df = pd.DataFrame(embeddings.cpu().numpy(), index=X_df.index, columns=[f"dae_embed_{subset_name}_{i}" for i in range(embeddings.shape[1])]); return embed_df

def manage_tabnet_model(X_df, y_df, subset_name, mode='train'):
    print(f"--- Managing TabNet for subset: {subset_name} (mode: {mode}) ---")
    model_dir = os.path.join(LOCAL_OUTPUT_DIR, "tabnet_models"); os.makedirs(model_dir, exist_ok=True); model_path_prefix = os.path.join(model_dir, f"tabnet_model_{subset_name}")
    if X_df.empty or X_df.shape[1] == 0: print("  - Skipping TabNet, no features provided."); return None
    X, y = X_df.values, y_df.values.reshape(-1, 1)
    if mode == 'train':
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42); model = TabNetRegressor(device_name=DEVICE)
        # --- FIX: Removed the unsupported 'verbose' argument ---
        model.fit(
            X_train=X_train, y_train=y_train,
            eval_set=[(X_val, y_val)],
            max_epochs=50,
            patience=15,
            batch_size=1024
        )
        model.save_model(model_path_prefix); print(f"  - TabNet model for {subset_name} saved."); return None
    elif mode == 'predict':
        model_zip_path = model_path_prefix + ".zip"
        if not os.path.exists(model_zip_path): print(f"  - WARNING: TabNet model not found at {model_zip_path}. Skipping."); return None
        model = TabNetRegressor(); model.load_model(model_zip_path); predictions = model.predict(X); return pd.DataFrame(predictions, index=X_df.index, columns=[f"tabnet_pred_{subset_name}"])

def manage_generic_model(model_class, X_df, y_df, subset_name, model_name, mode='train', **kwargs):
    print(f"--- Managing {model_name} for subset: {subset_name} (mode: {mode}) ---")
    model_dir = os.path.join(LOCAL_OUTPUT_DIR, "general_models"); os.makedirs(model_dir, exist_ok=True); model_path = os.path.join(model_dir, f"{model_name}_{subset_name}.joblib")
    if X_df.empty or X_df.shape[1] == 0: print(f"  - Skipping {model_name}, no features provided."); return None
    X, y = X_df.values, y_df.values
    if mode == 'train':
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        if model_name == 'CatBoost': kwargs['verbose'] = 0
        model = model_class(**kwargs); model.fit(X_train, y_train)
        preds_log = model.predict(X_test); r2 = r2_score(y_test, preds_log); print(f"  - {model_name} R2 (log) on validation: {r2:.4f}"); joblib.dump(model, model_path); return None
    elif mode == 'predict':
        if not os.path.exists(model_path): print(f"  - WARNING: Model not found for {model_name} on {subset_name}. Skipping."); return None
        model = joblib.load(model_path); predictions = model.predict(X); return pd.DataFrame(predictions, index=X_df.index, columns=[f"{model_name.lower()}_pred_{subset_name}"])

def main():
    """Main execution function for the entire modeling pipeline."""
    warnings.filterwarnings('ignore')
    os.makedirs(LOCAL_OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOCAL_DATA_DIR, exist_ok=True)
    print(f"Using device: {DEVICE}")

    # --- Data Loading and Preparation ---
    print("\n=============== DATA LOADING & PREPARATION ===============")
    master_df = pd.read_parquet(os.path.join(LOCAL_DATA_DIR, "master_dataset.parquet"))
    raw_rightmove_df = pd.read_csv(os.path.join(LOCAL_DATA_DIR, "Rightmove.csv"))
    processed_rightmove_df = process_rightmove_data(raw_rightmove_df)
    min_len = min(len(master_df), len(processed_rightmove_df))
    full_df = pd.concat([
        master_df.iloc[:min_len].reset_index(drop=True),
        processed_rightmove_df.iloc[:min_len].reset_index(drop=True)
    ], axis=1)
    TARGET_COLUMN = 'most_recent_sale_price'
    full_df.dropna(subset=[TARGET_COLUMN], inplace=True)
    full_df = full_df[full_df[TARGET_COLUMN] > 0]
    y_full_log = np.log1p(full_df[TARGET_COLUMN])

    # --- NEW STRATEGY: Use ALL available numeric features together ---
    print("\n=============== PREPARING FULL FEATURE SET ===============")
    
    # Select all columns that are numeric, excluding the target variable itself
    all_numeric_cols = [c for c in full_df.columns if c != TARGET_COLUMN and pd.api.types.is_numeric_dtype(full_df[c])]
    X_full_numeric = full_df[all_numeric_cols].fillna(0)
    
    print(f"Using a single feature set with {X_full_numeric.shape[1]} numeric features.")
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_full_numeric), index=X_full_numeric.index, columns=X_full_numeric.columns)
    
    # Define a single name for this model set
    model_suite_name = "full_dataset"

    # --- PHASE 1: MODEL TRAINING ON THE FULL DATASET ---
    print("\n\n=============== PHASE 1: MODEL TRAINING ON FULL DATASET ===============")
    manage_dae_model(X_scaled, model_suite_name, mode='train')
    manage_tabnet_model(X_scaled, y_full_log, model_suite_name, mode='train')
    manage_generic_model(lgb.LGBMRegressor, X_scaled, y_full_log, model_suite_name, "LightGBM", random_state=42)
    manage_generic_model(xgb.XGBRegressor, X_scaled, y_full_log, model_suite_name, "XGBoost", random_state=42)
    manage_generic_model(cb.CatBoostRegressor, X_scaled, y_full_log, model_suite_name, "CatBoost", random_state=42)
    gc.collect()

    # --- PHASE 2: META-MODEL ASSEMBLY ---
    print("\n\n=============== PHASE 2: META-MODEL ASSEMBLY ===============")
    print("Collecting Predictions and Embeddings from all trained models...")
    meta_features_list = []
    meta_features_list.append(manage_dae_model(X_scaled, model_suite_name, mode='predict'))
    meta_features_list.append(manage_tabnet_model(X_scaled, y_full_log, model_suite_name, mode='predict'))
    meta_features_list.append(manage_generic_model(lgb.LGBMRegressor, X_scaled, y_full_log, model_suite_name, "LightGBM", mode='predict'))
    meta_features_list.append(manage_generic_model(xgb.XGBRegressor, X_scaled, y_full_log, model_suite_name, "XGBoost", mode='predict'))
    meta_features_list.append(manage_generic_model(cb.CatBoostRegressor, X_scaled, y_full_log, model_suite_name, "CatBoost", mode='predict'))
    gc.collect()

    print("\nAssembling final meta-dataset...")
    meta_df = pd.concat([f for f in meta_features_list if f is not None], axis=1)
    meta_df = meta_df.loc[:, ~meta_df.columns.duplicated()]
    meta_df.fillna(meta_df.mean(), inplace=True)

    # --- PHASE 3: FINAL META-MODEL TRAINING ---
    print("\n\n=============== PHASE 3: FINAL META-MODEL TRAINING ===============")
    X_meta = meta_df
    y_meta = y_full_log.loc[X_meta.index]
    X_meta_train, X_meta_test, y_meta_train, y_meta_test = train_test_split(X_meta, y_meta, test_size=0.2, random_state=42)
    
    print("--- Training Final LightGBM Meta-Model ---")
    meta_model = lgb.LGBMRegressor(random_state=42, n_estimators=500, learning_rate=0.05, num_leaves=31)
    meta_model.fit(X_meta_train, y_meta_train)
    
    preds_meta = meta_model.predict(X_meta_test)
    final_r2 = r2_score(y_meta_test, preds_meta)
    y_test_orig = np.expm1(y_meta_test)
    preds_orig = np.expm1(preds_meta)
    final_mae_orig = mean_absolute_error(y_test_orig, preds_orig)

    print("\n--- FINAL META-MODEL PERFORMANCE ---")
    print(f"  - R2 Score (log): {final_r2:.4f}")
    print(f"  - MAE (original scale): £{final_mae_orig:,.2f}")
    
    joblib.dump(meta_model, os.path.join(LOCAL_OUTPUT_DIR, "FINAL_META_MODEL.joblib"))
    print("Final meta-model saved.")
    print("\n\nPipeline finished successfully.")

if __name__ == "__main__":
    main()
EOF

# --- Pipeline Execution ---
echo "Downloading master dataset from GCS..."
gsutil cp "${MASTER_DATA_GCS_PATH}" "${MASTER_DATA_LOCAL_PATH}"

echo "Downloading Rightmove dataset from GCS..."
gsutil cp "${RIGHTMOVE_GCS_PATH}" "${RIGHTMOVE_DATA_LOCAL_PATH}"

echo "Running Python worker script to train the comprehensive model suite..."
${PYTHON_CMD} "${SCRIPT_PATH}"

echo "Uploading all artifacts to GCS..."
gsutil -m cp -r "${OUTPUT_DIR}/*" "${OUTPUT_GCS_BASE_DIR}/"

echo "Comprehensive model factory run finished. Results are in ${OUTPUT_GCS_BASE_DIR}"