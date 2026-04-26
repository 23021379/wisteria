# These models will be identical in architecture to the other ones
# The data will be unencoded, allowing for interpretability; shap can be used and we can clearly understand the feature importances.








#!/bin/bash
#
# run_model_training_v5.sh
#
# V5 REVISION: Using row index matching between raw and processed files
#

# -- Strict Mode & Configuration --
set -e
set -o pipefail
set -x

# -- GCP & Project Configuration --
GCS_BUCKET="srgan-bucket-ace-botany-453819-t4"
TIMESTAMP=$(date +"%Y%m%d-%H%M%S")
OUTPUT_GCS_DIR="gs://${GCS_BUCKET}/model_training/run_${TIMESTAMP}"
N_TRIALS=50 # Number of Bayesian Optimization trials to run
MASTER_DATA_GCS_PATH="gs://${GCS_BUCKET}/imputation_pipeline/output_lgbm_20250727-102025/final_fully_imputed_dataset.parquet"
LATEST_FEATURE_SORTING_RUN_DIR=$(gsutil ls -d "gs://${GCS_BUCKET}/feature_sorting/run_*" | tail -n 1)
FEATURE_SETS_GCS_PATH="${LATEST_FEATURE_SORTING_RUN_DIR}feature_sets.json"
RIGHTMOVE_GCS_PATH="gs://${GCS_BUCKET}/house data scrape/Rightmove.csv"

# --- UPDATED: Use the unprocessed file as key map ---
KEY_MAP_GCS_PATH="gs://${GCS_BUCKET}/house data scrape/merged_property_data_with_coords.csv"

# -- Local Configuration --
PROJECT_DIR="./model_training_project_v5"
VENV_DIR="${PROJECT_DIR}/venv_mt"
OUTPUT_DIR="${PROJECT_DIR}/output"
DATA_DIR="${PROJECT_DIR}/data"
MASTER_DATA_LOCAL_PATH="${DATA_DIR}/master_dataset.parquet"
FEATURE_SETS_LOCAL_PATH="${DATA_DIR}/feature_sets.json"
RIGHTMOVE_DATA_LOCAL_PATH="${DATA_DIR}/Rightmove.csv"
SCRIPT_PATH="${PROJECT_DIR}/02_train_multi_head_model_v5.py"
KEY_MAP_LOCAL_PATH="${DATA_DIR}/key_map.csv"

# --- Create Local Directory Structure ---
mkdir -p "${PROJECT_DIR}" "${OUTPUT_DIR}" "${DATA_DIR}"
cd "${PROJECT_DIR}"

# --- Environment Setup ---
if [ ! -d "${VENV_DIR}" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv "${VENV_DIR}"
fi
source "${VENV_DIR}/bin/activate"

echo "Installing required Python packages for deep learning..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pandas pyarrow gcsfs google-cloud-storage scikit-learn lightgbm fuzzywuzzy optuna

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
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.metrics import mean_absolute_error, r2_score
import lightgbm as lgb
import hashlib
from fuzzywuzzy import fuzz, process
import optuna

# --- Configuration ---
MASTER_DATA_PATH = os.environ.get("MASTER_DATA_LOCAL_PATH")
FEATURE_SETS_PATH = os.environ.get("FEATURE_SETS_LOCAL_PATH")
RIGHTMOVE_DATA_PATH = os.environ.get("RIGHTMOVE_DATA_LOCAL_PATH")
KEY_MAP_PATH = os.environ.get("KEY_MAP_LOCAL_PATH")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR")
N_TRIALS = int(os.environ.get("N_TRIALS", 50))
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_FOLDS_OPTUNA = 3 # Use fewer folds for fast hyperparameter search.
NUM_FOLDS_FINAL = 10 # Use more folds for robust final model training.
BATCH_SIZE = 256 # This will be optimized
EPOCHS = 150 # Reduced for faster trials, can be increased later

# --- Rightmove Parsing Functions ---
def parse_sales_history(sales_str):
    if pd.isna(sales_str) or sales_str == '' or sales_str == '[]': return pd.DataFrame()
    try:
        sales_list = ast.literal_eval(str(sales_str)); sales_data = []
        if not isinstance(sales_list, list): return pd.DataFrame()
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
    processed_data = []; sales_history_col = None
    if 'sales_history' in raw_rightmove_df.columns: sales_history_col = 'sales_history'
    elif raw_rightmove_df.shape[1] > 2: sales_history_col = raw_rightmove_df.columns[2]
    else: raise ValueError("Sales history column not found in Rightmove data.")
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

def extract_address_components(address):
    """Extract meaningful components from address strings"""
    if pd.isna(address) or not isinstance(address, str):
        return {'full': '', 'house_num': '', 'street': '', 'area': '', 'postcode': ''}
    
    # Clean the address
    clean_addr = re.sub(r'[^\w\s]', '', address.lower().strip())
    
    # Extract house number (if present)
    house_num_match = re.match(r'^(\d+[a-z]?)', clean_addr)
    house_num = house_num_match.group(1) if house_num_match else ''
    
    # Extract postcode pattern (UK format)
    postcode_match = re.search(r'([a-z]{1,2}\d{1,2}\s?\d[a-z]{2})$', clean_addr)
    postcode = postcode_match.group(1) if postcode_match else ''
    
    # Extract area/city (common UK areas)
    uk_areas = ['london', 'birmingham', 'manchester', 'liverpool', 'leeds', 'sheffield', 
                'bristol', 'coventry', 'leicester', 'sunderland', 'reading', 'kingston',
                'bolton', 'luton', 'northampton', 'preston', 'milton keynes', 'aberdeen',
                'glasgow', 'edinburgh', 'cardiff', 'swansea', 'belfast', 'derry']
    
    area = ''
    for city in uk_areas:
        if city in clean_addr:
            area = city
            break
    
    # Extract street name (remove house number and postcode)
    street = clean_addr
    if house_num:
        street = street.replace(house_num, '', 1).strip()
    if postcode:
        street = street.replace(postcode, '').strip()
    if area:
        street = street.replace(area, '').strip()
    
    return {
        'full': clean_addr,
        'house_num': house_num,
        'street': street[:50],  # Limit length
        'area': area,
        'postcode': postcode
    }

def create_flexible_matching_keys(df, address_col):
    """Create multiple matching keys for flexible matching"""
    keys_data = []
    
    for idx, row in df.iterrows():
        address = str(row[address_col]) if address_col in row else ''
        components = extract_address_components(address)
        
        # Create multiple potential matching keys
        keys = []
        
        # Original normalized key
        original_key = re.sub(r'[^\w]', '', address.lower())
        if original_key:
            keys.append(original_key)
        
        # House number + street
        if components['house_num'] and components['street']:
            keys.append(components['house_num'] + components['street'])
        
        # Just street name
        if components['street'] and len(components['street']) > 3:
            keys.append(components['street'])
        
        # Street + area
        if components['street'] and components['area']:
            keys.append(components['street'] + components['area'])
        
        # Just house number (for simple addresses)
        if components['house_num']:
            keys.append(components['house_num'])
        
        # Area + postcode
        if components['area'] and components['postcode']:
            keys.append(components['area'] + components['postcode'])
        
        keys_data.append({
            'original_index': idx,
            'keys': keys,
            'components': components,
            'original_address': address
        })
    
    return keys_data

# --- PyTorch & Model Classes ---
class WisteriaDataset(Dataset):
    def __init__(self, data_dict, y_price, y_uncertainty):
        # FIX: Make a local copy to prevent mutating the original dictionary passed in.
        local_data_dict = data_dict.copy()
        msoa_df = local_data_dict.pop('msoa_id')
        self.msoa_id_tensor = torch.tensor(msoa_df.values, dtype=torch.long).squeeze()
        self.feature_tensors = {k: torch.tensor(v.values, dtype=torch.float32) for k, v in local_data_dict.items()}
        self.y_price = torch.tensor(y_price.values, dtype=torch.float32).unsqueeze(1)
        self.y_uncertainty = torch.tensor(y_uncertainty.values, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.y_price)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.feature_tensors.items()}
        item['msoa_id'] = self.msoa_id_tensor[idx]
        item['target_price'] = self.y_price[idx]
        item['target_uncertainty'] = self.y_uncertainty[idx]
        return item

class Chomp1d(nn.Module):
    def __init__(self, chomp_size): super(Chomp1d, self).__init__(); self.chomp_size = chomp_size
    def forward(self, x): return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2): super(TemporalBlock, self).__init__(); self.conv1 = nn.utils.parametrizations.weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)); self.chomp1 = Chomp1d(padding); self.relu1 = nn.ReLU(); self.dropout1 = nn.Dropout(dropout); self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1); self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None; self.relu = nn.ReLU(); self.init_weights()
    def init_weights(self): self.conv1.weight.data.normal_(0, 0.01); _ = self.downsample.weight.data.normal_(0, 0.01) if self.downsample is not None else None
    def forward(self, x): out = self.net(x); res = x if self.downsample is None else self.downsample(x); return self.relu(out + res)

class TCNHead(nn.Module):
    def __init__(self, input_dim, n_features_per_timestep, n_timesteps, num_channels, kernel_size=2, dropout=0.2):
        super(TCNHead, self).__init__()
        # Ensure the input dimension is correct
        assert input_dim == n_features_per_timestep * n_timesteps, \
            f"Input dimension ({input_dim}) must equal n_features_per_timestep ({n_features_per_timestep}) * n_timesteps ({n_timesteps})"
        
        self.n_features_per_timestep = n_features_per_timestep
        self.n_timesteps = n_timesteps
        
        layers = []
        num_levels = len(num_channels)
        # The first TemporalBlock input should be the number of features per timestep
        input_channels = n_features_per_timestep
        
        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = input_channels if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers.append(TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, padding=(kernel_size-1) * dilation_size, dropout=dropout))
        
        self.network = nn.Sequential(*layers)
        self.output_dim = num_channels[-1]

    def forward(self, x):
        # x is expected to have shape (batch_size, n_timesteps * n_features_per_timestep)
        batch_size = x.shape[0]
        
        # Reshape to (batch_size, n_timesteps, n_features_per_timestep)
        x_reshaped = x.view(batch_size, self.n_timesteps, self.n_features_per_timestep)
        
        # Permute to (batch_size, n_features_per_timestep, n_timesteps) for TCN input
        x_permuted = x_reshaped.permute(0, 2, 1)
        
        # The output of the TCN network will be (batch_size, num_output_channels, n_timesteps)
        output = self.network(x_permuted)
        
        # Return the output of the last timestep
        return output[:, :, -1]


class EnhancedTCNHead(nn.Module):
    def __init__(self, input_dim, n_features_per_timestep, n_timesteps, num_channels, kernel_size=2, dropout=0.2):
        super(EnhancedTCNHead, self).__init__()
        assert input_dim == n_features_per_timestep * n_timesteps, \
            f"Input dimension ({input_dim}) must equal n_features_per_timestep ({n_features_per_timestep}) * n_timesteps ({n_timesteps})"
        
        self.n_features_per_timestep = n_features_per_timestep
        self.n_timesteps = n_timesteps
        
        # --- TCN Backbone ---
        layers = []
        num_levels = len(num_channels)
        input_channels = n_features_per_timestep
        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = input_channels if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers.append(TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, padding=(kernel_size-1) * dilation_size, dropout=dropout))
        self.tcn_backbone = nn.Sequential(*layers)
        
        # --- Attention Mechanism ---
        tcn_output_channels = num_channels[-1]
        self.attention = nn.Sequential(
            nn.Linear(tcn_output_channels, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        self.output_dim = tcn_output_channels

    def forward(self, x):
        # x: (batch_size, n_timesteps * n_features_per_timestep)
        batch_size = x.shape[0]
        
        # Reshape and Permute for TCN
        x_reshaped = x.view(batch_size, self.n_timesteps, self.n_features_per_timestep)
        x_permuted = x_reshaped.permute(0, 2, 1) # -> (batch_size, features, timesteps)
        
        # Get TCN feature maps
        tcn_out = self.tcn_backbone(x_permuted) # -> (batch_size, tcn_channels, timesteps)
        tcn_out_permuted = tcn_out.permute(0, 2, 1) # -> (batch_size, timesteps, tcn_channels)
        
        # Calculate attention weights
        # attn_weights shape: (batch_size, timesteps, 1)
        attn_weights = self.attention(tcn_out_permuted)
        attn_weights = torch.softmax(attn_weights, dim=1)
        
        # Apply attention weights
        # context_vector shape: (batch_size, 1, tcn_channels)
        context_vector = torch.bmm(attn_weights.transpose(1, 2), tcn_out_permuted)
        
        # Squeeze to final output shape: (batch_size, tcn_channels)
        final_output = context_vector.squeeze(1)
        
        return final_output


class FusionModel(nn.Module):
    def __init__(self, feature_configs, msoa_cardinality, msoa_embedding_dim=16, hidden_dim=256, use_enhanced_tcn=True):
        super().__init__()
        
        self.heads = nn.ModuleDict()
        tcn_class = EnhancedTCNHead if use_enhanced_tcn else TCNHead

        for head_name, config in feature_configs.items():
            input_dim = config['input_dim']
            
            # Logic to determine head type based on name
            if 'spatio_temporal' in head_name or 'head_E_temporal' in head_name:
                # This is a TCN-style head
                print(f"  - Creating TCN Head: '{head_name}' with input dim {input_dim}")
                self.heads[head_name] = tcn_class(
                    input_dim=input_dim,
                    n_features_per_timestep=config['n_features_per_timestep'],
                    n_timesteps=config['n_timesteps'],
                    num_channels=[64, 128, hidden_dim]
                )
            else:
                # This is a standard MLP head
                print(f"  - Creating MLP Head: '{head_name}' with input dim {input_dim}")
                # Using a simple but effective MLP structure for all non-TCN heads
                self.heads[head_name] = nn.Sequential(
                    nn.Linear(input_dim, 512),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(512, hidden_dim)
                )

        num_heads = len(self.heads)
        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, batch_first=True, dropout=0.2)
        self.msoa_embedding = nn.Embedding(msoa_cardinality, msoa_embedding_dim)
        
        # The fusion layer's input size is now dynamically calculated
        fusion_input_dim = hidden_dim * num_heads + msoa_embedding_dim
        print(f"--- Fusion layer input dimension: {fusion_input_dim} ({num_heads} heads) ---")
        
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, 1024), 
            nn.ReLU(), 
            nn.Dropout(0.5), 
            nn.Linear(1024, 512), 
            nn.ReLU(), 
            nn.Dropout(0.5)
        )
        self.price_head = nn.Linear(512, 3)
        self.uncertainty_head = nn.Linear(512, 1)

    def forward(self, x):
        # Dynamically process each head
        head_outputs = []
        for head_name, head_module in self.heads.items():
            head_outputs.append(head_module(x[head_name]))
        
        # Stack all head outputs for the attention mechanism
        stacked_heads = torch.stack(head_outputs, dim=1)
        
        attn_output, _ = self.cross_attention(stacked_heads, stacked_heads, stacked_heads)
        fused_attention = attn_output.reshape(attn_output.size(0), -1)
        
        msoa_embed = self.msoa_embedding(x['msoa_id'])
        final_fused = torch.cat([fused_attention, msoa_embed], dim=1)
        
        final_representation = self.fusion_mlp(final_fused)
        return self.price_head(final_representation), self.uncertainty_head(final_representation)

def quantile_loss(preds, target, quantiles=[0.05, 0.5, 0.95]): losses = []; [losses.append(torch.max((q - 1) * (target - preds[:, i].unsqueeze(1)), q * (target - preds[:, i].unsqueeze(1)))) for i, q in enumerate(quantiles)]; return torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))


class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=32):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid() # Use Sigmoid if data is scaled to [0,1]
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def encode(self, x):
        return self.encoder(x)

def train_and_apply_autoencoder(df, cols, latent_dim=32, ae_epochs=100, ae_batch_size=256):
    """Trains an autoencoder on the given columns and returns the compressed features."""
    print(f"    - Training Autoencoder for {len(cols)} features -> {latent_dim} latent dims...")
    
    # 1. Prepare data for this specific head
    head_data = df[cols].copy().apply(pd.to_numeric, errors='coerce').fillna(0)
    
    # Use MinMaxScaler for autoencoders with Sigmoid output
    scaler = StandardScaler() # Using StandardScaler is fine if we dont use Sigmoid in decoder
    data_scaled = scaler.fit_transform(head_data)
    
    dataset = torch.utils.data.TensorDataset(torch.tensor(data_scaled, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=ae_batch_size, shuffle=True)
    
    # 2. Train the Autoencoder
    model = Autoencoder(input_dim=len(cols), latent_dim=latent_dim).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    model.train()
    for epoch in range(ae_epochs):
        for data in loader:
            inputs = data[0].to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()

    # 3. Apply the trained encoder
    model.eval()
    with torch.no_grad():
        encoded_data = model.encode(torch.tensor(data_scaled, dtype=torch.float32).to(DEVICE))
    
    print(f"    - Autoencoder training complete. Output shape: {encoded_data.shape}")
    
    # Return as a pandas DataFrame
    return pd.DataFrame(encoded_data.cpu().numpy(), index=df.index)


def train_one_epoch(model, loader, optimizer, uncertainty_weight):
    model.train(); total_loss = 0; price_loss_fn = quantile_loss; uncertainty_loss_fn = nn.MSELoss()
    for batch in loader: batch = {k: v.to(DEVICE) for k, v in batch.items()}; optimizer.zero_grad(); price_preds, uncertainty_preds = model(batch); loss = price_loss_fn(price_preds, batch['target_price']) + uncertainty_weight * uncertainty_loss_fn(uncertainty_preds, batch['target_uncertainty']); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader):
    model.eval(); all_price_preds, all_uncertainty_preds = [], []
    with torch.no_grad():
        for batch in loader: batch = {k: v.to(DEVICE) for k, v in batch.items()}; price_preds, uncertainty_preds = model(batch); all_price_preds.append(price_preds.cpu()); all_uncertainty_preds.append(uncertainty_preds.cpu())
    return torch.cat(all_price_preds), torch.cat(all_uncertainty_preds)

def find_msoa_column(df):
    """Find the MSOA column in the dataframe"""
    msoa_candidates = [col for col in df.columns if 'msoa' in col.lower()]
    if not msoa_candidates:
        # If no MSOA column found, create a dummy one
        print("WARNING: No MSOA column found. Creating dummy MSOA IDs.")
        df['dummy_msoa'] = pd.factorize(df.index % 100)[0]  # Create 100 dummy areas
        return 'dummy_msoa'
    else:
        print(f"Found MSOA candidates: {msoa_candidates}")
        return msoa_candidates[0]  # Use the first one found

def prepare_tcn_data(df, tcn_cols, pattern):
    """
    Parses column names to prepare data for a TCN head.
    Returns a scaled, flattened DataFrame and a config dictionary.
    This version is robust to handle two types of temporal formats:
    1. Standard temporal: YYYY_STEM (e.g., '2004_job_seekers')
    2. Spatio-temporal: PREFIX_YYYY_SUFFIX (e.g., 'compass_mean_2004_sale_count_F_n20')
    """
    if not tcn_cols:
        print(f"  - WARNING: No columns provided for TCN processing. Returning default.")
        return pd.DataFrame(np.zeros((len(df), 1)), index=df.index), {'n_timesteps': 1, 'n_features_per_timestep': 1}

    parsed_cols = {}
    spatio_pattern = re.compile(r"^(.*)_(\d{4})_(.*)$")
    temporal_pattern = re.compile(r"^(\d{4})_(.*)$")

    for col in tcn_cols:
        match_spatio = spatio_pattern.match(col)
        match_temporal = temporal_pattern.match(col)

        year, feature_stem = None, None

        # The patterns are mutually exclusive due to the '^' anchor
        if match_spatio:
            prefix, year, suffix = match_spatio.groups()
            feature_stem = f"{prefix}_{suffix}"
        elif match_temporal:
            year, feature_stem = match_temporal.groups()

        if year and feature_stem:
            if feature_stem not in parsed_cols:
                parsed_cols[feature_stem] = {}
            parsed_cols[feature_stem][year] = col

    if not parsed_cols:
        print(f"  - WARNING: No temporal patterns matched any columns in this set. Returning default.")
        return pd.DataFrame(np.zeros((len(df), 4)), index=df.index), {'n_timesteps': 2, 'n_features_per_timestep': 2}

    feature_stems = sorted(list(parsed_cols.keys()))
    all_years = sorted(list(set(y for f in parsed_cols.values() for y in f.keys())))
    n_timesteps = len(all_years) if all_years else 1
    n_features = len(feature_stems) if feature_stems else 1

    print(f"  - TCN Prep: Found {n_features} feature types across {n_timesteps} years.")

    tcn_array = np.zeros((len(df), n_timesteps, n_features))
    for i, stem in enumerate(feature_stems):
        for j, year in enumerate(all_years):
            col_name = parsed_cols[stem].get(year)
            if col_name and col_name in df.columns:
                tcn_array[:, j, i] = df[col_name].fillna(0).values

    tcn_flat = tcn_array.reshape(len(df), -1)
    scaler = QuantileTransformer(output_distribution='normal', n_quantiles=max(min(len(df) // 10, 100), 10), random_state=42)
    tcn_scaled = scaler.fit_transform(tcn_flat)
    
    config = {'n_timesteps': n_timesteps, 'n_features_per_timestep': n_features}
    
    return pd.DataFrame(tcn_scaled, index=df.index), config

def train_one_epoch(model, loader, optimizer, uncertainty_weight):
    model.train(); total_loss = 0; price_loss_fn = quantile_loss; uncertainty_loss_fn = nn.MSELoss()
    for batch in loader: batch = {k: v.to(DEVICE) for k, v in batch.items()}; optimizer.zero_grad(); price_preds, uncertainty_preds = model(batch); loss = price_loss_fn(price_preds, batch['target_price']) + uncertainty_weight * uncertainty_loss_fn(uncertainty_preds, batch['target_uncertainty']); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader):
    model.eval(); all_price_preds, all_uncertainty_preds = [], []
    with torch.no_grad():
        for batch in loader: batch = {k: v.to(DEVICE) for k, v in batch.items()}; price_preds, uncertainty_preds = model(batch); all_price_preds.append(price_preds.cpu()); all_uncertainty_preds.append(uncertainty_preds.cpu())
    return torch.cat(all_price_preds), torch.cat(all_uncertainty_preds)

def find_msoa_column(df):
    """Find the MSOA column in the dataframe"""
    msoa_candidates = [col for col in df.columns if 'msoa' in col.lower()]
    if not msoa_candidates:
        # If no MSOA column found, create a dummy one
        print("WARNING: No MSOA column found. Creating dummy MSOA IDs.")
        df['dummy_msoa'] = pd.factorize(df.index % 100)[0]  # Create 100 dummy areas
        return 'dummy_msoa'
    else:
        print(f"Found MSOA candidates: {msoa_candidates}")
        return msoa_candidates[0]  # Use the first one found

def validate_merge_quality(df, sample_size=10):
    """Validate that merged properties actually match by comparing original addresses."""
    print(f"\n--- MERGE VALIDATION (Sample of {sample_size}) ---")
    address_col_rm = [col for col in df.columns if 'address' in col.lower()][0] # Find the rightmove address col
    
    sample_df = df.sample(min(sample_size, len(df)))

    for i, (_, row) in enumerate(sample_df.iterrows()):
        feature_addr = row.get('property_id', 'N/A')
        rightmove_addr = row.get(address_col_rm, 'N/A')
        merge_key = row.get('final_merge_key', 'N/A')
        price = row.get('most_recent_sale_price', 0)

        # Use fuzzy matching to score the similarity of the original addresses
        similarity_score = fuzz.token_set_ratio(str(feature_addr), str(rightmove_addr))

        print(f"--- Record {i+1} / Price: £{price:,.0f} ---")
        print(f"  Merge Key   : '{merge_key}'")
        print(f"  Similarity  : {similarity_score}%")
        print(f"  Features Addr: {str(feature_addr)[:80]}")
        print(f"  Rightmove Addr: {str(rightmove_addr)[:80]}")
        if similarity_score < 70:
            print("  [!! WARNING: LOW SIMILARITY. THIS IS LIKELY A BAD MATCH !!]")
        print()


def objective(trial, data_for_model, y_price_for_training, y_uncertainty, feature_configs, msoa_cardinality, y_price_std, y_price_mean, df_for_eval):
    """
    The objective function for Optuna to optimize.
    It defines hyperparameters, trains the model K-Fold, and returns the validation score.
    """
    # --- 1. Define Hyperparameter Search Space ---
    print(f"\n--- Starting Trial {trial.number} ---")
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
        'uncertainty_loss_weight': trial.suggest_float('uncertainty_loss_weight', 0.1, 0.5),
        'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
        'fusion_hidden_dim': trial.suggest_categorical('fusion_hidden_dim', [128, 256, 512]),
        'msoa_embedding_dim': trial.suggest_categorical('msoa_embedding_dim', [8, 16, 32]),
        'batch_size': trial.suggest_categorical('batch_size', [128, 256, 512]),
    }
    
    # --- 2. K-Fold Cross-Validation Training Loop ---
    kf = KFold(n_splits=NUM_FOLDS_OPTUNA, shuffle=True, random_state=42)
    oof_price_preds = np.zeros((len(df_for_eval), 3))
    fold_maes = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(df_for_eval)):
        print(f"  ======== OPTUNA FOLD {fold+1}/{NUM_FOLDS_OPTUNA} (Trial {trial.number}) ========")
        train_data = {k: v.iloc[train_idx] for k, v in data_for_model.items()}
        val_data = {k: v.iloc[val_idx] for k, v in data_for_model.items()}
        
        train_dataset = WisteriaDataset(train_data, y_price_for_training.iloc[train_idx], y_uncertainty.iloc[train_idx])
        val_dataset = WisteriaDataset(val_data, y_price_for_training.iloc[val_idx], y_uncertainty.iloc[val_idx])
        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=params['batch_size'] * 2, shuffle=False, num_workers=2, pin_memory=True)
        
        # We need to update the model to accept the dynamic dropout rate
        # This requires a small change to the FusionModel definition (or pass it as an arg)
        # For simplicity, let's assume we can modify it during instantiation.
        # Note: A cleaner way is to pass dropout as an argument to FusionModel's __init__
        model = FusionModel(
            feature_configs=feature_configs, 
            msoa_cardinality=msoa_cardinality,
            msoa_embedding_dim=params['msoa_embedding_dim'],
            hidden_dim=params['fusion_hidden_dim'],
            use_enhanced_tcn=True
        ).to(DEVICE)
        
        optimizer = optim.AdamW(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=5)
        
        best_loss = float('inf')
        for epoch in range(EPOCHS):
            train_loss = train_one_epoch(model, train_loader, optimizer, params['uncertainty_loss_weight'])
            scheduler.step(train_loss)
            if train_loss < best_loss:
                best_loss = train_loss
        
        price_preds, _ = evaluate(model, val_loader)
        oof_price_preds[val_idx] = price_preds.numpy()

        # --- 3. Evaluate and Report ---
        median_preds_scaled = price_preds.numpy()[:, 1]
        median_preds_unscaled = np.expm1(median_preds_scaled * y_price_std + y_price_mean)
        true_values = df_for_eval.iloc[val_idx]['most_recent_sale_price']
        
        fold_mae = mean_absolute_error(true_values, median_preds_unscaled)
        fold_maes.append(fold_mae)
        print(f"    Fold {fold+1} MAE: £{fold_mae:,.2f}")

        # Optuna Pruning: Stop unpromising trials early
        trial.report(fold_mae, fold)
        if trial.should_prune():
            print("  -- Trial pruned --")
            raise optuna.exceptions.TrialPruned()

        del model, optimizer, scheduler; gc.collect(); torch.cuda.empty_cache()

    avg_mae = np.mean(fold_maes)
    print(f"--- Trial {trial.number} Finished. Average MAE: £{avg_mae:,.2f} ---")
    return avg_mae


def find_best_matches_fuzzy(keys1, keys2, threshold=85):
    """Finds the best fuzzy match for each key in keys1 from keys2."""
    print(f"  - Attempting to find fuzzy matches for {len(keys1)} keys...")
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
    Detects leakage by checking for near-perfect correlation between scaled AVM features and a scaled label.

    Args:
        df (pd.DataFrame): The dataframe, which will be modified.
        label_col (str): The raw, unscaled target variable column (e.g., 'most_recent_sale_price').
        correlation_threshold (float): The Pearson correlation coefficient above which a feature is flagged.
        noise_level (float): The standard deviation of the noise to add, as a fraction of the feature's std dev.

    Returns:
        pd.DataFrame: The dataframe with noise added to leaky columns.
    """
    print("\n--- STAGE 4.1: Mitigating AVM Target Leakage (Correlation Check) ---")
    avm_keywords = ['estimate', 'valuation', 'avm', 'homipi', 'zoopla', 'bnl', 'bricks', 'chimnie']
    mitigated_cols = []
    df_copy = df.copy()

    if label_col not in df_copy.columns:
        print(f"  - FATAL: Label column '{label_col}' not found. Cannot perform leakage check.")
        return df

    # 1. Scale the raw label to create a comparable series
    label_raw = pd.to_numeric(df_copy[label_col], errors='coerce').dropna()
    scaler = StandardScaler()
    label_scaled = pd.Series(scaler.fit_transform(label_raw.values.reshape(-1, 1)).flatten(), index=label_raw.index)

    # 2. Iterate through columns to find potential AVM features
    for col in df_copy.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in avm_keywords):
            print(f"  - Analyzing potential AVM feature: '{col}'")
            if not pd.api.types.is_numeric_dtype(df_copy[col]):
                print(f"    - Skipping non-numeric column.")
                continue

            # 3. Calculate correlation between the scaled feature and the scaled label
            #    We align them by index to handle any potential NaNs correctly.
            temp_df = pd.concat([df_copy[col], label_scaled], axis=1, keys=['feature', 'label']).dropna()
            if len(temp_df) < 2: continue # Not enough data to calculate correlation
            
            correlation = temp_df['feature'].corr(temp_df['label'])

            if pd.isna(correlation): continue

            # 4. Check if correlation exceeds the threshold
            if abs(correlation) > correlation_threshold:
                print(f"  [!! LEAKAGE DETECTED !!] Column '{col}' has a correlation of {correlation:.4f} with the scaled label.")
                col_std = df_copy[col].std()
                if pd.notna(col_std) and col_std > 0:
                    noise = np.random.normal(0, noise_level * col_std, size=len(df_copy))
                    df_copy.loc[df_copy[col].notna(), col] += noise
                    print(f"  - Added noise (std dev: {noise_level * col_std:.4f}) to column '{col}'.")
                    mitigated_cols.append(col)
                else:
                    print(f"  - Could not add noise to column '{col}' due to zero or NaN standard deviation.")
            else:
                print(f"  - Column '{col}' passed leakage check (Correlation: {correlation:.4f}).")

    if not mitigated_cols:
        print("  - No significant correlation leakage detected.")

    return df_copy


def main():
    warnings.filterwarnings('ignore')
    print(f"Using device: {DEVICE}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- STAGE 1: Load Data Sources (Simplified) ---
    print("--- STAGE 1: Loading Data Sources ---")
    try:
        # Load datasets
        df_features = pd.read_parquet(MASTER_DATA_PATH)
        df_rightmove_raw = pd.read_csv(RIGHTMOVE_DATA_PATH, on_bad_lines='skip')
        
        with open(FEATURE_SETS_PATH, 'r') as f:
            feature_sets = json.load(f)
        
        print(f"  - Loaded features: {df_features.shape}")
        print(f"  - Loaded Rightmove: {df_rightmove_raw.shape}")
        print(f"  - Features already has property_id column: {'property_id' in df_features.columns}")
        
    except Exception as e:
        raise IOError(f"FATAL: Could not load data files. Error: {e}")

    # --- STAGE 2: Direct Address Normalization (No Key Map Needed) ---
    print("\n--- STAGE 2: Direct Address Processing ---")
    
    def normalize_address_key_v3(address, is_rightmove_url=False):
        """Final address normalization focused on core matching"""
        if not isinstance(address, str) or address.strip() == '':
            return None
        
        # Handle Rightmove URL format
        if is_rightmove_url:
            match = re.search(r'^(.*?)https://www\.rightmove\.co\.uk', address, re.IGNORECASE)
            if match:
                address = match.group(1).strip()
        
        # Convert to lowercase and clean
        address = address.lower().strip()
        
        # Remove all punctuation and spaces for maximum matching potential
        address = re.sub(r'[^\w]', '', address)
        
        # Extract just the core components (first 20 characters for comparison)
        # This handles cases where one dataset has partial addresses
        if len(address) > 20:
            address = address[:20]
        
        return address
    
    # Process features addresses (using existing property_id)
    print("  - Normalizing feature addresses...")
    df_features['normalized_address_key'] = df_features['property_id'].apply(
        lambda x: normalize_address_key_v3(x, False)
    )
    
    # Remove rows with null addresses
    df_features = df_features[df_features['normalized_address_key'].notna()].copy()
    
    print(f"  - Features with valid addresses: {len(df_features)}")
    print("  - Sample normalized feature keys:")
    sample_feature_keys = df_features['normalized_address_key'].head(10).tolist()
    print("   ", sample_feature_keys)

    # Process Rightmove data
    df_rightmove_raw.columns = df_rightmove_raw.columns.str.lower()
    address_col_rm = 'address' if 'address' in df_rightmove_raw.columns else df_rightmove_raw.columns[0]

    print("  - Normalizing Rightmove addresses...")
    df_rightmove_raw['normalized_address_key'] = df_rightmove_raw[address_col_rm].apply(
        lambda x: normalize_address_key_v3(x, is_rightmove_url=True)
    )
    
    # Remove rows with null addresses
    df_rightmove_raw = df_rightmove_raw[df_rightmove_raw['normalized_address_key'].notna()].copy()
    
    print(f"  - Rightmove with valid addresses: {len(df_rightmove_raw)}")
    print("  - Sample normalized Rightmove keys:")
    sample_rm_keys = df_rightmove_raw['normalized_address_key'].head(10).tolist()
    print("   ", sample_rm_keys)

    # Process sales data
    df_rightmove_processed = process_rightmove_data(df_rightmove_raw)

    # Merge with address keys AND original address for validation
    address_col_rm = 'address' if 'address' in df_rightmove_raw.columns else df_rightmove_raw.columns[0]
    df_rightmove_with_address_key = pd.merge(
        df_rightmove_raw[['normalized_address_key', address_col_rm]],
        df_rightmove_processed,
        left_index=True,
        right_on='rightmove_row_id',
        how='inner'
    )
    df_rightmove_with_address_key.drop(columns=['rightmove_row_id'], inplace=True, errors='ignore')

    print(f"  - Rightmove processed: {df_rightmove_with_address_key.shape}")

    # --- STAGE 3: Enhanced Matching Strategy ---
    print("\n--- STAGE 3: Enhanced Address Matching ---")
    
    # Check initial overlap
    features_keys = set(df_features['normalized_address_key'])
    rightmove_keys = set(df_rightmove_with_address_key['normalized_address_key'])
    initial_overlap = len(features_keys & rightmove_keys)
    
    print(f"  - Initial exact overlap: {initial_overlap} properties")
    
    if initial_overlap == 0:
        print("  - No exact matches found. Fuzzy/partial matching will be attempted in STAGE 4.")
    else:
        print(f"  - Exact key overlap of {initial_overlap} found. Merge will proceed in STAGE 4.")

    
    
    print("\n--- STAGE 4: Definitive Merge and Validation ---")
    # Handle duplicates before merging
    df_features.drop_duplicates(subset=['normalized_address_key'], keep='first', inplace=True)
    df_rightmove_with_address_key.drop_duplicates(subset=['normalized_address_key'], keep='first', inplace=True)

    # Check for overlap to determine merge strategy
    features_keys = set(df_features['normalized_address_key'].dropna())
    rightmove_keys = set(df_rightmove_with_address_key['normalized_address_key'].dropna())
    overlap = len(features_keys & rightmove_keys)
    merge_on_key = 'final_merge_key'

    if overlap > 0:
        print(f"  - Using {overlap} exact matches for merge.")
        df_features[merge_on_key] = df_features['normalized_address_key']
        df_rightmove_with_address_key[merge_on_key] = df_rightmove_with_address_key['normalized_address_key']
    else:
        print("  - No exact matches found. Attempting fuzzy matching fallback...")
        fuzzy_matches = find_best_matches_fuzzy(features_keys, rightmove_keys, threshold=85)
        
        if fuzzy_matches:
            fuzzy_mapping = {fkey: rkey for fkey, rkey, score in fuzzy_matches}
            df_features[merge_on_key] = df_features['normalized_address_key'].map(fuzzy_mapping)
            df_rightmove_with_address_key[merge_on_key] = df_rightmove_with_address_key['normalized_address_key']
            print(f"  - Created {len(fuzzy_mapping)} mappings from fuzzy matching.")
        else:
            print("  - WARNING: Fuzzy matching failed. Falling back to exact keys (will likely result in an empty merge).")
            df_features[merge_on_key] = df_features['normalized_address_key']
            df_rightmove_with_address_key[merge_on_key] = df_rightmove_with_address_key['normalized_address_key']

    # Define all columns to bring in from the Rightmove dataset
    if address_col_rm not in df_rightmove_with_address_key.columns:
        raise ValueError(f"FATAL: Rightmove address column '{address_col_rm}' missing before final merge.")
    
    cols_to_merge_from_rightmove = [
        merge_on_key, 'most_recent_sale_price', 'most_recent_sale_year',
        'most_recent_sale_month', 'total_sales_count', 'days_since_last_sale',
        'price_change_since_last', address_col_rm
    ]

    # Prevent column name collisions that cause KeyErrors downstream
    cols_to_drop_from_features = [c for c in cols_to_merge_from_rightmove if c in df_features.columns and c != merge_on_key]
    if cols_to_drop_from_features:
        print(f"  - Dropping conflicting columns from features df before merge: {cols_to_drop_from_features}")
        df_features.drop(columns=cols_to_drop_from_features, inplace=True, errors='ignore')

    # Perform the single, definitive merge
    df = pd.merge(
        df_features.dropna(subset=[merge_on_key]),
        df_rightmove_with_address_key,
        on=merge_on_key,
        how='inner'
    )

    print(f"  - Final merged dataset shape: {df.shape}")
    if len(df) == 0:
        raise ValueError("FATAL: Merge resulted in an empty dataset. Check normalization and key matching.")

    # --- DIAGNOSTICS: Report on Merge Failures ---
    print("\n--- DIAGNOSTICS: Analyzing Merge Failures ---")
    original_feature_keys = set(df_features[merge_on_key].dropna())
    merged_keys = set(df[merge_on_key])
    failed_keys = original_feature_keys - merged_keys
    
    print(f"  - Total properties in source features (pre-merge): {len(original_feature_keys)}")
    print(f"  - Total properties successfully merged: {len(merged_keys)}")
    print(f"  - Total properties that failed to merge: {len(failed_keys)}")
    
    if failed_keys:
        failed_merges_df = df_features[df_features[merge_on_key].isin(failed_keys)]
        print("\n  --- Sample of Properties That FAILED to Merge ---")
        sample_size = min(10, len(failed_merges_df))
        sample_failures = failed_merges_df.head(sample_size)
        for index, row in sample_failures.iterrows():
            print(f"  - Original Address: '{row.get('property_id', 'N/A')[:70]}...'")
            print(f"    Normalized Key: '{row.get(merge_on_key, 'N/A')}'")
        print("\n  - DEBUGGING HINT: The 'Normalized Key' above could not be found in the Rightmove data.")

    # --- Final Validation ---
    validate_merge_quality(df)
    
    # --- AVM LEAKAGE MITIGATION ---
    df = mitigate_avm_leakage(df, label_col='most_recent_sale_price', correlation_threshold=0.995)

    # --- NEW STAGE 4.2: Autoencoder Feature Engineering (BEFORE Feature Selection) ---
    print("\n--- STAGE 4.2: Engineering Features with Specialized Autoencoders ---")
    
    # Define prefixes for different spatial feature groups
    spatial_prefixes = {'compass': 'compass_', 'atlas': 'atlas_', 'microscope': 'microscope_'}
    all_original_spatial_cols = []
    
    # This will hold the final set of features (encoded compass + raw atlas/microscope)
    combined_spatial_features_df = pd.DataFrame(index=df.index)

    # 1. Encode 'compass' features
    compass_cols = [col for col in df.columns if col.startswith(spatial_prefixes['compass'])]
    all_original_spatial_cols.extend(compass_cols)
    if compass_cols:
        print(f"  - Training 'compass' AE on {len(compass_cols)} features...")
        encoded_compass_data = train_and_apply_autoencoder(df, compass_cols, latent_dim=32)
        encoded_compass_data.columns = [f"compass_ae_{i}" for i in range(encoded_compass_data.shape[1])]
        combined_spatial_features_df = pd.concat([combined_spatial_features_df, encoded_compass_data], axis=1)
    else:
        print("  - No 'compass' features found to encode.")

    # 2. Append raw 'atlas' and 'microscope' features
    for name in ['atlas', 'microscope']:
        prefix = spatial_prefixes[name]
        raw_cols = [col for col in df.columns if col.startswith(prefix)]
        all_original_spatial_cols.extend(raw_cols)
        if raw_cols:
            print(f"  - Directly appending {len(raw_cols)} raw '{name}' features.")
            raw_data = df[raw_cols].copy()
            combined_spatial_features_df = pd.concat([combined_spatial_features_df, raw_data], axis=1)
        else:
            print(f"  - No '{name}' features found to append.")

    print(f"  - Created a total of {combined_spatial_features_df.shape[1]} new spatial features (encoded + raw).")
    df = pd.concat([df, combined_spatial_features_df], axis=1)

    # --- STAGE 4.5: Hybrid Feature Selection (LGBM for flat, preserve for temporal) ---
    print("\n--- STAGE 4.5: Performing Hybrid Feature Selection ---")
    
    with open(FEATURE_SETS_PATH, 'r') as f:
        feature_sets_for_selection = json.load(f)

    # Identify temporal columns to PRESERVE from selection
    temporal_cols_to_preserve = []
    spatio_temporal_cols_to_preserve = []
    for head_name, cols in feature_sets_for_selection.items():
        if 'head_E_temporal' in head_name:
            temporal_cols_to_preserve.extend(cols)
        elif 'head_F_spatio_temporal' in head_name:
            spatio_temporal_cols_to_preserve.extend(cols)

    # Stricter filtering for preservation: only keep columns that are in the dataframe and truly look temporal
    temporal_cols_to_preserve = [c for c in temporal_cols_to_preserve if c in df.columns and re.match(r"^\d{4}_", c)]
    spatio_temporal_cols_to_preserve = [c for c in spatio_temporal_cols_to_preserve if c in df.columns]
    preserved_cols = temporal_cols_to_preserve + spatio_temporal_cols_to_preserve
    print(f"  - Identified {len(preserved_cols)} temporal columns to preserve from selection.")

    # Define columns to EXCLUDE from selection
    excluded_cols = [
        'property_id', 'normalized_address_key', 'final_merge_key', 'matched_rightmove_key',
        'most_recent_sale_price', 'most_recent_sale_year', 'most_recent_sale_month',
        'total_sales_count', 'days_since_last_sale', 'price_change_since_last'
    ]
    address_col_rm = [col for col in df.columns if 'address' in col.lower()]
    excluded_cols.extend(address_col_rm)
    
    # The pool for LGBM selection now INCLUDES the new AE features but EXCLUDES the original spatial features
    lgbm_selection_pool = [
        c for c in df.columns 
        if c not in excluded_cols 
        and c not in preserved_cols 
        and c not in all_original_spatial_cols # Exclude the raw spatial features
        and df[c].dtype in ['int64', 'float64', 'int32', 'float32']
    ]
    
    print(f"  - Found {len(lgbm_selection_pool)} features for LGBM selection (including new AE features).")
    
    X_select = df[lgbm_selection_pool].copy().fillna(0)
    y_select = np.log1p(df['most_recent_sale_price'])

    lgbm_selector = lgb.LGBMRegressor(random_state=42, n_estimators=200, n_jobs=-1, verbose=-1)
    lgbm_selector.fit(X_select, y_select)

    importances = pd.DataFrame({
        'feature': X_select.columns,
        'importance': lgbm_selector.feature_importances_
    }).sort_values('importance', ascending=False)
    
    N_TOP_FEATURES = 500
    top_features_from_selection = importances.head(N_TOP_FEATURES)['feature'].tolist()
    
    print(f"  - Selected Top {N_TOP_FEATURES} features via LGBM.")
    print("  - Top 10 most important features (note: may include AE features):")
    print(importances.head(10).to_string(index=False))

    msoa_col_name = find_msoa_column(df)
    core_cols_to_keep = excluded_cols + [msoa_col_name, 'most_recent_sale_price']
    
    final_cols = list(set(top_features_from_selection + preserved_cols + [c for c in core_cols_to_keep if c in df.columns]))
    df = df[final_cols]
    print(f"  - DataFrame filtered to {df.shape[1]} total columns for model training.")

    # --- STAGE 5: Filter, Validate, and Prepare Data for Model ---
    print("\n--- STAGE 5: Filtering, Validating and Preparing Data for Sub-Head Architecture ---")
    Q1, Q3 = df['most_recent_sale_price'].quantile(0.01), df['most_recent_sale_price'].quantile(0.99)
    df = df[(df['most_recent_sale_price'] >= Q1) & (df['most_recent_sale_price'] <= Q3)].copy().reset_index(drop=True)
    print(f"  - Dataset after outlier removal: {len(df)} properties")

    y_price_log = np.log1p(df['most_recent_sale_price'])
    y_price_mean, y_price_std = y_price_log.mean(), y_price_log.std()
    y_price_for_training = pd.Series((y_price_log - y_price_mean) / y_price_std, index=df.index)
    y_uncertainty = pd.Series(np.abs(np.random.randn(len(df)) * 0.05), index=df.index)

    data_for_model = {}
    feature_configs = {}

    # Add the pre-processed AE heads to the model data
    for name in spatial_prefixes.keys():
        ae_head_cols = [c for c in df.columns if c.startswith(f"{name}_ae_")]
        if ae_head_cols:
            head_name = f"head_{name}_ae"
            print(f"\n--- Adding pre-encoded '{head_name}' to model input ---")
            data_for_model[head_name] = df[ae_head_cols]
            feature_configs[head_name] = {'input_dim': len(ae_head_cols)}

    # Dynamically process the remaining heads from the feature_sets JSON
    for head_name, cols in feature_sets.items():
        if head_name == 'unassigned_features' or not cols: continue
        
        available_cols = [c for c in cols if c in df.columns]
        
        if not available_cols: continue
            
        print(f"\n--- Processing data for head: {head_name} ---")

        if 'spatio_temporal' in head_name or 'head_E_temporal' in head_name:
            # --- FIX: Stricter "filter-in" approach for temporal features ---
            original_count = len(available_cols)
            temporal_cols = [c for c in available_cols if re.match(r"^\d{4}_", c)]
            filtered_count = len(temporal_cols)
            
            if original_count != filtered_count:
                print(f"  - CORRECTED: Kept {filtered_count} of {original_count} features that match temporal pattern.")

            if not temporal_cols:
                print(f"  - WARNING: No valid temporal columns left for head '{head_name}' after strict filtering. Skipping.")
                continue

            regex = r"(\d{4})_(.+)"
            processed_data, tcn_config = prepare_tcn_data(df, temporal_cols, regex)
            data_for_model[head_name] = processed_data
            feature_configs[head_name] = {**tcn_config, 'input_dim': processed_data.shape[1]}
        
        elif any(prefix in head_name for prefix in ['head_D_spatial', 'head_atlas', 'head_compass', 'head_microscope']):
            # Skip any spatial heads as they are now handled by the pre-encoded AE features
            print(f"  - Skipping '{head_name}' as its features were handled by AE pre-processing.")
            continue

        else:
            # Standard MLP Head Processing for all other flat features
            head_df = df[available_cols].copy().apply(pd.to_numeric, errors='coerce').fillna(0).replace([np.inf, -np.inf], 0)
            scaler = QuantileTransformer(output_distribution='normal', n_quantiles=max(min(len(df) // 10, 100), 10), random_state=42)
            processed_data = pd.DataFrame(scaler.fit_transform(head_df), columns=available_cols, index=df.index)
            data_for_model[head_name] = processed_data
            feature_configs[head_name] = {'input_dim': processed_data.shape[1]}

    # Process MSOA embeddings
    msoa_col = find_msoa_column(df)
    df[msoa_col] = pd.factorize(df[msoa_col].fillna('Unknown'))[0]
    data_for_model['msoa_id'] = df[[msoa_col]]
    msoa_cardinality = df[msoa_col].nunique() + 1
    print(f"\n--- MSOA cardinality set to: {msoa_cardinality} ---")

    # --- STAGE 6: Hyperparameter Optimization with Optuna ---
    print("\n--- STAGE 6: Starting Bayesian Hyperparameter Optimization ---")
    
    # Create a study object and optimize the objective function.
    study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner())
    
    # Use a lambda to pass additional arguments to the objective function
    study.optimize(lambda trial: objective(
        trial, 
        data_for_model, 
        y_price_for_training, 
        y_uncertainty, 
        feature_configs, 
        msoa_cardinality,
        y_price_std,
        y_price_mean,
        df  # Pass the dataframe for evaluation purposes
    ), n_trials=N_TRIALS)

    print("\nOptimization Finished!")
    print("Best trial:")
    best_trial = study.best_trial
    print(f"  Value (MAE): £{best_trial.value:,.2f}")
    print("  Best Hyperparameters: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    # --- STAGE 7: Final Model Training with Best Hyperparameters ---
    print("\n--- STAGE 7: Training Final Model using Best Hyperparameters ---")
    best_params = best_trial.params
    kf = KFold(n_splits=NUM_FOLDS_FINAL, shuffle=True, random_state=42)
    oof_price_preds = np.zeros((len(df), 3))
    oof_uncertainty_preds = np.zeros((len(df), 1))
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
        print(f"\n========== FINAL TRAINING: FOLD {fold+1}/{NUM_FOLDS_FINAL} ==========")
        train_data = {k: v.iloc[train_idx] for k, v in data_for_model.items()}
        val_data = {k: v.iloc[val_idx] for k, v in data_for_model.items()}
        
        train_dataset = WisteriaDataset(train_data, y_price_for_training.iloc[train_idx], y_uncertainty.iloc[train_idx])
        val_dataset = WisteriaDataset(val_data, y_price_for_training.iloc[val_idx], y_uncertainty.iloc[val_idx])
        train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=best_params['batch_size'] * 2, shuffle=False, num_workers=2, pin_memory=True)
        
        model = FusionModel(
            feature_configs=feature_configs, 
            msoa_cardinality=msoa_cardinality,
            msoa_embedding_dim=best_params['msoa_embedding_dim'],
            hidden_dim=best_params['fusion_hidden_dim'],
            use_enhanced_tcn=True
        ).to(DEVICE)
        
        optimizer = optim.AdamW(model.parameters(), lr=best_params['learning_rate'], weight_decay=best_params['weight_decay'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=5)
        
        best_loss = float('inf')
        for epoch in range(EPOCHS + 50): # Train for a bit longer on the final model
            train_loss = train_one_epoch(model, train_loader, optimizer, best_params['uncertainty_loss_weight'])
            if epoch % 10 == 0: print(f"  Epoch {epoch+1}/{EPOCHS+50}, Train Loss: {train_loss:.4f}")
            scheduler.step(train_loss)
            if train_loss < best_loss:
                best_loss = train_loss
                torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f"model_fold_{fold}_best_params.pt"))
        
        model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, f"model_fold_{fold}_best_params.pt")))
        price_preds, uncertainty_preds = evaluate(model, val_loader)
        oof_price_preds[val_idx] = price_preds.numpy()
        oof_uncertainty_preds[val_idx] = uncertainty_preds.numpy()
        del model, optimizer, scheduler; gc.collect(); torch.cuda.empty_cache()
    
    # --- STAGE 7: Evaluate & Save ---
    print("\n--- STAGE 7: Evaluating Performance and Saving Artifacts ---")
    oof_df = pd.DataFrame(oof_price_preds, columns=['price_q05', 'price_q50', 'price_q95'], index=df.index)
    oof_df['uncertainty_pred'] = oof_uncertainty_preds
    eval_df = df.join(oof_df)
    eval_df['predicted_price_median'] = np.expm1(eval_df['price_q50'] * y_price_std + y_price_mean)

    mae = mean_absolute_error(eval_df['most_recent_sale_price'], eval_df['predicted_price_median'])
    r2 = r2_score(eval_df['most_recent_sale_price'], eval_df['predicted_price_median'])
    print(f"\n--- FINAL DEEP LEARNING MODEL PERFORMANCE ---")
    print(f"  - R-squared (R²): {r2:.4f}")
    print(f"  - Mean Absolute Error (MAE): £{mae:,.2f}")

    # --- STAGE 8: Train Residual Model ---
    print("\n--- STAGE 8: Training Residual Refinement Model ---")
    eval_df['log_residual'] = y_price_for_training - eval_df['price_q50']
    
    # Dynamically get all head names from the data dictionary for the residual model
    residual_heads = [k for k in data_for_model.keys() if 'head' in k]
    print(f"  - Using the following heads for residual model: {residual_heads}")
    X_residual_list = [data_for_model[h].reset_index(drop=True) for h in residual_heads]
    X_residual_raw = pd.concat(X_residual_list, axis=1)
    
    # Handle potential duplicate columns from features being in multiple heads
    X_residual = X_residual_raw.loc[:,~X_residual_raw.columns.duplicated()]
    print(f"  - Created residual feature set. Original columns: {X_residual_raw.shape[1]}, Unique columns: {X_residual.shape[1]}")

    y_residual = eval_df['log_residual']

    lgbm = lgb.LGBMRegressor(random_state=42, n_estimators=1000, learning_rate=0.05, num_leaves=31, n_jobs=-1)
    lgbm.fit(X_residual, y_residual)
    
    import joblib
    joblib.dump(lgbm, os.path.join(OUTPUT_DIR, "lgbm_residual_model.joblib"))
    eval_df.to_csv(os.path.join(OUTPUT_DIR, "oof_predictions_with_residuals.csv"), index=False)
    print("Residual model and final predictions saved successfully.")
    print("\nPipeline finished successfully.")

if __name__ == "__main__":
    main()
EOF

# --- Pipeline Execution ---
echo "Downloading master dataset from GCS..."
gsutil cp "${MASTER_DATA_GCS_PATH}" "${MASTER_DATA_LOCAL_PATH}"

echo "Downloading feature set definitions from GCS..."
gsutil cp "${FEATURE_SETS_GCS_PATH}" "${FEATURE_SETS_LOCAL_PATH}"

echo "Downloading Rightmove dataset from GCS..."
gsutil cp "${RIGHTMOVE_GCS_PATH}" "${RIGHTMOVE_DATA_LOCAL_PATH}"

echo "Downloading Key Mapping dataset from GCS..."
gsutil cp "${KEY_MAP_GCS_PATH}" "${KEY_MAP_LOCAL_PATH}"

echo "Running Python worker script to train the multi-head model (V5)..."
export MASTER_DATA_LOCAL_PATH
export FEATURE_SETS_LOCAL_PATH
export RIGHTMOVE_DATA_LOCAL_PATH
export OUTPUT_DIR
export KEY_MAP_LOCAL_PATH
export N_TRIALS

python3 "${SCRIPT_PATH}"

echo "Uploading all training artifacts to GCS..."
gsutil -m cp -r "${OUTPUT_DIR}/*" "${OUTPUT_GCS_DIR}/"

echo "Model training finished. OOF predictions and models are in ${OUTPUT_GCS_DIR}"