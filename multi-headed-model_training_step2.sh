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
OUTPUT_GCS_DIR="gs://${GCS_BUCKET}/model_training_valuation/run_${TIMESTAMP}"
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

echo "Updating system packages and installing L4 GPU dependencies..."
sudo apt-get update
# Install a modern NVIDIA driver compatible with the L4 GPU's Ada Lovelace architecture
sudo apt-get install -y libgomp1 nvidia-driver-535

echo "Installing required Python packages for deep learning..."
pip install --upgrade pip
# IMPORTANT: Force re-install PyTorch compiled for CUDA 12.1 to overwrite any old, cached versions.
pip install --force-reinstall --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
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
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, r2_score
import lightgbm as lgb
import torch.nn.functional as F
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


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.projection = nn.Linear(input_dim, embed_dim)
        self.dropout = nn.Dropout(p=dropout)
        
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-np.log(10000.0) / embed_dim))
        self.register_buffer('div_term', div_term)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projection(x)
        pe = torch.zeros_like(x)
        pe[:, 0::2] = torch.sin(x[:, 0::2] * self.div_term)
        pe[:, 1::2] = torch.cos(x[:, 1::2] * self.div_term)
        return self.dropout(x + pe)

class SpatialFusion(nn.Module):
    def __init__(self, msoa_embed_dim, pos_encode_dim, spatial_head_dim, fusion_output_dim, num_heads=4, dropout=0.2):
        super().__init__()
        self.fusion_output_dim = fusion_output_dim
        self.msoa_proj = nn.Linear(msoa_embed_dim, fusion_output_dim)
        self.pos_proj = nn.Linear(pos_encode_dim, fusion_output_dim)
        self.spatial_head_proj = nn.Linear(spatial_head_dim, fusion_output_dim)
        
        self.attention = nn.MultiheadAttention(embed_dim=fusion_output_dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(fusion_output_dim)
        
    def forward(self, msoa_embed, pos_encoding, spatial_head_output):
        msoa_p = self.msoa_proj(msoa_embed).unsqueeze(1)
        pos_p = self.pos_proj(pos_encoding).unsqueeze(1)
        spatial_p = self.spatial_head_proj(spatial_head_output).unsqueeze(1)
        
        stacked_inputs = torch.cat([msoa_p, pos_p, spatial_p], dim=1)
        attn_output, _ = self.attention(stacked_inputs, stacked_inputs, stacked_inputs)
        fused_vector = self.norm(attn_output.mean(dim=1))
        return fused_vector


class FusionModel(nn.Module):
    def __init__(self, feature_configs, msoa_cardinality, head_params, fusion_embed_dim, msoa_embedding_dim=16, fusion_dropout_rate=0.5, use_enhanced_tcn=True, fusion_input_cap=2048):
        super().__init__()
        
        self.specialist_heads = nn.ModuleDict()
        self.base_head = None
        tcn_class = EnhancedTCNHead if use_enhanced_tcn else TCNHead
        common_output_dim = fusion_embed_dim

        # Define the hierarchical structure
        self.HEAD_GROUPS = {
            'spatio_temporal': [f'head_F_spatio_temporal_{p_type}' for p_type in ['D', 'S', 'T', 'F']],
            'static_context': ['head_C_census'],
            'visual': ['head_B_aesthetic', 'head_G_gemini_quantitative']
        }
        self.INDIVIDUAL_HEADS = ['head_A_dna', 'head_E_temporal', 'head_AVM']

        # --- 1. Create all head modules ---
        for head_name, config in feature_configs.items():
            if head_name not in head_params: continue
            input_dim, h_params = config['input_dim'], head_params[head_name]
            dropout, hidden_dim = h_params['dropout_rate'], h_params['hidden_dim']

            if head_name == 'head_base':
                self.base_head = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, common_output_dim))
            elif head_name == 'head_A_dna':
                self.specialist_heads[head_name] = TabNetHead(input_dim, common_output_dim, n_d=32, n_a=32, n_steps=5, gamma=1.3, n_independent=2, n_shared=2, dropout=dropout)
            elif 'temporal' in head_name:
                self.specialist_heads[head_name] = tcn_class(input_dim, config['n_features_per_timestep'], config['n_timesteps'], [hidden_dim, common_output_dim], dropout=dropout)
            else:
                self.specialist_heads[head_name] = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, common_output_dim))

        # --- 1.5 Create Spatial Fusion Components ---
        pos_encoding_dim = 32
        self.positional_encoder = SinusoidalPositionalEncoding(input_dim=2, embed_dim=pos_encoding_dim)
        self.spatial_fusion_block = SpatialFusion(
            msoa_embed_dim=msoa_embedding_dim,
            pos_encode_dim=pos_encoding_dim,
            spatial_head_dim=common_output_dim,
            fusion_output_dim=64
        )

        # --- 2. Create Intra-Group Attention Layers (Stage 1) ---
        self.group_attentions = nn.ModuleDict({
            group_name: nn.MultiheadAttention(embed_dim=common_output_dim, num_heads=4, batch_first=True, dropout=head_params['head_base']['dropout_rate'])
            for group_name in self.HEAD_GROUPS
        })

        # --- 3. Create Inter-Group Attention Layer (Stage 2) ---
        num_top_level_concepts = len(self.HEAD_GROUPS) + len(self.INDIVIDUAL_HEADS)
        self.final_attention = nn.MultiheadAttention(embed_dim=common_output_dim, num_heads=8, batch_first=True, dropout=head_params['head_base']['dropout_rate'])

        # --- 4. Create Final Fusion Components ---
        self.msoa_embedding = nn.Embedding(msoa_cardinality, msoa_embedding_dim)
        final_specialist_dim = num_top_level_concepts * common_output_dim

        self.gating_network = nn.Sequential(nn.Linear(final_specialist_dim, 1), nn.Sigmoid())
        
        fusion_input_dim = final_specialist_dim + common_output_dim + self.spatial_fusion_block.fusion_output_dim
        print(f"--- Hierarchical Fusion: Final MLP input dimension: {fusion_input_dim} ---")
        
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, 1024), nn.ReLU(), nn.Dropout(fusion_dropout_rate),
            nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(fusion_dropout_rate)
        )
        self.evidential_head = nn.Linear(512, 4) # Outputs: gamma, nu, alpha, beta

    def forward(self, x):
        # --- Stage 0: Process all heads to get initial embeddings ---
        head_outputs = {name: module(x[name]) for name, module in self.specialist_heads.items() if name in x}

        # --- Stage 0.5: Dedicated Spatial Fusion ---
        msoa_embed = self.msoa_embedding(x['msoa_id'])
        pos_encoding = self.positional_encoder(x['lat_lon'])
        spatial_engineered_output = head_outputs['head_spatial_engineered']
        hyperlocal_context_vector = self.spatial_fusion_block(msoa_embed, pos_encoding, spatial_engineered_output)

        # --- Stage 1: Intra-Group Attention ---
        group_vectors = []
        for group_name, head_names in self.HEAD_GROUPS.items():
            group_head_outputs = [head_outputs[h_name] for h_name in head_names if h_name in head_outputs]
            if not group_head_outputs: continue
            stacked_group = torch.stack(group_head_outputs, dim=1)
            attn_output, _ = self.group_attentions[group_name](stacked_group, stacked_group, stacked_group)
            group_vectors.append(attn_output.mean(dim=1))

        # --- Stage 2: Inter-Group Attention ---
        top_level_concepts = group_vectors + [head_outputs[h_name] for h_name in self.INDIVIDUAL_HEADS if h_name in head_outputs]
        stacked_top_level = torch.stack(top_level_concepts, dim=1)
        final_attn_output, _ = self.final_attention(stacked_top_level, stacked_top_level, stacked_top_level)
        final_specialist_representation = final_attn_output.reshape(final_attn_output.size(0), -1)

        # --- Stage 3: Final Gated Fusion ---
        base_output = self.base_head(x['head_base'])
        gate_weight = self.gating_network(final_specialist_representation)
        gated_base_output = base_output * gate_weight
        
        final_fused = torch.cat([final_specialist_representation, gated_base_output, hyperlocal_context_vector], dim=1)
        
        final_representation = self.fusion_mlp(final_fused)
        
        # --- Stage 4: Evidential Output Layer ---
        evidential_output = self.evidential_head(final_representation)
        gamma, nu, alpha, beta = torch.split(evidential_output, 1, dim=-1)
        
        # Apply activations to ensure valid NIG parameters
        nu = F.softplus(nu) + 1e-6
        alpha = F.softplus(alpha) + 1.0 # alpha > 1
        beta = F.softplus(beta) + 1e-6
        
        return torch.cat([gamma, nu, alpha, beta], dim=-1)

class EvidentialRegressionLoss(nn.Module):
    def __init__(self, coeff=1.0):
        super(EvidentialRegressionLoss, self).__init__()
        self.coeff = coeff

    def forward(self, pred, target):
        gamma, nu, alpha, beta = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        
        # Negative Log-Likelihood of the Normal-Inverse-Gamma distribution
        two_beta_lambda = 2 * beta * (1 + nu)
        nll = 0.5 * torch.log(np.pi / nu) \
            - alpha * torch.log(two_beta_lambda) \
            + (alpha + 0.5) * torch.log(nu * (target - gamma)**2 + two_beta_lambda) \
            + torch.lgamma(alpha) \
            - torch.lgamma(alpha + 0.5)
        
        # Regularizer to prevent the model from predicting zero evidence
        error = torch.abs(target - gamma)
        reg = error * (2 * nu + alpha)
        
        loss = nll + self.coeff * reg
        return loss.mean()

def calculate_evidential_uncertainty(nig_params):
    """Calculates aleatoric and epistemic uncertainty from NIG parameters."""
    gamma, nu, alpha, beta = nig_params[:, 0], nig_params[:, 1], nig_params[:, 2], nig_params[:, 3]
    
    # Ensure alpha > 1 for variance calculation
    alpha_safe = torch.clamp(alpha, min=1.0001)
    
    aleatoric = beta / (alpha_safe - 1)
    epistemic = beta / (nu * (alpha_safe - 1))
    
    return aleatoric, epistemic


# --- New TabNet Implementation ---
# Ghost Batch Normalization
class GBN(nn.Module):
    def __init__(self, input_dim, virtual_batch_size=128, momentum=0.01):
        super(GBN, self).__init__()
        self.input_dim = input_dim
        self.virtual_batch_size = virtual_batch_size
        self.bn = nn.BatchNorm1d(self.input_dim, momentum=momentum)

    def forward(self, x):
        chunks = x.chunk(int(np.ceil(x.shape[0] / self.virtual_batch_size)), 0)
        res = [self.bn(chunk) for chunk in chunks]
        return torch.cat(res, dim=0)

def sparsemax(z, dim=-1):
    # Sort z in descending order
    z_sorted, _ = torch.sort(z, dim=dim, descending=True)
    # Calculate the cumulative sum of sorted z
    z_cumsum = torch.cumsum(z_sorted, dim=dim)
    # Create a range tensor [1, 2, ..., N]
    k = torch.arange(1, z.size(dim) + 1, device=z.device).view(1, -1)
    # Find the largest k such that 1 + k * z_k > sum(z_i for i=1 to k)
    z_check = 1 + k * z_sorted > z_cumsum
    # Get the k_max (the number of non-zero probabilities)
    k_max = torch.sum(z_check, dim=dim, keepdim=True)
    # Get the threshold tau
    k_max_values = k_max.long() - 1
    tau = (torch.gather(z_cumsum, dim, k_max_values) - 1) / k_max.float()
    # Apply the sparsemax transformation
    return torch.max(torch.zeros_like(z), z - tau)

# Attentive Transformer for feature selection
class AttentiveTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, momentum=0.02, virtual_batch_size=128):
        super(AttentiveTransformer, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim, bias=False)
        self.bn = GBN(output_dim, virtual_batch_size=virtual_batch_size, momentum=momentum)

    def forward(self, a, priors):
        a = self.fc(a)
        a = self.bn(a)
        mask = a * priors
        return sparsemax(mask, dim=-1)

# Feature Transformer for processing selected features
class FeatureTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, n_shared, n_independent, momentum=0.02, virtual_batch_size=128):
        super(FeatureTransformer, self).__init__()
        self.n_independent = n_independent
        self.shared_layers = nn.ModuleList()
        self.independent_layers = nn.ModuleList()

        if n_shared > 0:
            shared_galu = nn.Linear(input_dim, 2 * output_dim, bias=False)
            self.shared_layers.append(shared_galu)
            self.shared_layers.append(GBN(2 * output_dim, virtual_batch_size, momentum))

        for _ in range(self.n_independent):
            ind_galu = nn.Linear(input_dim, 2 * output_dim, bias=False)
            self.independent_layers.append(nn.ModuleList([ind_galu, GBN(2 * output_dim, virtual_batch_size, momentum)]))

    def forward(self, x):
        shared_output = x
        if self.shared_layers:
            shared_output = self.shared_layers[0](shared_output)
            shared_output = self.shared_layers[1](shared_output)
            shared_output = F.glu(shared_output)
        
        outputs = []
        for i in range(self.n_independent):
            ind_out = self.independent_layers[i][0](x)
            ind_out = self.independent_layers[i][1](ind_out)
            ind_out = F.glu(ind_out)
            outputs.append(ind_out)
        
        # Aggregate shared and independent outputs
        return torch.stack([shared_output] + outputs, dim=1)

# Main TabNet Head Module
class TabNetHead(nn.Module):
    def __init__(self, input_dim, output_dim, n_d=32, n_a=32, n_steps=5, gamma=1.3, n_independent=2, n_shared=2, dropout=0.2):
        super(TabNetHead, self).__init__()
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.dropout = dropout

        self.initial_bn = nn.BatchNorm1d(input_dim, momentum=0.01)
        self.feature_transformer = FeatureTransformer(input_dim, n_d + n_a, n_shared, n_independent)
        self.attentive_transformers = nn.ModuleList()
        for _ in range(n_steps):
            self.attentive_transformers.append(AttentiveTransformer(n_a, input_dim))
        self.final_map = nn.Linear(n_d, output_dim)

    def forward(self, x):
        x = self.initial_bn(x)
        priors = torch.ones(x.shape).to(x.device)
        total_output = 0.
        
        # Split initial features into decision and attention parts
        x_transformed = self.feature_transformer(x)
        x_a = x_transformed[:, :, self.n_d:]
        
        for step in range(self.n_steps):
            mask = self.attentive_transformers[step](x_a.sum(1), priors)
            priors = priors * (self.gamma - mask)
            
            # Apply mask to original features and transform
            masked_x = mask * x
            x_step_transformed = self.feature_transformer(masked_x)
            
            # Get decision part for this step
            d_step = F.relu(x_step_transformed[:, :, :self.n_d].sum(1))
            total_output += d_step
            
            # Update attention part for next step
            x_a = x_step_transformed[:, :, self.n_d:]
            
        # Final projection to common output dim
        return self.final_map(total_output)


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

def train_and_apply_pca(df, cols, latent_dim=32):
    """Applies PCA to the given columns and returns the compressed features."""
    print(f"    - Applying PCA for {len(cols)} features -> {latent_dim} latent dims...")
    
    if not cols:
        print("    - No columns to process for PCA. Returning empty DataFrame.")
        return pd.DataFrame(index=df.index)

    # 1. Prepare data
    head_data = df[cols].copy().apply(pd.to_numeric, errors='coerce').fillna(0)
    
    # 2. Scale data before PCA
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(head_data)
    
    # 3. Fit and transform with PCA
    pca = PCA(n_components=latent_dim, random_state=42)
    encoded_data = pca.fit_transform(data_scaled)
    
    print(f"    - PCA complete. Explained variance: {np.sum(pca.explained_variance_ratio_):.4f}. Output shape: {encoded_data.shape}")
    
    # Return as a pandas DataFrame
    return pd.DataFrame(encoded_data, index=df.index)


def train_one_epoch(model, loader, optimizer, loss_fn):
    model.train(); total_loss = 0
    for batch in loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        optimizer.zero_grad()
        evidential_preds = model(batch)
        loss = loss_fn(evidential_preds, batch['target_price'].squeeze(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader):
    model.eval(); all_preds = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            preds = model(batch)
            all_preds.append(preds.cpu())
    return torch.cat(all_preds)

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
    hpi_pattern = re.compile(r"^(HPI_Adjusted_Median_Price|Sale_Count)_(D|S|T|F)$")

    for col in tcn_cols:
        year, feature_stem = None, None
        match_spatio = spatio_pattern.match(col)
        match_temporal = temporal_pattern.match(col)
        match_hpi = hpi_pattern.match(col)

        if match_spatio:
            prefix, year, suffix = match_spatio.groups(); feature_stem = f"{prefix}_{suffix}"
        elif match_temporal:
            year, feature_stem = match_temporal.groups()
        elif match_hpi:
            year, _ = "all", match_hpi.groups(); feature_stem = col
        
        if feature_stem:
            if feature_stem not in parsed_cols: parsed_cols[feature_stem] = {}
            if year != "all": parsed_cols[feature_stem][year] = col
            else: parsed_cols[feature_stem]['all_years_col'] = col

    feature_stems = sorted(list(parsed_cols.keys()))
    all_years = sorted(list(set(y for f in parsed_cols.values() for y in f.keys() if y != 'all_years_col')))
    n_timesteps = len(all_years) if all_years else 1
    n_features = len(feature_stems) if feature_stems else 1

    print(f"  - TCN Prep: Found {n_features} feature types across {n_timesteps} years.")

    tcn_array = np.zeros((len(df), n_timesteps, n_features))
    for i, stem in enumerate(feature_stems):
        if 'all_years_col' in parsed_cols[stem]:
            col_name = parsed_cols[stem]['all_years_col']
            if col_name in df.columns:
                tcn_array[:, :, i] = df[col_name].fillna(0).values[:, np.newaxis]
        else:
            for j, year in enumerate(all_years):
                col_name = parsed_cols[stem].get(year)
                if col_name and col_name in df.columns:
                    tcn_array[:, j, i] = df[col_name].fillna(0).values

    tcn_flat = tcn_array.reshape(len(df), -1)
    scaler = QuantileTransformer(output_distribution='normal', n_quantiles=max(min(len(df) // 10, 100), 10), random_state=42)
    tcn_scaled = scaler.fit_transform(tcn_flat)
    
    config = {'n_timesteps': n_timesteps, 'n_features_per_timestep': n_features}
    
    return pd.DataFrame(tcn_scaled, index=df.index), config

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


def objective(trial, data_for_model, y_price_for_training, feature_configs, msoa_cardinality, y_price_std, y_price_mean, df_for_eval):
    """
    The objective function for Optuna to optimize.
    It defines hyperparameters, trains the model K-Fold, and returns the validation score.
    """
    # --- 1. Define Hyperparameter Search Space ---
    print(f"\n--- Starting Trial {trial.number} ---")
    
    # Define head groups for parameter tuning
    LARGE_HEADS = ['head_A_dna', 'head_C_census', 'head_E_temporal', 'head_F_spatio_temporal_D', 'head_F_spatio_temporal_S', 'head_F_spatio_temporal_T', 'head_F_spatio_temporal_F']
    SMALL_HEADS = ['head_B_aesthetic', 'head_G_gemini_quantitative', 'head_spatial_engineered', 'head_AVM']

    # Define search space for grouped heads and other parameters
    trial_params = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
        'evidential_reg_coeff': trial.suggest_float('evidential_reg_coeff', 0.01, 0.5),
        'fusion_dropout_rate': trial.suggest_float('fusion_dropout_rate', 0.2, 0.6),
        'msoa_embedding_dim': trial.suggest_categorical('msoa_embedding_dim', [8, 16, 32]),
        'batch_size': trial.suggest_categorical('batch_size', [128, 256, 512]),
        'fusion_embed_dim': trial.suggest_categorical('fusion_embed_dim', [128, 256]),
        'large_head_hidden_dim': trial.suggest_categorical('large_head_hidden_dim', [512, 768]),
        'large_head_dropout': trial.suggest_float('large_head_dropout', 0.2, 0.5),
        'small_head_hidden_dim': trial.suggest_categorical('small_head_hidden_dim', [128, 256]),
        'small_head_dropout': trial.suggest_float('small_head_dropout', 0.1, 0.4),
    }

    # Build the structured head_params dictionary for the model
    head_params = {}
    for head_name in list(feature_configs.keys()):
        if head_name in LARGE_HEADS:
            head_params[head_name] = {'hidden_dim': trial_params['large_head_hidden_dim'], 'dropout_rate': trial_params['large_head_dropout']}
        elif head_name in SMALL_HEADS:
            head_params[head_name] = {'hidden_dim': trial_params['small_head_hidden_dim'], 'dropout_rate': trial_params['small_head_dropout']}
    # Add base head params (we'll group it with large heads)
    head_params['head_base'] = {'hidden_dim': trial_params['large_head_hidden_dim'], 'dropout_rate': trial_params['large_head_dropout']}
    
    # --- 2. K-Fold Cross-Validation Training Loop ---
    kf = KFold(n_splits=NUM_FOLDS_OPTUNA, shuffle=True, random_state=42)
    oof_evidential_preds = np.zeros((len(df_for_eval), 4)) # gamma, nu, alpha, beta
    fold_maes = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(df_for_eval)):
        print(f"  ======== OPTUNA FOLD {fold+1}/{NUM_FOLDS_OPTUNA} (Trial {trial.number}) ========")
        train_data = {k: v.iloc[train_idx] for k, v in data_for_model.items()}
        val_data = {k: v.iloc[val_idx] for k, v in data_for_model.items()}
        
        # NOTE: y_uncertainty is no longer needed for WisteriaDataset
        train_dataset = WisteriaDataset(train_data, y_price_for_training.iloc[train_idx], y_price_for_training.iloc[train_idx])
        val_dataset = WisteriaDataset(val_data, y_price_for_training.iloc[val_idx], y_price_for_training.iloc[val_idx])
        train_loader = DataLoader(train_dataset, batch_size=trial_params['batch_size'], shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=trial_params['batch_size'] * 2, shuffle=False, num_workers=2, pin_memory=True)
        
        model = FusionModel(
            feature_configs=feature_configs,
            msoa_cardinality=msoa_cardinality,
            head_params=head_params,
            fusion_embed_dim=trial_params['fusion_embed_dim'],
            msoa_embedding_dim=trial_params['msoa_embedding_dim'],
            fusion_dropout_rate=trial_params['fusion_dropout_rate'],
            use_enhanced_tcn=True,
            fusion_input_cap=2048
        ).to(DEVICE)
        
        optimizer = optim.AdamW(model.parameters(), lr=trial_params['learning_rate'], weight_decay=trial_params['weight_decay'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=5)
        loss_fn = EvidentialRegressionLoss(coeff=trial_params['evidential_reg_coeff'])
        
        best_loss = float('inf')
        for epoch in range(EPOCHS):
            train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn)
            scheduler.step(train_loss)
            if train_loss < best_loss:
                best_loss = train_loss
        
        evidential_preds = evaluate(model, val_loader)
        oof_evidential_preds[val_idx] = evidential_preds.numpy()

        # --- 3. Evaluate and Report ---
        # The point prediction is the 'gamma' parameter (index 0)
        point_preds_scaled = evidential_preds.numpy()[:, 0]
        point_preds_unscaled = np.expm1(point_preds_scaled * y_price_std + y_price_mean)
        true_values = df_for_eval.iloc[val_idx]['most_recent_sale_price']
        
        fold_mae = mean_absolute_error(true_values, point_preds_unscaled)
        fold_maes.append(fold_mae)
        print(f"    Fold {fold+1} MAE: £{fold_mae:,.2f}")

        # Optuna Pruning: Stop unpromising trials early
        trial.report(fold_mae, fold)
        if trial.should_prune():
            print("  -- Trial pruned --")
            raise optuna.exceptions.TrialPruned()

        del model, optimizer, scheduler; gc.collect(); torch.cuda.empty_cache()

    avg_mae = np.mean(fold_maes)
    sample_preds = oof_evidential_preds[:5, :]
    trial.set_user_attr("sample_predictions", sample_preds.tolist())

    print(f"--- Trial {trial.number} Finished. Average MAE: £{avg_mae:,.2f} ---")
    print(f"  - Sample Evidential Outputs (gamma, nu, alpha, beta):")
    print(sample_preds)
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

def engineer_temporal_summary_features(df, feature_sets):
    """
    Performs longitudinal compression on temporal features.
    For each temporal feature stem, it calculates trend, mean, and std dev over time,
    creating new static features.
    """
    print("\n--- STAGE 4.1.5: Temporal Summary Feature Engineering (Longitudinal Compression) ---")
    
    # 1. Identify all temporal columns from the feature sets
    temporal_cols = []
    for head_name, cols in feature_sets.items():
        if 'temporal' in head_name:
            temporal_cols.extend(cols)
    temporal_cols = sorted(list(set([c for c in temporal_cols if c in df.columns])))
    
    if not temporal_cols:
        print("  - No temporal columns found to engineer. Skipping.")
        return df, []

    # 2. Group columns by feature stem (e.g., 'job_seekers')
    feature_stems = {}
    # Pattern to capture year and the rest of the stem
    pattern = re.compile(r"(\d{4})_(.+)") 
    # More complex spatio-temporal pattern
    spatio_pattern = re.compile(r"^(.*?)_(\d{4})_(.*)$")

    for col in temporal_cols:
        match = pattern.match(col)
        if match:
            year, stem = match.groups()
            if stem not in feature_stems: feature_stems[stem] = []
            feature_stems[stem].append({'year': int(year), 'col': col})
        else:
            spatio_match = spatio_pattern.match(col)
            if spatio_match:
                prefix, year, suffix = spatio_match.groups()
                stem = f"{prefix}_{suffix}"
                if stem not in feature_stems: feature_stems[stem] = []
                feature_stems[stem].append({'year': int(year), 'col': col})
    
    # 3. For each stem, calculate summary stats
    new_summary_features = pd.DataFrame(index=df.index)
    new_feature_names = []

    print(f"  - Found {len(feature_stems)} unique temporal feature stems to process.")
    for i, (stem, year_cols) in enumerate(feature_stems.items()):
        if (i+1) % 50 == 0: print(f"    - Processing stem {i+1}/{len(feature_stems)}: {stem}")
        if len(year_cols) < 3: continue # Need at least 3 data points to calculate a meaningful trend/std

        # Create a temporary dataframe for this feature's history
        history_df = pd.DataFrame({
            info['year']: df[info['col']] for info in year_cols
        }).sort_index(axis=1)
        
        years = np.array(history_df.columns).reshape(-1, 1)
        values = history_df.values
        
        # Calculate Trend (slope of linear regression)
        # We use np.polyfit which is robust to NaNs in the values array
        trends = np.apply_along_axis(lambda y: np.polyfit(years.flatten(), y, 1)[0] if np.isfinite(y).sum() > 1 else 0, 1, values)
        
        # Calculate other stats
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean_vals = np.nanmean(values, axis=1)
            std_vals = np.nanstd(values, axis=1)

        # Add to our new features dataframe
        trend_col_name = f"ts_summary_trend_{stem}"
        mean_col_name = f"ts_summary_mean_{stem}"
        std_col_name = f"ts_summary_std_{stem}"
        
        new_summary_features[trend_col_name] = trends
        new_summary_features[mean_col_name] = mean_vals
        new_summary_features[std_col_name] = std_vals
        
        new_feature_names.extend([trend_col_name, mean_col_name, std_col_name])

    # 4. Concatenate new features to the main dataframe
    df = pd.concat([df, new_summary_features.fillna(0)], axis=1)
    print(f"  - Successfully created {len(new_feature_names)} new temporal summary features.")
    
    return df, new_feature_names


def main():
    warnings.filterwarnings('ignore')
    print(f"Using device: {DEVICE}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Configuration for Per-Head Feature Selection ---
    N_TOP_FEATURES_PER_HEAD = {
        "head_A_dna": 225, # Note: This will be ignored due to TabNet using all features
        "head_B_aesthetic": 50,
        "head_C_census": 225,
        "head_G_gemini_quantitative": 100,
        "DEFAULT": 125  # Default for any other heads
    }
    UNIMPORTANT_LATENT_DIM = 48 # Latent dim for unimportant features via Autoencoder
    PCA_LATENT_DIM = 64 # Latent dim for unimportant census features via PCA

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

    # --- STAGE 1.5: Refactor Feature Sets for New AVM Head ---
    print("\n--- STAGE 1.5: Re-allocating AVM features to dedicated head ---")
    AVM_FEATURES = [
        'num__bricksandlogic_estimated_price_gbp__numeric_extracted_or_original_text__bnl',
        'num__homipi_estimated_price_range_lower_bound_gbp_hm',
        'num__homipi_estimated_price_range_upper_bound_gbp_hm',
        'num__mouseprice_estimated_value_gbp__numeric_string__mp',
        'num__last_sold_price_gbp_hm',
        'num__StreetScan_past_sales_avg_price_gbp_for_All_Properties_ss',
        'num__StreetScan_past_sales_avg_price_gbp_for_Detached_ss',
        'num__StreetScan_past_sales_avg_price_gbp_for_Flats_ss',
        'num__StreetScan_past_sales_avg_price_gbp_for_New_Builds_ss',
        'num__StreetScan_past_sales_avg_price_gbp_for_Semi_Detached_ss',
        'num__StreetScan_past_sales_avg_price_gbp_for_Terraced_ss',
        'num__INT_avg_estimated_price',
        'num__INT_std_dev_estimated_price',
        'num__INT_bnl_price_per_sqft',
        'num__INT_mp_price_per_sqm',
        'num__INT_hm_price_range_spread_gbp',
        'num__INT_hm_price_vs_last_sold_ratio',
        'num__INT_hm_price_vs_ss_area_avg_price_ratio',
        'num__homipi_value_change_percentage_hm',
        'cat__mouseprice_estimated_rental_value_text__string_or_empty__mp_infrequent_sklearn',
        'cat__mouseprice_estimated_rental_value_text__string_or_empty__mp_nan'
    ]
    
    # Create the new head
    feature_sets['head_AVM'] = AVM_FEATURES
    print(f"  - Created 'head_AVM' with {len(AVM_FEATURES)} features.")

    # Remove these features from all other heads to prevent duplication
    avm_feature_set = set(AVM_FEATURES)
    for head_name, cols in feature_sets.items():
        if head_name == 'head_AVM':
            continue
        original_count = len(feature_sets[head_name])
        feature_sets[head_name] = [col for col in cols if col not in avm_feature_set]
        removed_count = original_count - len(feature_sets[head_name])
        if removed_count > 0:
            print(f"  - Removed {removed_count} AVM features from '{head_name}'.")

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

    # --- NEW STAGE 4.1.5: Temporal Summary Feature Engineering (Longitudinal Compression) ---
    df, new_temporal_summary_names = engineer_temporal_summary_features(df, feature_sets)
    if 'head_C_census' in feature_sets:
        feature_sets['head_C_census'].extend(new_temporal_summary_names)
        print(f"  - Added {len(new_temporal_summary_names)} summary features to 'head_C_census'.")
    else:
        # Fallback in case the head doesn't exist for some reason
        feature_sets['head_C_census'] = new_temporal_summary_names

    # --- NEW STAGE 4.2: Autoencoder Feature Engineering (BEFORE Feature Selection) ---
    print("\n--- STAGE 4.2: Engineering Features with Specialized Autoencoders ---")
    
    spatial_prefixes = {'compass': 'compass_', 'atlas': 'atlas_', 'microscope': 'microscope_'}
    spatial_cols_to_drop = []
    engineered_spatial_cols = []

    # 1. Encode STATIC 'compass' features
    static_compass_cols = [c for c in df.columns if c.startswith(spatial_prefixes['compass']) and not re.search(r'_\d{4}_', c)]
    if static_compass_cols:
        print(f"  - Training 'compass' AE on {len(static_compass_cols)} STATIC features...")
        encoded_compass_df = train_and_apply_autoencoder(df, static_compass_cols, latent_dim=80)
        encoded_compass_df.columns = [f"spatial_eng_compass_ae_{i}" for i in range(encoded_compass_df.shape[1])]
        df = pd.concat([df, encoded_compass_df], axis=1)
        engineered_spatial_cols.extend(encoded_compass_df.columns)
        spatial_cols_to_drop.extend(static_compass_cols)

    # 2. Identify raw 'atlas' and 'microscope' features to KEEP
    for name in ['atlas', 'microscope']:
        prefix = spatial_prefixes[name]
        static_raw_cols_to_keep = [c for c in df.columns if c.startswith(prefix) and not re.search(r'_\d{4}_', c)]
        engineered_spatial_cols.extend(static_raw_cols_to_keep)
    
    print(f"  - Created/identified a total of {len(engineered_spatial_cols)} engineered spatial features to be used in 'head_spatial_engineered'.")


    # --- STAGE 4.5: [REMOVED] Global Feature Selection is now handled by the per-head logic in Stage 5 ---
    
    # Now we can safely drop the raw spatial columns that were encoded or are no longer needed
    print(f"\n--- Dropping {len(spatial_cols_to_drop)} raw spatial columns that have been engineered...")
    df.drop(columns=spatial_cols_to_drop, inplace=True, errors='ignore')
    print(f"  - DataFrame now has {df.shape[1]} total columns before entering head-specific processing.")

    # --- STAGE 5: Filter, Validate, and Prepare Data for Model ---
    print("\n--- STAGE 5: Filtering, Validating and Preparing Data for Sub-Head Architecture ---")
    Q1, Q3 = df['most_recent_sale_price'].quantile(0.01), df['most_recent_sale_price'].quantile(0.99)
    df = df[(df['most_recent_sale_price'] >= Q1) & (df['most_recent_sale_price'] <= Q3)].copy().reset_index(drop=True)
    print(f"  - Dataset after outlier removal: {len(df)} properties")

    y_price_log = np.log1p(df['most_recent_sale_price'])
    y_price_mean, y_price_std = y_price_log.mean(), y_price_log.std()
    y_price_for_training = pd.Series((y_price_log - y_price_mean) / y_price_std, index=df.index)

    # --- HYBRID CONTEXT STRATEGY ---
    # --- DEDICATED head_base ARCHITECTURE ---
    print("\n--- Creating dedicated `head_base` and specialist heads ---")
    LAT_LON_COLS = ['latitude', 'longitude']
    UNIVERSAL_PREDICTORS = [
        'num__floor_area_sqm_from_homipi__numeric_or_empty__hm', 'num__number_of_bedrooms_from_homipi__numeric_or_empty__hm', 'num__number_of_reception_rooms_from_homipi__numeric_or_empty__hm',
        'num__number_of_storeys_from_homipi__numeric_or_empty__hm', 'num__property_construction_year_from_homipi__YYYY_or_2025NewBuildOrError__hm', 'num__property_main_type_encoded__1Flat_hm',
        'num__property_sub_type_code_from_homipi__4Detached_hm', 'num__property_tenure_code_from_homipi__2Freehold_hm', 'primary_MainGarden_area_sqm', 'primary_MainGarage_area_sqm',
        'primary_MainDrivewayParking_area_sqm', 'other_OtherBathroomsWCs_count', 'num_rooms_identified_in_step5', 'atlas_cluster_id',
        'num__StreetScan_average_household_income_gbp_ss', 'num__StreetScan_deprivation_rank_for_Overall_Multiple_Deprivation_ss', 'num__StreetScan_deprivation_rank_for_Crime_ss',
        'num__chimnie_local_area_safety_index_score_ch', 'num__primary_or_first_type_school_1_distance__numeric_extracted_or_original_text__bnl', 'num__primary_or_first_type_school_1_ofsted_rating_encoded__numeric_extracted_or_original_text__bnl',
        'num__train_station_1_distance__numeric_extracted_or_original_text__bnl', 'num__homipi_nearest_hospital_distance_or_number_1_hm', 'LSOA_Property_Density', 'ah4gpas', 'ah4no2',
        'ah4pm10', 'compass_mean_LSOA_Price_Growth_5Year_n100', 'num__last_sold_date_year__YYYY__hm', 'HPI_Adjusted_Median_Price_D',
        'HPI_Adjusted_Median_Price_F', 'LSOA_MedPrice_Recent', 'LSOA_Price_Growth_5Year', 'LSOA_Annual_Transaction_Rate', 'Sale_Count_D', 'num__current_epc_value_extracted_by_initial_homipi_script_hm_numeric',
        'num__potential_epc_value_extracted_by_initial_homipi_script_hm_numeric', 'primary_MainKitchen_renovation_score', 'primary_MainBathroom_renovation_score', 'primary_MainExteriorFront_renovation_score',
        'primary_MainLivingArea_renovation_score', 'primary_MainKitchen_num_features', 'primary_MainBathroom_num_flaws', 'Composite_CoreLiving_Strength', 'Opportunity_Good_Bones_Score',
        'Risk_Cosmetic_Burden_X_Age', 'Thesis_FutureProofingScore', 'avg_persona_rating_overall'
    ]
    universal_cols_present = [col for col in UNIVERSAL_PREDICTORS if col in df.columns]
    
    data_for_model = {}
    feature_configs = {}

    # 0. Create dedicated lat/lon input for the spatial fusion block
    if all(c in df.columns for c in LAT_LON_COLS):
        print(f"  - Creating dedicated 'lat_lon' input for spatial fusion.")
        data_for_model['lat_lon'] = df[LAT_LON_COLS].copy().apply(pd.to_numeric, errors='coerce').fillna(0)
    else:
        raise ValueError(f"FATAL: Latitude/Longitude columns not found in dataframe. Needed for spatial fusion.")

    # 1. Create the dedicated `head_base`
    print(f"  - Assigning {len(universal_cols_present)} raw universal predictors to `head_base`.")
    base_df_raw = df[universal_cols_present].copy().apply(pd.to_numeric, errors='coerce').fillna(0)
    scaler_base = QuantileTransformer(output_distribution='normal', n_quantiles=max(min(len(df) // 10, 100), 10), random_state=42)
    data_for_model['head_base'] = pd.DataFrame(scaler_base.fit_transform(base_df_raw), columns=base_df_raw.columns, index=df.index)
    feature_configs['head_base'] = {'input_dim': len(universal_cols_present)}

    # 2. Create the consolidated `head_spatial_engineered`
    if engineered_spatial_cols:
        print(f"  - Assigning {len(engineered_spatial_cols)} features to `head_spatial_engineered`.")
        spatial_df = df[engineered_spatial_cols].copy().apply(pd.to_numeric, errors='coerce').fillna(0)
        scaler_spatial = QuantileTransformer(output_distribution='normal', n_quantiles=max(min(len(df) // 10, 100), 10), random_state=42)
        data_for_model['head_spatial_engineered'] = pd.DataFrame(scaler_spatial.fit_transform(spatial_df), columns=spatial_df.columns, index=df.index)
        feature_configs['head_spatial_engineered'] = {'input_dim': len(engineered_spatial_cols)}

    # 3. Dynamically process the remaining "specialist" heads
    for head_name, cols in feature_sets.items():
        if head_name in ['unassigned_features', 'head_base', 'head_atlas', 'head_compass', 'head_microscope'] or not cols: continue
        
        available_cols = [c for c in cols if c in df.columns and c not in universal_cols_present]
        if not available_cols: continue
            
        print(f"\n--- Processing data for specialist head: {head_name} ---")

        if 'spatio_temporal' in head_name or 'head_E_temporal' in head_name:
            final_head_data, tcn_config = prepare_tcn_data(df, available_cols, pattern=r"(\d{4})_(.+)")
            data_for_model[head_name] = final_head_data
            feature_configs[head_name] = {**tcn_config, 'input_dim': final_head_data.shape[1]}
        else: # Standard MLP Head
            head_df_raw = df[available_cols].copy().apply(pd.to_numeric, errors='coerce').fillna(0).replace([np.inf, -np.inf], 0)
            
            # --- Tailored Preprocessing Logic ---
            if head_name == 'head_A_dna':
                print("    - Using TabNet strategy: Passing all features directly to the head.")
                final_head_df = head_df_raw
            
            elif head_name == 'head_C_census':
                print("    - Using PCA strategy for census data.")
                N_TOP_PER_HEAD = N_TOP_FEATURES_PER_HEAD.get(head_name, N_TOP_FEATURES_PER_HEAD["DEFAULT"])
                if len(available_cols) > N_TOP_PER_HEAD:
                    lgbm_head_selector = lgb.LGBMRegressor(random_state=42, n_estimators=100, n_jobs=-1, verbose=-1)
                    lgbm_head_selector.fit(head_df_raw, y_price_log)
                    importances = pd.DataFrame({'feature': head_df_raw.columns, 'importance': lgbm_head_selector.feature_importances_}).sort_values('importance', ascending=False)
                    important_cols = importances.head(N_TOP_PER_HEAD)['feature'].tolist()
                    unimportant_cols = importances.tail(len(importances) - N_TOP_PER_HEAD)['feature'].tolist()
                    
                    # Use PCA for the long tail of census features
                    encoded_unimportant_df = train_and_apply_pca(df, unimportant_cols, latent_dim=PCA_LATENT_DIM)
                    encoded_unimportant_df.columns = [f"{head_name}_unimp_pca_{i}" for i in range(encoded_unimportant_df.shape[1])]
                    final_head_df = pd.concat([head_df_raw[important_cols].reset_index(drop=True), encoded_unimportant_df.reset_index(drop=True)], axis=1)
                else:
                    final_head_df = head_df_raw

            else: # Default: Use Autoencoder for other large MLP heads
                print(f"    - Using default Autoencoder strategy for {head_name}.")
                N_TOP_PER_HEAD = N_TOP_FEATURES_PER_HEAD.get(head_name, N_TOP_FEATURES_PER_HEAD["DEFAULT"])
                if len(available_cols) > N_TOP_PER_HEAD:
                    lgbm_head_selector = lgb.LGBMRegressor(random_state=42, n_estimators=100, n_jobs=-1, verbose=-1)
                    lgbm_head_selector.fit(head_df_raw, y_price_log)
                    importances = pd.DataFrame({'feature': head_df_raw.columns, 'importance': lgbm_head_selector.feature_importances_}).sort_values('importance', ascending=False)
                    important_cols = importances.head(N_TOP_PER_HEAD)['feature'].tolist()
                    unimportant_cols = importances.tail(len(importances) - N_TOP_PER_HEAD)['feature'].tolist()
                    encoded_unimportant_df = train_and_apply_autoencoder(df, unimportant_cols, latent_dim=UNIMPORTANT_LATENT_DIM, ae_epochs=50)
                    encoded_unimportant_df.columns = [f"{head_name}_unimp_ae_{i}" for i in range(encoded_unimportant_df.shape[1])]
                    final_head_df = pd.concat([head_df_raw[important_cols].reset_index(drop=True), encoded_unimportant_df.reset_index(drop=True)], axis=1)
                else:
                    final_head_df = head_df_raw
            
            # --- Common Scaling and Finalization Step ---
            scaler = QuantileTransformer(output_distribution='normal', n_quantiles=max(min(len(df) // 10, 100), 10), random_state=42)
            processed_data = pd.DataFrame(scaler.fit_transform(final_head_df), columns=final_head_df.columns, index=df.index)
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

    if 'sample_predictions' in best_trial.user_attrs:
        print("\n  --- Sample Evidential Outputs from Best Trial (OOF) ---")
        print("  Columns: [gamma, nu, alpha, beta]")
        sample_preds = np.array(best_trial.user_attrs['sample_predictions'])
        print(sample_preds)


    # --- STAGE 7: Final Model Training with Best Hyperparameters ---
    print("\n--- STAGE 7: Training Final Model using Best Hyperparameters ---")
    best_params = best_trial.params
    
    # Reconstruct the optimal head_params from the best trial
    LARGE_HEADS = ['head_A_dna', 'head_C_census', 'head_E_temporal', 'head_F_spatio_temporal_D', 'head_F_spatio_temporal_S', 'head_F_spatio_temporal_T', 'head_F_spatio_temporal_F']
    SMALL_HEADS = ['head_B_aesthetic', 'head_G_gemini_quantitative', 'head_spatial_engineered', 'head_AVM']
    best_head_params = {}
    for head_name in list(feature_configs.keys()):
        if head_name in LARGE_HEADS:
            best_head_params[head_name] = {'hidden_dim': best_params['large_head_hidden_dim'], 'dropout_rate': best_params['large_head_dropout']}
        elif head_name in SMALL_HEADS:
            best_head_params[head_name] = {'hidden_dim': best_params['small_head_hidden_dim'], 'dropout_rate': best_params['small_head_dropout']}
    best_head_params['head_base'] = {'hidden_dim': best_params['large_head_hidden_dim'], 'dropout_rate': best_params['large_head_dropout']}

    kf = KFold(n_splits=NUM_FOLDS_FINAL, shuffle=True, random_state=42)
    oof_evidential_preds = np.zeros((len(df), 4)) # gamma, nu, alpha, beta
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
        print(f"\n========== FINAL TRAINING: FOLD {fold+1}/{NUM_FOLDS_FINAL} ==========")
        train_data = {k: v.iloc[train_idx] for k, v in data_for_model.items()}
        val_data = {k: v.iloc[val_idx] for k, v in data_for_model.items()}
        train_dataset = WisteriaDataset(train_data, y_price_for_training.iloc[train_idx], y_price_for_training.iloc[train_idx])
        val_dataset = WisteriaDataset(val_data, y_price_for_training.iloc[val_idx], y_price_for_training.iloc[val_idx])
        train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=best_params['batch_size'] * 2, shuffle=False, num_workers=2, pin_memory=True)
        
        model = FusionModel(
            feature_configs=feature_configs,
            msoa_cardinality=msoa_cardinality,
            head_params=best_head_params,
            fusion_embed_dim=best_params['fusion_embed_dim'],
            msoa_embedding_dim=best_params['msoa_embedding_dim'],
            fusion_dropout_rate=best_params['fusion_dropout_rate'],
            use_enhanced_tcn=True,
            fusion_input_cap=2048
        ).to(DEVICE)
        
        optimizer = optim.AdamW(model.parameters(), lr=best_params['learning_rate'], weight_decay=best_params['weight_decay'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=5)
        loss_fn = EvidentialRegressionLoss(coeff=best_params['evidential_reg_coeff'])

        best_loss = float('inf')
        for epoch in range(EPOCHS + 50): # Train for a bit longer on the final model
            train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn)
            if epoch % 10 == 0: print(f"  Epoch {epoch+1}/{EPOCHS+50}, Train Loss: {train_loss:.4f}")
            scheduler.step(train_loss)
            if train_loss < best_loss:
                best_loss = train_loss
                torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f"model_fold_{fold}_best_params.pt"))
        
        model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, f"model_fold_{fold}_best_params.pt")))
        evidential_preds = evaluate(model, val_loader)
        oof_evidential_preds[val_idx] = evidential_preds.numpy()
        del model, optimizer, scheduler; gc.collect(); torch.cuda.empty_cache()
    
    # --- STAGE 7: Evaluate & Save ---
    print("\n--- STAGE 7: Evaluating Performance and Saving Artifacts ---")
    oof_df = pd.DataFrame(oof_evidential_preds, columns=['pred_gamma', 'pred_nu', 'pred_alpha', 'pred_beta'], index=df.index)
    eval_df = df.join(oof_df)
    
    # Calculate point prediction and uncertainties
    eval_df['predicted_price'] = np.expm1(eval_df['pred_gamma'] * y_price_std + y_price_mean)
    aleatoric_unc, epistemic_unc = calculate_evidential_uncertainty(torch.tensor(eval_df[['pred_gamma', 'pred_nu', 'pred_alpha', 'pred_beta']].values))
    eval_df['aleatoric_uncertainty'] = aleatoric_unc.numpy()
    eval_df['epistemic_uncertainty'] = epistemic_unc.numpy()
    eval_df['total_uncertainty'] = eval_df['aleatoric_uncertainty'] + eval_df['epistemic_uncertainty']

    mae = mean_absolute_error(eval_df['most_recent_sale_price'], eval_df['predicted_price'])
    r2 = r2_score(eval_df['most_recent_sale_price'], eval_df['predicted_price'])
    print(f"\n--- FINAL DEEP LEARNING MODEL PERFORMANCE ---")
    print(f"  - R-squared (R²): {r2:.4f}")
    print(f"  - Mean Absolute Error (MAE): £{mae:,.2f}")

    # --- STAGE 8: Train Residual Model ---
    print("\n--- STAGE 8: Training Residual Refinement Model ---")
    eval_df['log_residual'] = y_price_for_training - eval_df['pred_gamma']
    
    # Dynamically get all head names from the data dictionary for the residual model
    
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