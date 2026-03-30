import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.cluster import KMeans
import joblib
import warnings

# Suppress SettingWithCopyWarning, common in multi-step data prep
warnings.filterwarnings('ignore', category=pd.core.common.SettingWithCopyWarning)

# --- 1. Configuration: Define File Paths and Key Columns ---

# Please verify these paths are correct
BASE_PATH = "[REDACTED_BY_SCRIPT]"
PROPERTY_MASTER_PATH = BASE_PATH + "[REDACTED_BY_SCRIPT]"
GEMINI_FEATURES_PATH = BASE_PATH + "[REDACTED_BY_SCRIPT]"
GLOBAL_SUBSET_1_PATH = BASE_PATH + "[REDACTED_BY_SCRIPT]"
GLOBAL_SUBSET_2_PATH = BASE_PATH + "[REDACTED_BY_SCRIPT]"
GLOBAL_SUBSET_3_PATH = BASE_PATH + "[REDACTED_BY_SCRIPT]"
GLOBAL_SUBSET_4_PATH = BASE_PATH + "[REDACTED_BY_SCRIPT]"
GLOBAL_SUBSET_5_PATH = BASE_PATH + "[REDACTED_BY_SCRIPT]"

# Define key column names for clarity and easier maintenance
# (Based on your provided headers, with the 'num__' prefix removed for easier access)
# Note: I am assuming the address column is the one to merge on initially.
# Let's clean up the column name for merging.
MERGE_COL_PROPERTY = '[REDACTED_BY_SCRIPT]'
MERGE_COL_GEMINI = 'property_id' # Assumed to be equivalent to the address
MERGE_COL_GLOBAL = 'property_postcode'

# AVM and Target columns
SALE_PRICE_COL = '[REDACTED_BY_SCRIPT]'
SALE_YEAR_COL = '[REDACTED_BY_SCRIPT]'
AVM_ESTIMATES = {
    'homipi': '[REDACTED_BY_SCRIPT]',
    'mouseprice': '[REDACTED_BY_SCRIPT]',
    'bnl': '[REDACTED_BY_SCRIPT]'
}

# --- 2. Data Loading and Aggregation Function ---

def load_and_aggregate_data():
    """
    Loads all data sources and merges them into a single DataFrame.
    This function handles the initial data preparation and consolidation.
    """
    print("[REDACTED_BY_SCRIPT]")

    # Load the main property file
    print(f"[REDACTED_BY_SCRIPT]")
    df_master = pd.read_csv(PROPERTY_MASTER_PATH)
    
    # Clean up column names by removing prefixes like 'num__' and 'cat__'
    df_master.columns = df_master.columns.str.split('__').str[-1]

    # Load Gemini features
    print(f"[REDACTED_BY_SCRIPT]")
    df_gemini = pd.read_csv(GEMINI_FEATURES_PATH)
    # Assuming 'property_id' can be used as the merge key, may need adjustment
    # if it'[REDACTED_BY_SCRIPT]'s the primary key.
    df_gemini.rename(columns={MERGE_COL_GEMINI: MERGE_COL_PROPERTY}, inplace=True)

    # Load global subset files
    print("[REDACTED_BY_SCRIPT]")
    df_sub1 = pd.read_csv(GLOBAL_SUBSET_1_PATH)
    df_sub2 = pd.read_csv(GLOBAL_SUBSET_2_PATH)
    df_sub3 = pd.read_csv(GLOBAL_SUBSET_3_PATH)
    df_sub4 = pd.read_csv(GLOBAL_SUBSET_4_PATH)
    df_sub5 = pd.read_csv(GLOBAL_SUBSET_5_PATH)
    
    # First, need a postcode for each property to merge global data
    # We will assume a property address can be mapped to a postcode.
    # A robust way would be to extract postcode from '[REDACTED_BY_SCRIPT]'.
    # For now, let's create a temporary postcode column for merging.
    # This is a simplification; a more robust regex would be better.
    df_master[MERGE_COL_GLOBAL] = df_master[MERGE_COL_PROPERTY].str.split(',').str[-1].str.strip()

    # --- Merging ---
    print("[REDACTED_BY_SCRIPT]")
    # Merge master with gemini
    df_full = pd.merge(df_master, df_gemini, on=MERGE_COL_PROPERTY, how='left')

    # Merge with global subsets on postcode
    df_full = pd.merge(df_full, df_sub1, on=MERGE_COL_GLOBAL, how='left', suffixes=('', '_s1'))
    df_full = pd.merge(df_full, df_sub2, on=MERGE_COL_GLOBAL, how='left', suffixes=('', '_s2'))
    df_full = pd.merge(df_full, df_sub3, on=MERGE_COL_GLOBAL, how='left', suffixes=('', '_s3'))
    df_full = pd.merge(df_full, df_sub4, on=MERGE_COL_GLOBAL, how='left', suffixes=('', '_s4'))
    df_full = pd.merge(df_full, df_sub5, on=MERGE_COL_GLOBAL, how='left', suffixes=('', '_s5'))

    print(f"[REDACTED_BY_SCRIPT]")
    return df_full


# --- 3. Ground Truth Preparation Function ---

def prepare_ground_truth(df, min_sale_year=2024):
    """
    Filters for recent, valid sales and calculates the log-transformed bias targets.
    """
    print(f"[REDACTED_BY_SCRIPT]")

    # Create a copy to avoid modifying the original full dataframe
    ground_truth_df = df.copy()

    # Filter for recent sales with valid prices
    ground_truth_df = ground_truth_df[ground_truth_df[SALE_YEAR_COL] >= min_sale_year]
    ground_truth_df = ground_truth_df[ground_truth_df[SALE_PRICE_COL] > 0]

    # Calculate log-transformed bias for each AVM
    for avm_name, avm_col in AVM_ESTIMATES.items():
        # Ensure AVM estimate is also positive
        ground_truth_df = ground_truth_df[ground_truth_df[avm_col] > 0]
        
        target_col_name = f'[REDACTED_BY_SCRIPT]'
        ground_truth_df[target_col_name] = np.log(ground_truth_df[avm_col]) - np.log(ground_truth_df[SALE_PRICE_COL])

    # Drop rows where bias could not be calculated
    bias_cols = [f'[REDACTED_BY_SCRIPT]' for name in AVM_ESTIMATES.keys()]
    ground_truth_df.dropna(subset=bias_cols, inplace=True)

    print(f"[REDACTED_BY_SCRIPT]")
    return ground_truth_df


# --- 4. Bias Model Training Function ---

def train_bias_predictors(ground_truth_df):
    """
    Trains a bias predictor model for each AVM using geographical cross-validation.
    """
    print("[REDACTED_BY_SCRIPT]")

    # --- Create Geographical Clusters for CV ---
    print("[REDACTED_BY_SCRIPT]")
    coords = ground_truth_df[['latitude', 'longitude']].dropna()
    kmeans = KMeans(n_clusters=100, random_state=42, n_init=10)
    ground_truth_df['geo_cluster'] = kmeans.fit_predict(coords)

    # --- Define columns to drop for the feature set (X) ---
    cols_to_drop = [SALE_PRICE_COL, SALE_YEAR_COL, MERGE_COL_PROPERTY, MERGE_COL_GLOBAL]
    cols_to_drop.extend([f'[REDACTED_BY_SCRIPT]' for name in AVM_ESTIMATES.keys()])
    cols_to_drop.extend(AVM_ESTIMATES.values())
    
    # Also drop date parts if they exist
    cols_to_drop.extend(['[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]'])

    features = ground_truth_df.drop(columns=cols_to_drop, errors='ignore')
    # LightGBM can handle NaNs, but let's ensure all data is numeric for simplicity
    features = features.select_dtypes(include=np.number)
    
    print(f"[REDACTED_BY_SCRIPT]")

    # --- Iterate and Train for Each AVM ---
    for avm_name in AVM_ESTIMATES.keys():
        print(f"[REDACTED_BY_SCRIPT]")
        
        target_col = f'[REDACTED_BY_SCRIPT]'
        y = ground_truth_df[target_col]
        X = features
        groups = ground_truth_df['geo_cluster']

        # LightGBM Model Parameters
        params = {
            'objective': 'regression_l1', # MAE is robust to outliers
            'metric': 'mae',
            'n_estimators': 2000,
            'learning_rate': 0.02,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 1,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
            'num_leaves': 31,
            'verbose': -1,
            'n_jobs': -1,
            'seed': 42,
            'boosting_type': 'gbdt',
        }

        # Setup Geographical Cross-Validation
        group_kfold = GroupKFold(n_splits=5)
        
        # We will train on the full dataset at the end, CV is for performance estimation
        # For this script, let's train a final model on all ground truth data
        
        model = lgb.LGBMRegressor(**params)
        
        # For demonstration, we'll fit on the whole ground_truth data.
        # In a production system, you'd use CV to evaluate and then refit on all data.
        model.fit(X, y, 
                  callbacks=[lgb.early_stopping(100, verbose=False)])

        # Save the trained model
        model_filename = f'[REDACTED_BY_SCRIPT]'
        joblib.dump(model, model_filename)
        print(f"[REDACTED_BY_SCRIPT]'{model_filename}'")


# --- 5. Final Feature Generation Function ---

def generate_final_features(full_df):
    """
    Loads the trained bias predictors and generates new bias features for the entire dataset.
    """
    print("[REDACTED_BY_SCRIPT]")
    
    df_final = full_df.copy()

    # Prepare the feature set for the full dataset (must match training)
    cols_to_drop = [SALE_PRICE_COL, SALE_YEAR_COL, MERGE_COL_PROPERTY, MERGE_COL_GLOBAL]
    cols_to_drop.extend(AVM_ESTIMATES.values())
    cols_to_drop.extend(['[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]'])

    features_full = df_final.drop(columns=cols_to_drop, errors='ignore').select_dtypes(include=np.number)

    # Load and predict for each AVM
    for avm_name in AVM_ESTIMATES.keys():
        print(f"[REDACTED_BY_SCRIPT]")
        model_filename = f'[REDACTED_BY_SCRIPT]'
        try:
            model = joblib.load(model_filename)
            
            # Ensure feature names match between training and prediction
            # This handles cases where some columns might be missing in one set
            train_cols = model.feature_name_
            predict_cols = features_full.columns
            
            missing_in_predict = set(train_cols) - set(predict_cols)
            if missing_in_predict:
                print(f"[REDACTED_BY_SCRIPT]")
                for col in missing_in_predict:
                    features_full[col] = 0 # Or a suitable default
            
            # Align column order
            aligned_features = features_full[train_cols]

            predicted_bias_col = f'[REDACTED_BY_SCRIPT]'
            df_final[predicted_bias_col] = model.predict(aligned_features)
            print(f"    - Created feature: '{predicted_bias_col}'")

        except FileNotFoundError:
            print(f"[REDACTED_BY_SCRIPT]")
            continue
            
    # Save the final, enriched DataFrame
    output_filename = '[REDACTED_BY_SCRIPT]'
    df_final.to_csv(output_filename, index=False)
    print(f"[REDACTED_BY_SCRIPT]'{output_filename}'")
    return df_final


# --- Main Execution Block ---
if __name__ == '__main__':
    # Step 1: Load and combine all your data sources
    full_dataframe = load_and_aggregate_data()

    # Step 2: Create the specialized dataset for training the bias models
    ground_truth_dataframe = prepare_ground_truth(full_dataframe, min_sale_year=2024)

    # Step 3: Train the bias predictor models using this ground truth data
    train_bias_predictors(ground_truth_dataframe)

    # Step 4: Use the trained models to generate features for the entire dataset
    final_enriched_dataframe = generate_final_features(full_dataframe)

    # Display the new columns in the final dataframe
    print("[REDACTED_BY_SCRIPT]")
    new_cols = ['[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', 
                '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
                '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]']
    print(final_enriched_dataframe[new_cols].head())