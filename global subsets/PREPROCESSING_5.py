import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.utils.validation import NotFittedError # For checking if transformer is fitted
from sklearn.pipeline import Pipeline # Add this import

# --- Configuration Constants ---
SUBSET5_RAW_FILE = '[REDACTED_BY_SCRIPT]'  # Assuming this is your input file name
SUBSET5_PROCESSED_FILE = '[REDACTED_BY_SCRIPT]'  # Changed to .parquet
N_ROWS_FOR_FITTING = 200000  # Number of rows to load for fitting transformers
CHUNK_SIZE = 50000         # Number of rows to process at a time for the full dataset

# Imputation and replacement values
NUMERICAL_NAN_IMPUTE_VALUE = -1.0 # Using float for consistency
CATEGORICAL_NAN_REPLACE_VALUE = 'Missing_Category'

# Thresholds for feature removal
# Low variance thresholds
LOW_VARIANCE_THRESHOLD_REPORT = 0.000001 # For reporting columns with very low variance
NUMERICAL_LOW_VARIANCE_THRESHOLD_FOR_REMOVAL = 0.0001 # To remove numerical columns
OHE_LOW_VARIANCE_THRESHOLD_FOR_REMOVAL = 0.002 # To remove OHE columns (usually 0/1, so variance is p(1-p))

# High correlation threshold
HIGH_CORRELATION_THRESHOLD = 0.98 # For identifying highly correlated features

# --- Define Columns for Subset 5 ---
# These lists will be populated after inspecting the actual columns from the file
# For now, making educated guesses based on the description of Subset 5

# Columns to drop right at the beginning
COLS_TO_DROP_INITIAL_S5 = ['pcds', 'lsoa11cd'] # Assuming lsoa11cd was for joining

# Categorical columns - 'usertype_numeric' is the main one.
# Even if it'[REDACTED_BY_SCRIPT]'if_binary' is efficient.
CATEGORICAL_COLS_S5 = ['usertype_numeric']

# Numerical columns will be all others after initial drops and excluding categorical ones.
# This will be determined dynamically later.
NUMERICAL_COLS_S5 = [] # To be populated

# Columns identified for manual dropping after reviewing correlation reports (if any)
MANUAL_DROPS_S5 = [
    '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]',
    'SFU_Y3_if_LargeUser',
    '[REDACTED_BY_SCRIPT]'
]


# --- Preprocessing Pipelines ---

# Numerical features pipeline
numerical_pipeline = Pipeline(steps=[ # Changed to Pipeline
    ('imputer', SimpleImputer(strategy='constant', fill_value=NUMERICAL_NAN_IMPUTE_VALUE)),
    ('transformer', PowerTransformer(method='yeo-johnson', standardize=True)),
    ('scaler', MinMaxScaler())
]) # Removed remainder='passthrough'

# Categorical features pipeline
categorical_pipeline = Pipeline(steps=[ # Changed to Pipeline
    ('imputer', SimpleImputer(strategy='constant', fill_value=CATEGORICAL_NAN_REPLACE_VALUE)),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='if_binary'))
]) # Removed remainder='passthrough'


# --- Helper Functions ---

def load_data_s5(file_path, nrows=None, usecols=None):
    """[REDACTED_BY_SCRIPT]"""
    if nrows:
        print(f"[REDACTED_BY_SCRIPT]")
    else:
        print(f"[REDACTED_BY_SCRIPT]")
    try:
        df = pd.read_csv(file_path, nrows=nrows, usecols=usecols)
        print(f"[REDACTED_BY_SCRIPT]")
        return df
    except FileNotFoundError:
        print(f"[REDACTED_BY_SCRIPT]")
        # Create a dummy dataframe for demonstration purposes if file not found
        print("[REDACTED_BY_SCRIPT]")
        dummy_data = {
            'pcds': [f'PC{i}' for i in range(nrows if nrows else 100)],
            'lsoa11cd': [f'LSOA{i%10}' for i in range(nrows if nrows else 100)],
            'usertype_numeric': np.random.choice([0, 1, np.nan], size=nrows if nrows else 100, p=[0.7, 0.2, 0.1]),
            'bba225_dow': np.random.rand(nrows if nrows else 100) * 100 + np.random.choice([0, np.nan], size=nrows if nrows else 100, p=[0.95,0.05]),
            'bba215_dow': np.random.rand(nrows if nrows else 100) * 80,
            'bba205_dow': np.random.rand(nrows if nrows else 100) * 60,
            'bba225_uf': np.random.rand(nrows if nrows else 100) * 100,
            'bba215_uf': np.random.rand(nrows if nrows else 100) * 70,
            'bba205_uf': np.random.rand(nrows if nrows else 100) * 50,
            'bba225_sfu': np.random.rand(nrows if nrows else 100) * 100,
            'bba215_sfu': np.random.rand(nrows if nrows else 100) * 90,
            'bba205_sfu': np.random.rand(nrows if nrows else 100) * 80,
            # Simplified FI features for dummy data
            'Dow_AbsChange_Y3_Y2': np.random.rand(nrows if nrows else 100) * 20 - 10,
            'UF_to_SFU_Ratio_Y3': np.random.rand(nrows if nrows else 100),
            'Dow_Y3_if_LargeUser': np.random.rand(nrows if nrows else 100) * 100,
            'Dow_StdDev_AllYears': np.random.rand(nrows if nrows else 100) * 5,
        }
        # Add more FI columns to mimic the structure, up to ~40 as described
        for i in range(15, 35): # Simplified additional FI features
            dummy_data[f'FI_Example_{i}'] = np.random.rand(nrows if nrows else 100) * np.random.randint(1,50) + np.random.choice([0, np.nan, 1, -1], size=nrows if nrows else 100, p=[0.9,0.05,0.025,0.025])

        df = pd.DataFrame(dummy_data)
        if nrows:
            df = df.head(nrows)
        return df
    except Exception as e:
        print(f"[REDACTED_BY_SCRIPT]")
        return pd.DataFrame() # Return empty dataframe on other errors


def preprocess_sample_s5(df_sample):
    """
    Preprocesses the sample data:
    1. Drops initial unwanted columns.
    2. Identifies numerical and categorical columns.
    3. Converts identified categorical columns to object type. # <-- NEW STEP
    4. Fits the numerical and categorical pipelines.
    5. Combines processed numerical and categorical features.
    Returns the fitted ColumnTransformer, processed DataFrame, and OHE feature names.
    """
    print("[REDACTED_BY_SCRIPT]")
    
    # 1. Drop initial columns
    df_sample_processed = df_sample.drop(columns=COLS_TO_DROP_INITIAL_S5, errors='ignore')
    print(f"[REDACTED_BY_SCRIPT]")

    # 2. Identify actual numerical and categorical columns from the sample
    current_categorical_cols = [col for col in CATEGORICAL_COLS_S5 if col in df_sample_processed.columns]
    
    potential_numerical_cols = [col for col in df_sample_processed.columns if col not in current_categorical_cols]
    
    global NUMERICAL_COLS_S5 
    NUMERICAL_COLS_S5 = []
    for col in potential_numerical_cols:
        if pd.api.types.is_numeric_dtype(df_sample_processed[col]):
            NUMERICAL_COLS_S5.append(col)
        else:
            print(f"Warning: Column '{col}'[REDACTED_BY_SCRIPT]")

    print(f"[REDACTED_BY_SCRIPT]")
    print(f"[REDACTED_BY_SCRIPT]")

    # 3. Convert categorical columns to object type to ensure string imputation works
    # This must be done BEFORE fitting the preprocessor
    if current_categorical_cols:
        for col in current_categorical_cols:
            df_sample_processed[col] = df_sample_processed[col].astype(object)
            print(f"Converted column '{col}'[REDACTED_BY_SCRIPT]")

    # Create the main ColumnTransformer
    transformers_list = []
    if NUMERICAL_COLS_S5:
        transformers_list.append(('num', numerical_pipeline, NUMERICAL_COLS_S5))
    if current_categorical_cols:
        transformers_list.append(('cat', categorical_pipeline, current_categorical_cols))

    if not transformers_list:
        print("[REDACTED_BY_SCRIPT]")
        return None, pd.DataFrame(), []
        
    preprocessor = ColumnTransformer(transformers=transformers_list, remainder='drop')

    # 4. Fit the preprocessor
    print("[REDACTED_BY_SCRIPT]")
    preprocessor.fit(df_sample_processed) # df_sample_processed now has correct dtypes for categorical_cols
    print("[REDACTED_BY_SCRIPT]")

    # 5. Transform the sample data
    print("[REDACTED_BY_SCRIPT]")
    processed_sample_np = preprocessor.transform(df_sample_processed)
    
    # Get feature names after transformation
    ohe_feature_names = []
    try:
        if 'cat' in preprocessor.named_transformers_ and current_categorical_cols:
            cat_transformer = preprocessor.named_transformers_['cat']
            ohe_step = cat_transformer.named_steps['onehot'] # .named_steps for Pipeline
            if hasattr(ohe_step, 'get_feature_names_out'):
                 ohe_feature_names = list(ohe_step.get_feature_names_out(current_categorical_cols))
            else:
                 print("[REDACTED_BY_SCRIPT]")
    except NotFittedError:
        print("[REDACTED_BY_SCRIPT]")
    except Exception as e:
        print(f"[REDACTED_BY_SCRIPT]")

    final_feature_names = list(NUMERICAL_COLS_S5) + ohe_feature_names
    
    processed_sample_df = pd.DataFrame(processed_sample_np, columns=final_feature_names, index=df_sample_processed.index)
    print(f"[REDACTED_BY_SCRIPT]")
    
    return preprocessor, processed_sample_df, ohe_feature_names



def identify_low_variance_features_s5(df_processed_sample, ohe_feature_names):
    """[REDACTED_BY_SCRIPT]"""
    print("[REDACTED_BY_SCRIPT]")
    features_to_drop_low_var = []
    variances = df_processed_sample.var()

    print("[REDACTED_BY_SCRIPT]")
    for col_name, var_val in variances.items():
        if var_val < LOW_VARIANCE_THRESHOLD_REPORT:
            print(f"[REDACTED_BY_SCRIPT]")

        # Determine removal based on type of column
        threshold_for_removal = OHE_LOW_VARIANCE_THRESHOLD_FOR_REMOVAL if col_name in ohe_feature_names else NUMERICAL_LOW_VARIANCE_THRESHOLD_FOR_REMOVAL
        if var_val < threshold_for_removal:
            print(f"[REDACTED_BY_SCRIPT]")
            features_to_drop_low_var.append(col_name)
        
    print(f"[REDACTED_BY_SCRIPT]")
    return features_to_drop_low_var


def identify_highly_correlated_features_s5(df_processed_sample, features_already_dropping):
    """
    Identifies highly correlated features from the processed sample.
    For Subset 5, this will need careful consideration of rules, similar to Subset 2's revised strategy.
    Initial version: report pairs, manually decide drops.
    """
    print("[REDACTED_BY_SCRIPT]")
    df_to_correlate = df_processed_sample.drop(columns=features_already_dropping, errors='ignore')
    
    # Ensure only numeric columns are used for correlation
    numeric_df_to_correlate = df_to_correlate.select_dtypes(include=np.number)
    
    corr_matrix = numeric_df_to_correlate.corr().abs()
    
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    features_to_drop_correlated = set()
    highly_correlated_pairs = []

    for column in upper.columns:
        for index in upper.index:
            if upper.loc[index, column] > HIGH_CORRELATION_THRESHOLD:
                highly_correlated_pairs.append((index, column, upper.loc[index, column]))

    print(f"[REDACTED_BY_SCRIPT]")
    # Basic strategy: for each pair, if neither is already set to be dropped,
    # add the second one to the drop list. This is naive and needs refinement.
    # For now, let's just report and let manual decisions guide via MANUAL_DROPS_S5
    
    unresolved_correlated_pairs_for_manual_review = []

    for feat1, feat2, corr_val in highly_correlated_pairs:
        # Check if either is already in features_to_drop_correlated or features_already_dropping
        # (features_already_dropping are typically low variance ones at this stage)
        in_features_to_drop_correlated_set = feat1 in features_to_drop_correlated or feat2 in features_to_drop_correlated
        in_features_already_dropping_set = feat1 in features_already_dropping or feat2 in features_already_dropping
        
        if not in_features_to_drop_correlated_set and not in_features_already_dropping_set:
            print(f"[REDACTED_BY_SCRIPT]")
            unresolved_correlated_pairs_for_manual_review.append((feat1, feat2, corr_val))
            # Default decision for now if not in MANUAL_DROPS_S5: drop feat2
            # This is where refined logic would go for Subset 5.
            # E.g., prefer 'change' features over raw, prefer latest year, etc.
            # if feat1 not in MANUAL_DROPS_S5 and feat2 not in MANUAL_DROPS_S5:
            # features_to_drop_correlated.add(feat2) # Example: drop the second one
            # print(f"[REDACTED_BY_SCRIPT]'{feat2}'[REDACTED_BY_SCRIPT]")
    
    print("[REDACTED_BY_SCRIPT]")
    print("[REDACTED_BY_SCRIPT]")
    print("[REDACTED_BY_SCRIPT]")
    for feat1, feat2, corr_val in unresolved_correlated_pairs_for_manual_review:
         print(f"[REDACTED_BY_SCRIPT]")

    # Features from MANUAL_DROPS_S5 are the primary source for correlated drops now
    for f_manual in MANUAL_DROPS_S5:
        if f_manual in df_to_correlate.columns: # ensure it's a valid column
             features_to_drop_correlated.add(f_manual)
        else:
            print(f"Warning: Column '{f_manual}'[REDACTED_BY_SCRIPT]")

    print(f"[REDACTED_BY_SCRIPT]")
    return list(features_to_drop_correlated)


# --- Main Processing Logic ---
if __name__ == '__main__':
    # 1. Load and preprocess the sample to fit transformers and identify columns
    df_sample = load_data_s5(SUBSET5_RAW_FILE, nrows=N_ROWS_FOR_FITTING)

    if df_sample.empty:
        print("[REDACTED_BY_SCRIPT]")
        exit()
        
    # Dynamically define NUMERICAL_COLS_S5 based on what's left after drops and categorical
    # This is handled inside preprocess_sample_s5 now

    preprocessor_fitted, processed_sample_df, ohe_feature_names = preprocess_sample_s5(df_sample)

    if preprocessor_fitted is None or processed_sample_df.empty:
        print("[REDACTED_BY_SCRIPT]")
        exit()

    # 2. Identify features to remove based on the processed sample
    features_to_drop_low_var = identify_low_variance_features_s5(processed_sample_df, ohe_feature_names)
    
    # Pass low variance drops to correlation function to avoid re-evaluating them
    features_to_drop_correlated = identify_highly_correlated_features_s5(processed_sample_df, features_to_drop_low_var)

    # Combine all features to drop
    # Ensure no duplicates and that we're dropping from the *final processed column names*
    all_features_to_drop = set(features_to_drop_low_var) | set(features_to_drop_correlated)
    
    print("[REDACTED_BY_SCRIPT]")
    print(f"[REDACTED_BY_SCRIPT]")
    print(f"[REDACTED_BY_SCRIPT]")
    print(f"[REDACTED_BY_SCRIPT]")

    # Determine final columns to keep
    # These are columns from the *processed_sample_df*
    final_columns_to_keep = [col for col in processed_sample_df.columns if col not in all_features_to_drop]
    print(f"[REDACTED_BY_SCRIPT]")
    if not final_columns_to_keep:
        print("[REDACTED_BY_SCRIPT]")
        exit()

    # 3. Process the full dataset in chunks
    print(f"[REDACTED_BY_SCRIPT]")
    
    # Store all processed chunks in a list for concatenation
    processed_chunks = []
    
    # Get all original column names to select when reading chunks
    # (except those dropped initially, to save memory)
    try:
        all_original_cols = pd.read_csv(SUBSET5_RAW_FILE, nrows=0).columns.tolist()
        cols_to_read_in_chunks = [col for col in all_original_cols if col not in COLS_TO_DROP_INITIAL_S5]
    except Exception as e:
        print(f"[REDACTED_BY_SCRIPT]")
        cols_to_read_in_chunks = None

    for chunk_num, chunk_df in enumerate(pd.read_csv(SUBSET5_RAW_FILE, chunksize=CHUNK_SIZE, usecols=cols_to_read_in_chunks, iterator=True)):
        print(f"[REDACTED_BY_SCRIPT]")
        
        # Ensure correct columns are present for transformation, similar to sample prep
        # (COLS_TO_DROP_INITIAL_S5 are already excluded by usecols)
        
        # Transform the chunk
        processed_chunk_np = preprocessor_fitted.transform(chunk_df)
        
        # Create DataFrame with the correct feature names (those used for processed_sample_df)
        processed_chunk_df = pd.DataFrame(processed_chunk_np, columns=processed_sample_df.columns, index=chunk_df.index)
        
        # Select only the final columns to keep
        final_chunk_df = processed_chunk_df[final_columns_to_keep]
        
        # Append chunk to list instead of writing to CSV
        processed_chunks.append(final_chunk_df)
        
        print(f"[REDACTED_BY_SCRIPT]")

    # Concatenate all chunks and save as Parquet
    print("[REDACTED_BY_SCRIPT]")
    final_df = pd.concat(processed_chunks, ignore_index=True)
    
    print(f"[REDACTED_BY_SCRIPT]")
    final_df.to_parquet(SUBSET5_PROCESSED_FILE, index=False, engine='pyarrow')
    
    print(f"[REDACTED_BY_SCRIPT]")
    print(f"[REDACTED_BY_SCRIPT]")