import pandas as pd
import numpy as np

# --- Helper Function for Reading CSVs ---
def read_csv_with_multiple_encodings(file_path, encodings=['utf-8', 'latin-1', 'cp1252'], **kwargs):
    """
    Attempts to read a CSV file using a list of specified encodings.
    """
    ex_enc = None
    for encoding in encodings:
        try:
            return pd.read_csv(file_path, encoding=encoding, **kwargs)
        except UnicodeDecodeError as e:
            ex_enc = e
            # print(f"[REDACTED_BY_SCRIPT]")
            continue
    print(f"[REDACTED_BY_SCRIPT]")
    raise Exception(f"[REDACTED_BY_SCRIPT]")


# --- Configuration: File Paths (USER MUST UPDATE THESE) ---
# Main postcode to LSOA lookup file
BASE_DIR = "[REDACTED_BY_SCRIPT]"  # Base directory where your data files are located
pcd_lsoa_lookup_file = BASE_DIR + "[REDACTED_BY_SCRIPT]"

# LSOA Level Data Files
ahah_data_file = BASE_DIR + "/AHAH_V4.csv" # Assumed file for AHAH features
lsoa_veg_file = BASE_DIR + "[REDACTED_BY_SCRIPT]"
lsoa_pwc_file = BASE_DIR + "[REDACTED_BY_SCRIPT]"
lsoa_boundaries_file = BASE_DIR + "[REDACTED_BY_SCRIPT]"

chunk_size = 50000
epsilon = 1e-6 # Small number to prevent division by zero

# --- Load LSOA-level Datasets (Lookup Tables) ---
print("[REDACTED_BY_SCRIPT]")

# 1. AHAH Data
# Replace '[REDACTED_BY_SCRIPT]' with the actual LSOA code column name if not 'LSOA21CD'
# Define required AHAH columns before try block to ensure it's always available
required_ahah_cols = [
    'ah4blue', 'ah4dent', 'ah4gp', 'ah4hosp', 'ah4phar', 'ah4leis', 
    'ah4pubs', 'ah4ffood', 'ah4tob', 'ah4gamb', 'ah4gpas', 'ah4no2', 
    'ah4so2', 'ah4pm10', 
    'ah4gp_rnk', 'ah4dent_rnk', 'ah4phar_rnk', 'ah4hosp_rnk', 'ah4leis_rnk', 
    'ah4gpas_rnk', 'ah4blue_rnk', 'ah4no2_rnk', 'ah4so2_rnk', 'ah4pm10_rnk',
    'ah4gamb_rnk', 'ah4pubs_rnk', 'ah4tob_rnk', 'ah4ffood_rnk',
    'ah4gp_pct', 'ah4dent_pct', 'ah4phar_pct', 'ah4hosp_pct', 'ah4leis_pct',
    'ah4gpas_pct', 'ah4blue_pct', 'ah4no2_pct', 'ah4so2_pct', 'ah4pm10_pct',
    'ah4gamb_pct', 'ah4pubs_pct', 'ah4tob_pct', 'ah4ffood_pct',
    'ah4h', 'ah4g', 'ah4e', 'ah4r',
    'ah4h_rnk', 'ah4g_rnk', 'ah4e_rnk', 'ah4r_rnk',
    'ah4h_pct', 'ah4g_pct', 'ah4e_pct', 'ah4r_pct',
    'ah4ahah', 'ah4ahah_rnk', 'ah4ahah_pct', 'LSOA21CD' # Ensure LSOA21CD is here
]

try:
    df_ahah = read_csv_with_multiple_encodings(ahah_data_file)
    # Standardize LSOA column name for merging
    # Example: if AHAH file uses 'lsoa21cd', rename to 'LSOA21CD'
    # df_ahah.rename(columns={'[REDACTED_BY_SCRIPT]': 'LSOA21CD'}, inplace=True)
    if 'LSOA21CD' not in df_ahah.columns and 'lsoacode' in df_ahah.columns: # Adjust as per your AHAH file
         df_ahah.rename(columns={'lsoacode': 'LSOA21CD'}, inplace=True)
    elif '\ufeffLSOA21CD' in df_ahah.columns:
         df_ahah.rename(columns={'\ufeffLSOA21CD': 'LSOA21CD'}, inplace=True)

    # Ensure all required ah4 columns exist, otherwise create them as NaN
    # This is a precaution; ideally, the file has them.
    for col in required_ahah_cols:
        if col not in df_ahah.columns and col != 'LSOA21CD': # LSOA21CD is the key
            print(f"Warning: Column '{col}'[REDACTED_BY_SCRIPT]")
            df_ahah[col] = np.nan
    df_ahah = df_ahah[[c for c in required_ahah_cols if c in df_ahah.columns]] # Select only existing or created cols
    df_ahah.drop_duplicates(subset=['LSOA21CD'], inplace=True)
except FileNotFoundError:
    print(f"[REDACTED_BY_SCRIPT]")
    df_ahah = pd.DataFrame(columns=required_ahah_cols) # Empty df to allow script to run for structure check
except Exception as e:
    print(f"[REDACTED_BY_SCRIPT]")
    df_ahah = pd.DataFrame(columns=required_ahah_cols)

# 2. LSOA Vegetation Data
lsoa_veg_cols = ['LSOA21CD', 'NDVI_MEAN', 'NDVI_MEDIAN', 'NDVI_STD', 'NDVI_MAX', 
                 'NDVI_MIN', 'NDVI_CV', 'VEG_FRAC', 'EVI_MEAN', 'EVI_MEDIAN', 
                 'EVI_STD', 'EVI_MAX', 'EVI_MIN', 'PXL_COUNT', 
                 'FINAL_PXL_COUNT', 'PCT_FILTERED']
try:
    df_lsoa_veg = read_csv_with_multiple_encodings(lsoa_veg_file)
    if '\ufeffLSOA21CD' in df_lsoa_veg.columns:
         df_lsoa_veg.rename(columns={'\ufeffLSOA21CD': 'LSOA21CD'}, inplace=True)
    df_lsoa_veg = df_lsoa_veg[[col for col in lsoa_veg_cols if col in df_lsoa_veg.columns]]
    df_lsoa_veg.drop_duplicates(subset=['LSOA21CD'], inplace=True)
except FileNotFoundError:
    print(f"[REDACTED_BY_SCRIPT]")
    df_lsoa_veg = pd.DataFrame(columns=lsoa_veg_cols)
except Exception as e:
    print(f"[REDACTED_BY_SCRIPT]")
    df_lsoa_veg = pd.DataFrame(columns=lsoa_veg_cols)


# 3. LSOA Population Weighted Centroids (PWC)
pwc_cols = ['LSOA21CD', 'x', 'y']
try:
    df_lsoa_pwc = read_csv_with_multiple_encodings(lsoa_pwc_file)
    if '\ufeffFID' in df_lsoa_pwc.columns: # Handle potential byte order mark in first column name
        df_lsoa_pwc.rename(columns={'\ufeffFID': 'FID'}, inplace=True)
    df_lsoa_pwc = df_lsoa_pwc[[col for col in pwc_cols if col in df_lsoa_pwc.columns]]
    df_lsoa_pwc.drop_duplicates(subset=['LSOA21CD'], inplace=True)
except FileNotFoundError:
    print(f"[REDACTED_BY_SCRIPT]")
    df_lsoa_pwc = pd.DataFrame(columns=pwc_cols)
except Exception as e:
    print(f"[REDACTED_BY_SCRIPT]")
    df_lsoa_pwc = pd.DataFrame(columns=pwc_cols)

# 4. LSOA Boundaries (for Area/Length)
boundaries_cols = ['LSOA21CD', 'Shape__Area', 'Shape__Length']
try:
    df_lsoa_boundaries = read_csv_with_multiple_encodings(lsoa_boundaries_file, low_memory=False) # low_memory for mixed types if any
    if '\ufeffFID' in df_lsoa_boundaries.columns:
        df_lsoa_boundaries.rename(columns={'\ufeffFID': 'FID'}, inplace=True)
    df_lsoa_boundaries = df_lsoa_boundaries[[col for col in boundaries_cols if col in df_lsoa_boundaries.columns]]
    df_lsoa_boundaries.drop_duplicates(subset=['LSOA21CD'], inplace=True)
except FileNotFoundError:
    print(f"[REDACTED_BY_SCRIPT]")
    df_lsoa_boundaries = pd.DataFrame(columns=boundaries_cols)
except Exception as e:
    print(f"[REDACTED_BY_SCRIPT]")
    df_lsoa_boundaries = pd.DataFrame(columns=boundaries_cols)

print("[REDACTED_BY_SCRIPT]")

# --- Function to Apply Feature Interactions ---
def apply_feature_interactions_subset4(df):
    """
    Applies the 40 feature interactions for Subset 4 to the DataFrame.
    The DataFrame should already contain all the base features from AHAH, LSOA Veg, etc.
    """
    # I. Green Space & Air Quality Interactions
    df['[REDACTED_BY_SCRIPT]'] = df['NDVI_MEAN'] * (1 / (df['ah4pm10'] + epsilon))
    df['[REDACTED_BY_SCRIPT]'] = df['ah4gpas'] * (1 / (df['ah4no2'] + epsilon))
    df['[REDACTED_BY_SCRIPT]'] = df['VEG_FRAC'] * df['ah4e']
    df['[REDACTED_BY_SCRIPT]'] = df['ah4pm10'] / (df['ah4no2'] + epsilon)
    df['[REDACTED_BY_SCRIPT]'] = df['NDVI_STD'] * (df['ah4no2'] + df['ah4pm10'] + df['ah4so2'])

    # II. Access to Services & Environmental Context Interactions
    df['[REDACTED_BY_SCRIPT]'] = (1 / (df['ah4gp'] + epsilon)) * df['ah4e']
    df['[REDACTED_BY_SCRIPT]'] = (1 / (df['ah4leis'] + epsilon)) * df['ah4g']
    df['[REDACTED_BY_SCRIPT]'] = (1 / (df['ah4blue'] + epsilon)) * df['ah4e']
    df['[REDACTED_BY_SCRIPT]'] = df['ah4h'] * df['ah4e']
    df['[REDACTED_BY_SCRIPT]'] = df['ah4h'] / (df['ah4no2'] + df['ah4pm10'] + df['ah4so2'] + epsilon)

    # III. Hazard Proximity & Mitigating/Compounding Factor Interactions
    df['[REDACTED_BY_SCRIPT]'] = (1 / (df['ah4ffood'] + epsilon)) * df['ah4g']
    df['[REDACTED_BY_SCRIPT]'] = (1 / (df['ah4gamb'] + epsilon)) * df['ah4leis']
    df['[REDACTED_BY_SCRIPT]'] = (1 / (df['ah4pubs'] + epsilon)) + (1 / (df['ah4ffood'] + epsilon))
    df['[REDACTED_BY_SCRIPT]'] = (df['ah4no2'] + df['ah4pm10'] + df['ah4so2']) * df['ah4gp']
    df['[REDACTED_BY_SCRIPT]'] = (1 / (df['ah4tob'] + epsilon)) / (df['ah4g'] + epsilon)
    
    # IV. Relative Performance & Domain Interactions
    df['[REDACTED_BY_SCRIPT]'] = df['ah4h_rnk'] - df['ah4e_rnk']
    df['[REDACTED_BY_SCRIPT]'] = df['ah4g_rnk'] / (df['ah4r_rnk'] + epsilon)
    df['[REDACTED_BY_SCRIPT]'] = df['ah4ahah'] * df['NDVI_MEAN']
    df['[REDACTED_BY_SCRIPT]'] = df['ah4gp_rnk'] - df['ah4hosp_rnk']
    df['[REDACTED_BY_SCRIPT]'] = df['ah4no2_pct'] - df['ah4pm10_pct']

    # V. Spatial Characteristics & Environmental Interactions
    df['[REDACTED_BY_SCRIPT]'] = df['NDVI_MEAN'] * df['Shape__Area']
    df['[REDACTED_BY_SCRIPT]'] = df['ah4ahah'] / (df['Shape__Area'] + epsilon)
    if 'FINAL_PXL_COUNT' in df.columns and 'PXL_COUNT' in df.columns: # Check if veg quality cols exist
        df['[REDACTED_BY_SCRIPT]'] = df['FINAL_PXL_COUNT'] / (df['PXL_COUNT'] + epsilon)
    else:
        df['[REDACTED_BY_SCRIPT]'] = np.nan
    if 'x' in df.columns and 'y' in df.columns: # Check if PWC coords exist
         df['CoordProduct_Xcoord_x_Ycoord'] = df['x'] * df['y']
    else:
        df['CoordProduct_Xcoord_x_Ycoord'] = np.nan
        
    # VI. Advanced Composite & Balance Metrics
    df['[REDACTED_BY_SCRIPT]'] = (df['ah4h'] + df['ah4g']) / (df['ah4e'] + epsilon)
    df['[REDACTED_BY_SCRIPT]'] = (1/(df['ah4gp']+epsilon)) + (1/(df['ah4dent']+epsilon)) + (1/(df['ah4phar']+epsilon))
    df['[REDACTED_BY_SCRIPT]'] = \
        ((1/(df['ah4ffood']+epsilon)) + (1/(df['ah4pubs']+epsilon))) / ((1/(df['ah4leis']+epsilon)) + epsilon)
    df['[REDACTED_BY_SCRIPT]'] = df['NDVI_MEAN'] / (df['NDVI_STD'] + epsilon)
    if 'EVI_MEAN' in df.columns and 'NDVI_MEAN' in df.columns:
        df['[REDACTED_BY_SCRIPT]'] = df['EVI_MEAN'] / (df['NDVI_MEAN'] + epsilon)
    else:
        df['[REDACTED_BY_SCRIPT]'] = np.nan
    df['[REDACTED_BY_SCRIPT]'] = df['ah4gpas_pct'] - df['ah4blue_pct']
    df['[REDACTED_BY_SCRIPT]'] = df['ah4gp'] * df['ah4gp_rnk']
    df['[REDACTED_BY_SCRIPT]'] = df['ah4no2'] * df['ah4no2_pct']
    df['[REDACTED_BY_SCRIPT]'] = df['NDVI_MAX'] - df['NDVI_MIN']
    df['[REDACTED_BY_SCRIPT]'] = df['NDVI_MEAN'] / ((df['NDVI_MAX'] - df['NDVI_MIN']) + epsilon)
    df['[REDACTED_BY_SCRIPT]'] = df['NDVI_MAX'] * df['ah4e']
    df['[REDACTED_BY_SCRIPT]'] = (1 / (df['ah4pubs'] + epsilon)) * (1 / (df['ah4leis'] + epsilon))
    df['[REDACTED_BY_SCRIPT]'] = df['ah4ahah'] * df['ah4ahah']
    df['[REDACTED_BY_SCRIPT]'] = df['ah4h'] / (df['ah4r'] + epsilon)
    df['[REDACTED_BY_SCRIPT]'] = df['ah4no2'] - df['ah4pm10']
    df['[REDACTED_BY_SCRIPT]'] = \
        (1/(df['ah4ffood']+epsilon)) + (1/(df['ah4gamb']+epsilon)) + (1/(df['ah4tob']+epsilon))
    
    return df

# --- Process Postcode Data in Chunks ---
processed_chunks = []
print(f"[REDACTED_BY_SCRIPT]")

try:
    chunk_iter = read_csv_with_multiple_encodings(
        pcd_lsoa_lookup_file,
        chunksize=chunk_size,
        usecols=['pcds', 'lsoa21cd'], # Only load necessary columns
        dtype={'pcds': str, 'lsoa21cd': str} # Specify dtype for join keys
    )

    for i, pcd_chunk in enumerate(chunk_iter):
        print(f"[REDACTED_BY_SCRIPT]")

        # Prepare base chunk
        pcd_chunk.rename(columns={'lsoa21cd': 'LSOA21CD'}, inplace=True) # Standardize LSOA col name

        # Merge with LSOA-level data
        merged_chunk = pcd_chunk
        if not df_ahah.empty:
            merged_chunk = pd.merge(merged_chunk, df_ahah, on='LSOA21CD', how='left', sort=False)
        if not df_lsoa_veg.empty:
            merged_chunk = pd.merge(merged_chunk, df_lsoa_veg, on='LSOA21CD', how='left', sort=False)
        if not df_lsoa_pwc.empty:
            merged_chunk = pd.merge(merged_chunk, df_lsoa_pwc, on='LSOA21CD', how='left', sort=False)
        if not df_lsoa_boundaries.empty:
            merged_chunk = pd.merge(merged_chunk, df_lsoa_boundaries, on='LSOA21CD', how='left', sort=False)
        
        # Apply feature interactions
        if not merged_chunk.empty:
            # Fill NaNs for numeric columns that are expected by interactions before calculation
            # This is a basic strategy; more sophisticated imputation might be needed
            numeric_cols_for_interactions = [
                'NDVI_MEAN', 'ah4pm10', 'ah4gpas', 'ah4no2', 'VEG_FRAC', 'ah4e', 'NDVI_STD', 'ah4so2',
                'ah4gp', 'ah4leis', 'ah4g', 'ah4blue', 'ah4h', 'ah4ffood', 'ah4gamb', 'ah4pubs',
                'ah4tob', 'ah4h_rnk', 'ah4e_rnk', 'ah4g_rnk', 'ah4r_rnk', 'ah4ahah', 'ah4gp_rnk',
                'ah4hosp_rnk', 'ah4no2_pct', 'ah4pm10_pct', 'Shape__Area', 'x', 'y', 'FINAL_PXL_COUNT',
                'PXL_COUNT', 'EVI_MEAN', 'ah4gpas_pct', 'ah4blue_pct', 'NDVI_MAX', 'NDVI_MIN'
            ]
            for col in numeric_cols_for_interactions:
                if col in merged_chunk.columns:
                    # Attempt to convert to numeric, coercing errors. Then fillna.
                    merged_chunk[col] = pd.to_numeric(merged_chunk[col], errors='coerce')
                   # merged_chunk[col].fillna(0, inplace=True) # Example: fill with 0, adjust if needed
                else: # If a column is missing entirely after merges, add it as NaN before interactions
                     merged_chunk[col] = np.nan
            
            # Re-ensure all required ah4 columns (from the extensive list) are present before interactions
            # This is important if df_ahah was initially empty or missing columns
            for col in required_ahah_cols:
                if col not in merged_chunk.columns and col != 'LSOA21CD':
                    merged_chunk[col] = np.nan
            
            # Fill NA for key columns before interaction (choose a sensible default, e.g. 0 or mean)
            # Example:
            # cols_to_fill_zero = ['NDVI_MEAN', 'ah4pm10', 'ah4gpas', ...]
            # for col in cols_to_fill_zero:
            #    if col in merged_chunk.columns:
            #        merged_chunk[col].fillna(0, inplace=True)


            merged_chunk_with_interactions = apply_feature_interactions_subset4(merged_chunk.copy()) # Use .copy() to avoid SettingWithCopyWarning
            processed_chunks.append(merged_chunk_with_interactions)
            print(f"[REDACTED_BY_SCRIPT]")
        else:
            print(f"[REDACTED_BY_SCRIPT]")

except FileNotFoundError:
    print(f"[REDACTED_BY_SCRIPT]")
except Exception as e:
    print(f"[REDACTED_BY_SCRIPT]")


# --- Combine Processed Chunks (Optional) ---
if processed_chunks:
    final_df_subset4 = pd.concat(processed_chunks, ignore_index=True)
    print(f"[REDACTED_BY_SCRIPT]")
    print(f"[REDACTED_BY_SCRIPT]")
    print("[REDACTED_BY_SCRIPT]")
    print(final_df_subset4.head())
    
    #--- Example: Save to CSV ---
    final_df_subset4.to_csv("[REDACTED_BY_SCRIPT]", index=False)
    print("[REDACTED_BY_SCRIPT]")
else:
    print("[REDACTED_BY_SCRIPT]")

print("\nScript finished.")