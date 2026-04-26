import pandas as pd
import numpy as np
import os
import re
from tqdm import tqdm

# --- Configuration (Same as before) ---
BASE_DIR = "[REDACTED_BY_SCRIPT]"
PROPERTY_FOLDERS_DIR = os.path.join(BASE_DIR, "JSONrightmove")
GLOBAL_SUBSETS_DIR = os.path.join(BASE_DIR, "global subsets")
OUTPUT_DIR = GLOBAL_SUBSETS_DIR
CHUNK_SIZE = 50000

UNPROCESSED_LOOKUP_FILE = {
    "path": os.path.join(GLOBAL_SUBSETS_DIR, "[REDACTED_BY_SCRIPT]"),
    "postcode_col": "pcds_original"
}

SUBSET_FILES_CONFIG = {
    "subset1": {
        "processed": os.path.join(GLOBAL_SUBSETS_DIR, "[REDACTED_BY_SCRIPT]"),
        "output": os.path.join(OUTPUT_DIR, "[REDACTED_BY_SCRIPT]")
    },
    "subset2": {
        "processed": os.path.join(GLOBAL_SUBSETS_DIR, "[REDACTED_BY_SCRIPT]"),
        "output": os.path.join(OUTPUT_DIR, "[REDACTED_BY_SCRIPT]")
    },
    "subset3": {
        "processed": os.path.join(GLOBAL_SUBSETS_DIR, "[REDACTED_BY_SCRIPT]"),
        "output": os.path.join(OUTPUT_DIR, "[REDACTED_BY_SCRIPT]")
    },
    "subset4": {
        "processed": os.path.join(GLOBAL_SUBSETS_DIR, "[REDACTED_BY_SCRIPT]"),
        "output": os.path.join(OUTPUT_DIR, "[REDACTED_BY_SCRIPT]")
    },
    "subset5": {
        "processed": os.path.join(GLOBAL_SUBSETS_DIR, "[REDACTED_BY_SCRIPT]"),
        "output": os.path.join(OUTPUT_DIR, "[REDACTED_BY_SCRIPT]")
    },
}

# --- Helper Functions (Same as before) ---
def standardize_postcode(pc):
    if not isinstance(pc, str): return None
    pc_nospace = re.sub(r'\s+', '', pc).upper()
    if len(pc_nospace) < 4: return None
    return f"[REDACTED_BY_SCRIPT]"

def extract_postcode_from_address(address_string):
    match = re.search(r'[REDACTED_BY_SCRIPT]', address_string.strip(), re.IGNORECASE)
    if match: return standardize_postcode(match.group(1))
    return None

def build_postcode_lookup_map(filepath, postcode_col):
    print(f"[REDACTED_BY_SCRIPT]'{os.path.basename(filepath)}'...")
    try:
        # First, check what columns are available
        df_sample = pd.read_csv(filepath, nrows=0)  # Read just the header
        print(f"[REDACTED_BY_SCRIPT]")
        
        if postcode_col not in df_sample.columns:
            print(f"ERROR: Column '{postcode_col}'[REDACTED_BY_SCRIPT]")
            return None
        
        # Only load the necessary column to save memory
        df_lookup = pd.read_csv(filepath, usecols=[postcode_col], low_memory=False, on_bad_lines='skip')
        df_lookup.dropna(subset=[postcode_col], inplace=True)
        postcode_map = {
            standardize_postcode(pc): index
            for index, pc in tqdm(df_lookup[postcode_col].items(), desc="Mapping postcodes")
            if standardize_postcode(pc) is not None
        }
        print(f"[REDACTED_BY_SCRIPT]")
        return postcode_map
    except FileNotFoundError:
        print(f"[REDACTED_BY_SCRIPT]'{filepath}'"); return None
    except KeyError:
        print(f"[REDACTED_BY_SCRIPT]'{postcode_col}' not found in '{filepath}'"); return None

# --- UPDATED Core Logic Function for Parquet ---
def extract_rows_by_index_parquet(filepath, target_indices_set):
    """
    Reads a parquet file and extracts only the rows specified by their indices.
    Returns a dictionary mapping {index: row_data}.
    """
    print(f"[REDACTED_BY_SCRIPT]'{os.path.basename(filepath)}'...")
    
    try:
        # Load the entire parquet file (parquet is already optimized for this)
        df = pd.read_parquet(filepath)
        
        # Get header for creating null rows if needed
        header = df.columns.tolist()
        
        # Find which indices exist in the dataframe
        available_indices = set(df.index) & target_indices_set
        
        if not available_indices:
            print(f"[REDACTED_BY_SCRIPT]")
            return {}, header
        
        # Extract the matching rows
        matched_rows = df.loc[list(available_indices)]
        
        # Convert to dictionary mapping index to row data
        found_rows = {idx: row for idx, row in matched_rows.iterrows()}
        
        print(f"[REDACTED_BY_SCRIPT]")
        return found_rows, header
        
    except FileNotFoundError:
        print(f"[REDACTED_BY_SCRIPT]'{filepath}'")
        return {}, []
    except Exception as e:
        print(f"[REDACTED_BY_SCRIPT]'{filepath}': {str(e)}")
        return {}, []


def main():
    """[REDACTED_BY_SCRIPT]"""
    
    # STEP 1: Build the postcode-to-row_number map
    postcode_to_index_map = build_postcode_lookup_map(UNPROCESSED_LOOKUP_FILE["path"], UNPROCESSED_LOOKUP_FILE["postcode_col"])
    if not postcode_to_index_map: return

    # STEP 2: Identify all target properties and their global row indices
    print(f"[REDACTED_BY_SCRIPT]'{PROPERTY_FOLDERS_DIR}'[REDACTED_BY_SCRIPT]")
    try:
        property_folders = [f for f in os.listdir(PROPERTY_FOLDERS_DIR) if os.path.isdir(os.path.join(PROPERTY_FOLDERS_DIR, f))]
    except FileNotFoundError:
        print(f"[REDACTED_BY_SCRIPT]'{PROPERTY_FOLDERS_DIR}'"); return
    
    ordered_properties = [] # List of tuples: (full_address, postcode)
    target_indices_map = {} # {postcode: global_index}
    unmatched_postcodes = set()
    
    for folder_name in tqdm(property_folders, desc="Identifying Targets"):
        postcode = extract_postcode_from_address(folder_name)
        if not postcode: continue
        
        ordered_properties.append((folder_name, postcode))
        global_index = postcode_to_index_map.get(postcode)
        
        if global_index is not None:
            # Only add to target_indices_map if not already present to avoid overwriting
            if postcode not in target_indices_map:
                 target_indices_map[postcode] = global_index
        else:
            unmatched_postcodes.add(postcode)
    
    target_indices_set = set(target_indices_map.values())
    print(f"[REDACTED_BY_SCRIPT]")
    if unmatched_postcodes:
        print(f"[REDACTED_BY_SCRIPT]")

    # STEP 3: Process each parquet file sequentially to gather data
    aggregated_data = {key: [] for key in SUBSET_FILES_CONFIG.keys()}
    
    for subset_key, config in SUBSET_FILES_CONFIG.items():
        print(f"[REDACTED_BY_SCRIPT]'{os.path.basename(config['processed'])}' ---")
        
        found_rows_for_subset, header = extract_rows_by_index_parquet(config['processed'], target_indices_set)
        
        print(f"[REDACTED_BY_SCRIPT]")
        
        # Assemble the data in the correct order for this subset
        data_for_this_subset = []
        null_row = pd.Series([-1] * len(header), index=header) if header else pd.Series()
        
        for _, postcode in tqdm(ordered_properties, desc=f"[REDACTED_BY_SCRIPT]"):
            if postcode in unmatched_postcodes:
                data_for_this_subset.append(null_row)
            else:
                global_index = target_indices_map[postcode]
                row_data = found_rows_for_subset.get(global_index)
                if row_data is not None:
                    data_for_this_subset.append(row_data)
                else:
                    # This case should not happen if logic is correct, but is a good safeguard
                    print(f"[REDACTED_BY_SCRIPT]")
                    data_for_this_subset.append(null_row)
        
        aggregated_data[subset_key] = data_for_this_subset

    # STEP 4: Save the final assembled dataframes
    print("[REDACTED_BY_SCRIPT]")
    ordered_addresses = [prop[0] for prop in ordered_properties] # Extract full addresses for the output column
    for subset_key, config in tqdm(SUBSET_FILES_CONFIG.items(), desc="Saving Files"):
        if not aggregated_data[subset_key]:
            print(f"[REDACTED_BY_SCRIPT]")
            continue
            
        final_df = pd.DataFrame(aggregated_data[subset_key])
        final_df.insert(0, 'property_address', ordered_addresses)
        final_df.to_csv(config["output"], index=False)
        
    print(f"[REDACTED_BY_SCRIPT]")
    print(f"'{os.path.abspath(OUTPUT_DIR)}'")

if __name__ == '__main__':
    main()