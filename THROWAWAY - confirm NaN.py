# import pyarrow.parquet as pq

# # Open parquet file and read in batches
# parquet_file = pq.ParquetFile('[REDACTED_BY_SCRIPT]')
# total_non_nan = 0

# # Process in batches to save memory
# for batch in parquet_file.iter_batches(batch_size=10000, columns=['latitude']):
#     latitude_series = batch['latitude'].to_pandas()
#     total_non_nan += latitude_series.notna().sum()

# print(f"[REDACTED_BY_SCRIPT]'latitude' column: {total_non_nan}")




import pandas as pd
import os

# --- Configuration ---
# The original dataset that was the 'left' side of the merge
main_input_path = r"[REDACTED_BY_SCRIPT]"

# The file that was supposed to provide the coordinates
coords_input_path = r"[REDACTED_BY_SCRIPT]"

# --- Main Diagnostic Function ---
def diagnose_postcode_mismatch():
    """
    Loads postcodes from two files, normalizes them, and checks for overlap.
    """
    print("[REDACTED_BY_SCRIPT]")
    
    try:
        # --- Load Postcodes from Main Dataset ---
        print(f"[REDACTED_BY_SCRIPT]")
        # Efficiently load only the 'postcode' column
        main_postcodes = pd.read_parquet(main_input_path, columns=['postcode'])['postcode']
        # Normalize: Convert to uppercase and remove all whitespace
        main_postcodes_normalized = set(main_postcodes.str.upper().str.replace(r'\s+', '', regex=True).dropna())
        print(f"[REDACTED_BY_SCRIPT]")

        # --- Load Postcodes from Coordinates File ---
        print(f"[REDACTED_BY_SCRIPT]")
        # The postcode column in this file is named 'pcds'
        coords_postcodes = pd.read_csv(coords_input_path, usecols=['pcds'])['pcds']
        # Normalize: Convert to uppercase and remove all whitespace
        coords_postcodes_normalized = set(coords_postcodes.str.upper().str.replace(r'\s+', '', regex=True).dropna())
        print(f"[REDACTED_BY_SCRIPT]")

        # --- Perform the Check ---
        print("[REDACTED_BY_SCRIPT]")
        # Find the intersection between the two sets
        matching_postcodes = main_postcodes_normalized.intersection(coords_postcodes_normalized)
        
        # --- Report Findings ---
        print("[REDACTED_BY_SCRIPT]")
        if not matching_postcodes:
            print("[REDACTED_BY_SCRIPT]")
            print("[REDACTED_BY_SCRIPT]")
        else:
            match_percentage = (len(matching_postcodes) / len(main_postcodes_normalized)) * 100
            print(f"[REDACTED_BY_SCRIPT]")
            print(f"[REDACTED_BY_SCRIPT]")
            if match_percentage < 95:
                print("[REDACTED_BY_SCRIPT]")

    except FileNotFoundError as e:
        print(f"[REDACTED_BY_SCRIPT]")
    except KeyError as e:
        print(f"[REDACTED_BY_SCRIPT]")
    except Exception as e:
        print(f"[REDACTED_BY_SCRIPT]")

if __name__ == "__main__":
    diagnose_postcode_mismatch()