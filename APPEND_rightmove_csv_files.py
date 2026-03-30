import pandas as pd
import ast
import re
import os
import time

# --- Configuration Block ---
# This is where you can toggle features and define file paths.
CONFIG = {
    "input_file_path": r"[REDACTED_BY_SCRIPT]",
    "target_file_path": r"[REDACTED_BY_SCRIPT]", # The file you want to append features to
    "output_file_path": r"[REDACTED_BY_SCRIPT]",

    # --- Feature Toggles ---
    # Set to True to include the feature, False to exclude.
    "features_to_append": {
        "property_type": False,
        "bedrooms": False,
        "bathrooms": False,
        "[REDACTED_BY_SCRIPT]": True,
        "most_recent_sale_date": False,
        "most_recent_sale_tenure": False,
        "historic_sales": False, # Master toggle for all historic sales data
    },

    # --- ML Consistency Control ---
    # Define the maximum number of past sales records to extract.
    # This ensures the output file has a consistent number of columns.
    "max_historic_sales": 0,
}

# --- Helper Functions for Data Parsing ---

def safe_save_csv(df, filepath, max_retries=3, delay=2):
    """
    Safely save a CSV file with retry logic for permission issues.
    """
    for attempt in range(max_retries):
        try:
            # First, try to create a backup filename if the original exists
            if os.path.exists(filepath):
                backup_path = filepath.replace('.csv', f'[REDACTED_BY_SCRIPT]')
                print(f"[REDACTED_BY_SCRIPT]")
                os.rename(filepath, backup_path)
            
            # Save the file
            df.to_csv(filepath, index=False)
            print(f"[REDACTED_BY_SCRIPT]")
            return True
            
        except PermissionError as e:
            print(f"[REDACTED_BY_SCRIPT]")
            if attempt < max_retries - 1:
                print(f"[REDACTED_BY_SCRIPT]'s open in Excel)")
                time.sleep(delay)
            else:
                # Final attempt - try with a different filename
                timestamp = int(time.time())
                alternative_path = filepath.replace('.csv', f'[REDACTED_BY_SCRIPT]')
                print(f"[REDACTED_BY_SCRIPT]")
                try:
                    df.to_csv(alternative_path, index=False)
                    print(f"[REDACTED_BY_SCRIPT]")
                    return True
                except Exception as final_e:
                    print(f"[REDACTED_BY_SCRIPT]")
                    return False
        except Exception as e:
            print(f"[REDACTED_BY_SCRIPT]")
            return False
    
    return False

def standardize_address(address):
    """[REDACTED_BY_SCRIPT]"""
    if not isinstance(address, str):
        return None
    return re.sub(r'[^a-z0-9]', '', address.lower())

def parse_price(price_str):
    """[REDACTED_BY_SCRIPT]"""
    if not isinstance(price_str, str):
        return None
    # Remove £, ,, and any other non-digit characters
    cleaned_price = re.sub(r'[^\d.]', '', price_str)
    return pd.to_numeric(cleaned_price, errors='coerce')

# --- !! UPDATED AND MORE ROBUST PARSING FUNCTION !! ---
def parse_sale_entry(entry_str):
    """
    Parses a single sale history entry string.
    Returns a tuple of (date, price, tenure).
    This version robustly finds the price by looking for '£' instead of relying on position.
    """
    if not entry_str or not isinstance(entry_str, str):
        return None, None, None

    # Case 1: Simple entry that is only a price (e.g., '£280,000')
    if '\n' not in entry_str and '£' in entry_str:
        price = parse_price(entry_str)
        return None, price, None

    # Case 2: Full entry with multiple lines
    if '\n' in entry_str:
        parts = entry_str.strip().split('\n')
        
        # Initialize results
        date = None
        price = None
        tenure = None

        # The date is reliably the first part
        if parts:
            date = parts[0]

        # **Smarter Price Finding**: Iterate through parts to find the one with the price
        for part in parts:
            if '£' in part:
                price = parse_price(part)
                break  # Stop after finding the first price

        # **Smarter Tenure Finding**: Check the last part for tenure keywords
        if parts:
            last_part = parts[-1]
            if 'Freehold' in last_part:
                tenure = 'Freehold'
            elif 'Leasehold' in last_part:
                tenure = 'Leasehold'
        
        return date, price, tenure
    
    # Fallback for any other format
    return None, None, None

def parse_rightmove_row(row):
    """[REDACTED_BY_SCRIPT]"""
    parsed_data = {}
    try:
        col0_list = ast.literal_eval(row.iloc[0])
        raw_address = col0_list[0]
        parsed_data['join_key'] = standardize_address(raw_address)
        parsed_data['rightmove_url'] = col0_list[1]
    except (ValueError, SyntaxError, IndexError):
        parsed_data['join_key'] = None
        parsed_data['rightmove_url'] = None

    try:
        col1_list = ast.literal_eval(row.iloc[1])
        parsed_data['full_address'] = col1_list[0].strip()
        parsed_data['property_type'] = col1_list[1].strip() if len(col1_list) > 1 and col1_list[1] else None
        parsed_data['bedrooms'] = pd.to_numeric(col1_list[2].strip(), errors='coerce') if len(col1_list) > 2 else None
        parsed_data['bathrooms'] = pd.to_numeric(col1_list[3].strip(), errors='coerce') if len(col1_list) > 3 else None
    except (ValueError, SyntaxError, IndexError):
        parsed_data.update({'full_address': None, 'property_type': None, 'bedrooms': None, 'bathrooms': None})

    try:
        col2_list = ast.literal_eval(row.iloc[2])
        sales_history = [s for s in col2_list if s]
        if sales_history:
            date, price, tenure = parse_sale_entry(sales_history[0])
            parsed_data['most_recent_sale_date'] = date
            parsed_data['[REDACTED_BY_SCRIPT]'] = price
            parsed_data['most_recent_sale_tenure'] = tenure
            historic_sales_to_process = sales_history[1:]
            for i in range(CONFIG['max_historic_sales']):
                price_col, date_col, tenure_col = f'[REDACTED_BY_SCRIPT]', f'historic_date_{i+1}', f'[REDACTED_BY_SCRIPT]'
                if i < len(historic_sales_to_process):
                    h_date, h_price, h_tenure = parse_sale_entry(historic_sales_to_process[i])
                    parsed_data[date_col], parsed_data[price_col], parsed_data[tenure_col] = h_date, h_price, h_tenure
                else:
                    parsed_data[date_col], parsed_data[price_col], parsed_data[tenure_col] = None, None, None
        else:
            parsed_data.update({'most_recent_sale_date': None, '[REDACTED_BY_SCRIPT]': None, 'most_recent_sale_tenure': None})
            for i in range(CONFIG['max_historic_sales']):
                parsed_data[f'historic_date_{i+1}'], parsed_data[f'[REDACTED_BY_SCRIPT]'], parsed_data[f'[REDACTED_BY_SCRIPT]'] = None, None, None
    except (ValueError, SyntaxError, IndexError):
        pass
    return pd.Series(parsed_data)


def main():
    """[REDACTED_BY_SCRIPT]"""
    if not os.path.exists(CONFIG["input_file_path"]) or not os.path.exists(CONFIG["target_file_path"]):
        exit()

    print("Loading data...")
    try:
        raw_df = pd.read_csv(CONFIG["input_file_path"], header=None, engine='python')
        target_df = pd.read_csv(CONFIG["target_file_path"])
        print(f"[REDACTED_BY_SCRIPT]")
        print(f"[REDACTED_BY_SCRIPT]")
    except FileNotFoundError as e:
        print(f"[REDACTED_BY_SCRIPT]")
        return

    print("[REDACTED_BY_SCRIPT]")
    parsed_df = raw_df.apply(parse_rightmove_row, axis=1)
    parsed_df.dropna(subset=['join_key'], inplace=True)
    print("Parsing complete.")

    # --- 4. Prepare for Merging (UPDATED LOGIC) ---
    print("[REDACTED_BY_SCRIPT]")
    
    # --- ** NEW: Dynamically find the address column in the target file ** ---
    address_col_name = None
    # Priority 1: Search for a column with "address" in its name
    for col in target_df.columns:
        if 'address' in col.lower():
            address_col_name = col
            print(f"[REDACTED_BY_SCRIPT]'{address_col_name}'")
            break
    
    # Priority 2: If not found, fall back to the first column
    if address_col_name is None:
        if not target_df.columns.empty:
            address_col_name = target_df.columns[0]
            print(f"[REDACTED_BY_SCRIPT]'address'[REDACTED_BY_SCRIPT]'{address_col_name}'[REDACTED_BY_SCRIPT]")
        else:
            print("[REDACTED_BY_SCRIPT]")
            return
            
    # Create the standardized join key using the dynamically found column
    target_df['join_key'] = target_df[address_col_name].apply(standardize_address)
    # --- ** END OF NEW LOGIC ** ---

    print("[REDACTED_BY_SCRIPT]")
    # Keep track of original columns before merge
    original_target_columns = target_df.columns.drop('join_key').tolist()
    merged_df = pd.merge(target_df, parsed_df, on='join_key', how='left')
    print(f"[REDACTED_BY_SCRIPT]")

    print("[REDACTED_BY_SCRIPT]")
    # Start with the original columns from the target file
    final_columns = original_target_columns

    toggles = CONFIG['features_to_append']
    if toggles.get('property_type'): final_columns.append('property_type')
    if toggles.get('bedrooms'): final_columns.append('bedrooms')
    if toggles.get('bathrooms'): final_columns.append('bathrooms')
    if toggles.get('[REDACTED_BY_SCRIPT]'): final_columns.append('[REDACTED_BY_SCRIPT]')
    if toggles.get('most_recent_sale_date'): final_columns.append('most_recent_sale_date')
    if toggles.get('most_recent_sale_tenure'): final_columns.append('most_recent_sale_tenure')

    if toggles.get('historic_sales'):
        for i in range(CONFIG['max_historic_sales']):
            final_columns.extend([f'[REDACTED_BY_SCRIPT]', f'historic_date_{i+1}', f'[REDACTED_BY_SCRIPT]'])

    final_columns_exist = [col for col in final_columns if col in merged_df.columns]
    final_df = merged_df[final_columns_exist]

    output_path = CONFIG["output_file_path"]
    print(f"[REDACTED_BY_SCRIPT]'{output_path}'...")
    
    # Use the safe save function instead of direct to_csv
    if safe_save_csv(final_df, output_path):
        print("[REDACTED_BY_SCRIPT]")
        print(f"[REDACTED_BY_SCRIPT]")
    else:
        print("[REDACTED_BY_SCRIPT]")
        print("You can try:")
        print("1. Close the file if it's open in Excel or another program")
        print("[REDACTED_BY_SCRIPT]")
        print("[REDACTED_BY_SCRIPT]")

if __name__ == "__main__":
    main()