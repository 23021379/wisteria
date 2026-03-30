import pandas as pd
import numpy as np
import re
import ast # For safely evaluating string literals like lists
from functools import reduce # For merging multiple dataframes

# --- Constants ---
ADDRESS_COL = '[REDACTED_BY_SCRIPT]'
SQFT_TO_SQM = 0.092903
CURRENT_YEAR = 2024

# --- File Paths (REPLACE THESE WITH YOUR ACTUAL PATHS if different from the error log) ---
original_rightmove_file = 'Rightmove.csv' # Your main source file
file_paths = {
    'homipi': '[REDACTED_BY_SCRIPT]',
    'mouseprice': '[REDACTED_BY_SCRIPT]',
    'bnl': '[REDACTED_BY_SCRIPT]',
    'chimnie': '[REDACTED_BY_SCRIPT]',
    'streetscan': '[REDACTED_BY_SCRIPT]'
}

# --- Suffixes for merged columns ---
suffixes = {
    'homipi': '_hm',
    'mouseprice': '_mp',
    'bnl': '_bnl',
    'chimnie': '_ch',
    'streetscan': '_ss'
}

# --- Helper Functions ---
def to_numeric_safe(series):
    return pd.to_numeric(series, errors='coerce')

def parse_epc_value(epc_str):
    if pd.isna(epc_str):
        return np.nan
    epc_str = str(epc_str).strip().upper()
    epc_map = {'A': 7, 'B': 6, 'C': 5, 'D': 4, 'E': 3, 'F': 2, 'G': 1}
    if epc_str in epc_map:
        return epc_map[epc_str]
    try:
        val = int(epc_str)
        return val
    except ValueError:
        try:
            val = float(epc_str)
            return val
        except ValueError:
            return np.nan

def parse_rental_value_pcm(rental_text):
    if pd.isna(rental_text):
        return np.nan
    match = re.search(r'[REDACTED_BY_SCRIPT]', str(rental_text), re.IGNORECASE)
    if match:
        try:
            return float(match.group(1).replace(',', ''))
        except ValueError:
            return np.nan
    return np.nan

def clean_address(address_str):
    """
    Cleans and standardizes address strings.
    The target format is the "processed" address format from Rightmove.csv's second field,
    which is already well-formatted (e.g., "[REDACTED_BY_SCRIPT]").
    """
    if pd.isna(address_str):
        return np.nan
    # Convert to string, make lowercase, strip leading/trailing whitespace
    cleaned = str(address_str).lower().strip()
    # Normalize all whitespace (spaces, tabs, newlines) to a single space
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned


# --- Load Original Rightmove Data and Extract Addresses ---
print(f"[REDACTED_BY_SCRIPT]")
try:
    # The Rightmove CSV has string representations of lists in its columns.
    # We need to read all columns initially to access the second field.
    # We expect at least 2 "fields" (string representations of lists) per row.
    df_original_rm_raw = pd.read_csv(original_rightmove_file, header=None, dtype=str) # Read all as string

    def extract_processed_address_from_row(row):
        try:
            # The second field (index 1) contains the list with the processed address
            if len(row) > 1 and pd.notna(row[1]):
                # raw_field_1 is something like "['45, Inkerman Road...', '\nHouse', ...]"
                raw_field_1_str = row[1]
                list_val = ast.literal_eval(raw_field_1_str)
                if isinstance(list_val, list) and len(list_val) > 0:
                    return list_val[0] # The first element of *this* list is the processed address
        except (ValueError, SyntaxError, IndexError) as e:
            # print(f"[REDACTED_BY_SCRIPT]")
            return np.nan
        return np.nan

    # Apply the extraction function row-wise
    df_original_rm = pd.DataFrame()
    df_original_rm[ADDRESS_COL] = df_original_rm_raw.apply(extract_processed_address_from_row, axis=1)
    df_original_rm[ADDRESS_COL] = df_original_rm[ADDRESS_COL].apply(clean_address)
    df_original_rm = df_original_rm[[ADDRESS_COL]].drop_duplicates().dropna(subset=[ADDRESS_COL]) # Keep only unique, non-null cleaned addresses
    print(f"[REDACTED_BY_SCRIPT]")

except FileNotFoundError:
    print(f"[REDACTED_BY_SCRIPT]'{original_rightmove_file}'[REDACTED_BY_SCRIPT]")
    df_original_rm = pd.DataFrame(columns=[ADDRESS_COL])
except Exception as e:
    print(f"[REDACTED_BY_SCRIPT]")
    df_original_rm = pd.DataFrame(columns=[ADDRESS_COL])


# The BNL input file is headerless and tab-separated with 59 columns.
# These are the base names (without suffix) derived from your final output header.

# --- Load Derived DataFrames ---
dfs = {}
for name, path in file_paths.items():
    try:
        if name == 'bnl':
            # --- Definitive Fix for the Malformed BNL File ---

            # Step 1: Read just the header row to get the 58 original names.
            header_names = pd.read_csv(path, nrows=0).columns.tolist()

            # Step 2: Create a new list of 59 names by adding a placeholder for the missing column.
            # We assume the extra column is at the end.
            full_column_names = header_names + ['[REDACTED_BY_SCRIPT]']
            print(f"[REDACTED_BY_SCRIPT]")

            # Step 3: Read the actual data, skipping the original (short) header
            # and applying our new, correct list of 59 names.
            current_df = pd.read_csv(
                path,
                header=None,         # The data has no header
                skiprows=1,          # Skip the original malformed header row
                names=full_column_names  # Apply our corrected list of names
            )

        else:
            # Use the default parser for the other, well-formed files.
            current_df = pd.read_csv(path)

        if current_df.empty:
            continue

        # --- This common logic now applies correctly to ALL dataframes ---

        # 1. Standardize the address column name
        if ADDRESS_COL not in current_df.columns and len(current_df.columns) > 0:
            current_df.rename(columns={current_df.columns[0]: ADDRESS_COL}, inplace=True)

        # 2. Clean the address column for a reliable merge key
        current_df[ADDRESS_COL] = current_df[ADDRESS_COL].apply(clean_address)

        # 3. Remove duplicates within this file
        current_df.drop_duplicates(subset=[ADDRESS_COL], keep='first', inplace=True)

        # 4. Add the suffix to all columns (except address) and store
        dfs[name] = current_df.rename(columns=lambda c: c + suffixes[name] if c != ADDRESS_COL else c)
        print(f"[REDACTED_BY_SCRIPT]")

    except FileNotFoundError:
        print(f"[REDACTED_BY_SCRIPT]")
    except Exception as e:
        print(f"[REDACTED_BY_SCRIPT]")


# --- [DEBUG] Inspect the dataframes before merging ---
print("[REDACTED_BY_SCRIPT]")
for name, df in dfs.items():
    if not df.empty:
        print(f"DataFrame '{name}':")
        print(f"  Shape: {df.shape}")
        print(f"[REDACTED_BY_SCRIPT]")
        # For bnl, also show some actual data to confirm it loaded
        if name == 'bnl' and len(df.columns) > 4:
            print(f"[REDACTED_BY_SCRIPT]")
        print("-" * 20)

# Check for matching addresses between the two key dataframes
if 'homipi' in dfs and 'bnl' in dfs and not dfs['homipi'].empty and not dfs['bnl'].empty:
    homipi_addresses = set(dfs['homipi'][ADDRESS_COL].dropna())
    bnl_addresses = set(dfs['bnl'][ADDRESS_COL].dropna())
    common_addresses = homipi_addresses.intersection(bnl_addresses)
    
    print(f"[REDACTED_BY_SCRIPT]'homipi' and 'bnl'.")
    
    if len(common_addresses) > 0:
        print(f"[REDACTED_BY_SCRIPT]")
    else:
        print("[REDACTED_BY_SCRIPT]")
        print("[REDACTED_BY_SCRIPT]")
        
print("--- End Diagnostics ---\n")

# --- Merge Derived DataFrames ---
valid_dfs = [df for df in dfs.values() if isinstance(df, pd.DataFrame) and (not df.empty or ADDRESS_COL in df.columns)]


if not valid_dfs:
    print("[REDACTED_BY_SCRIPT]")
    if not df_original_rm.empty:
        df_merged = df_original_rm.copy()
        print(f"[REDACTED_BY_SCRIPT]")
    else:
        print("[REDACTED_BY_SCRIPT]")
        exit()
elif len(valid_dfs) == 1:
    df_merged = valid_dfs[0].copy()
    df_key = [key for key, val in dfs.items() if val is valid_dfs[0]][0]
    print(f"[REDACTED_BY_SCRIPT]")
else:
    # Ensure all DataFrames in valid_dfs have the ADDRESS_COL before merging
    for i, df_to_merge in enumerate(valid_dfs):
        if ADDRESS_COL not in df_to_merge.columns:
            # This case should ideally be handled during loading, but as a fallback:
            print(f"[REDACTED_BY_SCRIPT]'{ADDRESS_COL}'[REDACTED_BY_SCRIPT]")
            # Create a placeholder to avoid merge error if it's truly empty without the column
            valid_dfs[i] = pd.DataFrame(columns=[ADDRESS_COL]) if df_to_merge.empty else df_to_merge

    # Filter again in case placeholders were added
    valid_dfs = [df for df in valid_dfs if ADDRESS_COL in df.columns and not df.empty]
    if not valid_dfs:
        print("[REDACTED_BY_SCRIPT]")
        if not df_original_rm.empty:
            df_merged = df_original_rm.copy()
            print(f"[REDACTED_BY_SCRIPT]")
        else:
            print("[REDACTED_BY_SCRIPT]")
            exit()
    elif len(valid_dfs) == 1: # Could become 1 after filtering
        df_merged = valid_dfs[0].copy()
        print(f"[REDACTED_BY_SCRIPT]")
    else:
        df_merged = reduce(lambda left, right: pd.merge(left, right, on=ADDRESS_COL, how='outer'), valid_dfs)


print(f"[REDACTED_BY_SCRIPT]")

if df_merged.empty and df_original_rm.empty:
    print("[REDACTED_BY_SCRIPT]")
    exit()
elif df_merged.empty and not df_original_rm.empty:
    print("[REDACTED_BY_SCRIPT]")
    df_merged = df_original_rm.copy()


# --- Add Missing Addresses from Original Rightmove Data ---
if not df_original_rm.empty:
    if df_merged.empty or ADDRESS_COL not in df_merged.columns:
        print("[REDACTED_BY_SCRIPT]")
        df_merged = df_original_rm.copy()
    else:
        # Ensure ADDRESS_COL is present for the set operation
        merged_addresses_series = df_merged[ADDRESS_COL].dropna().astype(str).unique()
        original_rm_addresses_series = df_original_rm[ADDRESS_COL].dropna().astype(str).unique()

        merged_addresses_set = set(merged_addresses_series)
        original_rm_addresses_set = set(original_rm_addresses_series)

        missing_addresses = list(original_rm_addresses_set - merged_addresses_set)

        if missing_addresses:
            print(f"[REDACTED_BY_SCRIPT]")
            df_missing = pd.DataFrame(missing_addresses, columns=[ADDRESS_COL])
            df_merged = pd.concat([df_merged, df_missing], ignore_index=True, sort=False)
            print(f"[REDACTED_BY_SCRIPT]")
        else:
            print(f"[REDACTED_BY_SCRIPT]")
else:
    print(f"[REDACTED_BY_SCRIPT]")


# --- Define specific column names (after suffixing) ---
# (Copied from previous script, ensure they are still relevant)
# Homipi
hm_price_col = '[REDACTED_BY_SCRIPT]' + suffixes['homipi']
hm_price_low_col = '[REDACTED_BY_SCRIPT]' + suffixes['homipi']
hm_price_high_col = '[REDACTED_BY_SCRIPT]' + suffixes['homipi']
hm_last_sold_price_col = 'last_sold_price_gbp' + suffixes['homipi']
hm_last_sold_year_col = '[REDACTED_BY_SCRIPT]' + suffixes['homipi']
hm_curr_epc_col = '[REDACTED_BY_SCRIPT]' + suffixes['homipi']
hm_pot_epc_col = '[REDACTED_BY_SCRIPT]' + suffixes['homipi']
hm_bedrooms_col = '[REDACTED_BY_SCRIPT]' + suffixes['homipi']
hm_area_sqm_col = '[REDACTED_BY_SCRIPT]' + suffixes['homipi']

# Mouseprice
mp_price_col = '[REDACTED_BY_SCRIPT]' + suffixes['mouseprice']
mp_rental_text_col = '[REDACTED_BY_SCRIPT]' + suffixes['mouseprice']
mp_area_sqm_col = '[REDACTED_BY_SCRIPT]' + suffixes['mouseprice']
mp_built_start_col = '[REDACTED_BY_SCRIPT]' + suffixes['mouseprice']
mp_built_end_col = '[REDACTED_BY_SCRIPT]' + suffixes['mouseprice']

# BNL
bnl_price_col = '[REDACTED_BY_SCRIPT]' + suffixes['bnl']
bnl_area_sqft_col = '[REDACTED_BY_SCRIPT]' + suffixes['bnl']
bnl_school_rating_col = '[REDACTED_BY_SCRIPT]' + suffixes['bnl']

# Chimnie
ch_safety_local_col = '[REDACTED_BY_SCRIPT]' + suffixes['chimnie']
ch_safety_wider_col = '[REDACTED_BY_SCRIPT]' + suffixes['chimnie']
ch_family_local_col = '[REDACTED_BY_SCRIPT]' + suffixes['chimnie']

# Streetscan
ss_income_col = '[REDACTED_BY_SCRIPT]' + suffixes['streetscan']
ss_avg_price_all_col = '[REDACTED_BY_SCRIPT]' + suffixes['streetscan']
ss_crime_rank_col = '[REDACTED_BY_SCRIPT]' + suffixes['streetscan']


# --- Pre-process specific columns (apply parsing) ---
print("[REDACTED_BY_SCRIPT]")
if hm_curr_epc_col in df_merged.columns:
    df_merged[hm_curr_epc_col + '_numeric'] = df_merged[hm_curr_epc_col].apply(parse_epc_value)
if hm_pot_epc_col in df_merged.columns:
    df_merged[hm_pot_epc_col + '_numeric'] = df_merged[hm_pot_epc_col].apply(parse_epc_value)
if mp_rental_text_col in df_merged.columns:
    df_merged[mp_rental_text_col + '_pcm_numeric'] = df_merged[mp_rental_text_col].apply(parse_rental_value_pcm)

# --- Feature Interaction Creation ---
print("[REDACTED_BY_SCRIPT]")

def safe_divide(numerator_series, denominator_series):
    num = pd.to_numeric(numerator_series, errors='coerce')
    den = pd.to_numeric(denominator_series, errors='coerce')
    den_safe = den.replace(0, np.nan)
    return num / den_safe

def all_cols_exist(df, cols_list):
    return all(c in df.columns for c in cols_list)

# == Within-Dataset Interactions ==
if all_cols_exist(df_merged, [hm_price_high_col, hm_price_low_col]):
    df_merged['[REDACTED_BY_SCRIPT]'] = to_numeric_safe(df_merged[hm_price_high_col]) - to_numeric_safe(df_merged[hm_price_low_col])
if all_cols_exist(df_merged, [hm_price_col, hm_last_sold_price_col]):
    df_merged['[REDACTED_BY_SCRIPT]'] = safe_divide(df_merged[hm_price_col], df_merged[hm_last_sold_price_col])
if hm_last_sold_year_col in df_merged.columns:
    df_merged['[REDACTED_BY_SCRIPT]'] = CURRENT_YEAR - to_numeric_safe(df_merged[hm_last_sold_year_col])

hm_pot_epc_numeric_col = hm_pot_epc_col + '_numeric'
hm_curr_epc_numeric_col = hm_curr_epc_col + '_numeric'
if all_cols_exist(df_merged, [hm_pot_epc_numeric_col, hm_curr_epc_numeric_col]):
    df_merged['[REDACTED_BY_SCRIPT]'] = to_numeric_safe(df_merged[hm_pot_epc_numeric_col]) - to_numeric_safe(df_merged[hm_curr_epc_numeric_col])
if all_cols_exist(df_merged, [hm_bedrooms_col, hm_area_sqm_col]):
    df_merged['[REDACTED_BY_SCRIPT]'] = safe_divide(df_merged[hm_bedrooms_col], df_merged[hm_area_sqm_col])

mp_rental_pcm_numeric_col = mp_rental_text_col + '_pcm_numeric'
if all_cols_exist(df_merged, [mp_rental_pcm_numeric_col, mp_price_col]):
    mp_monthly_rental_num = to_numeric_safe(df_merged[mp_rental_pcm_numeric_col])
    df_merged['[REDACTED_BY_SCRIPT]'] = mp_monthly_rental_num * 12
    df_merged['[REDACTED_BY_SCRIPT]'] = safe_divide(df_merged['[REDACTED_BY_SCRIPT]'], to_numeric_safe(df_merged[mp_price_col])) * 100
if all_cols_exist(df_merged, [mp_built_end_col, mp_built_start_col]):
    start_year = to_numeric_safe(df_merged[mp_built_start_col].replace('0', np.nan).replace(0, np.nan))
    end_year = to_numeric_safe(df_merged[mp_built_end_col].replace('0', np.nan).replace(0, np.nan))
    df_merged['[REDACTED_BY_SCRIPT]'] = end_year - start_year
if all_cols_exist(df_merged, [mp_price_col, mp_area_sqm_col]):
    df_merged['INT_mp_price_per_sqm'] = safe_divide(df_merged[mp_price_col], df_merged[mp_area_sqm_col])

bnl_area_sqm_converted_col = bnl_area_sqft_col + '_converted_sqm'
if bnl_area_sqft_col in df_merged.columns:
    df_merged[bnl_area_sqm_converted_col] = to_numeric_safe(df_merged[bnl_area_sqft_col]) * SQFT_TO_SQM
if all_cols_exist(df_merged, [bnl_price_col, bnl_area_sqft_col]):
    df_merged['[REDACTED_BY_SCRIPT]'] = safe_divide(df_merged[bnl_price_col], df_merged[bnl_area_sqft_col])

if all_cols_exist(df_merged, [ch_safety_local_col, ch_safety_wider_col]):
    df_merged['[REDACTED_BY_SCRIPT]'] = to_numeric_safe(df_merged[ch_safety_local_col]) - to_numeric_safe(df_merged[ch_safety_wider_col])

if all_cols_exist(df_merged, [ss_income_col, ss_avg_price_all_col]):
    df_merged['[REDACTED_BY_SCRIPT]'] = safe_divide(df_merged[ss_income_col], df_merged[ss_avg_price_all_col])

# == Cross-Dataset Interactions ==
price_cols_for_avg = [col for col in [hm_price_col, mp_price_col, bnl_price_col] if col in df_merged.columns]
if len(price_cols_for_avg) > 0:
    numeric_prices_df = df_merged[price_cols_for_avg].apply(pd.to_numeric, errors='coerce')
    df_merged['[REDACTED_BY_SCRIPT]'] = numeric_prices_df.mean(axis=1, skipna=True)
    if len(price_cols_for_avg) > 1:
         df_merged['[REDACTED_BY_SCRIPT]'] = numeric_prices_df.std(axis=1, skipna=True)

area_sqm_cols_for_avg = [col for col in [hm_area_sqm_col, mp_area_sqm_col, bnl_area_sqm_converted_col] if col in df_merged.columns]
if len(area_sqm_cols_for_avg) > 0:
    numeric_areas_df = df_merged[area_sqm_cols_for_avg].apply(pd.to_numeric, errors='coerce')
    df_merged['[REDACTED_BY_SCRIPT]'] = numeric_areas_df.mean(axis=1, skipna=True)

if all_cols_exist(df_merged, [hm_price_col, ss_avg_price_all_col]):
    df_merged['[REDACTED_BY_SCRIPT]'] = safe_divide(df_merged[hm_price_col], df_merged[ss_avg_price_all_col])

if all_cols_exist(df_merged, [bnl_school_rating_col, ch_family_local_col]):
    bnl_rating_numeric = to_numeric_safe(df_merged[bnl_school_rating_col])
    inverted_bnl_rating = 5 - bnl_rating_numeric
    chimnie_family_numeric = to_numeric_safe(df_merged[ch_family_local_col])
    df_merged['[REDACTED_BY_SCRIPT]'] = inverted_bnl_rating * chimnie_family_numeric

if all_cols_exist(df_merged, [ch_safety_local_col, ss_crime_rank_col]):
    chimnie_safety_numeric = to_numeric_safe(df_merged[ch_safety_local_col])
    ss_crime_rank_numeric = to_numeric_safe(df_merged[ss_crime_rank_col])
    streetscan_safety_component = (11 - ss_crime_rank_numeric)
    df_merged['[REDACTED_BY_SCRIPT]'] = chimnie_safety_numeric + streetscan_safety_component

# --- Save Output ---
output_file = '[REDACTED_BY_SCRIPT]' # Changed output name
try:
    if not df_merged.empty:
        df_merged.to_csv(output_file, index=False)
        print(f"[REDACTED_BY_SCRIPT]")
        print(f"[REDACTED_BY_SCRIPT]")
        print(f"[REDACTED_BY_SCRIPT]")
        print(f"[REDACTED_BY_SCRIPT]")
    else:
        print("[REDACTED_BY_SCRIPT]")
except Exception as e:
    print(f"[REDACTED_BY_SCRIPT]")