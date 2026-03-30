import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import time
import os

# --- Configuration ---
# IMPORTANT: Use raw string (r"...") or double backslashes (\\) for Windows paths.
FILE_PATH = r"[REDACTED_BY_SCRIPT]"
HPI_FILE_PATH = r"[REDACTED_BY_SCRIPT]"

OUTPUT_FILE_PATH = r"[REDACTED_BY_SCRIPT]"

# Columns from the property sales data file
SALES_COLUMN_NAMES = [
    'TransactionID', 'Price', 'DateOfTransfer', 'Postcode', 'PropertyType',
    'OldNew', 'Duration', 'PAON', 'SAON', 'Street', 'Locality', 'TownCity',
    'District', 'County', 'PPD_Category', 'RecordStatus'
]

# Mapping from sales data PropertyType to HPI data columns
HPI_COLUMN_MAP = {
    'D': '[REDACTED_BY_SCRIPT]',
    'S': '[REDACTED_BY_SCRIPT]',
    'T': '[REDACTED_BY_SCRIPT]',
    'F': '[REDACTED_BY_SCRIPT]',
    'ALL': '[REDACTED_BY_SCRIPT]' # Fallback
}


CHUNK_SIZE = 1_000_000
START_YEAR = 1995
END_YEAR = 2023
MAX_PRICE_FILTER = 20_000_000

# --- HPI Data Loading ---
def load_hpi_data(file_path):
    """
    Loads and processes the UK HPI data from the official CSV file.

    Returns:
        A dictionary structured as {year: {property_type_code: hpi_value}}.
    """
    print(f"[REDACTED_BY_SCRIPT]'{file_path}'...")
    try:
        hpi_df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"[REDACTED_BY_SCRIPT]'{file_path}'.")
        return None

    hpi_df['Year'] = pd.to_datetime(hpi_df['Period'], errors='coerce').dt.year
    hpi_df.dropna(subset=['Year'], inplace=True)
    hpi_df['Year'] = hpi_df['Year'].astype(int)

    # Filter for UK-wide data and relevant years
    hpi_df = hpi_df[hpi_df['Name'] == 'United Kingdom']
    hpi_df = hpi_df[hpi_df['Year'].between(START_YEAR, END_YEAR)]

    hpi_columns = list(HPI_COLUMN_MAP.values())
    for col in hpi_columns:
        hpi_df[col] = pd.to_numeric(hpi_df[col], errors='coerce')

    # Calculate the mean HPI for each year
    yearly_hpi = hpi_df.groupby('Year')[hpi_columns].mean().reset_index()

    # Convert the DataFrame to the desired nested dictionary format
    hpi_data = {}
    type_code_map = {v: k for k, v in HPI_COLUMN_MAP.items()} # Reverse map for lookup

    for _, row in yearly_hpi.iterrows():
        year = row['Year']
        hpi_data[year] = {}
        for hpi_col_name, type_code in type_code_map.items():
            if pd.notna(row[hpi_col_name]):
                hpi_data[year][type_code] = row[hpi_col_name]
    
    print("[REDACTED_BY_SCRIPT]")
    return hpi_data

# --- Main Processing Logic ---
def analyze_house_prices():
    """[REDACTED_BY_SCRIPT]"""
    if not os.path.exists(FILE_PATH):
        print(f"[REDACTED_BY_SCRIPT]'{FILE_PATH}'")
        return

    # Load HPI data first
    hpi_data = load_hpi_data(HPI_FILE_PATH)
    if not hpi_data:
        print("[REDACTED_BY_SCRIPT]")
        return
        
    LATEST_HPI_BY_TYPE = hpi_data.get(END_YEAR)
    if not LATEST_HPI_BY_TYPE:
        print(f"[REDACTED_BY_SCRIPT]")
        return

    print(f"[REDACTED_BY_SCRIPT]'{FILE_PATH}'...")
    start_time = time.time()

    postcode_data = defaultdict(lambda: defaultdict(list))

    print("[REDACTED_BY_SCRIPT]")
    try:
        chunk_iterator = pd.read_csv(
            FILE_PATH,
            header=None,
            names=SALES_COLUMN_NAMES,
            usecols=['Price', 'DateOfTransfer', 'Postcode', 'PropertyType'],
            chunksize=CHUNK_SIZE,
            encoding='latin-1',
            low_memory=False
        )

        for i, chunk in enumerate(chunk_iterator):
            print(f"[REDACTED_BY_SCRIPT]")

            chunk['Price'] = pd.to_numeric(chunk['Price'], errors='coerce')
            chunk.dropna(subset=['Postcode', 'Price', 'PropertyType'], inplace=True)
            chunk['Price'] = chunk['Price'].astype(np.int64)

            if MAX_PRICE_FILTER:
                chunk = chunk[chunk['Price'] <= MAX_PRICE_FILTER].copy()

            chunk['Year'] = pd.to_datetime(chunk['DateOfTransfer'], errors='coerce').dt.year
            chunk.dropna(subset=['Year'], inplace=True)
            chunk['Year'] = chunk['Year'].astype(int)

            # --- FILTERS ---
            chunk = chunk[chunk['Year'].between(START_YEAR, END_YEAR)]
            chunk = chunk[chunk['PropertyType'] != 'O']

            chunk['Postcode'] = chunk['Postcode'].str.strip()

            for row in chunk.itertuples(index=False):
                postcode_data[row.Postcode][row.PropertyType].append((row.Price, row.Year))

    except Exception as e:
        print(f"[REDACTED_BY_SCRIPT]")
        return

    if not postcode_data:
        print("No data was aggregated.")
        return

    print(f"[REDACTED_BY_SCRIPT]")
    print("[REDACTED_BY_SCRIPT]")

    results = []
    for postcode, type_data in postcode_data.items():
        for prop_type, sales in type_data.items():
            if not sales:
                continue

            prices, years = map(np.array, zip(*sales))
            
            # --- HPI Adjustment Logic ---
            # Get the latest HPI for this property type, fallback to 'ALL'
            latest_hpi = LATEST_HPI_BY_TYPE.get(prop_type, LATEST_HPI_BY_TYPE.get('ALL'))
            
            # Create adjustment factors for each sale year
            hpi_factors = []
            for year in years:
                year_hpi_data = hpi_data.get(year, {})
                # Use HPI for the specific type, or fallback to 'ALL' for that year
                current_hpi = year_hpi_data.get(prop_type, year_hpi_data.get('ALL', latest_hpi))
                factor = latest_hpi / current_hpi if current_hpi else 1.0
                hpi_factors.append(factor)

            adjusted_prices = prices * np.array(hpi_factors)
            # --- End HPI Adjustment ---

            count = len(prices)
            avg_price = np.mean(adjusted_prices)
            median_price = np.median(adjusted_prices)
            yearly_counts = Counter(years)

            result_row = {
                'Postcode': postcode,
                'PropertyType': prop_type,
                'Sale_Count': count,
                '[REDACTED_BY_SCRIPT]': round(avg_price, 2),
                '[REDACTED_BY_SCRIPT]': round(median_price, 2),
            }

            for year in range(START_YEAR, END_YEAR + 1):
                result_row[f'{year}_sale_count'] = yearly_counts.get(year, 0)

            results.append(result_row)

    print("[REDACTED_BY_SCRIPT]")

    if not results:
        print("[REDACTED_BY_SCRIPT]")
        return

    final_df = pd.DataFrame(results)
    
    stat_cols = ['Postcode', 'PropertyType', 'Sale_Count', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]']
    year_cols = [f'{year}_sale_count' for year in range(START_YEAR, END_YEAR + 1)]
    final_df = final_df[stat_cols + year_cols]

    final_df.sort_values(by=['Postcode', 'PropertyType'], inplace=True)

    print("[REDACTED_BY_SCRIPT]")
    print(f"[REDACTED_BY_SCRIPT]")
    
    print("[REDACTED_BY_SCRIPT]")
    print(final_df.head().to_string())

    try:
        final_df.to_csv(OUTPUT_FILE_PATH, index=False)
        print(f"[REDACTED_BY_SCRIPT]'{OUTPUT_FILE_PATH}'")
    except Exception as e:
        print(f"[REDACTED_BY_SCRIPT]")

if __name__ == "__main__":
    analyze_house_prices()
