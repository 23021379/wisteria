import pandas as pd
from pyproj import Transformer

# --- CONFIGURATION ---
# Path to your large CSV with 2.7 million postcodes
your_data_file = r"[REDACTED_BY_SCRIPT]"

# Path to the master lookup file you created in Step 2
master_lookup_file = r"[REDACTED_BY_SCRIPT]"

# --- PROCESSING ---

# 1. Load your data
print("[REDACTED_BY_SCRIPT]")
main_df = pd.read_csv(your_data_file, encoding='latin1', usecols=['pcds'])
# Remove any spaces from postcodes for a clean merge
main_df['pcds'] = main_df['pcds'].str.replace(' ', '')
print(f"[REDACTED_BY_SCRIPT]")

# 2. Load the master postcode lookup
print("[REDACTED_BY_SCRIPT]")
lookup_df = pd.read_csv(master_lookup_file)
# Remove spaces from the lookup postcodes as well
lookup_df['postcode'] = lookup_df['postcode'].str.replace(' ', '')
print(f"[REDACTED_BY_SCRIPT]")


# 3. Merge the two dataframes to add eastings and northings
print("[REDACTED_BY_SCRIPT]")
merged_df = pd.merge(main_df, lookup_df, left_on='pcds', right_on='postcode', how='left')


# 4. Convert coordinates
print("[REDACTED_BY_SCRIPT]")
# Define the transformation from British National Grid (EPSG:27700) to WGS84 (EPSG:4326)
transformer = Transformer.from_crs("EPSG:27700", "EPSG:4326")

# Filter out rows where the merge might have failed (i.e., eastings are null)
valid_coords = merged_df.dropna(subset=['eastings', 'northings'])
eastings = valid_coords['eastings'].values
northings = valid_coords['northings'].values

# Perform the transformation
lat, lon = transformer.transform(eastings, northings)

# Add the new coordinates back to the dataframe
merged_df.loc[valid_coords.index, 'latitude'] = lat
merged_df.loc[valid_coords.index, 'longitude'] = lon

# 5. Save the final result
output_filename = "[REDACTED_BY_SCRIPT]"
print(f"[REDACTED_BY_SCRIPT]")
# Select relevant columns before saving
final_df = merged_df[['pcds', 'latitude', 'longitude']]
final_df.to_csv(output_filename, index=False)

print("[REDACTED_BY_SCRIPT]")
print(f"[REDACTED_BY_SCRIPT]")