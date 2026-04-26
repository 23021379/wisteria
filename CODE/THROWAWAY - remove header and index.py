import pandas as pd

# --- Configuration ---
# The name of your original, messy file
input_filename = r'[REDACTED_BY_SCRIPT]'

# The name for the new, cleaned-up file
output_filename = r'[REDACTED_BY_SCRIPT]'

# --- Script Logic ---
try:
    # Step 1: Read the CSV file. 
    # We use 'header=None' to treat the first row as data, not as column names.
    print(f"[REDACTED_BY_SCRIPT]'{input_filename}'...")
    df = pd.read_csv(input_filename, header=None)

    # Step 2: Remove the first row (the incorrect header 'column00,...')
    # The first row is at index 0. 'inplace=True' modifies the dataframe directly.
    df.drop(index=0, inplace=True)
    print("[REDACTED_BY_SCRIPT]")

    # Step 3: Remove the first column (the unwanted index)
    # The first column is at index 0.
    df.drop(columns=0, inplace=True)
    print("[REDACTED_BY_SCRIPT]")

    # Step 4: Save the cleaned DataFrame to a new CSV file.
    # 'header=False' ensures no new header is written.
    # 'index=False' prevents pandas from writing its own row index.
    df.to_csv(output_filename, header=False, index=False)
    
    print("\nSuccess!")
    print(f"Cleaned data has been saved to '{output_filename}'")

except FileNotFoundError:
    print(f"Error: The file '{input_filename}' was not found.")
    print("[REDACTED_BY_SCRIPT]")
except Exception as e:
    print(f"[REDACTED_BY_SCRIPT]")
