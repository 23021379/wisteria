"""
Main Components
Step 1: Data Loading (Lines 22-34)
Loads the combined housing dataset from the previous script (combined_housing_data.parquet)
Includes error handling for missing files
Reports initial dataset dimensions
Step 2: Column Type Analysis (Lines 36-56)
The script analyzes all columns and categorizes them into three groups:

Categorical columns for One-Hot Encoding:

Columns with ≤20 unique values
Creates binary dummy variables (0/1) for each category
Good for low-cardinality categorical data
Categorical columns for Label Encoding:

Columns with >20 unique values
Assigns integer labels to each unique category
Prevents dimension explosion for high-cardinality data
Numeric columns:

Already numeric but may need type conversion
Excludes 'postcode' (identifier, not a feature)
Step 3: Encoding Transformations (Lines 58-83)
One-Hot Encoding
Creates binary columns for each category
dummy_na=True creates explicit columns for missing values
Example: A column with values ['A', 'B', 'C'] becomes three columns: col_A, col_B, col_C
Label Encoding
Converts categorical values to integers
Example: ['London', 'Manchester', 'Birmingham'] → [0, 1, 2]
Saves encoders for potential future use
Numeric Conversion
Forces conversion to numeric type
errors='coerce' converts invalid values to NaN instead of throwing errors
Step 4: Final Processing (Lines 85-106)
Performs final data type verification
Deliberately leaves NaN values instead of filling them (many ML models like XGBoost handle NaNs natively)
Saves the ML-ready dataset as ml_ready_combined_housing_data.parquet
Key Design Decisions
Adaptive Encoding Strategy: Uses the 20-unique-value threshold to automatically choose between one-hot and label encoding
Missing Value Handling: Preserves NaNs rather than imputing, allowing ML algorithms to handle them
Identifier Preservation: Keeps 'postcode' as a string identifier
Error Resilience: Uses errors='coerce' to handle data conversion issues gracefully
"""


import pandas as pd
import os
import re
from sklearn.preprocessing import LabelEncoder

# --- Configuration ---
BASE_DIR = r"[REDACTED_BY_SCRIPT]"
# INPUT FILE: The parquet file we created in the last step
INPUT_FILE = os.path.join(BASE_DIR, "[REDACTED_BY_SCRIPT]")
# OUTPUT FILE: The new, fully encoded, ML-ready dataset
OUTPUT_FILE_ENCODED = os.path.join(BASE_DIR, "[REDACTED_BY_SCRIPT]")

# Threshold for using One-Hot Encoding vs. Label Encoding
# If a column has 20 or fewer unique values, it will be one-hot encoded.
# Otherwise, it will be label encoded.
ONE_HOT_ENCODING_THRESHOLD = 20

print("[REDACTED_BY_SCRIPT]")

# --- Step 1: Load the Assembled Dataset ---
print(f"[REDACTED_BY_SCRIPT]'{os.path.basename(INPUT_FILE)}'...")
try:
    df = pd.read_parquet(INPUT_FILE)
    print("[REDACTED_BY_SCRIPT]")
    print(f"[REDACTED_BY_SCRIPT]")
except FileNotFoundError:
    print(f"[REDACTED_BY_SCRIPT]")
    print(f"[REDACTED_BY_SCRIPT]")
    exit()
except Exception as e:
    print(f"[REDACTED_BY_SCRIPT]")
    exit()

# --- Step 2: Identify Column Types for Encoding ---
print("[REDACTED_BY_SCRIPT]")

# Identify all columns that are currently non-numeric (i.e., 'object' type)
# The postcode is our identifier, not a feature, so we exclude it.
categorical_cols = df.select_dtypes(include=['object']).columns.drop('postcode')

# Separate them into two groups: those for one-hot encoding and those for label encoding
cols_to_one_hot_encode = []
cols_to_label_encode = []

for col in categorical_cols:
    # Check the number of unique values in the column
    if df[col].nunique(dropna=True) <= ONE_HOT_ENCODING_THRESHOLD:
        cols_to_one_hot_encode.append(col)
    else:
        cols_to_label_encode.append(col)

# Identify all columns that are NOT categorical to convert them to numbers
numeric_cols = df.columns.difference(categorical_cols).drop('postcode', errors='ignore')

print(f"[REDACTED_BY_SCRIPT]")
print(f"[REDACTED_BY_SCRIPT]")
print(f"[REDACTED_BY_SCRIPT]")

# --- Step 3: Apply Encoding and Type Conversion ---
print("[REDACTED_BY_SCRIPT]")

# Apply One-Hot Encoding
if cols_to_one_hot_encode:
    print(f"[REDACTED_BY_SCRIPT]")
    # The 'dummy_na=True' argument creates an explicit column for missing values, which can be a useful feature for models.
    df = pd.get_dummies(df, columns=cols_to_one_hot_encode, dummy_na=True, prefix=cols_to_one_hot_encode)
else:
    print("[REDACTED_BY_SCRIPT]")

# Apply Label Encoding
if cols_to_label_encode:
    print(f"[REDACTED_BY_SCRIPT]")
    # Using LabelEncoder for high-cardinality features
    label_encoders = {}
    for col in cols_to_label_encode:
        le = LabelEncoder()
        # Convert the column to string type and fill missing values before encoding
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le # Optionally save the encoder
else:
    print("[REDACTED_BY_SCRIPT]")

# Convert remaining columns to numeric, coercing errors
print(f"[REDACTED_BY_SCRIPT]")
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# --- Step 4: Final Check and Save ---
print("[REDACTED_BY_SCRIPT]")

# At this point, all columns (except the 'postcode' identifier) should be numeric.
# We can fill any remaining missing values (NaNs) that were created during numeric conversion.
# A simple strategy is to fill with 0, but for ML you might use mean, median, or a more complex imputer.
# For now, we will leave them as NaN, as many models (like XGBoost) can handle them automatically.
# print("[REDACTED_BY_SCRIPT]")
# df.fillna(0, inplace=True)

# Final check on data types
print("[REDACTED_BY_SCRIPT]")
print(df.info(verbose=False, show_counts=True))

# Save the final, ML-ready dataset
df.to_parquet(OUTPUT_FILE_ENCODED, index=False)

print(f"[REDACTED_BY_SCRIPT]")
print(f"[REDACTED_BY_SCRIPT]")