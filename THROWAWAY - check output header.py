import pandas as pd

# Path to the file
file_path = r"[REDACTED_BY_SCRIPT]"

# Most efficient way to get just the column names using parquet metadata
import pyarrow.parquet as pq
parquet_file = pq.ParquetFile(file_path)
columns = parquet_file.schema.names

# Print all column names
print("Column names:")
for i, col in enumerate(columns, 1):
    print(f"{i}. {col}")

print(f"[REDACTED_BY_SCRIPT]")