"""
This code is a data cleanup script designed to resolve duplicate column issues that commonly occur when merging multiple datasets. The specific problem it solves is the presence of duplicate columns with "_x" and "_y" suffixes that pandas creates during merges when column names overlap.

The script operates on a large Parquet file containing the final enriched housing dataset from previous processing steps. When multiple datasets are merged and they contain columns with the same names, pandas automatically renames them by adding "_x" and "_y" suffixes to distinguish between the different sources.

The configuration section sets up the input file path which points to the messy dataset with duplicate columns, defines the output path for the cleaned version, and establishes a chunk size of 500,000 rows to handle large files efficiently without running out of memory.

The main cleanup function follows a systematic approach. First, it reads the schema of the input Parquet file to identify all column names without loading the actual data. It then scans through all column names to find those ending with "_x" and "_y" suffixes. From these suffixed columns, it extracts the base names to understand which columns were duplicated during the merge process.

The script then processes the data in chunks to maintain memory efficiency when dealing with large datasets. For each chunk, it applies a coalescing logic where it creates a new clean column using the base name. The coalescing prioritizes the "_x" version of each column, but if that value is null or missing, it takes the value from the "_y" version instead. This ensures no data is lost while eliminating the duplication.

After creating the clean merged columns, the script drops all the original "_x" and "_y" columns since they are no longer needed. Each processed chunk is then written to a new clean Parquet file using PyArrow for efficient I/O operations.

The script includes robust error handling throughout the process. It checks if it can read the input file schema and exits gracefully if there are issues. It also handles cases where no duplicate columns are found and removes any existing output files before starting to ensure a clean slate.

The memory management is particularly important here because housing datasets can be extremely large with millions of rows and hundreds of columns. By processing in chunks and using PyArrow's efficient Parquet operations, the script can handle datasets that would otherwise exceed available RAM.

This cleanup step is essential because machine learning algorithms and data analysis tools expect clean, non-duplicated column names. Having "_x" and "_y" suffixed columns would create confusion and potentially lead to errors in downstream analysis or modeling work.
"""


import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os
import sys
import traceback
import re

# --- Configuration ---
# The messy input file with duplicate _x and _y columns
MESSY_INPUT_FILE = r"[REDACTED_BY_SCRIPT]"

# The new, clean output file we will create
CLEAN_OUTPUT_FILE = r"[REDACTED_BY_SCRIPT]"

CHUNK_SIZE = 500_000

# --- Main Cleanup Function ---
def clean_duplicate_columns():
    """
    Reads a Parquet file in chunks, finds _x and _y columns,
    coalesces them into single columns, and saves a new, clean file.
    """
    print(f"[REDACTED_BY_SCRIPT]")

    # --- Step 1: Identify Duplicate Columns from Schema ---
    try:
        all_columns = pq.read_schema(MESSY_INPUT_FILE).names
        cols_x = {col for col in all_columns if col.endswith('_x')}
        cols_y = {col for col in all_columns if col.endswith('_y')}
        
        # Find the base names of the duplicated columns
        base_names = {col[:-2] for col in cols_x}
        print(f"[REDACTED_BY_SCRIPT]")
        if not base_names:
            print("[REDACTED_BY_SCRIPT]")
            return

    except Exception as e:
        print(f"[REDACTED_BY_SCRIPT]")
        sys.exit(1)

    # --- Step 2: Process in Chunks to Create Clean File ---
    try:
        parquet_file = pq.ParquetFile(MESSY_INPUT_FILE)
        batch_iterator = parquet_file.iter_batches(batch_size=CHUNK_SIZE)
        writer = None

        if os.path.exists(CLEAN_OUTPUT_FILE):
            os.remove(CLEAN_OUTPUT_FILE)
            print(f"[REDACTED_BY_SCRIPT]")

        print("[REDACTED_BY_SCRIPT]")
        for i, batch in enumerate(batch_iterator):
            print(f"[REDACTED_BY_SCRIPT]")
            df_chunk = batch.to_pandas(split_blocks=True, self_destruct=True)

            # --- Coalesce Logic ---
            for base in base_names:
                col_x = f"{base}_x"
                col_y = f"{base}_y"
                
                # Create the new, clean column. Take the value from _x first.
                # If _x is null, take the value from _y.
                if col_x in df_chunk and col_y in df_chunk:
                    df_chunk[base] = df_chunk[col_x].fillna(df_chunk[col_y])

            # Drop the old _x and _y columns
            cols_to_drop = list(cols_x.union(cols_y))
            df_chunk.drop(columns=cols_to_drop, inplace=True, errors='ignore')

            # --- Writing Logic ---
            table = pa.Table.from_pandas(df_chunk, preserve_index=False)
            if writer is None:
                print(f"[REDACTED_BY_SCRIPT]")
                writer = pq.ParquetWriter(CLEAN_OUTPUT_FILE, table.schema)
            
            writer.write_table(table)
        
        if writer:
            writer.close()

    except Exception as e:
        print(f"[REDACTED_BY_SCRIPT]")
        traceback.print_exc()
        sys.exit(1)

    print(f"[REDACTED_BY_SCRIPT]")

if __name__ == "__main__":
    clean_duplicate_columns()