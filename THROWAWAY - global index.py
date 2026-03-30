import pandas as pd
import os

# --- Configuration ---
# This should be the same path as in your main pipeline script
BASE_DATA_PATH = "[REDACTED_BY_SCRIPT]"
INPUT_FILE = os.path.join(BASE_DATA_PATH, "[REDACTED_BY_SCRIPT]")
OUTPUT_FILE = os.path.join(BASE_DATA_PATH, "[REDACTED_BY_SCRIPT]")
CHUNK_SIZE = 100000  # Adjust chunk size based on your system's memory

def create_postcode_index_file():
    """
    Reads a large CSV file in chunks, extracts the 'pcds' column,
    and saves it to a new file.
    """
    if not os.path.exists(INPUT_FILE):
        print(f"[REDACTED_BY_SCRIPT]")
        return

    print(f"[REDACTED_BY_SCRIPT]'pcds'[REDACTED_BY_SCRIPT]")

    try:
        # Use an iterator to read the file in chunks
        chunk_iterator = pd.read_csv(
            INPUT_FILE,
            usecols=['pcds'],
            chunksize=CHUNK_SIZE,
            low_memory=False,
            encoding='utf-8'
        )

        # Process the first chunk to create the file with a header
        first_chunk = next(chunk_iterator)
        first_chunk.to_csv(OUTPUT_FILE, index=False)
        print(f"[REDACTED_BY_SCRIPT]")

        # Process remaining chunks and append to the file without the header
        for i, chunk in enumerate(chunk_iterator, start=2):
            chunk.to_csv(OUTPUT_FILE, mode='a', header=False, index=False)
            print(f"[REDACTED_BY_SCRIPT]")

        print("[REDACTED_BY_SCRIPT]")
        print(f"[REDACTED_BY_SCRIPT]")

    except Exception as e:
        print(f"[REDACTED_BY_SCRIPT]")
        print("[REDACTED_BY_SCRIPT]")


if __name__ == "__main__":
    create_postcode_index_file()