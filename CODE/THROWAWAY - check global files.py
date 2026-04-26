import pyarrow.parquet as pq

# Check a few of the files
paths_to_check = [
    '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]'
]

for path in paths_to_check:
    try:
        num_rows = pq.ParquetFile(path).metadata.num_rows
        print(f"[REDACTED_BY_SCRIPT]")
    except Exception as e:
        print(f"[REDACTED_BY_SCRIPT]")