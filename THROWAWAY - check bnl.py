import csv
import os

def check_csv_integrity(file_path):
    """
    Reads a CSV file row by row to check if each data row has the same
    number of fields (commas) as the header row.
    
    This helps diagnose "ragged" or "malformed" CSV files that cause
    pandas parsing errors.
    """
    print(f"[REDACTED_BY_SCRIPT]")

    if not os.path.exists(file_path):
        print(f"[REDACTED_BY_SCRIPT]'{file_path}'")
        return

    mismatched_rows = []

    try:
        with open(file_path, mode='r', encoding='utf-8', newline='') as infile:
            reader = csv.reader(infile)
            
            # Get the header and its length
            try:
                header = next(reader)
                header_length = len(header)
                print(f"[REDACTED_BY_SCRIPT]")
                print("-" * 30)
            except StopIteration:
                print("[REDACTED_BY_SCRIPT]")
                return

            # Check every subsequent row
            for line_number, row in enumerate(reader, start=2):
                row_length = len(row)
                if row_length != header_length:
                    # If a mismatch is found, record the details
                    mismatched_rows.append({
                        "line": line_number,
                        "expected": header_length,
                        "found": row_length,
                        "content": str(row) # Store the row content for inspection
                    })

    except Exception as e:
        print(f"[REDACTED_BY_SCRIPT]")
        return

    # --- Final Report ---
    print("[REDACTED_BY_SCRIPT]")
    if not mismatched_rows:
        print("[REDACTED_BY_SCRIPT]")
    else:
        print(f"[REDACTED_BY_SCRIPT]")
        print("[REDACTED_BY_SCRIPT]")
        
        # Print details for the first 5 problematic rows
        for i, error in enumerate(mismatched_rows):
            if i >= 5:
                print(f"[REDACTED_BY_SCRIPT]")
                break
            print(f"\n[Line {error['line'[REDACTED_BY_SCRIPT]")
            print(f"[REDACTED_BY_SCRIPT]'expected']}")
            print(f"  - Found Fields:    {error['found']}")
            print(f"[REDACTED_BY_SCRIPT]'content'][:100]}...")


if __name__ == "__main__":
    # Set the path to the file you want to check
    bnl_file_path = r'[REDACTED_BY_SCRIPT]'
    
    # Run the check
    check_csv_integrity(bnl_file_path)