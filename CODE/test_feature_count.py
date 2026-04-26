import csv

def check_csv_row_lengths(filepath, delimiter=','):
    """
    Checks if all rows in a CSV file have the same number of features as the header.

    Args:
        filepath (str): The path to the CSV file.
        delimiter (str, optional): The delimiter used in the CSV file. Defaults to ','.

    Returns:
        bool: True if all rows are consistent, False otherwise.
    """
    consistent = True
    try:
        with open(filepath, 'r', newline='', encoding='utf-8') as csvfile:
            # Using csv.reader to handle CSV parsing complexities
            reader = csv.reader(csvfile, delimiter=delimiter)

            try:
                # Get the header row
                header = next(reader)
                expected_num_features = len(header)
                print(f"[REDACTED_BY_SCRIPT]")
                print(f"Header: {header}\n")
            except StopIteration:
                print(f"Error: The file '{filepath}'[REDACTED_BY_SCRIPT]")
                return False # Or True, depending on how you define consistency for an empty file

            # Iterate through the rest of the rows
            # Start row_number from 2 because header is row 1
            for row_number, row in enumerate(reader, start=2):
                actual_num_features = len(row)
                if actual_num_features != expected_num_features:
                    print(f"[REDACTED_BY_SCRIPT]")
                    print(f"[REDACTED_BY_SCRIPT]")
                    print(f"[REDACTED_BY_SCRIPT]")
                    print(f"  Row content: {row}")
                    consistent = False
            
            if consistent:
                print(f"All data rows in '{filepath}'[REDACTED_BY_SCRIPT]")

    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        return False
    except Exception as e:
        print(f"[REDACTED_BY_SCRIPT]")
        return False
        
    return consistent

if __name__ == "__main__":
    result_semicolon = check_csv_row_lengths("[REDACTED_BY_SCRIPT]", delimiter=',')
    print(f"[REDACTED_BY_SCRIPT]")