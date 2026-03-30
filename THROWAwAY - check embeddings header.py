import csv
import os

# --- Step 1: Set the path to your local CSV file ---
# IMPORTANT: Replace the placeholder below with the actual, full path to your file.
# Example for Windows: '[REDACTED_BY_SCRIPT]'
# Example for Mac/Linux: '[REDACTED_BY_SCRIPT]'

local_csv_path = '[REDACTED_BY_SCRIPT]'
# --- Step 2: Run the script ---

def find_string_column(filepath):
    """
    Analyzes the first 3 data rows of a CSV to find the column containing string data.
    """
    print("[REDACTED_BY_SCRIPT]")
    if not os.path.exists(filepath):
        print(f"[REDACTED_BY_SCRIPT]")
        print(f"[REDACTED_BY_SCRIPT]")
        return

    print(f"[REDACTED_BY_SCRIPT]")

    try:
        with open(filepath, mode='r', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)

            # Read the header row to get column names
            header = next(reader, None)
            if header is None:
                print("[REDACTED_BY_SCRIPT]")
                return

            # Read the first 3 data rows
            data_rows = []
            for i in range(1):
                row = next(reader, None)
                if row is not None:
                    data_rows.append(row)
                else:
                    break

            if not data_rows:
                print("[REDACTED_BY_SCRIPT]")
                return

            print(f"[REDACTED_BY_SCRIPT]")
            print(f"Header: {header}")
            for i, row in enumerate(data_rows):
                print(f"Row {i+1}: {row}")

            # Add this to your script
            with open('header_analysis.txt', 'w') as f:
                f.write(f"Header: {header}\n")
                for i, row in enumerate(data_rows):
                    f.write(f"Row {i+1}: {row}\n")

            string_column_index = None
            found_multiple_strings = False

            # Analyze the first data row to check data types
            first_data_row = data_rows[0]
            
            # Iterate through each column in the first data row
            for index, value in enumerate(first_data_row):
                try:
                    # If this conversion succeeds, it's a number.
                    float(value)
                except ValueError:
                    # If it fails, it's a string (our ID column).
                    if string_column_index is not None:
                        # We've already found a string column, this is the second one.
                        print(f"[REDACTED_BY_SCRIPT]")
                        found_multiple_strings = True
                        break
                    
                    string_column_index = index

            print("\n--- Results ---")
            if found_multiple_strings:
                 print("[REDACTED_BY_SCRIPT]")
            elif string_column_index is not None:
                column_name = header[string_column_index]
                print(f"[REDACTED_BY_SCRIPT]")
                print(f"[REDACTED_BY_SCRIPT]")
                print(f"[REDACTED_BY_SCRIPT]'{column_name}'")
                
                # Show the string values from the first 3 rows
                print(f"   - Sample values:")
                for i, row in enumerate(data_rows):
                    if len(row) > string_column_index:
                        print(f"     Row {i+1}: '{row[string_column_index]}'")
            else:
                print("[REDACTED_BY_SCRIPT]")


    except Exception as e:
        print(f"[REDACTED_BY_SCRIPT]")

# Run the analysis
if __name__ == "__main__":
    find_string_column(local_csv_path)