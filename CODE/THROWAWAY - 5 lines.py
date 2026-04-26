import csv
from pathlib import Path
import chardet

def detect_encoding(file_path):
    """[REDACTED_BY_SCRIPT]"""
    with open(file_path, 'rb') as f:
        raw_data = f.read(10000)  # Read first 10KB
        result = chardet.detect(raw_data)
        return result['encoding']

def print_csv_preview(directory_path_str):
    """
    Prints the header and first 5 data lines of CSV files in a given directory.
    """
    directory_path = Path(directory_path_str)

    if not directory_path.is_dir():
        print(f"Error: Directory '{directory_path_str}' not found.")
        return

    print(f"[REDACTED_BY_SCRIPT]")

    processed_any_file = False
    for file_path in directory_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() == '.csv':
            print(f"[REDACTED_BY_SCRIPT]")
            processed_any_file = True
            
            # Try to detect encoding first
            try:
                detected_encoding = detect_encoding(file_path)
                print(f"Detected encoding: {detected_encoding}")
            except Exception as e:
                print(f"[REDACTED_BY_SCRIPT]")
                detected_encoding = 'utf-8'
            
            # Try multiple encodings
            encodings_to_try = [detected_encoding, 'utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            
            success = False
            for encoding in encodings_to_try:
                try:
                    with file_path.open('r', newline='', encoding=encoding, errors='replace') as f:
                        reader = csv.reader(f)
                        
                        try:
                            header = next(reader)
                            print(f"[REDACTED_BY_SCRIPT]")
                            print("Header:", header)
                            
                            # Check for corrupted column names
                            corrupted_cols = [col for col in header if any(ord(char) > 127 and char not in 'áéíóúñü' for char in col)]
                            if corrupted_cols:
                                print(f"[REDACTED_BY_SCRIPT]")
                                for col in corrupted_cols[:3]:  # Show first 3
                                    print(f"[REDACTED_BY_SCRIPT]")
                            
                        except StopIteration:
                            print("[REDACTED_BY_SCRIPT]")
                            break

                        print("[REDACTED_BY_SCRIPT]")
                        lines_printed_count = 0
                        for i, row in enumerate(reader):
                            if i < 1:
                                # Check for corrupted data in rows
                                clean_row = []
                                for cell in row:
                                    if isinstance(cell, str) and any(ord(char) > 127 and char not in 'áéíóúñü' for char in cell):
                                        clean_row.append(f"[REDACTED_BY_SCRIPT]")
                                    else:
                                        clean_row.append(cell)
                                print(clean_row)
                                lines_printed_count += 1
                            else:
                                break
                        
                        if lines_printed_count == 0:
                            print("[REDACTED_BY_SCRIPT]")
                        
                        success = True
                        break  # Successfully read file
                        
                except UnicodeDecodeError as e:
                    print(f"[REDACTED_BY_SCRIPT]")
                    continue
                except Exception as e:
                    print(f"[REDACTED_BY_SCRIPT]")
                    continue
            
            if not success:
                print("[REDACTED_BY_SCRIPT]")
            
            print("-" * 40)

    if not processed_any_file:
        print(f"[REDACTED_BY_SCRIPT]'{directory_path_str}'.")

# --- Main execution ---
if __name__ == "__main__":
    target_directory = r"[REDACTED_BY_SCRIPT]"
    print_csv_preview(target_directory)