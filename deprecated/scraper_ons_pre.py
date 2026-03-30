import csv
import os
import sys
from collections import defaultdict

# --- Configuration ---
ons_input_folder_path = 'ons' # *** CONFIRM THIS PATH ***
output_pivoted_csv_path = 'ons_pivoted.csv' # *** NEW FILENAME ***

# --- Main Pre-processing Logic ---
def pivot_ons_data_by_oa(input_folder, output_file):
    print(f"[REDACTED_BY_SCRIPT]")
    # Structure: {oa_code: {category_description: observation_value}}
    oa_data = defaultdict(dict)
    all_categories = set()
    processed_files = 0
    skipped_files = 0
    skipped_rows_total = 0
    # Expect OA code in first column generally
    possible_oa_col_names = ['Output Areas Code', 'Geography code', 'output areas code', 'oa21cd'] # Added oa21cd
    possible_value_col_names = ['Observation', 'observation', 'Value', 'value']

    print("[REDACTED_BY_SCRIPT]")
    if not os.path.isdir(input_folder):
        print(f"[REDACTED_BY_SCRIPT]")
        sys.exit(1)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.csv'):
            file_path = os.path.join(input_folder, filename)
            print(f"[REDACTED_BY_SCRIPT]")
            file_skipped = False
            rows_skipped_this_file = 0
            try:
                current_encoding = 'utf-8'
                try:
                    f_ons = open(file_path, 'r', newline='', encoding=current_encoding)
                    reader_ons = csv.reader(f_ons)
                    header = next(reader_ons)
                    header = [h.strip() for h in header]

                    # --- Dynamically Find Columns ---
                    oa_col_idx = -1
                    value_col_idx = -1
                    category_desc_col_idx = -1
                    category_code_col_idx = -1

                    # Find OA Column
                    for i, h in enumerate(header):
                        if h.lower() in possible_oa_col_names:
                            oa_col_idx = i
                            break
                    if oa_col_idx == -1:
                         print(f"[REDACTED_BY_SCRIPT]")
                         oa_col_idx = 0

                    # Find Value Column
                    for i, h in enumerate(header):
                         if h.lower() in possible_value_col_names:
                             value_col_idx = i
                             break
                    if value_col_idx == -1:
                        print(f"[REDACTED_BY_SCRIPT]")
                        value_col_idx = len(header) - 1

                    # Deduce Category Columns
                    if len(header) >= 3 and value_col_idx > 0:
                         category_desc_col_idx = value_col_idx - 1
                         if value_col_idx > 1: category_code_col_idx = value_col_idx - 2
                         print(f"[REDACTED_BY_SCRIPT]")
                    else:
                         print(f"[REDACTED_BY_SCRIPT]")
                         skipped_files += 1; file_skipped = True; f_ons.close(); continue

                    if not (0 <= oa_col_idx < len(header) and 0 <= category_desc_col_idx < len(header) and 0 <= value_col_idx < len(header)):
                         print(f"[REDACTED_BY_SCRIPT]")
                         skipped_files += 1; file_skipped = True; f_ons.close(); continue

                    # Process rows
                    for row_num, row in enumerate(reader_ons, start=2):
                        if len(row) == len(header):
                            try:
                                oa_code = row[oa_col_idx].strip() # Use OA code
                                category = row[category_desc_col_idx].strip()
                                value = row[value_col_idx].strip()

                                skip_this_row = False
                                if category_code_col_idx != -1 and 0 <= category_code_col_idx < len(row):
                                     if row[category_code_col_idx].strip() == '-8': skip_this_row = True
                                elif "does not apply" in category.lower(): skip_this_row = True
                                if skip_this_row: continue

                                if oa_code and category: # Check OA code here
                                    unique_category_key = category # Keep simple for now
                                    all_categories.add(unique_category_key)
                                    oa_data[oa_code][unique_category_key] = value # Use OA code as primary key
                                else:
                                    rows_skipped_this_file += 1
                            except IndexError: rows_skipped_this_file += 1
                        else:
                            if any(row): rows_skipped_this_file += 1
                    f_ons.close()

                except UnicodeDecodeError:
                    print(f"[REDACTED_BY_SCRIPT]")
                    current_encoding = 'latin-1'
                    # --- Repeat logic with latin-1 ---
                    try:
                        with open(file_path, 'r', newline='', encoding=current_encoding) as f_ons_latin:
                            # ... (Copy/paste the column finding and row processing logic from above, using oa_code) ...
                           reader_ons = csv.reader(f_ons_latin)
                           header = next(reader_ons); header = [h.strip() for h in header]
                           oa_col_idx = -1; value_col_idx = -1; category_desc_col_idx = -1; category_code_col_idx = -1
                           for i, h in enumerate(header):
                               if h.lower() in possible_oa_col_names: oa_col_idx = i; break
                           if oa_col_idx == -1: oa_col_idx = 0
                           for i, h in enumerate(header):
                               if h.lower() in possible_value_col_names: value_col_idx = i; break
                           if value_col_idx == -1: value_col_idx = len(header) - 1
                           if len(header) >= 3 and value_col_idx > 0:
                               category_desc_col_idx = value_col_idx - 1
                               if value_col_idx > 1: category_code_col_idx = value_col_idx - 2
                           else: print(f"[REDACTED_BY_SCRIPT]"); skipped_files += 1; file_skipped = True; continue
                           if not (0 <= oa_col_idx < len(header) and 0 <= category_desc_col_idx < len(header) and 0 <= value_col_idx < len(header)):
                                print(f"[REDACTED_BY_SCRIPT]"); skipped_files += 1; file_skipped = True; continue

                           for row_num, row in enumerate(reader_ons, start=2):
                                if len(row) == len(header):
                                    try:
                                        oa_code = row[oa_col_idx].strip()
                                        category = row[category_desc_col_idx].strip()
                                        value = row[value_col_idx].strip()
                                        skip_this_row = False
                                        if category_code_col_idx != -1 and 0 <= category_code_col_idx < len(row):
                                             if row[category_code_col_idx].strip() == '-8': skip_this_row = True
                                        elif "does not apply" in category.lower(): skip_this_row = True
                                        if skip_this_row: continue
                                        if oa_code and category:
                                            unique_category_key = category
                                            all_categories.add(unique_category_key)
                                            oa_data[oa_code][unique_category_key] = value
                                        else: rows_skipped_this_file += 1
                                    except IndexError: rows_skipped_this_file += 1
                                else:
                                    if any(row): rows_skipped_this_file += 1
                    except Exception as e_latin: print(f"[REDACTED_BY_SCRIPT]"); skipped_files += 1; file_skipped = True; continue

                except StopIteration: print(f"[REDACTED_BY_SCRIPT]")
                except Exception as e: print(f"[REDACTED_BY_SCRIPT]"); skipped_files += 1; file_skipped = True;
                finally:
                     if 'f_ons' in locals() and not f_ons.closed: f_ons.close()

            except IOError as ioe: print(f"[REDACTED_BY_SCRIPT]"); skipped_files += 1; file_skipped = True;

            if not file_skipped:
                 processed_files += 1
                 if rows_skipped_this_file > 0:
                     print(f"[REDACTED_BY_SCRIPT]")
                     skipped_rows_total += rows_skipped_this_file

    print(f"[REDACTED_BY_SCRIPT]")
    print(f"[REDACTED_BY_SCRIPT]")
    if skipped_rows_total > 0: print(f"[REDACTED_BY_SCRIPT]")

    if not all_categories: print("[REDACTED_BY_SCRIPT]"); sys.exit(1)

    # --- Second Pass: Write the pivoted data to the output CSV ---
    print(f"[REDACTED_BY_SCRIPT]")
    sorted_categories = sorted(list(all_categories))
    header_row = ['OA21_Code'] + sorted_categories # *** Use OA code header ***

    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as f_out:
            writer = csv.writer(f_out)
            writer.writerow(header_row)
            written_oa_count = 0
            for oa_code_key in sorted(oa_data.keys()): # Use OA code key
                data_dict = oa_data[oa_code_key]
                output_row = [oa_code_key] + [data_dict.get(cat, '') for cat in sorted_categories]
                writer.writerow(output_row)
                written_oa_count += 1
        print(f"[REDACTED_BY_SCRIPT]")
    except IOError as ioe: print(f"[REDACTED_BY_SCRIPT]")
    except Exception as e: print(f"[REDACTED_BY_SCRIPT]")

# --- Run the Pivot Process ---
if __name__ == "__main__":
    if ons_input_folder_path == '[REDACTED_BY_SCRIPT]' or not os.path.exists(ons_input_folder_path):
         print(f"Error: Update 'ons_input_folder_path'[REDACTED_BY_SCRIPT]")
    else:
        pivot_ons_data_by_oa(ons_input_folder_path, output_pivoted_csv_path)