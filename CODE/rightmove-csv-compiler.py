import csv
import os
import logging

# --- Configuration ---
INPUT_FILES = [
    r"[REDACTED_BY_SCRIPT]",
    r"[REDACTED_BY_SCRIPT]",
    r"[REDACTED_BY_SCRIPT]",
    r"[REDACTED_BY_SCRIPT]"
]
MASTER_OUTPUT_FILE = r"[REDACTED_BY_SCRIPT]"

# --- Logging ---
logging.basicConfig(filename="[REDACTED_BY_SCRIPT]", level=logging.INFO, encoding='utf-8',
                    format='[REDACTED_BY_SCRIPT]')

def combine_and_deduplicate():
    """[REDACTED_BY_SCRIPT]"""
    logging.info(f"[REDACTED_BY_SCRIPT]")

    # Create master file if it doesn't exist
    if not os.path.exists(MASTER_OUTPUT_FILE):
        with open(MASTER_OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # Write header row - adjust this based on your actual header structure
            writer.writerow(['Address+URL', 'address+type+beds+baths', 'prevsold'])
    
    # Get existing addresses from master file to avoid duplicates
    existing_addresses = set()
    try:
        with open(MASTER_OUTPUT_FILE, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                if row and len(row) > 0:
                    # Use first column as address identifier - adjust if needed
                    existing_addresses.add(row[0].lower().replace(",", ""))
    except Exception as e:
        logging.error(f"[REDACTED_BY_SCRIPT]")

    row_count = 0
    for input_file in INPUT_FILES:
        if os.path.exists(input_file):
            try:
                with open(input_file, 'r', encoding='utf-8') as infile, \
                     open(MASTER_OUTPUT_FILE, 'a', newline='', encoding='utf-8') as outfile:
                    reader = csv.reader(infile)
                    writer = csv.writer(outfile)

                    try:
                        next(reader)  # Skip header if present
                    except StopIteration:
                        continue  # Empty file

                    for row in reader:
                        if row and len(row) > 0:
                            # Check if this address is already in the master file
                            address_key = row[0].lower().replace(",", "")
                            if address_key not in existing_addresses:
                                writer.writerow(row)
                                existing_addresses.add(address_key)
                                row_count += 1
                logging.info(f"[REDACTED_BY_SCRIPT]")
            except Exception as e:
                logging.error(f"[REDACTED_BY_SCRIPT]")
        else:
            logging.warning(f"[REDACTED_BY_SCRIPT]")

    logging.info(f"[REDACTED_BY_SCRIPT]")

    # Deduplicate the master file
    deduplicate_csv(MASTER_OUTPUT_FILE)

    # Delete the input files
    for input_file in INPUT_FILES:
        try:
            os.remove(input_file)
            logging.info(f"[REDACTED_BY_SCRIPT]")
        except Exception as e:
            logging.error(f"[REDACTED_BY_SCRIPT]")

def deduplicate_csv(csv_filepath):
    """[REDACTED_BY_SCRIPT]"""
    logging.info(f"[REDACTED_BY_SCRIPT]")
    
    unique_rows = {}
    row_count = 0
    
    try:
        with open(csv_filepath, 'r', newline='', encoding='utf-8') as infile:
            reader = csv.reader(infile)
            header = next(reader)  # Read the header row
            
            for row in reader:
                if row:
                    address_key = row[0].lower().replace(",", "")  # Normalize address
                    if address_key not in unique_rows:
                        unique_rows[address_key] = row
                        row_count += 1
        
        # Write the unique rows back to the CSV file
        with open(csv_filepath, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(header)  # Write the header row
            for row in unique_rows.values():
                writer.writerow(row)
        
        logging.info(f"[REDACTED_BY_SCRIPT]")
    
    except Exception as e:
        logging.error(f"[REDACTED_BY_SCRIPT]")

if __name__ == "__main__":
    combine_and_deduplicate()
    logging.info("Script finished.")