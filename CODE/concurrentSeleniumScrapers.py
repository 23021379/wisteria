import multiprocessing
import subprocess
import time
import logging
import os
import csv
import shutil
import tempfile

# --- Configuration ---
NUM_PROCESSES = 4  # Number of concurrent scraper instances - easily changed
BASE_INPUT_FILE = r"[REDACTED_BY_SCRIPT]"  # Your main input CSV
BASE_OUTPUT_FILE = r"[REDACTED_BY_SCRIPT]"  # Base name.  Will become Rightmove_1.csv, etc.
SCRAPER_SCRIPT = r"[REDACTED_BY_SCRIPT]"  # Your scraper script
CENTRAL_CHROMEDRIVER = r"[REDACTED_BY_SCRIPT]"
MASTER_OUTPUT_FILE = r"[REDACTED_BY_SCRIPT]"  # Master output file

# --- Logging ---
logging.basicConfig(filename="launcher.log", level=logging.INFO, encoding='utf-8',
                    format='[REDACTED_BY_SCRIPT]')

def combine_previous_results():
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
    
    # Check for previous run output files
    row_count = 0
    for i in range(1, NUM_PROCESSES + 1):
        process_output_file = f"[REDACTED_BY_SCRIPT]"
        if os.path.exists(process_output_file):
            try:
                with open(process_output_file, 'r', encoding='utf-8') as infile, \
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
            except Exception as e:
                logging.error(f"[REDACTED_BY_SCRIPT]")
    
    logging.info(f"[REDACTED_BY_SCRIPT]")
    return existing_addresses

def split_csv(input_file, num_processes, already_scraped=None):
    """Splits the input CSV file into smaller CSV files for each process, 
    excluding addresses that have already been scraped."""
    if already_scraped is None:
        already_scraped = set()
        
    logging.info(f"[REDACTED_BY_SCRIPT]")

    try:
        with open(input_file, 'r', encoding='utf-8') as infile:
            reader = csv.reader(infile)
            header = next(reader)  # Preserve header row
            
            # Skip rows that were already processed from previous runs
            data = []
            for row in reader:
                if not row:  # Skip empty rows
                    continue
                    
                # Extract the address identifier - adjust column index if needed
                address_key = row[0].lower().replace(",", "")
                
                # Only include addresses not already scraped
                if address_key not in already_scraped:
                    data.append(row)
            
            if not data:
                logging.info("[REDACTED_BY_SCRIPT]")
                return []
                
            logging.info(f"[REDACTED_BY_SCRIPT]")
            
            chunk_size = len(data) // num_processes
            remainder = len(data) % num_processes
            
            if chunk_size == 0:
                # If fewer addresses than processes, adjust the number of processes
                num_processes = len(data)
                chunk_size = 1
                remainder = 0
                logging.info(f"[REDACTED_BY_SCRIPT]")
            
            for i in range(num_processes):
                start_index = i * chunk_size
                end_index = (i + 1) * chunk_size
                
                if i == num_processes - 1:  # Distribute remainder
                    end_index += remainder
                    
                output_file = f"[REDACTED_BY_SCRIPT]"
                with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
                    writer = csv.writer(outfile)
                    writer.writerow(header)  # Write the header row
                    writer.writerows(data[start_index:end_index])
                logging.info(f"[REDACTED_BY_SCRIPT]")
                
            return [f"[REDACTED_BY_SCRIPT]" for i in range(num_processes)]  # Return file list
            
    except Exception as e:
        logging.exception(f"[REDACTED_BY_SCRIPT]")
        return []  # Return empty list on error

def run_scraper(process_id, input_file, output_file, lock):
    """[REDACTED_BY_SCRIPT]"""
    logging.info(f"[REDACTED_BY_SCRIPT]")
    
    # Create a temporary directory for this process's ChromeDriver
    with tempfile.TemporaryDirectory() as temp_dir:
        # Copy the ChromeDriver executable to the temporary directory
        chromedriver_path = os.path.join(temp_dir, "chromedriver.exe")
        
        with lock:  # Use lock when copying the driver to prevent race conditions
            shutil.copy2(CENTRAL_CHROMEDRIVER, chromedriver_path)  # copy2 preserves metadata
        
        # Set the UC_CHROMEDRIVER_PATH environment variable for this process
        my_env = os.environ.copy()
        my_env["[REDACTED_BY_SCRIPT]"] = chromedriver_path
        
        try:
            command = ["python", SCRAPER_SCRIPT, "--input", input_file, "--output", output_file]
            subprocess.run(command, check=True, env=my_env)  # Pass environment
            logging.info(f"[REDACTED_BY_SCRIPT]")
        except subprocess.CalledProcessError as e:
            logging.error(f"[REDACTED_BY_SCRIPT]")
        except Exception as e:
            logging.exception(f"[REDACTED_BY_SCRIPT]")

def combine_csv_files(base_output_file, num_processes, final_output_file):
    """[REDACTED_BY_SCRIPT]"""
    logging.info(f"[REDACTED_BY_SCRIPT]")
    try:
        with open(final_output_file, 'w', newline='', encoding='utf-8') as final_outfile:
            writer = csv.writer(final_outfile)
            header_written = False  # Flag to write header only once
            
            for i in range(num_processes):
                process_id = i + 1
                output_file = f"[REDACTED_BY_SCRIPT]"
                
                try:  # Handle missing files (if a process crashed)
                    with open(output_file, 'r', encoding='utf-8') as infile:
                        reader = csv.reader(infile)
                        try:
                            header = next(reader)
                            
                            if not header_written:
                                writer.writerow(header)
                                header_written = True
                                
                            for row in reader:
                                writer.writerow(row)
                        except StopIteration:
                            logging.warning(f"[REDACTED_BY_SCRIPT]")
                            
                    logging.info(f"[REDACTED_BY_SCRIPT]")
                except FileNotFoundError:
                    logging.warning(f"[REDACTED_BY_SCRIPT]")
                    
    except Exception as e:
        logging.exception(f"[REDACTED_BY_SCRIPT]")

if __name__ == "__main__":
    start_time = time.time()
    logging.info("[REDACTED_BY_SCRIPT]")
    
    # Check if ChromeDriver exists
    if not os.path.exists(CENTRAL_CHROMEDRIVER):
        logging.fatal(f"[REDACTED_BY_SCRIPT]")
        exit(1)
        
    # First, combine all previous results into the master file and get already scraped addresses
    already_scraped = combine_previous_results()
    
    # Split the input file, excluding already scraped addresses
    input_files = split_csv(BASE_INPUT_FILE, NUM_PROCESSES, already_scraped)
    
    # If there's nothing new to scrape, exit
    if not input_files:
        logging.info("[REDACTED_BY_SCRIPT]")
        exit(0)
        
    # Create a lock for synchronized access to shared resources
    lock = multiprocessing.Lock()
    
    # Start the scraper processes
    processes = []
    for i in range(len(input_files)):  # Use actual number of files (might be less than NUM_PROCESSES)
        process_id = i + 1
        input_file = input_files[i]
        output_file = f"[REDACTED_BY_SCRIPT]"
        
        p = multiprocessing.Process(target=run_scraper, args=(process_id, input_file, output_file, lock))
        processes.append(p)
        p.start()
        
    # Wait for all processes to complete
    for p in processes:
        p.join()
        
    end_time = time.time()
    logging.info(f"[REDACTED_BY_SCRIPT]")
    
    # Combine the new output files
    FINAL_OUTPUT = r"[REDACTED_BY_SCRIPT]"
    combine_csv_files(BASE_OUTPUT_FILE, len(input_files), FINAL_OUTPUT)
    
    # Update the master file with new results
    combine_previous_results()
    
    logging.info("[REDACTED_BY_SCRIPT]")