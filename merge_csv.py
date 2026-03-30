import csv
import os

def merge_csv_files(file1_path, file2_path, output_path):
    try:
        # Read the first CSV file
        with open(file1_path, 'r', encoding='utf-8') as f1, \
             open(file2_path, 'r', encoding='utf-8') as f2, \
             open(output_path, 'w', newline='', encoding='utf-8') as output:
            
            reader1 = csv.reader(f1)
            reader2 = csv.reader(f2)
            writer = csv.writer(output)
            
            # Merge lines
            for line1, line2 in zip(reader1, reader2):
                # Combine the lines from both files
                merged_line = line1 + line2
                writer.writerow(merged_line)
                
        print(f"[REDACTED_BY_SCRIPT]")
        
    except Exception as e:
        print(f"[REDACTED_BY_SCRIPT]")

if __name__ == "__main__":
    # Define file paths
    file1_path = r"[REDACTED_BY_SCRIPT]"
    file2_path = r"[REDACTED_BY_SCRIPT]"
    output_path = r"[REDACTED_BY_SCRIPT]"
    
    merge_csv_files(file1_path, file2_path, output_path)


#streetscan,chimnie,BnL,Homipi