import pandas as pd
import os

def create_address_index_files():
    """
    Extracts the 'property_address' column from specified CSV files
    and saves it to a new index file.
    """
    # Define the directory where the subset files are located
    source_directory = r'[REDACTED_BY_SCRIPT]'

    # Define the files to process
    subset_filenames = [
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]'
    ]

    print(f"[REDACTED_BY_SCRIPT]")

    for filename in subset_filenames:
        input_path = os.path.join(source_directory, filename)
        output_filename = filename.replace('.csv', '_index.csv')
        output_path = os.path.join(source_directory, output_filename)

        try:
            # Check if the source file exists
            if not os.path.exists(input_path):
                print(f"[REDACTED_BY_SCRIPT]")
                continue

            print(f"[REDACTED_BY_SCRIPT]")
            
            # Read only the 'property_address' column to save memory
            address_df = pd.read_csv(input_path, usecols=['property_address'], low_memory=False)
            
            # Save the extracted addresses to the new index file
            address_df.to_csv(output_path, index=False)
            
            print(f"[REDACTED_BY_SCRIPT]")

        except KeyError:
            print(f"Error: 'property_address'[REDACTED_BY_SCRIPT]")
        except Exception as e:
            print(f"[REDACTED_BY_SCRIPT]")

    print("[REDACTED_BY_SCRIPT]")

if __name__ == '__main__':
    create_address_index_files()