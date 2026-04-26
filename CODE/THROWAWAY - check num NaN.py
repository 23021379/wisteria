import pandas as pd
import os

def analyze_csv_files(folder_path):
    """
    Analyzes all CSV files in a given folder to find their shape and NaN count,
    including custom NaN-like values.

    Args:
        folder_path (str): The absolute path to the folder containing CSV files.
    """
    if not os.path.isdir(folder_path):
        print(f"Error: The directory '{folder_path}' does not exist.")
        return

    print(f"[REDACTED_BY_SCRIPT]")

    # Define a list of values to be treated as NaN
    missing_value_formats = ['0', '0.0', '-1']

    # Loop through all files in the specified directory
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            try:
                # Read the csv file into a DataFrame, converting specified values to NaN
                df = pd.read_csv(file_path, na_values=missing_value_formats)

                # Get the shape (rows, columns)
                rows, cols = df.shape

                # Calculate the total number of NaN values
                nan_count = df.isnull().sum().sum()

                # Print the collected information
                print(f"--- {filename} ---")
                print(f"[REDACTED_BY_SCRIPT]")
                print(f"[REDACTED_BY_SCRIPT]'0', '0.0', '-1'): {nan_count}\n")

            except Exception as e:
                print(f"[REDACTED_BY_SCRIPT]")

# Specify the path to your folder
directory_path = r'[REDACTED_BY_SCRIPT]'

# Run the analysis
analyze_csv_files(directory_path)