import pandas as pd
import ast

# --- Configuration ---
QUARANTINE_THRESHOLD = 15
TOXIC_OMISSIONS = [
    "[REDACTED_BY_SCRIPT]",
    "[REDACTED_BY_SCRIPT]",
    "[REDACTED_BY_SCRIPT]",
    "[REDACTED_BY_SCRIPT]",
    "[REDACTED_BY_SCRIPT]",
    "[REDACTED_BY_SCRIPT]",
    "[REDACTED_BY_SCRIPT]",
    "[REDACTED_BY_SCRIPT]",
    "[REDACTED_BY_SCRIPT]",
    "[REDACTED_BY_SCRIPT]",
    "[REDACTED_BY_SCRIPT]",
    "[REDACTED_BY_SCRIPT]",
    "[REDACTED_BY_SCRIPT]",
    "[REDACTED_BY_SCRIPT]",
    "[REDACTED_BY_SCRIPT]",
    "[REDACTED_BY_SCRIPT]",
    "[REDACTED_BY_SCRIPT]",
    "[REDACTED_BY_SCRIPT]",
    "[REDACTED_BY_SCRIPT]",
    "[REDACTED_BY_SCRIPT]"
]
DATA_FILE_PATH = r"[REDACTED_BY_SCRIPT]"

def count_properties_with_toxic_omissions(file_path, toxic_columns, threshold):
    """
    Counts properties with a high number of missing critical data points
    by analyzing the data integrity manifest.

    Args:
        file_path (str): The path to the data_integrity_manifest.csv file.
        toxic_columns (list): A list of column names considered 'toxic omissions'.
        threshold (int): The minimum number of toxic omissions to flag a property.

    Returns:
        int: The number of properties meeting or exceeding the threshold.
    """
    try:
        # Load the dataset
        df = pd.read_csv(file_path)
        
        # Use a set for efficient lookup
        toxic_omissions_set = set(toxic_columns)

        def count_toxic(imputed_list_str):
            try:
                # Safely evaluate the string representation of a list
                imputed_list = ast.literal_eval(imputed_list_str)
                # Count the intersection between the two sets of columns
                count = len(toxic_omissions_set.intersection(imputed_list))
                return count
            except (ValueError, SyntaxError):
                # Handle cases where the string is not a valid list
                return 0

        # Apply the counting function to each row
        df['toxic_count'] = df['imputed_vitals_list'].apply(count_toxic)

        # Count how many properties have a toxic count >= the threshold
        quarantined_properties_count = (df['toxic_count'] >= threshold).sum()

        return quarantined_properties_count

    except FileNotFoundError:
        print(f"[REDACTED_BY_SCRIPT]")
        return None
    except KeyError as e:
        print(f"[REDACTED_BY_SCRIPT]")
        return None
    except Exception as e:
        print(f"[REDACTED_BY_SCRIPT]")
        return None

if __name__ == "__main__":
    flagged_count = count_properties_with_toxic_omissions(
        DATA_FILE_PATH, 
        TOXIC_OMISSIONS, 
        QUARANTINE_THRESHOLD
    )

    if flagged_count is not None:
        print(f"[REDACTED_BY_SCRIPT]")
