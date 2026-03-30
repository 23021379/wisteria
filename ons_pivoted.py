import pandas as pd
from pathlib import Path
import re
import traceback # For more detailed error printing if needed

def clean_col_name(name):
    """[REDACTED_BY_SCRIPT]"""
    name = str(name)  # Ensure it's a string
    # Remove text in parentheses (e.g., "(13 categories)")
    name = re.sub(r'\([^)]*\)', '', name)
    name = name.strip()
    # Replace whitespace and non-alphanumeric characters (except underscore) with a single underscore
    name = re.sub(r'[\s\W]+', '_', name)
    name = re.sub(r'_+', '_', name)      # Replace multiple underscores with a single one
    name = name.strip('_')               # Remove leading/trailing underscores
    return name

def get_variable_prefix(file_path, df_columns_list):
    """
    Generates a prefix for new columns based on filename or DataFrame content.
    df_columns_list is the list of original column names from the current CSV.
    """
    filename = file_path.name
    
    if filename.startswith("ons-"):
        # e.g., "[REDACTED_BY_SCRIPT]" -> "dependent_children"
        prefix_base = filename.replace("ons-", "").replace(".csv", "")
    elif "custom-filtered" in filename:
        # For the "custom-filtered..." file, use its specific variable description
        # This assumes the 4th column header (index 3) is descriptive of the variable
        if len(df_columns_list) > 3:
            prefix_base = df_columns_list[3] 
        else:
            # Fallback if the file has fewer than 4 columns
            prefix_base = filename.replace(".csv","") 
    else:
        # For any other filenames
        prefix_base = filename.replace(".csv", "")
    
    # Clean the generated prefix_base
    prefix = clean_col_name(prefix_base)
    return prefix


def compile_ons_files(directory_path_str, output_file_path_str):
    directory_path = Path(directory_path_str)
    output_file_path = Path(output_file_path_str)

    if not directory_path.is_dir():
        print(f"Error: Directory '{directory_path_str}' not found.")
        return

    all_csv_files = sorted(list(directory_path.glob("*.csv")))

    if not all_csv_files:
        print(f"[REDACTED_BY_SCRIPT]'{directory_path_str}'.")
        return

    # Prevent processing the output file if it's in the same directory and already exists
    all_csv_files = [f for f in all_csv_files if f.resolve() != output_file_path.resolve()]
    
    if not all_csv_files:
        print(f"[REDACTED_BY_SCRIPT]'{output_file_path.name}').")
        return
        
    print(f"[REDACTED_BY_SCRIPT]")

    # These are the standard names we expect for the key columns for merging.
    standard_key_cols = ['Output Areas Code', 'Output Areas']
    
    list_of_dataframes_to_merge = []

    for i, file_path in enumerate(all_csv_files):
        print(f"[REDACTED_BY_SCRIPT]")
        try:
            # Read only the header row first to get original column names reliably
            # This helps if pandas might otherwise infer numeric headers or alter spaces
            df_original_columns = pd.read_csv(file_path, encoding='utf-8', nrows=0).columns.tolist()

            # Now read the full CSV
            try:
                df = pd.read_csv(file_path, encoding='utf-8', low_memory=False)
            except UnicodeDecodeError:
                print(f"[REDACTED_BY_SCRIPT]")
                df = pd.read_csv(file_path, encoding='latin-1', low_memory=False)

            if df.empty:
                print(f"[REDACTED_BY_SCRIPT]")
                continue
            
            # Based on your output, the structure is consistently:
            # Col 0 (index 0): Output Areas Code
            # Col 1 (index 1): Output Areas
            # Col 2 (index 2): Variable Specific Code (e.g., '[REDACTED_BY_SCRIPT]')
            # Col 3 (index 3): Variable Specific Description (e.g., '[REDACTED_BY_SCRIPT]')
            # Col 4 (index 4): Observation
            
            if len(df.columns) < 5:
                print(f"[REDACTED_BY_SCRIPT]")
                continue

            # Use original column names for processing this file
            current_oa_code_col = df_original_columns[0]
            current_oa_col = df_original_columns[1]
            # The *values* in this column will become the new pivoted column names
            category_desc_col_for_pivot_values = df_original_columns[3] 
            # The *values* in this column are the data for the pivoted columns
            observation_col_for_pivot_data = df_original_columns[4]   

            current_index_cols_for_pivot = [current_oa_code_col, current_oa_col]

            # Ensure all identified columns are actually present in the loaded DataFrame
            critical_cols_for_file = current_index_cols_for_pivot + \
                                     [category_desc_col_for_pivot_values, observation_col_for_pivot_data]
            missing_cols_in_df = [col for col in critical_cols_for_file if col not in df.columns]
            if missing_cols_in_df:
                print(f"[REDACTED_BY_SCRIPT]"
                      f"[REDACTED_BY_SCRIPT]"
                      f"[REDACTED_BY_SCRIPT]")
                continue
            
            # Ensure the column used for new column names in pivot is suitable (e.g., string)
            if not pd.api.types.is_string_dtype(df[category_desc_col_for_pivot_values]) and \
               not pd.api.types.is_object_dtype(df[category_desc_col_for_pivot_values]):
                 df[category_desc_col_for_pivot_values] = df[category_desc_col_for_pivot_values].astype(str)

            try:
                df_pivoted = df.pivot_table(
                    index=current_index_cols_for_pivot,
                    columns=category_desc_col_for_pivot_values,
                    values=observation_col_for_pivot_data,
                    fill_value=0  # Fill missing category combinations for an OA within this file with 0
                )
            except Exception as e:
                print(f"[REDACTED_BY_SCRIPT]")
                print(f"[REDACTED_BY_SCRIPT]"
                      f"Columns from: '{category_desc_col_for_pivot_values}'[REDACTED_BY_SCRIPT]"
                      f"Values from: '{observation_col_for_pivot_data}'[REDACTED_BY_SCRIPT]")
                print(f"[REDACTED_BY_SCRIPT]'{category_desc_col_for_pivot_values}': "
                      f"[REDACTED_BY_SCRIPT]")
                traceback.print_exc()
                continue
            
            df_pivoted.reset_index(inplace=True) # Turn index columns back into regular columns

            # Generate a prefix for the new (pivoted) data columns based on filename/variable
            variable_prefix = get_variable_prefix(file_path, df_original_columns)
            
            # Rename pivoted columns: prefix + cleaned_category_value
            # The columns to rename are those that are NOT the original index columns
            renamed_pivoted_cols_map = {}
            for col_name_from_pivot in df_pivoted.columns:
                if col_name_from_pivot not in current_index_cols_for_pivot:
                    clean_category_value = clean_col_name(col_name_from_pivot)
                    renamed_pivoted_cols_map[col_name_from_pivot] = f"[REDACTED_BY_SCRIPT]"
            df_pivoted.rename(columns=renamed_pivoted_cols_map, inplace=True)

            # Standardize the key column names (originally index columns) to `standard_key_cols` for merging
            key_rename_map = {}
            if current_oa_code_col != standard_key_cols[0]: # e.g. if original was "Output Areas Code " with a space
                key_rename_map[current_oa_code_col] = standard_key_cols[0]
            if current_oa_col != standard_key_cols[1]:
                key_rename_map[current_oa_col] = standard_key_cols[1]
            if key_rename_map:
                df_pivoted.rename(columns=key_rename_map, inplace=True)
            
            list_of_dataframes_to_merge.append(df_pivoted)

        except pd.errors.EmptyDataError:
            print(f"[REDACTED_BY_SCRIPT]")
        except Exception as e:
            print(f"[REDACTED_BY_SCRIPT]")
            traceback.print_exc()

    if not list_of_dataframes_to_merge:
        print("[REDACTED_BY_SCRIPT]")
        return

    # Iteratively merge all processed DataFrames
    print(f"[REDACTED_BY_SCRIPT]")
    merged_df = list_of_dataframes_to_merge[0]
    for i in range(1, len(list_of_dataframes_to_merge)):
        # Ensure the DataFrame being merged has the standard key column names
        df_to_merge = list_of_dataframes_to_merge[i]
        if not all(key_col in df_to_merge.columns for key_col in standard_key_cols):
            print(f"[REDACTED_BY_SCRIPT]'s missing standard key columns: {standard_key_cols}. "
                  f"[REDACTED_BY_SCRIPT]")
            continue
        merged_df = pd.merge(merged_df, df_to_merge, on=standard_key_cols, how='outer')

    # Optional: Fill NaNs that result from outer merges.
    # These NaNs mean an Output Area didn't have data from one of the source files.
    # Filling with 0 assumes missing data means 0 observations for that entire variable/file block.
    # Consider if this is appropriate for your data; otherwise, leave as NaN.
    # merged_df.fillna(0, inplace=True) 

    # Ensure key columns are first, and sort other columns alphabetically for consistency
    if standard_key_cols[0] in merged_df.columns and standard_key_cols[1] in merged_df.columns:
        cols_to_front = standard_key_cols
        other_cols = sorted([col for col in merged_df.columns if col not in cols_to_front])
        merged_df = merged_df[cols_to_front + other_cols]
    else:
        print("[REDACTED_BY_SCRIPT]")


    try:
        merged_df.to_csv(output_file_path, index=False, encoding='utf-8')
        print(f"[REDACTED_BY_SCRIPT]")
        print(f"[REDACTED_BY_SCRIPT]")
        num_cols_to_show = min(10, len(merged_df.columns))
        print(f"[REDACTED_BY_SCRIPT]")
    except Exception as e:
        print(f"[REDACTED_BY_SCRIPT]")
        traceback.print_exc()

# --- Main execution ---
if __name__ == "__main__":
    # Path to the directory containing the ONS CSV files
    source_directory = r"[REDACTED_BY_SCRIPT]"
    
    # Path for the output compiled CSV file
    # It's good practice to save it outside the source directory or with a distinct name.
    compiled_output_file = r"[REDACTED_BY_SCRIPT]" 

    compile_ons_files(source_directory, compiled_output_file)