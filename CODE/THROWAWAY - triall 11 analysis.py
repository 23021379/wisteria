import pandas as pd
import sys

# =============================================================================
# --- Configuration ---
# Architect's Note: Encapsulate all user-configurable parameters here for clarity.
# This ensures a single source of truth for the script's operation.
# =============================================================================
EVALUATION_CSV_PATH = r"[REDACTED_BY_SCRIPT]"
SHAP_CSV_PATH = r"[REDACTED_BY_SCRIPT]"
OUTPUT_REPORT_PATH = r"[REDACTED_BY_SCRIPT]"

# --- Analysis Thresholds ---
# The dual-threshold system is mandated to prevent misleading results.
PERCENTAGE_ERROR_THRESHOLD = 0.15  # Properties with >15% error are flagged.
MIN_ABSOLUTE_ERROR_THRESHOLD = 5000 # Ignore high-percentage errors if the monetary value is insignificant.
TOP_N_PROPERTIES_TO_REPORT = 500    # Limit the report to the top N worst offenders for focused analysis.

def load_and_prepare_data(eval_path: str, shap_path: str) -> pd.DataFrame:
    """
    Loads evaluation and SHAP data, performs a robust inner merge, and calculates
    the primary analysis metric (percentage_error).
    
    Args:
        eval_path (str): Path to the holdout evaluation results CSV.
        shap_path (str): Path to the structured SHAP report CSV.

    Returns:
        pd.DataFrame: A merged and prepared DataFrame ready for analysis.
    """
    print(f"[REDACTED_BY_SCRIPT]")
    try:
        df_eval = pd.read_csv(eval_path)
    except FileNotFoundError:
        print(f"[REDACTED_BY_SCRIPT]", file=sys.stderr)
        sys.exit(1)

    print(f"[REDACTED_BY_SCRIPT]")
    try:
        df_shap = pd.read_csv(shap_path)
    except FileNotFoundError:
        print(f"[REDACTED_BY_SCRIPT]", file=sys.stderr)
        sys.exit(1)

    # Use only the necessary columns to keep memory usage low and define a clear contract.
    eval_cols = ['property_id', '[REDACTED_BY_SCRIPT]', 'predicted_price', 'absolute_error']
    shap_cols = ['property_id', 'contrib_feat_1', 'contrib_feat_1_shap', 
                 'contrib_feat_2', 'contrib_feat_2_shap', 'contrib_feat_3', 'contrib_feat_3_shap']

    # Defensive check for required columns
    for col in eval_cols:
        if col not in df_eval.columns:
            print(f"[REDACTED_BY_SCRIPT]'{col}'[REDACTED_BY_SCRIPT]", file=sys.stderr)
            sys.exit(1)
    for col in shap_cols:
         if col not in df_shap.columns:
            print(f"[REDACTED_BY_SCRIPT]'{col}'[REDACTED_BY_SCRIPT]", file=sys.stderr)
            sys.exit(1)

    print("[REDACTED_BY_SCRIPT]'property_id'[REDACTED_BY_SCRIPT]")
    # This inner merge is a critical validation gate, preventing analysis of misaligned data.
    df_merged = pd.merge(df_eval[eval_cols], df_shap[shap_cols], on='property_id', how='inner')
    
    if df_merged.empty:
        print("[REDACTED_BY_SCRIPT]'property_id'[REDACTED_BY_SCRIPT]", file=sys.stderr)
        sys.exit(1)

    # Calculate the key metric for defining "blind spots".
    # Replace zeros in sale price with a small number to avoid division by zero errors.
    df_merged['percentage_error'] = df_merged['absolute_error'] / df_merged['[REDACTED_BY_SCRIPT]'].replace(0, 1e-6)
    
    print(f"[REDACTED_BY_SCRIPT]")
    return df_merged

def analyze_blind_spots(df: pd.DataFrame, pct_threshold: float, abs_threshold: int, top_n: int) -> pd.DataFrame:
    """
    Filters the dataframe to identify the top N properties that represent the model's
    blind spots, based on dual error thresholds.

    Args:
        df (pd.DataFrame): The prepared and merged DataFrame.
        pct_threshold (float): The minimum percentage error to be considered a blind spot.
        abs_threshold (int): The minimum absolute error to be considered a blind spot.
        top_n (int): The number of top blind spot properties to return.

    Returns:
        pd.DataFrame: A sorted DataFrame containing the top N worst-performing properties.
    """
    print(f"[REDACTED_BY_SCRIPT]")
    # Dual-threshold filtering for robust identification.
    blind_spots_df = df[
        (df['percentage_error'] > pct_threshold) &
        (df['absolute_error'] > abs_threshold)
    ].copy() # .copy() is used to prevent SettingWithCopyWarning.

    print(f"[REDACTED_BY_SCRIPT]")
    # Sort to find the absolute worst offenders for focused review.
    blind_spots_df.sort_values(by='percentage_error', ascending=False, inplace=True)

    return blind_spots_df.head(top_n)

def generate_report(df_report: pd.DataFrame, output_path: str):
    """
    Generates a human-readable text file summarizing the blind spot analysis.

    Args:
        df_report (pd.DataFrame): The filtered and sorted DataFrame of blind spots.
        output_path (str): The path to write the final report file.
    """
    print(f"[REDACTED_BY_SCRIPT]")
    with open(output_path, 'w') as f:
        f.write("=====================================================\n")
        f.write("[REDACTED_BY_SCRIPT]")
        f.write("=====================================================\n\n")

        if df_report.empty:
            f.write("[REDACTED_BY_SCRIPT]")
            f.write(f"[REDACTED_BY_SCRIPT]")
            print("[REDACTED_BY_SCRIPT]")
            return

        f.write(f"[REDACTED_BY_SCRIPT]")
        f.write(f"[REDACTED_BY_SCRIPT]")
        f.write("-----------------------------------------------------\n\n")

        for index, row in df_report.iterrows():
            f.write(f"Property ID: {row['property_id']}\n")
            f.write(f"  - Sale Price:      £{row['most_recent_sale_price']:,.0f}\n")
            f.write(f"[REDACTED_BY_SCRIPT]'predicted_price']:,.0f}\n")
            f.write(f"[REDACTED_BY_SCRIPT]'absolute_error']:,.0f}\n")
            f.write(f"[REDACTED_BY_SCRIPT]'percentage_error']:.1%}\n")
            f.write("[REDACTED_BY_SCRIPT]")
            f.write(f"    1. {row['contrib_feat_1']} ({row['contrib_feat_1_shap']:+.4f})\n")
            f.write(f"    2. {row['contrib_feat_2']} ({row['contrib_feat_2_shap']:+.4f})\n")
            f.write(f"    3. {row['contrib_feat_3']} ({row['contrib_feat_3_shap']:+.4f})\n")
            f.write("\n-----------------------------------------------------\n\n")
            
    print(f"[REDACTED_BY_SCRIPT]")

if __name__ == "__main__":
    prepared_data = load_and_prepare_data(EVALUATION_CSV_PATH, SHAP_CSV_PATH)
    top_blind_spots = analyze_blind_spots(
        prepared_data, 
        PERCENTAGE_ERROR_THRESHOLD, 
        MIN_ABSOLUTE_ERROR_THRESHOLD,
        TOP_N_PROPERTIES_TO_REPORT
    )
    generate_report(top_blind_spots, OUTPUT_REPORT_PATH)