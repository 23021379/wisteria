import pandas as pd
import numpy as np
import re
from datetime import datetime
import os
import matplotlib.pyplot as plt

# ==============================================================================
# 1. CONFIGURATION - SINGLE SOURCE OF TRUTH
# ==============================================================================
# --- Analysis Parameters ---
N_MIN = 5
Z_SCORE_THRESHOLD_HIGH = 0.5
Z_SCORE_THRESHOLD_LOW = -0.5

# --- File Paths ---
# NOTE: Assumes '[REDACTED_BY_SCRIPT]' is in the same directory.
#       Replace with the actual path if different.
INPUT_DATA_PATH = r'[REDACTED_BY_SCRIPT]'
FAILURE_LOG_PATH = r'[REDACTED_BY_SCRIPT]'

# --- Regular Expressions ---
# Canonical UK Postcode Regex for validation (conforms to BS 7666)
UK_POSTCODE_REGEX = re.compile(r'[REDACTED_BY_SCRIPT]')

# Parsing regex to extract components from the `property_id` string.
# It is designed to be robust to variations, like a missing county.
# It looks for the postcode at the end and works backwards.
ADDRESS_PARSER_REGEX = re.compile(
    r"""
    ^
    (?P<street>.*?)
    \s+
    (?P<town_city>[a-z\s]+?)
    \s+
    (?:(?P<county>[a-z\s]+?)\s+)?  # Optional county group
    (?P<postcode>[a-z]{1,2}\d[a-z\d]?\s?\d[a-z]{2})
    $
    """,
    re.IGNORECASE | re.VERBOSE
)

# ==============================================================================
# 2. COMPONENT: ADDRESS PARSING & VALIDATION SERVICE (Directive 2.1)
# ==============================================================================
def parse_address(property_id: str) -> dict | None:
    """
    Parses a property_id string into structured components.

    Performs critical validation and sanitization on the extracted postcode.

    Args:
        property_id: The raw property identifier string.

    Returns:
        A dictionary with keys {'street', 'town_city', 'county', 'postcode'}
        on successful parsing and validation, otherwise None.
    """
    if not isinstance(property_id, str) or not property_id.strip():
        return None

    match = ADDRESS_PARSER_REGEX.match(property_id.strip())
    if not match:
        return None

    components = match.groupdict()

    # --- Postcode Sanitization & Validation ---
    raw_postcode = components.get('postcode', '')
    if not raw_postcode:
        return None

    # Sanitize: Uppercase and ensure standard single space format
    sanitized_postcode = raw_postcode.upper().replace(" ", "")
    if len(sanitized_postcode) > 4:
         # Insert space before the last 3 characters (the "inward" code)
        sanitized_postcode = f"[REDACTED_BY_SCRIPT]"
    else: # Malformed postcode, will fail regex
        sanitized_postcode = raw_postcode.upper()

    # Validate: Check against the canonical UK postcode regex
    if not UK_POSTCODE_REGEX.match(sanitized_postcode):
        return None

    sanitized_postcode = sanitized_postcode.replace("0","").replace("1","").replace("2","").replace("3","").replace("4","").replace("5","").replace("6","").replace("7","").replace("8","").replace("9","")
    return {
        'street': components.get('street', '').strip().title(),
        'town_city': components.get('town_city', '').strip().title(),
        'county': components.get('county', '').strip().title() if components.get('county') else None,
        'postcode': sanitized_postcode
    }

# ==============================================================================
# 3. COMPONENT: HIERARCHICAL AGGREGATION & ANOMALY SCORING (Directive 3 & 4)
# ==============================================================================
def calculate_anomaly_scores(df: pd.DataFrame, group_by_col: str) -> pd.DataFrame:
    """
    Performs hierarchical aggregation and calculates anomaly Z-scores.

    Args:
        df: The feature-engineered DataFrame.
        group_by_col: The column to group by (e.g., 'postcode_district').

    Returns:
        A DataFrame with aggregated metrics and Z-scores for each group.
    """
    # --- 3.1 & 3.2: Hierarchical Aggregation ---
    agg_metrics = {
        'absolute_error': ['mean', 'median'],
        'property_id': 'count',
        'mape': 'mean'
    }
    grouped_df = df.groupby(group_by_col).agg(agg_metrics).reset_index()

    # Flatten MultiIndex columns
    grouped_df.columns = [
        group_by_col,
        'mean_ae',
        'median_ae',
        'property_count',
        'mean_mape'
    ]

    # --- 4.1: Minimum Viability Threshold (N_min) ---
    significant_groups = grouped_df[grouped_df['property_count'] >= N_MIN].copy()

    if significant_groups.empty:
        # Fail-fast: If no groups meet the threshold, return an empty frame
        significant_groups['z_score_mape'] = pd.Series(dtype='float64')
        return significant_groups

    # --- 4.2: Anomaly Scoring via Z-Score ---
    # CRITICAL: Calculate stats on the filtered (significant) population only
    total_mean_mape = significant_groups['mean_mape'].mean()
    total_std_dev_mape = significant_groups['mean_mape'].std()
    
    # Avoid division by zero if std dev is zero (all groups have same mean_mape)
    if total_std_dev_mape == 0:
        significant_groups['z_score_mape'] = 0.0
    else:
        significant_groups['z_score_mape'] = (
            (significant_groups['mean_mape'] - total_mean_mape) / total_std_dev_mape
        )

    return significant_groups

# ==============================================================================
# 4. COMPONENT: REPORT GENERATION (Directive 5)
# ==============================================================================
def generate_markdown_report(
    scored_districts: pd.DataFrame,
    full_df: pd.DataFrame,
    total_records: int,
    failed_records: int
) -> str:
    """
    Generates the final, stakeholder-ready Markdown report.
    """
    # --- Filter for anomalies ---
    high_error = scored_districts[scored_districts['z_score_mape'] >= Z_SCORE_THRESHOLD_HIGH].sort_values('z_score_mape', ascending=False).head(5)
    low_error = scored_districts[scored_districts['z_score_mape'] <= Z_SCORE_THRESHOLD_LOW].sort_values('z_score_mape', ascending=True).head(5)

    # --- Prepare tables for markdown ---
    def format_table(df, columns):
        df_copy = df[columns].copy()
        df_copy['mean_ae'] = df_copy['mean_ae'].apply(lambda x: f"{x:,.2f}")
        df_copy['mean_mape'] = df_copy['mean_mape'].apply(lambda x: f"{x * 100:.2f}%")
        df_copy['z_score_mape'] = df_copy['z_score_mape'].apply(lambda x: f"{x:.2}")
        return df_copy.to_markdown(index=False)

    high_error_table_cols = ['postcode_district', 'property_count', 'mean_ae', 'mean_mape', 'z_score_mape']
    low_error_table_cols = ['postcode_district', 'property_count', 'mean_ae', 'mean_mape', 'z_score_mape']

    # --- Find example properties ---
    def find_example(df, district_df, full_df, sort_col, ascending):
        if not district_df.empty:
            target_district = district_df.iloc[0]['postcode_district']
            example = full_df[full_df['postcode_district'] == target_district].sort_values(sort_col, ascending=ascending).iloc[0]
            return (
                f"[REDACTED_BY_SCRIPT]'property_id']}`, "
                f"[REDACTED_BY_SCRIPT]'most_recent_sale_price']):,}, "
                f"[REDACTED_BY_SCRIPT]'absolute_error']):,} "
                f"({example['mape'[REDACTED_BY_SCRIPT]"
            )
        return "[REDACTED_BY_SCRIPT]"

    high_error_example = find_example(high_error, high_error, full_df, 'mape', ascending=False)
    low_error_example = find_example(low_error, low_error, full_df, 'mape', ascending=True)

    # --- Assemble the final report ---
    report = f"""
### **Wisteria Model Performance Report: Geospatial Error Analysis**

**Analysis Date:** {datetime.now().strftime('%Y-%m-%d')}
**Methodology:** This report identifies geographic areas of abnormal model performance by calculating the Mean Absolute Percentage Error (MAPE) grouped by Postcode District. Anomaly detection is performed using a Z-score statistical test, with a significance threshold of |Z| ≥ {Z_SCORE_THRESHOLD_HIGH} and a minimum sample size of {N_MIN} properties per group.

**Key Findings:**
*   A total of **{total_records:,}** properties were analyzed. **{failed_records:,}** properties failed the initial address parsing and validation stage.

---

#### **Top 5 Most Abnormally High-Error (Underperforming) Postcode Districts**

{format_table(high_error, high_error_table_cols)}

**Example High-Error Property:** {high_error_example}

---

#### **Top 5 Most Abnormally Low-Error (Overperforming) Postcode Districts**

{format_table(low_error, low_error_table_cols)}

**Example Low-Error Property:** {low_error_example}
"""
    return report.strip()

# ==============================================================================
# 5. MAIN EXECUTION ORCHESTRATOR
# ==============================================================================
def main():
    """
    Main function to orchestrate the entire analysis pipeline.
    """
    # --- Define Output File ---
    REPORT_FILE_PATH = 'analysis_report.txt'
    MARKDOWN_REPORT_PATH = '[REDACTED_BY_SCRIPT]'

    # --- 1. Data Ingestion ---
    if not os.path.exists(INPUT_DATA_PATH):
        print(f"[REDACTED_BY_SCRIPT]'{INPUT_DATA_PATH}'[REDACTED_BY_SCRIPT]")
        return

    # --- UPDATED: Custom CSV Parsing to handle commas in property_id ---
    with open(INPUT_DATA_PATH, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    header = lines[0].strip().split(',')
    data = []
    for line in lines[1:]:
        parts = line.strip().split(',')
        # The last 6 columns are fixed, the rest belong to property_id
        property_id = ','.join(parts[:-6])
        other_data = parts[-6:]
        row = [property_id] + other_data
        data.append(row)

    df = pd.DataFrame(data, columns=header)

    # Convert columns to appropriate data types
    numeric_cols = ['[REDACTED_BY_SCRIPT]', 'is_data_deficient', '[REDACTED_BY_SCRIPT]', 'predicted_price', 'absolute_error']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    # --- END UPDATE ---

    total_records = len(df)
    
    # --- 2. Address Parsing & Validation Pipeline ---
    parsed_data = []
    failures = []
    
    for _, row in df.iterrows():
        property_id = row['property_id']
        parsed_address = parse_address(property_id)
        
        if parsed_address:
            # Combine original data with new structured address
            new_row = {**row.to_dict(), **parsed_address}
            parsed_data.append(new_row)
        else:
            failures.append({
                'property_id': property_id,
                'failure_reason': '[REDACTED_BY_SCRIPT]'
            })

    # CRITICAL: Log failures for diagnostic review
    if failures:
        pd.DataFrame(failures).to_csv(FAILURE_LOG_PATH, index=False)
    
    if not parsed_data:
        print("[REDACTED_BY_SCRIPT]")
        return

    # Create the main working DataFrame
    structured_df = pd.DataFrame(parsed_data)
    
    # --- 3. Feature Engineering & Metric Calculation ---
    structured_df['postcode_area'] = structured_df['postcode'].str.split(' ').str[0].str.extract(r'([A-Z]+)')
    structured_df['postcode_district'] = structured_df['postcode'].str.split(' ').str[0]
    
    # Calculate MAPE, with defensive check for zero division
    structured_df['mape'] = np.where(
        structured_df['[REDACTED_BY_SCRIPT]'] > 0,
        structured_df['absolute_error'] / structured_df['[REDACTED_BY_SCRIPT]'],
        0
    )
    
    # Open the report file to start writing the analysis
    with open(REPORT_FILE_PATH, 'w', encoding='utf-8') as f:
        f.write(f"[REDACTED_BY_SCRIPT]")
        f.write(f"[REDACTED_BY_SCRIPT]'%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*50 + "\n\n")

        # --- NEW: Directional Error (Bias) Analysis ---
        # Calculate raw error and percentage error to check for bias
        structured_df['raw_error'] = structured_df['predicted_price'] - structured_df['[REDACTED_BY_SCRIPT]']
        structured_df['percentage_error'] = np.where(
            structured_df['[REDACTED_BY_SCRIPT]'] > 0,
            structured_df['raw_error'] / structured_df['[REDACTED_BY_SCRIPT]'],
            0
        )

        # Aggregate to find mean percentage error (a measure of bias)
        bias_analysis = structured_df.groupby('postcode_district').agg(
            mean_percentage_error=('percentage_error', 'mean'),
            property_count=('property_id', 'count')
        ).reset_index()

        # Filter for significant groups
        significant_bias = bias_analysis[bias_analysis['property_count'] >= N_MIN]

        f.write("[REDACTED_BY_SCRIPT]")
        f.write(significant_bias.sort_values('mean_percentage_error', ascending=False).head().to_markdown(index=False))
        f.write("\n\n")

        f.write("[REDACTED_BY_SCRIPT]")
        f.write(significant_bias.sort_values('mean_percentage_error', ascending=True).head().to_markdown(index=False))
        f.write("\n\n")
        # --- END NEW ---
        
        # --- Centralized Derived Feature Engineering ---
        # Create all derived columns here to ensure they are available for all subsequent analyses.
        
        # 1. Property Type Extraction
        def get_prop_type(prop_id):
            prop_id_lower = prop_id.lower()
            if 'flat' in prop_id_lower: return 'Flat'
            if 'apartment' in prop_id_lower: return 'Apartment'
            if 'maisonette' in prop_id_lower: return 'Maisonette'
            return 'House'
        structured_df['property_type'] = structured_df['property_id'].apply(get_prop_type)

        # 2. Price Band Categorization
        price_bins = [0, 250000, 500000, 1000000, float('inf')]
        price_labels = ['< £250k', '£250k-£500k', '£500k-£1M', '> £1M']
        structured_df['price_band'] = pd.cut(
            structured_df['[REDACTED_BY_SCRIPT]'], 
            bins=price_bins, 
            labels=price_labels, 
            right=False
        )
        # --- END Centralized Engineering ---

        # --- ENHANCED: Exhaustive Analysis by Routing Decision ---
        f.write("[REDACTED_BY_SCRIPT]")
        f.write("[REDACTED_BY_SCRIPT]")

        # 1. Detailed Overall Performance Metrics per Route
        # Add squared error for RMSE calculation
        structured_df['squared_error'] = structured_df['raw_error'] ** 2
        
        routing_analysis_detailed = structured_df.groupby('routing_decision').agg(
            property_count=('property_id', 'count'),
            mean_ae=('absolute_error', 'mean'),
            median_ae=('absolute_error', 'median'),
            mean_mape=('mape', 'mean'),
            bias_mean_pe=('percentage_error', 'mean'),
            rmse=('squared_error', lambda x: np.sqrt(x.mean()))
        ).reset_index()

        # Calculate percentage of total properties
        routing_analysis_detailed['percentage_of_total'] = (routing_analysis_detailed['property_count'] / total_records) * 100

        # Reorder for clarity
        routing_analysis_detailed = routing_analysis_detailed[[
            'routing_decision', 'property_count', 'percentage_of_total', 
            'mean_ae', 'median_ae', 'mean_mape', 'rmse', 'bias_mean_pe'
        ]]

        f.write("[REDACTED_BY_SCRIPT]")
        f.write(routing_analysis_detailed.sort_values('mean_mape', ascending=False).to_markdown(index=False, floatfmt=",.4f"))
        f.write("\n\n")

        # 2. Cross-Analysis: Performance by Price Band within each Route
        routing_price_band_cross = structured_df.groupby(['routing_decision', 'price_band'], observed=True).agg(
            property_count=('property_id', 'count'),
            mean_mape=('mape', 'mean')
        ).reset_index()

        pivot_routing_price = routing_price_band_cross.pivot_table(
            index='routing_decision',
            columns='price_band',
            values='mean_mape'
        )

        f.write("[REDACTED_BY_SCRIPT]")
        f.write(pivot_routing_price.applymap(lambda x: f'{x:.2%}' if pd.notna(x) else 'N/A').to_markdown())
        f.write("\n\n")

        # 3. Cross-Analysis: Performance by Property Type within each Route
        routing_prop_type_cross = structured_df.groupby(['routing_decision', 'property_type']).agg(
            property_count=('property_id', 'count'),
            mean_mape=('mape', 'mean')
        ).reset_index()

        pivot_routing_prop_type = routing_prop_type_cross.pivot_table(
            index='routing_decision',
            columns='property_type',
            values='mean_mape'
        )

        f.write("[REDACTED_BY_SCRIPT]")
        f.write(pivot_routing_prop_type.applymap(lambda x: f'{x:.2%}' if pd.notna(x) else 'N/A').to_markdown())
        f.write("\n\n")
        # --- END ENHANCED ---

        # --- Standard Analysis by Property Type ---
        prop_type_analysis = structured_df.groupby('property_type').agg(
            mean_mape=('mape', 'mean'),
            median_ae=('absolute_error', 'median'),
            property_count=('property_id', 'count')
        ).reset_index()

        f.write("[REDACTED_BY_SCRIPT]")
        f.write(prop_type_analysis.to_markdown(index=False))
        f.write("\n\n")

        # --- Standard Analysis by Price Band ---
        price_band_analysis = structured_df.groupby('price_band', observed=True).agg(
            mean_mape=('mape', 'mean'),
            median_ae=('absolute_error', 'median'),
            property_count=('property_id', 'count')
        ).reset_index()
        
        f.write("[REDACTED_BY_SCRIPT]")
        f.write(price_band_analysis.to_markdown(index=False))
        f.write("\n\n")

        try:
            # --- NEW: Analysis by Jurisdiction ID ---
            jurisdiction_analysis = structured_df.groupby('jurisdiction_id').agg(
                mean_mape=('mape', 'mean'),
                median_ae=('absolute_error', 'median'),
                property_count=('property_id', 'count')
            ).reset_index()

            f.write("[REDACTED_BY_SCRIPT]")
            f.write(jurisdiction_analysis.to_markdown(index=False))
            f.write("\n\n")
            # --- END NEW ---
        except KeyError:
            f.write("[REDACTED_BY_SCRIPT]")

        # --- NEW: Cross-Dimensional Analysis ---
        try:
            cross_analysis = structured_df.groupby(['jurisdiction_id', 'price_band']).agg(
                mean_mape=('mape', 'mean'),
                property_count=('property_id', 'count')
            ).reset_index()

            # Pivot for better readability
            pivot_cross_analysis = cross_analysis.pivot_table(
                index='jurisdiction_id', 
                columns='price_band', 
                values='mean_mape'
            )

            f.write("[REDACTED_BY_SCRIPT]")
            # Format as percentage for printing
            f.write(pivot_cross_analysis.applymap(lambda x: f'{x:.2%}' if pd.notna(x) else 'N/A').to_markdown())
            f.write("\n\n")
        except KeyError:
            f.write("[REDACTED_BY_SCRIPT]'jurisdiction_id' column. ---\n\n")
        # --- END NEW ---

        # --- NEW: Identify Top 5 Individual Outliers by MAPE ---
        top_5_outliers = structured_df.sort_values('mape', ascending=False).head(5)
        f.write("[REDACTED_BY_SCRIPT]")
        f.write(top_5_outliers[['property_id', '[REDACTED_BY_SCRIPT]', 'predicted_price', 'mape']].to_markdown(index=False))
        f.write("\n\n")
        # --- END NEW ---

        # --- 4. Hierarchical Aggregation & Scoring ---
        scored_districts = calculate_anomaly_scores(structured_df, 'postcode_district')
        # Note: Analysis by 'postcode_area' or 'town_city' would follow the same pattern:
        scored_areas = calculate_anomaly_scores(structured_df, 'postcode_area')
        scored_towns = calculate_anomaly_scores(structured_df, 'town_city')
        
        # You can then print or generate reports for these as well
        f.write("[REDACTED_BY_SCRIPT]")
        f.write(str(scored_areas.sort_values('z_score_mape', ascending=False).head()))
        f.write("\n\n")

        # --- 5. Report Generation ---
        report = generate_markdown_report(
            scored_districts=scored_districts,
            full_df=structured_df,
            total_records=total_records,
            failed_records=len(failures)
        )
        
        # Write the stakeholder markdown report to its own file
        with open(MARKDOWN_REPORT_PATH, 'w', encoding='utf-8') as md_file:
            md_file.write(report)
        
        f.write("[REDACTED_BY_SCRIPT]")
        f.write("[REDACTED_BY_SCRIPT]")
        f.write(report)

    print(f"[REDACTED_BY_SCRIPT]'{REPORT_FILE_PATH}'")
    print(f"[REDACTED_BY_SCRIPT]'{MARKDOWN_REPORT_PATH}'")

    # --- NEW: Visualization of MAPE Distribution by Price Band ---
    plt.figure(figsize=(10, 6))
    # Limit MAPE to 1 (100%) for better visualization of the main distribution
    structured_df['mape_capped'] = structured_df['mape'].clip(upper=1)
    structured_df.boxplot(column='mape_capped', by='price_band', grid=False)
    plt.title('[REDACTED_BY_SCRIPT]')
    plt.suptitle('') # Suppress the default title
    plt.xlabel('Price Band')
    plt.ylabel('[REDACTED_BY_SCRIPT]')
    plt.tight_layout()
    
    # Save the plot to a file
    output_plot_path = '[REDACTED_BY_SCRIPT]'
    plt.savefig(output_plot_path)
    print(f"[REDACTED_BY_SCRIPT]'{output_plot_path}'")
    # --- END NEW ---

    # --- NEW: Visualization of Absolute Error vs. Sale Price ---
    plt.figure(figsize=(10, 6))
    plt.scatter(structured_df['[REDACTED_BY_SCRIPT]'], structured_df['absolute_error'], alpha=0.5)
    plt.title('[REDACTED_BY_SCRIPT]')
    plt.xlabel('[REDACTED_BY_SCRIPT]')
    plt.ylabel('Absolute Error (£)')
    plt.grid(True)
    plt.tight_layout()
    
    # Save the plot to a file
    output_scatter_path = '[REDACTED_BY_SCRIPT]'
    plt.savefig(output_scatter_path)
    print(f"[REDACTED_BY_SCRIPT]'{output_scatter_path}'")
    # --- END NEW ---

    # --- NEW: Visualization of MAPE Distribution by Routing Decision ---
    plt.figure(figsize=(12, 7))
    structured_df['mape_capped'] = structured_df['mape'].clip(upper=1) # Cap for better visualization
    structured_df.boxplot(column='mape_capped', by='routing_decision', grid=False)
    plt.title('[REDACTED_BY_SCRIPT]')
    plt.suptitle('') # Suppress default title
    plt.xlabel('Routing Decision')
    plt.ylabel('[REDACTED_BY_SCRIPT]')
    plt.xticks(rotation=10)
    plt.tight_layout()
    
    output_boxplot_routing_path = '[REDACTED_BY_SCRIPT]'
    plt.savefig(output_boxplot_routing_path)
    print(f"[REDACTED_BY_SCRIPT]'{output_boxplot_routing_path}'")
    # --- END NEW ---

    # --- NEW: Visualization of Model Bias by Routing Decision ---
    bias_by_route = structured_df.groupby('routing_decision')['percentage_error'].mean().sort_values()
    
    plt.figure(figsize=(12, 7))
    bias_by_route.plot(kind='bar', color=(bias_by_route > 0).map({True: 'r', False: 'g'}))
    plt.axhline(0, color='k', linestyle='--', linewidth=0.8)
    plt.title('[REDACTED_BY_SCRIPT]')
    plt.xlabel('Routing Decision')
    plt.ylabel('[REDACTED_BY_SCRIPT]')
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.2%}'.format(y)))
    plt.xticks(rotation=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    output_bias_barchart_path = '[REDACTED_BY_SCRIPT]'
    plt.savefig(output_bias_barchart_path)
    print(f"[REDACTED_BY_SCRIPT]'{output_bias_barchart_path}'")
    # --- END NEW ---


if __name__ == "__main__":
    main()