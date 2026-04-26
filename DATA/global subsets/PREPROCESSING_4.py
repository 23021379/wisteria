import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import re

# --- Configuration for Subset 4 ---
FILE_PATH_SUBSET4 = "[REDACTED_BY_SCRIPT]"  # <--- *** USER: PLEASE UPDATE THIS PATH ***
SAMPLE_SIZE = 50000  # Or your preferred sample size

# Initial columns to drop (identifiers, etc.)
COLS_TO_DROP_SUBSET4 = [
    'pcds', 
    'LSOA21CD' 
    # Consider adding PXL_COUNT, FINAL_PXL_COUNT, PCT_FILTERED later if they consistently show low variance
    # and are deemed unnecessary after confirming overall data quality of vegetation metrics.
]

# Subset 4 is predominantly numerical. If there are any true categorical features beyond IDs, list them here.
INITIAL_CATEGORICAL_COLS_SUBSET4 = [
    # Example: '[REDACTED_BY_SCRIPT]'
]

# --- USER-DEFINED LIST FOR MANUAL REMOVALS AFTER REVIEW ---
# Populate this list after reviewing the output of the first run(s)
MANUAL_DROPS_S4 = [
    '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]',
    'NDVI_MEDIAN',
    'PCT_FILTERED',
    'PXL_COUNT',
    '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]'
]


# Thresholds
LOW_VARIANCE_THRESHOLD = 0.01  # For general numerical features post-scaling
HIGH_CORRELATION_THRESHOLD = 0.95 # For identifying collinear pairs
# OHE_LOW_VARIANCE_THRESHOLD_FOR_REMOVAL is less relevant here if no major OHE happens
OHE_LOW_VARIANCE_THRESHOLD_FOR_REMOVAL = 0.002 

def load_data(file_path):
    encodings_to_try = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
    for encoding in encodings_to_try:
        try:
            df = pd.read_csv(file_path, low_memory=False, encoding=encoding)
            print(f"[REDACTED_BY_SCRIPT]")
            return df
        except UnicodeDecodeError:
            print(f"[REDACTED_BY_SCRIPT]")
        except FileNotFoundError:
            print(f"[REDACTED_BY_SCRIPT]")
            return None
    print(f"[REDACTED_BY_SCRIPT]")
    return None

def preprocess_data_s4(df, cols_to_drop_initial, initial_categorical_cols_config):
    print(f"[REDACTED_BY_SCRIPT]")
    
    df_processed = df.drop(columns=cols_to_drop_initial, errors='ignore')
    print(f"[REDACTED_BY_SCRIPT]")

    # No specific feature engineering like 'is_terminated' for S4
    # Ensure specified categorical columns are string type
    for col in initial_categorical_cols_config:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].astype(str)

    numerical_cols = df_processed.select_dtypes(include=np.number).columns.tolist()
    categorical_cols_from_dtype = df_processed.select_dtypes(include=['object', 'category']).columns.tolist()
    
    final_categorical_cols = []
    final_numerical_cols = list(numerical_cols)

    for col in initial_categorical_cols_config:
        if col in df_processed.columns:
            if col not in final_categorical_cols:
                final_categorical_cols.append(col)
            if col in final_numerical_cols:
                final_numerical_cols.remove(col)
    
    for col in categorical_cols_from_dtype:
        if col not in final_categorical_cols:
            final_categorical_cols.append(col)
            
    all_cols_set = set(df_processed.columns)
    processed_cols_set = set(final_numerical_cols + final_categorical_cols)
    if all_cols_set != processed_cols_set:
        missing_from_typed_lists = all_cols_set - processed_cols_set
        print(f"[REDACTED_BY_SCRIPT]")
        for m_col in missing_from_typed_lists:
            if m_col not in final_numerical_cols and m_col not in final_categorical_cols:
                if pd.api.types.is_numeric_dtype(df_processed[m_col]):
                    final_numerical_cols.append(m_col)
                else:
                    final_categorical_cols.append(m_col)
                    df_processed[m_col] = df_processed[m_col].astype(str)

    print(f"[REDACTED_BY_SCRIPT]")
    print(f"[REDACTED_BY_SCRIPT]")

    numerical_pipeline = Pipeline([
        ('imputer_num', SimpleImputer(strategy='constant', fill_value=-1)),
        ('power_transform', PowerTransformer(method='yeo-johnson', standardize=False)),
        ('scaler_num', MinMaxScaler())
    ])
    
    transformers_list = []
    if final_numerical_cols:
        transformers_list.append(('num', numerical_pipeline, final_numerical_cols))
    
    if final_categorical_cols: # Only add categorical pipeline if there are categorical columns
        categorical_pipeline = Pipeline([
            ('imputer_cat', SimpleImputer(strategy='constant', fill_value='Missing_Category')),
            ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='if_binary'))
        ])
        transformers_list.append(('cat', categorical_pipeline, final_categorical_cols))

    if not transformers_list:
        print("[REDACTED_BY_SCRIPT]")
        return df_processed, []

    preprocessor = ColumnTransformer(
        transformers=transformers_list,
        remainder='passthrough',
        verbose_feature_names_out=False
    )
    preprocessor.set_output(transform="pandas")

    print("[REDACTED_BY_SCRIPT]")
    df_transformed = preprocessor.fit_transform(df_processed)
    
    ohe_feature_names = []
    if final_categorical_cols and 'cat' in preprocessor.named_transformers_:
        try:
            ohe_feature_names = preprocessor.named_transformers_['cat'].named_steps['one_hot_encoder'].get_feature_names_out(final_categorical_cols).tolist()
        except Exception as e:
            print(f"[REDACTED_BY_SCRIPT]")

    print(f"[REDACTED_BY_SCRIPT]")
    return df_transformed, ohe_feature_names

def get_low_variance_features_report(df, threshold):
    variances = df.var(ddof=0)
    low_variance_info = {feature: variances[feature] for feature in variances[variances < threshold].index}
    return low_variance_info

def get_collinearity_report(df, threshold):
    numeric_df = df.copy()
    for col in numeric_df.columns:
        try:
            numeric_df[col] = pd.to_numeric(numeric_df[col])
        except ValueError:
            print(f"[REDACTED_BY_SCRIPT]")
            # Decide on a strategy: fill with a numeric placeholder or drop for correlation analysis
            numeric_df[col] = 0 # Example: fill non-convertible with 0
            
    numeric_df = numeric_df.fillna(0) 
    
    corr_matrix = numeric_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    highly_correlated_pairs = []
    for column in upper.columns:
        correlated_features_in_col = upper.index[upper[column] > threshold].tolist()
        for feature in correlated_features_in_col:
            highly_correlated_pairs.append((feature, column, upper.loc[feature, column]))
    return highly_correlated_pairs

def get_ahah_metric_parts(feature_name):
    """[REDACTED_BY_SCRIPT]'ah4gp_rnk' or 'ah4pm10'."""
    match = re.match(r"[REDACTED_BY_SCRIPT]", feature_name)
    if match:
        base = match.group(1)
        suffix_type = match.group(2)
        if suffix_type == "_rnk":
            return base, "rank"
        elif suffix_type == "_pct":
            return base, "percentile"
        else: # No suffix, it's a base metric
            return base, "base"
    return feature_name, "other" # Not a standard AHAH metric name structure

def get_veg_metric_parts(feature_name):
    """[REDACTED_BY_SCRIPT]'NDVI_MEAN', 'EVI_STD'."""
    match = re.match(r"[REDACTED_BY_SCRIPT]", feature_name, re.IGNORECASE)
    if match:
        veg_index = match.group(1).upper() # NDVI or EVI
        stat_type = match.group(2).upper() # MEAN, MEDIAN, etc.
        return veg_index, stat_type
    return feature_name, "other"


def identify_features_to_remove_subset4(df, ohe_feature_names, low_variance_info, collinear_pairs, manual_drops):
    cols_to_remove = set(manual_drops) # Start with manually specified drops
    removal_reasons = {col: "Manual drop" for col in manual_drops}
    
    unresolved_correlated_pairs = []
    potentially_redundant_fis = {} # Store FIs that are correlated with base features

    # 1. Low Variance Features
    for feature, variance in low_variance_info.items():
        if feature in cols_to_remove: continue
        threshold_to_check = OHE_LOW_VARIANCE_THRESHOLD_FOR_REMOVAL if feature in ohe_feature_names else LOW_VARIANCE_THRESHOLD
        if variance < threshold_to_check:
            cols_to_remove.add(feature)
            removal_reasons[feature] = f"Low variance ({'OHE '[REDACTED_BY_SCRIPT]''[REDACTED_BY_SCRIPT]"

    # 2. Collinearity Removals - Rule Based
    processed_in_pair = set() # To avoid processing a pair twice if f1,f2 then f2,f1

    for f1_orig, f2_orig, corr_value in sorted(collinear_pairs, key=lambda x: x[2], reverse=True):
        f1, f2 = sorted((f1_orig, f2_orig)) # Process pair consistently
        if (f1,f2) in processed_in_pair: continue
        processed_in_pair.add((f1,f2))

        if f1 in cols_to_remove or f2 in cols_to_remove: continue

        # --- Rule: AHAH Metric vs. Rank vs. Percentile (Prefer Raw Metric) ---
        base1, type1 = get_ahah_metric_parts(f1)
        base2, type2 = get_ahah_metric_parts(f2)

        if base1 == base2 and type1 != type2 and type1 != "other" and type2 != "other":
            if type1 == "base": # f1 is base, f2 is rank/percentile
                cols_to_remove.add(f2)
                removal_reasons[f2] = f"[REDACTED_BY_SCRIPT]"
            elif type2 == "base": # f2 is base, f1 is rank/percentile
                cols_to_remove.add(f1)
                removal_reasons[f1] = f"[REDACTED_BY_SCRIPT]"
            elif type1 == "percentile" and type2 == "rank": # f1 is pct, f2 is rank
                cols_to_remove.add(f2) # Prefer percentile over rank
                removal_reasons[f2] = f"[REDACTED_BY_SCRIPT]"
            elif type2 == "percentile" and type1 == "rank": # f2 is pct, f1 is rank
                cols_to_remove.add(f1)
                removal_reasons[f1] = f"[REDACTED_BY_SCRIPT]"
            continue # Move to next pair once this rule is applied

        # --- Rule: AHAH Domain/Overall Scores vs. Individual Metrics (Prefer Individual) ---
        # Domain scores often end with 'h', 'g', 'e', 'r'. Overall is 'ahah'.
        # Example: f1='ah4h' (domain), f2='ah4gp' (individual)
        f1_is_domain_or_overall = (type1 == "base" and (f1.endswith('h') or f1.endswith('g') or f1.endswith('e') or f1.endswith('r') or f1.endswith('ahah'))) and len(f1) <= len("ah4ahah") + 2 # avoid matching long FI names
        f2_is_domain_or_overall = (type2 == "base" and (f2.endswith('h') or f2.endswith('g') or f2.endswith('e') or f2.endswith('r') or f2.endswith('ahah'))) and len(f2) <= len("ah4ahah") + 2

        if f1_is_domain_or_overall and not f2_is_domain_or_overall and type2 == "base": # f1 is domain/overall, f2 is individual base
            cols_to_remove.add(f1)
            removal_reasons[f1] = f"[REDACTED_BY_SCRIPT]"
            continue
        if f2_is_domain_or_overall and not f1_is_domain_or_overall and type1 == "base": # f2 is domain/overall, f1 is individual base
            cols_to_remove.add(f2)
            removal_reasons[f2] = f"[REDACTED_BY_SCRIPT]"
            continue
        # If both are domain/overall scores (e.g. ah4ahah vs ah4h) prefer the more granular one (shorter name here is a proxy)
        if f1_is_domain_or_overall and f2_is_domain_or_overall:
            if len(f1) > len(f2): # f1 is more "overall" (e.g. ah4ahah vs ah4h)
                cols_to_remove.add(f1)
                removal_reasons[f1] = f"[REDACTED_BY_SCRIPT]"
            else: # f2 is more "overall"
                cols_to_remove.add(f2)
                removal_reasons[f2] = f"[REDACTED_BY_SCRIPT]"
            continue

        # --- Rule: NDVI vs. EVI (Prefer NDVI) ---
        veg_index1, stat1 = get_veg_metric_parts(f1)
        veg_index2, stat2 = get_veg_metric_parts(f2)

        if stat1 == stat2 and stat1 != "other": # Both are same type of stat (MEAN, MEDIAN etc.)
            if veg_index1 == "EVI" and veg_index2 == "NDVI":
                cols_to_remove.add(f1) # Drop EVI
                removal_reasons[f1] = f"[REDACTED_BY_SCRIPT]"
                continue
            elif veg_index2 == "EVI" and veg_index1 == "NDVI":
                cols_to_remove.add(f2) # Drop EVI
                removal_reasons[f2] = f"[REDACTED_BY_SCRIPT]"
                continue
        
        # --- Rule: Feature Interactions (FIs) vs. Derivatives ---
        # Heuristic: if FI (often longer name, contains 'x' or 'div' or specific prefixes)
        # is correlated with a simpler base feature that might be one of its components.
        # This is hard to generalize perfectly. We'll flag for review.
        f1_is_fi_heuristic = ("_x_" in f1 or "_div_" in f1 or f1.startswith("FI_") or "RankDiff" in f1 or "Ratio" in f1 or "ScaledBy" in f1 or "Exposure" in f1)
        f2_is_fi_heuristic = ("_x_" in f2 or "_div_" in f2 or f2.startswith("FI_") or "RankDiff" in f2 or "Ratio" in f2 or "ScaledBy" in f2 or "Exposure" in f2)
        
        # Check if one is an FI and the other is a plausible base component
        # A simple check: if the base feature name is part of the FI name
        if f1_is_fi_heuristic and not f2_is_fi_heuristic and (f2.lower() in f1.lower() or base2.lower() in f1.lower()):
            if f1 not in potentially_redundant_fis: potentially_redundant_fis[f1] = []
            potentially_redundant_fis[f1].append((f2, corr_value, "[REDACTED_BY_SCRIPT]"))
            # Don't automatically drop yet, flag for review
        elif f2_is_fi_heuristic and not f1_is_fi_heuristic and (f1.lower() in f2.lower() or base1.lower() in f2.lower()):
            if f2 not in potentially_redundant_fis: potentially_redundant_fis[f2] = []
            potentially_redundant_fis[f2].append((f1, corr_value, "[REDACTED_BY_SCRIPT]"))
            # Don't automatically drop yet
            
        # If no specific rule applied, add to unresolved for manual review
        # Only add if neither has been handled by a rule that added to cols_to_remove
        # (The continue statements above handle this implicitly for ruled pairs)
        # We need to make sure we don't add pairs where one was just dropped.
        # This check is tricky because cols_to_remove is populated *within* this loop.
        # For now, let's add all pairs not caught by a specific rule.
        # We will filter this list *after* the loop based on final cols_to_remove.
        unresolved_correlated_pairs.append((f1_orig, f2_orig, corr_value))


    # Filter unresolved_correlated_pairs: only show pairs where BOTH features are still in the dataset
    # This happens AFTER all rules (including the FI heuristic loop which might also add to cols_to_remove)
    # have had a chance to populate 'cols_to_remove'.
    
    truly_final_unresolved_for_manual_review = []
    # unresolved_correlated_pairs was populated with (f1_orig, f2_orig, corr_value)
    for u_f1, u_f2, u_corr in unresolved_correlated_pairs: 
        if u_f1 not in cols_to_remove and u_f2 not in cols_to_remove:
            truly_final_unresolved_for_manual_review.append((u_f1, u_f2, u_corr))

    print("[REDACTED_BY_SCRIPT]")
    print(f"[REDACTED_BY_SCRIPT]")
    print("[REDACTED_BY_SCRIPT]")
    if truly_final_unresolved_for_manual_review:
        # Sort for consistent review
        for f1_p, f2_p, corr_p in sorted(list(set(truly_final_unresolved_for_manual_review)), key=lambda x: (x[0], x[1])):
            print(f"[REDACTED_BY_SCRIPT]")
    else:
        print("[REDACTED_BY_SCRIPT]")
            
    # Ensure the function returns this correctly filtered list
    return list(cols_to_remove), removal_reasons, truly_final_unresolved_for_manual_review


    # Decision for FIs based on user preference: if FI is highly correlated with a base component, drop FI.
    # This is applied AFTER initial rules, to what's left.
    for fi_cand, base_corrs in potentially_redundant_fis.items():
        if fi_cand in cols_to_remove: continue
        for base_feat, corr_val, reason in base_corrs:
            if base_feat in cols_to_remove: continue # Base feature was already removed for other reasons
            # If the FI is still present and highly correlated with a base feature that is also still present:
            if corr_val > HIGH_CORRELATION_THRESHOLD : # Double check threshold
                cols_to_remove.add(fi_cand)
                removal_reasons[fi_cand] = f"[REDACTED_BY_SCRIPT]"
                break # Remove FI once due to one strong base correlation

    print("[REDACTED_BY_SCRIPT]")
    print("[REDACTED_BY_SCRIPT]'t add much new info, consider adding the FI to MANUAL_DROPS_S4.")
    if potentially_redundant_fis:
        for fi, details in potentially_redundant_fis.items():
            if fi not in cols_to_remove: # Only show if not already auto-removed
                print(f"  FI: {fi}")
                for base_feature, corr, reason_text in details:
                    if base_feature not in cols_to_remove: # Only show if base also not auto-removed
                        print(f"[REDACTED_BY_SCRIPT]")
    else:
        print("[REDACTED_BY_SCRIPT]")

    print("[REDACTED_BY_SCRIPT]")
    print(f"[REDACTED_BY_SCRIPT]")
    print("[REDACTED_BY_SCRIPT]")
    if final_unresolved_pairs:
        # Sort for consistent review
        for f1, f2, corr in sorted(list(set(final_unresolved_pairs)), key=lambda x: (x[0], x[1])):
             # Check again if they ended up in cols_to_remove by the FI redundancy rule applied after the loop
            if f1 not in cols_to_remove and f2 not in cols_to_remove:
                print(f"[REDACTED_BY_SCRIPT]")
    else:
        print("[REDACTED_BY_SCRIPT]")
            
    return list(cols_to_remove), removal_reasons, final_unresolved_pairs


# --- Main Execution ---
if __name__ == '__main__':
    subset4_df_full = load_data(FILE_PATH_SUBSET4)

    if subset4_df_full is not None:
        if len(subset4_df_full) > SAMPLE_SIZE:
            print(f"[REDACTED_BY_SCRIPT]")
            subset4_df_to_process = subset4_df_full.sample(n=SAMPLE_SIZE, random_state=42).copy()
        else:
            subset4_df_to_process = subset4_df_full.copy()
        
        df_processed_s4, ohe_cols_s4 = preprocess_data_s4(
            subset4_df_to_process, 
            COLS_TO_DROP_SUBSET4, 
            INITIAL_CATEGORICAL_COLS_SUBSET4
        )
        
        print("[REDACTED_BY_SCRIPT]")
        df_processed_s4.fillna(-1, inplace=True) # Safeguard after processing

        low_var_info_report_s4 = get_low_variance_features_report(df_processed_s4, LOW_VARIANCE_THRESHOLD)
        # For OHE specific threshold, if any OHE features exist:
        low_var_ohe_info_report_s4 = get_low_variance_features_report(df_processed_s4, OHE_LOW_VARIANCE_THRESHOLD_FOR_REMOVAL)

        collinear_pairs_report_s4 = get_collinearity_report(df_processed_s4, HIGH_CORRELATION_THRESHOLD)

        print(f"[REDACTED_BY_SCRIPT]")
        for f, v in low_var_info_report_s4.items():
            if f not in ohe_cols_s4 and v < LOW_VARIANCE_THRESHOLD:
                 print(f"  - {f}: {v:.6f}")
        
        if ohe_cols_s4:
            print(f"[REDACTED_BY_SCRIPT]")
            for f, v in low_var_ohe_info_report_s4.items(): # Use the OHE specific threshold report
                if f in ohe_cols_s4 and v < OHE_LOW_VARIANCE_THRESHOLD_FOR_REMOVAL:
                     print(f"  - {f}: {v:.6f}")

        print(f"[REDACTED_BY_SCRIPT]")
        # for f1, f2, corr in collinear_pairs_report_s4[:5]: # Print first 5
        #     print(f"[REDACTED_BY_SCRIPT]")


        print("[REDACTED_BY_SCRIPT]")
        features_to_drop_final_s4, reasons_s4, unresolved_pairs_for_review = identify_features_to_remove_subset4(
            df_processed_s4,
            ohe_cols_s4,
            low_var_info_report_s4, # Main low var report
            collinear_pairs_report_s4,
            MANUAL_DROPS_S4
        )
        
        # Ensure drops are from existing columns
        features_to_drop_final_s4_existing = [col for col in features_to_drop_final_s4 if col in df_processed_s4.columns]
        
        df_final_sample_s4 = df_processed_s4.copy() 
        if features_to_drop_final_s4_existing:
            print(f"[REDACTED_BY_SCRIPT]")
            sorted_features_to_drop = sorted(list(set(features_to_drop_final_s4_existing))) 
            for col in sorted_features_to_drop:
                 print(f"[REDACTED_BY_SCRIPT]'N/A')})")
            
            df_final_sample_s4.drop(columns=sorted_features_to_drop, inplace=True, errors='ignore')
            print(f"[REDACTED_BY_SCRIPT]")
            print(f"[REDACTED_BY_SCRIPT]")
        else:
            print("[REDACTED_BY_SCRIPT]")
            
        print("[REDACTED_BY_SCRIPT]")
        print(f"[REDACTED_BY_SCRIPT]")
        # print(df_final_sample_s4.head())

        output_filename_s4 = "[REDACTED_BY_SCRIPT]"
        try:
            df_final_sample_s4.to_parquet(output_filename_s4, index=False)
            print(f"[REDACTED_BY_SCRIPT]")
            print(f"[REDACTED_BY_SCRIPT]")
            print(f"Review the 'Unresolved Highly Correlated Pairs' and 'Potentially Redundant FIs'[REDACTED_BY_SCRIPT]")
            print(f"[REDACTED_BY_SCRIPT]")
        except Exception as e:
            print(f"[REDACTED_BY_SCRIPT]")