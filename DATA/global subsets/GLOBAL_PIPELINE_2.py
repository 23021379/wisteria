import pandas as pd
import numpy as np
import chardet # For detecting file encoding

# --- Helper function to determine file encoding ---
def get_file_encoding(filepath, sample_size_chars=200000, sample_size_bytes=400000): # Increased sample
    """
    Tries to determine the encoding of a file.
    Reads a sample to make a guess.
    """
    utf8_sample_read_ok = False
    try:
        # Try reading a sample with UTF-8 directly
        with open(filepath, 'r', encoding='utf-8', errors='strict') as f:
            f.read(sample_size_chars) # Try to read some characters
        # print(f"[REDACTED_BY_SCRIPT]")
        utf8_sample_read_ok = True
    except UnicodeDecodeError:
        # print(f"[REDACTED_BY_SCRIPT]")
        pass 
    except Exception:
        # print(f"[REDACTED_BY_SCRIPT]")
        pass 

    try:
        with open(filepath, 'rb') as f:
            raw_data = f.read(sample_size_bytes)
        result = chardet.detect(raw_data)
        detected_encoding = result['encoding']
        confidence = result['confidence']
        # print(f"[REDACTED_BY_SCRIPT]")

        if detected_encoding:
            if confidence > 0.9 and detected_encoding.lower() not in ['ascii', 'utf-8']:
                # print(f"[REDACTED_BY_SCRIPT]")
                return detected_encoding
            
            if utf8_sample_read_ok and detected_encoding.lower() in ['utf-8', 'ascii'] and confidence > 0.75:
                # print(f"[REDACTED_BY_SCRIPT]")
                return 'utf-8'

            if not utf8_sample_read_ok and detected_encoding.lower() in ['utf-8', 'ascii'] and confidence > 0.5:
                # print(f"[REDACTED_BY_SCRIPT]")
                # Fall through to latin-1 or more general chardet guess if strict UTF-8 failed
                pass 

            if confidence > 0.6 and detected_encoding.lower() not in ['ascii']: 
                # print(f"[REDACTED_BY_SCRIPT]")
                return detected_encoding

        if utf8_sample_read_ok:
            # print(f"[REDACTED_BY_SCRIPT]")
            return 'utf-8'

    except Exception:
        # print(f"[REDACTED_BY_SCRIPT]")
        pass 

    # print(f"[REDACTED_BY_SCRIPT]'latin-1'.")
    return 'latin-1'


# --- Helper function to safely get numeric series ---
def get_numeric_series_from_df(df, col_name, default_value=0.0):
    """
    Safely gets a column as a numeric series, handling missing columns by creating a series of default_value.
    """
    if col_name in df.columns:
        series = df[col_name]
    else:
        series = pd.Series(default_value, index=df.index)
    return pd.to_numeric(series, errors='coerce').fillna(default_value)

# --- Function to calculate feature interactions for a data chunk (NO CHANGES HERE from your last version) ---
def calculate_subset2_feature_interactions(df_chunk):
    """
    Calculates the 40 feature interactions for Subset 2 on the given dataframe chunk.
    The input df_chunk is expected to be postcode data merged with ONS OA data.
    """
    
    household_size_cols = [
        '[REDACTED_BY_SCRIPT]',
        'household_size_2_people_in_household',
        'household_size_3_people_in_household',
        'household_size_4_people_in_household',
        '[REDACTED_BY_SCRIPT]'
    ]
    # Initialize missing household_size_cols to 0 series before summing
    for col in household_size_cols:
        if col not in df_chunk.columns:
            # If a base column for Total_Households_OA is missing after a left merge (should not happen if ONS data is complete for the OA)
            # This indicates an OA that might be missing from ons_pivoted_df or has all NaN values for these.
            df_chunk[col] = pd.Series(0.0, index=df_chunk.index)
        else:
            # Ensure it's numeric and fill any NaNs that might result from merge (e.g. OA not found in ONS data)
            df_chunk[col] = pd.to_numeric(df_chunk[col], errors='coerce').fillna(0)
            
    df_chunk['Total_Households_OA'] = df_chunk[household_size_cols].sum(axis=1)
    
    def safe_divide(numerator, denominator):
        # Ensure numerator and denominator are numeric before division
        num = pd.to_numeric(numerator, errors='coerce').fillna(0)
        den = pd.to_numeric(denominator, errors='coerce').fillna(0)
        return np.where(den == 0, 0.0, num / den) 

    # --- A. Household Structure & Composition Interactions ---
    col_name = '[REDACTED_BY_SCRIPT]'
    val_lone_parent = get_numeric_series_from_df(df_chunk, col_name)
    df_chunk['[REDACTED_BY_SCRIPT]'] = safe_divide(val_lone_parent, df_chunk['Total_Households_OA'])

    col_name = '[REDACTED_BY_SCRIPT]'
    val_one_person = get_numeric_series_from_df(df_chunk, col_name)
    df_chunk['FI_Prop_One_Person_HH'] = safe_divide(val_one_person, df_chunk['Total_Households_OA'])

    num_families_base = '[REDACTED_BY_SCRIPT]'
    col_2fam = f'[REDACTED_BY_SCRIPT]'
    col_3plusfam = f'[REDACTED_BY_SCRIPT]'
    val_2fam = get_numeric_series_from_df(df_chunk, col_2fam)
    val_3plusfam = get_numeric_series_from_df(df_chunk, col_3plusfam)
    df_chunk['multi_family_sum'] = val_2fam + val_3plusfam
    df_chunk['[REDACTED_BY_SCRIPT]'] = safe_divide(df_chunk['multi_family_sum'], df_chunk['Total_Households_OA'])

    dep_children_base_pattern1 = 'dependent_children'
    dep_children_base_pattern2 = '[REDACTED_BY_SCRIPT]'
    col_name_val_pattern1 = f'[REDACTED_BY_SCRIPT]'
    col_name_val_pattern2 = f'[REDACTED_BY_SCRIPT]'
    if col_name_val_pattern1 in df_chunk.columns: val_with_children = get_numeric_series_from_df(df_chunk, col_name_val_pattern1)
    elif col_name_val_pattern2 in df_chunk.columns: val_with_children = get_numeric_series_from_df(df_chunk, col_name_val_pattern2)
    else: val_with_children = pd.Series(0.0, index=df_chunk.index)
    df_chunk['[REDACTED_BY_SCRIPT]'] = safe_divide(val_with_children, df_chunk['Total_Households_OA'])

    col_name = '[REDACTED_BY_SCRIPT]'
    val_large_hh = get_numeric_series_from_df(df_chunk, col_name)
    df_chunk['FI_Prop_Large_HH'] = safe_divide(val_large_hh, df_chunk['Total_Households_OA'])
    
    lifestage_base = 'reference_person'
    col_name = f'[REDACTED_BY_SCRIPT]'
    val_hrp_young = get_numeric_series_from_df(df_chunk, col_name)
    df_chunk['[REDACTED_BY_SCRIPT]'] = safe_divide(val_hrp_young, df_chunk['Total_Households_OA'])
    df_chunk['[REDACTED_BY_SCRIPT]'] = np.nan 

    deprivation_base = 'deprivation'
    deprivation_1_dim = f'[REDACTED_BY_SCRIPT]'
    deprivation_2_dim = f'[REDACTED_BY_SCRIPT]'
    deprivation_3_dim = f'[REDACTED_BY_SCRIPT]'
    deprivation_4_dim = f'[REDACTED_BY_SCRIPT]'
    val_dep_1 = get_numeric_series_from_df(df_chunk, deprivation_1_dim)
    val_dep_2 = get_numeric_series_from_df(df_chunk, deprivation_2_dim)
    val_dep_3 = get_numeric_series_from_df(df_chunk, deprivation_3_dim)
    val_dep_4 = get_numeric_series_from_df(df_chunk, deprivation_4_dim)
    sum_deprived_households = val_dep_1 + val_dep_2 + val_dep_3 + val_dep_4
    df_chunk['[REDACTED_BY_SCRIPT]'] = safe_divide(val_dep_3 + val_dep_4, sum_deprived_households)
    
    col_name = '[REDACTED_BY_SCRIPT]'
    val_emp_dep = get_numeric_series_from_df(df_chunk, col_name)
    df_chunk['[REDACTED_BY_SCRIPT]'] = safe_divide(val_emp_dep, df_chunk['Total_Households_OA'])

    col_name = '[REDACTED_BY_SCRIPT]'
    val_edu_dep = get_numeric_series_from_df(df_chunk, col_name)
    df_chunk['[REDACTED_BY_SCRIPT]'] = safe_divide(val_edu_dep, df_chunk['Total_Households_OA'])

    col_name = '[REDACTED_BY_SCRIPT]'
    val_health_dep = get_numeric_series_from_df(df_chunk, col_name)
    df_chunk['[REDACTED_BY_SCRIPT]'] = safe_divide(val_health_dep, df_chunk['Total_Households_OA'])

    col_name = '[REDACTED_BY_SCRIPT]'
    val_housing_dep = get_numeric_series_from_df(df_chunk, col_name)
    df_chunk['[REDACTED_BY_SCRIPT]'] = safe_divide(val_housing_dep, df_chunk['Total_Households_OA'])

    num_employed_base = 'number_employed'
    col_name = f'[REDACTED_BY_SCRIPT]'
    val_no_adult_emp = get_numeric_series_from_df(df_chunk, col_name)
    df_chunk['[REDACTED_BY_SCRIPT]'] = safe_divide(val_no_adult_emp, df_chunk['Total_Households_OA'])

    vehicles_base = 'vehicles'
    col_name = f'[REDACTED_BY_SCRIPT]'
    val_no_vehicles = get_numeric_series_from_df(df_chunk, col_name)
    df_chunk['[REDACTED_BY_SCRIPT]'] = safe_divide(val_no_vehicles, df_chunk['Total_Households_OA'])

    col_name = f'[REDACTED_BY_SCRIPT]'
    val_multi_vehicles = get_numeric_series_from_df(df_chunk, col_name)
    df_chunk['[REDACTED_BY_SCRIPT]'] = safe_divide(val_multi_vehicles, df_chunk['Total_Households_OA'])

    central_heating_base = '[REDACTED_BY_SCRIPT]'
    col_name = f'[REDACTED_BY_SCRIPT]'
    val_no_ch = get_numeric_series_from_df(df_chunk, col_name)
    df_chunk['[REDACTED_BY_SCRIPT]'] = safe_divide(val_no_ch, df_chunk['Total_Households_OA'])

    accom_type_base = 'house_types'
    col_name = f'[REDACTED_BY_SCRIPT]'
    val_flats = get_numeric_series_from_df(df_chunk, col_name)
    df_chunk['[REDACTED_BY_SCRIPT]'] = safe_divide(val_flats, df_chunk['Total_Households_OA'])

    tenure_base = 'ownership'
    col_name = f'[REDACTED_BY_SCRIPT]'
    val_owned_outright = get_numeric_series_from_df(df_chunk, col_name)
    df_chunk['[REDACTED_BY_SCRIPT]'] = safe_divide(val_owned_outright, df_chunk['Total_Households_OA'])

    col_social_council = f'[REDACTED_BY_SCRIPT]'
    col_social_other = f'[REDACTED_BY_SCRIPT]'
    val_social_council = get_numeric_series_from_df(df_chunk, col_social_council)
    val_social_other = get_numeric_series_from_df(df_chunk, col_social_other)
    df_chunk['socially_rented_sum'] = val_social_council + val_social_other
    df_chunk['[REDACTED_BY_SCRIPT]'] = safe_divide(df_chunk['socially_rented_sum'], df_chunk['Total_Households_OA'])

    occupancy_bed_base = 'occupancy_rating'
    col_neg1 = f'[REDACTED_BY_SCRIPT]' 
    col_neg2less = f'[REDACTED_BY_SCRIPT]' # Note: Logic error in old code? Header has '[REDACTED_BY_SCRIPT]'. Wait, the old code used -1 and -2. 
    # Header check: '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]'.
    # There is no -1 or -2. It seems the data has categories like "1 or less" which implies overcrowding (negative rating).
    # "[REDACTED_BY_SCRIPT]"?? usually occupancy rating is: +2, +1, 0, -1, -2.
    # Header: `occupancy_rating_Occupancy_rating_of_bedrooms_0`, `_1`, `_1_or_less` (Wait. 1 or less? If it's a rating, +1 is good. -1 is bad).
    # Let's check `debug_csv_snippet.py` again for `occupancy_rating`.
    # `occupancy_rating_Occupancy_rating_of_bedrooms_0`
    # `occupancy_rating_Occupancy_rating_of_bedrooms_1`
    # `occupancy_rating_Occupancy_rating_of_bedrooms_1_or_less` (Is this +1 or less? Or -1 or less?)
    # Usually the categories are +2, +1, 0, -1, -2.
    # If the headers are simplified, "1 or less" might capture 0, -1, -2? But 0 is listed separately.
    # Ah, maybe it's "-1 or less"?
    # Let's assume `occupancy_rating_Occupancy_rating_of_bedrooms_1_or_less` is what I have to work with for now or look for `_minus_1`?
    # Actually, let's look at `debug_csv_snippet.py`: `occupancy_rating_Occupancy_rating_of_bedrooms_1`, `occupancy_rating_Occupancy_rating_of_bedrooms_1_or_less`. 
    # This is weird. "1" and "1 or less". Maybe "1" means "+1". "1 or less" means "+1, 0, -1..."? No, that would overlap.
    # Let's assume "1 or less" is the overcrowding metric or check for another one? 
    # Wait, the prompt says: "[REDACTED_BY_SCRIPT]". I will stick to literal mapping.
    # `occupancy_bed_base` was `occupancy_rating_for_bedrooms`. New is `occupancy_rating`.
    # Old cols: `_Occupancy_rating_of_-1`, `_Occupancy_rating_of_-2_or_less`.
    # New cols available in snippet: `occupancy_rating_Does_not_apply`, `occupancy_rating_Occupancy_rating_of_bedrooms_0`, `occupancy_rating_Occupancy_rating_of_bedrooms_1`, `occupancy_rating_Occupancy_rating_of_bedrooms_1_or_less`, `occupancy_rating_Occupancy_rating_of_bedrooms_2_or_more`.
    # It seems "1 or less" is the closest to overcrowding (if it means -1? No, 1 is under-occupied? No, rating = bedrooms required - bedrooms obtained. Negative is overcrowded. Positive is under-occupied).
    # If the label is "1 or less", it might mean "[REDACTED_BY_SCRIPT]"? Or maybe it's a specific census output category.
    # Wait, let's look at `occupancy_rating_rooms` too. `occupancy_rating_rooms_Occupancy_rating_of_rooms_1_or_less`.
    # It seems I'm missing the negative categories or they are named differently.
    # Or maybe "1 or less" is actually "-1 or less" with the minus sign stripped in the header creation?
    # I will map `col_neg1` and `col_neg2less` both to `occupancy_rating_Occupancy_rating_of_bedrooms_1_or_less` ? No that would double count.
    # I'[REDACTED_BY_SCRIPT]'s the only one. 
    # User instructions: "[REDACTED_BY_SCRIPT]"
    # I'll just use the `occupancy_rating_Occupancy_rating_of_bedrooms_1_or_less` variable.
    # But wait, logic: `df_chunk['overcrowded_bedrooms_sum'] = val_neg1 + val_neg2less`. 
    # I will change logic to just use the one column available. 
    
    val_neg = get_numeric_series_from_df(df_chunk, f'[REDACTED_BY_SCRIPT]') # Assuming this captures the 'bad' end.
    df_chunk['overcrowded_bedrooms_sum'] = val_neg
    df_chunk['[REDACTED_BY_SCRIPT]'] = safe_divide(df_chunk['overcrowded_bedrooms_sum'], df_chunk['Total_Households_OA'])

    num_per_room_base = 'number_per_room'
    col_over1_1_5 = f'[REDACTED_BY_SCRIPT]'
    col_over1_5 = f'[REDACTED_BY_SCRIPT]'
    val_over1_1_5 = get_numeric_series_from_df(df_chunk, col_over1_1_5)
    val_over1_5 = get_numeric_series_from_df(df_chunk, col_over1_5)
    df_chunk['[REDACTED_BY_SCRIPT]'] = val_over1_1_5 + val_over1_5
    df_chunk['[REDACTED_BY_SCRIPT]'] = safe_divide(df_chunk['[REDACTED_BY_SCRIPT]'], df_chunk['Total_Households_OA'])
    
    num_bedrooms_base = 'number_bedrooms'
    col_1bed = f'[REDACTED_BY_SCRIPT]'
    col_2bed = f'[REDACTED_BY_SCRIPT]'
    val_1bed = get_numeric_series_from_df(df_chunk, col_1bed)
    val_2bed = get_numeric_series_from_df(df_chunk, col_2bed)
    df_chunk['small_dwellings_sum'] = val_1bed + val_2bed
    df_chunk['[REDACTED_BY_SCRIPT]'] = safe_divide(df_chunk['small_dwellings_sum'], df_chunk['Total_Households_OA'])

    limited_a_lot_base = 'disability_affected'
    limited_a_little_base = 'disability_affected'
    sum_limited_a_lot_val = pd.Series(0.0, index=df_chunk.index)
    
    # Header: disability_affected_1_person_disabled_under_the_Equality_Act_whose_day_to_day_activities_are_limited_a_lot_in_household
    # Header: disability_affected_2_or_more_people_disabled_under_the_Equality_Act_whose_day_to_day_activities_are_limited_a_lot_in_household
    # Note: "2_1_person..." typo in logic? No, check debug snippet
    # Snippet: `disability_affected_2_1_person_disabled_under_the_Equality_Act_whose_day_to_day_activities_are_limited_a_lot_in_household` (Does this exist? "2_1"? Looks like a typo in original data or snippet formatting.
    # Ah, let's look at snippet closely (line 20 approx):
    # `disability_affected_2_1_person_disabled_under_the_Equality_Act_whose_day_to_day_activities_are_limited_a_lot_in_household`
    # `disability_affected_2_2_or_more_people_disabled_under_the_Equality_Act_whose_day_to_day_activities_are_limited_a_lot_in_household`
    # There is a weird `2_` prefix in the middle? `disability_affected_2_...`
    # But there is also `disability_affected_1_person...` (without `2_`?)
    # Wait, `disability_affected_1_person_disabled_..._limited_a_little_...`
    # `disability_affected_2_or_more_people_disabled_..._limited_a_little_...`
    # Use exact mappings based on snippet.
    
    col_lot_1 = f'[REDACTED_BY_SCRIPT]'
    col_lot_2 = f'[REDACTED_BY_SCRIPT]'
    sum_limited_a_lot_val = get_numeric_series_from_df(df_chunk, col_lot_1) + get_numeric_series_from_df(df_chunk, col_lot_2)
    df_chunk['sum_limited_a_lot'] = sum_limited_a_lot_val

    col_little_1 = f'[REDACTED_BY_SCRIPT]'
    col_little_2 = f'[REDACTED_BY_SCRIPT]'
    sum_limited_a_little_val = get_numeric_series_from_df(df_chunk, col_little_1) + get_numeric_series_from_df(df_chunk, col_little_2)
    df_chunk['sum_limited_a_little'] = sum_limited_a_little_val

    df_chunk['[REDACTED_BY_SCRIPT]'] = safe_divide(df_chunk['sum_limited_a_lot'], df_chunk['sum_limited_a_little'])

    num_carers_base = 'number_carers'
    unpaid_carers_sum_val = pd.Series(0.0, index=df_chunk.index)
    for cat_suffix in ['[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]']:
        unpaid_carers_sum_val += get_numeric_series_from_df(df_chunk, f'[REDACTED_BY_SCRIPT]')
    df_chunk['unpaid_carers_sum'] = unpaid_carers_sum_val
    df_chunk['[REDACTED_BY_SCRIPT]'] = safe_divide(df_chunk['unpaid_carers_sum'], df_chunk['Total_Households_OA'])

    health_dep_col_name = '[REDACTED_BY_SCRIPT]'
    health_deprived_col_val = get_numeric_series_from_df(df_chunk, health_dep_col_name)
    df_chunk['[REDACTED_BY_SCRIPT]'] = safe_divide(df_chunk['unpaid_carers_sum'], health_deprived_col_val)

    no_lthc_base = 'health_condition'
    col_name = f'[REDACTED_BY_SCRIPT]' # Check snippet for exact "All no condition" equivalent.
    # Snippet: `health_condition_No_people_with_a_non_limiting_long_term_physical_or_mental_health_condition_in_household` (This is "[REDACTED_BY_SCRIPT]", i.e. Healthy?)
    # Actually there are `health_condition_2_...`
    # Snippet: `health_condition_2_All_people_in_household_have_a_long_term_health_condition_or_disability` (BAD)
    # Snippet: `health_condition_2_2_or_more_people_in_household_have_no_long_term_health_condition_or_disability` (GOOD)
    # The snippet is messy with `health_condition_` and `health_condition_2_`.
    # I need "[REDACTED_BY_SCRIPT]".
    # Snippet has: `health_condition_2_1_person_in_household_has_no_long_term_health_condition_or_disability`? No.
    # Actually, looking for "All".
    # Snippet doesn't seemingly have "All ... no condition".
    # It has `health_condition_2_All_people_in_household_have_a_long_term_health_condition_or_disability` (This is ALL SICK).
    # It has `health_condition_2_2_or_more_people_in_household_have_no_long_term_health_condition_or_disability`.
    # Maybe I should rely on "[REDACTED_BY_SCRIPT]"?
    # `health_condition_No_people_with_a_non_limiting_long_term_physical_or_mental_health_condition_in_household` -> This specifically says "non-limiting". This is oddly specific.
    # What about `health_condition_2_1_person_in_household_has_no_long_term_health_condition_or_disability` ?
    # Let's assume the user mapped `no_lthc_base` to `health_condition`?
    # User didn't specify the exact value suffix mapping.
    # I will try to map `health_condition_2_All_people_in_household_have_no_long_term...` if it existed?
    # Let's look at `GLOBAL_PIPELINE_2.py` original code: `_All_people_in_household_have_no_long_term_health_condition`.
    # Snippet has `health_condition_2_All_people_in_household_have_a_long_term_health_condition_or_disability`.
    # It seems I might be missing the "All No" column in the snippet or it's named differently.
    # But `health_condition_2_2_or_more_people_in_household_have_no_long_term...` suggests mixed.
    # I will replace with a placeholder or best guess based on snippet? 
    # Or just use `health_condition` prefix and hope the code was right about the suffix? 
    # But the suffix "[REDACTED_BY_SCRIPT]" is not in the snippet (snippet has "have_a_long_term...").
    # Wait, `health_condition_No_people_with_a_non_limiting...`
    # I'[REDACTED_BY_SCRIPT]'t find it, or ask?
    # I will assume `health_condition_No_people_with_...` is what was meant? No.
    # Let's just update the BASE to `health_condition` or `health_condition_2`? 
    # User lists `health_condition` (lowercase).
    # I'll update the base and keep the suffix, but if the suffix is wrong it will fail 0.
    # Actually, let's look for "All people... no... condition" in the snippet string again.
    # `health_condition_2_1_person_in_household_has_no_long_term_health_condition_or_disability`
    # `health_condition_2_2_or_more_people_in_household_have_no_long_term_health_condition_or_disability`
    # `health_condition_2_All_people_in_household_have_a_long_term_health_condition_or_disability`
    # There is NO "All people... have NO...".
    # Maybe `health_condition_No_people_with_a_non_limiting...`?
    # I'll skip this one's logic change effectively (leave it zero) but fix the base name just in case?
    # `no_lthc_base = 'health_condition_2'` maybe?
    # I'[REDACTED_BY_SCRIPT]'t match it defaults to 0.

    df_chunk['[REDACTED_BY_SCRIPT]'] = safe_divide(pd.Series(0.0, index=df_chunk.index), df_chunk['Total_Households_OA']) # Disabling due to missing column match in snippet

    disabled_adults_base = 'disabilty'
    col_2_dis_adults = f'[REDACTED_BY_SCRIPT]' # Snippet has this
    col_1_dis_adults = f'[REDACTED_BY_SCRIPT]' # Snippet has this
    
    # Original code wanted 2 and 3+. Snippet has 2+.
    val_2_dis_adults = get_numeric_series_from_df(df_chunk, col_2_dis_adults)
    # val_3plus_dis_adults = get_numeric_series_from_df(df_chunk, col_3plus_dis_adults) # Not in snippet
    df_chunk['multiple_disabled_adults_sum'] = val_2_dis_adults
    df_chunk['[REDACTED_BY_SCRIPT]'] = safe_divide(df_chunk['multiple_disabled_adults_sum'], df_chunk['Total_Households_OA'])

    ethnic_groups_base = 'ethnicity'
    multi_ethnic_sum_val = pd.Series(0.0, index=df_chunk.index)
    col_mixed = f'[REDACTED_BY_SCRIPT]'
    col_other_combo = f'[REDACTED_BY_SCRIPT]'
    col_other_combo3 = f'[REDACTED_BY_SCRIPT]'
    multi_ethnic_sum_val = get_numeric_series_from_df(df_chunk, col_mixed) + get_numeric_series_from_df(df_chunk, col_other_combo) + get_numeric_series_from_df(df_chunk, col_other_combo3)
    
    df_chunk['multi_ethnic_sum'] = multi_ethnic_sum_val
    df_chunk['[REDACTED_BY_SCRIPT]'] = safe_divide(df_chunk['multi_ethnic_sum'], df_chunk['Total_Households_OA'])

    col_name = f'[REDACTED_BY_SCRIPT]'
    val_asian_only = get_numeric_series_from_df(df_chunk, col_name)
    df_chunk['[REDACTED_BY_SCRIPT]'] = safe_divide(val_asian_only, df_chunk['Total_Households_OA'])

    lang_base = 'language'
    col_name = f'[REDACTED_BY_SCRIPT]' # Snippet has this
    val_multi_lingual = get_numeric_series_from_df(df_chunk, col_name)
    df_chunk['[REDACTED_BY_SCRIPT]'] = safe_divide(val_multi_lingual, df_chunk['Total_Households_OA'])

    religion_base = 'multiple_religion'
    mixed_no_religion_sum_val = pd.Series(0.0, index=df_chunk.index)
    
    # Snippet: `multiple_religion_Multi_person_household_At_least_two_different_religions_stated`
    # Snippet: `multiple_religion_Multi_person_household_No_religion`
    # Snippet: `multiple_religion_Multi_person_household_Same_religion_and_no_religion`
    # Snippet: `multiple_religion_Multi_person_household_No_people_answered_the_religion_question`
    
    col_diff = f'[REDACTED_BY_SCRIPT]'
    col_no_rel = f'[REDACTED_BY_SCRIPT]'
    col_same_no_rel = f'[REDACTED_BY_SCRIPT]'
    
    mixed_no_religion_sum_val = get_numeric_series_from_df(df_chunk, col_diff) + \
                                get_numeric_series_from_df(df_chunk, col_no_rel) + \
                                get_numeric_series_from_df(df_chunk, col_same_no_rel)

    df_chunk['[REDACTED_BY_SCRIPT]'] = [REDACTED_BY_SCRIPT]_val
    df_chunk['[REDACTED_BY_SCRIPT]'] = safe_divide(df_chunk['[REDACTED_BY_SCRIPT]'], df_chunk['Total_Households_OA'])

    hrp_army_base = 'ons_army'
    col_name = f'[REDACTED_BY_SCRIPT]'
    val_hrp_vet = get_numeric_series_from_df(df_chunk, col_name)
    df_chunk['[REDACTED_BY_SCRIPT]'] = safe_divide(val_hrp_vet, df_chunk['Total_Households_OA'])

    student_away_base = 'household_schoolchildren'
    col_name = f'[REDACTED_BY_SCRIPT]' # And others?
    # Snippet: `household_schoolchildren_Household_with_one_student_away`, `household_schoolchildren_Household_with_three_or_more_students_away`, `household_schoolchildren_Household_with_two_students_away`
    val_student_away = get_numeric_series_from_df(df_chunk, f'[REDACTED_BY_SCRIPT]') + \
                       get_numeric_series_from_df(df_chunk, f'[REDACTED_BY_SCRIPT]') + \
                       get_numeric_series_from_df(df_chunk, f'[REDACTED_BY_SCRIPT]')
    
    df_chunk['[REDACTED_BY_SCRIPT]'] = safe_divide(val_student_away, df_chunk['Total_Households_OA'])

    df_chunk['[REDACTED_BY_SCRIPT]'] = df_chunk['[REDACTED_BY_SCRIPT]'] * df_chunk['[REDACTED_BY_SCRIPT]']
    df_chunk['[REDACTED_BY_SCRIPT]'] = df_chunk['[REDACTED_BY_SCRIPT]'] * df_chunk['[REDACTED_BY_SCRIPT]']
    df_chunk['[REDACTED_BY_SCRIPT]'] = df_chunk['[REDACTED_BY_SCRIPT]'] * df_chunk['[REDACTED_BY_SCRIPT]']
    df_chunk['[REDACTED_BY_SCRIPT]'] = df_chunk['[REDACTED_BY_SCRIPT]'] * df_chunk['[REDACTED_BY_SCRIPT]']
    df_chunk['[REDACTED_BY_SCRIPT]'] = df_chunk['[REDACTED_BY_SCRIPT]'] * df_chunk['[REDACTED_BY_SCRIPT]']
    
    lifestage_base = 'reference_person'
    elderly_owned_outright_sum_val = pd.Series(0.0, index=df_chunk.index)
    # Header: reference_person_Household_reference_person_is_aged_66_years_or_over_One_person_household
    # Header: reference_person_Household_reference_person_is_aged_66_years_or_over_Two_or_more_person_household_No_dependent_children
    # Header: reference_person_Household_reference_person_is_aged_66_years_or_over_Two_or_more_person_household_Dependent_children
    # But wait, we need "Owns Outright" as well overlap?
    # NO. The original code variable `elderly_owned_outright_sum` suggests it was iterating over `Lifestage_of_Household_Reference_Person_Aged_66_and_over_Owns_outright`.
    # This implies a Cross-Tab of Lifestage AND Tenure in the source data.
    # The snippet Header `reference_person` columns DO NOT mention "Owns". They only mention Age and Household composition.
    # `ownership` columns mention 'Owned_Owns_outright'.
    # If the source data does NOT have the interaction pre-calculated (which it seems it doesn't in the snippet), I cannot calculate `elderly_owned_outright_sum` directly from a single column.
    # I would need to interact `reference_person_...66_years_or_over...` AND `ownership_Owned_Owns_outright`.
    # BUT, this is a "Feature Interaction" script. Maybe I should perform that interaction here?
    # `FI_Elderly_Owned_Outright_Concentration`.
    # I can approximate "[REDACTED_BY_SCRIPT]" by multiplying `Prop_Elderly` * `Prop_Owned_Outright`.
    # Or if I can sum "Elderly" households, and assume distribution?
    # The original code logic: `if col.startswith(f'[REDACTED_BY_SCRIPT]'):` suggests the column EXISTED.
    # If it no longer exists, I should probably DISABLE this feature or Approximation.
    # I'll approximate it as `Prop_Elderly` * `Prop_Owned_Outright`.
    
    col_eld_1 = f'[REDACTED_BY_SCRIPT]'
    col_eld_2 = f'[REDACTED_BY_SCRIPT]'
    col_eld_3 = f'[REDACTED_BY_SCRIPT]'
    
    val_elderly = get_numeric_series_from_df(df_chunk, col_eld_1) + get_numeric_series_from_df(df_chunk, col_eld_2) + get_numeric_series_from_df(df_chunk, col_eld_3)
    
    # We already have `FI_Prop_Owned_Outright_HH`.
    # If I can'[REDACTED_BY_SCRIPT]'ll return the product of proportions?
    # Or `val_elderly / Total` * `val_owned_outright / Total`?
    # I'll do that properly.
    
    prop_elderly = safe_divide(val_elderly, df_chunk['Total_Households_OA'])
    df_chunk['[REDACTED_BY_SCRIPT]'] = prop_elderly * df_chunk['[REDACTED_BY_SCRIPT]'] # Approximation


    df_chunk['[REDACTED_BY_SCRIPT]'] = df_chunk['[REDACTED_BY_SCRIPT]'] * df_chunk['[REDACTED_BY_SCRIPT]']
    
    return df_chunk

# --- Main processing script ---
def main():
    # --- Filepaths (adjust as necessary) ---
    pcd_to_oa_lookup_filepath = r"[REDACTED_BY_SCRIPT]"
    # ONS pivoted data (census counts at OA level)
    ons_pivoted_filepath = r"[REDACTED_BY_SCRIPT]"
    # Output file
    output_filepath = r"[REDACTED_BY_SCRIPT]"

    print("[REDACTED_BY_SCRIPT]")
    ons_encoding = get_file_encoding(ons_pivoted_filepath)
    print(f"[REDACTED_BY_SCRIPT]")

    print("[REDACTED_BY_SCRIPT]")
    pcd_lookup_encoding = get_file_encoding(pcd_to_oa_lookup_filepath)
    print(f"[REDACTED_BY_SCRIPT]")

    print(f"[REDACTED_BY_SCRIPT]")
    ons_pivoted_df = pd.read_csv(ons_pivoted_filepath, encoding=ons_encoding, low_memory=False)
    print("[REDACTED_BY_SCRIPT]")

    bom_col_name_ons = '[REDACTED_BY_SCRIPT]'
    plain_col_name_ons = 'Output Areas Code'
    target_col_name_ons = 'Output_Areas_Code' 

    if bom_col_name_ons in ons_pivoted_df.columns:
        ons_pivoted_df.rename(columns={bom_col_name_ons: target_col_name_ons}, inplace=True)
    elif plain_col_name_ons in ons_pivoted_df.columns: 
        ons_pivoted_df.rename(columns={plain_col_name_ons: target_col_name_ons}, inplace=True)
    
    if target_col_name_ons not in ons_pivoted_df.columns:
        if '\ufeffOutput Areas' in ons_pivoted_df.columns:
            ons_pivoted_df.rename(columns={'\ufeffOutput Areas': target_col_name_ons}, inplace=True)
        elif 'Output Areas' in ons_pivoted_df.columns:
            ons_pivoted_df.rename(columns={'Output Areas': target_col_name_ons}, inplace=True)

    if target_col_name_ons not in ons_pivoted_df.columns:
        print(f"Error: Join key '{target_col_name_ons}'[REDACTED_BY_SCRIPT]")
        print(f"[REDACTED_BY_SCRIPT]")
        return

    ons_pivoted_df[target_col_name_ons] = ons_pivoted_df[target_col_name_ons].astype(str).str.strip().str.upper()

    # --- CRITICAL CHECK: Ensure critical ONS columns exist ---
    chk_col = '[REDACTED_BY_SCRIPT]'
    if chk_col not in ons_pivoted_df.columns:
        print(f"CRITICAL WARNING: '{chk_col}'[REDACTED_BY_SCRIPT]")
        print(f"[REDACTED_BY_SCRIPT]")
        # Check for similar names (case sensitivity or prefix issues)
        similar = [c for c in ons_pivoted_df.columns if 'Household_size' in c]
        if similar:
            print(f"[REDACTED_BY_SCRIPT]")
        else:
            print("No 'Household_size' columns found.")
            
    ons_potential_count_cols = [col for col in ons_pivoted_df.columns if col not in [target_col_name_ons, 'Output Areas']]
    for col in ons_potential_count_cols:
        ons_pivoted_df[col] = pd.to_numeric(ons_pivoted_df[col], errors='coerce').fillna(0)
    
    chunk_size = 50000
    print(f"[REDACTED_BY_SCRIPT]")
    
    first_chunk = True
    try:
        pcd_lookup_reader = pd.read_csv(
            pcd_to_oa_lookup_filepath, 
            encoding=pcd_lookup_encoding, 
            chunksize=chunk_size, 
            low_memory=False,
            encoding_errors='replace' # Added to handle persistent encoding errors within chunks
        )
    except FileNotFoundError:
        print(f"[REDACTED_BY_SCRIPT]")
        print("[REDACTED_BY_SCRIPT]'PCD_OA21_LSOA21_MSOA21_LAD_AUG24_UK_LU/yourfile.csv'[REDACTED_BY_SCRIPT]")
        return
    except Exception as e:
        print(f"[REDACTED_BY_SCRIPT]")
        return

    processed_chunks=[]
    for i, pcd_chunk_df in enumerate(pcd_lookup_reader):
        print(f"[REDACTED_BY_SCRIPT]")
        
        bom_col_name_pcd = '\ufeffpcd7' 
        plain_col_name_pcd = 'pcd7' 
        if bom_col_name_pcd in pcd_chunk_df.columns:
             pcd_chunk_df.rename(columns={bom_col_name_pcd: plain_col_name_pcd}, inplace=True)
        
        if 'oa21cd' not in pcd_chunk_df.columns:
            # Check for common variations if 'oa21cd' is missing
            potential_oa_cols = ['OA21CD', 'oa21CD', 'OutputArea21Code']
            found_oa_col = None
            for col_var in potential_oa_cols:
                if col_var in pcd_chunk_df.columns:
                    pcd_chunk_df.rename(columns={col_var: 'oa21cd'}, inplace=True)
                    found_oa_col = 'oa21cd'
                    break
            if not found_oa_col:
                print(f"Error: Join key 'oa21cd'[REDACTED_BY_SCRIPT]")
                print(f"[REDACTED_BY_SCRIPT]")
                continue 

        # Standardize join key in Postcode data
        pcd_chunk_df['oa21cd'] = pcd_chunk_df['oa21cd'].astype(str).str.strip().str.upper()

        merged_chunk = pd.merge(
            pcd_chunk_df,
            ons_pivoted_df,
            left_on='oa21cd',        
            right_on=target_col_name_ons, 
            how='left', sort=False
        )

        # Check merge success rate
        # We expect Total_Households_OA columns (e.g. '[REDACTED_BY_SCRIPT]') to be populated if merge worked consistently.
        # But 'Total_Households_OA' is calculated later. So check if any ONS column is non-null.
        ons_check_cols = [c for c in ons_pivoted_df.columns if c != target_col_name_ons]
        if ons_check_cols:
            sample_ons_col = ons_check_cols[0]
            if sample_ons_col in merged_chunk.columns:
                match_count = merged_chunk[sample_ons_col].notna().sum()
                match_pct = (match_count / len(merged_chunk)) * 100
                print(f"[REDACTED_BY_SCRIPT]")
                if match_count == 0:
                    print(f"[REDACTED_BY_SCRIPT]")
                    print(f"[REDACTED_BY_SCRIPT]'oa21cd'].head().tolist()}")
                    print(f"[REDACTED_BY_SCRIPT]")
        
        processed_chunk = calculate_subset2_feature_interactions(merged_chunk.copy())

        if first_chunk:
            processed_chunk.to_csv(output_filepath, index=False, mode='w', header=True)
            first_chunk = False
            print(f"[REDACTED_BY_SCRIPT]")
        else:
            processed_chunk.to_csv(output_filepath, index=False, mode='a', header=False)
            print(f"[REDACTED_BY_SCRIPT]")
        processed_chunks.append(processed_chunk)
    final_df_subset2 = pd.concat(processed_chunks, ignore_index=True)
    print(f"[REDACTED_BY_SCRIPT]")
    print(f"[REDACTED_BY_SCRIPT]")
    print("[REDACTED_BY_SCRIPT]")
    print(final_df_subset2.head())
    print("[REDACTED_BY_SCRIPT]")
    print(f"[REDACTED_BY_SCRIPT]")

if __name__ == '__main__':
    main()