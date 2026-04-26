import ast
import os
import csv
import io
import re
from datetime import datetime

# --- Configuration ---
# List your input CSV filenames here
input_filenames = [
    '[REDACTED_BY_SCRIPT]',
    'mPFromRightmove.csv',
    '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]'
]

output_filenames = [f.replace('.csv', '[REDACTED_BY_SCRIPT]') for f in input_filenames] # New suffix

# --- Encoding Helper Functions ---
def encode_date(date_str):
    """Encodes 'DD Mon YYYY'[REDACTED_BY_SCRIPT]'','','') on failure."""
    if not date_str or not isinstance(date_str, str):
        return ('', '', '')
    try:
        dt_obj = datetime.strptime(date_str.strip(), '%d %b %Y')
        return (dt_obj.strftime('%d'), dt_obj.strftime('%m'), dt_obj.strftime('%Y'))
    except ValueError:
        return ('', '', '')

# Global property type encoder (used by Homipi)
def encode_property_type(type_str):
    """Encodes 'Flat' to 1, 'House'[REDACTED_BY_SCRIPT]"""
    type_str_lower = str(type_str).strip().lower()
    if type_str_lower == 'flat': return '1'
    elif type_str_lower == 'house': return '2'
    else: return '0'

# Global A-G rating encoder (used by Homipi, MP)
def encode_rating(rating_str):
    """[REDACTED_BY_SCRIPT]"""
    rating_str_upper = str(rating_str).strip().upper()
    if len(rating_str_upper) == 1 and 'A' <= rating_str_upper <= 'G': # Check length
        return str(ord(rating_str_upper) - ord('A') + 1)
    else: return '0'

# Chimnie specific risk encoder
def encode_chimnie_risk(risk_str):
    """[REDACTED_BY_SCRIPT]"""
    risk_str_upper = str(risk_str).strip().upper()
    if risk_str_upper == "VERY LOW": return '1'
    elif risk_str_upper == "LOW": return '2'
    elif risk_str_upper == "MED": return '3'
    elif risk_str_upper == "HIGH": return '4'
    else: return '0'

# --- MP Specific Encoding/Helper Functions ---
def encode_mp_risk_level(risk_str):
    """[REDACTED_BY_SCRIPT]"""
    risk_str_lower = str(risk_str).strip().lower()
    if risk_str_lower == "lower": return '1'
    elif risk_str_lower == "low-medium": return '2' # Corrected
    elif risk_str_lower == "medium": return '3'
    elif risk_str_lower == "medium-high": return '4'
    elif risk_str_lower == "high": return '5'
    else: return '0'

def encode_mp_property_type(type_str):
    """[REDACTED_BY_SCRIPT]"""
    type_str_lower = str(type_str).strip().lower()
    if type_str_lower == 'flat': return '1'
    elif type_str_lower == 'house': return '2'
    elif type_str_lower == 'semi-detached': return '3'
    elif type_str_lower == 'detached': return '4'
    elif type_str_lower == 'terraced': return '5'
    elif type_str_lower == 'end of terrace': return '6'
    elif type_str_lower == 'bungalow': return '7'
    else: return '0'

def split_mp_built_year(year_text_str):
    """Splits MP 'Built YYYY-YYYY' or 'Built YYYY' or 'Built before YYYY'[REDACTED_BY_SCRIPT]"""
    year_text_str = str(year_text_str).strip()
    match_range = re.search(r'[REDACTED_BY_SCRIPT]', year_text_str, re.IGNORECASE)
    if match_range: return (match_range.group(1), match_range.group(2))
    match_before = re.search(r'[REDACTED_BY_SCRIPT]', year_text_str, re.IGNORECASE)
    if match_before: return (match_before.group(1), '0')
    match_single = re.search(r'Built\s*(\d{4})', year_text_str, re.IGNORECASE)
    if match_single: return ('0', match_single.group(1))
    match_after = re.search(r'[REDACTED_BY_SCRIPT]', year_text_str, re.IGNORECASE)
    if match_after: return (match_after.group(1), '0')
    if year_text_str.lower() == 'built unavailable': return ('', '') # Handle specific text
    return ('', '')

def encode_mp_comparison_phrase(phrase_str):
    """[REDACTED_BY_SCRIPT]'Most common: [Type/Rating]'."""
    phrase_str_norm = str(phrase_str).strip().lower()
    if not phrase_str_norm or phrase_str_norm == "[REDACTED_BY_SCRIPT]": return '0'
    if phrase_str_norm.startswith("most common:"):
        val_part = phrase_str_norm.split("most common:", 1)[1].strip()
        prop_type_encoded = encode_mp_property_type(val_part)
        if prop_type_encoded != '0': return prop_type_encoded
        if len(val_part) == 1:
            epc_encoded = encode_rating(val_part)
            if epc_encoded != '0': return epc_encoded
        return '200'
    elif phrase_str_norm == "[REDACTED_BY_SCRIPT]": return '101'
    elif phrase_str_norm == "typical": return '102'
    elif phrase_str_norm == "lower than average": return '103'
    elif phrase_str_norm == "typical for type": return '104'
    elif phrase_str_norm == "higher than average": return '105'
    elif phrase_str_norm == "lower than most common": return '106'
    else: return '0'

def clean_monetary_value(money_str):
    """Removes '£' and ','[REDACTED_BY_SCRIPT]"""
    if not isinstance(money_str, str): money_str = str(money_str)
    cleaned = money_str.replace('£', '').replace(',', '').strip()
    num_pattern = re.compile(r'^-?\d+(\.\d+)?$')
    if num_pattern.fullmatch(cleaned): return cleaned
    return ''

def process_mp_generic_list(raw_element, target_size):
    """[REDACTED_BY_SCRIPT]"""
    items = []
    temp_items_str = []
    if isinstance(raw_element, (list, tuple)):
        for sub_elem in raw_element: temp_items_str.append(str(sub_elem).strip())
    elif isinstance(raw_element, str):
        elem_strip = raw_element.strip()
        if (elem_strip.startswith('[') and elem_strip.endswith(']')) or \
           (elem_strip.startswith('(') and elem_strip.endswith(')')):
            try:
                parsed_list = ast.literal_eval(elem_strip)
                if isinstance(parsed_list, (list, tuple)):
                    for sub_elem in parsed_list: temp_items_str.append(str(sub_elem).strip())
                else: temp_items_str.append(str(parsed_list).strip())
            except: temp_items_str.append(elem_strip)
        else: temp_items_str.append(elem_strip)
    elif raw_element is not None: temp_items_str.append(str(raw_element).strip())

    processed_idx = 0
    while processed_idx < len(temp_items_str) and len(items) < target_size:
        item_str = temp_items_str[processed_idx]
        item_processed = False
        if not item_str:
            items.append('')
            item_processed = True

        if not item_processed and item_str.startswith("Built ") and (target_size - len(items)) >= 1 :
            year_s, year_e = split_mp_built_year(item_str)
            if year_s or year_e:
                items.append(year_s)
                if len(items) < target_size: items.append(year_e)
                item_processed = True

        if not item_processed:
            encoded_val = encode_mp_comparison_phrase(item_str)
            if encoded_val != '0': items.append(encoded_val); item_processed = True

        if not item_processed:
            encoded_val = encode_mp_property_type(item_str)
            if encoded_val != '0': items.append(encoded_val); item_processed = True

        if not item_processed and len(item_str) == 1:
            encoded_val = encode_rating(item_str)
            if encoded_val != '0': items.append(encoded_val); item_processed = True

        if not item_processed:
            num_match = re.match(r'[REDACTED_BY_SCRIPT]', item_str)
            if num_match:
                items.append(num_match.group(1).replace(',', ''))
                item_processed = True

        if not item_processed:
            items.append(item_str) # Fallback

        processed_idx += 1

    while len(items) < target_size: items.append('')
    return items[:target_size]

# --- Helper to check for empty list representation ---
def is_empty_list_repr(element):
    # (Implementation unchanged)
    if element is None: return True
    if isinstance(element, list) and not element: return True
    if isinstance(element, str) and element.strip() == '[]': return True
    return False

# --- Helper Function for Row Processing (v22) ---
def process_row_data(row_field_content, input_filename):
    """
    Processes row content. Applies specific schemas for MP, BNL, Homipi, Chimnie.
    Dynamic flatten for others.
    """
    # --- Inner recursive flatten ---
    def _recursive_flatten(element, output_list):
        if isinstance(element, (list, tuple)):
            for sub_element in element: _recursive_flatten(sub_element, output_list)
        elif isinstance(element, str):
            element_stripped = element.strip()
            if (element_stripped.startswith('[') and element_stripped.endswith(']')) or \
               (element_stripped.startswith('(') and element_stripped.endswith(')')):
                try:
                    sub_parsed = ast.literal_eval(element_stripped)
                    if isinstance(sub_parsed, (list, tuple)): _recursive_flatten(sub_parsed, output_list)
                    else: output_list.append(str(sub_parsed).strip())
                except: output_list.append(element_stripped)
            else: output_list.append(element_stripped)
        else: output_list.append(str(element).strip())

    # --- Main logic ---
    final_processed_row = []
    is_mp_file = 'mpfromrightmove.csv' in input_filename.lower()
    is_bnl_file = '[REDACTED_BY_SCRIPT]' in input_filename.lower()
    is_homipi_file = '[REDACTED_BY_SCRIPT]' in input_filename.lower()
    is_chimnie_file = '[REDACTED_BY_SCRIPT]' in input_filename.lower()
    MP_TARGET_FIELDS = 38
    HOMIPI_TARGET_FIELDS = 125
    CHIMNIE_TARGET_COLS = 31

    try:
        initial_flattened_row = []

        # --- Data Preparation for initial_flattened_row (NOT for MP.csv) ---
        if not is_mp_file:
            if is_homipi_file:
                temp_homipi_list = []
                if isinstance(row_field_content, list):
                    temp_homipi_list = [str(field).strip() for field in row_field_content]
                else:
                    print(f"[REDACTED_BY_SCRIPT]")
                    temp_homipi_list = [str(row_field_content).strip()]
                pre_cleaned_homipi_list = []
                for field_content in temp_homipi_list:
                    cleaned_field = field_content
                    if isinstance(field_content, str):
                        if field_content.startswith("['"[REDACTED_BY_SCRIPT]"']"):
                             try:
                                  parsed_val = ast.literal_eval(field_content)
                                  if isinstance(parsed_val, list) and len(parsed_val) == 1:
                                       cleaned_field = str(parsed_val[0])
                             except: pass
                        if cleaned_field == "-": cleaned_field = ""
                    pre_cleaned_homipi_list.append(cleaned_field)
                initial_flattened_row = pre_cleaned_homipi_list
            else: # Covers BNL, Chimnie, StreetScan
                data_source_for_flattening = None
                if isinstance(row_field_content, str):
                    content_to_parse = row_field_content.strip()
                    if content_to_parse.startswith('"'[REDACTED_BY_SCRIPT]'"') and len(content_to_parse) > 1:
                        content_to_parse = content_to_parse[1:-1].replace('""', '"')
                    if content_to_parse == '[]': data_source_for_flattening = []
                    else:
                        try: data_source_for_flattening = ast.literal_eval(content_to_parse)
                        except: data_source_for_flattening = [row_field_content.replace('""', '"')]
                elif isinstance(row_field_content, list):
                    data_source_for_flattening = row_field_content
                else:
                    data_source_for_flattening = [str(row_field_content)]
                if data_source_for_flattening is not None:
                    _recursive_flatten(data_source_for_flattening, initial_flattened_row)


        # --- MP.CSV SPECIAL SCHEMA-ENFORCED PROCESSING (v22) ---
        if is_mp_file:
            if not isinstance(row_field_content, str):
                print(f"[REDACTED_BY_SCRIPT]")
                return ['MP_INPUT_ERROR'] * MP_TARGET_FIELDS
            content_to_parse = row_field_content.strip()
            if content_to_parse.startswith('"'[REDACTED_BY_SCRIPT]'"') and len(content_to_parse) > 1:
                content_to_parse = content_to_parse[1:-1].replace('""', '"')
            parsed_main_list = []
            if content_to_parse == '[]': parsed_main_list = []
            else:
                try: parsed_main_list = ast.literal_eval(content_to_parse)
                except Exception as e_ast:
                    print(f"[REDACTED_BY_SCRIPT]")
                    return ['MP_AST_ERROR'] * MP_TARGET_FIELDS
            if not isinstance(parsed_main_list, list):
                print(f"[REDACTED_BY_SCRIPT]")
                return ['MP_PARSE_TYPE_ERROR'] * MP_TARGET_FIELDS

            output_row = [''] * MP_TARGET_FIELDS

            def get_elem(data_list, index, default_val=None):
                try: return data_list[index] if index < len(data_list) else default_val
                except IndexError: return default_val
            def parse_list_element(raw_element):
                 if isinstance(raw_element, (list, tuple)): return raw_element
                 if isinstance(raw_element, str):
                     element_stripped = raw_element.strip()
                     if (element_stripped.startswith('[') and element_stripped.endswith(']')) or \
                        (element_stripped.startswith('(') and element_stripped.endswith(')')):
                         try:
                             parsed = ast.literal_eval(element_stripped)
                             return parsed if isinstance(parsed, (list, tuple)) else None
                         except: return None
                 return None

            output_row[0] = str(get_elem(parsed_main_list, 0, '')).strip()
            output_row[1] = encode_mp_risk_level(get_elem(parsed_main_list, 1, ''))
            output_row[2] = str(get_elem(parsed_main_list, 2, '')).strip()
            output_row[3] = str(get_elem(parsed_main_list, 3, '')).strip()

            elem4_raw = get_elem(parsed_main_list, 4); elem5_raw = get_elem(parsed_main_list, 5)
            elem9_raw = get_elem(parsed_main_list, 9); elem10_raw = get_elem(parsed_main_list, 10)
            use_alt_indices = False
            if is_empty_list_repr(elem4_raw) and is_empty_list_repr(elem5_raw) and \
               not is_empty_list_repr(elem9_raw) and not is_empty_list_repr(elem10_raw):
                use_alt_indices = True

            source_list_1_raw = elem9_raw if use_alt_indices else elem4_raw
            source_list_2_raw = elem10_raw if use_alt_indices else elem5_raw
            idx_list3 = 11 if use_alt_indices else 7
            idx_list4 = 12 if use_alt_indices else 8
            idx_list6 = 14 if use_alt_indices else 10
            source_list_3_raw = get_elem(parsed_main_list, idx_list3)
            source_list_4_raw = get_elem(parsed_main_list, idx_list4)
            source_list_6_raw = get_elem(parsed_main_list, idx_list6)
            main_prop_type_str = str(get_elem(parsed_main_list, 6, '')).strip()

            actual_list_1 = parse_list_element(source_list_1_raw)
            list1_items = [str(i).strip() for i in actual_list_1] if isinstance(actual_list_1, (list, tuple)) else []
            actual_list_3 = parse_list_element(source_list_3_raw)
            list3_items = [str(i).strip() for i in actual_list_3] if isinstance(actual_list_3, (list, tuple)) else []

            found_prop_type = ''
            found_built_year = ''
            found_epc = ''
            found_floor_area = ''
            score1 = ''
            score2 = ''
            processed_indices_list1 = set()
            processed_indices_list3 = set()

            # Process Built Year
            for idx, item in enumerate(list1_items):
                 if item.startswith('Built '): found_built_year = item; processed_indices_list1.add(idx); break
            if not found_built_year:
                 for idx, item in enumerate(list3_items):
                     if item.startswith('Built '): found_built_year = item; processed_indices_list3.add(idx); break
            year_start, year_end = split_mp_built_year(found_built_year)
            output_row[10] = year_start
            output_row[11] = year_end

            # Process Floor Area
            num_extract_pattern = re.compile(r'([\d,]+(\.\d+)?)')
            for idx, item in enumerate(list1_items):
                if idx in processed_indices_list1: continue
                if item.endswith(' sqm floor area'):
                    match = num_extract_pattern.search(item)
                    if match: found_floor_area = match.group(1).replace(',', ''); processed_indices_list1.add(idx); break
            if not found_floor_area:
                 for idx, item in enumerate(list3_items):
                     if idx in processed_indices_list3: continue
                     if item.endswith(' sqm floor area'):
                         match = num_extract_pattern.search(item)
                         if match: found_floor_area = match.group(1).replace(',', ''); processed_indices_list3.add(idx); break
            output_row[6] = found_floor_area

            # Process Property Type
            for idx, item in enumerate(list1_items):
                if idx in processed_indices_list1: continue
                if encode_mp_property_type(item) != '0': found_prop_type = item; processed_indices_list1.add(idx); break
            if not found_prop_type and encode_mp_property_type(main_prop_type_str) != '0':
                found_prop_type = main_prop_type_str
            if not found_prop_type:
                 for idx, item in enumerate(list3_items):
                     if idx in processed_indices_list3: continue
                     if encode_mp_property_type(item) != '0': found_prop_type = item; processed_indices_list3.add(idx); break
            output_row[8] = encode_mp_property_type(found_prop_type)

            # Process EPC
            epc_candidates = []
            for idx, item in enumerate(list1_items):
                 if idx in processed_indices_list1: continue
                 if item.startswith('Tax band '): rating = item.replace('Tax band ', '').strip(); epc_candidates.append((rating, idx, "tax_band", 1))
                 elif item.endswith(' energy rating'): rating = item.replace(' energy rating', '').strip(); epc_candidates.append((rating, idx, "energy_rating", 1))
                 elif re.fullmatch(r'[A-G]', item): epc_candidates.append((item, idx, "standalone", 1))
            for idx, item in enumerate(list3_items):
                 if idx in processed_indices_list3: continue
                 if item.startswith('Tax band '): rating = item.replace('Tax band ', '').strip(); epc_candidates.append((rating, idx, "tax_band", 3))
                 elif item.endswith(' energy rating'): rating = item.replace(' energy rating', '').strip(); epc_candidates.append((rating, idx, "energy_rating", 3))
                 elif re.fullmatch(r'[A-G]', item): epc_candidates.append((item, idx, "standalone", 3))

            best_epc_tuple = None
            if epc_candidates:
                if any(c[2] == "energy_rating" for c in epc_candidates): best_epc_tuple = next(c for c in epc_candidates if c[2] == "energy_rating")
                elif any(c[2] == "tax_band" for c in epc_candidates): best_epc_tuple = next(c for c in epc_candidates if c[2] == "tax_band")
                elif any(c[2] == "standalone" for c in epc_candidates): best_epc_tuple = next(c for c in epc_candidates if c[2] == "standalone")

            if best_epc_tuple:
                found_epc = best_epc_tuple[0]
                epc_idx, epc_source_list = best_epc_tuple[1], best_epc_tuple[3]
                if epc_source_list == 1: processed_indices_list1.add(epc_idx)
                elif epc_source_list == 3: processed_indices_list3.add(epc_idx)
            output_row[12] = encode_rating(found_epc) # Use global encode_rating

            # Process Scores
            num_pattern = re.compile(r'^-?\d+(\.\d+)?$')
            scores_candidates = []
            for idx, item in enumerate(list1_items):
                if idx in processed_indices_list1: continue
                if num_pattern.fullmatch(item.replace(',', '')): scores_candidates.append(item.replace(',', ''))
            if scores_candidates: score1 = scores_candidates.pop(0)
            if scores_candidates: score2 = scores_candidates.pop(0)
            output_row[4] = score1
            output_row[5] = score2

            # Comparison Phrases (list 2)
            actual_list_2 = parse_list_element(source_list_2_raw)
            comparison_items_encoded = []
            if isinstance(actual_list_2, (list, tuple)):
                for item in actual_list_2: comparison_items_encoded.append(encode_mp_comparison_phrase(str(item)))
            while len(comparison_items_encoded) < 7: comparison_items_encoded.append('0')
            output_row[13:20] = comparison_items_encoded[:7]

            # Generic List 3 & 4 (filtered)
            filtered_list3 = [item for idx, item in enumerate(list3_items) if idx not in processed_indices_list3]
            actual_list_4 = parse_list_element(source_list_4_raw) # List 4 wasn't used for fallbacks, process directly

            output_row[20:24] = process_mp_generic_list(filtered_list3, 4)
            output_row[24:28] = process_mp_generic_list(actual_list_4, 4)

            # Monetary values (list 6)
            temp_list6_str = []
            if isinstance(source_list_6_raw, (list, tuple)):
                for sub_elem in source_list_6_raw: temp_list6_str.append(str(sub_elem).strip())
            elif isinstance(source_list_6_raw, str):
                elem_strip = source_list_6_raw.strip()
                if (elem_strip.startswith('[') and elem_strip.endswith(']')) or \
                   (elem_strip.startswith('(') and elem_strip.endswith(')')):
                    try:
                        parsed_l6 = ast.literal_eval(elem_strip)
                        if isinstance(parsed_l6, (list, tuple)):
                            for sub_elem in parsed_l6: temp_list6_str.append(str(sub_elem).strip())
                        else: temp_list6_str.append(str(parsed_l6).strip())
                    except: temp_list6_str.append(elem_strip)
                else: temp_list6_str.append(elem_strip)
            elif source_list_6_raw is not None:
                temp_list6_str.append(str(source_list_6_raw).strip())
            list6_items_cleaned = [clean_monetary_value(item_str) for item_str in temp_list6_str]
            while len(list6_items_cleaned) < 5: list6_items_cleaned.append('')
            output_row[33:38] = list6_items_cleaned[:5]

            final_processed_row = output_row

        # --- BNL.CSV SPECIFIC CLEANING ---
        elif is_bnl_file:
            if not initial_flattened_row and isinstance(row_field_content, list) and not row_field_content:
                 final_processed_row = []
            elif not initial_flattened_row:
                 print(f"[REDACTED_BY_SCRIPT]")
                 final_processed_row = ["BNL_FLATTEN_ERROR"]
            else:
                num_extract_pattern = re.compile(r'([-+]?[\d,]+(?:\.\d+)?)')
                cleaned_bnl_row = []
                for i_bnl, field_bnl in enumerate(initial_flattened_row):
                    field_str_bnl = str(field_bnl).strip()
                    if i_bnl == 0: cleaned_bnl_row.append(field_str_bnl); continue
                    match = num_extract_pattern.search(field_str_bnl)
                    if match: cleaned_bnl_row.append(match.group(1).replace(',', ''))
                    else: cleaned_bnl_row.append(field_str_bnl)
                final_processed_row = cleaned_bnl_row

        # --- HOMIPI.CSV FIXED COLUMN CLEANING & ENCODING ---
        elif is_homipi_file:
             if not initial_flattened_row and isinstance(row_field_content, list) and not row_field_content:
                  final_processed_row = [''] * HOMIPI_TARGET_FIELDS
             elif not initial_flattened_row:
                  print(f"[REDACTED_BY_SCRIPT]")
                  final_processed_row = ["HOMIPI_PREP_ERROR"] * HOMIPI_TARGET_FIELDS
             else:
                 cleaned_homipi_row = [''] * HOMIPI_TARGET_FIELDS
                 pattern_col3 = re.compile(r'[REDACTED_BY_SCRIPT]'); pattern_col4 = re.compile(r'[REDACTED_BY_SCRIPT]')
                 num_extract_first = re.compile(r'([-+]?[\d,]+(?:\.\d+)?)'); num_extract_all = re.compile(r"(\d[\d,]*(?:\.\d+)?)")
                 current_out_col = 0
                 for i, field_content_original in enumerate(initial_flattened_row):
                     field_str = str(field_content_original)
                     if i > 22: break
                     if current_out_col >= 25: break
                     target_output_index = current_out_col
                     match = None
                     if i == 0: cleaned_homipi_row[target_output_index] = field_str; current_out_col += 1
                     elif i == 1:
                         match = num_extract_first.search(field_str); cleaned_homipi_row[target_output_index] = match.group(1).replace(',', '') if match else ''; current_out_col += 1
                     elif i == 2:
                         match = pattern_col3.search(field_str)
                         if match: cleaned_homipi_row[target_output_index] = match.group(1).replace(',', ''); cleaned_homipi_row[target_output_index + 1] = match.group(2).replace(',', '');
                         current_out_col += 2
                     elif i == 3:
                         match = pattern_col4.search(field_str)
                         if match: cleaned_homipi_row[target_output_index] = match.group(1).replace(',', ''); cleaned_homipi_row[target_output_index + 1] = match.group(2).replace(',', '');
                         current_out_col += 2
                     elif i == 6: day, month, year = encode_date(field_str); cleaned_homipi_row[target_output_index] = day; cleaned_homipi_row[target_output_index + 1] = month; cleaned_homipi_row[target_output_index + 2] = year; current_out_col += 3
                     elif i == 7: cleaned_homipi_row[target_output_index] = encode_property_type(field_str); current_out_col += 1 # Global property type
                     elif i == 18: cleaned_homipi_row[target_output_index] = encode_rating(field_str); current_out_col += 1 # Global EPC rating
                     elif i == 19: cleaned_homipi_row[target_output_index] = encode_rating(field_str); current_out_col += 1 # Global EPC rating
                     elif i == 21 or i == 22: pass
                     else:
                         match = num_extract_first.search(field_str); cleaned_homipi_row[target_output_index] = match.group(1).replace(',', '') if match else field_str; current_out_col += 1
                 out_col_for_location = 25
                 if len(initial_flattened_row) > 23:
                     for j_homipi in range(23, len(initial_flattened_row)):
                         if out_col_for_location >= HOMIPI_TARGET_FIELDS: break
                         field_str_loc = str(initial_flattened_row[j_homipi])
                         numbers_found = num_extract_all.findall(field_str_loc)
                         output_cols_for_field = [num.replace(',', '') for num in numbers_found[:5]]
                         while len(output_cols_for_field) < 5: output_cols_for_field.append('')
                         slots_to_fill = min(5, HOMIPI_TARGET_FIELDS - out_col_for_location)
                         if slots_to_fill > 0: cleaned_homipi_row[out_col_for_location : out_col_for_location + slots_to_fill] = output_cols_for_field[:slots_to_fill]
                         out_col_for_location += slots_to_fill
                 final_processed_row = cleaned_homipi_row

        # --- CHIMNIE.CSV SPECIFIC PROCESSING ---
        elif is_chimnie_file:
            processed_chimnie_row = list(initial_flattened_row)
            risk_keywords = {"VERY LOW", "LOW", "MED", "HIGH"}
            sections_data_indices_map = [ list(range(1, 10)), list(range(11, 20)), list(range(21, 30)) ]
            for section_indices in sections_data_indices_map:
                nullify_this_section = False
                for data_idx in section_indices:
                    if data_idx < len(processed_chimnie_row):
                        if str(processed_chimnie_row[data_idx]).strip().upper() in risk_keywords:
                            nullify_this_section = True; break
                if nullify_this_section:
                    for data_idx_to_nullify in section_indices:
                        if data_idx_to_nullify < len(processed_chimnie_row):
                            processed_chimnie_row[data_idx_to_nullify] = '0'
            risk_text_column_actual_indices = [10, 20, 30]
            for text_col_idx in risk_text_column_actual_indices:
                if text_col_idx < len(processed_chimnie_row):
                    processed_chimnie_row[text_col_idx] = encode_chimnie_risk(processed_chimnie_row[text_col_idx])
            while len(processed_chimnie_row) < CHIMNIE_TARGET_COLS: processed_chimnie_row.append('0')
            final_processed_row = processed_chimnie_row[:CHIMNIE_TARGET_COLS]

        # --- OTHER FILES (StreetScan) ---
        else:
            final_processed_row = initial_flattened_row

    # --- Error Handling ---
    except (ValueError, SyntaxError, MemoryError, TypeError) as e:
        print(f"[REDACTED_BY_SCRIPT]")
        error_val_list = ['PARSE_ERROR']
        target_len = MP_TARGET_FIELDS if is_mp_file else (HOMIPI_TARGET_FIELDS if is_homipi_file else (CHIMNIE_TARGET_COLS if is_chimnie_file else 1))
        final_processed_row = error_val_list * target_len
        if final_processed_row:
             if isinstance(row_field_content, str): final_processed_row[0] = row_field_content[:100]
             elif isinstance(row_field_content, list) and row_field_content: final_processed_row[0] = str(row_field_content[0])[:100]
    except Exception as e:
        print(f"[REDACTED_BY_SCRIPT]")
        error_val_list = ['UNEXPECTED_ERROR']
        target_len = MP_TARGET_FIELDS if is_mp_file else (HOMIPI_TARGET_FIELDS if is_homipi_file else (CHIMNIE_TARGET_COLS if is_chimnie_file else 1))
        final_processed_row = error_val_list * target_len
        if final_processed_row:
             if isinstance(row_field_content, str): final_processed_row[0] = row_field_content[:100]
             elif isinstance(row_field_content, list) and row_field_content: final_processed_row[0] = str(row_field_content[0])[:100]

    return final_processed_row if final_processed_row else []


# --- Deduplication and Flattening Logic (Main loop) ---
# --- (Ensure the main loop from v17+ is used here) ---
print("[REDACTED_BY_SCRIPT]") # Version bump
total_errors_all_files = 0
for i, input_filename in enumerate(input_filenames):
    output_filename = output_filenames[i]
    if not os.path.exists(input_filename):
        print(f"[REDACTED_BY_SCRIPT]")
        continue
    print(f"\nProcessing '{input_filename}' -> '{output_filename}'...")
    seen_normalized_addresses_this_file = set()
    lines_written = 0; lines_skipped = 0; lines_error = 0; line_num = 0
    try:
        with open(input_filename, 'r', encoding='utf-8', newline='') as infile, \
             open(output_filename, 'w', encoding='utf-8', newline='') as outfile:
            csv_reader = csv.reader(infile)
            csv_writer = csv.writer(outfile)
            for line_num, row in enumerate(csv_reader, 1):
                if line_num % 5000 == 0: print(f"[REDACTED_BY_SCRIPT]")
                if not row: continue

                # Heuristic for Format A (single string field needing ast.literal_eval) vs
                # Format B (already a list of strings from CSV reader).
                # Homipi and BNL examples suggest they are Format B. MP is Format A.
                is_format_a = (len(row) == 1) and not any(fname_part in input_filename.lower() for fname_part in ['homipi', 'bnl'])

                data_to_process = row[0] if is_format_a else row

                try:
                    final_row_data = process_row_data(data_to_process, input_filename)
                except Exception as proc_err:
                     print(f"[REDACTED_BY_SCRIPT]")
                     print(f"[REDACTED_BY_SCRIPT]")
                     lines_error += 1; continue

                address_string_to_check = None
                if isinstance(final_row_data, list) and final_row_data:
                    # Use str() conversion for safety, in case first element isn't string
                    address_string_to_check = str(final_row_data[0])

                if address_string_to_check and address_string_to_check.strip() and "ERROR" not in address_string_to_check.upper():
                    normalized_address = ' '.join(address_string_to_check.lower().split())
                    if normalized_address not in seen_normalized_addresses_this_file:
                        csv_writer.writerow(final_row_data)
                        seen_normalized_addresses_this_file.add(normalized_address)
                        lines_written += 1
                    else:
                        lines_skipped += 1
                else:
                    # Only print skip message if it's not an explicitly marked ERROR row
                    if not (isinstance(final_row_data, list) and final_row_data and "ERROR" in str(final_row_data[0]).upper()):
                         print(f"[REDACTED_BY_SCRIPT]")
                    lines_error += 1
    except FileNotFoundError: print(f"[REDACTED_BY_SCRIPT]"); total_errors_all_files +=1
    except csv.Error as e: print(f"CSV Error reading '{input_filename}' near line {line_num}: {e}"); lines_error += 1; total_errors_all_files +=1
    except IOError as e: print(f"[REDACTED_BY_SCRIPT]'{input_filename}' or '{output_filename}': {e}"); total_errors_all_files +=1
    except Exception as e: print(f"[REDACTED_BY_SCRIPT]'{input_filename}' near line {line_num}: {e}"); lines_error +=1; total_errors_all_files +=1
    finally:
        print(f"Finished '{input_filename}'[REDACTED_BY_SCRIPT]")
        if lines_error > 0: print(f"[REDACTED_BY_SCRIPT]")
        total_errors_all_files += lines_error

print("[REDACTED_BY_SCRIPT]")
if total_errors_all_files > 0: print(f"[REDACTED_BY_SCRIPT]")