import os
import json
import re
import logging
from collections import defaultdict

# --- Configuration ---
MAIN_OUTPUT_DIR = r"[REDACTED_BY_SCRIPT]"
LOG_FILE = os.path.join(MAIN_OUTPUT_DIR, "json_update_log.txt")

logging.basicConfig(
    level=logging.INFO,
    format='[REDACTED_BY_SCRIPT]',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w'),
        logging.StreamHandler()
    ]
)

# --- START: COPIED FROM gemini_property_feature_generator.py ---
ROOM_TOKENS = {
    "KITCHEN": "KITCHEN", "LIVING_ROOM": "LIVING_ROOM", "DINING_AREA": "DINING_AREA",
    "BEDROOM": "BEDROOM", "BATHROOM": "BATHROOM", "HALLWAY_LANDING": "HALLWAY_LANDING",
    "UTILITY_ROOM": "UTILITY_ROOM", "GARAGE": "GARAGE", "CONSERVATORY_SUNROOM": "CONSERVATORY_SUNROOM",
    "OFFICE_STUDY": "OFFICE_STUDY", "GARDEN_YARD": "GARDEN_YARD", "PATIO_DECKING": "PATIO_DECKING",
    "DRIVEWAY": "DRIVEWAY", "EXTERIOR_FRONT": "EXTERIOR_FRONT", "EXTERIOR_REAR": "EXTERIOR_REAR",
    "AERIAL_VIEW": "AERIAL_VIEW", "VIEW_FROM_PROPERTY": "VIEW_FROM_PROPERTY",
    "OUTBUILDING_SHED": "OUTBUILDING_SHED", "STORAGE_AREA": "STORAGE_AREA",
    "DETAIL_SHOT": "DETAIL_SHOT", "OTHER_INDOOR_SPACE": "OTHER_INDOOR_SPACE",
    "FLOORPLAN": "FLOORPLAN", "SITE_PLAN": "SITE_PLAN", "STORM_PORCH": "STORM_PORCH",
    "EAVES": "EAVES"
}
UNKNOWN_TOKEN = "UNKNOWN"

def get_room_token(label_text):
    if not label_text or not isinstance(label_text, str):
        return UNKNOWN_TOKEN
    text = label_text.lower().strip().replace("_", " ")
    if "primary bedroom" in text: return ROOM_TOKENS["BEDROOM"]
    if "other bedrooms" in text: return ROOM_TOKENS["BEDROOM"]
    if "kitchen" in text: return ROOM_TOKENS["KITCHEN"]
    if "lounge" in text or "living room" in text or "sitting room" in text or "reception room" in text or "play room" in text: return ROOM_TOKENS["LIVING_ROOM"]
    if "dining" in text: return ROOM_TOKENS["DINING_AREA"]
    if "bedroom" in text: return ROOM_TOKENS["BEDROOM"]
    if "bathroom" in text or "shower room" in text or "ensuite" in text or "wc" in text or "toilet" in text: return ROOM_TOKENS["BATHROOM"]
    if "hall" in text or "landing" in text: return ROOM_TOKENS["HALLWAY_LANDING"]
    if "utility" in text: return ROOM_TOKENS["UTILITY_ROOM"]
    if "garage" in text: return ROOM_TOKENS["GARAGE"]
    if "conservatory" in text or "sun room" in text: return ROOM_TOKENS["CONSERVATORY_SUNROOM"]
    if "office" in text or "study" in text: return ROOM_TOKENS["OFFICE_STUDY"]
    if "garden" in text or "yard" in text: return ROOM_TOKENS["GARDEN_YARD"]
    if "patio" in text or "decking" in text: return ROOM_TOKENS["PATIO_DECKING"]
    if "driveway" in text: return ROOM_TOKENS["DRIVEWAY"]
    if "front exterior" in text or "front of house" in text: return ROOM_TOKENS["EXTERIOR_FRONT"]
    if "rear exterior" in text or "rear of house" in text: return ROOM_TOKENS["EXTERIOR_REAR"]
    if "aerial view" in text: return ROOM_TOKENS["AERIAL_VIEW"]
    if "view from property" in text: return ROOM_TOKENS["VIEW_FROM_PROPERTY"]
    if "outbuilding" in text or "shed" in text: return ROOM_TOKENS["OUTBUILDING_SHED"]
    if "storage" in text or "wardrobe" in text and "built" not in text: return ROOM_TOKENS["STORAGE_AREA"]
    if "detail" in text or "close-up" in text: return ROOM_TOKENS["DETAIL_SHOT"]
    if "games room" in text or "mezzanine" in text or "other" in text: return ROOM_TOKENS["OTHER_INDOOR_SPACE"]
    if "floorplan" in text: return ROOM_TOKENS["FLOORPLAN"]
    if "site plan" in text: return ROOM_TOKENS["SITE_PLAN"]
    if "storm porch" in text: return ROOM_TOKENS["STORM_PORCH"]
    if "eaves" in text: return ROOM_TOKENS["EAVES"]
    # Do not log unknown here, as this script is for updating based on existing data
    # logging.warning(f"[REDACTED_BY_SCRIPT]'{label_text}'[REDACTED_BY_SCRIPT]")
    return UNKNOWN_TOKEN
# --- END: COPIED FROM gemini_property_feature_generator.py ---


# --- Helper Functions (parse_dimensions_and_area, find_latest_year_suffix, update_step1_floorplan) ---
# (These remain mostly the same as in the previous version you approved,
#  parse_dimensions_and_area and update_step1_floorplan are included below for completeness)

def parse_dimensions_and_area(dimension_string):
    if not dimension_string or not isinstance(dimension_string, str):
        return None
    match = re.search(r'[REDACTED_BY_SCRIPT]', dimension_string, re.IGNORECASE)
    if match:
        try:
            width = float(match.group(1))
            depth = float(match.group(2))
            return width * depth
        except ValueError:
            logging.warning(f"[REDACTED_BY_SCRIPT]'{dimension_string}'")
            return None
    else:
        other_match = re.search(r'[REDACTED_BY_SCRIPT]', dimension_string, re.IGNORECASE)
        if other_match:
            logging.info(f"    Dimension string '{dimension_string}' seems to be already processed 'Other Bedrooms'[REDACTED_BY_SCRIPT]")
            return None
        logging.warning(f"[REDACTED_BY_SCRIPT]")
        return None

def find_latest_year_suffix(property_output_dir_path):
    latest_year = None
    if not os.path.isdir(property_output_dir_path):
        return None
    years_found = []
    for filename in os.listdir(property_output_dir_path):
        match = re.search(r'_y(\d{4})\.json$', filename)
        if match:
            try:
                year = int(match.group(1))
                years_found.append(year)
            except ValueError:
                continue
    if years_found:
        return str(max(years_found))
    return None

def update_step1_floorplan(filepath):
    # This function remains the same as the previous version I provided.
    # It correctly identifies bedrooms, relabels to "Primary Bedroom" / "Other Bedrooms",
    # and returns a label_map of {original_raw_label: new_label}.
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        return None, {}
    except json.JSONDecodeError:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        return None, {}

    if 'rooms_with_dimensions' not in data or not isinstance(data['rooms_with_dimensions'], list):
        logging.warning(f"  'rooms_with_dimensions'[REDACTED_BY_SCRIPT]")
        return data, {}

    original_rooms_with_dimensions = data.get('rooms_with_dimensions', [])
    bedrooms_details = []
    non_bedroom_rooms = []
    label_map = {} # Stores original_raw_label -> new_label ("Primary Bedroom" or "Other Bedrooms")

    for room in original_rooms_with_dimensions:
        if not isinstance(room, dict) or 'label' not in room:
            non_bedroom_rooms.append(room)
            continue
        label = room['label']
        dimensions_str = room.get('dimensions')
        if label.lower() == "primary bedroom" or label.lower() == "other bedrooms":
            non_bedroom_rooms.append(room)
            label_map[label] = label # Maps to itself if already processed
            continue
        if label.lower().startswith("bedroom"): # Catches "Bedroom", "Bedroom 1", "Master Bedroom" etc.
            area = parse_dimensions_and_area(dimensions_str)
            if area is not None:
                bedrooms_details.append({
                    "original_label": label, # This is the raw label from the file
                    "dimensions": dimensions_str,
                    "area": area,
                })
            else:
                logging.warning(f"    Bedroom '{label}'[REDACTED_BY_SCRIPT]'{dimensions_str}'[REDACTED_BY_SCRIPT]")
                non_bedroom_rooms.append(room)
        else:
            non_bedroom_rooms.append(room)

    if not bedrooms_details:
        logging.info(f"[REDACTED_BY_SCRIPT]")
        return data, label_map # label_map will be empty or contain only self-maps

    bedrooms_details.sort(key=lambda x: x['area'], reverse=True)
    new_rooms_with_dimensions = list(non_bedroom_rooms)
    
    primary_bedroom_info = bedrooms_details[0]
    new_rooms_with_dimensions.append({
        "label": "Primary Bedroom", # The new standardized label
        "dimensions": primary_bedroom_info['dimensions']
    })
    label_map[primary_bedroom_info['original_label']] = "Primary Bedroom"
    logging.info(f"    '{primary_bedroom_info['original_label']}'[REDACTED_BY_SCRIPT]'area']:.2f}) "
                 f"re-labeled to 'Primary Bedroom'[REDACTED_BY_SCRIPT]")

    other_bedrooms_list = bedrooms_details[1:]
    if other_bedrooms_list:
        num_other_beds = len(other_bedrooms_list)
        avg_other_area = sum(b['area'] for b in other_bedrooms_list) / num_other_beds
        other_bedrooms_entry = {
            "label": "Other Bedrooms", # The new standardized label
            "dimensions": f"[REDACTED_BY_SCRIPT]"
        }
        new_rooms_with_dimensions.append(other_bedrooms_entry)
        for b_info in other_bedrooms_list:
            label_map[b_info['original_label']] = "Other Bedrooms"
        logging.info(f"[REDACTED_BY_SCRIPT]'Other Bedrooms' "
                     f"[REDACTED_BY_SCRIPT]")
        logging.info(f"[REDACTED_BY_SCRIPT]'Other Bedrooms': {[b['original_label'[REDACTED_BY_SCRIPT]")

    data['rooms_with_dimensions'] = new_rooms_with_dimensions
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logging.info(f"[REDACTED_BY_SCRIPT]")
    except Exception as e:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        return None, {} # Return None for data to indicate critical save failure
    return data, label_map


def update_subsequent_json(filepath, label_map_from_step1, data_key_to_update=None, 
                           is_list_of_dicts_with_label=False, step1_data_content=None):
    """
    Updates JSON files (Steps 2, 3, 4, 5, 6).
    - label_map_from_step1: Maps original raw bedroom labels to "Primary Bedroom" / "Other Bedrooms".
    - step1_data_content: The content of the (updated) Step 1 JSON.
    """
    # Check if there's any work to do
    needs_update_for_step2_source = "step2_assignments" in filepath.lower() and step1_data_content is not None
    if not label_map_from_step1 and not needs_update_for_step2_source:
        logging.info(f"[REDACTED_BY_SCRIPT]")
        return True

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        logging.warning(f"[REDACTED_BY_SCRIPT]")
        return False
    except json.JSONDecodeError:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        return False

    changed = False

    # --- Logic for Step 2: Correcting 'source' and applying bedroom label_map ---
    if "step2_assignments" in filepath.lower() and data_key_to_update == 'room_assignments' and is_list_of_dicts_with_label:
        known_floorplan_tokens = set() # Store TOKENIZED labels from Step 1
        if step1_data_content:
            for room_dim in step1_data_content.get('rooms_with_dimensions', []):
                if isinstance(room_dim, dict) and 'label' in room_dim:
                    # The labels in updated step1_data_content are already "Primary Bedroom", "Other Bedrooms", or original non-bedroom labels
                    known_floorplan_tokens.add(get_room_token(room_dim['label']))
            for label_no_dim_raw in step1_data_content.get('[REDACTED_BY_SCRIPT]', []):
                if isinstance(label_no_dim_raw, str):
                    known_floorplan_tokens.add(get_room_token(label_no_dim_raw))
            # Remove UNKNOWN_TOKEN if it got added, as it's not a valid floorplan source indicator
            if UNKNOWN_TOKEN in known_floorplan_tokens:
                known_floorplan_tokens.remove(UNKNOWN_TOKEN)
        
        logging.info(f"[REDACTED_BY_SCRIPT]'None'}")

        if isinstance(data.get(data_key_to_update), list):
            new_assignments_list = []
            other_bedrooms_image_indices = []
            other_bedrooms_source_candidate = None

            for item in data[data_key_to_update]:
                if isinstance(item, dict) and 'label' in item:
                    raw_assignment_label = item['label'] # The label as it is in Step 2 file
                    
                    # 1. Apply bedroom remapping using label_map_from_step1 (which maps raw original Step 1 labels)
                    final_label_for_item = label_map_from_step1.get(raw_assignment_label, raw_assignment_label)
                    if final_label_for_item != raw_assignment_label:
                        item['label'] = final_label_for_item
                        changed = True
                        logging.info(f"    Step 2: Remapped raw label '{raw_assignment_label}' to '{final_label_for_item}' using Step 1 map.")
                    
                    # 2. Correct the source based on known_floorplan_tokens
                    # Tokenize the label *after* potential remapping
                    tokenized_final_label = get_room_token(final_label_for_item)

                    if known_floorplan_tokens and tokenized_final_label != UNKNOWN_TOKEN and tokenized_final_label in known_floorplan_tokens:
                        if item.get('source') != "Floorplan":
                            item['source'] = "Floorplan"
                            changed = True
                            logging.info(f"    Step 2: Corrected source to 'Floorplan' for label '{final_label_for_item}'[REDACTED_BY_SCRIPT]")
                    
                    # 3. Consolidate "Other Bedrooms"
                    if final_label_for_item == "Other Bedrooms":
                        if "image_indices" in item and isinstance(item['image_indices'], list):
                            other_bedrooms_image_indices.extend(item['image_indices'])
                        if item.get('source') == "Floorplan":
                            other_bedrooms_source_candidate = "Floorplan"
                        elif other_bedrooms_source_candidate is None:
                            other_bedrooms_source_candidate = item.get('source', "Generated")
                    else:
                        new_assignments_list.append(item)
                else:
                    new_assignments_list.append(item)
            
            if other_bedrooms_image_indices:
                consolidated_other_bedrooms = {
                    "label": "Other Bedrooms",
                    "source": other_bedrooms_source_candidate or "Generated",
                    "image_indices": sorted(list(set(other_bedrooms_image_indices)))
                }
                new_assignments_list.append(consolidated_other_bedrooms)
                logging.info(f"[REDACTED_BY_SCRIPT]'Other Bedrooms'[REDACTED_BY_SCRIPT]")
            data[data_key_to_update] = new_assignments_list
        else:
            logging.warning(f"  Key '{data_key_to_update}'[REDACTED_BY_SCRIPT]")

    # --- Logic for Steps 3, 4, 5, 6 (key renaming using label_map_from_step1) ---
    elif data_key_to_update is None: # Assumes dict keyed by raw room labels
        if isinstance(data, dict):
            # ... (The rest of the logic for Steps 3, 4, 5, 6 remains largely the same
            #      as the previous version, as it operates on the label_map_from_step1
            #      which maps original raw labels to new "[REDACTED_BY_SCRIPT]" labels.)
            new_data_dict_keys = {}
            if '[REDACTED_BY_SCRIPT]' in data and isinstance(data['[REDACTED_BY_SCRIPT]'], dict):
                new_data_dict_keys['[REDACTED_BY_SCRIPT]'] = data['[REDACTED_BY_SCRIPT]']

            if '[REDACTED_BY_SCRIPT]' in data and 'room_ratings_final' in data:
                old_eval_labels = data['[REDACTED_BY_SCRIPT]']
                old_ratings_final = data['room_ratings_final']
                temp_merged_ratings = defaultdict(list)

                for i, old_raw_label in enumerate(old_eval_labels):
                    # Use label_map_from_step1 which maps original raw Step 1 labels
                    new_label_for_list = label_map_from_step1.get(old_raw_label, old_raw_label)
                    if i < len(old_ratings_final) and old_ratings_final[i] is not None:
                        temp_merged_ratings[new_label_for_list].append(old_ratings_final[i])
                
                final_eval_labels = []
                final_room_ratings = []
                for new_label, list_of_rating_lists in temp_merged_ratings.items():
                    final_eval_labels.append(new_label)
                    if new_label == "Other Bedrooms" and len(list_of_rating_lists) > 1:
                        num_rating_sets = len(list_of_rating_lists)
                        if num_rating_sets > 0 and list_of_rating_lists[0] and len(list_of_rating_lists[0]) > 0: # Added check for list_of_rating_lists[0]
                            num_ratings_per_set = len(list_of_rating_lists[0])
                            averaged_ratings = [None] * num_ratings_per_set
                            for j in range(num_ratings_per_set):
                                valid_ratings_for_j = [rs[j] for rs in list_of_rating_lists if rs and len(rs) > j and isinstance(rs[j], (int, float))]
                                if valid_ratings_for_j:
                                    avg_val = sum(valid_ratings_for_j) / len(valid_ratings_for_j)
                                    averaged_ratings[j] = int(avg_val) if avg_val.is_integer() else round(avg_val, 1)
                            final_room_ratings.append(averaged_ratings)
                        else:
                            final_room_ratings.append(list_of_rating_lists[0] if list_of_rating_lists else [])
                    else:
                        final_room_ratings.append(list_of_rating_lists[0] if list_of_rating_lists else [])
                
                new_data_dict_keys['[REDACTED_BY_SCRIPT]'] = final_eval_labels
                new_data_dict_keys['room_ratings_final'] = final_room_ratings
                if list(old_eval_labels) != new_data_dict_keys['[REDACTED_BY_SCRIPT]'] or \
                   repr(old_ratings_final) != repr(new_data_dict_keys['room_ratings_final']): # More robust change check
                    changed = True
            
            else: # For Steps 3, 4, 6 (dict keyed by room labels)
                temp_other_data_accumulator = defaultdict(list)
                for old_raw_label_key, room_data_value in data.items():
                    if old_raw_label_key in ['[REDACTED_BY_SCRIPT]', 'room_ratings_final', '[REDACTED_BY_SCRIPT]']:
                        if old_raw_label_key not in new_data_dict_keys:
                             new_data_dict_keys[old_raw_label_key] = room_data_value
                        continue

                    # Use label_map_from_step1 which maps original raw Step 1 labels
                    new_label_key_mapped = label_map_from_step1.get(old_raw_label_key)
                    
                    if new_label_key_mapped: # A bedroom label was remapped
                        changed = True
                        if new_label_key_mapped == "Other Bedrooms":
                            temp_other_data_accumulator[new_label_key_mapped].append(room_data_value)
                        else: # Primary Bedroom
                            if new_label_key_mapped in new_data_dict_keys:
                                logging.warning(f"[REDACTED_BY_SCRIPT]'{new_label_key_mapped}'[REDACTED_BY_SCRIPT]")
                            new_data_dict_keys[new_label_key_mapped] = room_data_value
                    else: # Not a bedroom label that was remapped, or no mapping found
                        new_data_dict_keys[old_raw_label_key] = room_data_value
                
                if "Other Bedrooms" in temp_other_data_accumulator:
                    accumulated_list_other = temp_other_data_accumulator["Other Bedrooms"]
                    if accumulated_list_other:
                        # ... (Merging logic for "Other Bedrooms" for Steps 3, 4, 6 - keep as before) ...
                        if "step3_features" in filepath.lower() and all(isinstance(item, list) for item in accumulated_list_other):
                            all_features = set()
                            for f_list in accumulated_list_other: all_features.update(f_list)
                            new_data_dict_keys["Other Bedrooms"] = sorted(list(all_features))
                        elif ("step4_flaws" in filepath.lower() or "[REDACTED_BY_SCRIPT]" in filepath.lower()) and all(isinstance(item, dict) and "selling_points" in item for item in accumulated_list_other):
                            logging.warning(f"  Merging 'Other Bedrooms'[REDACTED_BY_SCRIPT]")
                            new_data_dict_keys["Other Bedrooms"] = accumulated_list_other[0]
                        elif "step6_renovation" in filepath.lower() and all(isinstance(item, (int,float)) for item in accumulated_list_other):
                             avg_reno = sum(accumulated_list_other) / len(accumulated_list_other)
                             new_data_dict_keys["Other Bedrooms"] = int(round(avg_reno))
                        else:
                             logging.warning(f"  Don't know how to merge 'Other Bedrooms' data for {os.path.basename(filepath)}. Taking first instance.")
                             if accumulated_list_other: new_data_dict_keys["Other Bedrooms"] = accumulated_list_other[0]
                data = new_data_dict_keys
        else:
            logging.warning(f"[REDACTED_BY_SCRIPT]")


    if changed:
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logging.info(f"[REDACTED_BY_SCRIPT]")
            return True
        except Exception as e:
            logging.error(f"[REDACTED_BY_SCRIPT]")
            return False
    else:
        logging.info(f"[REDACTED_BY_SCRIPT]")
        return True


if __name__ == "__main__":
    # ... (The main execution block remains the same as the previous version) ...
    # It will call update_step1_floorplan first, then update_subsequent_json
    # for each of the other files, passing the label_map and step1_data_content.
    logging.info(f"[REDACTED_BY_SCRIPT]")
    updated_properties_count = 0
    skipped_properties_count = 0

    address_folders = [
        f for f in os.listdir(MAIN_OUTPUT_DIR)
        if os.path.isdir(os.path.join(MAIN_OUTPUT_DIR, f))
    ]
    logging.info(f"[REDACTED_BY_SCRIPT]")

    for property_address in address_folders:
        property_output_dir = os.path.join(MAIN_OUTPUT_DIR, property_address)
        logging.info(f"[REDACTED_BY_SCRIPT]")

        year_suffix = find_latest_year_suffix(property_output_dir)
        year_suffix="_y"+year_suffix if year_suffix else None
        if not year_suffix:
            logging.warning(f"[REDACTED_BY_SCRIPT]'{property_address}'. Skipping.")
            skipped_properties_count += 1
            continue
        
        logging.info(f"[REDACTED_BY_SCRIPT]")

        step1_filename = f"[REDACTED_BY_SCRIPT]"
        step1_filepath = os.path.join(property_output_dir, step1_filename)
        
        updated_step1_data, bedroom_label_map_from_s1 = update_step1_floorplan(step1_filepath)

        if updated_step1_data is None : 
            logging.error(f"[REDACTED_BY_SCRIPT]")
            skipped_properties_count +=1
            continue
        
        # bedroom_label_map_from_s1 now maps original Step 1 raw labels to "Primary Bedroom" or "Other Bedrooms"

        # Update Step 2
        step2_filename = f"[REDACTED_BY_SCRIPT]"
        step2_filepath = os.path.join(property_output_dir, step2_filename)
        update_subsequent_json(step2_filepath, bedroom_label_map_from_s1, 
                               data_key_to_update='room_assignments', 
                               is_list_of_dicts_with_label=True,
                               step1_data_content=updated_step1_data)

        # Update Step 3
        step3_filename = f"[REDACTED_BY_SCRIPT]"
        step3_filepath = os.path.join(property_output_dir, step3_filename)
        update_subsequent_json(step3_filepath, bedroom_label_map_from_s1, step1_data_content=updated_step1_data) # Pass step1_data just in case, though not directly used by current step3 logic

        # Update Step 4
        step4_filename = f"[REDACTED_BY_SCRIPT]" # Corrected filename
        step4_filepath = os.path.join(property_output_dir, step4_filename)
        update_subsequent_json(step4_filepath, bedroom_label_map_from_s1, step1_data_content=updated_step1_data)
        
        # Update Step 5
        step5_filename = f"[REDACTED_BY_SCRIPT]"
        step5_filepath = os.path.join(property_output_dir, step5_filename)
        update_subsequent_json(step5_filepath, bedroom_label_map_from_s1, step1_data_content=updated_step1_data)

        # Update Step 6
        step6_filename = f"[REDACTED_BY_SCRIPT]"
        step6_filepath = os.path.join(property_output_dir, step6_filename)
        update_subsequent_json(step6_filepath, bedroom_label_map_from_s1, step1_data_content=updated_step1_data)

        updated_properties_count +=1

    logging.info("[REDACTED_BY_SCRIPT]")
    logging.info(f"[REDACTED_BY_SCRIPT]")
    logging.info(f"[REDACTED_BY_SCRIPT]")
    logging.info(f"[REDACTED_BY_SCRIPT]")