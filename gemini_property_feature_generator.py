import json
import csv
import os
import re
import math
from collections import defaultdict
import logging # Optional: for better logging

# --- Configuration ---
# Define standardized room tokens
ROOM_TOKENS = {
    "KITCHEN": "KITCHEN",
    "LIVING_ROOM": "LIVING_ROOM",
    "DINING_AREA": "DINING_AREA",
    "BEDROOM": "BEDROOM", # This will be the token for "Primary Bedroom" and "Other Bedrooms" too
    "BATHROOM": "BATHROOM",
    "HALLWAY_LANDING": "HALLWAY_LANDING",
    "UTILITY_ROOM": "UTILITY_ROOM",
    "GARAGE": "GARAGE",
    "CONSERVATORY_SUNROOM": "CONSERVATORY_SUNROOM",
    "OFFICE_STUDY": "OFFICE_STUDY",
    "GARDEN_YARD": "GARDEN_YARD",
    "PATIO_DECKING": "PATIO_DECKING",
    "DRIVEWAY": "DRIVEWAY",
    "EXTERIOR_FRONT": "EXTERIOR_FRONT",
    "EXTERIOR_REAR": "EXTERIOR_REAR",
    "AERIAL_VIEW": "AERIAL_VIEW",
    "VIEW_FROM_PROPERTY": "VIEW_FROM_PROPERTY",
    "OUTBUILDING_SHED": "OUTBUILDING_SHED",
    "STORAGE_AREA": "STORAGE_AREA",
    "DETAIL_SHOT": "DETAIL_SHOT",
    "OTHER_INDOOR_SPACE": "OTHER_INDOOR_SPACE",
    "FLOORPLAN": "FLOORPLAN",
    "SITE_PLAN": "SITE_PLAN",
    "STORM_PORCH": "STORM_PORCH",
    "EAVES": "EAVES"
}
UNKNOWN_TOKEN = "UNKNOWN"

# --- Room Categorization ---
MAJOR_ROOMS_PRIMARY_BED = {"BEDROOM"} # Primary bedroom will be identified from this token
MAJOR_ROOMS_MULTI_INSTANCE = {
    "KITCHEN",
    "LIVING_ROOM",
    "DINING_AREA",
    "BATHROOM",
    "OFFICE_STUDY"
}
MAJOR_ROOMS_SINGLE_INSTANCE = {
    "CONSERVATORY_SUNROOM"
}
EXCLUDED_TOKENS = {
    "FLOORPLAN", "SITE_PLAN", "DETAIL_SHOT", "AERIAL_VIEW",
    "EXTERIOR_FRONT", "EXTERIOR_REAR", "DRIVEWAY",
    "VIEW_FROM_PROPERTY", "UNKNOWN"
}

# --- High-Signal Token Definition ---
PROPERTY_LEVEL_TOKENS = {
    "EXT_MATERIAL_BRICK", "EXT_MATERIAL_STONE", "EXT_MATERIAL_RENDERED",
    "[REDACTED_BY_SCRIPT]", "PROP_TYPE_TERRACED", "[REDACTED_BY_SCRIPT]",
    "PROP_TYPE_DETACHED", "PROP_TYPE_BUNGALOW", "PROP_SIZE_TWO_STORY",
    "[REDACTED_BY_SCRIPT]", "CONDITION_GOOD", "CONDITION_NEEDS_MAINTENANCE",
    "CONDITION_NEEDS_UPDATE", "CONDITION_UNDER_CONSTRUCTION", "[REDACTED_BY_SCRIPT]",
    "CONDITION_POOR", "PARKING_DRIVEWAY", "PARKING_GARAGE", "[REDACTED_BY_SCRIPT]",
    "PARKING_GARAGE_ATTACHED", "[REDACTED_BY_SCRIPT]", "LAYOUT_OPEN_PLAN",
    "[REDACTED_BY_SCRIPT]", "GARDEN_LAWN", "GARDEN_PATIO", "GARDEN_DECKING",
    "GARDEN_FENCED", "GARDEN_SHED", "GARDEN_OUTBUILDING", "[REDACTED_BY_SCRIPT]",
    "GARDEN_MATURE_TREES", "GARDEN_RAISED_BEDS", "[REDACTED_BY_SCRIPT]",
    "GARDEN_PERGOLA", "[REDACTED_BY_SCRIPT]"
}
HIGH_SIGNAL_ROOM_FLAWS = {
    "FLAW_CONDITION", "FLAW_NEEDS_UPDATE", "FLAW_LAYOUT", "[REDACTED_BY_SCRIPT]",
    "FLAW_DATED", "FLAW_POOR_FINISH", "FLAW_BASIC_STYLE", "FLAW_STORAGE",
    "FLAW_LIGHT", "FLAW_UNATTRACTIVE", "FLAW_SPACE", "FLAW_FUNCTIONAL", # Renamed from FLAW_FUNCTIONAL in original
    "FLAW_NOISE", "FLAW_ACCESSIBILITY", "FLAW_PRIVACY" # Added FLAW_PRIVACY
}
HIGH_SIGNAL_ROOM_SPS = {
    "SP_CHARACTER", "SP_STYLE", "SP_FEATURE", "SP_POTENTIAL", "SP_PRIVACY",
    "SP_CONDITION", "SP_LIGHT", "SP_SPACE", "SP_FUNCTIONAL", "SP_LAYOUT", # Added SP_LAYOUT
    "SP_GARDEN_ACCESS", "SP_MODERN", "SP_STORAGE", "SP_LOW_MAINTENANCE",
    "SP_QUALITY_FINISH"
}
ALL_HIGH_SIGNAL_ROOM_TOKENS = HIGH_SIGNAL_ROOM_FLAWS.union(HIGH_SIGNAL_ROOM_SPS)

# --- Helper Functions ---
def get_room_token(label_text):
    """[REDACTED_BY_SCRIPT]"""
    if not label_text or not isinstance(label_text, str):
        return UNKNOWN_TOKEN
    text = label_text.lower().strip().replace("_", " ")

    # Handle "Primary Bedroom" and "Other Bedrooms" first
    if "primary bedroom" in text: return ROOM_TOKENS["BEDROOM"]
    if "other bedrooms" in text: return ROOM_TOKENS["BEDROOM"] # Treat as general bedroom for initial categorization

    # Original mappings
    if "kitchen" in text: return ROOM_TOKENS["KITCHEN"]
    if "lounge" in text or "living room" in text or "sitting room" in text or "reception room" in text or "play room" in text: return ROOM_TOKENS["LIVING_ROOM"]
    if "dining" in text: return ROOM_TOKENS["DINING_AREA"]
    if "bedroom" in text: return ROOM_TOKENS["BEDROOM"] # General catch-all if not primary/other
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

    logging.warning(f"[REDACTED_BY_SCRIPT]'{label_text}'[REDACTED_BY_SCRIPT]")
    return UNKNOWN_TOKEN

def calculate_area(dimensions_str, original_label=""): # Added original_label for context
    """[REDACTED_BY_SCRIPT]"""
    if not dimensions_str or not isinstance(dimensions_str, str):
        return None

    # Special handling for "Other Bedrooms" if its dimensions string is the descriptive one
    # e.g., "[REDACTED_BY_SCRIPT]"
    if original_label.lower() == "other bedrooms":
        other_match = re.search(r'[REDACTED_BY_SCRIPT]', dimensions_str, re.IGNORECASE)
        if other_match:
            try:
                num_rooms = int(other_match.group(1))
                avg_area = float(other_match.group(2))
                # Return an *estimated total area* for the group
                return num_rooms * avg_area
            except ValueError:
                logging.warning(f"Could not parse 'Other Bedrooms'[REDACTED_BY_SCRIPT]")
                return None

    # Standard dimension parsing: "W m x D m"
    match = re.search(r'[REDACTED_BY_SCRIPT]', dimensions_str)
    if match:
        try:
            dim1 = float(match.group(1))
            dim2 = float(match.group(2))
            return dim1 * dim2
        except ValueError:
            logging.warning(f"[REDACTED_BY_SCRIPT]")
            return None
    else:
        logging.info(f"[REDACTED_BY_SCRIPT]'{dimensions_str}'[REDACTED_BY_SCRIPT]")
        return None

def safe_average(num_list):
    """[REDACTED_BY_SCRIPT]"""
    if not num_list: return 0.0 # Return 0.0 or None, depending on desired CSV output for missing data
    return sum(num_list) / len(num_list)

def process_property(property_id, input_dir=".", year_suffix=""):
    """[REDACTED_BY_SCRIPT]"""
    try:
        floorplan_data = None
        step1_path = os.path.join(input_dir, f'[REDACTED_BY_SCRIPT]')
        if os.path.exists(step1_path):
            try:
                with open(step1_path, 'r', encoding='utf-8') as f:
                    floorplan_data = json.load(f)
            except Exception as e:
                logging.warning(f"[REDACTED_BY_SCRIPT]")
        else:
            logging.info(f"[REDACTED_BY_SCRIPT]")

        step4_path = os.path.join(input_dir, f'[REDACTED_BY_SCRIPT]')
        step5_path = os.path.join(input_dir, f'[REDACTED_BY_SCRIPT]')

        with open(step4_path, 'r', encoding='utf-8') as f: flaws_sp_data = json.load(f)
        with open(step5_path, 'r', encoding='utf-8') as f: ratings_data = json.load(f)

    except FileNotFoundError as e:
        logging.error(f"[REDACTED_BY_SCRIPT]'_y{year_suffix}': {e}")
        return None
    except json.JSONDecodeError as e:
        logging.error(f"[REDACTED_BY_SCRIPT]'_y{year_suffix}': {e}")
        return None
    except Exception as e:
        logging.error(f"[REDACTED_BY_SCRIPT]'_y{year_suffix}': {e}")
        return None

    temp_room_storage = defaultdict(lambda: {'std_token': UNKNOWN_TOKEN, 'area': None, 'rating': None, 'tags': set(), 'num_other_rooms_in_group': 0})
    all_property_tags = set()

    # Process Floorplan (Areas) from Step 1 output
    # This is where "Primary Bedroom" and "Other Bedrooms" from your main script's modification will be read
    if floorplan_data:
        for room in floorplan_data.get('rooms_with_dimensions', []):
            label = room.get('label')
            if label:
                std_token = get_room_token(label) # This will map "[REDACTED_BY_SCRIPT]" to "BEDROOM"
                temp_room_storage[label]['std_token'] = std_token
                temp_room_storage[label]['area'] = calculate_area(room.get('dimensions'), label) # Pass label for context
                # If it's the "Other Bedrooms" group from floorplan, store its count
                if label.lower() == "other bedrooms":
                    other_match = re.search(r'(\d+)\s*room\(s\)', room.get('dimensions', ""), re.IGNORECASE)
                    if other_match:
                        temp_room_storage[label]['num_other_rooms_in_group'] = int(other_match.group(1))


        for label in floorplan_data.get('[REDACTED_BY_SCRIPT]', []):
            if label and label not in temp_room_storage:
                temp_room_storage[label]['std_token'] = get_room_token(label)

    # Process Ratings (Step 5) - These labels should align with floorplan if Step 2 worked
    evaluated_labels = ratings_data.get('[REDACTED_BY_SCRIPT]', [])
    final_ratings = ratings_data.get('room_ratings_final', []) # This is your list of lists

    if len(evaluated_labels) == len(final_ratings):
        for i, label in enumerate(evaluated_labels):
            if not label: continue # Skip if label is empty
            std_token = get_room_token(label) # Get token for consistent storage
            # Ensure entry exists from floorplan or create one
            if label not in temp_room_storage:
                temp_room_storage[label]['std_token'] = std_token
            
            # Ensure the sublist final_ratings[i] is long enough before slicing
            if final_ratings[i] and len(final_ratings[i]) >= 21:
                # Slice to get persona ratings (indices 1 through 20)
                persona_ratings_slice = final_ratings[i][1:21] 
                ratings_list_for_avg = [r for r in persona_ratings_slice if isinstance(r, (int, float))]
                temp_room_storage[label]['rating'] = safe_average(ratings_list_for_avg)
            else:
                # Handle cases where the ratings list for a room is unexpectedly short or missing
                temp_room_storage[label]['rating'] = None # Or some other default like 0.0
                logging.warning(f"Room '{label}'[REDACTED_BY_SCRIPT]")
    else:
        logging.warning(f"[REDACTED_BY_SCRIPT]")

    # Process Flaws/SPs (Step 4)
    for label, data in flaws_sp_data.items():
        if not label: continue
        room_tags = set()
        for sp in data.get('selling_points', []): room_tags.update(sp.get('tags', []))
        for flaw in data.get('flaws', []): room_tags.update(flaw.get('tags', []))

        std_token = get_room_token(label)
        if label not in temp_room_storage: # Ensure entry exists
            temp_room_storage[label]['std_token'] = std_token
        temp_room_storage[label]['tags'].update(room_tags)
        all_property_tags.update(room_tags)

    # --- Convert to processed_rooms list ---
    processed_rooms = []
    for original_label, data in temp_room_storage.items():
        if data['std_token'] not in EXCLUDED_TOKENS:
            processed_rooms.append({
                'original_label': original_label, # Keep original label for debugging/reference
                'std_token': data['std_token'],
                'area': data['area'],
                'rating': data['rating'],
                'tags': data['tags'],
                'num_other_rooms_in_group': data.get('num_other_rooms_in_group', 0) # For "Other Bedrooms"
            })

    features = {'property_id': property_id}
    for prop_token in PROPERTY_LEVEL_TOKENS:
        features[f'Prop_{prop_token}'] = 1 if prop_token in all_property_tags else 0

    # --- Process Rooms by Category ---
    primary_bedroom_data = None
    other_bedrooms_group_data = None # This will hold the single "Other Bedrooms" entry from floorplan
    individual_other_bedrooms_from_ratings = [] # If "Other Bedrooms" wasn't in floorplan but individual ones were in ratings
    
    multi_instance_data = defaultdict(list)
    single_instance_data = defaultdict(list)
    misc_data = []

    # Categorize rooms from the `processed_rooms` list
    # This list now contains entries like "Primary Bedroom" and "Other Bedrooms" if they came from step1.
    for room in processed_rooms:
        original_label = room['original_label'].lower()
        std_token = room['std_token'] # Should be "BEDROOM" for both primary and other

        if original_label == "primary bedroom":
            primary_bedroom_data = room
            continue
        elif original_label == "other bedrooms":
            other_bedrooms_group_data = room
            continue
        
        # If not explicitly "Primary" or "Other Bedrooms" group, categorize based on std_token
        if std_token == 'BEDROOM':
            # These would be bedrooms that weren't part of the primary/other split,
            # or if primary/other labels weren't present in step1 output.
            # We'[REDACTED_BY_SCRIPT]"other" group.
            individual_other_bedrooms_from_ratings.append(room)
        elif std_token in MAJOR_ROOMS_MULTI_INSTANCE:
            multi_instance_data[std_token].append(room)
        elif std_token in MAJOR_ROOMS_SINGLE_INSTANCE:
            single_instance_data[std_token].append(room)
        else:
            misc_data.append(room)

    # Finalize "Other Bedrooms" features
    num_other_bedrooms_final = 0
    other_bedrooms_total_area_final = 0.0
    other_bedrooms_avg_area_final = 0.0
    other_ratings_final = []
    other_tags_flat_final = []

    if other_bedrooms_group_data:
        num_other_bedrooms_final = other_bedrooms_group_data.get('num_other_rooms_in_group', 0)
        other_bedrooms_total_area_final = other_bedrooms_group_data.get('area', 0.0) or 0.0 # Area is already sum
        if num_other_bedrooms_final > 0:
             other_bedrooms_avg_area_final = other_bedrooms_total_area_final / num_other_bedrooms_final
        if other_bedrooms_group_data.get('rating') is not None:
             other_ratings_final.append(other_bedrooms_group_data['rating'])
        other_tags_flat_final.extend(list(other_bedrooms_group_data.get('tags', set())))
    elif individual_other_bedrooms_from_ratings:
        # Fallback if "Other Bedrooms" group wasn't in floorplan data,
        # but individual bedrooms were found (e.g., from ratings directly)
        logging.info(f"[REDACTED_BY_SCRIPT]'Other Bedrooms'[REDACTED_BY_SCRIPT]")
        num_other_bedrooms_final = len(individual_other_bedrooms_from_ratings)
        other_areas = [d['area'] for d in individual_other_bedrooms_from_ratings if d['area'] is not None]
        other_bedrooms_total_area_final = sum(other_areas) if other_areas else 0.0
        other_bedrooms_avg_area_final = safe_average(other_areas)
        other_ratings_final = [d['rating'] for d in individual_other_bedrooms_from_ratings if d['rating'] is not None]
        other_tags_flat_final = [tag for d in individual_other_bedrooms_from_ratings for tag in d['tags']]


    # --- Calculate Features for Each Category ---
    features['PrimaryBedroom_Area'] = primary_bedroom_data['area'] if primary_bedroom_data and primary_bedroom_data['area'] is not None else None
    features['[REDACTED_BY_SCRIPT]'] = primary_bedroom_data['rating'] if primary_bedroom_data and primary_bedroom_data['rating'] is not None else None
    primary_tags = primary_bedroom_data['tags'] if primary_bedroom_data else set()
    for flaw in HIGH_SIGNAL_ROOM_FLAWS: features[f'[REDACTED_BY_SCRIPT]'] = 1 if flaw in primary_tags else 0
    for sp in HIGH_SIGNAL_ROOM_SPS: features[f'[REDACTED_BY_SCRIPT]'] = 1 if sp in primary_tags else 0

    # Other Bedrooms Features (using finalized values)
    features['Num_Other_Bedrooms'] = num_other_bedrooms_final
    features['[REDACTED_BY_SCRIPT]'] = other_bedrooms_total_area_final
    features['[REDACTED_BY_SCRIPT]'] = other_bedrooms_avg_area_final
    features['[REDACTED_BY_SCRIPT]'] = safe_average(other_ratings_final)
    features['[REDACTED_BY_SCRIPT]'] = sum(1 for tag in other_tags_flat_final if tag in HIGH_SIGNAL_ROOM_FLAWS)
    features['[REDACTED_BY_SCRIPT]'] = sum(1 for tag in other_tags_flat_final if tag in HIGH_SIGNAL_ROOM_SPS)


    # Multi-Instance Major Rooms Features
    for room_type_token_val in MAJOR_ROOMS_MULTI_INSTANCE: # Use the values like "KITCHEN"
        data_list = multi_instance_data[room_type_token_val]
        features[f'[REDACTED_BY_SCRIPT]'] = len(data_list)
        areas = [d['area'] for d in data_list if d['area'] is not None]
        ratings = [d['rating'] for d in data_list if d['rating'] is not None]
        tags_flat = [tag for d in data_list for tag in d['tags']]
        features[f'[REDACTED_BY_SCRIPT]'] = sum(areas) if areas else 0.0
        features[f'[REDACTED_BY_SCRIPT]'] = safe_average(areas)
        features[f'[REDACTED_BY_SCRIPT]'] = safe_average(ratings)
        features[f'[REDACTED_BY_SCRIPT]'] = sum(1 for tag in tags_flat if tag in HIGH_SIGNAL_ROOM_FLAWS)
        features[f'[REDACTED_BY_SCRIPT]'] = sum(1 for tag in tags_flat if tag in HIGH_SIGNAL_ROOM_SPS)

    # Single-Instance Major Rooms Features
    for room_type_token_val in MAJOR_ROOMS_SINGLE_INSTANCE:
        data_list = single_instance_data[room_type_token_val]
        data = data_list[0] if data_list else None
        features[f'[REDACTED_BY_SCRIPT]'] = 1 if data else 0
        features[f'[REDACTED_BY_SCRIPT]'] = data['area'] if data and data['area'] is not None else None
        features[f'[REDACTED_BY_SCRIPT]'] = data['rating'] if data and data['rating'] is not None else None
        tags = data['tags'] if data else set()
        for flaw in HIGH_SIGNAL_ROOM_FLAWS: features[f'[REDACTED_BY_SCRIPT]'] = 1 if flaw in tags else 0
        for sp in HIGH_SIGNAL_ROOM_SPS: features[f'[REDACTED_BY_SCRIPT]'] = 1 if sp in tags else 0

    # Misc Rooms Features
    features['Misc_Room_Count'] = len(misc_data)
    misc_areas = [d['area'] for d in misc_data if d['area'] is not None]
    misc_ratings = [d['rating'] for d in misc_data if d['rating'] is not None]
    misc_tags_flat = [tag for d in misc_data for tag in d['tags']]
    features['Misc_Total_Area'] = sum(misc_areas) if misc_areas else 0.0
    features['Misc_Average_Rating'] = safe_average(misc_ratings)
    features['[REDACTED_BY_SCRIPT]'] = sum(1 for tag in misc_tags_flat if tag in HIGH_SIGNAL_ROOM_FLAWS)
    features['[REDACTED_BY_SCRIPT]'] = sum(1 for tag in misc_tags_flat if tag in HIGH_SIGNAL_ROOM_SPS)

    return features

def get_feature_header():
    """[REDACTED_BY_SCRIPT]"""
    header = ['property_id']
    header.extend(sorted([f'Prop_{token}' for token in PROPERTY_LEVEL_TOKENS]))
    # Primary Bedroom
    header.extend(['PrimaryBedroom_Area', '[REDACTED_BY_SCRIPT]'])
    header.extend(sorted([f'[REDACTED_BY_SCRIPT]' for token in HIGH_SIGNAL_ROOM_FLAWS]))
    header.extend(sorted([f'[REDACTED_BY_SCRIPT]' for token in HIGH_SIGNAL_ROOM_SPS]))
    # Other Bedrooms
    header.extend(['Num_Other_Bedrooms', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
                   '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
                   '[REDACTED_BY_SCRIPT]'])
    # Multi-Instance Majors (use the values of ROOM_TOKENS for consistency if MAJOR_ROOMS_MULTI_INSTANCE stores keys)
    multi_instance_keys_sorted = sorted(list(MAJOR_ROOMS_MULTI_INSTANCE))
    for room_type_key in multi_instance_keys_sorted:
        # room_type_val = ROOM_TOKENS.get(room_type_key, room_type_key) # Get actual token value
        room_type_val = room_type_key # Assuming MAJOR_ROOMS_MULTI_INSTANCE stores the direct token value like "KITCHEN"
        header.extend([f'[REDACTED_BY_SCRIPT]', f'[REDACTED_BY_SCRIPT]', f'[REDACTED_BY_SCRIPT]',
                       f'[REDACTED_BY_SCRIPT]', f'[REDACTED_BY_SCRIPT]',
                       f'[REDACTED_BY_SCRIPT]'])
    # Single-Instance Majors
    single_instance_keys_sorted = sorted(list(MAJOR_ROOMS_SINGLE_INSTANCE))
    for room_type_key in single_instance_keys_sorted:
        # room_type_val = ROOM_TOKENS.get(room_type_key, room_type_key)
        room_type_val = room_type_key
        header.extend([f'Has_{room_type_val}', f'[REDACTED_BY_SCRIPT]', f'[REDACTED_BY_SCRIPT]'])
        header.extend(sorted([f'[REDACTED_BY_SCRIPT]' for token in HIGH_SIGNAL_ROOM_FLAWS]))
        header.extend(sorted([f'[REDACTED_BY_SCRIPT]' for token in HIGH_SIGNAL_ROOM_SPS]))
    # Misc Rooms
    header.extend(['Misc_Room_Count', 'Misc_Total_Area', 'Misc_Average_Rating',
                   '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]'])
    return header

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    PROPERTY_ID_EXAMPLE = "[REDACTED_BY_SCRIPT]" # Example
    INPUT_DIRECTORY_EXAMPLE = r"[REDACTED_BY_SCRIPT]" # Example
    INPUT_YEAR_EXAMPLE = "2024" # Example
    OUTPUT_CSV_FILE_EXAMPLE = "[REDACTED_BY_SCRIPT]"

    logging.info(f"[REDACTED_BY_SCRIPT]")
    property_features = process_property(f"[REDACTED_BY_SCRIPT]", 
                                         INPUT_DIRECTORY_EXAMPLE, 
                                         INPUT_YEAR_EXAMPLE)

    if property_features:
        header = get_feature_header()
        # Safety check for keys not in header
        for key_gen in property_features.keys():
            if key_gen not in header:
                logging.warning(f"Feature '{key_gen}'[REDACTED_BY_SCRIPT]")
                header.append(key_gen)

        file_exists = os.path.isfile(OUTPUT_CSV_FILE_EXAMPLE)
        try:
            with open(OUTPUT_CSV_FILE_EXAMPLE, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=header, extrasaction='ignore')
                if not file_exists:
                    writer.writeheader()
                writer.writerow(property_features)
            logging.info(f"[REDACTED_BY_SCRIPT]")
        except IOError as e:
            logging.error(f"[REDACTED_BY_SCRIPT]")
    else:
        logging.error(f"[REDACTED_BY_SCRIPT]")