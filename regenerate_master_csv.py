import os
import json
import re
import pandas as pd
import numpy as np
# from sentence_transformers import SentenceTransformer # Embeddings commented out
# from sklearn.metrics.pairwise import cosine_similarity 
from collections import defaultdict

# --- CONFIGURATION ---
BASE_PROPERTY_DATA_DIR = r"[REDACTED_BY_SCRIPT]" 
OUTPUT_CSV_FILE = r"[REDACTED_BY_SCRIPT]"

POSSIBLE_YEARS_FALLBACK = ["2025", "2024", "2023", "2022"] 

PRIMARY_ROOM_HEURISTIC_WEIGHT_IMAGES = 0.6
PRIMARY_ROOM_HEURISTIC_WEIGHT_SPS = 0.4

# --- EMBEDDINGS CONFIG (COMMENTED OUT) ---
# EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
# print(f"[REDACTED_BY_SCRIPT]")
# MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)
# EMBEDDING_DIM = MODEL.get_sentence_embedding_dimension()
# print(f"[REDACTED_BY_SCRIPT]")
EMBEDDING_DIM = 384 # Placeholder if MODEL is not loaded, for key generation if uncommented

# --- DYNAMIC MASTER TOKEN SETS ---
MASTER_ALL_ROOM_TYPE_TOKENS = set() # Still useful for has_room_type_X features
MASTER_ALL_FEATURE_TOKENS = set()
MASTER_ALL_SP_THEME_TOKENS = set()
MASTER_ALL_FLAW_THEME_TOKENS = set()
ALL_GENERATED_FEATURE_KEYS = set()

# --- CSV SAVING ---
SAVE_CHUNK_SIZE = 100 
HEADER_WRITTEN = False

# --- CANONICAL MAPPING ---
GENERIC_MAPPING_ORDERED = [
    ("KITCHEN", ("kitchen",)),
    ("LIVING_AREA", ("living room", "lounge", "sitting room", "reception room", "family room", "snug", "drawing room")),
    ("DINING_AREA", ("dining room", "dining area", "breakfast room")),
    ("BEDROOM", ("bedroom", "bed room")), 
    ("BATHROOM_WC", ("bathroom", "shower room", "en-suite", "ensuite", "en suite", "wc", "toilet", "cloakroom", "washroom")),
    ("[REDACTED_BY_SCRIPT]", ("hall", "hallway", "landing", "entrance", "stairs", "staircase", "inner hall", "reception hall")),
    ("UTILITY_ROOM", ("utility", "laundry")),
    ("GARAGE", ("garage",)),
    ("CONSERVATORY_SUNROOM", ("conservatory", "sun room", "orangery", "garden room")),
    ("OFFICE_STUDY", ("office", "study", "workroom")),
    ("GARDEN_YARD", ("garden", "yard", "grounds", "rear garden", "front garden", "side garden")),
    ("[REDACTED_BY_SCRIPT]", ("patio", "terrace", "decking", "balcony")),
    ("DRIVEWAY_PARKING", ("driveway", "drive", "parking")),
    ("EXTERIOR_FRONT", ("front exterior", "front elevation", "external front", "front of house")),
    ("EXTERIOR_REAR", ("rear exterior", "rear elevation", "external rear", "rear of house")),
    ("EXTERIOR_SIDE", ("side exterior", "side elevation", "external side")),
    ("OUTBUILDING", ("outbuilding", "shed", "workshop", "store", "summer house", "annex")),
    ("STORAGE", ("storage", "store room", "cupboard", "wardrobe", "airing cupboard", "eaves", "loft", "attic", "cellar", "basement")),
    ("PORCH", ("porch", "storm porch")),
    ("MISC_INDOOR", ("games room", "playroom", "cinema room", "gym", "mezzanine", "boot room")),
    ("AERIAL_VIEW", ("aerial view", "drone shot")),
    ("FLOORPLAN", ("floorplan", "floor plan")),
    ("SITE_PLAN", ("site plan",)),
    ("DETAIL_SHOT", ("detail", "close up", "feature")),
    ("VIEW_FROM_PROPERTY", ("view from property", "outlook", "view")),
    ("COMMUNAL_AREA", ("communal",))
]
# These are the roots for the final DataFrame column prefixes
CANONICAL_PREFIX_ROOT_MAP = {
    "KITCHEN": "MainKitchen",
    "LIVING_AREA": "MainLivingArea",
    "DINING_AREA": "MainDiningArea",
    "BEDROOM": "PrimaryBedroom", # Special handling: one "PrimaryBedroom", rest go to "OtherBedrooms"
    "BATHROOM_WC": "MainBathroom", # Default, refined below
    "[REDACTED_BY_SCRIPT]": "[REDACTED_BY_SCRIPT]",
    "UTILITY_ROOM": "MainUtilityRoom",
    "GARAGE": "MainGarage",
    "CONSERVATORY_SUNROOM": "[REDACTED_BY_SCRIPT]",
    "OFFICE_STUDY": "StudyOffice",
    "GARDEN_YARD": "MainGarden",
    "[REDACTED_BY_SCRIPT]":"[REDACTED_BY_SCRIPT]",
    "DRIVEWAY_PARKING": "MainDrivewayParking", # Or consider not making this a "room" with all features
    "EXTERIOR_FRONT": "MainExteriorFront",
    "EXTERIOR_REAR": "MainExteriorRear",
    "EXTERIOR_SIDE": "MainExteriorSide",
    "OUTBUILDING": "MainOutbuilding",
    "STORAGE": "[REDACTED_BY_SCRIPT]", # Grouping various storage concepts
    "PORCH": "MainPorch",
    "MISC_INDOOR": "MainMiscIndoor",
    # Roots for "Other" aggregated features
    "KITCHEN_OTHER": "OtherKitchens",
    "LIVING_AREA_OTHER": "OtherLivingAreas",
    "DINING_AREA_OTHER": "OtherDiningAreas",
    "BEDROOM_OTHER": "OtherBedrooms",
    "BATHROOM_WC_OTHER": "OtherBathroomsWCs",
    "GARDEN_YARD_OTHER": "OtherGardens",
    # If a generic type doesn'[REDACTED_BY_SCRIPT]"Misc"
}
# Define which generic types typically only have one "Main" instance (no "Other" category needed for them)
SINGLE_INSTANCE_CANONICAL_TYPES = {
    "[REDACTED_BY_SCRIPT]", "MainUtilityRoom", "StudyOffice", "MainGarage",
    "[REDACTED_BY_SCRIPT]", "MainExteriorFront", "MainExteriorRear", "MainExteriorSide",
    "MainDrivewayParking", "MainPorch", "MainMiscIndoor", "[REDACTED_BY_SCRIPT]", "MainOutbuilding",
    "[REDACTED_BY_SCRIPT]"
}


# --- HELPER FUNCTIONS (get_property_specific_year_suffix, find_json_file_for_property, load_json_safe, parse_dimensions_to_area, normalize_room_label_for_feature_name, get_text_embeddings - slightly adapted or embeddings part commented out) ---

def get_property_specific_year_suffix(property_path):
    files_to_check_for_year = ["output_step5_merged", "[REDACTED_BY_SCRIPT]"]
    for year_val_str in POSSIBLE_YEARS_FALLBACK:
        for prefix in files_to_check_for_year:
            path_y_y = os.path.join(property_path, f"[REDACTED_BY_SCRIPT]")
            if os.path.exists(path_y_y): return f"_y_y{year_val_str}"
            path_y = os.path.join(property_path, f"[REDACTED_BY_SCRIPT]")
            if os.path.exists(path_y): return f"_y{year_val_str}"
    return None

def find_json_file_for_property(base_path, file_prefix_no_year, property_year_suffix, file_suffix=".json"):
    filenames_to_try = []
    if property_year_suffix: filenames_to_try.append(f"[REDACTED_BY_SCRIPT]")
    for year_val_str in POSSIBLE_YEARS_FALLBACK:
        filenames_to_try.append(f"[REDACTED_BY_SCRIPT]")
        filenames_to_try.append(f"[REDACTED_BY_SCRIPT]")
    filenames_to_try.append(f"[REDACTED_BY_SCRIPT]")
    for filename in set(filenames_to_try):
        filepath = os.path.join(base_path, filename)
        if os.path.exists(filepath): return filepath
    return None

def load_json_safe(filepath, silent_if_not_found=True):
    if not filepath or not os.path.exists(filepath):
        if not silent_if_not_found: print(f"[REDACTED_BY_SCRIPT]")
        return None
    try:
        with open(filepath, 'r', encoding='utf-8') as f: return json.load(f)
    except Exception: # More general exception for JSON loading
        if not silent_if_not_found: print(f"[REDACTED_BY_SCRIPT]")
        return None

def parse_dimensions_to_area(dim_str): # Kept as before
    if not dim_str or str(dim_str).lower() == "null" or str(dim_str).strip() == "": return None
    dim_str = str(dim_str)
    try:
        matches = re.findall(r"([\d\.]+)\s*(m|ft|['\"[REDACTED_BY_SCRIPT]'\"])?", dim_str)
        if matches:
            d1_val_str, d1_unit_str, d2_val_str, d2_unit_str = matches[0]
            d1, d2 = float(d1_val_str), float(d2_val_str)
            if d1_unit_str and (d1_unit_str.lower() == "ft" or d1_unit_str in ["'", '"']): d1 *= 0.3048
            if d2_unit_str and (d2_unit_str.lower() == "ft" or d2_unit_str in ["'", '"']): d2 *= 0.3048
            elif d1_unit_str and (d1_unit_str.lower() == "ft" or d1_unit_str in ["'", '"']) and not d2_unit_str: d2 *=0.3048
            elif d2_unit_str and (d2_unit_str.lower() == "ft" or d2_unit_str in ["'", '"']) and not d1_unit_str: d1 *=0.3048
            return round(d1 * d2, 2)
        area_match = re.search(r"[REDACTED_BY_SCRIPT]", dim_str, re.IGNORECASE)
        if area_match:
            val, unit = float(area_match.group(1)), area_match.group(2).lower()
            if "ft" in unit: return round(val * 0.092903, 2)
            return round(val, 2)
    except Exception: pass
    return None

def normalize_room_label_for_feature_name(label): # Kept as before
    if not label: return "UnknownRoom"
    label = re.sub(r'[^\w\s-]', '', label) 
    return "".join(word.capitalize() for word in re.split(r"[\s-]+", label.strip()))

# def get_text_embeddings(text_list, model): # Embeddings commented out
#     if not text_list: return np.zeros(model.get_sentence_embedding_dimension())
#     processed_texts = [str(t).strip() for t in text_list if t and str(t).strip()]
#     if not processed_texts: return np.zeros(model.get_sentence_embedding_dimension())
#     full_text = ". ".join(processed_texts)
#     return model.encode(full_text)

def map_eval_label_to_generic_canonical(eval_label_from_step5):
    if not eval_label_from_step5 or not isinstance(eval_label_from_step5, str):
        return "UNKNOWN_ROOM_TYPE"
    text_lower = eval_label_from_step5.lower().strip()
    if "kitchen/diner" in text_lower or "kitchen diner" in text_lower: return "KITCHEN"
    if "reception hall" in text_lower: return "[REDACTED_BY_SCRIPT]"
    if "master bedroom" in text_lower or "principal bedroom" in text_lower : return "BEDROOM"
    if "bedroom 1" in text_lower or "bedroom one" in text_lower: return "BEDROOM"
    for generic_name, keywords_tuple in GENERIC_MAPPING_ORDERED:
        for keyword in keywords_tuple:
            if keyword in ["hall", "wc", "view", "store", "drive", "gym", "loft", "attic", "eaves", "cellar", "basement"]: 
                if re.search(r'\b' + re.escape(keyword) + r'\b', text_lower): return generic_name
            elif keyword in text_lower: return generic_name
    # print(f"[REDACTED_BY_SCRIPT]'{eval_label_from_step5}'[REDACTED_BY_SCRIPT]")
    return "UNKNOWN_ROOM_TYPE"

# --- MAIN PROCESSING ---
all_properties_features_list_of_dicts = []
property_id_counter = 0

print(f"[REDACTED_BY_SCRIPT]")
property_folder_names = os.listdir(BASE_PROPERTY_DATA_DIR) # Get list once for last item check

for idx, property_folder_name in enumerate(property_folder_names):
    property_path = os.path.join(BASE_PROPERTY_DATA_DIR, property_folder_name)
    if not os.path.isdir(property_path): continue

    property_id_counter += 1
    current_property_features = defaultdict(lambda: 0) 
    current_property_features["property_id"] = property_folder_name
    print(f"[REDACTED_BY_SCRIPT]")

    prop_year_suffix = get_property_specific_year_suffix(property_path)
    
    token_summary_path = find_json_file_for_property(property_path, "[REDACTED_BY_SCRIPT]", prop_year_suffix)
    property_token_summary = load_json_safe(token_summary_path)
    if property_token_summary:
        MASTER_ALL_ROOM_TYPE_TOKENS.update(property_token_summary.get("room_labels", []))
        MASTER_ALL_FEATURE_TOKENS.update(property_token_summary.get("features", []))
        MASTER_ALL_SP_THEME_TOKENS.update(property_token_summary.get("sp_themes", []))
        MASTER_ALL_FLAW_THEME_TOKENS.update(property_token_summary.get("flaw_themes", []))

    image_paths_map_data = load_json_safe(find_json_file_for_property(property_path, "image_paths_map", prop_year_suffix))
    step1_data = load_json_safe(find_json_file_for_property(property_path, "[REDACTED_BY_SCRIPT]", prop_year_suffix))
    step2_data = load_json_safe(find_json_file_for_property(property_path, "[REDACTED_BY_SCRIPT]", prop_year_suffix))
    step3_data = load_json_safe(find_json_file_for_property(property_path, "output_step3_features", prop_year_suffix))
    step4_data_prop = load_json_safe(find_json_file_for_property(property_path, "[REDACTED_BY_SCRIPT]", prop_year_suffix))
    step5_data = load_json_safe(find_json_file_for_property(property_path, "output_step5_merged", prop_year_suffix))
    step6_data = load_json_safe(find_json_file_for_property(property_path, "[REDACTED_BY_SCRIPT]", prop_year_suffix))
    untagged_text_data = load_json_safe(find_json_file_for_property(property_path, "[REDACTED_BY_SCRIPT]", prop_year_suffix))

    if not step5_data:
        print(f"[REDACTED_BY_SCRIPT]")
        continue
        
    current_property_features["num_images_total"] = len(image_paths_map_data) if image_paths_map_data else 0
    [REDACTED_BY_SCRIPT]_step5 = step5_data.get("[REDACTED_BY_SCRIPT]", [])
    current_property_features["[REDACTED_BY_SCRIPT]"] = len(evaluated_room_labels_step5)

    all_step2_room_assignments = step2_data.get("room_assignments", []) if step2_data else []
    num_total_bedrooms_s2, num_bathrooms_s2 = 0, 0
    for s2_room in all_step2_room_assignments:
        s2_label_lower = s2_room.get("label", "").lower()
        if "bedroom" in s2_label_lower: num_total_bedrooms_s2 += 1
        if any(term in s2_label_lower for term in ["bathroom", "en-suite", "ensuite", "shower room", "wc", "cloakroom"]):
            num_bathrooms_s2 +=1
    current_property_features["[REDACTED_BY_SCRIPT]"] = num_total_bedrooms_s2
    current_property_features["[REDACTED_BY_SCRIPT]"] = num_bathrooms_s2
    current_property_features["[REDACTED_BY_SCRIPT]"] = 1 if num_total_bedrooms_s2 > 0 else 0
    current_property_features["[REDACTED_BY_SCRIPT]"] = max(0, num_total_bedrooms_s2 - 1)

    map_eval_label_to_step2_instances = defaultdict(list)
    for s2_room_instance in all_step2_room_assignments:
        s2_label, s2_label_lower = s2_room_instance.get("label", ""), s2_room_instance.get("label", "").lower()
        if not s2_label_lower: continue
        matched_eval_label = None
        for eval5_label in evaluated_room_labels_step5:
            eval5_main_keyword = eval5_label.split(" ")[0].split("-")[0].lower()
            if eval5_main_keyword in s2_label_lower or eval5_label == s2_label : # Added direct match
                matched_eval_label = eval5_label
                break
        if matched_eval_label: map_eval_label_to_step2_instances[matched_eval_label].append(s2_room_instance)

    property_room_data_collection = []
    for room_eval_label in evaluated_room_labels_step5:
        generic_type = map_eval_label_to_generic_canonical(room_eval_label)
        if generic_type == "UNKNOWN_ROOM_TYPE" or generic_type in ["FLOORPLAN", "AERIAL_VIEW", "SITE_PLAN", "DETAIL_SHOT", "VIEW_FROM_PROPERTY", "COMMUNAL_AREA"]:
            continue
        area_val, image_val = None, 0
        if step1_data and "rooms_with_dimensions" in step1_data:
            for rdi in step1_data["rooms_with_dimensions"]:
                if rdi.get("label") == room_eval_label: area_val = parse_dimensions_to_area(rdi.get("dimensions")); break
        temp_s2_list = map_eval_label_to_step2_instances.get(room_eval_label, [])
        for s2i in temp_s2_list: image_val += len(s2i.get("image_indices",[]))
        
        property_room_data_collection.append({
            "eval_label": room_eval_label, "generic_type": generic_type,
            "reno_score": step6_data.get(room_eval_label) if step6_data else None,
            "sps": (step4_data_prop.get(room_eval_label, {}).get("selling_points", [])) if step4_data_prop else [],
            "flaws": (step4_data_prop.get(room_eval_label, {}).get("flaws", [])) if step4_data_prop else [],
            "features_text": step3_data.get(room_eval_label, []) if step3_data else [],
            "image_count": image_val, "area": area_val
        })

    rooms_grouped_by_generic_type = defaultdict(list)
    for rd_item in property_room_data_collection: rooms_grouped_by_generic_type[rd_item["generic_type"]].append(rd_item)

    for generic_type, instances_data_list in rooms_grouped_by_generic_type.items():
        if not instances_data_list: continue
        main_instance_data, other_instances_data_list = None, []
        if len(instances_data_list) == 1: main_instance_data = instances_data_list[0]
        else:
            def sort_key(room_d):
                area = room_d.get("area") if room_d.get("area") is not None else -1
                return (area > 0, area, room_d.get("image_count", 0), len(room_d.get("sps", [])), room_d.get("reno_score") if room_d.get("reno_score") is not None else -1)
            sorted_instances = sorted(instances_data_list, key=sort_key, reverse=True)
            main_instance_data = sorted_instances[0]
            other_instances_data_list = sorted_instances[1:]
        
        primary_prefix_root, other_prefix_root = CANONICAL_PREFIX_ROOT_MAP.get(generic_type), CANONICAL_PREFIX_ROOT_MAP.get(f"[REDACTED_BY_SCRIPT]")
        if generic_type == "BATHROOM_WC" and main_instance_data: # Special handling for BATHROOM_WC sub-types
            eval_label_lower_main = main_instance_data["eval_label"].lower()
            if "en-suite" in eval_label_lower_main or "ensuite" in eval_label_lower_main: primary_prefix_root = "MainEnsuite"
            elif "cloakroom" in eval_label_lower_main or " wc" in eval_label_lower_main or "toilet" in eval_label_lower_main: primary_prefix_root = "MainWC"
            else: primary_prefix_root = "MainBathroom" # Default for BATHROOM_WC generic type
            other_prefix_root = "OtherBathroomsWCs" # All others go to one bucket
        
        if not primary_prefix_root: primary_prefix_root = "Misc" # Fallback for primary as per your request
        if not other_prefix_root and primary_prefix_root not in SINGLE_INSTANCE_CANONICAL_TYPES:
             other_prefix_root = f"[REDACTED_BY_SCRIPT]'Main','').replace('Primary','')}Other" # e.g. MiscOther
             if primary_prefix_root == "Misc": other_prefix_root = "OtherMisc"


        if main_instance_data:
            cpf_pfix = f"[REDACTED_BY_SCRIPT]"
            # current_property_features[f"[REDACTED_BY_SCRIPT]"] = main_instance_data["eval_label"] # For debugging
            current_property_features[f"{cpf_pfix}_area_sqm"] = main_instance_data.get("area", 0) if main_instance_data.get("area") is not None else 0
            current_property_features[f"[REDACTED_BY_SCRIPT]"] = main_instance_data.get("image_count", 0)
            reno_m = main_instance_data.get("reno_score"); current_property_features[f"[REDACTED_BY_SCRIPT]"] = reno_m if reno_m is not None else 0
            sps_m, flaws_m = main_instance_data.get("sps", []), main_instance_data.get("flaws", [])
            n_sps_m, n_flaws_m = len(sps_m), len(flaws_m)
            current_property_features[f"{cpf_pfix}_num_sps"], current_property_features[f"[REDACTED_BY_SCRIPT]"] = n_sps_m, n_flaws_m
            current_property_features[f"[REDACTED_BY_SCRIPT]"] = n_sps_m / (n_flaws_m + 1e-6)
            for tk in MASTER_ALL_SP_THEME_TOKENS: current_property_features[f"[REDACTED_BY_SCRIPT]"] = sum(1 for sp in sps_m if tk in sp.get("tags", []))
            for tk in MASTER_ALL_FLAW_THEME_TOKENS: current_property_features[f"[REDACTED_BY_SCRIPT]"] = sum(1 for fl in flaws_m if tk in fl.get("tags", []))
            m_ft_text = main_instance_data.get("features_text", [])
            current_property_features[f"[REDACTED_BY_SCRIPT]"] = len(set(m_ft_text))
            for tk in MASTER_ALL_FEATURE_TOKENS:
                tk_var = tk.lower().replace("_", " ")
                current_property_features[f"[REDACTED_BY_SCRIPT]"] = 1 if any(tk_var in ft.lower() for ft in m_ft_text) else 0
            # Embeddings for main_instance_data commented out

        if other_instances_data_list and other_prefix_root:
            cpf_opfix = f"other_{other_prefix_root}"
            current_property_features[f"{cpf_opfix}_count"] = len(other_instances_data_list)
            other_reno = [r.get("reno_score") for r in other_instances_data_list if r.get("reno_score") is not None]
            if other_reno: current_property_features[f"[REDACTED_BY_SCRIPT]"] = np.mean(other_reno)
            current_property_features[f"[REDACTED_BY_SCRIPT]"] = sum(len(r.get("sps",[])) for r in other_instances_data_list)
            current_property_features[f"[REDACTED_BY_SCRIPT]"] = sum(len(r.get("flaws",[])) for r in other_instances_data_list)
            for tk in MASTER_ALL_SP_THEME_TOKENS: current_property_features[f"[REDACTED_BY_SCRIPT]"] = sum(sum(1 for sp in r.get("sps",[]) if tk in sp.get("tags",[])) for r in other_instances_data_list)
            for tk in MASTER_ALL_FLAW_THEME_TOKENS: current_property_features[f"[REDACTED_BY_SCRIPT]"] = sum(sum(1 for fl in r.get("flaws",[]) if tk in fl.get("tags",[])) for r in other_instances_data_list)
            for tk in MASTER_ALL_FEATURE_TOKENS:
                tk_var = tk.lower().replace("_", " ")
                current_property_features[f"[REDACTED_BY_SCRIPT]"] = 1 if any(any(tk_var in ft.lower() for ft in r.get("features_text",[])) for r in other_instances_data_list) else 0
            # Embeddings for other_instances_data_list (aggregated) commented out
            
    # Property-wide aggregates from original eval_labels (for general overview)
    all_reno_eval = [step6_data.get(rl) for rl in evaluated_room_labels_step5 if step6_data and isinstance(step6_data.get(rl), (int,float))]
    if all_reno_eval: current_property_features["[REDACTED_BY_SCRIPT]"] = np.mean(all_reno_eval) # etc. for min, max, std
    current_property_features["total_sps_property_eval_labels"] = sum(len(step4_data_prop.get(rl, {}).get("selling_points",[])) for rl in evaluated_room_labels_step5 if step4_data_prop)
    current_property_features["[REDACTED_BY_SCRIPT]"] = sum(len(step4_data_prop.get(rl, {}).get("flaws",[])) for rl in evaluated_room_labels_step5 if step4_data_prop)

    # Aggregated property-wide SP/Flaw themes from CANONICAL features
    for tk in MASTER_ALL_SP_THEME_TOKENS:
        total = sum(current_property_features.get(f"[REDACTED_BY_SCRIPT]", 0) for pfx_val in CANONICAL_PREFIX_ROOT_MAP.values() for pfx_root in [f"primary_{pfx_val}", f"[REDACTED_BY_SCRIPT]'Main','').replace('Primary','')}Other", f"[REDACTED_BY_SCRIPT]'Main','').replace('Primary','')}s"]) # Approximate other keys
        current_property_features[f"[REDACTED_BY_SCRIPT]"] = total
    for tk in MASTER_ALL_FLAW_THEME_TOKENS:
        total = sum(current_property_features.get(f"[REDACTED_BY_SCRIPT]", 0) for pfx_val in CANONICAL_PREFIX_ROOT_MAP.values() for pfx_root in [f"primary_{pfx_val}", f"[REDACTED_BY_SCRIPT]'Main','').replace('Primary','')}Other", f"[REDACTED_BY_SCRIPT]'Main','').replace('Primary','')}s"])
        current_property_features[f"[REDACTED_BY_SCRIPT]"] = total
        
    # Persona features
    if step5_data and "[REDACTED_BY_SCRIPT]" in step5_data:
        persona_ratings_dict = step5_data["[REDACTED_BY_SCRIPT]"]
        all_p_scores = []
        if persona_ratings_dict:
            for p_name, p_item in persona_ratings_dict.items():
                if not isinstance(p_item, dict): continue
                norm_p = normalize_room_label_for_feature_name(p_name)
                rating = p_item.get("rating") if p_item else None
                current_property_features[f"[REDACTED_BY_SCRIPT]"] = rating if rating is not None else 0
                if isinstance(rating, (int, float)): all_p_scores.append(rating)
        if all_p_scores:
            current_property_features["[REDACTED_BY_SCRIPT]"] = np.mean(all_p_scores)
            current_property_features["[REDACTED_BY_SCRIPT]"] = np.min(all_p_scores)
            current_property_features["[REDACTED_BY_SCRIPT]"] = np.max(all_p_scores)
            current_property_features["[REDACTED_BY_SCRIPT]"] = np.std(all_p_scores)
        # else: default to 0 from defaultdict

    # Untagged text basic features
    if untagged_text_data:
        current_property_features["len_raw_features_list"] = len(untagged_text_data.get("raw_features", []))
        current_property_features["[REDACTED_BY_SCRIPT]"] = sum(len(str(s)) for s in untagged_text_data.get("[REDACTED_BY_SCRIPT]", []))
    
    # Property-wide S4 text embeddings (COMMENTED OUT)
    # all_s4_sp_texts, all_s4_flaw_texts = [], []
    # if step4_data_prop:
    #     for room_data_s4 in step4_data_prop.values():
    #         all_s4_sp_texts.extend([sp.get("text","") for sp in room_data_s4.get("selling_points",[])])
    #         all_s4_flaw_texts.extend([fl.get("text","") for fl in room_data_s4.get("flaws",[])])
    # prop_s4_sp_emb = get_text_embeddings(all_s4_sp_texts, MODEL)
    # prop_s4_flaw_emb = get_text_embeddings(all_s4_flaw_texts, MODEL)
    # for i in range(EMBEDDING_DIM):
    #     current_property_features[f"[REDACTED_BY_SCRIPT]"] = prop_s4_sp_emb[i]
    #     current_property_features[f"[REDACTED_BY_SCRIPT]"] = prop_s4_flaw_emb[i]

    # Interaction Features
    current_property_features["[REDACTED_BY_SCRIPT]"] = current_property_features["[REDACTED_BY_SCRIPT]"] * current_property_features["[REDACTED_BY_SCRIPT]"]
    current_property_features["[REDACTED_BY_SCRIPT]"] = current_property_features["[REDACTED_BY_SCRIPT]"] ** 2
    if current_property_features.get("[REDACTED_BY_SCRIPT]",0) > 0 : # Check to prevent division by zero if all are 0
        current_property_features["[REDACTED_BY_SCRIPT]"] = current_property_features.get("[REDACTED_BY_SCRIPT]",0) / \
                                (1 + current_property_features.get("[REDACTED_BY_SCRIPT]",0)) # Use new agg key

    ALL_GENERATED_FEATURE_KEYS.update(current_property_features.keys())
    all_properties_features_list_of_dicts.append(dict(current_property_features))

    # Incremental Save Logic
    is_last_property = (idx == len(property_folder_names) - 1)
    if (property_id_counter % SAVE_CHUNK_SIZE == 0) or is_last_property:
        if all_properties_features_list_of_dicts:
            print(f"[REDACTED_BY_SCRIPT]")
            current_chunk_keys = set()
            for d_item in all_properties_features_list_of_dicts: current_chunk_keys.update(d_item.keys())
            ALL_GENERATED_FEATURE_KEYS.update(current_chunk_keys) # Ensure master key list is up to date
            
            final_cols_for_chunk = sorted(list(ALL_GENERATED_FEATURE_KEYS)) # Use the most complete list of columns

            # Standardize dicts in current chunk to have all columns from final_cols_for_chunk
            # This is important because ALL_GENERATED_FEATURE_KEYS might have grown *before* this chunk's dicts were created.
            # When creating the DataFrame, ensure all dicts effectively have all columns.
            # Pandas DataFrame constructor from list of dicts handles this by creating NaNs.
            
            chunk_df = pd.DataFrame(all_properties_features_list_of_dicts) # Let pandas align columns
            # Reindex to ensure all columns from master list are present, fill new ones with 0
            chunk_df = chunk_df.reindex(columns=final_cols_for_chunk, fill_value=0)
            chunk_df = chunk_df.fillna(0) # Fill any other NaNs

            if not HEADER_WRITTEN:
                chunk_df.to_csv(OUTPUT_CSV_FILE, mode='w', header=True, index=False)
                HEADER_WRITTEN = True
                print(f"[REDACTED_BY_SCRIPT]")
            else:
                # When appending, we need to make sure the chunk_df has columns compatible with the existing file.
                # The reindex above should ensure this.
                chunk_df.to_csv(OUTPUT_CSV_FILE, mode='a', header=False, index=False)
            
            print(f"[REDACTED_BY_SCRIPT]")
            all_properties_features_list_of_dicts = [] 
    
print("[REDACTED_BY_SCRIPT]")
print(f"[REDACTED_BY_SCRIPT]")
print(f"[REDACTED_BY_SCRIPT]")
print(f"[REDACTED_BY_SCRIPT]")
print(f"[REDACTED_BY_SCRIPT]")
print(f"[REDACTED_BY_SCRIPT]")
if HEADER_WRITTEN:
    print(f"[REDACTED_BY_SCRIPT]")
else:
    print("[REDACTED_BY_SCRIPT]")

print("Script finished.")