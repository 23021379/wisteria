import os
import json
import re
import pandas as pd
import numpy as np
from collections import defaultdict
from sentence_transformers import SentenceTransformer

# ==============================================================================
# I. CONSTANTS AND CONFIGURATION
# ==============================================================================

# --- File Paths ---
BASE_PROPERTY_DATA_DIR = r"[REDACTED_BY_SCRIPT]"
OUTPUT_QUAL_CSV_FILE = r"[REDACTED_BY_SCRIPT]"
OUTPUT_QUANT_CSV_FILE = r"[REDACTED_BY_SCRIPT]"

# --- Processing Configuration ---
POSSIBLE_YEARS_FALLBACK = ["2025", "2024", "2023", "2022","2021","2020","2019","2018","2017","2016","2015","2014","2013","2012","2011","2010"]
SAVE_CHUNK_SIZE = 50

# --- Embedding Model Configuration ---
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
# EMBEDDING_DIM will be set dynamically after loading the model

# --- Canonical Mapping Configuration (from original script) ---
GENERIC_MAPPING_ORDERED = [("KITCHEN", ("kitchen",)),("LIVING_AREA", ("living room", "lounge", "sitting room", "reception room", "family room", "snug", "drawing room")),("DINING_AREA", ("dining room", "dining area", "breakfast room")),("BEDROOM", ("bedroom", "bed room")),("BATHROOM_WC", ("bathroom", "shower room", "en-suite", "ensuite", "en suite", "wc", "toilet", "cloakroom", "washroom")),("[REDACTED_BY_SCRIPT]", ("hall", "hallway", "landing", "entrance", "stairs", "staircase", "inner hall", "reception hall")),("UTILITY_ROOM", ("utility", "laundry")),("GARAGE", ("garage",)),("CONSERVATORY_SUNROOM", ("conservatory", "sun room", "orangery", "garden room")),("OFFICE_STUDY", ("office", "study", "workroom")),("GARDEN_YARD", ("garden", "yard", "grounds", "rear garden", "front garden", "side garden")),("[REDACTED_BY_SCRIPT]", ("patio", "terrace", "decking", "balcony")),("DRIVEWAY_PARKING", ("driveway", "drive", "parking")),("EXTERIOR_FRONT", ("front exterior", "front elevation", "external front", "front of house")),("EXTERIOR_REAR", ("rear exterior", "rear elevation", "external rear", "rear of house")),("EXTERIOR_SIDE", ("side exterior", "side elevation", "external side")),("OUTBUILDING", ("outbuilding", "shed", "workshop", "store", "summer house", "annex")),("STORAGE", ("storage", "store room", "cupboard", "wardrobe", "airing cupboard", "eaves", "loft", "attic", "cellar", "basement")),("PORCH", ("porch", "storm porch")),("MISC_INDOOR", ("games room", "playroom", "cinema room", "gym", "mezzanine", "boot room")),("AERIAL_VIEW", ("aerial view", "drone shot")),("FLOORPLAN", ("floorplan", "floor plan")),("SITE_PLAN", ("site plan",)),("DETAIL_SHOT", ("detail", "close up", "feature")),("VIEW_FROM_PROPERTY", ("view from property", "outlook", "view")),("COMMUNAL_AREA", ("communal",))]
CANONICAL_PREFIX_ROOT_MAP = {"KITCHEN": "MainKitchen", "LIVING_AREA": "MainLivingArea", "DINING_AREA": "MainDiningArea","BEDROOM": "PrimaryBedroom", "BATHROOM_WC": "MainBathroom", "[REDACTED_BY_SCRIPT]": "[REDACTED_BY_SCRIPT]","UTILITY_ROOM": "MainUtilityRoom", "GARAGE": "MainGarage", "CONSERVATORY_SUNROOM": "[REDACTED_BY_SCRIPT]","OFFICE_STUDY": "StudyOffice", "GARDEN_YARD": "MainGarden", "[REDACTED_BY_SCRIPT]":"[REDACTED_BY_SCRIPT]","DRIVEWAY_PARKING": "MainDrivewayParking", "EXTERIOR_FRONT": "MainExteriorFront", "EXTERIOR_REAR": "MainExteriorRear","EXTERIOR_SIDE": "MainExteriorSide", "OUTBUILDING": "MainOutbuilding", "STORAGE": "[REDACTED_BY_SCRIPT]","PORCH": "MainPorch", "MISC_INDOOR": "MainMiscIndoor","KITCHEN_OTHER": "OtherKitchens", "LIVING_AREA_OTHER": "OtherLivingAreas", "DINING_AREA_OTHER": "OtherDiningAreas","BEDROOM_OTHER": "OtherBedrooms", "BATHROOM_WC_OTHER": "OtherBathroomsWCs", "GARDEN_YARD_OTHER": "OtherGardens",}
SINGLE_INSTANCE_CANONICAL_TYPES = {"[REDACTED_BY_SCRIPT]", "MainUtilityRoom", "StudyOffice", "MainGarage", "[REDACTED_BY_SCRIPT]","MainExteriorFront", "MainExteriorRear", "MainExteriorSide", "MainDrivewayParking", "MainPorch","MainMiscIndoor", "[REDACTED_BY_SCRIPT]", "MainOutbuilding", "[REDACTED_BY_SCRIPT]"}


# ==============================================================================
# II. HELPER FUNCTIONS
# ==============================================================================

def load_embedding_model():
    print(f"[REDACTED_BY_SCRIPT]")
    try:
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print("[REDACTED_BY_SCRIPT]")
        return model
    except Exception as e:
        print(f"[REDACTED_BY_SCRIPT]")
        exit()

def get_property_specific_year_suffix(property_path):
    ### MODIFIED ### - Added step3, step4, untagged review to the check
    files_to_check_for_year = ["output_step5_merged", "[REDACTED_BY_SCRIPT]", "output_step3_features", "[REDACTED_BY_SCRIPT]"]
    for year_val_str in POSSIBLE_YEARS_FALLBACK:
        for prefix in files_to_check_for_year:
            if os.path.exists(os.path.join(property_path, f"[REDACTED_BY_SCRIPT]")): return f"_y_y{year_val_str}"
    return None

def find_json_file_for_property(base_path, file_prefix_no_year, property_year_suffix, file_suffix=".json"):
    filenames_to_try = []
    if property_year_suffix: filenames_to_try.append(f"[REDACTED_BY_SCRIPT]")
    # Fallback to general year formats if property specific suffix fails
    for year_val_str in POSSIBLE_YEARS_FALLBACK:
        filenames_to_try.append(f"[REDACTED_BY_SCRIPT]")
    filenames_to_try.append(f"[REDACTED_BY_SCRIPT]")
    for filename in set(filenames_to_try):
        filepath = os.path.join(base_path, filename)
        if os.path.exists(filepath): return filepath
    return None

def load_json_safe(filepath):
    if not filepath or not os.path.exists(filepath): return None
    try:
        with open(filepath, 'r', encoding='utf-8') as f: return json.load(f)
    except Exception: return None

def parse_dimensions_to_area(dim_str):
    if not dim_str or str(dim_str).lower() == "null" or str(dim_str).strip() == "": return None
    dim_str = str(dim_str)
    try:
        matches = re.findall(r"([\d\.]+)\s*(m|ft|['\"[REDACTED_BY_SCRIPT]'\"])?", dim_str)
        if matches:
            d1_val_str, _, d2_val_str, _ = matches[0]
            d1, d2 = float(d1_val_str), float(d2_val_str)
            if 'ft' in dim_str.lower(): return round((d1 * 0.3048) * (d2 * 0.3048), 2)
            return round(d1 * d2, 2)
        area_match = re.search(r"[REDACTED_BY_SCRIPT]", dim_str, re.IGNORECASE)
        if area_match:
            val, unit = float(area_match.group(1)), area_match.group(2).lower()
            if "ft" in unit: return round(val * 0.092903, 2)
            return round(val, 2)
    except Exception: pass
    return None

def map_eval_label_to_generic_canonical(eval_label):
    if not eval_label or not isinstance(eval_label, str): return "UNKNOWN_ROOM_TYPE"
    text_lower = eval_label.lower().strip()
    if "kitchen/diner" in text_lower or "kitchen diner" in text_lower: return "KITCHEN"
    if "reception hall" in text_lower: return "[REDACTED_BY_SCRIPT]"
    for generic_name, keywords in GENERIC_MAPPING_ORDERED:
        for keyword in keywords:
            if keyword in ["hall", "wc", "view", "store", "drive", "gym", "loft", "attic", "eaves", "cellar", "basement"]:
                if re.search(r'\b' + re.escape(keyword) + r'\b', text_lower): return generic_name
            elif keyword in text_lower: return generic_name
    return "UNKNOWN_ROOM_TYPE"

def generate_text_embedding(text_document, model, embedding_dim):
    if isinstance(text_document, list):
        text_document = ". ".join(str(item) for item in text_document if item)
    if not text_document or not text_document.strip():
        return np.zeros(embedding_dim, dtype=np.float32)
    return model.encode(text_document)

def is_embedding_column(column_name):
    return re.search(r'_embedding_\d+$', column_name) is not None

def separate_features_by_type(features_dict):
    qualitative_features = {"property_id": features_dict["property_id"]}
    quantitative_features = {"property_id": features_dict["property_id"]}
    for key, value in features_dict.items():
        if key == "property_id":
            continue
        if is_embedding_column(key):
            qualitative_features[key] = value
        else:
            quantitative_features[key] = value
    return qualitative_features, quantitative_features

def create_empty_property_features(property_id, embedding_dim):
    empty_features = defaultdict(lambda: 0)
    empty_features["property_id"] = property_id
    
    # Define all embedding column prefixes to ensure they are created
    embedding_prefixes = []

    # Persona justification embeddings
    persona_names = [f"persona_{i+1}" for i in range(20)]
    for p_name in persona_names:
        norm_p_name = "".join(word.capitalize() for word in re.split(r"[\s-]+", p_name.strip()))
        embedding_prefixes.append(f"[REDACTED_BY_SCRIPT]")

    ### NEW ### - Add property-wide embedding prefixes
    embedding_prefixes.extend([
        "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]"
    ])
    
    # Room text embeddings
    text_types_room = ["features", "sps", "flaws"]
    for generic_type in CANONICAL_PREFIX_ROOT_MAP.keys():
        if "_OTHER" in generic_type: continue
        p_prefix_root = CANONICAL_PREFIX_ROOT_MAP[generic_type]
        p_prefix = f"[REDACTED_BY_SCRIPT]"
        for text_type in text_types_room:
            embedding_prefixes.append(f"[REDACTED_BY_SCRIPT]")
        if p_prefix_root not in SINGLE_INSTANCE_CANONICAL_TYPES:
            o_prefix = f"[REDACTED_BY_SCRIPT]'{generic_type}_OTHER', 'MiscOther')}"
            for text_type in text_types_room:
                embedding_prefixes.append(f"[REDACTED_BY_SCRIPT]")

    # Create all embedding columns with zero values
    for prefix in embedding_prefixes:
        for i in range(embedding_dim):
            empty_features[f"{prefix}_{i}"] = 0.0
            
    return dict(empty_features)

# ==============================================================================
# III. MAIN PROCESSING SCRIPT
# ==============================================================================

def main():
    MODEL = load_embedding_model()
    EMBEDDING_DIM = MODEL.get_sentence_embedding_dimension()
    print(f"[REDACTED_BY_SCRIPT]")

    # ==========================================================================
    # --- Pre-define column headers for both types ---
    # ==========================================================================
    quantitative_columns = [
        "property_id", "num_images_total", "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]", "[REDACTED_BY_SCRIPT]"
    ]
    qualitative_columns = ["property_id"]

    # Persona columns
    persona_names = [f"persona_{i+1}" for i in range(20)]
    for p_name in persona_names:
        norm_p_name = "".join(word.capitalize() for word in re.split(r"[\s-]+", p_name.strip()))
        quantitative_columns.append(f"[REDACTED_BY_SCRIPT]")
        for i in range(EMBEDDING_DIM):
            qualitative_columns.append(f"[REDACTED_BY_SCRIPT]")

    ### NEW ### Add property-wide aggregated text embedding columns
    property_wide_embedding_types = ["raw_features", "sps", "flaws"]
    for pw_type in property_wide_embedding_types:
        for i in range(EMBEDDING_DIM):
            qualitative_columns.append(f"[REDACTED_BY_SCRIPT]")

    # Room feature columns
    text_types = ["features", "sps", "flaws"]
    for generic_type in CANONICAL_PREFIX_ROOT_MAP.keys():
        if "_OTHER" in generic_type: continue
        p_prefix_root = CANONICAL_PREFIX_ROOT_MAP[generic_type]
        p_prefix = f"[REDACTED_BY_SCRIPT]"
        quantitative_columns.extend([f"{p_prefix}_area_sqm", f"[REDACTED_BY_SCRIPT]", f"[REDACTED_BY_SCRIPT]", f"{p_prefix}_num_sps", f"[REDACTED_BY_SCRIPT]"])
        for text_type in text_types:
            for i in range(EMBEDDING_DIM):
                qualitative_columns.append(f"[REDACTED_BY_SCRIPT]")
        if p_prefix_root not in SINGLE_INSTANCE_CANONICAL_TYPES:
            o_prefix = f"[REDACTED_BY_SCRIPT]'{generic_type}_OTHER', 'MiscOther')}"
            quantitative_columns.extend([f"{o_prefix}_count", f"[REDACTED_BY_SCRIPT]", f"[REDACTED_BY_SCRIPT]", f"[REDACTED_BY_SCRIPT]", f"[REDACTED_BY_SCRIPT]", f"[REDACTED_BY_SCRIPT]", f"[REDACTED_BY_SCRIPT]"])
            for text_type in text_types:
                for i in range(EMBEDDING_DIM):
                    qualitative_columns.append(f"[REDACTED_BY_SCRIPT]")

    all_properties_features_list_of_dicts = []
    
    if not os.path.exists(BASE_PROPERTY_DATA_DIR):
        print(f"[REDACTED_BY_SCRIPT]")
        return

    processed_ids = set()
    QUAL_HEADER_WRITTEN = False
    QUANT_HEADER_WRITTEN = False
    
    for output_file, file_type in [(OUTPUT_QUAL_CSV_FILE, "qualitative"), (OUTPUT_QUANT_CSV_FILE, "quantitative")]:
        if os.path.exists(output_file):
            print(f"[REDACTED_BY_SCRIPT]")
            try:
                df_existing = pd.read_csv(output_file, usecols=['property_id'], low_memory=False)
                existing_ids = set(df_existing['property_id'])
                processed_ids.update(existing_ids)
                if file_type == "qualitative": QUAL_HEADER_WRITTEN = True
                else: QUANT_HEADER_WRITTEN = True
                print(f"[REDACTED_BY_SCRIPT]")
            except Exception as e:
                print(f"[REDACTED_BY_SCRIPT]")
    
    if processed_ids:
        print(f"[REDACTED_BY_SCRIPT]")
    else:
        print("[REDACTED_BY_SCRIPT]")

    all_folder_names = os.listdir(BASE_PROPERTY_DATA_DIR)
    folders_to_process = [p for p in all_folder_names if os.path.isdir(os.path.join(BASE_PROPERTY_DATA_DIR, p)) and p not in processed_ids]
    total_to_process = len(folders_to_process)
    
    if total_to_process == 0:
        print("[REDACTED_BY_SCRIPT]")
        return
        
    print(f"[REDACTED_BY_SCRIPT]")
    
    for idx, property_folder_name in enumerate(folders_to_process):
        property_path = os.path.join(BASE_PROPERTY_DATA_DIR, property_folder_name)
        if not os.path.isdir(property_path): continue

        print(f"[REDACTED_BY_SCRIPT]")

        current_property_features = defaultdict(lambda: 0)
        current_property_features["property_id"] = property_folder_name

        prop_year_suffix = get_property_specific_year_suffix(property_path)
        
        ### MODIFIED ### Load all necessary files including the new ones
        step1_data = load_json_safe(find_json_file_for_property(property_path, "[REDACTED_BY_SCRIPT]", prop_year_suffix))
        step3_data = load_json_safe(find_json_file_for_property(property_path, "output_step3_features", prop_year_suffix))
        step4_data = load_json_safe(find_json_file_for_property(property_path, "[REDACTED_BY_SCRIPT]", prop_year_suffix))
        step5_data = load_json_safe(find_json_file_for_property(property_path, "output_step5_merged", prop_year_suffix))
        step6_data = load_json_safe(find_json_file_for_property(property_path, "[REDACTED_BY_SCRIPT]", prop_year_suffix))
        untagged_review_data = load_json_safe(find_json_file_for_property(property_path, "[REDACTED_BY_SCRIPT]", prop_year_suffix))
        image_paths_map_data = load_json_safe(find_json_file_for_property(property_path, "image_paths_map", prop_year_suffix))

        if not step5_data:
            print(f"[REDACTED_BY_SCRIPT]'output_step5_merged'[REDACTED_BY_SCRIPT]")
            empty_features = create_empty_property_features(property_folder_name, EMBEDDING_DIM)
            all_properties_features_list_of_dicts.append(empty_features)
        else:
            current_property_features["num_images_total"] = len(image_paths_map_data) if image_paths_map_data else 0
            [REDACTED_BY_SCRIPT]_step5 = step5_data.get("[REDACTED_BY_SCRIPT]", [])
            current_property_features["[REDACTED_BY_SCRIPT]"] = len(evaluated_room_labels_step5)
            
            ### NEW ### Process property-wide embeddings from untagged_review_data
            if untagged_review_data:
                raw_features_doc = ". ".join(untagged_review_data.get("raw_features", []))
                sps_doc = ". ".join(untagged_review_data.get("[REDACTED_BY_SCRIPT]", []))
                flaws_doc = ". ".join(untagged_review_data.get("raw_flaws_text", []))
                
                raw_features_emb = generate_text_embedding(raw_features_doc, MODEL, EMBEDDING_DIM)
                sps_emb = generate_text_embedding(sps_doc, MODEL, EMBEDDING_DIM)
                flaws_emb = generate_text_embedding(flaws_doc, MODEL, EMBEDDING_DIM)

                for i in range(EMBEDDING_DIM):
                    current_property_features[f"[REDACTED_BY_SCRIPT]"] = raw_features_emb[i]
                    current_property_features[f"[REDACTED_BY_SCRIPT]"] = sps_emb[i]
                    current_property_features[f"[REDACTED_BY_SCRIPT]"] = flaws_emb[i]

            persona_ratings = step5_data.get("[REDACTED_BY_SCRIPT]", {})
            if persona_ratings:
                all_p_scores = []
                for p_name, p_data in persona_ratings.items():
                    if not isinstance(p_data, dict): continue
                    match = re.match(r'persona_\d+', p_name)
                    if not match: continue
                    standardized_p_name = match.group(0)
                    norm_p_name = "".join(word.capitalize() for word in re.split(r"[\s-]+", standardized_p_name.strip()))
                    rating = p_data.get("rating")
                    if isinstance(rating, (int, float)):
                        current_property_features[f"[REDACTED_BY_SCRIPT]"] = rating
                        all_p_scores.append(rating)
                    justification_text = p_data.get("justification", "")
                    just_embedding = generate_text_embedding(justification_text, MODEL, EMBEDDING_DIM)
                    for i in range(EMBEDDING_DIM):
                        current_property_features[f"[REDACTED_BY_SCRIPT]"] = just_embedding[i]
                if all_p_scores:
                    current_property_features["[REDACTED_BY_SCRIPT]"] = np.mean(all_p_scores)
                    current_property_features["[REDACTED_BY_SCRIPT]"] = np.std(all_p_scores)

            property_room_data_collection = []
            for room_label in evaluated_room_labels_step5:
                generic_type = map_eval_label_to_generic_canonical(room_label)
                if generic_type == "UNKNOWN_ROOM_TYPE": continue
                
                ### MODIFIED ### This section now correctly uses step3 and step4 data
                sps_data = step4_data.get(room_label, {}).get("selling_points", []) if step4_data else []
                flaws_data = step4_data.get(room_label, {}).get("flaws", []) if step4_data else []
                property_room_data_collection.append({
                    "eval_label": room_label, "generic_type": generic_type,
                    "area": parse_dimensions_to_area(next((r.get("dimensions") for r in step1_data.get("rooms_with_dimensions",[]) if r.get("label") == room_label), None)) if step1_data else None,
                    "reno_score": step6_data.get(room_label) if step6_data else None,
                    "features_text": step3_data.get(room_label, []) if step3_data else [], # From step3
                    "sps_text": [sp.get("text", "") for sp in sps_data], # From step4
                    "flaws_text": [fl.get("text", "") for fl in flaws_data], # From step4
                })

            rooms_grouped_by_generic_type = defaultdict(list)
            for room_data in property_room_data_collection:
                rooms_grouped_by_generic_type[room_data["generic_type"]].append(room_data)

            for generic_type, instances in rooms_grouped_by_generic_type.items():
                main_instance, other_instances = None, []
                if len(instances) >= 1:
                    sorted_instances = sorted(instances, key=lambda r: (r.get("area") is not None, r.get("area", -1), len(r.get("sps_text", [])), r.get("eval_label", "")), reverse=True)
                    main_instance = sorted_instances[0]
                    other_instances = sorted_instances[1:]

                p_prefix_root = CANONICAL_PREFIX_ROOT_MAP.get(generic_type, 'Misc')
                if main_instance:
                    p_prefix = f"[REDACTED_BY_SCRIPT]"
                    current_property_features[f"{p_prefix}_area_sqm"] = main_instance.get("area", 0) if main_instance.get("area") is not None else 0
                    current_property_features[f"[REDACTED_BY_SCRIPT]"] = main_instance.get("reno_score", 0) if main_instance.get("reno_score") is not None else 0
                    current_property_features[f"[REDACTED_BY_SCRIPT]"] = len(main_instance.get("features_text", []))
                    current_property_features[f"{p_prefix}_num_sps"] = len(main_instance.get("sps_text", []))
                    current_property_features[f"[REDACTED_BY_SCRIPT]"] = len(main_instance.get("flaws_text", []))
                    for text_type in ["features", "sps", "flaws"]:
                        doc = ". ".join(main_instance.get(f"{text_type}_text", []))
                        emb = generate_text_embedding(doc, MODEL, EMBEDDING_DIM)
                        for i in range(EMBEDDING_DIM): current_property_features[f"[REDACTED_BY_SCRIPT]"] = emb[i]
                if other_instances and p_prefix_root not in SINGLE_INSTANCE_CANONICAL_TYPES:
                    o_prefix = f"[REDACTED_BY_SCRIPT]'{generic_type}_OTHER', 'MiscOther')}"
                    k = len(other_instances)
                    current_property_features[f"{o_prefix}_count"] = k
                    total_features = sum(len(r.get("features_text",[])) for r in other_instances)
                    total_sps = sum(len(r.get("sps_text",[])) for r in other_instances)
                    total_flaws = sum(len(r.get("flaws_text",[])) for r in other_instances)
                    current_property_features[f"[REDACTED_BY_SCRIPT]"] = total_features
                    current_property_features[f"[REDACTED_BY_SCRIPT]"] = total_sps
                    current_property_features[f"[REDACTED_BY_SCRIPT]"] = total_flaws
                    if k > 0:
                        current_property_features[f"[REDACTED_BY_SCRIPT]"] = total_features / k
                        current_property_features[f"[REDACTED_BY_SCRIPT]"] = total_sps / k
                        current_property_features[f"[REDACTED_BY_SCRIPT]"] = total_flaws / k
                    for text_type in ["features", "sps", "flaws"]:
                        all_texts = [text for r in other_instances for text in r.get(f"{text_type}_text", [])]
                        doc = ". ".join(all_texts)
                        emb = generate_text_embedding(doc, MODEL, EMBEDDING_DIM)
                        for i in range(EMBEDDING_DIM): current_property_features[f"[REDACTED_BY_SCRIPT]"] = emb[i]
            
            all_properties_features_list_of_dicts.append(dict(current_property_features))

        is_last_property = (idx == total_to_process - 1)
        if (len(all_properties_features_list_of_dicts) >= SAVE_CHUNK_SIZE) or (is_last_property and all_properties_features_list_of_dicts):
            print(f"[REDACTED_BY_SCRIPT]")
            
            qual_features_list = []
            quant_features_list = []
            for features_dict in all_properties_features_list_of_dicts:
                qual_features, quant_features = separate_features_by_type(features_dict)
                qual_features_list.append(qual_features)
                quant_features_list.append(quant_features)
            
            qual_df = pd.DataFrame(qual_features_list).reindex(columns=qualitative_columns, fill_value=0.0)
            for col in qual_df.columns:
                if col != 'property_id': qual_df[col] = qual_df[col].astype(float)
            qual_df.to_csv(OUTPUT_QUAL_CSV_FILE, mode='a' if QUAL_HEADER_WRITTEN else 'w', header=not QUAL_HEADER_WRITTEN, index=False)
            QUAL_HEADER_WRITTEN = True
            
            quant_df = pd.DataFrame(quant_features_list).reindex(columns=quantitative_columns, fill_value=0.0)
            for col in quant_df.columns:
                if col != 'property_id': quant_df[col] = quant_df[col].astype(float)
            quant_df.to_csv(OUTPUT_QUANT_CSV_FILE, mode='a' if QUANT_HEADER_WRITTEN else 'w', header=not QUANT_HEADER_WRITTEN, index=False)
            QUANT_HEADER_WRITTEN = True
            
            print(f"[REDACTED_BY_SCRIPT]")
            all_properties_features_list_of_dicts = []
    
    print("[REDACTED_BY_SCRIPT]")
    print(f"[REDACTED_BY_SCRIPT]")
    print(f"[REDACTED_BY_SCRIPT]")
    print(f"[REDACTED_BY_SCRIPT]")

if __name__ == "__main__":
    main()