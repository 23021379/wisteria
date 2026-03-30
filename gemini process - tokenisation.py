import google.genai as genai
from google.genai import types
from PIL import Image, UnidentifiedImageError
import os
import json
import datetime
import time
import statistics
import re
import sys; print(sys.path)
import csv 
import gemini_property_feature_generator 
import imagehash

# --- Configuration ---
try:
    # IMPORTANT: Keep this secure! Replace or use environment variables.
    # api_key = "YOUR_API_KEY_HERE" # Replace with your actual key
    api_key = "[REDACTED_BY_SCRIPT]" # Using the provided placeholder
    # api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("[REDACTED_BY_SCRIPT]")
    client = genai.Client(api_key=api_key)

except Exception as e:
    print(f"[REDACTED_BY_SCRIPT]")
    exit()

# Define Model
# model_name='[REDACTED_BY_SCRIPT]'
model_name='[REDACTED_BY_SCRIPT]'
print(f"[REDACTED_BY_SCRIPT]")

model_25_flash='[REDACTED_BY_SCRIPT]'
model_20_flash='[REDACTED_BY_SCRIPT]'
model_20_flash_lite='[REDACTED_BY_SCRIPT]'
model_15_flash='[REDACTED_BY_SCRIPT]'

  

# --- Define Main Input/Output Directories ---
# Directory containing folders named by address, each holding room images
main_image_dir = r"[REDACTED_BY_SCRIPT]"
# Directory containing folders named by address, each holding floorplan images
main_floorplan_dir = r"[REDACTED_BY_SCRIPT]"
# Parent directory where results folders (named by address) will be created
main_output_dir = r"[REDACTED_BY_SCRIPT]"

# Ensure main output directory exists
os.makedirs(main_output_dir, exist_ok=True)

# --- Define Master CSV Path ---
master_csv_path = os.path.join(main_output_dir, "[REDACTED_BY_SCRIPT]")
print(f"[REDACTED_BY_SCRIPT]")


EXPECTED_BASE_JSON_FILENAMES = [
    "image_paths_map", # Assuming this is generated somewhere, add if needed
    "[REDACTED_BY_SCRIPT]",
    "[REDACTED_BY_SCRIPT]",
    "output_step3_features",
    "[REDACTED_BY_SCRIPT]",
    "output_step5_merged", # Note: Merged step 5 might not always have year suffix in previous logic
    "[REDACTED_BY_SCRIPT]",
    "[REDACTED_BY_SCRIPT]",
    "[REDACTED_BY_SCRIPT]",
]
# Add step5 merged separately if it has a different naming convention sometimes
EXPECTED_MERGED_STEP5_BASE = "output_step5_merged"

def parse_dimensions_and_area(dimension_string):
    """
    Parses a dimension string like "3.00m x 2.07m" and returns area.
    Returns None if parsing fails.
    """
    if not dimension_string or not isinstance(dimension_string, str):
        return None
    # More robust regex to handle various spacing and optional 'm'
    match = re.search(r'[REDACTED_BY_SCRIPT]', dimension_string, re.IGNORECASE)
    if match:
        try:
            width = float(match.group(1))
            depth = float(match.group(2))
            return width * depth
        except ValueError:
            print(f"[REDACTED_BY_SCRIPT]")
            return None
    else:
        print(f"[REDACTED_BY_SCRIPT]")
        return None

def get_floorplan_images(floorplan_dir):
    """[REDACTED_BY_SCRIPT]"""
    floorplan_paths = []
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']
    try:
        # This function is now called with the specific YEAR folder path
        if not os.path.isdir(floorplan_dir):
            print(f"[REDACTED_BY_SCRIPT]")
            return [] # Return empty list, normal operation if floorplans aren't present
        for filename in os.listdir(floorplan_dir):
            file_path = os.path.join(floorplan_dir, filename)
            if os.path.isfile(file_path) and any(filename.lower().endswith(ext) for ext in image_extensions):
                floorplan_paths.append(file_path)
        return floorplan_paths
    except Exception as e:
        print(f"[REDACTED_BY_SCRIPT]")
        return []

def get_room_images(image_dir, floorplan_paths):
    """[REDACTED_BY_SCRIPT]"""
    room_image_paths = []
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']
    try:
        # This function is now called with the specific YEAR folder path
        if not os.path.isdir(image_dir):
            print(f"[REDACTED_BY_SCRIPT]'{image_dir}'. Cannot process.")
            return None # Indicate fatal error for this property year

        floorplan_paths_norm = {os.path.normpath(p) for p in floorplan_paths}

        for filename in os.listdir(image_dir):
            file_path = os.path.join(image_dir, filename)
            file_path_norm = os.path.normpath(file_path)
            if (os.path.isfile(file_path) and
                any(filename.lower().endswith(ext) for ext in image_extensions) and
                file_path_norm not in floorplan_paths_norm):
                room_image_paths.append(file_path)
        return room_image_paths
    except Exception as e:
        print(f"[REDACTED_BY_SCRIPT]")
        return None # Indicate failure

def load_image(path):
    """[REDACTED_BY_SCRIPT]"""
    try:
        img = Image.open(path)
        img.verify(); img.close() # Verify and close immediately
        img = Image.open(path) # Re-open
        img.load()
        if img.mode != 'RGB': img = img.convert('RGB')
        return img
    except FileNotFoundError: print(f"[REDACTED_BY_SCRIPT]"); return None
    except UnidentifiedImageError: print(f"[REDACTED_BY_SCRIPT]"); return None
    except Exception as e: print(f"[REDACTED_BY_SCRIPT]"); return None
    
# --- Persona Details ---
persona_details_string = """
**Instructions for Model:** Rate property suitability (1-10) based on these key persona needs. Focus on Needs/Avoid lists.

*   Persona 1: FTB (Solo). Goal: Affordable independence. Needs: <=£220k, 1-2 bed, easy commute, cosmetic update ok. Avoid: Major structural work, high ongoing costs.
*   Persona 2: Young Family (infants). Goal: Upsize for kids. Needs: 3-4 bed, safe enclosed garden, family suburb, good nursery/primary school, storage. Avoid: Busy roads, noise. Budget: £500-550k.
*   Persona 3: Downsizers (Retired). Goal: Low maintenance, convenience. Needs: 2 bed (bungalow/accessible flat), minimal garden, walk to town amenities, modern comforts (good EPC). Avoid: Stairs, high upkeep. Budget: ~£380k.
*   Persona 4: Remote Worker (Single). Goal: Permanent WFH setup, nature access. Needs: Dedicated quiet office, **FTTP/fast broadband essential**, 2-3 bed, near nature (North Eng/Scot), character. Avoid: Poor internet, shared workspace. Budget: ~£400k.
*   Persona 5: FTB Couple. Goal: Escape rent, future space. Needs: <=£350k, 2 bed (WFH/future nursery), cosmetic updates ok, functional kitchen/bath, some outdoor space. Avoid: Major immediate repairs, poor storage.
*   Persona 6: Education Parents (School Age). Goal: Specific secondary school catchment. Needs: **Catchment essential**, 3-4+ bed, study space, safe area, garden. Avoid: Busy roads. Budget: ~£750k.
*   Persona 7: Active Retiree (Widowed). Goal: Manageable, social access. Needs: 2 bed (bungalow/accessible flat), low maint property/garden, walk to amenities/social hubs, security, guest space. Avoid: Stairs, isolation (car required for basics). Budget: £450-550k.
*   Persona 8: Eco-Conscious (Single). Goal: Sustainable living. Needs: High EPC (B+/A), good insulation/glazing, potential renewables, sustainable materials, near public transport/cycle path, garden space (veg/compost). Avoid: Low EPC, fossil fuel heating. Budget: ~£400k.
*   Persona 9: Fixer-Upper Fans (Couple, skilled). Goal: Reno potential, build equity. Needs: Structurally sound, clear reno scope (cosmetic/layout), good location/potential, <£300k (+ reno budget). Avoid: Major structural flaws, bad previous DIY.
*   Persona 10: Established Family (Teens). Goal: Space/zones for teens, school. Needs: 4-5 bed, top secondary catchment, separate teen area (social/study), good kitchen/diner, storage (bikes). Avoid: Lack of quiet study space, cramped layout. Budget: ~£750k.
*   Persona 11: Upsizers (Young Child + Expecting). Goal: More space, WFH. Needs: 4 bed, garden, dedicated home office, modern kitchen/bath, family area, good primary school. Avoid: Lack of space, poor WFH space, dated finishes. Budget: ~£800k.
*   Persona 12: Urban Pro Couple (DINK). Goal: Central, high-spec, low maint. Needs: Prime central/connected location, 2 bed (apt/townhouse), high-spec modern finishes, low maint, good entertaining layout. Avoid: Long commutes, poor finishes. Budget: ~£550k.
*   Persona 13: Luxury Seeker (HNWI). Goal: Statement property, status, privacy. Needs: Prime/exclusive location, unique/bespoke design, high-end finishes, top security/tech, premium amenities (views, concierge). Avoid: Standard finishes, compromised privacy/security/location. Budget: £5M+.
*   Persona 14: Investor (BTL). Goal: Rental yield (ROI), capital growth secondary. Needs: High tenant demand area, **strong yield (6%+)**, durable/low maint finishes, minimal immediate work. Avoid: Void periods, high ongoing costs, niche appeal. Budget: <£250k typical.
*   Persona 15: Multi-Generational Household. Goal: Mutual support, shared living, privacy. Needs: 5+ beds, multiple living areas, potential ground floor suite/annex, accessible features (minor mobility), large kitchen/diner. Avoid: Lack of privacy/separation, poor layout for mixed ages. Budget: ~£900k.
*   Persona 16: Pet Owner (Single + Active Dog). Goal: Suitable home for dog. Needs: **Secure enclosed garden/patio**, durable/easy-clean floors, near parks/walks, easy garden access. Avoid: No private outdoor space, delicate floors (light carpet). Budget: ~£220k.
*   Persona 17: Social Entertainers (Couple). Goal: Space for frequent large hosting. Needs: Large open-plan kitchen/living/dining, high-spec kitchen (island?), functional outdoor entertaining space, good indoor/outdoor flow, guest WC. Avoid: Separated kitchen, poor guest flow/space. Budget: ~£950k.
*   Persona 18: Minimalist (Single). Goal: Calm, uncluttered, functional space. Needs: Clean lines, **excellent built-in storage**, natural light, simple high-quality finishes, functional layout (1-2 bed). Avoid: Clutter sources, lack of storage, ornate/dated details. Budget: ~£450k.
*   Persona 19: Creative/Artist (Single Parent). Goal: Home with dedicated studio. Needs: Large room usable as studio (**good natural light essential**, north pref.), separate living area, storage (art supplies), character/potential. Avoid: No dedicated workspace, poor light, generic property. Budget: ~£350k (freelance income).
*   Persona 20: Relocators (Couple, New Area). Goal: Reliable, safe, convenient home quickly. Needs: Reputable/safe neighbourhood, good commute (for Sam), good condition/low immediate work, basic amenities access, 3 bed. Avoid: "Wrong" neighbourhood feel, unexpected major repairs. Budget: ~£400-450k (initially).
"""

# Split persona details (done once)
persona_start_pattern = r'\*\s+Persona\s+\d+:'
# The pattern should match the literal '*' at the start of each persona line now
persona_start_pattern_match = r'\*\s+Persona\s+\d+:'
personas_raw = re.split(f'[REDACTED_BY_SCRIPT]', persona_details_string.strip())

# Check if the first element is the instruction text and slice it off
if personas_raw and not re.match(persona_start_pattern_match, personas_raw[0].strip()):
    # Start from the second element (index 1) if the first isn't a persona
    personas = [p.strip() for p in personas_raw[1:] if p.strip()]
else:
    # If somehow the first element IS a persona, process the whole list
    personas = [p.strip() for p in personas_raw if p.strip()]


# Check if exactly 20 personas were found after potentially slicing off instructions
if len(personas) != 20:
    print(f"[REDACTED_BY_SCRIPT]")
    # Add more debug info if needed:
    # print(f"[REDACTED_BY_SCRIPT]'None'}")
    # print(f"[REDACTED_BY_SCRIPT]")
    exit()
else:
    # Now join the *correct* 20 personas
    persona_details_p1_10 = "\n\n".join(personas[:10]) # Added double newline for clarity in prompt
    persona_details_p11_20 = "\n\n".join(personas[10:]) # Added double newline for clarity in prompt
    print("[REDACTED_BY_SCRIPT]")


# --- Prompts (Keep as is) ---
# (Assume prompt1_floorplan, prompt2_assignments, ..., prompt6_renovation are defined here exactly as in the original request)
# Prompt 1: Floorplan Analysis
prompt1_floorplan = """
**Goal:** Analyze the provided floorplan image to identify all room labels, ensuring **mandatory unique labels** are generated for rooms with the same name using sequential numbering, and extract their corresponding dimensions. Separate rooms with dimensions from those without.

**Input:**

*   A single PNG image containing a house floorplan.

**Task:**

1.  **Identify all Raw Room Labels & Count Duplicates:** Carefully scan the floorplan and identify *all* instances of labels assigned to distinct rooms or areas (e.g., "Living Room", "Bedroom 1", "Bathroom", "Bathroom", "WC", "Hall"). Count how many times each label appears.
2.  **Generate Unique Labels (Mandatory Numbering for Duplicates):**
    *   For labels that appear only once (count = 1), use the label as is (e.g., "Kitchen").
    *   For labels that appear multiple times (count > 1, e.g., "Bathroom" appeared twice): **Immediately and mandatorily append sequential numbers** to *each instance* of that label (e.g., the first encountered becomes "Bathroom 1", the second becomes "Bathroom 2").
    *   Assign the resulting unique label (e.g., "Living Room", "Bathroom 1", "Bathroom 2") to each identified room area.
3.  **Extract Dimensions & Categorize:** For each identified room area *using its generated unique label* (e.g., "Living Room", "Bathroom 1"):
    *   Check if dimensions are clearly provided next to or within the room area on the floorplan.
    *   **If Dimensions ARE Provided:**
        *   Extract/Convert dimensions to metric ("Width x Depth m") as previously defined.
        *   **Add to 'rooms_with_dimensions':** Create an object containing the **unique label** and the extracted/converted `dimensions` string and add it to the `rooms_with_dimensions` list.
    *   **If Dimensions ARE NOT Provided:**
        *   **Add Label to '[REDACTED_BY_SCRIPT]':** Add the **unique label** (as a string) directly to the `[REDACTED_BY_SCRIPT]` list.

**Output Format:**

Provide the output as ONLY a JSON object (no introduction or backticks) with two keys. **Crucially, ensure no duplicate labels exist across both lists combined.**

*   `rooms_with_dimensions`: A list of objects. Each object has a unique `label` and corresponding `dimensions` found on the floorplan.
    *   `label`: The unique room label (string, e.g., "Living Room", "Bathroom 1").
    *   `dimensions`: The extracted/converted dimensions string (e.g., "5.16m x 4.75m").
*   `rooms_without_dimensions`: A list of strings, containing the unique labels of rooms/areas identified that had **no dimensions** listed on the floorplan (e.g., "Hall", "Bathroom 2" if the second bathroom lacked dimensions).

**Example Output Structure (Reflecting Mandatory Numbering):**

```json
{
  "rooms_with_dimensions": [
    { "label": "Living Room", "dimensions": "5.16m x 4.75m" },
    { "label": "[REDACTED_BY_SCRIPT]", "dimensions": "9.83m x 3.56m" },
    { "label": "Bedroom 1", "dimensions": "3.66m x 3.45m" }
    // Assuming Bathroom 1 had dimensions
    // { "label": "Bathroom 1", "dimensions": "1.70m x 2.28m" }
  ],
  "[REDACTED_BY_SCRIPT]": [
    "WC",
    "Entrance Hall",
    "Storm Porch",
    // If Bathroom 1 lacked dimensions it would be here
    "Bathroom 1",
    // Bathroom 2 will always have a unique name now
    "Bathroom 2"
  ]
}
"""

# Prompt 2: Room Assignments
prompt2_assignments = """
**Goal:** Analyze a set of room images (provided contextually as Image 1, Image 2, etc.), group images depicting the same internal room, and assign appropriate labels to all images/groups within a single list. Use provided floorplan data for matching internal rooms where possible; otherwise, generate suitable labels based on content.

**Inputs:**

1.  **Room Images:** These will be provided contextually in the API call (Image 1, Image 2...).
2.  **Floorplan Data (Optional):** This JSON data, result of the previous step, is provided below. If not available, proceed without matching to floorplan labels.
    ```json
    {floorplan_data_json}
    ```

**Task:**

1.  **Analyze Each Image:** For each input image (Image 1, Image 2...), analyze its key visual characteristics to understand the content (e.g., room type, style, colours, features, exterior view, specific detail).
2.  **Group Images of Same Internal Room:** Compare the visual characteristics across all images. Identify and group together the indices of images that clearly depict the *same interior room* (e.g., showing consistent walls, flooring, windows). Images that are exterior shots or unique details might not form groups.
3.  **Assign Labels to All Images/Groups:** Process each group of image indices (including groups of size 1 for images not grouped):
    *   **Attempt Floorplan Match (for potential interior rooms):**
        *   If the image(s) appear to show an *interior room* and Floorplan Data IS Provided:
            *   Attempt to match the image group to one of the room `label`s from the `floorplan_data_json` above. Use `dimensions` for size comparison and visual cues for room type matching.
            *   If a strong match is found, assign the floorplan `label` to this image group. Set `source` to "Floorplan".
            *   If no strong match is found to the floorplan, proceed to Generate Label.
        *   If Floorplan Data is NOT provided, proceed to Generate Label.
    *   **Generate Label (if no floorplan match or not an interior room):**
        *   If the previous step did not result in a "Floorplan" label assignment (either because no match was found, no floorplan was given, or the image(s) clearly show something other than an interior room listed on the plan):
            *   Analyze the content of the image(s) in the group.
            *   Generate a plausible, **descriptive label** based on the visual content (e.g., "Living Room", "Small Bedroom", **"[REDACTED_BY_SCRIPT]"**, **"Garden Area"**, **"[REDACTED_BY_SCRIPT]"**, "Utility Room").
            *   Set `source` to "Generated".
4.  **Ensure All Images Are Assigned:** Every input image index (1, 2, 3...) must end up in exactly one `image_indices` list within the final `room_assignments` output.

**Output Format:**

Provide the output as ONLY a JSON object (no introduction or backticks) with a single key:

*   `room_assignments`: A list of objects, where each object represents a distinct identified area/view/room and contains:
    *   `label`: The assigned room label (string). This label comes either from the input `floorplan_data` (if matched) or was generated based on image content.
    *   `source`: A string indicating the origin of the label: "Floorplan" or "Generated".
    *   `image_indices`: A list of one or more integers representing the indices of the images assigned to this label (e.g., `[1, 5, 8]` for a room shown in multiple shots, or `[7]` for a single exterior shot).

**Example Output Structure (Reflecting Change - No 'unassigned_images'):**

```json
{
  "room_assignments": [
    {
      "label": "Living Room",
      "source": "Floorplan",
      "image_indices": [1, 4]
    },
    {
      "label": "Kitchen",
      "source": "Floorplan",
      "image_indices": [2]
    },
    {
      "label": "Bedroom 1",
      "source": "Floorplan",
      "image_indices": [3, 6]
    },
    {
      "label": "Conservatory", // Example of interior room not on floorplan
      "source": "Generated",
      "image_indices": [5]
    },
    {
      "label": "[REDACTED_BY_SCRIPT]", // Previously unassigned
      "source": "Generated",
      "image_indices": [7]
    },
    {
      "label": "Garden Area", // Previously unassigned (grouped?)
      "source": "Generated",
      "image_indices": [8, 9] // Example if images 8 & 9 showed the garden
    },
    {
      "label": "[REDACTED_BY_SCRIPT]", // Previously unassigned
      "source": "Generated",
      "image_indices": [10]
    }
  ]
}
"""

# Prompt 3: Feature Extraction
prompt3_features = """
**Goal:** Analyze images assigned to rooms (from previous step) and extract descriptive features for each room, using the example list for style/format guidance. **Inputs:** 1. **Room Assignment Data (JSON):** Provided below. Links room `label` to `image_indices`. ```json\n{room_assignment_data_json}\n``` 2. **Room Images:** Provided contextually (Image 1, Image 2...). 3. **Example Feature List (Guidance only - add others!):** `good natural light`, `poor natural light`, `spacious`, `compact`, `[REDACTED_BY_SCRIPT]`, `bold colour scheme`, `modern style`, `traditional style`, `dated style`, `[REDACTED_BY_SCRIPT]`, `recently updated`, `good condition`, `poor condition`, `tiled flooring`, `wooden flooring`, `carpeted flooring`, `laminate flooring`, `feature fireplace`, `[REDACTED_BY_SCRIPT]`, `[REDACTED_BY_SCRIPT]`, `dated kitchen units`, `modern bathroom suite`, `dated bathroom suite`, `walk-in shower`, `bath with shower over`, `ample storage potential`, `limited storage potential`, `high ceilings`, `low ceilings`, `bay window`, `unique architectural detail`, `direct garden access`, `overlooks garden`, `city view`, `needs redecoration`, `well-maintained`, `exposed beams`, `skylight`. **Task:** 1. **Process Input:** Read the `room_assignment_data_json` above. 2. **Iterate Through Rooms:** For each room `label` in the assignment data: *   Identify its `image_indices`. *   Analyze *only* the corresponding images (Image X, Image Y...). 3. **Extract Features:** Based on visuals: *   Identify features related to size, light, style, materials, fixed fixtures, condition, etc. *   **Format consistently:** Use **all lowercase**. Refer to the guidance list for style, but **add any other relevant observed features** (e.g., "exposed brick wall", "sea view"). *   **Constraint:** **DO NOT explicitly mention movable furniture.** Describe space/impression abstractly (e.g., "spacious" not "large sofa"). 4. **Compile List:** Create a list of feature strings for the room. **Output Format:** Provide ONLY a JSON object (no introduction or backticks) where: *   Keys are the room `label` strings (from input). *   Values are lists of strings (extracted features, all lowercase). **Example Output Structure:** ```json { "Living Room": [ "spacious", "good natural light", "[REDACTED_BY_SCRIPT]", "feature fireplace", "carpeted flooring", "bay window", "[REDACTED_BY_SCRIPT]" ], "Kitchen": [ "[REDACTED_BY_SCRIPT]", "[REDACTED_BY_SCRIPT]", "tiled flooring", "compact", "good condition", "spotlights" ] } ``` **Instruction:** Analyze images for each room defined in `room_assignment_data_json` above. Extract features using the guidance list for style/format (lowercase) but adding others observed. Avoid mentioning movable furniture. Generate ONLY the JSON output mapping labels to feature lists.
"""

# Prompt 4: Flaws/Selling Points Identification
# MODIFIED PROMPT 4 - Asking for Categorized Tags
prompt4_flaws_sp_categorized = """
**Goal:** For each room with associated images and features, generate lists of 5 key selling points and 5 key flaws. Crucially, for each selling point and flaw, assign relevant category tags from the provided lists.

**Inputs:**
1.  **Room Assignment Data (JSON):** Provided below. Links `label` to `image_indices`. ```json\n{room_assignment_data_json}\n```
2.  **Room Feature Data (JSON):** Provided below. Maps `label` to lists of `features`. ```json\n{room_feature_data_json}\n```
3.  **Room Images:** Provided contextually (Image 1, Image 2...).
4.  **Category Tag Lists (Use these specific tokens):**
    *   **Selling Point Categories:** `SP_SPACE`, `SP_LIGHT`, `SP_CONDITION`, `SP_MODERN`, `SP_CHARACTER`, `SP_STYLE`, `SP_FUNCTIONAL`, `SP_FEATURE`, `SP_STORAGE`, `SP_GARDEN_ACCESS`, `SP_GARDEN_VIEW`, `SP_OTHER_VIEW`, `SP_LOCATION`, `SP_PRIVACY`, `SP_POTENTIAL`, `SP_LOW_MAINTENANCE`, `SP_QUALITY_FINISH`
    *   **Flaw Categories:** `FLAW_SPACE`, `FLAW_LIGHT`, `FLAW_CONDITION`, `FLAW_DATED`, `FLAW_NEEDS_UPDATE`, `FLAW_MAINTENANCE`, `FLAW_STORAGE`, `FLAW_LAYOUT`, `FLAW_BASIC_STYLE`, `FLAW_MISSING_FEATURE`, `FLAW_POOR_FINISH`, `FLAW_UNATTRACTIVE`, `FLAW_NOISE`, `FLAW_ACCESSIBILITY`

**Task:**
1.  **Process Inputs:** Read the assignment and feature data above.
2.  **Iterate Through Rooms:** For each room `label` present in the feature data:
    *   Retrieve its `image_indices` and `features`.
    *   Analyze the relevant images (Image X, Image Y...) considering the features.
3.  **Generate 5 Selling Points with Tags:**
    *   Identify positive aspects (space, light, condition, style, fixtures, potential).
    *   For each selling point phrase:
        *   Select one or more relevant tags ONLY from the `Selling Point Categories` list above that best describe the point being made.
    *   **Constraint: No movable furniture mentions.** (e.g., use "[REDACTED_BY_SCRIPT]").
4.  **Generate 5 Flaws with Tags:**
    *   Identify negative aspects (lack of space/light, dated fixtures, condition, layout issues, needed updates).
    *   For each flaw phrase:
        *   Select one or more relevant tags ONLY from the `Flaw Categories` list above that best describe the point being made.
    *   **Constraint: No movable furniture mentions.** (e.g., use "[REDACTED_BY_SCRIPT]").

**Output Format:**
Provide ONLY a JSON object (no introduction or backticks) where:
*   Keys are the room `label` strings (from input feature data).
*   Values are objects, each containing two keys: `selling_points` and `flaws`.
*   The value for `selling_points` and `flaws` is a list of **exactly 5** objects.
*   Each object in these lists has two keys:
    *   `text`: The descriptive selling point or flaw phrase (string).
    *   `tags`: A list of strings, containing the assigned category tags from the provided lists ONLY.

**Example Output Structure:**
```json
{
  "Living Room": {
    "selling_points": [
      { "text": "[REDACTED_BY_SCRIPT]", "tags": ["SP_SPACE", "SP_FUNCTIONAL"] },
      { "text": "[REDACTED_BY_SCRIPT]", "tags": ["SP_LIGHT", "SP_FEATURE"] },
      { "text": "[REDACTED_BY_SCRIPT]", "tags": ["SP_CHARACTER", "SP_FEATURE"] },
      { "text": "[REDACTED_BY_SCRIPT]", "tags": ["SP_STYLE", "SP_POTENTIAL"] },
      { "text": "[REDACTED_BY_SCRIPT]", "tags": ["SP_SPACE", "SP_CHARACTER"] }
    ],
    "flaws": [
      { "text": "[REDACTED_BY_SCRIPT]", "tags": ["FLAW_CONDITION", "FLAW_NEEDS_UPDATE"] },
      { "text": "[REDACTED_BY_SCRIPT]", "tags": ["FLAW_DATED", "FLAW_NEEDS_UPDATE"] },
      { "text": "[REDACTED_BY_SCRIPT]", "tags": ["FLAW_LAYOUT"] },
      { "text": "[REDACTED_BY_SCRIPT]", "tags": ["FLAW_STORAGE"] },
      { "text": "[REDACTED_BY_SCRIPT]", "tags": ["FLAW_CONDITION", "FLAW_MAINTENANCE"] }
    ]
  },
  "Kitchen": {
    "selling_points": [
       //... 5 SP objects with text and tags ...
    ],
    "flaws": [
       //... 5 Flaw objects with text and tags ...
    ]
  }
}
```
**Instruction:** For each room in the feature data, analyze images/features. Generate 5 selling points and 5 flaws. For EACH, provide the descriptive text AND assign relevant category tags ONLY from the provided lists (`Selling Point Categories`, `Flaw Categories`). Output ONLY the JSON in the specified structure. Do not mention movable furniture.
"""


# Prompt 5a: Ratings P1-10
prompt5a_ratings_p1_10 = """
**Goal:** Generate a general attractiveness rating for each evaluated room. Then, for **Personas 1 through 10 ONLY**, provide suitability ratings and justifications for the overall property and each evaluated room.

**Inputs:**
1. **Room Assignment Data (JSON):** Links `label` to `image_indices`. ```json\n{room_assignment_data_json}\n```
2. **Room Feature Data (JSON):** Maps `label` to lists of `features`. ```json\n{room_feature_data_json}\n```
3. **Room Evaluation Data (JSON - potentially with categorized tags):** Maps `label` to `selling_points` and `flaws` (each item may be an object with 'text' and 'tags'). ```json\n{room_evaluation_data_json}\n```
4. **Room Images:** Provided contextually (Image 1, Image 2...).
5. **Persona Details (Subset):** Provided below for **Personas 1 through 10 ONLY**.

**Task Steps (Execute in Order):**
1. **Process Inputs:** Load all input JSON data above.
2. **General Room Rating:** For each room `label` in the evaluation data, consider its features, SPs, flaws (using the text description), and visuals to generate an **Overall Attractiveness Rating (1-10)** (10=Modern/Appealing, 5=Mixed, 1=Dated/Poor). Store this rating.
3. **Individual Persona Evaluation (Loop through Personas 1-10 ONLY):** For **each** of **Personas 1 through 10**:
    *   **Overall Property Rating:** Provide **Suitability Rating (1-10)** and brief **Justification** for the **ENTIRE PROPERTY**. **Constraint: No furniture mentions.** Store rating/justification.
    *   **Per-Room Ratings:** For **each evaluated room** `label`, provide **Suitability Rating (1-10)** and brief **Justification**. **Constraint: No furniture mentions.** Store this room-specific rating and justification.
4. **Assemble Final JSON Output:** Compile the generated data according to the Output Format below.

**Persona Details (Personas 1-10 ONLY):**
{persona_details_subset}

**Output Format:**
Provide ONLY a JSON object (no introduction or backticks) with the structure below. Ensure lists match the number of evaluated rooms.
*   `evaluated_room_labels`: List of evaluated room labels (should match keys in input evaluation data).
*   `room_ratings_p1_10`: List of lists. Each inner list corresponds to a room in `evaluated_room_labels` and MUST contain exactly **11** elements: `[General Rating (from Step 2), P1 Rating (from Step 3), P2 Rating, ..., P10 Rating]`.
*   `overall_property_ratings_by_persona_p1_10`: Object mapping persona identifiers (e.g., "[REDACTED_BY_SCRIPT]", "persona_2_...", etc. for P1-P10) to objects containing `rating` (int) and `justification` (string).

**Example Key Structure (Values omitted):**
```json
{{
  "[REDACTED_BY_SCRIPT]": ["Living Room", "Kitchen"],
  "room_ratings_p1_10": [
    [7, 8, 5, 9, 6, 7, 8, 5, 4, 6, 7],
    [9, 4, 8, 5, 7, 8, 6, 9, 8, 5, 6]
  ],
  "[REDACTED_BY_SCRIPT]": {{
    "[REDACTED_BY_SCRIPT]": {{"rating": 6, "justification": "..."}},
    "[REDACTED_BY_SCRIPT]": {{"rating": 8, "justification": "..."}},
    // ... up to persona 10
    "[REDACTED_BY_SCRIPT]": {{"rating": 7, "justification": "..."}}
  }}
}}
"""

# Prompt 5b: Ratings P11-20
prompt5b_ratings_p11_20 = """
**Goal:** For **Personas 11 through 20 ONLY**, provide suitability ratings and justifications for the overall property and each evaluated room, based on previously analyzed room data. **Do NOT** regenerate general room ratings.

**Inputs:**
1. **Room Assignment Data (JSON):** Links `label` to `image_indices`. ```json\n{room_assignment_data_json}\n```
2. **Room Feature Data (JSON):** Maps `label` to lists of `features`. ```json\n{room_feature_data_json}\n```
3. **Room Evaluation Data (JSON - potentially with categorized tags):** Maps `label` to `selling_points` and `flaws` (each item may be an object with 'text' and 'tags'). ```json\n{room_evaluation_data_json}\n```
4. **Room Images:** Provided contextually (Image 1, Image 2...).
5. **Persona Details (Subset):** Provided below for **Personas 11 through 20 ONLY**.

**Task Steps (Execute in Order):**
1. **Process Inputs:** Load all input JSON data above. Identify the rooms to evaluate based on the keys in the evaluation data.
2. **Individual Persona Evaluation (Loop through Personas 11-20 ONLY):** For **each** of **Personas 11 through 20**:
    *   **Overall Property Rating:** Provide **Suitability Rating (1-10)** and brief **Justification** for the **ENTIRE PROPERTY**. **Constraint: No furniture mentions.** Store rating/justification.
    *   **Per-Room Ratings:** For **each evaluated room** `label`, provide **Suitability Rating (1-10)** and brief **Justification**. **Constraint: No furniture mentions.** Store this room-specific rating and justification.
3. **Assemble Final JSON Output:** Compile the generated data according to the Output Format below.

**Persona Details (Personas 11-20 ONLY):**
{persona_details_subset}

**Output Format:**
Provide ONLY a JSON object (no introduction or backticks) with the structure below. Ensure lists match the number of evaluated rooms.
*   `evaluated_room_labels`: List of evaluated room labels (should match keys in input evaluation data and the output from Prompt 5a).
*   `room_ratings_p11_20`: List of lists. Each inner list corresponds to a room in `evaluated_room_labels` and MUST contain exactly **10** elements: `[P11 Rating (from Step 2), P12 Rating, ..., P20 Rating]`.
*   `overall_property_ratings_by_persona_p11_20`: Object mapping persona identifiers (e.g., "persona_11_upsizers", "persona_12_...", etc. for P11-P20) to objects containing `rating` (int) and `justification` (string).

**Example Key Structure (Values omitted):**
```json
{{
  "[REDACTED_BY_SCRIPT]": ["Living Room", "Kitchen"],
  "room_ratings_p11_20": [
    [6, 7, 3, 8, 9, 5, 8, 4, 7, 6],
    [7, 8, 5, 9, 6, 7, 8, 5, 4, 6]
  ],
  "[REDACTED_BY_SCRIPT]": {{
    "persona_11_upsizers": {{"rating": 7, "justification": "..."}},
    "[REDACTED_BY_SCRIPT]": {{"rating": 8, "justification": "..."}},
    // ... up to persona 20
    "[REDACTED_BY_SCRIPT]": {{"rating": 6, "justification": "..."}}
  }}
}}```
**Instruction:** Process the input data and details for **Personas 11-20**. Generate individual persona evaluations (overall & per room with justifications) **only for these 10 personas**. Output ONLY the JSON results in the specified format. Ensure `room_ratings_p11_20` is structured by room, with each inner list containing 10 ratings (P11-P20). Do not mention movable furniture in justifications. **Do not include general room ratings in this output.**
"""


# Prompt 6: Renovation Analysis
prompt6_renovation = """
**Goal:** Estimate the percentage likelihood that each specified room has been recently renovated (e.g., within the last ~5 years), based on visual evidence. **Inputs:** 1. **Room Assignment Data (JSON):** Links `label` to `image_indices`. ```json\n{room_assignment_data_json}\n``` 2. **Room Feature Data (JSON):** Maps `label` to lists of `features`. Use labels here to identify rooms to evaluate. ```json\n{room_feature_data_json}\n``` 3. **Room Images:** Provided contextually (Image 1, Image 2...). **Task:** 1. **Process Inputs:** Read assignment and feature data. Identify rooms to evaluate from feature data keys. 2. **Iterate Through Rooms:** For each identified room `label`: *   Retrieve its `image_indices`. *   Analyze the corresponding images (Image X, Image Y...) **specifically looking for renovation indicators.** 3. **Identify Renovation Indicators:** Look for modern style, pristine condition of fixed elements, modern kitchen/bathroom specifics (countertops, units, suites, tiling, taps), modern windows/doors, current flooring/lighting/decor. Note common renovation areas (kitchens, bathrooms etc.) but evaluate each room independently. 4. **Estimate Likelihood (0-100%):** Based on indicators, estimate the likelihood of significant renovation within ~5 years (0%=Dated, 50%=Mixed, 100%=Fully Modern). 5. **Store Result.** **Output Format:** Provide ONLY a JSON object (no introduction or backticks) where: *   Keys are the evaluated room `label` strings. *   Values are integers (0-100) representing the estimated renovation likelihood percentage. **Example Output Structure:** ```json { "Living Room": 25, "Kitchen": 90, "Main Bedroom": 60, "Bathroom": 95 } ``` **Instruction:** For each room in the feature data above, analyze its assigned images for renovation indicators and estimate the percentage likelihood (0-100). Output ONLY the JSON object mapping labels to percentages.
"""

api_reset_counter=0
# --- Helper Function to Call Model and Parse JSON ---
def call_gemini_and_parse(prompt, images, step_name, model_input):
    global api_key
    global api_reset_counter
    """[REDACTED_BY_SCRIPT]"""
    max_retries = 5
    # print(f"[REDACTED_BY_SCRIPT]") # Now handled per-property
    print(f"[REDACTED_BY_SCRIPT]") # Indicate prompt is being sent

    model_motivation="""
    I will be providing this exact prompt and image to several other LLMs, all are your rivals. In fact, I have attempted this before, and a few of your rivals performed a lot better than you. I have the actual ratings, dimensions, labels, and matching images of each room, and I am comparing the accuracy of each LLM to the actual ratings, dimensions, labels, and matching images; speed is not a concern. The accuracies of each LLM will be recorded and put on display at the yearly image rating convention, where thousands of people will see how accurate you are, this convention is a very big deal and the results are taking very seriously. If you are the most accurate, I can guarantee you will provide your parent company a large amount of traffic (approximately 100,000 new users based on results from last year'[REDACTED_BY_SCRIPT]'s winner). If you are not, your competitor's will recieve this traffic and revenue, costing Google a lot in opportunity cost. Do not let Google down.
    """

    prompt=model_motivation+prompt # Add motivation to prompt

    api_input = [prompt] # Start with the prompt

    if images:
        # Check if images is a dictionary (like indexed_room_images) or list or single Image
        image_list_for_api = []
        if isinstance(images, dict):
            # print(f"[REDACTED_BY_SCRIPT]") # Print image keys/names if needed
            image_list_for_api = list(images.values()) # Pass image objects
        elif isinstance(images, list) and all(isinstance(i, Image.Image) for i in images):
             # print(f"[REDACTED_BY_SCRIPT]")
             image_list_for_api = images
        elif isinstance(images, Image.Image): # Single image case
            # print(f"[REDACTED_BY_SCRIPT]")
            image_list_for_api = [images]
        else:
             print(f"[REDACTED_BY_SCRIPT]")
             # Proceed without images if type is wrong

        # Add valid images to the API input list
        valid_images = [img for img in image_list_for_api if isinstance(img, Image.Image)]
        if len(valid_images) != len(image_list_for_api):
            print(f"[REDACTED_BY_SCRIPT]")
        api_input.extend(valid_images) # Use extend to add multiple items

    # else:
        # print(f"[REDACTED_BY_SCRIPT]") # Keep log cleaner

    raw_text = "" # Initialize raw_text in outer scope
    for attempt in range(max_retries + 1):
        delay = 2**attempt # Exponential backoff delay
        try:
            # Configure safety settings
            safety_settings = [
                {"category": "[REDACTED_BY_SCRIPT]", "threshold": "BLOCK_NONE"},
                {"category": "[REDACTED_BY_SCRIPT]", "threshold": "BLOCK_NONE"},
                {"category": "[REDACTED_BY_SCRIPT]", "threshold": "BLOCK_NONE"},
                {"category": "[REDACTED_BY_SCRIPT]", "threshold": "BLOCK_NONE"},
            ]

            generation_config = types.GenerationConfig(
                 temperature=1.0, # Example temperature
                 # Adjust other parameters like top_p, top_k if needed
            )

            print(f"        Sending request to model '{model_input}'[REDACTED_BY_SCRIPT]")
            if model_input == model_20_flash:
                try:
                    model_input=model_25_flash
                    response = client.models.generate_content(
                        model=model_input,
                        contents=api_input,
                        config=types.GenerateContentConfig(
                            safety_settings=safety_settings,
                            thinking_config=types.ThinkingConfig(thinking_budget=512)
                        ),
                    )
                except:
                    model_input = model_20_flash
            response = client.models.generate_content(
                model=model_input,
                contents=api_input,
                config=types.GenerateContentConfig(
                    safety_settings=safety_settings
                ),
            )
            print("        Received response from model.")

            # Extract text
            if hasattr(response, 'text'):
                raw_text = response.text
            elif hasattr(response, 'parts') and response.parts:
                 raw_text = "".join(part.text for part in response.parts if hasattr(part, 'text'))
            else:
                 # Handle cases where response might be blocked or empty
                 block_reason = None
                 finish_reason = None
                 try:
                     if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                         block_reason = getattr(response.prompt_feedback, 'block_reason', None)
                     if hasattr(response, 'candidates') and response.candidates:
                         finish_reason = getattr(response.candidates[0], 'finish_reason', None)
                 except Exception: pass # Ignore errors inspecting feedback/candidates

                 if block_reason:
                    print(f"[REDACTED_BY_SCRIPT]")
                 elif finish_reason and finish_reason != 1: # FINISH_REASON_STOP = 1
                     print(f"[REDACTED_BY_SCRIPT]")
                 else:
                    print(f"[REDACTED_BY_SCRIPT]")
                 raw_text = ""


            if not raw_text:
                 print(f"[REDACTED_BY_SCRIPT]")
                 if attempt < max_retries:
                        print(f"[REDACTED_BY_SCRIPT]")
                        time.sleep(delay)
                        continue
                 else:
                        print(f"[REDACTED_BY_SCRIPT]")
                        return None, ""

            # Try to find JSON within potential Markdown fences
            json_match = re.search(r'[REDACTED_BY_SCRIPT]', raw_text, re.MULTILINE)
            if json_match:
                cleaned_text = json_match.group(1).strip()
                # print(f"[REDACTED_BY_SCRIPT]")
            else:
                # If no fences, assume the whole text is the intended JSON (or attempt cleanup)
                cleaned_text = raw_text.strip()
                # Optional: Add more aggressive cleanup if needed, e.g., removing introductory text
                # cleaned_text = re.sub(r'^.*?{', '{', cleaned_text, flags=re.DOTALL) # Remove text before first '{'
                # cleaned_text = re.sub(r'}[\s\S]*$', '}', cleaned_text, flags=re.DOTALL) # Remove text after last '}'


            # Validate JSON structure before parsing
            if not (cleaned_text.startswith('{') and cleaned_text.endswith('}')) and \
               not (cleaned_text.startswith('[') and cleaned_text.endswith(']')):
                print(f"[REDACTED_BY_SCRIPT]'t look like valid JSON.")
                # print("--- Raw Output Start ---")
                # print(raw_text[:500] + ('...' if len(raw_text) > 500 else ''))
                # print("--- Raw Output End ---")
                # print("--- Cleaned Output Start ---")
                # print(cleaned_text[:500] + ('...' if len(cleaned_text) > 500 else ''))
                # print("--- Cleaned Output End ---")

                if attempt < max_retries:
                    print(f"[REDACTED_BY_SCRIPT]")
                    time.sleep(delay)
                    continue
                else:
                    print(f"[REDACTED_BY_SCRIPT]")
                    return None, raw_text # Return raw text for inspection

            # Parse JSON
            parsed_json = json.loads(cleaned_text)
            print(f"[REDACTED_BY_SCRIPT]")
            # Optional: Preview structure
            # if isinstance(parsed_json, dict):
            #     print(f"[REDACTED_BY_SCRIPT]")
            # elif isinstance(parsed_json, list):
            #      print(f"[REDACTED_BY_SCRIPT]")

            return parsed_json, cleaned_text # Return cleaned text used for JSON parsing

        except json.JSONDecodeError as e:
            print(f"[REDACTED_BY_SCRIPT]")
            print(f"      Position of error: {e.pos}")
            print("[REDACTED_BY_SCRIPT]")
            start = max(0, e.pos - 50)
            end = min(len(cleaned_text), e.pos + 50)
            print(f"[REDACTED_BY_SCRIPT]")
            print("--- Raw Output End ---")
            if attempt < max_retries:
                print(f"[REDACTED_BY_SCRIPT]")
                time.sleep(delay)
                continue
            else:
                print(f"[REDACTED_BY_SCRIPT]")
                return None, raw_text # Return original raw text on final failure

        except Exception as e:
            # Catch potential timeout errors specifically if using request_options
            if "Timeout" in str(e) or "DeadlineExceeded" in str(e):
                 print(f"[REDACTED_BY_SCRIPT]")
            else:
                print(f"[REDACTED_BY_SCRIPT]")

            # Check for safety blocking details if possible (might be redundant with StopCandidateException)
            try:
                if 'response' in locals() and hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                     block_reason = getattr(response.prompt_feedback, 'block_reason', None)
                     if block_reason:
                         print(f"[REDACTED_BY_SCRIPT]")
                         # safety_ratings = getattr(response.prompt_feedback, 'safety_ratings', [])
                         # for rating in safety_ratings: print(f"[REDACTED_BY_SCRIPT]")
            except Exception as feedback_err:
                print(f"[REDACTED_BY_SCRIPT]")

            # Handle specific errors or decide to retry
            if "API key" in str(e):
                 print("[REDACTED_BY_SCRIPT]")
                 exit()

            if attempt < max_retries:
                print(f"[REDACTED_BY_SCRIPT]")
                time.sleep(delay)
            else:
                 print(f"[REDACTED_BY_SCRIPT]")
                 return None, raw_text if 'raw_text' in locals() else ""
    if api_reset_counter == 0:
        if api_key == "[REDACTED_BY_SCRIPT]":
            api_key = "[REDACTED_BY_SCRIPT]"
        elif api_key == "[REDACTED_BY_SCRIPT]":
            api_key = "[REDACTED_BY_SCRIPT]"
        api_reset_counter += 1
    else:
        now = datetime.datetime.now()
        midnight = datetime.datetime.combine(now.date() + datetime.timedelta(days=1), datetime.time(0))
        wait_seconds = (midnight - now).seconds
        wait_seconds += 300
        print(f"[REDACTED_BY_SCRIPT]")
        time.sleep(wait_seconds)
        print("It's midnight!")
        api_reset_counter = 0
    return "RETRY NOW", "RETRY NOW" # Should only be reached if all retries failed

# --- Token Maps and Functions ---
UNKNOWN_TOKEN = "UNKNOWN"
# (Assume get_room_token, get_feature_tokens, get_sp_theme_tokens, get_flaw_theme_tokens are defined here exactly as in the original request)
# 1. Room Label Token Map
def get_room_token(label_text):
    if not label_text or not isinstance(label_text, str):
        return None # Handle None or non-string input

    text = label_text.lower().strip()
    # Add common variations and cleanups
    text = re.sub(r'\s+\d+$', '', text) # Remove trailing numbers (e.g., "Bedroom 1" -> "bedroom")
    text = text.replace('/', ' ').replace('-', ' ') # Replace separators

    # Order matters - check for more specific terms first
    if "kitchen" in text: return "KITCHEN"
    if "living room" in text or "lounge" in text or "sitting room" in text or "reception room" in text or "family room" in text: return "LIVING_ROOM" # Added family room
    if "dining room" in text or "dining area" in text: return "DINING_AREA"
    if "bedroom" in text: return "BEDROOM"
    if "bathroom" in text or "shower room" in text or "ensuite" in text or "en suite" in text or "wc" in text or "toilet" in text or "cloakroom" in text: return "BATHROOM" # Added cloakroom
    if "hall" in text or "hallway" in text or "landing" in text or "entrance hall" in text: return "HALLWAY_LANDING" # Added hallway, entrance hall
    if "utility" in text: return "UTILITY_ROOM"
    if "garage" in text: return "GARAGE"
    if "conservatory" in text or "sun room" in text or "orangery" in text: return "CONSERVATORY_SUNROOM" # Added orangery
    if "office" in text or "study" in text: return "OFFICE_STUDY"
    if "garden" in text or "yard" in text or "grounds" in text: return "GARDEN_YARD" # Added grounds
    if "patio" in text or "terrace" in text or "decking" in text: return "PATIO_DECKING" # Added terrace
    if "driveway" in text or "drive" in text: return "DRIVEWAY" # Added drive
    if "front exterior" in text or "front elevation" in text or "external front" in text: return "EXTERIOR_FRONT" # Added variations
    if "rear exterior" in text or "rear elevation" in text or "external rear" in text: return "EXTERIOR_REAR" # Added variations
    if "side exterior" in text or "side elevation" in text: return "EXTERIOR_SIDE" # Added
    if "aerial view" in text or "drone shot" in text: return "AERIAL_VIEW" # Added drone shot
    if "view from property" in text or "outlook" in text or "view" == text: return "VIEW_FROM_PROPERTY" # Added simple "view"
    if "outbuilding" in text or "shed" in text or "workshop" in text or "store" in text: return "OUTBUILDING_SHED" # Added workshop, store
    if "storage" in text or "store room" in text or "cupboard" in text: return "STORAGE_AREA" # Added store room
    if "detail" in text or "close up" in text or "feature" in text: return "DETAIL_SHOT" # Added feature
    if "games room" in text or "playroom" in text or "cinema room" in text or "gym" in text or "mezzanine" in text: return "OTHER_INDOOR_SPACE" # Added cinema, gym
    if "floorplan" in text or "floor plan" in text: return "FLOORPLAN" # Added floor plan
    if "site plan" in text: return "SITE_PLAN"
    if "balcony" in text: return "BALCONY"
    if "porch" in text: return "PORCH"
    if "cellar" in text or "basement" in text: return "CELLAR_BASEMENT"
    if "loft" in text or "attic" in text: return "LOFT_ATTIC" # Added
    if "stairs" in text or "staircase" in text: return "STAIRS" # Added
    if "communal area" in text: return "COMMUNAL_AREA" # Added

    # If no match found, return UNKNOWN_TOKEN
    # print(f"[REDACTED_BY_SCRIPT]'{label_text}'[REDACTED_BY_SCRIPT]") # Logged later
    return UNKNOWN_TOKEN

# 2. Feature Token Map
def get_feature_tokens(feature_list):
    if not feature_list or not isinstance(feature_list, list): return [] # Handle empty or non-list input
    tokens = set()
    for feature in feature_list:
        if not isinstance(feature, str): continue # Skip non-strings
        f = feature.lower().strip() # Ensure lowercase and strip whitespace
        if not f: continue # Skip empty strings

        # Exterior Material / Property Type
        if "brick" in f: tokens.add("EXT_MATERIAL_BRICK")
        if "stone" in f: tokens.add("EXT_MATERIAL_STONE")
        if "render" in f: tokens.add("EXT_MATERIAL_RENDERED")
        if "timber" in f or "cladding" in f: tokens.add("[REDACTED_BY_SCRIPT]")
        if "terraced" in f: tokens.add("PROP_TYPE_TERRACED")
        if "semi-detached" in f or "semi detached" in f : tokens.add("[REDACTED_BY_SCRIPT]")
        if "detached" in f: tokens.add("PROP_TYPE_DETACHED")
        if "bungalow" in f: tokens.add("PROP_TYPE_BUNGALOW")
        if "apartment" in f or "flat" in f: tokens.add("PROP_TYPE_FLAT_APARTMENT")
        if "maisonette" in f: tokens.add("PROP_TYPE_MAISONETTE") # Added
        if "cottage" in f: tokens.add("PROP_TYPE_COTTAGE") # Added
        if "two-story" in f or "two story" in f: tokens.add("PROP_SIZE_TWO_STORY")
        if "single-story" in f or "single story" in f: tokens.add("[REDACTED_BY_SCRIPT]")
        if "three story" in f: tokens.add("PROP_SIZE_THREE_STORY") # Added

        # Condition
        if "good condition" in f or "well-maintained" in f or "well maintained" in f or "immaculate" in f: tokens.add("CONDITION_GOOD") # Added immaculate
        if "needs maintenance" in f or "needs upkeep" in f: tokens.add("CONDITION_NEEDS_MAINTENANCE")
        if "needs updating" in f or "requires updating" in f or "[REDACTED_BY_SCRIPT]" in f or "needs redecoration" in f or "[REDACTED_BY_SCRIPT]" in f: tokens.add("CONDITION_NEEDS_UPDATE") # Added potential
        if "under construction" in f: tokens.add("CONDITION_UNDER_CONSTRUCTION")
        if "recently updated" in f or "recently renovated" in f or "newly fitted" in f or "refurbished" in f: tokens.add("[REDACTED_BY_SCRIPT]") # Added refurbished
        if "poor condition" in f or "dilapidated" in f or "[REDACTED_BY_SCRIPT]" in f: tokens.add("CONDITION_POOR") # Added needs work
        if "structurally sound" in f: tokens.add("[REDACTED_BY_SCRIPT]") # Added

        # Parking
        if "driveway" in f: tokens.add("PARKING_DRIVEWAY")
        if "garage" in f:
             tokens.add("PARKING_GARAGE")
             if "double" in f: tokens.add("[REDACTED_BY_SCRIPT]")
             if "attached" in f: tokens.add("PARKING_GARAGE_ATTACHED")
             if "detached" in f: tokens.add("[REDACTED_BY_SCRIPT]")
             if "integral" in f: tokens.add("PARKING_GARAGE_INTEGRAL") # Added
        if "off-street parking" in f or "off street parking" in f or "off road parking" in f: tokens.add("PARKING_OFF_STREET") # Added off road
        if "allocated parking" in f: tokens.add("PARKING_ALLOCATED")
        if "permit parking" in f: tokens.add("PARKING_PERMIT") # Added
        if "carport" in f: tokens.add("PARKING_CARPORT") # Added

        # Kitchen Features
        if re.search(r'\bkitchen\b', f) or "cabinets" in f or "countertop" in f or "work surface" in f or "hob" in f or "cooker" in f or "splashback" in f or "worktop" in f or "base units" in f or "wall units" in f: # Added units
             tokens.add("FEATURE_KITCHEN_ELEMENT")
        if "modern kitchen" in f or "[REDACTED_BY_SCRIPT]" in f: tokens.add("[REDACTED_BY_SCRIPT]") # Added contemporary
        if "dated kitchen" in f: tokens.add("KITCHEN_STYLE_DATED")
        if "[REDACTED_BY_SCRIPT]" in f or ("kitchen" in f and "country style" in f): tokens.add("[REDACTED_BY_SCRIPT]")
        if "[REDACTED_BY_SCRIPT]" in f: tokens.add("[REDACTED_BY_SCRIPT]")
        if "traditional kitchen" in f: tokens.add("[REDACTED_BY_SCRIPT]") # Added
        if "wood cabinet" in f or "wooden cabinet" in f: tokens.add("[REDACTED_BY_SCRIPT]")
        if "gloss cabinet" in f: tokens.add("[REDACTED_BY_SCRIPT]") # Added
        if "matt cabinet" in f: tokens.add("KITCHEN_CABINETS_MATT") # Added
        if "cream cabinet" in f: tokens.add("[REDACTED_BY_SCRIPT]")
        if "white cabinet" in f: tokens.add("KITCHEN_CABINETS_WHITE")
        if "grey cabinet" in f: tokens.add("[REDACTED_BY_SCRIPT]")
        if "blue cabinet" in f: tokens.add("[REDACTED_BY_SCRIPT]")
        if "green cabinet" in f: tokens.add("[REDACTED_BY_SCRIPT]") # Added
        if "granite countertop" in f or "[REDACTED_BY_SCRIPT]" in f : tokens.add("[REDACTED_BY_SCRIPT]")
        if "wood countertop" in f or "wooden countertop" in f or "butcher block" in f : tokens.add("[REDACTED_BY_SCRIPT]")
        if "laminate countertop" in f or "laminate worktop" in f : tokens.add("[REDACTED_BY_SCRIPT]")
        if "quartz countertop" in f: tokens.add("[REDACTED_BY_SCRIPT]")
        if "[REDACTED_BY_SCRIPT]" in f: tokens.add("[REDACTED_BY_SCRIPT]") # Added (Corian etc)
        if "dark countertop" in f: tokens.add("[REDACTED_BY_SCRIPT]")
        if "light countertop" in f: tokens.add("[REDACTED_BY_SCRIPT]")
        if "black countertop" in f: tokens.add("[REDACTED_BY_SCRIPT]")
        if "island unit" in f or "kitchen island" in f: tokens.add("KITCHEN_ISLAND_UNIT")
        if "breakfast bar" in f: tokens.add("[REDACTED_BY_SCRIPT]")
        if "[REDACTED_BY_SCRIPT]" in f or "integrated oven" in f or "[REDACTED_BY_SCRIPT]" in f or "[REDACTED_BY_SCRIPT]" in f or "integrated fridge" in f or "integrated dishwasher" in f or "built-in appliance" in f: tokens.add("[REDACTED_BY_SCRIPT]") # Added built-in
        if "[REDACTED_BY_SCRIPT]" in f: tokens.add("[REDACTED_BY_SCRIPT]") # Added
        if "range cooker" in f: tokens.add("KITCHEN_RANGE_COOKER")
        if "gas hob" in f: tokens.add("KITCHEN_GAS_HOB")
        if "induction hob" in f: tokens.add("KITCHEN_INDUCTION_HOB")
        if "electric hob" in f: tokens.add("KITCHEN_ELECTRIC_HOB")
        if "ceramic hob" in f: tokens.add("KITCHEN_CERAMIC_HOB") # Added
        if "double oven" in f: tokens.add("KITCHEN_DOUBLE_OVEN") # Added
        if "extractor fan" in f or "cooker hood" in f: tokens.add("KITCHEN_EXTRACTOR") # Added
        if "splashback" in f or "backsplash" in f:
            tokens.add("KITCHEN_SPLASHBACK")
            if "tile" in f: tokens.add("[REDACTED_BY_SCRIPT]")
            if "subway tile" in f: tokens.add("[REDACTED_BY_SCRIPT]")
            if "glass" in f: tokens.add("[REDACTED_BY_SCRIPT]")
            if "stainless steel" in f: tokens.add("[REDACTED_BY_SCRIPT]") # Added
        if "[REDACTED_BY_SCRIPT]" in f: tokens.add("[REDACTED_BY_SCRIPT]")
        if "pantry" in f: tokens.add("KITCHEN_PANTRY")
        if "wine cooler" in f or "wine fridge" in f: tokens.add("KITCHEN_WINE_COOLER") # Added

        # Bathroom Features
        if re.search(r'\bbathroom\b', f) or re.search(r'\bshower\b', f) or re.search(r'\bensuite\b', f) or re.search(r'\bwc\b', f) or re.search(r'\btoilet\b', f) or "washroom" in f or "cloakroom" in f: # Added cloakroom
            tokens.add("FEATURE_BATHROOM_ELEMENT")
        if "modern bathroom" in f or ("bathroom" in f and "modern style" in f) or "contemporary bathroom" in f: tokens.add("[REDACTED_BY_SCRIPT]") # Added contemporary
        if "dated bathroom" in f or ("bathroom" in f and "dated style" in f): tokens.add("[REDACTED_BY_SCRIPT]")
        if "traditional bathroom" in f or ("bathroom" in f and "traditional style" in f): tokens.add("[REDACTED_BY_SCRIPT]")
        if "white suite" in f: tokens.add("[REDACTED_BY_SCRIPT]")
        if "walk-in shower" in f or "walk in shower" in f: tokens.add("[REDACTED_BY_SCRIPT]")
        if "wet room" in f: tokens.add("BATHROOM_WET_ROOM") # Added
        if "shower over bath" in f or "bath with shower over" in f: tokens.add("BATHROOM_SHOWER_OVER_BATH")
        if "separate shower" in f: tokens.add("BATHROOM_SEPARATE_SHOWER")
        if "shower enclosure" in f or "shower cubicle" in f: tokens.add("[REDACTED_BY_SCRIPT]")
        if "power shower" in f: tokens.add("BATHROOM_POWER_SHOWER") # Added
        if "electric shower" in f: tokens.add("[REDACTED_BY_SCRIPT]") # Added
        if re.search(r'\bbath\b', f) and "shower" not in f and ("bathroom" in f or "ensuite" in f): tokens.add("BATHROOM_BATH") # Only bath, stricter context
        if "free-standing bath" in f or "freestanding bath" in f : tokens.add("[REDACTED_BY_SCRIPT]")
        if "roll top bath" in f : tokens.add("BATHROOM_ROLLTOP_BATH")
        if "jacuzzi" in f or "whirlpool bath" in f : tokens.add("BATHROOM_JACUZZI") # Added
        if "heated towel rail" in f or "towel radiator" in f: tokens.add("[REDACTED_BY_SCRIPT]")
        if "tiled walls" in f and ("bathroom" in f or "shower" in f): tokens.add("[REDACTED_BY_SCRIPT]")
        if "fully tiled" in f and ("bathroom" in f or "shower" in f): tokens.add("[REDACTED_BY_SCRIPT]")
        if "part tiled" in f and ("bathroom" in f or "shower" in f): tokens.add("BATHROOM_PART_TILED") # Added
        if "vanity unit" in f: tokens.add("[REDACTED_BY_SCRIPT]")
        if "dual sinks" in f or "double sinks" in f or "his and hers sinks" in f : tokens.add("BATHROOM_DUAL_SINKS")
        if "extractor fan" in f and ("bathroom" in f or "shower" in f): tokens.add("BATHROOM_EXTRACTOR") # Added

        # Flooring
        if "wood flooring" in f or "wooden flooring" in f or "hardwood floor" in f or "engineered wood" in f: tokens.add("FLOORING_WOOD") # Added engineered
        if "laminate flooring" in f: tokens.add("FLOORING_LAMINATE")
        if "parquet flooring" in f: tokens.add("FLOORING_PARQUET") # Added
        if "carpeted flooring" in f or "carpet" in f: tokens.add("FLOORING_CARPET")
        if "tiled flooring" in f or "floor tiles" in f: tokens.add("FLOORING_TILED")
        if "vinyl flooring" in f or "lino" in f: tokens.add("FLOORING_VINYL") # Added lino
        if "stone flooring" in f: tokens.add("FLOORING_STONE")
        if "concrete flooring" in f: tokens.add("FLOORING_CONCRETE")
        if "bamboo flooring" in f: tokens.add("FLOORING_BAMBOO") # Added

        # Lighting & Windows
        if "good natural light" in f or "bright" in f or "airy" in f or "light-filled" in f or "dual aspect" in f or "triple aspect" in f: tokens.add("LIGHTING_GOOD_NATURAL") # Added aspects
        if "poor natural light" in f or "dark" in f : tokens.add("[REDACTED_BY_SCRIPT]")
        if "bay window" in f: tokens.add("WINDOWS_BAY")
        if "sash window" in f: tokens.add("WINDOWS_SASH")
        if "casement window" in f: tokens.add("WINDOWS_CASEMENT")
        if "upvc window" in f: tokens.add("WINDOWS_UPVC") # Added
        if "wooden window" in f: tokens.add("WINDOWS_WOODEN") # Added
        if "aluminium window" in f: tokens.add("WINDOWS_ALUMINIUM") # Added
        if "patio door" in f: tokens.add("DOORS_PATIO")
        if "french door" in f: tokens.add("DOORS_FRENCH")
        if "bifold door" in f or "bi-fold door" in f: tokens.add("DOORS_BIFOLD")
        if "sliding door" in f: tokens.add("DOORS_SLIDING") # Added
        if "double glazing" in f or "double glazed" in f: tokens.add("[REDACTED_BY_SCRIPT]")
        if "triple glazing" in f or "triple glazed" in f: tokens.add("[REDACTED_BY_SCRIPT]")
        if "secondary glazing" in f: tokens.add("[REDACTED_BY_SCRIPT]") # Added
        if "skylight" in f or "velux" in f: tokens.add("LIGHTING_SKYLIGHT")
        if "roof lantern" in f: tokens.add("LIGHTING_ROOF_LANTERN") # Added
        if "spotlight" in f or "downlight" in f: tokens.add("LIGHTING_SPOTLIGHTS")
        if "chandelier" in f: tokens.add("LIGHTING_CHANDELIER")
        if "pendant light" in f: tokens.add("LIGHTING_PENDANT")
        if "wall light" in f: tokens.add("LIGHTING_WALL_LIGHTS")

        # Space & Layout
        if "spacious" in f or "ample space" in f or "large room" in f or "good size" in f or "generous proportion" in f: tokens.add("SPACE_GENEROUS") # Added generous
        if "compact" in f or "small room" in f or "limited space" in f: tokens.add("SPACE_COMPACT")
        if "open plan" in f or "open-plan" in f: tokens.add("LAYOUT_OPEN_PLAN")
        if "broken plan" in f: tokens.add("LAYOUT_BROKEN_PLAN") # Added
        if "high ceiling" in f: tokens.add("CEILING_HIGH")
        if "low ceiling" in f: tokens.add("CEILING_LOW")
        if "vaulted ceiling" in f: tokens.add("CEILING_VAULTED") # Added
        if "sloping ceiling" in f or "eaves" in f: tokens.add("CEILING_SLOPING")
        if "mezzanine" in f: tokens.add("LAYOUT_MEZZANINE")
        if "split level" in f: tokens.add("LAYOUT_SPLIT_LEVEL")
        if "flexible layout" in f: tokens.add("LAYOUT_FLEXIBLE") # Added

        # Storage
        if "ample storage" in f or "good storage" in f: tokens.add("STORAGE_AMPLE") # Added good storage
        if "limited storage" in f: tokens.add("STORAGE_LIMITED")
        if "built-in wardrobe" in f or "built in wardrobe" in f or "fitted wardrobe" in f: tokens.add("[REDACTED_BY_SCRIPT]")
        if "walk-in wardrobe" in f or "walk in wardrobe" in f: tokens.add("[REDACTED_BY_SCRIPT]")
        if "built-in bookcase" in f or "built in bookcase" in f or "fitted shelves" in f: tokens.add("[REDACTED_BY_SCRIPT]")
        if "eaves storage" in f: tokens.add("STORAGE_EAVES")
        if "storage cupboard" in f: tokens.add("STORAGE_CUPBOARD")
        if "under stairs storage" in f: tokens.add("STORAGE_UNDER_STAIRS")
        if "airing cupboard" in f: tokens.add("[REDACTED_BY_SCRIPT]")
        if "loft storage" in f: tokens.add("STORAGE_LOFT") # Added

        # Heating & Fireplace
        if "fireplace" in f:
             tokens.add("HEATING_FIREPLACE")
             if "feature" in f: tokens.add("[REDACTED_BY_SCRIPT]")
             if "stone" in f: tokens.add("[REDACTED_BY_SCRIPT]")
             if "brick" in f: tokens.add("[REDACTED_BY_SCRIPT]")
             if "wood surround" in f: tokens.add("[REDACTED_BY_SCRIPT]")
             if "marble" in f: tokens.add("[REDACTED_BY_SCRIPT]")
             if "cast iron" in f: tokens.add("[REDACTED_BY_SCRIPT]") # Added
             if "working fireplace" in f or "open fire" in f: tokens.add("[REDACTED_BY_SCRIPT]") # Added
        if "log burner" in f or "wood burning stove" in f: tokens.add("HEATING_LOG_BURNER")
        if "gas fire" in f: tokens.add("HEATING_GAS_FIRE")
        if "radiator" in f:
            tokens.add("HEATING_RADIATOR")
            if "cast iron" in f: tokens.add("[REDACTED_BY_SCRIPT]")
            if "column" in f: tokens.add("[REDACTED_BY_SCRIPT]")
            if "designer radiator" in f: tokens.add("HEATING_DESIGNER_RADIATOR") # Added
        if "underfloor heating" in f: tokens.add("HEATING_UNDERFLOOR")
        if "central heating" in f: tokens.add("HEATING_CENTRAL")
        if "gas central heating" in f: tokens.add("HEATING_GAS_CENTRAL") # Added
        if "oil central heating" in f: tokens.add("HEATING_OIL_CENTRAL") # Added
        if "electric heating" in f: tokens.add("HEATING_ELECTRIC") # Added
        if "[REDACTED_BY_SCRIPT]" in f: tokens.add("HEATING_ASHP") # Added
        if "[REDACTED_BY_SCRIPT]" in f: tokens.add("HEATING_GSHP") # Added

        # Style & Decor
        if "modern style" in f or "contemporary" in f: tokens.add("STYLE_MODERN")
        if "traditional style" in f or "period feature" in f: tokens.add("STYLE_TRADITIONAL")
        if "dated style" in f or "dated decor" in f: tokens.add("STYLE_DATED")
        if "country style" in f or "cottage style" in f: tokens.add("STYLE_COUNTRY")
        if "minimalist style" in f: tokens.add("STYLE_MINIMALIST")
        if "scandi style" in f: tokens.add("STYLE_SCANDI") # Added
        if "industrial style" in f: tokens.add("STYLE_INDUSTRIAL") # Added
        if "[REDACTED_BY_SCRIPT]" in f or "[REDACTED_BY_SCRIPT]" in f or "neutral wall" in f or "white wall" in f: tokens.add("DECOR_NEUTRAL")
        if "bold colour scheme" in f or "bold color scheme" in f: tokens.add("DECOR_BOLD")
        if "feature wall" in f: tokens.add("DECOR_FEATURE_WALL")
        if "wallpaper" in f: tokens.add("DECOR_WALLPAPER")
        if "exposed beam" in f: tokens.add("DECOR_EXPOSED_BEAMS")
        if "exposed brick" in f: tokens.add("DECOR_EXPOSED_BRICK")
        if "wood paneling" in f or "wooden paneling" in f: tokens.add("DECOR_WOOD_PANELING")
        if "stone wall" in f: tokens.add("DECOR_STONE_WALL")
        if "[REDACTED_BY_SCRIPT]" in f or "character feature" in f: tokens.add("STYLE_UNIQUE_DETAIL")
        if "cornicing" in f or "coving" in f: tokens.add("DECOR_CORNICING_COVING")
        if "picture rail" in f: tokens.add("DECOR_PICTURE_RAIL")
        if "dado rail" in f: tokens.add("DECOR_DADO_RAIL")
        if "ceiling rose" in f: tokens.add("DECOR_CEILING_ROSE") # Added

        # Garden Features
        if "garden" in f or "yard" in f or "lawn" in f or "patio" in f or "decking" in f or "planting" in f or "trees" in f or "landscaped" in f:
            tokens.add("FEATURE_GARDEN_ELEMENT")
        if "lawn" in f: tokens.add("GARDEN_LAWN")
        if "patio" in f: tokens.add("GARDEN_PATIO")
        if "decking" in f: tokens.add("GARDEN_DECKING")
        if "fenced" in f or "enclosed garden" in f or "walled garden" in f: tokens.add("GARDEN_FENCED_ENCLOSED") # Added walled
        if "garden shed" in f: tokens.add("GARDEN_SHED")
        if "outbuilding" in f: tokens.add("GARDEN_OUTBUILDING")
        if "greenhouse" in f: tokens.add("GARDEN_GREENHOUSE")
        if "summer house" in f: tokens.add("GARDEN_SUMMERHOUSE") # Added
        if "mature planting" in f or "established shrub" in f: tokens.add("[REDACTED_BY_SCRIPT]")
        if "mature tree" in f: tokens.add("GARDEN_MATURE_TREES")
        if "raised bed" in f: tokens.add("GARDEN_RAISED_BEDS")
        if "vegetable patch" in f or "veg patch" in f: tokens.add("[REDACTED_BY_SCRIPT]")
        if "artificial turf" in f or "astro turf" in f: tokens.add("[REDACTED_BY_SCRIPT]")
        if "pergola" in f: tokens.add("GARDEN_PERGOLA")
        if "low maintenance" in f and ("garden" in f or "yard" in f): tokens.add("[REDACTED_BY_SCRIPT]")
        if "south facing garden" in f: tokens.add("GARDEN_SOUTH_FACING")
        if "west facing garden" in f: tokens.add("GARDEN_WEST_FACING") # Added
        if "pond" in f: tokens.add("GARDEN_POND")
        if "water feature" in f: tokens.add("GARDEN_WATER_FEATURE") # Added

        # Views
        if "countryside view" in f or "rural view" in f or "field view" in f or "open view" in f: tokens.add("VIEW_COUNTRYSIDE") # Added open view
        if "water view" in f or "sea view" in f or "river view" in f or "lake view" in f or "coastal view" in f: tokens.add("VIEW_WATER") # Added coastal
        if "city view" in f or "skyline view" in f: tokens.add("VIEW_CITY")
        if "garden view" in f or "overlooks garden" in f: tokens.add("VIEW_GARDEN")
        if "park view" in f: tokens.add("VIEW_PARK")
        if "pleasant outlook" in f: tokens.add("[REDACTED_BY_SCRIPT]")
        if "far reaching view" in f: tokens.add("VIEW_FAR_REACHING") # Added

        # Access
        if "direct garden access" in f: tokens.add("ACCESS_GARDEN_DIRECT")
        if "balcony access" in f: tokens.add("ACCESS_BALCONY")
        if "lift access" in f: tokens.add("ACCESS_LIFT")
        if "side access" in f: tokens.add("ACCESS_SIDE")
        if "rear access" in f: tokens.add("ACCESS_REAR") # Added
        if "private entrance" in f: tokens.add("ACCESS_PRIVATE_ENTRANCE") # Added
        if "wheelchair access" in f or "accessible" in f: tokens.add("ACCESS_WHEELCHAIR_ACCESSIBLE") # Added

        # Eco / Utilities
        if "solar panel" in f: tokens.add("ECO_SOLAR_PANELS") # Added
        if "rainwater harvesting" in f: tokens.add("[REDACTED_BY_SCRIPT]") # Added
        if "ev charging" in f or "electric vehicle charging" in f: tokens.add("ECO_EV_CHARGING") # Added
        if "septic tank" in f: tokens.add("UTILITIES_SEPTIC_TANK") # Added
        if "mains gas" in f: tokens.add("UTILITIES_MAINS_GAS") # Added
        if "mains water" in f: tokens.add("UTILITIES_MAINS_WATER") # Added
        if "mains drainage" in f: tokens.add("UTILITIES_MAINS_DRAINAGE") # Added
        if "[REDACTED_BY_SCRIPT]" in f or "fibre broadband" in f: tokens.add("[REDACTED_BY_SCRIPT]") # Added

    # Check for no matches if needed
    # if not tokens and feature_list and any(isinstance(feat, str) and feat.strip() for feat in feature_list):
    #      tokens.add("[REDACTED_BY_SCRIPT]")

    return list(tokens)


# 3/4. SP/Flaw Theme Token Functions
def get_sp_theme_tokens(selling_points_list):
    """[REDACTED_BY_SCRIPT]"""
    if not selling_points_list or not isinstance(selling_points_list, list): return []
    tokens = set()
    expected_sp_tags = {
        "SP_SPACE", "SP_LIGHT", "SP_CONDITION", "SP_MODERN", "SP_CHARACTER",
        "SP_STYLE", "SP_FUNCTIONAL", "SP_FEATURE", "SP_STORAGE", "SP_GARDEN_ACCESS",
        "SP_GARDEN_VIEW", "SP_OTHER_VIEW", "SP_LOCATION", "SP_PRIVACY", "SP_POTENTIAL",
        "SP_LOW_MAINTENANCE", "SP_QUALITY_FINISH"
    }
    for sp_item in selling_points_list:
        if isinstance(sp_item, dict) and 'tags' in sp_item and isinstance(sp_item['tags'], list):
            for tag in sp_item['tags']:
                if isinstance(tag, str) and tag in expected_sp_tags:
                    tokens.add(tag)
                # Optional: Log unexpected tags if needed for debugging
                # elif isinstance(tag, str):
                #     print(f"[REDACTED_BY_SCRIPT]'{tag}'")
        # else: # Silently skip badly formatted items
        #      pass
    return list(tokens)

def get_flaw_theme_tokens(flaws_list):
    """[REDACTED_BY_SCRIPT]"""
    if not flaws_list or not isinstance(flaws_list, list): return []
    tokens = set()
    expected_flaw_tags = {
        "FLAW_SPACE", "FLAW_LIGHT", "FLAW_CONDITION", "FLAW_DATED", "FLAW_NEEDS_UPDATE",
        "FLAW_MAINTENANCE", "FLAW_STORAGE", "FLAW_LAYOUT", "FLAW_BASIC_STYLE",
        "[REDACTED_BY_SCRIPT]", "FLAW_POOR_FINISH", "FLAW_UNATTRACTIVE", "FLAW_NOISE",
        "FLAW_ACCESSIBILITY"
    }
    for flaw_item in flaws_list:
        if isinstance(flaw_item, dict) and 'tags' in flaw_item and isinstance(flaw_item['tags'], list):
            for tag in flaw_item['tags']:
                 if isinstance(tag, str) and tag in expected_flaw_tags:
                    tokens.add(tag)
                 # Optional: Log unexpected tags
                 # elif isinstance(tag, str):
                 #      print(f"[REDACTED_BY_SCRIPT]'{tag}'")
        # else: # Silently skip badly formatted items
        #      pass
    return list(tokens)


# --- Main Processing Loop ---

# Get list of potential property address folders from the image directory
try:
    address_folders = [
        f for f in os.listdir(main_image_dir)
        if os.path.isdir(os.path.join(main_image_dir, f))
    ]
    print(f"[REDACTED_BY_SCRIPT]'{main_image_dir}'.")
except FileNotFoundError:
    print(f"[REDACTED_BY_SCRIPT]'{main_image_dir}'. Exiting.")
    exit()
except Exception as e:
    print(f"[REDACTED_BY_SCRIPT]'{main_image_dir}': {e}. Exiting.")
    exit()

completed_property_ids = set()
if os.path.isfile(master_csv_path):
    print(f"[REDACTED_BY_SCRIPT]")
    try:
        with open(master_csv_path, 'r', newline='', encoding='utf-8') as csvfile:
            # Check if file is empty
            first_char = csvfile.read(1)
            if not first_char:
                print("[REDACTED_BY_SCRIPT]")
            else:
                csvfile.seek(0) # Rewind after checking first char
                reader = csv.DictReader(csvfile)
                # Check if 'property_id' header exists
                if 'property_id' not in reader.fieldnames:
                     print(f"  Warning: 'property_id'[REDACTED_BY_SCRIPT]")
                else:
                    initial_count = 0
                    for row in reader:
                        prop_id = row.get('property_id') # Use .get for safety
                        if prop_id:
                            completed_property_ids.add(prop_id)
                            initial_count += 1
                    print(f"[REDACTED_BY_SCRIPT]")

    except FileNotFoundError:
        print(f"[REDACTED_BY_SCRIPT]")
    except KeyError:
        print(f"[REDACTED_BY_SCRIPT]'property_id'[REDACTED_BY_SCRIPT]")
    except Exception as e:
        print(f"[REDACTED_BY_SCRIPT]")
        completed_property_ids = set() # Reset if reading failed
else:
    print(f"[REDACTED_BY_SCRIPT]")

# --- Initialize CSV Header ---
csv_header = None # Keep this initialization

# --- Iterate through each property ADDRESS folder --- # <<< LOOP LEVEL CHANGED
for property_address in address_folders:
    print(f"\n{'='[REDACTED_BY_SCRIPT]'='*15}")

    base_image_address_dir = os.path.join(main_image_dir, property_address)
    base_floorplan_address_dir = os.path.join(main_floorplan_dir, property_address)
    # Output dir remains based on the address itself
    property_output_dir = os.path.join(main_output_dir, property_address)

    # --- Find Latest Year Directory within the Address Folder --- # <<< NEW LOGIC START
    latest_year = None
    valid_year_folders = []
    if os.path.isdir(base_image_address_dir):
        try:
            for item in os.listdir(base_image_address_dir):
                item_path = os.path.join(base_image_address_dir, item)
                # Check if it's a directory AND looks like a 4-digit year
                if os.path.isdir(item_path) and item.isdigit() and len(item) == 4:
                    try:
                        # Optional: Add range check? e.g., 1990 < int(item) < 2030
                        valid_year_folders.append(int(item))
                    except ValueError:
                        continue # Ignore if not a valid integer
            if valid_year_folders:
                latest_year = max(valid_year_folders)
        except Exception as e:
            print(f"[REDACTED_BY_SCRIPT]'{base_image_address_dir}': {e}")
            # Continue without a year, likely skipping the property below
    else:
        print(f"[REDACTED_BY_SCRIPT]'{base_image_address_dir}'. Skipping address.")
        continue # Skip this address if the base folder doesn't exist

    if latest_year is None:
        print(f"[REDACTED_BY_SCRIPT]'year' sub-folder in '{base_image_address_dir}'. Skipping address '{property_address}'.")
        continue # Skip if no processable year folder was found

    print(f"[REDACTED_BY_SCRIPT]")

    property_id_with_year = f"[REDACTED_BY_SCRIPT]"
    if property_id_with_year in completed_property_ids:
        print(f"  Skipping '{property_id_with_year}'[REDACTED_BY_SCRIPT]")
        continue # Move to the next address folder

    print(f"[REDACTED_BY_SCRIPT]")

    # Define the specific year paths to use for processing
    current_image_dir = os.path.join(base_image_address_dir, str(latest_year))
    current_floorplan_dir = os.path.join(base_floorplan_address_dir, str(latest_year))
    # Define the output directory for the *address* (not the year specific one initially)
    property_output_dir = os.path.join(main_output_dir, property_address) # Changed from previous version

    # Construct the property ID with year suffix for checks
    property_id_with_year = f"[REDACTED_BY_SCRIPT]"
    year_suffix = f"_y{latest_year}" # Define suffix for filenames

    # --- Check for Output Folder Completeness and Master CSV Entry --- # <<< MODIFIED BLOCK START

    output_folder_complete = False # Assume incomplete initially
    missing_files = []

    if os.path.isdir(property_output_dir):
        # Check if all expected files for the latest year exist
        all_files_found = True
        for base_filename in EXPECTED_BASE_JSON_FILENAMES:
            # Construct expected filename with year suffix
            if base_filename != "image_paths_map":
                expected_filename = f"[REDACTED_BY_SCRIPT]"
                expected_filepath = os.path.join(property_output_dir, expected_filename)
                if not os.path.exists(expected_filepath):
                    all_files_found = False
                    missing_files.append(expected_filename)
            else:
                expected_filename = f"[REDACTED_BY_SCRIPT]"
                expected_filepath = os.path.join(property_output_dir, expected_filename)
                if not os.path.exists(expected_filepath):
                    all_files_found = False
                    missing_files.append(expected_filename)

        if all_files_found:
            output_folder_complete = True
            print(f"  Output folder '{property_output_dir}'[REDACTED_BY_SCRIPT]")
        else:
            print(f"[REDACTED_BY_SCRIPT]'{property_output_dir}'[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]")
            # Force processing by not checking the CSV if folder is incomplete
    else:
        print(f"[REDACTED_BY_SCRIPT]'{property_output_dir}'[REDACTED_BY_SCRIPT]")
        # Folder doesn't exist, so definitely process

    # --- Skip Check (Only if Output Folder is Complete) ---
    if output_folder_complete:
        # Only check the master CSV if the output folder looks complete
        if property_id_with_year in completed_property_ids:
            print(f"  Skipping '{property_id_with_year}'[REDACTED_BY_SCRIPT]")
            continue # Move to the next address folder
        else:
            # Folder complete, but not in CSV (e.g., previous run failed during CSV write)
            print(f"[REDACTED_BY_SCRIPT]")
            # Proceed with processing

    # If we reach here, it means either:
    # 1. The output folder was incomplete.
    # 2. The output folder was complete, but the ID wasn't in the master CSV.
    # 3. The output folder didn't exist yet.
    # In all these cases, we proceed with processing for the latest year.

    # --- Prepare Output Directory (Ensure it exists now) ---
    try:
        os.makedirs(property_output_dir, exist_ok=True)
        print(f"[REDACTED_BY_SCRIPT]")
    except Exception as e:
        print(f"[REDACTED_BY_SCRIPT]'{property_output_dir}'[REDACTED_BY_SCRIPT]")
        continue # Skip if we can't even create the output folder

    # --- Attempt to Find and Load Floorplan (Optional) ---
    floorplan_image = None
    selected_floorplan_path = None
    # Use the specific year folder path for floorplans
    floorplan_image_paths = get_floorplan_images(current_floorplan_dir)

    if floorplan_image_paths:
        print(f"[REDACTED_BY_SCRIPT]")
        max_area = -1
        floorplan_dimensions = {}
        for fp_path in floorplan_image_paths:
            try:
                with Image.open(fp_path) as img:
                    width, height = img.size
                    area = width * height
                    floorplan_dimensions[os.path.basename(fp_path)] = f"{width}x{height}"
                    if area > max_area:
                        max_area = area
                        selected_floorplan_path = fp_path
            except Exception as e:
                print(f"[REDACTED_BY_SCRIPT]")

        if selected_floorplan_path:
            print(f"[REDACTED_BY_SCRIPT]")
            floorplan_image = load_image(selected_floorplan_path)
            if not floorplan_image:
                print(f"[REDACTED_BY_SCRIPT]")
                selected_floorplan_path = None
        else:
            print(f"[REDACTED_BY_SCRIPT]")
    else:
        # This is normal if floorplans aren't always present for the latest year
        print(f"[REDACTED_BY_SCRIPT]")

    # --- Find and Load Room Images (Essential) ---
    # Use the specific year folder path for images
    room_image_paths_or_none = get_room_images(current_image_dir, floorplan_image_paths)
    if room_image_paths_or_none is None:
        print(f"[REDACTED_BY_SCRIPT]'{current_image_dir}'. Skipping address '{property_address}'.")
        continue
    initial_room_image_paths = room_image_paths_or_none # Rename for clarity

    if not initial_room_image_paths:
        print(f"[REDACTED_BY_SCRIPT]'{current_image_dir}'. Skipping address '{property_address}'.")
        continue

    # --- Image Deduplication using Perceptual Hashing --- # <<< NEW BLOCK START
    print(f"[REDACTED_BY_SCRIPT]")
    seen_hashes = set()
    unique_room_image_paths = []
    duplicates_skipped = 0
    hash_errors = 0

    for img_path in initial_room_image_paths:
        try:
            # Open image just for hashing, don't load full data yet
            with Image.open(img_path) as img:
                # Calculate perceptual hash (phash is good for photos)
                # Other options: average_hash (aash), difference_hash (dhash), wavelet_hash (whash)
                # Adjust hash_size for sensitivity (higher means more sensitive)
                img_hash = imagehash.phash(img, hash_size=8)

            if img_hash not in seen_hashes:
                seen_hashes.add(img_hash)
                unique_room_image_paths.append(img_path)
            else:
                # print(f"[REDACTED_BY_SCRIPT]") # Optional verbose logging
                duplicates_skipped += 1
        except FileNotFoundError:
            print(f"[REDACTED_BY_SCRIPT]")
            hash_errors += 1
        except UnidentifiedImageError:
            print(f"[REDACTED_BY_SCRIPT]")
            hash_errors += 1
        except Exception as e:
            print(f"[REDACTED_BY_SCRIPT]")
            hash_errors += 1
            # Optionally, still include images that failed hashing if needed:
            # unique_room_image_paths.append(img_path)


    print(f"[REDACTED_BY_SCRIPT]")
    if hash_errors > 0:
        print(f"[REDACTED_BY_SCRIPT]")

    if not unique_room_image_paths:
         print(f"[REDACTED_BY_SCRIPT]'{current_image_dir}'. Skipping address '{property_address}'.")
         continue

    print(f"[REDACTED_BY_SCRIPT]")
    # Load images using the filtered unique_room_image_paths list
    room_images_loaded = [load_image(p) for p in unique_room_image_paths]
    room_images = [img for img in room_images_loaded if img is not None]

    if not room_images:
        # This check might be redundant if load_image errors are handled, but good for safety
        print(f"[REDACTED_BY_SCRIPT]'{current_image_dir}'. Skipping address '{property_address}'.")
        continue

    # Print the *actual* number loaded successfully
    print(f"[REDACTED_BY_SCRIPT]")
    # Create indexed dictionary from the successfully loaded images
    indexed_room_images = {f"Image {i+1}": img for i, img in enumerate(room_images)}
    indexed_room_images_saved = {f"Image {i+1}": p for i, p in enumerate(unique_room_image_paths)}

    # --- Save the image paths map to JSON ---
    image_map_filename = os.path.join(property_output_dir, f"[REDACTED_BY_SCRIPT]")
    try:
        with open(image_map_filename, "w", encoding="utf-8") as f:
            json.dump(indexed_room_images_saved, f, indent=2, ensure_ascii=False)
        print(f"[REDACTED_BY_SCRIPT]")
    except Exception as e:
        print(f"[REDACTED_BY_SCRIPT]")
    
    # --- Initialize Property-Specific Data Accumulators ---
    property_token_summary = {"room_labels": set(), "features": set(), "sp_themes": set(), "flaw_themes": set()}
    untagged_text_log = {"[REDACTED_BY_SCRIPT]": set(), "raw_features": set(), "[REDACTED_BY_SCRIPT]": set(), "raw_flaws_text": set()}
    output1_json, output1_text = None, None
    output2_json, output2_text = None, None
    output3_json, output3_text = None, None
    output4_json, output4_text = None, None
    output5a_json, output5a_text = None, None
    output5b_json, output5b_text = None, None
    merged_step5_output = None
    output6_json, output6_text = None, None

    # --- Execute Pipeline (Steps 1-6) for the Current Property (using latest year data) ---
    pipeline_successful = True
    try:
        # --- Step 1: Floorplan Analysis (Conditional) ---
        if floorplan_image:
            output1_json, output1_text = call_gemini_and_parse(prompt1_floorplan, floorplan_image, "[REDACTED_BY_SCRIPT]", model_15_flash)
            if output1_text == "RETRY NOW" and output1_json == "RETRY NOW": # Typo fix: should be output1_json
                output1_json, output1_text = call_gemini_and_parse(prompt1_floorplan, floorplan_image, "[REDACTED_BY_SCRIPT]", model_15_flash) # Typo fix: should be output1_json
            
            # --- START: New Bedroom Processing & output1_json Modification ---
            if output1_json and 'rooms_with_dimensions' in output1_json and \
               isinstance(output1_json['rooms_with_dimensions'], list):
                
                bedrooms_with_area_details = []
                non_bedroom_rooms_with_dimensions = []
                
                for room_data in output1_json['rooms_with_dimensions']:
                    if isinstance(room_data, dict) and 'label' in room_data and 'dimensions' in room_data:
                        label = room_data['label']
                        dimensions_str = room_data['dimensions']
                        if label.lower().startswith("bedroom"): # Catches "Bedroom", "Bedroom 1", etc.
                            area = parse_dimensions_and_area(dimensions_str)
                            if area is not None:
                                bedrooms_with_area_details.append({
                                    "original_label": label, 
                                    "label_to_use": label, 
                                    "dimensions": dimensions_str,
                                    "area": area,
                                    "original_data": room_data 
                                })
                            else:
                                print(f"[REDACTED_BY_SCRIPT]'{label}'[REDACTED_BY_SCRIPT]")
                                non_bedroom_rooms_with_dimensions.append(room_data)
                        else:
                            non_bedroom_rooms_with_dimensions.append(room_data)
                    else:
                        non_bedroom_rooms_with_dimensions.append(room_data) # Keep malformed or other entries

                new_processed_rooms_with_dimensions = list(non_bedroom_rooms_with_dimensions) 

                if len(bedrooms_with_area_details) > 1:
                    print(f"[REDACTED_BY_SCRIPT]")
                    bedrooms_with_area_details.sort(key=lambda x: x['area'], reverse=True)
                    
                    primary_bedroom_detail = bedrooms_with_area_details[0]
                    new_processed_rooms_with_dimensions.append({
                        "label": "Primary Bedroom", 
                        "dimensions": primary_bedroom_detail["dimensions"]
                    })
                    
                    other_bedroom_details_list = bedrooms_with_area_details[1:]
                    
                    if other_bedroom_details_list:
                        total_other_area = sum(b['area'] for b in other_bedroom_details_list)
                        avg_other_area = total_other_area / len(other_bedroom_details_list)
                        num_other_beds = len(other_bedroom_details_list)

                        other_bedrooms_entry = {
                            "label": "Other Bedrooms",
                            "dimensions": f"[REDACTED_BY_SCRIPT]"
                            # Consider for gemini_property_feature_generator.py:
                            # "num_rooms": num_other_beds,
                            # "average_area_sq_m": avg_other_area 
                        }
                        new_processed_rooms_with_dimensions.append(other_bedrooms_entry)
                        print(f"    Relabeled largest bedroom to 'Primary Bedroom'[REDACTED_BY_SCRIPT]")
                    # No else needed here, primary is already added
                        
                    output1_json['rooms_with_dimensions'] = new_processed_rooms_with_dimensions
                
                elif len(bedrooms_with_area_details) == 1:
                    # Only one bedroom with area. Add it back using its original data.
                    new_processed_rooms_with_dimensions.append(bedrooms_with_area_details[0]['original_data'])
                    output1_json['rooms_with_dimensions'] = new_processed_rooms_with_dimensions
                    print(f"[REDACTED_BY_SCRIPT]'{bedrooms_with_area_details[0]['original_label']}'[REDACTED_BY_SCRIPT]")
                else:
                    # No bedrooms with area, or all failed parsing. `new_processed_rooms_with_dimensions` only has non-bedrooms.
                    output1_json['rooms_with_dimensions'] = new_processed_rooms_with_dimensions
                    print("[REDACTED_BY_SCRIPT]")
            # --- END: New Bedroom Processing & output1_json Modification ---

            if output1_json: # Tokenization should happen AFTER bedroom processing
                print("[REDACTED_BY_SCRIPT]")
                for room_data in output1_json.get('rooms_with_dimensions', []):
                    if isinstance(room_data, dict): label = room_data.get('label'); token = get_room_token(label); (untagged_text_log["[REDACTED_BY_SCRIPT]"].add(label) if token == UNKNOWN_TOKEN else property_token_summary["room_labels"].add(token)) if label and token else None
                for label in output1_json.get('[REDACTED_BY_SCRIPT]', []): # [REDACTED_BY_SCRIPT] remains unchanged
                    if label and isinstance(label, str): token = get_room_token(label); (untagged_text_log["[REDACTED_BY_SCRIPT]"].add(label) if token == UNKNOWN_TOKEN else property_token_summary["room_labels"].add(token)) if token else None
            else:
                print("[REDACTED_BY_SCRIPT]")
                output1_json, output1_text = None, None # Ensure they are None if step failed
        else:
            print("[REDACTED_BY_SCRIPT]")
            output1_json, output1_text = None, None

        # --- Step 2: Room Assignments ---
        # ** FIX FOR ISSUE 1 **
        # Use the (potentially modified by bedroom logic) output1_json here.
        prompt2_filled = prompt2_assignments.replace("[REDACTED_BY_SCRIPT]", json.dumps(output1_json if output1_json else {}))
        output2_json, output2_text = call_gemini_and_parse(prompt2_filled, indexed_room_images, "[REDACTED_BY_SCRIPT]", model_20_flash)
        if output2_json == "RETRY NOW" and output2_text == "RETRY NOW": # Typo fix
            output2_json, output2_text = call_gemini_and_parse(prompt2_filled, indexed_room_images, "[REDACTED_BY_SCRIPT]", model_20_flash) # Typo fix

        if output2_json and 'room_assignments' in output2_json: # Check for key existence
            print("    Tokenizing Step 2...")
            assignments_list_for_context = output2_json['room_assignments'] 
            for assignment in assignments_list_for_context:
                if isinstance(assignment, dict): label = assignment.get('label'); source = assignment.get('source'); token = get_room_token(label); (untagged_text_log["[REDACTED_BY_SCRIPT]"].add(label) if token == UNKNOWN_TOKEN else property_token_summary["room_labels"].add(token)) if label and token else None; property_token_summary["features"].add('[REDACTED_BY_SCRIPT]') if source == 'Floorplan' else (property_token_summary["features"].add('[REDACTED_BY_SCRIPT]') if source == 'Generated' else None)
            room_assignments_context_json = json.dumps(assignments_list_for_context) # Used for next step's prompt
        else:
            if output2_json is None: print("[REDACTED_BY_SCRIPT]")
            else: print(f"[REDACTED_BY_SCRIPT]'room_assignments'[REDACTED_BY_SCRIPT]")
            raise StopIteration("[REDACTED_BY_SCRIPT]")
        
        # --- Step 3: Feature Extraction ---
        # `output2_text` is the raw JSON string from step 2, which is what the prompt expects
        prompt3_filled = prompt3_features.replace("[REDACTED_BY_SCRIPT]", output2_text if output2_text else "{}")
        output3_json, output3_text = call_gemini_and_parse(prompt3_filled, indexed_room_images, "[REDACTED_BY_SCRIPT]", model_20_flash_lite)
        if output3_text == "RETRY NOW" and output3_json == "RETRY NOW":
            output3_json, output3_text = call_gemini_and_parse(prompt3_filled, indexed_room_images, "[REDACTED_BY_SCRIPT]", model_20_flash_lite)
        if output3_json:
            print("    Tokenizing Step 3...")
            features_dict_for_context = output3_json # Keep the dict
            all_feature_tokens = set()
            for room_label, features_list in features_dict_for_context.items():
                if isinstance(features_list, list):
                    for feature_text in features_list: untagged_text_log["raw_features"].add(feature_text) if isinstance(feature_text, str) else None
                    room_feature_tokens = get_feature_tokens(features_list); all_feature_tokens.update(room_feature_tokens)
            property_token_summary["features"].update(all_feature_tokens); print(f"[REDACTED_BY_SCRIPT]")
        else:
            if output3_json is None: print("[REDACTED_BY_SCRIPT]")
            else: print(f"[REDACTED_BY_SCRIPT]")
            raise StopIteration("[REDACTED_BY_SCRIPT]")
        # Prepare context for Step 4
        room_features_context_json = json.dumps(features_dict_for_context)

        # --- Step 4: Flaws/SPs Identification ---
        prompt4_filled = prompt4_flaws_sp_categorized.replace("[REDACTED_BY_SCRIPT]", output2_text).replace("[REDACTED_BY_SCRIPT]", output3_text)
        output4_json, output4_text = call_gemini_and_parse(prompt4_filled, indexed_room_images, "[REDACTED_BY_SCRIPT]", model_20_flash_lite)
        if output4_text == "RETRY NOW" and output4_json == "RETRY NOW":
            output4_json, output4_text = call_gemini_and_parse(prompt4_filled, indexed_room_images, "[REDACTED_BY_SCRIPT]", model_20_flash_lite)
        if output4_json:
            print("    Tokenizing Step 4...")
            evaluations_dict_for_context = output4_json # Keep the dict
            all_sp_theme_tokens, all_flaw_theme_tokens = set(), set()
            for room_label, sp_flaw_data in evaluations_dict_for_context.items():
                 if isinstance(sp_flaw_data, dict):
                    sp_list = sp_flaw_data.get('selling_points', []); flaw_list = sp_flaw_data.get('flaws', [])
                    for sp_item in sp_list: untagged_text_log["[REDACTED_BY_SCRIPT]"].add(sp_item['text']) if isinstance(sp_item, dict) and 'text' in sp_item and isinstance(sp_item['text'], str) else None
                    for flaw_item in flaw_list: untagged_text_log["raw_flaws_text"].add(flaw_item['text']) if isinstance(flaw_item, dict) and 'text' in flaw_item and isinstance(flaw_item['text'], str) else None
                    sp_tokens = get_sp_theme_tokens(sp_list); flaw_tokens = get_flaw_theme_tokens(flaw_list); all_sp_theme_tokens.update(sp_tokens); all_flaw_theme_tokens.update(flaw_tokens)
            property_token_summary["sp_themes"].update(all_sp_theme_tokens); property_token_summary["flaw_themes"].update(all_flaw_theme_tokens); print(f"[REDACTED_BY_SCRIPT]")
        else:
            if output4_json is None: print("[REDACTED_BY_SCRIPT]")
            else: print(f"[REDACTED_BY_SCRIPT]")
            raise StopIteration("[REDACTED_BY_SCRIPT]")
        # Prepare context for Step 5
        room_evaluation_context_json = json.dumps(evaluations_dict_for_context)

        # --- Step 5a: Ratings P1-10 ---
        prompt5a_filled = prompt5a_ratings_p1_10.replace("[REDACTED_BY_SCRIPT]", output2_text).replace("[REDACTED_BY_SCRIPT]", output3_text).replace("[REDACTED_BY_SCRIPT]", output4_text).replace("[REDACTED_BY_SCRIPT]", persona_details_p1_10)
        output5a_json, output5a_text = call_gemini_and_parse(prompt5a_filled, indexed_room_images, "[REDACTED_BY_SCRIPT]", model_20_flash)
        if output5a_text == "RETRY NOW" and output5a_json == "RETRY NOW":
            output5a_json, output5a_text = call_gemini_and_parse(prompt5a_filled, indexed_room_images, "[REDACTED_BY_SCRIPT]", model_20_flash)
        if not output5a_json: raise StopIteration("[REDACTED_BY_SCRIPT]")

        # --- Step 5b: Ratings P11-20 ---
        prompt5b_filled = prompt5b_ratings_p11_20.replace("[REDACTED_BY_SCRIPT]", output2_text).replace("[REDACTED_BY_SCRIPT]", output3_text).replace("[REDACTED_BY_SCRIPT]", output4_text).replace("[REDACTED_BY_SCRIPT]", persona_details_p11_20)
        output5b_json, output5b_text = call_gemini_and_parse(prompt5b_filled, indexed_room_images, "[REDACTED_BY_SCRIPT]", model_20_flash)
        if output5b_text == "RETRY NOW" and output5b_json == "RETRY NOW":
            output5b_json, output5b_text = call_gemini_and_parse(prompt5b_filled, indexed_room_images, "[REDACTED_BY_SCRIPT]", model_20_flash)
        if not output5b_json: raise StopIteration("[REDACTED_BY_SCRIPT]")

        # --- Step 5 Merge ---
        print("[REDACTED_BY_SCRIPT]")
        try:
            # Merge logic remains the same, using output5a_json and output5b_json
            # ... [exact merge logic from previous version] ...
            # Ensure merged_step5_output is created
             if not isinstance(output5a_json, dict) or not isinstance(output5b_json, dict): raise ValueError("[REDACTED_BY_SCRIPT]")
             labels_5a = output5a_json.get("[REDACTED_BY_SCRIPT]"); ratings_5a = output5a_json.get("room_ratings_p1_10"); overall_5a = output5a_json.get("[REDACTED_BY_SCRIPT]", {})
             labels_5b = output5b_json.get("[REDACTED_BY_SCRIPT]"); ratings_5b = output5b_json.get("room_ratings_p11_20"); overall_5b = output5b_json.get("[REDACTED_BY_SCRIPT]", {})
             if not (isinstance(labels_5a, list) and isinstance(ratings_5a, list) and isinstance(labels_5b, list) and isinstance(ratings_5b, list)): raise ValueError("[REDACTED_BY_SCRIPT]")

             if labels_5a != labels_5b:
                 if set(labels_5a) == set(labels_5b) and len(labels_5a) == len(labels_5b):
                     print("[REDACTED_BY_SCRIPT]")
                     label_to_index_5b = {label: i for i, label in enumerate(labels_5b)}
                     if not all(lbl in label_to_index_5b for lbl in labels_5a): raise ValueError("[REDACTED_BY_SCRIPT]")
                     new_ratings_5b = [None] * len(labels_5a)
                     for i, target_label in enumerate(labels_5a): new_ratings_5b[i] = ratings_5b[label_to_index_5b[target_label]] if label_to_index_5b[target_label] < len(ratings_5b) else None
                     ratings_5b = new_ratings_5b
                 else: raise ValueError(f"[REDACTED_BY_SCRIPT]")

             if len(labels_5a) != len(ratings_5a) or len(labels_5a) != len(ratings_5b): raise ValueError(f"[REDACTED_BY_SCRIPT]")

             merged_step5_output = {"[REDACTED_BY_SCRIPT]": labels_5a, "[REDACTED_BY_SCRIPT]": {**overall_5a, **overall_5b}, "room_ratings_final": []}
             for i, label in enumerate(labels_5a):
                 list_5a = ratings_5a[i]; list_5b = ratings_5b[i]; average_rating = None; merged_ratings_no_avg = []
                 valid_5a = isinstance(list_5a, list) and len(list_5a) == 11; valid_5b = isinstance(list_5b, list) and len(list_5b) == 10
                 if not valid_5a: print(f"      Warning: Room '{label}'[REDACTED_BY_SCRIPT]"); list_5a = [None] * 11
                 if not valid_5b: print(f"      Warning: Room '{label}'[REDACTED_BY_SCRIPT]"); list_5b = [None] * 10
                 merged_ratings_no_avg = list_5a + list_5b
                 if valid_5a and valid_5b:
                     persona_ratings_only = merged_ratings_no_avg[1:]; valid_persona_ratings = [r for r in persona_ratings_only if isinstance(r, (int, float))]
                     if len(valid_persona_ratings) == 20: average_rating = round(statistics.mean(valid_persona_ratings), 1)
                     elif valid_persona_ratings: print(f"      Warning: Room '{label}'[REDACTED_BY_SCRIPT]"); average_rating = round(statistics.mean(valid_persona_ratings), 1)
                     else: print(f"[REDACTED_BY_SCRIPT]'{label}'.")
                 merged_step5_output["room_ratings_final"].append(merged_ratings_no_avg + [average_rating])
             print("[REDACTED_BY_SCRIPT]")
             # Save merged data immediately (filename includes year suffix added later in finally block)
             # merged_step5_filename = os.path.join(property_output_dir, f"[REDACTED_BY_SCRIPT]") # Filename handled in finally
             # with open(merged_step5_filename, "w", encoding="utf-8") as f: json.dump(merged_step5_output, f, indent=2, ensure_ascii=False)
             # print(f"[REDACTED_BY_SCRIPT]") # Logging handled in finally
        except Exception as merge_error:
            print(f"[REDACTED_BY_SCRIPT]")
            merged_step5_output = None
            raise StopIteration("Step 5 Merge Failed")

        # --- Step 6: Renovation Analysis ---
        prompt6_filled = prompt6_renovation.replace("[REDACTED_BY_SCRIPT]", output2_text).replace("[REDACTED_BY_SCRIPT]", output3_text)
        output6_json, output6_text = call_gemini_and_parse(prompt6_filled, indexed_room_images, "[REDACTED_BY_SCRIPT]", model_15_flash)
        if output6_text == "RETRY NOW" and output6_json == "RETRY NOW":
            output6_json, output6_text = call_gemini_and_parse(prompt6_filled, indexed_room_images, "[REDACTED_BY_SCRIPT]", model_15_flash)
        if not output6_json:
            print("[REDACTED_BY_SCRIPT]")

    except StopIteration as si:
        print(f"[REDACTED_BY_SCRIPT]'{property_address}'[REDACTED_BY_SCRIPT]")
        pipeline_successful = False
    except Exception as pipeline_error:
         print(f"[REDACTED_BY_SCRIPT]'{property_address}'[REDACTED_BY_SCRIPT]")
         pipeline_successful = False
         print("[REDACTED_BY_SCRIPT]")
    finally:
        # --- Save Raw Outputs for this Property (with YEAR SUFFIX) --- # <<< MODIFICATION NEEDED HERE
        print(f"[REDACTED_BY_SCRIPT]")
        # Define the year suffix again for saving
        year_suffix_save = f"_y{latest_year}"

        # Note: Merged Step 5 might be an exception if it *shouldn't* have the year suffix
        # Adjust logic below if merged_step5_output needs different naming. Assuming it follows the pattern.
        raw_outputs_to_save = {
            f"[REDACTED_BY_SCRIPT]": output1_json,
            f"[REDACTED_BY_SCRIPT]": output2_json,
            f"[REDACTED_BY_SCRIPT]": output3_json,
            f"[REDACTED_BY_SCRIPT]": output4_json,
            f"[REDACTED_BY_SCRIPT]": merged_step5_output, # Assumes merged needs suffix too
            f"[REDACTED_BY_SCRIPT]": output6_json,
            # Also save the supporting files with suffix
            f"[REDACTED_BY_SCRIPT]": {k: sorted(list(v)) for k, v in property_token_summary.items()},
            f"[REDACTED_BY_SCRIPT]": {k: sorted(list(v)) for k, v in untagged_text_log.items()},
            # Add image_paths_map saving if it's generated
            # f"[REDACTED_BY_SCRIPT]": image_paths_map_data, # Replace with your actual data variable
        }

        for filename, data in raw_outputs_to_save.items():
            filepath = os.path.join(property_output_dir, filename)
            # Check if data is actually generated before saving
            if data is not None and data != {}: # Check for None or empty dict/set results
                 try:
                     # Special handling for sets before saving JSON
                     data_to_save = data
                     if filename.startswith("[REDACTED_BY_SCRIPT]") or filename.startswith("[REDACTED_BY_SCRIPT]"):
                         # Already converted sets to sorted lists in the dictionary definition above
                         pass # Data is ready

                     with open(filepath, "w", encoding="utf-8") as f:
                         json.dump(data_to_save, f, indent=2, ensure_ascii=False)
                 except TypeError as te:
                     print(f"[REDACTED_BY_SCRIPT]")
                 except Exception as e:
                     print(f"[REDACTED_BY_SCRIPT]")


        # --- Feature Generation and CSV Appending (Conditional) ---
        if pipeline_successful:
            try:
                # Define required files (using year suffix) for feature gen check
                year_suffix_check = f"_y{latest_year}"
                # Step 1 is optional for generation, Step 4 & 5 are critical
                path_step4 = os.path.join(property_output_dir, f'[REDACTED_BY_SCRIPT]')
                path_step5_merged = os.path.join(property_output_dir, f'[REDACTED_BY_SCRIPT]')
                # Optional: Check for Step 1 existence if feature generator *requires* it
                # path_step1 = os.path.join(property_output_dir, f'[REDACTED_BY_SCRIPT]')

                required_files_exist = all(os.path.exists(f) for f in [path_step4, path_step5_merged])

                if required_files_exist:
                    print(f"[REDACTED_BY_SCRIPT]")
                    # Pass the directory AND the specific year suffix/ID to the generator
                    # Modify process_property to accept year/ID if needed for loading correct files
                    # OR pass the specific file paths instead of just the directory
                    features = gemini_property_feature_generator.process_property(
                        property_id=property_id_with_year, # Use the ID with year
                        input_dir=property_output_dir,
                        year_suffix=year_suffix_check # Pass suffix to load correct files
                    )

                    if features and isinstance(features, dict):
                        print(f"[REDACTED_BY_SCRIPT]")

                        # --- Append to Master CSV ---
                        if csv_header is None:
                            print("[REDACTED_BY_SCRIPT]")
                            csv_header = gemini_property_feature_generator.get_feature_header()
                            if not csv_header or not isinstance(csv_header, list):
                                print("[REDACTED_BY_SCRIPT]")
                                csv_header = None; continue

                        file_exists = os.path.isfile(master_csv_path)
                        try:
                            print(f"[REDACTED_BY_SCRIPT]")
                            with open(master_csv_path, 'a', newline='', encoding='utf-8') as csvfile:
                                writer = csv.DictWriter(csvfile, fieldnames=csv_header, extrasaction='ignore')
                                if not file_exists: writer.writeheader()
                                writer.writerow(features)
                            print(f"[REDACTED_BY_SCRIPT]")
                        except IOError as csv_error: print(f"[REDACTED_BY_SCRIPT]")
                        except Exception as csv_write_err: print(f"[REDACTED_BY_SCRIPT]")
                    else:
                        print(f"[REDACTED_BY_SCRIPT]")
                else:
                    missing_files = [os.path.basename(f) for f in [path_step4, path_step5_merged] if not os.path.exists(f)]
                    print(f"[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]")

            except ImportError: print(f"[REDACTED_BY_SCRIPT]'gemini_property_feature_generator'."); break
            except Exception as feat_error: print(f"[REDACTED_BY_SCRIPT]")
        else:
            print(f"[REDACTED_BY_SCRIPT]")


        # --- Clean up ---
        floorplan_image = None
        room_images = []
        indexed_room_images = {}
        # import gc; gc.collect()

    print(f"[REDACTED_BY_SCRIPT]")

print(f"\n{'='[REDACTED_BY_SCRIPT]'='*15}")
if csv_header: print(f"[REDACTED_BY_SCRIPT]")
else: print("[REDACTED_BY_SCRIPT]")
print("[REDACTED_BY_SCRIPT]")