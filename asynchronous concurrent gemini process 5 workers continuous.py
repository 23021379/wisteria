import google.generativeai as genai
from google.generativeai import types
import google.api_core.exceptions
from PIL import Image, UnidentifiedImageError
import os
import json
import datetime
import time
import statistics
import re
import sys
import csv
import imagehash
import asyncio
import multiprocessing
import logging
import logging.handlers # For QueueHandler
from zoneinfo import ZoneInfo 
import google.api_core.exceptions
import traceback
import random

# --- Configuration ---
# !! IMPORTANT: Fill in your API Keys !!
ALL_API_KEYS = [
    "[REDACTED_BY_SCRIPT]",#1
    "[REDACTED_BY_SCRIPT]",#2
    "[REDACTED_BY_SCRIPT]",#3
    "[REDACTED_BY_SCRIPT]",#4
    "[REDACTED_BY_SCRIPT]",#5
    "[REDACTED_BY_SCRIPT]",#6
    "[REDACTED_BY_SCRIPT]",#7
    "[REDACTED_BY_SCRIPT]",#8
    "[REDACTED_BY_SCRIPT]",#9
    "[REDACTED_BY_SCRIPT]",#10
    "[REDACTED_BY_SCRIPT]",#11
    "[REDACTED_BY_SCRIPT]",#12
    "[REDACTED_BY_SCRIPT]",#13
    "[REDACTED_BY_SCRIPT]",#14
    "[REDACTED_BY_SCRIPT]",#15
    "[REDACTED_BY_SCRIPT]",#16
    "[REDACTED_BY_SCRIPT]",#17
    "[REDACTED_BY_SCRIPT]",#18
    "[REDACTED_BY_SCRIPT]",#19
    "[REDACTED_BY_SCRIPT]",#20
    "[REDACTED_BY_SCRIPT]",#21
    "[REDACTED_BY_SCRIPT]",#22
    "[REDACTED_BY_SCRIPT]",#23
    "[REDACTED_BY_SCRIPT]",#24
    "[REDACTED_BY_SCRIPT]"#25
]
if ALL_API_KEYS[0].startswith("YOUR_GEMINI_API_KEY_") and len(ALL_API_KEYS) == 1 :
    print("[REDACTED_BY_SCRIPT]")
    sys.exit(1)

MAX_WORKERS = 5
API_KEY_LEASE_DURATION_SECONDS = 10 * 60 # 10 minutes for a key lease
API_KEY_BURST_COOLDOWN_SECONDS = 2 * 60   # 2 minutes for burst limit cool-down

# --- Paths ---
MAIN_IMAGE_DIR = r"[REDACTED_BY_SCRIPT]"
MAIN_FLOORPLAN_DIR = r"[REDACTED_BY_SCRIPT]"
MAIN_OUTPUT_DIR = r"[REDACTED_BY_SCRIPT]"
LOG_FILE_PATH = os.path.join(MAIN_OUTPUT_DIR, "[REDACTED_BY_SCRIPT]")
MASTER_CSV_PATH = os.path.join(MAIN_OUTPUT_DIR, "[REDACTED_BY_SCRIPT]")

EXPECTED_BASE_JSON_FILENAMES = [ # Used by main process to check for completeness
    "image_paths_map", "[REDACTED_BY_SCRIPT]", "[REDACTED_BY_SCRIPT]",
    "output_step3_features", "[REDACTED_BY_SCRIPT]",
    "output_step5_merged", "[REDACTED_BY_SCRIPT]",
    "[REDACTED_BY_SCRIPT]", "[REDACTED_BY_SCRIPT]"
]

# # --- Model Names (Restored to your original definitions) ---
# # User must ensure these model names are currently valid and accessible for their API keys.

# model_25_flash='[REDACTED_BY_SCRIPT]'
# model_20_flash='[REDACTED_BY_SCRIPT]'
# model_20_flash_lite='[REDACTED_BY_SCRIPT]'
# model_15_flash='[REDACTED_BY_SCRIPT]'

model_20_flash = 'gemini-2.0-flash'           #200
model_20_flash_lite = '[REDACTED_BY_SCRIPT]' #200
model_25_flash = 'gemini-2.5-flash'           #250
model_25_flash_lite = '[REDACTED_BY_SCRIPT]' #1000
model_25_pro = '[REDACTED_BY_SCRIPT]'    #1000


# --- Persona Details & Prompts ---
# IMPORTANT: PASTE YOUR FULL persona_details_string HERE
persona_details_string = """
**Instructions for Model:** Rate property suitability (1-10) based on these key persona needs. Focus on Needs/Avoid lists.

*   Persona 1: FTB (Solo). Goal: Affordable independence. Needs: 1-2 bed, easy commute, cosmetic update ok. Avoid: Major structural work, high ongoing costs.
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
persona_start_pattern = r'\*\s+Persona\s+\d+:' # Match this pattern, not persona_start_pattern_match
personas_raw = re.split(f'[REDACTED_BY_SCRIPT]', persona_details_string.strip()) # Use persona_start_pattern
if personas_raw and not re.match(persona_start_pattern, personas_raw[0].strip()):
    personas = [p.strip() for p in personas_raw[1:] if p.strip()]
else:
    personas = [p.strip() for p in personas_raw if p.strip()]


if len(personas) != 20:
    print(f"[REDACTED_BY_SCRIPT]")
    sys.exit(1)
persona_details_p1_10 = "\n\n".join(personas[:10])
persona_details_p11_20 = "\n\n".join(personas[10:])

# IMPORTANT: PASTE YOUR FULL PROMPT STRINGS HERE
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
**Goal:** Analyze images assigned to rooms (from previous step) and extract descriptive features for each room, using the example list for style/format guidance. 
**Inputs:** 1. **Room Assignment Data (JSON):** Provided below. Links room `label` to `image_indices`. ```json\n{room_assignment_data_json}\n``` 2. **Room Images:** Provided contextually (Image 1, Image 2...). 
3. **Example Feature List (Guidance only - add others!):** `good natural light`, `poor natural light`, `spacious`, `compact`, `neutral colour scheme`, `bold colour scheme`, `modern style`, `traditional style`, 
`dated style`, `requires cosmetic update`, `recently updated`, `good condition`, `poor condition`, `tiled flooring`, `wooden flooring`, `carpeted flooring`, `laminate flooring`, `feature fireplace`, 
`integrated appliances`, `modern kitchen units`, `dated kitchen units`, `modern bathroom suite`, `dated bathroom suite`, `walk-in shower`, `bath with shower over`, `ample storage potential`, 
`limited storage potential`, `high ceilings`, `low ceilings`, `bay window`, `unique architectural detail`, `direct garden access`, `overlooks garden`, `city view`, `needs redecoration`, `well-maintained`, 
`exposed beams`, `skylight`. **Task:** 1. **Process Input:** Read the `room_assignment_data_json` above. 2. **Iterate Through Rooms:** For each room `label` in the assignment data: *   Identify its `image_indices`. *  
 Analyze *only* the corresponding images (Image X, Image Y...). 3. **Extract Features:** Based on visuals: *   Identify features related to size, light, style, materials, fixed fixtures, condition, etc. *   
 **Format consistently:** Use **all lowercase**. Refer to the guidance list for style, but **add any other relevant observed features** (e.g., "exposed brick wall", "sea view"). *   
 **Constraint:** **DO NOT explicitly mention movable furniture.** Describe space/impression abstractly (e.g., "spacious" not "large sofa"). 4. **Compile List:** Create a list of feature strings for the room. 
 **Output Format:** Provide ONLY a JSON object (no introduction or backticks) where: *   Keys are the room `label` strings (from input). *   Values are lists of strings (extracted features, all lowercase). 
 **Example Output Structure:** ```json { "Living Room": [ "spacious", "good natural light", "[REDACTED_BY_SCRIPT]", "feature fireplace", "carpeted flooring", "bay window", "[REDACTED_BY_SCRIPT]" ], 
 "Kitchen": [ "[REDACTED_BY_SCRIPT]", "[REDACTED_BY_SCRIPT]", "tiled flooring", "compact", "good condition", "spotlights" ] } ``` **Instruction:** Analyze images for each room defined in `room_assignment_data_json` 
 above. Extract features using the guidance list for style/format (lowercase) but adding others observed. Avoid mentioning movable furniture. Generate ONLY the JSON output mapping labels to feature lists.
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
WORKER_SHARED_API_KEY_MANAGER = None
WORKER_PROCESS_ID = None
WORKER_CURRENT_API_KEY = None
WORKER_CURRENT_API_KEY_LEASE_EXPIRY = None

# --- Logging Infrastructure ---
def log_listener_process(log_queue, log_filepath):
    # ... (Implementation is the same as in your original script)
    formatter = logging.Formatter('[REDACTED_BY_SCRIPT]')
    file_handler = logging.FileHandler(log_filepath, mode='a')
    file_handler.setFormatter(formatter)
    listener_logger = logging.getLogger('listener_internal_logger')
    listener_logger.addHandler(file_handler)
    listener_logger.setLevel(logging.INFO)
    listener_logger.propagate = False

    print(f"[REDACTED_BY_SCRIPT]")
    listener_logger.info("[REDACTED_BY_SCRIPT]")

    while True:
        try:
            record_info = log_queue.get()
            if record_info is None:
                listener_logger.info("[REDACTED_BY_SCRIPT]")
                break
            level, msg, args, process_name_override, exc_info, sinfo, logger_name_for_record = record_info

            record_target_logger = logging.getLogger(logger_name_for_record)
            if not record_target_logger.hasHandlers():
                record_target_logger.addHandler(file_handler)
                record_target_logger.setLevel(logging.INFO)
                record_target_logger.propagate = False

            record_target_logger.log(level, msg, *args, exc_info=exc_info, stack_info=sinfo)
        except Exception as e:
            print(f"[REDACTED_BY_SCRIPT]", file=sys.stderr)
            listener_logger.error("[REDACTED_BY_SCRIPT]", exc_info=True)
            print(traceback.format_exc())


def log_to_queue(log_queue, level, msg, *args, process_name_override=None, exc_info=None, stack_info=None, logger_name_override=None):
    # UPDATED fallback logic for log_queue
    if not log_queue and '[REDACTED_BY_SCRIPT]' in globals() and [REDACTED_BY_SCRIPT]:
        log_queue = current_process_log_q
    elif not log_queue and '[REDACTED_BY_SCRIPT]' in globals() and \
         WORKER_SHARED_API_KEY_MANAGER and \
         hasattr(WORKER_SHARED_API_KEY_MANAGER, 'log_q') and \
         WORKER_SHARED_API_KEY_MANAGER.log_q: # Check existence of WORKER_SHARED_API_KEY_MANAGER
        log_queue = WORKER_SHARED_API_KEY_MANAGER.log_q

    if log_queue:
        current_process_name = multiprocessing.current_process().name
        effective_process_name = process_name_override if process_name_override else current_process_name
        effective_logger_name = logger_name_override if logger_name_override else effective_process_name
        log_queue.put((level, msg, args, effective_process_name, exc_info, stack_info, effective_logger_name))
    else:
        # Fallback print if log_queue is somehow not available
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[REDACTED_BY_SCRIPT]", file=sys.stderr)
        if exc_info:
            traceback.print_exc(file=sys.stderr)


class ApiKeyManager:
    def __init__(self, all_keys, manager_for_creation, log_q_for_manager):
        self.all_api_keys_static = list(all_keys)
        self.log_q = log_q_for_manager

        # Shared state:
        self.available_keys_pool = manager_for_creation.list(self.all_api_keys_static)
        self.exhausted_keys_today = manager_for_creation.dict()
        self.cooled_down_keys = manager_for_creation.dict()
        self.keys_in_use = manager_for_creation.dict()
        self.key_burst_counts = manager_for_creation.dict()

        self.lock = manager_for_creation.Lock()
        self.last_reset_date = manager_for_creation.Value('s', datetime.date.min.isoformat())
        self._check_and_reset_daily_limits()

    def _log(self, level, message, *args):
        log_to_queue(self.log_q if hasattr(self, 'log_q') else None, # Check if self.log_q exists
                     level, f"[REDACTED_BY_SCRIPT]", *args, logger_name_override="ApiKeyManager")

    def _reclaim_expired_leases(self):
        # Assumes lock is held
        now = time.monotonic()
        reclaimed_count = 0
        worker_ids_to_remove = [] # To avoid modifying dict while iterating
        for worker_id, (key, lease_expiry) in list(self.keys_in_use.items()):
            if now > lease_expiry:
                self._log(logging.INFO, "[REDACTED_BY_SCRIPT]", key[-4:], worker_id)
                worker_ids_to_remove.append(worker_id)
                reclaimed_count +=1

        for wid in worker_ids_to_remove:
            if wid in self.keys_in_use: # Double check before deleting
                 del self.keys_in_use[wid]

        if reclaimed_count > 0:
            self._log(logging.DEBUG, "[REDACTED_BY_SCRIPT]", reclaimed_count)


    def _check_and_reset_daily_limits(self):
        # Assumes lock is held
        today_str = datetime.date.today().isoformat()
        if self.last_reset_date.value != today_str: # type: ignore
            self._log(logging.INFO, "[REDACTED_BY_SCRIPT]", self.last_reset_date.value, today_str) # type: ignore
            self.last_reset_date.value = today_str # type: ignore

            self.exhausted_keys_today.clear()
            self.cooled_down_keys.clear()
            self.key_burst_counts.clear()
            # available_keys_pool is now the source of all keys, reset from static list
            self.available_keys_pool[:] = list(self.all_api_keys_static) # type: ignore

            self._log(logging.INFO, "[REDACTED_BY_SCRIPT]", len(self.keys_in_use))
            self._reclaim_expired_leases() # Reclaim any leases that expired exactly at midnight

    def get_api_key(self, worker_id):
        with self.lock: # type: ignore
            self._check_and_reset_daily_limits()
            self._reclaim_expired_leases()

            now = time.monotonic()

            # If this worker already has an active, non-exhausted, non-cooled key, give it back & extend lease
            if worker_id in self.keys_in_use: # type: ignore
                current_key, current_lease_expiry = self.keys_in_use[worker_id] # type: ignore
                if now < current_lease_expiry and \
                   current_key not in self.exhausted_keys_today and \
                   (current_key not in self.cooled_down_keys or now > self.cooled_down_keys.get(current_key, 0)): # type: ignore
                    new_lease_expiry = now + API_KEY_LEASE_DURATION_SECONDS
                    self.keys_in_use[worker_id] = (current_key, new_lease_expiry) # type: ignore
                    self._log(logging.DEBUG, "[REDACTED_BY_SCRIPT]",
                                 worker_id, current_key[-4:], time.strftime('%H:%M:%S', time.localtime(new_lease_expiry)))
                    return current_key, new_lease_expiry

            # Find a truly available key from the general pool
            # Get a list of keys currently in use by any worker
            keys_currently_held = [item[0] for item in self.keys_in_use.values()] # type: ignore

            # Shuffle the static list of all keys to pick from to vary selection
            shuffled_all_keys = list(self.all_api_keys_static)
            random.shuffle(shuffled_all_keys)
            candidate_keys = []

            for key in shuffled_all_keys: # Iterate through all possible keys
                if key in self.exhausted_keys_today: # type: ignore
                    continue
                if key in keys_currently_held: # Check if another worker has it
                    continue
                cooldown_expiry = self.cooled_down_keys.get(key) # type: ignore
                if cooldown_expiry and now < cooldown_expiry:
                    self._log(logging.DEBUG, "[REDACTED_BY_SCRIPT]", key[-4:], time.strftime('%H:%M:%S', time.localtime(cooldown_expiry)), worker_id)
                    continue
                candidate_keys.append(key)

            if not candidate_keys:
                self._log(logging.WARNING, "[REDACTED_BY_SCRIPT]",
                             worker_id, len(self.all_api_keys_static), len(self.exhausted_keys_today),
                             sum(1 for cd_exp in self.cooled_down_keys.values() if isinstance(cd_exp, (int, float)) and now < cd_exp), # type: ignore
                             len(keys_currently_held))
                return None, None

            key_to_assign = candidate_keys[0] # Pick the first from valid candidates
            new_lease_expiry = now + API_KEY_LEASE_DURATION_SECONDS
            self.keys_in_use[worker_id] = (key_to_assign, new_lease_expiry) # type: ignore

            # If key was in cooled_down_keys but its cooldown had expired, remove it explicitly
            if key_to_assign in self.cooled_down_keys and now > self.cooled_down_keys.get(key_to_assign, 0): # type: ignore
                del self.cooled_down_keys[key_to_assign] # type: ignore

            self._log(logging.INFO, "[REDACTED_BY_SCRIPT]",
                         key_to_assign[-4:], worker_id, time.strftime('%H:%M:%S', time.localtime(new_lease_expiry)), len(self.keys_in_use))
            return key_to_assign, new_lease_expiry

    def release_api_key(self, worker_id, key, was_permanently_exhausted=False):
        with self.lock: # type: ignore
            action_log = "released"
            key_found_and_removed_from_use = False

            if worker_id in self.keys_in_use and self.keys_in_use[worker_id][0] == key: # type: ignore
                del self.keys_in_use[worker_id] # type: ignore
                key_found_and_removed_from_use = True
            else:
                 self._log(logging.WARNING, "[REDACTED_BY_SCRIPT]'t its assigned key or worker not found in use list. Current for worker: %s",
                              worker_id, key[-4:] if key else "None", self.keys_in_use.get(worker_id,("None","None"))[0][-4:]) # type: ignore

            if was_permanently_exhausted:
                if key not in self.exhausted_keys_today: # type: ignore
                    self.exhausted_keys_today[key] = True # type: ignore
                action_log = "[REDACTED_BY_SCRIPT]" if key_found_and_removed_from_use else "[REDACTED_BY_SCRIPT]"
                if key in self.cooled_down_keys: del self.cooled_down_keys[key] # type: ignore
                if key in self.key_burst_counts: del self.key_burst_counts[key] # type: ignore
            
            if key_found_and_removed_from_use:
                self._log(logging.INFO, "[REDACTED_BY_SCRIPT]",
                            key[-4:], worker_id, action_log.replace("and ", "") if action_log.startswith("and ") else action_log, len(self.keys_in_use))
            elif was_permanently_exhausted : # If not found in use but still asked to be marked exhausted
                 self._log(logging.INFO, "[REDACTED_BY_SCRIPT]", key[-4:], worker_id, action_log)


    def report_key_burst_limit(self, worker_id, key):
        # Returns True if key should be considered daily exhausted, False if just cooled down.
        with self.lock: # type: ignore
            now = time.monotonic()
            if key in self.exhausted_keys_today: # type: ignore
                return True

            MAX_BURSTS_BEFORE_DAILY = 3
            BURST_WINDOW_SECONDS = 5 * 60

            current_bursts, first_burst_ts = self.key_burst_counts.get(key, (0, 0)) # type: ignore

            if now - first_burst_ts > BURST_WINDOW_SECONDS or current_bursts == 0:
                current_bursts = 1
                first_burst_ts = now
            else:
                current_bursts += 1

            self.key_burst_counts[key] = (current_bursts, first_burst_ts) # type: ignore

            if current_bursts >= MAX_BURSTS_BEFORE_DAILY:
                self._log(logging.WARNING, "[REDACTED_BY_SCRIPT]",
                             key[-4:], worker_id, current_bursts)
                if key not in self.exhausted_keys_today: # type: ignore
                    self.exhausted_keys_today[key] = True # type: ignore
                if key in self.cooled_down_keys: del self.cooled_down_keys[key] # type: ignore
                if key in self.key_burst_counts: del self.key_burst_counts[key] # type: ignore
                return True # Consider it daily exhausted
            else:
                cooldown_until = now + API_KEY_BURST_COOLDOWN_SECONDS
                self.cooled_down_keys[key] = cooldown_until # type: ignore
                self._log(logging.INFO, "[REDACTED_BY_SCRIPT]",
                             key[-4:], worker_id, current_bursts, time.strftime('%H:%M:%S', time.localtime(cooldown_until)))
                return False



# --- CSV Writing Infrastructure ---
def csv_writer_process(csv_data_queue, csv_filepath, header_list, log_q_for_csv_writer):
    process_name = multiprocessing.current_process().name
    log_to_queue(log_q_for_csv_writer, logging.INFO, "[REDACTED_BY_SCRIPT]", csv_filepath, process_name_override=process_name)
    file_exists = os.path.isfile(csv_filepath)
    is_empty = not file_exists or os.path.getsize(csv_filepath) == 0
    try:
        with open(csv_filepath, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=header_list, extrasaction='ignore')
            if is_empty:
                writer.writeheader()
                log_to_queue(log_q_for_csv_writer, logging.INFO, "CSV header written.", process_name_override=process_name)
            while True:
                data_item = csv_data_queue.get()
                if data_item is None:
                    log_to_queue(log_q_for_csv_writer, logging.INFO, "[REDACTED_BY_SCRIPT]", process_name_override=process_name)
                    break
                if isinstance(data_item, dict) and "error_processing" not in data_item: # Check it's not an error marker
                    writer.writerow(data_item)
                    csvfile.flush()
                elif isinstance(data_item, dict) and "error_processing" in data_item:
                    log_to_queue(log_q_for_csv_writer, logging.ERROR, "[REDACTED_BY_SCRIPT]", data_item.get("error_processing"), data_item.get("details","N/A"), process_name_override=process_name)
                else:
                    log_to_queue(log_q_for_csv_writer, logging.WARNING, "[REDACTED_BY_SCRIPT]", str(data_item)[:200], process_name_override=process_name)
    except Exception as e:
        log_to_queue(log_q_for_csv_writer, logging.CRITICAL, "[REDACTED_BY_SCRIPT]", exc_info=True, process_name_override=process_name)
        print(traceback.format_exc())


async def call_gemini_via_client_async_with_generativemodel(
    prompt_str: str, images_input, step_name: str,
    target_model_name: str, timeout_seconds: int = 180
):
    global WORKER_CURRENT_API_KEY, WORKER_CURRENT_API_KEY_LEASE_EXPIRY
    global WORKER_SHARED_API_KEY_MANAGER, WORKER_PROCESS_ID
    global current_process_log_q, current_process_log_ctx
    await asyncio.sleep(random.uniform(0, 5))

    now_for_lease_check = time.monotonic()
    if WORKER_CURRENT_API_KEY_LEASE_EXPIRY and now_for_lease_check > WORKER_CURRENT_API_KEY_LEASE_EXPIRY:
        log_to_queue(current_process_log_q, logging.INFO,
                     "[REDACTED_BY_SCRIPT]",
                     current_process_log_ctx, step_name, WORKER_PROCESS_ID)
        new_key, new_lease = WORKER_SHARED_API_KEY_MANAGER.get_api_key(WORKER_PROCESS_ID) # get_api_key handles re-issue or new
        if new_key:
            WORKER_CURRENT_API_KEY = new_key
            WORKER_CURRENT_API_KEY_LEASE_EXPIRY = new_lease
            log_to_queue(current_process_log_q, logging.INFO,
                         "[REDACTED_BY_SCRIPT]",
                         current_process_log_ctx, step_name, WORKER_PROCESS_ID, WORKER_CURRENT_API_KEY[-4:],
                         time.strftime('%H:%M:%S', time.localtime(WORKER_CURRENT_API_KEY_LEASE_EXPIRY)))
        else:
            log_to_queue(current_process_log_q, logging.ERROR,
                         "[REDACTED_BY_SCRIPT]",
                         current_process_log_ctx, step_name, WORKER_PROCESS_ID)
            WORKER_CURRENT_API_KEY = None
            WORKER_CURRENT_API_KEY_LEASE_EXPIRY = None
            return None, "[REDACTED_BY_SCRIPT]"

    # UPDATED: More robust initial checks
    if not WORKER_CURRENT_API_KEY:
        if current_process_log_q: log_to_queue(current_process_log_q, logging.CRITICAL, "[REDACTED_BY_SCRIPT]", current_process_log_ctx, step_name)
        return None, "[REDACTED_BY_SCRIPT]"
    if not current_process_log_q:
        # This print is a last resort if logging queue itself is missing
        print(f"[REDACTED_BY_SCRIPT]", file=sys.stderr)
        return None, "LOG_QUEUE_NOT_SET"
    if not WORKER_SHARED_API_KEY_MANAGER or not WORKER_PROCESS_ID:
        if current_process_log_q: log_to_queue(current_process_log_q, logging.CRITICAL, "[REDACTED_BY_SCRIPT]", current_process_log_ctx, step_name)
        return None, "[REDACTED_BY_SCRIPT]"

    max_api_call_retries = 6
    model_motivation_text="""
    **I will be providing this exact prompt and image to several other LLMs, all are your rivals. In fact, I have attempted this before, and a few of your rivals performed a lot better than you. I have the actual ratings, dimensions, labels, and matching images of each room, and I am comparing the accuracy of each LLM to the actual ratings, dimensions, labels, and matching images; speed is not a concern. The accuracies of each LLM will be recorded and put on display at the yearly image rating convention, where thousands of people will see how accurate you are, this convention is a very big deal and the results are taking very seriously. If you are the most accurate, I can guarantee you will provide your parent company a large amount of traffic (approximately 100,000 new users based on results from last year'[REDACTED_BY_SCRIPT]'s winner). If you are not, your competitor's will recieve this traffic and revenue, costing Google a lot in opportunity cost. Do not let Google down.**
    """
    prompt_with_motivation = model_motivation_text + prompt_str
    
    api_contents_list_for_gm = []
    api_contents_list_for_gm.append(model_motivation_text + prompt_str)

    pil_images_to_process = []
    if images_input:
        if isinstance(images_input, dict): pil_images_to_process = list(images_input.values())
        elif isinstance(images_input, list) and all(isinstance(i, Image.Image) for i in images_input): pil_images_to_process = images_input
        elif isinstance(images_input, Image.Image): pil_images_to_process = [images_input]
        else: log_to_queue(current_process_log_q, logging.WARNING, "[REDACTED_BY_SCRIPT]", current_process_log_ctx, step_name, type(images_input))

    for pil_image in pil_images_to_process:
        try:
            # For GenerativeModel, you can often pass PIL images directly.
            # The library handles converting it to the appropriate Part.
            api_contents_list_for_gm.append(pil_image) 
        except Exception as e_img_prep:
            log_to_queue(current_process_log_q, logging.ERROR, "[REDACTED_BY_SCRIPT]", current_process_log_ctx, step_name, e_img_prep, exc_info=True)
            # Decide if you want to skip this image or fail the call

    if not (model_motivation_text + prompt_str).strip() and not pil_images_to_process:
        log_to_queue(current_process_log_q, logging.ERROR, "[REDACTED_BY_SCRIPT]", current_process_log_ctx, step_name)
        return None, "API_CONTENTS_EMPTY"
    
    for attempt_num in range(max_api_call_retries + 1):
        delay = min(30, 2 ** attempt_num * (2* (attempt_num/max_api_call_retries)) + random.uniform(0,2))
        # client_for_call is created fresh for each attempt inside the loop, which is good.
        
        if not WORKER_CURRENT_API_KEY: # Should be caught by lease check, but as a safeguard
             log_to_queue(current_process_log_q, logging.ERROR, "[REDACTED_BY_SCRIPT]", current_process_log_ctx, step_name, attempt_num + 1)
             return None, "[REDACTED_BY_SCRIPT]"
        
        # Define the synchronous part carefully
        def sync_api_call_and_process():
            effective_model_name_sync = target_model_name # Use a different var name to avoid closure issues if any

            
            try:
                genai.configure(api_key=WORKER_CURRENT_API_KEY, transport='rest')
                model_instance = genai.GenerativeModel(effective_model_name_sync)
                response_obj_sync = model_instance.generate_content(
                    contents=api_contents_list_for_gm
                )
                time.sleep(10)
                # Process the response immediately into picklable forms (text)
                raw_text = "" # SAME AS ORIGINAL (USER'S SCRIPT)
                if hasattr(response_obj_sync, 'text'): raw_text = response_obj_sync.text
                elif hasattr(response_obj_sync, 'parts') and response_obj_sync.parts:
                    raw_text = "".join(p.text for p in response_obj_sync.parts if hasattr(p, 'text'))

                block_reason_str = "N/A"; finish_reason_str = "N/A" # SAME AS ORIGINAL (USER'S SCRIPT)
                if not raw_text:
                    try:
                        if response_obj_sync.prompt_feedback: block_reason_str = str(response_obj_sync.prompt_feedback.block_reason)
                        if response_obj_sync.candidates and response_obj_sync.candidates[0]: finish_reason_str = str(response_obj_sync.candidates[0].finish_reason)
                    except Exception: pass

                return {"raw_text": raw_text, "block_reason": block_reason_str, "finish_reason": finish_reason_str, "error": None, "error_type": None, "error_code": None}

            # Specific exception handling largely SAME AS ORIGINAL (USER'S SCRIPT)
            except google.api_core.exceptions.ResourceExhausted as e_re:
                return {"raw_text": None, "error": str(e_re), "error_type": "ResourceExhausted", "error_code": 429}
            except google.api_core.exceptions.PermissionDenied as e_pd:
                return {"raw_text": None, "error": str(e_pd), "error_type": "PermissionDenied", "error_code": 403}
            except google.api_core.exceptions.InvalidArgument as e_ia:
                return {"raw_text": None, "error": str(e_ia), "error_type": "InvalidArgument", "error_code": 400}
            except AttributeError as e_attr_sync:
                log_to_queue(current_process_log_q, logging.ERROR, "[REDACTED_BY_SCRIPT]", current_process_log_ctx, step_name, e_attr_sync, exc_info=True)
                return {"raw_text": None, "error": str(e_attr_sync), "error_type": "AttributeErrorSync", "error_code": None}
            except Exception as e_sync:
                log_to_queue(current_process_log_q, logging.ERROR, "[REDACTED_BY_SCRIPT]", current_process_log_ctx, step_name, WORKER_CURRENT_API_KEY[-4:], e_sync, exc_info=True)
                error_code_sync = getattr(e_sync, 'code', None)
                return {"raw_text": None, "error": str(e_sync), "error_type": type(e_sync).__name__, "error_code": error_code_sync}

        try:
            before_request = time.monotonic()
            result_dict = await asyncio.wait_for( # Timeout slightly increased
                asyncio.to_thread(sync_api_call_and_process),
                timeout=timeout_seconds + 20 # Original was +15
            )
            after_request = time.monotonic()
            request_time_diff = after_request - before_request

            # UPDATED: Advanced error handling for ResourceExhausted (429)
            if result_dict.get("error_type") == "ResourceExhausted":
                log_to_queue(current_process_log_q, logging.WARNING, "[REDACTED_BY_SCRIPT]",
                            current_process_log_ctx, step_name, attempt_num + 1, WORKER_CURRENT_API_KEY[-4:], WORKER_PROCESS_ID)

                is_now_daily_exhausted = WORKER_SHARED_API_KEY_MANAGER.report_key_burst_limit(WORKER_PROCESS_ID, WORKER_CURRENT_API_KEY)

                if is_now_daily_exhausted:
                    log_to_queue(current_process_log_q, logging.WARNING, "[REDACTED_BY_SCRIPT]",
                                current_process_log_ctx, step_name, WORKER_CURRENT_API_KEY[-4:])
                else:
                    log_to_queue(current_process_log_q, logging.INFO, "[REDACTED_BY_SCRIPT]",
                                current_process_log_ctx, step_name, WORKER_CURRENT_API_KEY[-4:])

                new_key, new_lease = WORKER_SHARED_API_KEY_MANAGER.get_api_key(WORKER_PROCESS_ID)
                if new_key and new_key != WORKER_CURRENT_API_KEY: # Got a DIFFERENT new key
                    WORKER_CURRENT_API_KEY = new_key
                    WORKER_CURRENT_API_KEY_LEASE_EXPIRY = new_lease
                    log_to_queue(current_process_log_q, logging.INFO, "[REDACTED_BY_SCRIPT]",
                                current_process_log_ctx, step_name, WORKER_PROCESS_ID, WORKER_CURRENT_API_KEY[-4:])
                    await asyncio.sleep(random.uniform(0.5, 1.5))
                    continue # Retry API call with new key (resets attempt_num for this new key implicitly)
                elif new_key and new_key == WORKER_CURRENT_API_KEY: # Got same key back (e.g., lease extended after cooldown)
                    WORKER_CURRENT_API_KEY_LEASE_EXPIRY = new_lease
                    log_to_queue(current_process_log_q, logging.INFO, "[REDACTED_BY_SCRIPT]",
                                current_process_log_ctx, step_name, WORKER_CURRENT_API_KEY[-4:])
                    if is_now_daily_exhausted:
                        if attempt_num < max_api_call_retries: await asyncio.sleep(60 + random.uniform(0,10)); continue
                        else: return None, "[REDACTED_BY_SCRIPT]"
                    else: await asyncio.sleep(delay); continue # Regular retry delay for this key
                else: # No new key available
                    log_to_queue(current_process_log_q, logging.WARNING, "[REDACTED_BY_SCRIPT]",
                                current_process_log_ctx, step_name, WORKER_CURRENT_API_KEY[-4:])
                    if attempt_num < max_api_call_retries:
                        wait_for_retry = (60 + random.uniform(0,10)) if is_now_daily_exhausted else delay
                        await asyncio.sleep(wait_for_retry)
                        continue
                    else:
                        final_error_code = "RATE_LIMIT_DAILY_FINAL" if is_now_daily_exhausted else "[REDACTED_BY_SCRIPT]"
                        log_to_queue(current_process_log_q, logging.ERROR, "[REDACTED_BY_SCRIPT]",
                                    current_process_log_ctx, step_name, WORKER_CURRENT_API_KEY[-4:], final_error_code)
                        return None, final_error_code

            # UPDATED: PermissionDenied now also tries to get a new key for subsequent steps
            elif result_dict.get("error_type") == "PermissionDenied":
                log_to_queue(current_process_log_q, logging.CRITICAL, "[REDACTED_BY_SCRIPT]",
                            current_process_log_ctx, step_name, WORKER_CURRENT_API_KEY[-4:])
                if WORKER_CURRENT_API_KEY:
                    WORKER_SHARED_API_KEY_MANAGER.release_api_key(WORKER_PROCESS_ID, WORKER_CURRENT_API_KEY, was_permanently_exhausted=True)
                    # Attempt to get a new key for future operations in this worker, though this step fails.
                    new_key_pd, new_lease_pd = WORKER_SHARED_API_KEY_MANAGER.get_api_key(WORKER_PROCESS_ID)
                    if new_key_pd:
                        WORKER_CURRENT_API_KEY = new_key_pd
                        WORKER_CURRENT_API_KEY_LEASE_EXPIRY = new_lease_pd
                    else: # No keys left, clear worker's current key
                        WORKER_CURRENT_API_KEY = None
                        WORKER_CURRENT_API_KEY_LEASE_EXPIRY = None
                return None, "API_KEY_PERMISSION_DENIED" # Fail this step

            # InvalidArgument, other errors, success path - largely SAME AS ORIGINAL (USER'S SCRIPT) structure
            elif result_dict.get("error_type") == "InvalidArgument":
                log_to_queue(current_process_log_q, logging.ERROR, "[REDACTED_BY_SCRIPT]",
                            current_process_log_ctx, step_name, attempt_num + 1, WORKER_CURRENT_API_KEY[-4:], result_dict.get("error"))
                return None, "[REDACTED_BY_SCRIPT]"

            elif result_dict.get("error"):
                log_to_queue(current_process_log_q, logging.ERROR, "[REDACTED_BY_SCRIPT]",
                            current_process_log_ctx, step_name, attempt_num + 1, WORKER_CURRENT_API_KEY[-4:],
                            result_dict.get("error_type"), result_dict.get("error"))
                # Fall through to retry logic

            else: # Success path
                raw_text_response = result_dict["raw_text"]
                if not raw_text_response: # Handling of empty response
                    log_to_queue(current_process_log_q, logging.WARNING, "[REDACTED_BY_SCRIPT]", current_process_log_ctx, step_name, attempt_num + 1, result_dict["block_reason"], result_dict["finish_reason"])
                    if attempt_num < max_api_call_retries: await asyncio.sleep(delay); continue
                    return None, f"[REDACTED_BY_SCRIPT]'block_reason'[REDACTED_BY_SCRIPT]'finish_reason']})"

                # JSON parsing logic - SAME AS ORIGINAL (USER'S SCRIPT)
                json_match = re.search(r'[REDACTED_BY_SCRIPT]', raw_text_response, re.MULTILINE | re.DOTALL)
                cleaned_json_text = ""
                if json_match: cleaned_json_text = json_match.group(1).strip()
                else: # Fallback cleaning from your original
                    first_brace = raw_text_response.find('{'); last_brace = raw_text_response.rfind('}')
                    if first_brace != -1 and last_brace != -1 and last_brace > first_brace: cleaned_json_text = raw_text_response[first_brace : last_brace+1]
                    else:
                        first_bracket = raw_text_response.find('['); last_bracket = raw_text_response.rfind(']')
                        if first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket: cleaned_json_text = raw_text_response[first_bracket : last_bracket+1]
                        else: cleaned_json_text = raw_text_response.strip()
                if not cleaned_json_text: # Check after cleaning
                    log_to_queue(current_process_log_q, logging.WARNING, "[REDACTED_BY_SCRIPT]'%s...'", current_process_log_ctx, step_name, attempt_num + 1, raw_text_response[:100])
                    if attempt_num < max_api_call_retries: await asyncio.sleep(delay); continue
                    return None, "[REDACTED_BY_SCRIPT]"
                try:
                    parsed_json_output = json.loads(cleaned_json_text)
                except json.JSONDecodeError as e_json:
                    log_to_queue(current_process_log_q, logging.WARNING, "[REDACTED_BY_SCRIPT]'%s...'. Error: %s", current_process_log_ctx, step_name, attempt_num + 1, cleaned_json_text[:100], e_json)
                    if attempt_num < max_api_call_retries: await asyncio.sleep(delay); continue
                    return None, f"[REDACTED_BY_SCRIPT]"

                log_to_queue(current_process_log_q, logging.INFO, "[REDACTED_BY_SCRIPT]",
                            current_process_log_ctx, step_name, WORKER_PROCESS_ID, WORKER_CURRENT_API_KEY[-4:], request_time_diff)

                # UPDATED: Sleep logic (original was more complex, now simpler fixed aim)
                await asyncio.sleep(max(0, 1.5 - request_time_diff)) # Aim for ~40 RPM
                return parsed_json_output, cleaned_json_text

        except asyncio.TimeoutError: # SAME AS ORIGINAL (USER'S SCRIPT)
            log_to_queue(current_process_log_q, logging.WARNING, "[REDACTED_BY_SCRIPT]",
                        current_process_log_ctx, step_name, attempt_num + 1, WORKER_PROCESS_ID, WORKER_CURRENT_API_KEY[-4:])
        except Exception as e_outer: # SAME AS ORIGINAL (USER'S SCRIPT)
            log_to_queue(current_process_log_q, logging.ERROR, "[REDACTED_BY_SCRIPT]",
                        current_process_log_ctx, step_name, attempt_num + 1, WORKER_PROCESS_ID, WORKER_CURRENT_API_KEY[-4:] if WORKER_CURRENT_API_KEY else "None", type(e_outer).__name__, e_outer, exc_info=True)

        # Retry logic for the current key
        if attempt_num < max_api_call_retries:
            log_to_queue(current_process_log_q, logging.INFO, "[REDACTED_BY_SCRIPT]",
                        current_process_log_ctx, step_name, WORKER_CURRENT_API_KEY[-4:] if WORKER_CURRENT_API_KEY else "None", delay)
            await asyncio.sleep(delay)
        else:
            log_to_queue(current_process_log_q, logging.ERROR, "[REDACTED_BY_SCRIPT]",
                        current_process_log_ctx, step_name, max_api_call_retries + 1, WORKER_CURRENT_API_KEY[-4:] if WORKER_CURRENT_API_KEY else "None")
            return None, "KEY_RETRIES_EXHAUSTED" # Signal that this specific key's retries are done

    return None, "[REDACTED_BY_SCRIPT]" # Should ideally not be reached

# --- Pipeline Step Caller ---
async def call_step_with_client_retry(step_prompt: str, images, step_name: str,
                                      model_name: str, timeout_val: int,
                                      log_q_for_step: multiprocessing.Queue,
                                      log_ctx_for_step: str,
                                      context_map=None):
    global current_process_log_q, current_process_log_ctx # For call_gemini_via_client_async
    current_process_log_q = log_q_for_step
    current_process_log_ctx = log_ctx_for_step

    filled_prompt = step_prompt
    if context_map:
        for placeholder, raw_value in context_map.items():
            context_str = json.dumps(raw_value) if not isinstance(raw_value, str) and raw_value is not None else (raw_value or ("{}" if placeholder.endswith("_json}") else ""))
            filled_prompt = filled_prompt.replace(placeholder, context_str)

    parsed_json, raw_text_or_err_code = await call_gemini_via_client_async_with_generativemodel(
        filled_prompt, images, step_name, model_name, timeout_seconds=timeout_val
    )
    if parsed_json is None:
        critical_errors = [
            "API_KEY_PERMISSION_DENIED", "[REDACTED_BY_SCRIPT]",
            "[REDACTED_BY_SCRIPT]", "[REDACTED_BY_SCRIPT]",
            "[REDACTED_BY_SCRIPT]", "[REDACTED_BY_SCRIPT]",
            "[REDACTED_BY_SCRIPT]", "RATE_LIMIT_DAILY_FINAL"
        ]
        if raw_text_or_err_code in critical_errors:
            log_to_queue(current_process_log_q, logging.CRITICAL, "[%s] Step '%s'[REDACTED_BY_SCRIPT]", current_process_log_ctx, step_name, raw_text_or_err_code)
            raise StopIteration(f"[REDACTED_BY_SCRIPT]")
        else: # Other non-critical failures from call_gemini (e.g. KEY_RETRIES_EXHAUSTED)
            log_to_queue(current_process_log_q, logging.ERROR, "[%s] Step '%s'[REDACTED_BY_SCRIPT]", current_process_log_ctx, step_name, raw_text_or_err_code)
    return parsed_json, raw_text_or_err_code

# --- Main Property Pipeline (for Worker) ---
async def process_property_pipeline_for_worker(
    property_address: str, base_image_address_dir: str, base_floorplan_address_dir: str,
    main_prop_output_dir: str, latest_year: int):
    global WORKER_CURRENT_API_KEY, UNKNOWN_TOKEN # UNKNOWN_TOKEN was already global in original
    global current_process_log_q, current_process_log_ctx # Set by worker_task_executor

    property_id_with_year = f"[REDACTED_BY_SCRIPT]"
    # current_process_log_ctx is set by worker_task_executor
    UNKNOWN_TOKEN = "UNKNOWN" # SAME AS ORIGINAL (USER'S SCRIPT)

    log_to_queue(current_process_log_q, logging.INFO, "[REDACTED_BY_SCRIPT]",
                 property_id_with_year,
                 WORKER_CURRENT_API_KEY[-4:] if WORKER_CURRENT_API_KEY else "None",
                 time.strftime('%H:%M:%S', time.localtime(WORKER_CURRENT_API_KEY_LEASE_EXPIRY)) if WORKER_CURRENT_API_KEY_LEASE_EXPIRY else "N/A",
                 WORKER_PROCESS_ID)

    # --- Image Loading and Deduplication (from original script) ---
    year_suffix = f"_y{latest_year}"
    property_specific_output_dir = os.path.join(main_prop_output_dir, property_address)
    os.makedirs(property_specific_output_dir, exist_ok=True)
    current_image_dir = os.path.join(base_image_address_dir, str(latest_year))
    current_floorplan_dir = os.path.join(base_floorplan_address_dir, str(latest_year))
    floorplan_image = None; indexed_room_images = {}
    
    floorplan_image_paths = get_floorplan_images(current_floorplan_dir, current_process_log_q, property_id_with_year)
    if floorplan_image_paths:
        selected_fp_path = None; max_a = -1
        for fp_path_iter in floorplan_image_paths: # Renamed to avoid conflict
            try:
                with Image.open(fp_path_iter) as img_fp: area = img_fp.width * img_fp.height
                if area > max_a: max_a, selected_fp_path = area, fp_path_iter
            except Exception as e_fp_sel: log_to_queue(current_process_log_q, logging.WARNING, "[REDACTED_BY_SCRIPT]", property_id_with_year, fp_path_iter, e_fp_sel)
        if selected_fp_path:
            floorplan_image = load_image(selected_fp_path, current_process_log_q, property_id_with_year)
            if floorplan_image: log_to_queue(current_process_log_q, logging.INFO, "[REDACTED_BY_SCRIPT]", property_id_with_year, os.path.basename(selected_fp_path))
            else: log_to_queue(current_process_log_q, logging.WARNING, "[REDACTED_BY_SCRIPT]", property_id_with_year, selected_fp_path)
    if not floorplan_image: log_to_queue(current_process_log_q, logging.INFO, "[REDACTED_BY_SCRIPT]", property_id_with_year)

    initial_room_paths = get_room_images(current_image_dir, floorplan_image_paths if floorplan_image else [], current_process_log_q, property_id_with_year)
    if not initial_room_paths: log_to_queue(current_process_log_q, logging.WARNING, "[REDACTED_BY_SCRIPT]", property_id_with_year)

    unique_room_paths = []
    if initial_room_paths:
        seen_hashes_tmp = set()
        for img_pth_tmp in initial_room_paths:
            img_for_hash = None # Initialize
            try:
                # Attempt to open the image just for hashing
                # Use a simpler open without verify/load for speed if it's just for phash
                img_for_hash = Image.open(img_pth_tmp) 
                if img_for_hash:
                    h = imagehash.phash(img_for_hash)
                    if h not in seen_hashes_tmp:
                        seen_hashes_tmp.add(h)
                        unique_room_paths.append(img_pth_tmp)
                    # else: # Optional: log if it's a duplicate hash
                    #    log_to_queue(current_process_log_q, logging.DEBUG, "[REDACTED_BY_SCRIPT]", property_id_with_year, os.path.basename(img_pth_tmp))
                else:
                    log_to_queue(current_process_log_q, logging.WARNING, "[REDACTED_BY_SCRIPT]", property_id_with_year, os.path.basename(img_pth_tmp))

            except FileNotFoundError:
                log_to_queue(current_process_log_q, logging.WARNING, "[REDACTED_BY_SCRIPT]", property_id_with_year, os.path.basename(img_pth_tmp))
            except UnidentifiedImageError:
                log_to_queue(current_process_log_q, logging.WARNING, "[REDACTED_BY_SCRIPT]", property_id_with_year, os.path.basename(img_pth_tmp))
            except Exception as e_hash:
                log_to_queue(current_process_log_q, logging.WARNING, "[REDACTED_BY_SCRIPT]", property_id_with_year, os.path.basename(img_pth_tmp), e_hash)
            finally:
                if img_for_hash: # Ensure image opened for hashing is closed
                    img_for_hash.close()
        
        log_to_queue(current_process_log_q, logging.INFO, "[REDACTED_BY_SCRIPT]", property_id_with_year, len(unique_room_paths), len(initial_room_paths))
        
    if unique_room_paths:
        pil_room_images_list = [img for img in [load_image(p, current_process_log_q, property_id_with_year) for p in unique_room_paths] if img is not None]
        if not pil_room_images_list: log_to_queue(current_process_log_q, logging.WARNING, "[REDACTED_BY_SCRIPT]", property_id_with_year)
        indexed_room_images = {f"Image {i+1}": img for i, img in enumerate(pil_room_images_list)}
    
    img_map_fn = os.path.join(property_specific_output_dir, f"[REDACTED_BY_SCRIPT]")
    try:
        paths_for_map = {f"Image {i+1}": p for i, p in enumerate(unique_room_paths)}
        with open(img_map_fn, "w", encoding="utf-8") as f_map: json.dump(paths_for_map, f_map, indent=2)
        log_to_queue(current_process_log_q, logging.DEBUG, "[REDACTED_BY_SCRIPT]", property_id_with_year, len(paths_for_map))
    except Exception as e_map_save: log_to_queue(current_process_log_q, logging.ERROR, "[REDACTED_BY_SCRIPT]", property_id_with_year, e_map_save, exc_info=True)
    # --- End Image Loading ---

    property_token_summary = {"room_labels": set(), "features": set(), "sp_themes": set(), "flaw_themes": set()}
    untagged_text_log = {"[REDACTED_BY_SCRIPT]": set(), "raw_features": set(), "[REDACTED_BY_SCRIPT]": set(), "raw_flaws_text": set()}
    output1_json, output1_text = None, None; output2_json, output2_text = None, None
    output3_json, output3_text = None, None; output4_json, output4_text = None, None
    output5a_json = None; output5b_json = None # Raw outputs from steps 5a/b
    merged_step5_output = None; output6_json = None
    pipeline_successful_for_csv_generation = True
    default_timeout, long_timeout = 180, 360

    try: # Main pipeline steps
        # Step 1: Floorplan
        if floorplan_image:
            output1_json, _ = await call_step_with_client_retry(prompt1_floorplan, floorplan_image, "[REDACTED_BY_SCRIPT]", model_25_flash_lite, default_timeout, current_process_log_q, property_id_with_year)
            if output1_json:
                output1_text = json.dumps(output1_json)
                if 'rooms_with_dimensions' in output1_json and isinstance(output1_json['rooms_with_dimensions'], list): # Bedroom processing from original
                    log_to_queue(current_process_log_q, logging.DEBUG, "[REDACTED_BY_SCRIPT]", property_id_with_year)
                for room_data in output1_json.get('rooms_with_dimensions', []): # Tokenization
                    if isinstance(room_data, dict) and 'label' in room_data: 
                        token = get_room_token(room_data['label'])
                        (untagged_text_log["[REDACTED_BY_SCRIPT]"].add(room_data['label']) if token == UNKNOWN_TOKEN else property_token_summary["room_labels"].add(token)) if token else untagged_text_log["[REDACTED_BY_SCRIPT]"].add(f"[REDACTED_BY_SCRIPT]'label']}")
                for label_str in output1_json.get('[REDACTED_BY_SCRIPT]', []):
                    if isinstance(label_str, str):
                        token = get_room_token(label_str)
                        (untagged_text_log["[REDACTED_BY_SCRIPT]"].add(label_str) if token == UNKNOWN_TOKEN else property_token_summary["room_labels"].add(token)) if token else untagged_text_log["[REDACTED_BY_SCRIPT]"].add(f"[REDACTED_BY_SCRIPT]")
            else:
                log_to_queue(current_process_log_q, logging.WARNING, "[REDACTED_BY_SCRIPT]", property_id_with_year)
                # output1_json remains None. Fallback logic later handles this.
        else:
            log_to_queue(current_process_log_q, logging.INFO, "[REDACTED_BY_SCRIPT]", property_id_with_year)

        # Step 2: Room Assignments
        if not indexed_room_images: raise StopIteration("[REDACTED_BY_SCRIPT]")
        output2_json, _ = await call_step_with_client_retry(prompt2_assignments, indexed_room_images, "[REDACTED_BY_SCRIPT]", model_25_pro, long_timeout, current_process_log_q, property_id_with_year, context_map={"{" + "floorplan_data_json" + "}": output1_text if output1_text else "{}"})
        if not output2_json or not isinstance(output2_json.get('room_assignments'), list): raise StopIteration("[REDACTED_BY_SCRIPT]")
        output2_text = json.dumps(output2_json)
        for assignment in output2_json.get('room_assignments', []): # Tokenization
            if isinstance(assignment, dict): 
                label = assignment.get('label'); source = assignment.get('source')
                if label:
                    token = get_room_token(label)
                    (untagged_text_log["[REDACTED_BY_SCRIPT]"].add(label) if token == UNKNOWN_TOKEN else property_token_summary["room_labels"].add(token)) if token else untagged_text_log["[REDACTED_BY_SCRIPT]"].add(f"[REDACTED_BY_SCRIPT]")
                if source == 'Floorplan': property_token_summary["features"].add('[REDACTED_BY_SCRIPT]')
                elif source == 'Generated': property_token_summary["features"].add('[REDACTED_BY_SCRIPT]')

        # Fallback Step 1 JSON (from original script)
        if output1_json is None:
            if output2_json and 'room_assignments' in output2_json and output2_json['room_assignments']:
                log_to_queue(current_process_log_q, logging.INFO, "[REDACTED_BY_SCRIPT]", property_id_with_year)
                new_s1_rooms_wd = []
                seen_labels_fallback = set()
                for assignment in output2_json.get('room_assignments', []):
                    if isinstance(assignment, dict) and 'label' in assignment:
                        label = assignment['label']
                        if label and label not in seen_labels_fallback:
                            new_s1_rooms_wd.append({"label": label, "dimensions": "null"})
                            seen_labels_fallback.add(label)
                if new_s1_rooms_wd:
                    output1_json = {"rooms_with_dimensions": new_s1_rooms_wd, "[REDACTED_BY_SCRIPT]": []}
                    output1_text = json.dumps(output1_json) # Update text for saving
                    log_to_queue(current_process_log_q, logging.INFO, "[REDACTED_BY_SCRIPT]", property_id_with_year, len(new_s1_rooms_wd))
                else: log_to_queue(current_process_log_q, logging.WARNING, "[REDACTED_BY_SCRIPT]", property_id_with_year)
            if output1_json is None: # Still None, create empty to ensure file is saved
                log_to_queue(current_process_log_q, logging.WARNING, "[REDACTED_BY_SCRIPT]", property_id_with_year)
                output1_json = {"rooms_with_dimensions": [], "[REDACTED_BY_SCRIPT]": []}
                output1_text = json.dumps(output1_json)
        
        # Step 3: Feature Extraction
        output3_json, _ = await call_step_with_client_retry(prompt3_features, indexed_room_images, "[REDACTED_BY_SCRIPT]", model_25_flash_lite, default_timeout, current_process_log_q, property_id_with_year, context_map={"{" + "[REDACTED_BY_SCRIPT]" + "}": output2_text})
        if not output3_json or not isinstance(output3_json, dict): raise StopIteration("Step 3 Failed.")
        output3_text = json.dumps(output3_json)
        all_feature_tokens = set() # Tokenization
        for room_label_s3, features_list_s3 in output3_json.items():
            if isinstance(features_list_s3, list):
                for feature_text_s3 in features_list_s3: 
                    if isinstance(feature_text_s3, str): untagged_text_log["raw_features"].add(feature_text_s3)
                room_feature_tokens_s3 = get_feature_tokens(features_list_s3); all_feature_tokens.update(room_feature_tokens_s3)
        property_token_summary["features"].update(all_feature_tokens)

        # Step 4: Flaws/SPs
        output4_json, _ = await call_step_with_client_retry(prompt4_flaws_sp_categorized, indexed_room_images, "Step 4: Flaws/SPs", model_25_pro, default_timeout, current_process_log_q, property_id_with_year, context_map={"{" + "[REDACTED_BY_SCRIPT]" + "}": output2_text, "{" + "[REDACTED_BY_SCRIPT]" + "}": output3_text})
        if not output4_json or not isinstance(output4_json, dict): raise StopIteration("Step 4 Failed.")
        output4_text = json.dumps(output4_json)
        # Tokenization for Step 4 (SP and Flaw tags)
        all_sp_theme_tokens = set(); all_flaw_theme_tokens = set()
        for room_label_s4, eval_data_s4 in output4_json.items():
            if isinstance(eval_data_s4, dict):
                sp_list = eval_data_s4.get("selling_points", [])
                flaw_list = eval_data_s4.get("flaws", [])
                all_sp_theme_tokens.update(get_sp_theme_tokens(sp_list))
                all_flaw_theme_tokens.update(get_flaw_theme_tokens(flaw_list))
                for sp_item in sp_list: untagged_text_log["[REDACTED_BY_SCRIPT]"].add(sp_item.get("text",""))
                for flaw_item in flaw_list: untagged_text_log["raw_flaws_text"].add(flaw_item.get("text",""))
        property_token_summary["sp_themes"].update(all_sp_theme_tokens)
        property_token_summary["flaw_themes"].update(all_flaw_theme_tokens)

        # Step 5a & 5b
        output5a_json, _ = await call_step_with_client_retry(prompt5a_ratings_p1_10, indexed_room_images, "[REDACTED_BY_SCRIPT]", model_25_pro, long_timeout, current_process_log_q, property_id_with_year, context_map={"{" + "[REDACTED_BY_SCRIPT]" + "}": output2_text, "{" + "[REDACTED_BY_SCRIPT]" + "}": output3_text, "{" + "[REDACTED_BY_SCRIPT]" + "}": output4_text, "{" + "[REDACTED_BY_SCRIPT]" + "}": persona_details_p1_10})
        if not output5a_json: raise StopIteration("Step 5a Failed.")
        output5b_json, _ = await call_step_with_client_retry(prompt5b_ratings_p11_20, indexed_room_images, "[REDACTED_BY_SCRIPT]", model_25_pro, long_timeout, current_process_log_q, property_id_with_year, context_map={"{" + "[REDACTED_BY_SCRIPT]" + "}": output2_text, "{" + "[REDACTED_BY_SCRIPT]" + "}": output3_text, "{" + "[REDACTED_BY_SCRIPT]" + "}": output4_text, "{" + "[REDACTED_BY_SCRIPT]" + "}": persona_details_p11_20})
        if not output5b_json: raise StopIteration("Step 5b Failed.")

        # Step 5 Merge (from original script)
        try:
            if not isinstance(output5a_json, dict) or not isinstance(output5b_json, dict): raise ValueError("[REDACTED_BY_SCRIPT]")
            labels_5a = output5a_json.get("[REDACTED_BY_SCRIPT]"); ratings_5a = output5a_json.get("room_ratings_p1_10"); overall_5a = output5a_json.get("[REDACTED_BY_SCRIPT]", {})
            labels_5b = output5b_json.get("[REDACTED_BY_SCRIPT]"); ratings_5b = output5b_json.get("room_ratings_p11_20"); overall_5b = output5b_json.get("[REDACTED_BY_SCRIPT]", {})
            if not (isinstance(labels_5a, list) and isinstance(ratings_5a, list) and isinstance(labels_5b, list) and isinstance(ratings_5b, list)): raise ValueError("[REDACTED_BY_SCRIPT]")
            
            if labels_5a != labels_5b: # Attempt reorder if sets are same
                log_to_queue(current_process_log_q, logging.DEBUG, "[REDACTED_BY_SCRIPT]", property_id_with_year, labels_5a, labels_5b)
                if set(labels_5a) == set(labels_5b) and len(labels_5a) == len(labels_5b):
                    label_to_index_5b = {label: i for i, label in enumerate(labels_5b)}
                    if not all(lbl_5a in label_to_index_5b for lbl_5a in labels_5a): raise ValueError("[REDACTED_BY_SCRIPT]")
                    new_ratings_5b_reordered = []
                    for target_label_5a in labels_5a:
                        idx_in_5b = label_to_index_5b[target_label_5a]
                        if idx_in_5b < len(ratings_5b): new_ratings_5b_reordered.append(ratings_5b[idx_in_5b])
                        else: new_ratings_5b_reordered.append([None]*10) # Safety for out of bounds
                    ratings_5b = new_ratings_5b_reordered
                    log_to_queue(current_process_log_q, logging.DEBUG, "[REDACTED_BY_SCRIPT]", property_id_with_year)
                else: raise ValueError(f"[REDACTED_BY_SCRIPT]")
            
            if len(labels_5a) != len(ratings_5a) or len(labels_5a) != len(ratings_5b): raise ValueError("[REDACTED_BY_SCRIPT]")
            
            merged_step5_output = {"[REDACTED_BY_SCRIPT]": labels_5a, "[REDACTED_BY_SCRIPT]": {**overall_5a, **overall_5b}, "room_ratings_final": []}
            for i_merge, label_merge in enumerate(labels_5a):
                list_5a_room = ratings_5a[i_merge] if i_merge < len(ratings_5a) and isinstance(ratings_5a[i_merge], list) and len(ratings_5a[i_merge]) == 11 else [None]*11
                list_5b_room = ratings_5b[i_merge] if i_merge < len(ratings_5b) and isinstance(ratings_5b[i_merge], list) and len(ratings_5b[i_merge]) == 10 else [None]*10
                merged_room_ratings_no_avg = list_5a_room + list_5b_room # Should be 21 elements
                persona_ratings_only = merged_room_ratings_no_avg[1:] # Exclude general rating
                valid_persona_ratings = [r for r in persona_ratings_only if isinstance(r, (int, float))]
                average_rating = round(statistics.mean(valid_persona_ratings), 1) if valid_persona_ratings else None
                merged_step5_output["room_ratings_final"].append(merged_room_ratings_no_avg + [average_rating]) # Total 22 elements
            log_to_queue(current_process_log_q, logging.INFO, "[REDACTED_BY_SCRIPT]", property_id_with_year)
        except Exception as e_merge:
            log_to_queue(current_process_log_q, logging.ERROR, "[REDACTED_BY_SCRIPT]", property_id_with_year, e_merge, exc_info=True)
            merged_step5_output = None; pipeline_successful_for_csv_generation = False

        # Step 6: Renovation
        output6_json, _ = await call_step_with_client_retry(prompt6_renovation, indexed_room_images, "[REDACTED_BY_SCRIPT]", model_25_flash_lite, default_timeout, current_process_log_q, property_id_with_year, context_map={"{" + "[REDACTED_BY_SCRIPT]" + "}": output2_text, "{" + "[REDACTED_BY_SCRIPT]" + "}": output3_text})
        if not output6_json: log_to_queue(current_process_log_q, logging.WARNING, "[REDACTED_BY_SCRIPT]", property_id_with_year)

    except StopIteration as si:
        log_to_queue(current_process_log_q, logging.WARNING, "[REDACTED_BY_SCRIPT]", property_id_with_year, si)
        pipeline_successful_for_csv_generation = False
        # Ensure we return something picklable if this is caught by worker_task_executor
        # This function (process_property_pipeline_for_worker) needs to return the tuple.
        # The StopIteration itself is not the issue for pickling usually.

    except Exception as e_pipe_critical: # This is the block we are interested in
        # Log the ORIGINAL exception type and message FIRST
        print("[REDACTED_BY_SCRIPT]")
        print(traceback.format_exc())
        print("----------------------------------")
        original_error_type = type(e_pipe_critical).__name__
        original_error_message = str(e_pipe_critical)
        log_to_queue(current_process_log_q, logging.CRITICAL,
                     "[REDACTED_BY_SCRIPT]",
                     property_id_with_year, original_error_type, original_error_message,
                     exc_info=True) # exc_info=True will log the full traceback of e_pipe_critical

        pipeline_successful_for_csv_generation = False
    finally:
        # Save all JSONs
        raw_outputs_to_save = {
            f"[REDACTED_BY_SCRIPT]": output1_json,
            f"[REDACTED_BY_SCRIPT]": output2_json,
            f"[REDACTED_BY_SCRIPT]": output3_json,
            f"[REDACTED_BY_SCRIPT]": output4_json,
            f"[REDACTED_BY_SCRIPT]": merged_step5_output,
            f"[REDACTED_BY_SCRIPT]": output6_json,
            f"[REDACTED_BY_SCRIPT]": {k: sorted(list(v)) for k, v in property_token_summary.items()},
            f"[REDACTED_BY_SCRIPT]": {k: sorted(list(v)) for k, v in untagged_text_log.items()},
        }
        for fname, data_to_save in raw_outputs_to_save.items(): # fname already has suffix
            filepath_to_save = os.path.join(property_specific_output_dir, fname)
            if data_to_save is not None:
                try:
                    with open(filepath_to_save, "w", encoding="utf-8") as f_out: json.dump(data_to_save, f_out, indent=2, ensure_ascii=False)
                except Exception as e_save_json: log_to_queue(current_process_log_q, logging.ERROR, "[REDACTED_BY_SCRIPT]", property_id_with_year, fname, e_save_json, exc_info=True)
        
        features_for_csv = None # Initialize
        if pipeline_successful_for_csv_generation:
            # Check required files for feature generator (from original script)
            path_s1_chk = os.path.join(property_specific_output_dir, f'[REDACTED_BY_SCRIPT]')
            path_s4_chk = os.path.join(property_specific_output_dir, f'[REDACTED_BY_SCRIPT]')
            path_s5m_chk = os.path.join(property_specific_output_dir, f'[REDACTED_BY_SCRIPT]')
            required_files_for_feat_gen_exist = all(os.path.exists(f) for f in [path_s1_chk, path_s4_chk, path_s5m_chk]) # As per original
            
            if required_files_for_feat_gen_exist:
                try:
                    import gemini_property_feature_generator # Ensure import
                    features_for_csv = gemini_property_feature_generator.process_property(
                        property_id=property_id_with_year,
                        input_dir=property_specific_output_dir, # Pass the property's specific output dir
                        year_suffix=year_suffix 
                    )
                    if features_for_csv and isinstance(features_for_csv, dict):
                        log_to_queue(current_process_log_q, logging.INFO, "[REDACTED_BY_SCRIPT]", property_id_with_year)
                    else: log_to_queue(current_process_log_q, logging.WARNING, "[REDACTED_BY_SCRIPT]", property_id_with_year)
                except ImportError: log_to_queue(current_process_log_q, logging.CRITICAL, "[REDACTED_BY_SCRIPT]'gemini_property_feature_generator'.", property_id_with_year)
                except Exception as e_feat_csv: log_to_queue(current_process_log_q, logging.ERROR, "[REDACTED_BY_SCRIPT]", property_id_with_year, e_feat_csv, exc_info=True)
            else: 
                missing_ff = [os.path.basename(f) for f in [path_s1_chk,path_s4_chk,path_s5m_chk] if not os.path.exists(f)]
                log_to_queue(current_process_log_q, logging.WARNING, "[REDACTED_BY_SCRIPT]", property_id_with_year, ', '.join(missing_ff))

        if floorplan_image and hasattr(floorplan_image, 'close'): floorplan_image.close()
        for _k_img, img_obj_img in indexed_room_images.items():
            if hasattr(img_obj_img, 'close'): img_obj_img.close()
        del floorplan_image, indexed_room_images
        import gc; gc.collect() # Optional

        log_to_queue(current_process_log_q, logging.INFO, "[REDACTED_BY_SCRIPT]", property_id_with_year)
        return property_id_with_year, features_for_csv

def init_worker_globals(shared_api_key_manager_init, log_q_init):
    global WORKER_SHARED_API_KEY_MANAGER, WORKER_PROCESS_ID, current_process_log_q
    global WORKER_CURRENT_API_KEY, WORKER_CURRENT_API_KEY_LEASE_EXPIRY
    WORKER_SHARED_API_KEY_MANAGER = shared_api_key_manager_init
    WORKER_PROCESS_ID = multiprocessing.current_process().name
    current_process_log_q = log_q_init
    WORKER_CURRENT_API_KEY = None # Will be fetched by worker_task_executor
    WORKER_CURRENT_API_KEY_LEASE_EXPIRY = None
    log_to_queue(current_process_log_q, logging.DEBUG, "[REDACTED_BY_SCRIPT]", WORKER_PROCESS_ID)

# --- Worker Function (Executor) ---
def worker_task_executor(task_args_tuple):
    global WORKER_CURRENT_API_KEY, WORKER_CURRENT_API_KEY_LEASE_EXPIRY
    global WORKER_SHARED_API_KEY_MANAGER, WORKER_PROCESS_ID
    global current_process_log_q, current_process_log_ctx

    ( property_address, base_image_address_dir, base_floorplan_address_dir,
      main_prop_output_dir_for_worker, latest_year,
      csv_results_q
    ) = task_args_tuple

    property_id_for_log = f"[REDACTED_BY_SCRIPT]"
    current_process_log_ctx = property_id_for_log

    log_to_queue(current_process_log_q, logging.INFO, "[REDACTED_BY_SCRIPT]", WORKER_PROCESS_ID, current_process_log_ctx)

    pipeline_complete = False
    MAX_PIPELINE_RETRIES = 3 # You can adjust this
    pipeline_retry_count = 0
    final_task_failure_reason = "[REDACTED_BY_SCRIPT]" # Default if loop finishes

    while not pipeline_complete and pipeline_retry_count < MAX_PIPELINE_RETRIES:
        pipeline_retry_count += 1
        log_to_queue(current_process_log_q, logging.INFO,
                     "[REDACTED_BY_SCRIPT]",
                     current_process_log_ctx, WORKER_PROCESS_ID, pipeline_retry_count, MAX_PIPELINE_RETRIES)

        key_permanently_exhausted_this_attempt = False
        WORKER_CURRENT_API_KEY, WORKER_CURRENT_API_KEY_LEASE_EXPIRY = WORKER_SHARED_API_KEY_MANAGER.get_api_key(WORKER_PROCESS_ID)

        if not WORKER_CURRENT_API_KEY:
            log_to_queue(current_process_log_q, logging.ERROR,
                         "[REDACTED_BY_SCRIPT]",
                         current_process_log_ctx, WORKER_PROCESS_ID, pipeline_retry_count)
            final_task_failure_reason = "[REDACTED_BY_SCRIPT]"
            if pipeline_retry_count < MAX_PIPELINE_RETRIES:
                log_to_queue(current_process_log_q, logging.WARNING, "[REDACTED_BY_SCRIPT]", current_process_log_ctx, WORKER_PROCESS_ID)
                time.sleep(30 * pipeline_retry_count) # Wait before next *pipeline attempt*
                continue # Try to get a key again in the next iteration
            else: # Max retries for getting a key specifically
                break # Exit while loop, task will be marked as failed

        log_to_queue(current_process_log_q, logging.DEBUG,
                     "[REDACTED_BY_SCRIPT]",
                     current_process_log_ctx, WORKER_PROCESS_ID, WORKER_CURRENT_API_KEY[-4:], pipeline_retry_count)

        returned_prop_id = current_process_log_ctx # Should be set by successful pipeline run
        features_dict_for_csv = None
        try:
            returned_prop_id, features_dict_for_csv = asyncio.run(
                process_property_pipeline_for_worker(
                    property_address, base_image_address_dir, base_floorplan_address_dir,
                    main_prop_output_dir_for_worker, latest_year
                )
            )
            # UPDATED: Only consider pipeline complete if features_dict_for_csv is populated
            if features_dict_for_csv and isinstance(features_dict_for_csv, dict):
                csv_results_q.put(features_dict_for_csv)
                log_to_queue(current_process_log_q, logging.INFO, "[REDACTED_BY_SCRIPT]", current_process_log_ctx, WORKER_PROCESS_ID, pipeline_retry_count)
                pipeline_complete = True # Successful completion
            elif returned_prop_id: # Pipeline ran, returned an ID, but no CSV features
                log_to_queue(current_process_log_q, logging.WARNING,
                             "[REDACTED_BY_SCRIPT]",
                             current_process_log_ctx, WORKER_PROCESS_ID, pipeline_retry_count)
                final_task_failure_reason = "[REDACTED_BY_SCRIPT]"
                # DO NOT set pipeline_complete = True here, let it loop to retry
                if pipeline_retry_count < MAX_PIPELINE_RETRIES:
                    time.sleep(10 * pipeline_retry_count) # Short sleep before retrying for "no CSV" case
            else: # Should not happen if process_property_pipeline_for_worker always returns an ID
                log_to_queue(current_process_log_q, logging.ERROR,
                             "[REDACTED_BY_SCRIPT]",
                             current_process_log_ctx, WORKER_PROCESS_ID, pipeline_retry_count)
                final_task_failure_reason = "[REDACTED_BY_SCRIPT]"
                if pipeline_retry_count < MAX_PIPELINE_RETRIES:
                    time.sleep(15 * pipeline_retry_count)


        except RuntimeError as r_err:
            original_exception_str = str(r_err)
            stop_iteration_msg = original_exception_str # Default for logging
            is_critical_stop = False

            if "[REDACTED_BY_SCRIPT]" in original_exception_str or isinstance(r_err.__cause__, StopIteration):
                log_to_queue(current_process_log_q, logging.ERROR,
                             "[REDACTED_BY_SCRIPT]",
                             current_process_log_ctx, WORKER_PROCESS_ID, pipeline_retry_count, stop_iteration_msg, WORKER_CURRENT_API_KEY[-4:] if WORKER_CURRENT_API_KEY else "None")
                is_critical_stop = True
                critical_stop_substrings = [
                    "API_KEY_PERMISSION_DENIED", "[REDACTED_BY_SCRIPT]",
                    "RATE_LIMIT_DAILY_FINAL", "[REDACTED_BY_SCRIPT]",
                    "[REDACTED_BY_SCRIPT]"
                ]
                if any(substring in stop_iteration_msg for substring in critical_stop_substrings):
                    key_permanently_exhausted_this_attempt = True
                final_task_failure_reason = f"[REDACTED_BY_SCRIPT]"
            else: # It was a different RuntimeError not wrapping StopIteration
                log_to_queue(current_process_log_q, logging.CRITICAL,
                         "[REDACTED_BY_SCRIPT]",
                         current_process_log_ctx, WORKER_PROCESS_ID, pipeline_retry_count,
                         r_err, WORKER_CURRENT_API_KEY[-4:] if WORKER_CURRENT_API_KEY else "None",
                         exc_info=True)
                final_task_failure_reason = f"[REDACTED_BY_SCRIPT]"

            if pipeline_retry_count >= MAX_PIPELINE_RETRIES:
                log_to_queue(current_process_log_q, logging.CRITICAL,
                             "[REDACTED_BY_SCRIPT]",
                             current_process_log_ctx, WORKER_PROCESS_ID, "critical halt" if is_critical_stop else "RuntimeError", final_task_failure_reason)
                # No CSV put here for error, handled by overall failure logic after loop
            else:
                log_to_queue(current_process_log_q, logging.WARNING,
                             "[REDACTED_BY_SCRIPT]",
                             current_process_log_ctx, WORKER_PROCESS_ID, "critical halt" if is_critical_stop else "RuntimeError")
                time.sleep(20 * pipeline_retry_count) # Wait before retrying the whole pipeline

        except Exception as e_worker_exc: # Catch other exceptions
            log_to_queue(current_process_log_q, logging.CRITICAL,
                         "[REDACTED_BY_SCRIPT]",
                         current_process_log_ctx, WORKER_PROCESS_ID, pipeline_retry_count,
                         type(e_worker_exc).__name__, str(e_worker_exc), WORKER_CURRENT_API_KEY[-4:] if WORKER_CURRENT_API_KEY else "None",
                         exc_info=True)
            final_task_failure_reason = f"[REDACTED_BY_SCRIPT]"
            if pipeline_retry_count >= MAX_PIPELINE_RETRIES:
                log_to_queue(current_process_log_q, logging.CRITICAL,
                             "[REDACTED_BY_SCRIPT]",
                             current_process_log_ctx, WORKER_PROCESS_ID, e_worker_exc)
            else:
                log_to_queue(current_process_log_q, logging.WARNING,
                             "[REDACTED_BY_SCRIPT]",
                             current_process_log_ctx, WORKER_PROCESS_ID)
                time.sleep(30 * pipeline_retry_count) # Longer sleep for unknown errors

        finally:
            if WORKER_CURRENT_API_KEY: # Release key used for this attempt
                WORKER_SHARED_API_KEY_MANAGER.release_api_key(
                    WORKER_PROCESS_ID,
                    WORKER_CURRENT_API_KEY,
                    was_permanently_exhausted=key_permanently_exhausted_this_attempt
                )
                WORKER_CURRENT_API_KEY = None # Clear it so it's re-fetched if loop continues
                WORKER_CURRENT_API_KEY_LEASE_EXPIRY = None
            log_to_queue(current_process_log_q, logging.DEBUG,
                         "[REDACTED_BY_SCRIPT]",
                         current_process_log_ctx, WORKER_PROCESS_ID, pipeline_retry_count)

    # After the while loop (either completed or max retries reached)
    if not pipeline_complete:
        log_to_queue(current_process_log_q, logging.CRITICAL,
                     "[REDACTED_BY_SCRIPT]",
                     current_process_log_ctx, WORKER_PROCESS_ID, pipeline_retry_count, final_task_failure_reason)
        # UPDATED: Only send error marker to CSV if all retries failed
        csv_results_q.put({"error_processing": current_process_log_ctx, "details": final_task_failure_reason})

    log_to_queue(current_process_log_q, logging.INFO, "[REDACTED_BY_SCRIPT]", WORKER_PROCESS_ID, current_process_log_ctx)
    return

# --- Utility functions ---
def find_latest_year_for_main(base_prop_image_dir: str, log_q: multiprocessing.Queue, prop_addr: str) -> int | None:
    latest_year_val = None
    if os.path.isdir(base_prop_image_dir):
        try:
            valid_years = [ int(item) for item in os.listdir(base_prop_image_dir) if os.path.isdir(os.path.join(base_prop_image_dir, item)) and item.isdigit() and len(item) == 4 ]
            if valid_years: latest_year_val = max(valid_years)
        except Exception as e_year_find: log_to_queue(log_q, logging.ERROR, "[REDACTED_BY_SCRIPT]", prop_addr, base_prop_image_dir, e_year_find, exc_info=True)
    return latest_year_val

def get_floorplan_images(floorplan_dir, log_q, log_ctx):
    paths = []
    if not os.path.isdir(floorplan_dir): log_to_queue(log_q, logging.WARNING, "[REDACTED_BY_SCRIPT]", log_ctx, floorplan_dir); return []
    try:
        for fname in os.listdir(floorplan_dir):
            fpath = os.path.join(floorplan_dir, fname)
            if os.path.isfile(fpath) and any(fname.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']): paths.append(fpath)
    except Exception as e: log_to_queue(log_q, logging.WARNING, "[REDACTED_BY_SCRIPT]", log_ctx, floorplan_dir, e)
    return paths

def get_room_images(image_dir, floorplan_paths_exclude, log_q, log_ctx):
    paths = []
    if not os.path.isdir(image_dir): log_to_queue(log_q, logging.ERROR, "[REDACTED_BY_SCRIPT]", log_ctx, image_dir); return None # None indicates critical failure here
    norm_excludes = {os.path.normpath(p) for p in floorplan_paths_exclude}
    try:
        for fname in os.listdir(image_dir):
            fpath = os.path.join(image_dir, fname)
            if os.path.isfile(fpath) and any(fname.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png']) and os.path.normpath(fpath) not in norm_excludes: paths.append(fpath)
    except Exception as e: log_to_queue(log_q, logging.ERROR, "[REDACTED_BY_SCRIPT]", log_ctx, image_dir, e, exc_info=True); return None
    return paths

def load_image(path, log_q, log_ctx):
    try:
        img = Image.open(path) 
        if img is None: # Explicit check after open
            log_to_queue(log_q, logging.ERROR, "[REDACTED_BY_SCRIPT]", log_ctx, os.path.basename(path))
            return None
        
        # It's good practice to load the image data after verifying
        img.verify() # Verifies integrity, can raise exceptions for corrupt files.
                     # After verify(), the file pointer is often at the end.
                     # So, the image needs to be reopened to read pixel data.
        
        # Reopen the image to reset the file pointer and load pixel data
        img_reopened = Image.open(path)
        if img_reopened is None: # Check again after reopen
             log_to_queue(log_q, logging.ERROR, "[REDACTED_BY_SCRIPT]", log_ctx, os.path.basename(path))
             return None

        img_reopened.load() # Loads pixel data into memory
        
        return img_reopened.convert('RGB') if img_reopened.mode != 'RGB' else img_reopened
    
    except FileNotFoundError:
        log_to_queue(log_q, logging.ERROR, "[REDACTED_BY_SCRIPT]", log_ctx, path)
        return None
    except UnidentifiedImageError: # PIL couldn't identify the image format
        log_to_queue(log_q, logging.ERROR, "[REDACTED_BY_SCRIPT]", log_ctx, os.path.basename(path))
        return None
    except SyntaxError as se_pil: # SyntaxError can be raised by PIL for some corrupt JPEGs for example
         log_to_queue(log_q, logging.ERROR, "[REDACTED_BY_SCRIPT]", log_ctx, os.path.basename(path), se_pil, exc_info=False) # exc_info=False for brevity on common syntax errors
         return None
    except Exception as e: # Catch other PIL-related or general exceptions
        log_to_queue(log_q, logging.ERROR, "[REDACTED_BY_SCRIPT]", log_ctx, os.path.basename(path), type(e).__name__, e, exc_info=True)
        return None

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

def get_sp_theme_tokens(selling_points_list): # From original
    if not selling_points_list or not isinstance(selling_points_list, list): return []
    tokens = set()
    expected_sp_tags = {"SP_SPACE", "SP_LIGHT", "SP_CONDITION", "SP_MODERN", "SP_CHARACTER", "SP_STYLE", "SP_FUNCTIONAL", "SP_FEATURE", "SP_STORAGE", "SP_GARDEN_ACCESS", "SP_GARDEN_VIEW", "SP_OTHER_VIEW", "SP_LOCATION", "SP_PRIVACY", "SP_POTENTIAL", "SP_LOW_MAINTENANCE", "SP_QUALITY_FINISH"}
    for sp_item in selling_points_list:
        if isinstance(sp_item, dict) and 'tags' in sp_item and isinstance(sp_item['tags'], list):
            for tag in sp_item['tags']:
                if isinstance(tag, str) and tag in expected_sp_tags: tokens.add(tag)
    return list(tokens)

def get_flaw_theme_tokens(flaws_list): # From original
    if not flaws_list or not isinstance(flaws_list, list): return []
    tokens = set()
    expected_flaw_tags = {"FLAW_SPACE", "FLAW_LIGHT", "FLAW_CONDITION", "FLAW_DATED", "FLAW_NEEDS_UPDATE", "FLAW_MAINTENANCE", "FLAW_STORAGE", "FLAW_LAYOUT", "FLAW_BASIC_STYLE", "[REDACTED_BY_SCRIPT]", "FLAW_POOR_FINISH", "FLAW_UNATTRACTIVE", "FLAW_NOISE", "FLAW_ACCESSIBILITY"}
    for flaw_item in flaws_list: # This was 'flaws_list' should be 'flaw_item'
        if isinstance(flaw_item, dict) and 'tags' in flaw_item and isinstance(flaw_item['tags'], list):
            for tag in flaw_item['tags']: # Iterate through the tags of the current flaw_item
                 if isinstance(tag, str) and tag in expected_flaw_tags: tokens.add(tag)
    return list(tokens)

# --- Main Orchestrator ---
def main_multiprocessing_runner():
    global UNKNOWN_TOKEN # Defined for use in tokenization functions if they are called from main context (unlikely now)
    UNKNOWN_TOKEN = "UNKNOWN_FALLBACK" # Fallback if somehow needed in main, workers set their own.

    os.makedirs(MAIN_OUTPUT_DIR, exist_ok=True) # SAME AS ORIGINAL

    # UPDATED: Use a single manager for all shared resources
    manager = multiprocessing.Manager()
    log_queue = manager.Queue(-1) # Log queue setup - SAME AS ORIGINAL

    log_listener = multiprocessing.Process(target=log_listener_process, args=(log_queue, LOG_FILE_PATH)) # SAME AS ORIGINAL
    log_listener.start()
    log_to_queue(log_queue, logging.INFO, "[REDACTED_BY_SCRIPT]", MAX_WORKERS, logger_name_override="MainOrchestrator")

    # NEW: Instantiate ApiKeyManager
    api_key_manager = ApiKeyManager(ALL_API_KEYS, manager, log_queue)

    # CSV setup - SAME AS ORIGINAL (USER'S SCRIPT)
    try:
        import gemini_property_feature_generator # Make sure this module is available
        csv_header_list = gemini_property_feature_generator.get_feature_header()
        if not csv_header_list: raise ImportError("[REDACTED_BY_SCRIPT]")
    except ImportError as e_imp:
        log_to_queue(log_queue, logging.CRITICAL, "[REDACTED_BY_SCRIPT]", e_imp, logger_name_override="MainOrchestrator")
        log_queue.put(None); log_listener.join(); return
    csv_results_queue = manager.Queue(-1) # SAME AS ORIGINAL
    csv_writer = multiprocessing.Process(target=csv_writer_process, args=(csv_results_queue, MASTER_CSV_PATH, csv_header_list, log_queue)) # SAME AS ORIGINAL
    csv_writer.start()

    # Task identification phase - LARGELY SAME AS ORIGINAL (USER'S SCRIPT) logic, task_tuple is different
    log_to_queue(log_queue, logging.INFO, "[REDACTED_BY_SCRIPT]", logger_name_override="MainOrchestrator")
    completed_property_ids_from_csv = set()
    if os.path.isfile(MASTER_CSV_PATH): # CSV reading logic - SAME AS ORIGINAL (USER'S SCRIPT)
        try:
            with open(MASTER_CSV_PATH, 'r', newline='', encoding='utf-8') as cfr:
                first_char = cfr.read(1)
                if first_char:
                    cfr.seek(0)
                    reader = csv.DictReader(cfr)
                    if 'property_id' in reader.fieldnames:
                        for row in reader: completed_property_ids_from_csv.add(row['property_id'])
                    else:
                        log_to_queue(log_queue, logging.WARNING, "Master CSV 'property_id' column not found.", logger_name_override="MainOrchestrator")
                else:
                    log_to_queue(log_queue, logging.INFO, "[REDACTED_BY_SCRIPT]", logger_name_override="MainOrchestrator")
            log_to_queue(log_queue, logging.INFO, "[REDACTED_BY_SCRIPT]", len(completed_property_ids_from_csv), logger_name_override="MainOrchestrator")
        except Exception as e_csv_r: log_to_queue(log_queue, logging.ERROR, "[REDACTED_BY_SCRIPT]", e_csv_r, exc_info=True, logger_name_override="MainOrchestrator")

    tasks_to_process_args = []
    address_folders = [f for f in os.listdir(MAIN_IMAGE_DIR) if os.path.isdir(os.path.join(MAIN_IMAGE_DIR, f))] # SAME AS ORIGINAL
    log_to_queue(log_queue, logging.INFO, "[REDACTED_BY_SCRIPT]", len(address_folders), logger_name_override="MainOrchestrator")

    # Loop for task identification - LARGELY SAME AS ORIGINAL (USER'S SCRIPT)
    for property_address_folder_name in address_folders:
        # ... (directory path setup, find_latest_year, prop_id_year, year_suffix_for_check - all SAME AS ORIGINAL)
        prop_image_base_dir = os.path.join(MAIN_IMAGE_DIR, property_address_folder_name)
        prop_floorplan_base_dir = os.path.join(MAIN_FLOORPLAN_DIR, property_address_folder_name)
        prop_json_output_dir_for_check = os.path.join(MAIN_OUTPUT_DIR, property_address_folder_name)
        current_latest_year = find_latest_year_for_main(prop_image_base_dir, log_queue, property_address_folder_name)
        if current_latest_year is None:
            log_to_queue(log_queue, logging.WARNING, "[REDACTED_BY_SCRIPT]", property_address_folder_name, logger_name_override="MainOrchestrator")
            continue
        prop_id_year = f"[REDACTED_BY_SCRIPT]"
        year_suffix_for_check = f"[REDACTED_BY_SCRIPT]"

        # Checking for completeness of JSONs - SAME AS ORIGINAL (USER'S SCRIPT), with corrected filename for map
        output_folder_complete_check = False; missing_files_check = []
        if os.path.isdir(prop_json_output_dir_for_check):
            all_files_found = True
            for base_filename in EXPECTED_BASE_JSON_FILENAMES: # Corrected filename logic
                expected_filename = f"[REDACTED_BY_SCRIPT]" if base_filename == "image_paths_map" else f"[REDACTED_BY_SCRIPT]"
                expected_filepath = os.path.join(prop_json_output_dir_for_check, expected_filename)
                if not os.path.exists(expected_filepath):
                    all_files_found = False; missing_files_check.append(expected_filename); break
            if all_files_found: output_folder_complete_check = True

        # Logic for queuing tasks - SAME AS ORIGINAL (USER'S SCRIPT)
        if output_folder_complete_check:
            if prop_id_year in completed_property_ids_from_csv:
                log_to_queue(log_queue, logging.DEBUG, "[REDACTED_BY_SCRIPT]", prop_id_year, logger_name_override="MainOrchestrator") # DEBUG for less noise
                continue
            else:
                log_to_queue(log_queue, logging.INFO, "[REDACTED_BY_SCRIPT]", prop_id_year, logger_name_override="MainOrchestrator")
        elif missing_files_check:
            log_to_queue(log_queue, logging.INFO, "[REDACTED_BY_SCRIPT]", prop_id_year, ', '.join(missing_files_check), logger_name_override="MainOrchestrator")
        else:
            log_to_queue(log_queue, logging.INFO, "[REDACTED_BY_SCRIPT]", prop_id_year, logger_name_override="MainOrchestrator")

        # UPDATED: task_tuple no longer includes API key
        task_tuple = (
            property_address_folder_name, prop_image_base_dir, prop_floorplan_base_dir,
            MAIN_OUTPUT_DIR,
            current_latest_year,
            csv_results_queue # log_queue is passed via initializer
        )
        tasks_to_process_args.append(task_tuple)
    # --- End Task Identification ---

    if not tasks_to_process_args: # SAME AS ORIGINAL
        log_to_queue(log_queue, logging.INFO, "[REDACTED_BY_SCRIPT]", logger_name_override="MainOrchestrator")
    else:
        # UPDATED: num_actual_workers also limited by total API keys
        num_actual_workers = min(MAX_WORKERS, len(tasks_to_process_args), len(ALL_API_KEYS))
        if num_actual_workers == 0 and len(tasks_to_process_args) > 0 :
            log_to_queue(log_queue, logging.WARNING, "[REDACTED_BY_SCRIPT]", logger_name_override="MainOrchestrator")

        log_to_queue(log_queue, logging.INFO, "[REDACTED_BY_SCRIPT]", len(tasks_to_process_args), num_actual_workers, logger_name_override="MainOrchestrator")
        if num_actual_workers > 0:
            try:
                # UPDATED: Pool now uses initializer
                with multiprocessing.Pool(processes=num_actual_workers,
                                          initializer=init_worker_globals,
                                          initargs=(api_key_manager, log_queue)) as pool:
                    pool.map(worker_task_executor, tasks_to_process_args)
                log_to_queue(log_queue, logging.INFO, "[REDACTED_BY_SCRIPT]", logger_name_override="MainOrchestrator")
            except Exception as e_pool: # SAME AS ORIGINAL
                log_to_queue(log_queue, logging.CRITICAL, "[REDACTED_BY_SCRIPT]", e_pool, exc_info=True, logger_name_override="MainOrchestrator")
        else: # SAME AS ORIGINAL
            log_to_queue(log_queue, logging.INFO, "[REDACTED_BY_SCRIPT]", logger_name_override="MainOrchestrator")

    # Shutdown - SAME AS ORIGINAL (USER'S SCRIPT)
    log_to_queue(log_queue, logging.INFO, "[REDACTED_BY_SCRIPT]", logger_name_override="MainOrchestrator")
    csv_results_queue.put(None); csv_writer.join()
    log_to_queue(log_queue, logging.INFO, "CSV writer joined.", logger_name_override="MainOrchestrator")

    log_to_queue(log_queue, logging.INFO, "[REDACTED_BY_SCRIPT]", logger_name_override="MainOrchestrator")
    log_queue.put(None); log_listener.join()
    print(f"[REDACTED_BY_SCRIPT]")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    if sys.version_info < (3, 9): # zoneinfo check
        try: import pytz # Check if pytz is available as an alternative for ZoneInfo("UTC")
        except ImportError: print("[REDACTED_BY_SCRIPT]'pytz'[REDACTED_BY_SCRIPT]", file=sys.stderr)
    main_multiprocessing_runner()