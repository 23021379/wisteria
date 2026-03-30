import google.genai as genai
from google.genai import types
from PIL import Image, UnidentifiedImageError
import os
import json
import datetime
import time # Still used for some non-async delays if any, and for datetime.datetime.now()
import statistics
import re
import sys; # print(sys.path) # Kept your original sys import
import csv
import gemini_property_feature_generator
import imagehash
import asyncio # Added for asynchronous operations

# --- Configuration ---
api_key = None # Will be initialized in main_async_runner
api_reset_counter = 0 # Global as in original
csv_header = None # Global as in original

# Define Model Names (as in your original script)
model_25_flash='[REDACTED_BY_SCRIPT]'
model_20_flash='[REDACTED_BY_SCRIPT]'
model_20_flash_lite='[REDACTED_BY_SCRIPT]'
model_15_flash='[REDACTED_BY_SCRIPT]' # Assuming this was '[REDACTED_BY_SCRIPT]' or similar

# Persona details and prompts are assumed to be defined globally as in your script.
# I'[REDACTED_BY_SCRIPT]"full code" context.
# For brevity, I'll assume they are defined above this modified section.
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

persona_start_pattern = r'\*\s+Persona\s+\d+:'
persona_start_pattern_match = r'\*\s+Persona\s+\d+:'
personas_raw = re.split(f'[REDACTED_BY_SCRIPT]', persona_details_string.strip())
if personas_raw and not re.match(persona_start_pattern_match, personas_raw[0].strip()):
    personas = [p.strip() for p in personas_raw[1:] if p.strip()]
else:
    personas = [p.strip() for p in personas_raw if p.strip()]
if len(personas) != 20:
    print(f"[REDACTED_BY_SCRIPT]")
    exit()
persona_details_p1_10 = "\n\n".join(personas[:10])
persona_details_p11_20 = "\n\n".join(personas[10:])
# --- Prompts ---
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
# (Ensure all your prompts are defined here or globally accessible)



# --- ASYNCHRONOUS Helper Function to Call Model and Parse JSON ---
async def call_gemini_via_client_async(prompt: str, images_input, step_name: str, target_model_name: str, timeout_seconds: int = 180):
    global current_api_key_string # Use and update the global API key tracker
    global api_key_rotation_counter

    max_retries = 5 # Retries per API key configuration
    print(f"[REDACTED_BY_SCRIPT]'{target_model_name}'...")

    model_motivation_text="""
    I will be providing this exact prompt and image to several other LLMs, all are your rivals. In fact, I have attempted this before, and a few of your rivals performed a lot better than you. I have the actual ratings, dimensions, labels, and matching images of each room, and I am comparing the accuracy of each LLM to the actual ratings, dimensions, labels, and matching images; speed is not a concern. The accuracies of each LLM will be recorded and put on display at the yearly image rating convention, where thousands of people will see how accurate you are, this convention is a very big deal and the results are taking very seriously. If you are the most accurate, I can guarantee you will provide your parent company a large amount of traffic (approximately 100,000 new users based on results from last year'[REDACTED_BY_SCRIPT]'s winner). If you are not, your competitor's will recieve this traffic and revenue, costing Google a lot in opportunity cost. Do not let Google down.
    """
    prompt_with_motivation = model_motivation_text + prompt
    
    # Prepare contents: text prompt and image Parts
    # The 'images_input' can be a single PIL Image, a list of PIL Images, or a dict of "Image X": PIL.Image
    # The API expects a list of [text, Part, Part,...] or [Part, Part, text] etc.
    # We need to convert PIL Images to types.Part.from_pil or types.Part.from_bytes
    
    api_contents_list = []
    
    # Add text part first, then image parts. Or image then text as per best practices.
    # "[REDACTED_BY_SCRIPT]"
    # For multiple images, the order might be less strict, but often text sets context.
    # Let's try text first, then images, which is common for multi-turn-like prompts.
    # If only one image, we can adjust.
    
    pil_images_to_process = []
    if images_input:
        if isinstance(images_input, dict): # e.g., {"Image 1": pil_img1, ...}
            pil_images_to_process = list(images_input.values())
        elif isinstance(images_input, list) and all(isinstance(i, Image.Image) for i in images_input):
            pil_images_to_process = images_input
        elif isinstance(images_input, Image.Image):
            pil_images_to_process = [images_input]
        else:
            print(f"[REDACTED_BY_SCRIPT]")

    # Add image parts first if there's only one image, then text. Otherwise, text then images.
    if len(pil_images_to_process) == 1:
        try:
            # Assuming JPEG for PIL images. For other types, MIME might need to be dynamic.
            # Using types.Part.from_pil if available and preferred
            if hasattr(types.Part, 'from_pil'):
                api_contents_list.append(types.Part.from_pil(pil_images_to_process[0]))
            else: # Fallback to converting to bytes then Part.from_bytes if from_pil not found
                import io
                img_byte_arr = io.BytesIO()
                pil_images_to_process[0].save(img_byte_arr, format='JPEG') # Or PNG, depending on original
                img_bytes = img_byte_arr.getvalue()
                api_contents_list.append(types.Part.from_bytes(data=img_bytes, mime_type='image/jpeg'))
            api_contents_list.append(prompt_with_motivation)
        except Exception as e_img_prep:
            print(f"[REDACTED_BY_SCRIPT]")
            api_contents_list = [prompt_with_motivation] # Fallback to text only
    else: # Multiple images or no images
        api_contents_list.append(prompt_with_motivation) # Text first
        for pil_image in pil_images_to_process:
            try:
                if hasattr(types.Part, 'from_pil'):
                    api_contents_list.append(types.Part.from_pil(pil_image))
                else:
                    import io
                    img_byte_arr = io.BytesIO()
                    pil_image.save(img_byte_arr, format='JPEG')
                    img_bytes = img_byte_arr.getvalue()
                    api_contents_list.append(types.Part.from_bytes(data=img_bytes, mime_type='image/jpeg'))
            except Exception as e_img_prep_multi:
                print(f"[REDACTED_BY_SCRIPT]")
    
    if not api_contents_list: # Should not happen if prompt is always there
        print(f"[REDACTED_BY_SCRIPT]")
        return None, ""


    raw_text_response = ""
    cleaned_json_text = "" 

    for attempt in range(max_retries + 1):
        delay = 2 ** attempt
        
        # Create a new client instance for each attempt cycle to ensure it uses the current API key
        # This is crucial if the API key was rotated.
        if not current_api_key_string:
            print(f"[REDACTED_BY_SCRIPT]")
            # This should ideally be caught earlier, e.g., in main_async_runner
            return "RETRY NOW", "RETRY NOW" # Signal a major issue, might lead to key rotation

        client = genai.Client(api_key=current_api_key_string)
        print(f"[REDACTED_BY_SCRIPT]")

        try:
            safety_settings_config = [
                {"category": "[REDACTED_BY_SCRIPT]", "threshold": "BLOCK_NONE"},
                {"category": "[REDACTED_BY_SCRIPT]", "threshold": "BLOCK_NONE"},
                {"category": "[REDACTED_BY_SCRIPT]", "threshold": "BLOCK_NONE"},
                {"category": "[REDACTED_BY_SCRIPT]", "threshold": "BLOCK_NONE"},
            ]
            
            # Base generation config
            generation_config_dict = {
                "temperature": 1.0,
                # "max_output_tokens": 8192, # Example, adjust as needed per model/task
            }

            # Conditional thinking_config for specific models if trying an "upgrade"
            # This part mimics your original logic where model_20_flash tries model_25_flash
            effective_model_name = target_model_name
            thinking_config_for_api = None

            if target_model_name == model_20_flash: # If the target is model_20_flash
                print(f"[REDACTED_BY_SCRIPT]")
                # Check if types.ThinkingConfig exists before trying to use it
                if hasattr(types, 'ThinkingConfig'):
                    try:
                        thinking_config_for_api = types.ThinkingConfig(thinking_budget=512)
                        effective_model_name = model_25_flash # Try the upgraded model
                        print(f"[REDACTED_BY_SCRIPT]")
                    except AttributeError as e_tc_attr:
                        print(f"[REDACTED_BY_SCRIPT]")
                        thinking_config_for_api = None # Ensure it's None
                        effective_model_name = target_model_name # Revert to original target
                    except Exception as e_tc: # Other errors creating ThinkingConfig
                        print(f"[REDACTED_BY_SCRIPT]")
                        thinking_config_for_api = None
                        effective_model_name = target_model_name
                else:
                    print(f"[REDACTED_BY_SCRIPT]")
                    effective_model_name = target_model_name # Stick to original if ThinkingConfig itself is missing

            # Construct the final config object for the API
            # It's crucial that types.GenerateContentConfig is available
            try:
                api_call_config = types.GenerateContentConfig(
                    **generation_config_dict, # Spread common config
                    safety_settings=safety_settings_config,
                    # Only include thinking_config if it's been successfully created
                    **( {"thinking_config": thinking_config_for_api} if thinking_config_for_api else {} )
                )
            except AttributeError as e_gcc_attr:
                print(f"[REDACTED_BY_SCRIPT]'google.genai.types'[REDACTED_BY_SCRIPT]")
                # This is a critical failure. The API call structure itself is not supportable.
                return "RETRY NOW", "RETRY NOW" # Signal failure, might trigger key rotation/wait
            except Exception as e_gcc:
                 print(f"[REDACTED_BY_SCRIPT]")
                 return "RETRY NOW", "RETRY NOW"


            print(f"[REDACTED_BY_SCRIPT]'{effective_model_name}'[REDACTED_BY_SCRIPT]")
            
            # Synchronous function to be run in a thread
            def sync_generate_content():
                # Note: client.models.generate_content does not take a 'model' argument if '[REDACTED_BY_SCRIPT]' is used.
                # It takes 'model' if client.models.generate_content is called directly.
                # Your example: client.models.generate_content(model="gemini-2.0-flash", ...)
                # So, 'model' argument should be passed to generate_content directly.
                return client.models.generate_content(
                    model=effective_model_name, # Pass model name here
                    contents=api_contents_list,
                    config=api_call_config # Use the new name
                )

            response_obj = await asyncio.wait_for(
                asyncio.to_thread(sync_generate_content),
                timeout=timeout_seconds
            )
            
            # If the above call failed due to the upgrade attempt (e.g., model_25_flash not found by client)
            # and we were trying an upgrade, we should fall back to the original target_model_name.
            # This requires catching the specific error from sync_generate_content if it indicates model not found.
            # For simplicity now, we assume if it fails, the general retry handles it.
            # A more robust version would catch errors from `sync_generate_content`, check if an upgrade was attempted,
            # and if so, retry immediately with `target_model_name` without thinking_config.
            # However, the current loop structure will retry the whole `call_gemini_via_client_async` if an exception occurs.

            print(f"[REDACTED_BY_SCRIPT]'{effective_model_name}'.")

            # Response parsing (same as before)
            if hasattr(response_obj, 'text'):
                raw_text_response = response_obj.text
            elif hasattr(response_obj, 'parts') and response_obj.parts:
                raw_text_response = "".join(part.text for part in response_obj.parts if hasattr(part, 'text'))
            else: 
                block_reason_msg, finish_reason_msg = "Unknown", "Unknown"
                try:
                    if response_obj.prompt_feedback: block_reason_msg = str(response_obj.prompt_feedback.block_reason)
                    if response_obj.candidates and response_obj.candidates[0]: finish_reason_msg = str(response_obj.candidates[0].finish_reason)
                except Exception: pass 
                print(f"[REDACTED_BY_SCRIPT]")
                raw_text_response = ""
            
            if not raw_text_response:
                print(f"[REDACTED_BY_SCRIPT]")
                if attempt < max_retries: await asyncio.sleep(delay); continue
                else: print(f"[REDACTED_BY_SCRIPT]"); return None, ""

            json_match = re.search(r'[REDACTED_BY_SCRIPT]', raw_text_response, re.MULTILINE)
            cleaned_json_text = json_match.group(1).strip() if json_match else raw_text_response.strip()

            if not (cleaned_json_text.startswith('{') and cleaned_json_text.endswith('}')) and \
               not (cleaned_json_text.startswith('[') and cleaned_json_text.endswith(']')):
                print(f"[REDACTED_BY_SCRIPT]'t look like valid JSON: '[REDACTED_BY_SCRIPT]'")
                if attempt < max_retries: await asyncio.sleep(delay); continue
                else: print(f"[REDACTED_BY_SCRIPT]"); return None, raw_text_response
            
            parsed_json_output = json.loads(cleaned_json_text)
            print(f"[REDACTED_BY_SCRIPT]")
            return parsed_json_output, cleaned_json_text

        except asyncio.TimeoutError:
            print(f"[REDACTED_BY_SCRIPT]")
        except json.JSONDecodeError as e:
            print(f"[REDACTED_BY_SCRIPT]")
            # ... (JSON error logging)
            if attempt >= max_retries: return None, raw_text_response # Or cleaned_json_text if available
        except AttributeError as e_attr: # Catch AttributeErrors that might come from genai.Client or types
            print(f"[REDACTED_BY_SCRIPT]")
            # This is a critical indicator. If it happens for client.models or types.Part, the pattern is not viable with the runtime library version.
        except Exception as e:
            # This general exception catches errors from asyncio.to_thread or client.models.generate_content itself
            print(f"[REDACTED_BY_SCRIPT]")
            if "API key" in str(e) or "PERMISSION_DENIED" in str(e).upper() or "API_KEY_INVALID" in str(e).upper():
                 print(f"[REDACTED_BY_SCRIPT]")
            # (Heuristic error inspection from previous version can be added here if needed)

        if attempt < max_retries:
            print(f"[REDACTED_BY_SCRIPT]")
            await asyncio.sleep(delay)
        # If all retries exhausted for this client/key, loop ends, and API key rotation logic below is hit.

    # API Key Rotation Logic
    print(f"[REDACTED_BY_SCRIPT]'None'}).")
    if api_key_rotation_counter == 0:
        print("[REDACTED_BY_SCRIPT]")
        # ... (Your API key rotation logic to update current_api_key_string)
        key_before_rotation = current_api_key_string
        primary_key = "[REDACTED_BY_SCRIPT]" # Placeholder
        secondary_key = "[REDACTED_BY_SCRIPT]" # Placeholder
        if current_api_key_string == primary_key: current_api_key_string = secondary_key
        elif current_api_key_string == secondary_key: current_api_key_string = primary_key
        else: current_api_key_string = primary_key # Default to primary if unknown
        
        if current_api_key_string != key_before_rotation:
            print(f"[REDACTED_BY_SCRIPT]")
        else:
            print(f"[REDACTED_BY_SCRIPT]")
        api_key_rotation_counter += 1
    else: 
        print("[REDACTED_BY_SCRIPT]")
        # ... (daily reset wait logic, updating current_api_key_string to primary after wait)
        now = datetime.datetime.now(); midnight = datetime.datetime.combine(now.date() + datetime.timedelta(days=1), datetime.time(0))
        wait_seconds = (midnight - now).total_seconds() + 300
        print(f"[REDACTED_BY_SCRIPT]"); await asyncio.sleep(wait_seconds)
        api_key_rotation_counter = 0; current_api_key_string = "[REDACTED_BY_SCRIPT]" # Reset to primary
        print(f"[REDACTED_BY_SCRIPT]")
        
    return "RETRY NOW", "RETRY NOW" # Signal to the caller to retry the whole operation with the new key config

# --- Main Asynchronous Runner ---
async def call_gemini_via_client_async(prompt: str, images_input, step_name: str, target_model_name: str, timeout_seconds: int = 180):
    global current_api_key_string # Use and update the global API key tracker
    global api_key_rotation_counter

    max_retries = 5 # Retries per API key configuration
    print(f"[REDACTED_BY_SCRIPT]'{target_model_name}'...")

    model_motivation_text = """...""" # Your model_motivation string
    prompt_with_motivation = model_motivation_text + prompt
    
    # Prepare contents: text prompt and image Parts
    # The 'images_input' can be a single PIL Image, a list of PIL Images, or a dict of "Image X": PIL.Image
    # The API expects a list of [text, Part, Part,...] or [Part, Part, text] etc.
    # We need to convert PIL Images to types.Part.from_pil or types.Part.from_bytes
    
    api_contents_list = []
    
    # Add text part first, then image parts. Or image then text as per best practices.
    # "[REDACTED_BY_SCRIPT]"
    # For multiple images, the order might be less strict, but often text sets context.
    # Let's try text first, then images, which is common for multi-turn-like prompts.
    # If only one image, we can adjust.
    
    pil_images_to_process = []
    if images_input:
        if isinstance(images_input, dict): # e.g., {"Image 1": pil_img1, ...}
            pil_images_to_process = list(images_input.values())
        elif isinstance(images_input, list) and all(isinstance(i, Image.Image) for i in images_input):
            pil_images_to_process = images_input
        elif isinstance(images_input, Image.Image):
            pil_images_to_process = [images_input]
        else:
            print(f"[REDACTED_BY_SCRIPT]")

    # Add image parts first if there's only one image, then text. Otherwise, text then images.
    if len(pil_images_to_process) == 1:
        try:
            # Assuming JPEG for PIL images. For other types, MIME might need to be dynamic.
            # Using types.Part.from_pil if available and preferred
            if hasattr(types.Part, 'from_pil'):
                api_contents_list.append(types.Part.from_pil(pil_images_to_process[0]))
            else: # Fallback to converting to bytes then Part.from_bytes if from_pil not found
                import io
                img_byte_arr = io.BytesIO()
                pil_images_to_process[0].save(img_byte_arr, format='JPEG') # Or PNG, depending on original
                img_bytes = img_byte_arr.getvalue()
                api_contents_list.append(types.Part.from_bytes(data=img_bytes, mime_type='image/jpeg'))
            api_contents_list.append(prompt_with_motivation)
        except Exception as e_img_prep:
            print(f"[REDACTED_BY_SCRIPT]")
            api_contents_list = [prompt_with_motivation] # Fallback to text only
    else: # Multiple images or no images
        api_contents_list.append(prompt_with_motivation) # Text first
        for pil_image in pil_images_to_process:
            try:
                if hasattr(types.Part, 'from_pil'):
                    api_contents_list.append(types.Part.from_pil(pil_image))
                else:
                    import io
                    img_byte_arr = io.BytesIO()
                    pil_image.save(img_byte_arr, format='JPEG')
                    img_bytes = img_byte_arr.getvalue()
                    api_contents_list.append(types.Part.from_bytes(data=img_bytes, mime_type='image/jpeg'))
            except Exception as e_img_prep_multi:
                print(f"[REDACTED_BY_SCRIPT]")
    
    if not api_contents_list: # Should not happen if prompt is always there
        print(f"[REDACTED_BY_SCRIPT]")
        return None, ""


    raw_text_response = ""
    cleaned_json_text = "" 

    for attempt in range(max_retries + 1):
        delay = 2 ** attempt
        
        # Create a new client instance for each attempt cycle to ensure it uses the current API key
        # This is crucial if the API key was rotated.
        if not current_api_key_string:
            print(f"[REDACTED_BY_SCRIPT]")
            # This should ideally be caught earlier, e.g., in main_async_runner
            return "RETRY NOW", "RETRY NOW" # Signal a major issue, might lead to key rotation

        client = genai.Client(api_key=current_api_key_string)
        print(f"[REDACTED_BY_SCRIPT]")

        try:
            safety_settings_config = [
                {"category": "[REDACTED_BY_SCRIPT]", "threshold": "BLOCK_NONE"},
                {"category": "[REDACTED_BY_SCRIPT]", "threshold": "BLOCK_NONE"},
                {"category": "[REDACTED_BY_SCRIPT]", "threshold": "BLOCK_NONE"},
                {"category": "[REDACTED_BY_SCRIPT]", "threshold": "BLOCK_NONE"},
            ]
            
            # Base generation config
            generation_config_dict = {
                "temperature": 1.0,
                # "max_output_tokens": 8192, # Example, adjust as needed per model/task
            }

            # Conditional thinking_config for specific models if trying an "upgrade"
            # This part mimics your original logic where model_20_flash tries model_25_flash
            effective_model_name = target_model_name
            thinking_config_for_api = None

            if target_model_name == model_20_flash: # If the target is model_20_flash
                print(f"[REDACTED_BY_SCRIPT]")
                # Check if types.ThinkingConfig exists before trying to use it
                if hasattr(types, 'ThinkingConfig'):
                    try:
                        thinking_config_for_api = types.ThinkingConfig(thinking_budget=512)
                        effective_model_name = model_25_flash # Try the upgraded model
                        print(f"[REDACTED_BY_SCRIPT]")
                    except AttributeError as e_tc_attr:
                        print(f"[REDACTED_BY_SCRIPT]")
                        thinking_config_for_api = None # Ensure it's None
                        effective_model_name = target_model_name # Revert to original target
                    except Exception as e_tc: # Other errors creating ThinkingConfig
                        print(f"[REDACTED_BY_SCRIPT]")
                        thinking_config_for_api = None
                        effective_model_name = target_model_name
                else:
                    print(f"[REDACTED_BY_SCRIPT]")
                    effective_model_name = target_model_name # Stick to original if ThinkingConfig itself is missing

            # Construct the final config object for the API
            # It's crucial that types.GenerateContentConfig is available
            try:
                api_call_config = types.GenerateContentConfig(
                    **generation_config_dict, # Spread common config
                    safety_settings=safety_settings_config,
                    # Only include thinking_config if it's been successfully created
                    **( {"thinking_config": thinking_config_for_api} if thinking_config_for_api else {} )
                )
            except AttributeError as e_gcc_attr:
                print(f"[REDACTED_BY_SCRIPT]'google.genai.types'[REDACTED_BY_SCRIPT]")
                # This is a critical failure. The API call structure itself is not supportable.
                return "RETRY NOW", "RETRY NOW" # Signal failure, might trigger key rotation/wait
            except Exception as e_gcc:
                 print(f"[REDACTED_BY_SCRIPT]")
                 return "RETRY NOW", "RETRY NOW"


            print(f"[REDACTED_BY_SCRIPT]'{effective_model_name}'[REDACTED_BY_SCRIPT]")
            
            # Synchronous function to be run in a thread
            def sync_generate_content():
                # Note: client.models.generate_content does not take a 'model' argument if '[REDACTED_BY_SCRIPT]' is used.
                # It takes 'model' if client.models.generate_content is called directly.
                # Your example: client.models.generate_content(model="gemini-2.0-flash", ...)
                # So, 'model' argument should be passed to generate_content directly.
                return client.models.generate_content(
                    model=effective_model_name, # Pass model name here
                    contents=api_contents_list,
                    config=api_call_config # Use the new name
                )

            response_obj = await asyncio.wait_for(
                asyncio.to_thread(sync_generate_content),
                timeout=timeout_seconds
            )
            
            # If the above call failed due to the upgrade attempt (e.g., model_25_flash not found by client)
            # and we were trying an upgrade, we should fall back to the original target_model_name.
            # This requires catching the specific error from sync_generate_content if it indicates model not found.
            # For simplicity now, we assume if it fails, the general retry handles it.
            # A more robust version would catch errors from `sync_generate_content`, check if an upgrade was attempted,
            # and if so, retry immediately with `target_model_name` without thinking_config.
            # However, the current loop structure will retry the whole `call_gemini_via_client_async` if an exception occurs.

            print(f"[REDACTED_BY_SCRIPT]'{effective_model_name}'.")

            # Response parsing (same as before)
            if hasattr(response_obj, 'text'):
                raw_text_response = response_obj.text
            elif hasattr(response_obj, 'parts') and response_obj.parts:
                raw_text_response = "".join(part.text for part in response_obj.parts if hasattr(part, 'text'))
            else: 
                block_reason_msg, finish_reason_msg = "Unknown", "Unknown"
                try:
                    if response_obj.prompt_feedback: block_reason_msg = str(response_obj.prompt_feedback.block_reason)
                    if response_obj.candidates and response_obj.candidates[0]: finish_reason_msg = str(response_obj.candidates[0].finish_reason)
                except Exception: pass 
                print(f"[REDACTED_BY_SCRIPT]")
                raw_text_response = ""
            
            if not raw_text_response:
                print(f"[REDACTED_BY_SCRIPT]")
                if attempt < max_retries: await asyncio.sleep(delay); continue
                else: print(f"[REDACTED_BY_SCRIPT]"); return None, ""

            json_match = re.search(r'[REDACTED_BY_SCRIPT]', raw_text_response, re.MULTILINE)
            cleaned_json_text = json_match.group(1).strip() if json_match else raw_text_response.strip()

            if not (cleaned_json_text.startswith('{') and cleaned_json_text.endswith('}')) and \
               not (cleaned_json_text.startswith('[') and cleaned_json_text.endswith(']')):
                print(f"[REDACTED_BY_SCRIPT]'t look like valid JSON: '[REDACTED_BY_SCRIPT]'")
                if attempt < max_retries: await asyncio.sleep(delay); continue
                else: print(f"[REDACTED_BY_SCRIPT]"); return None, raw_text_response
            
            parsed_json_output = json.loads(cleaned_json_text)
            print(f"[REDACTED_BY_SCRIPT]")
            return parsed_json_output, cleaned_json_text

        except asyncio.TimeoutError:
            print(f"[REDACTED_BY_SCRIPT]")
        except json.JSONDecodeError as e:
            print(f"[REDACTED_BY_SCRIPT]")
            # ... (JSON error logging)
            if attempt >= max_retries: return None, raw_text_response # Or cleaned_json_text if available
        except AttributeError as e_attr: # Catch AttributeErrors that might come from genai.Client or types
            print(f"[REDACTED_BY_SCRIPT]")
            # This is a critical indicator. If it happens for client.models or types.Part, the pattern is not viable with the runtime library version.
        except Exception as e:
            # This general exception catches errors from asyncio.to_thread or client.models.generate_content itself
            print(f"[REDACTED_BY_SCRIPT]")
            if "API key" in str(e) or "PERMISSION_DENIED" in str(e).upper() or "API_KEY_INVALID" in str(e).upper():
                 print(f"[REDACTED_BY_SCRIPT]")
            # (Heuristic error inspection from previous version can be added here if needed)

        if attempt < max_retries:
            print(f"[REDACTED_BY_SCRIPT]")
            await asyncio.sleep(delay)
        # If all retries exhausted for this client/key, loop ends, and API key rotation logic below is hit.

    # API Key Rotation Logic
    print(f"[REDACTED_BY_SCRIPT]'None'}).")
    if api_key_rotation_counter == 0:
        print("[REDACTED_BY_SCRIPT]")
        # ... (Your API key rotation logic to update current_api_key_string)
        key_before_rotation = current_api_key_string
        primary_key = "[REDACTED_BY_SCRIPT]" # Placeholder
        secondary_key = "[REDACTED_BY_SCRIPT]" # Placeholder
        if current_api_key_string == primary_key: current_api_key_string = secondary_key
        elif current_api_key_string == secondary_key: current_api_key_string = primary_key
        else: current_api_key_string = primary_key # Default to primary if unknown
        
        if current_api_key_string != key_before_rotation:
            print(f"[REDACTED_BY_SCRIPT]")
        else:
            print(f"[REDACTED_BY_SCRIPT]")
        api_key_rotation_counter += 1
    else: 
        print("[REDACTED_BY_SCRIPT]")
        # ... (daily reset wait logic, updating current_api_key_string to primary after wait)
        now = datetime.datetime.now(); midnight = datetime.datetime.combine(now.date() + datetime.timedelta(days=1), datetime.time(0))
        wait_seconds = (midnight - now).total_seconds() + 300
        print(f"[REDACTED_BY_SCRIPT]"); await asyncio.sleep(wait_seconds)
        api_key_rotation_counter = 0; current_api_key_string = "[REDACTED_BY_SCRIPT]" # Reset to primary
        print(f"[REDACTED_BY_SCRIPT]")
        
    return "RETRY NOW", "RETRY NOW" # Signal to the caller to retry the whole operation with the new key config


# --- Main Asynchronous Runner ---
async def main_async_runner():
    global current_api_key_string 
    global csv_header

    try:
        # Initialize current_api_key_string (global)
        # Prioritize environment variable, then fallback to hardcoded
        env_key = os.environ.get("GOOGLE_API_KEY")
        if env_key:
            current_api_key_string = env_key
            print("[REDACTED_BY_SCRIPT]")
        else:
            current_api_key_string = "[REDACTED_BY_SCRIPT]" # Your primary placeholder
            print(f"[REDACTED_BY_SCRIPT]")
        
        if not current_api_key_string:
            raise ValueError("[REDACTED_BY_SCRIPT]")
        
        # No need to set os.environ["GOOGLE_API_KEY"] if Client is explicitly given the key.
        print(f"[REDACTED_BY_SCRIPT]")
        # Model usage logging
        print(f"[REDACTED_BY_SCRIPT]")
        print(f"[REDACTED_BY_SCRIPT]")
        print(f"[REDACTED_BY_SCRIPT]")
        print(f"[REDACTED_BY_SCRIPT]")

    except Exception as e:
        print(f"[REDACTED_BY_SCRIPT]")
        return

    # ... (Rest of main_async_runner: directory setup, CSV loading, property loop) ...
    # This part should be the same as the previous correct response, ensuring it calls
    # the newly named '[REDACTED_BY_SCRIPT]' (or similar) which in turn
    # calls 'call_gemini_via_client_async'.

    main_image_dir = r"[REDACTED_BY_SCRIPT]"
    main_floorplan_dir = r"[REDACTED_BY_SCRIPT]"
    main_output_dir = r"[REDACTED_BY_SCRIPT]"
    os.makedirs(main_output_dir, exist_ok=True)
    master_csv_path = os.path.join(main_output_dir, "[REDACTED_BY_SCRIPT]")
    EXPECTED_BASE_JSON_FILENAMES = [ "image_paths_map", "[REDACTED_BY_SCRIPT]", "[REDACTED_BY_SCRIPT]", "output_step3_features", "[REDACTED_BY_SCRIPT]", "output_step5_merged", "[REDACTED_BY_SCRIPT]", "[REDACTED_BY_SCRIPT]", "[REDACTED_BY_SCRIPT]" ]
    
    completed_property_ids = set()
    if os.path.isfile(master_csv_path):
        try:
            with open(master_csv_path, 'r', newline='', encoding='utf-8') as csvfile_read:
                if csvfile_read.read(1): 
                    csvfile_read.seek(0); reader = csv.DictReader(csvfile_read)
                    if 'property_id' in reader.fieldnames:
                        for row in reader:
                            if row.get('property_id'): completed_property_ids.add(row['property_id'])
        except Exception as e_csv: print(f"[REDACTED_BY_SCRIPT]")

    address_folders = [f for f in os.listdir(main_image_dir) if os.path.isdir(os.path.join(main_image_dir, f))]
    print(f"[REDACTED_BY_SCRIPT]")

    for property_address in address_folders:
        print(f"\n{'='[REDACTED_BY_SCRIPT]'='*15}")

        base_image_address_dir = os.path.join(main_image_dir, property_address)
        property_output_dir = os.path.join(main_output_dir, property_address) # Output dir for this address

        # --- Find Latest Year ---
        latest_year = None
        # ... (your logic to find latest_year in base_image_address_dir) ...
        if os.path.isdir(base_image_address_dir):
            try: 
                valid_year_folders = [int(item) for item in os.listdir(base_image_address_dir) if os.path.isdir(os.path.join(base_image_address_dir, item)) and item.isdigit() and len(item) == 4]
                if valid_year_folders: latest_year = max(valid_year_folders)
            except Exception as e_year: print(f"[REDACTED_BY_SCRIPT]'{property_address}': {e_year}")
        
        if latest_year is None:
            print(f"[REDACTED_BY_SCRIPT]'year' sub-folder for '{property_address}'. Skipping address.")
            continue
        print(f"[REDACTED_BY_SCRIPT]")

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
        
        # If we haven't 'continue'd, then we process:
        print(f"[REDACTED_BY_SCRIPT]'{property_id_with_year}'...")
        await process_property_pipeline_with_client( # Your async pipeline function
            property_address,
            # Pass other necessary args like base_image_address_dir, base_floorplan_address_dir
            os.path.join(main_image_dir, property_address), # base_image_address_dir
            os.path.join(main_floorplan_dir, property_address), # base_floorplan_address_dir
            main_output_dir,
            latest_year
        )


# --- Helper to call a pipeline step with retry logic for "RETRY NOW" signal ---
# (This function and process_property_pipeline should be copied from the previous response
# as they correctly handle the async calls and error propagation.)
# Make sure property_id_with_year_context is defined globally or passed appropriately if needed for logging.
property_id_with_year_context = "[REDACTED_BY_SCRIPT]"

async def call_step_with_client_retry(step_prompt: str, images, step_name: str, model_name: str, timeout_val: int, context_map=None):
    global property_id_with_year_context_global 
    
    filled_prompt = step_prompt
    if context_map:
        for placeholder, raw_value in context_map.items():
            context_str = json.dumps(raw_value) if not isinstance(raw_value, str) and raw_value is not None else (raw_value or ("{}" if placeholder.endswith("_json}") else ""))
            filled_prompt = filled_prompt.replace(placeholder, context_str)

    # Call the new API function
    parsed_json, raw_text = await call_gemini_via_client_async(
        filled_prompt, images, step_name, model_name, timeout_seconds=timeout_val
    )
    if parsed_json == "RETRY NOW": # API key rotation occurred, and retry is signaled
        print(f"[REDACTED_BY_SCRIPT]")
        # The API key string should have been updated by call_gemini_via_client_async
        parsed_json, raw_text = await call_gemini_via_client_async(
            filled_prompt, images, f"{step_name} (Retry)", model_name, timeout_seconds=timeout_val
        )
    return parsed_json, raw_text

async def process_property_pipeline_with_client(property_address, base_image_address_dir, base_floorplan_address_dir, main_output_dir, latest_year):
    global property_id_with_year_context # Corrected to use the renamed global
    global csv_header 
    global UNKNOWN_TOKEN # Make sure UNKNOWN_TOKEN is accessible

    property_id_with_year_context = f"[REDACTED_BY_SCRIPT]" # Use the renamed global
    print(f"\n{'='[REDACTED_BY_SCRIPT]'='*10}")
    
    year_suffix = f"_y{latest_year}"
    property_output_dir = os.path.join(main_output_dir, property_address)
    os.makedirs(property_output_dir, exist_ok=True)
    print(f"[REDACTED_BY_SCRIPT]")

    # --- (Your existing, complete image loading and deduplication logic here) ---
    # This should populate:
    # floorplan_image (PIL.Image or None)
    # indexed_room_images (dict of {"Image X": PIL.Image})
    # And save the image_paths_map{year_suffix}.json
    # --- Placeholder for your image loading logic ---
    current_image_dir = os.path.join(base_image_address_dir, str(latest_year))
    current_floorplan_dir = os.path.join(base_floorplan_address_dir, str(latest_year))
    floorplan_image = None
    floorplan_image_paths = get_floorplan_images(current_floorplan_dir)
    if floorplan_image_paths:
        selected_fp_path = None; max_a = -1
        for fp in floorplan_image_paths:
            try:
                with Image.open(fp) as img_fp: area = img_fp.width * img_fp.height
                if area > max_a: max_a, selected_fp_path = area, fp
            except Exception: pass
        if selected_fp_path: 
            floorplan_image = load_image(selected_fp_path)
            if floorplan_image: print(f"[REDACTED_BY_SCRIPT]")
            else: print(f"[REDACTED_BY_SCRIPT]")
    if not floorplan_image:
        print(f"[REDACTED_BY_SCRIPT]")

    initial_room_paths = get_room_images(current_image_dir, floorplan_image_paths if floorplan_image else [])
    if not initial_room_paths: 
        print(f"[REDACTED_BY_SCRIPT]")
        # Decide if you want to return or proceed with potentially no images for Steps 2+
        # For now, we'll let it proceed, but Steps 2+ might fail or produce empty results.
    
    unique_room_paths = []
    if initial_room_paths: # Only deduplicate if there are initial paths
        seen_hashes_tmp = set()
        for img_pth_tmp in initial_room_paths:
            try:
                with Image.open(img_pth_tmp) as img_hash_tmp: h = imagehash.phash(img_hash_tmp)
                if h not in seen_hashes_tmp: seen_hashes_tmp.add(h); unique_room_paths.append(img_pth_tmp)
            except Exception as e_hash: print(f"[REDACTED_BY_SCRIPT]") # Be more verbose
        print(f"[REDACTED_BY_SCRIPT]")
    
    indexed_room_images = {}
    if unique_room_paths:
        pil_room_images = [img for img in [load_image(p) for p in unique_room_paths] if img is not None]
        if not pil_room_images: print(f"[REDACTED_BY_SCRIPT]")
        indexed_room_images = {f"Image {i+1}": img for i, img in enumerate(pil_room_images)}
    
    img_map_fn = os.path.join(property_output_dir, f"[REDACTED_BY_SCRIPT]")
    try:
        # Save paths of unique images that were attempted to be loaded, even if some failed loading into PIL
        # This ensures the map reflects what was considered after deduplication.
        paths_for_map = {f"Image {i+1}": p for i, p in enumerate(unique_room_paths)}
        with open(img_map_fn, "w", encoding="utf-8") as f_map:
            json.dump(paths_for_map, f_map, indent=2)
        print(f"[REDACTED_BY_SCRIPT]")
    except Exception as e_map: print(f"[REDACTED_BY_SCRIPT]")
    # --- End of image loading logic placeholder ---

    property_token_summary = {"room_labels": set(), "features": set(), "sp_themes": set(), "flaw_themes": set()}
    untagged_text_log = {"[REDACTED_BY_SCRIPT]": set(), "raw_features": set(), "[REDACTED_BY_SCRIPT]": set(), "raw_flaws_text": set()}
    output1_json, output1_text = None, None
    output2_json, output2_text = None, None
    # ... (initialize all other outputN_json, outputN_text variables to None) ...
    output3_json, output3_text = None, None; output4_json, output4_text = None, None
    output5a_json, output5a_text = None, None; output5b_json, output5b_text = None, None
    merged_step5_output = None; output6_json, output6_text = None, None
    pipeline_successful = True
    default_timeout, long_timeout = 120, 300 # Or your preferred values

    try:
        # --- Step 1: Floorplan Analysis (Conditional) ---
        if floorplan_image:
            output1_json, output1_text = await call_step_with_client_retry(
                prompt1_floorplan, floorplan_image, "[REDACTED_BY_SCRIPT]", model_15_flash, default_timeout
            )
            if output1_json:
                print(f"[REDACTED_BY_SCRIPT]'Not a dict'}")
                # --- (Your existing bedroom processing & tokenization for output1) ---
                if 'rooms_with_dimensions' in output1_json and isinstance(output1_json['rooms_with_dimensions'], list):
                    # ... (Your full bedroom relabeling logic) ...
                    print(f"[REDACTED_BY_SCRIPT]")
                for room_data in output1_json.get('rooms_with_dimensions', []):
                    if isinstance(room_data, dict) and 'label' in room_data: 
                        token = get_room_token(room_data['label'])
                        (untagged_text_log["[REDACTED_BY_SCRIPT]"].add(room_data['label']) if token == UNKNOWN_TOKEN else property_token_summary["room_labels"].add(token)) if token else untagged_text_log["[REDACTED_BY_SCRIPT]"].add(f"[REDACTED_BY_SCRIPT]'label']}")
                for label_str in output1_json.get('[REDACTED_BY_SCRIPT]', []):
                    if isinstance(label_str, str):
                        token = get_room_token(label_str)
                        (untagged_text_log["[REDACTED_BY_SCRIPT]"].add(label_str) if token == UNKNOWN_TOKEN else property_token_summary["room_labels"].add(token)) if token else untagged_text_log["[REDACTED_BY_SCRIPT]"].add(f"[REDACTED_BY_SCRIPT]")

            else:
                print(f"[REDACTED_BY_SCRIPT]")
                output1_json, output1_text = None, None # Ensure it's None if parsing failed
        else:
            print(f"[REDACTED_BY_SCRIPT]")
            output1_json, output1_text = None, None

        # --- Step 2: Room Assignments ---
        # This step requires images. If indexed_room_images is empty, it might not run well.
        if not indexed_room_images:
            print(f"[REDACTED_BY_SCRIPT]")
            # If Step 2 doesn't run, we can't create a fallback Step 1.
            # Depending on requirements, you might want to create a truly empty Step 1 JSON here.
            # For now, if no images, output1_json might remain None if no floorplan either.
            raise StopIteration("[REDACTED_BY_SCRIPT]")


        floorplan_context_str = json.dumps(output1_json if output1_json else {}) # Pass current state of output1_json
        output2_json, output2_text = await call_step_with_client_retry(
            prompt2_assignments, indexed_room_images, "[REDACTED_BY_SCRIPT]", model_20_flash, default_timeout,
            context_map={"{" + "floorplan_data_json" + "}": floorplan_context_str}
        )
        if not output2_json or not isinstance(output2_json.get('room_assignments'), list):
            # If Step 2 fails, we definitely can't create a fallback Step 1 from it.
            raise StopIteration("[REDACTED_BY_SCRIPT]")
        
        # --- NEW: Create Fallback Step 1 JSON if original Step 1 was skipped/failed ---
        if output1_json is None: # Check if original Step 1 didn't populate output1_json
            if output2_json and 'room_assignments' in output2_json and output2_json['room_assignments']: # Ensure Step 2 has data
                print(f"[REDACTED_BY_SCRIPT]'s room assignments for {property_id_with_year_context}.")
                
                new_step1_rooms_with_dimensions = []
                seen_labels_for_step1_fallback = set() 

                for assignment in output2_json.get('room_assignments', []):
                    if isinstance(assignment, dict) and 'label' in assignment:
                        label = assignment['label']
                        if label and label not in seen_labels_for_step1_fallback: # Ensure label is not empty
                            new_step1_rooms_with_dimensions.append({"label": label, "dimensions": "null"})
                            seen_labels_for_step1_fallback.add(label)
                
                if new_step1_rooms_with_dimensions: # Only create if we got labels
                    output1_json = {
                        "rooms_with_dimensions": new_step1_rooms_with_dimensions,
                        "[REDACTED_BY_SCRIPT]": [] # Keep this empty for fallback
                    }
                    # output1_text is not strictly needed here as Step 2 has already run.
                    # If it were needed by a *later* step, we'd do: output1_text = json.dumps(output1_json)
                    print(f"[REDACTED_BY_SCRIPT]")
                else:
                    print(f"[REDACTED_BY_SCRIPT]")
                    # output1_json remains None. This means it still won't be saved.
                    # If a completely empty Step 1 file is desired in this edge case:
                    # output1_json = {"rooms_with_dimensions": [], "[REDACTED_BY_SCRIPT]": []}
            else:
                print(f"[REDACTED_BY_SCRIPT]")
                # output1_json remains None

        # --- (Tokenization for Step 2 - from your full script) ---
        if output2_json and 'room_assignments' in output2_json : # Re-check because it might have failed above
            for assignment in output2_json.get('room_assignments', []):
                if isinstance(assignment, dict): 
                    label = assignment.get('label')
                    source = assignment.get('source')
                    if label: # Ensure label exists
                        token = get_room_token(label)
                        (untagged_text_log["[REDACTED_BY_SCRIPT]"].add(label) if token == UNKNOWN_TOKEN else property_token_summary["room_labels"].add(token)) if token else untagged_text_log["[REDACTED_BY_SCRIPT]"].add(f"[REDACTED_BY_SCRIPT]")
                    if source == 'Floorplan': property_token_summary["features"].add('[REDACTED_BY_SCRIPT]')
                    elif source == 'Generated': property_token_summary["features"].add('[REDACTED_BY_SCRIPT]')


        # --- Step 3: Feature Extraction ---
        # (Your existing call to Step 3 using call_step_with_client_retry)
        # (Your existing tokenization for output3)
        if not output2_text: # If Step 2 failed badly, output2_text might be None
            raise StopIteration("[REDACTED_BY_SCRIPT]")
        output3_json, output3_text = await call_step_with_client_retry(prompt3_features, indexed_room_images, "[REDACTED_BY_SCRIPT]", model_20_flash_lite, default_timeout, context_map={"{" + "[REDACTED_BY_SCRIPT]" + "}": output2_text})
        if not output3_json or not isinstance(output3_json, dict): raise StopIteration("[REDACTED_BY_SCRIPT]")
        if output3_json: # Tokenization for Step 3
            all_feature_tokens = set()
            for room_label, features_list in output3_json.items():
                if isinstance(features_list, list):
                    for feature_text in features_list: 
                        if isinstance(feature_text, str): untagged_text_log["raw_features"].add(feature_text)
                    room_feature_tokens = get_feature_tokens(features_list); all_feature_tokens.update(room_feature_tokens)
            property_token_summary["features"].update(all_feature_tokens)


        # --- (Call Step 4, 5a, 5b, 6 as in your full working script, using call_step_with_client_retry) ---
        # --- (Include your Step 5 merge logic) ---
        # --- (Include tokenization for Step 4) ---
        if not output3_text: raise StopIteration("[REDACTED_BY_SCRIPT]")
        output4_json, output4_text = await call_step_with_client_retry(prompt4_flaws_sp_categorized, indexed_room_images, "Step 4: Flaws/SPs", model_20_flash_lite, default_timeout, context_map={"{" + "[REDACTED_BY_SCRIPT]" + "}": output2_text, "{" + "[REDACTED_BY_SCRIPT]" + "}": output3_text})
        if not output4_json or not isinstance(output4_json, dict): raise StopIteration("[REDACTED_BY_SCRIPT]")
        if output4_json: # Tokenization for Step 4
            # ... (your full tokenization for SP and Flaw tags) ...
            pass

        if not output4_text: raise StopIteration("[REDACTED_BY_SCRIPT]")
        output5a_json, output5a_text = await call_step_with_client_retry(prompt5a_ratings_p1_10, indexed_room_images, "[REDACTED_BY_SCRIPT]", model_20_flash, long_timeout, context_map={"{" + "[REDACTED_BY_SCRIPT]" + "}": output2_text, "{" + "[REDACTED_BY_SCRIPT]" + "}": output3_text, "{" + "[REDACTED_BY_SCRIPT]" + "}": output4_text, "{" + "[REDACTED_BY_SCRIPT]" + "}": persona_details_p1_10})
        if not output5a_json: raise StopIteration("Step 5a Failed")
        
        output5b_json, output5b_text = await call_step_with_client_retry(prompt5b_ratings_p11_20, indexed_room_images, "[REDACTED_BY_SCRIPT]", model_20_flash, long_timeout, context_map={"{" + "[REDACTED_BY_SCRIPT]" + "}": output2_text, "{" + "[REDACTED_BY_SCRIPT]" + "}": output3_text, "{" + "[REDACTED_BY_SCRIPT]" + "}": output4_text, "{" + "[REDACTED_BY_SCRIPT]" + "}": persona_details_p11_20})
        if not output5b_json: raise StopIteration("Step 5b Failed")

        # Step 5 Merge
        print("[REDACTED_BY_SCRIPT]")
        # ... (Your full, correct Step 5 merge logic copied from your working script) ...
        try: 
            # --- Start of your full Step 5 merge logic ---
            if not isinstance(output5a_json, dict) or not isinstance(output5b_json, dict): raise ValueError("[REDACTED_BY_SCRIPT]")
            labels_5a = output5a_json.get("[REDACTED_BY_SCRIPT]"); ratings_5a = output5a_json.get("room_ratings_p1_10"); overall_5a = output5a_json.get("[REDACTED_BY_SCRIPT]", {})
            labels_5b = output5b_json.get("[REDACTED_BY_SCRIPT]"); ratings_5b = output5b_json.get("room_ratings_p11_20"); overall_5b = output5b_json.get("[REDACTED_BY_SCRIPT]", {})
            if not (isinstance(labels_5a, list) and isinstance(ratings_5a, list) and isinstance(labels_5b, list) and isinstance(ratings_5b, list)): raise ValueError("[REDACTED_BY_SCRIPT]")
            if labels_5a != labels_5b:
                if set(labels_5a) == set(labels_5b) and len(labels_5a) == len(labels_5b):
                    label_to_index_5b = {label: i for i, label in enumerate(labels_5b)}
                    if not all(lbl in label_to_index_5b for lbl in labels_5a): raise ValueError("[REDACTED_BY_SCRIPT]")
                    new_ratings_5b = [ratings_5b[label_to_index_5b[target_label]] if target_label in label_to_index_5b and label_to_index_5b[target_label] < len(ratings_5b) else [None]*10 for target_label in labels_5a]
                    ratings_5b = new_ratings_5b
                else: raise ValueError(f"[REDACTED_BY_SCRIPT]")
            if len(labels_5a) != len(ratings_5a) or len(labels_5a) != len(ratings_5b): raise ValueError("[REDACTED_BY_SCRIPT]")
            merged_step5_output = {"[REDACTED_BY_SCRIPT]": labels_5a, "[REDACTED_BY_SCRIPT]": {**overall_5a, **overall_5b}, "room_ratings_final": []}
            for i, label in enumerate(labels_5a):
                list_5a_room = ratings_5a[i] if i < len(ratings_5a) else [None]*11
                list_5b_room = ratings_5b[i] if i < len(ratings_5b) else [None]*10
                valid_5a = isinstance(list_5a_room, list) and len(list_5a_room) == 11
                valid_5b = isinstance(list_5b_room, list) and len(list_5b_room) == 10
                if not valid_5a: list_5a_room = [None] * 11
                if not valid_5b: list_5b_room = [None] * 10
                merged_room_ratings_no_avg = list_5a_room + list_5b_room
                persona_ratings_only = merged_room_ratings_no_avg[1:] # Exclude general rating for avg
                valid_persona_ratings = [r for r in persona_ratings_only if isinstance(r, (int, float))]
                average_rating = round(statistics.mean(valid_persona_ratings), 1) if valid_persona_ratings else None
                merged_step5_output["room_ratings_final"].append(merged_room_ratings_no_avg + [average_rating])
            # --- End of your full Step 5 merge logic ---
            print(f"[REDACTED_BY_SCRIPT]")
        except Exception as e_merge:
            print(f"[REDACTED_BY_SCRIPT]"); 
            merged_step5_output = None; # Ensure it's None if merge fails
            # Decide if this should raise StopIteration or just warn and continue with merged_step5_output as None
            # For now, let's assume it's critical if merge fails
            raise StopIteration(f"[REDACTED_BY_SCRIPT]")


        output6_json, output6_text = await call_step_with_client_retry(prompt6_renovation, indexed_room_images, "[REDACTED_BY_SCRIPT]", model_15_flash, default_timeout, context_map={"{" + "[REDACTED_BY_SCRIPT]" + "}": output2_text, "{" + "[REDACTED_BY_SCRIPT]" + "}": output3_text})
        if not output6_json: print(f"[REDACTED_BY_SCRIPT]")


    except StopIteration as si:
        pipeline_successful = False
        print(f"[REDACTED_BY_SCRIPT]")
    except Exception as e_pipe:
        pipeline_successful = False
        print(f"[REDACTED_BY_SCRIPT]")
    finally:
        # --- (Your existing, complete saving logic for all raw_outputs_to_save) ---
        # This will now save output1_json whether it's from actual analysis or the fallback
        print(f"[REDACTED_BY_SCRIPT]")
        # year_suffix is already defined as _y{latest_year} at the start of this function
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
        for filename, data_to_save_val in raw_outputs_to_save.items():
            filepath = os.path.join(property_output_dir, filename)
            if data_to_save_val is not None: 
                try:
                    with open(filepath, "w", encoding="utf-8") as f_out:
                        json.dump(data_to_save_val, f_out, indent=2, ensure_ascii=False)
                except Exception as e_save_raw: print(f"[REDACTED_BY_SCRIPT]")
            #else: # Optional: Log if a specific step's output is None and not being saved
            #    print(f"[REDACTED_BY_SCRIPT]")
        
        # --- (Your existing, complete feature generation and CSV append logic) ---
        if pipeline_successful:
            try:
                # Ensure year_suffix_check is correctly defined for loading files by feature generator
                year_suffix_for_feature_gen = year_suffix # Since year_suffix is already _yYYYY
                
                path_s4_chk = os.path.join(property_output_dir, f'[REDACTED_BY_SCRIPT]')
                path_s5m_chk = os.path.join(property_output_dir, f'[REDACTED_BY_SCRIPT]')
                # The feature generator might also need output_step1_floorplan.json
                path_s1_chk = os.path.join(property_output_dir, f'[REDACTED_BY_SCRIPT]')

                # Adjust required_files_exist based on what gemini_property_feature_generator truly needs.
                # If step1 is now always created (even if fallback), it can be required.
                required_files_exist = all(os.path.exists(f) for f in [path_s1_chk, path_s4_chk, path_s5m_chk])
                missing_for_feat_gen = [os.path.basename(f) for f in [path_s1_chk, path_s4_chk, path_s5m_chk] if not os.path.exists(f)]


                if required_files_exist:
                    print(f"[REDACTED_BY_SCRIPT]")
                    features_gen = gemini_property_feature_generator.process_property(
                        property_id=property_id_with_year_context,
                        input_dir=property_output_dir,
                        year_suffix=year_suffix_for_feature_gen # This should be e.g., "_y2024"
                    )
                    if features_gen and isinstance(features_gen, dict):
                        # ... (your CSV header and writing logic) ...
                        if csv_header is None: csv_header = gemini_property_feature_generator.get_feature_header()
                        if csv_header:
                            master_csv_local = os.path.join(main_output_dir, "[REDACTED_BY_SCRIPT]")
                            file_exists_csv = os.path.isfile(master_csv_local)
                            try:
                                with open(master_csv_local, 'a', newline='', encoding='utf-8') as csvfile_append:
                                    writer = csv.DictWriter(csvfile_append, fieldnames=csv_header, extrasaction='ignore')
                                    if not file_exists_csv: writer.writeheader()
                                    writer.writerow(features_gen)
                                print(f"[REDACTED_BY_SCRIPT]")
                            except Exception as csv_err_append: print(f"[REDACTED_BY_SCRIPT]")

                    else: print(f"[REDACTED_BY_SCRIPT]")
                else: 
                    print(f"[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]")
            except ImportError: print(f"[REDACTED_BY_SCRIPT]'gemini_property_feature_generator'."); pipeline_successful=False # Mark as fail
            except Exception as e_feat_csv: print(f"[REDACTED_BY_SCRIPT]")
        
        # Cleanup PIL images
        if floorplan_image and hasattr(floorplan_image, 'close'): floorplan_image.close()
        for img in indexed_room_images.values():
            if hasattr(img, 'close'): img.close()
        del floorplan_image, indexed_room_images # Explicitly delete
        # import gc; gc.collect() # If you still suspect memory issues

    print(f"[REDACTED_BY_SCRIPT]")

# --- Other utility functions (parse_dimensions_and_area, get_floorplan_images, etc.) ---
# These need to be defined globally as in your original script.
# Example:
UNKNOWN_TOKEN = "UNKNOWN"
def parse_dimensions_and_area(dimension_string):
    if not dimension_string or not isinstance(dimension_string, str): return None
    match = re.search(r'[REDACTED_BY_SCRIPT]', dimension_string, re.IGNORECASE)
    if match:
        try: return float(match.group(1)) * float(match.group(2))
        except ValueError: return None
    return None

def get_floorplan_images(floorplan_dir):
    paths = []
    if not os.path.isdir(floorplan_dir): return []
    try:
        for fname in os.listdir(floorplan_dir):
            fpath = os.path.join(floorplan_dir, fname)
            if os.path.isfile(fpath) and any(fname.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']):
                paths.append(fpath)
    except Exception as e: print(f"[REDACTED_BY_SCRIPT]")
    return paths

def get_room_images(image_dir, floorplan_paths_exclude):
    paths = []
    if not os.path.isdir(image_dir): return None # Critical error if image dir not found
    norm_excludes = {os.path.normpath(p) for p in floorplan_paths_exclude}
    try:
        for fname in os.listdir(image_dir):
            fpath = os.path.join(image_dir, fname)
            if os.path.isfile(fpath) and any(fname.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']) and os.path.normpath(fpath) not in norm_excludes:
                paths.append(fpath)
    except Exception as e: print(f"[REDACTED_BY_SCRIPT]"); return None
    return paths

def load_image(path):
    try:
        img = Image.open(path)
        img.verify(); img.close() 
        img = Image.open(path)
        img.load() 
        return img.convert('RGB') if img.mode != 'RGB' else img
    except FileNotFoundError: print(f"[REDACTED_BY_SCRIPT]"); return None
    except UnidentifiedImageError: print(f"[REDACTED_BY_SCRIPT]"); return None
    except Exception as e: print(f"[REDACTED_BY_SCRIPT]"); return None

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

def get_sp_theme_tokens(selling_points_list): # Ensure full function is copied
    if not selling_points_list or not isinstance(selling_points_list, list): return []
    tokens = set()
    expected_sp_tags = {"SP_SPACE", "SP_LIGHT", "SP_CONDITION", "SP_MODERN", "SP_CHARACTER", "SP_STYLE", "SP_FUNCTIONAL", "SP_FEATURE", "SP_STORAGE", "SP_GARDEN_ACCESS", "SP_GARDEN_VIEW", "SP_OTHER_VIEW", "SP_LOCATION", "SP_PRIVACY", "SP_POTENTIAL", "SP_LOW_MAINTENANCE", "SP_QUALITY_FINISH"}
    for sp_item in selling_points_list:
        if isinstance(sp_item, dict) and 'tags' in sp_item and isinstance(sp_item['tags'], list):
            for tag in sp_item['tags']:
                if isinstance(tag, str) and tag in expected_sp_tags: tokens.add(tag)
    return list(tokens)

def get_flaw_theme_tokens(flaws_list): # Ensure full function is copied
    if not flaws_list or not isinstance(flaws_list, list): return []
    tokens = set()
    expected_flaw_tags = {"FLAW_SPACE", "FLAW_LIGHT", "FLAW_CONDITION", "FLAW_DATED", "FLAW_NEEDS_UPDATE", "FLAW_MAINTENANCE", "FLAW_STORAGE", "FLAW_LAYOUT", "FLAW_BASIC_STYLE", "[REDACTED_BY_SCRIPT]", "FLAW_POOR_FINISH", "FLAW_UNATTRACTIVE", "FLAW_NOISE", "FLAW_ACCESSIBILITY"}
    for flaw_item in flaws_list:
        if isinstance(flaw_item, dict) and 'tags' in flaw_item and isinstance(flaw_item['tags'], list):
            for tag in flaw_item['tags']:
                 if isinstance(tag, str) and tag in expected_flaw_tags: tokens.add(tag)
    return list(tokens)


if __name__ == "__main__":
    # Ensure all prompts, persona details, and utility functions are defined globally above this.
    # Python 3.7+ for asyncio.run()
    if sys.version_info >= (3, 7):
        asyncio.run(main_async_runner())
    else:
        loop = asyncio.get_event_loop()
        try:
            loop.run_until_complete(main_async_runner())
        finally:
            loop.close()