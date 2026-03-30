import os
import cv2
import json
import numpy as np
import base64
from PIL import Image
from io import BytesIO
from google import genai
from google.genai import types

class WisteriaGeminiSegmenter:
    def __init__(self, api_key):
        if not api_key:
            raise ValueError("[REDACTED_BY_SCRIPT]")
        self.client = genai.Client(api_key=api_key)
        self.model_id = "gemini-2.5-flash"
        #self.model_id = "[REDACTED_BY_SCRIPT]"

    def _clean_json_response(self, text):
        """
        Removes markdown formatting if the model adds it, and handles common JSON string issues.
        """
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()

    def _fix_base64_padding(self, s):
        """
        Ensures the base64 string length is a multiple of 4 by adding '=' padding.
        """
        return s + '=' * (-len(s) % 4)

    def _parse_svg_path_to_mask(self, svg_path, width, height):
        """
        Parses a simplified SVG path string (M x y L x y Z) and draws it onto a mask.
        Assumes coordinates are normalized 0-1000.
        """
        import re
        
        # Regex to find commands and coordinate pairs.
        # Looking for 'M' or 'L' followed by coordinates. 
        # Example: "M 100 100 L 200 100 L 200 200 Z"
        # We normalize everything to a list of points.
        
        # Clean string
        svg_path = svg_path.replace(',', ' ').strip()
        
        # Extract all numbers
        tokens = re.findall(r'[0-9.]+', svg_path)
        
        # Convert to floats
        coords = [float(t) for t in tokens]
        
        # Pair them up (y, x) or (x, y) - Instructions say Normalized 0-1000
        # We will assume pairs are [y, x] based on previous context, 
        # BUT standard SVG is [x, y]. 
        # *Correction*: We will instruct the model to use [x, y] in the prompt to match standard SVG syntax,
        # ensuring 2.5's coding training data logic applies.
        
        pixel_points = []
        if len(coords) < 4: # Need at least 2 points
            return np.zeros((height, width), dtype=np.uint8)
            
        for i in range(0, len(coords), 2):
            if i + 1 >= len(coords): break
            
            # Standard SVG is X, Y. We will prompt for this.
            x_n = coords[i]
            y_n = coords[i+1]
            
            x = int((x_n / 1000) * width)
            y = int((y_n / 1000) * height)
            pixel_points.append([x, y])
            
        full_image_mask = np.zeros((height, width), dtype=np.uint8)
        
        if len(pixel_points) > 2:
            pts = np.array(pixel_points, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.fillPoly(full_image_mask, [pts], 255)
            
        return full_image_mask

    def get_room_mask(self, image_path, room_label, room_dimensions=None):
        """
        Generates a binary mask for a specific room using Gemini's native segmentation.
        """
        # 1. Load Image & Dimensions
        try:
            img_pil = Image.open(image_path)
        except Exception as e:
            print(f"[REDACTED_BY_SCRIPT]")
            return None

        width, height = img_pil.size
        
        # 2. Construct the Strict Schema Prompt
        dim_context = f"[REDACTED_BY_SCRIPT]" if room_dimensions else ""
        
        prompt = f"""
        Analyze the floorplan and trace the floor area of the room labeled '{room_label}'{dim_context}.
        Ignore furniture, text, and icons inside the room. Trace only the walls defining the room's shape.
        
        Output a STRICT JSON list where each entry contains:
        - "label": The name of the room found
        - "svg_path": A simplified SVG path string (`d` attribute) using normalized coordinates (0-1000).
          Format: "M x1 y1 L x2 y2 L x3 y3 ... Z"
          Use 'M' for Move To (start) and 'L' for Line To.
          Ensure the coordinates follow the walls precisely to define the polygon.
          Coordinates are [x, y] where 0,0 is top-left and 1000,1000 is bottom-right.
        """
        
        # 3. Call Gemini API
        try:
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=[prompt, img_pil],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.0,
                    max_output_tokens=65536 
                )
            )

            print(f"[REDACTED_BY_SCRIPT]'{room_label}': {response.text}")
            
            # SANITIZATION STEP
            clean_text = self._clean_json_response(response.text)
            response_data = json.loads(clean_text)
            
        except json.JSONDecodeError as e:
            print(f"[REDACTED_BY_SCRIPT]'{room_label}': {e}")
            # Debug: print the start of the bad response to diagnose
            print(f"[REDACTED_BY_SCRIPT]")
            return np.zeros((height, width), dtype=np.uint8)
        except Exception as e:
            print(f"[REDACTED_BY_SCRIPT]'{room_label}': {e}")
            return np.zeros((height, width), dtype=np.uint8)

        # 4. Process the Response
        full_image_mask = np.zeros((height, width), dtype=np.uint8)
        
        if not response_data:
            print(f"[WARN] No room found for '{room_label}'")
            return full_image_mask

        item = response_data[0] 
        
        try:
            # A. Extract SVG Path
            svg_path = item.get('svg_path', "")
            
            if not svg_path:
                print(f"[REDACTED_BY_SCRIPT]'{room_label}'")
                return full_image_mask

            # B. Parse and Draw
            full_image_mask = self._parse_svg_path_to_mask(svg_path, width, height)
            
            print(f"[REDACTED_BY_SCRIPT]'{room_label}'")
            
        except Exception as e:
            print(f"[REDACTED_BY_SCRIPT]'{room_label}': {e}")
            
        return full_image_mask

def main():
    # --- PATHS PROVIDED IN DIRECTIVE ---
    # Using raw strings for Windows paths to avoid escape character issues
    image_path = r"[REDACTED_BY_SCRIPT]"
    json_path = r"[REDACTED_BY_SCRIPT]"
    
    # Verify file existence
    if not os.path.exists(image_path):
        print(f"[REDACTED_BY_SCRIPT]")
        return
    if not os.path.exists(json_path):
        print(f"[REDACTED_BY_SCRIPT]")
        return

    # Create output directory for masks relative to the script or image
    output_dir = os.path.join(os.path.dirname(image_path), "generated_masks")
    os.makedirs(output_dir, exist_ok=True)

    # Initialize Architect's Segmenter
    api_key = "[REDACTED_BY_SCRIPT]"
    if not api_key:
        print("[REDACTED_BY_SCRIPT]")
        return
        
    segmenter = WisteriaGeminiSegmenter(api_key=api_key)

    with open(json_path, 'r') as f:
        data = json.load(f)

    rooms = data.get("rooms_with_dimensions", [])
    
    print(f"[REDACTED_BY_SCRIPT]")
    
    for room in rooms:
        label = room.get("label")
        dimensions = room.get("dimensions")
        
        if not label:
            continue
            
        mask = segmenter.get_room_mask(image_path, label, dimensions)
        
        # VALIDATION: Only save if the mask actually contains data
        if mask is not None and np.count_nonzero(mask) > 0:
            safe_label = "".join([c if c.isalnum() else "_" for c in label])
            
            # 1. Save Binary Mask
            mask_filename = f"[REDACTED_BY_SCRIPT]"
            mask_path = os.path.join(output_dir, mask_filename)
            cv2.imwrite(mask_path, mask)
            
            # 2. Create Zillow-Style Overlay (Yellow)
            # Load original
            original_img = cv2.imread(image_path)
            
            if original_img is not None:
                # Create a solid yellow image (BGR format: Blue=0, Green=255, Red=255)
                yellow_layer = np.zeros_like(original_img)
                yellow_layer[:] = [0, 255, 255]
                
                # Apply the binary mask to the yellow layer
                # We only want yellow where the mask is white
                yellow_masked = cv2.bitwise_and(yellow_layer, yellow_layer, mask=mask)
                
                # Blend: Original Image + Yellow Mask
                # alpha=1.0 (Original), beta=0.8 (Overlay Transparency), gamma=0
                overlay_img = cv2.addWeighted(original_img, 1.0, yellow_masked, 0.8, 0)
                
                # Draw the specific Room Label text on the center of the contour for clarity
                # Calculate center of mass for text placement
                M = cv2.moments(mask)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    cv2.putText(overlay_img, label, (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

                overlay_filename = f"[REDACTED_BY_SCRIPT]"
                overlay_path = os.path.join(output_dir, overlay_filename)
                cv2.imwrite(overlay_path, overlay_img)
                print(f"[REDACTED_BY_SCRIPT]")
            else:
                print(f"[REDACTED_BY_SCRIPT]")
        else:
            print(f"[REDACTED_BY_SCRIPT]'{label}'")

    print("[REDACTED_BY_SCRIPT]")

if __name__ == "__main__":
    main()