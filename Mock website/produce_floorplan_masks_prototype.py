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
        #self.model_id = "gemini-2.5-flash"
        self.model_id = "[REDACTED_BY_SCRIPT]"

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
        Find the room labeled '{room_label}'{dim_context} in this floorplan.
        
        Output a STRICT JSON list where each entry contains:
        - "label": The name of the room found
        - "polygon": A list of [y, x] coordinates (normalized 0-1000) representing the vertices of the room's floor area. 
          Trace the specific shape of the room (including L-shapes or alcoves) in clockwise order.
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
            # A. Extract Polygon Coordinates
            points_n = item.get('polygon', [])
            
            if not points_n or len(points_n) < 3:
                print(f"[REDACTED_BY_SCRIPT]'{room_label}'")
                return full_image_mask

            # B. De-normalize Coordinates (0-1000 -> Pixels)
            pixel_points = []
            for y_n, x_n in points_n:
                y = int((y_n / 1000) * height)
                x = int((x_n / 1000) * width)
                pixel_points.append([x, y]) # OpenCV expects [x, y]
            
            # Convert to NumPy array of shape (N, 1, 2) required by polylines/fillPoly
            pts = np.array(pixel_points, np.int32)
            pts = pts.reshape((-1, 1, 2))
            
            # C. Draw Polygon on Mask
            # Fill the polygon with white (255)
            cv2.fillPoly(full_image_mask, [pts], 255)
            
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