import os
import csv
import PIL.Image
import google.generativeai as genai
import time
import ast
import logging
from collections import defaultdict

# Configure logging
logging.basicConfig(filename="house_analysis.log", level=logging.INFO, 
                    format='[REDACTED_BY_SCRIPT]')

# Configuration
PICS_ROOT_DIR = r"[REDACTED_BY_SCRIPT]"
FLOORPLAN_ROOT_DIR = r"[REDACTED_BY_SCRIPT]"
OUTPUT_FILE = r"[REDACTED_BY_SCRIPT]"
API_KEY = "YOUR_GOOGLE_API_KEY_HERE"  # Replace with your actual API key
MODEL_LIST = ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-1.0-pro"]

# Configure Gemini API
genai.configure(api_key=API_KEY)

# Safety settings to avoid content filtering issues
safety_settings = [
    {"category": "[REDACTED_BY_SCRIPT]", "threshold": "BLOCK_NONE"},
    {"category": "[REDACTED_BY_SCRIPT]", "threshold": "BLOCK_NONE"},
    {"category": "[REDACTED_BY_SCRIPT]", "threshold": "BLOCK_NONE"},
    {"category": "[REDACTED_BY_SCRIPT]", "threshold": "BLOCK_NONE"}
]

def get_addresses_with_years(root_dir):
    """[REDACTED_BY_SCRIPT]"""
    addresses = {}
    
    if not os.path.exists(root_dir):
        logging.error(f"[REDACTED_BY_SCRIPT]")
        return addresses
        
    for address in os.listdir(root_dir):
        address_path = os.path.join(root_dir, address)
        if os.path.isdir(address_path):
            addresses[address] = []
            for year in os.listdir(address_path):
                year_path = os.path.join(address_path, year)
                if os.path.isdir(year_path):
                    addresses[address].append(year)
    
    return addresses

def get_images_for_address_year(address, year, is_floorplan=False):
    """[REDACTED_BY_SCRIPT]"""
    root_dir = FLOORPLAN_ROOT_DIR if is_floorplan else PICS_ROOT_DIR
    image_dir = os.path.join(root_dir, address, year)
    images = []
    
    if os.path.exists(image_dir):
        for file in os.listdir(image_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp')):
                image_path = os.path.join(image_dir, file)
                images.append(image_path)
    
    return images

def generate_prompt(address, years, has_floorplan):
    """[REDACTED_BY_SCRIPT]"""
    floorplan_text = "with floorplan" if has_floorplan else "without floorplan"
    years_text = ", ".join(years)
    
    prompt = f"""
    I will be analyzing images of a property at {address}, {floorplan_text}, from the following years: {years_text}.

    Please analyze these images and provide the following information:
    1. Property type (detached, semi-detached, terraced, apartment, etc.)
    2. Estimated number of bedrooms
    3. Estimated number of bathrooms
    4. Presence of a garden or outdoor space
    5. Overall condition rating (1-10)
    6. Evidence of renovations between the different years shown
    7. Key selling points of this property
    8. Potential issues or areas for improvement
    9. Estimated market value range (based on visible features, not location)

    If multiple years are present, please compare the images between years and note any significant changes or renovations.

    If a floorplan is included, use it to better understand the layout and dimensions of the property.

    Present your analysis in a structured format with clear headings for each section.
    """
    return prompt

def process_with_gemini(images, floorplans, prompt, max_retries=3):
    """[REDACTED_BY_SCRIPT]"""
    current_model_index = 0
    
    for attempt in range(max_retries * len(MODEL_LIST)):
        model_name = MODEL_LIST[current_model_index]
        
        try:
            model = genai.GenerativeModel(model_name=model_name, 
                                         generation_config={"temperature": 0.7, "top_p": 0.95, "top_k": 40, "max_output_tokens": 8192},
                                         safety_settings=safety_settings)
            
            # Load images
            image_objects = []
            for img_path in images:
                try:
                    img = PIL.Image.open(img_path)
                    image_objects.append(img)
                except Exception as e:
                    logging.error(f"[REDACTED_BY_SCRIPT]")
            
            # Load floorplans
            for plan_path in floorplans:
                try:
                    plan = PIL.Image.open(plan_path)
                    image_objects.append(plan)
                except Exception as e:
                    logging.error(f"[REDACTED_BY_SCRIPT]")
            
            # If we have images to process
            if image_objects:
                response = model.generate_content([prompt] + image_objects)
                return response.text
            else:
                logging.warning("[REDACTED_BY_SCRIPT]")
                return "[REDACTED_BY_SCRIPT]"
                
        except Exception as e:
            logging.warning(f"[REDACTED_BY_SCRIPT]")
            current_model_index = (current_model_index + 1) % len(MODEL_LIST)
            time.sleep(2 ** (attempt % max_retries))  # Exponential backoff
    
    return "[REDACTED_BY_SCRIPT]"

def main():
    """[REDACTED_BY_SCRIPT]"""
    # Get all addresses from both directories
    pic_addresses = get_addresses_with_years(PICS_ROOT_DIR)
    floorplan_addresses = get_addresses_with_years(FLOORPLAN_ROOT_DIR)
    
    # Combine addresses from both sources
    all_addresses = set(pic_addresses.keys()) | set(floorplan_addresses.keys())
    
    logging.info(f"[REDACTED_BY_SCRIPT]")
    
    # Initialize the output CSV if it doesn't exist
    if not os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Address', 'Years', 'Has Floorplan', 'Analysis'])
    
    # Process each address
    processed_count = 0
    for address in all_addresses:
        try:
            logging.info(f"[REDACTED_BY_SCRIPT]")
            
            # Get all years for this address from both sources
            years = set()
            if address in pic_addresses:
                years.update(pic_addresses[address])
            if address in floorplan_addresses:
                years.update(floorplan_addresses[address])
            
            years = sorted(years)  # Sort years chronologically
            
            all_images = []
            all_floorplans = []
            
            # Collect all images and floorplans for this address across all years
            for year in years:
                images = get_images_for_address_year(address, year, is_floorplan=False)
                floorplans = get_images_for_address_year(address, year, is_floorplan=True)
                
                all_images.extend(images)
                all_floorplans.extend(floorplans)
            
            has_floorplan = len(all_floorplans) > 0
            
            # Generate prompt
            prompt = generate_prompt(address, years, has_floorplan)
            
            # Process with Gemini
            analysis = process_with_gemini(all_images, all_floorplans, prompt)
            
            # Save results
            with open(OUTPUT_FILE, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([address, ', '.join(years), has_floorplan, analysis])
            
            processed_count += 1
            logging.info(f"[REDACTED_BY_SCRIPT]")
            
            # Add a delay to avoid rate limiting
            time.sleep(5)
            
        except Exception as e:
            logging.error(f"[REDACTED_BY_SCRIPT]")
    
    logging.info(f"[REDACTED_BY_SCRIPT]")

if __name__ == "__main__":
    main()