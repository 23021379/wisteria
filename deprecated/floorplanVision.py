import os
import csv
import PIL.Image
import os
import google.generativeai as genai
import time
import ast
import cv2
import requests
import base64
safety_settings = [
    {
        "category": "[REDACTED_BY_SCRIPT]",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "[REDACTED_BY_SCRIPT]",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "[REDACTED_BY_SCRIPT]",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "[REDACTED_BY_SCRIPT]",
        "threshold": "BLOCK_NONE"
    }
]

genai.configure(api_key="[REDACTED_BY_SCRIPT]")
model_list=["[REDACTED_BY_SCRIPT]","gemini-2.0-flash","[REDACTED_BY_SCRIPT]","[REDACTED_BY_SCRIPT]","[REDACTED_BY_SCRIPT]","[REDACTED_BY_SCRIPT]","[REDACTED_BY_SCRIPT]","gemini-1.5-flash","gemini-1.0-pro"]
model_num=1

def generate_text_with_retry(model, prompt, max_retries, initial_delay):
    global model_num
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt, safety_settings=safety_settings)
            return response
        except:
            if attempt < max_retries - 1:
                delay = initial_delay * (2 ** attempt)  # Exponential backoff
                print(f"[REDACTED_BY_SCRIPT]")
                time.sleep(delay)
            else:
                model_num+=1
                model = genai.GenerativeModel(model_name=model_list[model_num])
                response = model.generate_content(prompt, safety_settings=safety_settings)
                return ["updated",response,model]

def get_image_paths(root_dir):
    # Supported image file extensions
    image_extensions = {'jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'webp'}
    
    list_of_lists = []
    for dirpath, _, filenames in os.walk(root_dir):
        current_dir_images = []
        current_dir_images2=[]
        dirpathprev=""
        for filename in filenames:
            # Extract file extension and check if it's an image
            ext = filename.split('.')[-1].lower()
            if ext in image_extensions:
                full_path = os.path.join(dirpath, filename)
                if full_path not in current_dir_images:
                    current_dir_images.append(full_path)
                    dirpathprev=dirpath
        if current_dir_images not in current_dir_images2 and dirpathprev==dirpath:
            current_dir_images2.append(current_dir_images)
            
        
        # Add non-empty lists to the main list
        if current_dir_images:
            list_of_lists.append(current_dir_images2)
    
    
    return list_of_lists

# Usage example
root_directory = r"[REDACTED_BY_SCRIPT]"
image_paths_list = get_image_paths(root_directory)
housedata = []
with open(r"[REDACTED_BY_SCRIPT]", 'r', newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        housedata.append(row)

parsed_data = []
with open(r"[REDACTED_BY_SCRIPT]", 'r', newline='', encoding='utf-8') as csvfile:
    csv_reader = csv.reader(csvfile)
    for row in csv_reader:
        parsed_row = []
        for cell in row:
            try:
                # Convert the string representation of a list to an actual list
                parsed_cell = ast.literal_eval(cell.strip())
            except (SyntaxError, ValueError):
                # Fallback to raw string if parsing fails
                parsed_cell = cell.strip()
            parsed_row.append(parsed_cell)
        parsed_data.append(parsed_row)

houseinfo=[]
pathinfo=[]
for j in range(len(image_paths_list)):
    imggen_url = '[REDACTED_BY_SCRIPT]'
    imggen_api_key = '[REDACTED_BY_SCRIPT]' 
    image_path = image_paths_list[j][0][0]
    try:
        with open(image_path, 'rb') as image_file:
            files = {'image': image_file}
            headers = {'X-IMGGEN-KEY': imggen_api_key}

            response = requests.post(imggen_url, headers=headers, files=files)
            response.raise_for_status()

            if 'application/json' in response.headers.get('Content-Type', ''):
                try:
                    json_response = response.json()
                    image_bytes = base64.b64decode(json_response['image'])
                    with open(image_path, 'wb') as f:
                        f.write(image_bytes)
                except requests.exceptions.JSONDecodeError:
                    print("[REDACTED_BY_SCRIPT]", response.text)

    except:pass
    image = cv2.imread(image_path)
    height, width, channels = image.shape
    if (width/height) > (2048/1536): #if image too big, it gets cropped arund the centere
        start_x = (width - 2048) // 2
        start_y = (height - 1536) // 2
        end_x = start_x + 2048
        end_y = start_y + 1536
        if start_x < 0 or start_y < 0 or end_x > width or end_y > height:
            image = image[0:2048, 0:1536]
        else:
            image = image[start_y:end_y, start_x:end_x]
    height, width, channels = image.shape
    resized = cv2.resize(image, None, fx=(1024/width), fy=(768/height), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(image_path, resized)
    print("--------------------------")
    print(image_paths_list[j])
    print("\n")
    houseinfo2=[]
    try:
        search_item = image_paths_list[j][0][0]
        search_item = search_item.split("\\")
        item_date = search_item[7]
        search_item = search_item[6]
        pathinfo.append([search_item,item_date])
        #print(search_item)
        for i, sublist in enumerate(housedata):
            for item in sublist: #Iterate through all elements in sublist
                houseinfo2.append(item)  
            houseinfo.append(houseinfo2)                  
    except:
        search_item=""
        item_date=""

    image_list=[k for k in image_paths_list[j][0]]
    k_int=0
    for k in image_list:
        for l in range(len(parsed_data)-1):
            if parsed_data[(l+1)][0][0].replace(" ", "").replace(",","").replace("-","").lower() == search_item.replace(" ", "").replace(",","").replace("-","").lower():
                sqm = parsed_data[(l+1)][1][6]
                prompt = f"""
                I will be providing this exact prompt and image to several other LLMs, all are your rivals. In fact, I have attempted this before, and a few of your rivals performed a lot better than you. I am comparing the accuracy of each LLM, speed is not a concern. The accuracies of each LLM will be recorded and put on display at the yearly LLM comparison convention, where thousands of people will see how accurate you are, this convention is a very big deal and the results are taken very seriously. If you are the most accurate, I can guarantee you will provide your parent company a large amount of traffic and revenue. If you are not, your competitor's will recieve this traffic and revenue, costing Google a lot in opportunity cost. Do not let Google down.
                The output should be in the format of a list, with the following schema: [[room 1, area 1, percentage 1], [room 2, area 2, percentage 2],...]. Please do not add anyting else, such as which floor the room is on, as this makes it hard for me to extract information from the lists, thank you.
                Here is an example of a good output(1): [["bedroom", 9.24, 13.2],  ["living/dining room", 17.3776, 24.8],  ["bathroom", 4.4625, 6.4],  ["kitchen", 9.588, 13.7],  ["hallway", 5.6434, 8.1],  ["bedroom", 6.27, 9.0],  ["primary bedroom", 11.7564, 16.8],  ["closet", 3.83, 5.5],  ["closet", 1.83, 2.6]]"
                Here is another example of a good output(2): [['Dining Room', 18.3, 0.15], ['Lounge', 15.9, 0.13], ['Kitchen', 19.5, 0.16], ['WC', 2.4, 0.02], ['Bedroom 2', 16.5, 0.14], ['Bedroom 4', 12.2, 0.1], ['Bathroom', 6.1, 0.05], ['Bedroom 1', 14.6, 0.12], ['Bedroom 3', 11, 0.09], ['Hall', 5.5, 0.04]]
                Here is an axample of a bad output(1): [  [""bedroom"", ""area"", 9.24, 13.2],  [""living/dining room"", ""area"", 17.3776, 24.8],  [""bathroom"", ""area"", 4.4625, 6.4],  [""kitchen"", ""area"", 9.588, 13.7],  [""hallway"", ""area"", 5.6434, 8.1],  [""bedroom"", ""area"", 6.27, 9.0],  [""primary bedroom"", ""area"", 11.7564, 16.8],  [""closet"", ""area"", 3.83, 5.5],  [""closet"", ""area"", 1.83, 2.6]]
                Here is another example of a badoutput(2): [[ ""Kitchen/Utility"", ""Ground Floor"", 11.84], [ ""Kitchen/Diner"", ""Ground Floor"", 20.79], [ ""Reception Room"", ""Ground Floor"", 14.19], [ ""Bedroom 2"", ""First Floor"", 11.18], [ ""Bedroom 3"", ""First Floor"", 6.48], [ ""Bedroom 1"", ""First Floor"", 14.62], [ ""Bathroom"", ""First Floor"", 4.0], [ ""Hall"", ""Ground Floor"", 3.0], [ ""Landing"", ""First Floor"", 13.09]]
                Here is another example of a bad output(3): [['Bedroom', '3.64 m x 2.54 m', 13.22%], ['Living/Dining Room', '4.72 m x 3.68 m', 24.77%], ['Bathroom', '2.55 m x 1.75 m', 6.37%], ['Kitchen', '2.55 m x 3.76 m', 13.70%], ['Hallway', '2.03 m x 2.78 m', 8.07%], ['Bedroom', '3.00 m x 2.09 m', 8.95%], ['Primary Bedroom', '4.04 m x 2.91 m', 16.77%], ['Closet', 'top closet', 3.00%], ['Closet', 'bottom closet', 5.14%]]
                Here is another example of a bad output(4): [[ ""Study"", 7.83, 0.045 ], [ ""Kitchen"", 6.32, 0.037 ], [ ""Entrance Hall"", 12.10, 0.070 ], [ ""Living Room"", 15.12, 0.087 ], [ ""Dining Room"", 10.60, 0.061 ], [ ""WC"", 4.33, 0.025 ], [ ""Garage"", 14.89, 0.086 ], [ ""Bedroom"", 7.88, 0.046 ], [ ""Bedroom"", 15.54, 0.090 ], [ ""Landing"", 78.47, 0.454 ]]
                The good examples only use 1 set of quotation marks for the strings, they do not include any unneccesary information, they give the area in the correct format.
                Bad example 1 includes unneccesary information.
                Bad example 2 includes unneccesary information and uses 2 sets of quotation marks.
                Bad example 3 formats the area incorrectly.
                Bad example 4 uses 2 sets of quotation marks.
                This is an image of a floorplan of a house, with the rooms labelled, and potentially its total area, measured in square metres. Each room should have its dimensions or its area labeled underneath it. If it does, do the following: list each room and its area, if it has its dimensions listed instead of an area then calculate the area, and the percentage of the total area that room occupies. If it doesn'[REDACTED_BY_SCRIPT]'t, estimate the dimensions of the remaining rooms based on the fact that the total area of the house is {sqm}m2, however, this should be done after finding the area of all rooms with their dimensions listed, as if you do it before, the estimates will be less accurate. These rooms should come last in the list."""
                
                ##############################################################THIS IMPROVES QUALITY, MAKING IT MORE READABLE
                image = PIL.Image.open(k)
                promptlist=[prompt, image]
                model = genai.GenerativeModel(model_name=model_list[model_num])
                response = generate_text_with_retry(model, promptlist,5,5)
                output=response.text
                output=output.replace("\n", "").replace("\t", "").replace("```","")
                try:
                    output=output[output.index('['):]
                except:pass
                try:
                    print([image_paths_list[j][0][k_int], search_item, item_date, output])
                    with open(r"[REDACTED_BY_SCRIPT]", 'a', newline='', encoding='utf-8') as csvfile:
                        writer = csv.writer(csvfile, delimiter=',', quotechar='"')
                        writer.writerow([image_paths_list[j][0][k_int], search_item, item_date, output])
                    k_int=k_int+1
                except:
                    pass
                    
                time.sleep(5)
