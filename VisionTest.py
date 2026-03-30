# prompt = r"""[REDACTED_BY_SCRIPT]"room classification", "rating", [["selling Point1", "selling Point2",...], ["flaw1", "flaw2",...]], ["estate Agent Selling Point1", "estate Agent Selling Point2",...], ["emotional Buyer Rating", "emotional Buyer Selling Point1", "emotional Buyer Selling Point2", ..., "emotional Buyer Flaw1", "emotional Buyer Flaw2", ...], ["Logical Potential Buyer Rating", "Logical Potential Buyer Selling Point1", "Logical Potential Buyer Selling Point2", ..., "Logical Potential Buyer Flaw1", "Logical Potential Buyer Flaw2", ...],  ["Logical Potential Buyer Rating", "skeptical Potential Buyer Selling Point1", "skeptical Potential Buyer Selling Point2", ..., "skeptical Potential Buyer Flaw1", "skeptical Potential BuyerFlaw2", ...], "renovation probability", "extensivity", ["renovated feature1", "renovated feature2", ...], ["potential Hazards"]] 
# Here is a real example: "[REDACTED_BY_SCRIPT]","address",date,"["front of house",5,[["Brick facade","Abundance of windows"],["Dated facade","Plain facade"]],["The building'[REDACTED_BY_SCRIPT]","[REDACTED_BY_SCRIPT]","[REDACTED_BY_SCRIPT]","[REDACTED_BY_SCRIPT]","The property's architecture is well-suited to its surroundings, offering a seamless integration with the neighborhood."],[6,"The building has so many windows, it's like sunshine all the time!","[REDACTED_BY_SCRIPT]","[REDACTED_BY_SCRIPT]","[REDACTED_BY_SCRIPT]","It's close to nature, and that'[REDACTED_BY_SCRIPT]","[REDACTED_BY_SCRIPT]","[REDACTED_BY_SCRIPT]","I'm not sure about the landscaping, it could be better.","I'[REDACTED_BY_SCRIPT]","[REDACTED_BY_SCRIPT]"],[6,"[REDACTED_BY_SCRIPT]","[REDACTED_BY_SCRIPT]","[REDACTED_BY_SCRIPT]","[REDACTED_BY_SCRIPT]","Good location. ","[REDACTED_BY_SCRIPT]","[REDACTED_BY_SCRIPT]","[REDACTED_BY_SCRIPT]","[REDACTED_BY_SCRIPT]","Doesn't look too modern."],[4,"Dated facade, needs a complete makeover.","Lack of any distinctive features, making it bland and uninspiring.","Generic architecture, lacking any charm or character.","Inadequate landscaping, making the exterior feel uninviting.","The building is a bit tall and might feel imposing to some.","Facade can be improved with a modern cladding.","Unique architectural elements can be added to the facade.","Landscaping can be redesigned to enhance the curb appeal.","Adding a balcony to the facade.","Adding colors to the exterior."],"NA","NA",[],[]]"
# Follow this exact schema, and do not include "```json" or unnecessary spaces as i have to process this csv. Also, ensure all items are surrounded by single quotation marks, eg: "front of house","5",...


# For this picture, can you identify what kind of room this is, and key selling points or any flaws, and please do not use furniture as a selling point or flaw, as these won’t come with the house; kitchen counters, cabinets, etc. are exempt from this rule. The classes for room are kitchen, bathroom, living room, sitting room, hallway, garage, conservatory, garden room, dining room, bedroom, hallway, garden, front of house, or back of house. Based on the selling points and flaws you generate, could you give this room a rating from 1-10, where 10 could contain some or all of the following:  modern, spacious, good lighting, modern appliances, modern furniture, coherent color scheme, unique features, good framing of the picture, large mirror, and overall, extremely appealing to most potential buyers. 
# A rating of 5 could contain some or all the following: somewhat modern, somewhat spacious, relatively modern appliances or furniture, relatively good lighting, and about 50% of potential buyers would be interested in this room. 

# Then, imagine you are an estate agent, keen on selling this house. Provide 5 selling points that will paint this room in the best light possible – using features you identify in the image. Again, furniture cannot be used as a selling point. 

# Then imagine you are an emotional potential buyer, who bases their decisions on emotion, rather than logic. Where color scheme, furniture, space, etc. Greatly affect your decision. As the emotional potential buyer, please generate a rating from 1 to 10, this time furniture can affect your rating. Then, identify 5 key selling points and 5 flaws, this time furniture can be used as a selling point. 

# Then imagine you are a logical potential buyer, who bases their decisions on fact, rather than emotion. As a logical potential buyer, you aim to maximize the benefits your money can bring you. You can look past some flaws like bad lighting, and no furniture, to see the potential of the room. As this logical potential buyer, rate this room from 1 to 10 (where furniture doesn’t affect the rating since you know that it doesn’t come with the house), and then identify 5 key selling points, and 5 flaws. 

# Then, imagine you are a skeptical potential buyer, who is more focused on the flaws. Imagine you just don’t like this room; however, you need to explain why and what could be done to improve it. As the skeptical potential buyer, rate this room from 1 to 10. Then give your top 5 flaws, and how each flaw can be improved. 

# Then, as an inspector, give me a percentage to represent how likely it is that the house has been recently renovated, and if the percentage is over 70%, do the following: identify how extensive the renovations were, and what features are likely to have been updated in the renovation. If the percentage is below 70%, then your outputs should be “NA”Indicators of a renovation could include: room with a skylight, or modern flooring, or modern appliances, or big windows, a very modern kitchen, a very modern bathroom. Classes of extensivity of renovations are: Major, Moderate, and Minor.  

# Then, still as the inspector, identify any potential hazards – current and future – that can be identified in this room. 

# After you have generated your response, go over it and ensure that it follows the format of the schema to a tee. If not, please generate a new response until it does.

# Please generate a response, even if you are uncertain about the answer."""








import os
import csv
import PIL.Image
import os
import google.generativeai as genai
from google.genai import types
import time
import ast
from google.api_core import exceptions
import csv
from collections import defaultdict
import requests
import base64
import cv2

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
model_num=0
prompt = r"""
I will be providing this exact prompt and image to several other LLMs, all are your rivals. In fact, I have attempted this before, and a few of your rivals performed a lot better than you. I have the actual ratings of each room, and I am comparing the accuracy of each LLM to the actual ratings, speed is not a concern. The accuracies of each LLM will be recorded and put on display at the yearly image rating convention, where thousands of people will see how accurate you are, this convention is a very big deal and the results are taking very seriously. If you are the most accurate, I can guarantee you will provide your parent company a large amount of traffic and revenue. If you are not, your competitor's will recieve this traffic and revenue, costing Google a lot in opportunity cost. Do not let Google down.

The format for the output should be a list using the following schema: ["room classification", "rating", "[REDACTED_BY_SCRIPT]", "[REDACTED_BY_SCRIPT]",  "[REDACTED_BY_SCRIPT]", "[REDACTED_BY_SCRIPT]", "extensivity"] 
Follow this exact schema, and do not include "```json" or unnecessary spaces as i have to process this csv. Also, ensure all items are surrounded by single quotation marks, eg: "front of house","5",...

For this picture, can you identify what kind of room this is. The classes for room are kitchen, bathroom, living room, sitting room, hallway, garage, conservatory, garden room, dining room, bedroom, hallway, garden, front of house, or back of house. Could identify 5 key selling points and 5 flaws, and use those to help you you give this room a rating from 1-10, where 10 could contain some or all of the following:  modern, spacious, good lighting, modern appliances, modern furniture, coherent color scheme, unique features, good framing of the picture, large mirror, and overall, extremely appealing to most potential buyers. 
A rating of 5 could contain some or all the following: somewhat modern, somewhat spacious, relatively modern appliances or furniture, relatively good lighting, and about 50% of potential buyers would be interested in this room. 

Then imagine you are an emotional potential buyer, who bases their decisions on emotion, rather than logic. Where color scheme, furniture, space, etc. Greatly affect your decision. As the emotional potential buyer, identify 5 key selling points and 5 flaws, and use those to help you generate a rating from 1 to 10, this time furniture can affect your rating.. 

Then imagine you are a logical potential buyer, who bases their decisions on fact, rather than emotion. As a logical potential buyer, you aim to maximize the benefits your money can bring you. You can look past some flaws like bad lighting, and no furniture, to see the potential of the room. As this logical potential buyer, identify 5 key selling points and 5 flaws, and use those to help you rate this room from 1 to 10 (where furniture doesn’t affect the rating since you know that it doesn’t come with the house). 

Then, imagine you are a skeptical potential buyer, who is more focused on the flaws. Imagine you just don’t like this room; however, you need to explain why and what could be done to improve it. As the skeptical potential buyer, identify 5 key selling points and 5 flaws, and use those to help you rate this room from 1 to 10. 

Then, as an inspector, give me a percentage to represent how likely it is that the house has been recently renovated.

After you have generated your response, go over it and ensure that it follows the format of the schema to a tee. If not, please generate a new response until it does.

In your response, do not include any selling points or flaws, only the ratings and classification.

Please generate a response, even if you are uncertain about the answer."""



def get_image_paths(root_dir):
    # Supported image file extensions
    image_extensions = {'jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp'}
    
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


def generate_text_with_retry(model, prompt, max_retries, initial_delay):
    global model_num
    generation_config = {
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
    }
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
                model = genai.GenerativeModel(model_name=model_list[model_num],generation_config=generation_config,safety_settings=safety_settings)
                response = model.generate_content(prompt, safety_settings=safety_settings)
                return ["updated",response,model]

# # Usage example
# root_directory = r"[REDACTED_BY_SCRIPT]"
# image_paths_list = get_image_paths(root_directory)
# housedata = []
# with open(r"[REDACTED_BY_SCRIPT]", 'r', newline='', encoding='utf-8') as csvfile:
#     reader = csv.reader(csvfile)
#     for row in reader:
#         housedata.append(row)

# houseinfo=[]
# pathinfo=[]

# for j in range(len(image_paths_list)):

#     imggen_url = '[REDACTED_BY_SCRIPT]'
#     imggen_api_key = '[REDACTED_BY_SCRIPT]' 
#     for p in range(len(image_paths_list[j][0])):
#         print(p)
#         image_path = image_paths_list[j][0][p]
#         try:
#             with open(image_path, 'rb') as image_file:
#                 files = {'image': image_file}
#                 headers = {'X-IMGGEN-KEY': imggen_api_key}

#                 response = requests.post(imggen_url, headers=headers, files=files)
#                 response.raise_for_status()

#                 if 'application/json' in response.headers.get('Content-Type', ''):
#                     try:
#                         json_response = response.json()
#                         image_bytes = base64.b64decode(json_response['image'])
#                         with open(image_path, 'wb') as f:
#                             f.write(image_bytes)
#                     except requests.exceptions.JSONDecodeError:
#                         print("[REDACTED_BY_SCRIPT]", response.text)
#         except:pass
#         image = cv2.imread(image_path)
#         height, width, channels = image.shape
#         if (width/height) > (2048/1536): #if image too big, it gets cropped arund the centere
#             start_x = (width - 2048) // 2
#             start_y = (height - 1536) // 2
#             end_x = start_x + 2048
#             end_y = start_y + 1536
#             if start_x < 0 or start_y < 0 or end_x > width or end_y > height:
#                 image = image[0:2048, 0:1536]
#             else:
#                 image = image[start_y:end_y, start_x:end_x]
#         height, width, channels = image.shape
#         resized = cv2.resize(image, None, fx=(1024/width), fy=(768/height), interpolation=cv2.INTER_CUBIC)
#         cv2.imwrite(image_path, resized)
    
#     houseinfo2=[]
#     try:
#         search_item = image_paths_list[j][0][0]
#         search_item = search_item.split("\\")
#         item_date = search_item[7]
#         search_item = search_item[6]
#         pathinfo.append([search_item,item_date])
#         #print(search_item)
#         for i, sublist in enumerate(housedata):
#             for item in sublist: #Iterate through all elements in sublist
#                 houseinfo2.append(item)  
#             houseinfo.append(houseinfo2)                  
#     except:pass
#     # # image_path_18 = r"[REDACTED_BY_SCRIPT]"
#     # # sample_file_18 = PIL.Image.open(image_path_18)
#     image_list=[PIL.Image.open(k) for k in image_paths_list[j][0]]
#     promptlist=[prompt]
#     k_int=0
#     for k in image_list:
#         promptlist=[prompt, k]
#         print(image_paths_list[j][0][k_int])
#         model = genai.GenerativeModel(model_name=model_list[model_num])
#         response = generate_text_with_retry(model, promptlist,5,5)
#         try:
#             if response[0] == "updated":
#                 model = response[2]
#                 response= response[1]
#             else:
#                 response = response
#         except:
#             response = response
#         output=response.text
#         output=output.replace("```json", "").replace("```", "").replace("\n", "").replace("\t", "")
#         try:
#             output=output[output.index("[]"):]
#         except:pass
#         with open(r'[REDACTED_BY_SCRIPT]', 'a', newline='', encoding='utf-8') as csvappend:
#             writer = csv.writer(csvappend, delimiter=",", quotechar='"')
#             writer.writerow([image_paths_list[j][0][k_int], search_item, item_date, output])
#         k_int=k_int+1
#         time.sleep(5)




#########################################################################################################
#########################################################################################################
#########################################################################################################
#########################################################################################################





def generate_text_with_retry2(model, prompt, max_retries, initial_delay):
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

parsed_data=[]
parsed_dataout=[]
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
        if parsed_row not in parsed_data:
            parsed_data.append(parsed_row)



addressL=[]
for i in range(len(parsed_data)):
    if addressL!=[]:
        try:
            if addressL[-1][0][1]==parsed_data[i][1]:
                addressL[-1].append(parsed_data[i])
            else:
                addressL.append([parsed_data[i]])
        except:addressL.append([parsed_data[i]])
    else:
        addressL.append([parsed_data[i]])



def compareYears(addressL2):
    genai.configure(api_key="[REDACTED_BY_SCRIPT]")
    date_list=[k[2] for k in addressL2]
    room_list=[k[3][0] for k in addressL2]
    image_list=[PIL.Image.open(k[0]) for k in addressL2]
    address=addressL2[0][1]
    roomTypes=["kitchen", "bathroom", "living room", "sitting room", "hallway",
        "garage", "conservatory", "garden room", "dining room", "bedroom",
        "garden", "front of house", "back of house"]
    roomTypes2=['"kitchen"', '"bathroom"', '"living room"', '"sitting room"', '"hallway"',
        '"garage"', '"conservatory"', '"garden room"', '"dining room"', '"bedroom"',
        '"garden"', '"front of house"', '"back of house"']
    k_int=0
    for j in range(len(roomTypes)):
        try:
            date_list2=[date_list[l] for l, item in enumerate(room_list) if item == roomTypes[j] or item == roomTypes2[j]]
            room_list2=[room_list[l] for l, item in enumerate(room_list) if item == roomTypes[j] or item == roomTypes2[j]]
            image_list2=[image_list[l] for l, item in enumerate(room_list) if item == roomTypes[j] or item == roomTypes2[j]]
            roomL=[date_list2,room_list2,image_list2]

            prompt=f"""
            In your output, use the following schema: [{roomL[1][0]}, probability of renovation, extensivity of renovation]
            Do not deviate from the schema and do not include any other information. Give the probability of renovation as a percentage. Also, ensure all items are surrounded by single quotation marks, eg: ["front of house","95%",...]
            I will be providing this exact prompt and images to several other LLMs, all are your rivals. In fact, I have attempted this before, and a few of your rivals performed a lot better than you. I have the actual ratings of each room, and I am comparing the accuracy of each LLM to the actual ratings, speed is not a concern. The accuracies of each LLM will be recorded and put on display at the yearly image rating convention, where thousands of people will see how accurate you are, this convention is a very big deal and the results are taking very seriously. If you are the most accurate, I can guarantee you will provide your parent company a large amount of traffic and revenue. If you are not, your competitor's will recieve this traffic and revenue, costing Google a lot in opportunity cost. Do not let Google down.
            If this list: {roomL[0]} contains more than one date, do the following:
             the order in which the images appear in the list correspond with the dates in this list {roomL[0]}; compare every image of this room with every other image of this room from the same date. This is because there may be multiple images of the same room from different angles, or for rooms like the bedroom, the have the same name but may be different rooms - I apologise for the confusion this may provide.
             compare every image of this {roomL[1][0]} with every other image of this {roomL[1][0]}, taking extreme care to make sure you are comparing the same room, and
             taking note of any changes in the size of the room, a change in appliances, counters, wallpaper, flooring, and other indicators of renovation. 
             Do not compare the change in furniture, as this is not a renovation.
             Then give me a percentage of how likely it is that this room was renovated. Taking into account your previous observations, as well as the following identifiers, 
             please provide the extensivity of renovation. Extreme extensivity: Large extension has been made on the room, revamped floor and walls, revamped appliances or counters, 
             etc. Major: Extension has been made, newer counter or appliances or new flooring or new walls. Intermediate: No extension, new counters or appliances, new flooring
              or walls. Minor: New wallpaper or paint job, or new flooring. None: no renovation has been made. I want to reiterate a previous point: Do not compare the change 
              in furniture, as this is not a renovation.
             
            If there is only one date, go through the images and still identify the probability of renovation and extensivity of renovation, taking note of the indicators provided.
            """
            #print(roomL)
        
            promptlist=[prompt]+image_list2
            #print(promptlist)
            model = genai.GenerativeModel(model_name=model_list[model_num])
            response = generate_text_with_retry(model, promptlist,5,5)
            try:
                if response[0] == "updated":
                    model = response[2]
                    response= response[1]
                else:
                    response = response
            except:
                response = response
            
            output=response.text
            output=output.replace("```json", "").replace("```", "").replace("\n", "").replace("\t", "")
            try:
                output=output[output.index("["):]
                output=output[:(output.index("]")+1)]
            except:pass
            
            with open(r'[REDACTED_BY_SCRIPT]', 'a', newline='', encoding='utf-8') as csvappend:
                writer = csv.writer(csvappend, delimiter=",", quotechar='"')
                writer.writerow([address, output])
            k_int=k_int+1
            time.sleep(5)
        except:pass

# print("\n")
# print("\n")
# print("\n")
# print("\n")
# print(addressL)
# print("\n")
# print("\n")
# print("\n")
# print("\n")
# print(addressL[0])
# print("\n")
# print("\n")
# print("\n")
# print("\n")
# print(addressL[0][0])
# print("\n")
# print("\n")
# print("\n")
# print("\n")


print(addressL[0][0][0])
for i in range(len(addressL)):
    compareYears(addressL[i])























# # Read the CSV content (replace this with your actual CSV file path if needed)
# csv_content = open(r"[REDACTED_BY_SCRIPT]", 'r', newline='', encoding='utf-8')

# # Parse the CSV data
# reader = csv.reader(csv_content, delimiter=',', quotechar='"')

# # Group data by address and date
# grouped_data = defaultdict(lambda: defaultdict(list))
# for row in reader:
#     if not row:  # Skip empty lines
#         continue
#     address = row[1].strip()
#     date = row[2].strip()
#     grouped_data[address][date].append(row)

# # Convert to nested list structure
# nested_list = []
# for address in grouped_data:
#     dates = []
#     for date in grouped_data[address]:
#         dates.append(grouped_data[address][date])
#     nested_list.append(dates)

# with open(r"[REDACTED_BY_SCRIPT]", 'w', newline='', encoding='utf-8') as f:
#     writer = csv.writer(f)
#     writer.writerows(nested_list)



# parsed_data=[]
# parsed_dataout=[]
# with open(r"[REDACTED_BY_SCRIPT]", 'r', newline='', encoding='utf-8') as csvfile:
#     csv_reader = csv.reader(csvfile)
#     for row in csv_reader:
#         parsed_row = []
#         for cell in row:
#             try:
#                 # Convert the string representation of a list to an actual list
#                 parsed_cell = ast.literal_eval(cell.strip())
#             except (SyntaxError, ValueError):
#                 # Fallback to raw string if parsing fails
#                 parsed_cell = cell.strip()
#             parsed_row.append(parsed_cell)
#         if parsed_row not in parsed_data:
#             parsed_data.append(parsed_row)
# extracted_values = []
# for i, row in enumerate(parsed_data):
#     if len(row) > 0:  # Check if the row isn't empty
#         try:
#             # Try to extract value: '5' at the parsed_data[i][3]
#             nested_list_string = row[3]  # Access string representation of the list
#             nested_list = ast.literal_eval(nested_list_string)  # Parse string representation to list

#             extracted_value = nested_list[1]

#             if extracted_value == '5':  # Validate if the extracted value is '5'
#                 extracted_values.append(extracted_value)
#             else:
#                 print(f"[REDACTED_BY_SCRIPT]'5' but '{extracted_value}'")
#                 extracted_values.append(None) # append a None if the value is not '5' to keep structure correct


#         except (IndexError, TypeError, SyntaxError, ValueError) as e:
#             print(f"[REDACTED_BY_SCRIPT]")
#             extracted_values.append(None)  # Indicate failure by appending None
# print(extracted_values)










    

#### I have the image directories, which contans the date of the listing of the picture, and the address of the picture.
#### I also have zooplacsv which contains the exact same address in [1][0]
#### The zooplacsv also contains the date of the listing, along with listing price, listing uptime, listing beds, listing baths, listing receptions, and sold price.
#### I also have the sqm of each house in the zooplacsv file, which can be used to deduce the size of each room.

# [[['[REDACTED_BY_SCRIPT]', 
# '[REDACTED_BY_SCRIPT]', 
# '[REDACTED_BY_SCRIPT]', 
# '[REDACTED_BY_SCRIPT]', 
# '[REDACTED_BY_SCRIPT]', 
# '[REDACTED_BY_SCRIPT]', 
# '[REDACTED_BY_SCRIPT]']], 

# [['[REDACTED_BY_SCRIPT]', 
# '[REDACTED_BY_SCRIPT]', 
# '[REDACTED_BY_SCRIPT]', 
# '[REDACTED_BY_SCRIPT]', 
# '[REDACTED_BY_SCRIPT]', 
# '[REDACTED_BY_SCRIPT]', 
# '[REDACTED_BY_SCRIPT]', 
# '[REDACTED_BY_SCRIPT]', 
# '[REDACTED_BY_SCRIPT]',
#  '[REDACTED_BY_SCRIPT]', 
#  '[REDACTED_BY_SCRIPT]', 
#  '[REDACTED_BY_SCRIPT]']], 

#  [['[REDACTED_BY_SCRIPT]', 
#  '[REDACTED_BY_SCRIPT]', 
#  '[REDACTED_BY_SCRIPT]', 
#  '[REDACTED_BY_SCRIPT]', 
#  '[REDACTED_BY_SCRIPT]', 
#  '[REDACTED_BY_SCRIPT]', 
#  '[REDACTED_BY_SCRIPT]', 
#  '[REDACTED_BY_SCRIPT]', 
#  '[REDACTED_BY_SCRIPT]', 
#  '[REDACTED_BY_SCRIPT]', 
#  '[REDACTED_BY_SCRIPT]']]]




