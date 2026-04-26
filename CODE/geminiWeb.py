from google import genai
from google.genai import types
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch

client = genai.Client(api_key="[REDACTED_BY_SCRIPT]")

response = client.models.generate_content(
    model='gemini-2.0-flash',
    contents="[REDACTED_BY_SCRIPT]'3 Sandhoe Walk, Wallsend, NE28 6JL'[REDACTED_BY_SCRIPT]'https://www.bricksandlogic.co.uk/'[REDACTED_BY_SCRIPT]",
    config=types.GenerateContentConfig(
        tools=[types.Tool(google_search=types.GoogleSearchRetrieval(dynamic_retrieval_config=types.DynamicRetrievalConfig(dynamic_threshold=0)))],
        response_modalities=["TEXT"]),
)
for each in response.candidates[0].content.parts:
    print(each.text)





# from tavily import TavilyClient
# import time
# tavily_client = TavilyClient(api_key="[REDACTED_BY_SCRIPT]")
# response = tavily_client.search(
#     query="[REDACTED_BY_SCRIPT]'3 Sandhoe Walk, Wallsend, NE28 6JL'[REDACTED_BY_SCRIPT]",
#     search_depth="advanced",
#     include_answer="basic",
#     include_raw_content=True,
#     max_results=3,
#     include_domains=["[REDACTED_BY_SCRIPT]"]
# )
# print(response['results'][0]['url'])
# try:
#     print(response['results'][1]['url'])
# except:
#     pass
# try:
#     print(response['results'][2]['url'])
# except:
#     pass

# url = response['results'][0]['url']




# url="[REDACTED_BY_SCRIPT]"
# import undetected_chromedriver as uc
# from selenium import webdriver
# from selenium.webdriver.common.by import By
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
# from selenium.common.exceptions import TimeoutException
# import time
# import pyscreenshot
# from PIL import Image
# import cv2
# import numpy as np
# import mouse
# import os, os.path
# from os import listdir
# from os.path import isfile, join
# import math
# import random


# options = uc.ChromeOptions()
# options.add_argument('--no-sandbox')
# options.add_argument('--disable-extensions')
# options.add_argument('--disable-gpu')


# driver = uc.Chrome(options=options)

# driver.get(url)
# screenshot_interval = 0.05 
# match=False
# while match == False:
#     screenshot = pyscreenshot.grab()
#     screenshot = pyscreenshot.grab(bbox=(513, 351, 816, 416))
#     filename = f"[REDACTED_BY_SCRIPT]"
#     screenshot.save(filename)
    
#     onlyfiles = [f for f in listdir("[REDACTED_BY_SCRIPT]") if isfile(join("[REDACTED_BY_SCRIPT]", f))]
#     for imagecomp in onlyfiles:
#         compare = cv2.imread(filename)
#         compare2 = cv2.imread("[REDACTED_BY_SCRIPT]"+imagecomp)
#         result = cv2.matchTemplate(compare, compare2, cv2.TM_CCOEFF_NORMED)
#         _, max_val, _, max_loc = cv2.minMaxLoc(result)
#         if imagecomp == "2.png" and max_val > 0.98:
#             x0,y0=mouse.get_position() 
#             x1,y1=541, 383
#             xdiff=x1-x0
#             ydiff=y1-y0
#             for moveseg in range(1,50):
#                 xmove = x0 + moveseg*(xdiff/50)
#                 ymove = y0 + moveseg*(ydiff/50)
#                 movetime=0.4*math.sin(5*(xmove**3)+xmove**4-xmove**5)+.5
#                 movetime=movetime/random.uniform(25,50)
#                 mouse.move(xmove,ymove,absolute=True, duration=movetime)
#             mouse.click()
                
#             x0,y0=mouse.get_position() 
#             x1,y1 = random.uniform(200,800), random.uniform(200,600)
#             xdiff=x1-x0
#             ydiff=y1-y0
#             for moveseg in range(1,50):
#                 xmove = x0 + moveseg*(xdiff/50)
#                 ymove = y0 + moveseg*(ydiff/50)
#                 movetime=0.4*math.sin(5*(xmove**3)+xmove**4-xmove**5)+.5
#                 movetime=movetime/random.uniform(25,50)
#                 mouse.move(xmove,ymove,absolute=True, duration=movetime)
#         if imagecomp == "44.png" and max_val > .9 or imagecomp == "46.png" and max_val > .7 or imagecomp == "47.png" and max_val > .7 or imagecomp == "48.png" and max_val > .7:
#             match = True
#             break
#     time.sleep(screenshot_interval)
# driver.implicitly_wait(2)
# html_content = driver.page_source

# # Print the HTML content
# print(html_content)
# for i in range(8):
#     print("\n")
# image_elements = driver.find_elements(By.TAG_NAME, 'img')

# # Extract and print the 'src' attributes
# for img in image_elements:
#     img_src = img.get_attribute('src')
#     if img_src:  # Check if src attribute exists
#         print(img_src)




















#THIS IS THE ORIGINAL CODE FOR THE IMGEN API 
######################################################################
######################################################################
######################################################################
######################################################################
# import requests
# import base64
# import cv2
# import numpy as np
# from skimage import io, restoration
# from skimage.color import rgb2gray, rgba2rgb
# url = '[REDACTED_BY_SCRIPT]'
# api_key = '[REDACTED_BY_SCRIPT]' 
# image_path = r'[REDACTED_BY_SCRIPT]' 
# try:
#     with open(image_path, 'rb') as image_file:
#         files = {'image': image_file}
#         headers = {'X-IMGGEN-KEY': api_key}

#         response = requests.post(url, headers=headers, files=files)
#         response.raise_for_status()

#         if 'application/json' in response.headers.get('Content-Type', ''):
#             try:
#                 json_response = response.json()
#                 print("JSON Response:")
#                 output_image_path = r'[REDACTED_BY_SCRIPT]'
#                 image_bytes = base64.b64decode(json_response['image'])
#                 with open(output_image_path, 'wb') as f:
#                     f.write(image_bytes)
#             except requests.exceptions.JSONDecodeError:
#                 print("[REDACTED_BY_SCRIPT]", response.text)
# except:pass

# path = output_image_path
# image = cv2.imread(path)
# height, width, channels = image.shape
# resized = cv2.resize(image, None, fx=(1024/width), fy=(768/height), interpolation=cv2.INTER_CUBIC)
# cv2.imwrite(f'{path[:-4]}2.jpeg', resized)














