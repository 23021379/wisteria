"""[REDACTED_BY_SCRIPT]"""

import cv2
import numpy as np
from skimage import io, restoration
from skimage.color import rgb2gray, rgba2rgb

path = r"[REDACTED_BY_SCRIPT]"
image = cv2.imread(f'{path}')
height, width, channels = image.shape
# Apply Non-Local Means Denoising
denoised_image = cv2.fastNlMeansDenoisingColored(image, None, 2, 2, 1, 3)
smoothed = cv2.bilateralFilter(denoised_image, d=3, sigmaColor=25, sigmaSpace=25)
resized = cv2.resize(image, None, fx=(2048/width), fy=(1536/height), interpolation=cv2.INTER_CUBIC)
denoised_image = cv2.fastNlMeansDenoisingColored(resized, None, 4, 4, 1, 6)

blurred = cv2.GaussianBlur(denoised_image, (0, 0), 3)
sharpened = cv2.addWeighted(denoised_image, 1.1, blurred, -0.1, 0)

# Save the denoised imagedenoised_image
cv2.imwrite(f'{path}2.jpeg', sharpened)



image = io.imread(f'{path}2.jpeg')

# If the image has an alpha channel (RGBA), convert it to RGB
if image.shape[2] == 4:
    image = rgba2rgb(image)

red_channel = image[:, :, 0]
green_channel = image[:, :, 1]
blue_channel = image[:, :, 2]

# Denoise each channel independently
red_denoised = restoration.denoise_wavelet(red_channel, mode='soft', wavelet='db1', rescale_sigma=True)
green_denoised = restoration.denoise_wavelet(green_channel, mode='soft', wavelet='db1', rescale_sigma=True)
blue_denoised = restoration.denoise_wavelet(blue_channel, mode='soft', wavelet='db1', rescale_sigma=True)

# Stack the denoised channels back into a single image
denoised_image = np.stack([red_denoised, green_denoised, blue_denoised], axis=-1)

# Save the denoised image
io.imsave(f'{path}3.jpeg', (denoised_image * 255).astype(np.uint8))




##########################################################
#PICSART API
# import requests
# import json
# image_path = r"[REDACTED_BY_SCRIPT]"
# image_path.replace(" ","%20")

# url = "[REDACTED_BY_SCRIPT]"

# files = { "image": (image_path, open(image_path, "rb"), "image/jpeg") }
# payload = {
#     "upscale_factor": "2",
#     "format": "JPG"
# }
# headers = {
#     "accept": "application/json",
#     "X-Picsart-API-Key": "[REDACTED_BY_SCRIPT]"
# }

# response = requests.post(url, data=payload, files=files, headers=headers)
# outputurl=response.text
# outputurl = json.loads(outputurl)
# outputurl=outputurl["data"]["url"]
# print(outputurl)

# response = requests.get(outputurl)
# if response.status_code == 200:
#     with open((image_path[:-4]+"2.jpg"), 'wb') as file:
#         file.write(response.content)
#     print("[REDACTED_BY_SCRIPT]")
# else:
#     print(f"[REDACTED_BY_SCRIPT]")