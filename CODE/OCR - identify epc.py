from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import pytesseract
import cv2
import numpy as np
import easyocr

finalepc1=""
finalepc2=""
for i in range(1):
    image_path = r'[REDACTED_BY_SCRIPT]'
    image = Image.open(image_path)
    width, height = image.size
    if width == 275 and height == 258:
        crop_box = (5 * width // 7, 0,width,9 * height // 10)
        image = image.crop(crop_box)
        
        image = image.resize((711, 1200), Image.Resampling.LANCZOS)
        image = image.filter(ImageFilter.SMOOTH)
        image = image.filter(ImageFilter.EDGE_ENHANCE)
        image = image.filter(ImageFilter.SHARPEN)
        image = image.convert("L")
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        dist = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY)[1]
        image = Image.fromarray(dist)
        

        image = image.filter(ImageFilter.SMOOTH)
        image.save(r'[REDACTED_BY_SCRIPT]')

        image_path = r'[REDACTED_BY_SCRIPT]'
        image = Image.open(image_path)
        
        width, height = image.size
        crop_box = (0, 1.1*height//6,2 * width // 5,height)
        image1 = image.crop(crop_box)
        image1.save(r'[REDACTED_BY_SCRIPT]')    
        crop_box = (width // 2, 1.1*height//6,width,height)
        image2 = image.crop(crop_box)
        image2.save(r'[REDACTED_BY_SCRIPT]')
    elif width == 838 and height == 546:
        crop_box = (4.5 * width // 7, 0,width,9 * height // 10)
        image = image.crop(crop_box)
        
        
        image = image.convert("L")
        image = image.filter(ImageFilter.SMOOTH)
        image = image.filter(ImageFilter.EDGE_ENHANCE)
        image = image.filter(ImageFilter.SHARPEN)
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        dist = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)[1]
        image = Image.fromarray(dist)
        image = image.resize((1191, 2000), Image.Resampling.LANCZOS)
        image = image.filter(ImageFilter.SMOOTH)
        image = image.filter(ImageFilter.EDGE_ENHANCE)

        image.save(r'[REDACTED_BY_SCRIPT]')
        image_path = r'[REDACTED_BY_SCRIPT]'
        image = Image.open(image_path)
        
        width, height = image.size
        crop_box = (0, height//6,2 * width // 5,5*height//6)
        image1 = image.crop(crop_box)
        image1.save(r'[REDACTED_BY_SCRIPT]')    
        crop_box = (width // 2, height//6,width,5*height//6)
        image2 = image.crop(crop_box)
        image2.save(r'[REDACTED_BY_SCRIPT]')
    elif width == 614 and height == 371:
        crop_box = (4.5 * width // 7, 0,width,9 * height // 10)
        image = image.crop(crop_box)
        image = image.resize((237, 464), Image.Resampling.LANCZOS)

        image = image.convert("L")
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        dist = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)[1]
        image = Image.fromarray(dist)

        image = image.filter(ImageFilter.SMOOTH)
        image = image.filter(ImageFilter.EDGE_ENHANCE)
        image = image.filter(ImageFilter.SHARPEN)
        image.save(r'[REDACTED_BY_SCRIPT]')

        image_path = r'[REDACTED_BY_SCRIPT]'
        image = Image.open(image_path)
        
        image = image.resize((237, 464), Image.Resampling.LANCZOS)
        width, height = image.size
        crop_box = (0, height//6,2 * width // 5,height)
        image1 = image.crop(crop_box)
        image1.save(r'[REDACTED_BY_SCRIPT]')    
        crop_box = (width // 2, height//6,width,height)
        image2 = image.crop(crop_box)
        image2.save(r'[REDACTED_BY_SCRIPT]')
    
    options = f'[REDACTED_BY_SCRIPT]'
    extracted_text1 = pytesseract.image_to_string(image1, config=options)
    extracted_text1=extracted_text1.replace(" ","").replace("\n","")
    extracted_text2 = pytesseract.image_to_string(image2, config=options)
    extracted_text2=extracted_text2.replace(" ","").replace("\n","")
    if extracted_text1!="" and finalepc1=="":
        finalepc1=extracted_text1
        image1.save(r'[REDACTED_BY_SCRIPT]')
        print("finalepc1",finalepc1)
    if extracted_text2!="" and finalepc2=="":
        finalepc2=extracted_text2
        image2.save(r'[REDACTED_BY_SCRIPT]')
        print("finalepc2",finalepc2)
