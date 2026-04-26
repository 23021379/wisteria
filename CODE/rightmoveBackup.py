import ast
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import time
import csv
import undetected_chromedriver as uc
import math
import random
import time
from datetime import datetime
import Levenshtein
import mouse
import base64
import requests
import os, os.path
from os import listdir
from os.path import isfile, join
import pyscreenshot
from PIL import Image
import cv2
import numpy as np
import pandas as pd
import asyncio
import argparse
import logging
import shutil
import urllib
import multiprocessing

def parse_date(date_str):
    return datetime.strptime(date_str, "%B %Y")

def parse_date2(date_str):
    return datetime.strptime(date_str, "%b %Y")

def calculate_time_diff(sold_date, listed_date):
    delta = sold_date - listed_date
    months = delta.days // 30  # Approximate months
    years, months = divmod(months, 12)
    
    if years > 0 and months > 0:
        return f"[REDACTED_BY_SCRIPT]"
    elif years > 0:
        return f"{years} years"
    else:
        return f"{months} months"

async def scrape_with_timeout(timeout_seconds=420):  # 5 minutes = 300 seconds
    global driver
    global row
    global iteraCount
    global loopcount
    input_file = r"[REDACTED_BY_SCRIPT]"
    output_file = r"[REDACTED_BY_SCRIPT]"
    parsed_data = []
    parsed_dataout=[]
    with open(input_file, 'r', newline='', encoding='utf-8') as csvfile:
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
    iteraCount=0
    for row in parsed_data[4386:]:
        completedScrape = False
        retryattempt=0
        while completedScrape == False:
            if not row:  # Skip empty rows
                continue

            addressa=row[3].replace(",","")
            addressa=addressa.lower()
            address=row[0]
            price=row[1]
            date=row[2]
            if not driver:
                time.sleep(1)
                driver = initialize_driver()
            try:
                await asyncio.wait_for(asyncio.to_thread(scrape_page, addressa, address,price,date,output_file), timeout=timeout_seconds)
                completedScrape = True
            except asyncio.TimeoutError:
                print("[REDACTED_BY_SCRIPT]")
                try:
                    driver.quit()  # Quit the current driver
                    loopcount=0
                except:
                    pass
            except:
                try:
                    driver.quit()  # Quit the current driver
                    loopcount=0
                except:
                    pass
                if retryattempt < 3:
                    retryattempt += 1
                else:
                    completedScrape = True
                pass

def initialize_driver(): #Removed pathing arguments
    global driver
    global loopcount
    global iteraCount
    options = uc.ChromeOptions()
    # Add your Chrome options here
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    #Do not copy, since the environment has been set up and we are trusting that the environment variable UC_CHROMEDRIVER_PATH, is valid, with a valid chromedriver

    driver = uc.Chrome(options=options)
    return driver

loopcount=0
def scrape_page(addressa, address,price,date,output_file):
    global driver
    global loopcount
    global iteraCount
    """
    Your scraping logic goes here.  This function should be synchronous.
    """
    try:    
        try:
            if loopcount==0:
                driver.get("[REDACTED_BY_SCRIPT]")
        except:
            driver.quit()
            loopcount=0
            time.sleep(1)
            driver = initialize_driver() #Removed pathing arguments
            time.sleep(1)
            driver.get("[REDACTED_BY_SCRIPT]")

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
        #         if imagecomp == "49.png" and max_val > .9:
        #             match = True
        #             break
        #     time.sleep(screenshot_interval)

        timeout = 15

        #time.sleep(1000)
        if loopcount==0:
            ####clicks cookie thing
            #accept
            try:
                wait = WebDriverWait(driver, 15)
                element_locator = (By.CSS_SELECTOR, '[REDACTED_BY_SCRIPT]')
                element = wait.until(EC.presence_of_element_located(element_locator))
                driver.find_element(By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]").click()
            except:
                pass
            loopcount+=1
                
        element_present = EC.presence_of_element_located((By.CSS_SELECTOR, '[REDACTED_BY_SCRIPT]'))
        WebDriverWait(driver, timeout).until(element_present)
        driver.get(addressa)

        wait = WebDriverWait(driver, 15)
        element_locator = (By.CSS_SELECTOR, '[REDACTED_BY_SCRIPT]')
        wait.until(EC.presence_of_element_located(element_locator))
        housetype=driver.find_element(By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]").text
        housetype=housetype[housetype.index("\n"):]
        try:
            beds=driver.find_element(By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]").text
            beds=beds[beds.index("\n"):]
        except:beds=""
        try:
            baths=driver.find_element(By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]").text
            baths=baths[baths.index("\n"):]
        except:baths=""
        #housetype: #root > main > div > div._3Un-8ZyFYcWASK6819Y3R9 > div > p
        #beds: #root > main > div > div._3Un-8ZyFYcWASK6819Y3R9 > div:nth-child(2) > p
        #baths: #root > main > div > div._3Un-8ZyFYcWASK6819Y3R9 > div:nth-child(3) > p
        
        prevsold=["","","","","","","","","",""]
        for j in range(5):
            try:                                                         #root > main > section._34ugrcwpFfcPT_HVb-Z87F > div:nth-child(1) > div.gTIxzFvFbIU616ZHdtWkZ > div > div > div:nth-child(2) > table > tbody > tr:nth-child(2)
                prevsold[(0*j)+j]=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
            except:
                prevsold[(0*j)+j]=""
            try:
                prevsold[(0*j)+1]=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
            except:
                prevsold[(0*j)+1]=""
        #prev sold date1: #root > main > section._34ugrcwpFfcPT_HVb-Z87F > div:nth-child(1) > div.gTIxzFvFbIU616ZHdtWkZ > div > div > div:nth-child(2) > table > tbody > tr:nth-child(2) > td._2Dz8cX76Q51EJE_1aidJrI
        #prev sold price1:#root > main > section._34ugrcwpFfcPT_HVb-Z87F > div:nth-child(1) > div.gTIxzFvFbIU616ZHdtWkZ > div > div > div:nth-child(2) > table > tbody > tr:nth-child(2) > td:nth-child(3)
        #prev sold date2: #root > main > section._34ugrcwpFfcPT_HVb-Z87F > div:nth-child(1) > div.gTIxzFvFbIU616ZHdtWkZ > div > div > div:nth-child(2) > table > tbody > tr:nth-child(3) > td._2Dz8cX76Q51EJE_1aidJrI
        #prev sold price2:#root > main > section._34ugrcwpFfcPT_HVb-Z87F > div:nth-child(1) > div.gTIxzFvFbIU616ZHdtWkZ > div > div > div:nth-child(2) > table > tbody > tr:nth-child(3) > td:nth-child(3)
        hawktuah=address.replace(",","").lower()
        for j in range(50):
            try:
                imgsrc = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").get_attribute("src")                
                response = requests.get(imgsrc)
                if response.status_code == 200:
                    newpath = f'[REDACTED_BY_SCRIPT]' 
                    if not os.path.exists(newpath):
                        os.makedirs(newpath)
                    with open(f"[REDACTED_BY_SCRIPT]", "wb") as file:
                        file.write(response.content)
                
            except:pass
        #img1: #root > main > section._34ugrcwpFfcPT_HVb-Z87F > div.bYoLoaXRRVXBkrr3chN9W.PsCHfl-PfPOMY6X0wKv2g > div._3tSzCBUhsBPrgIVWAZJwQU > div > div > div:nth-child(1) > a > div > img
        #img2: #root > main > section._34ugrcwpFfcPT_HVb-Z87F > div.bYoLoaXRRVXBkrr3chN9W.PsCHfl-PfPOMY6X0wKv2g > div._3tSzCBUhsBPrgIVWAZJwQU > div > div > div:nth-child(2) > a > div > img
        #img3: #root > main > section._34ugrcwpFfcPT_HVb-Z87F > div.bYoLoaXRRVXBkrr3chN9W.PsCHfl-PfPOMY6X0wKv2g > div._3tSzCBUhsBPrgIVWAZJwQU > div > div > div:nth-child(3) > a > div > img
        #img4: #root > main > section._34ugrcwpFfcPT_HVb-Z87F > div.bYoLoaXRRVXBkrr3chN9W.PsCHfl-PfPOMY6X0wKv2g > div._3tSzCBUhsBPrgIVWAZJwQU > div > div > div:nth-child(4) > a > div > img
        #img5: #root > main > section._34ugrcwpFfcPT_HVb-Z87F > div.bYoLoaXRRVXBkrr3chN9W.PsCHfl-PfPOMY6X0wKv2g > div._3tSzCBUhsBPrgIVWAZJwQU > div > div > div:nth-child(5) > a > div > img
        #img6: #root > main > section._34ugrcwpFfcPT_HVb-Z87F > div.bYoLoaXRRVXBkrr3chN9W.PsCHfl-PfPOMY6X0wKv2g > div._3tSzCBUhsBPrgIVWAZJwQU > div > div > div:nth-child(6) > a > div > img
        #img7: #root > main > section._34ugrcwpFfcPT_HVb-Z87F > div.bYoLoaXRRVXBkrr3chN9W.PsCHfl-PfPOMY6X0wKv2g > div._3tSzCBUhsBPrgIVWAZJwQU > div > div > div:nth-child(7) > a > div > img
        #img8: #root > main > section._34ugrcwpFfcPT_HVb-Z87F > div.bYoLoaXRRVXBkrr3chN9W.PsCHfl-PfPOMY6X0wKv2g > div._3tSzCBUhsBPrgIVWAZJwQU > div > div > div:nth-child(8) > a > div > img
        hasPics=False
        try:
            floorplansrc=driver.find_element(By.CSS_SELECTOR,f"[REDACTED_BY_SCRIPT]").get_attribute("href")
            imagessrc=driver.find_element(By.CSS_SELECTOR,f"[REDACTED_BY_SCRIPT]").get_attribute("href")
            driver.get(floorplansrc)
            wait = WebDriverWait(driver, 15)
            element_locator = (By.CSS_SELECTOR, '[REDACTED_BY_SCRIPT]')
            wait.until(EC.presence_of_element_located(element_locator))
            floorplansrc=driver.find_element(By.CSS_SELECTOR,f"[REDACTED_BY_SCRIPT]").get_attribute("src")
            response = requests.get(floorplansrc)
            hawktuah=address.replace(",","").lower()
            if response.status_code == 200:
                newpath = f'[REDACTED_BY_SCRIPT]' 
                if not os.path.exists(newpath):
                    os.makedirs(newpath)
                with open(f"[REDACTED_BY_SCRIPT]", "wb") as file:
                    file.write(response.content)
            driver.get(imagessrc)
            wait = WebDriverWait(driver, 15)
            element_locator = (By.CSS_SELECTOR, '#media0 > img')
            wait.until(EC.presence_of_element_located(element_locator))
            for j in range(50):
                try:
                    imgsrc = driver.find_element(By.CSS_SELECTOR, f"#media{j} > img").get_attribute("src")                
                    response = requests.get(imgsrc)
                    if response.status_code == 200:
                        newpath = f'[REDACTED_BY_SCRIPT]' 
                        if not os.path.exists(newpath):
                            os.makedirs(newpath)
                        with open(f"[REDACTED_BY_SCRIPT]", "wb") as file:
                            file.write(response.content)
                except:pass
            hasPics=True
        except:
            try:
                imagesrc=driver.find_element(By.CSS_SELECTOR,f"[REDACTED_BY_SCRIPT]").get_attribute("href")
                driver.get(imagesrc)
                wait = WebDriverWait(driver, 15)
                element_locator = (By.CSS_SELECTOR, '#media0 > img')
                wait.until(EC.presence_of_element_located(element_locator))
                for j in range(50):
                    try:
                        imgsrc = driver.find_element(By.CSS_SELECTOR, f"#media{j} > img").get_attribute("src")                
                        response = requests.get(imgsrc)
                        if response.status_code == 200:
                            newpath = f'[REDACTED_BY_SCRIPT]' 
                            if not os.path.exists(newpath):
                                os.makedirs(newpath)
                            with open(f"[REDACTED_BY_SCRIPT]", "wb") as file:
                                file.write(response.content)
                    except:pass
                hasPics=True
            except:
                hasPics=False
                pass
        #floorplan href with images: #root > main > section._34ugrcwpFfcPT_HVb-Z87F > div.bYoLoaXRRVXBkrr3chN9W.PsCHfl-PfPOMY6X0wKv2g > div.G2aWqvf4JahNCQ0g_GJ4K > a:nth-child(3)
        #pics href with floorplan: #root > main > section._34ugrcwpFfcPT_HVb-Z87F > div.bYoLoaXRRVXBkrr3chN9W.PsCHfl-PfPOMY6X0wKv2g > div.G2aWqvf4JahNCQ0g_GJ4K > a:nth-child(1)
        #load href, then floorplan: #root > div > div._2fiMlyBpvAqYR2h4Rui7LI > div._3MkGY7CXNh8RgOGezWOOSz > div > div > div > div.react-transform-wrapper.transform-component-module_wrapper__SPB86 > div > img

        #pics href no floorplan: #root > main > section._34ugrcwpFfcPT_HVb-Z87F > div.bYoLoaXRRVXBkrr3chN9W.PsCHfl-PfPOMY6X0wKv2g > div.G2aWqvf4JahNCQ0g_GJ4K > a
        if hasPics == True:
            with open(output_file, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile, delimiter=",", quotechar='"')
                writer.writerow([[address.replace(",","").lower(),addressa],[address,housetype,beds,baths],prevsold])
        
        if iteraCount == 7:
            try:driver.quit()
            except:pass
            loopcount=0
        else: iteraCount+=1
    except Exception as e:
        try:
            driver.quit()
            loopcount=0
        except:
            pass
        print(f"[REDACTED_BY_SCRIPT]")
        return False


async def main(): #Removed argument and default
    global driver
    global row
    global iteraCount
    """[REDACTED_BY_SCRIPT]"""
    driver = initialize_driver() #removed argument

    await scrape_with_timeout(420) #Removed driver and argument

    driver.quit()  # Ensure the driver is closed after scraping
    
# logging.basicConfig(level=logging.INFO, encoding='utf-8',
#                     format='[REDACTED_BY_SCRIPT]')

if __name__ == "__main__":
    # chromedriver_lock = multiprocessing.Lock()
    # parser = argparse.ArgumentParser(description="Selenium scraper")
    # parser.add_argument("--input", required=True, help="[REDACTED_BY_SCRIPT]")
    # parser.add_argument("--output", required=True, help="Output CSV file")
    # args = parser.parse_args()
    # args.input, args.output
    asyncio.run(main()) #Remove path argument and default