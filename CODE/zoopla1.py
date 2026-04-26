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
import ast
import traceback

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

async def scrape_with_timeout(addressa, timeout_seconds=300):  # 5 minutes = 300 seconds
    global driver
    global row
    parsed_data = []
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
    
    for row in parsed_data:
        completedScrape = False
        retryattempt=0
        while completedScrape == False:
            if not row:  # Skip empty rows
                continue
            print(row)
            addressa=row[3]
            print(addressa)
            address=row[0]
            price=row[1]
            date=row[2]
            if not driver:
                time.sleep(1)
                driver = initialize_driver()
            try:
                await asyncio.wait_for(asyncio.to_thread(scrape_page, addressa,address), timeout=timeout_seconds)
                completedScrape = True
            except asyncio.TimeoutError:
                print("[REDACTED_BY_SCRIPT]")
                try:
                    driver.quit()  # Quit the current driver
                except:
                    pass
            except:
                try:
                    driver.quit()  # Quit the current driver
                except:
                    pass
                if retryattempt < 3:
                    retryattempt += 1
                else:
                    completedScrape = True
                pass

def initialize_driver():
    global driver
    global row
    """[REDACTED_BY_SCRIPT]"""
    options = uc.ChromeOptions()
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.page_load_strategy = 'eager'
    time.sleep(1)
    driver = uc.Chrome(options=options)
    return driver

loopcount=0
def scrape_page(addressa,address):
    global driver
    global row
    """
    Your scraping logic goes here.  This function should be synchronous.
    """
    try:    
        try:
            driver.get("[REDACTED_BY_SCRIPT]")
        except:
            driver.quit()
            time.sleep(1)
            driver = initialize_driver()
            time.sleep(1)
            driver.get("[REDACTED_BY_SCRIPT]")

        screenshot_interval = 0.05 
        match=False
        while match == False:
            screenshot = pyscreenshot.grab()
            screenshot = pyscreenshot.grab(bbox=(513, 351, 816, 416))
            filename = f"[REDACTED_BY_SCRIPT]"
            screenshot.save(filename)
            
            onlyfiles = [f for f in listdir("[REDACTED_BY_SCRIPT]") if isfile(join("[REDACTED_BY_SCRIPT]", f))]
            for imagecomp in onlyfiles:
                compare = cv2.imread(filename)
                compare2 = cv2.imread("[REDACTED_BY_SCRIPT]"+imagecomp)
                result = cv2.matchTemplate(compare, compare2, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)
                if imagecomp == "2.png" and max_val > 0.98:
                    x0,y0=mouse.get_position() 
                    x1,y1=541, 383
                    xdiff=x1-x0
                    ydiff=y1-y0
                    for moveseg in range(1,50):
                        xmove = x0 + moveseg*(xdiff/50)
                        ymove = y0 + moveseg*(ydiff/50)
                        movetime=0.4*math.sin(5*(xmove**3)+xmove**4-xmove**5)+.5
                        movetime=movetime/random.uniform(25,50)
                        mouse.move(xmove,ymove,absolute=True, duration=movetime)
                    mouse.click()
                        
                    x0,y0=mouse.get_position() 
                    x1,y1 = random.uniform(200,800), random.uniform(200,600)
                    xdiff=x1-x0
                    ydiff=y1-y0
                    for moveseg in range(1,50):
                        xmove = x0 + moveseg*(xdiff/50)
                        ymove = y0 + moveseg*(ydiff/50)
                        movetime=0.4*math.sin(5*(xmove**3)+xmove**4-xmove**5)+.5
                        movetime=movetime/random.uniform(25,50)
                        mouse.move(xmove,ymove,absolute=True, duration=movetime)
                if imagecomp == "44.png" and max_val > .9:
                    match = True
                    break
            time.sleep(screenshot_interval)

        timeout = 45

        # time.sleep(10)
        if loopcount==0:
            wait = WebDriverWait(driver, 15)
            try:
                #usercentrics-cmp-ui
                element_locator = (By.CSS_SELECTOR, '#usercentrics-cmp-ui')
                element = wait.until(EC.presence_of_element_located(element_locator))
                shadow_host = driver.find_element(By.CSS_SELECTOR, "#usercentrics-cmp-ui")
                shadow_root = driver.execute_script("[REDACTED_BY_SCRIPT]", shadow_host)
                #accept
                wait = WebDriverWait(shadow_root, 15)
                element_locator = (By.CSS_SELECTOR, '#accept')
                element = wait.until(EC.presence_of_element_located(element_locator))
                shadow_root.find_element(By.CSS_SELECTOR, "#accept").click()
            except:
                element_locator = (By.CSS_SELECTOR, '#usercentrics-root')
                element = wait.until(EC.presence_of_element_located(element_locator))
                shadow_host = driver.find_element(By.CSS_SELECTOR, "#usercentrics-root")
                shadow_root = driver.execute_script("[REDACTED_BY_SCRIPT]", shadow_host)
                wait = WebDriverWait(shadow_root, 15)
                element_locator = (By.CSS_SELECTOR, '[REDACTED_BY_SCRIPT]')
                element = wait.until(EC.presence_of_element_located(element_locator))
                shadow_root.find_element(By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]").click()
                

            
            try:
                #__next > div > div._18rbmd81 > div > header > div > div > div > nav > div:nth-child(3) > div._14bi3x32i._14bi3x32d._14bi3x31f > ul > li:nth-child(3) > span > a > span > span._1womgj9e._1womgj9g
                wait = WebDriverWait(driver, 5)
                element_locator = (By.CSS_SELECTOR, '[REDACTED_BY_SCRIPT]')
                element = wait.until(EC.presence_of_element_located(element_locator))
                driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").click()

            except:
                wait = WebDriverWait(driver, 5)
                element_locator = (By.CSS_SELECTOR, '[REDACTED_BY_SCRIPT]')
                element = wait.until(EC.presence_of_element_located(element_locator))
                driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").click()

            
            try:
                element_present = EC.presence_of_element_located((By.CSS_SELECTOR, '[REDACTED_BY_SCRIPT]'))
                WebDriverWait(driver, 15).until(element_present)
                driver.find_element(By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]").click()
                driver.find_element(By.CSS_SELECTOR, "#email").send_keys("[REDACTED_BY_SCRIPT]")
                driver.find_element(By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]").click()
                driver.find_element(By.CSS_SELECTOR, "#current-password").send_keys("Password1!")
                driver.find_element(By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]").click()
                #signin #__next > div > div._18rbmd81 > div > header > div > div > div > nav > div:nth-child(3) > div._14bi3x32h._14bi3x32c._14bi3x31e > ul > li:nth-child(3) > span > a
                #signin2 #main-content > div > div > form > div:nth-child(2) > div > div._1i23ci69 > div._14bi3x3v._14bi3x329._14bi3x323 > div
                #signin3 #main-content > div > div > form > div:nth-child(3) > div > div._1i23ci69 > div > div._1i23ci66._194zg6t8
                #signin4 #main-content > div > div > form > div:nth-child(6) > button
            except:
                element_present = EC.presence_of_element_located((By.CSS_SELECTOR, '[REDACTED_BY_SCRIPT]'))
                WebDriverWait(driver, 15).until(element_present)
                driver.find_element(By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]").click()
                driver.find_element(By.CSS_SELECTOR, "#email").send_keys("[REDACTED_BY_SCRIPT]")
                driver.find_element(By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]").click()
                driver.find_element(By.CSS_SELECTOR, "#current-password").send_keys("Password1!")
                driver.find_element(By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]").click()
                #main-content > div > div > form > div:nth-child(2) > div > div._1i23ci69 > div > div
                #main-content > div > div > form > div:nth-child(3) > div > div._1i23ci69 > div > div._1i23ci66._194zg6t8

            
        time.sleep(5)
        driver.get(addressa)

        house=[[],[],[],[],"",""]
        house[0]=[address.replace(",","").lower()]
        print(house)
        try:
            element_present = EC.presence_of_element_located((By.XPATH, '/html/body/div[4]/div/div[2]/main/div[1]/div/div/div[1]/div/div[2]/div[2]/div/div/div/h1'))
            WebDriverWait(driver, timeout).until(element_present)
            address = driver.find_element(By.XPATH, "/html/body/div[4]/div/div[2]/main/div[1]/div/div/div[1]/div/div[2]/div[2]/div/div/div/h1").text
            #                                        /html/body/div[4]/div/div[2]/main/div[1]/div/div/div[1]/div/div/div[2]/div/div/div/h1
            houseType = driver.find_element(By.XPATH, "/html/body/div[4]/div/div[2]/main/div[1]/div/div/div[1]/div/div[2]/div[3]/div/div[1]/div/p").text
            #                                          /html/body/div[4]/div/div[2]/main/div[1]/div/div/div[1]/div/div/div[3]/div/div/div/p
            try:
                beds = driver.find_element(By.XPATH, "[REDACTED_BY_SCRIPT]").text
            except:
                beds = ""
            try:
                baths = driver.find_element(By.XPATH, "[REDACTED_BY_SCRIPT]").text
            except:
                baths = ""
            try:
                receps = driver.find_element(By.XPATH, "[REDACTED_BY_SCRIPT]").text
            except:
                receps = ""
            
            try:
                ownership = driver.find_element(By.XPATH, "/html/body/div[4]/div/div[2]/main/div[1]/div/div/div[1]/div/div[2]/div[4]/div/div[1]/div").text
                sqm = driver.find_element(By.XPATH, "/html/body/div[4]/div/div[2]/main/div[1]/div/div/div[1]/div/div[2]/div[4]/div/div[2]/div").text
                epc = driver.find_element(By.XPATH, "/html/body/div[4]/div/div[2]/main/div[1]/div/div/div[1]/div/div[2]/div[4]/div/div[3]/div").text
            except:
                ownership = ""
                sqm = ""
                epc = ""
        except:
            element_present = EC.presence_of_element_located((By.XPATH, '/html/body/div[4]/div/div[2]/main/div[1]/div/div/div[1]/div/div/div[2]/div/div/div/h1'))
            WebDriverWait(driver, timeout).until(element_present)
            address = driver.find_element(By.XPATH, "/html/body/div[4]/div/div[2]/main/div[1]/div/div/div[1]/div/div/div[2]/div/div/div/h1").text
            houseType = driver.find_element(By.XPATH, "/html/body/div[4]/div/div[2]/main/div[1]/div/div/div[1]/div/div/div[3]/div/div[1]/div/p").text
            try:
                beds = driver.find_element(By.XPATH, "/html/body/div[4]/div/div[2]/main/div[1]/div/div/div[1]/div/div/div[3]/div/div[2]/p").text
            except:
                beds = ""
            try:
                baths = driver.find_element(By.XPATH, "/html/body/div[4]/div/div[2]/main/div[1]/div/div/div[1]/div/div/div[3]/div/div[3]/p").text
            except:
                baths = ""
            try:
                receps = driver.find_element(By.XPATH, "/html/body/div[4]/div/div[2]/main/div[1]/div/div/div[1]/div/div/div[3]/div/div[4]/p").text
            except:
                receps = ""
            
            try:
                ownership = driver.find_element(By.XPATH, "/html/body/div[4]/div/div[2]/main/div[1]/div/div/div[1]/div/div/div[4]/div/div[1]/div").text
                sqm = driver.find_element(By.XPATH, "/html/body/div[4]/div/div[2]/main/div[1]/div/div/div[1]/div/div/div[4]/div/div[2]/div").text
                epc = driver.find_element(By.XPATH, "/html/body/div[4]/div/div[2]/main/div[1]/div/div/div[1]/div/div/div[4]/div/div[3]/div").text
            except:
                ownership = ""
                sqm = ""
                epc = ""
        house[1]=[address, houseType, beds, baths, receps, ownership, sqm, epc]

        houseDetails=[]
        for j in range(15):
            try:
                saletype = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
                date = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
                price = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
                try:listingbeds=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
                except:listingbeds=""
                try:listingbaths=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
                except:listingbaths=""
                try:listingreceps=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
                except:listingreceps=""
            except:
                try:
                    saletype = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
                    date = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
                    price = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
                    if beds=="":
                        beds = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
                        baths = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
                        receps = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
                        house[1][2] = beds
                        house[1][3] = baths
                        house[1][4] = receps
                    try:listingbeds=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
                    except:listingbeds=""
                    try:listingbaths=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
                    except:listingbaths=""
                    try:listingreceps=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
                    except:listingreceps=""
                except:pass
            # if not any("Listed" in s for s in houseDetails):
            if [saletype,date,price,listingbeds,listingbaths,listingreceps] not in houseDetails:
                houseDetails.append([saletype,date,price,listingbeds,listingbaths,listingreceps])
        house[2]=houseDetails




        

        for i in range(0, len(house[2])-1, 1):
            if house[2][i][0] == 'Sold' and house[2][i+1][0] == 'Listed':
                try:
                    sold_date = parse_date(house[2][i][1])
                    listed_date = parse_date(house[2][i+1][1])
                    #print(sold_date, "\n", listed_date)

                    time_diff = calculate_time_diff(sold_date, listed_date)
                    house[3]=[house[2][i+1][1],house[2][i][1],time_diff, house[2][i][2]]
                except:pass

        ##converts strings to ints
        try:house[1][2] = house[1][2][:house[1][2].index(" ")]
        except:pass
        try:house[1][3] = house[1][3][:house[1][3].index(" ")]
        except:pass
        try:house[1][4] = house[1][4][:house[1][4].index(" ")]
        except:pass
        try:house[1][5] = house[1][5][:house[1][5].index(" ")]
        except:pass
        try:
            if "semi-detached" in house[1][1].lower():
                house[1][1] = "4"
            elif "detached" in house[1][1].lower():
                house[1][1] = "3"
            elif "terrace" in house[1][1].lower():
                house[1][1] = "2"
            else:
                house[1][1] = "1"
        except:house[1][1]="0"
        
        try:
            if "sqm" in house[1][6]:
                house[1][6] = house[1][6][:house[1][6].index(" ")]
        except:house[1][6]="0"

        try:epc = house[1][7][house[1][7].index(" "):]
        except:pass
        try:
            if "A" in epc:epc="6"
            elif "B" in epc:epc="5"
            elif "C" in epc:epc="4"
            elif "D" in epc:epc="3"
            elif "E" in epc:epc="2"
            elif "F" in epc:epc="1"
            else:epc="0"
        except:epc="0"
        house[1][7]=epc
        
        try:
            if "Freehold" in house[1][5]:house[1][5]="2"
            else:house[1][5]="1"
        except:house[1][5]="0"
        
        for i in range(10):
            try:
                try:driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").click()
                except:
                    try:driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").click()
                    except:
                        try:driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").click()
                        except:
                            try:driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").click()
                            except:pass
                dateimg = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
                try:driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").click()
                except:pass
                for j in range(50):
                    try:
                        imgsrc = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").get_attribute("src")
                        imgatt = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").get_attribute("alt")
                        img = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]")
                        #print(imgsrc,imgatt)
                        if "Floorplan" not in imgatt:
                            #img.screenshot(f"[REDACTED_BY_SCRIPT]")
                            response = requests.get(imgsrc)
                            if response.status_code == 200:
                                newpath = f'[REDACTED_BY_SCRIPT]' 
                                if not os.path.exists(newpath):
                                    os.makedirs(newpath)
                                with open(f"[REDACTED_BY_SCRIPT]", "wb") as file:
                                    file.write(response.content)
                        else:
                            response = requests.get(imgsrc)
                            if response.status_code == 200:
                                newpath = f'[REDACTED_BY_SCRIPT]' 
                                if not os.path.exists(newpath):
                                    os.makedirs(newpath)
                                with open(f"[REDACTED_BY_SCRIPT]", "wb") as file:
                                    file.write(response.content)
                    except:pass
                x0,y0=mouse.get_position() 
                x1,y1=300, 500
                xdiff=x1-x0
                ydiff=y1-y0
                for moveseg in range(1,50):
                    xmove = x0 + moveseg*(xdiff/50)
                    ymove = y0 + moveseg*(ydiff/50)
                    movetime=0.4*math.sin(5*(xmove**3)+xmove**4-xmove**5)+.5
                    movetime=movetime/random.uniform(25,50)
                    mouse.move(xmove,ymove,absolute=True, duration=movetime)
                mouse.click()
                time.sleep(random.uniform(1, 2))
            except:pass
        houseL=[]
        houseweb=0
        print(house)
        print(house[2])
        for i in range(len(house[2])-1):
            if house[2][i][1][-4:] == '2024':
                if house[2][i][0] == 'Sold':
                    try:
                        if houseweb == 0:
                            if house[2][i+1][0] != 'Listed':
                                #### calculate time data using sold date from recently sold and average sell time from home.couk
                                commacount = house[1][0].count(",")
                                postcode= house[1][0][house[1][0].index(",")+2:]
                                postcode = postcode[postcode.index(",")+2:]
                                if commacount >=3:
                                    try:postcode = postcode[postcode.index(",")+2:]
                                    except:pass
                                if commacount >=4:
                                    try:postcode = postcode[postcode.index(",")+2:]
                                    except:pass
                                postcode=postcode[:postcode.index(" ")]
                                driver.get(f"[REDACTED_BY_SCRIPT]")
                                try:
                                    element_present = EC.element_to_be_clickable((By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]"))
                                    element_present = EC.presence_of_element_located((By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]"))
                                    WebDriverWait(driver, timeout).until(element_present)
                                except:
                                    wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
                                    element_present = EC.element_to_be_clickable((By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]"))
                                    element_present = EC.presence_of_element_located((By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]"))
                                    WebDriverWait(driver, timeout).until(element_present)


                                driver.find_element(By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]").click()

                                soldprice = int(row[1][1:].replace(",",""))
                                if soldprice <= 100000:
                                    meansoldprice = driver.find_element(By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]").text
                                elif soldprice > 100000 and soldprice <= 200000:
                                    meansoldprice = driver.find_element(By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]").text
                                elif soldprice > 200000 and soldprice <= 300000:
                                    meansoldprice = driver.find_element(By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]").text
                                elif soldprice > 300000 and soldprice <= 400000:
                                    meansoldprice = driver.find_element(By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]").text
                                elif soldprice > 400000 and soldprice <= 500000:
                                    meansoldprice = driver.find_element(By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]").text
                                elif soldprice > 500000 and soldprice <= 1000000:
                                    meansoldprice = driver.find_element(By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]").text
                                else:
                                    meansoldprice = driver.find_element(By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]").text
                                meansoldprice = meansoldprice[:meansoldprice.index(" ")]
                                meanbeds = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]")
                                meanbeds = meanbeds[:meanbeds.index(" ")]
                                meantype = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]")
                                meantype = meantype[:meantype.index(" ")]
                                #under 100k body > section.container-fluid.mb-5 > div > main > div.homeco_pr_content > div:nth-child(6) > table > tbody > tr:nth-child(2) > td:nth-child(3)
                                #100k-200k  body > section.container-fluid.mb-5 > div > main > div.homeco_pr_content > div:nth-child(6) > table > tbody > tr:nth-child(3) > td:nth-child(3)

                                #1bed body > section.container-fluid.mb-5 > div > main > div.homeco_pr_content > div:nth-child(9) > table > tbody > tr:nth-child(2) > td:nth-child(3)
                                #2bed body > section.container-fluid.mb-5 > div > main > div.homeco_pr_content > div:nth-child(9) > table > tbody > tr:nth-child(3) > td:nth-child(3)

                                #flat body > section.container-fluid.mb-5 > div > main > div.homeco_pr_content > div:nth-child(12) > table > tbody > tr:nth-child(2) > td:nth-child(3)
                                #terr body > section.container-fluid.mb-5 > div > main > div.homeco_pr_content > div:nth-child(12) > table > tbody > tr:nth-child(3) > td:nth-child(3)
                                meantts = (meansoldprice + meanbeds + meantype)/3
                                meantts = int(meantts)/30
                                houseL.append(meantts)
                            houseweb+=1
                    except:pass
                if house[2][i][0] == 'Listed':
                    try:
                        int(row[2][(row[2].index(" "))+1:])
                        skibidirizz=row[2]
                    except:
                        skibidirizz=row[2][(row[2].index(" "))+1:]

                    try:
                        sold_date = parse_date(skibidirizz)
                    except:
                        sold_date = parse_date2(skibidirizz)
                    listed_date = parse_date(house[2][i][1])
                    time_diff = calculate_time_diff(sold_date, listed_date)
                    houseL.append(time_diff)
            else:
                #### calculate time data using sold date from recently sold and average sell time from home.couk
                if houseweb == 0:
                    houseweb+=1 
                    commacount = house[1][0].count(",")
                    postcode= house[1][0][house[1][0].index(",")+2:]
                    postcode = postcode[postcode.index(",")+2:]
                    if commacount >=3:
                        try:postcode = postcode[postcode.index(",")+2:]
                        except:pass
                    if commacount >=4:
                        try:postcode = postcode[postcode.index(",")+2:]
                        except:pass
                    postcode=postcode[:postcode.index(" ")]
                    print(postcode)
                    driver.get(f"[REDACTED_BY_SCRIPT]")
                    try:
                        time.sleep(1)
                        driver.find_element(By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]").click()
                    except:
                        try:
                            element_present = EC.presence_of_element_located((By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]"))
                            WebDriverWait(driver, 10).until(element_present)

                            driver.find_element(By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]").click()
                        except:pass
                    try:soldprice = int(row[1][1:].replace(",",""))
                    except:
                        try:soldprice = int(row[1][2:].replace(",",""))
                        except:
                            try:soldprice = int(row[1][3:].replace(",",""))
                            except:pass
                    try:
                        if soldprice <= 100000:
                            meansoldprice = driver.find_element(By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]").text
                        elif soldprice > 100000 and soldprice <= 200000:
                            meansoldprice = driver.find_element(By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]").text
                        elif soldprice > 200000 and soldprice <= 300000:
                            meansoldprice = driver.find_element(By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]").text
                        elif soldprice > 300000 and soldprice <= 400000:
                            meansoldprice = driver.find_element(By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]").text
                        elif soldprice > 400000 and soldprice <= 500000:
                            meansoldprice = driver.find_element(By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]").text
                        elif soldprice > 500000 and soldprice <= 1000000:
                            meansoldprice = driver.find_element(By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]").text
                        else:
                            meansoldprice = driver.find_element(By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]").text
                    except:
                        time.sleep(1000)
                    try:meansoldprice = meansoldprice[:meansoldprice.index(" ")]
                    except:meansoldprice="90"
                    try:
                        meanbeds = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]")
                        meanbeds = meanbeds[:meanbeds.index(" ")]
                    except:meanbeds="90"
                    try:
                        meantype = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]")
                        meantype = meantype[:meantype.index(" ")]
                    except:meantype="90"
                    #under 100k body > section.container-fluid.mb-5 > div > main > div.homeco_pr_content > div:nth-child(6) > table > tbody > tr:nth-child(2) > td:nth-child(3)
                    #100k-200k  body > section.container-fluid.mb-5 > div > main > div.homeco_pr_content > div:nth-child(6) > table > tbody > tr:nth-child(3) > td:nth-child(3)

                    #1bed body > section.container-fluid.mb-5 > div > main > div.homeco_pr_content > div:nth-child(9) > table > tbody > tr:nth-child(2) > td:nth-child(3)
                    #2bed body > section.container-fluid.mb-5 > div > main > div.homeco_pr_content > div:nth-child(9) > table > tbody > tr:nth-child(3) > td:nth-child(3)

                    #flat body > section.container-fluid.mb-5 > div > main > div.homeco_pr_content > div:nth-child(12) > table > tbody > tr:nth-child(2) > td:nth-child(3)
                    #terr body > section.container-fluid.mb-5 > div > main > div.homeco_pr_content > div:nth-child(12) > table > tbody > tr:nth-child(3) > td:nth-child(3)
                    meantts = (int(meansoldprice) + int(meanbeds) + int(meantype))/3
                    meantts = int(meantts)/30
                    houseL.append(meantts)
        house[4]=houseL    



        ####house[2] = sale history.
        ####house[4] = time between listing and sale.
        ##### Need to check if most recent item is in 2024. And if it is, if it is a listing or a sale. If it is a sale, need to check if it has lising date.
        ##### If it has no listing date, need to go to home.co.uk to get average selling time in postcode.
        ##### If it has neither, will still have to go to house.co.uk, and use the info from the recentlySold.csv file.

        print(house)
        try:driver.quit()
        except:pass
        with open(r'[REDACTED_BY_SCRIPT]', 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile, delimiter=",", quotechar='"')
            writer.writerow(house)
    except Exception as e:
        try:
            driver.quit()
        except:
            pass
        print(f"[REDACTED_BY_SCRIPT]")
        traceback.print_exc()
        return False


async def main():
    global driver
    global row
    """[REDACTED_BY_SCRIPT]"""
    driver = initialize_driver()

    await scrape_with_timeout(driver, 360)

    driver.quit()  # Ensure the driver is closed after scraping

if __name__ == "__main__":
    asyncio.run(main())

df = pd.read_csv(r"[REDACTED_BY_SCRIPT]", encoding='latin-1')
df_cleaned = df.drop_duplicates()
df_cleaned.to_csv(r"[REDACTED_BY_SCRIPT]", index=False, encoding='latin-1')
