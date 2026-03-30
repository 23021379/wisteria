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
#"AL", "B", "BA", "BB", "BD", "BH", "BL", "BN", "BR", "BS", "BT", "CA", "CB", "CF", "CH", "CM", "CO", "CR", "CT", "CV", "CW", "DA", "DD", "DE", "DG", "DH", "DL", "DN", "DT", "DY", "E", "EC", "EH", "EN", "EX", "FK", "FY", "G", "GL", "GU", "HA", "HD", "HG", "HP", "HR", "HS", "HU", "HX", "IG", "IP", "IV", "KA", "KT", "KW", "KY", "L", "LA", "LD", "LE", "LL", "LN", "LS", "LU", "M", "ME", "MK", "ML", "N", "NE", "NG", "NN", "NP", "NR", "NW", "OL", "OX", "PA", "PE", "PH", "PL", "PO", "PR", "RG", 
postcodes=["RH", "RM", "S", "SA", "SE", "SG", "SK", "SL", "SM", "SN", "SO", "SP", "SR", "SS", "ST", "SW", "SY", "TA", "TD", "TF", "TN", "TQ", "TR", "TS", "TW", "UB", "W", "WA", "WC", "WD", "WF", "WN", "WR", "WS", "WV", "YO", "ZE"]




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
    global checkloops
    for postcode_item in postcodes:
        completedScrape = False
        retryattempt=0
        while completedScrape == False:
            if not driver:
                time.sleep(1)
                driver = initialize_driver()
            try:
                await asyncio.wait_for(asyncio.to_thread(scrape_page, postcode_item), timeout=timeout_seconds)
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
def scrape_page(postcode_item):
    global driver
    global checkloops
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
                if imagecomp == "49.png" and max_val > .9:
                    match = True
                    break
            time.sleep(screenshot_interval)

        timeout = 45

        #time.sleep(1000)
        if loopcount==0:
            ####clicks cookie thing
            #accept
            wait = WebDriverWait(driver, 15)
            element_locator = (By.CSS_SELECTOR, '[REDACTED_BY_SCRIPT]')
            element = wait.until(EC.presence_of_element_located(element_locator))
            driver.find_element(By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]").click()
                
        element_present = EC.presence_of_element_located((By.CSS_SELECTOR, '[REDACTED_BY_SCRIPT]'))
        WebDriverWait(driver, timeout).until(element_present)
        for subpostcode_item in range(1,22):
            randomFrom=int(random.uniform(1,10))
            driver.get(f"[REDACTED_BY_SCRIPT]")
            #time.sleep(1000)
            
            try:
                time.sleep(2)
                element_present = EC.presence_of_element_located((By.CSS_SELECTOR, '[REDACTED_BY_SCRIPT]'))
                WebDriverWait(driver, 15).until(element_present)
                number_of_props=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
                number_of_props=number_of_props[:number_of_props.index(" ")]
                try:
                    number_of_props=number_of_props.replace(" ","")
                except:pass
                print(number_of_props)
                if int(number_of_props)<=25:
                    #main-content > div._14bi3x329._14bi3x316._14bi3x31d._14bi3x365._14bi3x36b._14bi3x37p._14bi3x37v > div > div > section > div:nth-child(1) > div._17smgnt0 > div:nth-child(19) > div > div > div > div._1hzil3o4 > a
                    #main-content > div._14bi3x329._14bi3x316._14bi3x31d._14bi3x365._14bi3x36b._14bi3x37p._14bi3x37v > div > div > section > div:nth-child(1) > div._17smgnt0 > div:nth-child(19) > div > div > div > div._1hzil3o4
                    for k in range(int(number_of_props)):
                        try:                                                 #root > div > div > div.obk37mjjq8gxI2WI0pOG > div > div.sbjbExlAlGsUFzukvfnA > a:nth-child(2) > div.U_bIwRI2FjLNv4ZUFwda
                            address = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
                        except:
                            try:address = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
                            except:address=""
                        try:                                             #root > div > div > div.obk37mjjq8gxI2WI0pOG > div > div.sbjbExlAlGsUFzukvfnA > a:nth-child(25) > div.xxm425gkCI9Xl4acPrzw > div:nth-child(2) > div > table > tbody > tr:nth-child(2) > td.eAeEgAzoQt7uJJ0H4Mom.atz5tYnaqoJ2uRM7nl_f
                                                                         #root > div > div > div.obk37mjjq8gxI2WI0pOG > div > div.sbjbExlAlGsUFzukvfnA > a:nth-child(25) > div.xxm425gkCI9Xl4acPrzw > div:nth-child(2) > div > table > tbody > tr:nth-child(2) > td.eAeEgAzoQt7uJJ0H4Mom.undefined
                            date=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
                        except:
                            try:
                                date=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
                            except:date=""
                        try:#                                           #root > div > div > div.obk37mjjq8gxI2WI0pOG > div > div.sbjbExlAlGsUFzukvfnA > a:nth-child({k+1}) > div.xxm425gkCI9Xl4acPrzw > div:nth-child(2) > div > table > tbody > tr:nth-child(2) > td.eAeEgAzoQt7uJJ0H4Mom.undefined
                            price=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
                        except:
                            try:
                                price=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
                            except:price=""
                        try:#                                           #root > div > div > div.obk37mjjq8gxI2WI0pOG > div > div.sbjbExlAlGsUFzukvfnA > a:nth-child(2)
                            link=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").get_attribute("href")
                        except:
                            try:
                                link=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").get_attribute("href")
                            except:link=""
                        if date != "":
                            if "2024" in date:
                                if "dec" in date.lower() or "nov" in date.lower():
                                    with open(r'[REDACTED_BY_SCRIPT]', 'a', newline='', encoding='utf-8') as csvfile:
                                        writer = csv.writer(csvfile, delimiter=",", quotechar='"')
                                        writer.writerow([address,price,date,link])
                            elif "2025" in date:
                                with open(r'[REDACTED_BY_SCRIPT]', 'a', newline='', encoding='utf-8') as csvfile:
                                    writer = csv.writer(csvfile, delimiter=",", quotechar='"')
                                    writer.writerow([address,price,date,link])
                            else:break
                else:
                    for l in range(int(int(number_of_props)/25)):
                        if l == int(int(number_of_props)/25)-1:
                            num_left=((int(number_of_props)/25)-l)*25
                            for k in range(num_left):
                                #main-content > div._14bi3x329._14bi3x316._14bi3x31d._14bi3x365._14bi3x36b._14bi3x37p._14bi3x37v > div > div > section > div:nth-child(1) > div._17smgnt0 > div:nth-child(19) > div > div > div > div._1hzil3o4 > a
                                #main-content > div._14bi3x329._14bi3x316._14bi3x31d._14bi3x365._14bi3x36b._14bi3x37p._14bi3x37v > div > div > section > div:nth-child(1) > div._17smgnt0 > div:nth-child(19) > div > div > div > div._1hzil3o4
                                try:                                                 #root > div > div > div.obk37mjjq8gxI2WI0pOG > div > div.sbjbExlAlGsUFzukvfnA > a:nth-child(2) > div.U_bIwRI2FjLNv4ZUFwda
                                    address = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
                                except:
                                    try:address = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
                                    except:address=""
                                try:                                             #root > div > div > div.obk37mjjq8gxI2WI0pOG > div > div.sbjbExlAlGsUFzukvfnA > a:nth-child(1) > div.xxm425gkCI9Xl4acPrzw > div:nth-child(2) > div > table > tbody > tr:nth-child(2) > td.eAeEgAzoQt7uJJ0H4Mom.atz5tYnaqoJ2uRM7nl_f
                                    date=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
                                except:
                                    try:
                                        date=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
                                    except:date=""
                                try:#                                           #root > div > div > div.obk37mjjq8gxI2WI0pOG > div > div.sbjbExlAlGsUFzukvfnA > a:nth-child(3) > div.xxm425gkCI9Xl4acPrzw > div:nth-child(2) > div > table > tbody > tr:nth-child(2) > td.eAeEgAzoQt7uJJ0H4Mom.undefined
                                    price=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
                                except:
                                    try:
                                        price=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
                                    except:price=""
                                try:#                                           #root > div > div > div.obk37mjjq8gxI2WI0pOG > div > div.sbjbExlAlGsUFzukvfnA > a:nth-child(2)
                                    link=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").get_attribute("href")
                                except:
                                    try:
                                        link=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").get_attribute("href")
                                    except:link=""
                                if date != "":
                                    if "2024" in date:
                                        if "dec" in date.lower() or "nov" in date.lower():
                                            with open(r'[REDACTED_BY_SCRIPT]', 'a', newline='', encoding='utf-8') as csvfile:
                                                writer = csv.writer(csvfile, delimiter=",", quotechar='"')
                                                writer.writerow([address,price,date,link])
                                    elif "2025" in date:
                                        with open(r'[REDACTED_BY_SCRIPT]', 'a', newline='', encoding='utf-8') as csvfile:
                                            writer = csv.writer(csvfile, delimiter=",", quotechar='"')
                                            writer.writerow([address,price,date,link])
                                    else:break
                                
                        else:
                            for k in range(25):
                                #main-content > div._14bi3x329._14bi3x316._14bi3x31d._14bi3x365._14bi3x36b._14bi3x37p._14bi3x37v > div > div > section > div:nth-child(1) > div._17smgnt0 > div:nth-child(19) > div > div > div > div._1hzil3o4 > a
                                #main-content > div._14bi3x329._14bi3x316._14bi3x31d._14bi3x365._14bi3x36b._14bi3x37p._14bi3x37v > div > div > section > div:nth-child(1) > div._17smgnt0 > div:nth-child(19) > div > div > div > div._1hzil3o4
                                try:                                                 #root > div > div > div.obk37mjjq8gxI2WI0pOG > div > div.sbjbExlAlGsUFzukvfnA > a:nth-child(2) > div.U_bIwRI2FjLNv4ZUFwda
                                    address = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
                                except:
                                    try:address = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
                                    except:address=""
                                try:                                             #root > div > div > div.obk37mjjq8gxI2WI0pOG > div > div.sbjbExlAlGsUFzukvfnA > a:nth-child(1) > div.xxm425gkCI9Xl4acPrzw > div:nth-child(2) > div > table > tbody > tr:nth-child(2) > td.eAeEgAzoQt7uJJ0H4Mom.atz5tYnaqoJ2uRM7nl_f
                                    date=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
                                except:
                                    try:
                                        date=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
                                    except:date=""
                                try:#                                           #root > div > div > div.obk37mjjq8gxI2WI0pOG > div > div.sbjbExlAlGsUFzukvfnA > a:nth-child(3) > div.xxm425gkCI9Xl4acPrzw > div:nth-child(2) > div > table > tbody > tr:nth-child(2) > td.eAeEgAzoQt7uJJ0H4Mom.undefined
                                    price=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
                                except:
                                    try:
                                        price=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
                                    except:price=""
                                try:#                                           #root > div > div > div.obk37mjjq8gxI2WI0pOG > div > div.sbjbExlAlGsUFzukvfnA > a:nth-child(2)
                                    link=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").get_attribute("href")
                                except:
                                    try:
                                        link=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").get_attribute("href")
                                    except:link=""
                                if date != "":
                                    if "2024" in date:
                                        if "dec" in date.lower() or "nov" in date.lower():
                                            with open(r'[REDACTED_BY_SCRIPT]', 'a', newline='', encoding='utf-8') as csvfile:
                                                writer = csv.writer(csvfile, delimiter=",", quotechar='"')
                                                writer.writerow([address,price,date,link])
                                    elif "2025" in date:
                                        with open(r'[REDACTED_BY_SCRIPT]', 'a', newline='', encoding='utf-8') as csvfile:
                                            writer = csv.writer(csvfile, delimiter=",", quotechar='"')
                                            writer.writerow([address,price,date,link])
                                    else:break
                        if date != "":
                            if "2024" in date or "2025" in date:
                                pass
                            else:break
                        wait=WebDriverWait(driver, 20)
                        try:
                            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]")))
                        except:pass
                        driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").click()
                        wait=WebDriverWait(driver, 10)
                        try:
                            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]")))
                        except:pass
                        
            except Exception as e:
                print(e)
                pass
        if checkloops==7:
            try:driver.quit()
            except:pass
        else:
            checkloops+=1
    except Exception as e:
        try:
            driver.quit()
        except:
            pass
        print(f"[REDACTED_BY_SCRIPT]")
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
import pandas as pd

df = pd.read_csv(r"[REDACTED_BY_SCRIPT]", encoding='latin-1')
df_cleaned = df.drop_duplicates()
df_cleaned.to_csv(r"[REDACTED_BY_SCRIPT]", index=False, encoding='latin-1')




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
#"AL", "B", "BA", "BB", "BD", "BH", "BL", "BN", "BR", "BS", "BT", "CA", "CB", "CF", "CH", "CM", "CO", "CR", "CT", "CV", "CW", 
postcodes=["DA", "DD", "DE", "DG", "DH", "DL", "DN", "DT", "DY", "E", "EC", "EH", "EN", "EX", "FK", "FY", "G", "GL", "GU", "HA", "HD", "HG", "HP", "HR", "HS", "HU", "HX", "IG", "IP", "IV", "KA", "KT", "KW", "KY", "L", "LA", "LD", "LE", "LL", "LN", "LS", "LU", "M", "ME", "MK", "ML", "N", "NE", "NG", "NN", "NP", "NR", "NW", "OL", "OX", "PA", "PE", "PH", "PL", "PO", "PR", "RG", "RH", "RM", "S", "SA", "SE", "SG", "SK", "SL", "SM", "SN", "SO", "SP", "SR", "SS", "ST", "SW", "SY", "TA", "TD", "TF", "TN", "TQ", "TR", "TS", "TW", "UB", "W", "WA", "WC", "WD", "WF", "WN", "WR", "WS", "WV", "YO", "ZE"]




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

async def scrape_with_timeout2(addressa, timeout_seconds=300):  # 5 minutes = 300 seconds
    global driver
    global row
    with open(r'[REDACTED_BY_SCRIPT]', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
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
                    driver = initialize_driver2()
                try:
                    await asyncio.wait_for(asyncio.to_thread(scrape_page2, addressa, address,price,date), timeout=timeout_seconds)
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

def initialize_driver2():
    global driver
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
def scrape_page2(addressa, address,price,date):
    global driver
    """
    Your scraping logic goes here.  This function should be synchronous.
    """
    try:    
        try:
            driver.get("[REDACTED_BY_SCRIPT]")
        except:
            driver.quit()
            time.sleep(1)
            driver = initialize_driver2()
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
                if imagecomp == "49.png" and max_val > .9:
                    match = True
                    break
            time.sleep(screenshot_interval)

        timeout = 45

        #time.sleep(1000)
        if loopcount==0:
            ####clicks cookie thing
            #accept
            wait = WebDriverWait(driver, 15)
            element_locator = (By.CSS_SELECTOR, '[REDACTED_BY_SCRIPT]')
            element = wait.until(EC.presence_of_element_located(element_locator))
            driver.find_element(By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]").click()
                
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
            with open(r'[REDACTED_BY_SCRIPT]', 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile, delimiter=",", quotechar='"')
                writer.writerow([[address.replace(",","").lower(),addressa],[address,housetype,beds,baths],prevsold])
        try:driver.quit()
        except:pass
    except Exception as e:
        try:
            driver.quit()
        except:
            pass
        print(f"[REDACTED_BY_SCRIPT]")
        return False


async def main2():
    global driver
    global row
    """[REDACTED_BY_SCRIPT]"""
    driver = initialize_driver2()

    await scrape_with_timeout2(driver, 360)

    driver.quit()  # Ensure the driver is closed after scraping

if __name__ == "__main__":
    asyncio.run(main2())
import pandas as pd

df = pd.read_csv(r"[REDACTED_BY_SCRIPT]", encoding='latin-1')
df_cleaned = df.drop_duplicates()
df_cleaned.to_csv(r"[REDACTED_BY_SCRIPT]", index=False, encoding='latin-1')
