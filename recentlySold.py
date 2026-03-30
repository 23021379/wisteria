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

postcodes=["AL", "B", "BA", "BB", "BD", "BH", "BL", "BN", "BR", "BS", "BT", "CA", "CB", "CF", "CH", "CM", "CO", "CR", "CT", "CV", "CW", "DA", "DD", "DE", "DG", "DH", "DL", "DN", "DT", "DY", "E", "EC", "EH", "EN", "EX", "FK", "FY", "G", "GL", "GU", "HA", "HD", "HG", "HP", "HR", "HS", "HU", "HX", "IG", "IP", "IV", "KA", "KT", "KW", "KY", "L", "LA", "LD", "LE", "LL", "LN", "LS", "LU", "M", "ME", "MK", "ML", "N", "NE", "NG", "NN", "NP", "NR", "NW", "OL", "OX", "PA", "PE", "PH", "PL", "PO", "PR", "RG", "RH", "RM", "S", "SA", "SE", "SG", "SK", "SL", "SM", "SN", "SO", "SP", "SR", "SS", "ST", "SW", "SY", "TA", "TD", "TF", "TN", "TQ", "TR", "TS", "TW", "UB", "W", "WA", "WC", "WD", "WF", "WN", "WR", "WS", "WV", "YO", "ZE"]




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
                        try:                                                 #main-content > div._14bi3x329._14bi3x316._14bi3x31d._14bi3x365._14bi3x36b._14bi3x37p._14bi3x37v > div > div > section > div:nth-child(1) > div._17smgnt0 > div:nth-child({k+1}) > div._1hzil3o0 > div > div > div._1hzil3o4
                            address = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
                        except:
                            try:address = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
                            except:address=""
                        try:                                             #main-content > div._14bi3x329._14bi3x316._14bi3x31d._14bi3x365._14bi3x36b._14bi3x37p._14bi3x37v > div > div > section > div:nth-child(1) > div._17smgnt0 > div:nth-child({k+1}) > div._1hzil3o0 > div > div > div._1hzil3o3 > div:nth-child(1)
                            price=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
                            try:
                                price=price.split("\n")
                                print(price)
                                price=price[1]
                                print(price)
                            except:pass
                        except:
                            try:
                                price=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
                                try:
                                    price=price.split("\n")
                                    print(price)
                                    price=price[1]
                                    print(price)
                                except:pass
                            except:price=""
                        try:
                            date=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
                            try:
                                date=date.split("\n")
                                date=date[0]
                                date=date.split(" ")
                                date=date[2] + " " + date[3]
                                print(date)
                            except:pass
                            date=f'"{date}"'
                        except:
                            try:
                                date=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
                                try:
                                    date=date.split("\n")
                                    date=date[0]
                                    date=date.split(" ")
                                    date=date[2] + " " + date[3]
                                    print(date)
                                except:pass
                                date=f'"{date}"'
                            except:date=""
                        try:
                            link=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").get_attribute("href")
                            link=f'"{link}"'
                        except:
                            try:
                                link=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").get_attribute("href")
                                link=f'"{link}"'
                            except:link=""
                        with open(r'[REDACTED_BY_SCRIPT]', 'a', newline='', encoding='utf-8') as csvfile:
                            writer = csv.writer(csvfile, delimiter=",", quotechar='"')
                            writer.writerow([address,price,date,link])
                else:
                    for l in range(int(int(number_of_props)/25)):
                        if l == int(int(number_of_props)/25)-1:
                            num_left=((int(number_of_props)/25)-l)*25
                            for k in range(num_left):
                                try:                                                 #main-content > div._14bi3x329._14bi3x316._14bi3x31d._14bi3x365._14bi3x36b._14bi3x37p._14bi3x37v > div > div > section > div:nth-child(1) > div._17smgnt0 > div:nth-child({k+1}) > div._1hzil3o0 > div > div > div._1hzil3o4
                                    address = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
                                except:
                                    try:address = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
                                    except:address=""
                                try:                                             #main-content > div._14bi3x329._14bi3x316._14bi3x31d._14bi3x365._14bi3x36b._14bi3x37p._14bi3x37v > div > div > section > div:nth-child(1) > div._17smgnt0 > div:nth-child({k+1}) > div._1hzil3o0 > div > div > div._1hzil3o3 > div:nth-child(1)
                                    price=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
                                    try:
                                        price=price.split("\n")
                                        print(price)
                                        price=price[1]
                                        print(price)
                                    except:pass
                                except:
                                    try:
                                        price=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
                                        try:
                                            price=price.split("\n")
                                            print(price)
                                            price=price[1]
                                            print(price)
                                        except:pass
                                    except:price=""
                                try:
                                    date=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
                                    try:
                                        date=date.split("\n")
                                        date=date[0]
                                        date=date.split(" ")
                                        date=date[2] + " " + date[3]
                                        print(date)
                                    except:pass
                                    date=f'"{date}"'
                                except:
                                    try:
                                        date=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
                                        try:
                                            date=date.split("\n")
                                            date=date[0]
                                            date=date.split(" ")
                                            date=date[2] + " " + date[3]
                                            print(date)
                                        except:pass
                                        date=f'"{date}"'
                                    except:date=""
                                try:
                                    link=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").get_attribute("href")
                                    link=f'"{link}"'
                                except:
                                    try:
                                        link=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").get_attribute("href")
                                        link=f'"{link}"'
                                    except:link=""
                                with open(r'[REDACTED_BY_SCRIPT]', 'a', newline='', encoding='utf-8') as csvfile:
                                    writer = csv.writer(csvfile, delimiter=",", quotechar='"')
                                    writer.writerow([address,price,date,link])
                        else:
                            for k in range(25):
                                try:                                                 #main-content > div._14bi3x329._14bi3x316._14bi3x31d._14bi3x365._14bi3x36b._14bi3x37p._14bi3x37v > div > div > section > div:nth-child(1) > div._17smgnt0 > div:nth-child({k+1}) > div._1hzil3o0 > div > div > div._1hzil3o4
                                    address = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
                                except:
                                    try:address = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
                                    except:address=""
                                try:                                             #main-content > div._14bi3x329._14bi3x316._14bi3x31d._14bi3x365._14bi3x36b._14bi3x37p._14bi3x37v > div > div > section > div:nth-child(1) > div._17smgnt0 > div:nth-child({k+1}) > div._1hzil3o0 > div > div > div._1hzil3o3 > div:nth-child(1)
                                    price=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
                                    try:
                                        price=price.split("\n")
                                        print(price)
                                        price=price[1]
                                        print(price)
                                    except:pass
                                except:
                                    try:
                                        price=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
                                        try:
                                            price=price.split("\n")
                                            print(price)
                                            price=price[1]
                                            print(price)
                                        except:pass
                                    except:price=""
                                try:
                                    date=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
                                    try:
                                        date=date.split("\n")
                                        date=date[0]
                                        date=date.split(" ")
                                        date=date[2] + " " + date[3]
                                        print(date)
                                    except:pass
                                    date=f'"{date}"'
                                except:
                                    try:
                                        date=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
                                        try:
                                            date=date.split("\n")
                                            date=date[0]
                                            date=date.split(" ")
                                            date=date[2] + " " + date[3]
                                            print(date)
                                        except:pass
                                        date=f'"{date}"'
                                    except:date=""
                                try:
                                    link=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").get_attribute("href")
                                    link=f'"{link}"'
                                except:
                                    try:
                                        link=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").get_attribute("href")
                                        link=f'"{link}"'
                                    except:link=""
                                with open(r'[REDACTED_BY_SCRIPT]', 'a', newline='', encoding='utf-8') as csvfile:
                                    writer = csv.writer(csvfile, delimiter=",", quotechar='"')
                                    writer.writerow([address,price,date,link])
                        driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").click()
            except Exception as e:
                print(e)
                pass
        try:driver.quit()
        except:pass
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
