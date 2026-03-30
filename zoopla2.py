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
import asyncio
import ast
import traceback
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity



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
    global parsed_dataout
    global cookiecount
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
    
    for i in range(len(parsed_data)):
        completedScrape = False
        retryattempt=0
        while completedScrape == False:
            parsed_datain=parsed_data[i]
            addressa=parsed_data[i][0][1].replace(",","")
            addressa=addressa.lower()
            if not driver:
                time.sleep(1)
                driver = initialize_driver()
            try:
                await asyncio.wait_for(asyncio.to_thread(scrape_page, addressa,parsed_datain), timeout=timeout_seconds)
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
    global parsed_dataout
    global cookiecount
    """[REDACTED_BY_SCRIPT]"""
    options = uc.ChromeOptions()
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    
    
    
    
    options.page_load_strategy = 'eager'
    time.sleep(1)
    driver = uc.Chrome(options=options)
    cookiecount=0
    return driver

loopcount=0
def scrape_page(addressa,parsed_datain):
    global driver
    global row
    global parsed_dataout
    global cookiecount
    try:    
        try:
            driver.get(addressa)
        except:
            driver.quit()
            time.sleep(1)
            driver = initialize_driver()
            time.sleep(1)
            driver.get(addressa) 

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
                if imagecomp == "44.png" and max_val > .9 or imagecomp == "46.png" and max_val > .7 or imagecomp == "47.png" and max_val > .7 or imagecomp == "48.png" and max_val > .7:
                    match = True
                    break
            time.sleep(screenshot_interval)

        #time.sleep(1000)
        timeout = 15
        if cookiecount==0:
            try:
                wait = WebDriverWait(driver, 45)
                element_locator = (By.CSS_SELECTOR, '#usercentrics-cmp-ui')
                wait.until(EC.presence_of_element_located(element_locator))
                shadow_host = driver.find_element(By.CSS_SELECTOR, "#usercentrics-cmp-ui")
                shadow_root = driver.execute_script("[REDACTED_BY_SCRIPT]", shadow_host)

                wait = WebDriverWait(shadow_root, timeout)
                element_locator = (By.CSS_SELECTOR, '#accept')
                wait.until(EC.presence_of_element_located(element_locator))
                shadow_root.find_element(By.CSS_SELECTOR, "#accept").click()
                cookiecount=1
            except:
                wait = WebDriverWait(driver, 30)
                element_locator = (By.CSS_SELECTOR, '#usercentrics-root')
                wait.until(EC.presence_of_element_located(element_locator))
                shadow_host = driver.find_element(By.CSS_SELECTOR, "#usercentrics-root")
                shadow_root = driver.execute_script("[REDACTED_BY_SCRIPT]", shadow_host)

                wait = WebDriverWait(shadow_root, timeout)
                element_locator = (By.CSS_SELECTOR, '[REDACTED_BY_SCRIPT]')
                element = wait.until(EC.presence_of_element_located(element_locator))
                shadow_root.find_element(By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]").click()
                cookiecount=1
        

        try:
            element_present = EC.presence_of_element_located((By.XPATH, '/html/body/div[4]/div/div[2]/main/div[1]/div/div/div[1]/div/div[2]/div[2]/div/div/div/h1'))
            WebDriverWait(driver, timeout).until(element_present)
            houseType = driver.find_element(By.XPATH, "/html/body/div[4]/div/div[2]/main/div[1]/div/div/div[1]/div/div[2]/div[3]/div/div[1]/div/p").text
            #                                          /html/body/div[4]/div/div[2]/main/div[1]/div/div/div[1]/div/div/div[3]/div/div/div/p
            try:
                beds = driver.find_element(By.XPATH, "[REDACTED_BY_SCRIPT]").text
            except:
                beds = ""            
        except:
            element_present = EC.presence_of_element_located((By.XPATH, '/html/body/div[4]/div/div[2]/main/div[1]/div/div/div[1]/div/div/div[2]/div/div/div/h1'))
            WebDriverWait(driver, timeout).until(element_present)
            houseType = driver.find_element(By.XPATH, "/html/body/div[4]/div/div[2]/main/div[1]/div/div/div[1]/div/div/div[3]/div/div[1]/div/p").text
            try:
                beds = driver.find_element(By.XPATH, "/html/body/div[4]/div/div[2]/main/div[1]/div/div/div[1]/div/div/div[3]/div/div[2]/p").text
            except:
                beds = ""
        
        try:
            bedsSorted=False
            while bedsSorted == False:
                if beds.index(" ") == 0:
                    beds = beds[1:]
                else:
                    bedsSorted = True
                    break
            parsed_datain[1][2]=beds[:beds.index(" ")]
        except:
            parsed_datain[1][2]=""
        
        try:
            if "semi-detached" in houseType.lower():
                parsed_datain[1][1] = "4"
            elif "detached" in houseType.lower():
                parsed_datain[1][1] = "3"
            elif "terrace" in houseType.lower():
                parsed_datain[1][1] = "2"
            else:
                parsed_datain[1][1] = "1"
        except:parsed_datain[1][1]="0"
        parsed_dataout.append(parsed_datain)
        with open(r"[REDACTED_BY_SCRIPT]", 'a', newline='', encoding='utf-8') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(parsed_datain)

    except Exception as e:
        try:
            driver.quit()
        except:
            pass
        error_message = traceback.format_exc()
        print(f"[REDACTED_BY_SCRIPT]")
        return False


async def main():
    global driver
    global row
    global parsed_dataout
    global cookiecount
    """[REDACTED_BY_SCRIPT]"""
    driver = initialize_driver()
    await scrape_with_timeout(300)
    driver.quit()  # Ensure the driver is closed after scraping





############################################################################################################################################
############################################################################################################################################
############################################################################################################################################
############################################################################################################################################







async def scrape_with_timeout2(timeout_seconds=120):  
    global driver
    global row
    global parsed_dataout
    global cookiecount
    global tryCount
    global i
    global driverReset
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
    
    tryCount=0
    i=0
    driverReset=0
    for loop in range(len(parsed_data)):
        completedScrape = False
        retryattempt=0
        while completedScrape == False:
            if not driver:
                time.sleep(1)
                driver = initialize_driver()
            try:
                await asyncio.wait_for(asyncio.to_thread(scrape_page2,parsed_data), timeout=timeout_seconds)
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





def scrape_page2(parsed_data):
    global driver
    global row
    global parsed_dataout
    global tryCount
    global i
    global driverReset

    # options = uc.ChromeOptions()
    # options.add_argument('--disable-gpu')
    # options.add_argument('--no-sandbox')
    # options.add_argument('--disable-dev-shm-usage')
    # options.page_load_strategy = 'eager'
    # time.sleep(1)
    # driver = uc.Chrome(options=options)


    # parsed_data = []
    # with open(r"[REDACTED_BY_SCRIPT]", 'r', newline='', encoding='utf-8') as csvfile:
    #     next(csvfile)
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


    
    #for loop in range(len(parsed_data)):
    if not driver:
        options = uc.ChromeOptions()
        options.add_argument('--disable-gpu')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        
        
        
        
        options.page_load_strategy = 'eager'
        time.sleep(1)
        driver = uc.Chrome(options=options)
        driverReset=1
    try:
        if tryCount == 0:
            addressin=parsed_data[i][1][0]
            addressin=addressin.split(",")
            addressin2=addressin[-2][1:].lower()+"/"+addressin[-1][1:].lower()+"/"+addressin[0].lower()+"/"
            addressin2=addressin2.replace(" ","-")
        elif tryCount == 1:
            addressin=parsed_data[i][1][0]
            addressin=addressin.split(",")
            addressin2=(addressin[-2][1:].replace("-","")).lower()+"/"+addressin[-1][1:].lower()+"/"+addressin[0].lower()+"/"
            addressin2=addressin2.replace(" ","-")
        elif tryCount == 2:
            addressin=parsed_data[i][1][0]
            addressin=addressin.split(",")
            addressin2=addressin[-2][1:].lower()+"/"+addressin[-1][1:].lower()+"/"+addressin[0].lower()+"/"
            addressin2=addressin2.replace(" ","")
        elif tryCount == 3:
            addressin=parsed_data[i][1][0]
            addressin=addressin.split(",")
            addressin2=addressin[-2][1:].lower()+"/"+addressin[-1][1:].lower()+"/"+addressin[0].lower()+"/"
            addressin2=addressin2.replace(" ","-")
        elif tryCount == 4:
            addressin=parsed_data[i][1][0]
            addressin=addressin.split(",")
            addressin2=addressin[2][1:].lower()+"/"+addressin[1][1:].lower()+"/"+addressin[3][1:].lower()+"/"+addressin[0].lower()+"/"
            addressin2=addressin2.replace(" ","")
        elif tryCount == 5:
            addressin=parsed_data[i][1][0]
            addressin=addressin.split(",")
            addressin2=addressin[2][1:].lower()+"/"+addressin[1][1:].lower()+"/"+addressin[3][1:].lower()+"/"+addressin[0].lower()+"/"
            addressin2=addressin2.replace(" ","-")
        elif tryCount == 6:
            addressin=parsed_data[i][1][0]
            addressin=addressin.split(",")
            addressin2=addressin[-1][1:].lower()+"/"+addressin[-2][1:].lower()+"/"+[addressin[k+3][1:] if addressin[k+3] == " "else addressin[k+3] for k in range(len(addressin)-2)].lower()+"/" 
            addressin2=addressin2.replace(" ","-")
        elif tryCount == 7:
            addressin=parsed_data[i][1][0]
            addressin=addressin.split(",")
            addressin2=addressin[-1][1:].lower()+"/"+addressin[-2][1:].lower()+"/"+[addressin[k+3][1:] if addressin[k+3] == " "else addressin[k+3] for k in range(len(addressin)-2)].lower()+"/" 
            addressin2=addressin2.replace(" ","")
        elif tryCount == 8:
            addressin=parsed_data[i][1][0]
            addressin=addressin.split(",")
            addressin2=addressin[2][1:].lower()+"/"+addressin[3][1:].lower()+"/"+addressin[0].lower()+"-"+addressin[1][1:].lower()+"/"
            addressin2=addressin2.replace(" ","-")
        else:
            addressin=parsed_data[i][1][0]
            addressin=addressin.split(",")
            addressin2=addressin[1][1:].lower()+"/"+addressin[2][1:].lower()+"/"+addressin[0].lower()+"/"
            addressin2=addressin2.replace(" ","-")
        #3 Bowes-Lyon Court, Dryden Road, Gateshead, NE9 5BX
        #gateshead/ne9-5bx/3-bowes-lyon-court-dryden-road/
        addressurl = "[REDACTED_BY_SCRIPT]"+addressin2
        driver.get(addressurl)

        #time.sleep(1000)
        if driverReset == 0:
            try:
                element_present = EC.presence_of_element_located((By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]"))
                WebDriverWait(driver, 5).until(element_present)
                driver.find_element(By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]").click()
            except:
                pass
            driverReset=1
        #time.sleep(1000)

        time.sleep(random.uniform(0.8, 1.5))
        try:
            est = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
            est = est.split("\n")
            est=int(est[1].replace("£","").replace(",",""))
        except:
            est = ""

        try:
            change = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
            change = change.split("\n")
        except:
            change = []

        try:
            range = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
            range = range.split("\n")
        except:
            range = []

        try:
            conf = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
            conf = conf.split("\n")
            conf = conf[1]
            if "High" in conf:conf=3
            elif "Moderate" in conf:conf=2
            elif "Low" in conf:conf=1
            else:conf=0
        except:
            conf = 0

        try:
            lastp = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
            lastp = lastp.split("\n")
            lastp=lastp[1].replace("£","").replace(",","")
        except:
            lastp = ""

        try:
            lastd = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
            lastd = lastd.split("\n")
            lastd=lastd[1]
        except:
            lastd = ""

        try:
            type = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
            type = type.split("\n")
        except:
            type = []

        try:
            subtype = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
            subtype = subtype.split("\n")
            subtype=subtype[1]
            if "Detached" in subtype:subtype=4
            elif "Semi-Detached" in subtype:subtype=3
            elif "Terraced" in subtype:subtype=2
            else:subtype=1
        except:
            subtype = 0

        try:
            beds = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
            beds = beds.split("\n")
            beds=beds[1]
        except:
            beds = ""

        try:
            receps = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
            receps = receps.split("\n")
            receps=receps[1]
        except:
            receps = ""

        try:
            extens = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
            extens = extens.split("\n")
            extens=extens[1]
        except:
            extens = ""

        try:
            storey = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
            storey = storey.split("\n")
            storey=storey[1]
        except:
            storey = ""

        try:
            sqm = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
            sqm = sqm.split("\n")
            sqm=sqm[1]
        except:
            sqm = ""

        try:
            tenure = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
            tenure = tenure.split("\n")
            tenure=tenure[1]
            if "Freehold" in tenure:tenure=2
            else:tenure=1
        except:
            tenure = 0

        try:
            epccurrent = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
            epccurrent = epccurrent.split("\n")
            epccurrent=epccurrent[1]
            epccurrent=epccurrent[epccurrent.index("/")+2:]
        except:
            epccurrent = ""

        try:
            epcpotential = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
            epcpotential = epcpotential.split("\n")
            epcpotential=epcpotential[1]
            epcpotential=epcpotential[epcpotential.index("/")+2:]
        except:
            epcpotential = ""

        try:
            council = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
            council = council.split("\n")
        except:
            council = []

        try:
            councilband = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
            councilband = councilband.split("\n")
        except:
            councilband = []

        try:
            permission = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
            permission = permission.split("\n")
        except:
            permission = []

        try:
            age = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
            age = age.split("\n")
            age=age[1]
            if "No" in age:age = age[(age.index("-")+1):-1]
            else: age = "2025"
        except:
            age = ""

        try:
            flood = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
            flood = flood.split("\n")
        except:
            flood = []

        try:
            la = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
            la = la.split("\n")
        except:
            la = []
        try:
            rail = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
            rail = rail.split("\n")
        except:
            rail=[]
        try:
            bus = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
            bus = bus.split("\n")
        except:
            bus=[]
        
        try:
            Pschool = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
            Pschool = Pschool.split("\n")
        except:
            Pschool=[]
        try:
            Sschool = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
            Sschool = Sschool.split("\n")
        except:
            Sschool=[]
        try:
            Nursery = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
            Nursery = Nursery.split("\n")
        except:
            Nursery=[]
        try:
            special = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
            special = special.split("\n")
        except:
            special=[]

        try:    
            Churches = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
            Churches = Churches.split("\n")
        except:
            Churches=[]
        try:
            Mosque = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
            Mosque = Mosque.split("\n")
        except:
            Mosque=[]
        try:
            Gurdwara = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
            Gurdwara = Gurdwara.split("\n")
        except:
            Gurdwara=[]
        try:
            Synagogue = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
            Synagogue = Synagogue.split("\n")
        except:
            Synagogue=[]
        try:
            Mandir = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
            Mandir = Mandir.split("\n")
        except:
            Mandir=[]

        try:
            gp = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
            gp = gp.split("\n")
        except:
            gp=[]
        try:
            dent = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
            dent = dent.split("\n")
        except:
            dent=[]
        try:
            hosp = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
            hosp=hosp.split("\n")
        except:
            hosp=[]
        try:
            pharm = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
            pharm = pharm.split("\n")
        except:pharm=[]
        try:
            opt = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
            opt = opt.split("\n")
        except:opt=[]
        try:
            clin = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
            clin = clin.split("\n")
        except:
            clin=[]
        try:
            other = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
            other = other.split("\n")
        except:
            other=[]

        
        parsed_dataout=[[[parsed_data[i][1][0]], [est], change, range, [conf], [lastp], [lastd], type, [subtype], [beds], [receps], [extens], [storey], [sqm], [tenure], [epccurrent], [epcpotential], council, councilband, permission, age, flood, la, rail, bus, Pschool, Sschool, Nursery, special, Churches, Mosque, Gurdwara, Synagogue, Mandir, gp, dent, hosp, pharm, opt, clin, other]]
        if parsed_dataout == [[[parsed_data[i][1][0]], [''], [], [], [0], [''], [''], [], [0], [''], [''], [''], [''], [''], [0], [''], [''], [], [], [], '', [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]]:
            raise Exception
        print(parsed_dataout)
        if "Price Range" in range or "0" in lastd:
            with open(r"[REDACTED_BY_SCRIPT]", 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(parsed_dataout)
        tryCount=0
        i+=1
        time.sleep(10)
    except:
        if tryCount < 9:
            tryCount +=1
        else:
            i+=1
            tryCount=0
        
        if not driver:
            options = uc.ChromeOptions()
            options.add_argument('--disable-gpu')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            
            
            
            
            options.page_load_strategy = 'eager'
            time.sleep(1)
            driver = uc.Chrome(options=options)
            driverReset=0

async def main2():
    global driver
    global row
    global parsed_dataout
    global cookiecount
    """[REDACTED_BY_SCRIPT]"""
    driver = initialize_driver()
    await scrape_with_timeout2(300)
    driver.quit()  # Ensure the driver is closed after scraping




############################################################################################################################################
############################################################################################################################################
############################################################################################################################################
############################################################################################################################################

async def scrape_with_timeout3(timeout_seconds=300):  
    global driver
    global row
    global parsed_dataout
    global cookiecount
    global tryCount
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
    tryCount=0
    for i in range(len(parsed_data)):
        completedScrape = False
        retryattempt=0
        while completedScrape == False:
            if not driver:
                time.sleep(1)
                driver = initialize_driver()
            try:
                await asyncio.wait_for(asyncio.to_thread(scrape_page3,parsed_data,i), timeout=timeout_seconds)
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

# tryCount=0
# parsed_data = []
# with open(r"[REDACTED_BY_SCRIPT]", 'r', newline='', encoding='utf-8') as csvfile:
#     next(csvfile)
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


def scrape_page3(parsed_data,i):
    global driver
    global row
    global parsed_dataout
    global cookiecount
    global tryCount
    try:
        try:
            if not driver:
                options = uc.ChromeOptions()
                options.add_argument('--disable-gpu')
                options.add_argument('--no-sandbox')
                options.add_argument('--disable-dev-shm-usage')
                
                
                
                
                options.page_load_strategy = 'eager'
                time.sleep(1)
                driver = uc.Chrome(options=options)
        except:
            options = uc.ChromeOptions()
            options.add_argument('--disable-gpu')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            
            
            
            
            options.page_load_strategy = 'eager'
            time.sleep(1)
            driver = uc.Chrome(options=options)
        addressin=parsed_data[i][1][0]
        addressin=addressin.replace(" ", "-").replace(",","").lower()
        addressurl = "[REDACTED_BY_SCRIPT]"+addressin
        driver.get(addressurl)

        timeout=20
        try:
            if cookiecount == 0:
                element_present = EC.presence_of_element_located((By.CSS_SELECTOR, '[REDACTED_BY_SCRIPT]'))
                WebDriverWait(driver, timeout).until(element_present)
                driver.find_element(By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]").click()
                cookiecount+=1
            else:
                element_present = EC.presence_of_element_located((By.XPATH, '[REDACTED_BY_SCRIPT]'))
                WebDriverWait(driver, timeout).until(element_present)
        except:pass
        time.sleep(random.uniform(0.8, 1.5))
        

        #time.sleep(1000)
        estimPrice, floorarea, rent, pricechange, rentchange, mortgage, plotsize, plotsdata1, plotsdata2 = "", "", "", "", "", "", "", "", ""
        try:
            floorarea = driver.find_element(By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]").text
            floorarea=floorarea.replace(" ft2","")
            #print(floorarea)
        except:pass
        try:
            estimPrice = driver.find_element(By.XPATH, "[REDACTED_BY_SCRIPT]").text
            estimPrice=estimPrice.replace("£","").replace(",","")
            rent = driver.find_element(By.XPATH, "[REDACTED_BY_SCRIPT]").text
            pricechange = driver.find_element(By.XPATH, "[REDACTED_BY_SCRIPT]").text
            rentchange = driver.find_element(By.XPATH, "/html/body/div/section[2]/div[3]/div/div/div/div[2]/div[2]/div/div[2]/div[2]/div/div[2]").text
            #print(estimPrice)
        except:pass
        try:
            mortgage = driver.find_element(By.XPATH, "[REDACTED_BY_SCRIPT]").text
            #print(mortgage)
        except:pass
        try:
            plotsize = driver.find_element(By.XPATH, "[REDACTED_BY_SCRIPT]").text
            plotsdata1 = driver.find_element(By.XPATH, "[REDACTED_BY_SCRIPT]").text
            plotsdata2 = driver.find_element(By.XPATH, "[REDACTED_BY_SCRIPT]").text
            plotsize = plotsize[(plotsize.index("plot of ")+len("plot of ")):]
            plotsize = plotsize.replace("m2.","")
            plotsdata1=plotsdata1[(plotsdata1.index("[REDACTED_BY_SCRIPT]")+len("[REDACTED_BY_SCRIPT]")):]
            plotsdata1=plotsdata1[:plotsdata1.index(" ")]
            plotsdata2=plotsdata2[(plotsdata2.index("out of ")+len("out of ")):]
            plotsdata2=plotsdata2[:plotsdata2.index(" ")]
            #print(plotsize)
            #print(plotsdata1)
            #print(plotsdata2)
        except:pass
        schooldist1,schooldist2,schooldist3,schoolrating1,schoolrating2,schoolrating3,schooldist21,schooldist22,schooldist23,schoolrating21,schoolrating22,schoolrating23="","","","","","","","","","","",""
        for k in range(4):
            try:
                schooldist1 = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
                schooldist2 = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
                schooldist3 = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
                schoolrating1 = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
                schoolrating2 = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
                schoolrating3 = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
                if "According" not in schooldist1:
                    if " m" in schooldist1:schooldist1=schooldist1.replace(" m","")
                    else:schooldist1=schooldist1.replace(" km","")
                    if " m" in schooldist2:schooldist2=schooldist2.replace(" m","")
                    else:schooldist2=schooldist2.replace(" km","")
                    if " m" in schooldist3:schooldist3=schooldist3.replace(" m","")
                    else:schooldist3=schooldist3.replace(" km","")
                    if "Outstanding" in schoolrating1:schoolrating1= 1
                    elif "Good" in schoolrating1:schoolrating1= 2
                    elif "Requires improvemen" in schoolrating1:schoolrating1= 3
                    else: schoolrating1= 4
                    if "Outstanding" in schoolrating2:schoolrating2= 1
                    elif "Good" in schoolrating2:schoolrating2= 2
                    elif "Requires improvemen" in schoolrating2:schoolrating2= 3
                    else: schoolrating2= 4
                    if "Outstanding" in schoolrating3:schoolrating3= 1
                    elif "Good" in schoolrating3:schoolrating3= 2
                    elif "Requires improvemen" in schoolrating3:schoolrating3= 3
                    else: schoolrating3= 4
                    #print(schooldist1, schooldist2, schooldist3, schoolrating1, schoolrating2, schoolrating3)
                    break
            except:
                try:
                    schooldist1 = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
                    schooldist2 = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
                    schooldist3 = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
                    schoolrating1 = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
                    schoolrating2 = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
                    schoolrating3 = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
                    if "According" not in schooldist1:  
                        if " m" in schooldist1:schooldist1=schooldist1.replace(" m","")
                        else:schooldist1=schooldist1.replace(" km","")
                        if " m" in schooldist2:schooldist2=schooldist2.replace(" m","")
                        else:schooldist2=schooldist2.replace(" km","")
                        if " m" in schooldist3:schooldist3=schooldist3.replace(" m","")
                        else:schooldist3=schooldist3.replace(" km","")
                        if "Outstanding" in schoolrating1:schoolrating1= 1
                        elif "Good" in schoolrating1:schoolrating1= 2
                        elif "Requires improvemen" in schoolrating1:schoolrating1= 3
                        else: schoolrating1= 4
                        if "Outstanding" in schoolrating2:schoolrating2= 1
                        elif "Good" in schoolrating2:schoolrating2= 2
                        elif "Requires improvemen" in schoolrating2:schoolrating2= 3
                        else: schoolrating2= 4
                        if "Outstanding" in schoolrating3:schoolrating3= 1
                        elif "Good" in schoolrating3:schoolrating3= 2
                        elif "Requires improvemen" in schoolrating3:schoolrating3= 3
                        else: schoolrating3= 4
                        #print(schooldist1, schooldist2, schooldist3, schoolrating1, schoolrating2, schoolrating3)
                        break
                except:pass
        for k in range(3):
            try:
                schooldist21 = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
                schooldist22 = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
                schooldist23 = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
                schoolrating21 = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
                schoolrating22 = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
                schoolrating23 = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
                if "According" not in schooldist21:
                    if " m" in schooldist21:schooldist21=schooldist21.replace(" m","")
                    else:schooldist21=schooldist21.replace(" km","")
                    if " m" in schooldist22:schooldist22=schooldist22.replace(" m","")
                    else:schooldist22=schooldist22.replace(" km","")
                    if " m" in schooldist23:schooldist23=schooldist23.replace(" m","")
                    else:schooldist23=schooldist23.replace(" km","")
                    if "Outstanding" in schoolrating21:schoolrating21= 1
                    elif "Good" in schoolrating21:schoolrating21= 2
                    elif "Requires improvemen" in schoolrating21:schoolrating21= 3
                    else: schoolrating21= 4
                    if "Outstanding" in schoolrating22:schoolrating22= 1
                    elif "Good" in schoolrating22:schoolrating22= 2
                    elif "Requires improvemen" in schoolrating22:schoolrating22= 3
                    else: schoolrating22= 4
                    if "Outstanding" in schoolrating23:schoolrating23= 1
                    elif "Good" in schoolrating23:schoolrating23= 2
                    elif "Requires improvemen" in schoolrating23:schoolrating23= 3
                    else: schoolrating23= 4

                    #print(schooldist21, schooldist22, schooldist23, schoolrating21, schoolrating22, schoolrating23)
                    break
            except:
                try:
                    schooldist21 = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
                    schooldist22 = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
                    schooldist23 = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
                    schoolrating21 = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
                    schoolrating22 = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
                    schoolrating23 = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
                    if "According" not in schooldist21:
                        if " m" in schooldist21:schooldist21=schooldist21.replace(" m","")
                        else:schooldist21=schooldist21.replace(" km","")
                        if " m" in schooldist22:schooldist22=schooldist22.replace(" m","")
                        else:schooldist22=schooldist22.replace(" km","")
                        if " m" in schooldist23:schooldist23=schooldist23.replace(" m","")
                        else:schooldist23=schooldist23.replace(" km","")
                        if "Outstanding" in schoolrating21:schoolrating21= 1
                        elif "Good" in schoolrating21:schoolrating21= 2
                        elif "Requires improvemen" in schoolrating21:schoolrating21= 3
                        else: schoolrating21= 4
                        if "Outstanding" in schoolrating22:schoolrating22= 1
                        elif "Good" in schoolrating22:schoolrating22= 2
                        elif "Requires improvemen" in schoolrating22:schoolrating22= 3
                        else: schoolrating22= 4
                        if "Outstanding" in schoolrating23:schoolrating23= 1
                        elif "Good" in schoolrating23:schoolrating23= 2
                        elif "Requires improvemen" in schoolrating23:schoolrating23= 3
                        else: schoolrating23= 4


                        #print(schooldist21, schooldist22, schooldist23, schoolrating21, schoolrating22, schoolrating23)
                        break
                except:pass
        brdbndStore=0
        for k in range(3):
            try:
                traindistance1 = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
                traindistance2 = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
                traindistance3 = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
                brdbndStore=k
                break
            except:
                try:
                    traindistance1 = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
                    traindistance2 = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
                    traindistance3 = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
                    brdbndStore=k
                    break
                except:pass
        if " m" in traindistance1:traindistance1=traindistance1.replace(" m","")
        else:traindistance1=traindistance1.replace(" km","")
        if " m" in traindistance2:traindistance2=traindistance2.replace(" m","")
        else:traindistance2=traindistance2.replace(" km","")
        if " m" in traindistance3:traindistance3=traindistance3.replace(" m","")
        else:traindistance3=traindistance3.replace(" km","")
        #print(traindistance1, traindistance2, traindistance3)

        broadband=""
        for k in range(4-brdbndStore):
            for j in range(4):
                try:
                    trybroadband = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]")
                    broadband=j+1
                    break
                except:
                    try:
                        trybroadband = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]")
                        broadband=j+1
                        break
                    except:pass
        #print(broadband)
        houseList=[]
        educationList=[]
        #
        #
        #body > div.bg-white > section:nth-child(28) > div.relative.z-1 > div > div > div > div.CardBody > div:nth-child(2) > div > div:nth-child(1) > div.flex-\[5\] > div > div:nth-child(1) > div > div.text-bold
        #body > div.bg-white > section:nth-child(28) > div.relative.z-1 > div > div > div > div.CardBody > div:nth-child(2) > div > div:nth-child(1) > div.flex-\[5\] > div > div:nth-child(2) > div > div.text-bold
        for j in range(3):
            for k in range(5):
                try:
                    houseinfo = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]")
                    houseinfo = driver.execute_script("[REDACTED_BY_SCRIPT]", houseinfo)
                    if k<4:
                        eduinfo = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]")
                        eduinfo = driver.execute_script("[REDACTED_BY_SCRIPT]", eduinfo)
                except:
                    try:
                        houseinfo = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]")
                        houseinfo = driver.execute_script("[REDACTED_BY_SCRIPT]", houseinfo)
                        if k<4:
                            eduinfo = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]")
                            eduinfo = driver.execute_script("[REDACTED_BY_SCRIPT]", eduinfo)
                    except:
                        try:
                            houseinfo = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]")
                            houseinfo = driver.execute_script("[REDACTED_BY_SCRIPT]", houseinfo)
                            if k<4:
                                eduinfo = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]")
                                eduinfo = driver.execute_script("[REDACTED_BY_SCRIPT]", eduinfo)
                        except:pass
               
                houseList.append(houseinfo)
                educationList.append(eduinfo)
        #print(houseList,educationList)
        avgAge,modeAge="",""
        for k in range(4):
            try:
                avgAge = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
                modeAge = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
                break
            except:
                try:
                    avgAge = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
                    modeAge = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
                    break
                except:pass
        #print(avgAge)
        modeAge=modeAge[(modeAge.index("was ")+4):]
        modeAge=modeAge[:modeAge.index(" and")]
        #print(modeAge)

        secondryHouse,carinfo="",""
        for k in range(4):
            try:
                secondryHouse = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
                carinfo = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
                break
                #/html/body/div[1]/section[18]/div[3]/div/div/div/div[3]/div[2]/div/div[1]/div/div[1]
                #/html/body/div[1]/section[19]/div[3]/div/div/div/div[3]/div[2]/div/div[1]/div/div[1]
            except:
                try:
                    secondryHouse = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
                    carinfo = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
                    break
                except:pass
        secondryHouse=secondryHouse[:secondryHouse.index("%")]
        #print(secondryHouse,carinfo)

        quality = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
        if "High" in quality:quality=quality.replace("High","3")
        elif "Medium" in quality:quality=quality.replace("Medium","2")
        else:quality="1"
        quality2 = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
        quality2=quality2.replace(" / 5","")
        #print(quality,quality2)
        if "[REDACTED_BY_SCRIPT]" in plotsize:
            parsed_dataout =[[parsed_data[i][1][0], floorarea, estimPrice, mortgage, "", "", "", schooldist1, schooldist2, schooldist3, schoolrating1, schoolrating2, schoolrating3, schooldist21, schooldist22, schooldist23, schoolrating21, schoolrating22, schoolrating23, traindistance1, traindistance2, traindistance3, houseList, educationList, avgAge, modeAge, secondryHouse, carinfo, quality, quality2]]
        else:
            parsed_dataout =[[parsed_data[i][1][0], floorarea, estimPrice, mortgage, plotsize, plotsdata1, plotsdata2, schooldist1, schooldist2, schooldist3, schoolrating1, schoolrating2, schoolrating3, schooldist21, schooldist22, schooldist23, schoolrating21, schoolrating22, schoolrating23, traindistance1, traindistance2, traindistance3, houseList, educationList, avgAge, modeAge, secondryHouse, carinfo, quality, quality2]]
        with open(r"[REDACTED_BY_SCRIPT]", 'a', newline='', encoding='utf-8') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(parsed_dataout)
        time.sleep(5)
    except:
        if tryCount < 3:
            i-=1
            tryCount +=1
        elif not driver:
            options = uc.ChromeOptions()
            options.add_argument('--disable-gpu')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            
            
            
            
            options.page_load_strategy = 'eager'
            time.sleep(1)
            driver = uc.Chrome(options=options)
        else:pass

async def main3():
    global driver
    global row
    global parsed_dataout
    global cookiecount
    """[REDACTED_BY_SCRIPT]"""
    driver = initialize_driver()
    await scrape_with_timeout3(300)
    driver.quit()  # Ensure the driver is closed after scraping







############################################################################################################################################
############################################################################################################################################
############################################################################################################################################
############################################################################################################################################






async def scrape_with_timeout4(timeout_seconds=300):  
    global driver
    global row
    global parsed_dataout
    global cookiecount
    global tryCount
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
    tryCount=0
    for i in range(len(parsed_data)):
        completedScrape = False
        retryattempt=0
        while completedScrape == False:
            if not driver:
                time.sleep(1)
                driver = initialize_driver()
            try:
                await asyncio.wait_for(asyncio.to_thread(scrape_page4,parsed_data,i), timeout=timeout_seconds)
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


# parsed_data = []
# with open(r"[REDACTED_BY_SCRIPT]", 'r', newline='', encoding='utf-8') as csvfile:
#     next(csvfile)
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

cookiecount=0
def scrape_page4(parsed_data,i):
    global driver
    global row
    global parsed_dataout
    global cookiecount
    global tryCount
    try:
        try:
            if not driver:
                options = uc.ChromeOptions()
                options.add_argument('--disable-gpu')
                options.add_argument('--no-sandbox')
                time.sleep(1)
                driver = uc.Chrome(options=options)
        except:
            options = uc.ChromeOptions()
            options.add_argument('--disable-gpu')
            options.add_argument('--no-sandbox')
            time.sleep(1)
            driver = uc.Chrome(options=options)
        addressin=parsed_data[i][1][0]
        addressin=addressin.split(",")
        if len(addressin)==3:
            addressin2=addressin[1][1:].replace(" ","-")+"/"+addressin[0][addressin[0].index(" ")+1:].replace(" ","+")+"/"+addressin[2][1:].replace(" ","+")+"/"+addressin[0].replace(" ","+")+"/"
            addressin2=addressin2.lower()
        elif len(addressin)==4:
            addressin2=addressin[2][1:].replace(" ","-")+addressin[1][1:].replace(" ","-")+"/"+addressin[0][addressin[0].index(" ")+1:].replace(" ","+")+"/"+addressin[3][1:].replace(" ","+")+"/"+addressin[0].replace(" ","+")+"/"
            addressin2=addressin2.lower()

        #30 Oakfield Drive, Killingworth, Newcastle upon Tyne, NE12 6YY
        #newcastle+upon+tyne/killingworth/oakfield+drive/ne12+6yy/30+oakfield+drive

        #108 Chilside Road, Gateshead, NE10 9EA
        #/gateshead/felling/chilside+road/ne10+9ea/108+chilside+road
        addressurl = "[REDACTED_BY_SCRIPT]"+addressin2
        driver.get(addressurl)
        # script = """
        # // Select all <img> tags
        # const images = document.querySelectorAll('img');
        # images.forEach(img => {
        #     // Remove the <img> element from the DOM
        #     img.remove();
        # });

        # // Optionally, log a message to confirm the script ran
        # console.log('[REDACTED_BY_SCRIPT]');
        # """
        # driver.execute_script(script)

        timeout=30
        if cookiecount==0:
            shadow_host = driver.find_element(By.CSS_SELECTOR, "#cmpwrapper")
            shadow_root = driver.execute_script("[REDACTED_BY_SCRIPT]", shadow_host)

            wait = WebDriverWait(shadow_root, timeout)
            element_locator = (By.CSS_SELECTOR, '[REDACTED_BY_SCRIPT]')
            element = wait.until(EC.presence_of_element_located(element_locator))
            shadow_root.find_element(By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]").click()
            cookiecount=1

        try:
            estimPrice0 = driver.find_element(By.XPATH, "[REDACTED_BY_SCRIPT]").text
            estimPrice0= estimPrice0.replace("£","").replace(",","")
        except:
            try:
                estimPrice0 = driver.find_element(By.XPATH, "[REDACTED_BY_SCRIPT]").text
                estimPrice0= estimPrice0.replace("£","").replace(",","")
            except:
                estimPrice0 = ""

        try:#                                            /html/body/div[5]/div[2]/div[5]/div[2]/div[1]/span[1]
            confidence0 = driver.find_element(By.XPATH, "[REDACTED_BY_SCRIPT]").text
            confidence0=confidence0.replace("We have ","")
            confidence0=confidence0[:(confidence0.index(" "))]
        except:
            try:
                confidence0 = driver.find_element(By.XPATH, "[REDACTED_BY_SCRIPT]").text
                confidence0=confidence0.replace("We have ","")
                confidence0=confidence0[:(confidence0.index(" "))]
            except:
                confidence0 = 0

        try:
            rent0 = driver.find_element(By.XPATH, "[REDACTED_BY_SCRIPT]").text
            rent0 = rent0.replace("Or ", "")
            rent0 = rent0[:rent0.index(" ")]
            try:
                rent0 = driver.find_element(By.XPATH, "[REDACTED_BY_SCRIPT]").text
                rent0 = rent0.replace("Or ", "")
                rent0 = rent0[:rent0.index(" ")]
            except:
                rent0 = ""
        except:
            rent0 = ""

        data000=""
        data001=""
        data00=[]
        data01=[]
        housetype=""
        for j in range(10):
            try:
                data000 = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
                data001 = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
                data002 = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").get_attribute('src')
                if data002 == "[REDACTED_BY_SCRIPT]":
                    housetype = data000.replace("\n","").replace("\t","")
                elif data002 == "[REDACTED_BY_SCRIPT]":
                    data000 = data000.replace(" per interior metre","").replace("£","")
                    try:
                        data000 = data000.replace(",","")
                    except:pass
                elif data002 == "[REDACTED_BY_SCRIPT]":
                    data000 = data000.replace(" sqm plot","")
                    try:
                        data000 = data000.replace(",","")
                    except:pass
                if data002 == "[REDACTED_BY_SCRIPT]":
                    data000 = data000[-1:]
                if data002 == "[REDACTED_BY_SCRIPT]":
                    data000 = data000[0]
                    beds = data000
                if data002 == "[REDACTED_BY_SCRIPT]":
                    data000 = data000.replace(" sqm floor area","")
                    try:
                        data000 = data000.replace(",","")
                    except:pass
                    SqFt=data000
                if data002 == "[REDACTED_BY_SCRIPT]":
                    data000 = data000
                if data002 == "[REDACTED_BY_SCRIPT]":
                    data000 = data000[0]
                
                data00.append(data000)
                data01.append(data001)
            except:
                try:
                    data000 = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
                    data001 = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
                    data002 = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").get_attribute('src')
                    if data002 == "[REDACTED_BY_SCRIPT]":
                        housetype = data000.replace("\n","").replace("\t","")
                    elif data002 == "[REDACTED_BY_SCRIPT]":
                        data000 = data000.replace(" per interior metre","").replace("£","")
                        try:
                            data000 = data000.replace(",","")
                        except:pass
                    elif data002 == "[REDACTED_BY_SCRIPT]":
                        data000 = data000.replace(" sqm plot","")
                        try:
                            data000 = data000.replace(",","")
                        except:pass
                    if data002 == "[REDACTED_BY_SCRIPT]":
                        data000 = data000[-1:]
                    if data002 == "[REDACTED_BY_SCRIPT]":
                        data000 = data000[0]
                        beds = data000
                    if data002 == "[REDACTED_BY_SCRIPT]":
                        data000 = data000.replace(" sqm floor area","")
                        try:
                            data000 = data000.replace(",","")
                        except:pass
                        SqFt=data000
                    if data002 == "[REDACTED_BY_SCRIPT]":
                        data000 = data000
                    if data002 == "[REDACTED_BY_SCRIPT]":
                        data000 = data000[0]
                    
                    data00.append(data000)
                    data01.append(data001)
                except:pass

        #print(data00)
        #print(data01)
        data000=""
        data001=""
        data10=[]
        data11=[]

        for j in range(10):
            try:
                data000 = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
                data001 = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
                data002 = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").get_attribute('src')
                #print(data002)
                if data002 == "[REDACTED_BY_SCRIPT]":
                    housetype = data000
                elif data002 == "[REDACTED_BY_SCRIPT]":
                    data000 = data000.replace(" per interior metre","").replace("£","")
                    try:
                        data000 = data000.replace(",","")
                    except:pass
                elif data002 == "[REDACTED_BY_SCRIPT]":
                    data000 = data000.replace(" sqm plot","")
                    try:
                        data000 = data000.replace(",","")
                    except:pass
                if data002 == "[REDACTED_BY_SCRIPT]":
                    data000 = data000[-1:]
                if data002 == "[REDACTED_BY_SCRIPT]":
                    data000 = data000[0]
                    beds = data000
                if data002 == "[REDACTED_BY_SCRIPT]":
                    data000 = data000.replace(" sqm floor area","")
                    try:
                        data000 = data000.replace(",","")
                    except:pass
                    SqFt=data000
                if data002 == "[REDACTED_BY_SCRIPT]":
                    data000 = data000
                if data002 == "[REDACTED_BY_SCRIPT]":
                    data000 = data000[0]
                data10.append(data000)
                data11.append(data001)
            except:
                try:
                    data000 = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
                    data001 = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
                    data002 = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").get_attribute('src')
                    #print(data002)
                    if data002 == "[REDACTED_BY_SCRIPT]":
                        housetype = data000
                    elif data002 == "[REDACTED_BY_SCRIPT]":
                        data000 = data000.replace(" per interior metre","").replace("£","")
                        try:
                            data000 = data000.replace(",","")
                        except:pass
                    elif data002 == "[REDACTED_BY_SCRIPT]":
                        data000 = data000.replace(" sqm plot","")
                        try:
                            data000 = data000.replace(",","")
                        except:pass
                    if data002 == "[REDACTED_BY_SCRIPT]":
                        data000 = data000[-1:]
                    if data002 == "[REDACTED_BY_SCRIPT]":
                        data000 = data000[0]
                        beds = data000
                    if data002 == "[REDACTED_BY_SCRIPT]":
                        data000 = data000.replace(" sqm floor area","")
                        try:
                            data000 = data000.replace(",","")
                        except:pass
                        SqFt=data000
                    if data002 == "[REDACTED_BY_SCRIPT]":
                        data000 = data000
                    if data002 == "[REDACTED_BY_SCRIPT]":
                        data000 = data000[0]
                    data10.append(data000)
                    data11.append(data001)
                except:pass
        #print(data10)
        #print(data11)
        data000=""
        data001=""
        data20=[]
        data21=[]
        for j in range(10):
            try:
                data000 = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
                data001 = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
                data20.append(data000)
                data21.append(data001)
            except:
                try:
                    data000 = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
                    data001 = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
                    data20.append(data000)
                    data21.append(data001)
                except:pass
        #print(data20)
        #print(data21)

        #print([[parsed_data[i][1][0],confidence0, estimPrice0, rent0, data00, data01, housetype, data10, data11, data20, data21]])
        parsed_dataout =[[parsed_data[i][1][0],confidence0, estimPrice0, rent0, data00, data01, housetype, data10, data11, data20, data21]]
        with open(r"[REDACTED_BY_SCRIPT]", 'a', newline='', encoding='utf-8') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(parsed_dataout)
        time.sleep(5)
    except:
        if tryCount < 3:
            i-=1
            tryCount +=1
        elif not driver:
            options = uc.ChromeOptions()
            options.add_argument('--disable-gpu')
            options.add_argument('--no-sandbox')
            time.sleep(1)
            driver = uc.Chrome(options=options)
        else:pass




async def main4():
    global driver
    global row
    global parsed_dataout
    global cookiecount
    """[REDACTED_BY_SCRIPT]"""
    driver = initialize_driver()
    await scrape_with_timeout4(300)
    driver.quit()  # Ensure the driver is closed after scraping








############################################################################################################################################
############################################################################################################################################
############################################################################################################################################
############################################################################################################################################






async def scrape_with_timeout5(timeout_seconds=300):  
    global driver
    global row
    global parsed_dataout
    global cookiecount
    global tryCount
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
    tryCount=0
    for i in range(len(parsed_data)):
        completedScrape = False
        retryattempt=0
        while completedScrape == False:
            if not driver:
                time.sleep(1)
                driver = initialize_driver()
            try:
                await asyncio.wait_for(asyncio.to_thread(scrape_page5,parsed_data,i), timeout=timeout_seconds)
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


cookiecount=0
def scrape_page5(parsed_data,i):
    global driver
    global row
    global parsed_dataout
    global cookiecount
    global tryCount
    try:
        try:
            if not driver:
                options = uc.ChromeOptions()
                options.add_argument('--disable-gpu')
                options.add_argument('--no-sandbox')
                time.sleep(1)
                driver = uc.Chrome(options=options)
        except:
            options = uc.ChromeOptions()
            options.add_argument('--disable-gpu')
            options.add_argument('--no-sandbox')
            time.sleep(1)
            driver = uc.Chrome(options=options)
        #print("1")
        addressin=parsed_data[i][0][0]
        parsed_data2 = []
        with open(r"[REDACTED_BY_SCRIPT]", 'r', newline='', encoding='utf-8') as csvfile2:
            csv_reader2 = csv.reader(csvfile2)
            #print("2")
            for row2 in csv_reader2:
                parsed_row2 = []
                #print("3")
                for cell in row2:
                    try:
                        # Convert the string representation of a list to an actual list
                        parsed_cell = ast.literal_eval(cell.strip())
                    except (SyntaxError, ValueError):
                        # Fallback to raw string if parsing fails
                        parsed_cell = cell.strip()
                    parsed_row2.append(parsed_cell)
                #print("4")
                #print(parsed_row2)
                addressChim=str(parsed_row2[0])+" "+str(parsed_row2[1])+" "+str(parsed_row2[2])+" "+str(parsed_row2[3])+" "+str(parsed_row2[4])+" "+str(parsed_row2[5])
                addressChim1=addressChim.replace(" ","").lower()
                addressin1=addressin.replace(" ","").lower()
                #print(addressChim1,addressin1)
                if addressChim1 == addressin1:
                    addressurl=parsed_row2[6]
                    break
        driver.get(addressurl)
        timeout=30
        chimnieaddressInfo=["","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","",""]
        element_present = EC.presence_of_element_located((By.XPATH, "/html/body/div[1]/div[1]/div[4]/div/div[1]/div[1]/div/div/div[1]/h3"))
        WebDriverWait(driver, timeout).until(element_present)
        estimPrice = driver.find_element(By.XPATH, f"/html/body/div[1]/div[1]/div[4]/div/div[1]/div[1]/div/div/div[1]/h3").text
        estimPrice=estimPrice.replace("£","").replace("K","000")
        estimPrice=estimPrice.split("-")
        chimnieaddressInfo[0]=estimPrice

        for i in range(4):
            try:
                svg = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]")
                path_element = svg.find_element(By.TAG_NAME, "path")
                d_attribute = path_element.get_attribute("d")
                print(d_attribute[:3])
                if d_attribute[:3] == "M48":
                    beds = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
                    chimnieaddressInfo[1]=beds
                elif d_attribute[:3] == "M41":
                    sqft = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
                    sqft=sqft.replace(" sqft","")
                    sqm=str(int(int(sqft)*0.09290304))
                    chimnieaddressInfo[3]=sqm
                elif d_attribute[:3] == "M57":
                    types = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
                    if "Detached" in types:types=4
                    elif "Semi-Detached" in types:types=3
                    elif "Terraced" in types:types=2
                    else:types=1
                    chimnieaddressInfo[4]=types
            except:pass

        try:
            bedsvg = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]")
            path_element = bedsvg.find_element(By.TAG_NAME, "path")
            d_attribute = path_element.get_attribute("d")
            print(d_attribute[:3])
        except:pass
        try:
            bathsvg = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]")
            path_element = bathsvg.find_element(By.TAG_NAME, "path")
            d_attribute = path_element.get_attribute("d")
            print(d_attribute[:3])
        except:pass
        try:
            sqftsvg = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]")
            path_element = sqftsvg.find_element(By.TAG_NAME, "path")
            d_attribute = path_element.get_attribute("d")
            print(d_attribute[:3])
        except:pass
        try:
            typesvg = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]")
            path_element = typesvg  .find_element(By.TAG_NAME, "path")
            d_attribute = path_element.get_attribute("d")
            print(d_attribute[:3])
            type = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
            if "Detached" in type:type=4
            elif "Semi-Detached" in type:type=3
            elif "Terraced" in type:type=2
            else:type=1
            chimnieaddressInfo[5]=type
        except:pass
        try:
            lease = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
            if "Freehold" in lease:lease=2
            else:lease=1
        except:lease=0
        chimnieaddressInfo[6]=lease
        try:
            counciltax = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
        except:
            counciltax=""
        chimnieaddressInfo[7]=counciltax
        try:
            ctaxcost = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
        except:
            ctaxcost=""
        chimnieaddressInfo[8]=ctaxcost
        try:
            epc = driver.find_element(By.XPATH, f"/html/body/div[1]/div[1]/div[4]/div/div[1]/div[5]/div[1]/div").text
            epc=epc.split("\n")
            epc[1]=epc[1].replace("(Potential: )","").replace(")","")
            if "A" in epc[0]:epc[0]="6"
            elif "B" in epc[0]:epc[0]="5"
            elif "C" in epc[0]:epc[0]="4"
            elif "D" in epc[0]:epc[0]="3"
            elif "E" in epc[0]:epc[0]="2"
            elif "F" in epc[0]:epc[0]="1"
            else:epc[0]="0"
            if "A" in epc[1]:epc[1]="6"
            elif "B" in epc[1]:epc[1]="5"
            elif "C" in epc[1]:epc[1]="4"
            elif "D" in epc[1]:epc[1]="3"
            elif "E" in epc[1]:epc[1]="2"
            elif "F" in epc[1]:epc[1]="1"
            else:epc[1]="0"
        except:
            epc=""
        chimnieaddressInfo[9]=epc
        try:
            bbandspeed = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
        except:
            bbandspeed=""
        chimnieaddressInfo[10]=bbandspeed
        try:
            builtsurface = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
        except:
            builtsurface=""
        try:
            builtsurface2 = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
            builtsurface=[builtsurface,builtsurface2]
        except:
            builtsurface=[builtsurface,""]
        chimnieaddressInfo[11]=builtsurface

        for i in range(3):
            try:
                lscore1 = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
                chimnieaddressInfo[((i+1)*10)+2+i] = lscore1
            except:pass
            try:
                escore1 = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
                chimnieaddressInfo[((i+1)*10)+3+i] = escore1
            except:pass
            try:# 
                sscore1 = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text   
                chimnieaddressInfo[((i+1)*10)+4+i] = sscore1
            except:pass
            try:
                fscore1 = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
                chimnieaddressInfo[((i+1)*10)+5+i] = fscore1
            except:pass
            try:
                cscore1 = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
                chimnieaddressInfo[((i+1)*10)+6+i] = cscore1
            except:pass
            try:
                l2score1 = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
                chimnieaddressInfo[((i+1)*10)+7+i] = l2score1
            except:pass
            try:
                e2score1 = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
                chimnieaddressInfo[((i+1)*10)+8+i] = e2score1
            except:pass
            try:
                s2score1 = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
                chimnieaddressInfo[((i+1)*10)+9+i] = s2score1
            except:pass
            try:
                f2score1 = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
                chimnieaddressInfo[((i+1)*10)+10+i] = f2score1
            except:pass
            try:
                c2score1 = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
                chimnieaddressInfo[((i+1)*10)+11+i] = c2score1
            except:pass
            print(chimnieaddressInfo,i)

            # time.sleep(random.uniform(1.2, 1.5))
            # try:
            #     driver.find_element(By.XPATH, "[REDACTED_BY_SCRIPT]").click()
            #     cleanair = driver.find_element(By.XPATH, f"/html/body/div[1]/div[1]/div[4]/div/div[1]/div[9]/div[6]/div/div/div/div/div/p").text
            #     chimnieaddressInfo[((i+1)*10)+12+i] = cleanair
            

            if i == 0:#                        /html/body/div[1]/div[1]/div[4]/div/div[1]/div[9]/div[1]/div[2]/div/div/div/button[2]
                driver.find_element(By.XPATH, "[REDACTED_BY_SCRIPT]").click()
                time.sleep(random.uniform(1.2, 1.5))
            elif i == 1:#                      /html/body/div[1]/div[1]/div[4]/div/div[1]/div[9]/div[1]/div[2]/div/div/div/button[3]
                driver.find_element(By.XPATH, "[REDACTED_BY_SCRIPT]").click()
                time.sleep(random.uniform(1.2, 1.5))


        #print(chimnieaddressInfo)
        chimnieaddressInfo=[addressin]+chimnieaddressInfo
        with open(r"[REDACTED_BY_SCRIPT]", 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([chimnieaddressInfo])
        
        time.sleep(5)
    except Exception as e:
        print(f"An error occurred: {e}")
        if tryCount < 3:
            i-=1
            tryCount +=1
        elif not driver:
            options = uc.ChromeOptions()
            options.add_argument('--disable-gpu')
            options.add_argument('--no-sandbox')
            time.sleep(1)
            driver = uc.Chrome(options=options)
        else:pass




async def main5():
    global driver
    global row
    global parsed_dataout
    global cookiecount
    """[REDACTED_BY_SCRIPT]"""
    driver = initialize_driver()
    await scrape_with_timeout5(300)
    driver.quit()  # Ensure the driver is closed after scraping










if __name__ == "__main__":
    # asyncio.run(main())
    # time.sleep(5)
    # asyncio.run(main2())
    # time.sleep(5)
    # asyncio.run(main3())
    # time.sleep(5)
    # asyncio.run(main4())
    # time.sleep(5)
    asyncio.run(main5())