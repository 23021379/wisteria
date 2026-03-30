from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.keys import Keys
#from selenium.webdriver.common.devtools import DevTools
from webdriver_manager.chrome import ChromeDriverManager
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
import tempfile
import shutil
import os
from fake_useragent import UserAgent

def create_temporary_chromedriver(original_chromedriver_path):
    """
    Creates a temporary copy of ChromeDriver and returns its path.
    
    Args:
        original_chromedriver_path: Path to the original ChromeDriver executable
        
    Returns:
        Path to the temporary ChromeDriver copy
    """
    # Create a temporary directory that will be automatically cleaned up
    temp_dir = tempfile.mkdtemp()
    
    # Create a copy of the ChromeDriver in the temporary directory
    temp_chromedriver_path = os.path.join(temp_dir, "chromedriver.exe")
    shutil.copy2(original_chromedriver_path, temp_chromedriver_path)
    
    # Return the path to the temporary ChromeDriver and the temp directory
    return temp_chromedriver_path, temp_dir


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

# Path to your original ChromeDriver
original_chromedriver = r"[REDACTED_BY_SCRIPT]"

# Create a temporary copy
temp_chromedriver_path, temp_dir = create_temporary_chromedriver(original_chromedriver)

"""
edit init
"""
def initialize_driver():
    global driver
    global loopcount
    global retryattempt
    global temp_chromedriver_path
    global start_whcih_website
    
    if start_whcih_website == "bricksandlogic":
        try:
            options = uc.ChromeOptions()
            options.add_argument('--disable-gpu')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            ua = UserAgent(browsers=['chrome'])
            user_agent = ua.random  # get a random UA
            options.add_argument(f'[REDACTED_BY_SCRIPT]')  # set it
            #options.add_argument('--headless')

            prefs = {"[REDACTED_BY_SCRIPT]": 2}
            options.add_experimental_option("prefs", prefs)
            
            # Use the temporary ChromeDriver
            driver = uc.Chrome(driver_executable_path=temp_chromedriver_path, options=options)
            
            return driver  # Return the driver and temp directory for cleanup later
        except Exception as e:
            # Clean up temp directory on error
            shutil.rmtree(temp_dir)
            raise e
    elif start_whcih_website == "homipi":
        try:
            options = uc.ChromeOptions()
            options.add_argument('--disable-gpu')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            ua = UserAgent(browsers=['chrome'])
            user_agent = ua.random  # get a random UA
            options.add_argument(f'[REDACTED_BY_SCRIPT]')  # set it
            #options.add_argument('--headless')

            prefs = {"[REDACTED_BY_SCRIPT]": 2}
            options.add_experimental_option("prefs", prefs)
            
            # Use the temporary ChromeDriver
            driver = uc.Chrome(driver_executable_path=temp_chromedriver_path, options=options)
            
            return driver  # Return the driver and temp directory for cleanup later
        except Exception as e:
            # Clean up temp directory on error
            shutil.rmtree(temp_dir)
            raise e
    else:
        try:
            options = uc.ChromeOptions()
            options.add_argument('--disable-gpu')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            ua = UserAgent(browsers=['chrome'])
            user_agent = ua.random  # get a random UA
            options.add_argument(f'[REDACTED_BY_SCRIPT]')  # set it
            #options.add_argument('--headless')

            prefs = {"[REDACTED_BY_SCRIPT]": 2}
            options.add_experimental_option("prefs", prefs)
            
            # Use the temporary ChromeDriver
            driver = uc.Chrome(driver_executable_path=temp_chromedriver_path, options=options)
            
            return driver  # Return the driver and temp directory for cleanup later
        except Exception as e:
            # Clean up temp directory on error
            shutil.rmtree(temp_dir)
            raise e

loopcount=0
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
    global completedScrape
    parsed_data = []
    parsed_dataout=[]
    with open(r"[REDACTED_BY_SCRIPT]", 'r', newline='', encoding='utf-8') as csvfile:
        csv_reader = csv.reader(csvfile)
        for jk in range(409):
            next(csv_reader)
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
                await asyncio.wait_for(asyncio.to_thread(scrape_page2,parsed_data,i), timeout=timeout_seconds)
                if completedScrape != True:
                    raise ValueError("[REDACTED_BY_SCRIPT]")
            except asyncio.TimeoutError:
                print("[REDACTED_BY_SCRIPT]")
                try:
                    driver.quit()  # Quit the current driver
                except:
                    pass
            except ValueError as e:
                if retryattempt < 3:
                    retryattempt += 1
                else:
                    completedScrape = True
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
    global temp_chromedriver_path
    global completedScrape
    
    #for loop in range(len(parsed_data)):
    if not driver:
        time.sleep(1)
        driver = initialize_driver()
        driverReset=1
    try:
        addressin=parsed_data[i][1][0]
        addressin2=addressin.split(" ")[-2]+"-"+addressin.split(" ")[-1]
        addressurl = "[REDACTED_BY_SCRIPT]"+addressin2+"?page=1"
        driver.get(addressurl)
        try:
            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]")))
            number_of_props=driver.find_element(By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]").text
            print(number_of_props)
        except:
            number_of_props="There are 10 properties"
        try:
            number_of_props=number_of_props[(number_of_props.index("There are ") + len("There are ")):(number_of_props.index(" properties"))]
        except:
            try:
                WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]")))
                number_of_props=driver.find_element(By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]").text
                print(number_of_props)
            except:
                number_of_props="There are 10 properties"
            try:
                number_of_props=number_of_props[(number_of_props.index("There are ") + len("There are ")):(number_of_props.index(" properties"))]
            except Exception as e:
                print(e)
        print(number_of_props)
        number_of_props=int(number_of_props)
        compare_addresses=[]
        compare_addresses_href=[]
        try:
            if (math.ceil(int(number_of_props)/10)-1)==0:
                blibidi=1
                blibidi_alert=True
            else:
                blibidi=math.ceil(int(number_of_props)/10)-1
                blibidi_alert=False
            for jk in range(blibidi):
                for kl in range(1,10):
                    try:
                        address_homipi=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]")
                        href_homipi=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]")
                        compare_addresses.append(address_homipi.text)
                        compare_addresses_href.append(href_homipi.get_attribute("href"))
                    except:
                        pass
                if blibidi_alert==False:
                    addressurl = "[REDACTED_BY_SCRIPT]"+addressin2+f"?page={jk+2}"
                    driver.get(addressurl)
        except:pass
        
        target_address = addressin
        most_similar = None
        most_similar_href=None
        min_distance = float('inf')  # Initialize with a very large distance

        for entry in compare_addresses:
            distance = Levenshtein.distance(target_address.lower().replace(",","").replace("-","").replace(" ","").replace(".",""), entry.lower().replace(",","").replace("-","").replace(" ","").replace(".",""))
            #print(target_address, entry, distance)
            if distance < min_distance:
                min_distance = distance
                most_similar = entry
                most_similar_href=compare_addresses_href[compare_addresses.index(entry)]

        if most_similar:
            addressurl=most_similar_href
        else:
            addressurl=""
        driver.get(addressurl)

        if driverReset == 0:
            try:
                element_present = EC.presence_of_element_located((By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]"))
                WebDriverWait(driver, 5).until(element_present)
                driver.find_element(By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]").click()
            except:
                pass
            driverReset=1

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
            range0 = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
            range0 = range.split("\n")
        except:
            range0 = []

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

        
        parsed_dataout=[[[parsed_data[i][1][0]], [est], change, range0, [conf], [lastp], [lastd], type, [subtype], [beds], [receps], [extens], [storey], [sqm], [tenure], [epccurrent], [epcpotential], council, councilband, permission, age, flood, la, rail, bus, Pschool, Sschool, Nursery, special, Churches, Mosque, Gurdwara, Synagogue, Mandir, gp, dent, hosp, pharm, opt, clin, other]]
        if parsed_dataout == [[[parsed_data[i][1][0]], [''], [], [], [0], [''], [''], [], [0], [''], [''], [''], [''], [''], [0], [''], [''], [], [], [], '', [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]]:
            raise Exception
        #print(parsed_dataout)
        if "Price Range" in range0 or "0" in lastd:
            with open(r"[REDACTED_BY_SCRIPT]", 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(parsed_dataout)
        tryCount=0
        i+=1
        completedScrape=True
    except:
        if tryCount < 9:
            tryCount +=1
        else:
            i+=1
            tryCount=0
        
        if not driver:
            driver = initialize_driver()
            driverReset=0

async def main2():
    global driver
    global row
    global parsed_dataout
    global cookiecount
    """[REDACTED_BY_SCRIPT]"""
    driver = initialize_driver()
    await scrape_with_timeout2(120)
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
        for jk in range(7669):
            next(csv_reader)
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
                time.sleep(1)
                try:
                    driver.quit()  # Quit the current driver
                except:
                    pass
                if retryattempt < 3:
                    retryattempt += 1
                else:
                    completedScrape = True
                pass

def scrape_page3(parsed_data,i):
    global driver
    global row
    global parsed_dataout
    global cookiecount
    global tryCount
    global temp_chromedriver_path
    try:
        try:
            if not driver:
                driver = initialize_driver()
        except:
            driver = initialize_driver()
        
        addressin=parsed_data[i][1][0]
        addressin2=addressin.split(" ")
        addressin2=addressin2[-2]+"-"+addressin2[-1]
        addressurl = "[REDACTED_BY_SCRIPT]"+addressin2

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
        
        BnLaddresses=[]
        BnLaddresses_href=[]
        BnLcount=1
        BnlFound=False
        #time.sleep(1000)
        while BnlFound==False:
            try:
                BnLaddresses0=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
                BnLaddresses.append(BnLaddresses0)                  #body > div > section:nth-child(3) > div.container.mx-auto.px-0.md\:px-\[2\.1rem\] > div > div.property-helpers > div:nth-child(3) > div > div.CardBody > div > ul > li:nth-child({BnLcount}) > span > span.flex-1.flex-grow.flex.items-center.px-\[1\.1rem\].max-w-full.overflow-hidden.bg-white.md\:px-\[1rem\].rounded-tr-md.rounded-br-md.border.border-l-0 > span.font-bold.block.mr-auto.max-w-full.overflow-hidden.whitespace-nowrap > a
                BnLaddresses_href0=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").get_attribute("href")
                BnLaddresses_href.append(BnLaddresses_href0)
                BnLcount+=1
            except:
                try:
                    BnLaddresses0=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
                    BnLaddresses.append(BnLaddresses0)                  #body > div > section:nth-child(3) > div.container.mx-auto.px-0.md\:px-\[2\.1rem\] > div > div.property-helpers > div:nth-child(3) > div > div.CardBody > div > ul > li:nth-child({BnLcount}) > span > span.flex-1.flex-grow.flex.items-center.px-\[1\.1rem\].max-w-full.overflow-hidden.bg-white.md\:px-\[1rem\].rounded-tr-md.rounded-br-md.border.border-l-0 > span.font-bold.block.mr-auto.max-w-full.overflow-hidden.whitespace-nowrap > a
                    BnLaddresses_href0=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").get_attribute("href")
                    BnLaddresses_href.append(BnLaddresses_href0)
                    BnLcount+=1
                except:
                    BnlFound=True
        
        target_address = addressin
        most_similar = None
        most_similar_href=None
        min_distance = float('inf')  # Initialize with a very large distance
        target_address_list=target_address.split(",")
        if "flat" in target_address_list[0].lower():
            target_address_list0=[target_address_list[1],target_address_list[2]]
        else:
            target_address_list0=[target_address_list[0],target_address_list[1],target_address_list[2]]
        for entry in BnLaddresses:
            entry_list=entry.split(",")
            if "flat" in target_address_list[0].lower():
                entry_list0=[entry_list[0],entry_list[1]]
            else:
                entry_list0=[entry_list[0],entry_list[1],entry_list[2]]
            sub_entry_count=0
            for jkl in range(len(entry_list0)):
                distance = Levenshtein.distance(target_address_list0[jkl].lower().replace(",","").replace("-","").replace(" ","").replace(".",""), entry_list0[jkl].lower().replace(",","").replace("-","").replace(" ","").replace(".",""))
                sub_entry_count+=distance
            distance=sub_entry_count
            print(entry_list0,"\n",target_address_list0,"\n",distance,"\n-----------------------------------------------------\n")
            #print(target_address, entry, distance)
            if distance < min_distance:
                min_distance = distance
                most_similar = entry
                most_similar_href=BnLaddresses_href[BnLaddresses.index(entry)]
                print("    ",most_similar,"\n    ",most_similar_href,"\n    ",min_distance,"\n-----------------------------------------------------\n")

        if most_similar:
            addressurl=most_similar_href
        else:
            print(BnLaddresses)
            print(BnLaddresses_href)
            pass
        try:
            str(addressurl)
        except:
            addressurl=addressurl[0]
        print(addressurl,addressin)
            
        driver.get(addressurl)
        time.sleep(random.uniform(1, 1.5))

        try:
            BnLaddresses=[]
            BnLaddresses_href=[]
            BnLcount=1
            BnlFound=False
            #time.sleep(1000)
            while BnlFound==False:
                try:
                    BnLaddresses0=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
                    BnLaddresses.append(BnLaddresses0)                  #body > div > section:nth-child(7) > div.container.mx-auto.px-0.md\:px-\[2\.1rem\] > div > div.property-helpers > div:nth-child(3) > div > div.CardBody > div > ul > li:nth-child(2) > span > span.flex-1.flex-grow.flex.items-center.px-\[1\.1rem\].max-w-full.overflow-hidden.bg-white.md\:px-\[1rem\].rounded-tr-md.rounded-br-md.border.border-l-0 > span > a
                    BnLaddresses_href0=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").get_attribute("href")
                    BnLaddresses_href.append(BnLaddresses_href0)
                    BnLcount+=1
                except:
                    try:
                        BnLaddresses0=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
                        BnLaddresses.append(BnLaddresses0)                  #body > div > section:nth-child(7) > div.container.mx-auto.px-0.md\:px-\[2\.1rem\] > div > div.property-helpers > div:nth-child(4) > div > div.CardBody > div > ul > li:nth-child(1) > span > span.flex-1.flex-grow.flex.items-center.px-\[1\.1rem\].max-w-full.overflow-hidden.bg-white.md\:px-\[1rem\].rounded-tr-md.rounded-br-md.border.border-l-0 > span.font-bold.block.mr-auto.max-w-full.overflow-hidden.whitespace-nowrap > a
                        BnLaddresses_href0=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").get_attribute("href")
                        BnLaddresses_href.append(BnLaddresses_href0)
                        BnLcount+=1
                    except:BnlFound=True
            
            target_address = addressin
            most_similar = None
            most_similar_href=None
            min_distance = float('inf')  # Initialize with a very large distance
            target_address_list=target_address.split(",")
            for entry in BnLaddresses:
                entry_list=entry.split(",")
                
                sub_entry_count=0
                for jkl in range(len(entry_list)):
                    distance = Levenshtein.distance(target_address_list[jkl].lower().replace(",","").replace("-","").replace(" ","").replace(".",""), entry_list[jkl].lower().replace(",","").replace("-","").replace(" ","").replace(".",""))
                    sub_entry_count+=distance
                distance=sub_entry_count
                print(entry_list0,"\n",target_address_list0,"\n",distance,"\n-----------------------------------------------------\n")
                #print(target_address, entry, distance)
                if distance < min_distance:
                    min_distance = distance
                    most_similar = entry
                    most_similar_href=BnLaddresses_href[BnLaddresses.index(entry)]

            if most_similar:
                addressurl=most_similar_href
            else:
                pass
            try:
                str(addressurl)
            except:
                addressurl=addressurl[0]
            print(addressurl,addressin)
                
            driver.get(addressurl)
            time.sleep(random.uniform(1, 1.5))
        except:pass


        try:
            element_present = EC.presence_of_element_located((By.CSS_SELECTOR, '[REDACTED_BY_SCRIPT]'))
            WebDriverWait(driver, 10).until(element_present)
        except Exception as e:
            pass

        time.sleep(random.uniform(0.8, 1.5))
        #time.sleep(1000)
        estimPrice, floorarea, rent, pricechange, rentchange, mortgage, plotsize, plotsdata1, plotsdata2 = "", "", "", "", "", "", "", "", ""
        try:
            floorarea = driver.find_element(By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]").text
            floorarea=floorarea.replace(" ft2","")           #body > div.bg-white > section.bg-white.py-0.relative.md\:bg-gradient-to-b.from-gray-400.via-white.to-white.md\:py-\[4rem\] > div.relative.z-1 > div.container.mx-auto.px-0.md\:px-\[2\.1rem\] > div.w-full.md\:max-w-\[33\.2rem\] > div > div.CardBody > div > p:nth-child(1) > span:nth-child(1)
            #print(floorarea)
        except:
            try:
                floorarea = driver.find_element(By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]").text
                floorarea=floorarea.replace(" ft2","")
            except:pass
        try:
            estimPrice = driver.find_element(By.XPATH, "[REDACTED_BY_SCRIPT]").text
            estimPrice=estimPrice.replace("£","").replace(",","") 
            rent = driver.find_element(By.XPATH, "[REDACTED_BY_SCRIPT]").text
        except:
            try:
                estimPrice = driver.find_element(By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]").text
                estimPrice=estimPrice.replace("£","").replace(",","") 
                rent = driver.find_element(By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]").text
                #
            except:pass
        try:
            mortgage = driver.find_element(By.XPATH, "[REDACTED_BY_SCRIPT]").text
            #print(mortgage)
        except Exception as e:
            pass
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
        except Exception as e:
            pass
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
                except Exception as e:
                    pass
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
                except Exception as e:
                    pass
        brdbndStore=0
        traindistance1=""
        traindistance2=""
        traindistance3=""
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
                except Exception as e:
                    pass
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
                    except Exception as e:
                        pass
        #print(broadband)
        houseList=[]
        educationList=[]
        #
        #
        #body > div.bg-white > section:nth-child(28) > div.relative.z-1 > div > div > div > div.CardBody > div:nth-child(2) > div > div:nth-child(1) > div.flex-\[5\] > div > div:nth-child(1) > div > div.text-bold
        #body > div.bg-white > section:nth-child(28) > div.relative.z-1 > div > div > div > div.CardBody > div:nth-child(2) > div > div:nth-child(1) > div.flex-\[5\] > div > div:nth-child(2) > div > div.text-bold
        for j in range(3):
            for k in range(5):
                houseinfo=""
                eduinfo=""
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
                        except Exception as e:
                            pass
               
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
                except Exception as e:
                    pass
        #print(avgAge)
        try:
            modeAge=modeAge[(modeAge.index("was ")+4):]
            modeAge=modeAge[:modeAge.index(" and")]
        except:pass
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
                except Exception as e:
                    pass
        try:
            secondryHouse=secondryHouse[:secondryHouse.index("%")]
        except:pass
        quality=""
        try:
            quality = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
            if "High" in quality:quality=quality.replace("High","3")
            elif "Medium" in quality:quality=quality.replace("Medium","2")
            else:quality="1"
        except Exception as e:
            pass
        quality2=""
        try:
            quality2 = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
            quality2=quality2.replace(" / 5","")
        except Exception as e:
            pass
        try:print(most_similar)
        except:pass
        try:
            print(parsed_data[i][1][0], floorarea, estimPrice, mortgage, "", "", "", schooldist1, schooldist2, schooldist3, schoolrating1, schoolrating2, schoolrating3, schooldist21, schooldist22, schooldist23, schoolrating21, schoolrating22, schoolrating23, traindistance1, traindistance2, traindistance3, houseList, educationList, avgAge, modeAge, secondryHouse, carinfo, quality, quality2)
        except Exception as e:
            pass
        if "[REDACTED_BY_SCRIPT]" in plotsize:
            parsed_dataout =[[parsed_data[i][1][0], floorarea, estimPrice, mortgage, "", "", "", schooldist1, schooldist2, schooldist3, schoolrating1, schoolrating2, schoolrating3, schooldist21, schooldist22, schooldist23, schoolrating21, schoolrating22, schoolrating23, traindistance1, traindistance2, traindistance3, houseList, educationList, avgAge, modeAge, secondryHouse, carinfo, quality, quality2,rent]]
        else:
            parsed_dataout =[[parsed_data[i][1][0], floorarea, estimPrice, mortgage, plotsize, plotsdata1, plotsdata2, schooldist1, schooldist2, schooldist3, schoolrating1, schoolrating2, schoolrating3, schooldist21, schooldist22, schooldist23, schoolrating21, schoolrating22, schoolrating23, traindistance1, traindistance2, traindistance3, houseList, educationList, avgAge, modeAge, secondryHouse, carinfo, quality, quality2,rent]]
        with open(r"[REDACTED_BY_SCRIPT]", 'a', newline='', encoding='utf-8') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(parsed_dataout)
        time.sleep(5)
    except:
        if tryCount < 3:
            i-=1
            tryCount +=1
        elif not driver:
            driver = initialize_driver()
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
        for jk in range(2905):
            next(csv_reader)
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
                if retryattempt < 3:
                    retryattempt += 1
                else:
                    completedScrape = True
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
    global temp_chromedriver_path
    try:
        try:
            if not driver:
                driver = initialize_driver()
        except:
            driver = initialize_driver()
        addressin=parsed_data[i][1][0]
        addressin2=addressin.split(" ")
        addressin2=addressin2[-2]+"%20"+addressin2[-1]
        addressurl = "[REDACTED_BY_SCRIPT]"+addressin2
        driver.get(addressurl)
        
        timeout=30
        if cookiecount==0:
            shadow_host = driver.find_element(By.CSS_SELECTOR, "#cmpwrapper")
            shadow_root = driver.execute_script("[REDACTED_BY_SCRIPT]", shadow_host)

            wait = WebDriverWait(shadow_root, timeout)
            element_locator = (By.CSS_SELECTOR, '[REDACTED_BY_SCRIPT]')
            element = wait.until(EC.presence_of_element_located(element_locator))
            shadow_root.find_element(By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]").click()
            cookiecount=1
        number_of_props = driver.find_element(By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]").text
        number_of_props = number_of_props.split(" ")
        number_of_props = number_of_props[0]
        compare_list_adrresses=[]
        compare_list_hrefs=[]
        for jk in range(int(number_of_props)):
            try:
                compare_list_adrresses.append(driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text)
                compare_list_hrefs.append(driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").get_attribute('href'))
            except:pass

        target_address = addressin
        most_similar = None
        most_similar_href=None
        min_distance = float('inf')  # Initialize with a very large distance

        for entry in compare_list_adrresses:
            distance = Levenshtein.distance(target_address.lower().replace(",","").replace("-","").replace(" ","").replace(".",""), entry.lower().replace(",","").replace("-","").replace(" ","").replace(".",""))
            #print(target_address, entry, distance)
            if distance < min_distance:
                min_distance = distance
                most_similar = entry
                most_similar_href=compare_list_hrefs[compare_list_adrresses.index(entry)]

        if most_similar:
            addressurl=most_similar_href
        else:
            pass

        #no. body > div.z > div.sp-dash-wrap > div:nth-child(7) > div.left > h2

        #href1: body > div.z > div.sp-dash-wrap > div:nth-child(7) > div.left > div.result-cards > a:nth-child(1)
        #address1: body > div.z > div.sp-dash-wrap > div:nth-child(7) > div.left > div.result-cards > a:nth-child(1) > div > div.lt > div.ltl

        #href2: body > div.z > div.sp-dash-wrap > div:nth-child(7) > div.left > div.result-cards > a:nth-child(3)
        #address2: body > div.z > div.sp-dash-wrap > div:nth-child(7) > div.left > div.result-cards > a:nth-child(3) > div > div.lt > div.ltl

        #href3: body > div.z > div.sp-dash-wrap > div:nth-child(7) > div.left > div.result-cards > a:nth-child(5)
        #address3: body > div.z > div.sp-dash-wrap > div:nth-child(7) > div.left > div.result-cards > a:nth-child(5) > div > div.lt > div.ltl
        
        driver.get(addressurl)
        wait = WebDriverWait(driver, timeout)
        element_locator = (By.CSS_SELECTOR, '[REDACTED_BY_SCRIPT]')
        element = wait.until(EC.presence_of_element_located(element_locator))
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
            driver = initialize_driver()
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
    global completedScrape
    parsed_data = []
    parsed_dataout=[]
    with open(r"[REDACTED_BY_SCRIPT]", 'r', newline='', encoding='utf-8') as csvfile:
        csv_reader = csv.reader(csvfile)
        for ki in range(4548):#THIS JUST SKIPS ROWS IT HAS DONE ALREADY. CHANGE THE NUMBER TO THE ROW NUMBER IT STOPPED AT
            next(csvfile)
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
                driver = initialize_driver5()
            try:
                await asyncio.wait_for(asyncio.to_thread(scrape_page5,parsed_data,i), timeout=timeout_seconds)
                if completedScrape != True:
                    raise ValueError("[REDACTED_BY_SCRIPT]")
            except asyncio.TimeoutError:
                print("[REDACTED_BY_SCRIPT]")
                try:
                    driver.quit()  # Quit the current driver
                except:
                    pass
            except ValueError as e:
                if retryattempt < 3:
                    retryattempt += 1
                else:
                    completedScrape = True
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
    global temp_chromedriver_path
    global completedScrape
    try:
        try:
            if not driver:
                driver = initialize_driver()
                
        except:
            options = uc.ChromeOptions()
            options.add_argument('--disable-gpu')
            options.add_argument('--no-sandbox')
            options.add_argument("[REDACTED_BY_SCRIPT]")
            options.add_argument("[REDACTED_BY_SCRIPT]")
            options.add_argument("--headless")
            time.sleep(1)
            driver = uc.Chrome(driver_executable_path=temp_chromedriver_path, options=options)
            
        for jk in range(5):
            try:
                addressin=parsed_data[i][1][0]
                get_website="www.chimnie.co.uk/"
                binginput="[REDACTED_BY_SCRIPT]"+addressin
                driver.get('[REDACTED_BY_SCRIPT]')
                element_present = EC.presence_of_element_located((By.CSS_SELECTOR, "#sb_form_c > div"))
                WebDriverWait(driver, 15).until(element_present)

                try:
                    driver.find_element(By.CSS_SELECTOR, "#sb_form_c > div").click()
                    driver.find_element(By.CSS_SELECTOR, "#sb_form_q").send_keys(binginput)
                    driver.find_element(By.CSS_SELECTOR, "#sb_form_q").send_keys(Keys.RETURN)
                except:
                    try:
                        driver.find_element(By.CSS_SELECTOR, "#bnp_btn_reject").click()
                    except:
                        driver.find_element(By.CSS_SELECTOR, "#bnp_btn_reject > a")
                    driver.find_element(By.CSS_SELECTOR, "#sb_form_c > div").click()
                    driver.find_element(By.CSS_SELECTOR, "#sb_form_q").send_keys(binginput)
                    driver.find_element(By.CSS_SELECTOR, "#sb_form_q").send_keys(Keys.RETURN)

                element_present = EC.presence_of_element_located((By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]"))
                WebDriverWait(driver, 15).until(element_present)
                potential_found=[]
                try:
                    driver.find_element(By.CSS_SELECTOR, "#bnp_btn_reject").click()
                except:
                    try:
                        driver.find_element(By.CSS_SELECTOR, "#bnp_btn_reject > a")
                    except:pass
                for k in range(10):
                    try:
                        domain=driver.find_element(By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]"+str(k+1)+"[REDACTED_BY_SCRIPT]").text
                        metadata=driver.find_element(By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]"+str(k+1)+") > h2").text
                        href=driver.find_element(By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]"+str(k+1)+") > h2 > a").get_attribute("href")
                    except:
                        try:
                            domain=driver.find_element(By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]"+str(k+1)+"[REDACTED_BY_SCRIPT]").text
                            metadata=driver.find_element(By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]"+str(k+1)+") > h2").text
                            href=driver.find_element(By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]"+str(k+1)+") > h2 > a").get_attribute("href")
                        except:
                            domain=""
                            metadata=""
                            href=""
                    if get_website in domain:
                        potential_found.append([domain,metadata,href])
                try:
                    driver.find_element(By.CSS_SELECTOR, "#bnp_btn_reject").click()
                except:
                    try:
                        driver.find_element(By.CSS_SELECTOR, "#bnp_btn_reject > a")
                    except:pass

                target_address = addressin
                target_address2=target_address.split(",")
                target_address=""
                for jkl in range(len(target_address2)-1):
                    target_address+=target_address2[jkl+1]
                most_similar = None
                min_distance = float('inf')  # Initialize with a very large distance

                for entry in potential_found:
                    domain, metadata, href = entry
                    distance = Levenshtein.distance(target_address.lower().replace(",","").replace("-","").replace(" ","").replace(".",""), metadata.lower().replace(",","").replace("-","").replace(" ","").replace(".",""))
                    print(target_address, metadata, distance)
                    if distance < min_distance:
                        min_distance = distance
                        most_similar = entry

                if most_similar:
                    domain, metadata, href = most_similar
                else:
                    pass

                
                addressurl = href
                break
            except Exception as e:
                print(e)
                pass
        driver.get(addressurl)
        timeout=30
        chimnieaddressInfo1=["","","","","","","","","",""]
        chimnieaddressInfo2=["","","","","","","","","",""]
        chimnieaddressInfo3=["","","","","","","","","",""]
        element_present = EC.presence_of_element_located((By.XPATH, "/html/body/div[1]/div[1]/div[4]/div/div[1]/div[1]/div/div/div[1]/h3"))
        WebDriverWait(driver, timeout).until(element_present)
        for i in range(3):
            try:
                lscore1 = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
                if i==0:chimnieaddressInfo1[0] = lscore1
                elif i==1:chimnieaddressInfo2[0] = lscore1
                elif i==2:chimnieaddressInfo3[0] = lscore1
            except:pass
            try:
                escore1 = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
                if i==0:chimnieaddressInfo1[1] = escore1
                elif i==1:chimnieaddressInfo2[1] = escore1
                elif i==2:chimnieaddressInfo3[1] = escore1
            except:pass
            try:# 
                sscore1 = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text   
                if i==0:chimnieaddressInfo1[2] = sscore1
                elif i==1:chimnieaddressInfo2[2] = sscore1
                elif i==2:chimnieaddressInfo3[2] = sscore1
            except:pass
            try:
                fscore1 = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
                if i==0:chimnieaddressInfo1[3] = fscore1
                elif i==1:chimnieaddressInfo2[3] = fscore1
                elif i==2:chimnieaddressInfo3[3] = fscore1
            except:pass
            try:
                cscore1 = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
                if i==0:chimnieaddressInfo1[4] = cscore1
                elif i==1:chimnieaddressInfo2[4] = cscore1
                elif i==2:chimnieaddressInfo3[4] = cscore1
            except:pass
            try:
                l2score1 = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
                if i==0:chimnieaddressInfo1[5] = l2score1
                elif i==1:chimnieaddressInfo2[5] = l2score1
                elif i==2:chimnieaddressInfo3[5] = l2score1
            except:pass
            try:
                e2score1 = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
                if i==0:chimnieaddressInfo1[6] = e2score1
                elif i==1:chimnieaddressInfo2[6] = e2score1
                elif i==2:chimnieaddressInfo3[6] = e2score1
            except:pass
            try:
                s2score1 = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
                if i==0:chimnieaddressInfo1[7] = s2score1
                elif i==1:chimnieaddressInfo2[7] = s2score1
                elif i==2:chimnieaddressInfo3[7] = s2score1
            except:pass
            try:
                f2score1 = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
                if i==0:chimnieaddressInfo1[8] = f2score1
                elif i==1:chimnieaddressInfo2[8] = f2score1
                elif i==2:chimnieaddressInfo3[8] = f2score1
            except:pass
            try:
                c2score1 = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text
                if i==0:chimnieaddressInfo1[9] = c2score1
                elif i==1:chimnieaddressInfo2[9] = c2score1
                elif i==2:chimnieaddressInfo3[9] = c2score1
            except:pass
            if i == 0:#                        /html/body/div[1]/div[1]/div[4]/div/div[1]/div[9]/div[1]/div[2]/div/div/div/button[2]
                driver.find_element(By.XPATH, "[REDACTED_BY_SCRIPT]").click()
                time.sleep(random.uniform(1.2, 1.5))
            elif i == 1:#                      /html/body/div[1]/div[1]/div[4]/div/div[1]/div[9]/div[1]/div[2]/div/div/div/button[3]
                driver.find_element(By.XPATH, "[REDACTED_BY_SCRIPT]").click()
                time.sleep(random.uniform(1.2, 1.5))
        chimnieaddressInfo=chimnieaddressInfo1+chimnieaddressInfo2+chimnieaddressInfo3
        chimnieaddressInfo=[addressin]+chimnieaddressInfo
        with open(r"[REDACTED_BY_SCRIPT]", 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([chimnieaddressInfo])
        completedScrape = True
        time.sleep(5)
    except Exception as e:
        print(f"An error occurred: {e}")
        if tryCount < 3:
            i-=1
            tryCount +=1
        elif not driver:
            driver = initialize_driver()
            
        else:pass




async def main5():
    global driver
    global row
    global parsed_dataout
    global cookiecount
    """[REDACTED_BY_SCRIPT]"""
    driver = initialize_driver5()
    await scrape_with_timeout5(300)
    driver.quit()  # Ensure the driver is closed after scraping


def initialize_driver5():
    global driver
    global row
    global parsed_dataout
    global cookiecount
    global temp_chromedriver_path
    options = uc.ChromeOptions()
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument("[REDACTED_BY_SCRIPT]")
    options.add_argument("[REDACTED_BY_SCRIPT]")
    options.add_argument("--headless")
    time.sleep(1)
    driver = uc.Chrome(driver_executable_path=temp_chromedriver_path, options=options)
    
    cookiecount=0
    return driver


##############################################################################################################################################


async def scrape_with_timeout6(timeout_seconds=120):  
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
        for jk in range(6086):
            next(csvfile)
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
                await asyncio.wait_for(asyncio.to_thread(scrape_page6,parsed_data), timeout=timeout_seconds)
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





def scrape_page6(parsed_data):
    global driver
    global row
    global parsed_dataout
    global tryCount
    global i
    global driverReset
    global temp_chromedriver_path
    if not driver:
        driver = initialize_driver()
        driverReset=1
    try:

        ##########################################################################################################
        #streetcan
        addressin=parsed_data[i][1][0]
        addressin2=addressin.split(" ")[-2]+"-"+addressin.split(" ")[-1]
        addressin2=addressin2.lower()
        addressurl=f"[REDACTED_BY_SCRIPT]"
        driver.get(addressurl)
        element_present = EC.presence_of_element_located((By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]"))
        WebDriverWait(driver, 15).until(element_present)

        #this gets the ratings
        ratings_list=[]
        for j in range(1,10):
            if j == 3 or j== 4 or j>5:
                star_count=0
                for k in range(5):
                    try:
                        star_check=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]")
                        if star_check.get_attribute("class") == "fas fa-star checked":
                            star_count+=1
                    except Exception as e:
                        print(f"[REDACTED_BY_SCRIPT]")
                        pass
                ratings_list.append(star_count)
        try:
            average_price_strt=driver.find_element(By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]").text
        except:pass
        

        income_list=[]
        try:
            income_num=driver.find_element(By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]").text
            income_num=income_num[income_num.index("\n")+2:]
            income_num=income_num.replace(",","").replace("£","")
            income_rank=driver.find_element(By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]").text
            income_rank=income_rank[:income_rank.index("/")]
        except:
            income_num=""
            income_rank=""
        income_list.append(income_num)
        income_list.append(income_rank)

        deprivation_list=[]
        for k in range(10):
            try:
                dep_item=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
                dep_item=dep_item[:dep_item.index("/")]
                dep_item=dep_item.replace(" ","").replace("-","")
            except:
                dep_item=""
            deprivation_list.append(dep_item)
        
        past_sales_list=[]
        for k in range(6):
            try:
                past_sales_item=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
                past_sales_item=past_sales_item.replace("£","").replace(",","")
            except:
                past_sales_item=""
            past_sales_list.append(past_sales_item)
        
       
        for k in range(10):
            for jk in range(10):
                try:
                    if "Retired" in driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text:
                        emp_col=k+1
                except:pass
        emp_type_list=["Retired","Full-Time Employee","Part-Time Employee","Self-Employed","[REDACTED_BY_SCRIPT]","[REDACTED_BY_SCRIPT]","[REDACTED_BY_SCRIPT]","Other","[REDACTED_BY_SCRIPT]","Unemployed"]
        employment_list=["","","","","","","","","",""]
        for k in range(10):
            try:
                for jk in range(len(emp_type_list)):                               #employment > div:nth-child(4) > div.col-md-4.col-lg-4.col-sm-12 > div > table > tbody > tr:nth-child(1) > td:nth-child(2)
                    if emp_type_list[jk] in driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text:
                        employment_list[jk]=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
            except:pass
        
        for k in range(10):
            for jk in range(10):
                try:
                    if "Education" in driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text:
                        emp_col=k+1
                except:pass
        emp_ind_type_list=["Education","[REDACTED_BY_SCRIPT]","Construction","[REDACTED_BY_SCRIPT]","Other","[REDACTED_BY_SCRIPT]","[REDACTED_BY_SCRIPT]","Manufacturing","Information and communication","Financial and insurance activities","[REDACTED_BY_SCRIPT]","[REDACTED_BY_SCRIPT]","Real estate activities","Transport and storage","[REDACTED_BY_SCRIPT]"]
        emp_ind_list=[""]*15
        for k in range(15):
            try:
                for jk in range(len(emp_ind_type_list)):                               
                    if emp_ind_type_list[jk] in driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text:
                        emp_ind_list[jk]=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
            except: pass

        streetscanaddressInfo=[addressin,ratings_list,income_list,deprivation_list,past_sales_list,employment_list,emp_ind_list]
        with open(r"[REDACTED_BY_SCRIPT]", 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([streetscanaddressInfo])       
        tryCount=0
        i+=1
    except Exception as e:
        print(f"An error occurred: {e}")
        if tryCount < 9:
            tryCount +=1
        else:
            i+=1
            tryCount=0
        
        if not driver:
            driver = initialize_driver()
            driverReset=0

async def main6():
    global driver
    global row
    global parsed_dataout
    global cookiecount
    """[REDACTED_BY_SCRIPT]"""
    driver = initialize_driver()
    await scrape_with_timeout6(300)
    driver.quit()  # Ensure the driver is closed after scraping




##############################################################################################################################################


async def scrape_with_timeout7(timeout_seconds=120):  
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
                await asyncio.wait_for(asyncio.to_thread(scrape_page7,parsed_data), timeout=timeout_seconds)
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





def scrape_page7(parsed_data):
    global driver
    global row
    global parsed_dataout
    global tryCount
    global i
    global driverReset
    global temp_chromedriver_path
    if not driver:
        driver = initialize_driver()
        driverReset=1
    try:

        ##########################################################################################################
        #streetcan
        addressin=parsed_data[i][1][0]
        addressin2=addressin.split(" ")[-2]
        addressurl=f"[REDACTED_BY_SCRIPT]"
        driver.get(addressurl)
        element_present = EC.presence_of_element_located((By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]"))
        WebDriverWait(driver, 15).until(element_present)

        """
        postcode district page gives me a list of all subdomains, and their hrefs.
        """
        subdomain_href_list=[]
        subdomain_name_list=[]
        href_to_name = {}  # Dictionary to map URLs to their names

        for k in range(1,1500):
            try:
                subdomain_name=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
                subdomain_href=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").get_attribute("href")
                subdomain_name_list.append(subdomain_name)
                subdomain_href_list.append(subdomain_href)
                href_to_name[subdomain_href] = subdomain_name  # Store mapping between URL and name
            except:
                break
        unique_links = list(set(subdomain_href_list))

        num_to_select = min(25, len(unique_links))
        selected_hrefs = random.sample(unique_links, num_to_select)
        subdomain_href_list = selected_hrefs
        subdomain_name_list = [href_to_name[href] for href in selected_hrefs]
        
        streetscan_avg_list=[]
        for jkl in range(len(subdomain_name_list)):
            driver.get(subdomain_name_list[jkl])
            element_present = EC.presence_of_element_located((By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]"))
            WebDriverWait(driver, 15).until(element_present)
            header_list=["Gender","Partnership Status","Health","[REDACTED_BY_SCRIPT]", "Ethnic Group","Country of Birth", "Length of Residence"]
            #this gets the ratings
            ratings_list=[]
            for j in range(10):
                if j == 3 or j== 4 or j>5:
                    star_count=0
                    for k in range(5):
                        try:
                            star_check=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]")
                            if star_check.get_attribute("class") == "fas fa-star checked":
                                star_count+=1
                        except Exception as e:
                            print(f"[REDACTED_BY_SCRIPT]")
                            pass
                    ratings_list.append(star_count)
            try:
                average_price_strt=driver.find_element(By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]").text
            except:pass
            tables_list=[]
            for k in range(8):
                if k//2 == 0:
                    if driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text == header_list[k/2]:
                        for jk in range(10):
                            female_1=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
                            female_2=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
                            tables_list.append(female_1)
                            tables_list.append(female_2)
                    else:
                        if k == 0:
                            for jk in range(4):
                                tables_list.append("")
                        elif k==2:
                            for jk in range(12):
                                tables_list.append("")
                        elif k==4:
                            for jk in range(10):
                                tables_list.append("")
                        elif k==6:
                            for jk in range(14):
                                tables_list.append("")
            for k in range(6):
                if k//2 == 0:
                    if driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text == header_list[k/2]:
                        for jk in range(10):
                            female_1=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
                            female_2=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
                            tables_list.append(female_1)
                            tables_list.append(female_2)
                    else:
                        if k==0:
                            for jk in range(16):
                                tables_list.append("")
                        elif k==2:
                            for jk in range(12):
                                tables_list.append("")
                        elif k==4:
                            for jk in range(10):
                                tables_list.append("")
            
            #the following is religion stats
            for k in range(4):
                try:
                    female_1=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
                    female_2=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
                except:
                    female_1=""
                    female_2=""
                tables_list.append(female_1)
                tables_list.append(female_2)

            income_list=[]
            try:
                income_num=driver.find_element(By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]").text
                income_num=income_num.replace(",","").replace("£","")
                income_rank=driver.find_element(By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]").text
                income_rank=income_rank[:income_rank.index("/")]
            except:
                income_num=""
                income_rank=""
            income_list.append(income_num)
            income_list.append(income_rank)

            for jk in range(4):
                try:
                    income_item=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
                except:
                    income_item=""
                income_list.append(income_item)
            for jk in range(4):
                try:
                    income_item=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
                except:
                    income_item=""
                income_list.append(income_item)
            

            deprivation_list=[]
            for k in range(10):
                try:
                    dep_item=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
                except:
                    dep_item=""
                deprivation_list.append(dep_item)
            
            past_sales_list=[]
            for k in range(6):
                try:
                    past_sales_item=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
                except:
                    past_sales_item=""
                past_sales_list.append(past_sales_item)
            
            for k in range(6):
                try:
                    past_sales_item=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
                except:
                    past_sales_item=""
                past_sales_list.append(past_sales_item)
            for k in range(6):
                try:
                    past_sales_item=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
                except:
                    past_sales_item=""
                past_sales_list.append(past_sales_item)

            emp_type_list=["Retired","Full-Time Employee","Part-Time Employee","Self-Employed","[REDACTED_BY_SCRIPT]","[REDACTED_BY_SCRIPT]","[REDACTED_BY_SCRIPT]","Other","[REDACTED_BY_SCRIPT]","Unemployed"]
            employment_list=["","","","","","","","","",""]
            for k in range(10):
                try:
                    for jk in range(len(emp_type_list)):
                        if emp_type_list[jk] in driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text:
                            employment_list[jk]=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
                except:
                    pass
            
            emp_ind_type_list=["Education","[REDACTED_BY_SCRIPT]","Construction","[REDACTED_BY_SCRIPT]","Other","[REDACTED_BY_SCRIPT]","[REDACTED_BY_SCRIPT]","Manufacturing","Information and communication","Financial and insurance activities","[REDACTED_BY_SCRIPT]","[REDACTED_BY_SCRIPT]","Real estate activities","Transport and storage","[REDACTED_BY_SCRIPT]"]
            emp_ind_list=[""]*15
            for k in range(15):
                try:
                    for jk in range(len(emp_ind_type_list)):
                        if emp_ind_type_list[jk] in driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text:
                            emp_ind_list[jk]=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
                except:
                    pass

            
            school_list=[]
            for k in range(5):
                try:
                    name = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
                    link= driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").get_attribute("href")
                    ofsted_score= driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
                    distance= driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
                    est_type= driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
                    nursery=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
                    gender=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
                    school_list.append([name,link,ofsted_score,distance,est_type,nursery,gender])
                except:
                    school_list.append(["","","","","","",""])
            
            for k in range(5):
                try:
                    name = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
                    link= driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").get_attribute("href")
                    ofsted_score= driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
                    distance= driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
                    est_type= driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
                    nursery=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
                    gender=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
                    school_list.append([name,link,ofsted_score,distance,est_type,nursery,gender])
                except:
                    school_list.append(["","","","","","",""])
            
            for k in range(5):
                try:
                    name = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
                    link= driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").get_attribute("href")
                    ofsted_score= driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
                    distance= driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
                    est_type= driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
                    nursery=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
                    gender=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
                    school_list.append([name,link,ofsted_score,distance,est_type,nursery,gender])
                except:
                    school_list.append(["","","","","","",""])
            
            food_drink_list=[]
            #restaurants
            try:
                dist=driver.find_element(By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]").text
                food_rating=0
                for jk in range(5):
                    food_rating_up=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]")
                    if food_rating_up.get_attribute("class") == "fas fa-star checked":
                        food_rating+=1
                food_drink_list.append([dist,food_rating])
                food_drink_list.append(["",""])
                food_drink_list.append(["",""])
                food_drink_list.append(["",""])
                food_drink_list.append(["",""])
            except:
                for k in range(5):
                    try:
                        dist=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
                        food_rating=0
                        for jk in range(5):
                            food_rating_up=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]")
                            if food_rating_up.get_attribute("class") == "fas fa-star checked":
                                food_rating+=1
                        food_drink_list.append([dist,food_rating])
                    except:
                        food_drink_list.append(["",""])
            #clubs
            try:
                dist=driver.find_element(By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]").text
                food_rating=0
                for jk in range(5):
                    food_rating_up=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]")
                    if food_rating_up.get_attribute("class") == "fas fa-star checked":
                        food_rating+=1
                food_drink_list.append([dist,food_rating])
                food_drink_list.append(["",""])
                food_drink_list.append(["",""])
                food_drink_list.append(["",""])
                food_drink_list.append(["",""])
            except:
                for k in range(5):
                    try:
                        dist=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
                        food_rating=0
                        for jk in range(5):
                            food_rating_up=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]")
                            if food_rating_up.get_attribute("class") == "fas fa-star checked":
                                food_rating+=1
                        food_drink_list.append([dist,food_rating])
                    except:
                        food_drink_list.append(["",""])
            #takeaways
            try:
                dist=driver.find_element(By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]").text
                food_rating=0
                for jk in range(5):
                    food_rating_up=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]")
                    if food_rating_up.get_attribute("class") == "fas fa-star checked":
                        food_rating+=1
                food_drink_list.append([dist,food_rating])
                food_drink_list.append(["",""])
                food_drink_list.append(["",""])
                food_drink_list.append(["",""])
                food_drink_list.append(["",""])
            except:
                for k in range(5):
                    try:
                        dist=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
                        food_rating=0
                        for jk in range(5):
                            food_rating_up=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]")
                            if food_rating_up.get_attribute("class") == "fas fa-star checked":
                                food_rating+=1
                        food_drink_list.append([dist,food_rating])
                    except:
                        food_drink_list.append(["",""])
            #supermarkets
            try:
                supermarket=driver.find_element(By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]").text
                dist=driver.find_element(By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]").text
                food_rating=0
                for jk in range(5):
                    food_rating_up=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]")
                    if food_rating_up.get_attribute("class") == "fas fa-star checked":
                        food_rating+=1
                food_drink_list.append([dist,food_rating])
                food_drink_list.append(["",""])
                food_drink_list.append(["",""])
                food_drink_list.append(["",""])
                food_drink_list.append(["",""])
            except:
                for k in range(5):
                    try:
                        supermarket=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
                        dist=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
                        food_rating=0
                        for jk in range(5):
                            food_rating_up=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]")
                            if food_rating_up.get_attribute("class") == "fas fa-star checked":
                                food_rating+=1
                        food_drink_list.append([supermarket,dist,food_rating])
                    except:
                        food_drink_list.append(["","",""])
            
            sports_amenities=[]
            try:
                dist=driver.find_element(By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]").text
                amenitie_type=driver.find_element(By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]").text
                sports_amenities.append([dist,amenitie_type])
                for k in range(9):
                    sports_amenities.append(["",""])
            except:
                for k in range(10):
                    try:
                        dist=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
                        amenitie_type=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
                        sports_amenities.append([dist,amenitie_type])
                    except:
                        sports_amenities.append(["",""])


            broadband_list=[]
            for k in range(4):
                try:
                    availability=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
                    speed=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
                    download=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
                    upload=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
                    broadband_list.append([availability,speed,download,upload])
                except:
                    broadband_list.append(["","","",""])







        ##########################################################################################################
        #end of streetcan



        ##########################################################################################################
        #movemarket
        addressurl=f"[REDACTED_BY_SCRIPT]"
        driver.get(addressurl)
        element_present = EC.presence_of_element_located((By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]"))
        WebDriverWait(driver, 15).until(element_present)

        detached_beds5, detached_beds4, detached_beds3, detached_beds2, detached_beds1, detached_sqft, detached_val = [], [], [], [], [], [], []
        semi_detached_beds5, semi_detached_beds4, semi_detached_beds3, semi_detached_beds2, semi_detached_beds1, semi_detached_sqft, semi_detached_val = [], [], [], [], [], [], []
        terraced_beds5, terraced_beds4, terraced_beds3, terraced_beds2, terraced_beds1, terraced_sqft, terraced_val = [], [], [], [], [], [], []
        flat_beds5, flat_beds4, flat_beds3, flat_beds2, flat_beds1, flat_sqft, flat_val = [], [], [], [], [], [], []
        for k in range(100):
            try:
                house_type=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
                beds=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
                sqft=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
                val=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").text
                if "semi-detached" in house_type.lower():
                    if "5" in beds:
                        semi_detached_beds5.append([sqft,val])
                    elif "4" in beds:
                        semi_detached_beds4.append([sqft,val])
                    elif "3" in beds:
                        semi_detached_beds3.append([sqft,val])
                    elif "2" in beds:
                        semi_detached_beds2.append([sqft,val])
                    elif "1" in beds:
                        semi_detached_beds1.append([sqft,val])
                    semi_detached_sqft.append(sqft)
                    semi_detached_val.append(val)
                elif "detached" in house_type.lower():
                    if "5" in beds:
                        detached_beds5.append([sqft,val])
                    elif "4" in beds:
                        detached_beds4.append([sqft,val])
                    elif "3" in beds:
                        detached_beds3.append([sqft,val])
                    elif "2" in beds:
                        detached_beds2.append([sqft,val])
                    elif "1" in beds:
                        detached_beds1.append([sqft,val])
                    detached_sqft.append(sqft)
                    detached_val.append(val)
                elif "terraced" in house_type.lower():
                    if "5" in beds:
                        terraced_beds5.append([sqft,val])
                    elif "4" in beds:
                        terraced_beds4.append([sqft,val])
                    elif "3" in beds:
                        terraced_beds3.append([sqft,val])
                    elif "2" in beds:
                        terraced_beds2.append([sqft,val])
                    elif "1" in beds:
                        terraced_beds1.append([sqft,val])
                    terraced_sqft.append(sqft)
                    terraced_val.append(val)
                elif "flat" in house_type.lower():
                    if "5" in beds:
                        flat_beds5.append([sqft,val])
                    elif "4" in beds:
                        flat_beds4.append([sqft,val])
                    elif "3" in beds:
                        flat_beds3.append([sqft,val])
                    elif "2" in beds:
                        flat_beds2.append([sqft,val])
                    elif "1" in beds:
                        flat_beds1.append([sqft,val])
                    flat_sqft.append(sqft)
                    flat_val.append(val)
            except:pass

        def calculate_averages(items_list):
            """[REDACTED_BY_SCRIPT]"""
            if items_list:
                sqft_values = [float(item[0]) for item in items_list]
                val_values = [float(item[1]) for item in items_list]
                avg_sqft = sum(sqft_values) / len(sqft_values)
                avg_val = sum(val_values) / len(val_values)
            else:
                avg_sqft = 0
                avg_val = 0
            return avg_sqft, avg_val

        # Calculate averages for detached houses
        avg_sqft_det5, avg_val_det5 = calculate_averages(detached_beds5)
        avg_sqft_det4, avg_val_det4 = calculate_averages(detached_beds4)
        avg_sqft_det3, avg_val_det3 = calculate_averages(detached_beds3)
        avg_sqft_det2, avg_val_det2 = calculate_averages(detached_beds2)
        avg_sqft_det1, avg_val_det1 = calculate_averages(detached_beds1)
        avg_sqft_det = sum(detached_sqft) / len(detached_sqft) if detached_sqft else 0
        avg_val_det = sum(detached_val) / len(detached_val) if detached_val else 0

        # Calculate averages for semi-detached houses
        avg_sqft_semi_det5, avg_val_semi_det5 = calculate_averages(semi_detached_beds5)
        avg_sqft_semi_det4, avg_val_semi_det4 = calculate_averages(semi_detached_beds4)
        avg_sqft_semi_det3, avg_val_semi_det3 = calculate_averages(semi_detached_beds3)
        avg_sqft_semi_det2, avg_val_semi_det2 = calculate_averages(semi_detached_beds2)
        avg_sqft_semi_det1, avg_val_semi_det1 = calculate_averages(semi_detached_beds1)
        avg_sqft_semi_det = sum(semi_detached_sqft) / len(semi_detached_sqft) if semi_detached_sqft else 0
        avg_val_semi_det = sum(semi_detached_val) / len(semi_detached_val) if semi_detached_val else 0

        # Calculate averages for terraced houses
        avg_sqft_ter5, avg_val_ter5 = calculate_averages(terraced_beds5)
        avg_sqft_ter4, avg_val_ter4 = calculate_averages(terraced_beds4)
        avg_sqft_ter3, avg_val_ter3 = calculate_averages(terraced_beds3)
        avg_sqft_ter2, avg_val_ter2 = calculate_averages(terraced_beds2)
        avg_sqft_ter1, avg_val_ter1 = calculate_averages(terraced_beds1)
        avg_sqft_terraced = sum(terraced_sqft) / len(terraced_sqft) if terraced_sqft else 0
        avg_val_terraced = sum(terraced_val) / len(terraced_val) if terraced_val else 0

        # Calculate averages for flats
        avg_sqft_flat5, avg_val_flat5 = calculate_averages(flat_beds5)
        avg_sqft_flat4, avg_val_flat4 = calculate_averages(flat_beds4)
        avg_sqft_flat3, avg_val_flat3 = calculate_averages(flat_beds3)
        avg_sqft_flat2, avg_val_flat2 = calculate_averages(flat_beds2)
        avg_sqft_flat1, avg_val_flat1 = calculate_averages(flat_beds1)
        avg_sqft_flat = sum(flat_sqft) / len(flat_sqft) if flat_sqft else 0
        avg_val_flat = sum(flat_val) / len(flat_val) if flat_val else 0

        """
        https://www.getagent.co.uk/postcode/CT21 gives me the average price of for the no. of beds for a postcode domain
        https://www.home.co.uk/guides/sold_house_prices.htm?location=ct21&latest=1 gives me the average price of a house type in a postcode domain

        https://themovemarket.com/tools/historicpropertysaledata/ct215pe gives me the house type, beds, size, and valuation of all houses in a postcode subdomain

        These can be used in conjunction with the data from chimnie to create a baseline for the area a house is in, so that it is easier for a model to predict the price of a house
        i.e. it knows the price of the average house in an area, and compares the current house to the average to help predict the price of the house.
        """

        
       
        tryCount=0
        i+=1
    except:
        if tryCount < 9:
            tryCount +=1
        else:
            i+=1
            tryCount=0
        
        if not driver:
            driver = initialize_driver()
            driverReset=0

async def main7():
    global driver
    global row
    global parsed_dataout
    global cookiecount
    """[REDACTED_BY_SCRIPT]"""
    driver = initialize_driver()
    await scrape_with_timeout7(300)
    driver.quit()  # Ensure the driver is closed after scraping


"""
start script

bricksandlogic

homipi

chimnie

mouseprice

streetscan
"""

start_whcih_website="homipi"
if __name__ == "__main__":
    # asyncio.run(main())
    # time.sleep(5)
    # asyncio.run(main2())
    # time.sleep(5)
    # asyncio.run(main3())
    # time.sleep(5)
    # asyncio.run(main4())
    # time.sleep(5)
    # asyncio.run(main5())
    # time.sleep(5)
    # asyncio.run(main6())
    # time.sleep(5)
    # asyncio.run(main6())
    if start_whcih_website == "bricksandlogic":
        asyncio.run(main3())
    elif start_whcih_website == "homipi":
        asyncio.run(main2())
    elif start_whcih_website == "misc":
        asyncio.run(main6())