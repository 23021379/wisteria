import ast
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException #Import exception
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
import argparse  # Import argparse
import logging
import shutil
import urllib
import multiprocessing
from fake_useragent import UserAgent

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

def scrape_with_timeout(input_file, output_file, timeout_seconds=420):
    global driver
    global loopcount
    global retryattempt
    parsed_data = []
    
    # Load the CSV data
    with open(input_file, 'r', newline='', encoding='utf-8') as csvfile:
        csv_reader = csv.reader(csvfile)
        header = next(csv_reader)  # skip header
        
        for row in csv_reader:
            parsed_row = []
            for cell in row:
                try:
                    parsed_cell = ast.literal_eval(cell.strip())
                except (SyntaxError, ValueError):
                    parsed_cell = cell.strip()
                parsed_row.append(parsed_cell)
            if parsed_row not in parsed_data:
                parsed_data.append(parsed_row)
    
    # Initialize driver
    driver = initialize_driver()
    if driver is None:
        logging.error("[REDACTED_BY_SCRIPT]")
        return
    
    scrape_counter = 0  # Add counter to track number of scrapes
    
    try:
        for row in parsed_data:
            completedScrape = False
            retryattempt = 0
            
            while completedScrape == False:
                if not row:  # Skip empty rows
                    completedScrape = True
                    continue
                
                addressa = row[3].replace(",", "")
                addressa = addressa.lower()
                address = row[0]
                price = row[1]
                date = row[2]
                
                try:
                    checkNotFail = scrape_page(addressa, address, price, date, output_file)
                    if checkNotFail == False:
                        raise WebDriverException("epic fail")
                    
                    # Successful scrape
                    completedScrape = True
                    scrape_counter += 1  # Increment counter
                    
                    # Only reinitialize every 10 scrapes or if there were errors
                    if scrape_counter % 10 == 0:
                        logging.info(f"[REDACTED_BY_SCRIPT]")
                        try:
                            driver.quit()
                            loopcount = 0
                            driver = initialize_driver()
                        except Exception as e:
                            logging.error(f"[REDACTED_BY_SCRIPT]")
                            driver = initialize_driver()  # Try again if it fails
                            
                except TimeoutException:
                    print("[REDACTED_BY_SCRIPT]")
                    try:
                        driver.quit()  # Quit the current driver
                        loopcount = 0
                        driver = initialize_driver()  # Re-initialize inside the loop
                    except:
                        pass
                    continue
                    
                except WebDriverException as e:
                    print(f"[REDACTED_BY_SCRIPT]")
                    try:
                        driver.quit()  # Quit and try again
                        loopcount = 0  # Reset
                        driver = initialize_driver()  # reinitialize
                    except:
                        pass  # Handle driver quit failure
                    
                    if retryattempt < 3:  # retry
                        retryattempt += 1
                    else:
                        completedScrape = True  # quit on more than 3 failures
                    continue
                    
                except Exception as e:
                    print(f"[REDACTED_BY_SCRIPT]")
                    try:
                        driver.quit()  # Quit the current driver
                        loopcount = 0
                        driver = initialize_driver()
                    except:
                        pass
                        
                    if retryattempt < 3:
                        retryattempt += 1
                    else:
                        completedScrape = True
                    continue
                    
            loopcount = 0  # Reset loopcount after each successful row scrape
            
        logging.info("Scraper exited loop")
        
    finally:
        # Ensure driver is always closed at the end
        if driver:
            try:
                driver.quit()
            except:
                pass
                
    logging.info("Scraper closed properly")
    return  # No return value is needed


def initialize_driver():
    global driver
    global loopcount
    global retryattempt
    options = uc.ChromeOptions()
    # Add your Chrome options here
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')

    # Add user-agent rotation:
    ua = UserAgent(browsers=['chrome'])
    user_agent = ua.random  # get a random UA
    options.add_argument(f'[REDACTED_BY_SCRIPT]')  # set it
    chromedriver_path = os.environ.get("[REDACTED_BY_SCRIPT]")
    if not chromedriver_path:
        raise ValueError("[REDACTED_BY_SCRIPT]")

    # Initialize Undetected Chromedriver
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            driver = uc.Chrome(driver_executable_path=chromedriver_path, options=options)
            print(f"[REDACTED_BY_SCRIPT]")
            return driver  # Return the driver if initialization is successful
        except WebDriverException as e:
            print(f"[REDACTED_BY_SCRIPT]")
            if attempt < max_attempts - 1:
                print("[REDACTED_BY_SCRIPT]")
                time.sleep(5)
            else:
                print("[REDACTED_BY_SCRIPT]")
                raise  # Re-raise the exception if max attempts are reached
        except Exception as e:
            print(f"[REDACTED_BY_SCRIPT]")
            raise

    return None  # Return None if all attempts fail (though exception will likely be raised first)

loopcount = 0

def scrape_page(addressa, address, price, date, output_file): #added driver as parameter
    global driver
    global loopcount
    global retryattempt
    """
    Your scraping logic goes here.  This function should be synchronous.
    """
    global loopcount #removed globals, use parameters
    try:
        if loopcount == 0:
            driver.get("[REDACTED_BY_SCRIPT]")

        timeout = 15

        if loopcount == 0:
            ####clicks cookie thing
            #accept
            try:
                wait = WebDriverWait(driver, 15)
                element_locator = (By.CSS_SELECTOR, '[REDACTED_BY_SCRIPT]')
                element = wait.until(EC.presence_of_element_located(element_locator))
                driver.find_element(By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]").click()
            except:
                pass
            loopcount += 1

        element_present = EC.presence_of_element_located((By.CSS_SELECTOR, '[REDACTED_BY_SCRIPT]'))
        WebDriverWait(driver, timeout).until(element_present)
        driver.get(addressa)

        wait = WebDriverWait(driver, 15)
        element_locator = (By.CSS_SELECTOR, '[REDACTED_BY_SCRIPT]')
        wait.until(EC.presence_of_element_located(element_locator))
        housetype = driver.find_element(By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]").text
        housetype = housetype[housetype.index("\n"):]
        try:
            beds = driver.find_element(By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]").text
            beds = beds[beds.index("\n"):]
        except:
            beds = ""
        try:
            baths = driver.find_element(By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]").text
            baths = baths[baths.index("\n"):]
        except:
            baths = ""

        prevsold = ["", "", "", "", "", "", "", "", "", ""]
        for j in range(5):
            try:
                prevsold[(0 * j) + j] = driver.find_element(By.CSS_SELECTOR,
                    f"[REDACTED_BY_SCRIPT]").text
            except:
                prevsold[(0 * j) + j] = ""
            try:
                prevsold[(0 * j) + 1] = driver.find_element(By.CSS_SELECTOR,
                    f"[REDACTED_BY_SCRIPT]").text
            except:
                prevsold[(0 * j) + 1] = ""
        try:
            lastListed=driver.find_element(By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]").text
            lastListed=lastListed[-4:]
        except:
            lastListed="2024"
        hawktuah = address.replace(",", "").lower()
        for j in range(50):
            try:
                imgsrc = driver.find_element(By.CSS_SELECTOR,
                    f"[REDACTED_BY_SCRIPT]").get_attribute(
                    "src")
                response = requests.get(imgsrc) #put this outside, put wait inside

                if response.status_code == 200:
                    #add delay here
                    delay = random.uniform(0.5, 2)  # Small delay, but crucial
                    time.sleep(delay)
                    newpath = f'[REDACTED_BY_SCRIPT]'
                    if not os.path.exists(newpath):
                        os.makedirs(newpath)
                    with open(f"[REDACTED_BY_SCRIPT]",
                              "wb") as file:
                        file.write(response.content)

            except:
                pass

        hasPics = False
        try:
            floorplansrc = driver.find_element(By.CSS_SELECTOR,
                f"[REDACTED_BY_SCRIPT]").get_attribute(
                "href")
            imagessrc = driver.find_element(By.CSS_SELECTOR,
                f"[REDACTED_BY_SCRIPT]").get_attribute(
                "href")
            driver.get(floorplansrc)
            wait = WebDriverWait(driver, 15)
            element_locator = (By.CSS_SELECTOR,
                '[REDACTED_BY_SCRIPT]')
            wait.until(EC.presence_of_element_located(element_locator))
            floorplansrc = driver.find_element(By.CSS_SELECTOR,
                f"[REDACTED_BY_SCRIPT]").get_attribute(
                "src")
            response = requests.get(floorplansrc)

            #add delay here
            delay = random.uniform(0.5, 2)  # Small delay, but crucial
            time.sleep(delay)

            hawktuah = address.replace(",", "").lower()
            if response.status_code == 200:
                newpath = f'[REDACTED_BY_SCRIPT]'
                if not os.path.exists(newpath):
                    os.makedirs(newpath)
                with open(f"[REDACTED_BY_SCRIPT]",
                          "wb") as file:
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
                        #add delay here
                        delay = random.uniform(0.5, 2)  # Small delay, but crucial
                        time.sleep(delay)
                        newpath = f'[REDACTED_BY_SCRIPT]'
                        if not os.path.exists(newpath):
                            os.makedirs(newpath)
                        with open(
                                f"[REDACTED_BY_SCRIPT]",
                                "wb") as file:
                            file.write(response.content)
                except:
                    pass
            hasPics = True
        except:
            try:
                imagesrc = driver.find_element(By.CSS_SELECTOR,
                    f"[REDACTED_BY_SCRIPT]").get_attribute(
                    "href")
                driver.get(imagesrc)
                wait = WebDriverWait(driver, 15)
                element_locator = (By.CSS_SELECTOR, '#media0 > img')
                wait.until(EC.presence_of_element_located(element_locator))
                for j in range(50):
                    try:
                        imgsrc = driver.find_element(By.CSS_SELECTOR, f"#media{j} > img").get_attribute("src")
                        response = requests.get(imgsrc)

                        if response.status_code == 200:
                            #add delay here
                            delay = random.uniform(0.5, 2)  # Small delay, but crucial
                            time.sleep(delay)

                            newpath = f'[REDACTED_BY_SCRIPT]'
                            if not os.path.exists(newpath):
                                os.makedirs(newpath)
                            with open(
                                    f"[REDACTED_BY_SCRIPT]",
                                    "wb") as file:
                                file.write(response.content)
                    except:
                        pass
                hasPics = True
            except:
                hasPics = False
                pass

        if hasPics == True:
            with open(output_file, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile, delimiter=",", quotechar='"')
                writer.writerow([[address.replace(",", "").lower(), addressa], [address, housetype, beds, baths], prevsold])
        # driver.quit()
        # loopcount =0
    except Exception as e:
        print(f"[REDACTED_BY_SCRIPT]")
        return False


def main(input_file, output_file):
    """[REDACTED_BY_SCRIPT]"""
    #Removed async

    scrape_with_timeout(input_file, output_file, 420)
    #Removed async

    #Removed driver quit

logging.basicConfig(level=logging.INFO, encoding='utf-8',
                    format='[REDACTED_BY_SCRIPT]')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Selenium scraper")
    parser.add_argument("--input", required=True, help="[REDACTED_BY_SCRIPT]")
    parser.add_argument("--output", required=True, help="Output CSV file")
    args = parser.parse_args()
    main(args.input, args.output) #Removed unused argument