from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.keys import Keys
import time
import csv
import undetected_chromedriver as uc
import math
import random
import time
import Levenshtein

options = uc.ChromeOptions()
options.add_argument('--disable-gpu')
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
options.page_load_strategy = 'eager'
time.sleep(1)
driver = uc.Chrome(options=options)
skib="[REDACTED_BY_SCRIPT]"
for k in range(100):
    try:
        driver.get('[REDACTED_BY_SCRIPT]')
        element_present = EC.presence_of_element_located((By.CSS_SELECTOR, "#bnp_btn_reject"))
        WebDriverWait(driver, 15).until(element_present)
        try:
            driver.find_element(By.CSS_SELECTOR, "#sb_form_c > div").click()
            driver.find_element(By.CSS_SELECTOR, "#sb_form_q").send_keys("site:homipi.co.uk "+skib)
            driver.find_element(By.CSS_SELECTOR, "#sb_form_q").send_keys(Keys.RETURN)
        except:
            try:
                driver.find_element(By.CSS_SELECTOR, "#bnp_btn_reject").click()
            except:
                driver.find_element(By.CSS_SELECTOR, "#bnp_btn_reject > a")
            driver.find_element(By.CSS_SELECTOR, "#sb_form_c > div").click()
            driver.find_element(By.CSS_SELECTOR, "#sb_form_q").send_keys("site:homipi.co.uk "+skib)
            driver.find_element(By.CSS_SELECTOR, "#sb_form_q").send_keys(Keys.RETURN)
                

        element_present = EC.presence_of_element_located((By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]"))
        WebDriverWait(driver, 15).until(element_present)
        potential_found=[]
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
            if "www.homipi.co.uk" in domain and "property" in domain:
                potential_found.append([domain,metadata,href])


        target_address = skib
        most_similar = None
        min_distance = float('inf')  # Initialize with a very large distance

        for entry in potential_found:
            domain, metadata, href = entry
            distance = Levenshtein.distance(target_address.lower().replace(",","").replace("-","").replace(" ","").replace(".",""), metadata.lower().replace(",","").replace("-","").replace(" ","").replace(".",""))  # Compare with metadata
            print(metadata, distance)
            if distance < min_distance:
                min_distance = distance
                most_similar = entry

        if most_similar:
            domain, metadata, href = most_similar
            print("[REDACTED_BY_SCRIPT]")
            print("Domain:", domain)
            print("Metadata:", metadata)
            print("Href:", href)
        else:
            print("[REDACTED_BY_SCRIPT]")
    except Exception as e:
        print(e)
        time.sleep(1000)