from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.keys import Keys
import time
import random
import csv
import undetected_chromedriver as uc
from difflib import get_close_matches
options = uc.ChromeOptions()
options.add_argument('--disable-gpu')
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
options.page_load_strategy = 'eager'
time.sleep(1)
driver = uc.Chrome(options=options)

inputsite="[REDACTED_BY_SCRIPT]"
inputsitechecker=inputsite
inputaddress="[REDACTED_BY_SCRIPT]"

driver.get("[REDACTED_BY_SCRIPT]")
time.sleep(random.uniform(0.5, 2))
driver.find_element(By.CSS_SELECTOR, "#sb_form_q").send_keys(f"[REDACTED_BY_SCRIPT]")
driver.find_element(By.CSS_SELECTOR, "#sb_form_q").send_keys(Keys.ENTER)
time.sleep(5)
checkL1=[]
checkL2=[]
for j in range(1,10):
    try:
        element_present = EC.presence_of_element_located((By.XPATH, f"[REDACTED_BY_SCRIPT]"))
        WebDriverWait(driver, timeout=1).until(element_present)
        
        for i in range(10):
            if driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text == inputsitechecker:
                checkL1.append(driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").get_attribute("href"))
                checkL2.append(driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]").text)
        break
    except:pass
print(checkL1)
print(checkL2)
matches = get_close_matches(inputaddress, checkL2, n=1, cutoff=0.1)
print(checkL1[checkL2.index(matches[0])])

for i in range(len(checkL2)):
    if inputaddress.replace(",","").replace(" ","").replace("-","").lower() in checkL2[i].replace(",","").replace(" ","").replace("-","").lower():
        print(checkL1[i])