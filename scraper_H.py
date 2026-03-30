from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

import time
import csv
import math
import random
import time
import Levenshtein
import asyncio
import ast
import traceback
from fake_useragent import UserAgent
import traceback
import logging

import scraper_misc
import scraper_driver







async def scrape_with_timeout2(timeout_seconds,log_filename,driver,row,tryCount,driverReset,completedScrape,i):  
    parsed_data = []
    parsed_dataout=[]
    with open(r"[REDACTED_BY_SCRIPT]", 'r', newline='', encoding='utf-8') as csvfile:
        csv_reader = csv.reader(csvfile)
        for jk in range(0):
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
    for loop in range(len(parsed_data)): # Assuming you process one item from parsed_data per loop
        completedScrape = False
        retryattempt = 0
        # Maybe reset tryCount here too if it's per outer item?
        # global tryCount
        # tryCount = 0
        print(f"[REDACTED_BY_SCRIPT]")
        while not completedScrape and retryattempt < 3: # Combine retry limit here
            driver_needs_refresh = False
            try:
                if not driver:
                    print("[REDACTED_BY_SCRIPT]")
                    driver = scraper_driver.initialize_driver(driver,driverReset,log_filename)
                    # Reset driver specific flags if needed, e.g., driverReset = 0

                # --- Core scraping attempt ---
                print(f"[REDACTED_BY_SCRIPT]")
                await asyncio.wait_for(asyncio.to_thread(scrape_page2, parsed_data, log_filename,driver,row,tryCount,driverReset,completedScrape,i), timeout=timeout_seconds)
                # --- End core scraping attempt ---

                # If scrape_pae2 finishes without error, it should set completedScrape = True
                if not completedScrape:
                     # This case implies scrape_pae2 finished but didn't set the flag - potential logic error?
                     # Or it hit its internal retry limit (tryCount)
                     print(f"[REDACTED_BY_SCRIPT]")
                     completedScrape = True # Force completion for this item if scrape_pae2 indicates failure/retry limit hit

            except asyncio.TimeoutError:
                print(f"[REDACTED_BY_SCRIPT]")
                driver_needs_refresh = True
                retryattempt += 1
            except Exception as e: # Catch specific exceptions from scrape_pe2 if possible
                print(f"[REDACTED_BY_SCRIPT]")
                print(traceback.format_exc())
                driver_needs_refresh = True
                retryattempt += 1
            finally:
                # --- GUARANTEED CLEANUP (if needed) ---
                if driver_needs_refresh:
                    print("[REDACTED_BY_SCRIPT]")
                    try:
                        if driver:
                            print("[REDACTED_BY_SCRIPT]")
                            driver.quit()
                            print("[REDACTED_BY_SCRIPT]")
                    except Exception as quit_err:
                        # Log the error from quit() but continue
                        print(f"[REDACTED_BY_SCRIPT]")
                    finally:
                        # Ensure driver variable is reset regardless of quit success
                        driver = None
                        # Always try to terminate processes after a failure/timeout
                        print("[REDACTED_BY_SCRIPT]")
                        found_pids = scraper_driver.find_chromedriver_processes()
                        scraper_driver.terminate_processes(found_pids)
                        print("[REDACTED_BY_SCRIPT]")
                        # Small delay before potential next attempt's initialization
                        time.sleep(2)

        if not completedScrape:
            print(f"[REDACTED_BY_SCRIPT]")


def check_homipi_cookie(driver):
    for kk in range(5):
        try:
            element_check=driver.find_element(By.CSS_SELECTOR, "#content > p")
            element_check=scraper_misc.get_element_text_robustly(driver, element_check, prefer_text_content=False)
        except:
            element_check=""
        if "Please click below" in element_check:
            try:
                element = WebDriverWait(driver, 0.5).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]"))
                )
                element.click()
            except:pass
    return

def scrape_page2(parsed_data,log_filename,driver,row,tryCount,driverReset,completedScrape,i):
    item_succeeded = False
    
    #for loop in range(len(parsed_data)):
    if not driver:
        time.sleep(1)
        driver = scraper_driver.initialize_driver(driver,driverReset,log_filename)
    try:
        addressin=parsed_data[i][1][0]
        address_parts = addressin.split(" ")
        if len(address_parts) < 2:
            print(f"[REDACTED_BY_SCRIPT]'{addressin}'")
            logging.error(f"[REDACTED_BY_SCRIPT]'{addressin}'")
            # Decide how to handle: skip or raise specific error
            raise ValueError(f"[REDACTED_BY_SCRIPT]") # Raise error to be caught below
        addressin2=addressin.split(" ")[-2]+"-"+addressin.split(" ")[-1]
        addressurl = "[REDACTED_BY_SCRIPT]"+addressin2+"?page=1"
        scraper_driver.check_driver_working(driver, addressurl)
        #time.sleep(10000)
        if driverReset == 0:
            try:
                element_present = EC.presence_of_element_located((By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]"))
                WebDriverWait(driver, 5).until(element_present)
                driver.find_element(By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]").click()
            except:
                pass
            driverReset=1
        #HOMIPI HAS A POPUP THAT BLOCKS THE PAGE WHEN IT DETECTS TOO MANY REQUESTS.
        
        check_homipi_cookie(driver)


        try:
            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]")))
            number_of_props_element = driver.find_element(By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]")
            number_of_props = scraper_misc.get_element_text_robustly(driver, number_of_props_element, prefer_text_content=False)
            print(number_of_props)
        except:
            number_of_props="There are 10 properties"
        try:
            number_of_props=number_of_props[(number_of_props.index("There are ") + len("There are ")):(number_of_props.index(" properties"))]
        except:
            try:
                WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]")))
                number_of_props_element = driver.find_element(By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]")
                number_of_props = scraper_misc.get_element_text_robustly(driver, number_of_props_element, prefer_text_content=False)
                print(number_of_props)
            except:
                number_of_props="There are 10 properties"
            try:
                number_of_props=number_of_props[(number_of_props.index("There are ") + len("There are ")):(number_of_props.index(" properties"))]
            except Exception as e:
                print(e)
        print(number_of_props)

        ##### SOMETIMES IT GETS STUCK ON THE COOKIE MENU, IDK WHY. THIS REPEATEDLY CHECKS IF IT IS STUCK ON THE COOKIE MENU; IF IT IS IT CLICKS THE BUTTON AND CHECKS AGAIN.
        contact_page=False
        if "[REDACTED_BY_SCRIPT]" in number_of_props:
            contact_page=True
        while contact_page == True:
            check_homipi_cookie(driver)
            try:
                WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]")))
                number_of_props_element = driver.find_element(By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]")
                number_of_props = scraper_misc.get_element_text_robustly(driver, number_of_props_element, prefer_text_content=False)
                print(number_of_props)
            except:
                number_of_props="There are 10 properties"
            try:
                number_of_props=number_of_props[(number_of_props.index("There are ") + len("There are ")):(number_of_props.index(" properties"))]
            except:
                try:
                    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]")))
                    number_of_props_element = driver.find_element(By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]")
                    number_of_props = scraper_misc.get_element_text_robustly(driver, number_of_props_element, prefer_text_content=False)
                    print(number_of_props)
                except:
                    number_of_props="There are 10 properties"
                try:
                    number_of_props=number_of_props[(number_of_props.index("There are ") + len("There are ")):(number_of_props.index(" properties"))]
                except Exception as e:
                    print(e)
            if "[REDACTED_BY_SCRIPT]" in number_of_props:
                contact_page=True
            else:
                contact_page=False
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
                for kl in range(1,20):
                    try:
                        address_homipi=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]")
                        href_homipi=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]")
                        address_text = scraper_misc.get_element_text_robustly(driver, address_homipi, prefer_text_content=False)
                        
                        print(address_text)
                        compare_addresses.append(address_text)
                        compare_addresses_href.append(href_homipi.get_attribute("href"))
                        try:
                            element_present = EC.presence_of_element_located((By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]"))
                            WebDriverWait(driver, 2).until(element_present)
                        except:
                            print("no more elements")
                            pass
                    except Exception as e:
                        print("no more elements")
                        pass
                if blibidi_alert==False:
                    addressurl = "[REDACTED_BY_SCRIPT]"+addressin2+f"?page={jk+2}"
                    scraper_driver.check_driver_working(driver, addressurl)
                    time.sleep(0.5)
                    url_confirmed=False
                    url_confirmed_count=0
                    while url_confirmed==False:
                        if addressurl not in driver.current_url:
                            if url_confirmed_count < 5:
                                print(f"[REDACTED_BY_SCRIPT]")
                                driver.get(addressurl)
                                time.sleep(0.5)
                                check_homipi_cookie(driver)
                                url_confirmed_count+=1
                            else:
                                url_confirmed=True
                        else:
                            url_confirmed=True
                        
        except Exception as e:
            print(e)
            pass
        
        target_address = addressin
        most_similar = None
        most_similar_href=None
        min_distance = float('inf')  # Initialize with a very large distance

        for entry in compare_addresses:
            distance = Levenshtein.distance(target_address.lower().replace(",","").replace("-","").replace(" ","").replace(".",""), entry.lower().replace(",","").replace("-","").replace(" ","").replace(".",""))
            print(target_address, entry, distance)
            if distance < min_distance:
                min_distance = distance
                most_similar = entry
                most_similar_href=compare_addresses_href[compare_addresses.index(entry)]

        if most_similar:
            addressurl=most_similar_href
        else:
            addressurl=""
            print("[REDACTED_BY_SCRIPT]")
        scraper_driver.check_driver_working(driver, addressurl)

        if driverReset == 0:
            try:
                element_present = EC.presence_of_element_located((By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]"))
                WebDriverWait(driver, 5).until(element_present)
                driver.find_element(By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]").click()
            except:
                pass
            driverReset=1

        time.sleep(random.uniform(0.8, 1.5))
        est, change, range0, conf, lastp, lastd, property_type, property_subtype, beds, receps, extens, storey, sqm, tenure, epccurrent, epcpotential, council, councilband, permission, age, flood, la = \
        [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], ['']
        for jk in range(1, 22):
            try:
                check_key_element = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]")
                check_key = scraper_misc.get_element_text_robustly(driver, check_key_element, prefer_text_content=False)
                check_key=check_key.split("\n")
                key_homipi=check_key[0]
                value_homipi=check_key[1]
                if "homipi price estimate" in key_homipi.lower():
                    est=value_homipi.replace("£","").replace(",","")
                    est=[est]
                if "value change" in key_homipi.lower():
                    change=value_homipi.replace("£","").replace(",","")
                    change=[change]
                if "price range" in key_homipi.lower():
                    range0=value_homipi.replace("£","").replace(",","")
                    range0=[range0]
                if "estimate confidence" in key_homipi.lower():
                    conf=value_homipi.replace("£","").replace(",","")
                    if "High" in conf:conf=3
                    elif "Moderate" in conf:conf=2
                    elif "Low" in conf:conf=1
                    else:conf=0
                    conf=[conf]
                if "last sold price" in key_homipi.lower():
                    lastp=value_homipi.replace("£","").replace(",","")
                    lastp=[lastp]
                if "last sold date" in key_homipi.lower():
                    lastd=value_homipi.replace("£","").replace(",","")
                    lastd=[lastd]
                if "type" in key_homipi.lower() and "sub type" not in key_homipi.lower():
                    property_type=value_homipi.replace("£","").replace(",","")
                    property_type=[property_type]
                if "sub type" in key_homipi.lower():
                    property_subtype=value_homipi.replace("£","").replace(",","")
                    if "Detached" in property_subtype:property_subtype=4
                    elif "Semi-Detached" in property_subtype:property_subtype=3
                    elif "Terraced" in property_subtype:property_subtype=2
                    else:property_subtype=1
                    property_subtype=[property_subtype]
                if "bedrooms" in key_homipi.lower():
                    beds=value_homipi.replace("£","").replace(",","")
                    beds=[beds]
                if "receptions" in key_homipi.lower():
                    receps=value_homipi.replace("£","").replace(",","")
                    receps=[receps]
                if "extensions" in key_homipi.lower():
                    extens=value_homipi.replace("£","").replace(",","")
                    extens=[extens]
                if "storey" in key_homipi.lower():
                    storey=value_homipi.replace("£","").replace(",","")
                    storey=[storey]
                if "floor area" in key_homipi.lower():
                    sqm=value_homipi.replace(",","")
                    sqm=[sqm]
                if "tenure" in key_homipi.lower():
                    tenure=value_homipi
                    if "Freehold" in tenure:tenure=2
                    else:tenure=1
                    tenure=[tenure]
                if "current epc" in key_homipi.lower():
                    epccurrent=value_homipi
                    epccurrent=epccurrent[epccurrent.index("/")+2:]
                    epccurrent=[epccurrent]
                if "potential epc" in key_homipi.lower():
                    epcpotential=value_homipi
                    epcpotential=epcpotential[epcpotential.index("/")+2:]
                    epcpotential=[epcpotential]
                if "council tax rate" in key_homipi.lower():
                    council=value_homipi
                    council=[council]
                if "council tax band" in key_homipi.lower():
                    councilband=value_homipi
                    councilband=[councilband]
                if "permission" in key_homipi.lower():
                    permission=value_homipi
                    permission=[permission]
                if "new build" in key_homipi.lower():
                    age=value_homipi
                    if "No" in age:
                        try:
                            age = age[(age.index("-")+1):-1]
                        except:
                            age = "2025"
                    else: age = "2025"
                    age=[age]
                if "flood risk" in key_homipi.lower():
                    flood=value_homipi
                    flood=[flood]
                if "local authority" in key_homipi.lower():
                    la=value_homipi  
                    la=[la]         
            except:pass

        try:
            rail_element = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]")
            rail = scraper_misc.get_element_text_robustly(driver, rail_element, prefer_text_content=False)
            rail = rail.split("\n")
        except:
            rail=[]
        try:
            bus_element = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]")
            bus = scraper_misc.get_element_text_robustly(driver, bus_element, prefer_text_content=False)
            bus = bus.split("\n")
        except:
            bus=[]
        
        try:
            Pschool_element = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]")
            Pschool = scraper_misc.get_element_text_robustly(driver, Pschool_element, prefer_text_content=False)
            Pschool = Pschool.split("\n")
        except:
            Pschool=[]
        try:
            Sschool_element = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]")
            Sschool = scraper_misc.get_element_text_robustly(driver, Sschool_element, prefer_text_content=False)
            Sschool = Sschool.split("\n")
        except:
            Sschool=[]
        try:
            Nursery_element = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]")
            Nursery = scraper_misc.get_element_text_robustly(driver, Nursery_element, prefer_text_content=False)
            Nursery = Nursery.split("\n")
        except:
            Nursery=[]
        try:
            special_element = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]")
            special = scraper_misc.get_element_text_robustly(driver, special_element, prefer_text_content=False)
            special = special.split("\n")
        except:
            special=[]

        try:    
            Churches_element = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]")
            Churches = scraper_misc.get_element_text_robustly(driver, Churches_element, prefer_text_content=False)
            Churches = Churches.split("\n")
        except:
            Churches=[]
        try:
            Mosque_element = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]")
            Mosque = scraper_misc.get_element_text_robustly(driver, Mosque_element, prefer_text_content=False)
            Mosque = Mosque.split("\n")
        except:
            Mosque=[]
        try:
            Gurdwara_element = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]")
            Gurdwara = scraper_misc.get_element_text_robustly(driver, Gurdwara_element, prefer_text_content=False)
            Gurdwara = Gurdwara.split("\n")
        except:
            Gurdwara=[]
        try:
            Synagogue_element = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]")
            Synagogue = scraper_misc.get_element_text_robustly(driver, Synagogue_element, prefer_text_content=False)
            Synagogue = Synagogue.split("\n")
        except:
            Synagogue=[]
        try:
            Mandir_element = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]")
            Mandir = scraper_misc.get_element_text_robustly(driver, Mandir_element, prefer_text_content=False)
            Mandir = Mandir.split("\n")
        except:
            Mandir=[]

        try:
            gp_element = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]")
            gp = scraper_misc.get_element_text_robustly(driver, gp_element, prefer_text_content=False)
            gp = gp.split("\n")
        except:
            gp=[]
        try:
            dent_element = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]")
            dent = scraper_misc.get_element_text_robustly(driver, dent_element, prefer_text_content=False)
            dent = dent.split("\n")
        except:
            dent=[]
        try:
            hosp_element = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]")
            hosp = scraper_misc.get_element_text_robustly(driver, hosp_element, prefer_text_content=False)
            hosp=hosp.split("\n")
        except:
            hosp=[]
        try:
            pharm_element = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]")
            pharm = scraper_misc.get_element_text_robustly(driver, pharm_element, prefer_text_content=False)
            pharm = pharm.split("\n")
        except:pharm=[]
        try:
            opt_element = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]")
            opt = scraper_misc.get_element_text_robustly(driver, opt_element, prefer_text_content=False)
            opt = opt.split("\n")
        except:opt=[]
        try:
            clin_element = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]")
            clin = scraper_misc.get_element_text_robustly(driver, clin_element, prefer_text_content=False)
            clin = clin.split("\n")
        except:
            clin=[]
        try:
            other_element = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]")
            other = scraper_misc.get_element_text_robustly(driver, other_element, prefer_text_content=False)
            other = other.split("\n")
        except:
            other=[]

        
        parsed_dataout=[[[parsed_data[i][1][0]], est, change, range0, conf, lastp, lastd, property_type, property_subtype, beds, receps, extens, storey, sqm, tenure, epccurrent, epcpotential, council, councilband, permission, age, flood, la, rail, bus, Pschool, Sschool, Nursery, special, Churches, Mosque, Gurdwara, Synagogue, Mandir, gp, dent, hosp, pharm, opt, clin, other]]
        if parsed_dataout == [[[parsed_data[i][1][0]], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], ['']]]:
            print(f"[REDACTED_BY_SCRIPT]")
            raise ValueError("[REDACTED_BY_SCRIPT]")
        with open(r"[REDACTED_BY_SCRIPT]", 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(parsed_dataout)
        tryCount=0
        i+=1
        completedScrape=True
        item_succeeded = True
    except Exception as e:
        print(f"[REDACTED_BY_SCRIPT]")
        print(traceback.format_exc()) # Keep traceback for debugging
        logging.error(f"[REDACTED_BY_SCRIPT]", exc_info=True) # Log detailed error
        tryCount += 1
        item_succeeded = False # Ensure it's False on error
    finally:
        # This block now runs reliably because item_succeeded is always defined
        print(f"[REDACTED_BY_SCRIPT]") # Debug print
        if item_succeeded:
            completedScrape = True # Signal outer loop to stop for this item
            print(f"[REDACTED_BY_SCRIPT]")
            i += 1 # Move to the next item index *only on success*
        elif tryCount >= 10: # Or max internal retries reached
            print(f"[REDACTED_BY_SCRIPT]")
            completedScrape = True # Also signal completion (failure) to the outer loop
            print(f"[REDACTED_BY_SCRIPT]")
            i += 1 # Move to the next item index *after giving up*
            tryCount = 0 # Reset counter for the *next* item
        # else: item failed, but internal retries remain (tryCount < 10), completedScrape stays False
        # No need to increment 'i' here, the outer loop in scrape_with_timeot2 will retry
        print(f"[REDACTED_BY_SCRIPT]") # Debug print




async def main2(driver,row,tryCount,driverReset,completedScrape,i):
    """[REDACTED_BY_SCRIPT]"""
    driver = None # Ensure driver starts as None
    log_filename='[REDACTED_BY_SCRIPT]'
    try:
        print("[REDACTED_BY_SCRIPT]")
        driver = scraper_driver.initialize_driver(driver,driverReset,log_filename) # Initial setup
        print("[REDACTED_BY_SCRIPT]")
        await scrape_with_timeout2(120,log_filename,driver,row,tryCount,driverReset,completedScrape,i) # Run the main loop
        print("[REDACTED_BY_SCRIPT]")
    except Exception as main_err:
        print(f"[REDACTED_BY_SCRIPT]")
        print(traceback.format_exc())
    finally:
        print("[REDACTED_BY_SCRIPT]")
        try:
            if driver:
                print("[REDACTED_BY_SCRIPT]")
                driver.quit()
                print("[REDACTED_BY_SCRIPT]")
        except Exception as quit_err:
            print(f"[REDACTED_BY_SCRIPT]")
        finally:
            # Ensure driver variable is reset
            driver = None
            # Always attempt final process termination
            print("[REDACTED_BY_SCRIPT]")
            found_pids = scraper_driver.find_chromedriver_processes()
            scraper_driver.terminate_processes(found_pids)
            print("[REDACTED_BY_SCRIPT]")