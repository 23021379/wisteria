"""
This script takes an address and finds the corresponding pages from chimnie, BnL, mouseprice, and streetscan and scrapes the data.
It uses Selenium for web scraping and handles various exceptions, including custom ones for specific error conditions.
"""


import selenium
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import StaleElementReferenceException

import time
import csv
import undetected_chromedriver as uc
import math
import random
import time
from datetime import datetime
import Levenshtein
import os, os.path
import asyncio
import ast
import traceback
import os
from fake_useragent import UserAgent
import traceback
import psutil
import logging

# --- 1. Set up Logging ---
LOG_FILENAME = '[REDACTED_BY_SCRIPT]'
LOG_FORMAT = '[REDACTED_BY_SCRIPT]'
error410switch=False

# Configure the logger
logging.basicConfig(
    level=logging.WARNING,  # Log WARNING, ERROR, CRITICAL messages
    format=LOG_FORMAT,
    filename=LOG_FILENAME,
    filemode='a'  # Append mode, so logs aren't overwritten each run
)

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


"""
edit init
"""
def initialize_driver():
    global driver
    global loopcount
    global retryattempt
    global start_whcih_website
    global driverReset
    driverReset=0
    # --- Cleanup FIRST ---
    print("[REDACTED_BY_SCRIPT]")
    try:
        if driver:
            print("[REDACTED_BY_SCRIPT]")
            driver.quit()
    except Exception as e:
        print(f"[REDACTED_BY_SCRIPT]")
    finally:
        quit_attempted = False
        quit_attempted_0=False
        while quit_attempted == False:
            try:print(driver)
            except:pass
            print(quit_attempted)
            try:
                driver.quit()
                time.sleep(0.1)
                if quit_attempted_0==True:
                    quit_attempted=True
                else:
                    quit_attempted_0=True
            except WebDriverException as e:
                print(f"[REDACTED_BY_SCRIPT]")
                quit_attempted = True
            except:
                print("[REDACTED_BY_SCRIPT]")
                if driver == None:
                    quit_attempted = True
        driver = None # Ensure driver is None before proceeding
        print("[REDACTED_BY_SCRIPT]")
        # Search for both potential names
        found_pids = find_chromedriver_processes(include_undetected=True)
        terminate_processes(found_pids)
        print("[REDACTED_BY_SCRIPT]")
        time.sleep(0.5) # Small delay

    # --- Attempt Initialization ---
    max_init_retries = 2
    for attempt in range(max_init_retries):
        print(f"[REDACTED_BY_SCRIPT]")
        try:
            # Clear UC cache/lock files if issues persist (Optional but can help)
            uc_cache_path = os.path.join(os.getenv('APPDATA'), 'undetected_chromedriver')
            uc_exe_path = os.path.join(uc_cache_path, '[REDACTED_BY_SCRIPT]')
            if os.path.exists(uc_exe_path) and attempt > 0: # Try removing on retry
                print(f"[REDACTED_BY_SCRIPT]")
                try:
                    os.remove(uc_exe_path)
                    print("[REDACTED_BY_SCRIPT]")
                except OSError as e:
                    print(f"[REDACTED_BY_SCRIPT]")
            # --- End Optional Cache Clear ---

            options = uc.ChromeOptions()
            # ... (your existing options: user-agent, prefs, etc.) ...
            options.add_argument('--disable-gpu')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            ua = UserAgent(browsers=['chrome'])
            user_agent = ua.random
            options.add_argument(f'[REDACTED_BY_SCRIPT]')
            prefs = {"[REDACTED_BY_SCRIPT]": 2}
            options.add_experimental_option("prefs", prefs)

            # Explicitly set executable path to potentially avoid patcher issues if needed
            # (though uc usually handles this)
            # driver_executable_path = r"[REDACTED_BY_SCRIPT]" # If needed
            # driver = uc.Chrome(options=options, use_subprocess=True, driver_executable_path=driver_executable_path)

            driver = uc.Chrome(options=options, use_subprocess=True)

            print("[REDACTED_BY_SCRIPT]")
            # Optional: Add a simple check immediately after init
            # driver.get("chrome://version/")
            # print("[REDACTED_BY_SCRIPT]")
            return driver # SUCCESS

        except FileExistsError as fee:
            print(f"[REDACTED_BY_SCRIPT]")
            logging.error(f"[REDACTED_BY_SCRIPT]", exc_info=True)
            # File is locked, try terminating again
            found_pids = find_chromedriver_processes(include_undetected=True)
            terminate_processes(found_pids)
            if attempt < max_init_retries - 1:
                print("[REDACTED_BY_SCRIPT]")
                time.sleep(3)
            else:
                print("[REDACTED_BY_SCRIPT]")
        except WebDriverException as wde:
             # Check if it's the service exit error
            if "unexpectedly exited" in str(wde):
                print(f"[REDACTED_BY_SCRIPT]")
                logging.error(f"[REDACTED_BY_SCRIPT]", exc_info=True)
                 # Try terminating again
                found_pids = find_chromedriver_processes(include_undetected=True)
                terminate_processes(found_pids)
                if attempt < max_init_retries - 1:
                     print("[REDACTED_BY_SCRIPT]")
                     time.sleep(3)
                else:
                     print("[REDACTED_BY_SCRIPT]")
            else:
                 # Different WebDriverException, treat as fatal for now
                 print(f"[REDACTED_BY_SCRIPT]")
                 logging.error(f"[REDACTED_BY_SCRIPT]", exc_info=True)
                 return None # Fatal error
        except Exception as e:
            print(f"[REDACTED_BY_SCRIPT]")
            print(traceback.format_exc())
            logging.error(f"[REDACTED_BY_SCRIPT]", exc_info=True)
            # Attempt cleanup before potential retry or failure
            found_pids = find_chromedriver_processes(include_undetected=True)
            terminate_processes(found_pids)
            if attempt < max_init_retries - 1:
                print("[REDACTED_BY_SCRIPT]")
                time.sleep(3)
            else:
                 print("[REDACTED_BY_SCRIPT]")

    # If loop finishes without returning, initialization failed
    print("[REDACTED_BY_SCRIPT]")
    return None # IMPORTANT: Return None clearly on failure

loopcount=0





def get_element_text_robustly(driver, element, prefer_text_content=False):
    """
    Attempts to get text from a Selenium WebElement, trying .text first
    and falling back to JavaScript's textContent if .text fails or is empty,
    or if prefer_text_content is True.

    Args:
        driver: The Selenium WebDriver instance.
        element: The Selenium WebElement.
        prefer_text_content: If True, skip .text and go straight to textContent.

    Returns:
        The extracted text string, or None if extraction fails.
    """
    text = None
    try:
        if not prefer_text_content:
            text = element.text

        # If .text returned nothing OR if we prefer textContent anyway
        if prefer_text_content or (text is not None and not text.strip()):
            try:
                # Use execute_script for potentially more reliable text extraction
                text = driver.execute_script("[REDACTED_BY_SCRIPT]", element)
            except Exception as js_error:
                # Log if JS execution fails for some reason
                print(f"[REDACTED_BY_SCRIPT]")
                # If .text had a value before, maybe keep it? Or set text to None.
                # Setting to None here to indicate failure.
                text = None # Fallback if JS fails

    except StaleElementReferenceException:
        print("[REDACTED_BY_SCRIPT]")
        return None # Element is no longer valid
    except Exception as e:
        print(f"[REDACTED_BY_SCRIPT]")
        return None # General error

    # Clean whitespace if text was found
    return text.strip() if text else None



####TERMINATES ALL CHROMEDRIVER INSTANCES RUNNING
PROCESS_NAME_TO_FIND = "chromedriver.exe" # Adjust if necessary for your OS
UC_PROCESS_NAME_TO_FIND = "[REDACTED_BY_SCRIPT]"
def find_chromedriver_processes(include_undetected=True):
    """[REDACTED_BY_SCRIPT]"""
    chromedriver_pids = []
    search_names = [PROCESS_NAME_TO_FIND.lower()]
    if include_undetected:
        search_names.append(UC_PROCESS_NAME_TO_FIND.lower())

    print(f"[REDACTED_BY_SCRIPT]")
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            proc_name_lower = proc.info['name'].lower() if proc.info['name'] else ''
            if any(search_name in proc_name_lower for search_name in search_names):
                print(f"  Found {proc.info['name'[REDACTED_BY_SCRIPT]'pid']}")
                chromedriver_pids.append(proc.info['pid'])
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            logging.warning(f"[REDACTED_BY_SCRIPT]", extra={'process': getattr(proc, 'pid', 0)})
            pass
        except Exception as e:
             logging.error(f"[REDACTED_BY_SCRIPT]", exc_info=True, extra={'process': getattr(proc, 'pid', 0)})
             pass
    # Remove duplicates if any process matched multiple names (unlikely)
    return list(set(chromedriver_pids))

def terminate_processes(pids):
    """[REDACTED_BY_SCRIPT]"""
    if not pids:
        print("[REDACTED_BY_SCRIPT]")
        return

    print("[REDACTED_BY_SCRIPT]")
    for pid in pids:
        log_extra = {'process': pid}
        try:
            process = psutil.Process(pid)
            print(f"[REDACTED_BY_SCRIPT]") # Log name
            process_name = process.name()
            log_extra['processName'] = process_name
            process.terminate()
            print(f"[REDACTED_BY_SCRIPT]")
            try:
                process.wait(timeout=2)
                print(f"[REDACTED_BY_SCRIPT]")
            except psutil.TimeoutExpired:
                print(f"[REDACTED_BY_SCRIPT]")
                logging.warning(f"[REDACTED_BY_SCRIPT]", extra=log_extra)
                process.kill()
                print(f"[REDACTED_BY_SCRIPT]")
                process.wait(timeout=1) # Brief wait after kill
                print(f"[REDACTED_BY_SCRIPT]")
                # Optional: Check if it *really* died
                if psutil.pid_exists(pid):
                    print(f"[REDACTED_BY_SCRIPT]")
                    logging.error(f"[REDACTED_BY_SCRIPT]", extra=log_extra)
                else:
                     print(f"[REDACTED_BY_SCRIPT]")

        except psutil.NoSuchProcess:
            # Log this as a warning, as it's not necessarily an error in the termination logic itself
            logging.warning(f"[REDACTED_BY_SCRIPT]", extra=log_extra)
            # print(f"[REDACTED_BY_SCRIPT]") # Optional: keep console message
        except psutil.AccessDenied:
            # Log this as an error
            logging.error(f"[REDACTED_BY_SCRIPT]", extra=log_extra)
            print(f"[REDACTED_BY_SCRIPT]") # Modify console message
        except Exception as e:
            try:
                # Log any other unexpected error
                logging.error(f"[REDACTED_BY_SCRIPT]", exc_info=True, extra=log_extra) # exc_info=True adds traceback
                print(f"[REDACTED_BY_SCRIPT]") # Modify console message
            except:pass



def check_driver_working(addressurl):
    global driver
    try:
        driver.get(addressurl)
    except:
        driver=initialize_driver()
        time.sleep(0.1)
        driver.get(addressurl)
    return

    ##CHECKS IF THE DRIVER IS WORKING OR NOT
    #DRIVER CAN BREAK, BUT DOESNT APPEAR AS NONETYPE; SO MUST ENSURE THAT IT IS FUNCTIONAL 
    # driver_confirmed=False
    # while driver_confirmed==False:
    #     if driver:
    #         try:
    #             driver.get(addressurl)
    #             driver_confirmed=True
    #         except Exception as e:
    #             print(traceback.format_exc())
    #             print(e)
    #             try:
    #                 driver.quit()
    #                 found_pids = find_chromedriver_processes()
    #                 terminate_processes(found_pids)
    #             except:pass
    #             driver=None
    #             time.sleep(1)
    #             driver = initialize_driver()
    #     else:
    #         while not driver:
    #             found_pids = find_chromedriver_processes()
    #             terminate_processes(found_pids)
    #             driver = initialize_driver()
    #             time.sleep(1)
    #     if driver_confirmed==False:
    #         try:
    #             driver.get(addressurl)
    #             driver_confirmed=True
    #         except Exception as e:
    #             print(traceback.format_exc())
    #             print(e)
    #             found_pids = find_chromedriver_processes()
    #             terminate_processes(found_pids)
    #     else:pass
    # return
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
    global error410switch
    global data_was_written_for_item
    global retryattempt
    global max_retries
    parsed_data = []
    parsed_dataout=[]
    with open(r"[REDACTED_BY_SCRIPT]", 'r', newline='', encoding='utf-8') as csvfile:
        csv_reader = csv.reader(csvfile)
        for jk in range(10351):
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
    
    i=0
    driverReset=0
    max_retries=3
    for loop in range(len(parsed_data)): # Assuming you process one item from parsed_data per loop
        completedScrape = False
        retryattempt = 0
        data_was_written_for_item = False
        tryCount = 0
        item_processed_successfully_or_skipped=False
        print(f"[REDACTED_BY_SCRIPT]")
        while not completedScrape or retryattempt < max_retries: # Combine retry limit here
            driver_needs_refresh = False
            try:
                if not driver:
                    print("[REDACTED_BY_SCRIPT]")
                    driver = initialize_driver()
                    time.sleep(0.1)
                    # Reset driver specific flags if needed, e.g., driverReset = 0

                # --- Core scraping attempt ---
                print(f"[REDACTED_BY_SCRIPT]")
                await asyncio.wait_for(asyncio.to_thread(scrape_page2, parsed_data), timeout=timeout_seconds)
                # --- End core scraping attempt ---

                # If scrape_page2 finishes without error, it should set completedScrape = True
                if not completedScrape:
                     # This case implies scrape_page2 finished but didn't set the flag - potential logic error?
                     # Or it hit its internal retry limit (tryCount)
                     print(f"[REDACTED_BY_SCRIPT]")
                     completedScrape = True # Force completion for this item if scrape_page2 indicates failure/retry limit hit
            except CustomError410 as e_410:
                print(f"[REDACTED_BY_SCRIPT]") # Add debug print
                print(f"[REDACTED_BY_SCRIPT]")
                item_processed_successfully_or_skipped = True
                retryattempt=max_retries
            except asyncio.TimeoutError:
                print(f"[REDACTED_BY_SCRIPT]")
                driver_needs_refresh = True
                retryattempt += 1
            except Exception as e:
                print(f"[REDACTED_BY_SCRIPT]") # Debug print

                # Check if the caught exception IS our custom one, or wraps it
                # This check is crucial
                is_custom_error = isinstance(e, CustomError410)
                # Sometimes the original exception is stored in 'args' or '__cause__'
                is_custom_error_cause = hasattr(e, '__cause__') and isinstance(e.__cause__, CustomError410)

                if is_custom_error or is_custom_error_cause:
                    # Handle as if the specific block caught it
                    actual_error = e if is_custom_error else e.__cause__
                    print(f"[REDACTED_BY_SCRIPT]")
                    print(f"[REDACTED_BY_SCRIPT]")
                    item_processed_successfully_or_skipped = True # Mark as done for retry loop
                    # Since we are skipping successfully, no driver refresh needed for *this specific error*
                    driver_needs_refresh = False
                    # Force exit from retry loop for this item
                    retryattempt = max_retries
                else:
                    # Handle truly unexpected errors
                    print(f"[REDACTED_BY_SCRIPT]")
                    # print(traceback.format_exc()) # Uncomment for full trace if needed
                    driver_needs_refresh = True
                    retryattempt += 1
            finally:
                # --- GUARANTEED CLEANUP (if needed) ---
                if driver_needs_refresh:
                    print("[REDACTED_BY_SCRIPT]")
                    try:
                        if driver:
                            driver.quit()
                    except Exception as quit_err:
                        # Log the error from quit() but continue
                        print(f"[REDACTED_BY_SCRIPT]")
                    finally:
                        # Ensure driver variable is reset regardless of quit success
                        quit_attempted = False
                        quit_attempted_0=False
                        while quit_attempted == False:
                            try:print(driver)
                            except:pass
                            print(quit_attempted)
                            try:
                                driver.quit()
                                time.sleep(0.1)
                                if quit_attempted_0==True:
                                    quit_attempted=True
                                else:
                                    quit_attempted_0=True
                            except WebDriverException as e:
                                print(f"[REDACTED_BY_SCRIPT]")
                                quit_attempted = True
                            except:
                                print("[REDACTED_BY_SCRIPT]")
                                if driver == None:
                                    quit_attempted = True
                        driver = None
                        # Always try to terminate processes after a failure/timeout
                        found_pids = find_chromedriver_processes()
                        terminate_processes(found_pids)
                        print("[REDACTED_BY_SCRIPT]")
                        # Small delay before potential next attempt's initialization
                        time.sleep(2)

        if data_was_written_for_item == False:
            print(f"[REDACTED_BY_SCRIPT]")
            try:
                # Construct the default empty row structure
                default_row_data = [[[parsed_data[i][1][0]]] + [[''] for _ in range(39)]] # Address + 39 empty lists containing empty string

                with open(r"[REDACTED_BY_SCRIPT]", 'a', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerows(default_row_data)


            except Exception as append_err:
                print(f"[REDACTED_BY_SCRIPT]")


        # --- Log outcome and increment i ---
        if completedScrape:
            if data_was_written_for_item:
                print(f"[REDACTED_BY_SCRIPT]")
            else: # Must have been skipped (410)
                print(f"[REDACTED_BY_SCRIPT]")
        else: # This means all retries failed (Timeout or General Error)
            print(f"[REDACTED_BY_SCRIPT]")
        
        if completedScrape == True or item_processed_successfully_or_skipped == True or retryattempt==max_retries:
            i+=1


def check_homipi_cookie():
    global driver
    element_check=""
    for kk in range(5):
        try:
            element_check=driver.find_element(By.CSS_SELECTOR, "#content > p")
            element_check=get_element_text_robustly(driver, element_check, prefer_text_content=False)
        except:
            element_check=""
        try:
            if "Please click below" in element_check:
                try:
                    scale_time=0.5-(kk*0.1)
                    element = WebDriverWait(driver, scale_time).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]"))
                    )
                    element.click()
                except:pass
        except:pass
    return

class CustomError410(Exception):
    """[REDACTED_BY_SCRIPT]"""
    global error410Switch
    error410switch=True
    pass

def scrape_page2(parsed_data):
    global driver
    global row
    global parsed_dataout
    global tryCount
    global i
    global driverReset
    global completedScrape
    global data_was_written_for_item
    global retryattempt
    global max_retries
    item_succeeded = False
    
    #for loop in range(len(parsed_data)):
    if not driver:
        driver = initialize_driver()
    try:
        addressin=parsed_data[i][1][0]
        address_parts = addressin.split(" ")
        if len(address_parts) < 2:
            logging.error(f"[REDACTED_BY_SCRIPT]'{addressin}'")
            # Decide how to handle: skip or raise specific error
            raise ValueError(f"[REDACTED_BY_SCRIPT]") # Raise error to be caught below
        addressin2=addressin.split(" ")[-2]+"-"+addressin.split(" ")[-1]
        addressurl = "[REDACTED_BY_SCRIPT]"+addressin2+"?page=1"
        check_driver_working(addressurl)
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
        
        check_homipi_cookie()


        try:
            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]")))
            number_of_props_element = driver.find_element(By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]")
            number_of_props = get_element_text_robustly(driver, number_of_props_element, prefer_text_content=False)
        except:
            number_of_props="There are 10 properties"
        try:
            number_of_props=number_of_props[(number_of_props.index("There are ") + len("There are ")):(number_of_props.index(" properties"))]
        except:
            try:
                WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]")))
                number_of_props_element = driver.find_element(By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]")
                number_of_props = get_element_text_robustly(driver, number_of_props_element, prefer_text_content=False)
            except:
                number_of_props="There are 10 properties"
            try:
                number_of_props=number_of_props[(number_of_props.index("There are ") + len("There are ")):(number_of_props.index(" properties"))]
            except:
                pass
        
        #quickly check for 410 error
        try:
            WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.CSS_SELECTOR, "#content > p")))
            check_error=driver.find_element(By.CSS_SELECTOR, "#content > p").text
            print(check_error)
            if check_error and "we're sorry" in check_error.lower():
                print(f"[REDACTED_BY_SCRIPT]")
                raise CustomError410("[REDACTED_BY_SCRIPT]")
        except CustomError410: # Catch it here just to re-raise immediately
            raise
        except:pass
        
        ##### SOMETIMES IT GETS STUCK ON THE COOKIE MENU, IDK WHY. THIS REPEATEDLY CHECKS IF IT IS STUCK ON THE COOKIE MENU; IF IT IS IT CLICKS THE BUTTON AND CHECKS AGAIN.
        contact_page=False
        if "[REDACTED_BY_SCRIPT]" in number_of_props:
            contact_page=True
        while contact_page == True:
            check_homipi_cookie()
            try:
                WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]")))
                number_of_props_element = driver.find_element(By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]")
                number_of_props = get_element_text_robustly(driver, number_of_props_element, prefer_text_content=False)
            except:
                number_of_props="There are 10 properties"
            if "[REDACTED_BY_SCRIPT]" in number_of_props:
                contact_page=True
            else:
                contact_page=False
                break
            try:
                number_of_props=number_of_props[(number_of_props.index("There are ") + len("There are ")):(number_of_props.index(" properties"))]
            except:
                try:
                    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]")))
                    number_of_props_element = driver.find_element(By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]")
                    number_of_props = get_element_text_robustly(driver, number_of_props_element, prefer_text_content=False)
                except:
                    number_of_props="There are 10 properties"
                try:
                    number_of_props=number_of_props[(number_of_props.index("There are ") + len("There are ")):(number_of_props.index(" properties"))]
                except:
                    pass
            if "[REDACTED_BY_SCRIPT]" in number_of_props:
                contact_page=True
            else:
                contact_page=False
                break
        try:
            number_of_props=int(number_of_props)
        except:
            try:
                number_of_props=number_of_props[(number_of_props.index("There are ") + len("There are ")):(number_of_props.index(" properties"))]
                number_of_props=int(number_of_props)
            except:
                number_of_props=10
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
                    if kl == 1:
                        try:
                            element_present = EC.presence_of_element_located((By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]"))
                            WebDriverWait(driver, 0.5).until(element_present)
                        except:pass
                    try:
                        address_homipi=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]")
                        href_homipi=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]")
                        address_text = get_element_text_robustly(driver, address_homipi, prefer_text_content=False)
                        
                        compare_addresses.append(address_text)
                        compare_addresses_href.append(href_homipi.get_attribute("href"))
                    except Exception as e:
                        pass
                if blibidi_alert==False:
                    addressurl = "[REDACTED_BY_SCRIPT]"+addressin2+f"?page={jk+2}"
                    check_driver_working(addressurl)
                    time.sleep(0.1)
                    url_confirmed=False
                    url_confirmed_count=0
                    while url_confirmed==False:
                        if addressurl not in driver.current_url:
                            if url_confirmed_count < 5:
                                print(f"[REDACTED_BY_SCRIPT]")
                                driver.get(addressurl)
                                time.sleep(0.1)
                                check_homipi_cookie()
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
            try:
                if entry:
                    distance = Levenshtein.distance(target_address.lower().replace(",","").replace("-","").replace(" ","").replace(".",""), entry.lower().replace(",","").replace("-","").replace(" ","").replace(".",""))
                    if distance < min_distance:
                        min_distance = distance
                        most_similar = entry
                        most_similar_href=compare_addresses_href[compare_addresses.index(entry)]
            except:pass

        if most_similar:
            addressurl=most_similar_href
        else:
            addressurl=""
            print("[REDACTED_BY_SCRIPT]")
        check_driver_working(addressurl)

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
            if jk == 1:
                element_present=EC.presence_of_element_located((By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]"))
                WebDriverWait(driver, 5).until(element_present)
            try:
                check_key_element = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]")
                check_key = get_element_text_robustly(driver, check_key_element, prefer_text_content=False)
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
            rail = get_element_text_robustly(driver, rail_element, prefer_text_content=False)
            rail = rail.split("\n")
        except:
            rail=[]
        try:
            bus_element = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]")
            bus = get_element_text_robustly(driver, bus_element, prefer_text_content=False)
            bus = bus.split("\n")
        except:
            bus=[]
        
        try:
            Pschool_element = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]")
            Pschool = get_element_text_robustly(driver, Pschool_element, prefer_text_content=False)
            Pschool = Pschool.split("\n")
        except:
            Pschool=[]
        try:
            Sschool_element = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]")
            Sschool = get_element_text_robustly(driver, Sschool_element, prefer_text_content=False)
            Sschool = Sschool.split("\n")
        except:
            Sschool=[]
        try:
            Nursery_element = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]")
            Nursery = get_element_text_robustly(driver, Nursery_element, prefer_text_content=False)
            Nursery = Nursery.split("\n")
        except:
            Nursery=[]
        try:
            special_element = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]")
            special = get_element_text_robustly(driver, special_element, prefer_text_content=False)
            special = special.split("\n")
        except:
            special=[]

        try:    
            Churches_element = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]")
            Churches = get_element_text_robustly(driver, Churches_element, prefer_text_content=False)
            Churches = Churches.split("\n")
        except:
            Churches=[]
        try:
            Mosque_element = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]")
            Mosque = get_element_text_robustly(driver, Mosque_element, prefer_text_content=False)
            Mosque = Mosque.split("\n")
        except:
            Mosque=[]
        try:
            Gurdwara_element = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]")
            Gurdwara = get_element_text_robustly(driver, Gurdwara_element, prefer_text_content=False)
            Gurdwara = Gurdwara.split("\n")
        except:
            Gurdwara=[]
        try:
            Synagogue_element = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]")
            Synagogue = get_element_text_robustly(driver, Synagogue_element, prefer_text_content=False)
            Synagogue = Synagogue.split("\n")
        except:
            Synagogue=[]
        try:
            Mandir_element = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]")
            Mandir = get_element_text_robustly(driver, Mandir_element, prefer_text_content=False)
            Mandir = Mandir.split("\n")
        except:
            Mandir=[]

        try:
            gp_element = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]")
            gp = get_element_text_robustly(driver, gp_element, prefer_text_content=False)
            gp = gp.split("\n")
        except:
            gp=[]
        try:
            dent_element = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]")
            dent = get_element_text_robustly(driver, dent_element, prefer_text_content=False)
            dent = dent.split("\n")
        except:
            dent=[]
        try:
            hosp_element = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]")
            hosp = get_element_text_robustly(driver, hosp_element, prefer_text_content=False)
            hosp=hosp.split("\n")
        except:
            hosp=[]
        try:
            pharm_element = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]")
            pharm = get_element_text_robustly(driver, pharm_element, prefer_text_content=False)
            pharm = pharm.split("\n")
        except:pharm=[]
        try:
            opt_element = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]")
            opt = get_element_text_robustly(driver, opt_element, prefer_text_content=False)
            opt = opt.split("\n")
        except:opt=[]
        try:
            clin_element = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]")
            clin = get_element_text_robustly(driver, clin_element, prefer_text_content=False)
            clin = clin.split("\n")
        except:
            clin=[]
        try:
            other_element = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]")
            other = get_element_text_robustly(driver, other_element, prefer_text_content=False)
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
        data_was_written_for_item=True
        tryCount=0
        completedScrape=True
        item_succeeded = True
        retryattempt=max_retries
    except CustomError410:
        print(f"[REDACTED_BY_SCRIPT]'succeeded' for progression.")
        item_succeeded=True
        raise
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
        elif tryCount >= 10: # Or max internal retries reached
            print(f"[REDACTED_BY_SCRIPT]")
            completedScrape = True # Also signal completion (failure) to the outer loop
            print(f"[REDACTED_BY_SCRIPT]")
            tryCount = 0 # Reset counter for the *next* item
        # else: item failed, but internal retries remain (tryCount < 10), completedScrape stays False
        # No need to increment 'i' here, the outer loop in scrape_with_timeout2 will retry
        print(f"[REDACTED_BY_SCRIPT]") # Debug print

async def main2():
    global driver
    global row
    global parsed_dataout
    global cookiecount
    """[REDACTED_BY_SCRIPT]"""
    driver = None # Ensure driver starts as None
    try:
        print("[REDACTED_BY_SCRIPT]")
        driver = initialize_driver() # Initial setup
        print("[REDACTED_BY_SCRIPT]")
        await scrape_with_timeout2(120) # Run the main loop
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
            quit_attempted = False
            quit_attempted_0=False
            while quit_attempted == False:
                try:print(driver)
                except:pass
                print(quit_attempted)
                try:
                    driver.quit()
                    time.sleep(0.1)
                    if quit_attempted_0==True:
                        quit_attempted=True
                    else:
                        quit_attempted_0=True
                except WebDriverException as e:
                    print(f"[REDACTED_BY_SCRIPT]")
                    quit_attempted = True
                except:
                    print("[REDACTED_BY_SCRIPT]")
                    if driver == None:
                        quit_attempted = True
            # Always attempt final process termination
            print("[REDACTED_BY_SCRIPT]")
            found_pids = find_chromedriver_processes()
            terminate_processes(found_pids)
            print("[REDACTED_BY_SCRIPT]")




############################################################################################################################################
############################################################################################################################################
############################################################################################################################################
############################################################################################################################################


async def scrape_with_timeout3(timeout_seconds=300): 
    global driver
    global row
    global parsed_dataout
    global parsed_data
    global cookiecount
    global tryCount
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
        check_driver_working(addressurl)

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
                BnLaddresses0_element = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]")
                BnLaddresses0 = get_element_text_robustly(driver, BnLaddresses0_element, prefer_text_content=False)
                BnLaddresses.append(BnLaddresses0)                  #body > div > section:nth-child(3) > div.container.mx-auto.px-0.md\:px-\[2\.1rem\] > div > div.property-helpers > div:nth-child(3) > div > div.CardBody > div > ul > li:nth-child({BnLcount}) > span > span.flex-1.flex-grow.flex.items-center.px-\[1\.1rem\].max-w-full.overflow-hidden.bg-white.md\:px-\[1rem\].rounded-tr-md.rounded-br-md.border.border-l-0 > span.font-bold.block.mr-auto.max-w-full.overflow-hidden.whitespace-nowrap > a
                BnLaddresses_href0=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").get_attribute("href")
                BnLaddresses_href.append(BnLaddresses_href0)
                BnLcount+=1
            except:
                try:
                    BnLaddresses0_element = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]")
                    BnLaddresses0 = get_element_text_robustly(driver, BnLaddresses0_element, prefer_text_content=False)
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
            
        check_driver_working(addressurl)
        time.sleep(random.uniform(1, 1.5))

        try:
            BnLaddresses=[]
            BnLaddresses_href=[]
            BnLcount=1
            BnlFound=False
            #time.sleep(1000)
            while BnlFound==False:
                try:
                    BnLaddresses0_element = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]")
                    BnLaddresses0 = get_element_text_robustly(driver, BnLaddresses0_element, prefer_text_content=False)
                    BnLaddresses.append(BnLaddresses0)                  #body > div > section:nth-child(7) > div.container.mx-auto.px-0.md\:px-\[2\.1rem\] > div > div.property-helpers > div:nth-child(3) > div > div.CardBody > div > ul > li:nth-child(2) > span > span.flex-1.flex-grow.flex.items-center.px-\[1\.1rem\].max-w-full.overflow-hidden.bg-white.md\:px-\[1rem\].rounded-tr-md.rounded-br-md.border.border-l-0 > span > a
                    BnLaddresses_href0=driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]").get_attribute("href")
                    BnLaddresses_href.append(BnLaddresses_href0)
                    BnLcount+=1
                except:
                    try:
                        BnLaddresses0_element = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]")
                        BnLaddresses0 = get_element_text_robustly(driver, BnLaddresses0_element, prefer_text_content=False)
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
                
            check_driver_working(addressurl)
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
            floorarea_element = driver.find_element(By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]")
            floorarea = get_element_text_robustly(driver, floorarea_element, prefer_text_content=False)
            floorarea=floorarea.replace(" ft2","")           #body > div.bg-white > section.bg-white.py-0.relative.md\:bg-gradient-to-b.from-gray-400.via-white.to-white.md\:py-\[4rem\] > div.relative.z-1 > div.container.mx-auto.px-0.md\:px-\[2\.1rem\] > div.w-full.md\:max-w-\[33\.2rem\] > div > div.CardBody > div > p:nth-child(1) > span:nth-child(1)
            #print(floorarea)
        except:
            try:
                floorarea_element = driver.find_element(By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]")
                floorarea = get_element_text_robustly(driver, floorarea_element, prefer_text_content=False)
                floorarea=floorarea.replace(" ft2","")
            except:pass
        try:
            estimPrice_element = driver.find_element(By.XPATH, "[REDACTED_BY_SCRIPT]")
            estimPrice = get_element_text_robustly(driver, estimPrice_element, prefer_text_content=False)
            estimPrice=estimPrice.replace("£","").replace(",","") 
            rent_element = driver.find_element(By.XPATH, "[REDACTED_BY_SCRIPT]")
            rent = get_element_text_robustly(driver, rent_element, prefer_text_content=False)
        except:
            try:
                estimPrice_element = driver.find_element(By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]")
                estimPrice = get_element_text_robustly(driver, estimPrice_element, prefer_text_content=False)
                estimPrice=estimPrice.replace("£","").replace(",","") 
                rent_element = driver.find_element(By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]")
                rent = get_element_text_robustly(driver, rent_element, prefer_text_content=False)
                #
            except:pass
        try:
            mortgage_element = driver.find_element(By.XPATH, "[REDACTED_BY_SCRIPT]")
            mortgage = get_element_text_robustly(driver, mortgage_element, prefer_text_content=False)
            #print(mortgage)
        except Exception as e:
            pass
        try:
            plotsize_element = driver.find_element(By.XPATH, "[REDACTED_BY_SCRIPT]")
            plotsize = get_element_text_robustly(driver, plotsize_element, prefer_text_content=False)
            plotsdata1_element = driver.find_element(By.XPATH, "[REDACTED_BY_SCRIPT]")
            plotsdata1 = get_element_text_robustly(driver, plotsdata1_element, prefer_text_content=False)
            plotsdata2_element = driver.find_element(By.XPATH, "[REDACTED_BY_SCRIPT]")
            plotsdata2 = get_element_text_robustly(driver, plotsdata2_element, prefer_text_content=False)
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
                schooldist1_element = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]")
                schooldist1 = get_element_text_robustly(driver, schooldist1_element, prefer_text_content=False)
                schooldist2_element = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]")
                schooldist2 = get_element_text_robustly(driver, schooldist2_element, prefer_text_content=False)
                schooldist3_element = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]")
                schooldist3 = get_element_text_robustly(driver, schooldist3_element, prefer_text_content=False)
                schoolrating1_element = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]")
                schoolrating1 = get_element_text_robustly(driver, schoolrating1_element, prefer_text_content=False)
                schoolrating2_element = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]")
                schoolrating2 = get_element_text_robustly(driver, schoolrating2_element, prefer_text_content=False)
                schoolrating3_element = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]")
                schoolrating3 = get_element_text_robustly(driver, schoolrating3_element, prefer_text_content=False)
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
                    schooldist1_element = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]")
                    schooldist1 = get_element_text_robustly(driver, schooldist1_element, prefer_text_content=False)
                    schooldist2_element = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]")
                    schooldist2 = get_element_text_robustly(driver, schooldist2_element, prefer_text_content=False)
                    schooldist3_element = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]")
                    schooldist3 = get_element_text_robustly(driver, schooldist3_element, prefer_text_content=False)
                    schoolrating1_element = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]")
                    schoolrating1 = get_element_text_robustly(driver, schoolrating1_element, prefer_text_content=False)
                    schoolrating2_element = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]")
                    schoolrating2 = get_element_text_robustly(driver, schoolrating2_element, prefer_text_content=False)
                    schoolrating3_element = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]")
                    schoolrating3 = get_element_text_robustly(driver, schoolrating3_element, prefer_text_content=False)
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
                schooldist21_element = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]")
                schooldist21 = get_element_text_robustly(driver, schooldist21_element, prefer_text_content=False)
                schooldist22_element = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]")
                schooldist22 = get_element_text_robustly(driver, schooldist22_element, prefer_text_content=False)
                schooldist23_element = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]")
                schooldist23 = get_element_text_robustly(driver, schooldist23_element, prefer_text_content=False)
                schoolrating21_element = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]")
                schoolrating21 = get_element_text_robustly(driver, schoolrating21_element, prefer_text_content=False)
                schoolrating22_element = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]")
                schoolrating22 = get_element_text_robustly(driver, schoolrating22_element, prefer_text_content=False)
                schoolrating23_element = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]")
                schoolrating23 = get_element_text_robustly(driver, schoolrating23_element, prefer_text_content=False)
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
                    schooldist21_element = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]")
                    schooldist21 = get_element_text_robustly(driver, schooldist21_element, prefer_text_content=False)
                    schooldist22_element = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]")
                    schooldist22 = get_element_text_robustly(driver, schooldist22_element, prefer_text_content=False)
                    schooldist23_element = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]")
                    schooldist23 = get_element_text_robustly(driver, schooldist23_element, prefer_text_content=False)
                    schoolrating21_element = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]")
                    schoolrating21 = get_element_text_robustly(driver, schoolrating21_element, prefer_text_content=False)
                    schoolrating22_element = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]")
                    schoolrating22 = get_element_text_robustly(driver, schoolrating22_element, prefer_text_content=False)
                    schoolrating23_element = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]")
                    schoolrating23 = get_element_text_robustly(driver, schoolrating23_element, prefer_text_content=False)
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
                traindistance1_element = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]")
                traindistance1 = get_element_text_robustly(driver, traindistance1_element, prefer_text_content=False)
                traindistance2_element = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]")
                traindistance2 = get_element_text_robustly(driver, traindistance2_element, prefer_text_content=False)
                traindistance3_element = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]")
                traindistance3 = get_element_text_robustly(driver, traindistance3_element, prefer_text_content=False)
                brdbndStore=k
                break
            except:
                try:
                    traindistance1_element = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]")
                    traindistance1 = get_element_text_robustly(driver, traindistance1_element, prefer_text_content=False)
                    traindistance2_element = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]")
                    traindistance2 = get_element_text_robustly(driver, traindistance2_element, prefer_text_content=False)
                    traindistance3_element = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]")
                    traindistance3 = get_element_text_robustly(driver, traindistance3_element, prefer_text_content=False)
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
                    houseinfo_element = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]")
                    houseinfo = driver.execute_script("[REDACTED_BY_SCRIPT]", houseinfo_element) # Using JS as .text might not work well here
                    if k<4:
                        eduinfo_element = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]")
                        eduinfo = driver.execute_script("[REDACTED_BY_SCRIPT]", eduinfo_element) # Using JS as .text might not work well here
                except:
                    try:
                        houseinfo_element = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]")
                        houseinfo = driver.execute_script("[REDACTED_BY_SCRIPT]", houseinfo_element) # Using JS as .text might not work well here
                        if k<4:
                            eduinfo_element = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]")
                            eduinfo = driver.execute_script("[REDACTED_BY_SCRIPT]", eduinfo_element) # Using JS as .text might not work well here
                    except:
                        try:
                            houseinfo_element = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]")
                            houseinfo = driver.execute_script("[REDACTED_BY_SCRIPT]", houseinfo_element) # Using JS as .text might not work well here
                            if k<4:
                                eduinfo_element = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]")
                                eduinfo = driver.execute_script("[REDACTED_BY_SCRIPT]", eduinfo_element) # Using JS as .text might not work well here
                        except Exception as e:
                            pass
               
                houseList.append(houseinfo)
                educationList.append(eduinfo)
        #print(houseList,educationList)
        avgAge,modeAge="",""
        for k in range(4):
            try:
                avgAge_element = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]")
                avgAge = get_element_text_robustly(driver, avgAge_element, prefer_text_content=False)
                modeAge_element = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]")
                modeAge = get_element_text_robustly(driver, modeAge_element, prefer_text_content=False)
                break
            except:
                try:
                    avgAge_element = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]")
                    avgAge = get_element_text_robustly(driver, avgAge_element, prefer_text_content=False)
                    modeAge_element = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]")
                    modeAge = get_element_text_robustly(driver, modeAge_element, prefer_text_content=False)
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
                secondryHouse_element = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]")
                secondryHouse = get_element_text_robustly(driver, secondryHouse_element, prefer_text_content=False)
                carinfo_element = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]")
                carinfo = get_element_text_robustly(driver, carinfo_element, prefer_text_content=False)
                break
                #/html/body/div[1]/section[18]/div[3]/div/div/div/div[3]/div[2]/div/div[1]/div/div[1]
                #/html/body/div[1]/section[19]/div[3]/div/div/div/div[3]/div[2]/div/div[1]/div/div[1]
            except:
                try:
                    secondryHouse_element = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]")
                    secondryHouse = get_element_text_robustly(driver, secondryHouse_element, prefer_text_content=False)
                    carinfo_element = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]")
                    carinfo = get_element_text_robustly(driver, carinfo_element, prefer_text_content=False)
                    break
                except Exception as e:
                    pass
        try:
            secondryHouse=secondryHouse[:secondryHouse.index("%")]
        except:pass
        quality=""
        try:
            quality_element = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]")
            quality = get_element_text_robustly(driver, quality_element, prefer_text_content=False)
            if "High" in quality:quality=quality.replace("High","3")
            elif "Medium" in quality:quality=quality.replace("Medium","2")
            else:quality="1"
        except Exception as e:
            pass
        quality2=""
        try:
            quality2_element = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]")
            quality2 = get_element_text_robustly(driver, quality2_element, prefer_text_content=False)
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
    try:
        if driver:
            driver.quit()  # Ensure the driver is closed after scraping
            driver=None
            found_pids = find_chromedriver_processes()
            terminate_processes(found_pids)
    except:
        driver=None
        found_pids = find_chromedriver_processes()
        terminate_processes(found_pids)







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
        check_driver_working(addressurl)
        
        timeout=30
        if cookiecount==0:
            shadow_host = driver.find_element(By.CSS_SELECTOR, "#cmpwrapper")
            shadow_root = driver.execute_script("[REDACTED_BY_SCRIPT]", shadow_host)

            wait = WebDriverWait(shadow_root, timeout)
            element_locator = (By.CSS_SELECTOR, '[REDACTED_BY_SCRIPT]')
            element = wait.until(EC.presence_of_element_located(element_locator))
            shadow_root.find_element(By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]").click()
            cookiecount=1
        number_of_props_element = driver.find_element(By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]")
        number_of_props = get_element_text_robustly(driver, number_of_props_element, prefer_text_content=False)
        number_of_props = number_of_props.split(" ")
        number_of_props = number_of_props[0]
        compare_list_adrresses=[]
        compare_list_hrefs=[]
        for jk in range(int(number_of_props)):
            try:
                address_element = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]")
                address_text = get_element_text_robustly(driver, address_element, prefer_text_content=False)
                compare_list_adrresses.append(address_text)
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
        
        check_driver_working(addressurl)
        wait = WebDriverWait(driver, timeout)
        element_locator = (By.CSS_SELECTOR, '[REDACTED_BY_SCRIPT]')
        element = wait.until(EC.presence_of_element_located(element_locator))
        try:
            estimPrice0_element = driver.find_element(By.XPATH, "[REDACTED_BY_SCRIPT]")
            estimPrice0 = get_element_text_robustly(driver, estimPrice0_element, prefer_text_content=False)
            estimPrice0= estimPrice0.replace("£","").replace(",","")
        except:
            try:
                estimPrice0_element = driver.find_element(By.XPATH, "[REDACTED_BY_SCRIPT]")
                estimPrice0 = get_element_text_robustly(driver, estimPrice0_element, prefer_text_content=False)
                estimPrice0= estimPrice0.replace("£","").replace(",","")
            except:
                estimPrice0 = ""

        try:#                                            /html/body/div[5]/div[2]/div[5]/div[2]/div[1]/span[1]
            confidence0_element = driver.find_element(By.XPATH, "[REDACTED_BY_SCRIPT]")
            confidence0 = get_element_text_robustly(driver, confidence0_element, prefer_text_content=False)
            confidence0=confidence0.replace("We have ","")
            confidence0=confidence0[:(confidence0.index(" "))]
        except:
            try:
                confidence0_element = driver.find_element(By.XPATH, "[REDACTED_BY_SCRIPT]")
                confidence0 = get_element_text_robustly(driver, confidence0_element, prefer_text_content=False)
                confidence0=confidence0.replace("We have ","")
                confidence0=confidence0[:(confidence0.index(" "))]
            except:
                confidence0 = 0

        try:
            rent0_element = driver.find_element(By.XPATH, "[REDACTED_BY_SCRIPT]")
            rent0 = get_element_text_robustly(driver, rent0_element, prefer_text_content=False)
            rent0 = rent0.replace("Or ", "")
            rent0 = rent0[:rent0.index(" ")]
            try:
                rent0_element = driver.find_element(By.XPATH, "[REDACTED_BY_SCRIPT]")
                rent0 = get_element_text_robustly(driver, rent0_element, prefer_text_content=False)
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
                data000_element = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]")
                data000 = get_element_text_robustly(driver, data000_element, prefer_text_content=False)
                data001_element = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]")
                data001 = get_element_text_robustly(driver, data001_element, prefer_text_content=False)
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
                    data000_element = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]")
                    data000 = get_element_text_robustly(driver, data000_element, prefer_text_content=False)
                    data001_element = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]")
                    data001 = get_element_text_robustly(driver, data001_element, prefer_text_content=False)
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
                data000_element = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]")
                data000 = get_element_text_robustly(driver, data000_element, prefer_text_content=False)
                data001_element = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]")
                data001 = get_element_text_robustly(driver, data001_element, prefer_text_content=False)
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
                    data000_element = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]")
                    data000 = get_element_text_robustly(driver, data000_element, prefer_text_content=False)
                    data001_element = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]")
                    data001 = get_element_text_robustly(driver, data001_element, prefer_text_content=False)
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
                data000_element = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]")
                data000 = get_element_text_robustly(driver, data000_element, prefer_text_content=False)
                data001_element = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]")
                data001 = get_element_text_robustly(driver, data001_element, prefer_text_content=False)
                data20.append(data000)
                data21.append(data001)
            except:
                try:
                    data000_element = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]")
                    data000 = get_element_text_robustly(driver, data000_element, prefer_text_content=False)
                    data001_element = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]")
                    data001 = get_element_text_robustly(driver, data001_element, prefer_text_content=False)
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
    try:
        if driver:
            driver.quit()  # Ensure the driver is closed after scraping
            driver=None
            found_pids = find_chromedriver_processes()
            terminate_processes(found_pids)
    except:
        driver=None
        found_pids = find_chromedriver_processes()
        terminate_processes(found_pids)








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
            driver = uc.Chrome(options=options, use_subprocess=True)
            
        for jk in range(5):
            try:
                addressin=parsed_data[i][1][0]
                get_website="www.chimnie.co.uk/"
                binginput="[REDACTED_BY_SCRIPT]"+addressin
                check_driver_working('[REDACTED_BY_SCRIPT]')
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
                        domain_element = driver.find_element(By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]"+str(k+1)+"[REDACTED_BY_SCRIPT]")
                        domain = get_element_text_robustly(driver, domain_element, prefer_text_content=False)
                        metadata_element = driver.find_element(By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]"+str(k+1)+") > h2")
                        metadata = get_element_text_robustly(driver, metadata_element, prefer_text_content=False)
                        href=driver.find_element(By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]"+str(k+1)+") > h2 > a").get_attribute("href")
                    except:
                        try:
                            domain_element = driver.find_element(By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]"+str(k+1)+"[REDACTED_BY_SCRIPT]")
                            domain = get_element_text_robustly(driver, domain_element, prefer_text_content=False)
                            metadata_element = driver.find_element(By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]"+str(k+1)+") > h2")
                            metadata = get_element_text_robustly(driver, metadata_element, prefer_text_content=False)
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
        check_driver_working(addressurl)
        timeout=30
        chimnieaddressInfo1=["","","","","","","","","",""]
        chimnieaddressInfo2=["","","","","","","","","",""]
        chimnieaddressInfo3=["","","","","","","","","",""]
        element_present = EC.presence_of_element_located((By.XPATH, "/html/body/div[1]/div[1]/div[4]/div/div[1]/div[1]/div/div/div[1]/h3"))
        WebDriverWait(driver, timeout).until(element_present)
        for i in range(3):
            try:
                lscore1_element = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]")
                lscore1 = get_element_text_robustly(driver, lscore1_element, prefer_text_content=False)
                if i==0:chimnieaddressInfo1[0] = lscore1
                elif i==1:chimnieaddressInfo2[0] = lscore1
                elif i==2:chimnieaddressInfo3[0] = lscore1
            except:pass
            try:
                escore1_element = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]")
                escore1 = get_element_text_robustly(driver, escore1_element, prefer_text_content=False)
                if i==0:chimnieaddressInfo1[1] = escore1
                elif i==1:chimnieaddressInfo2[1] = escore1
                elif i==2:chimnieaddressInfo3[1] = escore1
            except:pass
            try:# 
                sscore1_element = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]")   
                sscore1 = get_element_text_robustly(driver, sscore1_element, prefer_text_content=False)
                if i==0:chimnieaddressInfo1[2] = sscore1
                elif i==1:chimnieaddressInfo2[2] = sscore1
                elif i==2:chimnieaddressInfo3[2] = sscore1
            except:pass
            try:
                fscore1_element = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]")
                fscore1 = get_element_text_robustly(driver, fscore1_element, prefer_text_content=False)
                if i==0:chimnieaddressInfo1[3] = fscore1
                elif i==1:chimnieaddressInfo2[3] = fscore1
                elif i==2:chimnieaddressInfo3[3] = fscore1
            except:pass
            try:
                cscore1_element = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]")
                cscore1 = get_element_text_robustly(driver, cscore1_element, prefer_text_content=False)
                if i==0:chimnieaddressInfo1[4] = cscore1
                elif i==1:chimnieaddressInfo2[4] = cscore1
                elif i==2:chimnieaddressInfo3[4] = cscore1
            except:pass
            try:
                l2score1_element = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]")
                l2score1 = get_element_text_robustly(driver, l2score1_element, prefer_text_content=False)
                if i==0:chimnieaddressInfo1[5] = l2score1
                elif i==1:chimnieaddressInfo2[5] = l2score1
                elif i==2:chimnieaddressInfo3[5] = l2score1
            except:pass
            try:
                e2score1_element = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]")
                e2score1 = get_element_text_robustly(driver, e2score1_element, prefer_text_content=False)
                if i==0:chimnieaddressInfo1[6] = e2score1
                elif i==1:chimnieaddressInfo2[6] = e2score1
                elif i==2:chimnieaddressInfo3[6] = e2score1
            except:pass
            try:
                s2score1_element = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]")
                s2score1 = get_element_text_robustly(driver, s2score1_element, prefer_text_content=False)
                if i==0:chimnieaddressInfo1[7] = s2score1
                elif i==1:chimnieaddressInfo2[7] = s2score1
                elif i==2:chimnieaddressInfo3[7] = s2score1
            except:pass
            try:
                f2score1_element = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]")
                f2score1 = get_element_text_robustly(driver, f2score1_element, prefer_text_content=False)
                if i==0:chimnieaddressInfo1[8] = f2score1
                elif i==1:chimnieaddressInfo2[8] = f2score1
                elif i==2:chimnieaddressInfo3[8] = f2score1
            except:pass
            try:
                c2score1_element = driver.find_element(By.XPATH, f"[REDACTED_BY_SCRIPT]")
                c2score1 = get_element_text_robustly(driver, c2score1_element, prefer_text_content=False)
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
    try:
        if driver:
            driver.quit()  # Ensure the driver is closed after scraping
            driver=None
            found_pids = find_chromedriver_processes()
            terminate_processes(found_pids)
    except:
        driver=None
        found_pids = find_chromedriver_processes()
        terminate_processes(found_pids)


def initialize_driver5():
    global driver
    global row
    global parsed_dataout
    global cookiecount
    options = uc.ChromeOptions()
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument("[REDACTED_BY_SCRIPT]")
    options.add_argument("[REDACTED_BY_SCRIPT]")
    options.add_argument("--headless")
    time.sleep(1)
    driver = uc.Chrome(options=options, use_subprocess=True)
    
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
        for jk in range(10841):
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
                    if driver:
                        driver.quit()  # Quit the current driver
                        driver=None
                        found_pids = find_chromedriver_processes()
                        terminate_processes(found_pids)
                except:
                    driver=None
                    found_pids = find_chromedriver_processes()
                    terminate_processes(found_pids)
            except:
                try:
                    if driver:
                        driver.quit()
                        found_pids = find_chromedriver_processes()
                        terminate_processes(found_pids)
                except:
                    driver=None
                    found_pids = find_chromedriver_processes()
                    terminate_processes(found_pids)
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
        check_driver_working(addressurl)
        
        working=False
        try:
            try:
                element_present = EC.presence_of_element_located((By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]"))
                WebDriverWait(driver, 3).until(element_present)
            except:pass
            check_fail_element = driver.find_element(By.CSS_SELECTOR, "body > div > h1")
            check_fail = get_element_text_robustly(driver, check_fail_element, prefer_text_content=False)
            print(check_fail)
            if check_fail and "oops" in check_fail.lower(): # Added check_fail existence check
                print("Streetcan failed")
                working=False
            else: # Assume working if not oops or if check_fail is None/empty
                working=True
        except:
            try: # Added nested try-except for robustness
                element_present = EC.presence_of_element_located((By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]"))
                WebDriverWait(driver, 10).until(element_present)
                working=True
            except Exception as e_wait:
                 print(f"[REDACTED_BY_SCRIPT]")
                 working=False # Failed to find banner even after waiting

        if working==True:
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
                            # print(f"[REDACTED_BY_SCRIPT]") # Reduced verbosity
                            pass
                    ratings_list.append(star_count)
            average_price_strt = "" # Initialize
            try:
                average_price_strt_element = driver.find_element(By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]")
                average_price_strt = get_element_text_robustly(driver, average_price_strt_element, prefer_text_content=False)
            except:pass
            

            income_list=[]
            income_num="" # Initialize
            income_rank="" # Initialize
            try:
                income_num_element = driver.find_element(By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]")
                income_num_text = get_element_text_robustly(driver, income_num_element, prefer_text_content=False)
                if income_num_text and "\n" in income_num_text: # More robust splitting
                    income_num=income_num_text[income_num_text.index("\n")+1:].strip() # Get text after newline
                    income_num=income_num.replace(",","").replace("£","")
                elif income_num_text: # Handle case without newline
                     income_num = income_num_text.strip().replace(",","").replace("£","")

                income_rank_element = driver.find_element(By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]")
                income_rank_text = get_element_text_robustly(driver, income_rank_element, prefer_text_content=False)
                if income_rank_text and "/" in income_rank_text:
                    income_rank=income_rank_text[:income_rank_text.index("/")]
                elif income_rank_text: # Handle case without '/'
                     income_rank = income_rank_text.strip()
            except Exception as e_income: # Catch specific error
                print(f"[REDACTED_BY_SCRIPT]")
                income_num=""
                income_rank=""
            income_list.append(income_num)
            income_list.append(income_rank)

            deprivation_list=[]
            for k in range(10):
                dep_item="" # Initialize
                try:
                    dep_item_element = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]")
                    dep_item_text = get_element_text_robustly(driver, dep_item_element, prefer_text_content=False)
                    if dep_item_text and "/" in dep_item_text:
                        dep_item=dep_item_text[:dep_item_text.index("/")]
                        dep_item=dep_item.replace(" ","").replace("-","")
                    elif dep_item_text: # Handle case without '/'
                         dep_item = dep_item_text.strip().replace(" ","").replace("-","")
                except Exception as e_dep: # Catch specific error
                    # print(f"[REDACTED_BY_SCRIPT]") # Reduced verbosity
                    dep_item=""
                deprivation_list.append(dep_item)
            
            past_sales_list=[]
            for k in range(6):
                past_sales_item="" # Initialize
                try:
                    past_sales_item_element = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]")
                    past_sales_item_text = get_element_text_robustly(driver, past_sales_item_element, prefer_text_content=False)
                    if past_sales_item_text:
                        past_sales_item=past_sales_item_text.replace("£","").replace(",","")
                except Exception as e_sales: # Catch specific error
                    # print(f"[REDACTED_BY_SCRIPT]") # Reduced verbosity
                    past_sales_item=""
                past_sales_list.append(past_sales_item)
            
            emp_col = None # Initialize emp_col
            # Find the column containing "Retired" first
            for k_outer in range(1, 11): # Check potential parent divs 1 to 10
                try:
                    # Check if this div contains the employment table structure
                    header_check = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]")
                    header_text = get_element_text_robustly(driver, header_check, prefer_text_content=False)
                    # A simple check, might need refinement based on actual table headers
                    if header_text and ("Type" in header_text or "Status" in header_text):
                        # Now check rows within this potential column
                        for jk_inner in range(1, 11): # Check rows 1 to 10
                            try:
                                cell_element = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]")
                                cell_text = get_element_text_robustly(driver, cell_element, prefer_text_content=False)
                                if cell_text and "Retired" in cell_text:
                                    emp_col = k_outer
                                    print(f"[REDACTED_BY_SCRIPT]")
                                    break # Found the column
                            except:
                                continue # Try next row
                        if emp_col:
                            break # Exit outer loop once column is found
                except:
                    continue # Try next potential parent div

            emp_type_list=["Retired","Full-Time Employee","Part-Time Employee","Self-Employed","[REDACTED_BY_SCRIPT]","[REDACTED_BY_SCRIPT]","[REDACTED_BY_SCRIPT]","Other","[REDACTED_BY_SCRIPT]","Unemployed"]
            employment_list=[""] * len(emp_type_list) # Initialize based on list length

            if emp_col: # Only proceed if the column was found
                for k_row in range(1, 11): # Iterate through rows 1 to 10
                    try:
                        type_element = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]")
                        type_text = get_element_text_robustly(driver, type_element, prefer_text_content=False)
                        value_element = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]")
                        value_text = get_element_text_robustly(driver, value_element, prefer_text_content=False)

                        if type_text and value_text:
                            # Find which type it matches in our list
                            for jk, emp_type in enumerate(emp_type_list):
                                if emp_type in type_text: # Use 'in' for partial match
                                    employment_list[jk] = value_text
                                    break # Assume one match per row
                    except Exception as e_emp_type:
                        # print(f"[REDACTED_BY_SCRIPT]") # Reduced verbosity
                        pass # Continue to next row
            else:
                print("[REDACTED_BY_SCRIPT]")


            # Find the industry column similarly
            emp_ind_col = None
            for k_outer in range(1, 11):
                try:
                    header_check = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]")
                    header_text = get_element_text_robustly(driver, header_check, prefer_text_content=False)
                    if header_text and "Industry" in header_text: # Check for industry header
                         # Now check rows within this potential column for a known industry
                        for jk_inner in range(1, 16): # Check rows 1 to 15 (or more if needed)
                            try:
                                cell_element = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]")
                                cell_text = get_element_text_robustly(driver, cell_element, prefer_text_content=False)
                                if cell_text and "Education" in cell_text: # Check for a known entry like Education
                                    emp_ind_col = k_outer
                                    print(f"[REDACTED_BY_SCRIPT]")
                                    break
                            except:
                                continue
                        if emp_ind_col:
                            break
                except:
                    continue

            emp_ind_type_list=["Education","[REDACTED_BY_SCRIPT]","Construction","[REDACTED_BY_SCRIPT]","Other","[REDACTED_BY_SCRIPT]","[REDACTED_BY_SCRIPT]","Manufacturing","Information and communication","Financial and insurance activities","[REDACTED_BY_SCRIPT]","[REDACTED_BY_SCRIPT]","Real estate activities","Transport and storage","[REDACTED_BY_SCRIPT]"]
            emp_ind_list=[""] * len(emp_ind_type_list)

            if emp_ind_col: # Only proceed if the column was found
                for k_row in range(1, 16): # Iterate through rows 1 to 15
                    try:
                        type_element = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]")
                        type_text = get_element_text_robustly(driver, type_element, prefer_text_content=False)
                        value_element = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]")
                        value_text = get_element_text_robustly(driver, value_element, prefer_text_content=False)

                        if type_text and value_text:
                            for jk, emp_ind_type in enumerate(emp_ind_type_list):
                                # Use 'in' for robustness, might need exact match depending on data
                                if emp_ind_type in type_text:
                                    emp_ind_list[jk] = value_text
                                    break
                    except Exception as e_emp_ind:
                        # print(f"[REDACTED_BY_SCRIPT]") # Reduced verbosity
                        pass
            else:
                print("[REDACTED_BY_SCRIPT]")


            streetscanaddressInfo=[addressin,ratings_list,income_list,deprivation_list,past_sales_list,employment_list,emp_ind_list]
            with open(r"[REDACTED_BY_SCRIPT]", 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([streetscanaddressInfo])       
            tryCount=0
            i+=1
        else:
            print(f"[REDACTED_BY_SCRIPT]") # Added message
            tryCount=0
            i+=1
    except Exception as e:
        print(f"[REDACTED_BY_SCRIPT]") # Clarified function name
        print(traceback.format_exc()) # Print traceback for better debugging
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
    try:
        if driver:
            driver.quit()  # Ensure the driver is closed after scraping
            driver=None
            found_pids = find_chromedriver_processes()
            terminate_processes(found_pids)
    except:
        driver=None
        found_pids = find_chromedriver_processes()
        terminate_processes(found_pids)




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
                await asyncio.wait_for(asyncio.to_thread(scrape_page7,parsed_data), timeout=timeout_seconds)#print(traceback.format_exc())
                completedScrape = True
            except asyncio.TimeoutError:
                print("[REDACTED_BY_SCRIPT]")
                try:
                    if driver:
                        driver.quit()  # Quit the current driver
                        driver=None
                        found_pids = find_chromedriver_processes()
                        terminate_processes(found_pids)
                except:
                    driver=None
                    found_pids = find_chromedriver_processes()
                    terminate_processes(found_pids)
            except:
                try:
                    if driver:
                        driver.quit()  # Quit the current driver
                        driver=None
                        found_pids = find_chromedriver_processes()
                        terminate_processes(found_pids)
                except:
                    driver=None
                    found_pids = find_chromedriver_processes()
                    terminate_processes(found_pids)
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
    if not driver:
        driver = initialize_driver()
        driverReset=1
    try:

        ##########################################################################################################
        #streetcan
        addressin=parsed_data[i][1][0]
        addressin2=addressin.split(" ")[-2]
        addressurl=f"[REDACTED_BY_SCRIPT]"
        check_driver_working(addressurl)
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
            check_driver_working(subdomain_name_list[jkl])
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
        check_driver_working(addressurl)
        element_present = EC.presence_of_element_located((By.CSS_SELECTOR, "[REDACTED_BY_SCRIPT]"))
        WebDriverWait(driver, 15).until(element_present)

        detached_beds5, detached_beds4, detached_beds3, detached_beds2, detached_beds1, detached_sqft, detached_val = [], [], [], [], [], [], []
        semi_detached_beds5, semi_detached_beds4, semi_detached_beds3, semi_detached_beds2, semi_detached_beds1, semi_detached_sqft, semi_detached_val = [], [], [], [], [], [], []
        terraced_beds5, terraced_beds4, terraced_beds3, terraced_beds2, terraced_beds1, terraced_sqft, terraced_val = [], [], [], [], [], [], []
        flat_beds5, flat_beds4, flat_beds3, flat_beds2, flat_beds1, flat_sqft, flat_val = [], [], [], [], [], [], []
        for k in range(100):
            try:
                type_element = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]")
                house_type = get_element_text_robustly(driver, type_element, prefer_text_content=False)
                beds_element = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]")
                beds = get_element_text_robustly(driver, beds_element, prefer_text_content=False)
                sqft_element = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]")
                sqft = get_element_text_robustly(driver, sqft_element, prefer_text_content=False)
                val_element = driver.find_element(By.CSS_SELECTOR, f"[REDACTED_BY_SCRIPT]")
                val = get_element_text_robustly(driver, val_element, prefer_text_content=False)
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
    try:
        if driver:
            driver.quit()  # Ensure the driver is closed after scraping
            driver=None
            found_pids = find_chromedriver_processes()
            terminate_processes(found_pids)
    except:
        driver=None
        found_pids = find_chromedriver_processes()
        terminate_processes(found_pids)


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
        try:
            asyncio.run(main2())
        except KeyboardInterrupt:
            print("[REDACTED_BY_SCRIPT]")
        except Exception as e:
            print(f"[REDACTED_BY_SCRIPT]")
            print(f"[REDACTED_BY_SCRIPT]")
            print(f"Error Args: {e.args}")
            print(traceback.format_exc())
            print(f"--- END TOP LEVEL EXCEPTION ---")
        finally:
            print("[REDACTED_BY_SCRIPT]")
            # Optional: Add final check for stray processes again here, though main2 should do it.
            # print("[REDACTED_BY_SCRIPT]")
            # found_pids = find_chromedriver_processes(include_undetected=True)
            # terminate_processes(found_pids)
    elif start_whcih_website == "misc":
        asyncio.run(main6())