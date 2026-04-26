import nodriver
import asyncio
import csv
import math
import random
import time # Keep for non-async delays if absolutely necessary, but prefer asyncio.sleep
from datetime import datetime
import Levenshtein
import os, os.path
import ast
import traceback
import psutil # Still useful for browser.proc
import logging
from fake_useragent import UserAgent
import gc

# --- 1. Set up Logging ---
LOG_FILENAME = '[REDACTED_BY_SCRIPT]' # Changed log filename
LOG_FORMAT = '[REDACTED_BY_SCRIPT]'

# Configure the logger
logging.basicConfig(
    level=logging.WARNING,
    format=LOG_FORMAT,
    filename=LOG_FILENAME,
    filemode='a'
)

# Global variable to hold the browser instance (consider passing it instead if structure allows)
browser: nodriver.Browser = None
# Global variable to hold the main tab (consider passing it instead)
# main_tab: uc.Tab = None # Or initialize later

# --- Date Parsing Functions (Unchanged) ---
def parse_date(date_str):
    return datetime.strptime(date_str, "%B %Y")

def parse_date2(date_str):
    return datetime.strptime(date_str, "%b %Y")

def calculate_time_diff(sold_date, listed_date):
    delta = sold_date - listed_date
    months = delta.days // 30
    years, months = divmod(months, 12)
    if years > 0 and months > 0:
        return f"[REDACTED_BY_SCRIPT]"
    elif years > 0:
        return f"{years} years"
    else:
        return f"{months} months"

# --- Initialize Browser (Replaces initialize_driver) ---
async def initialize_browser():
    """[REDACTED_BY_SCRIPT]"""
    global browser # Use global browser instance
    # global main_tab # Use global tab instance

    # --- Cleanup FIRST ---
    print("[REDACTED_BY_SCRIPT]")
    if browser:
        print("[REDACTED_BY_SCRIPT]")
        proc_pid = None # Initialize pid variable
        try:
            # Attempt to get PID *before* trying to close, if possible and attributes exist
            if hasattr(browser, 'proc') and browser.proc and hasattr(browser.proc, 'pid'):
                 proc_pid = browser.proc.pid

            # Attempt to close gracefully
            if hasattr(browser, 'close') and callable(browser.close): # Check if close method exists and is callable
                 await browser.close()
                 print("[REDACTED_BY_SCRIPT]")
            else:
                 print("[REDACTED_BY_SCRIPT]'close'[REDACTED_BY_SCRIPT]")
                 # If no close method, maybe the browser object is invalid/incomplete

            # Give it a moment, then check if process still exists (if we got a PID)
            await asyncio.sleep(0.5)
            if proc_pid and psutil.pid_exists(proc_pid):
                 print(f"[REDACTED_BY_SCRIPT]")
                 try:
                     process_to_kill = psutil.Process(proc_pid)
                     process_to_kill.kill()
                     process_to_kill.wait(timeout=2)
                     print(f"[REDACTED_BY_SCRIPT]")
                 except (psutil.NoSuchProcess):
                     print(f"[REDACTED_BY_SCRIPT]")
                 except Exception as kill_e:
                     print(f"[REDACTED_BY_SCRIPT]")
                     logging.warning(f"[REDACTED_BY_SCRIPT]")
            elif proc_pid:
                 print(f"[REDACTED_BY_SCRIPT]")

        except AttributeError as ae:
            # Catch attribute errors specifically if browser object structure is unexpected
            print(f"[REDACTED_BY_SCRIPT]")
            logging.warning(f"[REDACTED_BY_SCRIPT]")
            # If we got a PID earlier, try killing it even if close failed
            if proc_pid and psutil.pid_exists(proc_pid):
                print(f"[REDACTED_BY_SCRIPT]")
                # ... (include kill logic here too, same as above) ...
                try:
                    process_to_kill = psutil.Process(proc_pid)
                    process_to_kill.kill(); process_to_kill.wait(timeout=2)
                    print(f"[REDACTED_BY_SCRIPT]")
                except Exception as kill_e: print(f"[REDACTED_BY_SCRIPT]")

        except Exception as e:
            # Catch other unexpected errors during cleanup
            print(f"[REDACTED_BY_SCRIPT]")
            logging.warning(f"[REDACTED_BY_SCRIPT]", exc_info=False)
            # Again, attempt kill if PID known
            if proc_pid and psutil.pid_exists(proc_pid):
                print(f"[REDACTED_BY_SCRIPT]")
                # ... (include kill logic here too, same as above) ...
                try:
                    process_to_kill = psutil.Process(proc_pid)
                    process_to_kill.kill(); process_to_kill.wait(timeout=2)
                    print(f"[REDACTED_BY_SCRIPT]")
                except Exception as kill_e: print(f"[REDACTED_BY_SCRIPT]")

        finally:
            browser = None # Ensure browser global is reset
            gc.collect()
            print("[REDACTED_BY_SCRIPT]")
            await asyncio.sleep(0.5) # Small delay
    else:
        print("[REDACTED_BY_SCRIPT]")

    # --- Attempt Initialization ---
    max_init_retries = 2
    for attempt in range(max_init_retries):
        print(f"[REDACTED_BY_SCRIPT]")
        try:
            ua = UserAgent(browsers=['chrome'])
            user_agent = ua.random

            # nodriver configuration
            browser_args = [
                '--disable-gpu',
                '--no-sandbox',
                '--disable-dev-shm-usage',
                # Try blocking images via blink settings
                '[REDACTED_BY_SCRIPT]',
                # Maybe proxy settings, window size etc. if needed
                # '[REDACTED_BY_SCRIPT]'
            ]

            browser = await nodriver.start(
                user_agent=user_agent,
                browser_args=browser_args,
                # You might need to specify browser_executable_path if not found
                # browser_executable_path=r"[REDACTED_BY_SCRIPT]",
                headless=False # Set to True if you want headless
            )

            print("[REDACTED_BY_SCRIPT]")
            # Get the first tab (or create one if needed)
            # main_tab = await browser.get("about:blank") # Get a specific tab
            # if not main_tab:
            #     main_tab = await browser.first_tab() # Or just get the first available tab
            # print(f"[REDACTED_BY_SCRIPT]")

            # Optional: Add a simple check immediately after init
            # await main_tab.get("chrome://version/")
            # print("[REDACTED_BY_SCRIPT]")

            initial_tab = await browser.get('about:blank') # Corrected call
            # Simply confirm you got a tab object, no need to access .id here
            if initial_tab:
                print(f"[REDACTED_BY_SCRIPT]") # <--- CORRECTED: Removed .id access
            else:
                # This shouldn't happen if browser.get succeeds, but good practice
                raise Exception("browser.get('about:blank'[REDACTED_BY_SCRIPT]")
            return browser
        except asyncio.TimeoutError:
            print(f"[REDACTED_BY_SCRIPT]")
            logging.error(f"[REDACTED_BY_SCRIPT]")
        except Exception as e:
            print(f"[REDACTED_BY_SCRIPT]")
            # print(traceback.format_exc()) # Keep traceback during debugging
            logging.error(f"[REDACTED_BY_SCRIPT]", exc_info=True)
            if browser: # Check if browser variable exists AT ALL before trying to close
                print("[REDACTED_BY_SCRIPT]")
                try:
                    # proc_pid = browser.proc.pid if hasattr(browser, 'proc') and browser.proc else None # Get PID before close if possible
                    await browser.close() # Use await
                    # ... kill process logic ...
                except AttributeError: # Specifically catch if close doesn't exist because 'browser' is wrong type
                     print("[REDACTED_BY_SCRIPT]")
                except Exception as kill_err:
                    print(f"[REDACTED_BY_SCRIPT]")
            browser = None # Reset browser variable for retry

        # --- Cleanup after failed attempt ---
        if browser: # If browser object exists but failed (e.g., timeout getting tab)
            print("[REDACTED_BY_SCRIPT]")
            try:
                proc_pid = browser.proc.pid if hasattr(browser, 'proc') and browser.proc else None
                await browser.close()
                if proc_pid and psutil.pid_exists(proc_pid):
                    process_to_kill = psutil.Process(proc_pid)
                    process_to_kill.kill()
                    process_to_kill.wait(1)
            except Exception as kill_err:
                print(f"[REDACTED_BY_SCRIPT]")
        browser = None # Reset browser variable for retry

        if attempt < max_init_retries - 1:
            print("[REDACTED_BY_SCRIPT]")
            await asyncio.sleep(3)
        else:
            print("[REDACTED_BY_SCRIPT]")

    # If loop finishes without returning, initialization failed
    print("[REDACTED_BY_SCRIPT]")
    return None

async def get_element_text_robustly(element: nodriver.Element):
    """
    Attempts to get text from a nodriver Element using JavaScript execution
    for increased robustness against stale elements.
    """
    if not element:
        return None
    text = None
    try:
        # Use JavaScript to get the textContent property
        # Pass the element itself as an argument to the JS function
        js_script = "[REDACTED_BY_SCRIPT]"
        text = await element.evaluate(js_script, args=[element]) # Pass element as argument
    except Exception as e:
        # Log the error if JavaScript execution fails
        print(f"[REDACTED_BY_SCRIPT]")
        logging.warning(f"[REDACTED_BY_SCRIPT]", exc_info=False)
        # Optionally, you could try falling back to element.text_content() here,
        # but if JS fails, text_content() is also likely to fail.
        # try:
        #     print("[REDACTED_BY_SCRIPT]")
        #     text = await element.text_content()
        # except Exception as e2:
        #     print(f"[REDACTED_BY_SCRIPT]")
        #     return None
        return None # Return None if JS fails

    # Clean and return the text
    return text.strip() if text else None

async def ensure_browser_and_tab(current_tab: nodriver.Tab = None):
    """[REDACTED_BY_SCRIPT]"""
    global browser # Need browser to re-initialize
    browser_ok = False
    if browser and hasattr(browser, 'proc') and browser.proc and browser.proc.is_running():
        browser_ok = True

    tab_ok = False
    if current_tab and not current_tab.closed: # Check if tab is explicitly closed
        try:
            # Simple check: Get the current URL without navigating
            await current_tab.url # Accessing property might implicitly check connection
            tab_ok = True
        except Exception: # Catches errors if tab is disconnected/invalid
            print(f"[REDACTED_BY_SCRIPT]")
            tab_ok = False

    if browser_ok and tab_ok:
        return current_tab # Existing browser and tab are fine

    # If browser or tab is not okay, re-initialize
    print("[REDACTED_BY_SCRIPT]")
    browser = await initialize_browser() # Use the potentially headless setting from main
    if not browser:
        raise Exception("[REDACTED_BY_SCRIPT]")

    # Get a new tab from the freshly initialized browser
    try:
        # Call browser.get without timeout to get the tab
        new_tab = await browser.get('about:blank') # <--- CORRECTED: Removed timeout kwarg
        print(f"[REDACTED_BY_SCRIPT]")
        # Optional: Set timeout on the new tab if desired
        # await new_tab.set_default_timeout(20)
        return new_tab
    except Exception as e:
        logging.error(f"[REDACTED_BY_SCRIPT]", exc_info=True)
        raise Exception("[REDACTED_BY_SCRIPT]")

#


async def check_homipi_cookie(tab) -> bool:
    """
    Checks for and clicks the Homipi 'I am Human' button, verifying by looking for expected content.
    No longer relies on reading the text of the anti-bot message itself.

    Returns:
        bool: True if the page seems okay (no button or click successful & verified),
              False if an error occurred or the click likely failed verification.
    """
    # Define selectors
    content_selector = "[REDACTED_BY_SCRIPT]" # Element expected on NORMAL page
    human_button_selector = '//input[@value="I am Human"[REDACTED_BY_SCRIPT]"I am Human")]' # Anti-bot button
    max_attempts = 5 # Allow retrying the check/click/verify sequence once

    for attempt in range(max_attempts):
        print(f"[REDACTED_BY_SCRIPT]")
        button_found = False
        try:
           # 1. Check if VISIBLE NORMAL content is ALREADY present
            content_is_visible = False # Flag
            try:
                print(f"[REDACTED_BY_SCRIPT]'{content_selector}'...")
                content_element = await tab.select(content_selector, timeout=2.0)
                if content_element:
                    try:
                        # *** ROBUST VISIBILITY CHECK ***
                        if await content_element.is_displayed():
                            content_is_visible = True
                        else:
                            print("[REDACTED_BY_SCRIPT]")
                    except (AttributeError, nodriver.ProtocolException) as disp_err:
                         print(f"[REDACTED_BY_SCRIPT]")
                    except Exception as disp_err_other:
                         print(f"[REDACTED_BY_SCRIPT]")

                if content_is_visible:
                     print("[REDACTED_BY_SCRIPT]")
                     return True # Success
                else:
                    # Element not found or not visible
                    if content_element: pass # Already printed message if found but not visible
                    else: print("[REDACTED_BY_SCRIPT]")

            except (asyncio.TimeoutError, nodriver.NoSuchElementException):
                print("[REDACTED_BY_SCRIPT]")
            # Keep other exceptions as they were (ProtocolException, general Exception)

            except (asyncio.TimeoutError, nodriver.NoSuchElementException): # Corrected import path
                print("[REDACTED_BY_SCRIPT]")
            except nodriver.ProtocolException as pe_content: # Corrected import path
                print(f"[REDACTED_BY_SCRIPT]")
            except Exception as e_content:
                 print(f"[REDACTED_BY_SCRIPT]")


            # 2. Look for the "I am Human" button
            print("Checking for 'I am Human' button...")
            try:
                button = await tab.select(human_button_selector, timeout=3.0)
                if button:
                    button_found = True
                    print("Detected 'I am Human' button.")
                else:
                    # Should not happen if select doesn't raise error, but check anyway
                    print("[REDACTED_BY_SCRIPT]")
                    # Treat as button not found

            except (asyncio.TimeoutError, nodriver.NoSuchElementException):
                print("'I am Human' button not found.")
                # If no button AND no normal content, state is unclear.
                # Let's try verifying content again after a pause in case it was slow loading.
                await asyncio.sleep(2.0)
                try:
                    await tab.select(content_selector, timeout=2.0)
                    print("[REDACTED_BY_SCRIPT]")
                    return True
                except:
                    print("[REDACTED_BY_SCRIPT]")
                    # If this is the last attempt, fail. Otherwise, loop will retry.
                    if attempt == max_attempts - 1: return False
                    await asyncio.sleep(2) # Wait before next attempt loop
                    continue # Go to next attempt loop

            except nodriver.core.error.ProtocolException as pe_button:
                 print(f"[REDACTED_BY_SCRIPT]")
                 return False # Fail due to instability
            except Exception as e_button:
                 print(f"[REDACTED_BY_SCRIPT]")
                 return False # Fail


            # 3. If Button Was Found, Click It
            if button_found and button: # Ensure button object exists
                print("Attempting to click 'I am Human' button...")
                try:
                    await asyncio.sleep(0.5) # Brief pause before click
                    await button.click(timeout=5)
                    print("[REDACTED_BY_SCRIPT]")
                    # *** INCREASED WAIT TIME ***
                    await asyncio.sleep(5.0)
                except nodriver.ProtocolException as pe_click:
                    # This is common if click causes navigation
                    print(f"[REDACTED_BY_SCRIPT]")
                    print("[REDACTED_BY_SCRIPT]")
                    await asyncio.sleep(3.0) # Extra wait
                except Exception as click_err:
                    print(f"[REDACTED_BY_SCRIPT]")
                    return False # Click itself failed


            print("[REDACTED_BY_SCRIPT]")
            verification_content_visible = False
            try:
                verify_element = await tab.select(content_selector, timeout=7.0)
                if verify_element:
                    try:
                        # *** ROBUST VISIBILITY CHECK ***
                        if await verify_element.is_displayed():
                             verification_content_visible = True
                        else:
                             print(f"[REDACTED_BY_SCRIPT]")
                    except (AttributeError, nodriver.ProtocolException) as disp_err:
                        print(f"[REDACTED_BY_SCRIPT]")
                    except Exception as disp_err_other:
                         print(f"[REDACTED_BY_SCRIPT]")

                if verification_content_visible:
                    print("[REDACTED_BY_SCRIPT]")
                    return True # SUCCESS
                else:
                     # Element not found or not visible
                     if verify_element: pass # Already printed if found but not visible
                     else: print(f"[REDACTED_BY_SCRIPT]")

                     if attempt < max_attempts - 1:
                         await asyncio.sleep(3)
                         continue
                     else:
                         print("[REDACTED_BY_SCRIPT]")
                         return False # FAIL

            except (asyncio.TimeoutError, nodriver.NoSuchElementException):
                print(f"[REDACTED_BY_SCRIPT]")
                if attempt < max_attempts - 1:
                    await asyncio.sleep(3)
                    continue
                else:
                    print("[REDACTED_BY_SCRIPT]")
                    return False # FAIL

            except (asyncio.TimeoutError, nodriver.NoSuchElementException):
                print(f"[REDACTED_BY_SCRIPT]")
                if attempt < max_attempts - 1:
                    await asyncio.sleep(3)
                    continue
                else:
                    print("[REDACTED_BY_SCRIPT]")
                    return False # FAIL
            except nodriver.core.error.ProtocolException as pe_verify:
                print(f"[REDACTED_BY_SCRIPT]")
                return False # FAIL
            except Exception as e_verify:
                # Catch potential errors from is_displayed() too
                print(f"[REDACTED_BY_SCRIPT]")
                return False # FAIL


        # --- Catch errors for the whole attempt ---
        except nodriver.ProtocolException as pe_outer: # Corrected import path
             print(f"[REDACTED_BY_SCRIPT]")
             if attempt < max_attempts - 1:
                 await asyncio.sleep(3)
                 continue
             return False
        except Exception as e_outer:
             print(f"[REDACTED_BY_SCRIPT]") # Print the error itself
             logging.error(f"[REDACTED_BY_SCRIPT]", exc_info=True) # Log with traceback
             return False # Explicitly return False on unexpected errors


    # Should not be reached if max_attempts > 0
    print("[REDACTED_BY_SCRIPT]")
    return False

async def scrape_page2(tab: nodriver.Tab, parsed_data_item: list):
    """
    Scrapes a single property page using nodriver.

    Args:
        tab: The nodriver Tab object to use.
        parsed_data_item: The list containing data for the current item.

    Returns:
        Tuple: (bool indicating success, list containing scraped data or None)
    """
    # Global flags are tricky with async, pass state or return values
    # global driverReset # Replaced by passing tab

    scraped_output = None # Initialize output for this item
    item_succeeded = False

    try:
        addressin = parsed_data_item[1][0]
        address_parts = addressin.split(" ")
        if len(address_parts) < 2:
            logging.error(f"Address format error: '{addressin}'")
            raise ValueError(f"[REDACTED_BY_SCRIPT]")

        addressin2 = addressin.split(" ")[-2] + "-" + addressin.split(" ")[-1]
        postcode_search_url = f"[REDACTED_BY_SCRIPT]"

        print(f"[REDACTED_BY_SCRIPT]")
        #tab = await ensure_browser_and_tab(tab, postcode_search_url) # Ensure tab is working, get potentially new tab
        await tab.get(postcode_search_url)

        # Handle initial consent cookie (assuming it's the same selector)
        # Note: This relies on the element being present quickly.
        # Consider a more robust check if it's dynamic.
        # The driverReset logic is removed, handle cookie once per session maybe?
        # Or check for it each time.
        try:
            consent_button = await tab.select(
                "[REDACTED_BY_SCRIPT]",
                timeout=5 # Wait up to 5 seconds for the button
            )
            if consent_button:
                print("[REDACTED_BY_SCRIPT]")
                await consent_button.click()
                await asyncio.sleep(0.5) # Wait for overlay to disappear
        except Exception: # Catches timeout if not found, or other errors
             print("[REDACTED_BY_SCRIPT]")
             pass # Continue if not found

        # Check for the anti-bot measure
        await check_homipi_cookie(tab)

        number_of_props_text = "There are 10 properties" # Default
        try:
            # Wait slightly longer for the main content area
            prop_count_element = await tab.select("[REDACTED_BY_SCRIPT]", timeout=15)
            if prop_count_element:
                number_of_props_text = await get_element_text_robustly(prop_count_element)
                if not number_of_props_text: number_of_props_text = "There are 10 properties" # Fallback
            else:
                 print("[REDACTED_BY_SCRIPT]")
                 # Decide how to handle: raise error, use default, etc.

        except Exception as e:
             print(f"[REDACTED_BY_SCRIPT]")
             # Log the error: logging.warning(f"[REDACTED_BY_SCRIPT]")

        # --- Anti-bot loop ---
        contact_page_check_count = 0
        while "[REDACTED_BY_SCRIPT]" in number_of_props_text and contact_page_check_count < 3:
            print("Detected 'Contact Support'[REDACTED_BY_SCRIPT]")
            contact_page_check_count += 1
            await check_homipi_cookie(tab) # Retry the cookie check/click
            await asyncio.sleep(1) # Wait a bit
            try:
                prop_count_element = await tab.select("[REDACTED_BY_SCRIPT]", timeout=10)
                if prop_count_element:
                    number_of_props_text = await get_element_text_robustly(prop_count_element)
                    if not number_of_props_text: number_of_props_text = "There are 10 properties" # Fallback
                else:
                     # If still not found after retry, maybe break or raise error
                     print("[REDACTED_BY_SCRIPT]")
                     number_of_props_text = "There are 10 properties" # Reset to default to potentially break loop if text changes
                     break
            except Exception as e:
                 print(f"[REDACTED_BY_SCRIPT]")
                 number_of_props_text = "There are 10 properties" # Reset to default
                 break # Exit loop on error

        if "[REDACTED_BY_SCRIPT]" in number_of_props_text:
             print("Still seeing 'Contact Support'[REDACTED_BY_SCRIPT]")
             raise Exception("[REDACTED_BY_SCRIPT]")

        # --- Extract number of properties ---
        number_of_props = 10 # Default
        try:
            match = [s for s in number_of_props_text.split() if s.isdigit()]
            if match:
                number_of_props = int(match[0])
            # Original parsing logic (less robust)
            # number_of_props = number_of_props_text[(number_of_props_text.index("There are ") + len("There are ")):(number_of_props_text.index(" properties"))]
            # number_of_props = int(number_of_props)
            print(f"[REDACTED_BY_SCRIPT]")
        except Exception as e:
             print(f"[REDACTED_BY_SCRIPT]'{number_of_props_text}'[REDACTED_BY_SCRIPT]")
             number_of_props = 10

        # --- Collect addresses and hrefs from all pages ---
        compare_addresses = []
        compare_addresses_href = []
        num_pages = math.ceil(number_of_props / 10) # 10 items per page usually
        if num_pages == 0: num_pages = 1
        print(f"[REDACTED_BY_SCRIPT]")

        for page_num in range(1, num_pages + 1):
            if page_num > 1:
                page_url = f"[REDACTED_BY_SCRIPT]"
                print(f"[REDACTED_BY_SCRIPT]")
                #tab = await ensure_browser_and_tab(tab, page_url)
                await tab.get(page_url)
                await check_homipi_cookie(tab) # Check cookie on each page load
                # Verify URL after navigation (optional but good practice)
                # current_url = await tab.url()
                # if page_url not in current_url:
                #    print(f"[REDACTED_BY_SCRIPT]")
                #    await asyncio.sleep(1) # Extra wait?

            # Scrape addresses on the current page
            # Wait for the container holding the links to be present
            try:
                await tab.select("[REDACTED_BY_SCRIPT]", timeout=10)
            except Exception:
                print(f"[REDACTED_BY_SCRIPT]")
                continue # Skip to next page if container isn't found

            # Get all address links within the container
            # Adjust selector if structure varies; this assumes direct children divs
            address_elements = await tab.select_all("[REDACTED_BY_SCRIPT]")

            print(f"[REDACTED_BY_SCRIPT]")
            for i, address_link in enumerate(address_elements):
                address_text = None
                href = None
                try:
                    # Check if the element handle seems valid before interacting
                    if not address_link:
                        print(f"[REDACTED_BY_SCRIPT]")
                        continue

                    # Try getting text
                    try:
                        address_text = await get_element_text_robustly(address_link)
                    except Exception as text_e:
                        print(f"[REDACTED_BY_SCRIPT]")
                        # Optionally continue to try getting href, or just continue the outer loop
                        # continue

                    # Try getting href
                    try:
                        href = await address_link.get_attribute("href")
                    except Exception as href_e:
                        # If the 'NoneType' error happened getting href, this might catch it
                        print(f"[REDACTED_BY_SCRIPT]")
                        # continue

                    # Only proceed if both were successful
                    if address_text and href:
                        compare_addresses.append(address_text)
                        compare_addresses_href.append(href)
                        # print(f"[REDACTED_BY_SCRIPT]")
                    else:
                        # Log if one or both failed but didn't raise an exception caught above
                        if not address_text:
                            print(f"[REDACTED_BY_SCRIPT]")
                        if not href:
                             print(f"[REDACTED_BY_SCRIPT]")

                except Exception as e:
                    # Catch any unexpected errors during the checks/calls for this specific link
                    print(f"[REDACTED_BY_SCRIPT]")
                    # Log traceback if needed for debugging:
                    # import traceback
                    # print(traceback.format_exc())
                    pass # Continue to the next link element

        # --- Find the most similar address ---
        target_address = addressin
        most_similar = None
        most_similar_href = None
        min_distance = float('inf')

        # Normalize target address once
        target_norm = target_address.lower().replace(",", "").replace("-", "").replace(" ", "").replace(".", "")

        for i, entry in enumerate(compare_addresses):
            try:
                if entry:
                    entry_norm = entry.lower().replace(",", "").replace("-", "").replace(" ", "").replace(".", "")
                    distance = Levenshtein.distance(target_norm, entry_norm)
                    if distance < min_distance:
                        min_distance = distance
                        most_similar = entry
                        most_similar_href = compare_addresses_href[i] # Get corresponding href
            except Exception as e:
                print(f"[REDACTED_BY_SCRIPT]'{entry}': {e}")
                pass

        if most_similar_href:
            print(f"[REDACTED_BY_SCRIPT]'{most_similar}'[REDACTED_BY_SCRIPT]")
            addressurl = most_similar_href
        else:
            print(f"[REDACTED_BY_SCRIPT]'{target_address}'[REDACTED_BY_SCRIPT]")
            raise ValueError(f"[REDACTED_BY_SCRIPT]") # Fail the item

        # --- Navigate to the specific property page ---
        #tab = await ensure_browser_and_tab(tab, addressurl)
        await tab.get(addressurl)
        await check_homipi_cookie(tab) # Check cookie again

        # Re-check consent (might reappear on property page?)
        try:
            consent_button = await tab.select(
                "[REDACTED_BY_SCRIPT]",
                timeout=3 # Shorter timeout here
            )
            if consent_button:
                print("[REDACTED_BY_SCRIPT]")
                await consent_button.click()
                await asyncio.sleep(0.5)
        except Exception:
            pass # Ignore if not found


        # --- Scrape Property Details ---
        print("[REDACTED_BY_SCRIPT]")
        await asyncio.sleep(random.uniform(0.8, 1.5)) # Keep random delay

        details = {
            'est': [''], 'change': [''], 'range0': [''], 'conf': [''], 'lastp': [''], 'lastd': [''],
            'property_type': [''], 'property_subtype': [''], 'beds': [''], 'receps': [''], 'extens': [''],
            'storey': [''], 'sqm': [''], 'tenure': [''], 'epccurrent': [''], 'epcpotential': [''],
            'council': [''], 'councilband': [''], 'permission': [''], 'age': [''], 'flood': [''], 'la': ['']
        }

        # Wait for the details section to load
        try:
             await tab.select("[REDACTED_BY_SCRIPT]", timeout=15)
        except Exception:
             print("[REDACTED_BY_SCRIPT]")
             raise TimeoutError("[REDACTED_BY_SCRIPT]") # Fail the item

        detail_items = await tab.select_all("[REDACTED_BY_SCRIPT]") # Get all list items' inner divs

        for item_div in detail_items:
            try:
                label_element = await item_div.select('label') # Find label within div
                if not label_element: continue # Skip if no label found

                # Get text robustly, splitting label and value
                full_text = await get_element_text_robustly(label_element)
                if not full_text or '\n' not in full_text: continue # Skip if format is wrong

                key_homipi, value_homipi = full_text.split('\n', 1) # Split only on the first newline
                key_lower = key_homipi.lower().strip()
                value_cleaned = value_homipi.replace("£", "").replace(",", "").strip()

                # Map keys to dictionary
                if "homipi price estimate" in key_lower: details['est'] = [value_cleaned]
                elif "value change" in key_lower: details['change'] = [value_cleaned]
                elif "price range" in key_lower: details['range0'] = [value_cleaned]
                elif "estimate confidence" in key_lower:
                    conf_val = 0
                    if "High" in value_homipi: conf_val = 3
                    elif "Moderate" in value_homipi: conf_val = 2
                    elif "Low" in value_homipi: conf_val = 1
                    details['conf'] = [conf_val]
                elif "last sold price" in key_lower: details['lastp'] = [value_cleaned]
                elif "last sold date" in key_lower: details['lastd'] = [value_homipi.strip()] # Keep original format for date?
                elif "type" in key_lower and "sub type" not in key_lower: details['property_type'] = [value_homipi.strip()]
                elif "sub type" in key_lower:
                    sub_val = 1
                    if "Detached" in value_homipi: sub_val = 4
                    elif "Semi-Detached" in value_homipi: sub_val = 3
                    elif "Terraced" in value_homipi: sub_val = 2
                    details['property_subtype'] = [sub_val]
                elif "bedrooms" in key_lower: details['beds'] = [value_cleaned]
                elif "receptions" in key_lower: details['receps'] = [value_cleaned]
                elif "extensions" in key_lower: details['extens'] = [value_cleaned]
                elif "storey" in key_lower: details['storey'] = [value_cleaned]
                elif "floor area" in key_lower: details['sqm'] = [value_cleaned.replace(" sq m", "").strip()] # Clean unit
                elif "tenure" in key_lower:
                    tenure_val = 1
                    if "Freehold" in value_homipi: tenure_val = 2
                    details['tenure'] = [tenure_val]
                elif "current epc" in key_lower and "/" in value_homipi: details['epccurrent'] = [value_homipi[value_homipi.find("/")+1:].strip()]
                elif "potential epc" in key_lower and "/" in value_homipi: details['epcpotential'] = [value_homipi[value_homipi.find("/")+1:].strip()]
                elif "council tax rate" in key_lower: details['council'] = [value_cleaned] # Keep rate as number string
                elif "council tax band" in key_lower: details['councilband'] = [value_homipi.strip()] # Keep band as letter
                elif "permission" in key_lower: details['permission'] = [value_homipi.strip()]
                elif "new build" in key_lower:
                    age_val = "2025" # Default if 'Yes'
                    if "No" in value_homipi and "-" in value_homipi:
                        try: age_val = value_homipi.split('-')[-1].strip()
                        except: age_val = "2025" # Fallback if parsing fails
                    details['age'] = [age_val]
                elif "flood risk" in key_lower: details['flood'] = [value_homipi.strip()]
                elif "local authority" in key_lower: details['la'] = [value_homipi.strip()]

            except Exception as e:
                print(f"[REDACTED_BY_SCRIPT]")
                # Log error: logging.warning(f"[REDACTED_BY_SCRIPT]", exc_info=True)
                pass

        # --- Helper function to scrape nearby lists ---
        async def scrape_nearby_list(selector):
            try:
                ul_element = await tab.select(selector, timeout=2) # Short timeout for these
                if ul_element:
                    text = await get_element_text_robustly(ul_element)
                    return text.split('\n') if text else []
                return []
            except Exception:
                # print(f"[REDACTED_BY_SCRIPT]")
                return []

        # --- Scrape Nearby Amenities ---
        print("[REDACTED_BY_SCRIPT]")
        nearby = {
            'rail': await scrape_nearby_list("[REDACTED_BY_SCRIPT]"),
            'bus': await scrape_nearby_list("[REDACTED_BY_SCRIPT]"),
            'Pschool': await scrape_nearby_list("[REDACTED_BY_SCRIPT]"),
            'Sschool': await scrape_nearby_list("[REDACTED_BY_SCRIPT]"),
            'Nursery': await scrape_nearby_list("[REDACTED_BY_SCRIPT]"),
            'special': await scrape_nearby_list("[REDACTED_BY_SCRIPT]"),
            'Churches': await scrape_nearby_list("[REDACTED_BY_SCRIPT]"),
            'Mosque': await scrape_nearby_list("[REDACTED_BY_SCRIPT]"),
            'Gurdwara': await scrape_nearby_list("[REDACTED_BY_SCRIPT]"),
            'Synagogue': await scrape_nearby_list("[REDACTED_BY_SCRIPT]"),
            'Mandir': await scrape_nearby_list("[REDACTED_BY_SCRIPT]"),
            'gp': await scrape_nearby_list("[REDACTED_BY_SCRIPT]"),
            'dent': await scrape_nearby_list("[REDACTED_BY_SCRIPT]"),
            'hosp': await scrape_nearby_list("[REDACTED_BY_SCRIPT]"),
            'pharm': await scrape_nearby_list("[REDACTED_BY_SCRIPT]"),
            'opt': await scrape_nearby_list("[REDACTED_BY_SCRIPT]"),
            'clin': await scrape_nearby_list("[REDACTED_BY_SCRIPT]"),
            'other': await scrape_nearby_list("[REDACTED_BY_SCRIPT]"),
        }

        # --- Format Output Data ---
        scraped_output = [[ [parsed_data_item[1][0]] ] + # Address
                          list(details.values()) +        # Property details
                          list(nearby.values())           # Nearby amenities
                         ]

        # --- Check if data seems empty/default ---
        # Create a default empty structure for comparison
        default_empty_details = {k: [''] for k in details}
        default_empty_nearby = {k: [] for k in nearby}
        default_output = [[ [parsed_data_item[1][0]] ] +
                           list(default_empty_details.values()) +
                           list(default_empty_nearby.values())
                         ]

        if scraped_output == default_output:
             print(f"[REDACTED_BY_SCRIPT]")
             raise ValueError("[REDACTED_BY_SCRIPT]")

        item_succeeded = True
        print(f"[REDACTED_BY_SCRIPT]")

    except Exception as e:
        print(f"[REDACTED_BY_SCRIPT]")
        print(traceback.format_exc()) # Keep traceback for debugging
        logging.error(f"[REDACTED_BY_SCRIPT]", exc_info=True) # Log detailed error
        item_succeeded = False # Ensure it's False on error

    finally:
        # Return success status and the scraped data (or None if failed)
        return item_succeeded, scraped_output


async def scrape_with_timeout(parsed_data: list, start_index: int, output_filepath: str, timeout_seconds=120):
    """[REDACTED_BY_SCRIPT]"""
    global browser # Need access to the global browser instance

    parsed_data = []
    processed_count = 0
    start_index = 1687 # Define the starting index

    # --- Load Data ---
    input_filepath = r"[REDACTED_BY_SCRIPT]"
    output_filepath = r"[REDACTED_BY_SCRIPT]" # New output file
    print(f"[REDACTED_BY_SCRIPT]")
    try:
        with open(input_filepath, 'r', newline='', encoding='utf-8') as csvfile:
            csv_reader = csv.reader(csvfile)
            # Skip header rows if necessary (adjust count)
            # next(csv_reader) # Example: skip one header row

            # Skip rows up to the start_index
            for _ in range(start_index):
                 try:
                     next(csv_reader)
                 except StopIteration:
                     print(f"[REDACTED_BY_SCRIPT]")
                     break # Stop if file ends before reaching start index

            # Read remaining rows
            for row_num, row in enumerate(csv_reader, start=start_index):
                parsed_row = []
                for cell in row:
                    try:
                        parsed_cell = ast.literal_eval(cell.strip())
                    except (SyntaxError, ValueError):
                        parsed_cell = cell.strip() # Fallback to raw string
                    parsed_row.append(parsed_cell)
                # Basic validation (e.g., check if expected address structure is present)
                if len(parsed_row) > 1 and isinstance(parsed_row[1], list) and len(parsed_row[1]) > 0:
                     parsed_data.append(parsed_row)
                else:
                     print(f"[REDACTED_BY_SCRIPT]") # +1 because enumerate is 0-based
    except FileNotFoundError:
        print(f"[REDACTED_BY_SCRIPT]")
        return # Stop execution if input file is missing
    except Exception as e:
        print(f"[REDACTED_BY_SCRIPT]")
        logging.error(f"[REDACTED_BY_SCRIPT]", exc_info=True)
        return # Stop execution on loading error

    print(f"[REDACTED_BY_SCRIPT]")
    if not parsed_data:
        print("No data to process.")
        return

    # --- Scraping Loop ---
    current_tab = None # Initialize tab variable

    for i, item_data in enumerate(parsed_data):
        item_index_original = start_index + i # Keep track of original index for logging
        print(f"[REDACTED_BY_SCRIPT]")
        print(f"[REDACTED_BY_SCRIPT]")

        max_item_retries = 3
        attempt = 0
        item_completed_successfully = False

        while attempt < max_item_retries and not item_completed_successfully:
            attempt += 1
            print(f"[REDACTED_BY_SCRIPT]")
            browser_needs_refresh = False
            scraped_data = None

            try:
                # --- Ensure Browser and Tab are Ready ---
                # Add hasattr check for robustness, but main fix is ensuring browser is valid *before* this point
                # The ensure_browser_and_tab function handles this better.
                # This specific check might become:
                # current_tab = await ensure_browser_and_tab(current_tab)
                # (See full code from previous answer for ensure_browser_and_tab implementation)

                # If keeping the check here temporarily for debugging:
                browser_ok = False
                if browser and hasattr(browser, 'proc') and browser.proc and browser.proc.is_running():
                    browser_ok = True

                if not browser_ok:
                    print("[REDACTED_BY_SCRIPT]")
                    browser = await initialize_browser()
                    if not browser:
                        print("[REDACTED_BY_SCRIPT]")
                        # No point retrying if browser init fails repeatedly
                        browser_needs_refresh = True # Signal to break outer loop? Or just log?
                        break # Break inner retry loop for this item
                    try:
                        current_tab = await browser.get('about:blank') # <--- CORRECTED: Removed timeout kwarg
                        if not current_tab:
                            raise Exception("browser.get('about:blank'[REDACTED_BY_SCRIPT]") # Check if tab is valid
                        # Accessing tab.id here might still fail if the tab is invalid, log cautiously
                        print(f"[REDACTED_BY_SCRIPT]") # Simplified log message

                    except Exception as tab_err:
                        print(f"[REDACTED_BY_SCRIPT]")
                        logging.error(f"[REDACTED_BY_SCRIPT]", exc_info=True)
                        # If we can't get a tab, the browser might be broken
                        browser_needs_refresh = True # Signal to kill this browser instance
                        # Attempt to close the potentially broken browser before breaking
                        if browser: await browser.close()
                        browser = None
                        break # Break inner retry loop
                    print("[REDACTED_BY_SCRIPT]")
                elif not current_tab: # Browser exists, but no tab? Get one.
                    print("Getting new tab...")
                    current_tab = await browser.first_tab()
                    if not current_tab:
                        print("[REDACTED_BY_SCRIPT]")
                        browser_needs_refresh = True # Maybe close browser?
                        break # Break inner retry loop

                # --- Perform Scraping within Timeout ---
                print("[REDACTED_BY_SCRIPT]")
                scrape_success, scraped_data = await asyncio.wait_for(
                    scrape_page2(current_tab, item_data),
                    timeout=timeout_seconds
                )
                # --- End core scraping ---

                if scrape_success and scraped_data:
                    print(f"[REDACTED_BY_SCRIPT]")
                    item_completed_successfully = True
                    processed_count += 1
                    # Write data immediately on success
                    try:
                        with open(output_filepath, 'a', newline='', encoding='utf-8') as csvfile:
                            writer = csv.writer(csvfile)
                            writer.writerows(scraped_data)
                        print(f"[REDACTED_BY_SCRIPT]")
                    except Exception as write_err:
                        print(f"[REDACTED_BY_SCRIPT]")
                        logging.error(f"[REDACTED_BY_SCRIPT]", exc_info=True)
                        # Decide if this should count as a failure for retry purposes?
                        # For now, we consider scrape successful but log write error.
                else:
                    # scrape_page2 returned False or None
                    print(f"[REDACTED_BY_SCRIPT]")
                    # No browser refresh needed unless scrape_page2 raised an unexpected exception
                    # If it returned False, it handled its internal logic failure.

            except asyncio.TimeoutError:
                print(f"[REDACTED_BY_SCRIPT]")
                logging.warning(f"[REDACTED_BY_SCRIPT]")
                browser_needs_refresh = True # Assume browser is stuck
            except Exception as e: # Catch errors from scrape_page2 or wait_for
                print(f"[REDACTED_BY_SCRIPT]")
                print(traceback.format_exc())
                logging.error(f"[REDACTED_BY_SCRIPT]", exc_info=True)
                browser_needs_refresh = True # Assume browser is unstable
            finally:
                # --- Cleanup After Attempt ---
                if browser_needs_refresh and attempt < max_item_retries:
                    print("[REDACTED_BY_SCRIPT]")
                    if browser: # Check if the variable holds *something*
                        browser_pid = browser.proc.pid if hasattr(browser, 'proc') and browser.proc else None
                        print(f"[REDACTED_BY_SCRIPT]'N/A'})...")
                        try:
                            await browser.close() # <--- Correct: Use await
                            print("[REDACTED_BY_SCRIPT]")
                        except Exception as close_err:
                             # This might happen if the browser already crashed hard
                            print(f"  Error during 'await browser.close()'[REDACTED_BY_SCRIPT]")
                            logging.warning(f"[REDACTED_BY_SCRIPT]")

                        # Check if process still exists after close attempt
                        if browser_pid and psutil.pid_exists(browser_pid):
                             print(f"[REDACTED_BY_SCRIPT]")
                             try:
                                 process = psutil.Process(browser_pid)
                                 process.kill()
                                 process.wait(1) # Brief wait
                                 print(f"[REDACTED_BY_SCRIPT]")
                             except Exception as kill_err:
                                 print(f"[REDACTED_BY_SCRIPT]")
                                 logging.warning(f"[REDACTED_BY_SCRIPT]")
                    else:
                         print("[REDACTED_BY_SCRIPT]")

                    # Reset globals for next attempt
                    browser = None
                    current_tab = None # Tab is invalid if browser is refreshed
                    gc.collect() # Encourage memory cleanup
                    print("[REDACTED_BY_SCRIPT]")
                    await asyncio.sleep(2) # Delay before next attempt

        # --- After all retries for an item ---
        if not item_completed_successfully:
            print(f"[REDACTED_BY_SCRIPT]")
            # Optionally log failed items to a separate file
            # with open("failed_items.log", "a") as f:
            #     f.write(f"[REDACTED_BY_SCRIPT]")

    print(f"[REDACTED_BY_SCRIPT]")
    print(f"[REDACTED_BY_SCRIPT]")


async def main():
    global browser # To allow final cleanup
    print("[REDACTED_BY_SCRIPT]")
    start_time = time.time()

    # --- Configuration ---
    start_index = 1687
    input_filepath = r"[REDACTED_BY_SCRIPT]"
    output_filepath = r"[REDACTED_BY_SCRIPT]"
    headless = False # Set to True for headless execution
    scrape_timeout = 150 # Seconds per item attempt

    # --- Load Data ---
    parsed_data = []
    print(f"[REDACTED_BY_SCRIPT]")
    try:
        with open(input_filepath, 'r', newline='', encoding='utf-8') as csvfile:
            csv_reader = csv.reader(csvfile)
            # Skip rows if header exists or start_index > 0
            for _ in range(start_index):
                 try: next(csv_reader)
                 except StopIteration: break
            # Read the rest
            for row_num, row in enumerate(csv_reader, start=start_index):
                # ... (your existing data parsing logic) ...
                parsed_row = []
                for cell in row:
                    try: parsed_cell = ast.literal_eval(cell.strip())
                    except: parsed_cell = cell.strip()
                    parsed_row.append(parsed_cell)
                # Add validation if needed
                if len(parsed_row) > 1 and isinstance(parsed_row[1], list) and len(parsed_row[1]) > 0:
                    parsed_data.append(parsed_row)
                else:
                     logging.warning(f"[REDACTED_BY_SCRIPT]")

        print(f"[REDACTED_BY_SCRIPT]")
        if not parsed_data: return # Exit if no data

    except FileNotFoundError:
        print(f"[REDACTED_BY_SCRIPT]")
        return
    except Exception as e:
        print(f"[REDACTED_BY_SCRIPT]")
        logging.critical(f"[REDACTED_BY_SCRIPT]", exc_info=True)
        return

    # --- Run Scraper ---
    try:
        # Browser initialization is handled within scrape_with_timeout's loop via ensure_browser_and_tab
        print("[REDACTED_BY_SCRIPT]")
        await scrape_with_timeout(parsed_data, start_index, output_filepath, timeout_seconds=scrape_timeout)

    except Exception as main_err:
        print(f"[REDACTED_BY_SCRIPT]")
        print(f"[REDACTED_BY_SCRIPT]")
        print(f"[REDACTED_BY_SCRIPT]")
        print(traceback.format_exc())
        logging.critical(f"[REDACTED_BY_SCRIPT]", exc_info=True)
    finally:
        print("[REDACTED_BY_SCRIPT]")
        if browser:
            print("[REDACTED_BY_SCRIPT]")
            await asyncio.sleep(1) # Short pause before final close
            try:
                proc_pid = browser.proc.pid if hasattr(browser, 'proc') and browser.proc else None
                await browser.close()
                if proc_pid and psutil.pid_exists(proc_pid):
                    print(f"[REDACTED_BY_SCRIPT]")
                    try:
                         psutil.Process(proc_pid).kill()
                    except: pass # Ignore errors killing already dead process
            except Exception as quit_err:
                print(f"[REDACTED_BY_SCRIPT]")
                logging.error(f"[REDACTED_BY_SCRIPT]", exc_info=False)
            finally:
                browser = None
        else:
            print("[REDACTED_BY_SCRIPT]")

        end_time = time.time()
        print(f"[REDACTED_BY_SCRIPT]")
        print("[REDACTED_BY_SCRIPT]")


if __name__ == "__main__":
    # Setup asyncio loop based on OS
    # if os.name == 'nt': # Windows specific event loop policy (optional)
    #     asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(main())