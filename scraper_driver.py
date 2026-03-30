import os
import time
import logging
import traceback
import psutil
import undetected_chromedriver as uc
from fake_useragent import UserAgent
from selenium.common.exceptions import WebDriverException



def initialize_driver(driver,driverReset,log_filename):
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
        driver = None # Ensure driver is None before proceeding
        print("[REDACTED_BY_SCRIPT]")
        # Search for both potential names
        found_pids = find_chromedriver_processes(include_undetected=True)
        terminate_processes(found_pids,log_filename)
        print("[REDACTED_BY_SCRIPT]")
        time.sleep(1) # Small delay

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
            terminate_processes(found_pids,log_filename)
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
                terminate_processes(found_pids,log_filename)
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
            terminate_processes(found_pids,log_filename)
            if attempt < max_init_retries - 1:
                print("[REDACTED_BY_SCRIPT]")
                time.sleep(3)
            else:
                 print("[REDACTED_BY_SCRIPT]")

    # If loop finishes without returning, initialization failed
    print("[REDACTED_BY_SCRIPT]")
    return None # IMPORTANT: Return None clearly on failure



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

def terminate_processes(pids,log_filename):
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
            # Log any other unexpected error
            logging.error(f"[REDACTED_BY_SCRIPT]", exc_info=True, extra=log_extra) # exc_info=True adds traceback
            print(f"[REDACTED_BY_SCRIPT]") # Modify console message



def check_driver_working(driver, addressurl):
    driver.get(addressurl)
    return