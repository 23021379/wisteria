import datetime
from selenium.common.exceptions import StaleElementReferenceException
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