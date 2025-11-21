#!/usr/bin/env python3
"""
Scraper for extracting ZRSZ (ess.gov.si) job listings
Export text into txt file
"""

import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from selenium.webdriver.chrome.options import Options


def scrape_job_listing(url, output_file='job_text.txt'):
    """
    Extract job listing from url
    
    Args:
        url: URL for job listing
        output_file: txt file for export
    """
    # Set Chrome options
    chrome_options = Options()
    # chrome_options.add_argument('--headless')  # Debug option
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-blink-features=AutomationControlled')
    chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])
    
    # Inicialize Chrome
    driver = webdriver.Chrome(options=chrome_options)
    
    try:
        print(f"Loading URL: {url}")
        driver.get(url)
        
        # Wait for URL to load
        wait = WebDriverWait(driver, 20)
        
        print("Waiting for DOM load 'scrolled-pdm-body'...")
        # Wait for text
        wait.until(EC.presence_of_element_located((By.CLASS_NAME, 'scrolled-pdm-body')))
        print("Element found!")
        time.sleep(3)  # add extra time for URL to load
        
        page_number = 1
        total_chars = 0
        
        # Open file for write
        with open(output_file, 'w', encoding='utf-8') as f:
            while True:
                print(f"Extracting page {page_number}...")
                
                try:
                    # Find element with text
                    body_element = driver.find_element(By.CLASS_NAME, 'scrolled-pdm-body')
                    text_content = body_element.text.strip()
                    
                    if text_content:
                        # Write to file
                        if page_number > 1:
                            f.write('\n\n')
                        f.write(f"--- Page {page_number} ---\n{text_content}\n")
                        f.flush()  # Save to disk
                        total_chars += len(text_content)
                        print(f"  ✓ Extracted {len(text_content)} characters → written in {output_file}")
                    
                    # Try next page
                    try:
                        next_button = driver.find_element(By.CSS_SELECTOR, 'button.bg-right.btn-sm')
                        
                        # Check if button on page exists (next page)
                        if next_button.is_enabled() and next_button.is_displayed():
                            print("  Clicking on next page button...")
                            driver.execute_script("arguments[0].click();", next_button)
                            time.sleep(2)  # Wait for URL to load
                            page_number += 1
                        else:
                            print("  Button for next page is disabled")
                            break
                            
                    except NoSuchElementException:
                        print("  Button for next page found")
                        break
                        
                except NoSuchElementException:
                    print(f"  Element 'scrolled-pdm-body' not found on page {page_number}")
                    break
        
        print(f"\n✓ Text saved to file: {output_file}")
        print(f"✓ Total pages extracted: {page_number}")
        print(f"✓ Total characters: {total_chars}")
        
        return True
        
    except TimeoutException:
        print("✗ Page was not loaded in time allocated")
        return None
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return None
        
    finally:
        driver.quit()


if __name__ == '__main__':
    # URL of job listing
    url = 'https://www.ess.gov.si/iskalci-zaposlitve/iskanje-zaposlitve/iskanje-dela/?idp=3356108/#/pdm/3356108'
    
    # Run job listing scrapper
    scrape_job_listing(url, 'job_text.txt')

