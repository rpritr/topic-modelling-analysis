# Topic Modelling Analysis on Slovenian Job Listings

Python script for topic modelling on Slovenian Job Listings from ZRSZ (ess.gov.si). It offers two scripts one for scrapping URL for job listing and the other for extracting topics. 

## Job Listing Scrapper 

Script for running scrapper on URL to extracting raw text for job listings.

### Install

```bash
conda activate topic-modelling
pip install -r requirements.txt
```

**Note:** Also intall ChromeDriver with:
- macOS: `brew install chromedriver`
- more info: https://chromedriver.chromium.org/

### Usage

```bash
python scrape_job_listing.py
```

Script will:
1. Open URL for job post
2. Extract text from specific job post
3. Save text into `job_text.txt`

### Adjusting scapper URL

URL can be set in `scrape_job_listing.py`:

```python
url = 'https://www.ess.gov.si/iskalci-zaposlitve/iskanje-zaposlitve/iskanje-dela/?idp=VAŠA_ID/#/pdm/VAŠA_ID'
```

## Topic Modelling on job listing

Python script for topic modelling on Slovenian Job Listings from ZRSZ (ess.gov.si).

```bash
python topic_modelling.py
```