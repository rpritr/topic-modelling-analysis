"""
Class for Data Loading from filename
    - init # init DataLoader, args: filename
    - load_job_descriptions # Load job desciptions from filename, args: filename, return job_description
"""

import re

class DataLoader:
    
    def __init__(self, filename):
        """Init DataLoader from filename"""
        self.filename = filename
    
    def load_plain(self, filename):
        with open(filename, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    
    def load_job_descriptions(self, filename):
        """Load job desciptions from filename"""
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # split job listings per page "--- Stran X ---"
        pages = re.split(r'--- Stran \d+ ---', content)
        
        job_descriptions = []
        for page in pages:
            if not page.strip():
                continue
            
            # Search for "Opis del in nalog" section
            match = re.search(r'Opis del in nalog\s*\n(.+?)(?=\nNudimo|\nPričakujemo|\n---|$)', 
                            page, re.DOTALL)
            if match:
                description = match.group(1).strip()
                if description:
                    job_descriptions.append(description)
        
        return job_descriptions
