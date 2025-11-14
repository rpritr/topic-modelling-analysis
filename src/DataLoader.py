import re

class DataLoader:
    
    def __init__(self, filename):
        self.filename = filename
        
    def load_job_descriptions(self, filename):
        """Naloži job opise iz datoteke"""
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Razdeli na posamezne oglase po "--- Stran X ---"
        pages = re.split(r'--- Stran \d+ ---', content)
        
        job_descriptions = []
        for page in pages:
            if not page.strip():
                continue
            
            # Išči "Opis del in nalog" sekcijo
            match = re.search(r'Opis del in nalog\s*\n(.+?)(?=\nNudimo|\nPričakujemo|\n---|$)', 
                            page, re.DOTALL)
            if match:
                description = match.group(1).strip()
                if description:
                    job_descriptions.append(description)
        
        return job_descriptions


    def preprocess_text(self, texts):
        """Osnovni preprocessing tekstov"""
        # Slovenian stopwords - osnovni nabor
        slovenian_stopwords = [
            'in', 'je', 'na', 'za', 'z', 'se', 'v', 'da', 'ki', 'po', 
            'so', 'od', 'pri', 'ni', 'ter', 'kot', 'ali', 'ima', 'bilo',
            'biti', 'tega', 'tudi', 'bo', 'več', 'če', 'vse', 'do', 'še'
        ]
        
        return texts, slovenian_stopwords
