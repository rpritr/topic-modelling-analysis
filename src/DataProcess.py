import re

class DataProcess:
    
    def __init__(self, texts):
        self.texts = texts
        

    def preprocess_text(self):
        """Osnovni preprocessing tekstov"""
        # Slovenian stopwords - osnovni nabor
        slovenian_stopwords = [
            'in', 'je', 'na', 'za', 'z', 'se', 'v', 'da', 'ki', 'po', 
            'so', 'od', 'pri', 'ni', 'ter', 'kot', 'ali', 'ima', 'bilo',
            'biti', 'tega', 'tudi', 'bo', 'več', 'če', 'vse', 'do', 'še'
        ]
        
        return self.texts, slovenian_stopwords
