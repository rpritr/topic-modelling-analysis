"""
Class for Data Processing
    - init # init DataProcess, args: texts
    - preprocess_text # Basic text preprocessing, return: texts, slovenian_stopwods
"""

class DataProcess:
    
    def __init__(self, texts):
        self.texts = texts
        
    def preprocess_text(self):
        """Basic text preprocessing"""
        # Slovenian stopwords - basic
        slovenian_stopwords = [
            'in', 'je', 'na', 'za', 'z', 'se', 'v', 'da', 'ki', 'po', 
            'so', 'od', 'pri', 'ni', 'ter', 'kot', 'ali', 'ima', 'bilo',
            'biti', 'tega', 'tudi', 'bo', 'več', 'če', 'vse', 'do', 'še'
        ]
        
        return self.texts, slovenian_stopwords
