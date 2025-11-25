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
        
        cleaned_texts = []
        for text in self.texts:
            words = text.lower().split()
            filtered = [w for w in words if w not in slovenian_stopwords]
            cleaned_texts.append(" ".join(filtered))
            
        return cleaned_texts, slovenian_stopwords