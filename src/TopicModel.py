import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
import matplotlib.pyplot as plt
from src.DataLoader import DataLoader
from src.DataProcess import DataProcess
from wordcloud import WordCloud

from src.DataProcess import DataProcess

class TopicModel:
    
    def __init__(self, texts, n_topics=5, n_top_words=10):
        self.texts = texts
        self.n_topics = n_topics
        self.n_top_words = n_top_words
        
    def perform_topic_modeling(self):
        """Izvede LDA topic modeling"""
        
        # Preprocessing
        dp = DataProcess(texts=self.texts)
        texts, stopwords = dp.preprocess_text()
        
        # Ustvari document-term matrix
        print(f"Analiziram {len(texts)} job oglasov...")
        
        vectorizer = CountVectorizer(
            max_df=0.95,  # Ignoriraj besede ki se pojavljajo v več kot 95% dokumentov
            min_df=2,      # Ignoriraj besede ki se pojavljajo v manj kot 2 dokumentih
            stop_words=stopwords,
            lowercase=True,
            max_features=1000
        )
        
        doc_term_matrix = vectorizer.fit_transform(texts)
        
        # LDA model
        print(f"\nGradem LDA model s {self.n_topics} temami...")
        lda_model = LatentDirichletAllocation(
            n_components=self.n_topics,
            random_state=42,
            max_iter=20,
            learning_method='online'
        )
        
        lda_model.fit(doc_term_matrix)
        
        # Prikaži teme
        print("\n" + "="*80)
        print(f"TOP {self.n_top_words} BESED ZA VSAKO TEMO")
        print("="*80)
        
        feature_names = vectorizer.get_feature_names_out()
        
        for topic_idx, topic in enumerate(lda_model.components_):
            top_words_idx = topic.argsort()[-self.n_top_words:][::-1]
            top_words = [feature_names[i] for i in top_words_idx]
            
            print(f"\nTEMA {topic_idx + 1}:")
            print(f"  {', '.join(top_words)}")
        
        print("\n" + "="*80)
        
        # Prikaži nekaj primerov job oglasov in njihove dominantne teme
        doc_topic_dist = lda_model.transform(doc_term_matrix)
        
        print("\nPRIMER OGLASOV IN NJIHOVE DOMINANTNE TEME:")
        print("="*80)
        
        for i in range(min(5, len(texts))):
            dominant_topic = np.argmax(doc_topic_dist[i])
            topic_prob = doc_topic_dist[i][dominant_topic]
            
            print(f"\nOglas {i+1} (Tema {dominant_topic + 1}, verjetnost: {topic_prob:.2f}):")
            print(f"  {texts[i][:200]}...")
        
        return lda_model, vectorizer, doc_term_matrix
