"""
Class for Data Processing
    - init # init DataProcess, args: texts
    - preprocess_text # Basic text preprocessing, return: texts, slovenian_stopwods
"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
from src.DataProcess import DataProcess
from src.DataProcess import DataProcess

class TopicModel:
    
    def __init__(self, texts, n_topics=5, n_top_words=10):
        self.texts = texts
        self.n_topics = n_topics
        self.n_top_words = n_top_words
        
    def perform_topic_modeling(self):
        """Execute LDA topic modeling"""
        
        # Preprocessing
        dp = DataProcess(texts=self.texts)
        texts, stopwords = dp.preprocess_text()
        
        # Create document-term matrix
        print(f"Analysing {len(texts)} job listings...")
        
        vectorizer = CountVectorizer(
            max_df=0.95,  # Ignore words that appear in more than 95% of documents
            min_df=2,     # Ignore words that appear in less than 2 documents
            stop_words=stopwords,
            lowercase=True,
            max_features=1000
        )
        
        doc_term_matrix = vectorizer.fit_transform(texts)
        
        # LDA model
        print(f"\nBuilding LDA model with {self.n_topics} topics...")
        lda_model = LatentDirichletAllocation(
            n_components=self.n_topics,
            random_state=42,
            max_iter=20,
            learning_method='online'
        )
        
        lda_model.fit(doc_term_matrix)
        
        # Display topics
        print("\n" + "="*80)
        print(f"TOP {self.n_top_words} WORDS FOR EACH TOPICS")
        print("="*80)
        
        feature_names = vectorizer.get_feature_names_out()
        
        for topic_idx, topic in enumerate(lda_model.components_):
            top_words_idx = topic.argsort()[-self.n_top_words:][::-1]
            top_words = [feature_names[i] for i in top_words_idx]
            
            print(f"\nTOPIC {topic_idx + 1}:")
            print(f"  {', '.join(top_words)}")
        
        print("\n" + "="*80)
        
        # Show some examples of job ads and their dominant themes
        doc_topic_dist = lda_model.transform(doc_term_matrix)
        
        print("EXAMPLE WITH JOB LISTINGS AND TOPICS:")
        print("="*80)
        
        for i in range(min(5, len(texts))):
            dominant_topic = np.argmax(doc_topic_dist[i])
            topic_prob = doc_topic_dist[i][dominant_topic]
            
            print(f"\nJob {i+1} (Topic {dominant_topic + 1}, probablity: {topic_prob:.2f}):")
            print(f"  {texts[i][:200]}...")
        
        return lda_model, vectorizer, doc_term_matrix
