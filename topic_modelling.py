"""
Script for topic modelling on text
"""

from src.DataLoader import DataLoader
from src.TopicModel import TopicModel
from src.TopicVisualise import TopicVisualise

def main():
    
    dl = DataLoader(filename='job_test.txt')
    
    # Load data
    print("Loading job descriptions from job_test.txt...")
    job_descriptions = dl.load_job_descriptions('job_test.txt')
    
    if not job_descriptions:
        print("No job desciptions found!")
        return
    
    print(f"Found {len(job_descriptions)} job descriptions\n")
    
    # Exectute topic modelling
    n_topics = 5  # Number of topics
    n_top_words = 10  # Number of top words in topic
    
    tm = TopicModel(job_descriptions,
        n_topics=n_topics,
        n_top_words=n_top_words)
    
    lda_model, vectorizer, doc_term_matrix = tm.perform_topic_modeling()
    
    tv = TopicVisualise(lda_model, vectorizer, n_topics)
    
    # Generate wordcloud visualization
    tv.visualize_topics_wordcloud()
    
    print("\n✓ Topic modeling done!")


if __name__ == "__main__":
    main()
