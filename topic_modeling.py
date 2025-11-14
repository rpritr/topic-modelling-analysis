"""
Enostavna skripta za topic modeling iz job oglasov
"""

from src.DataLoader import DataLoader
from src.TopicModel import TopicModel
from src.TopicVisualise import TopicVisualise

def main():
    
    dl = DataLoader(filename='job_test.txt')
    
    # Naloži podatke
    print("Nalagam job opise iz job_test.txt...")
    job_descriptions = dl.load_job_descriptions('job_test.txt')
    
    if not job_descriptions:
        print("Nisem našel nobenih job opisov!")
        return
    
    print(f"Najdenih {len(job_descriptions)} job opisov\n")
    
    # Izvedi topic modeling
    n_topics = 5  # Število tem
    n_top_words = 10  # Število top besed na temo
    
    tm = TopicModel(job_descriptions,
        n_topics=n_topics,
        n_top_words=n_top_words)
    
    lda_model, vectorizer, doc_term_matrix = tm.perform_topic_modeling()
    
    tv = TopicVisualise(lda_model, vectorizer, n_topics)
    # Generiraj wordcloud vizualizacijo
    tv.visualize_topics_wordcloud()
    
    print("\n✓ Topic modeling končan!")


if __name__ == "__main__":
    main()

