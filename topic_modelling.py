"""
Script for topic modelling on text
"""

from src.BERTopicVisualise import BertTopicVisualise
from src.DataLoader import DataLoader
from src.DataProcess import DataProcess
from src.TopicModel import TopicModel
from src.LDATopicVisualise import LDATopicVisualise
import argparse
import os

def main():
    parser = argparse.ArgumentParser(
        description="Topic Modelling Pipeline"
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        default="job_test.txt",
        help="Path to input file with job descriptions (default: job_test.txt)"
    )
    
    parser.add_argument(
        "--lemmatized",
        action="store_true",
        help="Treat input file as one lemmatized document per line"
    )
    
    args = parser.parse_args()
    filename = args.input

    if not os.path.isfile(filename):
        print(f"[ERROR] Input file not found: {filename}")
        return
    
    dl = DataLoader(filename=filename)
    
    if args.lemmatized:
        print(f"Loading lemmatized job descriptions from {filename}...")
        job_descriptions = dl.load_plain(filename)
    else:
        print(f"Loading job descriptions from {filename} (raw format)...")
        job_descriptions = dl.load_job_descriptions(filename)

    if not job_descriptions:
        print("[ERROR] No job descriptions found!")
        return
    
    print(f"Found {len(job_descriptions)} job descriptions\n")
    
    n_topics = 3  # Number of topics
    n_top_words = 5  # Number of top words in topic
    
    # Exectute Topic Modelling
    LDA(job_descriptions, n_topics, n_top_words)
    BERTopic(job_descriptions, n_topics, n_top_words)
    
def BERTopic(job_descriptions, n_topics, n_top_words):

    # Preprocessing
    dp = DataProcess(texts=job_descriptions)
    texts, stopwords = dp.preprocess_text()
        
    bert_tm = TopicModel(texts, n_topics, n_top_words)
    result = bert_tm.perform_topic_modeling_bert()
    bert_vis = BertTopicVisualise(
        result["topic_model"],
        n_topics
    )

    bert_vis.visualize_topics_wordcloud(output_file="wordclouds_bertopic.png")
    
def LDA(job_descriptions, n_topics, n_top_words):
    
    # Preprocessing
    dp = DataProcess(texts=job_descriptions)
    #texts = dp.lemmatize_texts()
    texts, stopwords = dp.preprocess_text()
    

    # Exectute topic modelling
    tm = TopicModel(texts,
        n_topics=n_topics,
        n_top_words=n_top_words)
    
    lda_model, vectorizer, doc_term_matrix = tm.perform_topic_modeling_lda()
    
    tv = LDATopicVisualise(lda_model, vectorizer, n_topics)
    
    # Generate wordcloud visualization
    tv.visualize_topics_wordcloud(output_file="wordclouds_lda.png")
    
    print("\n✓ Topic modeling done!")


if __name__ == "__main__":
    main()
