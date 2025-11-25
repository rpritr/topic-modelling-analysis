from src.DataLoader import DataLoader
from src.DataLemmatize import DataLemmatize


def main():
    dl = DataLoader(filename='job_test.txt')
    
    # Load data
    print("Loading job descriptions from job_test.txt...")
    
    texts = dl.load_job_descriptions("job_test.txt")
    dp = DataLemmatize(texts)
    lemmas = dp.lemmatize_texts()
    with open("lemmatized_jobs.txt", "w") as f:
        for line in lemmas:
            f.write(line.strip() + "\n")
            
if __name__ == "__main__":
    main()