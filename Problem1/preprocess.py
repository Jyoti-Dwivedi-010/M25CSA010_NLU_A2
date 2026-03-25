import os
import re
import string
import json
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Necessary downloads for the first run
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

# preprocessing function to clean and tokenize text, ensuring it is predominantly English and removing boilerplate content specific to the IIT Jodhpur website 
def preprocess_text(text):
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove Specific Repeating Boilerplate from IITJ Site
    text = re.sub(r'copyright © \d{4} all rights reserved.*?email the wim \.', '', text)
    text = re.sub(r'https?://[^\s<>"]+|www\.[^\s<>"]+', '', text)
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
    
    # 3. Remove non-alphabetical tokens (excessive punctuation/non-text)
    text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)
    text = re.sub(r'\d+', ' ', text) # Also strip out raw numbers from embeddings unless needed
    
    # 4. Tokenization (Convert continuous text to word list)
    tokens = word_tokenize(text)
    
    # 5. Stopword filtering & short word removal (Keep meaningful English text)
    stop_words = set(stopwords.words('english'))
   
    filtered_tokens = [w for w in tokens if w not in stop_words and len(w) > 1]
    
    return filtered_tokens

# function to generate statistics about the dataset and create a word cloud visualization based on the tokens extracted from the corpus, providing insights into the most common words and overall vocabulary richness of the IIT Jodhpur documents.
def generate_statistics_and_cloud(all_tokens, num_documents):
    vocab = set(all_tokens)
    vocab_size = len(vocab)
    total_tokens = len(all_tokens)
    
    # Report Dataset Statistics
    print("\n--- DATASET STATISTICS ---")
    print(f"Total Number of Documents: {num_documents}")
    print(f"Total Tokens: {total_tokens}")
    print(f"Vocabulary Size: {vocab_size}")
    
    # Generate Word Cloud based on tokens
    text_content = ' '.join(all_tokens)
    
    print("\nGenerating Word Cloud...")
    wordcloud = WordCloud(width=800, height=400, background_color='white', collocations=False).generate(text_content)
    
    # Plotting code
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title("IIT Jodhpur Corpus Word Cloud")
    
    # Save visualization to disk
    os.makedirs("visualizations", exist_ok=True)
    plt.savefig("visualizations/wordcloud.png")
    print("WordCloud saved at visualizations/wordcloud.png")


# load and preprocess the raw documents, applying the necessary cleaning steps to ensure that the resulting corpus is suitable for training word embeddings and performing analysis, while also generating statistics and visualizations to understand the dataset better.
def load_and_preprocess(raw_filepath, clean_filepath):
    all_tokens_for_stats = []
    cleaned_sentences = []
    
    documents_count = 0
    with open(raw_filepath, "r", encoding="utf-8") as f:
        for line in f:
            try:
                doc = json.loads(line.strip())
                documents_count += 1
                
                text = doc.get("text", "")
                tokens = preprocess_text(text)
                if tokens:
                    cleaned_sentences.append(tokens)
                    all_tokens_for_stats.extend(tokens)
            except json.JSONDecodeError:
                continue
                
    # Save fully clean corpus
    with open(clean_filepath, "w", encoding="utf-8") as cf:
        for sent in cleaned_sentences:
            cf.write(" ".join(sent) + "\n")
            
    print(f"Cleaned corpus successfully created at {clean_filepath}")
            
    # Visualize and output required assignment stats
    generate_statistics_and_cloud(all_tokens_for_stats, documents_count)
    
    return cleaned_sentences

if __name__ == "__main__":
    if not os.path.exists("data"):
        os.makedirs("data")
        
    RAW_FILE = "data/iitj_raw_documents.jsonl"
    CLEAN_FILE = "data/iitj_clean_corpus.txt"
    
    #  fallback if corpus doesn't exist
    if not os.path.exists(RAW_FILE):
        print(f"Could not find {RAW_FILE}, please run crawler.py first.")
    else:
        load_and_preprocess(RAW_FILE, CLEAN_FILE)
