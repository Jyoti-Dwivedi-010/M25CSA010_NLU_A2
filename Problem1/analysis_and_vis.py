import os
import numpy as np
import collections
from gensim.models import Word2Vec
import pickle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# to disable excessive warnings
import warnings
warnings.filterwarnings("ignore")
#to maintain reproducible results
seed = 42
#comparator class to train gensim models and compare with scratch models
class GensimComparator:
    def __init__(self, corpus_path):
        self.corpus_path = corpus_path
        self.sentences = []
        self.load_corpus()
# Load the corpus and prepare sentences for training
    def load_corpus(self):
        with open(self.corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                tokens = line.strip().split()
                if tokens:
                    self.sentences.append(tokens)
# Train Gensim CBOW model
    def train_gensim_cbow(self):
        print("Training Gensim CBOW...")
        model = Word2Vec(sentences=self.sentences, vector_size=50, window=3, min_count=1, sg=0, negative=5, workers=4, epochs=5)
        model.save("models/gensim_cbow.model")
        return model
# Train Gensim Skip-Gram model
    def train_gensim_sg(self):
        print("Training Gensim Skip-Gram...")
        model = Word2Vec(sentences=self.sentences, vector_size=50, window=3, min_count=1, sg=1, negative=5, workers=4, epochs=5)
        model.save("models/gensim_sg.model")
        return model

# Evaluator class to load from-scratch models and perform similarity and analogy tasks
class ScratchModelEvaluator:
    def __init__(self, model_path):
        with open(model_path, 'rb') as f:
            state = pickle.load(f)
            self.W1 = state['w1']
            self.word_to_id = state['word2id']
            self.id_to_word = state['id2word']
            self.vocab_size = len(self.id_to_word)
# Get the embedding vector for a given word funciton
    def get_vector(self, word):
        if word in self.word_to_id:
            idx = self.word_to_id[word]
            return self.W1[idx]
        return None
# Calculate cosine similarity between two vectors
    def cosine_similarity(self, v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
# finding the most similar words to a given word based on cosine similarity of their embeddings
    def most_similar(self, word, topn=5):
        if word not in self.word_to_id:
            print(f"Word '{word}' not in vocabulary.")
            return []
        
        target_vec = self.get_vector(word)
        similarities = []
        for w, idx in self.word_to_id.items():
            if w == word: continue
            sim = self.cosine_similarity(target_vec, self.W1[idx])
            similarities.append((w, sim))
            
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:topn]

#   analogy function to find the word that best completes the analogy w1 : w2 :: w3 : ?
    def analogy(self, w1, w2, w3, topn=1):
        
        for w in [w1, w2, w3]:
            if w not in self.word_to_id:
                print(f"Word '{w}' not in vocabulary.")
                return None
                
        target_vec = self.get_vector(w2) - self.get_vector(w1) + self.get_vector(w3)
        similarities = []
        
        for w, idx in self.word_to_id.items():
            if w in [w1, w2, w3]: continue
            sim = self.cosine_similarity(target_vec, self.W1[idx])
            similarities.append((w, sim))
            
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:topn]
# function to visualize embeddings using PCA or t-SNE plots
def visualize_embeddings(embeddings_dict, title, filename, use_tsne=True):
   
    words = list(embeddings_dict.keys())
    vectors = np.array(list(embeddings_dict.values()))
    
    if len(words) < 2: return
    
    # Use PCA or TSNE for dimensionality reduction based on the number of words
    reducer = PCA(n_components=2)
    if use_tsne and len(words) > 5:
        reducer = TSNE(n_components=2, perplexity=min(30, len(words)-1), random_state=seed)
        
    reduced_vectors = reducer.fit_transform(vectors)
    
    plt.figure(figsize=(10, 8))
    for i, word in enumerate(words):
        plt.scatter(reduced_vectors[i, 0], reduced_vectors[i, 1])
        plt.annotate(word, (reduced_vectors[i, 0], reduced_vectors[i, 1]), alpha=0.7)
        
    plt.title(title)
    plt.savefig(filename)
    print(f"Saved visualization to {filename}")

if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg") # Non-interactive backend
    
    os.makedirs("models", exist_ok=True)
    os.makedirs("visualizations", exist_ok=True)
    
    # 1. Comparative analysis using Gensim
    gensim_comp = GensimComparator("data/iitj_clean_corpus.txt")
    if gensim_comp.sentences:
        g_cbow = gensim_comp.train_gensim_cbow()
        g_sg = gensim_comp.train_gensim_sg()
    else:
        print("Warning: Corpus is empty. Skipping Gensim Training.")
        
    # 2. Evaluate from-scratch models
    try:
        scratch_sg = ScratchModelEvaluator("models/skipgram_model.pkl")
        scratch_cbow = ScratchModelEvaluator("models/cbow_model.pkl")
        
        target_words = ["research", "student", "phd", "exam"]
        print("\n--- Semantic Analysis: Nearest Neighbors ---")
        for word in target_words:
            print(f"\nTarget Word: '{word}'")
            
            print("  Skip-Gram:")
            res_sg = scratch_sg.most_similar(word)
            for r in res_sg: print(f"    {r[0]}: {r[1]:.4f}")
            
            print("  CBOW:")
            res_cbow = scratch_cbow.most_similar(word)
            for r in res_cbow: print(f"    {r[0]}: {r[1]:.4f}")
            
        print("\n--- Semantic Analysis: Analogies ---")
        analogy_combinations = [
            ("ug", "btech", "pg"),
            ("faculty", "teaching", "student"),
            ("research", "phd", "course")
        ]
        
        for w1, w2, w3 in analogy_combinations:
            print(f"\nAnalogy: {w1} : {w2} :: {w3} : ?")
            
            # SkipGram Analogy
            analogy_res_sg = scratch_sg.analogy(w1, w2, w3, topn=3)
            if analogy_res_sg:
                formatted_res_sg = ", ".join([f"'{word}' ({score:.4f})" for word, score in analogy_res_sg])
                print(f"  Skip-Gram  ->  {formatted_res_sg}")
            else:
                print(f"  Skip-Gram  ->  [Words not in vocab]")
                
            # CBOW Analogy
            analogy_res_cbow = scratch_cbow.analogy(w1, w2, w3, topn=3)
            if analogy_res_cbow:
                formatted_res_cbow = ", ".join([f"'{word}' ({score:.4f})" for word, score in analogy_res_cbow])
                print(f"  CBOW       ->  {formatted_res_cbow}")
            else:
                print(f"  CBOW       ->  [Words not in vocab]")
        
        # 3. Visualization
        # Select common academic words for visualization
        vocab_subset = ["research", "student", "phd", "exam", "course", "thesis", "faculty", "btech", "mtech", "ug", "pg", "science", "engineering"]
        
        embeddings_to_plot_sg = {w: scratch_sg.get_vector(w) for w in vocab_subset if scratch_sg.get_vector(w) is not None}
        visualize_embeddings(embeddings_to_plot_sg, "Skip-gram (Scratch) Embeddings PCA/t-SNE", "visualizations/skipgram_tsne.png")
        
        embeddings_to_plot_cbow = {w: scratch_cbow.get_vector(w) for w in vocab_subset if scratch_cbow.get_vector(w) is not None}
        visualize_embeddings(embeddings_to_plot_cbow, "CBOW (Scratch) Embeddings PCA/t-SNE", "visualizations/cbow_tsne.png")

    except FileNotFoundError:
        print("Scratch models not found. Please run word2vec_scratch.py first.")
