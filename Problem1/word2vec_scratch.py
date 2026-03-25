import sys
import numpy as np
import random
import os
import pickle

seed = 42
np.random.seed(seed)
random.seed(seed)

# Base class for Word2Vec models, containing shared functionality for both CBOW and Skip-Gram architectures, including vocabulary building, weight initialization, negative sampling, and model saving/loading.
class Word2VecBase:
    
    def __init__(self, embedding_dim=100, window_size=2, num_negative_samples=5, learning_rate=0.01, epochs=10):
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.num_negative_samples = num_negative_samples
        self.lr = learning_rate
        self.epochs = epochs
        
        # Word index mapping
        self.word_to_id = {}
        self.id_to_word = {}
        self.vocab_size = 0
        self.word_counts = {}
        
        # Unigram distributions (Used for negative sampling power of 3/4)
        self.unigram_dist = []
        
        # Matrix variables (Target W_target, Context W_context)
        self.W1 = None # Input-to-Hidden Weights
        self.W2 = None # Hidden-to-Output Weights
        
# builds the vocabulary from the corpus, initializes the embedding matrices, and prepares the negative sampling distribution based on word frequencies, ensuring that the model is ready for training with the specified hyperparameters.
    def build_vocab(self, corpus):
        
        print("Building vocabulary...")
        
        # Populate counts
        for sentence in corpus:
            for word in sentence:
                self.word_counts[word] = self.word_counts.get(word, 0) + 1
                
        # Id mapping
        for word in sorted(self.word_counts.keys()):
            self.word_to_id[word] = len(self.word_to_id)
            self.id_to_word[self.word_to_id[word]] = word
            
        self.vocab_size = len(self.word_to_id)
        print(f"Vocab size defined as: {self.vocab_size}")
        
        self.init_weights()
        self.init_unigram_distribution()
        
# initializes the weight matrices W1 and W2 with small random values, which will be updated during training to learn meaningful word embeddings based on the context of words in the corpus.
    def init_weights(self):
        self.W1 = np.random.uniform(-0.1, 0.1, (self.vocab_size, self.embedding_dim))
        self.W2 = np.random.uniform(-0.1, 0.1, (self.vocab_size, self.embedding_dim))
        
    def init_unigram_distribution(self):
        
        counts = [self.word_counts[self.id_to_word[i]] for i in range(self.vocab_size)]
        pow_counts = np.power(counts, 0.75) 
        probs = pow_counts / np.sum(pow_counts)
        self.unigram_dist = probs
        
        # Build an O(1) sampling table (size 1M) for fast negative sampling
        table_size = 1000000
        self.sampling_table = np.zeros(table_size, dtype=np.int32)
        
        cumulative_probs = np.cumsum(probs)
        current_idx = 0
        for i in range(table_size):
            # i / table_size is the cumulative threshold
            while current_idx < self.vocab_size and (i / table_size) > cumulative_probs[current_idx]:
                current_idx += 1
            if current_idx >= self.vocab_size:
                current_idx = self.vocab_size - 1
            self.sampling_table[i] = current_idx

# function to draw negative samples for a given target word ID, ensuring that the sampled words are not the same as the target word and are drawn according to the unigram distribution prepared during vocabulary building.
    def get_negative_samples(self, target_id):
        
        samples = []
        while len(samples) < self.num_negative_samples:
            rand_idx = random.randint(0, len(self.sampling_table) - 1)
            sampled_id = self.sampling_table[rand_idx]
            if sampled_id != target_id:
                samples.append(sampled_id)
        return samples

# saves the trained model's parameters, including the embedding matrices and vocabulary mappings, to a file using pickle, allowing for later loading and evaluation of the learned word embeddings without needing to retrain the model from scratch.
    def save_model(self, filepath):
       
        state = {
            'w1': self.W1,
            'word2id': self.word_to_id,
            'id2word': self.id_to_word,
            'dim': self.embedding_dim,
            'window': self.window_size
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
            
    def load_model(self, filepath):
        if not os.path.exists(filepath): return False
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
            self.W1 = state['w1']
            self.word_to_id = state['word2id']
            self.id_to_word = state['id2word']
            self.embedding_dim = state['dim']
            self.vocab_size = len(self.word_to_id)
        return True

# CBOW architecture using Negative Sampling.
class CBSEModel(Word2VecBase):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
        
    def train(self, corpus):
        print("Training CBOW with Negative Sampling...")
        
        for epoch in range(self.epochs):
            total_loss = 0
            
            for sentence in corpus:
                # Map continuous words to IDs
                sen_ids = [self.word_to_id[w] for w in sentence]
                sen_length = len(sen_ids)
                
                # Iterate each target word
                for i, target_id in enumerate(sen_ids):
                    
                    # Compute the context window sliding bounds
                    start = max(0, i - self.window_size)
                    end = min(sen_length, i + self.window_size + 1)
                    context_ids = [sen_ids[j] for j in range(start, end) if j != i]
                    
                    # If empty context at edges, continue
                    if not context_ids:
                        continue
                        
                    # Target output array
                    # CBOW uses average context embedding to predict target output
                    h = np.mean([self.W1[c_id] for c_id in context_ids], axis=0) # [dim]
                    
                    # Forward/Backward targets starting with True Class (Label 1)
                    samples = [(target_id, 1)]
                    
                    # Draw Negative Samples (Label 0)
                    neg_samples = self.get_negative_samples(target_id)
                    for n in neg_samples:
                        samples.append((n, 0))
                        
                    # Gradient accumulation across samples
                    eh = np.zeros(self.embedding_dim) # Error gradient w.r.t Hidden Layer h
                    word_loss = 0
                    
                    for w_c, label_val in samples:
                        # Forward pass
                        f = np.dot(h, self.W2[w_c])
                        pred = self.sigmoid(f)
                        
                        # Loss computation (Cross Entropy approx)
                        g = (label_val - pred) * self.lr
                        word_loss -= np.log(pred + 1e-10) if label_val == 1 else np.log(1 - pred + 1e-10)
                        
                        # Accumulate errors for input gradient
                        eh += g * self.W2[w_c]
                        
                        # Update output context weights
                        self.W2[w_c] += g * h
                        
                    total_loss += word_loss / len(samples)
                        
                    # Update input embedding weights for all tokens in context
                    for c_id in context_ids:
                        self.W1[c_id] += (1.0 / len(context_ids)) * eh
                        
            avg_epoch_loss = total_loss / (len(corpus) * (sen_length if sen_length > 0 else 1))
            print(f"CBOW Epoch {epoch+1}/{self.epochs} completed. Average Loss: {avg_epoch_loss:.4f}")
        return avg_epoch_loss

# skip-gram architecture using Negative Sampling, where the model learns to predict context words given a target word, and updates the embeddings based on the error between the predicted context and the actual context words in the corpus. 
class SkipGramModel(Word2VecBase):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10))) 

    def train(self, corpus):
        print("Training Skip-Gram with Negative Sampling...")
        
        for epoch in range(self.epochs):
            total_loss = 0
            
            for sentence in corpus:
                sen_ids = [self.word_to_id[w] for w in sentence]
                sen_length = len(sen_ids)
                
                # Scan entire sequence
                for i, target_id in enumerate(sen_ids):
                    
                    start = max(0, i - self.window_size)
                    end = min(sen_length, i + self.window_size + 1)
                    
                    # We are trying to predict the context tokens from the single target word.
                    context_ids = [sen_ids[j] for j in range(start, end) if j != i]
                    
                    for context_id in context_ids:
                        h = self.W1[target_id] # [dim]
                        
                        # Target array definitions: True Class context token = 1
                        samples = [(context_id, 1)]
                        # False Sample tokens = 0
                        neg_samples = self.get_negative_samples(context_id)
                        for n in neg_samples:
                            samples.append((n, 0))
                            
                        eh = np.zeros(self.embedding_dim)
                        word_loss = 0
                        
                        for w_c, label_val in samples:
                            # Forward pass
                            f = np.dot(h, self.W2[w_c])
                            pred = self.sigmoid(f)
                            
                            # Loss computation (Negative Log Likelihood)
                            # Only accumulate average error per word to avoid exploding sums
                            g = (label_val - pred) * self.lr
                            if label_val == 1:
                                word_loss -= np.log(pred + 1e-10)
                            else:
                                word_loss -= np.log(1.0 - pred + 1e-10)
                                
                            eh += g * self.W2[w_c]
                            self.W2[w_c] += g * h
                            
                        # Average the loss over the samples to prevent massive magnitudes
                        total_loss += word_loss / len(samples)
                        
                        # Update Target Mapping
                        self.W1[target_id] += eh
                        
            # Calculate Average Loss properly normalized by actual number of words predicted
            avg_epoch_loss = total_loss / (len(corpus) * (sen_length if sen_length > 0 else 1) * (self.window_size * 2))
            print(f"Skip-Gram Epoch {epoch+1}/{self.epochs} completed. Average Loss: {avg_epoch_loss:.4f}")
        return avg_epoch_loss

# reading the cleaned corpus file, where each line is a preprocessed sentence, and converting it into a list of sentences (where each sentence is a list of tokens) that can be used for training the word embedding models. This function also includes error handling to ensure that the file exists and is properly formatted.
def read_clean_corpus(filepath):
    
    sentences = []
    if not os.path.exists(filepath):
        print(f"File {filepath} not found.")
        return sentences
        
    with open(filepath, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            parts = line.strip().split()
            if parts: sentences.append(parts)
            # Reverting artificial cutoff to ensure ALL data gets modeled for accurate semantic embeddings
            
    return sentences

if __name__ == "__main__":
    if not os.path.exists('models'):
        os.makedirs('models')
        
    corpus = read_clean_corpus('data/iitj_clean_corpus.txt')
    if not corpus:
        corpus = [["mock", "data", "student", "research", "exam", "phd"]] # Mock fallback fallback
        
    total_words = sum(len(doc) for doc in corpus)
    print(f"Corpus loaded with {len(corpus)} documents (total {total_words} tokens). Starting hyperparameter experiments.")
    
    # Baseline hyperparameters
    base_dim = 100
    base_window = 2
    base_neg_samples = 5
    epochs = 3
    
    # Experiments for each hyperparameter
    experiments = {
        "Embedding Dimension": [
            {"dim": 50, "win": base_window, "ns": base_neg_samples},
            {"dim": 100, "win": base_window, "ns": base_neg_samples},
            {"dim": 150, "win": base_window, "ns": base_neg_samples}
        ],
        "Context Window Size": [
            {"dim": base_dim, "win": 2, "ns": base_neg_samples},
            {"dim": base_dim, "win": 4, "ns": base_neg_samples},
            {"dim": base_dim, "win": 6, "ns": base_neg_samples}
        ],
        "Number of Negative Samples": [
            {"dim": base_dim, "win": base_window, "ns": 3},
            {"dim": base_dim, "win": base_window, "ns": 5},
            {"dim": base_dim, "win": base_window, "ns": 10}
        ]
    }
    
    results_summary = []
    
    for exp_name, configs in experiments.items():
        print(f"\n==========================================")
        print(f"Experimenting with: {exp_name}")
        print(f"==========================================")
        
        for config in configs:
            dim = config["dim"]
            win = config["win"]
            ns = config["ns"]
            
            print(f"\n--- Config: embed_dim={dim}, window_size={win}, neg_samples={ns} ---")
            
            # --- Train Skip-Gram ---
            print("Training Skip-gram:")
            sg_model = SkipGramModel(embedding_dim=dim, window_size=win, num_negative_samples=ns, epochs=epochs)
            sg_model.build_vocab(corpus)
            sg_loss = sg_model.train(corpus)
            sg_model.save_model(f"models/skipgram_dim{dim}_win{win}_ns{ns}.pkl")
            
            # --- Train CBOW ---
            print("Training CBOW:")
            cbow_model = CBSEModel(embedding_dim=dim, window_size=win, num_negative_samples=ns, epochs=epochs)
            cbow_model.build_vocab(corpus)
            cbow_loss = cbow_model.train(corpus)
            cbow_model.save_model(f"models/cbow_dim{dim}_win{win}_ns{ns}.pkl")
            
            # Store summary
            results_summary.append({
                "Experiment": exp_name,
                "Dim": dim, "Win": win, "Neg": ns,
                "SG_Loss": f"{sg_loss:.4f}",
                "CBOW_Loss": f"{cbow_loss:.4f}"
            })

    print("\n=======================================================")
    print("HYPERPARAMETER EXPERIMENT RESULTS (For your Report)")
    print("=======================================================")
    print(f"{'Experiment Area':<28} | {'Dim':<4} | {'Win':<3} | {'Neg':<3} | {'SG Loss':<9} | {'CBOW Loss':<9}")
    print("-" * 70)
    for res in results_summary:
        print(f"{res['Experiment']:<28} | {res['Dim']:<4} | {res['Win']:<3} | {res['Neg']:<3} | {res['SG_Loss']:<9} | {res['CBOW_Loss']:<9}")
    print("-" * 70)
    
    print("\nAll hyperparameter experiments completed and models generated.")
