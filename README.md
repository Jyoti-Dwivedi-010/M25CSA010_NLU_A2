# CSL 7640: Natural Language Understanding  
## Assignment 2 — Word Embeddings and RNN-based Name Generation  

**Name:** Dwivedi Jyoti Rajeshbhai  
**Roll No:** M25CSA010  

---

## Overview

This project implements core Natural Language Processing models **from scratch** for two tasks:

1. **Word Embedding Learning** using CBOW and Skip-gram on IIT Jodhpur textual data
2. **Character-level Name Generation** using recurrent neural network architectures

The focus of the assignment is to understand model behavior, hyperparameter sensitivity, and qualitative vs quantitative evaluation in NLP systems.

---

# Problem 1 — Word Embeddings

## Dataset

Text data was collected from IIT Jodhpur academic sources including:

- Official website pages
- Academic regulations
- Faculty profiles
- Course syllabi

After preprocessing:

- Documents: **200**
- Tokens: **71,544**
- Vocabulary Size: **6,628**
---

## Models Implemented

Both models were implemented **from scratch**:

- CBOW (Continuous Bag of Words)
- Skip-gram with Negative Sampling
---

## Hyperparameters Explored

- Embedding Dimension: 50, 100, 150  
- Context Window Size: 2, 4, 6  
- Negative Samples: 3, 5, 10  

Total:
9 experimental configurations.

---

## Key Findings

- Skip-gram produced tighter semantic clusters
- CBOW trained faster but showed weaker semantic precision
- Increasing negative samples significantly improved performance
- Best configuration: Embedding Dimension = 100, Window Size = 6, Negative Samples = 10


Final losses: Skip-gram: 3.60, CBOW: 5.67

---

## Visualizations

The repository includes:

- Word cloud of corpus
- PCA / t-SNE embedding projections
- Nearest neighbor analysis
- Analogy evaluation

---

# Problem 2 — Character-Level Name Generation

## Dataset

A dataset of: 1000 Indian names was generated and used to train sequence models.

---

## Models Implemented

All models were implemented from scratch:

1. Vanilla RNN  
2. Bidirectional LSTM (BLSTM)  
3. RNN with Attention  

---

## Model Configuration


- Hidden Size: 256
Layers: 1
Training Epochs: 10
Loss Function: Cross Entropy


- Trainable parameters:
Vanilla RNN: 13,738
BLSTM: 74,666
RNN + Attention: 22,058


---

## Evaluation Metrics

Two metrics were used:

- Novelty Rate
- Diversity

Results:


-Vanilla RNN:
Diversity: 93%
Novelty: 93.55%

-BLSTM:
Diversity: 100%
Novelty: 100%

-RNN + Attention:
Diversity: 93%
Novelty: 88.17%

---

## Key Observations

- BLSTM overfitted despite perfect quantitative scores
- Vanilla RNN produced the most realistic names
- Attention improved generalization
- Qualitative evaluation is necessary alongside metrics

---

# Repository Structure

M25CSA010_NLU_A2/
│
├── Problem1/
│   ├── crawler.py              # Web scraping script to collect IITJ text data
│   ├── preprocess.py           # Data preprocessing and corpus cleaning
│   ├── word2vec_scratch.py     # Word2Vec implementation (CBOW & Skip-gram)
│   ├── analysis_and_vis.py     # Semantic analysis and visualization
│   │
│   ├── data/
│   │   └── iitj_clean_corpus.txt   # Processed text corpus
│   │   └── iitj_raw_documents.jsonl   # raw text corpus
│   │
│   ├── models/
│   │   └── saved embeddings / checkpoints
│   │
│   ├── visualizations/
│   │   ├── cbow_tsne.png
│   │   ├── skipgram_tsne.png
│   │   └── wordcloud.png
│   │
│   └── requirements.txt
│   └── README.md
│
├── Problem2/
│   ├── models.py               # RNN, BLSTM, and Attention model definitions
│   ├── train_evaluate.py       # Training loop and evaluation metrics
│   ├── patch.py                # Utility / helper functions
│   │
│   ├── checkpoints/
│   │   ├── VanillaRNN.pt
│   │   ├── BLSTM.pt
│   │   └── RNN_Attention.pt
│   │
│   ├── names_list.txt          # Training dataset (1000 Indian names)
│   ├── loss_curves.png         # Training loss visualization
│   └── requirements.txt
│   └── README.md
│
├── report.pdf                  # Final assignment report
├── README.md
├── images                      # results images
└── .gitignore
- report.pdf
- images folder

---

# How to Run

## Problem 1 — Word Embeddings
( see the problem1 folder's readme file)

- python crawler.py
- python preprocess.py
- python word2vec_scratch.py
- python analysis_and_vis.py


## Problem 2 — Name Generation
( see the problem2 folder's readme file )
- python train_evaluate.py
---

# Technologies Used


- Python
- NumPy
- PyTorch
- Matplotlib
- scikit-learn
- BeautifulSoup
