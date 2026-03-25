# Problem 1: Learning Word Embeddings from IIT Jodhpur Data

## Overview

This project trains Word2Vec models (Skip-gram and CBOW) on textual data collected from IIT Jodhpur institutional sources. It includes hyperparameter experimentation, semantic analysis, and visualization of learned embeddings.

## Dataset

- **Sources**: IIT Jodhpur website, academic regulations, faculty profiles, course syllabi
- **Total Documents**: 200
- **Total Tokens**: 71,544
- **Vocabulary Size**: 6,628 unique words

## Project Structure

```
Problem1/
├── crawler.py                 # Web scraping script to collect data
├── preprocess.py             # Data preprocessing and cleaning
├── word2vec_scratch.py       # Word2Vec training (Skip-gram & CBOW)
├── analysis_and_vis.py       # Semantic analysis and visualization
├── data/                     # Collected raw and processed data
├── models/                   # Trained model checkpoints
├── visualizations/           # Generated plots and visualizations
└── requirements.txt          # Python dependencies
```

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Data Collection
```bash
python crawler.py
```
Collects textual data from IIT Jodhpur sources and saves to `data/raw_corpus.txt`.

### Step 2: Data Preprocessing
```bash
python preprocess.py
```
Cleans corpus, removes boilerplate, performs tokenization/lowercasing. Output: `data/iitj_clean_corpus.txt`

### Step 3: Model Training
```bash
python word2vec_scratch.py
```
Trains Skip-gram and CBOW models with hyperparameter experiments:
- Embedding dimensions: 50, 100, 150
- Context windows: 2, 4, 6
- Negative samples: 3, 5, 10

Generated models saved to `models/`

### Step 4: Semantic Analysis & Visualization
```bash
python analysis_and_vis.py
```
Performs:
- Nearest neighbor analysis (top-5 for key words)
- Analogy experiments (UG:BTech::PG:?)
- t-SNE dimensionality reduction
- Visualization generation

Outputs saved to `visualizations/`

## Key Results

### Dataset Statistics
- Documents: 200
- Tokens: 71,544
- Vocabulary: 6,628 words
- Most frequent: student, program, course, degree, research

### Model Performance
**Skip-gram** (nicely performing config):
- Embedding Dim: 100
- Window Size: 6
- Negative Samples: 10
- Final Loss: 3.6045

**CBOW** (nicely performing config):
- Final Loss: 5.6747
- Stable across hyperparameters
- Computationally efficient

### Semantic Quality
- Skip-gram: Better semantic coherence, tight domain-specific clusters
- CBOW: Strong co-occurrence patterns, good on frequent words
- Both models successfully captured academic terminology

## Visualizations

- `wordcloud.png` - Most frequent terms
- `*_tsne.png` - t-SNE projections for Skip-gram and CBOW

## Key Findings

1. **Skip-gram advantages**: Better for semantic tasks, responsive to hyperparameters, effective on rare words
2. **CBOW advantages**: Faster training, stable performance, strong co-occurrence capture
3. **Optimal config**: Dim=100, Window=6, Neg=10
4. **Domain learning**: Models successfully learned institutional academic structures

## Hyperparameter Experiments

| Configuration | Embedding Dim | Window Size | Neg Samples | SG Loss | CBOW Loss |
|---------------|---------------|-------------|-------------|---------|-----------|
| Best Skip-gram| 100 | 6 | 10 | 3.6045 | 5.6747 |

Complete results in main report.

## Files Generated

- `data/iitj_clean_corpus.txt` - Cleaned corpus
- `models/*.pkl` - Trained embeddings
- `visualizations/` - All plots and visualizations

## Notes

- Preprocessing removes HTML, boilerplate, special characters
- English-only text (non-English filtered)
- Negative sampling used for efficiency
- Both models trained for 3 epochs
- Cosine similarity used for nearest neighbor analysis


