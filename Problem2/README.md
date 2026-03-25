# Problem 2: Character-Level Name Generation Using RNN Variants

## Overview

This project implements and compares three RNN architectures for character-level Indian name generation:
- Vanilla RNN
- Bidirectional LSTM (BLSTM)
- RNN with Attention Mechanism

## Dataset

- **Size**: 1000 Indian names (synthetically generated)
- **Format**: Character-level sequences
- **File**: `names_list.txt`
- **Characteristics**: Multi-regional diversity, variable length (3-15 characters), mixed gender

## Project Structure

```
Problem2/
├── models.py              # RNN, BLSTM, and Attention model definitions
├── train_evaluate.py        # Training loop and evaluation metrics
├── patch.py  
├── checkpoints/             # Saved model weights
├── names_list.txt          # Training dataset
├── loss_curves.png         # Training progression visualization
└── requirements.txt        # Python dependencies
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

### Training and Evaluation
```bash
python train_evaluate.py
```

This script:
1. Trains all three models for 10 epochs
2. Generates 100 names from each trained model
3. Computes quantitative metrics (diversity, novelty)
4. Saves trained models to `checkpoints/`
5. Produces `loss_curves.png` visualization
6. Prints evaluation report

## Models

### Vanilla RNN
- **Architecture**: Single-layer RNN with tanh activation
- **Parameters**: 13,738
- **Hidden Size**: 256
- **Embedding Dim**: 128
- **Performance**: 93% diversity, 93.55% novelty
- **Quality**: Good phonetics, reliable generalization

### Bidirectional LSTM (BLSTM)
- **Architecture**: Bidirectional LSTM processing sequences both directions
- **Parameters**: 74,666
- **Hidden Size**: 256 (per direction)
- **Embedding Dim**: 128
- **Performance**: 100% diversity, 100% novelty
- **Quality**: POOR - severe overfitting, gibberish output
- **Status**:  - metrics are artificial, output is unusable

### RNN with Attention
- **Architecture**: Vanilla RNN + scaled dot-product attention
- **Parameters**: 22,058
- **Hidden Size**: 256
- **Attention Dim**: 128
- **Embedding Dim**: 128
- **Performance**: 93% diversity, 88.17% novelty
- **Quality**: EXCELLENT - realistic, phonetically plausible names
- **Status**:  **RECOMMENDED** - best practical results

## Results

### Quantitative Metrics

| Model | Diversity | Novelty | Status |
|-------|-----------|---------|--------|
| Vanilla RNN | 93.00% | 93.55% | Best  |
| BLSTM | 100.00% | 100.00% |  Overfitted (reject) |
| RNN + Attention | 93.00% | 88.17% |  Second Best choice |

### Sample Outputs

**Vanilla RNN** (Good):
```
Krishna Pandey
Nikhetha
Arjun Sidhtu
Priya Verma
Karan Pandey
```

**BLSTM** (Poor - Overfitted):
```
AKral A Agang AdIyJ      [gibberish]
Agjit AgaxAng Aggta      [character repetition]
AabAma Arya ASvit Ag     [unnatural patterns]
```

**RNN + Attention** (Excellent):
```
Kajri
Senanya Mukherjee
Sindha
Arjun Verma
Reyansh
```

## Training Dynamics

- **BLSTM**: Loss 2.08 → 0.0042 (rapid convergence indicates overfitting)
- **Vanilla RNN**: Loss 2.64 → 0.64 (steady, realistic convergence)
- **Attention-RNN**: Loss 2.57 → 0.64 (comparable to vanilla RNN)

## Key Findings

###  What Works
- **RNN + Attention** produces realistic, phonetically plausible names
- Attention mechanism prevents overfitting while maintaining good metrics
- Vanilla RNN provides reliable baseline with genuine generalization

###  What Doesn't Work
- **BLSTM** severely overfits despite perfect metrics
- Bidirectional processing inappropriate for unidirectional generation
- Quantitative metrics alone insufficient for quality assessment

### Critical Insights
1. Perfect diversity/novelty scores ≠ good output quality
2. Human qualitative assessment is essential
3. Simpler models with attention > complex models without regularization
4. Architecture appropriateness matters more than complexity

## Failure Modes

| Mode | Frequency | Severity |
|------|-----------|----------|
| Character Repetition | BLSTM | Critical |
| Gibberish Sequences | BLSTM | Critical |
| Truncation | Vanilla RNN, Attention | Minor |
| Merged Names | Vanilla RNN | Minor |

## Evaluation Metrics Explained

**Diversity Rate**: % of unique names in 100 generated outputs
- High diversity = model not stuck in repetition loops
- 100% = all names unique (BLSTM metric gaming)

**Novelty Rate**: % of generated names not in training set
- Measures generalization vs memorization
- 100% = no names copied from training (not necessarily good!)

**Critical**: Both metrics can be gamed by models that generate random/garbage sequences. Must combine with qualitative assessment.

## Hyperparameters

- Learning Rate: 0.001 (Adam optimizer)
- Epochs: 10
- Batch Size: 32
- Loss Function: Cross-entropy
- Sequence Length: Variable (max 20 chars)

## Generated Files

- `checkpoints/VanillaRNN.pt` - Vanilla RNN weights
- `checkpoints/BLSTM.pt` - BLSTM weights
- `checkpoints/RNN_Attention.pt` - RNN + Attention weights
- `loss_curves.png` - Training visualization


## Common Issues

### BLSTM Producing Gibberish?
This is expected. BLSTM overfits to the training data.

### Repetitive Output?
- Vanilla RNN shows minimal repetition (93% diversity) - this is expected
- If using BLSTM, abandon it entirely

## Files Explained

- `models.py` - PyTorch model definitions with forward/backward passes
- `train_evaluate.py` - Training loop, generation, and metric computation
- `names_list.txt` - Raw training data (1000 Indian names)
- `checkpoints/` - Saved model weights for inference

