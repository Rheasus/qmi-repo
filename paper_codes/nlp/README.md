# Natural Language Processing Benchmark - Quantum-Inspired Optimizers

This directory contains the code for benchmarking quantum-inspired optimization methods against classical baseline optimizers on natural language processing tasks.

## Overview

### Optimizers
- **Classical Baselines:** Adam, AdamW, SGD, RMSprop, Adagrad
- **Quantum-Inspired:** SPSA, QNG, QPSO, COBYLA

### Datasets
- **AG News:** News article classification (4 classes)
- **IMDB:** Sentiment analysis (2 classes)
- **SST-2:** Sentiment analysis from GLUE benchmark (2 classes)

### Models
- **DistilBERT:** Distilled version of BERT
- **RoBERTa:** Robustly Optimized BERT
- **SimpleLSTM:** 2-layer bidirectional LSTM

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Training

```bash
python trainer.py \
    --dataset ag_news \
    --model distilbert \
    --optimizer adam \
    --seed 42 \
    --epochs 3
```

### Full Benchmark

```bash
python run_benchmark.py \
    --datasets ag_news imdb sst2 \
    --models distilbert roberta lstm \
    --optimizers adam adamw spsa qng \
    --seeds 42 1337 2025 \
    --epochs 3
```

## Command-Line Arguments

**Required:**
- `--dataset`: ag_news, imdb, sst2
- `--model`: distilbert, roberta, lstm
- `--optimizer`: adam, adamw, sgd, rmsprop, adagrad, spsa, qng, qpso, cobyla
- `--seed`: Random seed

**Optional:**
- `--epochs`: Number of epochs (default: 3)
- `--lr`: Learning rate (default: 1e-4)
- `--batch_size`: Batch size (default: 16)
- `--max_length`: Max sequence length (default: 256)

## Code Structure

```
nlp/
├── trainer.py          # Main training script
├── models.py           # Model implementations
├── optimizers.py       # Optimizer implementations
├── utils.py            # Utility functions
├── run_benchmark.py    # Batch execution script
├── README.md
└── requirements.txt
```

## Author

**Gorkem Yilmaz**  
University of Sussex  
gy74@sussex.ac.uk
