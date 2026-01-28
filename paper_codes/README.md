# Benchmarking Quantum-Inspired Optimization Methods

This repository contains the complete implementation code for the paper:

**"Benchmarking Quantum-Inspired Optimization Methods: A Critical Evaluation on Computer Vision, Natural Language Processing, and Tabular Data Tasks"**

Author: **Gorkem Yilmaz**  
Institution: University of Sussex  
Contact: gy74@sussex.ac.uk

---

## Overview

This repository provides clean, modular, and reproducible code for benchmarking quantum-inspired optimization methods (SPSA, QNG, QPSO, COBYLA) against classical baseline optimizers (Adam, AdamW, SGD, etc.) across three domains:

1. **Computer Vision (CV):** CIFAR-10, CIFAR-100, Fashion-MNIST with SimpleCNN and ResNet18
2. **Natural Language Processing (NLP):** AG News, IMDB, SST-2 with DistilBERT, RoBERTa, and SimpleLSTM
3. **Tabular Data:** Breast Cancer, Wine Quality with MLP and tree-based models (XGBoost, LightGBM, CatBoost)

---

## Repository Structure

```
paper_codes/
├── cv/                      # Computer Vision experiments
│   ├── trainer.py           # Main training script
│   ├── models.py            # SimpleCNN and ResNet18 models
│   ├── optimizers.py        # Optimizer implementations
│   ├── utils.py             # Utility functions
│   ├── run_benchmark.py     # Batch execution script
│   ├── README.md
│   └── requirements.txt
│
├── nlp/                     # Natural Language Processing experiments
│   ├── trainer.py           # Main training script
│   ├── models.py            # DistilBERT, RoBERTa, SimpleLSTM models
│   ├── optimizers.py        # Optimizer implementations
│   ├── utils.py             # Utility functions
│   ├── run_benchmark.py     # Batch execution script
│   ├── README.md
│   └── requirements.txt
│
├── tabular/                 # Tabular data experiments
│   ├── trainer.py           # Main training script
│   ├── models.py            # MLP and tree-based models
│   ├── optimizers.py        # Optimizer implementations
│   ├── utils.py             # Utility functions
│   ├── README.md
│   └── requirements.txt
│
└── README.md                # This file
```

---

## Quick Start

### 1. Installation

Clone the repository and install dependencies for each domain:

```bash
# Computer Vision
cd paper_codes/cv
pip install -r requirements.txt

# Natural Language Processing
cd ../nlp
pip install -r requirements.txt

# Tabular Data
cd ../tabular
pip install -r requirements.txt
```

### 2. Run Single Experiment

#### Computer Vision

```bash
cd paper_codes/cv
python trainer.py \
    --dataset cifar10 \
    --model SimpleCNN \
    --optimizer Adam \
    --seed 42 \
    --epochs 50
```

#### Natural Language Processing

```bash
cd paper_codes/nlp
python trainer.py \
    --dataset ag_news \
    --model distilbert \
    --optimizer adam \
    --seed 42 \
    --epochs 3
```

#### Tabular Data

```bash
cd paper_codes/tabular
python trainer.py \
    --dataset breast_cancer \
    --model mlp \
    --optimizer adam \
    --seed 42
```

### 3. Run Full Benchmark

Each domain includes a `run_benchmark.py` script for batch execution:

```bash
# Computer Vision
cd paper_codes/cv
python run_benchmark.py \
    --datasets cifar10 cifar100 fashion_mnist \
    --models SimpleCNN ResNet18 \
    --optimizers SGD Adam AdamW SPSA QNG \
    --seeds 42 1337 2025 \
    --epochs 50

# Natural Language Processing
cd paper_codes/nlp
python run_benchmark.py \
    --datasets ag_news imdb sst2 \
    --models distilbert roberta lstm \
    --optimizers adam adamw spsa qng \
    --seeds 42 1337 2025 \
    --epochs 3
```

---

## Experimental Setup

### Optimizers

**Classical Baselines:**
- **SGD:** Stochastic Gradient Descent with momentum (0.9)
- **Adam:** Adaptive Moment Estimation
- **AdamW:** Adam with weight decay decoupling (0.01)
- **RMSprop:** Root Mean Square Propagation
- **Adagrad:** Adaptive Gradient Algorithm

**Quantum-Inspired:**
- **SPSA:** Simultaneous Perturbation Stochastic Approximation (gradient-free)
- **QNG:** Quantum Natural Gradient
- **QPSO:** Quantum Particle Swarm Optimization
- **COBYLA:** Constrained Optimization BY Linear Approximations

### Datasets

#### Computer Vision
- **CIFAR-10:** 60,000 32×32 RGB images, 10 classes
- **CIFAR-100:** 60,000 32×32 RGB images, 100 classes
- **Fashion-MNIST:** 70,000 28×28 grayscale images, 10 classes

#### Natural Language Processing
- **AG News:** News article classification, 4 classes
- **IMDB:** Movie review sentiment analysis, 2 classes
- **SST-2:** Stanford Sentiment Treebank, 2 classes

#### Tabular Data
- **Breast Cancer:** Binary classification, 569 samples, 30 features
- **Wine Quality:** Multi-class classification, 1599 samples, 11 features

### Models

#### Computer Vision
- **SimpleCNN:** 3-layer CNN (32→64→128 channels) + 2 FC layers
- **ResNet18:** 18-layer residual network (~11.2M parameters)

#### Natural Language Processing
- **DistilBERT:** Distilled BERT (66M parameters)
- **RoBERTa:** Robustly Optimized BERT (125M parameters)
- **SimpleLSTM:** 2-layer bidirectional LSTM (embedding_dim=128, hidden_dim=256)

#### Tabular Data
- **MLP:** 2-layer perceptron (128→64 hidden units)
- **XGBoost:** Gradient boosting (100 estimators)
- **LightGBM:** Gradient boosting (100 estimators)
- **CatBoost:** Gradient boosting (100 iterations)

---

## Hyperparameters

### Computer Vision
- **Epochs:** 50
- **Batch Size:** 32
- **Learning Rate:** 0.001
- **Seeds:** 42, 1337, 2025

### Natural Language Processing
- **Epochs:** 3
- **Batch Size:** 16
- **Learning Rate:** 1e-4
- **Max Sequence Length:** 256
- **Seeds:** 42, 1337, 2025

### Tabular Data
- **Epochs (MLP):** 100
- **Batch Size (MLP):** 32
- **Learning Rate (MLP):** 0.001
- **Seeds:** 42, 1337, 2025

---

## Output Structure

Each experiment creates a results directory containing:

```
results/{run_id}/
├── run_summary.json              # Configuration and final metrics
├── history.csv                   # Epoch-wise training history
├── loss_curve.png                # Training/validation curves
├── model.pth                     # Trained model weights
├── classification_report.json    # Per-class metrics
├── confusion_matrix.csv          # Confusion matrix
└── confusion_matrix.json         # Confusion matrix (JSON)
```

---

## Hardware Requirements

- **GPU:** NVIDIA T4 or equivalent (16GB VRAM) for CV and NLP
- **CPU:** 4+ cores
- **RAM:** 16GB recommended
- **Storage:** ~50GB for datasets and results

---

## Reproducibility

All experiments use fixed random seeds (42, 1337, 2025) for reproducibility:

```python
import random
import numpy as np
import torch

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{yilmaz2025quantum,
  title={Benchmarking Quantum-Inspired Optimization Methods: 
         A Critical Evaluation on Computer Vision, Natural Language Processing, 
         and Tabular Data Tasks},
  author={Yilmaz, Gorkem},
  journal={[Journal Name]},
  year={2025},
  institution={University of Sussex}
}
```

---

## License

This code is provided for research purposes. Please see LICENSE file for details.

---

## Contact

For questions or issues, please contact:

**Gorkem Yilmaz**  
University of Sussex  
Email: gy74@sussex.ac.uk

---

## Acknowledgments

This research was conducted at the University of Sussex. All experiments were run on NVIDIA T4 GPUs.
