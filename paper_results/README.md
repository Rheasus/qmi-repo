# Experiment Results - Supplementary Data

**Paper:** Benchmarking Quantum-Inspired Optimization Methods: A Critical Evaluation on Computer Vision, Natural Language Processing, and Tabular Data Tasks

**Author:** Gorkem Yilmaz  
**Institution:** University of Sussex  
**Contact:** gy74@sussex.ac.uk

---

## Overview

This folder contains the complete experimental results supporting the paper. All results are provided in CSV format for transparency and reproducibility.

---

## File Descriptions

### 1. `cv_experiment_results.csv` (Computer Vision)

**Contents:** Complete results for all CV experiments.

| Column | Description |
|--------|-------------|
| `folder_name` | Unique experiment identifier |
| `dataset` | Dataset name: `cifar10`, `cifar100`, `fashion_mnist` |
| `model` | Model architecture: `SimpleCNN`, `ResNet18` |
| `optimizer` | Optimizer: `SGD`, `Adam`, `AdamW`, `SPSA`, `QNG` |
| `seed` | Random seed: 42, 1337, 2025 |
| `learning_rate` | Learning rate (0.001 for all CV experiments) |
| `epochs` | Number of training epochs (50) |
| `batch_size` | Batch size (32) |
| `momentum` | SGD momentum (0.9 for SGD, null otherwise) |
| `weight_decay` | Weight decay (0.01 for AdamW, null otherwise) |
| `spsa_a` | SPSA perturbation scale (0.1 for SPSA) |
| `spsa_c` | SPSA gradient scale (0.01 for SPSA) |
| `accuracy` | Test accuracy |
| `macro_f1` | Macro-averaged F1 score |
| `macro_precision` | Macro-averaged precision |
| `macro_recall` | Macro-averaged recall |
| `weighted_f1` | Weighted F1 score |
| `weighted_precision` | Weighted precision |
| `weighted_recall` | Weighted recall |
| `final_train_loss` | Final training loss |
| `final_val_loss` | Final validation loss |
| `final_val_acc` | Final validation accuracy |
| `best_val_acc` | Best validation accuracy achieved |
| `final_train_acc` | Final training accuracy |
| `training_time` | Training time in seconds |

**Experiment Coverage:**
- Datasets: CIFAR-10, CIFAR-100, Fashion-MNIST
- Models: SimpleCNN (3-layer CNN), ResNet18
- Optimizers: SGD (momentum=0.9), Adam, AdamW (wd=0.01), SPSA, QNG
- Seeds: 42, 1337, 2025
- Total runs: 209

---

### 2. `nlp_experiment_results.csv` (Natural Language Processing)

**Contents:** Complete results for all NLP experiments.

| Column | Description |
|--------|-------------|
| `folder_name` | Unique experiment identifier |
| `run_name` | Descriptive run name |
| `dataset` | Dataset: `ag_news`, `imdb`, `sst2` |
| `model` | Model: `distilbert`, `roberta`, `lstm` |
| `optimizer` | Optimizer used |
| `seed` | Random seed |
| `learning_rate` | Learning rate |
| `batch_size` | Batch size (16) |
| `epochs` | Training epochs (3 for transformers, 8 for LSTM) |
| `test_accuracy` | Test set accuracy |
| `test_f1` | Test set F1 score |
| `cls_accuracy` | Classification accuracy |
| `macro_precision` | Macro-averaged precision |
| `macro_recall` | Macro-averaged recall |
| `macro_f1` | Macro-averaged F1 |
| `weighted_precision` | Weighted precision |
| `weighted_recall` | Weighted recall |
| `weighted_f1` | Weighted F1 |
| `final_train_loss` | Final training loss |
| `final_train_acc` | Final training accuracy |
| `final_val_loss` | Final validation loss |
| `final_val_acc` | Final validation accuracy |
| `final_val_f1` | Final validation F1 |
| `best_epoch` | Epoch with best validation performance |
| `best_val_loss` | Best validation loss |
| `best_val_acc` | Best validation accuracy |
| `best_val_f1` | Best validation F1 |
| `training_time_seconds` | Training time in seconds |

**Experiment Coverage:**
- Datasets: AG News, IMDB, SST-2
- Models: DistilBERT, RoBERTa, SimpleLSTM
- Optimizers: SGD, Adam, AdamW, RMSprop, Adagrad, SPSA, QNG, QPSO, COBYLA
- Total runs: 234

---

### 3. `tabular_experiment_results.csv` (Tabular Data - Summary)

**Contents:** Summarized results for tabular experiments.

| Column | Description |
|--------|-------------|
| `folder_name` | Unique experiment identifier |
| `dataset` | Dataset: `adults`, `higgs` |
| `model` | Model: `XGBoost`, `LightGBM`, `CatBoost` |
| `optimizer` | Optimizer: `SGD`, `Adam`, `SPSA`, `QNG` |
| `seed` | Random seed |
| `learning_rate_grid` | Learning rates tested: 0.1, 0.01, 0.001 |
| `accuracy` | Test accuracy |
| `macro_f1` | Macro-averaged F1 |
| `macro_precision` | Macro-averaged precision |
| `macro_recall` | Macro-averaged recall |
| `weighted_f1` | Weighted F1 |
| `weighted_precision` | Weighted precision |
| `weighted_recall` | Weighted recall |
| `final_train_loss` | Final training loss |
| `final_val_loss` | Final validation loss |
| `final_val_metric` | Final validation metric |

**Experiment Coverage:**
- Datasets: Adult Census, Higgs Boson
- Models: XGBoost, LightGBM, CatBoost
- Optimizers: SGD, Adam, SPSA, QNG
- Seeds: 42, 1337, 2025
- Total runs: 201

---

### 4. `tabular_experiment_results_detailed.csv` (Tabular Data - Detailed)

**Contents:** Detailed results with full hyperparameter logging.

| Column | Description |
|--------|-------------|
| `run_id` | Unique run identifier (UUID) |
| `run_name` | Descriptive run name |
| `status` | Run status (FINISHED) |
| `model` | Model type |
| `dataset` | Dataset name |
| `optimizer` | Optimizer used |
| `seed` | Random seed |
| `lr` | Learning rate |
| `momentum` | Momentum (if applicable) |
| `weight_decay` | Weight decay (if applicable) |
| `eps` | Epsilon (if applicable) |
| `betas` | Beta parameters (if applicable) |
| `a` | SPSA 'a' parameter |
| `c` | SPSA 'c' parameter |
| `max_iter` | Maximum iterations |
| `task_type` | Task type (classification/regression) |
| `gpu_name` | GPU model (Tesla T4) |
| `gpu_available` | GPU availability |
| `gpu_count` | Number of GPUs |
| `cpu_count` | Number of CPUs |
| `accuracy` | Test accuracy |
| `f1_score` | F1 score |
| `precision` | Precision |
| `recall` | Recall |
| `roc_auc` | ROC AUC score |
| `rmse` | Root mean squared error (regression) |
| `mse` | Mean squared error (regression) |
| `mae` | Mean absolute error (regression) |
| `r2_score` | R² score (regression) |
| `training_time` | Training time |

**Total runs:** 270

---

## Experimental Protocol

### Computer Vision
- **Epochs:** 50
- **Batch size:** 32
- **Learning rate:** 0.001
- **Loss function:** CrossEntropyLoss
- **Data augmentation:** Normalization only
- **Learning rate schedule:** None (constant)
- **Early stopping:** None

### Natural Language Processing
- **Epochs:** 3 (transformers), 8 (LSTM)
- **Batch size:** 16
- **Learning rates:** 1e-5 to 1e-3 (optimizer-dependent)
- **Max sequence length:** 256

### Tabular
- **Train/Val/Test split:** 70% / 15% / 15%
- **Stratification:** Yes (classification), No (regression)
- **Preprocessing:** StandardScaler (fit on train only)
- **Learning rate grid:** {0.1, 0.01, 0.001}

---

## Reproducibility

All experiments use fixed random seeds for reproducibility:
- **Seeds used:** 42, 1337, 2025

Seed is set for:
- Python's `random` module
- NumPy
- PyTorch (including CUDA)
- CUDNN deterministic mode enabled

---

## Hardware

All experiments were conducted on:
- **GPU:** NVIDIA Tesla T4 (16GB VRAM)
- **CPU:** 8 cores
- **RAM:** 16GB+

---

## Citation

If you use this data, please cite:

```bibtex
@article{yilmaz2025quantum,
  title={Benchmarking Quantum-Inspired Optimization Methods: 
         A Critical Evaluation on Computer Vision, Natural Language Processing, 
         and Tabular Data Tasks},
  author={Yilmaz, Gorkem},
  journal={Quantum Machine Intelligence},
  year={2025},
  institution={University of Sussex}
}
```

---

## Contact

For questions about the experimental results:

**Gorkem Yilmaz**  
University of Sussex  
Email: gy74@sussex.ac.uk
