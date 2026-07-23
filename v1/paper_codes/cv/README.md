# Computer Vision Benchmark - Quantum-Inspired Optimizers

This directory contains the code for benchmarking quantum-inspired optimization methods against classical baseline optimizers on computer vision tasks.

## Overview

The benchmark compares five optimizers across three datasets and two neural network architectures:

### Optimizers
- **Classical Baselines:**
  - SGD: Stochastic Gradient Descent with momentum (0.9)
  - Adam: Adaptive Moment Estimation
  - AdamW: Adam with weight decay decoupling (0.01)

- **Quantum-Inspired:**
  - SPSA: Simultaneous Perturbation Stochastic Approximation
  - QNG: Quantum Natural Gradient

### Datasets
- **CIFAR-10:** 60,000 32x32 color images in 10 classes
- **CIFAR-100:** 60,000 32x32 color images in 100 classes  
- **Fashion-MNIST:** 70,000 28x28 grayscale images in 10 classes

### Models
- **SimpleCNN:** 3-layer convolutional neural network with dropout
- **ResNet18:** 18-layer residual network adapted for small images

## Installation

Install required packages:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Training

Train a single model:

```bash
python trainer.py \
    --dataset cifar10 \
    --model SimpleCNN \
    --optimizer Adam \
    --seed 42 \
    --epochs 50
```

### Full Benchmark

Run the complete benchmark across all configurations:

```bash
python run_benchmark.py \
    --datasets cifar10 cifar100 fashion_mnist \
    --models SimpleCNN ResNet18 \
    --optimizers SGD Adam AdamW SPSA QNG \
    --seeds 42 1337 2025 \
    --epochs 50
```

### Command-Line Arguments

**Required:**
- `--dataset`: Dataset name (cifar10, cifar100, fashion_mnist)
- `--model`: Model architecture (SimpleCNN, ResNet18)
- `--optimizer`: Optimizer name (SGD, Adam, AdamW, SPSA, QNG)
- `--seed`: Random seed for reproducibility

**Optional:**
- `--epochs`: Number of training epochs (default: 50)
- `--lr`: Learning rate (default: 0.001)
- `--batch_size`: Batch size (default: 32)
- `--data_dir`: Directory for datasets (default: ./datasets)
- `--results_dir`: Directory for results (default: ./results)

**Optimizer-Specific:**
- `--momentum`: SGD momentum (default: 0.9)
- `--weight_decay`: AdamW weight decay (default: 0.01)
- `--spsa_a`: SPSA parameter a (default: 0.1)
- `--spsa_c`: SPSA parameter c (default: 0.01)

## Code Structure

```
cv/
├── trainer.py          # Main training script
├── models.py           # Neural network architectures (SimpleCNN, ResNet18)
├── optimizers.py       # Optimizer implementations
├── utils.py            # Utility functions (evaluation, visualization, I/O)
├── run_benchmark.py    # Batch benchmark execution script
├── README.md           # This file
└── requirements.txt    # Python dependencies
```

## Output Structure

Each experiment creates a directory with the following structure:

```
results/{dataset}_{model}_{optimizer}_{seed}/
├── run_summary.json              # Experiment configuration and metrics
├── history.csv                   # Training history (epoch-wise)
├── loss_curve.png                # Training/validation curves
├── model.pth                     # Trained model weights
├── classification_report.json    # Detailed per-class metrics
├── confusion_matrix.csv          # Confusion matrix
├── confusion_matrix.json         # Confusion matrix (JSON format)
└── f1_scores.json                # F1 scores (macro and weighted)
```

## Results Format

### run_summary.json

```json
{
  "run_id": "cifar10_SimpleCNN_Adam_42",
  "dataset": "cifar10",
  "model": "SimpleCNN",
  "optimizer": "Adam",
  "seed": 42,
  "hyperparameters": {
    "lr": 0.001,
    "epochs": 50,
    "batch_size": 32
  },
  "metrics": {
    "test_accuracy": 75.23,
    "test_loss": 0.812,
    "best_val_accuracy": 75.89,
    "training_time_seconds": 856.3
  }
}
```

### history.csv

| epoch | train_acc | val_acc | train_loss | val_loss |
|-------|-----------|---------|------------|----------|
| 1     | 35.2      | 42.1    | 1.856      | 1.623    |
| 2     | 48.7      | 52.3    | 1.432      | 1.298    |
| ...   | ...       | ...     | ...        | ...      |
| 50    | 98.1      | 75.2    | 0.056      | 0.812    |

## Reproducibility

All experiments use fixed random seeds for reproducibility:
- Seeds used in paper: 42, 1337, 2025
- Results are averaged across these three seeds

To reproduce paper results:

```bash
# Run for each seed
for seed in 42 1337 2025; do
    python trainer.py \
        --dataset cifar10 \
        --model SimpleCNN \
        --optimizer Adam \
        --seed $seed \
        --epochs 50
done
```

## Model Architectures

### SimpleCNN

- **Conv1:** 3→32 channels, 3×3 kernel, ReLU, 2×2 MaxPool
- **Conv2:** 32→64 channels, 3×3 kernel, ReLU, 2×2 MaxPool
- **Conv3:** 64→128 channels, 3×3 kernel, ReLU, 2×2 MaxPool
- **FC1:** Flattened → 512 units, ReLU, Dropout(0.5)
- **FC2:** 512 → num_classes (softmax)

**Parameters:** ~1.2M (CIFAR), ~550K (Fashion-MNIST)

### ResNet18

Standard ResNet-18 architecture with:
- Modified first conv layer for grayscale inputs (Fashion-MNIST)
- Modified final FC layer for dataset-specific number of classes
- No pretrained weights (trained from scratch)

**Parameters:** ~11.2M

## Hardware Requirements

- **GPU:** NVIDIA T4 or equivalent (16GB VRAM)
- **CPU:** 4+ cores recommended
- **RAM:** 8GB minimum, 16GB recommended
- **Storage:** ~10GB for datasets + results

## Training Time

Approximate training times per configuration (50 epochs, batch size 32, T4 GPU):

| Model      | Dataset       | Time    |
|------------|---------------|---------|
| SimpleCNN  | CIFAR-10      | ~15 min |
| SimpleCNN  | CIFAR-100     | ~15 min |
| SimpleCNN  | Fashion-MNIST | ~5 min  |
| ResNet18   | CIFAR-10      | ~30 min |
| ResNet18   | CIFAR-100     | ~30 min |
| ResNet18   | Fashion-MNIST | ~10 min |

## Citation

If you use this code in your research, please cite:

```bibtex
@article{yilmaz2025quantum,
  title={Benchmarking Quantum-Inspired Optimization Methods: 
         A Critical Evaluation on Computer Vision, Natural Language Processing, 
         and Tabular Data Tasks},
  author={Yilmaz, Gorkem},
  journal={[Journal Name]},
  year={2025}
}
```

## Author

**Gorkem Yilmaz**  
University of Sussex  
gy74@sussex.ac.uk

## License

This code is provided for research purposes. Please see LICENSE file for details.
