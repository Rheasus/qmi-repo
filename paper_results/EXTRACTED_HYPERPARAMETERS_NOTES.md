EXTRACTED HYPERPARAMETERS (NOTES)
================================

Purpose
-------
This note extracts **configuration metadata only** (no performance results) from the logged
CSV artifacts in `experiment_results/`. It is intended to support writing the Methods / Setup
sections without re-parsing the raw result tables.

IMPORTANT
---------
- Do **not** treat any metric columns (accuracy/F1/loss/etc.) here as reportable in Methods.
- If a required detail is missing from the repository artifacts, it is marked as `[[INFO NEEDED]]`.


1) NLP (from `nlp_consildated_results.csv`)
------------------------------------------

**Coverage (as logged):**
- Datasets: `ag_news`, `imdb`, `sst2`
- Models: `distilbert`, `roberta`, `lstm`
- Optimizers: `sgd`, `adam`, `adamw`, `rmsprop`, `adagrad`, `spsa`, `qng`, `cobyla`, `qpso`

**Training hyperparameters (as logged):**
- Batch size: `16`
- Epochs: `3` (transformers), `8` (LSTM)
- Learning-rate values observed in the consolidated log:
  - `1e-05`, `2e-05`, `3e-05`, `0.0001`, `0.001`
- Seeds observed in the consolidated log:
  - `2025`

**NLP training protocol (as logged in `NLP – CURRENT EXPERIMENT TABLE.md` and `nlp_consildated_results.csv`):**
- Batch size: `16`
- Epochs:
  - DistilBERT / RoBERTa: `3`
  - LSTM: `8`

[[INFO NEEDED: If a learning-rate scheduler was used (e.g., linear warmup/decay), specify the exact configuration.]]


2) CV (from `cv_experiment_results.csv`)
---------------------------------------

**Coverage (robustly parsed from `folder_name`):**
- Datasets: `cifar10`, `cifar100`, `fashion_mnist` (dataset column is logged as `fashion`)
- Models: `ResNet18`, `SimpleCNN`
- Optimizers: `SGD`, `Adam`, `AdamW`, `SPSA`, `QNG`
- Seeds observed (parsed from `folder_name`): `42`, `123`, `1337`, `2025`

**Known logging issue (Fashion-MNIST rows):**
- For rows where `folder_name` starts with `fashion_mnist_...`, the CSV columns `model` and `optimizer`
  are shifted (e.g., `model` appears as `mnist`, and `optimizer` may incorrectly appear as `ResNet18`
  or `SimpleCNN`). For Methods-writing, prefer parsing `folder_name` as:
  - dataset = all tokens except last 4 (joined by `_`)
  - model   = token -4
  - optimizer = token -3
  - seed    = token -2
  - timestamp = token -1

**Author-provided CV training protocol (to be reflected in Methods):**
- Epochs: `50`
- Batch size: `32`
- Loss: `CrossEntropy`
- Data augmentation: none beyond normalization
- Learning-rate schedule: none
- Early stopping: none (all runs complete 50 epochs)
- Base learning rate: `0.001`
- Optimizer settings:
  - SGD: momentum = `0.9`
  - Adam: default configuration
  - AdamW: weight decay = `0.01`
  - SPSA: \(a=0.1\), \(c=0.01\)
  - QNG: Adam-based approximation (heuristic configuration)
- Seeds (CV): `42`, `1337`, `2025`

[[INFO NEEDED: CV run identifiers in `cv_experiment_results.csv` suggest additional seeds (e.g., `123`) may exist in logs.
Confirm whether only {42, 1337, 2025} are used for paper-reported CV results.]]


3) Tabular (from `tabular_experiment_results.csv` and `ALL_MODELS_consolidated_results.csv`)
--------------------------------------------------

**Coverage (as logged):**
- Datasets (classification): `adults`, `higgs`
- Datasets (regression): `california_housing` (present in `ALL_MODELS_consolidated_results.csv`)
- Models: `XGBoost`, `LightGBM`, `CatBoost`
- Optimizers: `SGD`, `Adam`, `SPSA`, `QNG`
- Seeds observed (parsed from `folder_name`): `42`, `1337`, `2025`

**Dataset naming note (tabular regression):**
- The global setup document names the dataset as `California Housing`, while the consolidated log uses
  the identifier `california_housing`.

**Tabular hyperparameter grid (from `ALL_MODELS_consolidated_results.csv`, using `param_*` fields only):**
- Learning-rate values (`param_lr`) used in tabular runs:
  - `0.1`, `0.01`, `0.001`
- Seeds used (tabular): `42`, `1337`, `2025`
- Task types present (`param_task_type`): `classification`, `regression`
- Hardware fields present in the consolidated log (tabular):
  - GPU: `Tesla T4` (`param_gpu_available=True`, `param_gpu_count=1`)
  - CPU count: `8`

**Tabular LR coverage nuance (from per-run identifiers in the consolidated log):**
- For `SGD` and `Adam`, the LR grid includes `{0.1, 0.01, 0.001}` across datasets and models.
- For `SPSA` and `QNG`, the LR grid includes `{0.1, 0.01}` across datasets and models.

**Missing/empty hyperparameter fields in the consolidated log (tabular):**
- `param_weight_decay`: empty in tabular rows
- `param_momentum`: empty in tabular rows
- `param_eps`: empty in tabular rows
- `param_betas`: empty in tabular rows

**Tabular data splitting + preprocessing (from `tabular-codes/fromazure/`):**
- This section is based on the **author-provided standard tabular protocol** (message dated 2026-01-08),
  and should be treated as the authoritative split + preprocessing description unless contradicted by
  other repository artifacts.

**Splitting protocol (tabular – standard):**
- Goal: reproducible + optimizer-fair comparisons.
- Principle: the **same split** is used across optimizers; validation is always separate.
- Split ratios (all tabular datasets): Train 70\% / Validation 15\% / Test 15\%.
- Seeds used: `42`, `1337`, `2025`.
- Seed is set globally for:
  - `train_test_split`
  - NumPy
  - PyTorch
  (via `utils/seed.py`).

**Stratification (classification vs regression):**
- Classification datasets (`adults`, `higgs`): stratified splits (`stratify=y`) are used both for the initial
  split and for the subsequent train→val/test split.
- Regression dataset (`california_housing`): no stratification (`stratify=None`).

**Tabular dataset schema notes (from `tabular-codes/fromazure/preprocessing_config.py`):**
- Adult (code name `uci_adult`): target column `income`; categorical + numerical features configured explicitly.
- California Housing (`california_housing`): target `MedHouseVal`; 8 numerical features; no categorical features.
- Higgs (`higgs`): target `Label`; numerical features listed as `feature_0`..`feature_27`.
- Confirmed by author:
  - `uci_adult` and `adults` refer to the same dataset.
  - Higgs feature naming matches `feature_{i}` in the executed dataset file.

**Preprocessing rule (critical, leakage control):**
- Fit transformations on **train only**, then transform validation/test.
- Numerical: `StandardScaler`.
- Categorical (Adult): `OneHotEncoder(handle_unknown='ignore')`.

**Training-related protocol (tabular):**
- max\_epochs = 50
- early\_stopping = True; patience = 5; monitor = val\_loss
- batch\_size = 32
- num\_runs = 3 (same split; same seed; different optimizer runs) → mean ± std reported.

**Clarification (MLP vs GBDT training):**
- The epoch/batch-size/validation-loss early-stopping protocol applies **only** to the neural tabular baseline (MLP).
- XGBoost/LightGBM/CatBoost are trained with their native boosting-iteration training and model-specific early-stopping
  criteria, without mini-batch training.

**Missing training hyperparameters:**
- [[INFO NEEDED]] The exact learning-rate / tuning grid and the precise training protocol for each GBDT model
  (objective, boosting rounds, early stopping, regularization, etc.) are not present in the CSV artifacts.

**Logging/aggregation note (regression):**
- Some downstream extraction/summary logic may assume classification-only artifacts (e.g., confusion-matrix/F1).
  For `california_housing` (regression), rely on the consolidated log and regression metrics (e.g., RMSE/MAE/\(R^2\))
  rather than classification metrics.


