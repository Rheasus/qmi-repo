"""NLP data pipeline — identical to v1 (paper_codes/nlp/utils.py).

HF datasets ag_news / imdb / glue-sst2 (SST-2 uses the GLUE validation split
as the evaluation set, standard practice since test labels are hidden).
Tokenization: max_length=256, fixed padding, truncation.

Protocol note (kept deliberately): as in v1, per-epoch curves are computed on
the evaluation set. No model selection or early stopping uses these metrics
(the final-epoch model is the reported model), so this does not leak into
results; v2 keeps it so new runs remain protocol-identical to the retained
v1 NLP results. `max_samples` exists for smoke tests only.
"""

from pathlib import Path

import torch
from torch.utils.data import Dataset
from datasets import load_dataset as hf_load_dataset

NUM_CLASSES = {"ag_news": 4, "imdb": 2, "sst2": 2}


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]), add_special_tokens=True,
            max_length=self.max_length, padding="max_length",
            truncation=True, return_tensors="pt")
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(int(self.labels[idx]), dtype=torch.long),
        }


def load_nlp_data(dataset_name, tokenizer, data_dir, max_length=256, max_samples=None):
    cache_dir = str(Path(data_dir))
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    # namespaced hub ids (canonical homes of the same v1 datasets; the legacy
    # short names no longer resolve on current huggingface_hub versions)
    if dataset_name == "ag_news":
        ds = hf_load_dataset("fancyzhx/ag_news", cache_dir=cache_dir)
        train = (ds["train"]["text"], ds["train"]["label"])
        test = (ds["test"]["text"], ds["test"]["label"])
    elif dataset_name == "imdb":
        ds = hf_load_dataset("stanfordnlp/imdb", cache_dir=cache_dir)
        train = (ds["train"]["text"], ds["train"]["label"])
        test = (ds["test"]["text"], ds["test"]["label"])
    elif dataset_name == "sst2":
        ds = hf_load_dataset("nyu-mll/glue", "sst2", cache_dir=cache_dir)
        train = (ds["train"]["sentence"], ds["train"]["label"])
        test = (ds["validation"]["sentence"], ds["validation"]["label"])
    else:
        raise ValueError(f"Unknown NLP dataset: {dataset_name}")

    if max_samples:
        train = (train[0][:max_samples], train[1][:max_samples])
        n_test = min(max_samples // 5, len(test[0]))
        test = (test[0][:n_test], test[1][:n_test])

    train_ds = TextDataset(*train, tokenizer, max_length)
    test_ds = TextDataset(*test, tokenizer, max_length)
    return train_ds, test_ds, NUM_CLASSES[dataset_name]
