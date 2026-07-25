"""NLP architectures — identical to v1 (paper_codes/nlp/models.py).

SimpleLSTM (2-layer bidirectional, ~6.3M params, of which 3.9M are embedding over the bert-base-uncased
vocabulary) plus HuggingFace DistilBERT / RoBERTa sequence classifiers.
"""

import torch
import torch.nn as nn
from transformers import (
    DistilBertTokenizer, DistilBertForSequenceClassification,
    RobertaTokenizer, RobertaForSequenceClassification,
    AutoTokenizer,
)


class SimpleLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2,
                            batch_first=True, bidirectional=True, dropout=0.3)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids, attention_mask=None):
        embedded = self.embedding(input_ids)
        _, (hidden, _) = self.lstm(embedded)
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        return self.fc(self.dropout(hidden))


def build_model(model_name, num_classes):
    if model_name == "distilbert":
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=num_classes)
    elif model_name == "roberta":
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        model = RobertaForSequenceClassification.from_pretrained(
            "roberta-base", num_labels=num_classes)
    elif model_name == "lstm":
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model = SimpleLSTM(vocab_size=tokenizer.vocab_size, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown NLP model: {model_name}")
    return model, tokenizer
