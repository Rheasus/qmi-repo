"""
Neural Network Models for NLP Benchmark
========================================

This module implements:
1. SimpleLSTM: 2-layer bidirectional LSTM with embedding
2. DistilBERT: Distilled BERT model for sequence classification
3. RoBERTa: Robustly Optimized BERT Pretraining Approach

Author: Gorkem Yilmaz
Institution: University of Sussex
Contact: gy74@sussex.ac.uk
"""

import torch
import torch.nn as nn
from transformers import (
    DistilBertTokenizer, DistilBertForSequenceClassification,
    RobertaTokenizer, RobertaForSequenceClassification,
    AutoTokenizer
)


class SimpleLSTM(nn.Module):
    """
    Bidirectional LSTM for text classification.
    
    Architecture:
        - Embedding layer (vocab_size → embedding_dim)
        - 2-layer bidirectional LSTM (dropout=0.3 between layers)
        - Concatenate final forward and backward hidden states
        - Dropout(0.5)
        - Fully connected layer → num_classes
    
    Note: This architecture does NOT include an intermediate FC projection layer.
          The LSTM outputs (2 * hidden_dim) are directly fed to the output layer.
    
    Args:
        vocab_size: Size of vocabulary
        embedding_dim: Dimension of word embeddings (default: 128)
        hidden_dim: Hidden dimension of LSTM (default: 256)
        num_classes: Number of output classes (default: 2)
    """
    
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_classes=2):
        super(SimpleLSTM, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=2, 
            batch_first=True, 
            bidirectional=True, 
            dropout=0.3
        )
        self.dropout = nn.Dropout(0.5)
        
        # Direct connection: concatenated hidden states → output
        # No intermediate FC layer
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
    
    def forward(self, input_ids, attention_mask=None):
        # Embed input tokens
        embedded = self.embedding(input_ids)
        
        # Pass through LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Concatenate final forward and backward hidden states from last layer
        # hidden shape: [num_layers * num_directions, batch, hidden_dim]
        # We take the last layer's forward (hidden[-2]) and backward (hidden[-1])
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        
        # Apply dropout
        hidden = self.dropout(hidden)
        
        # Output layer
        output = self.fc(hidden)
        
        return output


def build_model(model_name, num_classes):
    """
    Build model and tokenizer.
    
    Args:
        model_name: One of 'distilbert', 'roberta', 'lstm'
        num_classes: Number of output classes
        
    Returns:
        model, tokenizer
    """
    if model_name == 'distilbert':
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        model = DistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-uncased',
            num_labels=num_classes
        )
    
    elif model_name == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaForSequenceClassification.from_pretrained(
            'roberta-base',
            num_labels=num_classes
        )
    
    elif model_name == 'lstm':
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        model = SimpleLSTM(
            vocab_size=tokenizer.vocab_size, 
            num_classes=num_classes
        )
    
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model, tokenizer
