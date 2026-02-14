"""
GRU Sentiment Embedding Module
Author: Boris Letić (Student 1)
Date: 2025-02-13

This module extracts sentiment embeddings from BERT outputs using GRU:
1. Takes BERT [CLS] token embeddings (768-dim)
2. Processes rolling window sentiment features through GRU
3. Outputs 64-dim sentiment embedding for fusion

This is the final component of Student 1's contribution.
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
import argparse
import os
from tqdm import tqdm


class GRUSentimentEmbedding(nn.Module):
    """
    GRU-based sentiment embedding extractor
    
    Takes BERT sentiment embeddings + rolling features and produces
    a 64-dimensional embedding for multimodal fusion
    """
    
    def __init__(self, bert_dim=768, sentiment_features_dim=12, 
                 hidden_dim=64, num_layers=2, dropout=0.3):
        """
        Args:
            bert_dim: BERT [CLS] embedding dimension (768)
            sentiment_features_dim: Number of additional sentiment features
                                   (rolling mean, std, volume, engagement)
            hidden_dim: GRU hidden dimension (output embedding size)
            num_layers: Number of GRU layers
            dropout: Dropout probability
        """
        super(GRUSentimentEmbedding, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Input dimension = BERT embedding + sentiment features
        input_dim = bert_dim + sentiment_features_dim
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Output projection (optional, for additional transformation)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, bert_embeddings, sentiment_features, lengths=None):
        """
        Forward pass
        
        Args:
            bert_embeddings: BERT [CLS] embeddings (batch, seq_len, 768)
            sentiment_features: Additional features (batch, seq_len, sentiment_features_dim)
            lengths: Actual sequence lengths for packing (optional)
        
        Returns:
            sentiment_embedding: (batch, hidden_dim) - final sentiment representation
            all_hidden: (batch, seq_len, hidden_dim) - all GRU hidden states
        """
        # Concatenate BERT embeddings with sentiment features
        combined = torch.cat([bert_embeddings, sentiment_features], dim=-1)
        # (batch, seq_len, bert_dim + sentiment_features_dim)
        
        # Pack padded sequence if lengths provided
        if lengths is not None:
            combined = nn.utils.rnn.pack_padded_sequence(
                combined, lengths, batch_first=True, enforce_sorted=False
            )
        
        # GRU forward pass
        output, hidden = self.gru(combined)
        # output: (batch, seq_len, hidden_dim)
        # hidden: (num_layers, batch, hidden_dim)
        
        # Unpack if packed
        if lengths is not None:
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        
        # Take last hidden state from final layer
        sentiment_embedding = hidden[-1]  # (batch, hidden_dim)
        
        # Optional: Apply output projection
        sentiment_embedding = self.output_proj(sentiment_embedding)
        
        return sentiment_embedding, output


class SentimentEmbeddingDataset(Dataset):
    """
    Dataset for extracting sentiment embeddings with rolling window
    """
    
    def __init__(self, tweets_df, sentiment_features_df, window_size=168):
        """
        Args:
            tweets_df: DataFrame with individual tweets and BERT embeddings
            sentiment_features_df: DataFrame with hourly aggregated features
            window_size: Rolling window size in hours (default: 168 = 7 days)
        """
        self.tweets_df = tweets_df
        self.sentiment_features_df = sentiment_features_df
        self.window_size = window_size
        
        # Ensure sorted by datetime
        self.sentiment_features_df = self.sentiment_features_df.sort_values('datetime')
        
    def __len__(self):
        return len(self.sentiment_features_df) - self.window_size
    
    def __getitem__(self, idx):
        """
        Get a window of sentiment data
        
        Returns:
            Dictionary with:
                - bert_embeddings: (window_size, 768)
                - sentiment_features: (window_size, n_features)
                - datetime: Target datetime
        """
        # Get window of hourly data
        window_start = idx
        window_end = idx + self.window_size
        
        window_df = self.sentiment_features_df.iloc[window_start:window_end]
        
        # Extract features (example columns - adjust based on actual data)
        feature_cols = [
            'sentiment_score_mean', 'sentiment_score_std',
            'sentiment_mean_6h', 'sentiment_std_6h',
            'sentiment_mean_12h', 'sentiment_std_12h',
            'sentiment_mean_24h', 'sentiment_std_24h',
            'tweet_volume_24h', 'engagement_24h'
        ]
        
        # Fill missing columns with zeros if they don't exist
        for col in feature_cols:
            if col not in window_df.columns:
                window_df[col] = 0.0
        
        sentiment_features = window_df[feature_cols].values
        sentiment_features = torch.FloatTensor(sentiment_features)
        
        # For BERT embeddings, we'll use pre-computed embeddings or zeros
        # In practice, these would be extracted using the trained BERT model
        # For now, use placeholder
        bert_embeddings = torch.randn(self.window_size, 768)  # Placeholder
        
        # Target datetime
        target_datetime = window_df.iloc[-1]['datetime']
        
        return {
            'bert_embeddings': bert_embeddings,
            'sentiment_features': sentiment_features,
            'datetime': target_datetime
        }


def extract_bert_embeddings_batch(model, tokenizer, texts, device, max_length=128):
    """
    Extract BERT [CLS] embeddings for a batch of texts
    
    Args:
        model: Trained BERT model
        tokenizer: BERT tokenizer
        texts: List of text strings
        device: torch device
        max_length: Maximum sequence length
    
    Returns:
        Tensor of shape (batch_size, 768)
    """
    model.eval()
    
    encodings = tokenizer(
        texts,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encodings['input_ids'].to(device)
    attention_mask = encodings['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model.bert(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.pooler_output  # [CLS] token embeddings
    
    return embeddings


def main():
    parser = argparse.ArgumentParser(description='GRU Sentiment Embedding Extraction')
    
    parser.add_argument('--sentiment_features', type=str, required=True,
                       help='Path to sentiment features CSV (hourly aggregated)')
    parser.add_argument('--bert_model', type=str, required=True,
                       help='Path to trained BERT sentiment model')
    parser.add_argument('--output_dir', type=str, default='models/sentiment_embeddings/',
                       help='Output directory for embeddings')
    parser.add_argument('--window_size', type=int, default=168,
                       help='Rolling window size in hours (default: 168 = 7 days)')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for processing')
    
    args = parser.parse_args()
    
    print(f"\n{'#'*70}")
    print(f"# GRU Sentiment Embedding Extraction")
    print(f"# Author: Boris Letic (Student 1)")
    print(f"# Window Size: {args.window_size} hours")
    print(f"{'#'*70}\n")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Load sentiment features
    print(f"Loading sentiment features from: {args.sentiment_features}")
    sentiment_df = pd.read_csv(args.sentiment_features)
    sentiment_df['datetime'] = pd.to_datetime(sentiment_df['datetime'])
    print(f" Loaded {len(sentiment_df)} hourly data points")
    
    # Initialize GRU model
    print(f"\nInitializing GRU Sentiment Embedding model...")
    gru_model = GRUSentimentEmbedding(
        bert_dim=768,
        sentiment_features_dim=10,  # Adjust based on actual features
        hidden_dim=64,
        num_layers=2,
        dropout=0.3
    )
    gru_model = gru_model.to(device)
    
    print(f" GRU model initialized")
    print(f"  Input: BERT (768) + Sentiment Features (10) = 778")
    print(f"  Output: 64-dimensional embedding")
    
    # Create dataset
    # Note: In practice, you would have actual tweets aligned with hourly data
    # For demonstration, we create a simple dataset
    
    print(f"\nNote: This is a demonstration of the GRU architecture.")
    print(f"In the full implementation, you would:")
    print(f"  1. Extract BERT embeddings for tweets in each hour")
    print(f"  2. Aggregate them with sentiment features")
    print(f"  3. Process through GRU with {args.window_size}h rolling window")
    print(f"  4. Output 64-dim sentiment embedding for fusion module")
    
    # Save model architecture
    model_path = os.path.join(args.output_dir, 'gru_sentiment_model.pt')
    torch.save({
        'model_state_dict': gru_model.state_dict(),
        'config': {
            'bert_dim': 768,
            'sentiment_features_dim': 10,
            'hidden_dim': 64,
            'num_layers': 2,
            'dropout': 0.3,
            'window_size': args.window_size
        }
    }, model_path)
    
    print(f"\n GRU model architecture saved to: {model_path}")
    
    # Create a sample forward pass to demonstrate
    print(f"\nDemonstration - Sample forward pass:")
    batch_size = 4
    seq_len = args.window_size
    
    # Sample inputs
    sample_bert_emb = torch.randn(batch_size, seq_len, 768).to(device)
    sample_sent_feat = torch.randn(batch_size, seq_len, 10).to(device)
    
    # Forward pass
    gru_model.eval()
    with torch.no_grad():
        sentiment_emb, all_hidden = gru_model(sample_bert_emb, sample_sent_feat)
    
    print(f"  Input shapes:")
    print(f"    - BERT embeddings: {sample_bert_emb.shape}")
    print(f"    - Sentiment features: {sample_sent_feat.shape}")
    print(f"  Output shapes:")
    print(f"    - Sentiment embedding: {sentiment_emb.shape}")
    print(f"    - All GRU hidden states: {all_hidden.shape}")
    
    print(f"\n{'='*70}")
    print(f"GRU Sentiment Embedding module ready!")
    print(f"{'='*70}")
    print(f"\nThis 64-dim sentiment embedding will be concatenated with")
    print(f"the 64-dim price embedding from Bogdan's LSTM module,")
    print(f"then processed through the Multi-Head Attention fusion layer.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
