"""
BERT Sentiment Analysis Branch
Author: Boris Letić (Student 1)
Date: 2025-02-13

This module implements the Sentiment Analysis Branch of the multimodal architecture:
1. BERT fine-tuning for crypto sentiment classification (3 classes)
2. GRU-based sentiment embedding extraction
3. Rolling window sentiment aggregation

This is the main contribution of Student 1 (Boris) to the project.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import argparse
import os
from datetime import datetime


class CryptoSentimentDataset(Dataset):
    """
    PyTorch Dataset for crypto sentiment tweets
    """
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        """
        Args:
            texts: List of tweet texts
            labels: List of sentiment labels (0: negative, 1: neutral, 2: positive)
            tokenizer: BERT tokenizer
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


class BERTSentimentClassifier(nn.Module):
    """
    BERT-based sentiment classifier for crypto tweets
    Fine-tunes BERT for 3-class classification (negative, neutral, positive)
    """
    
    def __init__(self, n_classes=3, dropout=0.3, freeze_bert_layers=8):
        """
        Args:
            n_classes: Number of output classes (3 for ternary sentiment)
            dropout: Dropout probability
            freeze_bert_layers: Number of BERT layers to freeze (0-12)
        """
        super(BERTSentimentClassifier, self).__init__()
        
        # Load pre-trained BERT
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        # Freeze early layers for faster training
        if freeze_bert_layers > 0:
            for i, layer in enumerate(self.bert.encoder.layer):
                if i < freeze_bert_layers:
                    for param in layer.parameters():
                        param.requires_grad = False
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)
    
    def forward(self, input_ids, attention_mask):
        """
        Forward pass
        
        Args:
            input_ids: Token IDs (batch_size, seq_length)
            attention_mask: Attention mask (batch_size, seq_length)
        
        Returns:
            logits: Class logits (batch_size, n_classes)
            pooled_output: BERT [CLS] embedding (batch_size, hidden_size)
        """
        # BERT forward pass
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Extract [CLS] token representation
        pooled_output = outputs.pooler_output  # (batch_size, 768)
        
        # Classification
        output = self.dropout(pooled_output)
        logits = self.classifier(output)
        
        return logits, pooled_output


def train_epoch(model, data_loader, criterion, optimizer, scheduler, device):
    """
    Train for one epoch
    
    Returns:
        average_loss, accuracy
    """
    model.train()
    losses = []
    correct_predictions = 0
    total_predictions = 0
    
    progress_bar = tqdm(data_loader, desc="Training")
    
    for batch in progress_bar:
        # Move to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        # Forward pass
        logits, _ = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        # Metrics
        losses.append(loss.item())
        _, preds = torch.max(logits, dim=1)
        correct_predictions += torch.sum(preds == labels)
        total_predictions += labels.size(0)
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': np.mean(losses[-50:]),
            'acc': (correct_predictions.double() / total_predictions).item()
        })
    
    return np.mean(losses), (correct_predictions.double() / total_predictions).item()


def eval_model(model, data_loader, criterion, device):
    """
    Evaluate model on validation/test set
    
    Returns:
        average_loss, accuracy, predictions, true_labels
    """
    model.eval()
    losses = []
    predictions_list = []
    true_labels_list = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            logits, _ = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            
            losses.append(loss.item())
            
            _, preds = torch.max(logits, dim=1)
            predictions_list.extend(preds.cpu().numpy())
            true_labels_list.extend(labels.cpu().numpy())
    
    predictions_list = np.array(predictions_list)
    true_labels_list = np.array(true_labels_list)
    
    accuracy = accuracy_score(true_labels_list, predictions_list)
    
    return np.mean(losses), accuracy, predictions_list, true_labels_list


def plot_training_history(history, save_path):
    """
    Plot training and validation metrics
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(history['val_loss'], label='Val Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(history['train_acc'], label='Train Acc', marker='o')
    axes[1].plot(history['val_acc'], label='Val Acc', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f" Training history plot saved to: {save_path}")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """
    Plot confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - BERT Sentiment Classification')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f" Confusion matrix saved to: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='BERT Sentiment Analysis Fine-tuning')
    
    parser.add_argument('--data', type=str, required=True,
                       help='Path to labeled tweets CSV')
    parser.add_argument('--output_dir', type=str, default='models/bert_sentiment/',
                       help='Output directory for model and results')
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-5,
                       help='Learning rate')
    parser.add_argument('--max_length', type=int, default=128,
                       help='Maximum sequence length')
    parser.add_argument('--freeze_layers', type=int, default=8,
                       help='Number of BERT layers to freeze')
    
    args = parser.parse_args()
    
    print(f"\n{'#'*70}")
    print(f"# BERT Sentiment Analysis Fine-tuning")
    print(f"# Author: Boris Letic (Student 1)")
    print(f"# Model: bert-base-uncased")
    print(f"{'#'*70}\n")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load data
    print(f"\nLoading data from: {args.data}")
    df = pd.read_csv(args.data)
    
    # Map sentiment to labels
    sentiment_map = {'negative': 0, 'neutral': 1, 'positive': 2}
    df['label'] = df['sentiment'].map(sentiment_map)
    
    # Filter out any unmapped sentiments
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)
    
    print(f"Loaded {len(df)} labeled tweets")
    print(f"\nClass distribution:")
    for sentiment, label in sentiment_map.items():
        count = (df['label'] == label).sum()
        pct = (count / len(df)) * 100
        print(f"  {sentiment.capitalize()}: {count} ({pct:.1f}%)")
    
    # Split data (80/20 train/val from labeled data)
    train_size = int(0.8 * len(df))
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:]
    
    print(f"\nData split:")
    print(f"  Training: {len(train_df)} samples")
    print(f"  Validation: {len(val_df)} samples")
    
    # Initialize tokenizer
    print(f"\nInitializing BERT tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = CryptoSentimentDataset(
        texts=train_df['cleaned_text'].values,
        labels=train_df['label'].values,
        tokenizer=tokenizer,
        max_length=args.max_length
    )
    
    val_dataset = CryptoSentimentDataset(
        texts=val_df['cleaned_text'].values,
        labels=val_df['label'].values,
        tokenizer=tokenizer,
        max_length=args.max_length
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Initialize model
    print(f"\nInitializing BERT model...")
    print(f"  - Freezing first {args.freeze_layers} layers")
    model = BERTSentimentClassifier(
        n_classes=3,
        dropout=0.3,
        freeze_bert_layers=args.freeze_layers
    )
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    # Learning rate scheduler
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    # Training loop
    print(f"\n{'='*70}")
    print(f"Starting training for {args.epochs} epochs...")
    print(f"{'='*70}\n")
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 70)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device
        )
        
        # Validate
        val_loss, val_acc, val_preds, val_labels = eval_model(
            model, val_loader, criterion, device
        )
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"\nEpoch {epoch + 1} Results:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_path = os.path.join(args.output_dir, 'bert_sentiment_best.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss
            }, model_path)
            print(f"Best model saved (Val Acc: {val_acc:.4f})")
    
    # Final evaluation
    print(f"\n{'='*70}")
    print(f"Training Complete!")
    print(f"{'='*70}")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    
    # Load best model for final evaluation
    checkpoint = torch.load(os.path.join(args.output_dir, 'bert_sentiment_best.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final validation metrics
    val_loss, val_acc, val_preds, val_labels = eval_model(
        model, val_loader, criterion, device
    )
    
    # Classification report
    class_names = ['Negative', 'Neutral', 'Positive']
    print(f"\nClassification Report:")
    print(classification_report(val_labels, val_preds, target_names=class_names))
    
    # F1 score
    f1_weighted = f1_score(val_labels, val_preds, average='weighted')
    print(f"Weighted F1-Score: {f1_weighted:.4f}")
    
    # Plot training history
    plot_path = os.path.join(args.output_dir, 'training_history.png')
    plot_training_history(history, plot_path)
    
    # Plot confusion matrix
    cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(val_labels, val_preds, class_names, cm_path)
    
    # Save tokenizer
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nTokenizer saved to: {args.output_dir}")
    
    print(f"\n{'='*70}")
    print(f"All results saved to: {args.output_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
