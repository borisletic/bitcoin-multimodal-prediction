"""
VADER Sentiment Baseline Model
Author: Boris Letić (Student 1)
Date: 2025-02-13

This is the baseline sentiment analysis model using:
- VADER (Valence Aware Dictionary and sEntiment Reasoner) for rule-based sentiment
- Logistic Regression for classification

Purpose: Compare against BERT fine-tuned model to demonstrate value of deep learning
"""

import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import pickle


class VADERSentimentBaseline:
    """
    VADER-based sentiment classifier
    Uses rule-based sentiment scores + Logistic Regression
    """
    
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
        self.classifier = None
        
    def extract_vader_features(self, text):
        """
        Extract VADER sentiment features from text
        
        Returns:
            Dictionary with compound, positive, negative, neutral scores
        """
        scores = self.vader.polarity_scores(str(text))
        return scores
    
    def predict_vader_simple(self, text):
        """
        Simple VADER prediction using compound score thresholds
        
        Returns:
            0 (negative), 1 (neutral), 2 (positive)
        """
        scores = self.extract_vader_features(text)
        compound = scores['compound']
        
        if compound >= 0.05:
            return 2  # positive
        elif compound <= -0.05:
            return 0  # negative
        else:
            return 1  # neutral
    
    def create_feature_matrix(self, texts):
        """
        Create feature matrix from texts using VADER scores
        
        Args:
            texts: List of text strings
        
        Returns:
            numpy array of shape (n_samples, 4) with VADER features
        """
        features = []
        
        for text in texts:
            scores = self.extract_vader_features(text)
            features.append([
                scores['compound'],
                scores['pos'],
                scores['neu'],
                scores['neg']
            ])
        
        return np.array(features)
    
    def train(self, train_texts, train_labels):
        """
        Train Logistic Regression on VADER features
        
        Args:
            train_texts: List of training texts
            train_labels: List of training labels (0, 1, 2)
        """
        print("\nTraining VADER + Logistic Regression baseline...")
        
        # Extract features
        X_train = self.create_feature_matrix(train_texts)
        
        # Train classifier
        self.classifier = LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        )
        self.classifier.fit(X_train, train_labels)
        
        # Training accuracy
        train_preds = self.classifier.predict(X_train)
        train_acc = accuracy_score(train_labels, train_preds)
        
        print(f" Training complete")
        print(f"  Training Accuracy: {train_acc:.4f}")
        
        return self
    
    def predict(self, texts):
        """
        Predict sentiment using trained classifier
        
        Args:
            texts: List of text strings
        
        Returns:
            numpy array of predictions
        """
        if self.classifier is None:
            raise ValueError("Model not trained. Call train() first.")
        
        X = self.create_feature_matrix(texts)
        predictions = self.classifier.predict(X)
        
        return predictions
    
    def predict_simple(self, texts):
        """
        Predict sentiment using simple VADER thresholds (no training needed)
        
        Args:
            texts: List of text strings
        
        Returns:
            numpy array of predictions
        """
        predictions = [self.predict_vader_simple(text) for text in texts]
        return np.array(predictions)
    
    def save(self, filepath):
        """Save trained model"""
        if self.classifier is None:
            raise ValueError("Model not trained. Call train() first.")
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.classifier, f)
        
        print(f" Model saved to: {filepath}")
    
    def load(self, filepath):
        """Load trained model"""
        with open(filepath, 'rb') as f:
            self.classifier = pickle.load(f)
        
        print(f" Model loaded from: {filepath}")
        return self


def plot_confusion_matrix(y_true, y_pred, class_names, save_path, title):
    """
    Plot confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f" Confusion matrix saved to: {save_path}")
    plt.close()


def compare_models(simple_preds, trained_preds, true_labels, class_names):
    """
    Compare simple VADER vs trained VADER+LR
    """
    print(f"\n{'='*70}")
    print(f"Model Comparison")
    print(f"{'='*70}")
    
    # Simple VADER
    simple_acc = accuracy_score(true_labels, simple_preds)
    simple_f1 = f1_score(true_labels, simple_preds, average='weighted')
    
    print(f"\nSimple VADER (rule-based thresholds):")
    print(f"  Accuracy:  {simple_acc:.4f}")
    print(f"  F1-Score:  {simple_f1:.4f}")
    
    # Trained VADER + LR
    trained_acc = accuracy_score(true_labels, trained_preds)
    trained_f1 = f1_score(true_labels, trained_preds, average='weighted')
    
    print(f"\nVADER + Logistic Regression (trained):")
    print(f"  Accuracy:  {trained_acc:.4f}")
    print(f"  F1-Score:  {trained_f1:.4f}")
    
    print(f"\nImprovement from training:")
    print(f"  Accuracy:  +{(trained_acc - simple_acc):.4f} ({((trained_acc/simple_acc - 1)*100):.1f}%)")
    print(f"  F1-Score:  +{(trained_f1 - simple_f1):.4f} ({((trained_f1/simple_f1 - 1)*100):.1f}%)")


def main():
    parser = argparse.ArgumentParser(description='VADER Sentiment Baseline')
    
    parser.add_argument('--data', type=str, required=True,
                       help='Path to labeled tweets CSV')
    parser.add_argument('--output_dir', type=str, default='baseline/vader_results/',
                       help='Output directory for results')
    parser.add_argument('--mode', type=str, default='both', choices=['simple', 'trained', 'both'],
                       help='Mode: simple VADER, trained VADER+LR, or both')
    
    args = parser.parse_args()
    
    print(f"\n{'#'*70}")
    print(f"# VADER Sentiment Baseline")
    print(f"# Author: Boris Letic (Student 1)")
    print(f"# Purpose: Baseline comparison for BERT model")
    print(f"{'#'*70}\n")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print(f"Loading data from: {args.data}")
    df = pd.read_csv(args.data)
    
    # Map sentiment to labels
    sentiment_map = {'negative': 0, 'neutral': 1, 'positive': 2}
    df['label'] = df['sentiment'].map(sentiment_map)
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)
    
    print(f" Loaded {len(df)} labeled tweets")
    
    # Split data (80/20 train/val)
    train_size = int(0.8 * len(df))
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:]
    
    print(f"\nData split:")
    print(f"  Training: {len(train_df)} samples")
    print(f"  Validation: {len(val_df)} samples")
    
    # Initialize model
    model = VADERSentimentBaseline()
    
    class_names = ['Negative', 'Neutral', 'Positive']
    
    # Simple VADER predictions
    if args.mode in ['simple', 'both']:
        print(f"\n{'='*70}")
        print(f"Evaluating Simple VADER (rule-based)")
        print(f"{'='*70}")
        
        val_preds_simple = model.predict_simple(val_df['cleaned_text'].values)
        
        simple_acc = accuracy_score(val_df['label'].values, val_preds_simple)
        simple_f1 = f1_score(val_df['label'].values, val_preds_simple, average='weighted')
        
        print(f"\nResults:")
        print(f"  Validation Accuracy: {simple_acc:.4f}")
        print(f"  Weighted F1-Score:   {simple_f1:.4f}")
        
        print(f"\nClassification Report:")
        print(classification_report(val_df['label'].values, val_preds_simple, 
                                   target_names=class_names))
        
        # Plot confusion matrix
        cm_path = os.path.join(args.output_dir, 'confusion_matrix_simple.png')
        plot_confusion_matrix(val_df['label'].values, val_preds_simple, class_names, 
                            cm_path, 'VADER Simple - Confusion Matrix')
    
    # Trained VADER + LR
    if args.mode in ['trained', 'both']:
        print(f"\n{'='*70}")
        print(f"Training VADER + Logistic Regression")
        print(f"{'='*70}")
        
        model.train(
            train_texts=train_df['cleaned_text'].values,
            train_labels=train_df['label'].values
        )
        
        # Predictions
        val_preds_trained = model.predict(val_df['cleaned_text'].values)
        
        trained_acc = accuracy_score(val_df['label'].values, val_preds_trained)
        trained_f1 = f1_score(val_df['label'].values, val_preds_trained, average='weighted')
        
        print(f"\nResults:")
        print(f"  Validation Accuracy: {trained_acc:.4f}")
        print(f"  Weighted F1-Score:   {trained_f1:.4f}")
        
        print(f"\nClassification Report:")
        print(classification_report(val_df['label'].values, val_preds_trained, 
                                   target_names=class_names))
        
        # Plot confusion matrix
        cm_path = os.path.join(args.output_dir, 'confusion_matrix_trained.png')
        plot_confusion_matrix(val_df['label'].values, val_preds_trained, class_names, 
                            cm_path, 'VADER + Logistic Regression - Confusion Matrix')
        
        # Save model
        model_path = os.path.join(args.output_dir, 'vader_lr_model.pkl')
        model.save(model_path)
    
    # Compare both if running both modes
    if args.mode == 'both':
        val_preds_simple = model.predict_simple(val_df['cleaned_text'].values)
        val_preds_trained = model.predict(val_df['cleaned_text'].values)
        
        compare_models(val_preds_simple, val_preds_trained, 
                      val_df['label'].values, class_names)
    
    # Save predictions to CSV
    results_df = val_df.copy()
    
    if args.mode in ['simple', 'both']:
        results_df['vader_simple_pred'] = val_preds_simple
    
    if args.mode in ['trained', 'both']:
        results_df['vader_lr_pred'] = val_preds_trained
    
    results_path = os.path.join(args.output_dir, 'predictions.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\n Predictions saved to: {results_path}")
    
    print(f"\n{'='*70}")
    print(f"Baseline evaluation complete!")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
