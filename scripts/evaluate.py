"""
Comprehensive Evaluation Script
Evaluates BERT model vs VADER baseline and generates reports

Author: Boris Letić (Student 1)
Date: 2025-02-13
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, roc_auc_score
)
import argparse
import os
from datetime import datetime
import json

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


class ModelEvaluator:
    """
    Comprehensive model evaluation
    """
    
    def __init__(self, output_dir='results/evaluation'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.class_names = ['Negative', 'Neutral', 'Positive']
        self.results = {}
    
    def evaluate_model(self, y_true, y_pred, model_name):
        """
        Evaluate a single model
        """
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*60}")
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted'),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted')
        }
        
        # Print metrics
        print(f"\nMetrics:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name:20s}: {value:.4f}")
        
        # Classification report
        print(f"\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=self.class_names))
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Store results
        self.results[model_name] = {
            'metrics': metrics,
            'confusion_matrix': cm.tolist(),
            'classification_report': classification_report(
                y_true, y_pred, target_names=self.class_names, output_dict=True
            )
        }
        
        return metrics, cm
    
    def plot_confusion_matrix(self, cm, model_name):
        """
        Plot confusion matrix
        """
        plt.figure(figsize=(8, 6))
        
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Count'}
        )
        
        plt.xlabel('Predicted', fontweight='bold')
        plt.ylabel('True', fontweight='bold')
        plt.title(f'Confusion Matrix - {model_name}', fontweight='bold', fontsize=14)
        
        # Add accuracy text
        accuracy = np.trace(cm) / np.sum(cm)
        plt.text(
            1.5, -0.3,
            f'Accuracy: {accuracy:.2%}',
            ha='center',
            fontsize=12,
            fontweight='bold'
        )
        
        plt.tight_layout()
        
        save_path = os.path.join(self.output_dir, f'cm_{model_name.lower().replace(" ", "_")}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix saved: {save_path}")
        
        plt.close()
    
    def plot_comparison(self):
        """
        Plot comparison of all models
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Extract metrics for all models
        models = list(self.results.keys())
        metrics_to_plot = ['accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted']
        
        # Prepare data
        data = {metric: [] for metric in metrics_to_plot}
        
        for model in models:
            for metric in metrics_to_plot:
                data[metric].append(self.results[model]['metrics'][metric])
        
        # Plot 1: Bar chart
        x = np.arange(len(models))
        width = 0.2
        
        for i, metric in enumerate(metrics_to_plot):
            axes[0].bar(
                x + i * width,
                data[metric],
                width,
                label=metric.replace('_', ' ').title()
            )
        
        axes[0].set_xlabel('Model', fontweight='bold')
        axes[0].set_ylabel('Score', fontweight='bold')
        axes[0].set_title('Model Comparison - All Metrics', fontweight='bold', fontsize=14)
        axes[0].set_xticks(x + width * 1.5)
        axes[0].set_xticklabels(models)
        axes[0].legend()
        axes[0].set_ylim(0, 1)
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Accuracy comparison
        accuracy_scores = [self.results[model]['metrics']['accuracy'] for model in models]
        colors = ['#e74c3c', '#3498db', '#2ecc71'][:len(models)]
        
        axes[1].barh(models, accuracy_scores, color=colors, alpha=0.7)
        axes[1].set_xlabel('Accuracy', fontweight='bold')
        axes[1].set_title('Model Accuracy Comparison', fontweight='bold', fontsize=14)
        axes[1].set_xlim(0, 1)
        axes[1].grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (model, acc) in enumerate(zip(models, accuracy_scores)):
            axes[1].text(acc + 0.02, i, f'{acc:.2%}', va='center', fontweight='bold')
        
        plt.tight_layout()
        
        save_path = os.path.join(self.output_dir, 'model_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Model comparison saved: {save_path}")
        
        plt.close()
    
    def analyze_errors(self, y_true, y_pred, texts, model_name, top_k=20):
        """
        Analyze top-K errors
        """
        print(f"\n{'='*60}")
        print(f"Error Analysis - {model_name}")
        print(f"{'='*60}")
        
        # Find incorrect predictions
        incorrect_mask = y_true != y_pred
        incorrect_indices = np.where(incorrect_mask)[0]
        
        if len(incorrect_indices) == 0:
            print("No errors found! Perfect classification.")
            return
        
        print(f"\nTotal errors: {len(incorrect_indices)} ({len(incorrect_indices)/len(y_true)*100:.2f}%)")
        
        # Analyze error types
        error_types = {}
        for i in incorrect_indices:
            true_label = self.class_names[y_true[i]]
            pred_label = self.class_names[y_pred[i]]
            error_type = f"{true_label} → {pred_label}"
            
            if error_type not in error_types:
                error_types[error_type] = []
            error_types[error_type].append(i)
        
        print(f"\nError Distribution:")
        for error_type, indices in sorted(error_types.items(), key=lambda x: len(x[1]), reverse=True):
            count = len(indices)
            pct = (count / len(incorrect_indices)) * 100
            print(f"  {error_type:25s}: {count:4d} ({pct:5.1f}%)")
        
        # Sample errors
        print(f"\nTop {min(top_k, len(incorrect_indices))} Error Examples:")
        
        sample_indices = np.random.choice(incorrect_indices, min(top_k, len(incorrect_indices)), replace=False)
        
        for idx in sample_indices[:10]:  # Show first 10
            true_label = self.class_names[y_true[idx]]
            pred_label = self.class_names[y_pred[idx]]
            text = texts[idx] if idx < len(texts) else "N/A"
            
            print(f"\n  [{idx}] True: {true_label}, Predicted: {pred_label}")
            print(f"      Text: {text[:80]}...")
        
        # Save error analysis
        error_df = pd.DataFrame({
            'index': incorrect_indices,
            'true_label': [self.class_names[y_true[i]] for i in incorrect_indices],
            'pred_label': [self.class_names[y_pred[i]] for i in incorrect_indices],
            'text': [texts[i] if i < len(texts) else "N/A" for i in incorrect_indices]
        })
        
        error_path = os.path.join(self.output_dir, f'errors_{model_name.lower().replace(" ", "_")}.csv')
        error_df.to_csv(error_path, index=False)
        print(f"\n✓ Error analysis saved: {error_path}")
    
    def save_results(self):
        """
        Save evaluation results to JSON
        """
        results_path = os.path.join(
            self.output_dir,
            f'evaluation_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        )
        
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n✓ Results saved: {results_path}")
        
        return results_path
    
    def print_summary(self):
        """
        Print final summary
        """
        print(f"\n{'='*60}")
        print(f"EVALUATION SUMMARY")
        print(f"{'='*60}")
        
        print(f"\nModels Evaluated: {len(self.results)}")
        
        # Best model by accuracy
        best_model = max(self.results.items(), key=lambda x: x[1]['metrics']['accuracy'])
        print(f"\nBest Model (by accuracy): {best_model[0]}")
        print(f"  Accuracy: {best_model[1]['metrics']['accuracy']:.4f}")
        print(f"  F1-Score: {best_model[1]['metrics']['f1_weighted']:.4f}")
        
        # Comparison table
        print(f"\nModel Comparison:")
        print(f"{'Model':<25} {'Accuracy':>10} {'F1-Score':>10} {'Precision':>10} {'Recall':>10}")
        print(f"{'-'*70}")
        
        for model_name, result in self.results.items():
            metrics = result['metrics']
            print(f"{model_name:<25} {metrics['accuracy']:>10.4f} {metrics['f1_weighted']:>10.4f} "
                  f"{metrics['precision_weighted']:>10.4f} {metrics['recall_weighted']:>10.4f}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate sentiment analysis models')
    
    parser.add_argument('--predictions', type=str, required=True,
                       help='CSV file with predictions')
    parser.add_argument('--output_dir', type=str, default='results/evaluation',
                       help='Output directory for evaluation results')
    parser.add_argument('--top_k_errors', type=int, default=20,
                       help='Number of top errors to analyze')
    
    args = parser.parse_args()
    
    print(f"\n{'#'*60}")
    print(f"# Model Evaluation")
    print(f"# Author: Boris Letić (Student 1)")
    print(f"# Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*60}")
    
    # Load predictions
    print(f"\nLoading predictions from: {args.predictions}")
    df = pd.read_csv(args.predictions)
    
    print(f"✓ Loaded {len(df)} predictions")
    
    # Initialize evaluator
    evaluator = ModelEvaluator(output_dir=args.output_dir)
    
    # Evaluate each model
    y_true = df['label'].values
    texts = df['cleaned_text'].values if 'cleaned_text' in df.columns else df['text'].values
    
    # VADER Simple
    if 'vader_simple_pred' in df.columns:
        y_pred_vader_simple = df['vader_simple_pred'].values
        metrics, cm = evaluator.evaluate_model(y_true, y_pred_vader_simple, "VADER Simple")
        evaluator.plot_confusion_matrix(cm, "VADER Simple")
        evaluator.analyze_errors(y_true, y_pred_vader_simple, texts, "VADER Simple", args.top_k_errors)
    
    # VADER + LR
    if 'vader_lr_pred' in df.columns:
        y_pred_vader_lr = df['vader_lr_pred'].values
        metrics, cm = evaluator.evaluate_model(y_true, y_pred_vader_lr, "VADER + LR")
        evaluator.plot_confusion_matrix(cm, "VADER + LR")
        evaluator.analyze_errors(y_true, y_pred_vader_lr, texts, "VADER + LR", args.top_k_errors)
    
    # BERT (if available)
    if 'bert_pred' in df.columns:
        y_pred_bert = df['bert_pred'].values
        metrics, cm = evaluator.evaluate_model(y_true, y_pred_bert, "BERT Fine-tuned")
        evaluator.plot_confusion_matrix(cm, "BERT Fine-tuned")
        evaluator.analyze_errors(y_true, y_pred_bert, texts, "BERT Fine-tuned", args.top_k_errors)
    
    # Plot comparisons
    if len(evaluator.results) > 1:
        evaluator.plot_comparison()
    
    # Save results
    evaluator.save_results()
    
    # Print summary
    evaluator.print_summary()
    
    print(f"\n{'='*60}")
    print(f"Evaluation complete! 🎉")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
