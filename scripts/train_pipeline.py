"""
End-to-End Training Pipeline for Student 1 (Boris)
Runs complete workflow: preprocessing → baseline → BERT training → evaluation

Author: Boris Letić
Date: 2025-02-13
"""

import subprocess
import os
import sys
import time
from datetime import datetime
import json

class TrainingPipeline:
    """
    Complete training pipeline for sentiment analysis
    """
    
    def __init__(self, config):
        self.config = config
        self.results = {}
        self.start_time = None
        
    def print_header(self, title):
        """Print formatted header"""
        print(f"\n{'='*70}")
        print(f"  {title}")
        print(f"{'='*70}\n")
    
    def run_command(self, cmd, description):
        """Run shell command and track time"""
        self.print_header(description)
        print(f"Command: {cmd}\n")
        
        start = time.time()
        
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                check=True,
                capture_output=True,
                text=True
            )
            
            elapsed = time.time() - start
            
            print(result.stdout)
            if result.stderr:
                print("Warnings/Errors:")
                print(result.stderr)
            
            print(f"\n✓ Completed in {elapsed:.2f} seconds")
            
            return True, elapsed
            
        except subprocess.CalledProcessError as e:
            elapsed = time.time() - start
            print(f"\n✗ Failed after {elapsed:.2f} seconds")
            print(f"Error: {e.stderr}")
            return False, elapsed
    
    def step1_preprocessing(self):
        """Step 1: Preprocess tweets"""
        cmd = f"""python scripts/preprocessing.py \
            --input {self.config['tweets_file']} \
            --output_dir {self.config['processed_dir']} \
            --text_column text \
            --datetime_column datetime"""
        
        success, elapsed = self.run_command(cmd, "STEP 1: Preprocessing Tweets")
        
        self.results['preprocessing'] = {
            'success': success,
            'time': elapsed
        }
        
        return success
    
    def step2_vader_baseline(self):
        """Step 2: Train VADER baseline"""
        cmd = f"""python baseline/vader_baseline.py \
            --data {self.config['processed_dir']}/tweets_labeled.csv \
            --output_dir {self.config['baseline_dir']} \
            --mode both"""
        
        success, elapsed = self.run_command(cmd, "STEP 2: VADER Baseline Training")
        
        self.results['vader_baseline'] = {
            'success': success,
            'time': elapsed
        }
        
        return success
    
    def step3_bert_training(self):
        """Step 3: BERT fine-tuning"""
        cmd = f"""python scripts/sentiment_branch.py \
            --data {self.config['processed_dir']}/tweets_labeled.csv \
            --output_dir {self.config['bert_dir']} \
            --epochs {self.config['bert_epochs']} \
            --batch_size {self.config['bert_batch_size']} \
            --lr {self.config['bert_lr']} \
            --freeze_layers {self.config['bert_freeze_layers']}"""
        
        success, elapsed = self.run_command(cmd, "STEP 3: BERT Fine-tuning")
        
        self.results['bert_training'] = {
            'success': success,
            'time': elapsed
        }
        
        return success
    
    def step4_embedding_extraction(self):
        """Step 4: Extract sentiment embeddings"""
        cmd = f"""python models/gru_sentiment_embedding.py \
            --sentiment_features {self.config['processed_dir']}/sentiment_features_train.csv \
            --bert_model {self.config['bert_dir']}/bert_sentiment_best.pt \
            --window_size {self.config['window_size']} \
            --output_dir {self.config['embedding_dir']}"""
        
        success, elapsed = self.run_command(cmd, "STEP 4: Sentiment Embedding Extraction")
        
        self.results['embedding_extraction'] = {
            'success': success,
            'time': elapsed
        }
        
        return success
    
    def save_results(self):
        """Save pipeline results"""
        total_time = time.time() - self.start_time
        
        results_summary = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'steps': self.results,
            'total_time': total_time,
            'total_time_formatted': f"{total_time/60:.2f} minutes"
        }
        
        output_file = os.path.join(
            self.config['results_dir'],
            f'training_pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        )
        
        os.makedirs(self.config['results_dir'], exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        print(f"\n✓ Results saved to: {output_file}")
        
        return results_summary
    
    def print_summary(self, results_summary):
        """Print final summary"""
        self.print_header("TRAINING PIPELINE SUMMARY")
        
        print(f"Total Time: {results_summary['total_time_formatted']}")
        print(f"\nStep Results:")
        
        for step_name, step_result in self.results.items():
            status = "✓ SUCCESS" if step_result['success'] else "✗ FAILED"
            time_str = f"{step_result['time']:.2f}s"
            print(f"  {step_name:25s} {status:12s} ({time_str})")
        
        # Check if all succeeded
        all_success = all(step['success'] for step in self.results.values())
        
        if all_success:
            print(f"\n{'='*70}")
            print(f"  🎉 ALL STEPS COMPLETED SUCCESSFULLY!")
            print(f"{'='*70}\n")
            
            print("Next Steps:")
            print("  1. Check baseline/vader_results/ for VADER results")
            print("  2. Check models/bert_sentiment/ for BERT model & plots")
            print("  3. Check models/sentiment_embeddings/ for GRU embeddings")
            print("  4. Review training_history.png and confusion_matrix.png")
            print("  5. Integrate with Bogdan's price branch for multimodal fusion")
        else:
            print(f"\n{'='*70}")
            print(f"  ⚠️ SOME STEPS FAILED - CHECK LOGS ABOVE")
            print(f"{'='*70}\n")
    
    def run(self):
        """Run complete pipeline"""
        print(f"\n{'#'*70}")
        print(f"#  SENTIMENT ANALYSIS - END-TO-END TRAINING PIPELINE")
        print(f"#  Author: Boris Letić (Student 1)")
        print(f"#  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'#'*70}")
        
        self.start_time = time.time()
        
        # Step 1: Preprocessing
        if not self.step1_preprocessing():
            print("\n✗ Preprocessing failed. Aborting pipeline.")
            return False
        
        # Step 2: VADER Baseline
        if not self.step2_vader_baseline():
            print("\n⚠️ VADER baseline failed, but continuing...")
        
        # Step 3: BERT Training
        if not self.step3_bert_training():
            print("\n✗ BERT training failed. Aborting pipeline.")
            return False
        
        # Step 4: Embedding Extraction
        if not self.step4_embedding_extraction():
            print("\n⚠️ Embedding extraction failed, but models are trained.")
        
        # Save and print results
        results_summary = self.save_results()
        self.print_summary(results_summary)
        
        return True


def main():
    """Main entry point"""
    
    # Configuration
    config = {
        # Input files
        'tweets_file': 'data/bitcoin_tweets_10k.csv',
        
        # Output directories
        'processed_dir': 'data/processed',
        'baseline_dir': 'baseline/vader_results',
        'bert_dir': 'models/bert_sentiment',
        'embedding_dir': 'models/sentiment_embeddings',
        'results_dir': 'results',
        
        # BERT hyperparameters
        'bert_epochs': 3,
        'bert_batch_size': 32,
        'bert_lr': 2e-5,
        'bert_freeze_layers': 8,
        
        # GRU parameters
        'window_size': 168  # 7 days
    }
    
    # Create pipeline
    pipeline = TrainingPipeline(config)
    
    # Run
    success = pipeline.run()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
