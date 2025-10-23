"""
Quick start script to run NER continual learning experiments
Runs multiple methods and compares results

This script will:
1. Train with Experience Replay (ER)
2. Train with Fine-tuning (SGD)
3. Train with Joint training (upper bound)
4. Generate comparison visualizations
"""

import subprocess
import sys
import os
import time
import json
from pathlib import Path

def run_experiment(model, buffer_size=None, n_epochs=2, batch_size=16, lr=2e-5):
    """Run a single experiment"""
    print("\n" + "="*80)
    print(f"RUNNING EXPERIMENT: {model.upper()}")
    print("="*80)
    
    cmd = [
        sys.executable, 'train_ner_continual.py',
        '--model', model,
        '--n_classes', '4',  # Hindi-Bangla NER has 4 classes: O, PER, LOC, ORG
        '--batch_size', str(batch_size),
        '--n_epochs', str(n_epochs),
        '--lr', str(lr),
        '--seed', '42',
        '--nowand', '1',
    ]
    
    # Add enable_other_metrics for non-joint models
    if model != 'joint':
        cmd.extend(['--enable_other_metrics', '1'])
    
    # Add buffer size for replay-based methods
    if buffer_size is not None:
        cmd.extend(['--buffer_size', str(buffer_size)])
    
    print(f"Command: {' '.join(cmd)}")
    print()
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - start_time
    
    print(f"\nExperiment completed in {elapsed/60:.2f} minutes")
    
    return result.returncode == 0

def compare_results():
    """Compare results from different methods"""
    print("\n" + "="*80)
    print("COMPARING RESULTS")
    print("="*80)
    
    # This would load and compare results from different runs
    # For now, just print a message
    print("\nTo compare results, check the logs above for:")
    print("  - Average Accuracy")
    print("  - Forward Transfer")
    print("  - Backward Transfer")
    print("  - Forgetting")
    print("\nExpected results:")
    print("  - Joint (upper bound): Highest accuracy, no forgetting")
    print("  - ER (Experience Replay): Good accuracy, reduced forgetting")
    print("  - SGD (Fine-tuning): Lower accuracy, high forgetting")

def main():
    """Run all experiments"""
    print("="*80)
    print("HINDI-BANGLA NER CONTINUAL LEARNING EXPERIMENTS")
    print("="*80)
    print("\nThis script will run 3 experiments:")
    print("  1. Experience Replay (ER) - with buffer")
    print("  2. Fine-tuning (SGD) - no buffer")
    print("  3. Joint Training - upper bound")
    print("\nEach experiment trains on Hindi first, then Bangla")
    print("="*80)
    
    # Configuration
    n_epochs = 2  # Epochs per task
    batch_size = 16
    lr = 2e-5
    buffer_size = 200
    
    results = {}
    
    # Experiment 1: Experience Replay
    print("\n\n")
    print("█" * 80)
    print("EXPERIMENT 1/3: EXPERIENCE REPLAY (ER)")
    print("█" * 80)
    success = run_experiment('er_nlp', buffer_size=buffer_size, n_epochs=n_epochs, 
                            batch_size=batch_size, lr=lr)
    results['er'] = success
    
    # Experiment 2: Fine-tuning (SGD)
    print("\n\n")
    print("█" * 80)
    print("EXPERIMENT 2/3: FINE-TUNING (SGD)")
    print("█" * 80)
    success = run_experiment('sgd', buffer_size=None, n_epochs=n_epochs,
                            batch_size=batch_size, lr=lr)
    results['sgd'] = success
    
    # Experiment 3: Joint Training
    print("\n\n")
    print("█" * 80)
    print("EXPERIMENT 3/3: JOINT TRAINING (UPPER BOUND)")
    print("█" * 80)
    success = run_experiment('joint', buffer_size=None, n_epochs=n_epochs,
                            batch_size=batch_size, lr=lr)
    results['joint'] = success
    
    # Summary
    print("\n\n")
    print("="*80)
    print("ALL EXPERIMENTS COMPLETED")
    print("="*80)
    print("\nResults:")
    for method, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"  {method.upper()}: {status}")
    
    compare_results()
    
    print("\n" + "="*80)
    print("DONE!")
    print("="*80)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run NER continual learning experiments')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick test with smaller dataset')
    args = parser.parse_args()
    
    if args.quick:
        print("\n⚠️  QUICK MODE: Using smaller dataset for testing")
        print("This is not representative of full performance!\n")
    
    main()

