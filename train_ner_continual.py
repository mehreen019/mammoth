"""
Continual Learning Training Script for Hindi-Bangla NER
Uses Mammoth's framework for proper continual learning with forward/backward transfer analysis

Example usage:
    # Experience Replay with buffer
    python train_ner_continual.py --model er_nlp --dataset seq-hindi-bangla-ner --buffer_size 200 --batch_size 16 --n_epochs 2 --lr 2e-5 --enable_other_metrics 1
    
    # Fine-tuning (no replay)
    python train_ner_continual.py --model sgd --dataset seq-hindi-bangla-ner --batch_size 16 --n_epochs 2 --lr 2e-5 --enable_other_metrics 1
    
    # Joint training (upper bound)
    python train_ner_continual.py --model joint --dataset seq-hindi-bangla-ner --batch_size 16 --n_epochs 2 --lr 2e-5
"""

import sys
import os

# Add mammoth to path
mammoth_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, mammoth_path)

from main import main

if __name__ == '__main__':
    # Set default arguments for NER if not provided
    default_args = [
        '--backbone', 'bert-multilingual',
        '--dataset', 'seq-hindi-bangla-ner',
        '--model', 'er_nlp',
        '--buffer_size', '200',
        '--minibatch_size', '16',
        '--batch_size', '16',
        '--n_epochs', '2',
        '--lr', '2e-5',
        '--enable_other_metrics', '1',  # Enable forward/backward transfer metrics
        '--seed', '42',
        '--num_workers', '0',  # Avoid multiprocessing issues
    ]
    
    # Parse command line args, use defaults if not provided
    import argparse
    parser = argparse.ArgumentParser(description='Train Hindi-Bangla NER with Continual Learning')
    parser.add_argument('--model', type=str, default='er_nlp', 
                       help='Model to use (er_nlp, sgd, joint, etc.)')
    parser.add_argument('--buffer_size', type=int, default=200,
                       help='Buffer size for replay-based methods')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--n_epochs', type=int, default=2,
                       help='Number of epochs per task')
    parser.add_argument('--lr', type=float, default=2e-5,
                       help='Learning rate')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--enable_other_metrics', type=int, default=1,
                       help='Enable forward/backward transfer metrics')
    parser.add_argument('--nowand', type=int, default=1,
                       help='Disable wandb logging')
    
    # Parse known args
    args, unknown = parser.parse_known_args()
    
    # Build command line arguments
    cmd_args = [
        '--backbone', 'bert-multilingual',
        '--dataset', 'seq-hindi-bangla-ner',
        '--model', args.model,
        '--batch_size', str(args.batch_size),
        '--n_epochs', str(args.n_epochs),
        '--lr', str(args.lr),
        '--enable_other_metrics', str(args.enable_other_metrics),
        '--seed', str(args.seed),
        '--num_workers', '0',
        '--nowand', str(args.nowand),
    ]
    
    # Add buffer size for replay-based methods
    if args.model in ['er_nlp', 'er', 'der', 'derpp', 'xder']:
        cmd_args.extend(['--buffer_size', str(args.buffer_size)])
        cmd_args.extend(['--minibatch_size', str(args.batch_size)])
    
    # Add any unknown args
    cmd_args.extend(unknown)
    
    print("="*80)
    print("CONTINUAL LEARNING FOR HINDI-BANGLA NER")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Dataset: seq-hindi-bangla-ner")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs per task: {args.n_epochs}")
    print(f"Learning rate: {args.lr}")
    if args.model in ['er_nlp', 'er', 'der', 'derpp', 'xder']:
        print(f"Buffer size: {args.buffer_size}")
    print(f"Enable metrics: {args.enable_other_metrics}")
    print("="*80)
    print()
    
    # Override sys.argv and call main
    sys.argv = ['train_ner_continual.py'] + cmd_args
    
    main()

