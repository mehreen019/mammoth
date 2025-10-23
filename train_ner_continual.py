"""
Continual Learning Training Script for Hindi-Bangla NER
Uses Mammoth's framework for proper continual learning with forward/backward transfer analysis

Example usage:
    # Experience Replay with buffer
    python train_ner_continual.py --model er_nlp --buffer_size 200 --batch_size 16 --n_epochs 2 --lr 2e-5

    # Fine-tuning (no replay)
    python train_ner_continual.py --model sgd --batch_size 16 --n_epochs 2 --lr 2e-5

    # Joint training (upper bound)
    python train_ner_continual.py --model joint --batch_size 16 --n_epochs 2 --lr 2e-5

    # Quick test
    python train_ner_continual.py --model er_nlp --n_epochs 1 --batch_size 8
"""

import sys
import os

# Add mammoth to path
mammoth_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, mammoth_path)

from main import main

if __name__ == '__main__':
    # Disable wandb by default (can be overridden by setting WANDB_MODE env var)
    if 'WANDB_MODE' not in os.environ:
        os.environ['WANDB_MODE'] = 'disabled'

    # Check if user provided any arguments
    user_args = sys.argv[1:]

    # Extract model from user args to determine if we need buffer
    model = 'er_nlp'  # default
    for i, arg in enumerate(user_args):
        if arg == '--model' and i + 1 < len(user_args):
            model = user_args[i + 1]
            break

    # Models that don't need buffer
    NO_BUFFER_MODELS = ['sgd', 'joint', 'ewc_on', 'si', 'lwf', 'lwf-mc']

    # Set default arguments for NER
    default_args = {
        '--backbone': 'bert-multilingual',
        '--dataset': 'seq-hindi-bangla-ner',
        '--n_classes': '4',  # Hindi-Bangla NER has 4 classes: O, PER, LOC, ORG
        '--model': model,
        '--batch_size': '16',
        '--n_epochs': '2',
        '--lr': '2e-5',
        '--enable_other_metrics': '1',
        '--seed': '42',
        '--num_workers': '0',
    }

    # Add buffer args only for rehearsal-based methods
    if model not in NO_BUFFER_MODELS:
        default_args['--buffer_size'] = '200'
        default_args['--minibatch_size'] = '16'

    # Build final argument list
    final_args = []

    # Add defaults first
    for key, value in default_args.items():
        # Check if user provided this argument
        if key not in user_args:
            final_args.extend([key, value])

    # Add user arguments (they will override defaults)
    final_args.extend(user_args)

    # Parse to show user what's being used
    model = None
    buffer_size = None
    batch_size = None
    n_epochs = None
    lr = None

    i = 0
    while i < len(final_args):
        if final_args[i] == '--model' and i + 1 < len(final_args):
            model = final_args[i + 1]
        elif final_args[i] == '--buffer_size' and i + 1 < len(final_args):
            buffer_size = final_args[i + 1]
        elif final_args[i] == '--batch_size' and i + 1 < len(final_args):
            batch_size = final_args[i + 1]
        elif final_args[i] == '--n_epochs' and i + 1 < len(final_args):
            n_epochs = final_args[i + 1]
        elif final_args[i] == '--lr' and i + 1 < len(final_args):
            lr = final_args[i + 1]
        i += 1

    # Add minibatch_size for replay methods if not already present
    if model in ['er_nlp', 'er', 'der', 'derpp', 'xder']:
        if '--minibatch_size' not in final_args and batch_size:
            final_args.extend(['--minibatch_size', batch_size])

    print("="*80)
    print("CONTINUAL LEARNING FOR HINDI-BANGLA NER")
    print("="*80)
    print(f"Model: {model}")
    print(f"Dataset: seq-hindi-bangla-ner")
    print(f"Batch size: {batch_size}")
    print(f"Epochs per task: {n_epochs}")
    print(f"Learning rate: {lr}")
    if model in ['er_nlp', 'er', 'der', 'derpp', 'xder']:
        print(f"Buffer size: {buffer_size}")
    print("="*80)
    print()

    # Override sys.argv and call main
    sys.argv = ['train_ner_continual.py'] + final_args

    main()

