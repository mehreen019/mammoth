# Setup and Run Guide for Hindi-Bangla NER Continual Learning

## Quick Setup (5 minutes)

### Step 1: Install Dependencies

```bash
# Install base requirements
pip install -r requirements.txt

# Install NLP-specific requirements
pip install -r requirements_nlp.txt
```

### Step 2: Verify Installation

```bash
python test_nlp_setup.py
```

This should print:
```
✓ PyTorch installed
✓ Transformers installed
✓ Datasets installed
✓ BERT model can be loaded
All dependencies are correctly installed!
```

### Step 3: Run Quick Test (5 minutes)

```bash
# Quick test with 1 epoch
python train_ner_continual.py --model er_nlp --buffer_size 100 --n_epochs 1 --batch_size 8
```

Expected output:
```
Task 0 (Hindi): ~60-70% accuracy
Task 1 (Bangla): ~50-60% accuracy
Forward Transfer: ~+2-5%
Backward Transfer: ~-10 to -20%
```

## Full Experiments (1.5-2 hours)

### Option A: Run All Experiments

```bash
python run_ner_experiments.py
```

This runs:
1. Experience Replay (ER) - ~40 min
2. Fine-tuning (SGD) - ~30 min
3. Joint Training - ~40 min

### Option B: Run Individual Experiments

```bash
# Experience Replay (best continual learning method)
python train_ner_continual.py --model er_nlp --buffer_size 200 --n_epochs 2

# Fine-tuning (baseline - shows catastrophic forgetting)
python train_ner_continual.py --model sgd --n_epochs 2

# Joint training (upper bound - not continual learning)
python train_ner_continual.py --model joint --n_epochs 2
```

## Understanding the Output

### During Training

You'll see:
```
Task 0/2 - Epoch 1/2
Training: 100% |████████| 32/32 [00:15<00:00, 2.13it/s, loss=1.23, lr=2e-05]
Evaluating: 100% |████████| 7/7 [00:02<00:00, 3.45it/s]

Task 0 Accuracy: 68.5%
```

### After Each Task

```
Evaluation on all tasks:
  Task 0 (Hindi): 68.5%
  Task 1 (Bangla): 0.0%  ← Not trained yet
```

### Final Results

```
================================================================================
FINAL RESULTS
================================================================================

Task Accuracies:
  After Task 0: [68.5, 0.0]
  After Task 1: [55.2, 62.3]  ← Task 0 dropped (forgetting!)

Average Accuracy: 58.75%

Forward Transfer: +3.2%
  → Learning Hindi helped with Bangla

Backward Transfer: -13.3%
  → Forgot some Hindi after learning Bangla

Forgetting: -13.3%
  → Maximum accuracy drop on previous tasks
```

## Interpreting Results

### Good Results (Experience Replay)
- Average Accuracy: 60-70%
- Backward Transfer: -5 to -15% (low forgetting)
- Forward Transfer: +2 to +5% (positive transfer)

### Bad Results (Fine-tuning)
- Average Accuracy: 50-60%
- Backward Transfer: -20 to -40% (high forgetting)
- Forward Transfer: 0 to +3%

### Upper Bound (Joint Training)
- Average Accuracy: 70-80%
- Backward Transfer: 0% (no forgetting - sees all data)
- Forward Transfer: N/A (not continual learning)

## Troubleshooting

### Issue: Low Accuracy (~10-20%)

**Solution**: This was the original bug! Make sure you're using the updated files:
- `datasets/seq_hindi_bangla_ner.py` (with task IDs)
- `train_ner_continual.py` (proper Mammoth integration)

### Issue: CUDA Out of Memory

**Solution**: Reduce batch size
```bash
python train_ner_continual.py --model er_nlp --batch_size 4 --buffer_size 100
```

### Issue: Too Slow

**Solution**: Reduce dataset size

Edit `datasets/seq_hindi_bangla_ner.py` line 158-159:
```python
# Change from:
hindi_train = load_dataset('wikiann', 'hi', split='train[:500]')
# To:
hindi_train = load_dataset('wikiann', 'hi', split='train[:200]')
```

Or reduce epochs:
```bash
python train_ner_continual.py --model er_nlp --n_epochs 1
```

### Issue: ModuleNotFoundError

**Solution**: Install missing dependencies
```bash
pip install -r requirements.txt
pip install -r requirements_nlp.txt
```

### Issue: Dataset Download Fails

**Solution**: The code will automatically use dummy data if WikiANN fails to download. You'll see:
```
Error loading Hindi WikiANN: ...
Using dummy data for demonstration...
```

This is fine for testing, but results won't be meaningful.

## Command-Line Arguments Reference

### Essential Arguments

- `--model`: Method to use
  - `er_nlp`: Experience Replay (recommended)
  - `sgd`: Fine-tuning baseline
  - `joint`: Joint training upper bound
  - `der`, `derpp`: Advanced replay methods

- `--buffer_size`: Replay buffer size (for ER methods)
  - Small: 100 (faster, less memory)
  - Medium: 200 (balanced)
  - Large: 500 (better performance, slower)

- `--n_epochs`: Epochs per task
  - Quick test: 1
  - Normal: 2
  - Full: 3-5

- `--batch_size`: Batch size
  - Small GPU: 4-8
  - Medium GPU: 16
  - Large GPU: 32

- `--lr`: Learning rate
  - Default: 2e-5 (good for BERT)
  - Range: 1e-5 to 5e-5

### Optional Arguments

- `--seed`: Random seed (default: 42)
- `--enable_other_metrics`: Enable FWT/BWT (default: 1)
- `--nowand`: Disable wandb logging (default: 1)
- `--device`: Device to use (default: auto-detect)

### Examples

```bash
# Quick test
python train_ner_continual.py --model er_nlp --n_epochs 1 --batch_size 8

# Full training
python train_ner_continual.py --model er_nlp --buffer_size 500 --n_epochs 3

# CPU only
python train_ner_continual.py --model er_nlp --device cpu

# Different method
python train_ner_continual.py --model derpp --buffer_size 200
```

## Expected Runtime

| Configuration | Time | Accuracy |
|--------------|------|----------|
| Quick test (1 epoch, batch=8) | ~5 min | ~50-60% |
| Normal (2 epochs, batch=16) | ~30-40 min | ~60-70% |
| Full (3 epochs, batch=16, buffer=500) | ~60-90 min | ~65-75% |
| All experiments (3 methods) | ~1.5-2 hours | Comparison |

## Files Created

After running, you'll have:
- Console output with metrics
- Logs in `results/` (if enabled)
- Model checkpoints (if `--savecheck` enabled)

## Next Steps

1. Run quick test to verify setup
2. Run full experiments
3. Compare results between methods
4. Present findings to professor

## Key Points for Professor

1. **Proper Continual Learning**
   - Sequential tasks: Hindi → Bangla
   - Evaluation on all previous tasks
   - Quantitative metrics: FWT, BWT, Forgetting

2. **Clear Demonstration**
   - ER reduces forgetting vs. fine-tuning
   - Joint training is upper bound
   - Concrete numbers to show improvement

3. **Reproducible**
   - Fixed seed (42)
   - Clear command-line interface
   - Documented configuration

## Support

If you encounter issues:
1. Check this guide's troubleshooting section
2. Verify dependencies with `python test_nlp_setup.py`
3. Try quick test first before full experiments
4. Check `NER_CONTINUAL_LEARNING_README.md` for more details

