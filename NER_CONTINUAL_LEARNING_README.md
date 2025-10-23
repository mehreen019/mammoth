# Hindi-Bangla NER Continual Learning

This project implements continual learning for Named Entity Recognition (NER) across Hindi and Bangla languages using the Mammoth framework.

## Overview

The task is to learn NER sequentially:
1. **Task 0**: Hindi NER (recognize PER, LOC, ORG entities in Hindi text)
2. **Task 1**: Bangla NER (recognize PER, LOC, ORG entities in Bangla text)

The challenge is to learn Task 1 without forgetting Task 0 (catastrophic forgetting).

## Key Features

✅ **Proper Continual Learning**: Uses Mammoth's framework for sequential task learning  
✅ **Forward Transfer Analysis**: Measures if learning Hindi helps with Bangla  
✅ **Backward Transfer Analysis**: Measures forgetting of Hindi after learning Bangla  
✅ **Multiple Methods**: Compare Experience Replay, Fine-tuning, and Joint training  
✅ **Command-line Arguments**: Fully configurable via CLI  
✅ **Fast Training**: Optimized to run in ~2 hours  

## Quick Start

### Option 1: Run All Experiments (Recommended)

```bash
python run_ner_experiments.py
```

This will run 3 experiments:
1. **Experience Replay (ER)**: Uses a buffer to replay old examples
2. **Fine-tuning (SGD)**: Simple sequential training (baseline)
3. **Joint Training**: Train on all data together (upper bound)

### Option 2: Run Single Experiment

```bash
# Experience Replay
python train_ner_continual.py --model er_nlp --buffer_size 200 --n_epochs 2

# Fine-tuning (no buffer)
python train_ner_continual.py --model sgd --n_epochs 2

# Joint training (upper bound)
python train_ner_continual.py --model joint --n_epochs 2
```

### Option 3: Use Mammoth Directly

```bash
python main.py --model er_nlp --dataset seq-hindi-bangla-ner --buffer_size 200 --batch_size 16 --n_epochs 2 --lr 2e-5 --enable_other_metrics 1 --nowand 1
```

## Understanding the Results

### Metrics Explained

1. **Average Accuracy**: Overall performance across all tasks
   - Higher is better
   - Joint training should have the highest

2. **Forward Transfer (FWT)**: Does learning Hindi help with Bangla?
   - Positive = beneficial transfer
   - Negative = negative transfer
   - Zero = no transfer

3. **Backward Transfer (BWT)**: How much did we forget Hindi after learning Bangla?
   - Positive = improved on old tasks (rare)
   - Zero = no forgetting (ideal)
   - Negative = forgetting (common problem)

4. **Forgetting**: Maximum accuracy drop on previous tasks
   - Lower (more negative) = more forgetting
   - Zero = no forgetting (ideal)

### Expected Results

| Method | Avg Accuracy | Forgetting | Notes |
|--------|-------------|------------|-------|
| **Joint** | ~70-80% | 0% | Upper bound - sees all data |
| **ER (buffer=200)** | ~60-70% | -5 to -15% | Good balance |
| **SGD (fine-tune)** | ~50-60% | -20 to -40% | High forgetting |

### Sample Output

```
Task 0 (Hindi) Accuracy: 75.2%
Task 1 (Bangla) Accuracy: 68.4%

After Task 1:
  Task 0 (Hindi) Accuracy: 62.1%  ← Forgetting!
  Task 1 (Bangla) Accuracy: 68.4%

Metrics:
  Average Accuracy: 65.25%
  Forward Transfer: +2.3%  ← Learning Hindi helped with Bangla
  Backward Transfer: -13.1%  ← Forgot some Hindi
  Forgetting: -13.1%
```

## Configuration Options

### Key Arguments

- `--model`: Method to use (`er_nlp`, `sgd`, `joint`, `der`, `derpp`)
- `--buffer_size`: Size of replay buffer (for ER methods)
- `--batch_size`: Batch size (default: 16)
- `--n_epochs`: Epochs per task (default: 2)
- `--lr`: Learning rate (default: 2e-5)
- `--seed`: Random seed (default: 42)
- `--enable_other_metrics`: Enable FWT/BWT/Forgetting (default: 1)

### Example Configurations

```bash
# Fast test (smaller dataset)
python train_ner_continual.py --model er_nlp --buffer_size 100 --n_epochs 1

# Full training (better results)
python train_ner_continual.py --model er_nlp --buffer_size 500 --n_epochs 3

# Different methods
python train_ner_continual.py --model der --buffer_size 200  # Dark Experience Replay
python train_ner_continual.py --model derpp --buffer_size 200  # DER++
```

## Files Overview

- `train_ner_continual.py`: Main training script with CLI arguments
- `run_ner_experiments.py`: Run all experiments and compare
- `analyze_ner_results.py`: Generate visualizations (if needed)
- `datasets/seq_hindi_bangla_ner.py`: Dataset implementation
- `models/er_nlp.py`: Experience Replay model for NLP
- `backbone/bert_multilingual.py`: BERT multilingual backbone

## Dataset Details

- **Source**: WikiANN dataset (Hindi and Bangla)
- **Size**: 500 train + 100 test per language (for speed)
- **Classes**: O (Outside), PER (Person), LOC (Location), ORG (Organization)
- **Format**: Sentence-level classification (dominant entity type)

## Troubleshooting

### Low Accuracy (~10-20%)

This was the original problem! Fixed by:
- ✅ Proper task separation with task IDs
- ✅ Using Task-IL setting (same classes, different tasks)
- ✅ Correct data masking in store_masked_loaders

### Out of Memory

- Reduce `--batch_size` (try 8 or 4)
- Reduce buffer size (try 100)
- Use CPU: `--device cpu`

### Too Slow

- Reduce dataset size in `seq_hindi_bangla_ner.py` (change `split='train[:500]'` to `split='train[:200]'`)
- Reduce epochs: `--n_epochs 1`
- Reduce buffer: `--buffer_size 100`

## For Your Professor

### Key Points to Highlight

1. **Proper Continual Learning Setup**
   - Sequential learning: Hindi → Bangla
   - Proper task separation with task IDs
   - Evaluation on all previous tasks after each task

2. **Forward Transfer Analysis**
   - Measures if multilingual BERT's shared representations help
   - Positive FWT shows beneficial knowledge transfer

3. **Backward Transfer Analysis**
   - Measures catastrophic forgetting
   - ER significantly reduces forgetting vs. fine-tuning

4. **Concrete Results**
   - Quantitative metrics: Accuracy, FWT, BWT, Forgetting
   - Comparison of multiple methods
   - Clear demonstration of continual learning challenges

### Presentation Tips

1. Show the accuracy matrix (Task 0 vs Task 1 performance)
2. Highlight the forgetting in SGD vs. reduced forgetting in ER
3. Explain that Joint training is the upper bound (not continual learning)
4. Discuss the trade-off between memory (buffer size) and performance

## Time Estimate

- Single experiment: ~30-40 minutes
- All 3 experiments: ~1.5-2 hours
- Includes training + evaluation + metrics

## Citation

If using this code, cite:
- Mammoth framework: https://github.com/aimagelab/mammoth
- WikiANN dataset: Pan et al., 2017

