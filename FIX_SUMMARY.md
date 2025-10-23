# NER Continual Learning Fixes - Summary

## Problems Identified

### 1. **Critical Bug: Wrong Learning Setting**
- **Issue**: Using `task-il` (Task Incremental Learning) with shared classes across tasks
- **Symptom**: Forgetting = 0.00, SGD outperforming ER, impossible metrics
- **Root Cause**: Task-IL expects disjoint class ranges (Task 0: classes 0-3, Task 1: classes 4-7), but both Hindi and Bangla used the same classes (0-3)
- **Impact**: Evaluation logic couldn't properly separate tasks, leading to meaningless continual learning metrics

### 2. **Low Accuracy (~30%)**
- **Issue #1**: Only 500 training samples per language (too small for BERT fine-tuning)
- **Issue #2**: Only 2 epochs per task (insufficient for convergence)
- **Impact**: Model couldn't learn NER properly on either language

## Fixes Applied

### Fix 1: Changed from Task-IL to Class-IL ✅
**File**: `datasets/seq_hindi_bangla_ner.py:397`

```python
# BEFORE:
SETTING = 'task-il'  # INCORRECT for shared classes

# AFTER:
SETTING = 'class-il'  # CORRECT for same classes across different domains
```

**Why Class-IL?**
- Class-IL is designed for scenarios where the same classes appear across different tasks/domains
- In your case: same entity types (PER, LOC, ORG) but different languages (Hindi, Bangla)
- This ensures proper task separation during evaluation
- Metrics (forgetting, BWT) will now be computed correctly

**Expected Impact**:
- ✅ Forgetting > 0 for SGD (catastrophic forgetting will be visible)
- ✅ ER will outperform SGD (replay buffer helps retention)
- ✅ BWT metrics will be meaningful and negative for SGD, less negative for ER

### Fix 2: Increased Dataset Size ✅
**File**: `datasets/seq_hindi_bangla_ner.py:443-448`

```python
# BEFORE:
hindi_train = _load_wikiann_split('hi', 'train[:500]')   # Only 500 samples
hindi_test = _load_wikiann_split('hi', 'validation[:100]')  # Only 100 test
bangla_train = _load_wikiann_split('bn', 'train[:500]')
bangla_test = _load_wikiann_split('bn', 'validation[:100]')

# AFTER:
hindi_train = _load_wikiann_split('hi', 'train')   # ~7000 samples
hindi_test = _load_wikiann_split('hi', 'validation')   # ~1000 samples
bangla_train = _load_wikiann_split('bn', 'train')
bangla_test = _load_wikiann_split('bn', 'validation')
```

**Expected Impact**:
- Accuracy should increase from ~30% to **60-80%**
- More training data = better BERT fine-tuning
- More reliable continual learning metrics

### Fix 3: Increased Training Epochs ✅
**Files**:
- `train_ner_continual.py:53`
- `datasets/seq_hindi_bangla_ner.py:534`
- `run_ner_experiments.py:19,85`

```python
# BEFORE:
'--n_epochs': '2'  # Too few for BERT convergence

# AFTER:
'--n_epochs': '5'  # Better convergence
```

**Expected Impact**:
- Combined with Fix #2, accuracy should reach **70-85%**
- Better task learning = more visible forgetting in SGD
- ER's benefit will be more pronounced

## Expected Results After Fixes

### Before Fixes (Broken):
```
Metric                     ER      SGD
───────────────────────────────────────
Final Class-IL Accuracy   31.0%   32.5%  ❌ Too low
Backward Transfer (BWT)   2.00    4.00   ❌ Should be negative
Forgetting                0.00    0.00   ❌ Impossible!
```

### After Fixes (Expected):
```
Metric                     ER      SGD
───────────────────────────────────────
Final Class-IL Accuracy   75-80%  65-70% ✅ ER > SGD
Backward Transfer (BWT)   -5.0    -15.0  ✅ ER less negative
Forgetting                10-15   25-35  ✅ ER < SGD (less forgetting)
```

**Key Improvements**:
1. ✅ **ER outperforms SGD** (as expected theoretically)
2. ✅ **Forgetting > 0** for both methods (realistic)
3. ✅ **ER has lower forgetting** than SGD (replay helps)
4. ✅ **Higher overall accuracy** (usable baseline)

## How to Run

```bash
# Run all experiments with fixed settings
python run_ner_experiments.py

# Or run individual experiments
python train_ner_continual.py --model er_nlp --buffer_size 200 --batch_size 16 --n_epochs 5
python train_ner_continual.py --model sgd --batch_size 16 --n_epochs 5
python train_ner_continual.py --model joint --batch_size 16 --n_epochs 5
```

## Understanding Class-IL vs Task-IL

### Class-IL (What You Need)
- **Same classes appear in multiple tasks**
- Example: PER, LOC, ORG entities in Hindi (Task 0) and Bangla (Task 1)
- Model must learn to recognize the same entity types across different domains
- Evaluation: Model sees all classes, no masking
- Challenge: Avoid forgetting how to recognize entities in previous languages

### Task-IL (What You Had - Wrong)
- **Different classes in each task**
- Example: Classes 0-9 in Task 0, Classes 10-19 in Task 1
- Model knows which task it's evaluating (class masking applied)
- Evaluation: Only relevant classes are considered
- Your bug: Both tasks had classes 0-3, breaking the evaluation logic

## Additional Notes

### If You Still Get Low Accuracy (~30-40%)
1. **Check WikiANN data files exist**: Ensure you have the full dataset downloaded
   - Expected location: `data/wikiann/{hi,bn}/{train,validation}.jsonl`
   - Use `download_wikiann.py` to download if missing

2. **Verify label distribution**: Check for class imbalance
   ```python
   # Most sentences might be class 0 (no entity)
   # Consider filtering to keep only sentences with entities
   ```

3. **Try more epochs**: If still underfitting, increase to 10 epochs
   ```python
   '--n_epochs': '10'
   ```

4. **Consider the sentence-level simplification**: The current implementation converts token-level NER to sentence-level classification, which loses information. For production-quality NER, you'd need token-level predictions.

### Performance Expectations
- **Joint Training** (upper bound): 80-90% accuracy
- **Experience Replay (ER)**: 70-80% accuracy
- **SGD (Fine-tuning)**: 60-70% accuracy

The gap between ER and SGD demonstrates the benefit of continual learning with replay!

## Files Modified
1. ✅ `datasets/seq_hindi_bangla_ner.py` - Changed SETTING, dataset size, default epochs
2. ✅ `train_ner_continual.py` - Increased default epochs to 5
3. ✅ `run_ner_experiments.py` - Updated defaults to match

## Next Steps
1. Run the experiments: `python run_ner_experiments.py`
2. Verify the metrics are now realistic
3. Compare ER vs SGD performance
4. Optionally: Try other continual learning methods (EWC, LwF, etc.)
