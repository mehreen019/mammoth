# Running Hindi-Bangla NER on Google Colab

Follow these steps to run the BERT Multilingual NER model on Google Colab and generate visualizations for your professor.

---

## Step 1: Upload Files to Colab

1. Open Google Colab: https://colab.research.google.com/
2. Create a new notebook
3. Click on the folder icon on the left sidebar
4. Upload your entire `mammoth` project folder, OR use the method below:

### Option A: Upload via Google Drive (Recommended)
```python
from google.colab import drive
drive.mount('/content/drive')

# Navigate to your project folder
%cd /content/drive/MyDrive/mammoth
```

### Option B: Clone from GitHub (if you have it there)
```python
!git clone https://github.com/YOUR_USERNAME/mammoth.git
%cd mammoth
```

### Option C: Direct Upload
- Upload the entire `mammoth` folder using the file browser

---

## Step 2: Install Dependencies

Run this in a Colab cell:

```python
!pip install -q torch transformers scikit-learn matplotlib seaborn tqdm
```

OR install from requirements file:

```python
!pip install -q -r requirements_colab.txt
```

---

## Step 3: Verify GPU is Enabled

Check if GPU is available (recommended for faster training):

```python
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# If you see 'cpu', go to Runtime > Change runtime type > Hardware accelerator > GPU
```

---

## Step 4: Run the Training Script

Execute the training script:

```python
!python colab_train_ner.py
```

This will:
- Load the BERT Multilingual model
- Train on dummy Hindi-Bangla NER data (5 epochs)
- Generate training/validation metrics
- Create visualizations automatically
- Save everything to the `results/` folder

---

## Step 5: View the Results

After training completes, you'll see:

### Console Output:
- Training progress with loss and accuracy
- Per-epoch summaries
- Classification report
- Example predictions with tokens

### Generated Visualizations (saved in `results/` folder):

1. **`training_history.png`** - Training and validation loss/accuracy curves
2. **`confusion_matrix.png`** - Shows which NER tags are confused
3. **`per_class_metrics.png`** - Precision, Recall, F1-score for each tag
4. **`bert_multilingual_ner.pth`** - Saved model weights

---

## Step 6: Display Images in Colab

To display the generated images directly in your notebook:

```python
from IPython.display import Image, display

# Display training history
print("Training History:")
display(Image('results/training_history.png'))

print("\nConfusion Matrix:")
display(Image('results/confusion_matrix.png'))

print("\nPer-Class Metrics:")
display(Image('results/per_class_metrics.png'))
```

---

## Step 7: Download Results

Download all results to show your professor:

```python
# Zip the results folder
!zip -r results.zip results/

# Download in Colab
from google.colab import files
files.download('results.zip')
```

---

## Customization (Optional)

### Change Training Parameters

Edit these values in `colab_train_ner.py`:

```python
BATCH_SIZE = 16          # Increase if you have more GPU memory
EPOCHS = 5               # More epochs = longer training
LEARNING_RATE = 2e-5     # Learning rate for optimizer
NUM_TRAIN_SAMPLES = 200  # Number of training samples
NUM_VAL_SAMPLES = 50     # Number of validation samples
```

### Use Your Own Dataset

Replace the `DummyNERDataset` class in `colab_train_ner.py` with your actual Hindi-Bangla NER dataset. Make sure your dataset returns:
- `input_ids`: Tokenized input
- `attention_mask`: Attention mask
- `labels`: NER tags for each token

---

## Quick Commands Summary

Copy-paste these commands in order:

```python
# 1. Install dependencies
!pip install -q torch transformers scikit-learn matplotlib seaborn tqdm

# 2. Check GPU
import torch
print(f"GPU Available: {torch.cuda.is_available()}")

# 3. Run training
!python colab_train_ner.py

# 4. Display results
from IPython.display import Image, display
display(Image('results/training_history.png'))
display(Image('results/confusion_matrix.png'))
display(Image('results/per_class_metrics.png'))

# 5. Download results
!zip -r results.zip results/
from google.colab import files
files.download('results.zip')
```

---

## Expected Output

You should see:
- Training progress bars
- Epoch summaries showing loss and accuracy
- Final classification report
- 3 example predictions with tokens and tags
- All visualizations saved to `results/` folder

**Training time:** ~3-5 minutes with GPU, ~15-20 minutes without GPU

---

## Troubleshooting

### Error: "No module named 'backbone'"
- Make sure you're in the `mammoth` directory: `%cd mammoth`

### Error: "transformers library is required"
- Run: `!pip install transformers`

### Error: "CUDA out of memory"
- Reduce `BATCH_SIZE` in the script
- Or disable GPU and use CPU (slower)

### Images not showing
- Make sure the script completed successfully
- Check if `results/` folder exists: `!ls results/`

---

## Questions?

If you have any issues, check:
1. You're in the correct directory (`mammoth/`)
2. All files are uploaded correctly
3. Dependencies are installed
4. GPU is enabled (optional but recommended)

Good luck with your demo!
