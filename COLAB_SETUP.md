# Colab Setup Instructions for WikiANN NER Dataset

## Problem
Colab's Python 3.12 runtime has incompatible versions of `dill` and `multiprocess` that cause RLock pickling errors when loading HuggingFace datasets. The code has been modified to ONLY use local files and will NOT attempt to download from HuggingFace.

## Solution
Upload pre-downloaded local dataset files to Colab. The files are already prepared in your local `data/wikiann/` directory.

---

## Quick Start (Recommended)

### 1. Use Google Drive (Persistent storage)

This is the best approach because files persist across Colab sessions.

**In Colab:**
```python
from google.colab import drive
import os
import shutil

# Mount Google Drive
drive.mount('/content/drive')

# Create mammoth directory in Drive if needed
!mkdir -p /content/drive/MyDrive/mammoth_data/wikiann

# Then from your local machine, upload the files to Google Drive manually:
# - Go to Google Drive
# - Navigate to MyDrive/mammoth_data/wikiann/
# - Upload the 'hi' and 'bn' folders (with all their .jsonl files)

# Once uploaded, symlink to the expected location
!mkdir -p /content/mammoth/data
!ln -s /content/drive/MyDrive/mammoth_data/wikiann /content/mammoth/data/wikiann

# Verify
!ls -la /content/mammoth/data/wikiann/hi/
!ls -la /content/mammoth/data/wikiann/bn/
```

### 2. Direct Upload (Temporary - lost when runtime disconnects)

**In Colab:**
```python
from google.colab import files
import os

# Create directories
!mkdir -p /content/mammoth/data/wikiann/hi
!mkdir -p /content/mammoth/data/wikiann/bn

# Upload Hindi files
print("=" * 60)
print("Upload Hindi files (train.jsonl, validation.jsonl, test.jsonl)")
print("=" * 60)
os.chdir('/content/mammoth/data/wikiann/hi')
uploaded = files.upload()

# Upload Bangla files
print("=" * 60)
print("Upload Bangla files (train.jsonl, validation.jsonl, test.jsonl)")
print("=" * 60)
os.chdir('/content/mammoth/data/wikiann/bn')
uploaded = files.upload()

# Return to main directory
os.chdir('/content/mammoth')

# Verify
print("\n" + "=" * 60)
print("Verification:")
print("=" * 60)
!ls -la /content/mammoth/data/wikiann/hi/
!ls -la /content/mammoth/data/wikiann/bn/
```

---

## Required Files

You need these 6 files total (already on your local machine at `D:\coding\projects\mammoth\data\wikiann\`):

**Hindi (hi/):**
- `train.jsonl` (651 KB)
- `validation.jsonl` (129 KB)
- `test.jsonl` (132 KB)

**Bangla (bn/):**
- `train.jsonl` (1.15 MB)
- `validation.jsonl` (116 KB)
- `test.jsonl` (116 KB)

---

## File Format

Each `.jsonl` file contains one JSON object per line:
```json
{"tokens": ["word1", "word2", ...], "ner_tags": [0, 1, 2, ...]}
```

These files were generated using `download_wikiann.py`.

---

## Verify Upload Success

```python
import os
import json

def verify_dataset():
    """Verify that all required files are present and valid"""
    base = '/content/mammoth/data/wikiann'

    for lang in ['hi', 'bn']:
        for split in ['train', 'validation', 'test']:
            path = f'{base}/{lang}/{split}.jsonl'
            if not os.path.exists(path):
                print(f"❌ MISSING: {path}")
                return False

            # Check file is not empty and has valid JSON
            with open(path, 'r') as f:
                first_line = f.readline()
                if not first_line:
                    print(f"❌ EMPTY: {path}")
                    return False
                try:
                    data = json.loads(first_line)
                    if 'tokens' not in data or 'ner_tags' not in data:
                        print(f"❌ INVALID FORMAT: {path}")
                        return False
                except json.JSONDecodeError:
                    print(f"❌ INVALID JSON: {path}")
                    return False

            print(f"✅ {lang}/{split}.jsonl")

    print("\n🎉 All files verified successfully!")
    return True

verify_dataset()
```

---

## Run Experiments

Once files are uploaded and verified:

```python
!python run_ner_experiments.py
```

The code will automatically load from local files. If files are missing, you'll get a clear error message showing exactly which files are needed.
