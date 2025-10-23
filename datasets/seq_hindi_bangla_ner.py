# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sequential Hindi -> Bangla NER Dataset for Continual Learning
Uses WikiANN dataset from HuggingFace datasets library

This implementation simplifies NER to sentence-level classification to fit Mammoth's architecture:
- Task 1: Hindi NER (3 entity types: PER, LOC, ORG)
- Task 2: Bangla NER (3 entity types: PER, LOC, ORG)
"""

from typing import Tuple
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
import sys
import os
import itertools
import json

# Avoid multiprocessing issues with HuggingFace datasets in restricted environments
os.environ.setdefault("HF_DATASETS_DISABLE_MULTIPROCESSING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# IMPORTANT: Must import local datasets utilities BEFORE trying to import HuggingFace datasets
# to avoid import conflicts
from datasets.utils.continual_dataset import ContinualDataset, store_masked_loaders
from utils.conf import base_path
from datasets.utils import set_default_from_args

# Handle the HuggingFace import - must manipulate sys.path to avoid local datasets folder
load_dataset = None
AutoTokenizer = None
HF_DATASETS_LIB = None
HF_DOWNLOAD_CONFIG_CLS = None
HUGGINGFACE_AVAILABLE = False


def _parse_split(split: str):
    """Return (base_split, limit_or_None) for slicing expressions like 'train[:500]'."""
    base_split = split
    limit = None
    if '[' in split and ']' in split:
        base_split, bracket = split.split('[', 1)
        bracket = bracket.strip(']')
        if bracket:
            # Support train[:500] and train[100:200] (use right bound)
            if ':' in bracket:
                right = bracket.split(':', 1)[1]
                if right:
                    try:
                        limit = int(right)
                    except ValueError:
                        limit = None
            else:
                try:
                    limit = int(bracket)
                except ValueError:
                    limit = None
    return base_split, limit


def _load_local_wikiann_split(lang_code: str, split: str):
    """
    Load a split from locally cached files.
    Expected layout: <base_path>/data/wikiann/<lang>/<split>.jsonl (or .json)
    where each record has 'tokens' and 'ner_tags'.
    """
    dataset_root = os.path.join(base_path(), 'data', 'wikiann', lang_code)
    base_split, limit = _parse_split(split)

    candidates = [
        os.path.join(dataset_root, f"{base_split}.jsonl"),
        os.path.join(dataset_root, f"{base_split}.json"),
    ]

    for path in candidates:
        if os.path.exists(path):
            tokens, ner_tags = [], []
            try:
                if path.endswith('.jsonl'):
                    with open(path, 'r', encoding='utf-8') as handle:
                        for line in handle:
                            if not line.strip():
                                continue
                            sample = json.loads(line)
                            tokens.append(sample['tokens'])
                            ner_tags.append(sample['ner_tags'])
                            if limit is not None and len(tokens) >= limit:
                                break
                else:  # .json
                    with open(path, 'r', encoding='utf-8') as handle:
                        data = json.load(handle)
                    for sample in data:
                        tokens.append(sample['tokens'])
                        ner_tags.append(sample['ner_tags'])
                        if limit is not None and len(tokens) >= limit:
                            break
            except Exception as local_err:
                raise RuntimeError(f"Failed to parse local WikiANN file at {path}") from local_err

            if not tokens:
                raise RuntimeError(f"Local WikiANN file {path} is empty or missing required fields.")

            return {
                'tokens': tokens,
                'ner_tags': ner_tags
            }

    return None


def _patch_hf_dill_for_rlock(hf_datasets_module):
    """
    HuggingFace datasets <= 2.18 depends on dill<0.3.8 in some environments.
    On Python 3.12 this triggers RuntimeError when hashing configs containing RLock objects.
    This patch falls back to a deterministic placeholder when that specific error occurs.
    """
    try:
        import dill  # type: ignore
    except Exception:
        return

    version_str = getattr(dill, "__version__", "")
    version_parts = [int(part) for part in version_str.split('.') if part.isdigit()]
    while len(version_parts) < 3:
        version_parts.append(0)
    if tuple(version_parts[:3]) >= (0, 3, 8):
        # Patched in dill>=0.3.8
        return

    hf_dill_utils = getattr(getattr(hf_datasets_module, "utils", None), "_dill", None)
    if hf_dill_utils is None:
        try:
            from datasets.utils import _dill as hf_dill_utils  # type: ignore
        except Exception:
            return

    if getattr(hf_dill_utils, "_mammoth_rlock_patch", False):
        return

    original_dump = getattr(hf_dill_utils, "dump", None)
    original_dumps = getattr(hf_dill_utils, "dumps", None)
    if original_dump is None or original_dumps is None:
        return

    import pickle

    error_snippet = "RLock objects should only be shared between processes through inheritance"

    def _placeholder_bytes(obj):
        obj_type = getattr(type(obj), "__qualname__", type(obj).__name__)
        return pickle.dumps(f"__hf_rlock_placeholder__:{obj_type}", protocol=pickle.HIGHEST_PROTOCOL)

    def _safe_dump(obj, file, *args, **kwargs):
        try:
            return original_dump(obj, file, *args, **kwargs)
        except RuntimeError as err:
            if error_snippet not in str(err):
                raise
            file.write(_placeholder_bytes(obj))
            return None

    def _safe_dumps(obj, *args, **kwargs):
        try:
            return original_dumps(obj, *args, **kwargs)
        except RuntimeError as err:
            if error_snippet not in str(err):
                raise
            return _placeholder_bytes(obj)

    hf_dill_utils.dump = _safe_dump  # type: ignore
    hf_dill_utils.dumps = _safe_dumps  # type: ignore
    hf_dill_utils._mammoth_rlock_patch = True


def _import_hf_datasets():
    """Import HuggingFace datasets by temporarily removing mammoth dir from sys.path"""
    import sys
    import os

    # Save original state
    original_path = sys.path.copy()
    original_modules = {}

    try:
        # Remove '/content/mammoth' and similar from sys.path
        mammoth_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # Filter out the mammoth directory from sys.path
        new_path = []
        for p in sys.path:
            abs_p = os.path.abspath(p) if p else ''
            if abs_p != mammoth_dir and p not in ['', '.']:
                new_path.append(p)

        # Backup and remove 'datasets' from sys.modules if it points to local folder
        if 'datasets' in sys.modules:
            original_modules['datasets'] = sys.modules['datasets']
            if hasattr(sys.modules['datasets'], '__file__'):
                if 'mammoth' in sys.modules['datasets'].__file__:
                    del sys.modules['datasets']

        # Also remove any datasets.* submodules
        for key in list(sys.modules.keys()):
            if key.startswith('datasets.'):
                original_modules[key] = sys.modules[key]
                del sys.modules[key]

        # Apply new path
        sys.path = new_path

        # Now import HuggingFace datasets
        import datasets as hf_datasets
        import transformers

        # Verify we got the right module
        if not hasattr(hf_datasets, 'load_dataset'):
            raise ImportError("Got wrong datasets module")

        _patch_hf_dill_for_rlock(hf_datasets)

        return hf_datasets, transformers

    finally:
        # Always restore original state
        sys.path = original_path
        for key, value in original_modules.items():
            sys.modules[key] = value

try:
    hf_datasets_module, transformers_module = _import_hf_datasets()
    if hf_datasets_module is not None and transformers_module is not None:
        load_dataset = hf_datasets_module.load_dataset
        AutoTokenizer = transformers_module.AutoTokenizer
        HF_DATASETS_LIB = hf_datasets_module
        HF_DOWNLOAD_CONFIG_CLS = getattr(hf_datasets_module, "DownloadConfig", None)
        HUGGINGFACE_AVAILABLE = True
except Exception as e:
    # Failed to import, will try again at runtime
    pass


def _streaming_subset(lang_code: str, base_split: str, limit: int):
    """DEPRECATED: This function is no longer used. Local files are required."""
    raise RuntimeError(
        f"Cannot stream WikiANN data from HuggingFace. Please download the dataset locally using download_wikiann.py\n"
        f"Expected location: {os.path.join(base_path(), 'data', 'wikiann', lang_code)}\n"
        f"Run: python download_wikiann.py"
    )


def _load_wikiann_split(lang_code: str, split: str):
    """
    Load a WikiANN split from local files ONLY.

    This function no longer attempts to download from HuggingFace to avoid
    RLock pickling errors in Python 3.12+ environments (especially Colab).

    Use download_wikiann.py to prepare the dataset locally first.
    """
    local_split = _load_local_wikiann_split(lang_code, split)
    if local_split is not None:
        return local_split

    # If we get here, local files were not found
    expected_path = os.path.join(base_path(), 'data', 'wikiann', lang_code)
    raise FileNotFoundError(
        f"\n{'='*80}\n"
        f"WikiANN dataset not found locally!\n"
        f"{'='*80}\n"
        f"Language: {lang_code}\n"
        f"Split: {split}\n"
        f"Expected location: {expected_path}\n"
        f"\n"
        f"To fix this:\n"
        f"1. Run locally (not in Colab): python download_wikiann.py\n"
        f"2. This will create files in data/wikiann/{{hi,bn}}/\n"
        f"3. Upload these files to Colab at: /content/mammoth/data/wikiann/{{hi,bn}}/\n"
        f"\n"
        f"Required files:\n"
        f"  - {os.path.join(expected_path, 'train.jsonl')}\n"
        f"  - {os.path.join(expected_path, 'validation.jsonl')}\n"
        f"  - {os.path.join(expected_path, 'test.jsonl')}\n"
        f"{'='*80}\n"
    )


class NERDatasetWrapper(Dataset):
    """
    Wrapper for NER data that converts it to Mammoth-compatible format.
    Each sample is a tokenized sentence with aggregated entity label.
    """

    def __init__(self, texts, labels, tokenizer, task_ids=None, max_length=128):
        """
        Args:
            texts: List of token sequences (List[List[str]])
            labels: List of NER label sequences (List[List[int]])
            tokenizer: HuggingFace tokenizer
            task_ids: List of task IDs for each sample (for continual learning)
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Convert to sentence-level labels (dominant entity type)
        self.sentence_labels = []
        for label_seq in labels:
            # Map NER tags: O=0, B-PER/I-PER=1, B-LOC/I-LOC=2, B-ORG/I-ORG=3
            # Take the most frequent non-O entity type
            entity_counts = {1: 0, 2: 0, 3: 0}  # PER, LOC, ORG
            for tag in label_seq:
                if tag in [1, 2]:  # B-PER or I-PER
                    entity_counts[1] += 1
                elif tag in [3, 4]:  # B-LOC or I-LOC
                    entity_counts[2] += 1
                elif tag in [5, 6]:  # B-ORG or I-ORG
                    entity_counts[3] += 1

            # Get dominant entity or O (0) if no entities
            if sum(entity_counts.values()) == 0:
                self.sentence_labels.append(0)
            else:
                dominant = max(entity_counts, key=entity_counts.get)
                self.sentence_labels.append(dominant)

        # Tokenize all texts
        self.encodings = []
        for tokens in texts:
            # Join tokens back to text for tokenizer
            text = ' '.join(tokens) if isinstance(tokens, list) else tokens
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=max_length,
                return_tensors='pt'
            )
            # Flatten batch dimension (tokenizer returns [1, seq_len])
            encoding = {k: v.squeeze(0) for k, v in encoding.items()}
            self.encodings.append(encoding)

        # Store as numpy arrays for Mammoth compatibility
        self.data = np.array([enc['input_ids'].numpy() for enc in self.encodings])
        self.targets = np.array(self.sentence_labels)
        self.attention_masks = np.array([enc['attention_mask'].numpy() for enc in self.encodings])

        # Store task IDs for continual learning - must match the data length!
        if task_ids is not None:
            self.task_ids = np.array(task_ids[:len(self.data)])  # Trim to actual data length
        else:
            self.task_ids = np.zeros(len(self.data), dtype=np.int64)

        # Ensure all arrays have the same length
        assert len(self.data) == len(self.targets) == len(self.attention_masks) == len(self.task_ids), \
            f"Mismatched lengths: data={len(self.data)}, targets={len(self.targets)}, masks={len(self.attention_masks)}, task_ids={len(self.task_ids)}"

    def __len__(self):
        # Use the actual data length, not texts length (in case some failed to tokenize)
        return len(self.data)

    def __getitem__(self, index):
        """
        Returns: (input_ids, label, attention_mask)
        Matches Mammoth's (data, target, not_aug_data) pattern
        """
        if index >= len(self.data):
            raise IndexError(f"Index {index} out of bounds for dataset with {len(self.data)} samples")

        input_ids = torch.tensor(self.data[index], dtype=torch.long)
        label = torch.tensor(self.targets[index], dtype=torch.long)
        attention_mask = torch.tensor(self.attention_masks[index], dtype=torch.long)

        # Return (input, label, extra_info)
        return input_ids, label, attention_mask


class SequentialHindiBanglaNER(ContinualDataset):
    """
    Sequential Hindi -> Bangla NER Dataset

    Task 0: Hindi NER (classes 0-3: O, PER, LOC, ORG)
    Task 1: Bangla NER (classes 0-3: O, PER, LOC, ORG) - SAME classes, different language

    This creates a 2-task continual learning scenario where the model must learn
    to recognize the same entity types in different languages sequentially.
    """

    NAME = 'seq-hindi-bangla-ner'
    SETTING = 'task-il'  # Task-IL: same classes, different tasks (languages)
    N_CLASSES_PER_TASK = 4  # O, PER, LOC, ORG (same for both languages)
    N_TASKS = 2  # Hindi, Bangla
    N_CLASSES = 4  # Total unique classes (shared across tasks)
    SIZE = (128,)  # Sequence length (not used but required by framework)

    TRANSFORM = None
    TEST_TRANSFORM = None

    def __init__(self, args):
        super().__init__(args)

        # Ensure HuggingFace libraries are available
        global HUGGINGFACE_AVAILABLE, load_dataset, AutoTokenizer, HF_DATASETS_LIB, HF_DOWNLOAD_CONFIG_CLS
        if not HUGGINGFACE_AVAILABLE or load_dataset is None or AutoTokenizer is None:
            hf_datasets_module, transformers_module = _import_hf_datasets()
            if hf_datasets_module is None or transformers_module is None:
                raise ImportError(
                    "\n" + "="*60 + "\n"
                    "ERROR: Cannot import HuggingFace libraries!\n"
                    "="*60 + "\n"
                    "Please run in Colab:\n"
                    "  !pip install datasets transformers\n"
                    "Then restart the runtime.\n"
                    "="*60
                )
            load_dataset = hf_datasets_module.load_dataset
            AutoTokenizer = transformers_module.AutoTokenizer
            HF_DATASETS_LIB = hf_datasets_module
            HF_DOWNLOAD_CONFIG_CLS = getattr(hf_datasets_module, "DownloadConfig", None)
            HUGGINGFACE_AVAILABLE = True
            print(f"✅ HuggingFace libraries loaded successfully")

        # Use multilingual BERT tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
            self.max_length = 128
        except Exception as e:
            raise RuntimeError(f"Failed to load BERT multilingual tokenizer: {e}")

    def get_data_loaders(self) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """Load WikiANN Hindi and Bangla NER data"""

        print("Loading WikiANN dataset for Hindi and Bangla...")

        # Load Hindi data (Task 0)
        hindi_train = _load_wikiann_split('hi', 'train[:500]')  # Smaller for speed
        hindi_test = _load_wikiann_split('hi', 'validation[:100]')

        # Load Bangla data (Task 1)
        bangla_train = _load_wikiann_split('bn', 'train[:500]')
        bangla_test = _load_wikiann_split('bn', 'validation[:100]')

        def _extract_texts_and_labels(split, split_name: str):
            """Normalize HuggingFace or tuple-based split outputs into (texts, labels) lists."""
            # Tuple of (texts, labels) -> typically dummy data fallback
            if isinstance(split, tuple) and len(split) == 2:
                texts, labels = split
            # HuggingFace Dataset object
            elif hasattr(split, '__getitem__') and hasattr(split, 'column_names'):
                texts, labels = split['tokens'], split['ner_tags']
            # Dictionary-style access (e.g., DatasetDict split converted to dict)
            elif isinstance(split, dict) and 'tokens' in split and 'ner_tags' in split:
                texts, labels = split['tokens'], split['ner_tags']
            else:
                raise TypeError(f"Unsupported split type for {split_name}: {type(split)}")

            # Ensure we always work with regular Python lists for downstream processing
            return list(texts), list(labels)

        # Extract tokens and labels - handle both real data and streaming dictionaries uniformly
        hindi_train_texts, hindi_train_labels = _extract_texts_and_labels(hindi_train, "hindi_train")
        hindi_test_texts, hindi_test_labels = _extract_texts_and_labels(hindi_test, "hindi_test")
        bangla_train_texts, bangla_train_labels = _extract_texts_and_labels(bangla_train, "bangla_train")
        bangla_test_texts, bangla_test_labels = _extract_texts_and_labels(bangla_test, "bangla_test")

        # Combine datasets with task IDs (Hindi=0, Bangla=1)
        all_train_texts = hindi_train_texts + bangla_train_texts
        all_train_labels = hindi_train_labels + bangla_train_labels
        all_test_texts = hindi_test_texts + bangla_test_texts
        all_test_labels = hindi_test_labels + bangla_test_labels

        # Create task IDs: 0 for Hindi, 1 for Bangla
        train_task_ids = [0] * len(hindi_train_texts) + [1] * len(bangla_train_texts)
        test_task_ids = [0] * len(hindi_test_texts) + [1] * len(bangla_test_texts)

        # Create dataset wrappers with task IDs
        train_dataset = NERDatasetWrapper(
            all_train_texts,
            all_train_labels,
            self.tokenizer,
            task_ids=train_task_ids,
            max_length=self.max_length
        )
        test_dataset = NERDatasetWrapper(
            all_test_texts,
            all_test_labels,
            self.tokenizer,
            task_ids=test_task_ids,
            max_length=self.max_length
        )

        print(f"Dataset loaded: {len(train_dataset)} train, {len(test_dataset)} test samples")
        print(f"  Task 0 (Hindi): {sum(t == 0 for t in train_task_ids)} train, {sum(t == 0 for t in test_task_ids)} test")
        print(f"  Task 1 (Bangla): {sum(t == 1 for t in train_task_ids)} train, {sum(t == 1 for t in test_task_ids)} test")

        # Use Mammoth's store_masked_loaders to create task-specific loaders
        train_loader, test_loader = store_masked_loaders(train_dataset, test_dataset, self)

        return train_loader, test_loader

    @set_default_from_args("backbone")
    def get_backbone(self):
        return "bert-multilingual"  # Will need custom backbone

    @staticmethod
    def get_transform():
        return None

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        return None

    @staticmethod
    def get_denormalization_transform():
        return None

    @set_default_from_args('batch_size')
    def get_batch_size(self):
        return 16  # Smaller batch for BERT

    @set_default_from_args('n_epochs')
    def get_epochs(self):
        return 2  # Fast training for demo (2 epochs per task)

    def get_class_names(self):
        if self.class_names is not None:
            return self.class_names

        # Class names (same for both tasks in Task-IL setting)
        self.class_names = ['O', 'PER', 'LOC', 'ORG']
        return self.class_names
