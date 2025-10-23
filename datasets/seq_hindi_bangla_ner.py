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

# IMPORTANT: Must import local datasets utilities BEFORE trying to import HuggingFace datasets
# to avoid import conflicts
from datasets.utils.continual_dataset import ContinualDataset, store_masked_loaders
from utils.conf import base_path
from datasets.utils import set_default_from_args

# Now handle the HuggingFace datasets import
# The issue: there's a naming conflict between the local 'datasets' folder and HuggingFace's 'datasets' package
# Solution: Use importlib to directly import from site-packages, bypassing the local datasets folder
load_dataset = None
AutoTokenizer = None
HUGGINGFACE_AVAILABLE = False

try:
    import importlib.util
    import site

    # Method 1: Try to find HuggingFace datasets in site-packages
    hf_datasets_module = None
    for site_dir in site.getsitepackages() + [site.getusersitepackages()]:
        if site_dir is None:
            continue
        datasets_init = os.path.join(site_dir, 'datasets', '__init__.py')
        if os.path.exists(datasets_init):
            # Load the module directly from this path
            spec = importlib.util.spec_from_file_location("hf_datasets", datasets_init)
            if spec and spec.loader:
                hf_datasets_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(hf_datasets_module)
                # Verify it's the right module (has load_dataset)
                if hasattr(hf_datasets_module, 'load_dataset'):
                    break
                else:
                    hf_datasets_module = None

    if hf_datasets_module is None:
        raise ImportError("Could not find HuggingFace datasets package in site-packages")

    # Import transformers normally (no conflict with this one)
    from transformers import AutoTokenizer as _AutoTokenizer

    # Extract the function we need
    load_dataset = hf_datasets_module.load_dataset
    AutoTokenizer = _AutoTokenizer
    HUGGINGFACE_AVAILABLE = True

except (ImportError, AttributeError) as e:
    HUGGINGFACE_AVAILABLE = False
    load_dataset = None
    AutoTokenizer = None


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

        # Store task IDs for continual learning
        if task_ids is not None:
            self.task_ids = np.array(task_ids)
        else:
            self.task_ids = np.zeros(len(self.texts), dtype=np.int64)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        """
        Returns: (input_ids, label, attention_mask)
        Matches Mammoth's (data, target, not_aug_data) pattern
        """
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
        global HUGGINGFACE_AVAILABLE, load_dataset, AutoTokenizer
        if not HUGGINGFACE_AVAILABLE or load_dataset is None or AutoTokenizer is None:
            try:
                # Try importing at runtime using importlib to bypass naming conflict
                import importlib.util
                import site

                # Find HuggingFace datasets in site-packages
                hf_datasets_module = None
                for site_dir in site.getsitepackages() + [site.getusersitepackages()]:
                    if site_dir is None:
                        continue
                    datasets_init = os.path.join(site_dir, 'datasets', '__init__.py')
                    if os.path.exists(datasets_init):
                        spec = importlib.util.spec_from_file_location("hf_datasets_runtime", datasets_init)
                        if spec and spec.loader:
                            hf_datasets_module = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(hf_datasets_module)
                            if hasattr(hf_datasets_module, 'load_dataset'):
                                break
                            else:
                                hf_datasets_module = None

                if hf_datasets_module is None:
                    raise ImportError("Could not find HuggingFace datasets in site-packages")

                # Import transformers
                import transformers as _transformers_module

                load_dataset = hf_datasets_module.load_dataset
                AutoTokenizer = _transformers_module.AutoTokenizer
                HUGGINGFACE_AVAILABLE = True
                print(f"✅ HuggingFace libraries loaded: datasets={hf_datasets_module.__version__}, transformers={_transformers_module.__version__}")
            except (ImportError, AttributeError) as e:
                print("\n" + "="*60)
                print("ERROR: HuggingFace libraries not found!")
                print("="*60)
                print(f"Python executable: {sys.executable}")
                print(f"Python path (first 3): {sys.path[:3]}")
                print("\nThis dataset requires: datasets and transformers")
                print("\nInstall with:")
                print("  pip install datasets transformers")
                print("\nOr in Colab:")
                print("  !pip install datasets transformers")
                print("\nDebugging info:")
                print(f"  site.getsitepackages(): {site.getsitepackages()}")
                print("="*60)
                raise ImportError(f"Failed to import HuggingFace libraries: {e}")

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
        try:
            hindi_train = load_dataset('wikiann', 'hi', split='train[:500]')  # Smaller for speed
            hindi_test = load_dataset('wikiann', 'hi', split='validation[:100]')
        except Exception as e:
            print(f"Error loading Hindi WikiANN: {e}")
            print("Using dummy data for demonstration...")
            hindi_train, hindi_test = self._create_dummy_data('hindi', 500, 100)

        # Load Bangla data (Task 1)
        try:
            bangla_train = load_dataset('wikiann', 'bn', split='train[:500]')
            bangla_test = load_dataset('wikiann', 'bn', split='validation[:100]')
        except Exception as e:
            print(f"Error loading Bangla WikiANN: {e}")
            print("Using dummy data for demonstration...")
            bangla_train, bangla_test = self._create_dummy_data('bangla', 500, 100)

        # Extract tokens and labels
        if isinstance(hindi_train, tuple):
            # Dummy data
            hindi_train_texts, hindi_train_labels = hindi_train
            hindi_test_texts, hindi_test_labels = hindi_test
            bangla_train_texts, bangla_train_labels = bangla_train
            bangla_test_texts, bangla_test_labels = bangla_test
        else:
            # Real WikiANN data
            hindi_train_texts = hindi_train['tokens']
            hindi_train_labels = hindi_train['ner_tags']
            hindi_test_texts = hindi_test['tokens']
            hindi_test_labels = hindi_test['ner_tags']

            bangla_train_texts = bangla_train['tokens']
            bangla_train_labels = bangla_train['ner_tags']
            bangla_test_texts = bangla_test['tokens']
            bangla_test_labels = bangla_test['ner_tags']

        # Combine datasets with task IDs (Hindi=0, Bangla=1)
        all_train_texts = list(hindi_train_texts) + list(bangla_train_texts)
        all_train_labels = list(hindi_train_labels) + list(bangla_train_labels)
        all_test_texts = list(hindi_test_texts) + list(bangla_test_texts)
        all_test_labels = list(hindi_test_labels) + list(bangla_test_labels)

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

    def _create_dummy_data(self, lang, n_train, n_test):
        """Create dummy NER data for testing when WikiANN is unavailable"""
        import random

        vocab = {
            'hindi': ['राम', 'दिल्ली', 'भारत', 'गूगल', 'सीता', 'मुंबई'],
            'bangla': ['রাম', 'ঢাকা', 'বাংলাদেশ', 'গুগল', 'সীতা', 'কলকাতা']
        }

        def generate_sample():
            tokens = random.choices(vocab[lang], k=random.randint(5, 15))
            # Random NER tags (0=O, 1-2=PER, 3-4=LOC, 5-6=ORG)
            labels = [random.randint(0, 6) for _ in tokens]
            return tokens, labels

        train_data = [generate_sample() for _ in range(n_train)]
        test_data = [generate_sample() for _ in range(n_test)]

        train_texts, train_labels = zip(*train_data)
        test_texts, test_labels = zip(*test_data)

        return (train_texts, train_labels), (test_texts, test_labels)

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
