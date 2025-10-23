"""
Colab Training Script for Hindi-Bangla NER with BERT Multilingual
This script trains the model and generates visualizations for demonstration
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

# Import custom modules
from backbone.bert_multilingual import BERTMultilingualBackbone
from datasets.seq_hindi_bangla_ner import NERDatasetWrapper
from visualize_ner_results import (
    create_summary_report,
    visualize_predictions
)


# NER Tag Definitions (BIO format for Hindi-Bangla NER)
TAG_TO_IDX = {
    'O': 0,           # Outside
    'B-PER': 1,       # Beginning of Person
    'I-PER': 2,       # Inside Person
    'B-LOC': 3,       # Beginning of Location
    'I-LOC': 4,       # Inside Location
    'B-ORG': 5,       # Beginning of Organization
    'I-ORG': 6,       # Inside Organization
    'B-MISC': 7,      # Beginning of Miscellaneous
    'I-MISC': 8       # Inside Miscellaneous
}

IDX_TO_TAG = {v: k for k, v in TAG_TO_IDX.items()}
CLASS_NAMES = list(TAG_TO_IDX.keys())


def _create_dummy_ner_data(lang, n_train, n_test):
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





def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc='Training')
    for batch in pbar:
        # Unpack batch: (input_ids, label, attention_mask)
        input_ids = batch[0].to(device)
        labels = batch[1].to(device)
        attention_mask = batch[2].to(device)

        optimizer.zero_grad()

        # Forward pass through BERT backbone
        outputs = model((input_ids, attention_mask))

        # Compute loss
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item(), 'acc': correct/total*100})

    return total_loss / len(dataloader), correct / total * 100


def evaluate(model, dataloader, criterion, device):
    """Evaluate the model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Evaluating')
        for batch in pbar:
            # Unpack batch: (input_ids, label, attention_mask)
            input_ids = batch[0].to(device)
            labels = batch[1].to(device)
            attention_mask = batch[2].to(device)

            outputs = model((input_ids, attention_mask))
            loss = criterion(outputs, labels)

            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item()

            # Store for confusion matrix
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            pbar.set_postfix({'loss': loss.item(), 'acc': correct/total*100})

    return total_loss / len(dataloader), correct / total * 100, all_preds, all_labels


def get_sample_predictions(model, dataloader, tokenizer, device, num_samples=3):
    """Get sample predictions for visualization"""
    model.eval()

    tokens_list = []
    true_tags_list = []
    pred_tags_list = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_samples:
                break

            # Unpack batch: (input_ids, label, attention_mask)
            input_ids = batch[0][0].unsqueeze(0).to(device)
            attention_mask = batch[2][0].unsqueeze(0).to(device)
            label = batch[1][0].item()

            outputs = model((input_ids, attention_mask))
            prediction = outputs.argmax(dim=1).item()

            # Decode tokens
            tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy())
            tokens = [t for t in tokens if t not in ['[PAD]', '[CLS]', '[SEP]']][:10]  # First 10 tokens

            # Get corresponding labels
            true_tag = IDX_TO_TAG.get(label, 'O')
            pred_tag = IDX_TO_TAG.get(prediction, 'O')

            true_tags = [true_tag] * len(tokens)
            pred_tags = [pred_tag] * len(tokens)

            tokens_list.append(tokens)
            true_tags_list.append(true_tags)
            pred_tags_list.append(pred_tags)

    return tokens_list, true_tags_list, pred_tags_list


def main():
    """Main training and visualization pipeline"""
    print("="*80)
    print("HINDI-BANGLA NER TRAINING WITH BERT MULTILINGUAL")
    print("="*80)

    # Hyperparameters
    BATCH_SIZE = 16
    EPOCHS = 5
    LEARNING_RATE = 2e-5

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Load tokenizer
    print("\nLoading BERT tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')

    # Create datasets using actual Hindi-Bangla NER data
    print("Creating Hindi-Bangla NER datasets...")

    # Load Hindi data (Task 1)
    try:
        from datasets import load_dataset as hf_load_dataset
        hindi_train = hf_load_dataset('wikiann', 'hi', split='train[:500]')
        hindi_test = hf_load_dataset('wikiann', 'hi', split='validation[:100]')
    except Exception as e:
        print(f"Error loading Hindi WikiANN: {e}")
        print("Using dummy data for demonstration...")
        hindi_train, hindi_test = _create_dummy_ner_data('hindi', 500, 100)

    # Load Bangla data (Task 2)
    try:
        from datasets import load_dataset as hf_load_dataset
        bangla_train = hf_load_dataset('wikiann', 'bn', split='train[:500]')
        bangla_test = hf_load_dataset('wikiann', 'bn', split='validation[:100]')
    except Exception as e:
        print(f"Error loading Bangla WikiANN: {e}")
        print("Using dummy data for demonstration...")
        bangla_train, bangla_test = _create_dummy_ner_data('bangla', 500, 100)

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

    # Combine datasets (Hindi first, then Bangla for sequential learning)
    all_train_texts = list(hindi_train_texts) + list(bangla_train_texts)
    all_train_labels = list(hindi_train_labels) + list(bangla_train_labels)
    all_test_texts = list(hindi_test_texts) + list(bangla_test_texts)
    all_test_labels = list(hindi_test_labels) + list(bangla_test_labels)

    # Create dataset wrappers
    train_dataset = NERDatasetWrapper(
        all_train_texts,
        all_train_labels,
        tokenizer,
        max_length=128
    )
    val_dataset = NERDatasetWrapper(
        all_test_texts,
        all_test_labels,
        tokenizer,
        max_length=128
    )

    print(f"Dataset loaded: {len(train_dataset)} train, {len(val_dataset)} test samples")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Create model
    print("Creating BERT Multilingual model...")
    model = BERTMultilingualBackbone(n_classes=len(TAG_TO_IDX), device=device)
    model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }

    # Training loop
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80)

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print("-" * 80)

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        # Store history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

    print("\n" + "="*80)
    print("TRAINING COMPLETED")
    print("="*80)

    # Get final predictions for visualization
    print("\nGenerating predictions for visualization...")
    _, _, all_preds, all_labels = evaluate(
        model, val_loader, criterion, device
    )

    # Get sample predictions with tokens
    tokens_list, true_tags_list, pred_tags_list = get_sample_predictions(
        model, val_loader, tokenizer, device, num_samples=3
    )

    # Create visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)

    # Create comprehensive report
    create_summary_report(history, all_labels, all_preds, CLASS_NAMES, save_dir='results')

    # Show example predictions
    print("\n" + "="*80)
    print("EXAMPLE PREDICTIONS")
    print("="*80)
    visualize_predictions(tokens_list, true_tags_list, pred_tags_list, CLASS_NAMES, num_examples=3)

    # Save model
    print("\nSaving model...")
    torch.save(model.state_dict(), 'results/bert_multilingual_ner.pth')
    print("Model saved to 'results/bert_multilingual_ner.pth'")

    print("\n" + "="*80)
    print("ALL DONE! Check the 'results/' folder for visualizations.")
    print("="*80)


if __name__ == '__main__':
    main()
