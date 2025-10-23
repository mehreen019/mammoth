"""
Colab Training Script for Hindi-Bangla NER with BERT Multilingual
This script trains the model and generates visualizations for demonstration
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import numpy as np
from tqdm import tqdm
import os

# Import custom modules
from backbone.bert_multilingual import BERTMultilingualBackbone
from visualize_ner_results import create_summary_report, visualize_predictions


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


class DummyNERDataset(torch.utils.data.Dataset):
    """
    Dummy dataset for demonstration purposes
    Replace this with your actual Hindi-Bangla NER dataset
    """
    def __init__(self, tokenizer, num_samples=100, max_length=128):
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.max_length = max_length

        # Sample sentences (mix of Hindi, Bangla, and English for demo)
        self.sentences = [
            "राम दिल्ली में रहते हैं।",
            "সে ঢাকায় বাস করে।",
            "Google is a company in California.",
            "मोहन और सीता मुंबई गए।",
            "আমি কলকাতায় জন্মেছি।"
        ] * (num_samples // 5 + 1)

        # Dummy labels (random for demo - replace with real labels)
        self.labels = []
        for _ in range(num_samples):
            # Generate random sequence of tags
            seq_len = np.random.randint(10, max_length)
            tags = np.random.choice(len(TAG_TO_IDX), size=seq_len)
            self.labels.append(tags.tolist())

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        sentence = self.sentences[idx % len(self.sentences)]
        labels = self.labels[idx]

        # Tokenize
        encoding = self.tokenizer(
            sentence,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Pad labels to match tokenized length
        label_tensor = torch.zeros(self.max_length, dtype=torch.long)
        label_tensor[:len(labels)] = torch.tensor(labels[:self.max_length])

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': label_tensor
        }


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc='Training')
    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        # Forward pass (using CLS token classification for demo)
        outputs = model((input_ids, attention_mask))

        # For simplicity, we'll use the first token's label
        # In real NER, you'd classify each token
        loss = criterion(outputs, labels[:, 0])

        # Backward pass
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels[:, 0]).sum().item()
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
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model((input_ids, attention_mask))
            loss = criterion(outputs, labels[:, 0])

            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels[:, 0]).sum().item()
            total += labels.size(0)
            total_loss += loss.item()

            # Store for confusion matrix
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels[:, 0].cpu().numpy())

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

            input_ids = batch['input_ids'][0].unsqueeze(0).to(device)
            attention_mask = batch['attention_mask'][0].unsqueeze(0).to(device)
            labels = batch['labels'][0]

            outputs = model((input_ids, attention_mask))
            prediction = outputs.argmax(dim=1).item()

            # Decode tokens
            tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy())
            tokens = [t for t in tokens if t not in ['[PAD]', '[CLS]', '[SEP]']][:10]  # First 10 tokens

            # Get corresponding labels
            true_tags = [IDX_TO_TAG[labels[j].item()] for j in range(len(tokens))]
            pred_tags = [IDX_TO_TAG[prediction]] * len(tokens)  # Simplified for demo

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
    NUM_TRAIN_SAMPLES = 200
    NUM_VAL_SAMPLES = 50

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Load tokenizer
    print("\nLoading BERT tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')

    # Create datasets
    print("Creating datasets...")
    train_dataset = DummyNERDataset(tokenizer, num_samples=NUM_TRAIN_SAMPLES)
    val_dataset = DummyNERDataset(tokenizer, num_samples=NUM_VAL_SAMPLES)

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
        val_loss, val_acc, val_preds, val_labels = evaluate(model, val_loader, criterion, device)

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
    final_val_loss, final_val_acc, all_preds, all_labels = evaluate(
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
