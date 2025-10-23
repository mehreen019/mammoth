"""
Visualization utilities for NER results
Creates plots for training metrics, confusion matrix, and per-class performance
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import torch


def plot_training_history(history, save_path=None):
    """
    Plot training and validation metrics over epochs

    Args:
        history: dict with 'train_loss', 'val_loss', 'train_acc', 'val_acc' lists
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot losses
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', marker='s', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Plot accuracies
    axes[1].plot(history['train_acc'], label='Train Accuracy', marker='o', linewidth=2, color='green')
    axes[1].plot(history['val_acc'], label='Val Accuracy', marker='s', linewidth=2, color='orange')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """
    Plot confusion matrix for NER predictions

    Args:
        y_true: True labels (flat list)
        y_pred: Predicted labels (flat list)
        class_names: List of class names
        save_path: Optional path to save the figure
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues', values_format='d', xticks_rotation=45)
    plt.title('NER Tag Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_per_class_metrics(y_true, y_pred, class_names, save_path=None):
    """
    Plot per-class precision, recall, and F1 scores

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Optional path to save the figure
    """
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)

    # Extract metrics
    precision = [report[name]['precision'] for name in class_names]
    recall = [report[name]['recall'] for name in class_names]
    f1 = [report[name]['f1-score'] for name in class_names]

    # Create grouped bar chart
    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))

    bars1 = ax.bar(x - width, precision, width, label='Precision', color='skyblue')
    bars2 = ax.bar(x, recall, width, label='Recall', color='lightcoral')
    bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='lightgreen')

    ax.set_xlabel('NER Tags', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Per-Class Performance Metrics', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def visualize_predictions(tokens_list, true_tags_list, pred_tags_list, class_names, num_examples=3):
    """
    Display example predictions with color coding

    Args:
        tokens_list: List of token sequences
        true_tags_list: List of true tag sequences
        pred_tags_list: List of predicted tag sequences
        class_names: List of class names
        num_examples: Number of examples to show
    """
    # Color map for different tag types
    colors = plt.cm.Set3(np.linspace(0, 1, len(class_names)))
    tag_colors = {tag: colors[i] for i, tag in enumerate(class_names)}

    for idx in range(min(num_examples, len(tokens_list))):
        print(f"\n{'='*80}")
        print(f"EXAMPLE {idx + 1}")
        print(f"{'='*80}")

        tokens = tokens_list[idx]
        true_tags = true_tags_list[idx]
        pred_tags = pred_tags_list[idx]

        print(f"\n{'Token':<20} {'True Tag':<20} {'Predicted Tag':<20} {'Match'}")
        print("-" * 80)

        for token, true_tag, pred_tag in zip(tokens, true_tags, pred_tags):
            match = "✓" if true_tag == pred_tag else "✗"
            symbol = "✓" if match == "✓" else "✗"
            print(f"{token:<20} {true_tag:<20} {pred_tag:<20} {symbol}")

        # Calculate accuracy for this example
        correct = sum(1 for t, p in zip(true_tags, pred_tags) if t == p)
        accuracy = correct / len(true_tags) * 100
        print(f"\nExample Accuracy: {accuracy:.2f}%")


def print_classification_report(y_true, y_pred, class_names):
    """
    Print detailed classification report

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
    """
    print("\n" + "="*80)
    print("CLASSIFICATION REPORT")
    print("="*80)
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    print(report)


def create_summary_report(history, y_true, y_pred, class_names, save_dir='results'):
    """
    Create a comprehensive visualization summary

    Args:
        history: Training history dict
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_dir: Directory to save figures
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    print("\n" + "="*80)
    print("GENERATING VISUALIZATION REPORT")
    print("="*80)

    # 1. Training history
    print("\n[1/4] Plotting training history...")
    plot_training_history(history, save_path=f'{save_dir}/training_history.png')

    # 2. Confusion matrix
    print("[2/4] Plotting confusion matrix...")
    plot_confusion_matrix(y_true, y_pred, class_names, save_path=f'{save_dir}/confusion_matrix.png')

    # 3. Per-class metrics
    print("[3/4] Plotting per-class metrics...")
    plot_per_class_metrics(y_true, y_pred, class_names, save_path=f'{save_dir}/per_class_metrics.png')

    # 4. Classification report
    print("[4/4] Generating classification report...")
    print_classification_report(y_true, y_pred, class_names)

    print("\n" + "="*80)
    print(f"All visualizations saved to '{save_dir}/' directory")
    print("="*80)
