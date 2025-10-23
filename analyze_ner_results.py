"""
Analysis script for NER continual learning results
Generates comprehensive visualizations and metrics for forward/backward transfer
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import os
from pathlib import Path

def load_results(results_dir='results'):
    """Load results from Mammoth's output"""
    results_file = Path(results_dir) / 'results.json'
    if results_file.exists():
        with open(results_file, 'r') as f:
            return json.load(f)
    return None

def plot_accuracy_matrix(results, save_path='results/accuracy_matrix.png'):
    """
    Plot accuracy matrix showing performance on each task after training on each task
    
    Rows: Task trained on
    Columns: Task evaluated on
    """
    if results is None or 'task_accuracies' not in results:
        print("No task accuracies found in results")
        return
    
    task_accs = results['task_accuracies']
    n_tasks = len(task_accs)
    
    # Create matrix
    acc_matrix = np.zeros((n_tasks, n_tasks))
    for i, task_results in enumerate(task_accs):
        for j, acc in enumerate(task_results):
            acc_matrix[i, j] = acc
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(acc_matrix, annot=True, fmt='.2f', cmap='YlGnBu', 
                xticklabels=[f'Task {i}' for i in range(n_tasks)],
                yticklabels=[f'After Task {i}' for i in range(n_tasks)],
                vmin=0, vmax=100, ax=ax)
    ax.set_title('Accuracy Matrix: Performance on Each Task After Training', fontsize=14, fontweight='bold')
    ax.set_xlabel('Evaluated Task', fontsize=12)
    ax.set_ylabel('Training Progress', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved accuracy matrix to {save_path}")
    plt.close()

def plot_transfer_metrics(results, save_path='results/transfer_metrics.png'):
    """Plot forward and backward transfer metrics"""
    if results is None:
        print("No results found")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Forward Transfer
    if 'forward_transfer' in results:
        fwt = results['forward_transfer']
        axes[0].bar(['Forward Transfer'], [fwt], color='skyblue', edgecolor='black', linewidth=2)
        axes[0].set_ylabel('Accuracy Gain (%)', fontsize=12)
        axes[0].set_title('Forward Transfer\n(Benefit from previous tasks)', fontsize=14, fontweight='bold')
        axes[0].axhline(y=0, color='red', linestyle='--', linewidth=1)
        axes[0].set_ylim(-10, 10)
        axes[0].grid(axis='y', alpha=0.3)
        
        # Add value label
        axes[0].text(0, fwt + 0.5, f'{fwt:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Backward Transfer
    if 'backward_transfer' in results:
        bwt = results['backward_transfer']
        color = 'lightcoral' if bwt < 0 else 'lightgreen'
        axes[1].bar(['Backward Transfer'], [bwt], color=color, edgecolor='black', linewidth=2)
        axes[1].set_ylabel('Accuracy Change (%)', fontsize=12)
        axes[1].set_title('Backward Transfer\n(Forgetting of previous tasks)', fontsize=14, fontweight='bold')
        axes[1].axhline(y=0, color='red', linestyle='--', linewidth=1)
        axes[1].set_ylim(-30, 10)
        axes[1].grid(axis='y', alpha=0.3)
        
        # Add value label
        axes[1].text(0, bwt + 1, f'{bwt:.2f}%', ha='center', va='bottom' if bwt >= 0 else 'top', 
                    fontsize=12, fontweight='bold')
    
    # Forgetting
    if 'forgetting' in results:
        forgetting = results['forgetting']
        axes[2].bar(['Forgetting'], [forgetting], color='salmon', edgecolor='black', linewidth=2)
        axes[2].set_ylabel('Accuracy Drop (%)', fontsize=12)
        axes[2].set_title('Forgetting\n(Maximum accuracy drop)', fontsize=14, fontweight='bold')
        axes[2].axhline(y=0, color='red', linestyle='--', linewidth=1)
        axes[2].set_ylim(-30, 5)
        axes[2].grid(axis='y', alpha=0.3)
        
        # Add value label
        axes[2].text(0, forgetting + 1, f'{forgetting:.2f}%', ha='center', va='bottom' if forgetting >= 0 else 'top',
                    fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved transfer metrics to {save_path}")
    plt.close()

def plot_task_performance(results, save_path='results/task_performance.png'):
    """Plot performance on each task over training"""
    if results is None or 'task_accuracies' not in results:
        print("No task accuracies found")
        return
    
    task_accs = results['task_accuracies']
    n_tasks = len(task_accs)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for task_id in range(n_tasks):
        accs = [task_accs[i][task_id] if task_id < len(task_accs[i]) else None 
                for i in range(n_tasks)]
        # Filter out None values
        x_vals = [i for i, acc in enumerate(accs) if acc is not None]
        y_vals = [acc for acc in accs if acc is not None]
        
        ax.plot(x_vals, y_vals, marker='o', linewidth=2, markersize=8, 
               label=f'Task {task_id} ({"Hindi" if task_id == 0 else "Bangla"})')
    
    ax.set_xlabel('Training Progress (After Task)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Task Performance Over Training', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(n_tasks))
    ax.set_xticklabels([f'After Task {i}' for i in range(n_tasks)])
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved task performance to {save_path}")
    plt.close()

def print_summary(results):
    """Print a text summary of results"""
    print("\n" + "="*80)
    print("CONTINUAL LEARNING RESULTS SUMMARY")
    print("="*80)
    
    if results is None:
        print("No results found!")
        return
    
    if 'task_accuracies' in results:
        print("\nTask Accuracies:")
        for i, task_accs in enumerate(results['task_accuracies']):
            print(f"  After Task {i}: {task_accs}")
    
    if 'average_accuracy' in results:
        print(f"\nAverage Accuracy: {results['average_accuracy']:.2f}%")
    
    if 'forward_transfer' in results:
        print(f"\nForward Transfer: {results['forward_transfer']:.2f}%")
        print("  (Positive = benefit from previous tasks)")
    
    if 'backward_transfer' in results:
        print(f"\nBackward Transfer: {results['backward_transfer']:.2f}%")
        print("  (Negative = forgetting, Positive = improvement)")
    
    if 'forgetting' in results:
        print(f"\nForgetting: {results['forgetting']:.2f}%")
        print("  (Maximum accuracy drop on previous tasks)")
    
    print("\n" + "="*80)

def create_comprehensive_report(results_dir='results'):
    """Create all visualizations and summary"""
    os.makedirs(results_dir, exist_ok=True)
    
    results = load_results(results_dir)
    
    print("\n" + "="*80)
    print("GENERATING COMPREHENSIVE ANALYSIS")
    print("="*80)
    
    # Generate plots
    plot_accuracy_matrix(results, f'{results_dir}/accuracy_matrix.png')
    plot_transfer_metrics(results, f'{results_dir}/transfer_metrics.png')
    plot_task_performance(results, f'{results_dir}/task_performance.png')
    
    # Print summary
    print_summary(results)
    
    print("\n" + "="*80)
    print(f"All visualizations saved to '{results_dir}/' directory")
    print("="*80)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Analyze NER continual learning results')
    parser.add_argument('--results_dir', type=str, default='results',
                       help='Directory containing results')
    args = parser.parse_args()
    
    create_comprehensive_report(args.results_dir)

