"""
visualization.py
----------------
Generates all mandatory and supplemental publication-quality figures
for the Indonesian tweet sentiment classification project.
All figures are saved to outputs/figures/.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re

# Set publication style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

FIG_DIR = 'outputs/figures'
os.makedirs(FIG_DIR, exist_ok=True)

# Cohesive color palette for classes
CLASS_PALETTE = {
    'positive': '#2ECC71',  # green
    'negative': '#E74C3C',  # red
    'neutral': '#3498DB'    # blue
}

def plot_dataset_overview(audit_path: str = 'dataset/processed/audit_summary.json'):
    """
    1. Dataset overview: Record counts, missing values, duplicates, and unique records.
    """
    with open(audit_path, 'r', encoding='utf-8') as f:
        audit = json.load(f)
        
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    
    categories = ['Total Records', 'Raw Unique', 'Clean Unique', 'Exact Duplicates', 'Normalized Dups', 'Missing Values']
    values = [
        audit['total_records'],
        audit['raw_unique_records'],
        audit['clean_unique_records'],
        audit['raw_exact_duplicates'],
        audit['clean_duplicates'],
        audit['raw_missing_values']
    ]
    colors = ['#1F77B4', '#2CA02C', '#17BECF', '#FF7F0E', '#D62728', '#7F7F7F']
    
    bars = ax.bar(categories, values, color=colors, width=0.55, edgecolor='black', alpha=0.85)
    
    # Annotate values
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f'{int(h):,}',
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        
    ax.set_title('Dataset Overview and Quality Audit\n(Indonesian Sentiment Tweet Dataset)', fontsize=13, fontweight='bold', pad=15)
    ax.set_ylabel('Number of Records', fontsize=11)
    ax.set_ylim(0, max(values) * 1.15)
    plt.xticks(rotation=20, ha='right', fontsize=10)
    plt.tight_layout()
    
    out_path = os.path.join(FIG_DIR, 'dataset_overview.png')
    plt.savefig(out_path)
    plt.close()
    print(f"Generated {out_path}")

def plot_text_length_distribution(cleaned_csv: str = 'dataset/processed/cleaned_tweets.csv.gz'):
    """
    2. Text length distribution: Histogram and KDE of tweet lengths (raw vs cleaned).
    """
    df = pd.read_csv(cleaned_csv, compression='gzip', nrows=100000)
    
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    
    sns.histplot(df['raw_char_len'], bins=50, kde=True, color='#FF7F0E', label='Raw Characters', alpha=0.4, ax=ax)
    sns.histplot(df['clean_char_len'], bins=50, kde=True, color='#1F77B4', label='Cleaned Characters', alpha=0.5, ax=ax)
    
    raw_mean = df['raw_char_len'].mean()
    clean_mean = df['clean_char_len'].mean()
    
    ax.axvline(raw_mean, color='#D62728', linestyle='--', linewidth=1.5, label=f'Raw Mean ({raw_mean:.1f})')
    ax.axvline(clean_mean, color='#005580', linestyle=':', linewidth=1.5, label=f'Cleaned Mean ({clean_mean:.1f})')
    
    ax.set_title('Tweet Character Length Distribution (Pre vs Post Normalization)', fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel('Character Count', fontsize=11)
    ax.set_ylabel('Density / Frequency', fontsize=11)
    ax.set_xlim(0, 220)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=10)
    plt.tight_layout()
    
    out_path = os.path.join(FIG_DIR, 'text_length_distribution.png')
    plt.savefig(out_path)
    plt.close()
    print(f"Generated {out_path}")

def plot_frequent_terms(cleaned_csv: str = 'dataset/processed/cleaned_tweets.csv.gz'):
    """
    3. Frequent terms: Top meaningful terms after preprocessing.
    """
    df = pd.read_csv(cleaned_csv, compression='gzip', nrows=50000)
    
    # Common Indonesian stopwords to filter out for term frequency visualization
    stop_words = {
        'yang', 'aku', 'dan', 'di', 'ini', 'itu', 'dengan', 'untuk', 'pada', 'dari',
        'ke', 'karena', 'kalau', 'tapi', 'ada', 'sudah', 'bisa', 'akan', 'kita',
        'kamu', 'dia', 'mereka', 'sama', 'saja', 'juga', 'mau', 'jadi', 'lagi', 'lah',
        'kan', 'ya', 'kok', 'sih', 'nih', 'deh', 'dong', 'kah', 'pun', 'kalian'
    }
    
    all_words = []
    for text in df['text_clean'].dropna():
        words = text.split()
        all_words.extend([w for w in words if len(w) > 2 and w not in stop_words])
        
    counts = Counter(all_words).most_common(20)
    words, freqs = zip(*counts)
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    y_pos = np.arange(len(words))
    
    bars = ax.barh(y_pos, freqs, color='#2B5B84', edgecolor='black', alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(words, fontsize=10)
    ax.invert_yaxis()
    
    for bar in bars:
        w = bar.get_width()
        ax.annotate(f'{int(w):,}',
                    xy=(w, bar.get_y() + bar.get_height() / 2),
                    xytext=(4, 0), textcoords="offset points",
                    ha='left', va='center', fontsize=9)
        
    ax.set_title('Top 20 Frequent Content Terms in Tweet Corpus', fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel('Token Frequency', fontsize=11)
    ax.set_xlim(0, max(freqs) * 1.15)
    plt.tight_layout()
    
    out_path = os.path.join(FIG_DIR, 'frequent_terms.png')
    plt.savefig(out_path)
    plt.close()
    print(f"Generated {out_path}")

def plot_initial_label_distribution(seed_labels_path: str = 'dataset/seed_labels.csv'):
    """
    4. Initial label distribution: Distribution of seed labels with source identified.
    """
    df = pd.read_csv(seed_labels_path)
    
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    
    pivot = df.groupby(['label', 'label_source']).size().unstack(fill_value=0)
    
    colors = ['#4A90E2', '#50E3C2']
    bars = pivot.plot(kind='bar', stacked=True, ax=ax, color=colors, edgecolor='black', alpha=0.85)
    
    ax.set_title('Distribution of Seed and Reference Labels by Provenance', fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel('Sentiment Class', fontsize=11)
    ax.set_ylabel('Record Count', fontsize=11)
    plt.xticks(rotation=0, fontsize=10)
    ax.legend(title='Label Source', frameon=True, facecolor='white', framealpha=0.9)
    
    # Value annotations
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{int(height)}',
                        (p.get_x() + p.get_width() / 2., p.get_y() + height / 2.),
                        ha='center', va='center', fontsize=9, color='black', fontweight='bold')
            
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, 'initial_label_distribution.png')
    plt.savefig(out_path)
    plt.close()
    print(f"Generated {out_path}")

def plot_pseudo_label_distribution(pseudo_labels_path: str = 'dataset/processed/pseudo_labels.csv'):
    """
    5. Pseudo-label distribution: Class distribution after each self-training iteration.
    """
    if not os.path.exists(pseudo_labels_path):
        print(f"File {pseudo_labels_path} not found. Skipping.")
        return
        
    df = pd.read_csv(pseudo_labels_path)
    if 'predicted_label' in df.columns:
        label_col = 'predicted_label'
    else:
        label_col = 'label'
        
    pivot = df.groupby(['iteration', label_col]).size().unstack(fill_value=0)
    
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    
    palette = [CLASS_PALETTE.get(c, '#888888') for c in pivot.columns]
    bars = pivot.plot(kind='bar', ax=ax, color=palette, edgecolor='black', alpha=0.85, width=0.7)
    
    ax.set_title('Pseudo-Label Acquisition across Self-Training Iterations', fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel('Self-Training Iteration', fontsize=11)
    ax.set_ylabel('Acquired Pseudo-Labels Count', fontsize=11)
    plt.xticks(rotation=0, fontsize=10)
    ax.legend(title='Predicted Sentiment', frameon=True, facecolor='white')
    
    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(f'{int(h)}',
                        (p.get_x() + p.get_width() / 2., h),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
            
    ax.set_ylim(0, pivot.values.max() * 1.15)
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, 'pseudo_label_distribution.png')
    plt.savefig(out_path)
    plt.close()
    print(f"Generated {out_path}")

def plot_confidence_distribution(conf_sample_path: str = 'outputs/metrics/confidence_sample.csv.gz'):
    """
    6. Confidence distribution: Scores of accepted vs rejected pseudo-labels.
    """
    if not os.path.exists(conf_sample_path):
        print(f"File {conf_sample_path} not found. Skipping.")
        return
        
    df = pd.read_csv(conf_sample_path, compression='gzip')
    
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    
    accepted = df[df['accepted']]['confidence']
    rejected = df[~df['accepted']]['confidence']
    
    sns.histplot(rejected, bins=30, color='#E74C3C', label=f'Rejected (< Threshold, n={len(rejected):,})', alpha=0.5, ax=ax)
    sns.histplot(accepted, bins=20, color='#2ECC71', label=f'Accepted (>= Threshold, n={len(accepted):,})', alpha=0.7, ax=ax)
    
    ax.axvline(0.90, color='black', linestyle='--', linewidth=1.5, label='Optimal Threshold (0.90)')
    
    ax.set_title('Model Confidence Distribution for Unlabeled Tweet Pool', fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel('Max Class Probability Confidence', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.legend(frameon=True, facecolor='white')
    plt.tight_layout()
    
    out_path = os.path.join(FIG_DIR, 'confidence_distribution.png')
    plt.savefig(out_path)
    plt.close()
    print(f"Generated {out_path}")

def plot_model_comparison(metrics_path: str = 'outputs/metrics/test_metrics_comparison.json'):
    """
    7. Model comparison: Bar chart comparing baseline and self-trained model metrics.
    """
    if not os.path.exists(metrics_path):
        print(f"File {metrics_path} not found. Skipping.")
        return
        
    with open(metrics_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    metrics = ['Accuracy', 'Macro F1', 'Weighted F1']
    models = ['Dummy Majority', 'Seed-Only Baseline', 'Self-Trained Model']
    
    dummy_vals = [data['dummy_majority']['accuracy'], data['dummy_majority']['macro_f1'], data['dummy_majority']['weighted_f1']]
    base_vals = [data['seed_baseline']['accuracy'], data['seed_baseline']['macro_f1'], data['seed_baseline']['weighted_f1']]
    final_vals = [data['self_trained_final']['accuracy'], data['self_trained_final']['macro_f1'], data['self_trained_final']['weighted_f1']]
    
    x = np.arange(len(metrics))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    
    rects1 = ax.bar(x - width, dummy_vals, width, label='Dummy Majority', color='#95A5A6', edgecolor='black')
    rects2 = ax.bar(x, base_vals, width, label='Seed-Only Baseline', color='#E67E22', edgecolor='black')
    rects3 = ax.bar(x + width, final_vals, width, label='Self-Trained Model', color='#27AE60', edgecolor='black')
    
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('Test Set Performance Comparison Across Model Architectures', fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=True, facecolor='white')
    
    # Value annotations
    for rects in [rects1, rects2, rects3]:
        for rect in rects:
            h = rect.get_height()
            ax.annotate(f'{h:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, fontweight='bold')
            
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, 'model_comparison.png')
    plt.savefig(out_path)
    plt.close()
    print(f"Generated {out_path}")

def plot_confusion_matrix(metrics_path: str = 'outputs/metrics/test_metrics_comparison.json'):
    """
    8. Confusion matrix: Heatmap for the final model on the independent test set.
    """
    if not os.path.exists(metrics_path):
        print(f"File {metrics_path} not found. Skipping.")
        return
        
    with open(metrics_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    classes = data['classes']
    cm_base = np.array(data['seed_baseline']['confusion_matrix'])
    cm_final = np.array(data['self_trained_final']['confusion_matrix'])
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    
    sns.heatmap(cm_base, annot=True, fmt='d', cmap='Oranges', cbar=False,
                xticklabels=classes, yticklabels=classes, ax=axes[0])
    axes[0].set_title('Confusion Matrix: Seed-Only Baseline', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Predicted Label', fontsize=10)
    axes[0].set_ylabel('True Label (Silver Reference)', fontsize=10)
    
    sns.heatmap(cm_final, annot=True, fmt='d', cmap='Greens', cbar=False,
                xticklabels=classes, yticklabels=classes, ax=axes[1])
    axes[1].set_title('Confusion Matrix: Final Self-Trained Model', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Predicted Label', fontsize=10)
    axes[1].set_ylabel('True Label (Silver Reference)', fontsize=10)
    
    plt.suptitle('Test Set Confusion Matrices Comparison (N=500)', fontsize=14, fontweight='bold', y=1.03)
    plt.tight_layout()
    
    out_path = os.path.join(FIG_DIR, 'confusion_matrix.png')
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"Generated {out_path}")

def plot_per_class_performance(metrics_path: str = 'outputs/metrics/test_metrics_comparison.json'):
    """
    9. Per-class performance: Precision, recall, and F1-score by sentiment class.
    """
    if not os.path.exists(metrics_path):
        print(f"File {metrics_path} not found. Skipping.")
        return
        
    with open(metrics_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    final_rep = data['self_trained_final']['per_class']
    classes = ['positive', 'negative', 'neutral']
    
    prec = [final_rep[c]['precision'] for c in classes]
    rec = [final_rep[c]['recall'] for c in classes]
    f1 = [final_rep[c]['f1-score'] for c in classes]
    
    x = np.arange(len(classes))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    
    r1 = ax.bar(x - width, prec, width, label='Precision', color='#3498DB', edgecolor='black')
    r2 = ax.bar(x, rec, width, label='Recall', color='#F39C12', edgecolor='black')
    r3 = ax.bar(x + width, f1, width, label='F1-Score', color='#2ECC71', edgecolor='black')
    
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('Per-Class Performance Metrics (Final Self-Trained Model)', fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in classes], fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.legend(frameon=True, facecolor='white')
    
    for rects in [r1, r2, r3]:
        for rect in rects:
            h = rect.get_height()
            ax.annotate(f'{h:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, fontweight='bold')
            
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, 'per_class_performance.png')
    plt.savefig(out_path)
    plt.close()
    print(f"Generated {out_path}")

def plot_threshold_comparison(th_path: str = 'outputs/metrics/threshold_comparison.json'):
    """
    10. Threshold analysis: Validation Macro F1 and accepted volume across thresholds.
    """
    if not os.path.exists(th_path):
        print(f"File {th_path} not found. Skipping.")
        return
        
    with open(th_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    thresholds = sorted([float(k) for k in data.keys()])
    macro_f1s = [data[str(th)]['best_val_macro_f1'] for th in thresholds]
    pseudo_counts = [data[str(th)]['total_pseudo_labels'] for th in thresholds]
    
    fig, ax1 = plt.subplots(figsize=(8, 5), dpi=300)
    
    color1 = '#2980B9'
    ax1.set_xlabel('Confidence Threshold (tau)', fontsize=11)
    ax1.set_ylabel('Best Validation Macro F1', color=color1, fontsize=11)
    line1 = ax1.plot(thresholds, macro_f1s, color=color1, marker='o', linewidth=2.5, markersize=8, label='Validation Macro F1')
    ax1.tick_params(axis='y', labelcolor=color1)
    
    for t, f1 in zip(thresholds, macro_f1s):
        ax1.annotate(f'{f1:.4f}', (t, f1), xytext=(0, 8), textcoords='offset points', ha='center', fontweight='bold', color=color1)
        
    ax2 = ax1.twinx()
    color2 = '#8E44AD'
    ax2.set_ylabel('Acquired Pseudo-Labels Count', color=color2, fontsize=11)
    line2 = ax2.plot(thresholds, pseudo_counts, color=color2, marker='s', linestyle='--', linewidth=2, markersize=8, label='Pseudo-Labels Count')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    for t, cnt in zip(thresholds, pseudo_counts):
        ax2.annotate(f'{cnt:,}', (t, cnt), xytext=(0, -15), textcoords='offset points', ha='center', fontweight='bold', color=color2)
        
    plt.title('Self-Training Confidence Threshold Optimization (tau in {0.85, 0.90, 0.95})', fontsize=12, fontweight='bold', pad=15)
    plt.tight_layout()
    
    out_path = os.path.join(FIG_DIR, 'threshold_comparison.png')
    plt.savefig(out_path)
    plt.close()
    print(f"Generated {out_path}")

def plot_error_breakdown(error_path: str = 'outputs/metrics/error_analysis.json'):
    """
    11. Error breakdown: Qualitative linguistic error categories.
    """
    if not os.path.exists(error_path):
        print(f"File {error_path} not found. Skipping.")
        return
        
    with open(error_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    dist = data['error_distribution']
    cats = list(dist.keys())
    counts = list(dist.values())
    
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    
    colors = sns.color_palette('muted', len(cats))
    bars = ax.barh(cats, counts, color=colors, edgecolor='black', alpha=0.85)
    ax.invert_yaxis()
    
    for bar in bars:
        w = bar.get_width()
        pct = (w / data['total_errors']) * 100
        ax.annotate(f'{int(w)} ({pct:.1f}%)',
                    xy=(w, bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0), textcoords="offset points",
                    ha='left', va='center', fontsize=9, fontweight='bold')
        
    ax.set_title(f"Test Error Distribution by Linguistic Category\n(Total Errors: {data['total_errors']}/{data['total_test_samples']}, Error Rate: {data['error_rate']*100:.1f}%)",
                 fontsize=12, fontweight='bold', pad=15)
    ax.set_xlabel('Error Count', fontsize=11)
    ax.set_xlim(0, max(counts) * 1.25)
    plt.tight_layout()
    
    out_path = os.path.join(FIG_DIR, 'error_breakdown.png')
    plt.savefig(out_path)
    plt.close()
    print(f"Generated {out_path}")

def generate_all_visualizations():
    """
    Runs all visualization functions.
    """
    print("\n--- Generating Publication Visualizations ---")
    plot_dataset_overview()
    plot_text_length_distribution()
    plot_frequent_terms()
    plot_initial_label_distribution()
    plot_pseudo_label_distribution()
    plot_confidence_distribution()
    plot_model_comparison()
    plot_confusion_matrix()
    plot_per_class_performance()
    plot_threshold_comparison()
    plot_error_breakdown()
    print("All visualizations generated successfully.")

if __name__ == '__main__':
    generate_all_visualizations()
