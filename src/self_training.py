"""
self_training.py
----------------
Genuine iterative self-training pipeline with confidence thresholding,
class-imbalance monitoring, multi-threshold comparison, and comprehensive
tracking of accepted/rejected pseudo-labels.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

def build_pipeline():
    """
    Constructs the feature union and Logistic Regression classifier pipeline.
    """
    features = FeatureUnion([
        ('word_tfidf', TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_features=12000,
            sublinear_tf=True
        )),
        ('char_tfidf', TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=(3, 5),
            min_df=5,
            max_features=18000,
            sublinear_tf=True
        ))
    ])
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    return Pipeline([
        ('features', features),
        ('clf', clf)
    ])

def run_self_training_loop(
    train_seed_df: pd.DataFrame,
    val_df: pd.DataFrame,
    pool_df: pd.DataFrame,
    threshold: float = 0.90,
    max_iterations: int = 5,
    class_cap: int = 1500,
    save_confidence_sample: bool = True
):
    """
    Executes iterative self-training for a specific confidence threshold.
    Returns: best_model, history, all_pseudo_labels, confidence_records
    """
    print(f"\n==========================================")
    print(f"Starting Self-Training: Threshold = {threshold}")
    print(f"==========================================")
    
    current_train = train_seed_df[['record_id', 'text', 'label']].copy()
    current_train['origin'] = 'seed'
    current_train['confidence'] = 1.0
    current_train['iteration'] = 0
    
    current_pool = pool_df[['record_id', 'text_clean']].copy()
    
    history = []
    pseudo_labels_accumulated = []
    confidence_records = []
    
    best_macro_f1 = -1.0
    best_model = None
    best_iteration = 0
    
    for iteration in range(1, max_iterations + 1):
        print(f"\n--- Iteration {iteration} (Current Train Size: {len(current_train):,}) ---")
        
        # 1. Fit classifier on current training pool
        pipe = build_pipeline()
        pipe.fit(current_train['text'], current_train['label'])
        classes = pipe.classes_
        
        # 2. Evaluate on validation set
        val_preds = pipe.predict(val_df['text_clean'])
        val_acc = accuracy_score(val_df['label'], val_preds)
        val_macro_f1 = f1_score(val_df['label'], val_preds, average='macro', zero_division=0)
        val_weighted_f1 = f1_score(val_df['label'], val_preds, average='weighted', zero_division=0)
        rep = classification_report(val_df['label'], val_preds, output_dict=True, zero_division=0)
        
        print(f"Val Acc: {val_acc:.4f} | Macro F1: {val_macro_f1:.4f} | Weighted F1: {val_weighted_f1:.4f}")
        
        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            best_model = pipe
            best_iteration = iteration
            
        # 3. Predict on remaining unlabeled pool in batches
        print(f"Predicting on unlabeled pool ({len(current_pool):,} records remaining)...")
        pool_probs = pipe.predict_proba(current_pool['text_clean'].fillna(''))
        max_idx = np.argmax(pool_probs, axis=1)
        max_conf = np.max(pool_probs, axis=1)
        predicted_classes = classes[max_idx]
        
        # Log sample confidences for visualization
        if save_confidence_sample and iteration <= 2:
            sub_indices = np.random.choice(len(max_conf), size=min(10000, len(max_conf)), replace=False)
            for idx in sub_indices:
                confidence_records.append({
                    'threshold': threshold,
                    'iteration': iteration,
                    'confidence': float(max_conf[idx]),
                    'predicted_class': predicted_classes[idx],
                    'accepted': bool(max_conf[idx] >= threshold)
                })
                
        # 4. Filter by confidence threshold
        confident_mask = max_conf >= threshold
        total_confident = int(np.sum(confident_mask))
        total_rejected = len(current_pool) - total_confident
        print(f"Candidates meeting threshold {threshold}: {total_confident:,} | Rejected: {total_rejected:,}")
        
        if total_confident == 0:
            print("No confident samples found. Early stopping.")
            history.append({
                'iteration': iteration,
                'train_size': len(current_train),
                'val_accuracy': float(val_acc),
                'val_macro_f1': float(val_macro_f1),
                'val_weighted_f1': float(val_weighted_f1),
                'per_class': {c: rep[c] for c in ['positive', 'negative', 'neutral']},
                'total_confident': 0,
                'accepted_count': 0,
                'rejected_count': total_rejected,
                'accepted_by_class': {}
            })
            break
            
        cand_df = current_pool[confident_mask].copy()
        cand_df['predicted_label'] = predicted_classes[confident_mask]
        cand_df['confidence'] = max_conf[confident_mask]
        
        # 5. Class-imbalance monitoring and selection capping
        selected_slices = []
        accepted_class_counts = {}
        for c in classes:
            c_slice = cand_df[cand_df['predicted_label'] == c]
            if len(c_slice) > class_cap:
                c_slice = c_slice.sort_values(by='confidence', ascending=False).iloc[:class_cap]
            accepted_class_counts[c] = len(c_slice)
            selected_slices.append(c_slice)
            
        selected_batch = pd.concat(selected_slices, ignore_index=True)
        print(f"Accepted after class cap (max {class_cap}/class): {len(selected_batch):,} {accepted_class_counts}")
        
        # Save pseudo-label records
        selected_batch['iteration'] = iteration
        selected_batch['model_version'] = f"self_train_t{threshold}_iter{iteration}"
        pseudo_labels_accumulated.append(selected_batch)
        
        history.append({
            'iteration': iteration,
            'train_size': len(current_train),
            'val_accuracy': float(val_acc),
            'val_macro_f1': float(val_macro_f1),
            'val_weighted_f1': float(val_weighted_f1),
            'per_class': {c: rep[c] for c in ['positive', 'negative', 'neutral']},
            'total_confident': total_confident,
            'accepted_count': len(selected_batch),
            'rejected_count': total_rejected,
            'accepted_by_class': accepted_class_counts,
            'confidence_mean': float(cand_df['confidence'].mean()),
            'confidence_median': float(cand_df['confidence'].median())
        })
        
        # 6. Add accepted pseudo-labels to training set
        new_train_rows = pd.DataFrame({
            'record_id': selected_batch['record_id'],
            'text': selected_batch['text_clean'],
            'label': selected_batch['predicted_label'],
            'origin': f'pseudo_label_iter_{iteration}',
            'confidence': selected_batch['confidence'],
            'iteration': iteration
        })
        current_train = pd.concat([current_train, new_train_rows], ignore_index=True)
        
        # 7. Remove accepted samples from unlabeled pool
        accepted_ids = set(selected_batch['record_id'])
        current_pool = current_pool[~current_pool['record_id'].isin(accepted_ids)].copy()
        
    # Evaluate final fitted model
    final_pipe = build_pipeline()
    final_pipe.fit(current_train['text'], current_train['label'])
    final_val_preds = final_pipe.predict(val_df['text_clean'])
    final_acc = accuracy_score(val_df['label'], final_val_preds)
    final_macro_f1 = f1_score(val_df['label'], final_val_preds, average='macro', zero_division=0)
    final_weighted_f1 = f1_score(val_df['label'], final_val_preds, average='weighted', zero_division=0)
    
    if final_macro_f1 >= best_macro_f1:
        best_model = final_pipe
        best_macro_f1 = final_macro_f1
        best_iteration = max_iterations
        
    print(f"\nCompleted self-training for threshold {threshold}. Best Val Macro F1: {best_macro_f1:.4f} at Iteration {best_iteration}")
    
    all_pseudo_df = pd.concat(pseudo_labels_accumulated, ignore_index=True) if pseudo_labels_accumulated else pd.DataFrame()
    
    return {
        'threshold': threshold,
        'best_iteration': best_iteration,
        'best_val_macro_f1': float(best_macro_f1),
        'final_val_accuracy': float(final_acc),
        'final_val_macro_f1': float(final_macro_f1),
        'final_val_weighted_f1': float(final_weighted_f1),
        'history': history,
        'pseudo_df': all_pseudo_df,
        'best_model': best_model,
        'confidence_records': confidence_records
    }

def execute_self_training(
    train_seed_path: str = 'dataset/processed/train_seed.csv',
    val_path: str = 'dataset/processed/val_set.csv',
    pool_path: str = 'dataset/processed/unlabeled_pool.csv.gz',
    model_output_dir: str = 'models/final_model',
    metadata_output: str = 'models/model_metadata.json',
    pseudo_labels_output: str = 'dataset/processed/pseudo_labels.csv',
    history_output: str = 'outputs/metrics/self_training_history.json',
    threshold_output: str = 'outputs/metrics/threshold_comparison.json',
    conf_output: str = 'outputs/metrics/confidence_sample.csv.gz',
    pool_sample_size: int = 150000
):
    """
    Coordinates multi-threshold comparison, selects optimal self-training configuration,
    saves the final model, metadata, pseudo-labels, and logs.
    """
    print(f"Loading data: seed from {train_seed_path}, val from {val_path}, pool from {pool_path}...")
    train_seed_df = pd.read_csv(train_seed_path)
    val_df = pd.read_csv(val_path)
    
    # Load pool (sample 150,000 for balanced speed and representative coverage)
    print(f"Loading unlabeled pool sample ({pool_sample_size:,} records)...")
    pool_df = pd.read_csv(pool_path, compression='gzip', nrows=pool_sample_size)
    print(f"Pool loaded: {len(pool_df):,} records.")
    
    thresholds = [0.85, 0.90, 0.95]
    comparison_results = {}
    best_overall_res = None
    best_overall_f1 = -1.0
    all_conf_records = []
    
    for th in thresholds:
        res = run_self_training_loop(
            train_seed_df=train_seed_df,
            val_df=val_df,
            pool_df=pool_df,
            threshold=th,
            max_iterations=4,
            class_cap=1200
        )
        comparison_results[str(th)] = {
            'threshold': th,
            'best_iteration': res['best_iteration'],
            'best_val_macro_f1': res['best_val_macro_f1'],
            'final_val_accuracy': res['final_val_accuracy'],
            'final_val_macro_f1': res['final_val_macro_f1'],
            'final_val_weighted_f1': res['final_val_weighted_f1'],
            'total_pseudo_labels': len(res['pseudo_df']),
            'history': res['history']
        }
        all_conf_records.extend(res['confidence_records'])
        
        if res['best_val_macro_f1'] > best_overall_f1:
            best_overall_f1 = res['best_val_macro_f1']
            best_overall_res = res
            
    print(f"\n==========================================")
    print(f"Threshold Comparison Summary:")
    for th, stats in comparison_results.items():
        print(f"  Threshold {th}: Best Val Macro F1 = {stats['best_val_macro_f1']:.4f}, Total Pseudo-Labels = {stats['total_pseudo_labels']:,}")
    print(f"Optimal Threshold Selected: {best_overall_res['threshold']}")
    print(f"==========================================")
    
    # Save threshold comparison
    os.makedirs(os.path.dirname(threshold_output), exist_ok=True)
    with open(threshold_output, 'w', encoding='utf-8') as f:
        json.dump(comparison_results, f, indent=2)
    print(f"Saved threshold comparison to {threshold_output}")
    
    # Save confidence samples
    os.makedirs(os.path.dirname(conf_output), exist_ok=True)
    pd.DataFrame(all_conf_records).to_csv(conf_output, index=False, compression='gzip')
    print(f"Saved confidence score sample to {conf_output}")
    
    # Save best pseudo-labels
    best_pseudo_df = best_overall_res['pseudo_df']
    os.makedirs(os.path.dirname(pseudo_labels_output), exist_ok=True)
    best_pseudo_df.to_csv(pseudo_labels_output, index=False)
    print(f"Saved {len(best_pseudo_df):,} pseudo-labels to {pseudo_labels_output}")
    
    # Save history
    with open(history_output, 'w', encoding='utf-8') as f:
        json.dump(best_overall_res['history'], f, indent=2)
    print(f"Saved self-training history to {history_output}")
    
    # Save best model
    os.makedirs(model_output_dir, exist_ok=True)
    model_path = os.path.join(model_output_dir, 'model.joblib')
    joblib.dump(best_overall_res['best_model'], model_path)
    print(f"Saved final self-trained model to {model_path}")
    
    # Save metadata
    model_meta = {
        "model_type": "Self_Trained_TFIDF_LogisticRegression",
        "optimal_threshold": best_overall_res['threshold'],
        "best_iteration": best_overall_res['best_iteration'],
        "best_val_macro_f1": best_overall_res['best_val_macro_f1'],
        "final_val_accuracy": best_overall_res['final_val_accuracy'],
        "seed_training_samples": len(train_seed_df),
        "pseudo_labeled_samples": len(best_pseudo_df),
        "total_training_samples": len(train_seed_df) + len(best_pseudo_df),
        "pseudo_labels_class_distribution": best_pseudo_df['predicted_label'].value_counts().to_dict() if len(best_pseudo_df) > 0 else {},
        "vectorizer_config": {
            "word_ngram_range": [1, 2],
            "word_max_features": 12000,
            "char_ngram_range": [3, 5],
            "char_max_features": 18000
        },
        "classifier_config": {
            "C": 1.0,
            "max_iter": 1000,
            "solver": "lbfgs"
        }
    }
    with open(metadata_output, 'w', encoding='utf-8') as f:
        json.dump(model_meta, f, indent=2)
    print(f"Saved model metadata to {metadata_output}")
    
    return best_overall_res

if __name__ == '__main__':
    execute_self_training()
