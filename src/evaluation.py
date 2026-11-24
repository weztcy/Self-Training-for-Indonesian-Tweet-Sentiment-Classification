"""
evaluation.py
-------------
Evaluates the baseline vs self-trained models on the isolated test set,
computes publication-grade metrics, and performs structured error analysis.
"""

import os
import sys
import json
import joblib
import re
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from src.train_baseline import build_vectorizer
from sklearn.linear_model import LogisticRegression

def perform_error_categorization(row) -> str:
    """
    Categorizes the nature of a classification error.
    """
    if row['is_correct']:
        return 'Correct'
    
    true_lbl = row['true_label']
    pred_lbl = row['pred_label']
    text = str(row['text']).lower()
    
    # Check negation presence
    has_negation = bool(re.search(r'\b(tidak|bukan|belum|jangan|tak|kurang|ga|gak|nggak|nda)\b', text))
    
    # Polar inversion (pos <-> neg)
    if {true_lbl, pred_lbl} == {'positive', 'negative'}:
        if has_negation:
            return 'Negation Misinterpretation'
        return 'Polar Inversion'
        
    # Neutral vs Polar confusion
    if true_lbl == 'neutral' and pred_lbl in ['positive', 'negative']:
        if has_negation:
            return 'Negation / Ambiguity Error'
        return 'Neutral False Polar (Over-triggering)'
        
    if true_lbl in ['positive', 'negative'] and pred_lbl == 'neutral':
        if bool(re.search(r'\b(nak|je|ni|tengok|pon|takde|lah|kan|lorh)\b', text)):
            return 'Colloquial / Dialect Nuance'
        return 'Polar False Neutral (Under-triggering)'
        
    return 'Contextual Ambiguity'

def evaluate_models_on_test(
    test_csv: str = 'dataset/processed/test_set.csv',
    train_seed_csv: str = 'dataset/processed/train_seed.csv',
    pseudo_labels_csv: str = 'dataset/processed/pseudo_labels.csv',
    baseline_model_path: str = 'models/baseline_model.joblib',
    final_model_dir: str = 'models/final_model',
    output_metrics_json: str = 'outputs/metrics/test_metrics_comparison.json',
    output_predictions_csv: str = 'outputs/predictions/test_predictions.csv',
    output_error_analysis_json: str = 'outputs/metrics/error_analysis.json'
):
    """
    Comprehensive test set evaluation comparing Dummy, Baseline, and Self-Trained models.
    """
    print(f"Loading test set from {test_csv}...")
    test_df = pd.read_csv(test_csv)
    X_test = test_df['text_clean'].fillna('')
    y_test = test_df['label']
    
    # Load baseline model
    print(f"Loading baseline model from {baseline_model_path}...")
    baseline_model = joblib.load(baseline_model_path)
    
    # Train full self-trained model on Seed (3,000) + Pseudo-labels (14,400)
    print("Fitting Self-Trained Final Model on Seed + Acquired Pseudo-Labels...")
    train_seed_df = pd.read_csv(train_seed_csv)
    pseudo_df = pd.read_csv(pseudo_labels_csv)
    
    combined_train = pd.concat([
        train_seed_df[['record_id', 'text', 'label']],
        pseudo_df[['record_id', 'text_clean', 'predicted_label']].rename(
            columns={'text_clean': 'text', 'predicted_label': 'label'}
        )
    ], ignore_index=True)
    
    final_model = Pipeline([
        ('features', build_vectorizer()),
        ('clf', LogisticRegression(C=1.0, max_iter=1000, random_state=42))
    ])
    final_model.fit(combined_train['text'], combined_train['label'])
    
    # Save the retrained final model
    os.makedirs(final_model_dir, exist_ok=True)
    joblib.dump(final_model, os.path.join(final_model_dir, 'model.joblib'))
    print(f"Saved retrained final model to {os.path.join(final_model_dir, 'model.joblib')}")
    
    # Dummy majority baseline
    dummy = DummyClassifier(strategy='most_frequent')
    dummy.fit(train_seed_df['text'].fillna(''), train_seed_df['label'])
    dummy_preds = dummy.predict(X_test)
    
    # Predict
    base_preds = baseline_model.predict(X_test)
    base_probs = baseline_model.predict_proba(X_test)
    
    final_preds = final_model.predict(X_test)
    final_probs = final_model.predict_proba(X_test)
    
    classes = ['positive', 'negative', 'neutral']
    
    def get_summary(preds):
        return {
            'accuracy': float(accuracy_score(y_test, preds)),
            'macro_f1': float(f1_score(y_test, preds, average='macro', zero_division=0)),
            'weighted_f1': float(f1_score(y_test, preds, average='weighted', zero_division=0)),
            'per_class': classification_report(y_test, preds, output_dict=True, zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, preds, labels=classes).tolist()
        }
        
    dummy_summary = get_summary(dummy_preds)
    base_summary = get_summary(base_preds)
    final_summary = get_summary(final_preds)
    
    print("\n=== TEST SET PERFORMANCE COMPARISON ===")
    print(f"Dummy Majority Classifier   : Acc = {dummy_summary['accuracy']:.4f}, Macro F1 = {dummy_summary['macro_f1']:.4f}, Weighted F1 = {dummy_summary['weighted_f1']:.4f}")
    print(f"Seed-Only Baseline Model    : Acc = {base_summary['accuracy']:.4f}, Macro F1 = {base_summary['macro_f1']:.4f}, Weighted F1 = {base_summary['weighted_f1']:.4f}")
    print(f"Self-Trained Final Model    : Acc = {final_summary['accuracy']:.4f}, Macro F1 = {final_summary['macro_f1']:.4f}, Weighted F1 = {final_summary['weighted_f1']:.4f}")
    
    # Build detailed prediction table
    pred_df = pd.DataFrame({
        'record_id': test_df['record_id'],
        'text': test_df['text_clean'],
        'true_label': y_test,
        'baseline_pred': base_preds,
        'baseline_conf': np.max(base_probs, axis=1),
        'final_pred': final_preds,
        'final_conf': np.max(final_probs, axis=1),
        'is_correct_baseline': (base_preds == y_test),
        'is_correct_final': (final_preds == y_test)
    })
    
    pred_df['is_correct'] = pred_df['is_correct_final']
    pred_df['pred_label'] = pred_df['final_pred']
    pred_df['error_category'] = pred_df.apply(perform_error_categorization, axis=1)
    
    os.makedirs(os.path.dirname(output_predictions_csv), exist_ok=True)
    pred_df.to_csv(output_predictions_csv, index=False)
    print(f"Saved predictions with error tags to {output_predictions_csv}")
    
    # Error analysis breakdown
    errors_only = pred_df[~pred_df['is_correct']].copy()
    error_counts = errors_only['error_category'].value_counts().to_dict()
    
    examples_by_category = {}
    for cat in error_counts.keys():
        samples = errors_only[errors_only['error_category'] == cat].head(3)
        examples_by_category[cat] = [
            {
                'record_id': r['record_id'],
                'text': r['text'],
                'true_label': r['true_label'],
                'pred_label': r['final_pred'],
                'confidence': float(round(r['final_conf'], 4))
            }
            for _, r in samples.iterrows()
        ]
        
    error_analysis_data = {
        'total_test_samples': len(test_df),
        'total_errors': len(errors_only),
        'error_rate': float(len(errors_only) / len(test_df)),
        'error_distribution': error_counts,
        'category_examples': examples_by_category
    }
    
    os.makedirs(os.path.dirname(output_error_analysis_json), exist_ok=True)
    with open(output_error_analysis_json, 'w', encoding='utf-8') as f:
        json.dump(error_analysis_data, f, indent=2)
    print(f"Saved error analysis to {output_error_analysis_json}")
    
    # Save consolidated test metrics
    test_metrics = {
        'test_set_size': len(test_df),
        'test_class_distribution': y_test.value_counts().to_dict(),
        'classes': classes,
        'dummy_majority': dummy_summary,
        'seed_baseline': base_summary,
        'self_trained_final': final_summary
    }
    os.makedirs(os.path.dirname(output_metrics_json), exist_ok=True)
    with open(output_metrics_json, 'w', encoding='utf-8') as f:
        json.dump(test_metrics, f, indent=2)
    print(f"Saved test metrics to {output_metrics_json}")
    
    return test_metrics

if __name__ == '__main__':
    evaluate_models_on_test()
