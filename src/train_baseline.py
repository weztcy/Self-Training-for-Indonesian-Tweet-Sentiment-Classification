"""
train_baseline.py
-----------------
Trains the initial TF-IDF + Logistic Regression baseline on weak seed labels
and evaluates on the independent validation set.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix

def build_vectorizer():
    """
    Constructs a dual word and character-wb TF-IDF feature pipeline.
    """
    return FeatureUnion([
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

def train_baseline_models(
    train_seed_path: str = 'dataset/processed/train_seed.csv',
    val_path: str = 'dataset/processed/val_set.csv',
    model_output_path: str = 'models/baseline_model.joblib',
    metrics_output_path: str = 'outputs/metrics/baseline_metrics.json'
):
    """
    Trains baseline models (Dummy Majority and Seed-Only Logistic Regression)
    and logs evaluation metrics.
    """
    print(f"Loading seed training data from {train_seed_path}...")
    train_df = pd.read_csv(train_seed_path)
    val_df = pd.read_csv(val_path)
    
    X_train, y_train = train_df['text'].fillna(''), train_df['label']
    X_val, y_val = val_df['text_clean'].fillna(''), val_df['label']
    
    # 1. Majority class baseline
    dummy = DummyClassifier(strategy='most_frequent')
    dummy.fit(X_train, y_train)
    dummy_preds = dummy.predict(X_val)
    dummy_acc = accuracy_score(y_val, dummy_preds)
    dummy_macro_f1 = f1_score(y_val, dummy_preds, average='macro', zero_division=0)
    dummy_weighted_f1 = f1_score(y_val, dummy_preds, average='weighted', zero_division=0)
    
    print("\n--- Majority Class Dummy Baseline ---")
    print(f"Accuracy: {dummy_acc:.4f}, Macro F1: {dummy_macro_f1:.4f}, Weighted F1: {dummy_weighted_f1:.4f}")
    
    # 2. Seed-Only TF-IDF + Logistic Regression Baseline
    print("\nFitting Baseline TF-IDF + Logistic Regression on seed data...")
    vectorizer = build_vectorizer()
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    pipeline = Pipeline([
        ('features', vectorizer),
        ('clf', clf)
    ])
    
    pipeline.fit(X_train, y_train)
    val_preds = pipeline.predict(X_val)
    val_probs = pipeline.predict_proba(X_val)
    
    acc = accuracy_score(y_val, val_preds)
    macro_f1 = f1_score(y_val, val_preds, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_val, val_preds, average='weighted', zero_division=0)
    report_dict = classification_report(y_val, val_preds, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_val, val_preds, labels=['positive', 'negative', 'neutral']).tolist()
    
    print(f"\n--- Seed-Only Logistic Regression Baseline ---")
    print(f"Validation Accuracy: {acc:.4f}")
    print(f"Validation Macro F1: {macro_f1:.4f}")
    print(f"Validation Weighted F1: {weighted_f1:.4f}")
    print("\nClassification Report:\n", classification_report(y_val, val_preds, digits=4, zero_division=0))
    
    # Save artifacts
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(pipeline, model_output_path)
    print(f"Saved baseline model to {model_output_path}")
    
    metrics = {
        "model_name": "Seed_Only_Logistic_Regression",
        "training_samples": len(train_df),
        "validation_samples": len(val_df),
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class": {
            cls: {
                "precision": float(report_dict[cls]["precision"]),
                "recall": float(report_dict[cls]["recall"]),
                "f1_score": float(report_dict[cls]["f1-score"]),
                "support": int(report_dict[cls]["support"])
            }
            for cls in ['positive', 'negative', 'neutral']
        },
        "confusion_matrix_labels": ['positive', 'negative', 'neutral'],
        "confusion_matrix": cm,
        "dummy_majority_baseline": {
            "accuracy": float(dummy_acc),
            "macro_f1": float(dummy_macro_f1),
            "weighted_f1": float(dummy_weighted_f1)
        }
    }
    
    os.makedirs(os.path.dirname(metrics_output_path), exist_ok=True)
    with open(metrics_output_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved baseline metrics to {metrics_output_path}")
    
    return pipeline, metrics

if __name__ == '__main__':
    train_baseline_models()
