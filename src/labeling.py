"""
labeling.py
-----------
Module for weak supervision, lexicon-based scoring, independent reference set creation,
inter-annotator agreement measurement, and data splitting without leakage.
"""

import os
import json
import re
import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score

# Function words / pronouns / prepositions that cause false polarity if scored naively
FUNCTION_WORDS_BLACKLIST = {
    'yang', 'aku', 'kamu', 'dia', 'mereka', 'kita', 'kami', 'kalian', 'ia', 'anda',
    'ini', 'itu', 'di', 'ke', 'dari', 'dan', 'atau', 'tapi', 'tetapi', 'karena',
    'jika', 'kalau', 'bila', 'dengan', 'untuk', 'pada', 'dalam', 'atas', 'bawah',
    'oleh', 'tentang', 'seperti', 'sebagai', 'bagi', 'sampai', 'hingga', 'ada',
    'adalah', 'ialah', 'yaitu', 'yakni', 'pun', 'lah', 'kah', 'tah', 'apa', 'siapa',
    'kapan', 'dimana', 'bagaimana', 'kenapa', 'mengapa', 'berapa', 'saja', 'juga',
    'sudah', 'telah', 'sedang', 'lagi', 'akan', 'harus', 'mesti', 'bisa', 'dapat',
    'boleh', 'mau', 'ingin', 'hendak', 'jangan', 'tidak', 'tak', 'bukan', 'belum',
    'tiada', 'tanpa', 'lebih', 'paling', 'sangat', 'amat', 'terlalu', 'sekali',
    'semua', 'seluruh', 'segala', 'setiap', 'tiap', 'beberapa', 'banyak', 'sedikit',
    'satu', 'dua', 'tiga', 'empat', 'lima', 'enam', 'tujuh', 'delapan', 'sembilan', 'sepuluh'
}

NEGATION_TOKENS = {
    'tidak', 'tak', 'bukan', 'belum', 'jangan', 'kurang', 'ga', 'gak',
    'nggak', 'ngga', 'kagak', 'nda', 'ogah', 'x', 'tiada', 'tanpa'
}

INTENSIFIER_TOKENS = {
    'banget', 'sangat', 'amat', 'sekali', 'bener-bener', 'betul-betul',
    'terlalu', 'parah', 'gila', 'betul', 'paling', 'luar biasa'
}

DIMINISHER_TOKENS = {
    'agak', 'sedikit', 'mungkin', 'kayaknya', 'sepertinya', 'kira-kira'
}

class InSetSentimentScorer:
    """
    Lexicon-based sentiment scoring engine based on InSet (Koto & Rahmaningtyas, 2017)
    with syntactic negation windows, intensifiers, and grammatical filtering.
    """
    def __init__(self, pos_path: str = 'dataset/lexicon/inset_positive.tsv',
                 neg_path: str = 'dataset/lexicon/inset_negative.tsv'):
        pos_df = pd.read_csv(pos_path, sep='\t')
        neg_df = pd.read_csv(neg_path, sep='\t')
        
        # Build clean polarity dictionaries excluding grammatical function words
        self.pos_dict = {
            str(row['word']).lower().strip(): float(row['weight'])
            for _, row in pos_df.iterrows()
            if str(row['word']).lower().strip() not in FUNCTION_WORDS_BLACKLIST
        }
        self.neg_dict = {
            str(row['word']).lower().strip(): float(row['weight'])
            for _, row in neg_df.iterrows()
            if str(row['word']).lower().strip() not in FUNCTION_WORDS_BLACKLIST
        }
        
        # High-salience conversational polarity cues
        self.pos_dict.update({
            'haha': 3.0, 'wkwk': 3.0, 'hehe': 2.0, 'makasih': 4.0,
            'terima kasih': 4.0, 'alhamdulillah': 5.0, 'mantap': 4.0,
            'keren': 4.0, 'bagus': 4.0, 'enak': 3.0, 'sayang': 3.0,
            'suka': 3.0, 'cinta': 4.0, 'syukur': 4.0, 'asik': 3.0
        })
        self.neg_dict.update({
            'huhu': -3.0, 'kecewa': -4.0, 'kesal': -4.0, 'marah': -4.0,
            'benci': -4.0, 'bodoh': -4.0, 'jelek': -4.0, 'buruk': -4.0,
            'gagal': -4.0, 'rusak': -3.0, 'rugi': -3.0, 'sedih': -4.0,
            'anjing': -5.0, 'bangsat': -5.0, 'mampus': -5.0, 'sial': -4.0
        })
        
    def score_text(self, text: str) -> dict:
        """
        Calculates polarity score, word matches, and confidence.
        """
        if not isinstance(text, str) or not text.strip():
            return {'score': 0.0, 'pos_sum': 0.0, 'neg_sum': 0.0, 'label': 'neutral', 'confidence': 0.50}
        
        tokens = text.lower().split()
        pos_sum = 0.0
        neg_sum = 0.0
        
        for i, tok in enumerate(tokens):
            # Check negation window of 1-3 tokens preceding current token
            window_neg = tokens[max(0, i - 3):i]
            is_negated = any(t in NEGATION_TOKENS for t in window_neg)
            
            # Check intensifier or diminisher in local window
            window_adj = tokens[max(0, i - 2):min(len(tokens), i + 3)]
            is_intensified = any(t in INTENSIFIER_TOKENS for t in window_adj)
            is_diminished = any(t in DIMINISHER_TOKENS for t in window_adj)
            
            scale = 1.5 if is_intensified else (0.7 if is_diminished else 1.0)
            
            if tok in self.pos_dict:
                val = self.pos_dict[tok] * scale
                if is_negated:
                    neg_sum += abs(val) * 0.9  # negated positive becomes negative
                else:
                    pos_sum += val
            elif tok in self.neg_dict:
                val = abs(self.neg_dict[tok]) * scale
                if is_negated:
                    pos_sum += val * 0.7       # negated negative mitigates or turns mild positive
                else:
                    neg_sum += val
                    
        net_score = pos_sum - neg_sum
        
        # Decision logic with confidence estimation
        if pos_sum >= 3.0 and neg_sum <= 1.0 and net_score >= 2.5:
            label = 'positive'
            confidence = min(0.98, 0.70 + 0.04 * (net_score - 2.5))
        elif neg_sum >= 3.0 and pos_sum <= 1.0 and net_score <= -2.5:
            label = 'negative'
            confidence = min(0.98, 0.70 + 0.04 * (abs(net_score) - 2.5))
        elif pos_sum < 1.5 and neg_sum < 1.5 and len(tokens) >= 4:
            # Factual or conversational lack of strong polarity
            label = 'neutral'
            confidence = 0.85
        else:
            # Low margin or mixed polarity
            if net_score > 1.5:
                label = 'positive'
                confidence = 0.65
            elif net_score < -1.5:
                label = 'negative'
                confidence = 0.65
            else:
                label = 'neutral'
                confidence = 0.60
                
        return {
            'score': round(net_score, 2),
            'pos_sum': round(pos_sum, 2),
            'neg_sum': round(neg_sum, 2),
            'label': label,
            'confidence': round(confidence, 4)
        }

def rule_heuristic_annotator(text: str) -> dict:
    """
    Secondary independent rule/pattern annotator used to compute
    inter-annotator agreement and form an ensemble consensus silver reference.
    """
    if not isinstance(text, str):
        return {'label': 'neutral', 'confidence': 0.50}
    
    t = text.lower()
    
    # Positive heuristic patterns
    pos_patterns = [
        r'\b(alhamdulillah|syukur|mantap|keren|hebat|terima kasih|makasih|senang|bahagia|suka|cinta|beruntung|semangat|puas)\b',
        r'\b(haha|wkwk|hehe)\b',
        r'\b(tidak (kecewa|rusak|gagal|sedih|marah))\b'
    ]
    # Negative heuristic patterns
    neg_patterns = [
        r'\b(kecewa|sedih|marah|kesal|benci|anjing|bangsat|bodoh|gagal|parah|rugi|rusak|mampus|sial|sakit hati)\b',
        r'\b(huhu)\b',
        r'\b(tidak (suka|puas|senang|enak|bagus|bisa|ramah))\b'
    ]
    
    pos_matches = sum(len(re.findall(p, t)) for p in pos_patterns)
    neg_matches = sum(len(re.findall(p, t)) for p in neg_patterns)
    
    if pos_matches > neg_matches and pos_matches > 0:
        return {'label': 'positive', 'confidence': min(0.95, 0.70 + 0.1 * pos_matches)}
    elif neg_matches > pos_matches and neg_matches > 0:
        return {'label': 'negative', 'confidence': min(0.95, 0.70 + 0.1 * neg_matches)}
    else:
        return {'label': 'neutral', 'confidence': 0.75}

def partition_and_label_dataset(
    cleaned_csv: str = 'dataset/processed/cleaned_tweets.csv.gz',
    output_dir: str = 'dataset/processed',
    seed_labels_path: str = 'dataset/seed_labels.csv',
    random_state: int = 42
):
    """
    Partitions the unique tweets, constructs independent evaluation sets (500 Val, 500 Test),
    computes inter-annotator agreement, generates balanced seed labels (3,000) for training,
    and isolates the remaining pool for self-training.
    """
    print(f"Loading preprocessed tweets from {cleaned_csv}...")
    df = pd.read_csv(cleaned_csv, compression='gzip')
    
    # 1. Strict deduplication and validity filter
    df_valid = df[~df['is_clean_duplicate'] & (df['clean_char_len'] > 0)].copy()
    print(f"Deduplicated valid tweets: {len(df_valid):,} records.")
    
    # 2. Sample 1,000 independent evaluation records (500 Val, 500 Test)
    # Using fixed random_state for exact reproducibility
    eval_candidates = df_valid.sample(n=1000, random_state=random_state)
    eval_indices = eval_candidates.index
    
    val_df = eval_candidates.iloc[:500].copy()
    test_df = eval_candidates.iloc[500:].copy()
    
    # Remove evaluation records from training pool immediately to guarantee ZERO leakage
    train_pool_df = df_valid.drop(index=eval_indices).copy()
    print(f"Independent evaluation pool: 1,000 records (500 Val, 500 Test).")
    print(f"Remaining training pool: {len(train_pool_df):,} records.")
    
    # 3. Annotate evaluation sets with multi-method silver consensus and measure agreement
    scorer = InSetSentimentScorer()
    
    for split_name, s_df in [('val', val_df), ('test', test_df)]:
        labels_annotator1 = []
        labels_annotator2 = []
        confidences = []
        final_labels = []
        
        for text in s_df['text_clean']:
            res1 = scorer.score_text(text)
            res2 = rule_heuristic_annotator(text)
            
            l1 = res1['label']
            l2 = res2['label']
            labels_annotator1.append(l1)
            labels_annotator2.append(l2)
            
            # Consensus resolution:
            # If both agree -> unanimous consensus
            # If disagreement: if one is neutral and other has high confidence polarity, prioritize polarity
            # Otherwise default to the more conservative scorer
            if l1 == l2:
                final_l = l1
                conf = max(res1['confidence'], res2['confidence'])
            elif res1['confidence'] >= 0.85 and res2['confidence'] < 0.80:
                final_l = l1
                conf = res1['confidence'] * 0.90
            elif res2['confidence'] >= 0.85 and res1['confidence'] < 0.80:
                final_l = l2
                conf = res2['confidence'] * 0.90
            else:
                final_l = 'neutral'
                conf = 0.70
                
            final_labels.append(final_l)
            confidences.append(conf)
            
        s_df['label_annotator1'] = labels_annotator1
        s_df['label_annotator2'] = labels_annotator2
        s_df['label'] = final_labels
        s_df['confidence'] = confidences
        s_df['label_source'] = 'silver_reference_consensus'
        s_df['review_status'] = 'silver_reference_set'
        
        # Inter-annotator agreement
        kappa = cohen_kappa_score(labels_annotator1, labels_annotator2)
        agreement_pct = np.mean(np.array(labels_annotator1) == np.array(labels_annotator2)) * 100
        print(f"Evaluation split '{split_name}': Cohen's Kappa = {kappa:.4f}, Raw Agreement = {agreement_pct:.2f}%")
        print(f"  Class distribution:\n{s_df['label'].value_counts().to_dict()}")

    # 4. Generate Weak Seed Training Set (3,000 balanced records) strictly from train_pool_df
    print("\nScoring candidate seeds from training pool...")
    # To be efficient, evaluate first 50,000 tweets from training pool to find 1,000 high-confidence per class
    sample_for_seeds = train_pool_df.sample(n=min(50000, len(train_pool_df)), random_state=random_state)
    
    seed_records = []
    pos_seeds, neg_seeds, neu_seeds = [], [], []
    
    for _, row in sample_for_seeds.iterrows():
        res = scorer.score_text(row['text_clean'])
        lbl = res['label']
        conf = res['confidence']
        
        # High confidence selection threshold
        if lbl == 'positive' and conf >= 0.85 and len(pos_seeds) < 1000:
            pos_seeds.append({
                'record_id': row['record_id'],
                'text': row['text_clean'],
                'text_raw': row['text_raw'],
                'label': 'positive',
                'label_source': 'inset_lexicon_weak_supervision',
                'confidence': conf,
                'review_status': 'weak_seed'
            })
        elif lbl == 'negative' and conf >= 0.85 and len(neg_seeds) < 1000:
            neg_seeds.append({
                'record_id': row['record_id'],
                'text': row['text_clean'],
                'text_raw': row['text_raw'],
                'label': 'negative',
                'label_source': 'inset_lexicon_weak_supervision',
                'confidence': conf,
                'review_status': 'weak_seed'
            })
        elif lbl == 'neutral' and conf >= 0.85 and len(neu_seeds) < 1000:
            neu_seeds.append({
                'record_id': row['record_id'],
                'text': row['text_clean'],
                'text_raw': row['text_raw'],
                'label': 'neutral',
                'label_source': 'inset_lexicon_weak_supervision',
                'confidence': conf,
                'review_status': 'weak_seed'
            })
            
        if len(pos_seeds) == 1000 and len(neg_seeds) == 1000 and len(neu_seeds) == 1000:
            break
            
    train_seed_df = pd.DataFrame(pos_seeds + neg_seeds + neu_seeds)
    print(f"Seed set selected: {len(train_seed_df)} records (Pos: {len(pos_seeds)}, Neg: {len(neg_seeds)}, Neu: {len(neu_seeds)})")
    
    # 5. Form the Unlabeled Pool by removing seed records from train_pool_df
    seed_record_ids = set(train_seed_df['record_id'])
    unlabeled_pool_df = train_pool_df[~train_pool_df['record_id'].isin(seed_record_ids)].copy()
    print(f"Unlabeled training pool size: {len(unlabeled_pool_df):,} records.")
    
    # 6. Save split files
    os.makedirs(output_dir, exist_ok=True)
    val_df.to_csv(os.path.join(output_dir, 'val_set.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test_set.csv'), index=False)
    train_seed_df.to_csv(os.path.join(output_dir, 'train_seed.csv'), index=False)
    unlabeled_pool_df[['record_id', 'text_clean', 'clean_char_len', 'clean_word_count']].to_csv(
        os.path.join(output_dir, 'unlabeled_pool.csv.gz'), index=False, compression='gzip'
    )
    
    # 7. Save consolidated seed_labels.csv (containing weak seeds + reference sets with provenance)
    combined_labels = pd.concat([
        train_seed_df[['record_id', 'text', 'label', 'label_source', 'confidence', 'review_status']],
        val_df[['record_id', 'text_clean', 'label', 'label_source', 'confidence', 'review_status']].rename(columns={'text_clean': 'text'}),
        test_df[['record_id', 'text_clean', 'label', 'label_source', 'confidence', 'review_status']].rename(columns={'text_clean': 'text'})
    ], ignore_index=True)
    
    os.makedirs(os.path.dirname(seed_labels_path), exist_ok=True)
    combined_labels.to_csv(seed_labels_path, index=False)
    print(f"Consolidated seed_labels.csv saved to {seed_labels_path} with {len(combined_labels):,} records.")
    
    # 8. Save metadata
    splits_meta = {
        "random_state": random_state,
        "total_unique_deduplicated": len(df_valid),
        "validation_records": len(val_df),
        "test_records": len(test_df),
        "train_seed_records": len(train_seed_df),
        "unlabeled_pool_records": len(unlabeled_pool_df),
        "seed_class_distribution": train_seed_df['label'].value_counts().to_dict(),
        "val_class_distribution": val_df['label'].value_counts().to_dict(),
        "test_class_distribution": test_df['label'].value_counts().to_dict(),
        "val_cohen_kappa": float(cohen_kappa_score(val_df['label_annotator1'], val_df['label_annotator2'])),
        "test_cohen_kappa": float(cohen_kappa_score(test_df['label_annotator1'], test_df['label_annotator2'])),
        "seed_record_ids_sample": train_seed_df['record_id'].head(10).tolist(),
        "val_record_ids_sample": val_df['record_id'].head(10).tolist(),
        "test_record_ids_sample": test_df['record_id'].head(10).tolist()
    }
    
    with open(os.path.join(output_dir, 'splits_metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(splits_meta, f, indent=2)
    print(f"Saved splits metadata to {os.path.join(output_dir, 'splits_metadata.json')}")
    
    return train_seed_df, val_df, test_df, unlabeled_pool_df

if __name__ == '__main__':
    partition_and_label_dataset()
