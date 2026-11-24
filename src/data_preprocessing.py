"""
data_preprocessing.py
---------------------
Reproducible text cleaning, data audit, and normalization pipeline
for Indonesian tweet sentiment classification.
"""

import os
import json
import re
import unicodedata
import pandas as pd
import numpy as np

# Compile regexes once for performance
RE_URL = re.compile(r'https?://\S+|www\.\S+')
RE_MENTION = re.compile(r'@\w+')
RE_HASHTAG = re.compile(r'#(\w+)')
RE_LAUGHTER_HA = re.compile(r'\b(?:h+a+){2,}h*\b', re.IGNORECASE)
RE_LAUGHTER_WK = re.compile(r'\b(?:w+k+){2,}w*\b', re.IGNORECASE)
RE_LAUGHTER_HE = re.compile(r'\b(?:h+e+){2,}h*\b', re.IGNORECASE)
RE_CRYING_HU   = re.compile(r'\b(?:h+u+){2,}[hu]*\b', re.IGNORECASE)
RE_ELONGATION  = re.compile(r'([a-zA-Z])\1{2,}')
RE_NON_ALPHA   = re.compile(r'[^a-zA-Z0-9\s-]')
RE_SPACES      = re.compile(r'\s+')

# Standard Indonesian colloquial mappings (negations, intensifiers, frequent pronouns)
SLANG_MAP = {
    'yg': 'yang', 'dgn': 'dengan', 'klo': 'kalau', 'kalo': 'kalau',
    'bgt': 'banget', 'bgtu': 'begitu', 'krn': 'karena', 'sbb': 'sebab',
    'tp': 'tapi', 'sy': 'saya', 'km': 'kamu', 'jd': 'jadi',
    'udh': 'sudah', 'udah': 'sudah', 'blm': 'belum', 'dpt': 'dapat',
    'ga': 'tidak', 'gak': 'tidak', 'nggak': 'tidak', 'ngga': 'tidak',
    'kagak': 'tidak', 'gatau': 'tidak tahu', 'tdk': 'tidak', 'enggak': 'tidak',
    'bkn': 'bukan', 'jgn': 'jangan', 'bener': 'benar', 'beneran': 'benaran',
    'abis': 'habis', 'aja': 'saja', 'aje': 'saja', 'sm': 'sama',
    'dr': 'dari', 'utk': 'untuk', 'org': 'orang', 'bbrp': 'beberapa',
    'bisaa': 'bisa', 'makasii': 'makasih', 'makasi': 'makasih'
}

def clean_tweet(text: str) -> str:
    """
    Cleans a single tweet text conservatively while preserving sentiment
    cues, negation markers, and informal expressions.
    """
    if not isinstance(text, str):
        return ''
    
    # 1. Unicode normalization (NFKC)
    text = unicodedata.normalize('NFKC', text)
    
    # 2. Strip URLs and Mentions
    text = RE_URL.sub('', text)
    text = RE_MENTION.sub('', text)
    
    # 3. Hashtags: strip '#' symbol but preserve token
    text = RE_HASHTAG.sub(r'\1', text)
    
    # 4. Standardize laughter and crying
    text = RE_LAUGHTER_HA.sub('haha', text)
    text = RE_LAUGHTER_WK.sub('wkwk', text)
    text = RE_LAUGHTER_HE.sub('hehe', text)
    text = RE_CRYING_HU.sub('huhu', text)
    
    # 5. Elongation reduction: collapse 3+ consecutive identical chars to 2
    text = RE_ELONGATION.sub(r'\1\1', text)
    
    # 6. Lowercase
    text = text.lower()
    
    # 7. Normalize colloquial abbreviations
    tokens = [SLANG_MAP.get(tok, tok) for tok in text.split()]
    text = ' '.join(tokens)
    
    # 8. Punctuation cleanup (keep alphanumerics, hyphens, spaces)
    text = RE_NON_ALPHA.sub(' ', text)
    text = RE_SPACES.sub(' ', text).strip()
    
    return text

def run_preprocessing(
    input_csv: str = 'dataset/original/Indonesian Sentiment Tweet Dataset Unlabeled.csv',
    output_csv: str = 'dataset/processed/cleaned_tweets.csv.gz',
    audit_json: str = 'dataset/processed/audit_summary.json'
) -> pd.DataFrame:
    """
    Loads raw CSV without dropping the first row, assigns record_ids, cleans text,
    performs comprehensive data audit, and exports results.
    """
    print(f"Loading raw dataset from {input_csv}...")
    # CRITICAL: header=None to retain the first record
    df = pd.read_csv(input_csv, header=None, names=['text_raw'], dtype=str)
    n_total = len(df)
    print(f"Loaded {n_total:,} records.")
    
    # Assign persistent, deterministic record_id
    df['record_id'] = [f"tweet_{i:06d}" for i in range(n_total)]
    
    # Data audit on raw data
    raw_missing = int(df['text_raw'].isna().sum())
    raw_empty = int((df['text_raw'].str.strip() == '').sum())
    raw_duplicates = int(df.duplicated(subset=['text_raw']).sum())
    raw_unique = n_total - raw_duplicates
    
    raw_lens = df['text_raw'].fillna('').str.len()
    
    # Perform cleaning
    print("Running text normalization pipeline...")
    df['text_clean'] = [clean_tweet(t) for t in df['text_raw']]
    
    # Compute clean metrics
    df['raw_char_len'] = raw_lens
    df['clean_char_len'] = df['text_clean'].str.len()
    df['clean_word_count'] = df['text_clean'].apply(lambda s: len(s.split()) if s else 0)
    
    # Duplicate analysis on cleaned data
    df['is_raw_duplicate'] = df.duplicated(subset=['text_raw'], keep='first')
    df['is_clean_duplicate'] = df.duplicated(subset=['text_clean'], keep='first')
    
    clean_empty = int((df['clean_char_len'] == 0).sum())
    clean_duplicates = int(df['is_clean_duplicate'].sum())
    clean_unique = n_total - clean_duplicates
    
    audit_summary = {
        "dataset_name": "Indonesian Sentiment Tweet Dataset Unlabeled",
        "total_records": n_total,
        "first_record_preview": df.loc[0, 'text_raw'],
        "raw_missing_values": raw_missing,
        "raw_empty_strings": raw_empty,
        "raw_exact_duplicates": raw_duplicates,
        "raw_unique_records": raw_unique,
        "raw_char_len_mean": float(raw_lens.mean()),
        "raw_char_len_median": float(raw_lens.median()),
        "raw_char_len_std": float(raw_lens.std()),
        "raw_char_len_min": int(raw_lens.min()),
        "raw_char_len_max": int(raw_lens.max()),
        "clean_empty_strings": clean_empty,
        "clean_duplicates": clean_duplicates,
        "clean_unique_records": clean_unique,
        "clean_char_len_mean": float(df['clean_char_len'].mean()),
        "clean_char_len_median": float(df['clean_char_len'].median()),
        "clean_char_len_std": float(df['clean_char_len'].std()),
        "clean_char_len_min": int(df['clean_char_len'].min()),
        "clean_char_len_max": int(df['clean_char_len'].max()),
        "clean_word_count_mean": float(df['clean_word_count'].mean()),
        "clean_word_count_median": float(df['clean_word_count'].median())
    }
    
    print("\n=== Data Audit Summary ===")
    for k, v in audit_summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v}")
            
    os.makedirs(os.path.dirname(audit_json), exist_ok=True)
    with open(audit_json, 'w', encoding='utf-8') as f:
        json.dump(audit_summary, f, indent=2)
    print(f"Audit summary saved to {audit_json}")
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    print(f"Saving cleaned dataset to {output_csv}...")
    df.to_csv(output_csv, index=False, compression='gzip')
    print(f"Saved {len(df):,} records successfully.")
    
    return df

if __name__ == '__main__':
    run_preprocessing()
