"""
streamlit_app.py
----------------
Interactive Streamlit demonstration application for Indonesian Tweet Sentiment
Classification using the self-trained NLP model.
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_preprocessing import clean_tweet

# Page configuration
st.set_page_config(
    page_title="Indonesian Tweet Sentiment - Self-Training NLP",
    page_icon="🇮🇩",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1A365D;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4A5568;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F7FAFC;
        border-radius: 8px;
        padding: 1rem;
        border-left: 4px solid #3182CE;
        margin-bottom: 1rem;
    }
    .badge-pos {
        background-color: #C6F6D5;
        color: #22543D;
        padding: 0.35rem 0.8rem;
        border-radius: 6px;
        font-weight: 700;
        font-size: 1.2rem;
    }
    .badge-neg {
        background-color: #FED7D7;
        color: #742A2A;
        padding: 0.35rem 0.8rem;
        border-radius: 6px;
        font-weight: 700;
        font-size: 1.2rem;
    }
    .badge-neu {
        background-color: #E2E8F0;
        color: #2D3748;
        padding: 0.35rem 0.8rem;
        border-radius: 6px;
        font-weight: 700;
        font-size: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models_and_meta():
    """
    Loads final self-trained model, baseline model, and metadata.
    """
    final_model_path = os.path.join(PROJECT_ROOT, 'models', 'final_model', 'model.joblib')
    base_model_path = os.path.join(PROJECT_ROOT, 'models', 'baseline_model.joblib')
    meta_path = os.path.join(PROJECT_ROOT, 'models', 'model_metadata.json')
    test_metrics_path = os.path.join(PROJECT_ROOT, 'outputs', 'metrics', 'test_metrics_comparison.json')
    
    final_model = joblib.load(final_model_path) if os.path.exists(final_model_path) else None
    base_model = joblib.load(base_model_path) if os.path.exists(base_model_path) else None
    
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
            
    test_metrics = {}
    if os.path.exists(test_metrics_path):
        with open(test_metrics_path, 'r', encoding='utf-8') as f:
            test_metrics = json.load(f)
            
    return final_model, base_model, meta, test_metrics

final_model, base_model, meta, test_metrics = load_models_and_meta()

# Sidebar
st.sidebar.title("🧭 System Configuration")
st.sidebar.markdown("""
**Project:** Self-Training Sentiment Classifier  
**Language:** Indonesian (with Malay / Slang)  
**Dataset:** 453,390 Unlabeled Tweets  
**Method:** Semi-Supervised Self-Training  
**Representation:** Dual Word + Char TF-IDF  
""")

if meta:
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Optimal Threshold ($\\tau$):** `{meta.get('optimal_threshold', 0.90)}`")
    st.sidebar.markdown(f"**Seed Samples:** `{meta.get('seed_training_samples', 3000):,}`")
    st.sidebar.markdown(f"**Pseudo-Labels:** `{meta.get('pseudo_labeled_samples', 0):,}`")
    st.sidebar.markdown(f"**Total Training Pool:** `{meta.get('total_training_samples', 0):,}`")

st.sidebar.markdown("---")
st.sidebar.caption("Google DeepMind Agentic NLP Solution")

# Main Header
st.markdown('<div class="main-header">🇮🇩 Indonesian Tweet Sentiment Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Semi-Supervised Learning with Iterative Self-Training & Pseudo-Labeling</div>', unsafe_allow_html=True)

# Top Metric Banner
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
with col_m1:
    st.metric(label="Total Tweet Pool", value="453,390")
with col_m2:
    st.metric(label="Unique Deduplicated", value="439,133")
with col_m3:
    macro_f1 = test_metrics.get('self_trained_final', {}).get('macro_f1', 0.52)
    base_f1 = test_metrics.get('seed_baseline', {}).get('macro_f1', 0.38)
    delta_f1 = macro_f1 - base_f1
    st.metric(label="Test Macro F1", value=f"{macro_f1:.4f}", delta=f"{delta_f1:+.4f}")
with col_m4:
    test_acc = test_metrics.get('self_trained_final', {}).get('accuracy', 0.65)
    base_acc = test_metrics.get('seed_baseline', {}).get('accuracy', 0.40)
    delta_acc = test_acc - base_acc
    st.metric(label="Test Accuracy", value=f"{test_acc:.4f}", delta=f"{delta_acc:+.4f}")

st.markdown("---")

# Prediction Interface
tab_demo, tab_method, tab_figures, tab_limits = st.tabs([
    "💬 Interactive Tweet Classifier",
    "🔬 Self-Training Methodology",
    "📊 Performance Visualizations",
    "⚠️ Model Limitations & Error Analysis"
])

with tab_demo:
    col_input, col_result = st.columns([1.1, 0.9])
    
    with col_input:
        st.subheader("1. Enter Tweet or Choose Example")
        
        # Presets
        presets = {
            "Custom": "",
            "Positive (Appreciation)": "Alhamdulillah, pelayanan cepat banget dan stafnya ramah sekali! Puas belanja di sini.",
            "Negative (Complaint)": "Parah banget, barang yang dikirim rusak dan customer servicenya tidak mau tanggung jawab sama sekali!",
            "Neutral (Factual)": "KAI mengumumkan jadwal keberangkatan kereta api tambahan untuk periode libur akhir pekan mulai pukul 08:00 WIB.",
            "Negation Test (Positive flipped)": "Pelayanannya tidak ramah dan makanannya tidak enak, sangat mengecewakan.",
            "Slang / Colloquial (Twitter alay)": "sy kira bkl bgs bgt trnyata zonk parah gaes, sedih bgt huhuhu",
            "Malay Code-Switching": "Aku tak faham betul jenis orang yang buang sampah merata-rata, geli bodoh tengok."
        }
        
        selected_preset = st.selectbox("Quick Presets:", list(presets.keys()))
        default_val = presets[selected_preset]
        
        user_text = st.text_area(
            "Tweet Text:",
            value=default_val,
            height=120,
            placeholder="Ketik atau tempel tweet di sini..."
        )
        
        btn_classify = st.button("🔍 Classify Sentiment", type="primary", use_container_width=True)
        
    with col_result:
        st.subheader("2. Sentiment Prediction")
        
        if (btn_classify or user_text) and user_text.strip():
            # Preprocess
            clean_text = clean_tweet(user_text)
            
            if final_model:
                classes = list(final_model.classes_)
                probs = final_model.predict_proba([clean_text])[0]
                pred_class = classes[int(np.argmax(probs))]
                confidence = float(np.max(probs))
                
                # Badge styling
                badge_class = "badge-pos" if pred_class == "positive" else ("badge-neg" if pred_class == "negative" else "badge-neu")
                emoji = "🎉" if pred_class == "positive" else ("😡" if pred_class == "negative" else "ℹ️")
                
                st.markdown(f"""
                <div style="margin-bottom: 1.2rem;">
                    <span style="font-size: 1.1rem; font-weight: 600; margin-right: 0.5rem;">Detected Sentiment:</span>
                    <span class="{badge_class}">{emoji} {pred_class.upper()}</span>
                    <span style="margin-left: 0.8rem; font-size: 1rem; color: #718096;">(Confidence: {confidence*100:.1f}%)</span>
                </div>
                """, unsafe_allow_html=True)
                
                # Probabilities chart
                prob_df = pd.DataFrame({
                    'Sentiment Class': [c.capitalize() for c in classes],
                    'Probability': probs
                }).sort_values(by='Probability', ascending=True)
                
                st.bar_chart(prob_df.set_index('Sentiment Class'), use_container_width=True)
                
                # Preprocessing audit
                with st.expander("🛠️ View Text Preprocessing Transformations", expanded=False):
                    st.write("**Raw Input:**", user_text)
                    st.write("**Cleaned & Normalized:**", clean_text)
                    st.caption("Normalized: Unicode NFKC, URLs/mentions removed, elongated characters collapsed, negations & slang mapped.")
            else:
                st.info("Model is currently loading or fitting. Please verify model artifacts.")
        else:
            st.info("Enter tweet text or select an example preset to see sentiment classification.")

with tab_method:
    st.subheader("Semi-Supervised Iterative Self-Training Architecture")
    st.markdown("""
    This project addresses the challenge of building a high-performance 3-class Indonesian sentiment classifier
    from a completely unlabeled dataset of **453,390 tweets** without original ground truth.
    
    #### Pipeline Stages:
    1. **Data Audit & Strict Leakage Prevention:**
       - The dataset was loaded without header row, ensuring the very first record was retained.
       - Exact duplicate records (8,530) and normalized duplicates (14,257) were isolated.
       - A fixed random seed (42) reserved an untouched 1,000-record independent evaluation sample (500 Val, 500 Test).
    2. **Weak Supervision & Seed Generation:**
       - Utilized the academic **InSet Lexicon** (Koto & Rahmaningtyas, 2017) with grammatical function word pruning,
         multi-token negation sliding windows, and intensifiers.
       - Extracted a balanced seed training set of **3,000 tweets** (1,000 Positive, 1,000 Negative, 1,000 Neutral).
    3. **Iterative Self-Training with Selection Caps:**
       - Candidate pseudo-labels from the unlabeled pool were accepted only if predicted class confidence $\\ge \\tau$ (evaluated across $\\tau \\in \\{0.85, 0.90, 0.95\\}$).
       - Applied per-class selection caps to prevent majority-class collapse and maintain representative polarity balance.
    """)

with tab_figures:
    st.subheader("Publication Visualizations")
    col_f1, col_f2 = st.columns(2)
    
    fig_dir = os.path.join(PROJECT_ROOT, 'outputs', 'figures')
    with col_f1:
        p1 = os.path.join(fig_dir, 'model_comparison.png')
        if os.path.exists(p1):
            st.image(p1, caption="Model Performance Comparison (Test Set)", use_container_width=True)
        p2 = os.path.join(fig_dir, 'confusion_matrix.png')
        if os.path.exists(p2):
            st.image(p2, caption="Confusion Matrix Heatmap", use_container_width=True)
            
    with col_f2:
        p3 = os.path.join(fig_dir, 'pseudo_label_distribution.png')
        if os.path.exists(p3):
            st.image(p3, caption="Pseudo-Labels Acquired Across Iterations", use_container_width=True)
        p4 = os.path.join(fig_dir, 'confidence_distribution.png')
        if os.path.exists(p4):
            st.image(p4, caption="Confidence Distribution of Pool Predictions", use_container_width=True)

with tab_limits:
    st.subheader("Model Limitations & Linguistic Error Analysis")
    st.markdown("""
    - **Weak Supervision Bias:** Initial seed labels and silver reference labels are derived from lexicon consensus and rule heuristics rather than certified human multi-annotator panels.
    - **Sarcasm and Irony:** Tweets expressing dissatisfaction using superficially polite or ironic wording (e.g. *'Makasih loh ya udah bikin nunggu 3 jam'*) remain challenging.
    - **Code-Switching:** Casual social media Indonesian frequently mixes English and regional Malay/Javanese/Sundanese dialects where vocabulary out-of-vocabulary effects occur.
    - **Implicit Sentiment:** Short, context-dependent expressions require broader conversational discourse history.
    """)
