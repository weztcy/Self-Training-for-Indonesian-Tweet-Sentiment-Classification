# Portfolio Project Data — Self-Training for Indonesian Tweet Sentiment Classification

- **Category:** `machine-learning` (`Machine Learning`)
- **Accent Color:** `indigo`
- **Target Assets Folder:** `public/images/projects/self_training_for_indonesian/`
- **Required Local Images:** `banner.webp`, `image01.webp` .. `image09.webp`

## TypeScript Data (`ProjectCaseStudy`)

```typescript
{
  "id": "self-training-for-indonesian-tweet-sentiment-classification",
  "code": "PRO-30 // ML-15",
  "title": "Self-Training for Indonesian Tweet Sentiment Classification",
  "category": "machine-learning",
  "categoryLabel": "Semi-Supervised Learning",
  "year": "2024",
  "status": "Completed",
  "accentColor": "indigo",
  "repoUrl": "https://github.com/weztcy/Self-Training-for-Indonesian-Tweet-Sentiment-Classification",
  "license": "MIT License",
  "banner": "/images/projects/self_training_for_indonesian/banner.webp",
  "summary": "Semi-supervised self-training NLP pipeline classifying 453,390 unlabeled Indonesian tweets using curated InSet weak supervision and dual TF-IDF.",
  "projectDescription": "An end-to-end semi-supervised sentiment classification system trained on 453,390 unlabeled Indonesian tweets. Combines audited InSet weak supervision, dual word-and-character TF-IDF Logistic Regression, multi-threshold self-training, error analysis, and an interactive Streamlit demo.",
  "architectureSummary": "Headerless Raw Tweets -> Conservative Slang & Negation Normalization -> Seed-42 Leakage-Free Split -> InSet Weak Seeds -> Dual TF-IDF Logistic Regression -> Capped Iterative Self-Training -> Test Error Audit & Streamlit UI",
  "techStack": [
    "Python",
    "Scikit-Learn",
    "Pandas",
    "NumPy",
    "SciPy",
    "Matplotlib",
    "Seaborn",
    "Joblib",
    "Streamlit"
  ],
  "telemetryMatrix": {
    "projectType": "Semi-Supervised NLP Sentiment Pipeline",
    "targetPlatform": "Python CLI & Streamlit Web App",
    "primaryDomain": "Indonesian Social Media Text Mining",
    "architecturePipeline": "InSet Weak Seeds + Dual TF-IDF Self-Training",
    "testingAndAudit": "Verified",
    "projectStatus": "Completed",
    "projectYear": "2024"
  },
  "mlProjectInfo": {
    "projectType": "Semi-Supervised NLP Classification Pipeline",
    "learningParadigm": "Semi-Supervised Learning",
    "modelDomain": "Indonesian Tweet Sentiment Analysis",
    "projectStatus": "Completed",
    "developmentYear": "2024",
    "capabilityBadges": [
      "Weak Supervision Bootstrap",
      "Iterative Pseudo Labeling",
      "Linguistic Error Audit"
    ]
  },
  "mlProblemContext": "Large-scale Indonesian social media corpora often lack ground-truth sentiment labels while exhibiting informal slang, non-standard elongations, negation diversity, and Malay code-switching. Bootstrapping reliable classifiers without annotation leakage and diagnosing confirmation bias during iterative self-training remain core challenges.",
  "mlDatasetOverview": {
    "profileBadge": "453K Tweet Corpus",
    "source": "Indonesian Sentiment Tweet Dataset Unlabeled & InSet Lexicon",
    "modality": "NLP",
    "scale": "453,390 Raw Records / 439,132 Valid Unique Tweets",
    "inputShape": "N x 30,000 Sparse Word + Char-WB TF-IDF Features",
    "dataFormat": "CSV / Gzipped CSV / TSV",
    "description": "Headerless single-column corpus audited to remove 14,257 normalized duplicates and 11 empty strings, partitioned via seed 42 into 500 validation, 500 test, 3,000 balanced weak-seed training tweets, and 435,132 unlabeled pool records.",
    "provenanceNote": "Evaluation splits use dual-annotator silver consensus (InSet scorer and rule heuristics); seed training uses curated InSet weak supervision."
  },
  "mlFeaturesAndDataCharacteristics": {
    "spaceBadge": "Dual Sparse TF-IDF",
    "spaceNote": "30,000 sublinear TF-IDF dimensions merging word n-grams and character-boundary subwords",
    "objectiveOutput": {
      "label": "Target Output",
      "name": "3-Class Tweet Sentiment",
      "type": "Multi-Class",
      "distribution": "Seed: 1,000/class balanced; Test: 394 Neutral (78.8%), 82 Positive (16.4%), 24 Negative (4.8%)"
    },
    "inputSignals": [
      {
        "name": "Word N-Gram TF-IDF (1-2)",
        "type": "Sparse Text Feature",
        "importanceOrRole": "12,000 dims / Phrase & negation cues",
        "description": "Captures lexical sentiment tokens and negated bigrams such as tidak suka and tak reti after slang normalization."
      },
      {
        "name": "Character-WB TF-IDF (3-5)",
        "type": "Sparse Subword Feature",
        "importanceOrRole": "18,000 dims / Morphology & slang",
        "description": "Models Indonesian prefixes, suffixes, character elongations, and dialectal spelling mutations within word boundaries."
      },
      {
        "name": "InSet Polarity & Negation Window",
        "type": "Weak Supervision Signal",
        "importanceOrRole": "Seed & silver consensus generation",
        "description": "Scores tweets using pruned InSet weights (59 function words blacklisted), 3-token negation inversion, and intensifier scaling."
      }
    ],
    "dataCharacteristicsSummary": "Cleaned tweets average 74.42 characters and 12.67 words with heavy use of informal negations, expressive particles, and regional code-switching. While weak seeds are balanced at 1,000 per class, the natural held-out test split is 78.8% neutral, creating sharp prior shift."
  },
  "mlLearningApproach": {
    "paradigm": "Semi-Supervised Learning via Weak Supervision & Iterative Self-Training",
    "methodologyDescription": "An audited InSet weak-supervision scorer initializes a 3,000-sample balanced seed model using dual TF-IDF Logistic Regression. The classifier then runs four self-training iterations over 150,000 unlabeled tweets, ingesting capped high-confidence pseudo-labels across thresholds 0.85, 0.90, and 0.95.",
    "justification": "Combining linear L2-regularized probabilities over 30,000 sparse word-and-subword features with per-class selection caps provides transparent decision boundaries, rapid CPU iteration over 150,000 unlabeled tweets, and direct observability of confirmation-bias drift."
  },
  "mlDataPreparation": [
    {
      "step": "01",
      "title": "Headerless Audit & Text Normalization",
      "description": "Loaded 453,390 raw tweets with header=None, applied NFKC normalization, stripped URLs/mentions, collapsed elongations, standardized laughter/crying tokens, and mapped colloquial Indonesian negations and slang.",
      "badge": "Cleaning"
    },
    {
      "step": "02",
      "title": "Strict Deduplication & Zero-Leakage Split",
      "description": "Removed 14,257 normalized duplicates and 11 empty records, then isolated 1,000 tweets (500 validation, 500 test) under random_state=42 before any lexicon scoring or seed sampling.",
      "badge": "Splitting"
    },
    {
      "step": "03",
      "title": "Curated InSet Weak Supervision & Silver Consensus",
      "description": "Blacklisted 59 function words from InSet TSVs, applied 3-token negation windows and intensifier scaling, extracted 3,000 balanced seeds (>=0.85 confidence), and annotated eval splits via dual-annotator consensus.",
      "badge": "Labeling"
    },
    {
      "step": "04",
      "title": "Dual Word & Character-WB TF-IDF Vectorization",
      "description": "Constructed a 30,000-dimensional sparse FeatureUnion combining word n-grams (1-2, 12k max features) and char_wb n-grams (3-5, 18k max features) with sublinear TF scaling.",
      "badge": "Vectorization"
    }
  ],
  "mlDataPreparationNote": "All 1,000 validation and test tweets were removed from the deduplicated training candidate pool prior to weak-seed extraction and unlabeled-pool sampling to guarantee zero train-test leakage.",
  "mlModelArchitectureAndTraining": {
    "algorithmName": "Dual TF-IDF + Multinomial Logistic Regression (Self-Trained)",
    "architectureSummary": "A Scikit-learn Pipeline uniting 12,000 word TF-IDF features and 18,000 char_wb TF-IDF features into a 30,000-dimensional sparse vector fed to an L2-regularized multinomial Logistic Regression classifier.",
    "configurationBadge": "LBFGS // Tau 0.85",
    "hyperparameters": [
      {
        "name": "word_tfidf",
        "value": "ngram (1,2), max 12,000",
        "description": "Word unigram and bigram TF-IDF with min_df=2 and sublinear_tf=True."
      },
      {
        "name": "char_tfidf",
        "value": "char_wb (3,5), max 18,000",
        "description": "Subword character n-gram TF-IDF with min_df=5 and sublinear_tf=True."
      },
      {
        "name": "LogisticRegression C / solver",
        "value": "C=1.0, lbfgs, max_iter=1000",
        "description": "L2-regularized multinomial logistic regression with random_state=42."
      },
      {
        "name": "confidence_threshold (tau)",
        "value": "0.85 (swept 0.85, 0.90, 0.95)",
        "description": "Minimum predicted probability required to accept an unlabeled tweet as a pseudo-label."
      },
      {
        "name": "max_iterations / class_cap",
        "value": "4 iters / 1,200 per class",
        "description": "Caps pseudo-label ingestion at 3,600 balanced records per iteration over a 150,000-tweet pool."
      }
    ],
    "objectiveCriterion": "Multinomial Cross-Entropy Loss with L2 Regularization",
    "optimizationSolver": "L-BFGS",
    "trainingProcess": "The baseline fits 3,000 balanced weak seeds. Iterative self-training then runs 4 rounds over 150,000 unlabeled tweets at tau in {0.85, 0.90, 0.95}; tau=0.85 acquires 14,400 balanced pseudo-labels (4,800 per class), yielding a final 17,400-sample retrained classifier."
  },
  "mlEvaluationStrategy": {
    "validationMethod": "Isolated 500-Validation Sweep & 500-Test Silver-Reference Benchmark",
    "description": "Confidence thresholds were tuned on the held-out 500-tweet validation split, followed by a single final evaluation on the untouched 500-tweet test set against Dummy Majority and Seed-Only Baseline models.",
    "metrics": [
      {
        "label": "SEED BASELINE ACC",
        "value": "42.20%",
        "benchmark": "Dummy: 4.80%"
      },
      {
        "label": "SEED MACRO F1",
        "value": "0.3856",
        "benchmark": "Dummy: 0.0305"
      },
      {
        "label": "SELF-TRAINED ACC",
        "value": "38.00%",
        "benchmark": "Baseline: 42.20%"
      },
      {
        "label": "SELF-TRAIN MACRO F1",
        "value": "0.3522",
        "benchmark": "Baseline: 0.3856"
      },
      {
        "label": "SELF-TRAIN WT F1",
        "value": "0.4109",
        "benchmark": "Baseline: 0.4583"
      },
      {
        "label": "NEGATIVE RECALL",
        "value": "91.67%",
        "benchmark": "Target > 85.00%"
      }
    ]
  },
  "mlPerformanceVisualizations": [
    {
      "title": "Interactive Tweet Classifier — Slang Normalization & Inference",
      "badge": "Streamlit UI",
      "imageUrl": "/images/projects/self_training_for_indonesian/image01.webp",
      "telemetryNote": "Live slang tweet classified as Negative at 77.2% confidence with preprocessing transformation diff."
    },
    {
      "title": "Semi-Supervised Methodology Pipeline Overview",
      "badge": "Streamlit UI",
      "imageUrl": "/images/projects/self_training_for_indonesian/image02.webp",
      "telemetryNote": "Documents leakage-free splitting, InSet weak supervision, and capped multi-threshold self-training."
    },
    {
      "title": "Publication Visualizations Dashboard Grid",
      "badge": "Streamlit UI",
      "imageUrl": "/images/projects/self_training_for_indonesian/image03.webp",
      "telemetryNote": "Consolidated 2x2 dashboard comparing model metrics, pseudo-label counts, confusion matrices, and confidence."
    },
    {
      "title": "Model Limitations & Linguistic Error Analysis View",
      "badge": "Streamlit UI",
      "imageUrl": "/images/projects/self_training_for_indonesian/image04.webp",
      "telemetryNote": "Summarizes weak-supervision bias, sarcasm inversion, regional code-switching, and implicit sentiment."
    },
    {
      "title": "Corpus Terms, Character Length, Provenance & Confidence Audit",
      "badge": "Diagnostics Grid",
      "imageUrl": "/images/projects/self_training_for_indonesian/image05.webp",
      "telemetryNote": "Top term tidak reaches 10,355 occurrences; seed provenance balances 1,000 per class against silver consensus."
    },
    {
      "title": "Confidence Threshold Optimization & Iteration Acquisition",
      "badge": "Training Dynamics",
      "imageUrl": "/images/projects/self_training_for_indonesian/image06.webp",
      "telemetryNote": "Tau 0.85 yields 14,400 balanced pseudo-labels while tau 0.95 drops to 6,085 and starves the neutral class."
    },
    {
      "title": "Dataset Quality Audit & Test Error Breakdown",
      "badge": "Error Taxonomy",
      "imageUrl": "/images/projects/self_training_for_indonesian/image07.webp",
      "telemetryNote": "310/500 test errors categorized: 64.2% Neutral False Polar over-triggering and 28.1% Negation ambiguity."
    },
    {
      "title": "Per-Class Test Metrics & Model Architecture Comparison",
      "badge": "Benchmark Chart",
      "imageUrl": "/images/projects/self_training_for_indonesian/image08.webp",
      "telemetryNote": "Self-trained model achieves 0.92 Negative recall and 0.92 Neutral precision; Seed-Only leads at 0.386 Macro F1."
    },
    {
      "title": "Test Set Confusion Matrices (Baseline vs. Self-Trained)",
      "badge": "Confusion Matrix",
      "imageUrl": "/images/projects/self_training_for_indonesian/image09.webp",
      "telemetryNote": "Highlights neutral-to-polar spillover growing from 269/394 in baseline to 286/394 after pseudo-labeling."
    }
  ],
  "mlVisualizationTelemetryNote": "Validation Macro F1 peaked at 0.3761 in Iteration 1 and drifted to 0.3680 by Iteration 4, closely mirroring the test Macro F1 drop from 0.3856 (Seed-Only) to 0.3522 (Self-Trained).",
  "mlModelInterpretationAndLimitations": {
    "interpretationSummary": "Linear TF-IDF weights strongly anchor on high-salience polar tokens and negation particles, yielding 91.67% negative recall and 92.31% neutral precision. However, iterative pseudo-labeling amplifies polar token weights, causing polar over-triggering on neutral tweets.",
    "keyDrivers": [
      {
        "factor": "Balanced Seed vs. Skewed Test Prior",
        "impact": "Training on 33.3% balanced classes against a 78.8% neutral test distribution causes 199 neutral false-polar errors (64.2%)."
      },
      {
        "factor": "Non-Sentiment Negation Particles",
        "impact": "Discourse markers like tidak cuma and jangan tanggal trigger 87 negation/ambiguity errors (28.1%)."
      },
      {
        "factor": "Confidence Calibration Across Thresholds",
        "impact": "Setting tau=0.95 admits only 195 neutral pseudo-labels vs. 980 positive in 4 iterations, whereas tau=0.85 preserves 4,800 per class."
      },
      {
        "factor": "Malay & Regional Code-Switching",
        "impact": "Dialectal phrases such as tkdelah meroyan induce out-of-vocabulary or inverted polarity predictions."
      }
    ],
    "limitations": [
      {
        "title": "Silver-Reference Ground Truth Constraints",
        "description": "Validation and test sets rely on InSet plus rule-heuristic consensus (Kappa 0.0937-0.0998) rather than human multi-annotator gold labels."
      },
      {
        "title": "Confirmation Bias & Semantic Drift",
        "description": "Assimilating 14,400 pseudo-labels without external supervision reinforced polar boundaries and reduced test accuracy from 42.20% to 38.00%."
      },
      {
        "title": "Bag-of-Words Syntactic Bounds",
        "description": "TF-IDF n-grams cannot reliably resolve long-range poetic negations, additive discourse particles, or ironic praise."
      }
    ],
    "futureImprovements": [
      {
        "title": "IndoBERT Contextual Fine-Tuning",
        "target": "Replace sparse TF-IDF with pretrained Indonesian transformer embeddings to resolve long-range negation and sarcasm."
      },
      {
        "title": "Human-in-the-Loop Active Learning",
        "target": "Route low-margin borderline samples (0.45-0.75 confidence) to human annotators to curb pseudo-label drift."
      },
      {
        "title": "Prior-Calibrated Thresholding",
        "target": "Adjust class-specific thresholds and prior weighting to reflect the natural 78.8% neutral social-media distribution."
      }
    ],
    "discoveryNote": "Auditing function-word weights in raw academic lexicons (blacklisting 59 neutral grammatical tokens like yang with -5 weight) proved essential before bootstrapping any weak-supervision pipeline."
  },
  "mlModelArtifacts": [
    {
      "name": "models/baseline_model.joblib",
      "format": "JOBLIB"
    },
    {
      "name": "models/final_model/model.joblib",
      "format": "JOBLIB"
    },
    {
      "name": "models/model_metadata.json",
      "format": "JSON"
    }
  ],
  "mlMethodologySpecs": {
    "learningParadigm": "Semi-Supervised Learning",
    "modelArchitecture": "TF-IDF Logistic Regression",
    "coreFramework": "Scikit-Learn",
    "validationMethod": "Holdout Silver Consensus Benchmark"
  }
}
```
