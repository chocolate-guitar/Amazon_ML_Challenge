# Amazon ML Challenge - Business Entity Resolution

This repository contains the end-to-end pipeline for the Amazon ML Challenge: Business Entity Resolution. It is optimized for execution on local machines and **Amazon SageMaker**.

## Project Structure

```text
├── src/
│   ├── blocking.py       # Inverted-index candidate generation (FastBlocker)
│   ├── features.py       # Parallel RapidFuzz similarity feature extraction
│   ├── model.py          # HistGradientBoostingClassifier & F0.5 threshold optimizer
│   ├── preprocess.py     # Text cleaning, French/US legal suffixes & address normalizers
│   └── utils.py          # Macro F0.5 evaluator & submission TSV exporter
├── train_sagemaker.py    # Training script supporting SageMaker env vars & CLI flags
├── predict_pipeline.py   # Inference script to generate matching_results & candidate_pairs
├── requirements.txt      # Pinned Python dependencies
└── README.md
```

## Quick Start on Amazon SageMaker / Local

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Model
```bash
# Basic run:
python train_sagemaker.py --train <path_to_train_dir> --model-dir models

# Scale up with custom parameters:
python train_sagemaker.py --train <path_to_train_dir> --model-dir models --n-sample 300000 --max-iter 400
```

### 3. Generate Submission Predictions
```bash
python predict_pipeline.py --test-dir <path_to_test_dir> --model-path models/matching_lgbm.joblib --output-dir output
```
This produces:
- `output/candidate_pairs.tsv`
- `output/matching_results.tsv`
