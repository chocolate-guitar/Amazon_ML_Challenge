import os
import sys
import time
import argparse
import pandas as pd
import numpy as np
import polars as pl
from src.blocking import FastBlocker
from src.features import extract_pair_features
from src.model import MatchingModel
from src.utils import load_ground_truth, compute_f05_macro

def parse_args():
    parser = argparse.ArgumentParser(description="Train Business Entity Resolution Model on Amazon SageMaker")

    # SageMaker environment variables or command-line args
    parser.add_argument(
        "--train",
        type=str,
        default=os.environ.get("SM_CHANNEL_TRAIN", "/Users/vedant/Downloads/ML challenge/student_resource/dataset/train"),
        help="Path to training data directory containing train_source*.tsv and train_ground_truth.tsv"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=os.environ.get("SM_MODEL_DIR", "models"),
        help="Path to output model directory (SageMaker archives everything here into model.tar.gz)"
    )
    parser.add_argument(
        "--n-sample",
        type=int,
        default=250000,
        help="Number of S1 samples to train on (e.g. 250000 or -1 for full dataset)"
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=400,
        help="Number of iterations for HistGradientBoostingClassifier"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.05,
        help="Learning rate"
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=8,
        help="Max depth for trees"
    )

    return parser.parse_args()

def main():
    args = parse_args()
    print("=== STARTING SAGEMAKER TRAINING PIPELINE ===")
    print(f"Data directory: {args.train}")
    print(f"Model output directory: {args.model_dir}")
    print(f"Training parameters: n_sample={args.n_sample}, max_iter={args.max_iter}, lr={args.learning_rate}, max_depth={args.max_depth}")
    
    t0 = time.time()

    # 1. Load Train Ground Truth
    gt_path = os.path.join(args.train, "train_ground_truth.tsv")
    print(f"Loading Ground Truth from {gt_path}...")
    gt_dict = load_ground_truth(gt_path)

    # 2. Sample S1 for training
    s1_path = os.path.join(args.train, "train_source1.tsv")
    print(f"Loading Source 1 from {s1_path}...")
    df_s1_full = pd.read_csv(s1_path, sep="\t")
    
    if args.n_sample > 0 and args.n_sample < len(df_s1_full):
        df_s1 = df_s1_full.sample(n=args.n_sample, random_state=42).reset_index(drop=True)
    else:
        df_s1 = df_s1_full.reset_index(drop=True)
        
    sample_s1_ids = set(df_s1['entity_id'].values)
    gt_sample = {k: v for k, v in gt_dict.items() if k in sample_s1_ids}
    print(f"Using {len(df_s1):,} S1 entities for training.")

    # 3. Load S2 and S3
    s2_path = os.path.join(args.train, "train_source2.tsv")
    s3_path = os.path.join(args.train, "train_source3.tsv")
    print(f"Loading Source 2 ({s2_path}) and Source 3 ({s3_path})...")
    df_s2 = pd.read_csv(s2_path, sep="\t")
    df_s3 = pd.read_csv(s3_path, sep="\t")

    # 4. Generate Candidate Pairs via FastBlocker
    print("\n--- STEP 1: CANDIDATE GENERATION (BLOCKING) ---")
    blocker = FastBlocker(top_k_per_source=20)
    candidates = blocker.generate_candidates(df_s1, df_s2, df_s3)

    # 5. Build Pair Dataframe with Labels
    print("\n--- STEP 2: CREATING PAIR DATASET & LABELS ---")
    s1_name_map = dict(zip(df_s1['entity_id'], df_s1['clean_name']))
    s1_addr_map = dict(zip(df_s1['entity_id'], df_s1['clean_addr']))

    s23_df_concat = pd.concat([df_s2, df_s3], ignore_index=True)
    s23_name_map = dict(zip(s23_df_concat['entity_id'], s23_df_concat['clean_name']))
    s23_addr_map = dict(zip(s23_df_concat['entity_id'], s23_df_concat['clean_addr']))

    pairs_list = []
    for s1_id, c_set in candidates.items():
        gt_targets = gt_sample.get(s1_id, set())
        for target_id in c_set:
            is_match = 1 if target_id in gt_targets else 0
            pairs_list.append({
                's1_id': s1_id,
                'target_id': target_id,
                's1_clean_name': s1_name_map.get(s1_id, ''),
                's1_clean_addr': s1_addr_map.get(s1_id, ''),
                'target_clean_name': s23_name_map.get(target_id, ''),
                'target_clean_addr': s23_addr_map.get(target_id, ''),
                'label': is_match
            })

    pairs_df = pd.DataFrame(pairs_list)
    print(f"Generated {len(pairs_df):,} candidate pairs for training.")
    print(f"Positive pairs: {pairs_df['label'].sum():,}, Negative pairs: {(len(pairs_df) - pairs_df['label'].sum()):,}")

    # 6. Feature Extraction
    print("\n--- STEP 3: FEATURE EXTRACTION ---")
    X_features = extract_pair_features(pairs_df)
    y_labels = pairs_df['label'].values

    # 7. Train / Validation Split
    print("\n--- STEP 4: MODEL TRAINING & THRESHOLD TUNING ---")
    split_idx = int(len(pairs_df) * 0.7)
    
    X_train, y_train = X_features.iloc[:split_idx], y_labels[:split_idx]
    X_val, y_val = X_features.iloc[split_idx:], y_labels[split_idx:]
    df_val = pairs_df.iloc[split_idx:].reset_index(drop=True)

    model = MatchingModel(max_iter=args.max_iter, learning_rate=args.learning_rate, max_depth=args.max_depth)
    model.fit(X_train, y_train)

    # 8. Optimize Threshold on Validation Set
    val_probs = model.predict_proba(X_val)
    val_s1_set = set(df_val['s1_id'].unique())
    val_gt = {k: v for k, v in gt_sample.items() if k in val_s1_set}
    best_tau = model.optimize_threshold(df_val, val_probs, val_gt)

    # 9. Save Trained Model to SageMaker Model Directory
    os.makedirs(args.model_dir, exist_ok=True)
    save_path = os.path.join(args.model_dir, "matching_lgbm.joblib")
    model.save(save_path)
    print(f"\nModel successfully saved to {save_path}")
    print(f"Total training pipeline time: {time.time()-t0:.2f} seconds")

if __name__ == "__main__":
    main()
