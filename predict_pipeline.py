import os
import time
import pandas as pd
import numpy as np
import polars as pl
from src.blocking import FastBlocker
from src.features import extract_pair_features
from src.model import MatchingModel
from src.utils import save_submission_tsv

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run Entity Resolution Inference & Generate Submission")
    default_test_dir = "/Users/vedant/Downloads/ML challenge/student_resource/dataset/test" if os.path.exists("/Users/vedant/Downloads/ML challenge/student_resource/dataset/test") else "dataset/test"
    default_model_path = os.environ.get("SM_MODEL_DIR", "models")
    if os.path.isdir(default_model_path):
        candidate_model = os.path.join(default_model_path, "matching_lgbm.joblib")
        if os.path.exists(candidate_model):
            default_model_path = candidate_model
        else:
            default_model_path = "models/matching_lgbm.joblib"
    else:
        default_model_path = "models/matching_lgbm.joblib"

    parser.add_argument("--test-dir", type=str, default=os.environ.get("SM_CHANNEL_TEST", default_test_dir), help="Path to test dataset directory")
    parser.add_argument("--model-path", type=str, default=default_model_path, help="Path to trained model joblib file")
    parser.add_argument("--output-dir", type=str, default=os.environ.get("SM_OUTPUT_DIR", "output"), help="Output directory for TSV submission files")
    return parser.parse_args()

def main():
    args = parse_args()
    print("=== STARTING INFERENCE & SUBMISSION GENERATION ===")
    print(f"Test directory: {args.test_dir}")
    print(f"Model path: {args.model_path}")
    print(f"Output directory: {args.output_dir}")
    t0 = time.time()

    # 1. Load Test Files
    print("Loading test datasets...")
    df_s1 = pd.read_csv(os.path.join(args.test_dir, "test_source1.tsv"), sep="\t")
    df_s2 = pd.read_csv(os.path.join(args.test_dir, "test_source2.tsv"), sep="\t")
    df_s3 = pd.read_csv(os.path.join(args.test_dir, "test_source3.tsv"), sep="\t")

    print(f"Test S1: {len(df_s1):,}, Test S2: {len(df_s2):,}, Test S3: {len(df_s3):,}")

    # 2. Run Candidate Generation (Blocking)
    print("\n--- STEP 1: TEST CANDIDATE GENERATION (BLOCKING) ---")
    blocker = FastBlocker(top_k_per_source=20)
    candidates = blocker.generate_candidates(df_s1, df_s2, df_s3)

    # 3. Save candidate_pairs.tsv
    cand_output_path = os.path.join(args.output_dir, "candidate_pairs.tsv")
    save_submission_tsv(cand_output_path, candidates, is_candidate=True)

    # 4. Load Trained Matching Model
    print(f"\n--- STEP 2: PAIRWISE FEATURE EXTRACTION & MATCHING ---")
    model = MatchingModel.load(args.model_path)
    print(f"Loaded trained model with threshold tau = {model.best_threshold:.2f}")

    # 5. Build Pair Dataframe for Model Scoring
    s1_name_map = dict(zip(df_s1['entity_id'], df_s1['clean_name']))
    s1_addr_map = dict(zip(df_s1['entity_id'], df_s1['clean_addr']))

    s23_df_concat = pd.concat([df_s2, df_s3], ignore_index=True)
    s23_name_map = dict(zip(s23_df_concat['entity_id'], s23_df_concat['clean_name']))
    s23_addr_map = dict(zip(s23_df_concat['entity_id'], s23_df_concat['clean_addr']))

    pairs_list = []
    for s1_id, c_set in candidates.items():
        for target_id in c_set:
            pairs_list.append({
                's1_id': s1_id,
                'target_id': target_id,
                's1_clean_name': s1_name_map.get(s1_id, ''),
                's1_clean_addr': s1_addr_map.get(s1_id, ''),
                'target_clean_name': s23_name_map.get(target_id, ''),
                'target_clean_addr': s23_addr_map.get(target_id, ''),
            })

    pairs_df = pd.DataFrame(pairs_list)
    print(f"Generated {len(pairs_df):,} test candidate pairs for matching model evaluation.")

    if len(pairs_df) > 0:
        # Extract features batch-by-batch if large
        batch_size = 500000
        n_pairs = len(pairs_df)
        all_probs = []

        for start in range(0, n_pairs, batch_size):
            end = min(start + batch_size, n_pairs)
            batch_df = pairs_df.iloc[start:end]
            X_batch = extract_pair_features(batch_df)
            batch_probs = model.predict_proba(X_batch)
            all_probs.append(batch_probs)

        pairs_df['prob'] = np.concatenate(all_probs)
        
        # Filter predictions using optimal threshold tau
        df_matched = pairs_df[pairs_df['prob'] >= model.best_threshold]
        if len(df_matched) > 0:
            pl_matched = pl.from_pandas(df_matched[['s1_id', 'target_id']])
            grouped = pl_matched.group_by('s1_id').agg(pl.col('target_id'))
            matching_results = dict(zip(grouped['s1_id'].to_list(), grouped['target_id'].to_list()))
        else:
            matching_results = {}
    else:
        matching_results = {}

    # Ensure every test S1 entity is present in matching_results
    for s1_id in df_s1['entity_id'].values:
        if s1_id not in matching_results:
            matching_results[s1_id] = []

    # 6. Save matching_results.tsv
    matching_output_path = os.path.join(args.output_dir, "matching_results.tsv")
    save_submission_tsv(matching_output_path, matching_results, is_candidate=False)

    print(f"\nPrediction pipeline finished in {time.time()-t0:.2f} seconds.")

if __name__ == "__main__":
    main()
