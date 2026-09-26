import os
import pandas as pd
import numpy as np

def compute_f05_macro(y_true_dict: dict[str, set[str]], y_pred_dict: dict[str, set[str]]) -> tuple[float, float, float]:
    """
    Computes macro-averaged F_0.5, Precision, and Recall across all Source 1 entities.
    y_true_dict: {s1_id: set_of_gt_matched_ids}
    y_pred_dict: {s1_id: set_of_predicted_matched_ids}
    """
    precisions = []
    recalls = []
    f05_scores = []
    
    beta_sq = 0.25
    multiplier = 1.25

    for s1_id, gt_set in y_true_dict.items():
        pred_set = y_pred_dict.get(s1_id, set())
        
        # Case 1: Ground truth is empty (Singleton)
        if len(gt_set) == 0:
            if len(pred_set) == 0:
                pr, re, f05 = 1.0, 1.0, 1.0
            else:
                pr, re, f05 = 0.0, 0.0, 0.0
        # Case 2: Ground truth has matches
        else:
            if len(pred_set) == 0:
                pr, re, f05 = 0.0, 0.0, 0.0
            else:
                tp = len(gt_set.intersection(pred_set))
                pr = tp / len(pred_set)
                re = tp / len(gt_set)
                
                if (beta_sq * pr + re) > 0:
                    f05 = (multiplier * pr * re) / (beta_sq * pr + re)
                else:
                    f05 = 0.0
                    
        precisions.append(pr)
        recalls.append(re)
        f05_scores.append(f05)
        
    return float(np.mean(f05_scores)), float(np.mean(precisions)), float(np.mean(recalls))

def load_ground_truth(gt_tsv_path: str) -> dict[str, set[str]]:
    """Load train ground truth TSV into a dict: {s1_id: set(matched_ids)}"""
    gt_df = pd.read_csv(gt_tsv_path, sep="\t")
    gt_df['matched_entity_ids'] = gt_df['matched_entity_ids'].fillna('')
    
    gt_dict = {}
    for row in gt_df.itertuples(index=False):
        s1_id = row.source1_entity_id
        m_str = row.matched_entity_ids
        if m_str and isinstance(m_str, str):
            matches = set(x.strip() for x in m_str.split(',') if x.strip())
        else:
            matches = set()
        gt_dict[s1_id] = matches
    return gt_dict

def save_submission_tsv(output_path: str, results_dict: dict[str, list[str] | set[str]], is_candidate: bool = False):
    """
    Saves matching_results.tsv or candidate_pairs.tsv in the exact required format.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    header_col = "candidate_entity_ids" if is_candidate else "matched_entity_ids"
    
    rows = []
    for s1_id, matches in results_dict.items():
        if isinstance(matches, (set, list)):
            m_str = ",".join(sorted(list(matches)))
        else:
            m_str = ""
        rows.append(f"{s1_id}\t{m_str}\n")
        
    with open(output_path, "w", encoding="utf-8") as f:
        col_name = "candidate_entity_ids" if is_candidate else "matched_entity_ids"
        f.write(f"source1_entity_id\t{col_name}\n")
        f.writelines(rows)
    print(f"Saved {output_path} with {len(rows):,} rows.")
