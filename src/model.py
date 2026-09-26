import numpy as np
import pandas as pd
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier
import joblib
import os
from src.utils import compute_f05_macro

class MatchingModel:
    """
    High-speed, robust Pairwise Entity Matching Model using 
    HistGradientBoostingClassifier (native sklearn histogram GBDT).
    """
    def __init__(self, max_iter: int = 250, learning_rate: float = 0.05, max_depth: int = 8):
        self.clf = HistGradientBoostingClassifier(
            max_iter=max_iter,
            learning_rate=learning_rate,
            max_depth=max_depth,
            max_leaf_nodes=63,
            random_state=42,
            early_stopping=False
        )
        self.best_threshold = 0.5

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        """Train HistGradientBoostingClassifier on candidate pair features."""
        print(f"Training HistGradientBoostingClassifier on {len(X):,} candidate pairs (Positives: {y.sum():,})...", flush=True)
        self.clf.fit(X, y)
        print("Model training completed successfully.", flush=True)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probability of pair being a true match."""
        return self.clf.predict_proba(X)[:, 1]

    def optimize_threshold(self, df_pairs: pd.DataFrame, y_prob: np.ndarray, y_true_gt: dict[str, set[str]]) -> float:
        """Find decision threshold tau that maximizes macro F0.5 score using Polars."""
        print("Optimizing threshold for macro F0.5 with Polars...", flush=True)
        
        df_eval = pd.DataFrame({
            's1_id': df_pairs['s1_id'].values,
            'target_id': df_pairs['target_id'].values,
            'prob': y_prob
        })
        df_pl = pl.from_pandas(df_eval)

        best_score = -1.0
        best_tau = 0.5
        best_p, best_r = 0.0, 0.0

        for tau in np.arange(0.35, 0.90, 0.05):
            matched_pl = df_pl.filter(pl.col('prob') >= tau)
            if len(matched_pl) > 0:
                grouped = matched_pl.group_by('s1_id').agg(pl.col('target_id'))
                pred_dict = dict(zip(grouped['s1_id'].to_list(), [set(x) for x in grouped['target_id'].to_list()]))
            else:
                pred_dict = {}

            # Add singletons
            for s1_id in y_true_gt.keys():
                if s1_id not in pred_dict:
                    pred_dict[s1_id] = set()

            f05, p, r = compute_f05_macro(y_true_gt, pred_dict)
            print(f"Threshold tau = {tau:.2f} | F0.5: {f05:.4f} | Precision: {p:.4f} | Recall: {r:.4f}", flush=True)

            if f05 > best_score:
                best_score = f05
                best_tau = tau
                best_p, best_r = p, r

        self.best_threshold = float(best_tau)
        print(f"\nOptimal Threshold: {self.best_threshold:.2f} with Validation F0.5: {best_score:.4f} (Precision: {best_p:.4f}, Recall: {best_r:.4f})", flush=True)
        return self.best_threshold

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump((self.clf, self.best_threshold), path)

    @classmethod
    def load(cls, path: str):
        clf, threshold = joblib.load(path)
        model = cls()
        model.clf = clf
        model.best_threshold = threshold
        return model
