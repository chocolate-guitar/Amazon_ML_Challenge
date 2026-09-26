import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from concurrent.futures import ThreadPoolExecutor
import os

def _chunk_fuzz_ratio(pairs_chunk):
    s1_names, target_names = pairs_chunk
    return [fuzz.ratio(a, b) for a, b in zip(s1_names, target_names)]

def _chunk_fuzz_token_set(pairs_chunk):
    s1_names, target_names = pairs_chunk
    return [fuzz.token_set_ratio(a, b) for a, b in zip(s1_names, target_names)]

def _chunk_fuzz_token_sort(pairs_chunk):
    s1_names, target_names = pairs_chunk
    return [fuzz.token_sort_ratio(a, b) for a, b in zip(s1_names, target_names)]

def parallel_fuzz(func, arr1, arr2, n_jobs: int = 10) -> np.ndarray:
    """Run RapidFuzz string metric in parallel across CPU cores."""
    n = len(arr1)
    if n == 0:
        return np.array([], dtype=np.float32)

    chunk_size = int(np.ceil(n / n_jobs))
    chunks = [(arr1[i:i+chunk_size], arr2[i:i+chunk_size]) for i in range(0, n, chunk_size)]
    
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        results = list(executor.map(func, chunks))
        
    flat_res = [item for sublist in results for item in sublist]
    return np.array(flat_res, dtype=np.float32)

def extract_pair_features(pairs_df: pd.DataFrame, n_jobs: int = 10) -> pd.DataFrame:
    """
    Parallel feature extraction using RapidFuzz over 10 CPU cores.
    Extremely fast calculation across millions of pairs.
    """
    features = {}

    s1_names = pairs_df['s1_clean_name'].fillna('').values
    target_names = pairs_df['target_clean_name'].fillna('').values
    s1_addrs = pairs_df['s1_clean_addr'].fillna('').values
    target_addrs = pairs_df['target_clean_addr'].fillna('').values
    target_ids = pairs_df['target_id'].values

    # 1. Parallel Name Features
    features['name_ratio'] = parallel_fuzz(_chunk_fuzz_ratio, s1_names, target_names, n_jobs)
    features['name_token_set_ratio'] = parallel_fuzz(_chunk_fuzz_token_set, s1_names, target_names, n_jobs)
    features['name_token_sort_ratio'] = parallel_fuzz(_chunk_fuzz_token_sort, s1_names, target_names, n_jobs)
    features['name_exact'] = (s1_names == target_names).astype(np.float32)
    features['name_len_diff'] = np.abs(np.array([len(a) for a in s1_names]) - np.array([len(b) for b in target_names])).astype(np.float32)

    # 2. Parallel Address Features
    features['addr_ratio'] = parallel_fuzz(_chunk_fuzz_ratio, s1_addrs, target_addrs, n_jobs)
    features['addr_token_set_ratio'] = parallel_fuzz(_chunk_fuzz_token_set, s1_addrs, target_addrs, n_jobs)
    features['addr_exact'] = (s1_addrs == target_addrs).astype(np.float32)
    features['addr_is_null'] = ((s1_addrs == '') | (target_addrs == '')).astype(np.float32)

    # 3. Source Indicator (S2 vs S3)
    features['is_source3'] = np.array([1 if str(tid).startswith('S3-') else 0 for tid in target_ids], dtype=np.float32)

    return pd.DataFrame(features)
