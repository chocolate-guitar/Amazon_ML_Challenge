import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from concurrent.futures import ThreadPoolExecutor
from src.preprocess import extract_street_number, extract_postal_code

def _chunk_fuzz_ratio(pairs_chunk):
    s1_names, target_names = pairs_chunk
    return [fuzz.ratio(a, b) for a, b in zip(s1_names, target_names)]

def _chunk_fuzz_token_set(pairs_chunk):
    s1_names, target_names = pairs_chunk
    return [fuzz.token_set_ratio(a, b) for a, b in zip(s1_names, target_names)]

def _chunk_fuzz_token_sort(pairs_chunk):
    s1_names, target_names = pairs_chunk
    return [fuzz.token_sort_ratio(a, b) for a, b in zip(s1_names, target_names)]

def _chunk_jaro_winkler(pairs_chunk):
    s1_names, target_names = pairs_chunk
    return [JaroWinkler.similarity(a, b) * 100.0 for a, b in zip(s1_names, target_names)]

def _chunk_name_heuristics(pairs_chunk):
    s1_names, target_names = pairs_chunk
    first_word_matches = []
    acronym_matches = []
    for sn, tn in zip(s1_names, target_names):
        s1_words = sn.split()
        t_words = tn.split()
        first_word_matches.append(1.0 if (s1_words and t_words and s1_words[0] == t_words[0]) else 0.0)

        s1_acr = "".join(w[0] for w in s1_words if len(w) >= 2)
        t_acr = "".join(w[0] for w in t_words if len(w) >= 2)
        is_acr = 0.0
        if s1_acr and len(s1_acr) >= 2 and (s1_acr in tn.replace(" ", "") or s1_acr in t_words):
            is_acr = 1.0
        elif t_acr and len(t_acr) >= 2 and (t_acr in sn.replace(" ", "") or t_acr in s1_words):
            is_acr = 1.0
        acronym_matches.append(is_acr)
    return first_word_matches, acronym_matches

def _chunk_addr_heuristics(pairs_chunk):
    s1_addrs, target_addrs = pairs_chunk
    st_matches = []
    both_have_nums = []
    zip_matches = []
    zip_mismatches = []
    for sa, ta in zip(s1_addrs, target_addrs):
        s_num = extract_street_number(sa)
        t_num = extract_street_number(ta)
        has_s_num = bool(s_num)
        has_t_num = bool(t_num)
        both_have_nums.append(1.0 if (has_s_num and has_t_num) else 0.0)
        st_matches.append(1.0 if (has_s_num and has_t_num and s_num == t_num) else 0.0)

        s_zip = extract_postal_code(sa)
        t_zip = extract_postal_code(ta)
        has_s_zip = bool(s_zip)
        has_t_zip = bool(t_zip)
        if has_s_zip and has_t_zip:
            if s_zip == t_zip:
                zip_matches.append(1.0)
                zip_mismatches.append(0.0)
            else:
                zip_matches.append(0.0)
                zip_mismatches.append(1.0)
        else:
            zip_matches.append(0.0)
            zip_mismatches.append(0.0)

    return st_matches, both_have_nums, zip_matches, zip_mismatches

def parallel_map(func, arr1, arr2, n_jobs: int = 10):
    """Run worker function in parallel across chunks."""
    n = len(arr1)
    if n == 0:
        return []
    chunk_size = int(np.ceil(n / n_jobs))
    chunks = [(arr1[i:i+chunk_size], arr2[i:i+chunk_size]) for i in range(0, n, chunk_size)]
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        results = list(executor.map(func, chunks))
    return results

def extract_pair_features(pairs_df: pd.DataFrame, n_jobs: int = 10) -> pd.DataFrame:
    """
    Parallel feature extraction using RapidFuzz and domain-specific heuristics.
    Combines high-speed C string algorithms with precision-oriented address & acronym checks.
    """
    features = {}

    s1_names = pairs_df['s1_clean_name'].fillna('').values
    target_names = pairs_df['target_clean_name'].fillna('').values
    s1_addrs = pairs_df['s1_clean_addr'].fillna('').values
    target_addrs = pairs_df['target_clean_addr'].fillna('').values
    target_ids = pairs_df['target_id'].values

    # 1. Parallel Name String Similarities
    ratio_res = parallel_map(_chunk_fuzz_ratio, s1_names, target_names, n_jobs)
    features['name_ratio'] = np.array([x for chunk in ratio_res for x in chunk], dtype=np.float32)

    token_set_res = parallel_map(_chunk_fuzz_token_set, s1_names, target_names, n_jobs)
    features['name_token_set_ratio'] = np.array([x for chunk in token_set_res for x in chunk], dtype=np.float32)

    token_sort_res = parallel_map(_chunk_fuzz_token_sort, s1_names, target_names, n_jobs)
    features['name_token_sort_ratio'] = np.array([x for chunk in token_sort_res for x in chunk], dtype=np.float32)

    jw_res = parallel_map(_chunk_jaro_winkler, s1_names, target_names, n_jobs)
    features['name_jaro_winkler'] = np.array([x for chunk in jw_res for x in chunk], dtype=np.float32)

    features['name_exact'] = (s1_names == target_names).astype(np.float32)
    features['name_len_diff'] = np.abs(np.array([len(a) for a in s1_names]) - np.array([len(b) for b in target_names])).astype(np.float32)

    # 2. Name Domain Heuristics (Acronym & First Word)
    name_heur_res = parallel_map(_chunk_name_heuristics, s1_names, target_names, n_jobs)
    first_words = [x for chunk in name_heur_res for x in chunk[0]]
    acronyms = [x for chunk in name_heur_res for x in chunk[1]]
    features['name_first_word_match'] = np.array(first_words, dtype=np.float32)
    features['name_acronym_match'] = np.array(acronyms, dtype=np.float32)

    # 3. Parallel Address String Similarities
    addr_ratio_res = parallel_map(_chunk_fuzz_ratio, s1_addrs, target_addrs, n_jobs)
    features['addr_ratio'] = np.array([x for chunk in addr_ratio_res for x in chunk], dtype=np.float32)

    addr_tok_res = parallel_map(_chunk_fuzz_token_set, s1_addrs, target_addrs, n_jobs)
    features['addr_token_set_ratio'] = np.array([x for chunk in addr_tok_res for x in chunk], dtype=np.float32)

    features['addr_exact'] = (s1_addrs == target_addrs).astype(np.float32)
    features['addr_is_null'] = ((s1_addrs == '') | (target_addrs == '')).astype(np.float32)

    # 4. Address Domain Heuristics (Street Number & Postal Code Match / Mismatch)
    addr_heur_res = parallel_map(_chunk_addr_heuristics, s1_addrs, target_addrs, n_jobs)
    st_matches = [x for chunk in addr_heur_res for x in chunk[0]]
    both_nums = [x for chunk in addr_heur_res for x in chunk[1]]
    zip_matches = [x for chunk in addr_heur_res for x in chunk[2]]
    zip_mismatches = [x for chunk in addr_heur_res for x in chunk[3]]

    features['addr_street_number_match'] = np.array(st_matches, dtype=np.float32)
    features['addr_both_have_number'] = np.array(both_nums, dtype=np.float32)
    features['addr_postal_code_match'] = np.array(zip_matches, dtype=np.float32)
    features['addr_postal_code_mismatch'] = np.array(zip_mismatches, dtype=np.float32)

    # 5. Source Indicator (S2 vs S3)
    features['is_source3'] = np.array([1 if str(tid).startswith('S3-') else 0 for tid in target_ids], dtype=np.float32)

    return pd.DataFrame(features)
