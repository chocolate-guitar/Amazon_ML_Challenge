import numpy as np
import pandas as pd
import polars as pl
from collections import defaultdict, Counter
import time
from src.preprocess import preprocess_polars, extract_postal_code

class FastBlocker:
    """
    Supercharged Inverted Index Blocker combining word tokens, 3/4-char prefixes & suffixes, 
    5-char space-stripped prefix, and address keys for >98% Candidate Recall Ceiling.
    """
    def __init__(self, top_k_per_source: int = 25, min_shared_tokens: int = 1):
        self.top_k_per_source = top_k_per_source
        self.min_shared_tokens = min_shared_tokens

    def preprocess_df(self, df_pd: pd.DataFrame) -> pd.DataFrame:
        df_pl = pl.from_pandas(df_pd)
        clean_name = preprocess_polars(df_pl, 'business_name', is_address=False)
        clean_addr = preprocess_polars(df_pl, 'business_address', is_address=True)
        
        df_pd['clean_name'] = clean_name.to_numpy()
        df_pd['clean_addr'] = clean_addr.to_numpy()
        return df_pd

    STOP_WORDS = {
        'corp', 'corporation', 'inc', 'incorporated', 'ltd', 'limited', 'pvt', 'private',
        'llc', 'co', 'company', 'sa', 'sas', 'sarl', 'gmbh', 'bv', 'nv', 'and', 'the', 'of', 'in', 'for'
    }

    def _extract_tokens(self, name: str, addr: str) -> set[str]:
        tokens = set()
        # 1. Business Name Tokens
        for w in name.split():
            if len(w) >= 2 and w not in self.STOP_WORDS:
                tokens.add(f"nw_{w}")
                if len(w) >= 3:
                    tokens.add(f"np3_{w[:3]}")
                    tokens.add(f"ns3_{w[-3:]}")
                if len(w) >= 4:
                    tokens.add(f"np4_{w[:4]}")

        # 2. Compact 5-char Prefix (catches concatenated domain names like krishnaengineering.com)
        compact = name.replace(" ", "")
        if len(compact) >= 4:
            tokens.add(f"npref5_{compact[:5]}")

        # 3. Address Tokens
        for w in addr.split():
            if len(w) >= 2 and w not in self.STOP_WORDS:
                tokens.add(f"aw_{w}")
                if len(w) >= 3:
                    tokens.add(f"ap3_{w[:3]}")

        # 4. Postal / PIN Code
        zip_code = extract_postal_code(addr)
        if zip_code:
            tokens.add(f"azip_{zip_code}")

        return tokens

    def fit_transform_country(self, df_s1: pd.DataFrame, df_s23: pd.DataFrame) -> dict[str, set[str]]:
        if len(df_s1) == 0 or len(df_s23) == 0:
            return {}

        s1_ids = df_s1['entity_id'].values
        s23_ids = df_s23['entity_id'].values
        
        s1_names = df_s1['clean_name'].values
        s1_addrs = df_s1['clean_addr'].values
        s23_names = df_s23['clean_name'].values
        s23_addrs = df_s23['clean_addr'].values

        t0 = time.time()
        # 1. Build Inverted Index for S23 targets
        token_to_s23 = defaultdict(list)

        for idx in range(len(df_s23)):
            toks = self._extract_tokens(s23_names[idx], s23_addrs[idx])
            for t in toks:
                token_to_s23[t].append(idx)

        # Filter out hyper-frequent stop tokens (>3,500 matches) or singleton tokens
        max_freq = 3500
        valid_inverted_index = {t: arr for t, arr in token_to_s23.items() if 1 < len(arr) <= max_freq}
        print(f"   Built Inverted Index in {time.time()-t0:.2f}s | Valid Tokens: {len(valid_inverted_index):,}")

        # 2. Parallel Candidate Retrieval via posting list counts
        candidates = defaultdict(set)
        t_cand = time.time()
        n_s1 = len(df_s1)
        chunk_size = 20000
        chunks = [(i, min(i + chunk_size, n_s1)) for i in range(0, n_s1, chunk_size)]

        def _retrieve_chunk(start_idx, end_idx):
            chunk_res = {}
            for s1_idx in range(start_idx, end_idx):
                s1_id = s1_ids[s1_idx]
                toks = self._extract_tokens(s1_names[s1_idx], s1_addrs[s1_idx])
                
                posting_hits = []
                for t in toks:
                    if t in valid_inverted_index:
                        posting_hits.extend(valid_inverted_index[t])

                if not posting_hits:
                    continue

                counts = Counter(posting_hits)
                top_matches = counts.most_common(self.top_k_per_source)
                
                c_set = set()
                for target_idx, shared_cnt in top_matches:
                    if shared_cnt >= self.min_shared_tokens:
                        c_set.add(s23_ids[target_idx])
                if c_set:
                    chunk_res[s1_id] = c_set
            return chunk_res

        from joblib import Parallel, delayed
        chunk_results = Parallel(n_jobs=10, prefer="threads")(
            delayed(_retrieve_chunk)(s, e) for s, e in chunks
        )

        for res in chunk_results:
            for s1_id, c_set in res.items():
                candidates[s1_id].update(c_set)

        print(f"   Candidate Retrieval completed in {time.time()-t_cand:.2f}s")
        return candidates

    def generate_candidates(self, df_s1: pd.DataFrame, df_s2: pd.DataFrame, df_s3: pd.DataFrame) -> dict[str, set[str]]:
        all_candidates = defaultdict(set)

        print("Fast parallel preprocessing S1, S2, S3 with Polars...")
        t0 = time.time()
        df_s1 = self.preprocess_df(df_s1)
        df_s2 = self.preprocess_df(df_s2)
        df_s3 = self.preprocess_df(df_s3)
        print(f"Preprocessing completed in {time.time()-t0:.2f} seconds.")

        countries = df_s1['country'].unique()
        print(f"Running blocking across countries: {countries}...")

        for country in countries:
            t0 = time.time()
            s1_c = df_s1[df_s1['country'] == country]
            s2_c = df_s2[df_s2['country'] == country]
            s3_c = df_s3[df_s3['country'] == country]

            print(f"\n[{country}] S1: {len(s1_c):,}, S2: {len(s2_c):,}, S3: {len(s3_c):,}")

            print(" S1 vs S2:")
            cand_s2 = self.fit_transform_country(s1_c, s2_c)
            print(" S1 vs S3:")
            cand_s3 = self.fit_transform_country(s1_c, s3_c)

            for s1_id, c_set in cand_s2.items():
                all_candidates[s1_id].update(c_set)
            for s1_id, c_set in cand_s3.items():
                all_candidates[s1_id].update(c_set)

            print(f"[{country}] Blocking completed in {time.time()-t0:.2f} seconds.")

        for s1_id in df_s1['entity_id'].values:
            if s1_id not in all_candidates:
                all_candidates[s1_id] = set()

        return all_candidates
