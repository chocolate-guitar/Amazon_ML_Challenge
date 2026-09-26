import re
import unicodedata
import polars as pl

# Common legal suffix normalization map
LEGAL_SUFFIXES = {
    r'\bcorporation\b': 'corp',
    r'\bincorporated\b': 'inc',
    r'\blimited\b': 'ltd',
    r'\bprivate\b': 'pvt',
    r'\bcompany\b': 'co',
    r'\bproprietorship\b': 'prop',
    r'\bsociete a responsabilite limitee\b': 'sarl',
    r'\bsociete par actions simplifiee\b': 'sas',
    r'\bsociete anonyme\b': 'sa',
    r'\bentreprise unipersonnelle a responsabilite limitee\b': 'eurl',
    r'\bllc\b': 'llc',
    r'\bllp\b': 'llp',
    r'\binc\b': 'inc',
    r'\bltd\b': 'ltd',
    r'\bpvt\b': 'pvt',
    r'\bcorp\b': 'corp',
}

# Address shortcuts normalization
ADDRESS_MAP = {
    r'\broad\b': 'rd',
    r'\bstreet\b': 'st',
    r'\bavenue\b': 'ave',
    r'\bboulevard\b': 'blvd',
    r'\bapartment\b': 'apt',
    r'\bsuite\b': 'ste',
    r'\bbuilding\b': 'bldg',
    r'\bfloor\b': 'fl',
    r'\bnumber\b': 'no',
    r'\bpost office box\b': 'pobox',
    r'\bpo box\b': 'pobox',
    r'\bnear\b': 'nr',
}

def remove_accents(input_str: str) -> str:
    """Normalize unicode diacritics."""
    if not isinstance(input_str, str):
        return ""
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])

def normalize_text(text: str, is_address: bool = False) -> str:
    """Fast clean text normalization."""
    if not isinstance(text, str) or not text.strip():
        return ""
    text = remove_accents(text.lower().strip())
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    mapping = ADDRESS_MAP if is_address else LEGAL_SUFFIXES
    for pat, repl in mapping.items():
        text = re.sub(pat, repl, text)
    return re.sub(r'\s+', ' ', text).strip()

def get_char_ngrams(text: str, n: int = 3) -> list[str]:
    """Generate character n-grams."""
    text = f" {text} "
    if len(text) < n:
        return [text]
    return [text[i:i+n] for i in range(len(text) - n + 1)]

def preprocess_polars(df_pl: pl.DataFrame, col_name: str, is_address: bool = False) -> pl.Series:
    """Parallel text normalization using Polars regex."""
    expr = pl.col(col_name).fill_null("").str.to_lowercase()
    expr = expr.str.replace_all(r'[^a-z0-9\s]', ' ')
    mapping = ADDRESS_MAP if is_address else LEGAL_SUFFIXES
    for pat, repl in mapping.items():
        expr = expr.str.replace_all(pat, repl)
    expr = expr.str.replace_all(r'\s+', ' ').str.strip_chars()
    return df_pl.select(expr).to_series()
