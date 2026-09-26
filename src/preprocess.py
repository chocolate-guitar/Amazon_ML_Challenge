import re
import unicodedata
import polars as pl

# Expanded Legal Suffix Normalization Map
LEGAL_SUFFIXES = {
    # Multi-token patterns (match first)
    r'\bpvt\.?\s*ltd\.?\b': 'pvt ltd',
    r'\bpte\.?\s*ltd\.?\b': 'pte ltd',
    r'\bsoci[eé]t[eé]\s+[aà]\s+responsabilit[eé]\s+limit[eé]e\b': 'sarl',
    r'\bsoci[eé]t[eé]\s+par\s+actions\s+simplifi[eé]e\b': 'sas',
    r'\bentreprise\s+unipersonnelle\s+[aà]\s+responsabilit[eé]\s+limit[eé]e\b': 'eurl',
    # Single-token legal suffixes
    r'\bcorporation\b': 'corp',
    r'\bincorporated\b': 'inc',
    r'\blimited\b': 'ltd',
    r'\bprivate\b': 'pvt',
    r'\bpvt\.?\b': 'pvt',
    r'\bcompany\b': 'co',
    r'\bproprietorship\b': 'prop',
    r'\bsociete anonyme\b': 'sa',
    r'\bsarl\b': 'sarl',
    r'\bsas\b': 'sas',
    r'\bsa\b': 'sa',
    r'\bsnc\b': 'snc',
    r'\bsrl\b': 'srl',
    r'\bspa\b': 'spa',
    r'\bllc\b': 'llc',
    r'\bllp\b': 'llp',
    r'\blp\b': 'lp',
    r'\binc\b': 'inc',
    r'\bltd\b': 'ltd',
    r'\bcorp\b': 'corp',
    r'\b&\b': 'and',
}

# Expanded Address Shortcuts (US, Indian, and French tokens)
ADDRESS_MAP = {
    # US / General
    r'\broad\b': 'rd',
    r'\bstreet\b': 'st',
    r'\bavenue\b': 'ave',
    r'\bboulevard\b': 'blvd',
    r'\bdrive\b': 'dr',
    r'\blane\b': 'ln',
    r'\bapartment\b': 'apt',
    r'\bsuite\b': 'ste',
    r'\bbuilding\b': 'bldg',
    r'\bfloor\b': 'fl',
    r'\bhighway\b': 'hwy',
    r'\bnumber\b': 'no',
    r'\bpost office box\b': 'pobox',
    r'\bpo box\b': 'pobox',
    # Indian address patterns
    r'\bnear\b': 'nr',
    r'\bopposite\b': 'opp',
    r'\bcolony\b': 'clny',
    r'\bnagar\b': 'ngr',
    r'\bsector\b': 'sec',
    r'\bphase\b': 'ph',
    r'\bindustrial\s+area\b': 'indl area',
    r'\bc\s*/\s*o\b': 'co',
    # French address tokens (test set)
    r'\bimpasse\b': 'imp',
    r'\bchemin\b': 'ch',
    r'\ball[eé]e\b': 'all',
    r'\barrondissement\b': 'arr',
    r'\bcedex\b': 'cdx',
    r'\bbp\b': 'bp',
    r'\blieudit\b': 'ld',
    r'\bhameau\b': 'ham',
}

NULL_LITERAL_REGEX = r'\bnull\b'

def remove_accents(input_str: str) -> str:
    """Normalize unicode diacritics (crucial for French names/addresses)."""
    if not isinstance(input_str, str):
        return ""
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])

def extract_street_number(addr: str) -> str:
    """Extract leading/numeric token from address."""
    m = re.search(r"\b(\d+)\b", addr)
    return m.group(1) if m else ""

def extract_postal_code(addr: str) -> str:
    """Extract 5 or 6 digit postal code (US ZIP, India PIN, France code)."""
    m = re.search(r"\b(\d{5,6})\b", addr)
    return m.group(1) if m else ""

def preprocess_polars(df_pl: pl.DataFrame, col_name: str, is_address: bool = False) -> pl.Series:
    """Parallel text normalization using Polars regex."""
    expr = pl.col(col_name).fill_null("").str.to_lowercase()
    if is_address:
        # Strip literal 'null' placeholders found in raw address feeds
        expr = expr.str.replace_all(NULL_LITERAL_REGEX, ' ')
    expr = expr.str.replace_all(r'[^a-z0-9\s&]', ' ')
    mapping = ADDRESS_MAP if is_address else LEGAL_SUFFIXES
    for pat, repl in mapping.items():
        expr = expr.str.replace_all(pat, repl)
    expr = expr.str.replace_all(r'\s+', ' ').str.strip_chars()
    return df_pl.select(expr).to_series()
