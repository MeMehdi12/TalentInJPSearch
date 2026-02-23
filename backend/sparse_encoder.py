"""
Sparse Encoder - Encode search queries to BM25 sparse vectors
==============================================================
Loads vocabulary and IDF from the unified setup script.
Encodes query keywords into sparse vectors for hybrid search.
"""

import json
import re
import math
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import Counter

# Paths - uses the cache from 21_setup_qdrant_hybrid.py
BASE_DIR = Path(__file__).parent.parent
SPARSE_CACHE_DIR = BASE_DIR / 'Database' / 'sparse_cache'
VOCAB_PATH = SPARSE_CACHE_DIR / 'vocabulary.json'
IDF_PATH = SPARSE_CACHE_DIR / 'idf.json'
STATS_PATH = SPARSE_CACHE_DIR / 'stats.json'

# BM25 parameters - loaded from stats, with defaults
BM25_K1 = 1.5
BM25_B = 0.75
AVG_DOC_LEN = 50.0  # Will be updated from stats.json if available

# Japanese-aware tokenization pattern
TOKEN_PATTERN = re.compile(r'[a-zA-Z0-9]+|[\u3040-\u309F]+|[\u30A0-\u30FF]+|[\u4E00-\u9FFF]+')

# Stopwords
STOPWORDS = {'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
             'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
             'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
             'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
             'it', 'its', 'this', 'that', 'these', 'those', 'i', 'me', 'my',
             'we', 'our', 'you', 'your', 'he', 'she', 'they', 'them', 'their'}


class SparseEncoder:
    """
    Encodes text queries into BM25 sparse vectors.
    
    Usage:
        encoder = SparseEncoder()
        indices, values = encoder.encode(["python", "google", "san francisco"])
    """
    
    def __init__(self):
        self._vocab: Optional[Dict[str, int]] = None
        self._idf: Optional[Dict[str, float]] = None
        self._avg_doc_len: float = AVG_DOC_LEN
        self._loaded = False
    
    def _ensure_loaded(self):
        """Lazy load vocabulary and IDF"""
        if self._loaded:
            return
        
        if not VOCAB_PATH.exists():
            raise FileNotFoundError(
                f"Vocabulary not found: {VOCAB_PATH}\n"
                "Run Scripts/21_setup_qdrant_hybrid.py first to generate BM25 data."
            )
        if not IDF_PATH.exists():
            raise FileNotFoundError(
                f"IDF not found: {IDF_PATH}\n"
                "Run Scripts/21_setup_qdrant_hybrid.py first to generate BM25 data."
            )
        
        with open(VOCAB_PATH, 'r', encoding='utf-8') as f:
            self._vocab = json.load(f)
        
        with open(IDF_PATH, 'r', encoding='utf-8') as f:
            # IDF stored as {index_str: value}
            raw_idf = json.load(f)
            # Convert to {token: value} using vocab
            idx_to_token = {str(v): k for k, v in self._vocab.items()}
            self._idf = {}
            for idx_str, val in raw_idf.items():
                if idx_str in idx_to_token:
                    self._idf[idx_to_token[idx_str]] = val
        
        # Load stats if available
        if STATS_PATH.exists():
            with open(STATS_PATH, 'r', encoding='utf-8') as f:
                stats = json.load(f)
                self._avg_doc_len = stats.get('avg_doc_len', AVG_DOC_LEN)
        
        self._loaded = True
    
    @property
    def vocab(self) -> Dict[str, int]:
        self._ensure_loaded()
        return self._vocab
    
    @property
    def idf(self) -> Dict[str, float]:
        self._ensure_loaded()
        return self._idf
    
    @property
    def vocab_size(self) -> int:
        return len(self.vocab)
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text (handles both English and Japanese)"""
        if not text or not isinstance(text, str):
            return []
        
        text = text.lower()
        tokens = TOKEN_PATTERN.findall(text)
        
        # Filter: remove stopwords, single chars, pure digits
        filtered = []
        for t in tokens:
            if len(t) < 2:
                continue
            if t in STOPWORDS:
                continue
            if t.isdigit():
                continue
            filtered.append(t)
        
        return filtered
    
    def encode_tokens(self, tokens: List[str], boost: float = 1.0) -> Tuple[List[int], List[float]]:
        """
        Encode a list of tokens into sparse vector.
        
        Args:
            tokens: List of keywords to encode
            boost: Multiplier for weights (for boosting certain terms)
        
        Returns:
            indices: Vocabulary indices
            values: BM25-style weights
        """
        if not tokens:
            return [], []
        
        # Count term frequencies in query
        tf = Counter(tokens)
        doc_len = len(tokens)
        
        indices = []
        values = []
        
        for term, freq in tf.items():
            if term not in self.vocab:
                continue
            
            idx = self.vocab[term]
            idf_value = self.idf.get(term, 0)
            
            # BM25 formula (simplified for queries)
            numerator = freq * (BM25_K1 + 1)
            denominator = freq + BM25_K1 * (1 - BM25_B + BM25_B * doc_len / self._avg_doc_len)
            tf_component = numerator / denominator
            
            weight = idf_value * tf_component * boost
            
            if weight > 0:
                indices.append(idx)
                values.append(float(weight))
        
        return indices, values
    
    def encode(self, keywords: List[str], boost: float = 1.0) -> Tuple[List[int], List[float]]:
        """
        Encode a list of keywords into sparse vector.
        Keywords can be phrases like "san francisco" - they get tokenized.
        
        Args:
            keywords: List of keywords/phrases
            boost: Weight multiplier
        
        Returns:
            indices, values tuple for SparseVector
        """
        # Tokenize all keywords
        all_tokens = []
        for kw in keywords:
            tokens = self.tokenize(kw)
            all_tokens.extend(tokens)
        
        return self.encode_tokens(all_tokens, boost)
    
    def encode_text(self, text: str, boost: float = 1.0) -> Tuple[List[int], List[float]]:
        """
        Encode free-form text into sparse vector.
        
        Args:
            text: Free-form text to encode
            boost: Weight multiplier
        
        Returns:
            indices, values tuple for SparseVector
        """
        tokens = self.tokenize(text)
        return self.encode_tokens(tokens, boost)
    
    def encode_with_expansion(
        self, 
        keywords: List[str],
        skill_relations: Optional[Dict[str, List[Tuple[str, float]]]] = None,
        boost: float = 1.0
    ) -> Tuple[List[int], List[float]]:
        """
        Encode keywords with skill expansion.
        "python" expands to include "django", "flask", etc.
        
        Args:
            keywords: Original keywords
            skill_relations: Mapping of skill -> [(related_skill, confidence), ...]
            boost: Base weight multiplier
        
        Returns:
            indices, values tuple for SparseVector
        """
        all_tokens = []
        token_boosts = {}  # Track boost per token
        
        for kw in keywords:
            tokens = self.tokenize(kw)
            for token in tokens:
                all_tokens.append(token)
                token_boosts[token] = boost  # Original keyword gets full boost
                
                # Expand if we have relations
                if skill_relations and token in skill_relations:
                    for related, confidence in skill_relations[token][:5]:  # Top 5 related
                        related_lower = related.lower()
                        if related_lower not in token_boosts:
                            all_tokens.append(related_lower)
                            # Related skills get reduced boost based on confidence
                            token_boosts[related_lower] = boost * confidence * 0.5
        
        # Encode with individual boosts
        if not all_tokens:
            return [], []
        
        tf = Counter(all_tokens)
        doc_len = len(all_tokens)
        
        indices = []
        values = []
        
        for term, freq in tf.items():
            if term not in self.vocab:
                continue
            
            idx = self.vocab[term]
            idf_value = self.idf.get(term, 0)
            term_boost = token_boosts.get(term, boost)
            
            numerator = freq * (BM25_K1 + 1)
            denominator = freq + BM25_K1 * (1 - BM25_B + BM25_B * doc_len / self._avg_doc_len)
            tf_component = numerator / denominator
            
            weight = idf_value * tf_component * term_boost
            
            if weight > 0:
                indices.append(idx)
                values.append(float(weight))
        
        return indices, values
    
    def get_top_terms(self, indices: List[int], values: List[float], n: int = 10) -> List[Tuple[str, float]]:
        """
        Debug helper: get top N terms from a sparse vector.
        
        Returns:
            List of (term, weight) tuples
        """
        idx_to_word = {v: k for k, v in self.vocab.items()}
        
        term_weights = []
        for idx, val in zip(indices, values):
            if idx in idx_to_word:
                term_weights.append((idx_to_word[idx], val))
        
        return sorted(term_weights, key=lambda x: -x[1])[:n]


# Global instance for reuse
_encoder: Optional[SparseEncoder] = None


def get_sparse_encoder() -> SparseEncoder:
    """Get singleton sparse encoder instance"""
    global _encoder
    if _encoder is None:
        _encoder = SparseEncoder()
    return _encoder


if __name__ == '__main__':
    # Test the encoder
    encoder = get_sparse_encoder()
    
    print(f"Vocabulary size: {encoder.vocab_size}")
    
    # Test encoding
    test_keywords = ["python", "google", "machine learning", "san francisco"]
    indices, values = encoder.encode(test_keywords)
    
    print(f"\nQuery: {test_keywords}")
    print(f"Sparse vector: {len(indices)} non-zero terms")
    print(f"Top terms: {encoder.get_top_terms(indices, values, 10)}")
