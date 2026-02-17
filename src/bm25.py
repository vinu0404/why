import re
from rank_bm25 import BM25Okapi

STOP_WORDS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'in', 'on', 'at',
    'to', 'for', 'of', 'and', 'or', 'it', 'this', 'that', 'with',
    'by', 'from', 'as', 'be', 'has', 'had', 'have', 'been', 'its',
}


def tokenize(text):
    """Lowercase, extract words, drop stop-words."""
    tokens = re.findall(r'\b\w+\b', text.lower())
    return [t for t in tokens if t not in STOP_WORDS]


class BM25Index:
    def __init__(self):
        self.index = None
        self.chunk_ids = []

    def build(self, chunks):
        """Build BM25 index from a list of chunk dicts."""
        self.chunk_ids = [c["chunk_id"] for c in chunks]
        corpus = [tokenize(c["chunk_text"]) for c in chunks]
        self.index = BM25Okapi(corpus)
        print(f"  BM25 index built {len(corpus)} chunks")

    def search(self, query, top_k=5):
        tokens = tokenize(query)
        scores = self.index.get_scores(tokens)

        top_idx = sorted(range(len(scores)),
                         key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for rank, idx in enumerate(top_idx, 1):
            results.append({
                "chunk_id": self.chunk_ids[idx],
                "score": float(scores[idx]),
                "rank": rank,
            })
        return results
