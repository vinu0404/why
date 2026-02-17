import os
import numpy as np
import faiss

from src.embeddings import embed_query
from src.bm25 import BM25Index

FAISS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "faiss.index")


class DenseRetriever:
    """Cosine-similarity search over chunk embeddings using FAISS."""

    def __init__(self):
        self.index = None
        self.chunk_ids = []

    def build(self, chunks):
        embeddings = []
        self.chunk_ids = []

        for c in chunks:
            if c.get("embedding"):
                embeddings.append(c["embedding"])
                self.chunk_ids.append(c["chunk_id"])

        if not embeddings:
            print("  !! no embeddings to index")
            return

        vecs = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(vecs)       

        dim = vecs.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(vecs)
        print(f"  FAISS index built – {len(self.chunk_ids)} vectors, dim={dim}")

    def save(self, path=None):
        path = path or FAISS_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)
        faiss.write_index(self.index, path)
        np.save(path + ".ids.npy", np.array(self.chunk_ids))

    def load(self, path=None):
        path = path or FAISS_PATH
        self.index = faiss.read_index(path)
        self.chunk_ids = np.load(path + ".ids.npy", allow_pickle=True).tolist()

    def search(self, query, top_k=5):
        q = np.array([embed_query(query)], dtype=np.float32)
        faiss.normalize_L2(q)

        scores, indices = self.index.search(q, top_k)

        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), 1):
            if idx == -1:
                continue
            results.append({
                "chunk_id": self.chunk_ids[idx],
                "score": float(score),
                "rank": rank,
            })
        return results



def reciprocal_rank_fusion(result_lists, k=60):
    """
    Merge several ranked lists with RRF.
    """
    rrf_scores = {}

    for results in result_lists:
        for item in results:
            cid = item["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + item["rank"])

    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    return [
        {"chunk_id": cid, "rrf_score": score, "rank": rank}
        for rank, (cid, score) in enumerate(ranked, 1)
    ]


class HybridRetriever:
    def __init__(self):
        self.bm25 = BM25Index()
        self.dense = DenseRetriever()

    def build(self, chunks):
        self.bm25.build(chunks)
        self.dense.build(chunks)

    def search(self, query, top_k=5, method="rrf"):
        if method == "bm25":
            return self.bm25.search(query, top_k)
        elif method == "dense":
            return self.dense.search(query, top_k)
        elif method == "rrf":
            bm25_res = self.bm25.search(query, top_k * 2)
            dense_res = self.dense.search(query, top_k * 2)
            fused = reciprocal_rank_fusion([bm25_res, dense_res])
            return fused[:top_k]
        else:
            raise ValueError(f"unknown retrieval method: {method}")
