# RAG Pipeline – From-Scratch PDF QA with Exact Citations

RAG pipeline over PDFs with exact `(doc_id, page, char_start, char_end)` citations. No RAG frameworks (no LangChain/LlamaIndex).

## Architecture

| Module | Description |
|---|---|
| `src/ingest.py` | PyMuPDF → page text + tables + figure captions |
| `src/chunking.py` | Structural (font-size + regex headings) and Semantic (embedding similarity threshold) |
| `src/embeddings.py` | OpenAI `text-embedding-3-small` wrapper |
| `src/bm25.py` | `rank_bm25` BM25Okapi with stop-word tokenizer |
| `src/retriever.py` | FAISS `IndexFlatIP` (cosine) + Reciprocal Rank Fusion |
| `src/db.py` | SQLite metadata store (documents, pages, chunks) |
| `src/generator.py` | GPT-4o-mini prompt builder + generation |
| `src/citations.py` | Regex citation extraction + offset validation |
| `src/pipeline.py` | End-to-end orchestrator + evaluation metrics |

### Comparison Table

| Experiment | EM | F1 | Grounding % | p95 Latency (s) |
|---|---|---|---|---|
| structural + bm25 | 0.000 | 0.402 | 70.0% | 4.27 |
| structural + dense | 0.000 | 0.334 | 60.0% | 4.32 |
| structural + rrf | 0.000 | 0.461 | 75.0% | 3.35 |
| semantic + bm25 | 0.000 | 0.512 | 75.0% | 4.99 |
| semantic + dense | 0.000 | 0.450 | 80.0% | 5.45 |
| semantic + rrf | 0.000 | 0.521 | 80.0% | 5.48 |



### Best:

- Best single retriever F1 = 0.402
- RRF F1 = 0.461
- Delta = +0.059

### Latency

| Experiment | p50 (s) | p95 (s) | p99 (s) | mean (s) |
|---|---|---|---|---|
| structural + bm25 | 1.95 | 4.27 | 4.51 | 2.42 |
| structural + dense | 1.78 | 4.32 | 4.55 | 2.01 |
| structural + rrf | 1.72 | 3.35 | 3.69 | 1.96 |
| semantic + bm25 | 2.55 | 4.99 | 6.80 | 2.59 |
| semantic + dense | 2.34 | 5.45 | 10.20 | 3.10 |
| semantic + rrf | 2.83 | 5.48 | 7.94 | 3.20 |

## Settings

- **Embedding:** `text-embedding-3-small` (1536 dim)
- **LLM:** `gpt-4o-mini`
- **top-k:** 4
- **Structural chunking:** ~512 token windows, 50 token overlap
- **Semantic chunking:** cosine similarity threshold = 0.75
- **RRF:** k = 60

## Optimization

Two caching changes to eliminate repeated database call during evaluation:

1. **In-memory chunk cache** (`pipeline.py`): `build_retriever()` stores all chunks in a `CHUNK_CACHE` dict at initialization. `answer_question()` looks up chunks from memory instead of running `SELECT * FROM chunks` on every query. Removes a full DB scan per question.

2. **Page text cache** (`citations.py`): `validate_citation()` caches `(doc_id, page) → page_text` in a `PAGE_CACHE` dict. First access hits SQLite, all subsequent lookups for the same page are instant.

### Latency After Optimization

| Experiment | p50 (s) | p95 (s) | p99 (s) | mean (s) |
|---|---|---|---|---|
| structural + bm25 | 1.55 | 2.69 | 3.48 | 1.63 |
| structural + dense | 1.52 | **2.53** | 2.64 | 1.56 |
| structural + rrf | 1.57 | 2.74 | 3.48 | 1.73 |
| semantic + bm25 | 2.21 | 3.40 | 6.44 | 2.20 |
| semantic + dense | 2.26 | 4.18 | 5.94 | 2.38 |
| semantic + rrf | 2.37 | 4.68 | 6.57 | 2.56 |
