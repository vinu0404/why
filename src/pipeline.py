"""
End-to-end RAG pipeline:
  ingest PDFs  ->  chunk  ->  embed  ->  index  ->  retrieve  ->  generate
"""
import os
import re
import json
import time
import numpy as np
from collections import Counter

from src.ingest import ingest_pdf_folder
from src.chunking import structural_chunk, semantic_chunk
from src.embeddings import embed_texts
from src.db import (
    init_db, insert_document, insert_page,
    insert_chunks_batch, get_all_chunks, clear_chunks,
)
from src.retriever import HybridRetriever
from src.generator import generate_answer
from src.citations import compute_grounding

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PDF_DIR = os.path.join(BASE_DIR, "pdf")
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "rag.db")
QA_PATH = os.path.join(BASE_DIR, "eval", "qa.json")

def run_ingestion(chunk_mode="structural"):
    print(f"INGESTION chunk_mode = {chunk_mode}")
    documents = ingest_pdf_folder(PDF_DIR)
    conn = init_db(DB_PATH)
    clear_chunks(conn, chunk_mode=chunk_mode)
    all_chunks = []
    for doc in documents:
        insert_document(conn, doc["doc_id"], doc["filename"], doc["num_pages"])

        for page_data in doc["pages"]:
            insert_page(conn, doc["doc_id"], page_data["page"], page_data["text"])

            if chunk_mode == "structural":
                chunks = structural_chunk(page_data)
            else:
                chunks = semantic_chunk(page_data)

            all_chunks.extend(chunks)
    texts = [c["chunk_text"] for c in all_chunks]
    embeddings = embed_texts(texts)
    for i, c in enumerate(all_chunks):
        c["embedding"] = embeddings[i]

    insert_chunks_batch(conn, all_chunks)
    conn.close()
    return all_chunks

CHUNK_CACHE = {}

def build_retriever(chunk_mode="structural"):
    global CHUNK_CACHE

    conn = init_db(DB_PATH)
    chunks = get_all_chunks(conn, chunk_mode=chunk_mode)
    conn.close()

    if not chunks:
        print(f"no chunks for mode '{chunk_mode}' – run ingestion first")
        return None

    # cache chunks in memory so answer_question never hits the DB
    CHUNK_CACHE = {c["chunk_id"]: c for c in chunks}

    retriever = HybridRetriever()
    retriever.build(chunks)
    return retriever


def answer_question(question, retriever, chunk_mode="structural",
                    retrieval_method="rrf", top_k=4):
    conn = init_db(DB_PATH)

    results = retriever.search(question, top_k=top_k, method=retrieval_method)

    # use in-memory cache instead of loading all chunks from DB
    context_chunks = [
        CHUNK_CACHE[r["chunk_id"]]
        for r in results if r["chunk_id"] in CHUNK_CACHE
    ]

    answer = generate_answer(question, context_chunks)
    grounding = compute_grounding(answer, conn)
    conn.close()

    return {
        "question": question,
        "answer": answer,
        "retrieved": [
            {"chunk_id": c["chunk_id"], "doc_id": c["doc_id"],
             "page": c["page"], "char_start": c["char_start"],
             "char_end": c["char_end"]}
            for c in context_chunks
        ],
        "grounding": grounding,
    }

def run_evaluation(qa_path=None, retriever=None,
                   chunk_mode="structural", retrieval_method="rrf",
                   top_k=4):
    qa_path = qa_path or QA_PATH
    with open(qa_path, "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)

    results = []
    latencies = []

    for i, qa in enumerate(qa_pairs):
        print(f"  Q{i+1}/{len(qa_pairs)}: {qa['question'][:55]}…")

        t0 = time.time()
        res = answer_question(
            qa["question"], retriever,
            chunk_mode=chunk_mode,
            retrieval_method=retrieval_method,
            top_k=top_k,
        )
        elapsed = time.time() - t0
        latencies.append(elapsed)

        em = compute_exact_match(res["answer"], qa["gold_answer"])
        f1 = compute_f1(res["answer"], qa["gold_answer"])

        res["gold_answer"] = qa["gold_answer"]
        res["gold_doc"] = qa.get("gold_doc")
        res["gold_page"] = qa.get("gold_page")
        res["em"] = em
        res["f1"] = f1
        res["latency"] = elapsed
        results.append(res)

        g = res["grounding"]["grounding_pct"]
        print(f"    EM={em}  F1={f1:.3f}  grounding={g:.0f}%  time={elapsed:.2f}s")

    # aggregate
    summary = {
        "chunk_mode": chunk_mode,
        "retrieval_method": retrieval_method,
        "top_k": top_k,
        "n": len(results),
        "avg_em": np.mean([r["em"] for r in results]),
        "avg_f1": np.mean([r["f1"] for r in results]),
        "avg_grounding": np.mean([r["grounding"]["grounding_pct"]
                                   for r in results]),
        "p95_latency": np.percentile(latencies, 95),
    }

    print(f"\n  --- {chunk_mode} + {retrieval_method} ---")
    print(f"  EM:        {summary['avg_em']:.3f}")
    print(f"  F1:        {summary['avg_f1']:.3f}")
    print(f"  Grounding: {summary['avg_grounding']:.1f}%")
    print(f"  p95 lat:   {summary['p95_latency']:.2f}s")

    return summary, results


def normalize_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text) 
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def compute_exact_match(predicted, gold):
    return int(normalize_text(predicted) == normalize_text(gold))


def compute_f1(predicted, gold):
    pred_tokens = normalize_text(predicted).split()
    gold_tokens = normalize_text(gold).split()

    if not pred_tokens or not gold_tokens:
        return 0.0

    pred_counter = Counter(pred_tokens)
    gold_counter = Counter(gold_tokens)

    common = 0
    for tok in pred_counter:
        common += min(pred_counter[tok], gold_counter.get(tok, 0))

    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)
