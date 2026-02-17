import openai
import os
import numpy as np
from dotenv import load_dotenv

load_dotenv()

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = "text-embedding-3-small"


def embed_texts(texts, model=MODEL):
    """Embed a list of strings. Returns list of 1536-dim vectors."""
    if not texts:
        return []

    batch_size = 100
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch = [t if t.strip() else " " for t in batch]

        resp = client.embeddings.create(input=batch, model=model)
        all_embeddings.extend([item.embedding for item in resp.data])

    return all_embeddings


def embed_query(query, model=MODEL):
    """Embed a single query string."""
    resp = client.embeddings.create(input=[query], model=model)
    return resp.data[0].embedding


def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))
