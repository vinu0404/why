import openai
import os
from dotenv import load_dotenv

load_dotenv()

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = "gpt-4o-mini"


def build_prompt(question, context_chunks):
    """Assemble the RAG prompt with inline source metadata."""

    ctx_parts = []
    for i, c in enumerate(context_chunks, 1):
        src = (f"doc_id={c['doc_id']}, page={c['page']}, "
               f"char_start={c['char_start']}, char_end={c['char_end']}")
        ctx_parts.append(f"--- Context {i} ---\n{c['chunk_text']}\nSource: {src}\n")

    context_str = "\n".join(ctx_parts)

    prompt = f"""You are a precise document QA system. Answer ONLY using the provided context below.

CITATION RULES (follow exactly):
- Every factual sentence MUST end with a citation.
- Citation format: [doc_id | page | char_start:char_end]
- You MUST copy the doc_id, page, char_start, and char_end values EXACTLY as they appear in the Source line. Do NOT change, round, or invent any numbers.
- Each citation must use values from ONE Source line. Never combine values from different sources.

Example:
  Source: doc_id=cricket-rules, page=6, char_start=1204, char_end=1587
  Correct citation: [cricket-rules | 6 | 1204:1587]
  WRONG: [cricket-rules | 6 | 1200:1590]  ← do NOT alter the numbers

Context:
{context_str}

Question: {question}

Instructions:
1. Answer concisely using ONLY the context above.
2. End every factual sentence with a citation copied exactly from a Source line.
3. If the context does not contain the answer, say "Not enough information in the provided context."
4. Do NOT make up information or citation values.

Answer:"""

    return prompt


def generate_answer(question, context_chunks, model=MODEL, max_tokens=512):
    prompt = build_prompt(question, context_chunks)

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system",
             "content": "You are a helpful document QA assistant that always cites sources precisely."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
        temperature=0.0,
    )
    return resp.choices[0].message.content
