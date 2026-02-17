import re
from src.db import get_page_text, get_all_chunks

PAGE_CACHE = {}

def extract_citations(answer_text):
    """
    Pull citations out of the generated answer.
    Expected format:  [doc_id | page | char_start:char_end]
    """
    pattern = r'\[([^|\]]+)\|\s*(\d+)\s*\|\s*(\d+):(\d+)\]'
    citations = []
    for m in re.finditer(pattern, answer_text):
        citations.append({
            "doc_id": m.group(1).strip(),
            "page": int(m.group(2)),
            "char_start": int(m.group(3)),
            "char_end": int(m.group(4)),
            "raw": m.group(0),
        })
    return citations


def validate_citation(citation, conn):
    """
    Check that the cited (doc_id, page, char_start:char_end) actually
    falls inside a stored chunk and that the page text at those offsets
    contains real content.
    """
    key = (citation["doc_id"], citation["page"])
    if key not in PAGE_CACHE:
        PAGE_CACHE[key] = get_page_text(conn, citation["doc_id"], citation["page"])
    page_text = PAGE_CACHE[key]

    if page_text is not None:
        cs, ce = citation["char_start"], citation["char_end"]
        if 0 <= cs < ce <= len(page_text):
            snippet = page_text[cs:ce]
            if snippet.strip():
                return {"valid": True, "cited_text": snippet}
    chunks = get_all_chunks(conn)
    for chunk in chunks:
        if chunk["doc_id"] != citation["doc_id"]:
            continue
        if chunk["page"] != citation["page"]:
            continue
        if (citation["char_start"] >= chunk["char_start"] and
                citation["char_end"] <= chunk["char_end"]):
            offset = citation["char_start"] - chunk["char_start"]
            length = citation["char_end"] - citation["char_start"]
            cited = chunk["chunk_text"][offset:offset + length]
            return {"valid": True, "cited_text": cited}

    return {"valid": False, "cited_text": None}


def compute_grounding(answer_text, conn):
    """Return grounding % and per-citation details."""
    citations = extract_citations(answer_text)

    if not citations:
        return {
            "grounding_pct": 0.0,
            "total": 0,
            "valid": 0,
            "details": [],
        }

    valid_count = 0
    details = []
    for cit in citations:
        result = validate_citation(cit, conn)
        if result["valid"]:
            valid_count += 1
        details.append({
            "citation": cit["raw"],
            "valid": result["valid"],
            "preview": (result["cited_text"][:80]
                        if result["cited_text"] else None),
        })

    return {
        "grounding_pct": valid_count / len(citations) * 100,
        "total": len(citations),
        "valid": valid_count,
        "details": details,
    }
