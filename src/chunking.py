import re
import uuid
import tiktoken
import numpy as np
from src.embeddings import embed_texts, cosine_similarity

enc = tiktoken.encoding_for_model("gpt-4o-mini")


def count_tokens(text):
    return len(enc.encode(text))

#  STRUCTURAL CHUNKING (section-aware, 512 tokens) 

def detect_headings(text_blocks):
    """
    Return a list of heading strings found in the page's text blocks.
    """
    if not text_blocks:
        return []

    sizes = [b["font_size"] for b in text_blocks if b["text"].strip()]
    if not sizes:
        return []
    median_size = sorted(sizes)[len(sizes) // 2]

    headings = []
    for b in text_blocks:
        txt = b["text"].strip()
        if not txt:
            continue

        is_heading = False
        if b["font_size"] > median_size * 1.15:
            is_heading = True
        if b["font_flags"] & (1 << 4) and b["font_size"] >= median_size:
            is_heading = True
        if re.match(r'^(Article|Section|Chapter|Part|ARTICLE|SECTION)\s+\d+', txt):
            is_heading = True
        if re.match(r'^\d+\.\d+', txt):
            is_heading = True

        if is_heading:
            headings.append(txt)

    return headings


def structural_chunk(page_data, max_tokens=512, overlap_tokens=50):
    """
    Split page text into chunks aligned to section headings.
    If a section is longer than max_tokens it gets a sliding-window split.
    """
    page_text = page_data["text"]
    doc_id = page_data["doc_id"]
    page_num = page_data["page"]
    text_blocks = page_data.get("text_blocks", [])

    if not page_text.strip():
        return []

    heading_texts = detect_headings(text_blocks)
    heading_positions = []
    search_from = 0
    for h in heading_texts:
        idx = page_text.find(h, search_from)
        if idx >= 0:
            heading_positions.append((idx, h))
            search_from = idx + len(h)
    sections = []
    if heading_positions:
        first_pos = heading_positions[0][0]
        if first_pos > 0 and page_text[:first_pos].strip():
            sections.append({
                "heading": None,
                "char_start": 0,
                "char_end": first_pos,
            })
        for i, (pos, heading) in enumerate(heading_positions):
            end_pos = heading_positions[i + 1][0] if i + 1 < len(heading_positions) else len(page_text)
            sections.append({
                "heading": heading,
                "char_start": pos,
                "char_end": end_pos,
            })
    else:
        sections.append({
            "heading": None,
            "char_start": 0,
            "char_end": len(page_text),
        })
    chunks = []
    for sec in sections:
        raw_text = page_text[sec["char_start"]:sec["char_end"]]
        text, cs, ce = _trim_offsets(raw_text, sec["char_start"])
        if not text:
            continue
        tokens = count_tokens(text)

        if tokens <= max_tokens:
            chunks.append(_make_chunk(doc_id, page_num, cs, ce, text, tokens,
                                       sec["heading"], "structural"))
        else:
            for sub in _sliding_window_split(text, cs, max_tokens, overlap_tokens):
                chunks.append(_make_chunk(
                    doc_id, page_num, sub["char_start"], sub["char_end"],
                    sub["text"], count_tokens(sub["text"]),
                    sec["heading"], "structural"))
    return chunks


#  SEMANTIC CHUNKING (embedding-similarity threshold)

def _split_sentences(text):
    """Rough sentence splitter (period/!/? followed by space, or double newline)."""
    parts = re.split(r'(?<=[.!?])\s+|\n{2,}', text)
    return [p.strip() for p in parts if p.strip()]


def semantic_chunk(page_data, similarity_threshold=0.75, max_tokens=512):
    """
    Group consecutive sentences whose embeddings are similar.
    When similarity drops below the threshold a new chunk starts.
    """
    page_text = page_data["text"]
    doc_id = page_data["doc_id"]
    page_num = page_data["page"]

    if not page_text.strip():
        return []

    sentences = _split_sentences(page_text)
    if not sentences:
        return []
    spans = []     
    cursor = 0
    for sent in sentences:
        idx = page_text.find(sent, cursor)
        if idx == -1:
            idx = cursor           
        spans.append((idx, idx + len(sent)))
        cursor = idx + len(sent)
    if len(sentences) == 1:
        cs, ce = spans[0]
        txt = page_text[cs:ce]
        return [_make_chunk(doc_id, page_num, cs, ce, txt,
                            count_tokens(txt), None, "semantic")]

    embeddings = embed_texts(sentences)
    sims = []
    for i in range(len(embeddings) - 1):
        sims.append(cosine_similarity(embeddings[i], embeddings[i + 1]))

    # group: break whenever similarity < threshold
    groups = [[0]]
    for i, sim in enumerate(sims):
        if sim < similarity_threshold:
            groups.append([i + 1])
        else:
            groups[-1].append(i + 1)

    # merge tiny groups (< 2 sentences) into their neighbour
    merged = []
    for g in groups:
        if merged and len(g) < 2:
            merged[-1].extend(g)
        else:
            merged.append(g)
    chunks = []
    for group in merged:
        cs = spans[group[0]][0]
        ce = spans[group[-1]][1]
        chunk_text = page_text[cs:ce]
        tokens = count_tokens(chunk_text)

        if tokens > max_tokens:
            for sub in _sliding_window_split(chunk_text, cs, max_tokens, 50):
                chunks.append(_make_chunk(
                    doc_id, page_num, sub["char_start"], sub["char_end"],
                    sub["text"], count_tokens(sub["text"]), None, "semantic"))
        else:
            chunks.append(_make_chunk(doc_id, page_num, cs, ce,
                                       chunk_text, tokens, None, "semantic"))
    return chunks


def _trim_offsets(text, base_offset):
    """Strip leading/trailing whitespace and return adjusted offsets."""
    lstrip = len(text) - len(text.lstrip())
    stripped = text.strip()
    new_start = base_offset + lstrip
    new_end = new_start + len(stripped)
    return stripped, new_start, new_end


def _make_chunk(doc_id, page, cs, ce, text, token_count, heading, mode):
    return {
        "chunk_id": str(uuid.uuid4()),
        "doc_id": doc_id,
        "page": page,
        "char_start": cs,
        "char_end": ce,
        "heading": heading,
        "chunk_text": text,
        "token_count": token_count,
        "chunk_mode": mode,
    }


def _sliding_window_split(text, base_offset, max_tokens, overlap_tokens):
    """
    Split *text* into overlapping windows.
    Offsets returned are absolute (base_offset + position in *text*).
    """
    word_spans = [(m.start(), m.end(), m.group())
                  for m in re.finditer(r'\S+', text)]
    if not word_spans:
        return []

    chunks = []
    start_w = 0

    while start_w < len(word_spans):
        end_w = start_w
        while end_w < len(word_spans):
            window = text[word_spans[start_w][0]:word_spans[end_w][1]]
            if count_tokens(window) > max_tokens and end_w > start_w:
                end_w -= 1
                break
            end_w += 1
        else:
            end_w -= 1  

        if end_w < start_w:
            end_w = start_w

        cs = word_spans[start_w][0]
        ce = word_spans[end_w][1]

        chunks.append({
            "text": text[cs:ce],
            "char_start": base_offset + cs,
            "char_end": base_offset + ce,
        })
        next_start = end_w + 1
        for ow in range(end_w, start_w, -1):
            overlap_text = text[word_spans[ow][0]:word_spans[end_w][1]]
            if count_tokens(overlap_text) >= overlap_tokens:
                next_start = ow
                break

        if next_start <= start_w:
            next_start = end_w + 1 

        start_w = next_start

    return chunks
