import fitz
import os
import re


def extract_pages(pdf_path):
    """Extract text and metadata from each page of a PDF.
    """
    doc = fitz.open(pdf_path)
    doc_id = os.path.splitext(os.path.basename(pdf_path))[0]

    pages = []
    for page_num in range(len(doc)):
        page = doc[page_num]

        # full page text 
        page_text = page.get_text("text")
        text_blocks = []
        raw_dict = page.get_text("dict")
        for block in raw_dict.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text_blocks.append({
                        "text": span["text"],
                        "font_size": span["size"],
                        "font_flags": span["flags"],
                        "bbox": span["bbox"],
                    })

        # tables
        tables = []
        try:
            for tab in page.find_tables().tables:
                data = tab.extract()
                if data:
                    tables.append({
                        "markdown": _table_to_markdown(data),
                        "bbox": tab.bbox,
                    })
        except Exception:
            pass

        # figure / table captions
        captions = _find_figure_captions(page_text)

        pages.append({
            "doc_id": doc_id,
            "page": page_num + 1,  
            "text": page_text,
            "text_blocks": text_blocks,
            "tables": tables,
            "captions": captions,
        })

    doc.close()

    return {
        "doc_id": doc_id,
        "filename": os.path.basename(pdf_path),
        "num_pages": len(pages),
        "pages": pages,
    }


def _table_to_markdown(table_data):
    """Turn a 2-D list into a markdown table string."""
    if not table_data or not table_data[0]:
        return ""
    header = [str(c) if c else "" for c in table_data[0]]
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for row in table_data[1:]:
        cells = [str(c) if c else "" for c in row]
        while len(cells) < len(header):
            cells.append("")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _find_figure_captions(page_text):
    captions = []
    patterns = [
        r'(Figure\s+\d+[\.\:].*?)(?:\n|$)',
        r'(Fig\.\s+\d+[\.\:].*?)(?:\n|$)',
        r'(Table\s+\d+[\.\:].*?)(?:\n|$)',
    ]
    for pat in patterns:
        for m in re.finditer(pat, page_text, re.IGNORECASE):
            captions.append({
                "text": m.group(1).strip(),
                "char_start": m.start(),
                "char_end": m.end(),
            })
    return captions


def ingest_pdf_folder(pdf_folder):
    """Process every PDF in a folder and return list of doc dicts."""
    documents = []
    for fname in sorted(os.listdir(pdf_folder)):
        if not fname.lower().endswith(".pdf"):
            continue
        path = os.path.join(pdf_folder, fname)
        print(f"  extracting: {fname}")
        doc_data = extract_pages(path)
        documents.append(doc_data)
        print(f"    -> {doc_data['num_pages']} pages")

    print(f"  total docs: {len(documents)}")
    return documents
