import sqlite3
import os
import numpy as np

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "rag.db")


def get_connection(db_path=None):
    if db_path is None:
        db_path = DB_PATH
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path=None):
    conn = get_connection(db_path)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            doc_id      TEXT PRIMARY KEY,
            filename    TEXT,
            num_pages   INTEGER,
            ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS pages (
            doc_id      TEXT,
            page        INTEGER,
            page_text   TEXT,
            PRIMARY KEY (doc_id, page),
            FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id    TEXT PRIMARY KEY,
            doc_id      TEXT,
            page        INTEGER,
            char_start  INTEGER,
            char_end    INTEGER,
            heading     TEXT,
            chunk_text  TEXT,
            token_count INTEGER,
            chunk_mode  TEXT,
            embedding   BLOB,
            FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
        )
    """)

    conn.commit()
    return conn


def insert_document(conn, doc_id, filename, num_pages):
    conn.execute(
        "INSERT OR REPLACE INTO documents (doc_id, filename, num_pages) VALUES (?, ?, ?)",
        (doc_id, filename, num_pages)
    )
    conn.commit()


def insert_page(conn, doc_id, page, page_text):
    conn.execute(
        "INSERT OR REPLACE INTO pages (doc_id, page, page_text) VALUES (?, ?, ?)",
        (doc_id, page, page_text)
    )
    conn.commit()


def get_page_text(conn, doc_id, page):
    """Return the raw page text so we can verify citations."""
    row = conn.execute(
        "SELECT page_text FROM pages WHERE doc_id = ? AND page = ?",
        (doc_id, page)
    ).fetchone()
    return row["page_text"] if row else None


def insert_chunk(conn, chunk):
    emb_blob = None
    if chunk.get("embedding") is not None:
        emb_blob = np.array(chunk["embedding"], dtype=np.float32).tobytes()

    conn.execute(
        """INSERT OR REPLACE INTO chunks
           (chunk_id, doc_id, page, char_start, char_end,
            heading, chunk_text, token_count, chunk_mode, embedding)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            chunk["chunk_id"], chunk["doc_id"], chunk["page"],
            chunk["char_start"], chunk["char_end"],
            chunk.get("heading"), chunk["chunk_text"],
            chunk.get("token_count", 0), chunk["chunk_mode"],
            emb_blob,
        )
    )


def insert_chunks_batch(conn, chunks):
    for c in chunks:
        insert_chunk(conn, c)
    conn.commit()


def get_all_chunks(conn, chunk_mode=None):
    if chunk_mode:
        rows = conn.execute(
            "SELECT * FROM chunks WHERE chunk_mode = ?", (chunk_mode,)
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM chunks").fetchall()

    results = []
    for row in rows:
        chunk = dict(row)
        if chunk["embedding"] is not None:
            chunk["embedding"] = np.frombuffer(
                chunk["embedding"], dtype=np.float32
            ).tolist()
        results.append(chunk)
    return results


def get_chunk_by_id(conn, chunk_id):
    row = conn.execute(
        "SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)
    ).fetchone()
    if row:
        chunk = dict(row)
        if chunk["embedding"]:
            chunk["embedding"] = np.frombuffer(
                chunk["embedding"], dtype=np.float32
            ).tolist()
        return chunk
    return None


def clear_chunks(conn, chunk_mode=None):
    if chunk_mode:
        conn.execute("DELETE FROM chunks WHERE chunk_mode = ?", (chunk_mode,))
    else:
        conn.execute("DELETE FROM chunks")
    conn.commit()
