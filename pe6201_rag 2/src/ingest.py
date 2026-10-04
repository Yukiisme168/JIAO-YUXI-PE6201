"""ingest.py - PDF -> page text -> chunks -> embeddings -> saved index.

We OWN the chunking logic because it strongly affects retrieval quality.
Each chunk keeps its source file name and PHYSICAL page number (1-based, as
shown by a PDF viewer's page counter). That metadata is what makes
citation checking and citation-accuracy evaluation possible later.

Index layout (one folder per chunking setting): index/<tag>/
    chunks.json  list of {id, doc, page, text}
    vectors.npy  L2-normalised float32 matrix, row i <-> chunks[i]
    meta.json    counts, embedding tokens and cost, pages with no text
"""
import os, re, json
import numpy as np
from pypdf import PdfReader
from .config import PDF_DIR, INDEX_DIR, Config, embed_cost
from .llm import embed_texts

MIN_CHUNK_CHARS = 50


def extract_pages(pdf_path):
    """Return [(page_no, cleaned_text)] for one PDF. Empty text => scanned page."""
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        pages.append((i, re.sub(r"\s+", " ", text).strip()))
    return pages


def chunk_text(text, size, overlap):
    """Fixed-size character windows with overlap. Chunks never cross a page."""
    if len(text) <= size:
        return [text]
    chunks, start, step = [], 0, max(1, size - overlap)
    while start < len(text):
        chunks.append(text[start:start + size])
        if start + size >= len(text):
            break
        start += step
    return chunks


def build_index(cfg=None, force=False):
    """Build (or reuse) the index for cfg's chunking params. Returns meta dict."""
    cfg = cfg or Config()
    out = os.path.join(INDEX_DIR, cfg.tag)
    meta_path = os.path.join(out, "meta.json")
    if os.path.exists(meta_path) and not force:
        print(f"[ingest] reuse existing index {cfg.tag}")
        return json.load(open(meta_path))

    pdfs = sorted(f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in {PDF_DIR}")
    chunks, empty_pages, n_pages = [], [], 0
    for fname in pdfs:
        for page_no, text in extract_pages(os.path.join(PDF_DIR, fname)):
            n_pages += 1
            if len(text) < MIN_CHUNK_CHARS:
                empty_pages.append(f"{fname}:{page_no}")
                continue
            for piece in chunk_text(text, cfg.chunk_size, cfg.chunk_overlap):
                if len(piece) >= MIN_CHUNK_CHARS:
                    chunks.append({"id": len(chunks), "doc": fname, "page": page_no, "text": piece})

    print(f"[ingest] {len(pdfs)} PDFs, {n_pages} pages, {len(chunks)} chunks; embedding...")
    vecs, tokens = embed_texts([c["text"] for c in chunks])
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

    os.makedirs(out, exist_ok=True)
    json.dump(chunks, open(os.path.join(out, "chunks.json"), "w"), ensure_ascii=False)
    np.save(os.path.join(out, "vectors.npy"), vecs)
    meta = {"tag": cfg.tag, "n_docs": len(pdfs), "n_pages": n_pages, "n_chunks": len(chunks),
            "embed_tokens": tokens, "embed_cost_usd": embed_cost(tokens),
            "pages_without_text": empty_pages}
    json.dump(meta, open(meta_path, "w"), indent=2)
    if empty_pages:
        print(f"[ingest] WARNING {len(empty_pages)} pages had no extractable text (scanned?). "
              f"See meta.json. They cannot be answered from.")
    print(f"[ingest] done. embedding tokens={tokens}, cost=${meta['embed_cost_usd']:.4f}")
    return meta


def find_pages(keyword, cfg=None, max_results=15, width=120):
    """Labelling helper: show where a keyword appears (doc + physical page).

    Use it while writing the 50-question answer key to locate gold pages fast.
    """
    cfg = cfg or Config()
    chunks = json.load(open(os.path.join(INDEX_DIR, cfg.tag, "chunks.json")))
    seen, shown = set(), 0
    for c in chunks:
        m = re.search(re.escape(keyword), c["text"], flags=re.I)
        if m and (c["doc"], c["page"]) not in seen:
            seen.add((c["doc"], c["page"]))
            s = max(0, m.start() - width // 2)
            print(f"{c['doc']} | p.{c['page']} | ...{c['text'][s:s + width]}...")
            shown += 1
            if shown >= max_results:
                break
    if not shown:
        print("no match")
