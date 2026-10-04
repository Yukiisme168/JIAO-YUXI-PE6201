"""retrieve.py - embed the question, find the top-k most similar chunks.

Returns the hits plus the embedding tokens used (for costing). The caller
(pipeline.py) applies the similarity threshold: if the best score is below
cfg.sim_threshold the system refuses instead of letting the LLM guess.
Similarity = cosine (vectors are normalised, so a dot product).
"""
import os, json
import numpy as np
from .config import INDEX_DIR, Config
from .llm import embed_texts

_cache = {}


def load_index(cfg):
    d = os.path.join(INDEX_DIR, cfg.tag)
    if d not in _cache:
        chunks = json.load(open(os.path.join(d, "chunks.json")))
        vecs = np.load(os.path.join(d, "vectors.npy"))
        _cache[d] = (chunks, vecs)
    return _cache[d]


def retrieve(question, cfg=None):
    """Return (hits, embed_tokens). hits = top_k dicts {id, doc, page, text, score}."""
    cfg = cfg or Config()
    chunks, vecs = load_index(cfg)
    qv, tokens = embed_texts([question])
    qv = qv[0] / np.linalg.norm(qv[0])
    scores = vecs @ qv
    idx = np.argsort(-scores)[:cfg.top_k]
    hits = [{**chunks[i], "score": float(scores[i])} for i in idx]
    return hits, tokens
