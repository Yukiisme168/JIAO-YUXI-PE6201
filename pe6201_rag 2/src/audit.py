"""audit.py - append-only audit log (one JSON object per line).

Records every question, the retrieved chunks, the answer, the status and the
token usage / cost. This is the "audit log" mitigation from the proposal and
the source of the per-question cost numbers. Questions containing PII are
redacted before they are written.
"""
import os, json, datetime
from .config import LOG_DIR


def log_event(rec):
    os.makedirs(LOG_DIR, exist_ok=True)
    slim = dict(rec)
    slim["hits"] = [{"doc": h["doc"], "page": h["page"], "score": round(h["score"], 4),
                     "text": h["text"][:300]} for h in rec.get("hits", [])]
    slim["timestamp"] = datetime.datetime.now().isoformat(timespec="seconds")
    with open(os.path.join(LOG_DIR, "audit.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(slim, ensure_ascii=False) + "\n")
