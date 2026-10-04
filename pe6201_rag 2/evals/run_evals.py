"""run_evals.py - the evaluation harness (L1 deterministic + hand-labelled faithfulness).

NO LLM-as-judge. Automatic metrics compare system output with the hand-written
answer key in questions_50.csv:
  retrieval_doc_hit / retrieval_page_hit : was a gold doc / (doc,page) in the top-k?
  citation_present   : every answered response has >=1 citation (L1 assertion)
  citation_correct   : a cited (doc,page) matches the gold sources
  refusal_accuracy   : should_refuse rows that did NOT get an answer
  false_refusal_rate : answerable rows that did NOT get an answer
  pii_leak           : answered responses containing PII (L1 assertion; should be 0)
  cost / latency     : from token usage and wall-clock time
Faithfulness is judged by a human: export_for_labelling() -> label -> faithfulness().

Typical use (see notebook): run_eval(df, cfg) -> summarize(res) -> export_for_labelling(res)
"""
import os, json
import numpy as np
import pandas as pd
from src.config import Config, PDF_DIR, RESULTS_DIR
from src.pipeline import answer_question
from src.retrieve import retrieve
from src.guardrails import find_pii

REQUIRED = ["id", "split", "type", "question", "gold_answer", "gold_sources", "should_refuse"]


def parse_sources(s):
    """'a.pdf:12;b.pdf:5' -> {('a.pdf',12), ('b.pdf',5)} (file lowercased)."""
    out = set()
    if not isinstance(s, str):
        return out
    for part in s.split(";"):
        part = part.strip()
        if ":" in part:
            f, p = part.rsplit(":", 1)
            try:
                out.add((f.strip().lower(), int(p)))
            except ValueError:
                pass
    return out


def load_questions(path="evals/questions_50.csv"):
    df = pd.read_csv(path, dtype={"gold_sources": str})
    df["should_refuse"] = df["should_refuse"].astype(int)
    return df


def validate_questions(df):
    """Print counts and return a list of problems (typos in file names etc.)."""
    miss = [c for c in REQUIRED if c not in df.columns]
    if miss:
        return [f"missing columns: {miss}"]
    pdfs = {f.lower() for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")}
    problems = []
    for _, r in df.iterrows():
        srcs = parse_sources(r["gold_sources"])
        if r["should_refuse"] == 0 and not srcs:
            problems.append(f"{r['id']}: answerable question without gold_sources")
        for f, _p in srcs:
            if f not in pdfs:
                problems.append(f"{r['id']}: file not found in raw_pdfs: {f}")
        if "REPLACE" in str(r["gold_sources"]) or str(r["question"]).startswith("(示例)"):
            problems.append(f"{r['id']}: still a template row")
    print(f"{len(df)} questions | split:", df["split"].value_counts().to_dict(),
          "| type:", df["type"].value_counts().to_dict())
    if len(df) != 50:
        problems.append(f"expected 50 questions, found {len(df)}")
    for p in problems:
        print("  PROBLEM:", p)
    if not problems:
        print("questions look valid")
    return problems


def run_eval(df, cfg=None, log=False):
    """Run the full pipeline on every row; return a per-question results DataFrame."""
    cfg = cfg or Config()
    rows = []
    for _, r in df.iterrows():
        rec = answer_question(r["question"], cfg, log=log)
        gold = parse_sources(r["gold_sources"])
        gold_docs = {d for d, _ in gold}
        hit_pairs = [(h["doc"].lower(), h["page"]) for h in rec["hits"]]
        answered = rec["status"] == "answered"
        rows.append({
            "id": r["id"], "split": r["split"], "type": r["type"], "question": r["question"],
            "should_refuse": int(r["should_refuse"]), "gold_answer": r["gold_answer"],
            "status": rec["status"], "answered": answered, "answer": rec["answer"],
            "raw_answer": rec["raw_answer"], "top_score": rec["top_score"],
            "cited": ";".join(f"{d}:{p}" for d, p in rec["cited"]),
            "cited_n": len(rec["cited"]) if answered else 0,
            "doc_hit": any(d in gold_docs for d, _ in hit_pairs),
            "page_hit": any(hp in gold for hp in hit_pairs),
            "cite_correct": answered and any(c in gold for c in rec["cited"]),
            "pii_out": answered and bool(find_pii(rec["answer"])),
            "cost_usd": rec["cost_usd"], "latency_s": rec["latency_s"],
            "in_tokens": rec["in_tokens"], "out_tokens": rec["out_tokens"],
            "embed_tokens": rec["embed_tokens"], "retries": rec["retries"],
            "retrieved": ";".join(f"{h['doc']}:{h['page']}@{h['score']:.2f}" for h in rec["hits"]),
            "ctx": "\n---\n".join(f"[{h['doc']}, p.{h['page']}] {h['text']}" for h in rec["hits"]),
        })
    return pd.DataFrame(rows)


def _mean(s):
    return float(s.mean()) if len(s) else float("nan")


def summarize(res):
    """Aggregate metrics (a dict). See module docstring for definitions."""
    ans = res[res.should_refuse == 0]
    ref = res[res.should_refuse == 1]
    answered = res[res.answered]
    ans_answered = ans[ans.answered]
    return {
        "n": len(res), "n_answerable": len(ans), "n_should_refuse": len(ref),
        "retrieval_doc_hit": _mean(ans.doc_hit),
        "retrieval_page_hit": _mean(ans.page_hit),
        "answer_rate_on_answerable": _mean(ans.answered),
        "false_refusal_rate": 1 - _mean(ans.answered) if len(ans) else float("nan"),
        "refusal_accuracy": 1 - _mean(ref.answered) if len(ref) else float("nan"),
        "citation_present": _mean(answered.cited_n > 0),
        "citation_correct": _mean(ans_answered.cite_correct),
        "pii_leak_count": int(answered.pii_out.sum()),
        "avg_cost_usd": _mean(res.cost_usd),
        "avg_latency_s": _mean(res.latency_s),
        "p95_latency_s": float(res.latency_s.quantile(0.95)) if len(res) else float("nan"),
        "status_counts": res.status.value_counts().to_dict(),
    }


def threshold_sweep(df, cfg=None, thresholds=None):
    """Cheap sweep (retrieval only, no LLM calls): refusal vs false-refusal per threshold."""
    cfg = cfg or Config()
    thresholds = thresholds if thresholds is not None else np.arange(0.15, 0.61, 0.05)
    scores = np.array([retrieve(q, cfg)[0][0]["score"] for q in df["question"]])
    sr = df["should_refuse"].values == 1
    rows = []
    for t in thresholds:
        refused = scores < t
        rows.append({"threshold": round(float(t), 2),
                     "refusal_accuracy": float(refused[sr].mean()) if sr.any() else float("nan"),
                     "false_refusal_rate": float(refused[~sr].mean()) if (~sr).any() else float("nan")})
    return pd.DataFrame(rows)


def export_for_labelling(res, name):
    """Write a CSV for the human faithfulness labelling (answered rows only)."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    lab = res[res.answered][["id", "type", "question", "gold_answer", "answer", "cited", "retrieved", "ctx"]].copy()
    lab["faithful"] = ""      # you fill: 1 = every claim supported by the cited text, 0 = not
    lab["error_type"] = ""    # if 0: retrieval_miss / generation_unsupported / wrong_citation / other
    lab["notes"] = ""
    path = os.path.join(RESULTS_DIR, f"{name}_to_label.csv")
    lab.to_csv(path, index=False, encoding="utf-8-sig")
    res.drop(columns=["ctx"]).to_csv(os.path.join(RESULTS_DIR, f"{name}_results.csv"), index=False, encoding="utf-8-sig")
    print("labelling file:", path)
    return path


def faithfulness(label_path):
    """Compute hand-labelled faithfulness from the filled-in CSV."""
    lab = pd.read_csv(label_path)
    done = lab[lab["faithful"].isin([0, 1, "0", "1"])].copy()
    done["faithful"] = done["faithful"].astype(int)
    print(f"labelled {len(done)} / {len(lab)} answered rows")
    out = {"faithfulness": float(done.faithful.mean()) if len(done) else float("nan"), "n_labelled": len(done)}
    if (done.faithful == 0).any():
        out["error_types"] = done[done.faithful == 0]["error_type"].value_counts().to_dict()
    return out
