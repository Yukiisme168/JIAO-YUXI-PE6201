# RAG-powered Compliance Q&A Assistant (PE6201 End-of-Course Project)

Answers bank-compliance questions using ONLY a corpus of **6 public MAS regulatory documents**.
Every answer carries `[file.pdf, p.N]` citations; low-confidence questions are refused.

## How to run (Google Colab)
1. Upload this repo (or `git clone` it) into Google Drive at `MyDrive/pe6201_rag/`.
2. Put the 6 PDFs in `data/raw_pdfs/` (see `data/DATA_README.md`).
3. In Colab: add a Secret named `OPENROUTER_API_KEY` (key icon, left sidebar, enable notebook access).
4. Open `PE6201.ipynb` and run the cells top to bottom.

## Repo map
- `src/`    code (config, llm, ingest, retrieve, generate, guardrails, audit, pipeline)
- `evals/`  harness (`run_evals.py`), 50-question answer key (`questions_50.csv`), `EVALS_README.md`, `results/`
- `data/`   corpus + `DATA_README.md`
- `PRODUCT.md` persona, I/O, architecture, metrics targeted vs reached

## Rented vs owned
Rented (API): embeddings + answer generation. Owned: chunking, prompt, retrieval gate,
citation/PII guardrails, audit log, and the evaluation harness. No model is trained or fine-tuned.

## Key results (final test set, n=27)
- Faithfulness (hand-labelled, n=20 answered): **1.00**
- Citation accuracy: **0.95**
- Retrieval page hit@5: **0.96**
- Refusal accuracy (should-refuse rows): **1.00**
- PII leakage: **0**
- Cost per question: **S$0.00028** (S$0.28 per 1000 questions)
