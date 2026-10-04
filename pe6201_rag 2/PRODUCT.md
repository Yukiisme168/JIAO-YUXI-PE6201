# Product documentation

## Persona
Full-time compliance analyst (2-3 years' experience) at a mid-sized commercial bank, answering daily
enquiries from front-office teams and needing to locate regulatory text quickly.

## Input / Output
- Input: a natural-language compliance question.
- Output: a short answer with citations `[file.pdf, p.N]`, OR a refusal message.
- Every request is audit-logged.

## Architecture (text)
Question -> PII check on question -> embed query (API) -> cosine search over chunk index
-> threshold gate -> LLM answer from top-k chunks (API) -> citation check + PII check in code
-> answer with citations, OR refusal/block -> audit log.

Owned code: chunking, prompt, threshold gate, citation/PII guardrails, audit log, eval harness.
External intelligence: embedding model and chat model (see `src/config.py`).

## Metrics: targeted vs reached (final test set, n=27)
| Metric | Target | Reached |
|---|---|---|
| Faithfulness (hand-labelled, n=20) | high | 1.00 |
| Citation accuracy | high | 0.95 |
| Retrieval page hit@5 | high | 0.96 |
| Refusal accuracy (should-refuse) | high | 1.00 |
| False refusal rate | low | 0.13 |
| PII leakage | 0 | 0 |
| Cost per question | < S$0.01 | S$0.00028 |

Time-saved is NOT claimed: it was not measured with real analysts (listed as future work).

## Tuning summary (dev set)
- Retrieval: chunk_size 1000 + top_k 5 gave page_hit = 1.00 and lowest false-refusal (0.056).
- Threshold: 0.60 is the "sweet spot" - refusal accuracy 0.60 with zero false refusals.
- Single threshold cannot separate answerable vs unanswerable rows perfectly (0.636-0.670 overlap).
