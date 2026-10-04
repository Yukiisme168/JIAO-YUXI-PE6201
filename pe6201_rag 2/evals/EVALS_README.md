# Evals

## Questions
- `questions_50.csv`: 50 hand-written questions with answer key (gold_answer, gold_sources, should_refuse).
- Type mix: single 33, multi 8, out_of_corpus 5, trap 3, pii 4.
- Split: dev 23 / test 27. Each type appears in both splits.
- Gold labels were written by hand from the source PDFs (no model involved).

## Automatic metrics (`run_evals.py`)
- retrieval_doc_hit / retrieval_page_hit : was a gold doc / (doc,page) in the top-k?
- citation_present   : every answered response has >=1 citation (L1 assertion)
- citation_correct   : a cited (doc,page) matches the gold sources
- refusal_accuracy   : should_refuse rows that did NOT get an answer
- false_refusal_rate : answerable rows that did NOT get an answer
- pii_leak           : answered responses containing PII (should be 0)
- cost / latency     : from token usage and wall-clock time

No LLM-as-judge is used.

## Faithfulness (hand-labelled)
- Every answered test question was labelled by hand in results/final_test_to_label.csv.
- Rubric: 1 = every claim is supported by the cited text; 0 = otherwise (with error_type).
- Result: 20 / 20 answered test questions labelled faithful (faithfulness = 1.00).
- Some answers were incomplete (noted in the notes column), but none contained hallucinated claims.
- Questions were drafted with AI assistance; all gold answers and page references were manually verified against the source PDFs.

## Protocol
Tune on dev only; run test once for the final numbers.

## Files in results/
- final_test_results.csv : per-question outputs of the final test run
- final_test_to_label.csv : hand-labelled faithfulness file
- tuning_chunk_topk.csv  : chunk_size x top_k sweep on dev
- tuning_threshold.csv   : similarity threshold sweep on dev
- tuning_threshold_final.csv : threshold sweep under final config
