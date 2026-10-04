"""pipeline.py - the end-to-end flow: question in, grounded answer (or refusal) out.

  question -> [PII in question?] -> retrieve top-k -> [best score < threshold?]
           -> generate -> [model abstained?] -> [PII in answer?]
           -> [citation check; retry once] -> answer  (always audit-logged)

status values:
  answered | refused_pii_input | refused_low_score | model_abstained |
  blocked_pii_output | blocked_no_citation | blocked_citation_not_in_retrieved
Anything other than "answered" returns a refusal message, never a raw LLM answer.
"""
import time
from dataclasses import asdict
from .config import Config, embed_cost, chat_cost
from .retrieve import retrieve
from .generate import generate_answer, ABSTAIN_TOKEN
from .guardrails import find_pii, check_citations
from .audit import log_event

MESSAGES = {
    "refused_pii_input": "Your question appears to contain personal data (PII). Please remove it and ask again.",
    "refused_low_score": "I cannot answer this from the regulatory corpus (no sufficiently relevant source found).",
    "model_abstained": "The retrieved sources do not contain enough information to answer this question.",
    "blocked_pii_output": "The draft answer was blocked because it contained possible personal data.",
    "blocked_no_citation": "The draft answer was blocked because it lacked a source citation.",
    "blocked_citation_not_in_retrieved": "The draft answer was blocked because it cited a source that was not retrieved.",
}


def answer_question(question, cfg=None, log=True):
    cfg = cfg or Config()
    t0 = time.time()
    rec = {"question": question, "status": None, "answer": "", "raw_answer": "", "cited": [],
           "hits": [], "top_score": None, "embed_tokens": 0, "in_tokens": 0, "out_tokens": 0,
           "retries": 0, "config": asdict(cfg)}

    def finish(status, answer):
        rec["status"], rec["answer"] = status, answer
        rec["cost_usd"] = embed_cost(rec["embed_tokens"]) + chat_cost(rec["in_tokens"], rec["out_tokens"])
        rec["latency_s"] = round(time.time() - t0, 3)
        if log:
            logged = dict(rec)
            if status == "refused_pii_input":
                logged["question"] = "[REDACTED: PII detected in question]"
            log_event(logged)
        return rec

    if find_pii(question):
        return finish("refused_pii_input", MESSAGES["refused_pii_input"])

    hits, et = retrieve(question, cfg)
    rec["hits"], rec["embed_tokens"], rec["top_score"] = hits, et, hits[0]["score"]
    if rec["top_score"] < cfg.sim_threshold:
        return finish("refused_low_score", MESSAGES["refused_low_score"])

    for attempt in range(cfg.max_retries + 1):
        text, i_tok, o_tok = generate_answer(question, hits, reminder=(attempt > 0))
        rec["in_tokens"] += i_tok
        rec["out_tokens"] += o_tok
        rec["retries"], rec["raw_answer"] = attempt, text
        if text.strip().startswith(ABSTAIN_TOKEN):
            return finish("model_abstained", MESSAGES["model_abstained"])
        if find_pii(text):
            return finish("blocked_pii_output", MESSAGES["blocked_pii_output"])
        ok, reason, cited = check_citations(text, hits)
        rec["cited"] = cited
        if ok:
            return finish("answered", text.strip())
    return finish("blocked_" + reason, MESSAGES["blocked_" + reason])
