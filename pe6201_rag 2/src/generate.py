"""generate.py - prompt template and the LLM call that writes the answer.

The prompt is tuned for compliance questions: answer ONLY from the supplied
excerpts, cite every statement as [file.pdf, p.N], and output the sentinel
INSUFFICIENT_CONTEXT when the excerpts do not contain the answer.
NOTE: the prompt only *asks* for citations. Enforcement is done in code by
guardrails.py, so safety does not depend on the model obeying the prompt.
"""
from .llm import chat

ABSTAIN_TOKEN = "INSUFFICIENT_CONTEXT"

SYSTEM_PROMPT = (
    "You are a regulatory Q&A assistant for bank compliance analysts. "
    "Answer ONLY using the numbered context excerpts supplied by the user message.\n"
    "Rules:\n"
    "1. Every factual statement must end with a citation in EXACTLY this format: "
    "[filename.pdf, p.N] using the file name and page shown on the excerpt header.\n"
    "2. If the excerpts do not contain the answer, reply with exactly: " + ABSTAIN_TOKEN + "\n"
    "3. Never use outside knowledge. Never give formal legal advice.\n"
    "4. Be concise: at most about 150 words. Do not repeat personal data.\n"
    "5. If an excerpt comes from a document whose file name contains 'Consultation', state clearly that the position is only PROPOSED and not yet in force."
)

REMINDER = ("\n\nREMINDER: your previous answer broke the citation rules. Cite every statement as "
            "[filename.pdf, p.N] using ONLY files/pages that appear in the excerpt headers.")


def build_context(hits):
    return "\n\n".join(f"[{h['doc']}, p.{h['page']}]\n{h['text']}" for h in hits)


def generate_answer(question, hits, reminder=False):
    """Return (text, prompt_tokens, completion_tokens)."""
    user = f"Context excerpts:\n\n{build_context(hits)}\n\nQuestion: {question}"
    if reminder:
        user += REMINDER
    return chat([{"role": "system", "content": SYSTEM_PROMPT},
                 {"role": "user", "content": user}])
