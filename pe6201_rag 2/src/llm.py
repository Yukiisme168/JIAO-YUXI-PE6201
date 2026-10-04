"""llm.py - the ONLY place that talks to the rented OpenAI API.

Two functions: embed_texts() and chat(). Both retry on transient errors and
return token usage so that every call can be costed (see audit.py / pipeline.py).
Keeping API calls in one file makes the "build vs rent" boundary explicit:
everything else in the repo is code we own.
"""
import time
import numpy as np
from .config import get_client, EMBED_MODEL, CHAT_MODEL


def _retry(fn, attempts=4):
    for i in range(attempts):
        try:
            return fn()
        except Exception:
            if i == attempts - 1:
                raise
            time.sleep(2 ** i)


def embed_texts(texts):
    """Embed a list of strings. Returns (float32 array [n, dim], total_tokens)."""
    client = get_client()
    vectors, tokens = [], 0
    for i in range(0, len(texts), 100):
        batch = texts[i:i + 100]
        r = _retry(lambda: client.embeddings.create(model=EMBED_MODEL, input=batch))
        vectors.extend(d.embedding for d in r.data)
        tokens += r.usage.total_tokens
    return np.array(vectors, dtype="float32"), tokens


def chat(messages, temperature=0.0, max_tokens=500):
    """One chat completion. Returns (text, prompt_tokens, completion_tokens)."""
    client = get_client()
    r = _retry(lambda: client.chat.completions.create(
        model=CHAT_MODEL, messages=messages, temperature=temperature, max_tokens=max_tokens))
    return r.choices[0].message.content or "", r.usage.prompt_tokens, r.usage.completion_tokens
