"""config.py - central settings for the whole project.

Everything tunable lives here: model names, prices, chunking and retrieval
parameters, file paths. Other modules import from this file and never hard-code
numbers. Change a value here and the whole pipeline follows.

IMPORTANT: model names and prices below are PLACEHOLDERS. Check the current
OpenAI pricing page before you report any cost number, and write the date you
checked in your report.
"""
import os
from dataclasses import dataclass

# ---- paths (repo root = parent of src/) ------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF_DIR = os.path.join(ROOT, "data", "raw_pdfs")
INDEX_DIR = os.path.join(ROOT, "index")
LOG_DIR = os.path.join(ROOT, "logs")
RESULTS_DIR = os.path.join(ROOT, "evals", "results")

# ---- rented models (we only call APIs; no training) ---------------------------
EMBED_MODEL = "openai/text-embedding-3-small"
CHAT_MODEL = "openai/gpt-4o-mini"   # if you switch to a reasoning model, adapt llm.chat()

# ---- prices in USD per 1M tokens  (VERIFY on the official pricing page!) --------
PRICE_EMBED_PER_M = 0.02
PRICE_CHAT_IN_PER_M = 0.15
PRICE_CHAT_OUT_PER_M = 0.60
USD_TO_SGD = 1.30            # update to the rate on the day you write the report


@dataclass
class Config:
    """Tunable pipeline parameters (the knobs we tune on the dev set)."""
    chunk_size: int = 1000        # characters per chunk
    chunk_overlap: int = 150      # characters shared between neighbouring chunks
    top_k: int = 4                # chunks passed to the LLM
    sim_threshold: float = 0.30   # refuse if best cosine similarity < this
    max_retries: int = 1          # regenerate once if the citation check fails

    @property
    def tag(self) -> str:
        """Index folder name; chunking params decide which index is used."""
        return f"cs{self.chunk_size}_ov{self.chunk_overlap}"


def embed_cost(tokens: int) -> float:
    return tokens / 1e6 * PRICE_EMBED_PER_M


def chat_cost(in_tokens: int, out_tokens: int) -> float:
    return in_tokens / 1e6 * PRICE_CHAT_IN_PER_M + out_tokens / 1e6 * PRICE_CHAT_OUT_PER_M


_client = None


def get_client():
    """OpenAI-compatible client. Works with an OpenRouter key (secret OPENROUTER_API_KEY)
    or a native OpenAI key (secret OPENAI_API_KEY)."""
    global _client
    if _client is None:
        from openai import OpenAI
        key = None
        for name in ("OPENROUTER_API_KEY", "OPENAI_API_KEY"):
            key = os.environ.get(name)
            if not key:
                try:
                    from google.colab import userdata
                    key = userdata.get(name)
                except Exception:
                    key = None
            if key:
                break
        if not key:
            raise RuntimeError("No API key found. Add OPENROUTER_API_KEY (or OPENAI_API_KEY) in Colab Secrets.")
        base = "https://openrouter.ai/api/v1" if key.startswith("sk-or-") else None
        _client = OpenAI(api_key=key, base_url=base)
    return _client
