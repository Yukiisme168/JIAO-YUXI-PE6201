"""guardrails.py - deterministic, code-based safety rules (NO LLM involved).

Rule 1  citation check : answer must contain >=1 citation of the form
                         [file.pdf, p.N], and every cited (file, page) must be
                         among the chunks that were actually retrieved.
Rule 2  PII detection  : regexes for email, SG NRIC/FIN, SG phone, CN national
                         ID, and card numbers (Luhn-validated). Applied to the
                         user question (refuse before calling the API) and to
                         the generated answer (block).
These run regardless of what the model says. Known limitation: regex PII has
false positives (e.g. a public contact e-mail printed in a circular) and false
negatives (names, addresses). State this in the report.
"""
import re

CITATION_RE = re.compile(r"\[([^\[\]]+?),\s*p\.?\s*(\d+)\]", re.I)

PII_PATTERNS = {
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    "sg_nric_fin": re.compile(r"\b[STFGM]\d{7}[A-Z]\b", re.I),
    "sg_phone": re.compile(r"(?<!\d)(?:\+65[\s-]?)?[3689]\d{3}[\s-]?\d{4}(?!\d)"),
    "cn_national_id": re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)"),
}
CARD_CANDIDATE_RE = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")


def _luhn_ok(digits):
    total, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d *= 2
            if d > 9:
                d -= 9
        total += d
        alt = not alt
    return total % 10 == 0


def find_pii(text):
    """Return a list of (type, matched_text). Empty list = clean."""
    found = []
    for name, pat in PII_PATTERNS.items():
        for m in pat.finditer(text):
            found.append((name, m.group(0)))
    for m in CARD_CANDIDATE_RE.finditer(text):
        digits = re.sub(r"\D", "", m.group(0))
        if 13 <= len(digits) <= 19 and _luhn_ok(digits):
            found.append(("card_number", m.group(0)))
    return found


def extract_citations(answer):
    """Return [(file_lowercase, page_int)] cited in the answer."""
    return [(f.strip().lower(), int(p)) for f, p in CITATION_RE.findall(answer)]


def check_citations(answer, hits):
    """Return (ok, reason, cited). reason in {ok, no_citation, citation_not_in_retrieved}."""
    cited = extract_citations(answer)
    if not cited:
        return False, "no_citation", cited
    allowed = {(h["doc"].lower(), h["page"]) for h in hits}
    if any(c not in allowed for c in cited):
        return False, "citation_not_in_retrieved", cited
    return True, "ok", cited
