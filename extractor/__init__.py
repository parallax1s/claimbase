# Vendored and condensed from Episteme (MIT, github.com/parallax1s/Episteme).
"""claimbase.extractor — minimal, zero-dependency claim extraction engine.

Public API
----------
extract_claims(text, source_format="text") -> list[dict]
    Each dict: {"text", "type", "support_in_text", "quote"}

embed_texts(texts) -> list | None
    L2-normalised embeddings via model2vec, or None if unavailable.

jaccard(a, b) -> float
    Content-token Jaccard similarity.
"""

from extractor.engine import extract_claims, embed_texts, jaccard  # noqa: F401

__all__ = ["extract_claims", "embed_texts", "jaccard"]
