# Vendored and condensed from Episteme (MIT, github.com/parallax1s/Episteme).
"""
Minimal, zero-dependency claim extraction engine vendored from Episteme.

Sources used (all MIT):
  episteme/ingest.py           — text normalisation
  episteme/segment.py          — sentence splitting, genre detection
  episteme/lenses/epistemic.py — epistemic claim heuristics
  episteme/consolidate.py      — _content_toks / _jac, _embed

Only the plain-text / markdown / mdx normalisation path, the abbreviation-aware
sentence splitter, and the EpistemicLens heuristics are included.  All pydantic
models have been replaced with dataclasses or plain dicts.  Dependencies: stdlib
only (model2vec is optional).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

__all__ = ["extract_claims", "embed_texts", "jaccard"]

# ===========================================================================
# 1. Ingest / normalisation  (vendored from episteme/ingest.py)
# ===========================================================================

MARKDOWN_FORMATS = {"md", "markdown", "rst", "mdx"}

_FENCED_CODE_RE = re.compile(r"```.*?```", re.S)
_MATH_BLOCK_RE = re.compile(r"\$\$.*?\$\$", re.S)
_MATH_INLINE_RE = re.compile(r"(?<![\\$])\$[^$\n]+\$(?!\$)")
_CURRENCY_CONTENT_RE = re.compile(r"\d[\d,.]*\s")
_INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")
_TABLE_ROW_RE = re.compile(r"^[ \t]*\|.*\|[ \t]*$\n?", re.M)
_SUP_CITATION_RE = re.compile(r"[ \t]*<sup\b[^>]*>(?P<body>.*?)</sup>", re.I | re.S)
_SUP_ANCHOR_NUM_RE = re.compile(r"<a\b[^>]*>\s*\[?\s*(\d+)\s*\]?\s*</a>", re.I)
_FOOTNOTE_REF_RE = re.compile(r"[ \t]*\[\^([^\]\s]+)\]")
_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_BOLD_STARS_RE = re.compile(r"\*\*([^*\n]+)\*\*")
_ITALIC_STAR_RE = re.compile(r"\*([^*\n]+)\*")
_BOLD_UNDERSCORES_RE = re.compile(r"(?<![\w_])__([^_\n]+)__(?![\w_])")
_ITALIC_UNDERSCORE_RE = re.compile(r"(?<![\w_])_([^_\n]+)_(?![\w_])")
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.S)
_HTML_BREAK_RE = re.compile(r"<br\s*/?>", re.I)
_HTML_TAG_RE = re.compile(r"</?[A-Za-z][^>]*>")
_PLACEHOLDER_RE = re.compile(r"\x00(\d+)\x00")

_MDX_IMPORT_LINE_RE = re.compile(
    r"^\s*import\s+(?:type\s+)?(?:[\w$*\s{},]+\s+from\s+)?[\"'][^\"']+[\"']\s*;?\s*$"
)
_MDX_IMPORT_OPEN_RE = re.compile(r"^\s*import\s+(?:type\s+)?[\w$*\s,]*\{[^}]*$")
_MDX_EXPORT_RE = re.compile(r"^\s*export\s+(?:const|let|var)\s+\w")
_MDX_SELF_CLOSING_RE = re.compile(r"^[ \t]*<[A-Z][\w.]*(?:\s[^>]*?)?/>[ \t]*$", re.M)
_MDX_PAIRED_RE = re.compile(
    r"^[ \t]*<([A-Z][\w.]*)(?:\s[^>]*?)?>[ \t]*\n?(?P<body>.*?)\n?[ \t]*</\1\s*>[ \t]*$",
    re.M | re.S,
)


def _sup_citation_marker(match: re.Match) -> str:
    body = match.group("body")
    numbers = _SUP_ANCHOR_NUM_RE.findall(body)
    if not numbers and re.fullmatch(r"\s*\[[\d,\s]+\]\s*", _HTML_TAG_RE.sub("", body)):
        numbers = re.findall(r"\d+", body)
    if not numbers:
        return match.group(0)
    return "[" + ", ".join(numbers) + "]"


def _strip_mdx_statements(text: str) -> str:
    lines = text.split("\n")
    kept: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if _MDX_IMPORT_LINE_RE.match(line):
            i += 1
            continue
        if _MDX_IMPORT_OPEN_RE.match(line):
            i += 1
            while i < len(lines):
                tail = lines[i].rstrip()
                i += 1
                if not tail or tail.endswith((";", '"', "'")):
                    break
            continue
        if _MDX_EXPORT_RE.match(line):
            balance = line.count("{") - line.count("}")
            i += 1
            while balance > 0 and i < len(lines):
                balance += lines[i].count("{") - lines[i].count("}")
                i += 1
            continue
        kept.append(line)
        i += 1
    return "\n".join(kept)


def _strip_mdx_constructs(text: str) -> str:
    protected: list[str] = []

    def _protect(raw: str) -> str:
        protected.append(raw)
        return f"\x00{len(protected) - 1}\x00"

    for pattern in (_FENCED_CODE_RE, _MATH_BLOCK_RE):
        text = pattern.sub(lambda m: _protect(m.group(0)), text)
    text = _strip_mdx_statements(text)
    while True:
        unwrapped = _MDX_PAIRED_RE.sub(lambda m: m.group("body"), text)
        if unwrapped == text:
            break
        text = unwrapped
    text = _MDX_SELF_CLOSING_RE.sub("", text)
    return _PLACEHOLDER_RE.sub(lambda m: protected[int(m.group(1))], text)


def _normalize_markdown_inline(text: str) -> str:
    protected: list[str] = []

    def _protect(raw: str) -> str:
        protected.append(raw)
        return f"\x00{len(protected) - 1}\x00"

    for pattern in (_FENCED_CODE_RE, _MATH_BLOCK_RE):
        text = pattern.sub(lambda m: _protect(m.group(0)), text)

    def _protect_math(m: re.Match) -> str:
        if _CURRENCY_CONTENT_RE.match(m.group(0)[1:-1]):
            return m.group(0)
        return _protect(m.group(0))

    text = _MATH_INLINE_RE.sub(_protect_math, text)
    text = _INLINE_CODE_RE.sub(lambda m: _protect(m.group(1)), text)
    text = _TABLE_ROW_RE.sub("", text)
    text = _SUP_CITATION_RE.sub(_sup_citation_marker, text)
    text = _FOOTNOTE_REF_RE.sub(r"[\1]", text)
    text = _IMAGE_RE.sub(r"\1", text)
    text = _LINK_RE.sub(r"\1", text)
    for pattern in (_BOLD_STARS_RE, _ITALIC_STAR_RE, _BOLD_UNDERSCORES_RE, _ITALIC_UNDERSCORE_RE):
        text = pattern.sub(r"\1", text)
    text = _HTML_COMMENT_RE.sub("", text)
    text = _HTML_BREAK_RE.sub("\n", text)
    text = _HTML_TAG_RE.sub("", text)
    return _PLACEHOLDER_RE.sub(lambda m: protected[int(m.group(1))], text)


def _normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\t\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _ingest(text: str, source_format: str = "text") -> str:
    """Normalise *text* for the given *source_format* and return clean plain text."""
    if source_format == "mdx":
        text = _strip_mdx_constructs(text)
    if source_format in MARKDOWN_FORMATS:
        text = _normalize_markdown_inline(text)
    return _normalize_text(text)


# ===========================================================================
# 2. Sentence splitter  (vendored from episteme/segment.py)
# ===========================================================================

_CITATION_MARKER = r"\[\d+(?:\s*,\s*\d+)*\]"
_SENTENCE_END_RE = re.compile(
    rf"(?<=[.!?])(?P<citation>{_CITATION_MARKER})?\s+(?=[\"'“‘(]*[A-Z0-9])"
)
_CITATION_MARKER_RE = re.compile(_CITATION_MARKER)

# Periods after these tokens are abbreviations, not sentence boundaries.
_ABBREVIATIONS = (
    # English
    "Dr", "Mr", "Mrs", "Ms", "Prof", "St", "vs", "etc", "e.g", "i.e", "cf",
    "al", "Fig", "No", "Jr", "Sr", "Inc", "Ltd", "U.S", "U.K", "Ph.D",
    # German
    "z.B", "bzw", "ca", "Nr", "usw", "d.h", "u.a", "vgl", "Abs", "S", "Bd",
    "Hrsg", "Jh",
)
_ABBREVIATION_ALTERNATION = "|".join(
    re.escape(abbr) for abbr in sorted(_ABBREVIATIONS, key=len, reverse=True)
)
_NO_SPLIT_PERIOD_RE = re.compile(
    rf"(?<![\w.])(?:{_ABBREVIATION_ALTERNATION}|\d{{1,2}}|[A-Z])\.$"
)
_NO_SPLIT_LOOKBACK = 16

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")
_PAGE_MARKER_RE = re.compile(r"^\[Page \d+\]$")
_MD_HEADER_RE = re.compile(r"^#{1,6}\s+\S", re.M)
_ATTRIBUTION_VERB_RE = re.compile(
    r"\b(said|asked|replied|whispered|muttered|shouted|exclaimed|answered)\b"
)


def _is_abbreviation_period(paragraph: str, boundary: int) -> bool:
    if paragraph[boundary - 1] != ".":
        return False
    start = max(0, boundary - _NO_SPLIT_LOOKBACK)
    return _NO_SPLIT_PERIOD_RE.search(paragraph, start, boundary) is not None


def split_sentences(paragraph: str, base_start: int = 0) -> list[tuple[str, int, int]]:
    """Split *paragraph* into ``(text, start, end)`` sentence spans."""
    if not paragraph.strip():
        return []
    pieces: list[tuple[str, int, int]] = []
    cursor = 0
    for match in _SENTENCE_END_RE.finditer(paragraph):
        if _is_abbreviation_period(paragraph, match.start()):
            continue
        end = match.end("citation") if match.group("citation") else match.start()
        raw = paragraph[cursor:end].strip()
        if raw:
            leading = len(paragraph[cursor:end]) - len(paragraph[cursor:end].lstrip())
            s = base_start + cursor + leading
            e = base_start + end
            pieces.append((raw, s, e))
        cursor = match.end()
    tail = paragraph[cursor:].strip()
    if tail:
        leading = len(paragraph[cursor:]) - len(paragraph[cursor:].lstrip())
        pieces.append((tail, base_start + cursor + leading, base_start + len(paragraph)))
    return pieces


# ===========================================================================
# 3. Genre detection  (vendored from episteme/segment.py)
# ===========================================================================

def _guess_genre(text: str) -> str:
    """Return 'nonfiction', 'fiction', 'mixed', or 'unknown'."""
    lowered = text.lower()
    fiction_cues = len(re.findall(
        r"\b(he|she|they|her|him|his|said|asked|replied|whispered|thought|remembered|felt|"
        r"knew|understood|doubted|suspected|looked|opened|closed|took|found|lit|"
        r"slipped|tucked|checked|pressed|lifted|turned|paused|laughed|failed|"
        r"raised|reflected|answered|smiled|"
        r"door|room|coat|drawer|key|kitchen|lights|mirror|figure|window|"
        r"latch|lantern|coin|signal|reflection|letter|envelope|skylight|"
        r"cellar|hatch|chalk|mark|wall|radio|dial|static|voice|speaker|"
        r"ink|message|rewrote|name|basement|freezer|drain|clock|elevator|"
        r"hallway|bucket|rain|sky|water|orchard|trees|attic|map|paper|unfolded|"
        r"behind|hand|mouth)\b",
        lowered,
    ))
    nonfiction_cues = len(re.findall(
        r"\b(policy|study|studies|evidence|data|therefore|thus|hence|because|however|"
        r"moreover|furthermore|consequently|report|claim|claims|argument|premise|"
        r"conclusion|research|analysis|percent|%)\b",
        lowered,
    ))
    markdown_headers = len(_MD_HEADER_RE.findall(text))
    citation_markers = len(_CITATION_MARKER_RE.findall(text))
    structure_cues = 2 * markdown_headers + citation_markers
    nonfiction_signal = nonfiction_cues + structure_cues

    paragraphs = [p for p in re.split(r"\n\s*\n", lowered) if p.strip()]
    dialogue_paragraphs = sum(
        1 for p in paragraphs
        if (p.count('"') + p.count("“") + p.count("”")) >= 2
        and _ATTRIBUTION_VERB_RE.search(p)
    )
    dialogue_density = dialogue_paragraphs / max(1, len(paragraphs))
    dialogue_dashes = len(re.findall(r"(?:^|\n)\s*[—–]\s+", text))
    narrated_dialogue = (
        dialogue_paragraphs >= 2 and dialogue_density >= 0.3
    ) or dialogue_dashes >= 4

    if structure_cues >= 3 and not narrated_dialogue and fiction_cues < 3:
        return "nonfiction"
    if narrated_dialogue and fiction_cues >= nonfiction_signal:
        return "fiction"
    if fiction_cues >= 3 and nonfiction_signal == 0:
        return "fiction"
    if nonfiction_signal >= max(3, fiction_cues + 2):
        return "nonfiction"
    if fiction_cues and nonfiction_signal:
        return "mixed"
    return "unknown"


# ===========================================================================
# 4. Paragraph block segmentation  (vendored from episteme/segment.py)
# ===========================================================================

def _paragraph_blocks(text: str) -> list[tuple[str, int, int]]:
    blocks: list[tuple[str, int, int]] = []
    for match in re.finditer(r"\S(?:.*?\S)?(?=\n\s*\n|\Z)", text, flags=re.S):
        raw = match.group(0)
        start, end = match.span()
        block = raw.strip()
        leading = len(raw) - len(raw.lstrip())
        trailing = len(raw.rstrip())
        blocks.append((block, start + leading, start + trailing))
    return blocks


def _is_heading(block: str) -> bool:
    first_line = block.strip().splitlines()[0] if block.strip() else ""
    if _HEADING_RE.match(first_line):
        return True
    if _PAGE_MARKER_RE.match(first_line):
        return True
    if len(first_line) <= 80 and "\n" not in block and not re.search(r"[.!?]$", first_line):
        alpha = sum(ch.isalpha() for ch in first_line)
        if alpha >= 3 and (first_line.istitle() or first_line.isupper()):
            return True
    return False


# ===========================================================================
# 5. Sentence dataclass
# ===========================================================================

@dataclass(slots=True)
class _Sentence:
    text: str
    span_start: int
    span_end: int
    id: str


# ===========================================================================
# 6. Epistemic lens heuristics  (vendored from episteme/lenses/epistemic.py)
# ===========================================================================

_CLAIM_VERBS = re.compile(
    r"\b(is|are|was|were|has|have|had|will|would|can|could|does|do|did|"
    r"says?|said|states?|stated|reports?|reported|causes?|caused|fails?|failed|"
    r"declines?|declined|fell|falls|rose|rises|raises?|raised|grew|grows|"
    r"ends?|ended|retires?|retired|releases?|released|spends?|spent|refuses?|refused|"
    r"plays?|played|stars?|starred|appears?|appeared|works?|worked|competes?|competed|"
    r"participates?|participated|forces?|forced|makes?|made|drinks?|drank|"
    r"worsens?|worsened|shortens?|shortened|cuts?|cut|passes?|passed|"
    r"finishes?|finished|reduces?|reduced|increases?|increased|improves?|improved|"
    r"shows?|showed|proves?|proved|suggests?|solves?|solved|"
    r"contributes?|contributed|counteracts?|counteracted|occurs?|occurred|"
    r"activates?|activated|links?|linked|alters?|altered|mediates?|mediated|"
    r"diminishes?|diminished|prevents?|prevented|suppresses?|suppressed|"
    r"took place|takes? place|put out)\b",
    re.I,
)
_NUMBER = re.compile(
    r"(\d+(?:\.\d+)?\s?(?:%|percent|million|billion|thousand)?"
    r"|\b(?:one|two|three|four|five|ten|hundred)\b)",
    re.I,
)
_QUANT_NUMBER = re.compile(
    r"(\d+(?:\.\d+)?\s?(?:%|percent|million|billion|thousand)?"
    r"|\b(?:one|two|three|four|five|ten)\s+(?:hundred|thousand|million|billion|percent)\b)",
    re.I,
)
_CAUSAL = re.compile(
    r"\b(because|therefore|as a result|causes?|caused|led to|due to|drives?|results? in)\b",
    re.I,
)
_HEDGE = re.compile(
    r"\b(may|might|could|possibly|probably|likely|suggests?|appears|seems|"
    r"roughly|approximately)\b",
    re.I,
)
_ABSOLUTE = re.compile(
    r"\b(always|never|all|none|every|guarantees?|proves?|proved|certainly|impossible)\b",
    re.I,
)
_SOURCE = re.compile(
    r"\b(according to|study|paper|report|dataset|survey|citation|source|evidence|"
    r"table|figure)\b",
    re.I,
)
_NORMATIVE = re.compile(
    r"\b(should|ought|must|need to|ethical|unfair|better|worse|good|bad)\b",
    re.I,
)
_PREDICTIVE = re.compile(
    r"\b(will|is expected to|forecast|predict|by \d{4}|next year|future)\b",
    re.I,
)
_LEGAL_MED_FIN = re.compile(
    r"\b(illegal|legal|law|contract|lease|clause|court|statute|compliant|"
    r"medical|treatment|diagnosis|drug|vaccine|patient|clinical|clinic|sterile|"
    r"investment|stock|tax|financial)\b",
    re.I,
)
_CITATION_PAT = re.compile(r"\[\d+(?:\s*,\s*\d+)*\]")
_URL_PAT = re.compile(r"https?://|www\.", re.I)
_QUOTED_SPAN = re.compile(r'["“][^"“”]+["”]')
_ATTRIBUTION = re.compile(
    r"\b(said|says|argued|argues|claim(?:s|ed)?|wrote|writes|states?|stated|"
    r"reports?|reported|told|asserts?|asserted|insists?|insisted|"
    r"maintains?|maintained|contends?|contended|according to)\b",
    re.I,
)
_REFUTING = re.compile(
    r"\b(wrongly|falsely|incorrectly|mistakenly|debunked|erroneously|"
    r"disproven|disproved|refuted|untrue)\b",
    re.I,
)
_DIGIT = re.compile(r"\d")

_ANAPHORA_SET = frozenset({"it", "this", "that", "these", "those", "which", "they"})
_IMPERATIVE_VERBS = frozenset({
    "build", "make", "consider", "think", "do", "use", "start", "stop",
    "take", "let", "imagine", "remember", "note", "keep", "try", "ask",
    "avoid", "ensure", "create", "choose", "follow", "apply", "define",
    "describe", "explain", "suppose", "assume", "beware", "behold",
    "picture", "write",
})
_CONJUNCT_VERB_HEADS = frozenset({
    "say", "says", "argue", "argues", "show", "shows", "note", "notes",
    "add", "adds", "explain", "explains", "suggest", "suggests",
    "claim", "claims", "push", "pushes", "lean", "leans", "rate", "rates",
    "see", "sees", "call", "calls",
})
_ALARM_FLAGS = frozenset({
    "unsupported_major_claim",
    "overclaiming",
    "causal_inference_needs_evidence",
    "normative_warrant_needs_support",
})

# Claim type constants (matching SCHEMA.md)
_TYPE_NORMATIVE = "normative"
_TYPE_DEFINITIONAL = "definitional"
_TYPE_CAUSAL = "causal"
_TYPE_PREDICTIVE = "predictive"
_TYPE_STATISTICAL = "statistical"
_TYPE_LEGAL = "legal"
_TYPE_MEDICAL = "medical"
_TYPE_FINANCIAL = "financial"
_TYPE_HISTORICAL = "historical"
_TYPE_EMPIRICAL = "empirical"
_TYPE_DESCRIPTIVE = "descriptive"
_TYPE_OTHER = "other"

_NEEDS_VERIFICATION = {
    _TYPE_EMPIRICAL, _TYPE_STATISTICAL, _TYPE_CAUSAL, _TYPE_PREDICTIVE,
    _TYPE_HISTORICAL, _TYPE_LEGAL, _TYPE_MEDICAL, _TYPE_FINANCIAL,
}


def _strip_citations(text: str) -> str:
    return _CITATION_PAT.sub(" ", text)


def _has_number(text: str) -> bool:
    return bool(_NUMBER.search(_strip_citations(text)))


def _has_quant_number(text: str) -> bool:
    return bool(_QUANT_NUMBER.search(_strip_citations(text)))


def _claim_type(text: str) -> str:
    if _NORMATIVE.search(text):
        return _TYPE_NORMATIVE
    if re.search(r"\b(means|defined as|refers to)\b", text, re.I):
        return _TYPE_DEFINITIONAL
    if _CAUSAL.search(text):
        return _TYPE_CAUSAL
    if _PREDICTIVE.search(text):
        return _TYPE_PREDICTIVE
    if _has_quant_number(text):
        return _TYPE_STATISTICAL
    if _LEGAL_MED_FIN.search(text):
        lower = text.lower()
        if any(
            w in lower
            for w in ["medical", "treatment", "diagnosis", "drug", "vaccine",
                      "patient", "clinical", "clinic", "sterile"]
        ):
            return _TYPE_MEDICAL
        if any(w in lower for w in ["investment", "stock", "tax", "financial"]):
            return _TYPE_FINANCIAL
        return _TYPE_LEGAL
    if re.search(r"\b(in \d{3,4}|during|historically|century|war|election)\b", text, re.I):
        return _TYPE_HISTORICAL
    if _CLAIM_VERBS.search(text):
        return (
            _TYPE_EMPIRICAL
            if re.search(
                r"\b(real|world|people|system|policy|market|emissions|health|app|"
                r"worker|workers|productivity|inspection|apartment|apartments|safe|safety)\b",
                text, re.I,
            )
            else _TYPE_DESCRIPTIVE
        )
    return _TYPE_OTHER


def _support_in_text(claim_text: str, sentence_text: str) -> float:
    score = 0.25
    if _SOURCE.search(sentence_text):
        score += 0.25
    if _CITATION_PAT.search(sentence_text) or _URL_PAT.search(sentence_text):
        score += 0.3
    if _has_number(sentence_text):
        score += 0.1
    if _HEDGE.search(sentence_text):
        score += 0.07
    if _ABSOLUTE.search(claim_text) and not _SOURCE.search(sentence_text):
        score -= 0.12
    return max(0.0, min(1.0, score))


def _risk_flags(text: str, ctype: str, support: float) -> list[str]:
    flags: list[str] = []
    if support < 0.35 and ctype in {_TYPE_CAUSAL, _TYPE_STATISTICAL, _TYPE_EMPIRICAL}:
        flags.append("unsupported_major_claim")
    if ctype == _TYPE_NORMATIVE and _CAUSAL.search(text) and support < 0.35:
        flags.append("normative_warrant_needs_support")
    if _ABSOLUTE.search(text) and support < 0.55:
        flags.append("overclaiming")
    if ctype in {_TYPE_LEGAL, _TYPE_MEDICAL, _TYPE_FINANCIAL} or _LEGAL_MED_FIN.search(text):
        flags.append("high_stakes_domain")
    if ctype == _TYPE_CAUSAL and support < 0.5:
        flags.append("causal_inference_needs_evidence")
    return flags


def _build_reason(ctype: str, support: float, verification: bool, notes: list[str]) -> str:
    support_label = "strong" if support >= 0.7 else "moderate" if support >= 0.45 else "weak"
    verify = " It should be externally verified." if verification else ""
    reason = f"Detected a {ctype} claim with {support_label} internal support.{verify}"
    if notes:
        reason = f"{reason} {' '.join(notes)}"
    return reason


def _is_question(text: str) -> bool:
    # strip citations and trailing quote/bracket chars before checking
    tail = _strip_citations(text).rstrip(" \t\r\n\"“”‘’')]")
    return tail.endswith("?") or tail.endswith("?!") or tail.endswith("!?")


def _is_imperative(text: str) -> bool:
    stripped = text.lstrip(" \t\"“”‘’(")
    words = re.findall(r"[A-Za-z']+", stripped[:60])
    if not words:
        return False
    head = words[0].lower()
    if head in _IMPERATIVE_VERBS:
        return len(words) < 2 or words[1].lower() != "of"
    return False


def _is_narrative_description(sentence_text: str) -> bool:
    return not (
        _DIGIT.search(sentence_text)
        or _CITATION_PAT.search(sentence_text)
        or _URL_PAT.search(sentence_text)
        or _QUOTED_SPAN.search(sentence_text)
    )


def _attribution_info(claim_text: str, sentence_text: str) -> tuple[bool, bool]:
    attributed = bool(
        _QUOTED_SPAN.search(sentence_text) and _ATTRIBUTION.search(sentence_text)
    )
    if attributed:
        quote_in_claim = any(q in claim_text for q in ('"', "“", "”"))
        covers_sentence = len(claim_text) >= 0.8 * len(sentence_text)
        attributed = quote_in_claim or covers_sentence
    refuted = attributed and bool(_REFUTING.search(sentence_text))
    return attributed, refuted


def _is_independent_clause(segment: str, head: str) -> bool:
    if head in _CONJUNCT_VERB_HEADS or head in _IMPERATIVE_VERBS:
        return False
    return bool(
        _CLAIM_VERBS.search(segment)
        or _CAUSAL.search(segment)
        or _has_number(segment)
        or _ABSOLUTE.search(segment)
        or _NORMATIVE.search(segment)
        or re.search(r"\b(means|defined as|refers to)\b", segment, re.I)
    )


def _split_claims(sentence: str) -> list[str]:
    """Clause-aware split of one sentence into claim fragments."""
    if not (
        _CLAIM_VERBS.search(sentence)
        or _CAUSAL.search(sentence)
        or _has_number(sentence)
        or _ABSOLUTE.search(sentence)
        or _NORMATIVE.search(sentence)
        or re.search(r"\b(means|defined as|refers to)\b", sentence, re.I)
    ):
        return []
    tokens = re.split(r"(\s*(?:;|,\s+and\s+|\s+and\s+|,\s+but\s+|\s+but\s+)\s*)", sentence)
    segments: list[str] = []
    for i in range(0, len(tokens), 2):
        seg = tokens[i].strip()
        if not seg:
            continue
        delim = tokens[i - 1] if i > 0 else ""
        head_match = re.match(r"[A-Za-z']+", seg)
        head = head_match.group(0).lower() if head_match else ""
        # Anaphoric / too-short / bare-verb-conjunct fragments are re-merged.
        if segments and (
            head in _ANAPHORA_SET
            or len(seg.split()) < 4
            or not _is_independent_clause(seg, head)
        ):
            segments[-1] = f"{segments[-1]}{delim}{seg}"
        else:
            segments.append(seg)
    claims = []
    for part in segments:
        part = part.strip()
        word_count = len(part.split())
        has_claim_verb = bool(_CLAIM_VERBS.search(part))
        if word_count >= (3 if has_claim_verb else 4) and (
            _CLAIM_VERBS.search(part)
            or _CAUSAL.search(part)
            or _has_number(part)
            or _ABSOLUTE.search(part)
            or _NORMATIVE.search(part)
            or re.search(r"\b(means|defined as|refers to)\b", part, re.I)
        ):
            claims.append(part)
    return claims[:4]


def _extract_from_sentence(sentence: _Sentence, fiction: bool) -> list[dict[str, Any]]:
    """Return zero or more claim dicts extracted from one sentence."""
    text = sentence.text

    if _is_question(text):
        return []

    imperative = _is_imperative(text)
    normative_marker = bool(_NORMATIVE.search(text))
    if imperative and not normative_marker:
        return []

    claim_texts = _split_claims(text)
    if not claim_texts and _CLAIM_VERBS.search(text):
        claim_texts = [text]
    if not claim_texts and imperative and normative_marker:
        claim_texts = [text]

    results: list[dict[str, Any]] = []
    for claim_text in claim_texts:
        ctype = _claim_type(claim_text)
        if imperative:
            ctype = _TYPE_NORMATIVE

        support = _support_in_text(claim_text, text)
        verification = ctype in _NEEDS_VERIFICATION
        risks = _risk_flags(claim_text, ctype, support)
        notes: list[str] = []

        if imperative:
            notes.append(
                "Phrased in the imperative mood, so it is treated as a "
                "normative directive rather than a factual assertion."
            )

        attributed, refuted = _attribution_info(claim_text, text)
        if attributed:
            risks.append("attributed_quote")
            notes.append(
                "This restates attributed/quoted speech, not the document's own assertion."
            )
        if refuted:
            risks.append("refuted_in_text")
            notes.append(
                "The surrounding text marks the quoted claim as false, "
                "so the document asserts its negation."
            )

        if _CITATION_PAT.search(text) or _URL_PAT.search(text):
            notes.append(
                "The sentence carries an inline citation or link, "
                "raising its in-text support."
            )

        if fiction and _is_narrative_description(text):
            kept = [f for f in risks if f not in _ALARM_FLAGS]
            if len(kept) != len(risks) or verification:
                notes.append(
                    "Fiction narration: alarm-class risk flags and "
                    "external verification are suppressed for scene-setting description."
                )
            risks = kept
            verification = False

        results.append({
            "text": claim_text.strip(),
            "type": ctype,
            "support_in_text": round(support, 4),
            "quote": text[:300],
            # Internal fields available to tests.
            "_risk_flags": risks,
            "_reason": _build_reason(ctype, support, verification, notes),
            "_verification": verification,
        })

    return results


# ===========================================================================
# 7. Public API: extract_claims
# ===========================================================================

def extract_claims(text: str, source_format: str = "text") -> list[dict]:
    """Extract epistemic claims from *text*.

    Parameters
    ----------
    text:
        Raw source text (plain, markdown, html-light, mdx supported).
    source_format:
        ``"text"`` (default), ``"md"``, ``"markdown"``, ``"mdx"``, or ``"rst"``.

    Returns
    -------
    list[dict]
        Each dict: ``{"text", "type", "support_in_text", "quote"}``.
        ``quote`` is the source sentence truncated to 300 chars.
    """
    clean = _ingest(text, source_format)
    fiction = _guess_genre(clean) == "fiction"

    sentences: list[_Sentence] = []
    sent_counter = 0
    for block, block_start, _block_end in _paragraph_blocks(clean):
        if _is_heading(block):
            continue
        for sent_text, sent_start, sent_end in split_sentences(block, block_start):
            sent_counter += 1
            sentences.append(_Sentence(
                text=sent_text,
                span_start=sent_start,
                span_end=sent_end,
                id=f"sent_{sent_counter:04d}",
            ))

    claims: list[dict] = []
    for sentence in sentences:
        for raw in _extract_from_sentence(sentence, fiction):
            claims.append({
                "text": raw["text"],
                "type": raw["type"],
                "support_in_text": raw["support_in_text"],
                "quote": raw["quote"],
            })
    return claims


# ===========================================================================
# 8. embed_texts  (vendored from episteme/consolidate.py _embed)
# ===========================================================================

_EMBED_CACHE: dict[str, object] = {}
_EMBED_MODEL = "minishlab/potion-retrieval-32M"


def embed_texts(texts: list[str]) -> "list | None":
    """Return L2-normalised embeddings for *texts*, or ``None`` if unavailable.

    Uses ``minishlab/potion-retrieval-32M`` (static CPU-only model2vec model).
    Returns ``None`` when model2vec is not installed or the model cannot load.
    """
    try:
        import numpy as np
        from model2vec import StaticModel  # type: ignore[import]
    except ImportError:
        return None
    model = _EMBED_CACHE.get(_EMBED_MODEL)
    if model is None:
        try:
            model = StaticModel.from_pretrained(_EMBED_MODEL)
        except Exception:
            return None
        _EMBED_CACHE[_EMBED_MODEL] = model
    vecs = np.asarray(model.encode(texts), dtype="float32")  # type: ignore[union-attr]
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vecs / norms).tolist()


# ===========================================================================
# 9. jaccard  (vendored from episteme/consolidate.py _content_toks / _jac)
# ===========================================================================

_WORD_RE = re.compile(r"[a-z0-9']+")
_STOP_WORDS = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "for", "with",
    "as", "at", "by", "is", "are", "was", "were", "be", "been", "being", "that",
    "this", "it", "its", "they", "their", "them", "we", "our", "you", "your",
    "he", "she", "his", "her", "i", "not", "no", "so", "if", "then", "than",
    "there", "here", "what", "which", "who", "how", "why", "when", "from", "into",
    "out", "up", "down", "over", "about", "would", "could", "will", "can", "may",
    "might", "do", "does", "did", "have", "has", "had", "more", "most", "some",
    "any", "all", "one", "also", "such", "these", "those", "very", "just", "like",
}


def _content_toks(text: str) -> set[str]:
    """Lowercase content tokens: stopwords and length<=2 tokens removed."""
    return {
        t for t in _WORD_RE.findall(text.lower())
        if len(t) > 2 and t not in _STOP_WORDS
    }


def jaccard(a: str, b: str) -> float:
    """Content-token Jaccard similarity between strings *a* and *b*."""
    aa, bb = _content_toks(a), _content_toks(b)
    if not aa or not bb:
        return 0.0
    return len(aa & bb) / len(aa | bb)
