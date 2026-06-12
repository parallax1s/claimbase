# Vendored and condensed from Episteme (MIT, github.com/parallax1s/Episteme).
"""
Behavioural tests for extractor.engine — vendored from:
  - episteme/tests/test_epistemic_improvements.py
  - episteme/tests/test_segment.py

Covers: question guard, imperative guard, prepositional openers, anaphora
merge, citation-aware support, German/English abbreviations, sentence-boundary
correctness, and an end-to-end 3-paragraph extraction test.
"""

from __future__ import annotations

from extractor.engine import (
    _Sentence,
    _extract_from_sentence,
    _guess_genre,
    _ingest,
    _is_heading,
    _paragraph_blocks,
    extract_claims,
    jaccard,
    split_sentences,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _claims(text: str, source_format: str = "text"):
    """Run extract_claims and return the full internal dicts (with _risk_flags etc.)."""
    clean = _ingest(text, source_format)
    fiction = _guess_genre(clean) == "fiction"
    sentences = []
    sent_counter = 0
    for block, block_start, _block_end in _paragraph_blocks(clean):
        if _is_heading(block):
            continue
        prev_text = None
        for sent_text, sent_start, sent_end in split_sentences(block, block_start):
            sent_counter += 1
            sentences.append((
                _Sentence(
                    text=sent_text, span_start=sent_start, span_end=sent_end,
                    id=f"sent_{sent_counter:04d}",
                ),
                prev_text,
            ))
            prev_text = sent_text
    result = []
    for sentence, prev_text in sentences:
        result.extend(_extract_from_sentence(sentence, fiction, prev_text))
    return result


# ===========================================================================
# (1) Question guard
# ===========================================================================

def test_interrogative_is_not_a_claim():
    assert extract_claims("Is the moon made of cheese?") == []


def test_interrogative_variants_are_not_claims():
    for text in [
        "Did the policy reduce emissions?",
        "Really, did it work?!",
    ]:
        assert extract_claims(text) == [], text


def test_declarative_extracts_statistical_claim():
    claims = extract_claims("The policy reduced emissions by 20 percent.")
    assert len(claims) == 1
    assert claims[0]["type"] == "statistical"


def test_cited_question_is_still_a_question():
    assert extract_claims("Did crime really fall?[1]") == []


# ===========================================================================
# (2) Anaphora fragment merge
# ===========================================================================

def test_anaphoric_fragment_is_merged_not_emitted():
    full = _claims("The first study was solid, but that claim needs better evidence.")
    texts = [c["text"] for c in full]
    assert len(full) == 1
    assert not any(t.lower().startswith("that claim") for t in texts)
    assert "that claim needs better evidence" in texts[0]


def test_short_trailing_fragment_is_merged_back():
    full = _claims("Sales rose sharply and profits grew.")
    texts = [c["text"] for c in full]
    assert not any(t.strip().lower() == "profits grew" for t in texts)
    assert any("profits grew" in t for t in texts)


def test_bare_verb_conjunct_is_merged_not_emitted():
    full = _claims(
        "I lean on their framework throughout and say clearly where I push further."
    )
    texts = [c["text"] for c in full]
    assert len(full) == 1
    assert not any(t.strip().lower().startswith("say ") for t in texts)
    assert "say clearly where I push further" in texts[0]


def test_figures_bare_verb_show_conjunct_is_merged():
    full = _claims("The figures are stable across years and show no real decline.")
    texts = [c["text"] for c in full]
    assert len(full) == 1
    assert not any(t.strip().lower().startswith("show ") for t in texts)
    assert "show no real decline" in texts[0]


# ===========================================================================
# (3) Imperative guard
# ===========================================================================

def test_imperative_is_not_a_claim():
    assert extract_claims("Consider the long-run consequences now.") == []


def test_imperative_will_not_predictive():
    assert extract_claims("Build the funnel for the flame that it will become.") == []


def test_imperative_with_normative_marker_becomes_normative():
    full = _claims("Build a better safety culture across the company.")
    assert len(full) == 1
    assert full[0]["type"] == "normative"
    assert "imperative" in full[0]["_reason"].lower()


def test_prepositional_opener_is_not_skipped_as_imperative():
    assert len(extract_claims("During the pandemic, unemployment rose to 15 percent.")) > 0


def test_prepositional_opener_history():
    assert len(extract_claims(
        "Over the broad span of history, democracy is more the exception than the rule."
    )) > 0


def test_noun_use_of_is_not_imperative():
    assert len(extract_claims("Use of force declined sharply after the reform.")) > 0


# ===========================================================================
# (4) Attribution / refutation flags
# ===========================================================================

def test_attributed_quote_flagged_with_refutation():
    full = _claims('"The earth is flat," said the conspiracy theorist, wrongly.')
    assert len(full) == 1
    assert "attributed_quote" in full[0]["_risk_flags"]
    assert "refuted_in_text" in full[0]["_risk_flags"]
    assert "attribut" in full[0]["_reason"].lower()


def test_attribution_no_refutation_flag():
    full = _claims('"The bridge is safe," the engineer said.')
    assert len(full) == 1
    assert "attributed_quote" in full[0]["_risk_flags"]
    assert "refuted_in_text" not in full[0]["_risk_flags"]


# ===========================================================================
# (5) Citation-aware support
# ===========================================================================

def test_inline_citation_raises_support_and_is_not_statistical():
    full = _claims("The reform cut emissions sharply [2].")
    assert len(full) == 1
    assert full[0]["type"] != "statistical"
    assert full[0]["support_in_text"] >= 0.5
    assert "citation" in full[0]["_reason"].lower()
    assert "unsupported_major_claim" not in full[0]["_risk_flags"]


def test_bracketed_citation_pair_is_not_a_statistic():
    full = _claims("Productivity rose for remote workers [1, 2].")
    assert len(full) == 1
    assert full[0]["type"] != "statistical"
    assert full[0]["support_in_text"] >= 0.5


def test_real_statistic_with_citation_stays_statistical():
    full = _claims("The intervention reduced readmissions by 40% [3].")
    assert len(full) == 1
    assert full[0]["type"] == "statistical"
    assert full[0]["support_in_text"] >= 0.5


def test_bare_number_word_is_not_statistical():
    full = _claims("This essay is about a different one entirely, namely power.")
    assert all(c["type"] != "statistical" for c in full)


def test_url_counts_as_support():
    full = _claims(
        "The policy reduced emissions, documented at https://example.org/report."
    )
    assert len(full) == 1
    assert full[0]["support_in_text"] >= 0.5


# ===========================================================================
# (6) Sentence splitter — abbreviations and German ordinals
# ===========================================================================

def test_split_no_split_after_dr():
    pieces = split_sentences("Dr. Smith arrived. He left.", 0)
    assert [t for t, _, _ in pieces] == ["Dr. Smith arrived.", "He left."]


def test_split_no_split_after_eg():
    pieces = split_sentences("Fruit, e.g. apples, is cheap. The economy grew.", 0)
    texts = [t for t, _, _ in pieces]
    assert texts == ["Fruit, e.g. apples, is cheap.", "The economy grew."]


def test_split_german_18_Jahrhundert_one_sentence():
    paragraph = "Die Aufklärung war eine geistige Bewegung des 18. Jahrhunderts in Europa."
    pieces = split_sentences(paragraph, 0)
    assert [t for t, _, _ in pieces] == [paragraph]


def test_split_german_pair_splits_cleanly():
    paragraph = "Die Kriminalität sank im 18. Jahrhundert um 40 Prozent. Niemand weiß warum."
    pieces = split_sentences(paragraph, 0)
    assert [t for t, _, _ in pieces] == [
        "Die Kriminalität sank im 18. Jahrhundert um 40 Prozent.",
        "Niemand weiß warum.",
    ]


def test_split_keeps_citation_marker_with_preceding_sentence():
    paragraph = (
        "Emissions fell by 12 percent.[1, 2] However, the recession explains the decline.[3] "
        "Therefore, the policy claim is weak."
    )
    pieces = split_sentences(paragraph, 0)
    texts = [t for t, _, _ in pieces]
    assert texts == [
        "Emissions fell by 12 percent.[1, 2]",
        "However, the recession explains the decline.[3]",
        "Therefore, the policy claim is weak.",
    ]
    for text, start, end in pieces:
        assert paragraph[start:end] == text


def test_split_1945_year_still_splits():
    pieces = split_sentences("It ended in 1945. The next era began.", 0)
    assert [t for t, _, _ in pieces] == ["It ended in 1945.", "The next era began."]


# ===========================================================================
# (7) Public API shape
# ===========================================================================

def test_extract_claims_returns_correct_keys():
    claims = extract_claims("The policy reduced emissions by 20 percent.")
    assert claims
    for claim in claims:
        assert set(claim.keys()) == {"text", "type", "support_in_text", "quote"}


def test_extract_claims_quote_max_300_chars():
    long_sentence = "A " * 200 + "reduced emissions significantly."
    claims = extract_claims(long_sentence)
    for claim in claims:
        assert len(claim["quote"]) <= 300


# ===========================================================================
# (7b) Context quotes — previous sentence rides along as quote prefix
# ===========================================================================

def test_quote_includes_antecedent_sentence_for_dangling_referent():
    text = (
        "The asteroid will strike the planet within thirty years. "
        "There is nothing you can do about it."
    )
    claims = extract_claims(text)
    dangling = [c for c in claims if "nothing you can do" in c["text"]]
    assert len(dangling) == 1
    quote = dangling[0]["quote"]
    assert "asteroid will strike" in quote
    assert quote.endswith("There is nothing you can do about it.")


def test_quote_first_sentence_of_paragraph_has_no_prefix():
    text = "The policy reduced emissions by 20 percent. It also cut compliance costs."
    claims = extract_claims(text)
    first = [c for c in claims if c["text"].startswith("The policy")]
    assert first
    assert first[0]["quote"] == "The policy reduced emissions by 20 percent."
    second = [c for c in claims if c["text"].startswith("It also")]
    assert second
    assert second[0]["quote"] == (
        "The policy reduced emissions by 20 percent. It also cut compliance costs."
    )


def test_quote_respects_paragraph_boundaries():
    text = (
        "The committee endorsed the reform unanimously.\n\n"
        "It reduced emissions by 20 percent."
    )
    claims = extract_claims(text)
    target = [c for c in claims if c["text"].startswith("It reduced")]
    assert len(target) == 1
    assert target[0]["quote"] == "It reduced emissions by 20 percent."
    assert "committee" not in target[0]["quote"]


def test_quote_front_truncation_preserves_claim_sentence():
    prev = "The committee reviewed the evidence " + ("very " * 60) + "carefully."
    claim_sentence = "It reduced emissions by 20 percent."
    claims = extract_claims(f"{prev} {claim_sentence}")
    target = [c for c in claims if c["text"].startswith("It reduced")]
    assert len(target) == 1
    quote = target[0]["quote"]
    assert len(quote) <= 300
    # The claim sentence always survives whole, at the end of the quote.
    assert quote.endswith(claim_sentence)
    # The sacrificial prefix is front-truncated and ellipsis-marked.
    assert quote.startswith("…")
    # The cut lands on a word boundary: the first prefix word is a real word.
    first_word = quote[1:].lstrip().split(" ")[0]
    assert first_word in prev.split()


def test_quote_overlong_single_sentence_keeps_legacy_truncation():
    long_sentence = "The reform " + ("really " * 50) + "reduced emissions by 20 percent."
    claims = extract_claims(f"Short lead-in sentence first. {long_sentence}")
    target = [c for c in claims if "reduced emissions" in c["text"]]
    assert target
    for claim in target:
        assert len(claim["quote"]) <= 300
        # No prefix fits; the overlong claim sentence is tail-truncated as before.
        assert claim["quote"] == long_sentence[:300]


# ===========================================================================
# (8) jaccard
# ===========================================================================

def test_jaccard_identical():
    assert jaccard("emissions policy reduced", "emissions policy reduced") == 1.0


def test_jaccard_disjoint():
    assert jaccard("apple orange", "computer keyboard") == 0.0


def test_jaccard_partial():
    score = jaccard("the policy reduced emissions", "emissions were reduced by policy")
    assert 0.0 < score <= 1.0


def test_jaccard_stopwords_filtered():
    # "the" and "a" are stopwords; content tokens are "cat" and "dog"
    a = jaccard("the cat sat", "a dog sat")
    b = jaccard("cat sat", "dog sat")
    assert abs(a - b) < 0.01


# ===========================================================================
# (9) End-to-end: 3-paragraph sample
# ===========================================================================

_SAMPLE = """\
AI systems have achieved superhuman performance on many narrow tasks.
This progress raises urgent questions about safety and alignment.
The gap between capabilities and alignment research has widened over the past decade.

Critics argue that current safety techniques are insufficient for frontier models.
Some researchers claim we should pause development until alignment is solved.
Evidence suggests that interpretability tools remain rudimentary.

Policy responses have varied widely across jurisdictions.
The EU AI Act introduces binding requirements for high-risk systems.
Voluntary commitments from labs have had limited measurable impact so far.
"""


def test_end_to_end_extracts_multiple_claims():
    claims = extract_claims(_SAMPLE)
    assert len(claims) >= 5


def test_end_to_end_claim_types_are_valid():
    valid = {
        "empirical", "statistical", "causal", "predictive", "normative",
        "definitional", "descriptive", "historical", "other",
        # also allowed by SCHEMA: "legal", "medical", "financial"
        "legal", "medical", "financial",
    }
    for claim in extract_claims(_SAMPLE):
        assert claim["type"] in valid, claim


def test_end_to_end_support_in_range():
    for claim in extract_claims(_SAMPLE):
        assert 0.0 <= claim["support_in_text"] <= 1.0, claim


def test_end_to_end_no_questions_extracted():
    text = _SAMPLE + "\nIs AI safe? Will it ever be aligned?\n"
    claims = extract_claims(text)
    for claim in claims:
        assert not claim["text"].rstrip().endswith("?"), claim


def test_end_to_end_quote_is_source_sentence():
    # quote should be a non-empty string coming from the source
    for claim in extract_claims(_SAMPLE):
        assert isinstance(claim["quote"], str)
        assert len(claim["quote"]) > 0


# ===========================================================================
# (10) embed_texts (graceful None when model2vec absent)
# ===========================================================================

def test_embed_texts_returns_list_or_none():
    from extractor.engine import embed_texts
    result = embed_texts(["hello world", "AI alignment"])
    assert result is None or isinstance(result, list)


def test_embed_texts_none_on_import_error(monkeypatch):
    import extractor.engine as eng
    import builtins
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name in ("model2vec", "numpy"):
            raise ImportError
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    result = eng.embed_texts(["test"])
    assert result is None
