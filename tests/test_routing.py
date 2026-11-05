"""
Routing regression battery for Hieroglyph Explorer (25 queries).

Checks that Phase 0 (aggregate questions answered from pandas) and
Phase 1 (Gardiner sign-code lookup) catch the questions they should,
and leave everything else to FAISS + the LLM.

No API key or network needed: routing is deterministic.

Run from the repo root:
    pip install pytest
    pytest -v
"""
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)                      # data files are loaded by relative path
sys.path.insert(0, str(ROOT))

from rag.knowledge_base import load_all_signs                       # noqa: E402
from rag.retriever import (                                         # noqa: E402
    _answer_aggregate,
    _detect_aggregate_query,
    _extract_gardiner_codes,
)


@pytest.fixture(scope="module")
def df():
    return load_all_signs()


@pytest.fixture(scope="module")
def counts(df):
    """Signs per category, computed independently of the router."""
    return df.groupby("category_letter").size()


# ── 1. Count of signs in one category, by letter ────────────────────────────
@pytest.mark.parametrize("query, letter", [
    ("How many signs are in category A?", "A"),
    ("how many signs in category g", "G"),           # lowercase, no '?'
    ("How many signs are in the U category?", "U"),
    ("Count the signs in group Aa", "Aa"),           # two-letter category
])
def test_category_count_by_letter(df, counts, query, letter):
    assert _detect_aggregate_query(query, df) == ("category_specific", letter)
    answer = _answer_aggregate("category_specific", letter, df)["answer"]
    assert f"contains {counts[letter]} signs" in answer


# ── 2. Count of signs in one category, by name ──────────────────────────────
@pytest.mark.parametrize("query, letter", [
    ("How many bird signs are there?", "G"),         # not H (Parts of birds)
    ("How many mammals are in the list?", "E"),      # not F (Parts of mammals)
    ("How many fish signs are there?", "K"),
])
def test_category_count_by_name(df, query, letter):
    assert _detect_aggregate_query(query, df) == ("category_specific", letter)


# ── 3. Total number of signs ────────────────────────────────────────────────
@pytest.mark.parametrize("query", [
    "How many signs are there in total?",
    "What is the total number of hieroglyphs?",
    "Count all the signs",
])
def test_total_signs(df, query):
    assert _detect_aggregate_query(query, df) == ("total_signs", None)
    answer = _answer_aggregate("total_signs", None, df)["answer"]
    assert "770 total signs" in answer


# ── 4. Number of categories ─────────────────────────────────────────────────
@pytest.mark.parametrize("query", [
    "How many categories are there?",
    "What is the number of groups in the sign list?",
])
def test_category_count(df, counts, query):
    assert _detect_aggregate_query(query, df) == ("category_count", None)
    answer = _answer_aggregate("category_count", None, df)["answer"]
    assert f"There are {len(counts)} categories" in answer


# ── 5. Most / fewest superlatives ───────────────────────────────────────────
def test_most_signs(df, counts):
    q = "Which category has the most signs?"
    assert _detect_aggregate_query(q, df) == ("category_most", None)
    answer = _answer_aggregate("category_most", None, df)["answer"]
    top = counts[counts == counts.max()].index
    for letter in top:
        assert f"{letter} (" in answer
    assert f"{counts.max()}" in answer


@pytest.mark.parametrize("query", [
    "Which category has the fewest signs?",
    "Which group has the least number of signs?",
])
def test_fewest_signs(df, counts, query):
    assert _detect_aggregate_query(query, df) == ("category_least", None)
    answer = _answer_aggregate("category_least", None, df)["answer"]
    # Several categories can share the lowest count; every one must be named.
    tied = counts[counts == counts.min()].index
    for letter in tied:
        assert f"{letter} (" in answer
    assert f"{counts.min()}" in answer


# ── 6. A sign code takes precedence over aggregate routing ──────────────────
@pytest.mark.parametrize("query, codes", [
    ("How many strokes does Z1 have?", ["Z1"]),      # 'how many' but names a sign
    ("how many meanings does g17 have", ["G17"]),    # lowercase code
    ("Compare A1 and B1", ["A1", "B1"]),             # multiple codes
    ("what is difference between E13 and N 22?", ["E13", "N22"]),  # code typed with a space
])
def test_sign_code_precedence(df, query, codes):
    assert _detect_aggregate_query(query, df) is None
    assert _extract_gardiner_codes(query, df) == codes


def test_no_false_sign_code(df):
    # A lowercase letter followed by a number is ordinary text, not sign A2.
    assert _extract_gardiner_codes("give me a 2 line summary of the owl sign", df) == []


# ── 7. Negative cases: must NOT be routed to an aggregate answer ────────────
@pytest.mark.parametrize("query", [
    "What does the owl sign mean?",
    "What is the most common determinative in category G?",
    "Show me signs related to birds",
    "Which signs are most often used in royal names?",
    "Tell me about signs used to write the sound m",
])
def test_not_hijacked_by_router(df, query):
    assert _detect_aggregate_query(query, df) is None
