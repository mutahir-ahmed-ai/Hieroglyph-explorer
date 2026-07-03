import re
import streamlit as st
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from groq import Groq
from rag.knowledge_base import build_documents


@st.cache_resource
def get_vector_store(_df: pd.DataFrame) -> FAISS:
    """
    Build and cache the FAISS vector index from all 770 sign documents.

    NOTE: no text splitter here — the longest sign document is ~500
    characters, so a chunk_size=800 splitter never actually split
    anything. One sign = one document = one vector, guaranteed whole.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    docs = build_documents(_df)
    return FAISS.from_documents(docs, embeddings)


# ── Phase 1: Direct Gardiner-code lookup ────────────────────────────────────

def _extract_gardiner_codes(query: str, df: pd.DataFrame) -> list[str]:
    """
    Detect Gardiner codes in the query — case-insensitive, multi-code.

    Fixes over the previous version:
    1. 'g17' / 'v34' (lowercase, how real users type) now match.
    2. 'Compare A1 and B1' returns BOTH codes, not just the first.
    3. Every candidate is validated against the actual dataset, so a
       stray letter+digit token that isn't a real sign can never
       trigger a bogus direct lookup.
    """
    candidates = re.findall(r'\b(aa\d+[a-z]?|[a-z]\d+[a-z]?)\b', query.lower())

    valid = set(df["gardiner_code"].str.upper())
    codes, seen = [], set()
    for raw in candidates:
        if raw.startswith('aa'):
            code = 'Aa' + raw[2:]
        else:
            code = raw[0].upper() + raw[1:]
        if code.upper() in valid and code.upper() not in seen:
            codes.append(code)
            seen.add(code.upper())
    return codes


def _direct_lookup(code: str, df: pd.DataFrame) -> str | None:
    """
    Look up a Gardiner code directly in the DataFrame.
    100% reliable — no FAISS involved, just a filter.
    """
    match = df[df["gardiner_code"].str.upper() == code.upper()]
    if match.empty:
        return None
    row = match.iloc[0]

    def clean(val) -> str:
        s = str(val).strip()
        return "" if s.lower() in ("none", "nan", "") else s

    parts = [
        f"Gardiner sign {clean(row.get('gardiner_code'))} is called the "
        f"{clean(row.get('english_name'))}.",
        f"It belongs to category {clean(row.get('category_name'))}.",
    ]
    phonetic = clean(row.get("phonetic_value"))
    if phonetic:
        parts.append(
            f"Its phonetic value is '{phonetic}', meaning it represents "
            f"that sound in writing."
        )
    else:
        parts.append(
            "It has no phonetic value and is used purely as a determinative."
        )
    if clean(row.get("type")):
        parts.append(f"It functions as a {clean(row.get('type'))}.")
    if clean(row.get("primary_meaning")):
        parts.append(f"Primary meaning: {clean(row.get('primary_meaning'))}.")
    if clean(row.get("secondary_meanings")):
        parts.append(f"Secondary uses include: {clean(row.get('secondary_meanings'))}.")
    if clean(row.get("common_contexts")):
        parts.append(f"Commonly found in: {clean(row.get('common_contexts'))}.")
    if clean(row.get("notes")):
        parts.append(f"Notes: {clean(row.get('notes'))}.")
    if str(row.get("status", "")) == "Needs Review":
        parts.append(
            "Note: this sign is uncertain and needs further scholarly review."
        )

    return " ".join(parts)


# ── Phase 0: Aggregate / meta-question routing ──────────────────────────────
# Questions that require counting or grouping across the ENTIRE dataset.
# FAISS structurally cannot answer these (top-k retrieval never sees the
# whole table), so they are answered deterministically from pandas.
#
# ROUTING LAW: specific patterns are checked BEFORE general ones.
# 'how many signs in category A' must hit the category handler, and must
# NOT fall through to the total-signs handler.

_COUNT_WORDS = r'\bhow many\b|\btotal\b|\bcount\b|\bnumber of\b'


def _category_from_query(q: str, df: pd.DataFrame):
    """Find a category referenced by LETTER or by NAME. Returns letter or None."""
    letters = set(df["category_letter"].str.lower())

    # By letter: 'category A', 'the U category', 'group Aa'.
    # Each pattern runs SEPARATELY — a combined alternation lets an early
    # junk match (the word 'in' before 'category') consume the text that
    # the real match needed. Every candidate is validated against the data.
    for pat in (
        r'\bcategor(?:y|ies)\s+([a-z]{1,2})\b',   # 'category a'
        r'\b([a-z]{1,2})\s+category\b',           # 'u category'
        r'\bgroup\s+([a-z]{1,2})\b',              # 'group aa'
    ):
        for m in re.finditer(pat, q):
            letter = m.group(1)
            if letter in letters:
                return letter.upper() if len(letter) == 1 else letter.capitalize()

    # By name: 'bird signs', 'how many mammals', 'fish category'
    names = df.groupby("category_letter")["category_name"].first()
    for letter, name in names.items():
        for word in re.findall(r'[a-z]+', str(name).lower()):
            if len(word) < 4:          # skip 'and', 'his', 'her', 'of'...
                continue
            # match singular or plural form as a whole word
            if re.search(rf'\b{word}s?\b|\b{word.rstrip("s")}\b', q):
                return letter
    return None


def _detect_aggregate_query(query: str, df: pd.DataFrame):
    """Returns (kind, arg) if this is an aggregate question, else None."""
    q = query.lower()
    counting = bool(re.search(_COUNT_WORDS, q))

    # 0. If the query names a real sign code (Z1, G17...), it is a
    #    sign-specific question — never an aggregate. Bail out so
    #    'how many strokes does Z1 have' goes to direct lookup.
    if _extract_gardiner_codes(query, df):
        return None

    # 1. MOST SPECIFIC: count of signs in one category — by letter
    #    ('category A') or by name ('bird signs', 'how many mammals')
    if counting:
        letter = _category_from_query(q, df)
        if letter:
            return ('category_specific', letter)

    # 2. Number of categories
    if counting and re.search(r'\bcategor(?:y|ies)\b|\bgroups?\b', q) \
            and not re.search(r'\bsigns?\s+(?:are\s+)?in\b', q):
        if re.search(r'how many (?:categories|groups)\b|number of (?:categories|groups)\b', q):
            return ('category_count', None)

    # 3. Category with most/fewest signs — must run BEFORE total_signs,
    #    or 'least number of signs' gets swallowed by the general pattern. — requires explicit sign-counting
    #    intent, so 'most common determinative in category G' does NOT match.
    superlative = re.search(
        r'\b(most|fewest|least|largest|biggest|smallest)\b[^.?!]*\bsigns?\b'
        r'|\bsigns?\b[^.?!]*\b(most|fewest|least)\b'
        r'|which (?:category|group) (?:has|is) the '
        r'(most|fewest|least|largest|biggest|smallest)\b',
        q,
    )
    if superlative and re.search(r'\bcategor|\bgroup\b', q):
        word = next(g for g in superlative.groups() if g)
        if word in ('fewest', 'least', 'smallest'):
            return ('category_least', None)
        return ('category_most', None)

    # 4. Total signs (MOST GENERAL — always checked last)
    if counting and re.search(r'\bsigns?\b|\bhieroglyphs?\b', q):
        return ('total_signs', None)

    return None


def _answer_aggregate(kind: str, arg, df: pd.DataFrame) -> dict:
    if kind == 'category_specific':
        subset = df[df['category_letter'] == arg]
        name = subset['category_name'].iloc[0]
        verified = int((subset['status'] == 'Verified').sum())
        review = int((subset['status'] == 'Needs Review').sum())
        answer = (
            f"Category {arg} ({name}) contains {len(subset)} signs — "
            f"{verified} verified"
            + (f" and {review} flagged as needing scholarly review."
               if review else ".")
        )
        return {"answer": answer, "matched_codes": []}

    if kind == 'total_signs':
        total = len(df)
        verified = int((df['status'] == 'Verified').sum())
        review = int((df['status'] == 'Needs Review').sum())
        answer = (
            f"There are {total} total signs in the Gardiner Sign List used "
            f"by this system — {verified} verified and {review} "
            f"flagged as needing further scholarly review."
        )
        return {"answer": answer, "matched_codes": []}

    if kind == 'category_count':
        names = (
            df.groupby('category_letter')['category_name']
            .first()
            .sort_index()
        )
        listing = "\n".join(f"- {l}: {n}" for l, n in names.items())
        answer = (
            f"There are {len(names)} categories in the Gardiner Sign List:\n\n"
            f"{listing}"
        )
        return {"answer": answer, "matched_codes": []}

    if kind in ('category_least', 'category_most'):
        counts = df.groupby('category_letter').size()
        letter = counts.idxmin() if kind == 'category_least' else counts.idxmax()
        name = df[df['category_letter'] == letter]['category_name'].iloc[0]
        n = int(counts.min() if kind == 'category_least' else counts.max())
        adjective = "fewest" if kind == 'category_least' else "most"
        answer = (
            f"The category with the {adjective} signs is "
            f"{letter} ({name}) with {n} signs."
        )
        return {"answer": answer, "matched_codes": []}

    return {"answer": "Unable to compute that.", "matched_codes": []}


# ── Main entry point ────────────────────────────────────────────────────────

def query_knowledge_base(
    query: str,
    vector_store: FAISS,
    groq_api_key: str,
    df: pd.DataFrame,
    k: int = 5,
) -> dict:
    """
    Three-phase retrieval:
    0. Aggregate/meta questions (counts, most/fewest) → answered
       deterministically from pandas. No LLM call, zero hallucination risk.
    1. Gardiner code(s) detected → direct DataFrame lookup, guaranteed hit.
    2. FAISS semantic search fills the remaining context slots.
    """

    # Phase 0 — deterministic aggregate answers
    agg = _detect_aggregate_query(query, df)
    if agg:
        return _answer_aggregate(agg[0], agg[1], df)

    # Phase 1 — direct lookups for every code mentioned
    final_context_parts = []
    matched_codes = []
    for code in _extract_gardiner_codes(query, df):
        text = _direct_lookup(code, df)
        if text:
            final_context_parts.append(f"[Sign {code}]\n{text}")
            matched_codes.append(code)

    # Phase 2 — FAISS semantic search fills remaining slots
    semantic_results = vector_store.similarity_search(query, k=k)
    seen = set(c.upper() for c in matched_codes)
    for doc in semantic_results:
        if len(final_context_parts) >= k:
            break
        code = doc.metadata.get("gardiner_code", "")
        if code.upper() not in seen:
            final_context_parts.append(f"[Sign {code}]\n{doc.page_content}")
            matched_codes.append(code)
            seen.add(code.upper())

    context = "\n\n---\n\n".join(final_context_parts)

    system_prompt = """You are an expert Egyptologist and hieroglyph scholar.
Answer questions about ancient Egyptian hieroglyphs using the provided knowledge base context.

Your answers should:
- Be accurate, detailed, and educational
- Explain technical terms (phonogram, determinative, biliteral etc.) in plain language
- Reference specific Gardiner sign codes when relevant (e.g. G17 for the owl)
- Be engaging — hieroglyphs are fascinating and you should convey that

IMPORTANT rules:
- If the user asks about SPECIFIC sign codes (V34, G17, Aa10 etc.), the first context
  blocks contain those signs. Focus your answer on them.
- Use exact facts from the context — phonetic values, meanings, contexts.
- If the requested sign is genuinely absent from the context, say so honestly and briefly.
- Do NOT invent or guess information. Everything must come from the context.
- The context contains only the retrieved signs, never the full sign list —
  so never claim to give exhaustive counts or complete listings.

Base your answer on the provided context only."""

    user_message = f"""Knowledge base context:
{context}

Question: {query}

Please provide a detailed, accurate answer."""

    client = Groq(api_key=groq_api_key)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
        temperature=0.3,
        max_tokens=1024,
    )

    return {
        "answer":        response.choices[0].message.content,
        "matched_codes": matched_codes,
    }
