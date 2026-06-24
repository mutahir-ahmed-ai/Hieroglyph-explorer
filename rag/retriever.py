import re
import streamlit as st
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from groq import Groq
from rag.knowledge_base import build_documents


@st.cache_resource
def get_vector_store(_df: pd.DataFrame) -> FAISS:
    """
    Build and cache the FAISS vector index from all 770 sign documents.
    @st.cache_resource runs once per session — underscore prefix on _df
    tells Streamlit not to try hashing the DataFrame.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    docs = build_documents(_df)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(docs)
    return FAISS.from_documents(chunks, embeddings)


def _extract_gardiner_code(query: str):
    """
    Detect a Gardiner code in the query string.
    Handles: G17, V34, Aa10, N35, D21, Z1, A14a etc.
    The fix: use [A-Z] to match ANY uppercase letter — previous version
    used [A-HI-LM-PR-Z] which was missing N, O, S, T, U, V, W, X, Y.
    Returns the normalised code string, or None if no code found.
    """
    pattern = r'\b(Aa\d+[a-z]?|[A-Z]\d+[a-z]?)\b'
    match = re.search(pattern, query)
    if not match:
        return None
    raw = match.group(0)
    if raw[:2].lower() == 'aa':
        return 'Aa' + raw[2:]
    else:
        return raw[0].upper() + raw[1:]


def _direct_lookup(code: str, df: pd.DataFrame) -> str | None:
    """
    Look up a Gardiner code directly in the DataFrame.
    This is 100% reliable — no FAISS needed, just a DataFrame filter.
    Returns the sign's document text, or None if not found.
    """
    match = df[df["gardiner_code"].str.upper() == code.upper()]
    if match.empty:
        return None
    row = match.iloc[0]

    # Rebuild the same natural-language summary used in knowledge_base.py
    parts = [
        f"Gardiner sign {row.get('gardiner_code', '')} is called the "
        f"{row.get('english_name', '')}.",
        f"It belongs to category {row.get('category_name', '')}.",
    ]
    phonetic = row.get("phonetic_value", "")
    if phonetic and str(phonetic).lower() not in ("none", "nan", ""):
        parts.append(
            f"Its phonetic value is '{phonetic}', meaning it represents "
            f"that sound in writing."
        )
    else:
        parts.append(
            "It has no phonetic value and is used purely as a determinative."
        )
    sign_type = row.get("type", "")
    if sign_type:
        parts.append(f"It functions as a {sign_type}.")
    primary = row.get("primary_meaning", "")
    if primary:
        parts.append(f"Primary meaning: {primary}.")
    secondary = row.get("secondary_meanings", "")
    if secondary and str(secondary).lower() not in ("none", "nan", ""):
        parts.append(f"Secondary uses include: {secondary}.")
    contexts = row.get("common_contexts", "")
    if contexts and str(contexts).lower() not in ("none", "nan", ""):
        parts.append(f"Commonly found in: {contexts}.")
    notes = row.get("notes", "")
    if notes and str(notes).lower() not in ("none", "nan", ""):
        parts.append(f"Notes: {notes}.")
    status = row.get("status", "")
    if status == "Needs Review":
        parts.append(
            "Note: this sign is uncertain and needs further scholarly review."
        )

    return " ".join(parts)


def query_knowledge_base(
    query: str,
    vector_store: FAISS,
    groq_api_key: str,
    df: pd.DataFrame,
    k: int = 5,
) -> dict:
    """
    Two-phase retrieval:
    1. If a Gardiner code is detected, look it up DIRECTLY in the DataFrame
       — guaranteed to find the right sign every time.
    2. FAISS semantic search fills the remaining context slots.
    """

    # ── Phase 1: Direct DataFrame lookup ──────────────────────────────────
    direct_code = _extract_gardiner_code(query)
    direct_text = None
    if direct_code:
        direct_text = _direct_lookup(direct_code, df)

    # ── Phase 2: FAISS semantic search ────────────────────────────────────
    semantic_results = vector_store.similarity_search(query, k=k)

    # ── Merge results ─────────────────────────────────────────────────────
    # Direct match always first, then FAISS results (deduped)
    final_context_parts = []
    matched_codes = []

    if direct_text and direct_code:
        final_context_parts.append(f"[Sign {direct_code}]\n{direct_text}")
        matched_codes.append(direct_code)

    seen = set(c.upper() for c in matched_codes)
    for doc in semantic_results:
        code = doc.metadata.get("gardiner_code", "")
        if code.upper() not in seen:
            final_context_parts.append(f"[Sign {code}]\n{doc.page_content}")
            matched_codes.append(code)
            seen.add(code.upper())
        if len(final_context_parts) >= k:
            break

    context = "\n\n---\n\n".join(final_context_parts)

    # ── System prompt ──────────────────────────────────────────────────────
    system_prompt = """You are an expert Egyptologist and hieroglyph scholar.
Answer questions about ancient Egyptian hieroglyphs using the provided knowledge base context.

Your answers should:
- Be accurate, detailed, and educational
- Explain technical terms (phonogram, determinative, biliteral etc.) in plain language
- Reference specific Gardiner sign codes when relevant (e.g. G17 for the owl)
- Be engaging — hieroglyphs are fascinating and you should convey that

IMPORTANT rules:
- If the user asks about a SPECIFIC sign code (V34, G17, Aa10 etc.), the FIRST context
  block contains that sign. Focus your answer on it.
- Use exact facts from the context — phonetic values, meanings, contexts.
- If the requested sign is genuinely absent from the context, say so honestly and briefly.
- Do NOT invent or guess information. Everything must come from the context.

Base your answer on the provided context only."""

    user_message = f"""Knowledge base context:
{context}

Question: {query}

Please provide a detailed, accurate answer."""

    # ── Groq call ─────────────────────────────────────────────────────────
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
