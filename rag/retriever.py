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

    @st.cache_resource means this runs once per session.
    The underscore prefix on _df tells Streamlit not to hash the DataFrame
    (DataFrames are not hashable by default).

    Steps:
    1. Convert each DataFrame row to a LangChain Document
    2. Split documents (most are short but some notes are long)
    3. Embed with all-MiniLM-L6-v2
    4. Build FAISS index
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
    Handles formats like: G17, V34, Aa10, N35, D21, Z1, Aa1
    Returns the normalised code string, or None if no code found.
    """
    # Match Aa prefix (two letters) or single letter, followed by digits and
    # optional lowercase suffix (e.g. A14a, D27a)
    pattern = r'\b(Aa\d+[a-z]?|[A-HI-LM-PR-Z]\d+[a-z]?)\b'
    match = re.search(pattern, query, re.IGNORECASE)
    if not match:
        return None

    raw = match.group(0)

    # Normalise capitalisation
    if raw[:2].lower() == 'aa':
        # Aa-series: Aa10, Aa1, etc.
        return 'Aa' + raw[2:].lstrip('0') or '0'
    else:
        # Single letter category: G17, V34, N35 etc.
        return raw[0].upper() + raw[1:]


def query_knowledge_base(
    query: str,
    vector_store: FAISS,
    groq_api_key: str,
    k: int = 5,
) -> dict:
    """
    Search the knowledge base for relevant sign documents and generate an answer.

    Two-phase retrieval strategy:
    1. If the query contains a recognisable Gardiner code (V34, G17, Aa10 etc.),
       that sign is fetched directly by searching 'Gardiner sign V34' and matching
       the metadata — guaranteeing the right sign appears first regardless of
       what pure semantic search would return.
    2. Standard FAISS semantic search runs in parallel and fills remaining slots
       with related signs, providing useful context for the LLM.

    This solves the core limitation of pure vector search: short codes like
    'V34' share character patterns with 'V35', 'D34' etc., so semantic search
    alone returns the wrong signs when users ask about a specific code.
    """

    # ── Phase 1: Direct code detection and lookup ─────────────────────────
    direct_code = _extract_gardiner_code(query)
    direct_doc = None

    if direct_code:
        # Search using a natural-language phrase that will score high for
        # the specific sign's document
        candidates = vector_store.similarity_search(
            f"Gardiner sign {direct_code}", k=15
        )
        # Find the chunk whose metadata matches the code exactly
        for doc in candidates:
            if doc.metadata.get("gardiner_code", "").upper() == direct_code.upper():
                direct_doc = doc
                break

    # ── Phase 2: Semantic search for context ──────────────────────────────
    semantic_results = vector_store.similarity_search(query, k=k)

    # ── Merge: direct match first, then semantic, deduped ─────────────────
    seen = set()
    retrieved = []

    if direct_doc:
        retrieved.append(direct_doc)
        seen.add(direct_doc.metadata.get("gardiner_code", "").upper())

    for doc in semantic_results:
        code = doc.metadata.get("gardiner_code", "").upper()
        if code not in seen:
            retrieved.append(doc)
            seen.add(code)
        if len(retrieved) >= k:
            break

    # ── Extract ordered unique codes for glyph display ────────────────────
    matched_codes = []
    seen_codes = set()
    for doc in retrieved:
        code = doc.metadata.get("gardiner_code", "")
        if code and code not in seen_codes:
            matched_codes.append(code)
            seen_codes.add(code)

    # ── Build context string for the LLM ──────────────────────────────────
    context_parts = []
    for doc in retrieved:
        code = doc.metadata.get("gardiner_code", "")
        context_parts.append(f"[Sign {code}]\n{doc.page_content}")
    context = "\n\n---\n\n".join(context_parts)

    # ── System prompt ─────────────────────────────────────────────────────
    system_prompt = """You are an expert Egyptologist and hieroglyph scholar.
Answer questions about ancient Egyptian hieroglyphs using the provided knowledge base context.

Your answers should:
- Be accurate, detailed, and educational
- Explain technical terms (phonogram, determinative, biliteral, etc.) in plain language
- Reference specific Gardiner sign codes when relevant (e.g. G17 for the owl)
- Be engaging — hieroglyphs are fascinating and you should convey that

IMPORTANT rules:
- If the user asks about a SPECIFIC sign code (like V34, G17, Aa10), focus your answer 
  entirely on THAT sign. The first context block is always the most relevant sign.
- Use the exact facts from the context — phonetic values, meanings, contexts.
- If the requested sign is genuinely not in the context, say so honestly and briefly,
  then offer to explain the signs that ARE in the context.
- Do NOT invent information. Everything must come from the context provided.

Base your answer on the provided context only."""

    user_message = f"""Knowledge base context:
{context}

Question: {query}

Please provide a detailed, accurate answer."""

    # ── Groq API call ─────────────────────────────────────────────────────
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
