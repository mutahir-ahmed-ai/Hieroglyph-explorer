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

    # Most sign documents are 200-400 chars — no need for heavy splitting.
    # We split anyway to handle any long notes fields gracefully.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(docs)

    return FAISS.from_documents(chunks, embeddings)


def query_knowledge_base(
    query: str,
    vector_store: FAISS,
    groq_api_key: str,
    k: int = 5,
) -> dict:
    """
    Search the FAISS index for the k most relevant sign documents,
    then use Groq to synthesise a natural-language answer.

    Returns a dict with:
        answer:        the Groq-generated response text
        matched_codes: list of Gardiner codes from retrieved chunks
    """

    # Step 1: Retrieve the k most relevant chunks
    retrieved = vector_store.similarity_search(query, k=k)

    # Extract unique Gardiner codes from retrieved chunks
    matched_codes = []
    seen = set()
    for doc in retrieved:
        code = doc.metadata.get("gardiner_code", "")
        if code and code not in seen:
            matched_codes.append(code)
            seen.add(code)

    # Step 2: Build context for the LLM
    context_parts = []
    for doc in retrieved:
        code = doc.metadata.get("gardiner_code", "")
        context_parts.append(f"[Sign {code}]\n{doc.page_content}")
    context = "\n\n---\n\n".join(context_parts)

    # Step 3: Call Groq
    system_prompt = """You are an expert Egyptologist and hieroglyph scholar. 
Answer questions about ancient Egyptian hieroglyphs using the provided knowledge base context.

Your answers should:
- Be accurate, detailed, and educational
- Explain technical terms (phonogram, determinative, etc.) in plain language
- Reference the specific Gardiner sign codes when relevant (e.g. G17 for the owl)
- Be engaging — hieroglyphs are fascinating and you should convey that
- If the user's question matches specific signs in the context, describe them clearly
- If the question is about a concept (like 'what is a determinative'), explain it using examples from the context

Base your answer entirely on the provided context. If the context doesn't contain enough information, say so honestly."""

    user_message = f"""Knowledge base context:
{context}

Question: {query}

Please provide a detailed, accurate answer based on the knowledge base above."""

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

    answer = response.choices[0].message.content

    return {
        "answer":        answer,
        "matched_codes": matched_codes,
    }
