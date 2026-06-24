# 𓂀 Hieroglyph Explorer

An AI-powered reference tool for ancient Egyptian hieroglyphs built on Gardiner's complete Sign List. Browse all 770 signs with their Unicode characters, or ask anything in natural language and get a scholar-quality answer from the knowledge base.

🚀 Live Demo: https://hieroglyph-explorer-fus8jsejdqktgheeuyogyz.streamlit.app/

---

## What It Does

**Browse tab** — Select a category (e.g. "G — Birds") then a specific sign. See the hieroglyph displayed large, with every field from the Gardiner Sign List explained in plain English — phonetic value, sign type, primary and secondary meanings, common contexts, and scholarly notes.

**Ask tab** — Type any question in natural language. The system searches the complete 770-sign knowledge base using semantic similarity and generates a detailed answer. Examples:

- *"What does the owl hieroglyph mean?"* → finds G17, explains phonetic value m, uses in funerary texts
- *"Which signs represent water?"* → finds N35, N36, N37, explains ripple vs pool distinctions  
- *"What is a determinative?"* → explains the concept with concrete examples from the knowledge base

---

## The Hieroglyph Images

Signs are displayed as Unicode characters from the Egyptian Hieroglyphs block (U+13000–U+1342F), standardised in Unicode 5.2. All 770 signs in the database are mapped — 760 with confirmed Unicode characters, 10 rare variants with a fallback symbol.

No image files. No hosting. Clean, sharp, scalable — rendered by the browser using the Noto Sans Egyptian Hieroglyphs font.

---

## Architecture

```
User query
    ↓
all-MiniLM-L6-v2 embeds the query → 384-dim vector
    ↓
FAISS similarity search → top 5 most relevant sign documents
    ↓
Groq Llama 3.3 70B synthesises answer from retrieved context
    ↓
Matched Gardiner codes → Unicode glyphs displayed alongside answer
```

The knowledge base is built from two source files:
- `Gardiner_Sign_List_COMPLETE.xlsx` — 711 verified signs with full metadata
- `Gardiner_Uncertain_Signs.csv` — 59 needs-review signs

Each sign is converted to a natural-language document that FAISS can search semantically. "What makes an owl sound" finds G17 even though the user never typed "G17" or "phonetic m".

---

## Tech Stack

| Component | Tool |
|---|---|
| Vector search | FAISS in-process |
| Embeddings | HuggingFace all-MiniLM-L6-v2 |
| LLM | Llama 3.3 70B via Groq |
| Framework | LangChain |
| Hieroglyph display | Unicode U+13000–U+1342F |
| UI | Streamlit |
| Data | pandas + openpyxl |

---

## Deployment

### 1. Create repo and upload files

Create a new GitHub repo named `hieroglyph-explorer`. Upload all project files **and** the two data files:

```
hieroglyph-explorer/
├── rag/
│   ├── __init__.py
│   ├── knowledge_base.py
│   └── retriever.py
├── utils/
│   ├── __init__.py
│   └── unicode_map.py
├── Gardiner_Sign_List_COMPLETE.xlsx   ← upload this
├── Gardiner_Uncertain_Signs.csv       ← upload this
├── app.py
├── requirements.txt
└── README.md
```

### 2. Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) → New app
2. Repository: `mutahir-ahmed-ai/hieroglyph-explorer`
3. Main file: `app.py`
4. Advanced settings → Secrets:

```toml
GROQ_API_KEY = "your_groq_key_here"
```

5. Deploy — first build takes 3-4 minutes (downloads sentence-transformers model)

---

## Author

**Mutahir Ahmed** — AI Developer | Published Egyptology researcher (2021)

This project connects Mutahir's published research on computational analysis of Egyptian hieroglyphs to his 2026 AI development portfolio. The Gardiner Sign List knowledge base was compiled and structured specifically for this application.

[GitHub](https://github.com/mutahir-ahmed-ai)
