# 𓂀 Hieroglyph Explorer

An AI-powered reference tool for ancient Egyptian hieroglyphs built on Gardiner's complete Sign List. Browse all 770 signs with their Unicode characters, or ask anything in natural language and get a scholar-quality answer from the knowledge base.

Live Demo: https://hieroglyph-explorer-fus8jsejdqktgheeuyogyz.streamlit.app/

---

## What It Does

**Browse tab** — Select a category (e.g. "G — Birds") then a specific sign. See the hieroglyph displayed large, with every field from the Gardiner Sign List explained in plain English — phonetic value, sign type, primary and secondary meanings, common contexts, and scholarly notes.

**Ask tab** — Type any question in natural language. The system routes it to the right answering strategy and responds. Examples:

- *"What does the owl hieroglyph mean?"* → finds G17, explains phonetic value *m*, its uses across inscriptions
- *"Which signs represent water?"* → finds N35, N41, W14, N39, N4 — ripples, wells, jars, pools, and rain
- *"How many signs are in category A?"* → **62** (58 verified, 4 needing review) — answered instantly, computed directly from the dataset
- *"tell me about g17"* → lowercase and multi-code queries ("compare a1 and b1") resolve directly
- *"What is a determinative?"* → explains the concept with concrete examples from the knowledge base

---

## Architecture — Three-Phase Retrieval

```
User query
    ↓
Phase 0 — QUERY ROUTER
    Aggregate question? (counts, totals, most/fewest, per-category)
    → answered deterministically by pandas. No LLM call. No hallucination risk.
    ↓ otherwise
Phase 1 — DIRECT CODE LOOKUP
    Gardiner code(s) in the query? (case-insensitive, multi-code,
    validated against the real dataset)
    → exact DataFrame lookup, guaranteed hit
    ↓ plus
Phase 2 — SEMANTIC SEARCH
    all-MiniLM-L6-v2 embeds the query → 384-dim normalized vector
    → FAISS similarity search → top-k sign documents
    ↓
Groq Llama 3.3 70B synthesises the answer from retrieved context only
    ↓
Matched Gardiner codes → Unicode glyphs displayed alongside the answer
```

**Why the router exists — a real failure mode, found and fixed.** Earlier versions answered *"How many signs are in category A?"* with confidently invented numbers. The mechanism: top-k similarity search retrieves the k chunks that *mean* the most like the question — but a count is a property of the whole table, which no k chunks ever contain. The LLM, seeing only 5 sign entries and asked "how many," produced a fluent guess. A prompt instruction ("do not invent information") did not reliably prevent it, because prompt-level rules are probabilistic. The fix is architectural: counting questions never reach the LLM at all — pandas computes the answer with zero generation step left to fail.

**Grounded refusals.** For questions the router lets through but retrieval can't support (e.g. *"the most common determinative in category G"* when no G chunks are retrieved), the system prompt instructs the model that it never sees the full sign list and must not claim exhaustive knowledge — so it answers honestly that the context is insufficient rather than fabricating.

**Tested.** Routing behaviour is validated by a 23-query regression battery covering: per-category counts (by letter and by name), totals, category counts, most/fewest superlatives, sign-code precedence over aggregates, and negative cases that must *not* be hijacked by the router.

The knowledge base is built from two source files:

- `Gardiner_Sign_List_COMPLETE.xlsx` — 711 verified signs with full metadata
- `Gardiner_Uncertain_Signs.csv` — 59 needs-review signs

Each sign is converted to a natural-language document that FAISS searches semantically. "What makes an owl sound" finds G17 even though the user never typed "G17" or "phonetic m" — meaning-based retrieval, not keyword matching. One sign = one document = one vector; the longest document is ~500 characters, so no chunk splitting is needed and every sign is always retrieved whole.

---

## The Hieroglyph Images

Signs are displayed as Unicode characters from the Egyptian Hieroglyphs block (U+13000–U+1342F), standardised in Unicode 5.2. All 770 signs in the database are mapped — 760 with confirmed Unicode characters, 10 rare variants with a fallback symbol.

No image files. No hosting. Clean, sharp, scalable — rendered by the browser using the Noto Sans Egyptian Hieroglyphs font.

---

## Tech Stack

| Component | Tool |
|---|---|
| Query routing | pandas (deterministic aggregates) |
| Vector search | FAISS in-process (normalized embeddings — inner product = cosine similarity) |
| Embeddings | HuggingFace all-MiniLM-L6-v2 |
| LLM | Llama 3.3 70B via Groq |
| Framework | LangChain |
| Hieroglyph display | Unicode U+13000–U+1342F |
| UI | Streamlit |
| Data | pandas + openpyxl |

---

## Known Limitations

- Filtered aggregates ("most common determinative *within* category G") fall through to semantic search, which answers honestly when context is insufficient rather than computing the true answer — a metadata-filtered retrieval pass is the planned fix.
- Category detection by name uses word matching against category names; unusual phrasings may fall through to semantic search (which fails safe: the LLM is instructed never to claim exhaustive counts).
- FAISS index is rebuilt on cold start (~3–4 min first boot on Streamlit Cloud).

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

**Mutahir Ahmed** — AI Developer | Published researcher — deep-learning hieroglyph detection (ICGCET 2021)

This project connects Mutahir's published research on computational analysis of Egyptian hieroglyphs to his 2026 AI development portfolio. The Gardiner Sign List knowledge base was compiled and structured specifically for this application.

[GitHub](https://github.com/mutahir-ahmed-ai)
