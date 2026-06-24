import streamlit as st
import pandas as pd
from utils.unicode_map import get_glyph
from rag.knowledge_base import load_all_signs
from rag.retriever import get_vector_store, query_knowledge_base

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hieroglyph Explorer",
    page_icon="𓂀",
    layout="centered"
)

# ──────────────────────────────────────────────────────────────────────────────
# FONT — load Noto Sans Egyptian Hieroglyphs from Google Fonts
# Without this font, browsers may show boxes instead of hieroglyphs on some
# systems. With it, all 760 mapped signs render correctly everywhere.
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Noto+Sans+Egyptian+Hieroglyphs&display=swap" rel="stylesheet">
<style>
.glyph-display {
    font-family: 'Noto Sans Egyptian Hieroglyphs', 'Segoe UI Historic', serif;
    font-size: 96px;
    line-height: 1.1;
    text-align: center;
    padding: 12px 0;
}
.glyph-small {
    font-family: 'Noto Sans Egyptian Hieroglyphs', 'Segoe UI Historic', serif;
    font-size: 36px;
    line-height: 1.2;
}
.field-label {
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #888;
    margin-bottom: 2px;
}
.field-value {
    font-size: 15px;
    margin-bottom: 12px;
}
.field-explain {
    font-size: 12px;
    color: #999;
    font-style: italic;
    margin-top: 2px;
    margin-bottom: 14px;
}
.uncertain-badge {
    display: inline-block;
    background: #fff3cd;
    color: #856404;
    border: 1px solid #ffc107;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 12px;
    font-weight: 500;
    margin-bottom: 12px;
}
.verified-badge {
    display: inline-block;
    background: #d1e7dd;
    color: #0a3622;
    border: 1px solid #a3cfbb;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 12px;
    font-weight: 500;
    margin-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# Cached — loads Excel + CSV once per session.
# Returns a single DataFrame of all 770 signs.
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_data():
    return load_all_signs()

@st.cache_resource
def get_store(df):
    return get_vector_store(df)

df = get_data()
vector_store = get_store(df)

# ──────────────────────────────────────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(
    '<div style="text-align:center; font-family: Noto Sans Egyptian Hieroglyphs, serif; '
    'font-size: 48px; line-height: 1; padding: 8px 0 0;">𓂀 𓇳 𓅓 𓈖 𓋹</div>',
    unsafe_allow_html=True
)
st.title("Hieroglyph Explorer")
st.markdown(
    "Browse all 770 signs from Gardiner's Sign List, or ask anything "
    "about Egyptian hieroglyphs in plain English."
)

# ──────────────────────────────────────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────────────────────────────────────
tab_browse, tab_ask = st.tabs(["📖  Browse Signs", "💬  Ask About Hieroglyphs"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — BROWSE
# Two-step selector: category → sign within that category
# Displays the full sign card with all dataset fields explained
# ══════════════════════════════════════════════════════════════════════════════
with tab_browse:

    st.markdown("### Browse by Category")
    st.caption(
        "Select a category, then choose a sign. "
        "Each category groups signs by the type of object depicted."
    )

    # Step 1: Category selector
    # Build display labels like "A — Man and his occupations (59 signs)"
    categories = (
        df[["category_letter", "category_name"]]
        .drop_duplicates()
        .sort_values("category_letter")
    )
    cat_labels = {}
    for _, row in categories.iterrows():
        letter = row["category_letter"]
        name   = row["category_name"]
        count  = len(df[df["category_letter"] == letter])
        cat_labels[f"{letter} — {name} ({count} signs)"] = letter

    selected_cat_label = st.selectbox(
        "Category",
        options=list(cat_labels.keys()),
        help="Gardiner's Sign List groups hieroglyphs into 25 categories "
             "based on what they depict."
    )
    selected_cat = cat_labels[selected_cat_label]

    # Step 2: Sign selector within that category
    cat_df = df[df["category_letter"] == selected_cat].reset_index(drop=True)

    # Build sign labels like "A1 — Seated man (𓀀)"
    sign_labels = {}
    for _, row in cat_df.iterrows():
        code  = row["gardiner_code"]
        name  = row["english_name"]
        glyph = get_glyph(code)
        sign_labels[f"{code} — {name}  {glyph}"] = code

    selected_sign_label = st.selectbox(
        "Sign",
        options=list(sign_labels.keys()),
        help="Each sign has a unique Gardiner code. "
             "The letter is the category, the number identifies the sign within it."
    )
    selected_code = sign_labels[selected_sign_label]

    # Get the selected sign's row
    sign = df[df["gardiner_code"] == selected_code].iloc[0]

    st.divider()

    # ── Sign Card ─────────────────────────────────────────────────────────────
    # Large glyph display
    glyph_char = get_glyph(selected_code)
    st.markdown(
        f'<div class="glyph-display">{glyph_char}</div>',
        unsafe_allow_html=True
    )

    # Status badge
    if sign["status"] == "Verified":
        st.markdown('<span class="verified-badge">✓ Verified Sign</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="uncertain-badge">⚠ Needs Review</span>', unsafe_allow_html=True)
        if "review_note" in sign and pd.notna(sign.get("review_note")):
            st.caption(f"Review note: {sign['review_note']}")

    # Gardiner Code + Category
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="field-label">Gardiner Code</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="field-value"><strong>{sign["gardiner_code"]}</strong></div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="field-explain">The unique reference code assigned by '
            'Alan Gardiner in his 1927 sign list. The letter indicates category, '
            'the number identifies the specific sign.</div>',
            unsafe_allow_html=True
        )
    with col2:
        st.markdown('<div class="field-label">Category</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="field-value"><strong>{sign["category_letter"]}</strong> — {sign["category_name"]}</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<div class="field-explain">Gardiner grouped all ~750 signs into '
            '25 categories based on what the sign depicts — people, animals, '
            'plants, buildings, etc.</div>',
            unsafe_allow_html=True
        )

    # English Name
    st.markdown('<div class="field-label">English Name</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="field-value"><strong>{sign["english_name"]}</strong></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="field-explain">A plain English description of what '
        'the hieroglyph depicts visually.</div>',
        unsafe_allow_html=True
    )

    # Phonetic Value
    phonetic = sign["phonetic_value"] if pd.notna(sign["phonetic_value"]) else "none"
    st.markdown('<div class="field-label">Phonetic Value</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="field-value"><strong>{phonetic}</strong></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="field-explain">The sound(s) this sign represents when '
        'used as a phonogram — like letters in an alphabet. '
        '"none" means this sign is used for meaning only, not sound.</div>',
        unsafe_allow_html=True
    )

    # Type
    sign_type = sign["type"] if pd.notna(sign["type"]) else "unknown"
    st.markdown('<div class="field-label">Sign Type</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="field-value"><strong>{sign_type}</strong></div>', unsafe_allow_html=True)

    type_explanations = {
        "determinative": "Placed at the end of words to clarify meaning — adds no sound, just context.",
        "phonogram":     "Represents one or more consonant sounds, like a letter.",
        "logogram":      "Represents a whole word or concept directly.",
        "biliteral":     "Represents exactly two consonant sounds together.",
        "triliteral":    "Represents exactly three consonant sounds together.",
        "ideogram":      "Depicts the object it represents and stands for that word.",
    }
    explain = type_explanations.get(sign_type.lower(), "A functional category describing how this sign is used in writing.")
    st.markdown(f'<div class="field-explain">{explain}</div>', unsafe_allow_html=True)

    # Primary Meaning
    st.markdown('<div class="field-label">Primary Meaning</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="field-value">{sign["primary_meaning"]}</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="field-explain">The main function or meaning of this sign '
        'in standard Middle Egyptian (the classical form of the language).</div>',
        unsafe_allow_html=True
    )

    # Secondary Meanings
    if pd.notna(sign["secondary_meanings"]) and sign["secondary_meanings"]:
        st.markdown('<div class="field-label">Secondary Meanings</div>', unsafe_allow_html=True)
        # The field uses | as separator — display as a clean list
        secondaries = [s.strip() for s in str(sign["secondary_meanings"]).split("|")]
        for s in secondaries:
            st.markdown(f'<div class="field-value">• {s}</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="field-explain">Additional uses this sign has beyond '
            'its primary function — hieroglyphs often serve multiple purposes.</div>',
            unsafe_allow_html=True
        )

    # Common Contexts
    if pd.notna(sign["common_contexts"]) and sign["common_contexts"]:
        st.markdown('<div class="field-label">Common Contexts</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="field-value">{sign["common_contexts"]}</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<div class="field-explain">The types of ancient texts and inscriptions '
            'where this sign appears most frequently.</div>',
            unsafe_allow_html=True
        )

    # Notes
    if pd.notna(sign["notes"]) and sign["notes"]:
        st.markdown('<div class="field-label">Scholar Notes</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="field-value">{sign["notes"]}</div>',
            unsafe_allow_html=True
        )

    # Quick search prompt
    st.divider()
    st.caption(
        f"Want to learn more? Switch to the **Ask** tab and type: "
        f"*'Tell me everything about {sign['gardiner_code']} — {sign['english_name']}'*"
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ASK
# Natural language queries → FAISS retrieval → Groq synthesis
# ══════════════════════════════════════════════════════════════════════════════
with tab_ask:

    st.markdown("### Ask About Hieroglyphs")
    st.caption(
        "Ask anything in plain English. The system searches the complete "
        "Gardiner knowledge base and generates a detailed answer."
    )

    # Sample query buttons
    st.markdown("**Try an example:**")
    examples = [
        "What does the owl hieroglyph mean?",
        "Which signs represent water?",
        "Tell me about Gardiner sign G17",
        "What is a determinative?",
        "Which hieroglyph was used for the sound M?",
    ]

    cols = st.columns(len(examples))
    for i, (col, ex) in enumerate(zip(cols, examples)):
        with col:
            if st.button(ex, key=f"ex_{i}", use_container_width=True):
                st.session_state.ask_query = ex
                st.rerun()

    st.markdown("")

    # Text input
    if "ask_query" not in st.session_state:
        st.session_state.ask_query = ""

    query = st.text_input(
        "Your question:",
        key="ask_query",
        placeholder="e.g. What does the seated man hieroglyph represent?",
    )

    if st.button("🔍 Search", type="primary"):
        if query.strip():
            st.session_state.ask_query = query

            with st.spinner("Searching knowledge base..."):
                try:
                    results = query_knowledge_base(
                        query=query,
                        vector_store=vector_store,
                        groq_api_key=st.secrets["GROQ_API_KEY"],
                        df=df,
                    )
                except Exception as e:
                    if "rate_limit" in str(e).lower() or "429" in str(e):
                        st.warning(
                            "⏳ Groq API daily limit reached. "
                            "The Ask tab will work again after midnight US time. "
                            "The Browse tab works normally in the meantime."
                        )
                    else:
                        st.error(f"Something went wrong: {str(e)}")
                    st.stop()

            st.markdown("---")

            # Show matched signs with glyphs
            if results["matched_codes"]:
                st.markdown("**Signs referenced in this answer:**")
                cols = st.columns(min(len(results["matched_codes"]), 5))
                for col, code in zip(cols, results["matched_codes"][:5]):
                    with col:
                        glyph = get_glyph(code)
                        st.markdown(
                            f'<div style="text-align:center; '
                            f'font-family: Noto Sans Egyptian Hieroglyphs, serif; '
                            f'font-size: 48px;">{glyph}</div>'
                            f'<div style="text-align:center; font-size:12px; '
                            f'color:#888;">{code}</div>',
                            unsafe_allow_html=True
                        )

                st.markdown("")

            # Main answer
            st.markdown(results["answer"])

            # Source note
            if results["matched_codes"]:
                source_list = ", ".join(results["matched_codes"])
                st.caption(
                    f"Answer drawn from Gardiner's Sign List entries: {source_list}"
                )
        else:
            st.warning("Please enter a question first.")
