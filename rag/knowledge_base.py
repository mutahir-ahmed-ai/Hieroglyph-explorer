import pandas as pd
from langchain_core.documents import Document

def load_all_signs() -> pd.DataFrame:
    """
    Load all signs from the Excel file and CSV into a single DataFrame.
    The Excel has verified signs across multiple sheets.
    The CSV has uncertain/needs-review signs.
    Returns a unified DataFrame of all 770 signs.
    """
    # Load verified signs from Excel
    df_verified = pd.read_excel(
        "Gardiner_Sign_List_COMPLETE.xlsx",
        sheet_name="Verified Signs"
    )

    # Load uncertain signs from CSV
    df_uncertain = pd.read_csv("Gardiner_Uncertain_Signs.csv")

    # Combine — uncertain signs have an extra 'review_note' column
    # Fill missing review_note for verified signs with None
    if "review_note" not in df_verified.columns:
        df_verified["review_note"] = None

    df = pd.concat([df_verified, df_uncertain], ignore_index=True)
    return df


def build_documents(df: pd.DataFrame) -> list[Document]:
    """
    Convert each sign row into a LangChain Document for FAISS embedding.

    Each document's page_content is a plain-text summary of the sign —
    written so that semantic search finds it naturally. A user who asks
    "what makes an owl sound" will find G17 because the content mentions
    "owl" and "phonetic m" in natural language.

    The metadata carries the gardiner_code so we can retrieve it after search.
    """
    documents = []

    for _, row in df.iterrows():
        code     = row.get("gardiner_code", "")
        cat_name = row.get("category_name", "")
        eng_name = row.get("english_name", "")
        phonetic = row.get("phonetic_value", "none")
        sign_type= row.get("type", "")
        primary  = row.get("primary_meaning", "")
        secondary= row.get("secondary_meanings", "")
        contexts = row.get("common_contexts", "")
        notes    = row.get("notes", "")
        status   = row.get("status", "")

        # Build a natural-language paragraph about this sign.
        # This is what gets embedded and searched semantically.
        parts = [
            f"Gardiner sign {code} is called the {eng_name}.",
            f"It belongs to category {cat_name}.",
        ]

        if phonetic and str(phonetic).lower() not in ("none", "nan", ""):
            parts.append(f"Its phonetic value is '{phonetic}', meaning it represents that sound in writing.")
        else:
            parts.append("It has no phonetic value and is used purely as a determinative.")

        if sign_type:
            parts.append(f"It functions as a {sign_type}.")

        if primary:
            parts.append(f"Primary meaning: {primary}.")

        if secondary and str(secondary).lower() not in ("none", "nan", ""):
            parts.append(f"Secondary uses include: {secondary}.")

        if contexts and str(contexts).lower() not in ("none", "nan", ""):
            parts.append(f"Commonly found in: {contexts}.")

        if notes and str(notes).lower() not in ("none", "nan", ""):
            parts.append(f"Notes: {notes}.")

        if status == "Needs Review":
            parts.append("Note: this sign is uncertain and needs further scholarly review.")

        page_content = " ".join(parts)

        documents.append(Document(
            page_content=page_content,
            metadata={
                "gardiner_code": code,
                "english_name":  eng_name,
                "status":        status,
            }
        ))

    return documents
