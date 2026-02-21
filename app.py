import streamlit as st
import pandas as pd

st.title("TB Auto Group POC")

# -------------------------------
# CoA PREPROCESSING (Pass 1)
# -------------------------------

# Path to fixed CoA file
coa_path = "data/data/Univ Grouping List Sample.xlsx"

# Load raw CoA
coa_raw = pd.read_excel(coa_path)

st.subheader("Raw CoA Preview")
st.dataframe(coa_raw.head())


# ---- STEP 1: Keep only valid subgroup rows ----
# We only want leaf nodes where Subgroup ID and Subgroup name exist

coa_sub = coa_raw.dropna(subset=["Subgroup ID", "Subgroup"]).copy()


# ---- STEP 2: Normalize subgroup IDs ----
# Excel often reads numeric IDs as floats (1010.0)
# We convert everything to string without decimal

coa_sub["Subgroup ID"] = coa_sub["Subgroup ID"].astype(str).str.replace(".0", "", regex=False)


# ---- STEP 3: Validate duplicates ----
duplicate_mask = coa_sub["Subgroup ID"].duplicated(keep=False)

duplicates = coa_sub[duplicate_mask]

if len(duplicates) > 0:
    st.error(f"Duplicate Subgroup IDs detected: {duplicates['Subgroup ID'].nunique()}")
    st.write("Duplicate rows:")
    st.dataframe(duplicates[["Subgroup ID", "Subgroup", "Group", "Class", "Type"]])
else:
    st.success("No duplicate Subgroup IDs")


# ---- STEP 4: Build subgroup lookup table ----
# This becomes the canonical taxonomy used everywhere

subgroup_lookup = {}

for _, row in coa_sub.iterrows():
    subgroup_lookup[row["Subgroup ID"]] = {
        "subgroup_name": row["Subgroup"],
        "group": row["Group"],
        "class": row["Class"],
        "type": row["Type"],
    }


# ---- STEP 5: Create subgroup description text ----
# This text will later be used for embeddings + LLM context

coa_sub["description"] = (
    coa_sub["Subgroup"]
    + " — subgroup under "
    + coa_sub["Group"]
    + ", "
    + coa_sub["Class"]
    + ", "
    + coa_sub["Type"]
)


# ---- STEP 6: Show cleaned subgroup universe ----
st.subheader("Valid Subgroups (Engine Target Space)")
st.write(f"Total subgroups: {len(coa_sub)}")
st.dataframe(coa_sub[["Subgroup ID", "Subgroup", "Group", "Class", "Type"]].head())

# ---- User Upload for Trial Balance ----
uploaded_file = st.file_uploader("Upload Trial Balance (Excel)", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.subheader("Preview")
    st.dataframe(df)

# -------------------------------
# FEATURE EXTRACTION (Pass 2)
# -------------------------------

if uploaded_file:

    df = pd.read_excel(uploaded_file)

    # Normalize account name for analysis
    df["account_name_clean"] = (
        df["Account Name"]
        .astype(str)
        .str.lower()
        .str.replace(r"[^a-z0-9\s]", "", regex=True)
    )

    # ---- Keyword detection (very simple starter list) ----
    keywords = ["rent", "salary", "insurance", "advance", "deposit", "loan"]

    def find_keywords(text):
        return [k for k in keywords if k in text]

    df["keywords_found"] = df["account_name_clean"].apply(find_keywords)

    # ---- Lifecycle flag ----
    lifecycle_words = ["prepaid", "accrued", "advance", "deferred"]

    df["lifecycle_flag"] = df["account_name_clean"].apply(
        lambda x: any(w in x for w in lifecycle_words)
    )

    # ---- Generic / low-signal flag ----
    generic_words = ["misc", "others", "sundry", "general", "clearing", "suspense"]

    df["low_signal_flag"] = df["account_name_clean"].apply(
        lambda x: any(w in x for w in generic_words)
    )

    # ---- Account number band detection (soft) ----
    df["number_band"] = df["Account Number"].astype(str).str[0]

    # ---- Balance direction ----
    df["balance_direction"] = df["Balance"].apply(
        lambda x: "debit" if x >= 0 else "credit"
    )

    # ---- Anomaly flag (very basic starter logic) ----
    # Example: credit balance in 1xxx (asset band) looks suspicious
    df["anomaly_flag"] = (
        (df["number_band"] == "1") & (df["balance_direction"] == "credit")
    )

    st.subheader("Feature Extraction Preview")
    st.dataframe(
        df[
            [
                "Account Number",
                "Account Name",
                "Balance",
                "keywords_found",
                "lifecycle_flag",
                "low_signal_flag",
                "number_band",
                "balance_direction",
                "anomaly_flag",
            ]
        ].head(20)
    )

    # Accounting keyword → subgroup hint words
    accounting_boosts = {
        "revenue": ["revenue", "sales"],
        "sale": ["revenue", "sales"],
        "salary": ["payroll", "salary", "wages"],
        "wage": ["payroll", "salary", "wages"],
        "cogs": ["cost of sales", "cost of goods"],
        "cost of goods": ["cost of sales"],
        "depreciation": ["depreciation"],
        "interest": ["interest"],
        "tax": ["tax"],
        "rent": ["rent"],
        "inventory": ["inventory", "raw", "finished goods"],
    }

    category_triggers = {
    "revenue": ["revenue", "sales"],
    "salary": ["payroll", "salary", "wages"],
    "wage": ["payroll", "salary", "wages"],
    "cogs": ["cost of sales", "cost of revenue"],
    "cost of goods": ["cost of sales", "cost of revenue"],
    "depreciation": ["depreciation"],
    "interest": ["interest"],
    "tax": ["tax"],
    "debt": ["loan", "debt", "borrow", "notes payable"],
    "loan": ["loan", "debt", "borrow", "notes payable"], 
    }

    # -------------------------------
    # CANDIDATE GENERATION (Lightweight)
    # -------------------------------

    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np

    # Prepare subgroup texts (from CoA preprocessing)
    subgroup_texts = coa_sub["description"].tolist()
    subgroup_ids = coa_sub["Subgroup ID"].tolist()
    subgroup_names = coa_sub["Subgroup"].tolist()

    # Fit TF-IDF on subgroup descriptions once
    vectorizer = TfidfVectorizer()
    subgroup_matrix = vectorizer.fit_transform(subgroup_texts)

    candidate_rows = []

    for _, row in df.iterrows():

        account_text = row["account_name_clean"]

        acc_vec = vectorizer.transform([account_text])

        # Cosine similarity
        sims = (subgroup_matrix @ acc_vec.T).toarray().ravel()

        # Top 5 candidates
        top_idx = np.argsort(sims)[-5:][::-1]

        candidates = []

        for idx in top_idx:
            score = float(sims[idx])

            # ---- Accounting keyword boost ----
            for trigger, hints in accounting_boosts.items():
                if trigger in account_text:
                    if any(h in subgroup_names[idx].lower() for h in hints):
                        score += 0.25

            # ---- lifecycle boost ----
            if row["lifecycle_flag"]:
                if any(w in subgroup_names[idx].lower() for w in ["prepaid", "accrued", "deferred"]):
                    score += 0.2

            candidates.append({
                "subgroup_id": subgroup_ids[idx],
                "subgroup_name": subgroup_names[idx],
                "score": score
            })
            
        # ---- Category recall guardrail ----
        extra_candidates = []

        for trigger, hints in category_triggers.items():
            if trigger in account_text:
                for i, name in enumerate(subgroup_names):
                    if any(h in name.lower() for h in hints):

                        # avoid duplicates
                        if not any(c["subgroup_id"] == subgroup_ids[i] for c in candidates):

                            extra_candidates.append({
                                "subgroup_id": subgroup_ids[i],
                                "subgroup_name": subgroup_names[i],
                                "score": 0.35  # medium baseline so it enters shortlist
                            })

        # Add only a few to avoid flooding
        candidates.extend(extra_candidates[:2])
        
        candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)

        score_gap = None
        if len(candidates) > 1:
            score_gap = candidates[0]["score"] - candidates[1]["score"]

        candidate_rows.append({
            "Account Name": row["Account Name"],
            "Top Candidate": candidates[0]["subgroup_name"],
            "Top Score": candidates[0]["score"],
            "Score Gap": score_gap,
            "All Candidates": [c["subgroup_name"] for c in candidates]
        })

    candidate_df = pd.DataFrame(candidate_rows)

    st.subheader("Candidate Generation Preview")
    st.dataframe(candidate_df.head(20))

    import os
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv()
    client = OpenAI()


    # ---- Pick ambiguous rows only ----
    ambiguous = candidate_df[candidate_df["Score Gap"] < 0.1].copy().reset_index(drop=True)

    st.write(f"Ambiguous rows sent to LLM: {len(ambiguous)}")


    # ---- Structured LLM referee ----

    def build_batch_payload(rows):
        payload = []

        for i, r in rows.iterrows():
            payload.append({
                "id": int(i),
                "account": r["Account Name"],
                "candidates": r["All Candidates"]
            })

        return payload


    if len(ambiguous) > 0:

        payload = build_batch_payload(ambiguous)

        prompt = f"""
        You are an accounting expert.

        For each item choose the best subgroup from candidates.
        If none fit, subgroup = NONE.

        Return JSON array with:
        id
        subgroup
        confidence (high, moderate, low)
        rationale (one short sentence)

        Data:
        {payload}
        """

        response = client.responses.create(
            model="gpt-4.1-mini",
            input=prompt,
            temperature=0
        )

        text = response.output[0].content[0].text

        st.subheader("Structured LLM Response")
        st.code(text, language="json")
        
        import json

        clean_text = text.strip()

        if clean_text.startswith("```"):
            clean_text = clean_text.split("```")[1]
            if clean_text.startswith("json"):
                clean_text = clean_text[4:].strip()

        try:
            llm_results = json.loads(clean_text)
        except:
            llm_results = []   

        def compute_confidence(score_gap, subgroup, low_signal):

            if subgroup == "NONE":
                return "low"

            if score_gap is not None and score_gap > 0.15:
                return "high"

            if low_signal:
                return "low"

            return "moderate"

        final_rows = []

        for item in llm_results:

            row = ambiguous.loc[item["id"]]

            confidence = compute_confidence(
                row["Score Gap"],
                item["subgroup"],
                row.get("low_signal_flag", False)
            )

            final_rows.append({
                "Account": row["Account Name"],
                "Chosen Subgroup": item["subgroup"],
                "Confidence": confidence,
                "Rationale": item["rationale"]
            })

        final_df = pd.DataFrame(final_rows)

    st.subheader("Final Mapping Preview")
    st.dataframe(final_df)