import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import os
import json

load_dotenv()

st.title("TB Auto Group POC")

# -------------------------------
# CoA PREPROCESSING (Pass 1)
# -------------------------------

from pathlib import Path

# Path to fixed CoA file
BASE_DIR = Path(__file__).resolve().parent
coa_path = BASE_DIR / "data" / "data" / "Univ Grouping List Sample.xlsx"

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
    sub_id = int(row["Subgroup ID"])
    subgroup_lookup[sub_id] = {
        "subgroup_name": row["Subgroup"],
        "group": row["Group"],
        "class": row["Class"],
        "type": row["Type"],
    }

st.write("Sample lookup keys:", list(subgroup_lookup.keys())[:5])
st.write("Key type:", type(list(subgroup_lookup.keys())[0]))

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

# ---- Unified category signal config ----
    CATEGORY_SIGNALS = {
        "revenue": {
            "triggers": ["revenue", "sales"],
            "hints": ["revenue", "sales"],
        },
        "payroll": {
            "triggers": ["salary", "wages", "payroll"],
            "hints": ["payroll", "salary", "wages"],
        },
        "cogs": {
            "triggers": ["cost of sales", "cost of goods"],
            "hints": ["cost of sales", "cost of goods"],
        },
        "depreciation": {
            "triggers": ["depreciation"],
            "hints": ["depreciation"],
        },
        "interest": {
            "triggers": ["interest"],
            "hints": ["interest"],
        },
        "tax": {
            "triggers": ["tax"],
            "hints": ["tax"],
        },
        "debt": {
            "triggers": ["loan", "debt", "borrow", "notes payable"],
            "hints": ["loan", "debt", "borrow", "notes payable"],
        },
        "inventory": {
            "triggers": ["inventory", "raw", "finished goods"],
            "hints": ["inventory", "raw", "finished goods"],
        },
        "rent": {
            "triggers": ["rent"],
            "hints": ["rent"],
        },
    }

# -------------------------------
# FEATURE EXTRACTION (Pass 2)
# -------------------------------
    # Normalize account name for analysis
    df["account_name_clean"] = (
        df["Account Name"]
        .astype(str)
        .str.lower()
        .str.replace(r"[^a-z0-9\s]", "", regex=True)
    )

    # ---- Keyword detection (very simple starter list) ----
    def find_keywords(text):
        found = []
        for cat in CATEGORY_SIGNALS.values():
            for w in cat["triggers"]:
                if w in text:
                    found.append(w)
        return found

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
            
            candidates.append({
                "subgroup_id": subgroup_ids[idx],
                "subgroup_name": subgroup_names[idx],
                "score": float(sims[idx])   # ✅ base score
            })
            
        # ---- Category recall guardrail ----
        extra_candidates = []

        for cat in CATEGORY_SIGNALS.values():
            if any(t in account_text for t in cat["triggers"]):
                # ensure subgroup with hint words appears in candidates
                for idx2, name in enumerate(subgroup_names):
                    if any(h in name.lower() for h in cat["hints"]):
                        extra_candidates.append({
                            "subgroup_id": subgroup_ids[idx2],
                            "subgroup_name": subgroup_names[idx2],
                            "score": 0.05  # small base score so it survives shortlist
                        })

        # Add only a few to avoid flooding
        candidates.extend(extra_candidates[:2])

        # dedupe
        seen = {}
        for c in candidates:
            sid = c["subgroup_id"]
            if sid not in seen or c["score"] > seen[sid]["score"]:
                seen[sid] = c
        candidates = list(seen.values())

        # ranking loop with category and lifecycle boosts
        for c in candidates:

            idx_name = c["subgroup_name"].lower()

            # category boost
            for cat in CATEGORY_SIGNALS.values():
                if any(t in account_text for t in cat["triggers"]):
                    if any(h in idx_name for h in cat["hints"]):
                        c["score"] += 0.25

            # lifecycle boost
            if row["lifecycle_flag"]:
                if any(w in idx_name for w in ["prepaid", "accrued", "deferred"]):
                    c["score"] += 0.2
        
        candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)

        score_gap = None
        if len(candidates) > 1:
            score_gap = candidates[0]["score"] - candidates[1]["score"]

        candidate_rows.append({
            "Account Name": row["Account Name"],
            "Top Candidate": candidates[0]["subgroup_name"],
            "Top Candidate ID": candidates[0]["subgroup_id"],
            "Top Score": candidates[0]["score"],
            "Score Gap": score_gap,
            "All Candidates": [
                {"id": c["subgroup_id"], "name": c["subgroup_name"]}
                for c in candidates
            ]
        })

    candidate_df = pd.DataFrame(candidate_rows)

    st.subheader("Candidate Generation Preview")
    st.dataframe(candidate_df.head(20))
    
    st.write("API key loaded:", bool(os.getenv("OPENAI_API_KEY")))

    from openai import OpenAI

    client = OpenAI()


    # ---- Pick ambiguous rows only ----
    ambiguous = candidate_df[candidate_df["Score Gap"] < 0.1].copy().reset_index(drop=True)

    st.write(f"Ambiguous rows sent to LLM: {len(ambiguous)}")
    st.write("First ambiguous candidate IDs:",
         [c["id"] for c in ambiguous.iloc[0]["All Candidates"]])

    # ---- Structured LLM referee ----

    def build_batch_payload(rows):
        payload = []

        for i, r in rows.iterrows():
            payload.append({
                "id": int(i),
                "account": r["Account Name"],
                "candidates": r["All Candidates"]  # now id+name
            })

        return payload

    llm_results = []

    if len(ambiguous) > 0:

        if not os.getenv("OPENAI_API_KEY"):
            st.error("API key missing — LLM disabled")
            st.stop()
          
        payload = build_batch_payload(ambiguous)

        prompt = f"""
        You are an accounting expert.

        For each item choose the best subgroup from candidates.
        If none fit, subgroup = NONE.

        Return JSON array with:
        id
        subgroup_id (must be EXACTLY one of candidate ids or NONE)
        confidence (high, moderate, low)
        rationale (one short sentence explaining why this candidate is best fit for the account.)
        Choose using subgroup_id, not subgroup name.

        Data:
        {payload}
        """

        response = client.responses.create(
            model="gpt-4o-mini",
            temperature=0,
            input=prompt
        )

        text = ""

        for item in response.output:
            if item.type == "message":
                for c in item.content:
                    if c.type == "output_text":
                        text += c.text

        # Parse JSON (no markdown cleaning needed now)
        st.subheader("Structured LLM Response")
        st.code(text, language="json")
        
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
                item["subgroup_id"],
                row.get("low_signal_flag", False)
            )

           # normalize ID coming from LLM
            sub_id = item["subgroup_id"]

            if sub_id == "NONE":
                sub_id_int = None
            else:
                sub_id_int = int(sub_id)

            st.write("Chosen ID:", sub_id_int)
            st.write("Exists in lookup:", sub_id_int in subgroup_lookup)

            # resolve name via lookup
            sub_name = (
                "NONE"
                if sub_id_int is None
                else subgroup_lookup.get(sub_id_int, {}).get("subgroup_name", "UNKNOWN")
            )

            # append row
            final_rows.append({
                "Account": row["Account Name"],
                "Chosen Subgroup": sub_name,
                "Subgroup ID": "NONE" if sub_id_int is None else sub_id_int,
                "Confidence": confidence,
                "Rationale": item["rationale"]
            })

    # ==============================
    # UNIFIED FINAL TABLE ASSEMBLY
    # ==============================

    # Build LLM decision map by account name
    llm_map = {}

    for item in llm_results:
        row = ambiguous.iloc[item["id"]]

        sub_id = None if item["subgroup_id"] == "NONE" else int(item["subgroup_id"])

        llm_map[row["Account Name"]] = {
            "subgroup_id": sub_id,
            "rationale": item["rationale"],
            "confidence": item["confidence"],
        }


    final_records = []

    for _, row in df.iterrows():

        account_name = row["Account Name"]

        # ---------- If row went to LLM ----------
        if account_name in llm_map:

            decision = llm_map[account_name]
            sub_id = decision["subgroup_id"]

            lookup = subgroup_lookup.get(sub_id, {}) if sub_id else {}

            final_records.append({
                **row.to_dict(),
                "Chosen Subgroup": lookup.get("subgroup_name", "NONE"),
                "Group": lookup.get("group"),
                "Class": lookup.get("class"),
                "Type": lookup.get("type"),
                "Confidence": decision["confidence"],
                "Rationale": decision["rationale"],
            })

        # ---------- Deterministic path ----------
        else:

            cand_row = candidate_df[candidate_df["Account Name"] == account_name].iloc[0]

            sub_id_raw = cand_row["Top Candidate ID"]

            sub_id = None if pd.isna(sub_id_raw) else int(sub_id_raw)

            lookup = subgroup_lookup.get(sub_id, {}) if sub_id else {}

            # simple confidence rule
            gap = cand_row["Score Gap"]
            confidence = "high" if gap and gap > 0.15 else "moderate"

            final_records.append({
                **row.to_dict(),
                "Chosen Subgroup": lookup.get("subgroup_name"),
                "Group": lookup.get("group"),
                "Class": lookup.get("class"),
                "Type": lookup.get("type"),
                "Confidence": confidence,
                "Rationale": "Auto-mapped via deterministic signals",
            })

    final_df = pd.DataFrame(final_records)

    st.subheader("Final Mapping Preview")
    st.dataframe(final_df)