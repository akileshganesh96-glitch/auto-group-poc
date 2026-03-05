import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import os
import json
import numpy as np
import re
import html
from openai import OpenAI
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

try:
    api_key = st.secrets["OPENAI_API_KEY"]
except:
    api_key = os.environ.get("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

st.title("TB Auto Group POC")

# ============================================================
# SECTION 1: CoA PREPROCESSING (runs once on startup)
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
coa_path = BASE_DIR / "data" / "data" / "Univ Grouping List Sample.xlsx"
embedding_path = BASE_DIR / "data" / "coa_subgroup_embeddings.json"

with open(embedding_path) as f:
    subgroup_embeddings = json.load(f)

subgroup_ids = list(subgroup_embeddings.keys())
subgroup_matrix = np.array(list(subgroup_embeddings.values()))

coa_raw = pd.read_excel(coa_path)
coa_sub = coa_raw.dropna(subset=["Subgroup ID", "Subgroup"]).copy()

coa_sub["Subgroup ID"] = (
    coa_sub["Subgroup ID"]
    .astype(str)
    .str.replace(".0", "", regex=False)
    .str.strip()
)

TYPE_NORMALIZATION = {
    "assets": "asset", "liabilities": "liability",
    "equities": "equity", "revenues": "revenue", "expenses": "expense"
}
coa_sub["Type"] = (
    coa_sub["Type"].astype(str).str.strip().str.lower()
    .replace(TYPE_NORMALIZATION)
)

VALID_TYPES = {"asset", "liability", "equity", "revenue", "expense"}
invalid_types = coa_sub[~coa_sub["Type"].isin(VALID_TYPES)]
if len(invalid_types) > 0:
    st.error("Invalid Type values in CoA:")
    st.dataframe(invalid_types[["Subgroup ID", "Subgroup", "Type"]])
    st.stop()

subgroup_lookup = {}
for _, row in coa_sub.iterrows():
    sub_id = int(row["Subgroup ID"])
    subgroup_lookup[sub_id] = {
        "subgroup_name": row["Subgroup"],
        "group": row["Group"],
        "class": row["Class"],
        "type": row["Type"],
    }

def normalize_text(x):
    return str(x).lower().replace("&", "and").replace("-", " ").strip()

subgroup_name_map = {}
for sid, data in subgroup_lookup.items():
    normalized = normalize_text(data["subgroup_name"])
    subgroup_name_map[normalized] = sid

# ============================================================
# SECTION 2: HELPER FUNCTIONS (defined once, used inside pipeline)
# ============================================================

SYNONYM_MAP = {
    "ar": "accounts receivable",
    "a r": "accounts receivable",
    "ap": "accounts payable",
    "a p": "accounts payable",
    "debtors": "receivable",
    "creditors": "payable",
    "np": "notes payable",
    "r m": "repairs and maintenance",
    "fica": "payroll tax social security",
    "futa": "payroll tax unemployment federal",
    "mesc": "payroll tax unemployment state",
    "cos": "cost of sales",
    "cogs": "cost of goods sold",
    "pp e": "property plant equipment",
    "ppe": "property plant equipment",
}

def preprocess_account_name(raw_name):
    """Clean real-world TB account names before retrieval or LLM calls."""
    name = html.unescape(str(raw_name).strip())

    # Strip QuickBooks numeric prefix e.g. "1005 · "
    name = re.sub(r'^\d+\s*[·•]\s*', '', name)

    # Take last segment of hierarchy path e.g. "Overhead:Accounting" → "Accounting"
    if ':' in name:
        name = name.split(':')[-1].strip()

    # Remove entity name suffixes after " - " if suffix looks like a proper noun
    dash_parts = name.split(' - ')
    if len(dash_parts) > 1:
        suffix = dash_parts[-1].strip()
        suffix_words = suffix.split()
        if len(suffix_words) >= 2 and all(w[0].isupper() for w in suffix_words if w):
            name = ' - '.join(dash_parts[:-1]).strip()

    # Remove parentheses e.g. "Gain(Loss)" → "Gain Loss"
    name = re.sub(r'[()]', ' ', name)

    # Final cleanup
    name = name.lower()
    name = re.sub(r'[^a-z0-9\s]', ' ', name)
    name = ' '.join(name.split())
    return name

def normalize_synonyms(text):
    words = text.split()
    words = [SYNONYM_MAP.get(w, w) for w in words]
    return " ".join(words)

def classify_types_llm(df_batch, industry_context):
    """Classify account types using LLM. Returns list of {id, type} dicts."""
    payload = [
        {
            "id": int(idx),
            "account_number": str(row["Account Number"]),
            "account_name": str(row["account_name_clean"]),
            "balance_sign": (
                "credit" if pd.notna(row["Balance"]) and float(row["Balance"]) < 0
                else "debit" if pd.notna(row["Balance"]) and float(row["Balance"]) > 0
                else "zero"
            )
        }
        for idx, row in df_batch.iterrows()
    ]

    prompt = f"""You are a financial reporting expert with deep knowledge of IFRS and US GAAP.

Industry context: {industry_context}
Use this to interpret account names that may be industry-specific.
In construction, costs like materials, subcontractor costs, permits, and site expenses are Cost of Revenue, not operating expenses.

Classify each account into exactly one of these types:
asset | liability | equity | revenue | expense | unknown

Classification rules:
- asset: economic resources providing future benefit. Includes contra-assets (accumulated depreciation, allowance for doubtful debts).
- liability: present obligations requiring future outflow. Includes deferred revenue, tax payables, accruals, VAT payable, GST payable.
- equity: residual interest after liabilities. Includes share capital, retained earnings, dividends declared, distributions to owners, owner withdrawals.
- revenue: inflows from operations or gains. Includes sales, service income, interest income, gain on disposal, gain on sale.
- expense: outflows incurred to generate revenue. Includes cost of sales, salaries, depreciation, interest expense, tax expense, bad debt expense.
- unknown: ONLY when the account name contains no recognisable accounting terminology whatsoever — for example, a person's name, a property address, or a pure reference code with no descriptive words. If the name contains any accounting term, commit to your best classification.

Critical edge cases — follow these exactly:
- Dividends Paid, Distributions to Owners, Owner Drawings → EQUITY not expense
- Deferred Revenue, Unearned Revenue, Advance Billing → LIABILITY not revenue
- Accumulated Depreciation, Allowance for Doubtful Debts → ASSET (contra-asset)
- VAT Payable, GST Payable, Sales Tax Payable, Income Tax Payable → LIABILITY
- Sales Returns, Returns and Allowances, Sales Discounts → REVENUE (contra-revenue)
- Gain on Disposal, Gain on Sale, Loss on Disposal → REVENUE
- Intercompany Eliminations, Clearing Accounts, Suspense Accounts → UNKNOWN

Note: account_name has been normalized (lowercase, punctuation removed).
Account numbers are provided separately for context only.

Return strict JSON array only, no other text:
[{{"id": 1, "type": "asset"}}, {{"id": 2, "type": "expense"}}]

Accounts to classify:
{json.dumps(payload, indent=2)}"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )

    text = response.choices[0].message.content.strip()

    if "```" in text:
        parts = text.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("["):
                text = part
                break

    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1:
        text = text[start:end+1]

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        st.warning("Type classification JSON parse failed for a batch.")
        return []

def build_deterministic_rationale(account_name, subgroup_name, raw_score, type_match, predicted_type):
    """Explain what drove the deterministic match."""
    account_words = set(account_name.lower().split())
    subgroup_words = set(subgroup_name.lower().split())
    stop = {"and", "or", "the", "a", "an", "of", "for", "in", "on", "to", "other"}
    common_words = (account_words & subgroup_words) - stop

    score_pct = int(raw_score * 100)

    if common_words:
        word_match = f"Matched on: {', '.join(sorted(common_words))}"
    else:
        word_match = f"Matched via semantic similarity to '{subgroup_name}'"

    type_note = f"Type confirmed as {predicted_type} by LLM"
    return f"{word_match} | {type_note} | Similarity: {score_pct}%"

# ============================================================
# SECTION 3: UI — UPLOAD AND INDUSTRY CONTEXT
# ============================================================

industry_context = st.text_input(
    "Company industry (optional — helps classify industry-specific accounts)",
    value="Construction and infrastructure company"
)

uploaded_file = st.file_uploader("Upload Trial Balance (Excel)", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)

    with st.expander("Trial Balance Preview", expanded=False):
        st.dataframe(df)

    if not os.getenv("OPENAI_API_KEY"):
        st.error("API key missing")
        st.stop()

    # ============================================================
    # SECTION 4: MAIN PIPELINE inside st.status
    # ============================================================

    SCORE_THRESHOLD = 0.22
    GAP_THRESHOLD = 0.01

    with st.status("Analysing your Trial Balance...", expanded=True) as status:

        # --- Step 1: Clean account names ---
        st.write("🧹 Cleaning and normalising account names...")
        df["account_name_clean"] = df["Account Name"].apply(preprocess_account_name)
        df["account_name_clean"] = df["account_name_clean"].apply(normalize_synonyms)
        st.write(f"✅ {len(df)} accounts cleaned and normalised")

        # --- Step 2: Exact lexical match ---
        st.write("🔎 Checking for exact matches against Chart of Accounts...")
        df["lexical_subgroup_id"] = None
        for i, row in df.iterrows():
            normalized_account = normalize_text(row["account_name_clean"])
            if normalized_account in subgroup_name_map:
                df.at[i, "lexical_subgroup_id"] = subgroup_name_map[normalized_account]

        lexical_matches = df[df["lexical_subgroup_id"].notna()].copy()
        remaining_df = df[df["lexical_subgroup_id"].isna()].copy().reset_index(drop=True)
        st.write(f"✅ {len(lexical_matches)} exact matches found — {len(remaining_df)} accounts proceeding to AI pipeline")

        # --- Step 3: LLM type classification ---
        st.write("🤖 Classifying account types using AI...")
        all_type_results = []
        for i in range(0, len(remaining_df), 50):
            batch = remaining_df.iloc[i:i+50]
            all_type_results.extend(classify_types_llm(batch, industry_context))

        type_map = {r["id"]: r["type"] for r in all_type_results}
        remaining_df["predicted_type"] = remaining_df.index.map(type_map).fillna("unknown")

        type_counts = remaining_df["predicted_type"].value_counts().to_dict()
        type_summary = ", ".join([f"{v} {k}" for k, v in type_counts.items() if k != "unknown"])
        pending = type_counts.get("unknown", 0)
        st.write(f"✅ Type classification complete — {type_summary}, {pending} pending review")

        # --- Step 4: Semantic embeddings ---
        st.write("🔍 Generating semantic embeddings for account matching...")
        account_names = list(remaining_df["account_name_clean"])
        account_vectors = []
        for i in range(0, len(account_names), 200):
            batch = account_names[i:i+200]
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=batch
            )
            for emb in response.data:
                account_vectors.append(emb.embedding)
        account_vectors = np.array(account_vectors)
        similarity_matrix = cosine_similarity(account_vectors, subgroup_matrix)
        st.write(f"✅ Embeddings complete — matching {len(remaining_df)} accounts against {len(coa_sub)} subgroups")

        # --- Step 5: Candidate generation with soft containment ---
        st.write("⚖️ Running candidate retrieval and deterministic selection...")
        candidate_rows = []

        for vec_idx, (df_idx, row) in enumerate(remaining_df.iterrows()):
            sims = similarity_matrix[vec_idx]
            scores = sims.copy()
            predicted_type = row["predicted_type"]

            # Soft containment — boost candidates matching predicted type
            if predicted_type in VALID_TYPES:
                for i, sid in enumerate(subgroup_ids):
                    cand_type = subgroup_lookup[int(sid)]["type"]
                    if cand_type == predicted_type:
                        scores[i] *= 2.0

            top_idx = np.argsort(scores)[-7:][::-1]
            candidates = []

            for i in top_idx:
                subgroup_id = int(subgroup_ids[i])
                raw_score = float(sims[i])
                boosted_score = float(scores[i])
                cand_type = subgroup_lookup[subgroup_id]["type"]

                candidates.append({
                    "subgroup_id": subgroup_id,
                    "subgroup_name": subgroup_lookup[subgroup_id]["subgroup_name"],
                    "score": boosted_score,
                    "raw_score": raw_score,
                    "type": cand_type,
                    "type_match": cand_type == predicted_type
                })

            candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
            candidate_rows.append({
                "account_number": row["Account Number"],
                "account_name": row["Account Name"],
                "balance": row.get("Balance"),
                "predicted_type": predicted_type,
                "candidates": candidates
            })

        # --- Step 6: Deterministic selection ---
        final_rows = []

        for row in candidate_rows:
            candidates = row["candidates"]
            predicted_type = row["predicted_type"]

            if predicted_type == "unknown" or predicted_type not in VALID_TYPES:
                final_rows.append({
                    "Account Number": row["account_number"],
                    "Account Name": row["account_name"],
                    "Balance": row.get("balance"),
                    "Predicted Type": predicted_type,
                    "Selected Subgroup": None,
                    "Subgroup ID": None,
                    "Confidence": "low",
                    "Escalate to LLM": False,
                    "Rationale": "Type could not be determined — flagged for manual review"
                })
                continue

            if not candidates:
                final_rows.append({
                    "Account Number": row["account_number"],
                    "Account Name": row["account_name"],
                    "Balance": row.get("balance"),
                    "Predicted Type": predicted_type,
                    "Selected Subgroup": None,
                    "Subgroup ID": None,
                    "Confidence": "low",
                    "Escalate to LLM": True,
                    "Rationale": "No subgroup candidates found — escalated for review"
                })
                continue

            top1 = candidates[0]
            top2 = candidates[1] if len(candidates) > 1 else None
            gap = (top1["score"] - top2["score"]) if top2 else None

            if top1["score"] < SCORE_THRESHOLD * 2:
                escalate = True
                confidence = "low"
            elif gap is not None and gap < GAP_THRESHOLD:
                escalate = True
                confidence = "moderate"
            else:
                escalate = False
                confidence = "high"

            rationale = (
                build_deterministic_rationale(
                    row["account_name"], top1["subgroup_name"],
                    top1["raw_score"], top1["type_match"], predicted_type
                )
                if not escalate
                else "Score gap too narrow — escalated to AI referee for disambiguation"
            )

            final_rows.append({
                "Account Number": row["account_number"],
                "Account Name": row["account_name"],
                "Balance": row.get("balance"),
                "Predicted Type": predicted_type,
                "Selected Subgroup": top1["subgroup_name"],
                "Subgroup ID": top1["subgroup_id"],
                "Confidence": confidence,
                "Escalate to LLM": escalate,
                "Rationale": rationale
            })

        final_df = pd.DataFrame(final_rows)

        escalated_count = int(final_df["Escalate to LLM"].sum())
        confident_count = len(final_df) - escalated_count
        st.write(f"✅ Deterministic selection complete — {confident_count} mapped, {escalated_count} escalated to AI referee")

        auto_mapped = len(final_df[final_df["Subgroup ID"].notna()])
        manual_count = len(final_df[final_df["Subgroup ID"].isna()])

        status.update(
            label=f"✅ First pass complete — {auto_mapped + len(lexical_matches)} accounts mapped, {manual_count} pending AI review or manual review",
            state="complete",
            expanded=False
        )

    # ============================================================
    # SECTION 5: LLM REFEREE (button-triggered, outside status block)
    # ============================================================

    if "final_df" not in st.session_state:
        st.session_state.final_df = final_df

    if "candidate_rows" not in st.session_state:
        st.session_state.candidate_rows = candidate_rows

    if escalated_count > 0:
        if "llm_run" not in st.session_state:
            st.session_state.llm_run = False

        if st.button(f"🤖 Run AI Referee for {escalated_count} Ambiguous Accounts"):
            st.session_state.llm_run = True

        if st.session_state.llm_run:
            ambiguous_df = final_df[st.session_state.final_df["Escalate to LLM"] == True].copy().reset_index(drop=True)

            payload = []
            for i, row in ambiguous_df.iterrows():
                orig = next(
                    (x for x in st.session_state.candidate_rows if x["account_name"] == row["Account Name"]),
                    None
                )
                if orig is None or not orig["candidates"]:
                    payload.append({
                        "id": i,
                        "account_number": str(row["Account Number"]),
                        "account_name": row["Account Name"],
                        "balance_sign": "credit" if pd.notna(orig["balance"] if orig else None) and float(orig["balance"]) < 0 else "debit",
                        "predicted_type": row["Predicted Type"],
                        "candidates": []
                    })
                    continue

                payload.append({
                    "id": i,
                    "account_number": str(row["Account Number"]),
                    "account_name": row["Account Name"],
                    "balance": orig.get("balance"),
                    "predicted_type": row["Predicted Type"],
                    "candidates": [
                        {
                            "subgroup_id": c["subgroup_id"],
                            "subgroup_name": c["subgroup_name"],
                            "type": c["type"]
                        }
                        for c in orig["candidates"]
                    ]
                })

            if payload:
                with st.spinner("Running AI referee on ambiguous accounts..."):
                    prompt = f"""You are a senior accountant with expertise in IFRS and US GAAP financial statement preparation.

Industry context: {industry_context}

For each account below, select the most appropriate subgroup from the candidate list.

Rules:
- You MUST choose from the provided candidates only.
- Use the account name, account number, balance sign, and predicted type together.
- A negative balance may indicate a contra account or credit-nature account.
- If none of the candidates are appropriate, return subgroup_id as "NONE".
- Do NOT invent subgroup names. Only use the exact subgroup_id values provided.

Return strict JSON array only:
[
  {{
    "id": <id>,
    "subgroup_id": "<exact id from candidates or NONE>",
    "confidence": "high|moderate|low",
    "rationale": "<one sentence: state what the account represents and why this subgroup fits under IFRS/GAAP, under 25 words>"
  }}
]

Accounts:
{json.dumps(payload, indent=2)}"""

                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        temperature=0,
                        messages=[{"role": "user", "content": prompt}]
                    )

                    text = response.choices[0].message.content.strip()

                    if "```" in text:
                        parts = text.split("```")
                        for part in parts:
                            part = part.strip()
                            if part.startswith("json"):
                                part = part[4:].strip()
                            if part.startswith("["):
                                text = part
                                break

                    start = text.find("[")
                    end = text.rfind("]")
                    if start != -1 and end != -1:
                        text = text[start:end+1]

                    try:
                        llm_results = json.loads(text)
                    except json.JSONDecodeError:
                        llm_results = []
                        st.error("AI referee JSON parse failed.")

                    for item in llm_results:
                        idx = item.get("id")
                        chosen_id = str(item.get("subgroup_id", "NONE"))

                        if idx is None or idx >= len(ambiguous_df):
                            continue

                        orig = next(
                            (x for x in candidate_rows
                             if x["account_name"] == ambiguous_df.loc[idx, "Account Name"]),
                            None
                        )
                        valid_ids = {str(c["subgroup_id"]) for c in orig["candidates"]} if orig else set()

                        if chosen_id == "NONE" or chosen_id not in valid_ids:
                            ambiguous_df.loc[idx, "Selected Subgroup"] = None
                            ambiguous_df.loc[idx, "Subgroup ID"] = None
                            ambiguous_df.loc[idx, "Confidence"] = "low"
                            ambiguous_df.loc[idx, "Rationale"] = (
                                "No suitable subgroup found — flagged for manual review"
                                if chosen_id == "NONE"
                                else "AI returned invalid candidate — flagged for manual review"
                            )
                        else:
                            sub_id_int = int(chosen_id)
                            ambiguous_df.loc[idx, "Subgroup ID"] = sub_id_int
                            ambiguous_df.loc[idx, "Selected Subgroup"] = subgroup_lookup[sub_id_int]["subgroup_name"]
                            ambiguous_df.loc[idx, "Confidence"] = item.get("confidence", "moderate")
                            ambiguous_df.loc[idx, "Rationale"] = item.get("rationale", "AI referee decision")

                    # Fallback for any escalated rows with no LLM response
                    responded_ids = {item.get("id") for item in llm_results}
                    for i, row in ambiguous_df.iterrows():
                        if i not in responded_ids:
                            ambiguous_df.loc[i, "Rationale"] = "Escalated — no AI response received, flagged for manual review"
                            ambiguous_df.loc[i, "Confidence"] = "low"

                    # Merge back into final_df
                    for _, resolved_row in ambiguous_df.iterrows():
                        mask = final_df["Account Name"] == resolved_row["Account Name"]
                        for col in ["Selected Subgroup", "Subgroup ID", "Confidence", "Rationale"]:
                            final_df.loc[mask, col] = resolved_row[col]

                st.success(f"✅ AI referee complete")

    # ============================================================
    # SECTION 6: FINAL TABLE ASSEMBLY
    # ============================================================

    final_records = []
    for _, row in st.session_state.final_df.iterrows():
        sub_id = row.get("Subgroup ID")
        try:
            sub_id_int = int(sub_id) if sub_id is not None and not pd.isna(sub_id) else None
        except (ValueError, TypeError):
            sub_id_int = None

        lookup = subgroup_lookup.get(sub_id_int, {}) if sub_id_int else {}

        final_records.append({
            "Account Number": row.get("Account Number"),
            "Account Name": row.get("Account Name"),
            "Balance": row.get("Balance"),
            "Type": lookup.get("type") if lookup else row.get("Predicted Type"),
            "Class": lookup.get("class"),
            "Group": lookup.get("group"),
            "Subgroup": lookup.get("subgroup_name"),
            "Confidence": row.get("Confidence"),
            "Rationale": row.get("Rationale"),
        })

    # Assemble lexical match rows
    lexical_rows = []
    for _, row in lexical_matches.iterrows():
        sid = int(row["lexical_subgroup_id"])
        lookup = subgroup_lookup[sid]
        lexical_rows.append({
            "Account Number": row["Account Number"],
            "Account Name": row["Account Name"],
            "Balance": row.get("Balance"),
            "Type": lookup["type"],
            "Class": lookup["class"],
            "Group": lookup["group"],
            "Subgroup": lookup["subgroup_name"],
            "Confidence": "high",
            "Rationale": f"Exact match to '{lookup['subgroup_name']}' in Chart of Accounts",
        })

    lexical_df = pd.DataFrame(lexical_rows)
    ai_output_df = pd.DataFrame(final_records)
    final_output_df = pd.concat([lexical_df, ai_output_df], ignore_index=True)

    # ============================================================
    # SECTION 7: RESULTS DISPLAY
    # ============================================================

    auto_count = final_output_df["Subgroup"].notna().sum()
    manual_review_count = final_output_df["Subgroup"].isna().sum()
    high_conf = (final_output_df["Confidence"] == "high").sum()
    moderate_conf = (final_output_df["Confidence"] == "moderate").sum()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Accounts", len(final_output_df))
    col2.metric("Auto-Mapped", auto_count)
    col3.metric("High Confidence", high_conf)
    col4.metric("Manual Review", manual_review_count)

    # Color code confidence column
    def color_confidence(val):
        colors = {
            "high": "background-color: #1a472a; color: #a3d9a5",
            "moderate": "background-color: #4a3800; color: #ffd966",
            "low": "background-color: #4a1010; color: #f4a4a4",
        }
        return colors.get(val, "")

    styled_df = final_output_df.style.applymap(color_confidence, subset=["Confidence"])

    st.subheader("Final Mapping")
    st.dataframe(styled_df, use_container_width=True, height=600)

    csv = final_output_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Mapping CSV",
        data=csv,
        file_name="tb_mapping_output.csv",
        mime="text/csv"
    )