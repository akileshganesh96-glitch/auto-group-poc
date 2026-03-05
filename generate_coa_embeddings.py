import pandas as pd
import json
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

BASE_DIR = Path(__file__).resolve().parent

coa_path = BASE_DIR / "data" / "data" / "Univ Grouping List Sample.xlsx"
output_path = BASE_DIR / "data" / "coa_subgroup_embeddings.json"

coa = pd.read_excel(coa_path)
coa = coa.dropna(subset=["Subgroup ID", "Subgroup"]).copy()

# Build text to embed
texts = []
ids = []

for _, row in coa.iterrows():

    subgroup_id = int(str(row["Subgroup ID"]).replace(".0", "").strip())

    text = f"""
    Subgroup: {row['Subgroup']}
    Group: {row['Group']}
    Class: {row['Class']}
    Type: {row['Type']}
    """

    texts.append(text)
    ids.append(subgroup_id)

# Batch embedding
batch_size = 100
embeddings = {}

for i in range(0, len(texts), batch_size):

    batch = texts[i:i+batch_size]

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=batch
    )

    for j, emb in enumerate(response.data):
        embeddings[ids[i+j]] = emb.embedding

# Save embeddings
with open(output_path, "w") as f:
    json.dump(embeddings, f)

print("Saved subgroup embeddings:", output_path)