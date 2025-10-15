# One-time script to build the HW7 Chroma index
__import__("pysqlite3")
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
import pandas as pd, os
import chromadb
from openai import OpenAI

CSV_PATH = "/workspaces/Multi-Page-App/Example_news_info_for_testing.csv"
OUT_PATH = "./HW7_DB"
COLLECTION_NAME = "HW7_News"

client = chromadb.PersistentClient(path=OUT_PATH)
collection = client.get_or_create_collection(COLLECTION_NAME)

api_key = os.environ.get("OPENAI_API_KEY")  # or read from .streamlit/secrets.toml
openai_client = OpenAI(api_key=api_key)

df = pd.read_csv(CSV_PATH)
print(f"Building embeddings for {len(df)} rows…")

client.delete_collection(COLLECTION_NAME)
collection = client.get_or_create_collection(COLLECTION_NAME)

for i, row in df.iterrows():
    text = str(row["Document"])
    meta = {
        "company_name": str(row.get("company_name", "")),
        "date": str(row.get("Date", "")),
        "url": str(row.get("URL", "")),
        "days_since_2000": int(row.get("days_since_2000", 0))
        if pd.notna(row.get("days_since_2000", None))
        else 0,
    }
    emb = openai_client.embeddings.create(
        model="text-embedding-3-small", input=text
    ).data[0].embedding
    collection.add(
        ids=[f"doc_{i}"],
        embeddings=[emb],
        documents=[text],
        metadatas=[meta],
    )

print("Finished building HW7_DB/")
