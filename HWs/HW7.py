# HW7 — News Info Bot (final load-only version)
# -------------------------------------------------
# Loads prebuilt ChromaDB (HW7_DB) + answers:
#  1) “Find the most interesting news”
#  2) “Find news about <topic>”
# Uses RAG + prompt-engineering for a global law firm context

__import__("pysqlite3")
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import os, requests, pandas as pd, streamlit as st, chromadb
from openai import OpenAI

# ----------------------------
# Streamlit setup
# ----------------------------
st.set_page_config(page_title="HW7 — News Info Bot", layout="wide")
st.title("HW7 — News Info Bot")

st.markdown("""
This bot analyzes a CSV of **news stories** for a **global law firm**.

It can:
1. 🧭 *Find the most interesting legal news* (ranked list)  
2. 🗞️ *Find news about a specific topic*  
using **RAG + LLMs (OpenAI / Mistral)**.
""")

# ----------------------------
# Load CSV (for reference)
# ----------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CSV_PATH = os.path.join(BASE_DIR, "Example_news_info_for_testing.csv")

st.write("📂 Looking for CSV at:", CSV_PATH)
if not os.path.exists(CSV_PATH):
    st.error("❌ CSV file not found! Listing current directory:")
    st.write(os.listdir(BASE_DIR))
    st.stop()
else:
    df = pd.read_csv(CSV_PATH)
    st.success("✅ CSV loaded successfully!")

# ----------------------------
# Load prebuilt Chroma database
# ----------------------------
CHROMA_PATH = "./HW7_DB"
COLLECTION_NAME = "HW7_News"
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(COLLECTION_NAME)
st.success(f"Loaded prebuilt ChromaDB collection: {COLLECTION_NAME}")

# ----------------------------
# Setup OpenAI + helper functions
# ----------------------------
openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

LAW_KEYWORDS = [
    "lawsuit","litigation","sued","settlement","regulation","regulatory","fine",
    "penalty","CFPB","SEC","DOJ","FTC","antitrust","compliance","investigation",
    "merger","acquisition","bankruptcy","governance","sanction","enforcement"
]

def interest_score(text, days_since):
    """Keyword count + recency boost."""
    t = text.lower()
    k = sum(1 for kw in LAW_KEYWORDS if kw in t)
    recency = 1.0 - min(max(days_since, 0), 9500) / 9500.0
    return k * 1.0 + recency * 0.5

def retrieve_news(query, n=8):
    """Return top-n relevant news for a topic."""
    q_emb = openai_client.embeddings.create(
        model="text-embedding-3-small", input=query
    ).data[0].embedding
    res = collection.query(query_embeddings=[q_emb], n_results=n)
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    results = []
    for d, m in zip(docs, metas):
        results.append({
            "text": d,
            "company_name": m.get("company_name", ""),
            "date": m.get("date", ""),
            "url": m.get("url", ""),
            "days_since_2000": m.get("days_since_2000", 0),
        })
    return results

def most_interesting_news(k=8):
    """Rank by heuristic interest_score."""
    all_items = retrieve_news(
        "legal, regulatory, litigation, compliance, antitrust, penalties", n=40
    )
    for r in all_items:
        r["score"] = interest_score(r["text"], r["days_since_2000"])
    ranked = sorted(all_items, key=lambda x: x["score"], reverse=True)[:k]
    return ranked

def call_llm(vendor, model, messages):
    if vendor == "OpenAI":
        resp = openai_client.chat.completions.create(model=model, messages=messages)
        return resp.choices[0].message.content
    else:
        headers = {
            "Authorization": f"Bearer {st.secrets['MISTRAL_API_KEY']}",
            "Content-Type": "application/json",
        }
        payload = {"model": model, "messages": messages, "stream": False}
        r = requests.post("https://api.mistral.ai/v1/chat/completions",
                          headers=headers, json=payload, timeout=60)
        data = r.json()
        return data.get("choices", [{}])[0].get("message", {}).get("content", "[Mistral error]")

def summarize_for_law_firm(vendor, model, snippets, task):
    context = "\n\n---\n\n".join(
        [f"Company: {s['company_name']} | Date: {s['date']}\n{s['text'][:600]}" for s in snippets]
    )
    messages = [
        {"role": "system",
         "content": "You are a legal news analyst for a global law firm. Focus on legal, regulatory, and compliance implications."},
        {"role": "user",
         "content": f"Task: {task}\n\nUse only this context:\n{context}"}
    ]
    return call_llm(vendor, model, messages)

# ----------------------------
# Bot interface
# ----------------------------
st.subheader("Ask the Bot")

col1, col2 = st.columns(2)
with col1:
    task = st.radio("Select task:", ["Most interesting news", "News about a topic"])
with col2:
    vendor = st.selectbox("Vendor:", ["OpenAI", "Mistral"])
    model = st.selectbox(
        "Model:",
        ["gpt-5-mini", "gpt-4o-mini"] if vendor == "OpenAI"
        else ["mistral-small-latest", "mistral-large-latest"]
    )

topic = ""
if task == "News about a topic":
    topic = st.text_input("Enter topic (e.g. CFPB, merger, antitrust):")

if st.button("Run"):
    if task == "Most interesting news":
        ranked = most_interesting_news()
        st.markdown("### Ranked News (for a Global Law Firm)")
        for i, r in enumerate(ranked, start=1):
            st.markdown(
                f"**{i}. {r['company_name']}** ({r['date']}) "
                f"[link]({r['url']}) — *score:* {round(r['score'],2)}"
            )
        summary = summarize_for_law_firm(
            vendor, model, ranked,
            "Summarize why each news item matters legally or regulatorily."
        )
        st.markdown("#### LLM Summary")
        st.markdown(summary)
    else:
        if not topic.strip():
            st.warning("Enter a topic first.")
        else:
            hits = retrieve_news(topic)
            st.markdown(f"### News related to '{topic}'")
            for i, r in enumerate(hits, start=1):
                st.markdown(
                    f"**{i}. {r['company_name']}** ({r['date']}) "
                    f"[link]({r['url']})"
                )
            summary = summarize_for_law_firm(
                vendor, model, hits,
                f"Summarize key legal or regulatory insights about {topic}."
            )
            st.markdown("#### LLM Summary")
            st.markdown(summary)

# ----------------------------
# Architecture & Evaluation
# ----------------------------
st.subheader("Write-up Notes")

with st.expander("Architecture Summary"):
    st.markdown("""
**Architecture**
- Prebuilt Chroma vector DB (`HW7_DB`) using OpenAI embeddings (`text-embedding-3-small`).
- Each article’s text stored with metadata (`company_name`, `date`, `URL`).
- Retrieval: top-k snippets from Chroma → sent to LLM (RAG pipeline).
- Interest score = keyword frequency + recency boost.
- Prompt frames LLM as a **law-firm analyst** to highlight legal significance.
- Vendors compared: OpenAI (`gpt-5-mini`, `gpt-4o-mini`) vs Mistral (`small`, `large`).
""")

with st.expander("Evaluation Summary"):
    st.markdown("""
**Evaluation**
- Verified top-ranked news contains legal/regulatory terms.
- Confirmed newer, legally salient items rank higher.
- Compared model outputs:
  - OpenAI models more factual and structured.
  - Mistral faster but more generic.
- Overall: `gpt-4o-mini` = best precision; `gpt-5-mini` = best cost/performance.
""")
