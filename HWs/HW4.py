# HWs/HW4.py

# --- Fix for sqlite3 in Streamlit with ChromaDB ---
__import__("pysqlite3")
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import os, glob, re, json
import streamlit as st
from openai import OpenAI
import chromadb
from bs4 import BeautifulSoup

# -------------------------
# Page setup
# -------------------------
st.set_page_config(page_title="HW4 — iSchool RAG Chatbot", layout="wide")
st.title("HW4 — iSchool Student Organizations Chatbot (RAG)")

# -------------------------
# Model + API setup
# -------------------------
model_choice = st.sidebar.selectbox(
    "Choose LLM", ["gpt-5-mini", "gpt-3.5-turbo"], index=0
)

if "openai_client" not in st.session_state:
    api_key = st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        st.error("Missing OpenAI API key in secrets.toml")
        st.stop()
    st.session_state.openai_client = OpenAI(api_key=api_key)

openai_client = st.session_state.openai_client

# -------------------------
# Chroma setup
# -------------------------
CHROMA_PATH = "./HW4_Chromadb"
COLLECTION_NAME = "HW4_iSchool_Orgs"
MARKER_FILE = os.path.join(CHROMA_PATH, ".created")

# -------------------------
# HTML → text
# -------------------------
def html_to_text(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.extract()
    text = soup.get_text(" ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# -------------------------
# Chunking into exactly 2
# -------------------------
def chunk_into_two(text):
    words = text.split()
    if not words:
        return ["", ""]
    mid = len(words) // 2
    return [" ".join(words[:mid]), " ".join(words[mid:])]

# -------------------------
# Build vector DB (once)
# -------------------------
def build_vector_db(html_folder="./su_orgs"):
    os.makedirs(CHROMA_PATH, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(COLLECTION_NAME)

    files = sorted(glob.glob(os.path.join(html_folder, "*.html")))
    st.write(f"Found {len(files)} HTML files — building vector DB...")

    docs, ids, metas, embs = [], [], [], []
    for path in files:
        fname = os.path.basename(path)
        text = html_to_text(path)
        for i, chunk in enumerate(chunk_into_two(text), start=1):
            if not chunk.strip():
                continue
            eid = f"{fname}::part{i}"
            emb = openai_client.embeddings.create(
                input=chunk, model="text-embedding-3-small"
            ).data[0].embedding
            docs.append(chunk)
            ids.append(eid)
            metas.append({"file": fname, "part": i})
            embs.append(emb)

    if docs:
        collection.add(documents=docs, ids=ids, metadatas=metas, embeddings=embs)
        with open(MARKER_FILE, "w") as f:
            json.dump({"created": True}, f)
        st.success("Vector DB built.")
    return collection

# -------------------------
# Load or build collection
# -------------------------
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(COLLECTION_NAME)

if not os.path.exists(MARKER_FILE):
    st.info("No DB found — building from HTML...")
    collection = build_vector_db()
else:
    st.write("Using existing vector DB.")

# -------------------------
# Chat memory
# -------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -------------------------
# Chatbot
# -------------------------
if prompt := st.chat_input("Ask about iSchool student orgs..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # embed query + retrieve
    q_emb = openai_client.embeddings.create(
        input=prompt, model="text-embedding-3-small"
    ).data[0].embedding
    results = collection.query(query_embeddings=[q_emb], n_results=3)
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    evidence = []
    for d, m in zip(docs, metas):
        evidence.append(f"Source: {m.get('file')} (part {m.get('part')})\n{d[:500]}")
    evidence_block = "\n\n---\n\n".join(evidence) if evidence else "No relevant docs."

    system_prompt = f"""
You are an assistant about iSchool student organizations.
Use ONLY this info if relevant:
{evidence_block}

Rules:
- If supported by docs, start: "According to the uploaded materials:" and cite the file.
- If not, start: "I couldn't find an exact answer in the uploaded materials; here's general info:"
"""

    with st.chat_message("assistant"):
        stream = openai_client.chat.completions.create(
            model=model_choice,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            stream=True,
        )
        reply = st.write_stream(stream)

    st.session_state.messages.append({"role": "assistant", "content": reply})
