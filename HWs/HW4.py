# HWs/HW4.py
# ---------------------------
# HW4 — iSchool Student Organizations Chatbot (RAG)
# ---------------------------

__import__("pysqlite3")
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import os
import glob
import json
import re
import streamlit as st
from bs4 import BeautifulSoup
from openai import OpenAI
import chromadb
import requests

# -------------------------
# Page config + title
# -------------------------
st.set_page_config(page_title="HW4 — iSchool RAG Chatbot", layout="wide")
st.title("HW4 — iSchool Student Organizations Chatbot (RAG)")

# -------------------------
# Sidebar: Model selection
# -------------------------
st.sidebar.header("Settings")
model_choice = st.sidebar.selectbox(
    "Choose LLM for answers",
    ["gpt-5-mini", "Mistral", "Gemini 2.5"],
    index=0,
)
st.sidebar.markdown("""
- This app uses Retrieval-Augmented Generation (RAG).
- The assistant will explicitly state when an answer is supported by uploaded HTML materials.
""")

# -------------------------
# OpenAI / Mistral / Gemini setup
# -------------------------
if "openai_client" not in st.session_state:
    api_key = st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        st.error("OpenAI API key missing in secrets.toml")
        st.stop()
    st.session_state.openai_client = OpenAI(api_key=api_key)

openai_client = st.session_state.openai_client

# -------------------------
# ChromaDB settings
# -------------------------
CHROMA_DB_PATH = "./HW4_Chromadb"
CHROMA_CREATED_MARKER = os.path.join(CHROMA_DB_PATH, ".created")
COLLECTION_NAME = "HW4_iSchool_Orgs"

# -------------------------
# HTML -> text extraction
# -------------------------
def html_to_text(path):
    """Extract visible text from HTML using built-in parser to avoid lxml errors."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f, "html.parser")  # Option 2: html.parser
    # Remove scripts/styles
    for tag in soup(["script", "style", "noscript"]):
        tag.extract()
    text = soup.get_text(separator="\n")
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()

# -------------------------
# Chunking: exactly two mini-docs per file
# -------------------------
def chunk_into_two_by_words(text, filename):
    """
    Split document into two chunks by word count:
    - Guarantees exactly two chunks (requirement)
    - Keeps chunks roughly equal in size
    - Robust for HTML with inconsistent headings
    """
    words = text.split()
    if not words:
        return ["", ""]
    if len(words) < 400:
        return [" ".join(words), ""]  # small doc -> second chunk empty
    mid = len(words) // 2
    chunk1 = " ".join(words[:mid]).strip()
    chunk2 = " ".join(words[mid:]).strip()
    return [chunk1, chunk2]

# -------------------------
# Build vector DB (run once)
# -------------------------
def build_vector_db_from_html(html_folder="./su_orgs"):
    os.makedirs(CHROMA_DB_PATH, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_or_create_collection(COLLECTION_NAME)

    html_files = sorted(glob.glob(os.path.join(html_folder, "*.html")))
    if not html_files:
        st.warning(f"No HTML files found in `{html_folder}`. Unzip su_orgs.zip there.")
        return collection

    docs_to_add, ids_to_add, metadatas_to_add, embeddings_list = [], [], [], []

    st.write(f"Found {len(html_files)} HTML files — building vector DB...")

    for path in html_files:
        filename = os.path.basename(path)
        raw_text = html_to_text(path)
        chunk1, chunk2 = chunk_into_two_by_words(raw_text, filename)
        for i, chunk in enumerate([chunk1, chunk2], start=1):
            if not chunk.strip():
                continue
            doc_id = f"{filename}::part{i}"
            emb_resp = openai_client.embeddings.create(input=chunk, model="text-embedding-3-small")
            emb = emb_resp.data[0].embedding

            docs_to_add.append(chunk)
            ids_to_add.append(doc_id)
            metadatas_to_add.append({"source_file": filename, "part": i})
            embeddings_list.append(emb)

    if docs_to_add:
        collection.add(
            documents=docs_to_add,
            ids=ids_to_add,
            metadatas=metadatas_to_add,
            embeddings=embeddings_list,
        )
        with open(CHROMA_CREATED_MARKER, "w", encoding="utf-8") as f:
            json.dump({"created": True, "num_docs": len(docs_to_add)}, f)
        st.success("Vector DB created.")
    else:
        st.info("No chunks added (HTMLs may be empty).")

    return collection

# -------------------------
# Load or build collection
# -------------------------
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = client.get_or_create_collection(COLLECTION_NAME)
if not os.path.exists(CHROMA_CREATED_MARKER):
    st.info("Vector DB not found — building from HTML files (run once).")
    collection = build_vector_db_from_html(html_folder="./su_orgs")
else:
    st.write("Using existing vector DB (cached).")

# -------------------------
# Memory buffer & chat history
# -------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "memory" not in st.session_state:
    st.session_state.memory = []

# Show previous chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -------------------------
# Handle user prompt
# -------------------------
if prompt := st.chat_input("Ask about iSchool student organizations (e.g., 'How do I join a club?')"):
    st.session_state.messages.append({"role":"user","content":prompt})
    with st.chat_message("user"): st.markdown(prompt)

    # --- Embed query ---
    q_emb = openai_client.embeddings.create(input=prompt, model="text-embedding-3-small").data[0].embedding

    # --- Query vector DB ---
    results = collection.query(query_embeddings=[q_emb], n_results=4)
    retrieved_docs = results.get("documents", [[]])[0]
    retrieved_mds = results.get("metadatas", [[]])[0]

    evidence_items = []
    for doc_text, md in zip(retrieved_docs, retrieved_mds):
        src = md.get("source_file","?")
        part = md.get("part","?")
        evidence_items.append(f"Source: {src} (chunk {part})\n{doc_text[:1200]}")
    evidence_block = "\n\n---\n\n".join(evidence_items) if evidence_items else "No relevant documents found."

    # --- System prompt ---
    system_prompt = f"""
You are an assistant specialized in iSchool student organizations.
Use the retrieved document chunks to answer the user's question.
- If the answer is supported, start: "According to the uploaded materials:" and cite sources.
- If not, start: "I couldn't find an exact answer in the uploaded materials; here's general info:".
Retrieved evidence:
{evidence_block}
"""

    # --- Memory summary (last 5 Q/A) ---
    memory_summary = ""
    if st.session_state.memory:
        mem_lines = [f"Q: {pair['q']}\nA: {pair['a']}" for pair in st.session_state.memory[-5:]]
        memory_summary = "\n\nPrevious Q/A:\n" + "\n\n".join(mem_lines)

    messages_for_model = [{"role":"system","content":system_prompt + (memory_summary if memory_summary else "")},
                          {"role":"user","content":prompt}]

    # --- Call chosen LLM ---
    response_text = ""
    if model_choice=="gpt-5-mini":
        stream = openai_client.chat.completions.create(model="gpt-5-mini",
                                                      messages=messages_for_model,
                                                      stream=True)
        with st.chat_message("assistant"):
            response_text = st.write_stream(stream)

    elif model_choice=="Mistral":
        headers = {"Authorization": f"Bearer {st.secrets['MISTRAL_API_KEY']}",
                   "Content-Type": "application/json"}
        payload = {"model":"mistral-small","messages":messages_for_model,"stream":False}
        r = requests.post("https://api.mistral.ai/v1/chat/completions",
                          headers=headers, json=payload, timeout=60)
        data = r.json()
        response_text = data.get("choices",[{}])[0].get("message",{}).get("content","[Mistral error]")
        with st.chat_message("assistant"): st.markdown(response_text)

    else:  # Gemini 2.5
        import google.generativeai as genai
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        model = genai.GenerativeModel("gemini-2.5-pro")
        resp = model.generate_content([m["content"] for m in messages_for_model])
        response_text = resp.text
        with st.chat_message("assistant"): st.markdown(response_text)

    # --- Store in history & memory ---
    st.session_state.messages.append({"role":"assistant","content":response_text})
    st.session_state.memory.append({"q":prompt,"a":response_text})
    if len(st.session_state.memory)>5: st.session_state.memory=st.session_state.memory[-5:]
