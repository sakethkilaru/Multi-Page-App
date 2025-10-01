# HW5 — iSchool Organizations Chatbot with Function-based Retrieval
# -----------------------------------------------------------------

__import__("pysqlite3")
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import os, json, re, glob, requests
import streamlit as st
from bs4 import BeautifulSoup
from openai import OpenAI
import chromadb

# -------------------------
# Page config + title
# -------------------------
st.set_page_config(page_title="HW5 — iSchool Memory Chatbot", layout="wide")
st.title("HW5 — iSchool Student Organizations Chatbot (with Function Retrieval)")

# -------------------------
# Sidebar: Model selection
# -------------------------
st.sidebar.header("Settings")
model_choice = st.sidebar.selectbox(
    "Choose LLM for answers",
    ["gpt-5-mini", "Mistral", "Gemini 2.5"],
    index=0,
)

# -------------------------
# OpenAI setup
# -------------------------
if "openai_client" not in st.session_state:
    api_key = st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        st.error("Missing OpenAI API key in secrets.toml")
        st.stop()
    st.session_state.openai_client = OpenAI(api_key=api_key)

openai_client = st.session_state.openai_client

# -------------------------
# ChromaDB setup
# -------------------------
CHROMA_DB_PATH = "./HW4_Chromadb"
COLLECTION_NAME = "HW4_iSchool_Orgs"
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = client.get_or_create_collection(COLLECTION_NAME)

# -------------------------
# Retrieval function
# -------------------------
def retrieve_relevant_docs(query, n=3):
    """Take user query -> return relevant org info from vector DB."""
    q_emb = openai_client.embeddings.create(
        input=query, model="text-embedding-3-small"
    ).data[0].embedding

    results = collection.query(query_embeddings=[q_emb], n_results=n)
    docs = results.get("documents", [[]])[0]
    mds = results.get("metadatas", [[]])[0]

    if not docs:
        return "No relevant information found in the uploaded documents."

    evidence = []
    for doc_text, md in zip(docs, mds):
        src = md.get("source_file", "?")
        part = md.get("part", "?")
        evidence.append(f"Source: {src} (chunk {part})\n{doc_text[:800]}")

    return "\n\n---\n\n".join(evidence)

# -------------------------
# Memory
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
# Handle user query
# -------------------------
if prompt := st.chat_input("Ask about iSchool student organizations:"):
    st.session_state.messages.append({"role":"user","content":prompt})
    with st.chat_message("user"): st.markdown(prompt)

    # Step 1: vector search
    relevant_info = retrieve_relevant_docs(prompt)

    # Step 2: build system prompt
    memory_context = ""
    if st.session_state.memory:
        mem_lines = [f"Q: {m['q']} | A: {m['a']}" for m in st.session_state.memory[-5:]]
        memory_context = "\n\nPrevious Q/A:\n" + "\n".join(mem_lines)

    system_prompt = f"""
You are an assistant specialized in iSchool student organizations.
Use the following retrieved information to answer the user query.
- If supported, start: "According to the uploaded materials:" and cite sources.
- If not, start: "I couldn't find an exact answer in the uploaded materials; here's general info:".

Retrieved info:
{relevant_info}

{memory_context}
"""

    messages_for_model = [
        {"role":"system","content":system_prompt},
        {"role":"user","content":prompt}
    ]

    # Step 3: call LLM
    response_text = ""
    if model_choice=="gpt-5-mini":
        resp = openai_client.chat.completions.create(
            model="gpt-5-mini", messages=messages_for_model
        )
        response_text = resp.choices[0].message.content

    elif model_choice=="Mistral":
        headers = {"Authorization": f"Bearer {st.secrets['MISTRAL_API_KEY']}",
                   "Content-Type": "application/json"}
        payload = {"model":"mistral-small","messages":messages_for_model,"stream":False}
        r = requests.post("https://api.mistral.ai/v1/chat/completions",
                          headers=headers, json=payload, timeout=60)
        data = r.json()
        response_text = data.get("choices",[{}])[0].get("message",{}).get("content","[Mistral error]")

    else:  # Gemini 2.5
        import google.generativeai as genai
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        model = genai.GenerativeModel("gemini-2.5-pro")
        resp = model.generate_content([m["content"] for m in messages_for_model])
        response_text = resp.text

    # Step 4: display + update memory
    with st.chat_message("assistant"): st.markdown(response_text)
    st.session_state.messages.append({"role":"assistant","content":response_text})
    st.session_state.memory.append({"q":prompt,"a":response_text})
    if len(st.session_state.memory) > 5:
        st.session_state.memory = st.session_state.memory[-5:]
