# HWs/HW4.py
import streamlit as st
import sys
import os

# --- Patch sqlite import for FAISS ---
try:
    import pysqlite3
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except Exception:
    pass

from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

# --- Title ---
st.title("HW4 — iSchool Student Organizations Chatbot (RAG)")

# --- Sidebar ---
st.sidebar.header("Options")
vendor = st.sidebar.selectbox(
    "Select LLM Vendor",
    ["OpenAI (GPT-5)", "Mistral", "Google Gemini"]
)

memory_type = st.sidebar.radio(
    "Conversation Memory",
    ["Buffer (5 Qs)", "Conversation Summary"],
)

# --- Load Vector DB ---
@st.cache_resource
def load_vectorstore():
    return FAISS.load_local("vector_db", OpenAIEmbeddings())

vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k":3})

# --- Memory ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Display history ---
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# --- Handle input ---
if prompt := st.chat_input("Ask about student orgs:"):
    # store user msg
    st.session_state.messages.append({"role":"user","content":prompt})
    with st.chat_message("user"): st.markdown(prompt)

    # --- memory logic ---
    history = st.session_state.messages
    if memory_type == "Buffer (5 Qs)":
        trimmed, user_count = [], 0
        for m in reversed(history):
            if m["role"]=="user": user_count += 1
            trimmed.insert(0,m)
            if user_count==5: break
        history = trimmed
    elif memory_type == "Conversation Summary":
        summary = " ".join([m["content"] for m in history])
        history = [{"role":"system","content":f"Summary so far: {summary}"}]

    # --- retrieve context from vector db ---
    docs = retriever.get_relevant_documents(prompt)
    context = "\n".join([d.page_content for d in docs])
    system_prompt = {"role":"system","content":f"Use these org pages:\n{context}"}

    messages_with_system = [system_prompt] + history + [{"role":"user","content":prompt}]

    # --- pick LLM ---
    response_text = ""
    if vendor.startswith("OpenAI"):
        from openai import OpenAI
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        stream = client.chat.completions.create(
            model="gpt-5",
            messages=messages_with_system,
            stream=True,
        )
        with st.chat_message("assistant"):
            response_text = st.write_stream(stream)

    elif vendor=="Mistral":
        import requests, json
        model = "mistral-small"
        headers = {
            "Authorization": f"Bearer {st.secrets['MISTRAL_API_KEY']}",
            "Content-Type": "application/json"
        }
        payload = {"model": model, "messages": messages_with_system}
        r = requests.post("https://api.mistral.ai/v1/chat/completions",
                          headers=headers, data=json.dumps(payload), timeout=60)
        data = r.json()
        if "choices" not in data:
            response_text = f"Mistral error: {data.get('error', data)}"
        else:
            response_text = data["choices"][0]["message"]["content"]
        with st.chat_message("assistant"): st.markdown(response_text)

    else:  # Gemini
        import google.generativeai as genai
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        model = genai.GenerativeModel("gemini-2.5-pro")
        resp = model.generate_content([m["content"] for m in messages_with_system])
        response_text = resp.text
        with st.chat_message("assistant"): st.markdown(response_text)

    # --- store assistant reply ---
    st.session_state.messages.append({"role":"assistant","content":response_text})
