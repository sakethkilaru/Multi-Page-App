# HWs/HW3.py
import streamlit as st
import requests
from bs4 import BeautifulSoup

# --- Title ---
st.title("HW3 — URL Vendor Chatbot")

# --- Sidebar ---
st.sidebar.header("Options")
url1 = st.sidebar.text_input("Enter first URL")
url2 = st.sidebar.text_input("Enter second URL (optional)")

vendor = st.sidebar.selectbox(
    "Select LLM Vendor",
    ["OpenAI (GPT-5)", "Mistral", "Google Gemini"]
)

memory_type = st.sidebar.radio(
    "Conversation Memory",
    ["Buffer (6 Qs)", "Conversation Summary", "Buffer (2000 tokens)"],
)

# text fetcher
def fetch_text(url):
    if not url: return ""
    try:
        r = requests.get(url, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        for s in soup(["script","style","noscript"]): s.decompose()
        return soup.get_text(" ", strip=True)
    except Exception as e:
        return f"[Error fetching {url}: {e}]"

docs_context = ""
if url1: docs_context += f"\nFrom {url1}:\n{fetch_text(url1)}"
if url2: docs_context += f"\nFrom {url2}:\n{fetch_text(url2)}"


if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Display history ---
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# --- Handle input ---
if prompt := st.chat_input("Ask about the documents:"):
    # store user msg
    st.session_state.messages.append({"role":"user","content":prompt})
    with st.chat_message("user"): st.markdown(prompt)

    # memory logic 
    history = st.session_state.messages
    if memory_type == "Buffer (6 Qs)":
        # keep last 6 user turns + replies
        trimmed = []
        user_count = 0
        for m in reversed(history):
            if m["role"]=="user":
                user_count+=1
            trimmed.insert(0,m)
            if user_count==6: break
        history = trimmed
    elif memory_type == "Conversation Summary":
        # crude summary = concat all prev exchanges
        summary = " ".join([m["content"] for m in history])
        history = [{"role":"system","content":f"Summary so far: {summary}"}]

    # system prompt includes docs
    system_prompt = {"role":"system","content":f"Use these docs:\n{docs_context}"}

    # final msgs
    messages_with_system = [system_prompt] + history + [{"role":"user","content":prompt}]

    # pick client
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
        from mistralai.client import MistralClient
        client = MistralClient(api_key=st.secrets["MISTRAL_API_KEY"])
        resp = client.chat(model="mistral-large-latest", messages=messages_with_system)
        response_text = resp.choices[0].message["content"]
        with st.chat_message("assistant"): st.markdown(response_text)
    else:  # Gemini
        import google.generativeai as genai
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        model = genai.GenerativeModel("gemini-2.5-pro")
        resp = model.generate_content([m["content"] for m in messages_with_system])
        response_text = resp.text
        with st.chat_message("assistant"): st.markdown(response_text)

    # store assistant reply
    st.session_state.messages.append({"role":"assistant","content":response_text})
