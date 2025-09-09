import streamlit as st
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
import cohere
import google.generativeai as genai


#function
def read_url_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.get_text()
    except requests.RequestException as e:
        st.error(f"Error reading {url}: {e}")
        return None

# Streamlit App
st.title("HW2 - URL Summarizer")

#user URL
url = st.text_input("Enter a web page URL:")

#Sidebar options
summary_type = st.sidebar.radio(
    "Select summary type:",
    ["100 words", "2 paragraphs", "5 bullet points"]
)

language = st.sidebar.selectbox(
    "Output language:",
    ["English", "French", "Spanish"]
)

llm_choice = st.sidebar.selectbox(
    "Choose LLM:",
    ["OpenAI", "Cohere", "Gemini"]
)

advanced = st.sidebar.checkbox("Use advanced model?")


# Load API Clients
openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
cohere_client = cohere.Client(st.secrets["COHERE_API_KEY"])
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])


# Build Prompt
def build_prompt(text, summary_type, language):
    return f"Summarize the following text in {summary_type}, in {language}:\n\n{text}"



#summarize
if url:
    text = read_url_content(url)
    if text:
        prompt = build_prompt(text, summary_type, language)

        if llm_choice == "OpenAI":
            model = "gpt-4o" if advanced else "gpt-3.5-turbo"
            response = openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}]
            )
            summary = response.choices[0].message.content

        elif llm_choice == "Cohere":
            model = "command-r" if advanced else "command-light"
            response = cohere_client.generate(
                model=model,
                prompt=prompt
            )
            summary = response.generations[0].text

        elif llm_choice == "Gemini":
            model = "gemini-1.5-pro" if advanced else "gemini-1.5-flash"
            gemini_model = genai.GenerativeModel(model)
            response = gemini_model.generate_content(prompt)
            summary = response.text

        st.subheader("Summary")
        st.write(summary)
