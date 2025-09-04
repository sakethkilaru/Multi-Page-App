import streamlit as st
from openai import OpenAI

# Show title and description.
st.title("# My Document question answering")
st.write(
    "Upload a document below and ask a question about it – GPT will answer! "
    "To use this app, you need to provide an OpenAI API key, which you can get "
    "[here](https://platform.openai.com/account/api-keys). "
)

openai_api_key = st.secrets["OPENAI_API_KEY"]

# Validating key
client = None
if openai_api_key:
    try:
        client = OpenAI(api_key=openai_api_key)
        client.models.list()
        st.success("API key is valid")
    except Exception as e:
        st.error(f"Invalid API key: {e}")
        client = None
else:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")

if client:
    uploaded_file = st.file_uploader("Upload a document (.txt or .md)", type=("txt", "md"))

    question = st.text_area(
        "Now ask a question about the document!",
        placeholder="Can you give me a short summary?",
        disabled=not uploaded_file,
    )

    if uploaded_file and question:
        document = uploaded_file.read().decode()
        messages = [
            {"role": "user", "content": f"Here's a document: {document} \n\n---\n\n {question}"}
        ]

        try:
            stream = client.chat.completions.create(
                model="gpt-5-nano",
                messages=messages,
                stream=True,
            )
            st.write_stream(stream)
        except Exception as e:
            st.error(f"Error generating response: {e}")
