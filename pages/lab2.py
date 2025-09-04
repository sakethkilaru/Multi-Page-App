import streamlit as st
from openai import OpenAI

st.title("📝 Lab 2")


#client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

st.title("Document Summarizer")

#sidebar
summary_type = st.sidebar.selectbox(
    "Summary Type",
    ["100 words", "2 connecting paragraphs", "5 bullet points"]
)

use_advanced = st.sidebar.checkbox("Use Advanced Model (4o)")

model_name = "gpt-4o" if use_advanced else "gpt-4o-mini"

# User input document
document = st.text_area("Paste your document here:", height=300)

if document:
    prompt = f"Summarize the following document in the style '{summary_type}':\n\n{document}"
    
    with st.spinner("Generating summary..."):
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}]
        )

    summary = response.choices[0].message["content"]
    st.subheader("Summary")
    st.write(summary)
else:
    st.info("Paste a document above to see the summary.")
