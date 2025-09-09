import streamlit as st
from openai import OpenAI
import fitz  

# Show title and description.
st.title("📄 My Document question answering")
st.write(
    "Upload a document below and ask a question about it – GPT will answer! "
    "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
)

# Ask user for their OpenAI API key via `st.text_input`.
# Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
# via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management

openai_api_key = st.text_input("OpenAI API Key", type="password")

# --- VALIDATE KEY IMMEDIATELY ---
client = None
if openai_api_key:
    try:
        client = OpenAI(api_key=openai_api_key)
        # Simple test request to validate the key
        client.models.list()
        st.success("✅ API key is valid!")
    except Exception as e:
        st.error(f"Invalid API key: {e}")
        client = None
else:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
# --------------------------------

# Model selection (choose from the four requested)
model_choice = st.selectbox(
    "Choose a model",
    options=[
        "gpt-3.5-turbo",      # gpt-3.5
        "gpt-4.1",            # gpt-4.1
        "gpt-5-chat-latest",  # gpt-5-chat-latest
        "gpt-5-nano",         # gpt-5-nano
    ],
    index=3,  # default to gpt-5-nano 
)

if client:

    # Let the user upload a file via `st.file_uploader`.
    # Restrict to only txt and pdf per assignment.
    uploaded_file = st.file_uploader(
        "Upload a document (.txt or .pdf)", type=("txt", "pdf")
    )

    # helper to read pdf bytes using PyMuPDF
    def read_pdf(uploaded_file_obj):
        # uploaded_file_obj is a SpooledTemporaryFile / BytesIO-like object
        file_bytes = uploaded_file_obj.read()
        # open from bytes
        pdf = fitz.open(stream=file_bytes, filetype="pdf")
        text_parts = []
        for page in pdf:
            text_parts.append(page.get_text())
        pdf.close()
        return "\n".join(text_parts)

    # Ask the user for a question via `st.text_area`.
    # Disable if no file uploaded or key invalid.
    question = st.text_area(
        "Now ask a question about the document!",
        placeholder="Can you give me a short summary?",
        disabled=(not uploaded_file or not client),
    )

    # Make sure we only keep the document when uploaded_file exists.
    document = None
    if uploaded_file:
        file_extension = uploaded_file.name.split(".")[-1].lower()
        if file_extension == "txt":
            try:
                # read and decode text file
                document = uploaded_file.read().decode()
            except Exception as e:
                st.error(f"Failed to read .txt file: {e}")
                document = None
        elif file_extension == "pdf":
            try:
                document = read_pdf(uploaded_file)
            except Exception as e:
                st.error(f"Failed to read PDF file: {e}")
                document = None
        else:
            st.error("Unsupported file type.")
            document = None
    else:
        # If the file was removed from the UI, ensure we don't keep the data
        document = None

    if uploaded_file and document and question:

        # Process the uploaded file and question.
        messages = [
            {
                "role": "user",
                "content": f"Here's a document: {document} \n\n---\n\n {question}",
            }
        ]

        # Generate an answer using the OpenAI API.
        try:
            stream = client.chat.completions.create(
                model=model_choice,
                messages=messages,
                stream=True,
            )

            # Stream the response to the app using `st.write_stream`.
            st.write_stream(stream)

        except Exception as e:
            st.error(f"Error generating response: {e}")
