import streamlit as st

# Multi-page navigation (HW Manager)
pg = st.navigation(
    {
        "HW Manager": [
            st.Page("HWs/HW1.py", title="HW1 - Document Q&A", icon="📄"),
            st.Page("HWs/HW2.py", title="HW2 - URL Summarizer", icon="🌐"),
        ],
    }
)

pg.run()
