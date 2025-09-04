import streamlit as st

pg = st.navigation(
    {
        "Labs": [
            st.Page("pages/lab2.py", title="Lab 2 - Blank Page", icon="📝"),
            st.Page("pages/lab1.py", title="Lab 1 - Document Q&A", icon="📄"),
        ]
    }
)

pg.run()
