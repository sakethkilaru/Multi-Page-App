import streamlit as st

st.set_page_config(page_title="HWs")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["HW1", "HW2", "HW3", "HW4"])

if page == "HW1":
    exec(open("HWs/HW1.py").read())
elif page == "HW2":
    exec(open("HWs/HW2.py").read())
elif page == "HW3":
    exec(open("HWs/HW3.py").read())
elif page == "HW4":
    exec(open("HWs/HW4.py").read())