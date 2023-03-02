import streamlit as st

with open('intro.md','r+') as f:
    st.markdown(f,unsafe_allow_html=True)