import streamlit as st

with open('intro.md','r+') as f:
    st.markdown(f.read(),unsafe_allow_html=True)