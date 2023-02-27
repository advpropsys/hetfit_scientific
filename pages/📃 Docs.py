import streamlit as st

st.header('Welcome to Docs!')
with open('main.md','r+') as f:
    mdfile = f.read()
st.markdown(mdfile,unsafe_allow_html=True)
