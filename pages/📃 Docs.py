import streamlit as st

st.header('Welcome to Docs!')
with open('module_name.md','r+') as f:
    mdfile = f.read()
st.write(mdfile)
