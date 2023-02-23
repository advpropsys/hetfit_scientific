import streamlit as st

st.header('Welcome to Docs!')
with open('docs/nets/envs.html','r+') as f:
    html = f.read()
st.write(html)
