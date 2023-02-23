import streamlit as st

st.header('Welcome to Docs!')
with open('docs/nets.envs.md','r+') as f:
    mdfile = f.read()
st.write(mdfile)
