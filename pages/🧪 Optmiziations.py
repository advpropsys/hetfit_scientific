from nets.opti.blackbox import Hyper
import streamlit as st

st.code('from nets.opti.blackbox import Hyper')
st.code('api = Hyper(**kwargs)')
api = Hyper()
st.code('api.start_study(n_trials=n,neptune_id,neptune_api)')
run = None
def study():
    run = api.start_study(n=100)
st.button('Start study',on_click=study())
if run:
    st.markdown('### Most recent run:')
    st.write(run)
