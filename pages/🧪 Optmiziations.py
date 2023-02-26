from nets.opti.blackbox import Hyper
import streamlit as st

st.code('from nets.opti.blackbox import Hyper')
st.code('api = Hyper(**kwargs)')
api = Hyper()
st.code('api.start_study(n_trials=n,neptune_id,neptune_api)')
run = None
def study():
    run = api.start_study(100)
    st.markdown('### Most recent run:')
    st.write(run)
st.button('Start study',on_click=study())
    
