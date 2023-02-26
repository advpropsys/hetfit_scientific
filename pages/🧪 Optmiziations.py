from nets.opti.blackbox import Hyper
import streamlit as st



st.code('from nets.opti.blackbox import Hyper')
st.code('api = Hyper(**kwargs)')
api = Hyper()
st.code('api.start_study(n_trials=n,neptune_id,neptune_api)')
run = None
def study():
    run = api.start_study(50)
    st.success('Study Finished!',icon='✅')
    st.markdown('### :orange[Most recent run:]')
    st.write(run)
if st.button('Start study',use_container_width=True):
    study()
    
