from nets.opti.blackbox import Hyper
import streamlit as st

with open('bb.md','r+') as f:
    st.markdown(f.read(),unsafe_allow_html=True)

st.code('from nets.opti.blackbox import Hyper')
st.code('api = Hyper(**kwargs)')
api = Hyper()
st.code('api.start_study(n_trials=n,neptune_id,neptune_api)')
run = None
def study():
    run = api.start_study(50)
    st.success('Study Finished!',icon='✅')
    st.markdown('### :orange[Most recent run:]')
    st.write(run[0])
    st.write(run[1])
    st.write(run[2])
    st.write(run[3])
if st.button('Start study',use_container_width=True):
    st.info('Study is running',icon='🔄')
    study()
    
