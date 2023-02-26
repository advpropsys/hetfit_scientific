import streamlit as st
import plotly.io as pio
pio.renderers.default='jupyterlab'

st.markdown('## :orange[Finding optimal HET design]')
st.markdown('Firstly we import SCI environment from HETFit module as well as design design module which will plot magnetic flux on $d{B}/d{z}$ Magntically shielded HET configuration and function to get whole deisgn of HET via just $P,U$ as inputs')
st.markdown('We are generating new features and specifying new domain based on $n_t$ value ')
st.code("""
from nets.envs import SCI
import torch
from nets.design import B_field_norm,PUdesign
B = B_field_norm(0.0002,14,k=16)
a = SCI()
a.feature_gen()
a.df = a.df[(a.df.nu_t < 0.66) & (a.df.nu_t > 0)] 
        """)
import plotly.express as px
from nets.envs import SCI
import torch
from nets.design import B_field_norm
data = B_field_norm(0.0002,14,k=16)
fig = px.line(y=data[1],x=data[0],labels={'y':'B','x':'L'})
st.write(fig)
a = SCI()
a.feature_gen()
a.df = a.df[(a.df.nu_t < 0.66) & (a.df.nu_t > 0)] 
        
st.markdown('---\n As you can see it is possible to access every bit of data you are working on via simple HETFit interface \n ---')
st.code("""
        a.compile(idx=(1,2,3,4,5,7,-1))\na.train()
        """)
a.compile(idx=(1,2,3,4,5,7,-1))
a.train()
st.markdown("---\n"
            "#### We select the $P,U,d,h,L,T$ columns for this case. As we know the geometry and needed thrust."
            "\n---\n"
            "#### Now we will assemble 2d matrix where rows are $n_t$ values and i,j (U,d) are changing. $h = 0.242*d$ as per PINN, L is approximated to be 2h, T - const = 0.3")

st.code("""
from torch import tensor
import numpy as np

y=[]
for i in np.arange(0.1,0.8,0.01):
    x=[]
    for j in np.arange(0.1,0.8,0.01):
        x.append(a.inference(tensor([0.25,float(i),float(j),float(j*0.242),2*float(j*0.242),0.3])).item())
    y.append(x)
        
        """)
st.markdown('---')
from torch import tensor
import numpy as np

y=[]
for i in np.arange(0.1,0.8,0.01):
                x=[]
                for j in np.arange(0.1,0.8,0.01):
                        x.append(a.inference(tensor([0.25,float(i),float(j),float(j*0.242),2*float(j*0.242),0.3])).item())
                y.append(x)
                
st.markdown("Now we plot and analyze: Seems like you need higher voltages and HET diamater for higher efficiencies.\n---")
st.code("""
fig = px.imshow(np.array(y),labels={r'x':r'$d_s$',r'y':r'$U_s$',r'color':r'$n_t$'})
fig.update_layout(
    dragmode='drawrect', # define dragmode
    newshape=dict(line_color='cyan'))
# Add modebar buttons
fig.show(config={'modeBarButtonsToAdd':['drawline',
                                        'drawopenpath',
                                        'drawclosedpath',
                                        'drawrect',
                                        'eraseshape'
                                       ]})
        """)

fig = px.imshow(np.array(y),labels={r'x':r'$U_s$',r'y':r'$d_s$',r'color':r'$\eta_t$'},title=r'$U_s,d_s \mapsto \eta_t \text{at} P,h,L,T \text{Invariants}$')
fig.update_layout(
    dragmode='drawrect', # define dragmode
    newshape=dict(line_color='cyan'))
# Add modebar buttons
st.write(fig)

st.markdown('---\nUsing this strategy we just have assembled model for $U,d \mapsto n_t$ with other design variables as invariants. It also can be done another way by overlaying predictions of two varibles models.')

###
if st.button(r'Generate $f:R^2 \to R$ maps',use_container_width=True):
    a.compile(idx=(2,3,-1))
    a.train()

    y=[]
    for i in np.arange(0.1,0.8,0.01):
                    x=[]
                    for j in np.arange(0.1,0.8,0.01):
                            x.append(a.inference(tensor([float(i),float(j)])).item())
                    y.append(x)

    fig = px.imshow(np.array(y),labels={r'x':r'$U_s$',r'y':r'$d_s$',r'color':r'$\nu_t$'},title=r'$U_s,d_s \mapsto \nu_t$')
    fig.update_layout(
        dragmode='drawrect', # define dragmode
        newshape=dict(line_color='cyan'))
    # Add modebar buttons
    st.write(fig)

    a.compile(idx=(3,4,-1))
    a.train()

    y=[]
    for i in np.arange(0.1,0.8,0.01):
                    x=[]
                    for j in np.arange(0.1,0.8,0.01):
                            x.append(a.inference(tensor([float(i),float(j)])).item())
                    y.append(x)

    fig = px.imshow(np.array(y),labels={r'x':r'$d_s$',r'y':r'$h_s$',r'color':r'$\nu_t$'},title=r'$d_s,h_s \mapsto \nu_t$')
    fig.update_layout(
        dragmode='drawrect', # define dragmode
        newshape=dict(line_color='cyan'))
    # Add modebar buttons
    st.write(fig)


    ###

    a.compile(idx=(6,7,-1))
    a.train()

    y=[]
    for i in np.arange(0.1,0.8,0.01):
                    x=[]
                    for j in np.arange(0.1,0.8,0.01):
                            x.append(a.inference(tensor([float(i),float(j)])).item())
                    y.append(x)

    fig = px.imshow(np.array(y),labels={r'x':r'$m_{as}$',r'y':r'$T$',r'color':r'$\nu_t$'},title=r'$m_{as},T \mapsto \nu_t$')
    fig.update_layout(
        dragmode='drawrect', # define dragmode
        newshape=dict(line_color='cyan'))
    # Add modebar buttons
    st.write(fig)

    ###
    a.compile(idx=(7,8,-1))
    a.train()

    y=[]
    for i in np.arange(0.1,0.8,0.01):
                    x=[]
                    for j in np.arange(0.1,0.8,0.01):
                            x.append(a.inference(tensor([float(i),float(j)])).item())
                    y.append(x)

    fig = px.imshow(np.array(y),labels={r'x':r'$T$',r'y':r'$I_{sp}$',r'color':r'$\nu_t$'}, title=r'$T,I_{sp} \mapsto \nu_t$')
    fig.update_layout(
        dragmode='drawrect', # define dragmode
        newshape=dict(line_color='cyan'))
    # Add modebar buttons
    st.write(fig)

