import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from nets.normflows import nflow
import numpy as np
import seaborn as sns
import streamlit as st

def random_normal_samples(n, dim=2):
    return torch.zeros(n, dim).normal_(mean=0, std=3)

st.code('api = nflow(dim=8,latent=16)')
api = nflow(dim=8,latent=16)
st.code('api.compile(optim=torch.optim.ASGD,bw=3.05,lr=0.0001,wd=None)')
api.compile(optim=torch.optim.ASGD,bw=3.05,lr=0.0001,wd=None)
st.code('api.train(iters=8000)')


api.train(iters=10)

@st.cache_data
def samples():
    samples = np.array(api.model.sample(
        torch.tensor(api.scaled).float()).detach())
    return samples

fig = plt.figure(figsize=(10,8))
g = sns.jointplot(x=samples()[:, 0], y=samples()[:, 1], kind='kde',cmap=sns.color_palette("Blues", as_cmap=True),fill=True,label='Gaussian KDE')
fig = sns.scatterplot(x=api.scaled[:,0],y=api.scaled[:,1],ax=g.ax_joint,c='orange',marker='+',s=100,label='Real')

st.pyplot(fig.get_figure())
