import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from nets.normflows import nflow
import numpy as np
import seaborn as sns
import streamlit as st

st.markdown('## This is :orange[normalizing] flows approach to generating more data points, its most accurate and cheap method so far')
def random_normal_samples(n, dim=2):
    return torch.zeros(n, dim).normal_(mean=0, std=3)


st.markdown("""
Normalizing flows can be used for generating new samples for sensor or engineering data by modeling the distribution of the data using a series of invertible transformations. These transformations can capture the complex probability density function of the data, and the resulting model can be used to generate new samples.

For example, if we have a dataset of sensor readings from a machine, we can use normalizing flows to model the distribution of these readings. We can then generate new samples of sensor readings that are similar to the original dataset, but with some variation. This can be useful for testing machine learning algorithms, evaluating sensor performance, or generating synthetic data for training new models.

Normalizing flows can also be used for anomaly detection in sensor data. By learning the distribution of normal sensor readings, we can identify instances where the data deviates significantly from this normal distribution, indicating a potential anomaly or fault in the machine.

Overall, normalizing flows provide a powerful tool for generating new samples of sensor or engineering data, as well as for analyzing and understanding the underlying distribution of the data.
            """)
st.markdown('### Define API, kwargs: dimension,latent, dataset')
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
st.markdown('### 10 iters')
st.pyplot(fig.get_figure())

st.markdown('### 12000 iters')
st.image('output.png')
