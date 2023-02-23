import streamlit as st

st.header('Welcome to Docs!')
mdfile = """ envs.py 

envs.py
=======

[#](#section-0)

from utils.dataset\_loader import get\_dataset
from nets.dense import Net
from nets.deep\_dense import dmodel
from PINN.pinns import \*

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import os
import numpy as np
from torch import nn, tensor
import pandas as pd
import plotly.express as px
from sklearn.linear\_model import SGDRegressor
from sklearn.feature\_selection import SelectFromModel

[#](#section-1)

class SCI(): #Scaled Computing Interface

[#](#section-2)

    def \_\_init\_\_(self, hidden\_dim:int \= 200, dropout:bool \= True, epochs:int \= 10, dataset:str \= 'test.pkl',sample\_size:int\=1000,source:str\='dataset.csv',boundary\_conditions:list\=None):
        self.type:str \= 'legacy'
        self.seed:int \= 449
        self.dim \= hidden\_dim
        self.dropout \= dropout
        self.df \= get\_dataset(sample\_size\=sample\_size,source\=source,name\=dataset,boundary\_conditions\=boundary\_conditions)
        self.epochs \= epochs
        self.len\_idx \= 0
        self.input\_dim\_for\_check \= 0

[#](#section-3)

    def feature\_gen(self, base:bool\=True, fname:str\=None,index:int\=None,func\=None) \-> None:
        
        if base:
            self.df\['P\_sqrt'\] \= self.df.iloc\[:,1\].apply(lambda x: x \*\* 0.5)
            self.df\['j'\] \= self.df.iloc\[:,1\]/(self.df.iloc\[:,3\]\*self.df.iloc\[:,4\])
            self.df\['B'\] \= self.df.iloc\[:,\-1\].apply(lambda x: x \*\* 2).apply(lambda x:1 if x\>1 else x)
            self.df\['nu\_t'\] \= self.df.iloc\[:,7\]\*\*2/(2\*self.df.iloc\[:,6\]\*self.df.P)
            
        if fname and index and func:
            self.df\[fname\] \= self.df.iloc\[:,index\].apply(func)

[#](#section-4)

    def feature\_importance(self,X:pd.DataFrame,Y:pd.Series,verbose:int\=1):
        
        mod \= SGDRegressor()
        
        selector \= SelectFromModel(mod,threshold\='1.25\*mean')
        selector.fit(np.array(X),np.array(Y))
        
        if verbose:
            print(f'\\n Report of feature importance: {dict(zip(X.columns,selector.estimator\_.coef\_))}')
        for i in range(len(selector.get\_support())):
            if selector.get\_support()\[i\]:
                print(f'-rank 1 PASSED:',X.columns\[i\])
            else:
                print(f'-rank 0 REJECT:',X.columns\[i\])
        return f'\\n Report of feature importance: {dict(zip(X.columns,selector.estimator\_.coef\_))}'

[#](#section-5)

Data prep pipeline

    def data\_flow(self,columns\_idx:tuple \= (1,3,3,5), idx:tuple\=None, split\_idx:int \= 800) \-> torch.utils.data.DataLoader:

[#](#section-6)

        Args:
    

columns\_idx (tuple, optional): Columns to be selected (sliced 1:2 3:4) for feature fitting. Defaults to (1,3,3,5). idx (tuple, optional): 2|3 indexes to be selected for feature fitting. Defaults to None. Use either idx or columns\_idx (for F:R->R idx, for F:R->R2 columns\_idx) split\_idx (int) : Index to split for training

        Returns:
    

torch.utils.data.DataLoader: Torch native dataloader

        batch\_size\=2
        
        self.split\_idx\=split\_idx
        
        if idx!=None:
            self.len\_idx \= len(idx)
            if len(idx)\==2:
                self.X \= tensor(self.df.iloc\[:,idx\[0\]\].values\[:split\_idx\]).float()
                self.Y \= tensor(self.df.iloc\[:,idx\[1\]\].values\[:split\_idx\]).float()
                batch\_size \= 1
            else:
                self.X \= tensor(self.df.iloc\[:,\[idx\[0\],idx\[1\]\]\].values\[:split\_idx,:\]).float()
                self.Y \= tensor(self.df.iloc\[:,idx\[2\]\].values\[:split\_idx\]).float()
        else:
            self.X \= tensor(self.df.iloc\[:,columns\_idx\[0\]:columns\_idx\[1\]\].values\[:split\_idx,:\]).float()
            self.Y \= tensor(self.df.iloc\[:,columns\_idx\[2\]:columns\_idx\[3\]\].values\[:split\_idx\]).float()
            
        print('Shapes for debug: (X,Y)',self.X.shape, self.Y.shape)
        train\_data \= torch.utils.data.TensorDataset(self.X, self.Y)
        Xtrain \= torch.utils.data.DataLoader(train\_data,batch\_size\=batch\_size)
        self.input\_dim \= self.X.size(\-1)
        self.indexes \= idx if idx else columns\_idx
        self.column\_names \= \[ self.df.columns\[i\] for i in self.indexes \]
        return Xtrain

[#](#section-7)

Initializes seed for torch optional()

    def init\_seed(self,seed):

[#](#section-8)

        
        torch.manual\_seed(seed)

[#](#section-9)

    def train\_epoch(self,X, model, loss\_function, optim):
        for i,data in enumerate(X):
                Y\_pred \= model(data\[0\])
                loss \= loss\_function(Y\_pred, data\[1\])

[#](#section-10)

mean\_abs\_percentage\_error = MeanAbsolutePercentageError() ape = mean\_abs\_percentage\_error(Y\_pred, data\[1\])

                
                loss.backward()
                optim.step()
                optim.zero\_grad()
            
                
                ape\_norm \= abs(np.mean((Y\_pred.detach().numpy()\-data\[1\].detach().numpy())/(data\[1\].detach().numpy()+0.1)))
                if (i+1)%200\==0:
                    print(f'Iter {i+1} APE =',ape\_norm)
                self.loss\_history.append(loss.data.item())
                self.ape\_history.append(None if ape\_norm \>1 else ape\_norm)

[#](#section-11)

Builds model, loss, optimizer. Has defaults

    def compile(self,columns:tuple\=None,idx:tuple\=None, optim:torch.optim \= torch.optim.AdamW,loss:nn\=nn.L1Loss, model:nn.Module \= dmodel, custom:bool\=False) \-> None:

[#](#section-12)

        Args:
    

columns (tuple, optional): Columns to be selected for feature fitting. Defaults to (1,3,3,5). optim - torch Optimizer loss - torch Loss function (nn)

        
        self.columns \= columns

        
                
        if not(columns):
            self.len\_idx \= 0
        else:
            self.len\_idx \= len(columns)
            
        if not(self.columns) and not(idx):
            self.Xtrain \= self.data\_flow()
        elif not(idx): 
            self.Xtrain \= self.data\_flow(columns\_idx\=self.columns)
        else:
            self.Xtrain \= self.data\_flow(idx\=idx)
            
        if custom:
            self.model \= model()
            if self.len\_idx \== 2:
                self.input\_dim\_for\_check \= 1
        else: 
            if self.len\_idx \== 2:
                self.model \= model(in\_features\=1,hidden\_features\=self.dim).float()
                self.input\_dim\_for\_check \= 1
            if self.len\_idx \== 3:
                self.model \= Net(input\_dim\=2,hidden\_dim\=self.dim).float()
            if (self.len\_idx \== 0) or self.columns:
                self.model \= Net(input\_dim\=self.input\_dim,hidden\_dim\=self.dim).float()
                
            self.optim \= optim(self.model.parameters(), lr\=0.001)
            self.loss\_function \= loss()
            
        if self.input\_dim\_for\_check:
            self.X  \= self.X.reshape(\-1,1)

[#](#section-13)

Train model

    def train(self,epochs:int\=10) \-> None:

[#](#section-14)

        If sklearn instance uses .fit()
    

        if 'sklearn' in str(self.model.\_\_class\_\_):
            self.model.fit(np.array(self.X),np.array(self.Y))
            plt.scatter(self.X,self.model.predict(self.X))
            plt.scatter(self.X,self.Y)
            plt.xlabel('Xreal')
            plt.ylabel('Ypred/Yreal')
            plt.show()
            return print('Sklearn model fitted successfully')
        else:
            self.model.train()
            
        self.loss\_history \= \[\]
        self.ape\_history \= \[\]
        
        self.epochs \= epochs
        
        
        for j in range(self.epochs):
            self.train\_epoch(self.Xtrain,self.model,self.loss\_function,self.optim)
            
        plt.plot(self.loss\_history,label\='loss\_history')
        plt.legend()

[#](#section-15)

    def save(self,name:str\='model.pt') \-> None:
        torch.save(self.model,name)

[#](#section-16)

    def onnx\_export(self,path:str\='./models/model.onnx'):
        torch.onnx.export(self.model,self.X,path)

[#](#section-17)

    def jit\_export(self,path:str\='./models/model.pt'):
        torch.jit.save(self.model,path)

[#](#section-18)

Inference of (pre-)trained model

    def inference(self,X:tensor, model\_name:str\=None) \-> np.ndarray:

[#](#section-19)

        Args:
    

X (tensor): your data in domain of train

        Returns:
    

np.ndarray: predictions

        if model\_name is None:
            self.model.eval()
            
        if model\_name in os.listdir('./models'):
            model \= torch.load(f'./models/{model\_name}')
            model.eval()
            return model(X).detach().numpy()
        
        return self.model(X).detach().numpy()

[#](#section-20)

    def plot(self):
        self.model.eval()
        print(self.Y.shape,self.model(self.X).detach().numpy().shape,self.X.shape)
        if self.X.shape\[\-1\] != self.model(self.X).detach().numpy().shape\[\-1\]:
            print('Size mismatch, try 3d plot, plotting by second dim of largest tensor')
            plt.scatter(self.X\[:,1\],self.model(self.X).detach().numpy(),label\='predicted',s\=2)
            if self.Y.shape\[\-1\]!=1:
                plt.scatter(self.X\[:,1\],self.Y\[:,1\],s\=1,label\='real')
            else:
                plt.scatter(self.X\[:,1\],self.Y,s\=1,label\='real')
            plt.xlabel(rf'${self.column\_names\[0\]}$')
            plt.ylabel(rf'${self.column\_names\[1\]}$')
            plt.legend()
        else:
            plt.scatter(self.X,self.model(self.X).detach().numpy(),s\=2,label\='predicted')
            plt.scatter(self.X,self.Y,s\=1,label\='real')
            plt.xlabel(r'$X$')
            plt.ylabel(r'$Y$')
            plt.legend()

[#](#section-21)

    def plot3d(self):
        X \= self.X
        self.model.eval()
        x \= X\[:,0\].numpy().ravel()
        y \= X\[:,1\].numpy().ravel()
        z \= self.model(X).detach().numpy().ravel()
        surf \= px.scatter\_3d(x\=x, y\=y,z\=self.df.iloc\[:,self.indexes\[\-1\]\].values\[:self.split\_idx\],opacity\=0.3,
                             labels\={'x':f'{self.column\_names\[0\]}',
                                     'y':f'{self.column\_names\[1\]}',
                                     'z':f'{self.column\_names\[\-1\]}'
                                     },title\='Mesh prediction plot'
                            )

[#](#section-22)

fig.colorbar(surf, shrink=0.5, aspect=5)

        surf.update\_traces(marker\_size\=3)
        surf.update\_layout(plot\_bgcolor\='#888888')
        surf.add\_mesh3d(x\=x, y\=y, z\=z, opacity\=0.7,colorscale\='sunsetdark',intensity\=z,
            )

[#](#section-23)

surf.show()

        
        return surf

[#](#section-24)

    def performance(self,c\=0.4) \-> dict:
        a\=\[\]
        for i in range(1000):
            a.append(100\-abs(np.mean(self.df.iloc\[1:24,1:8\].values\-self.df.iloc\[24:,1:8\].sample(23).values)/(self.Y.numpy()\[1:\]+c))\*100)
    
        gen\_acc \= np.mean(a)
        ape \= (100\-abs(np.mean(self.model(self.X).detach().numpy()\-self.Y.numpy()\[1:\])\*100))
        abs\_ape \= ape\*gen\_acc/100
        return {'Generator\_Accuracy, %':np.mean(a),'APE\_abs, %':abs\_ape,'Model\_APE, %': ape}

[#](#section-25)

    def performance\_super(self,c\=0.4,real\_data\_column\_index:tuple \= (1,8),real\_data\_samples:int\=23, generated\_length:int\=1000) \-> dict:
        a\=\[\]
        for i in range(1000):
            a.append(100\-abs(np.mean(self.df.iloc\[1:real\_data\_samples+1,real\_data\_column\_index\[0\]:real\_data\_column\_index\[1\]\].values\-self.df.iloc\[real\_data\_samples+1:,real\_data\_column\_index\[0\]:real\_data\_column\_index\[1\]\].sample(real\_data\_samples).values)/(self.Y.numpy()\[1:\]+c))\*100)
        gen\_acc \= np.mean(a)
        ape \= (100\-abs(np.mean(self.model(self.X).detach().numpy()\-self.Y.numpy()\[1:\])\*100))
        abs\_ape \= ape\*gen\_acc/100
        return {'Generator\_Accuracy, %':np.mean(a),'APE\_abs, %':abs\_ape,'Model\_APE, %': ape}

[#](#section-26)

class RCI(SCI): #Real object interface

[#](#section-27)

    def \_\_init\_\_(self,\*args,\*\*kwargs):
        super(RCI,self).\_\_init\_\_()

[#](#section-28)

Data prep pipeline

    def data\_flow(self,columns\_idx:tuple \= (1,3,3,5), idx:tuple\=None, split\_idx:int \= 800) \-> torch.utils.data.DataLoader:

[#](#section-29)

            Args:
    

columns\_idx (tuple, optional): Columns to be selected (sliced 1:2 3:4) for feature fitting. Defaults to (1,3,3,5). idx (tuple, optional): 2|3 indexes to be selected for feature fitting. Defaults to None. Use either idx or columns\_idx (for F:R->R idx, for F:R->R2 columns\_idx) split\_idx (int) : Index to split for training

            Returns:
    

torch.utils.data.DataLoader: Torch native dataloader

            batch\_size\=2
            
            real\_scale \= pd.read\_csv('data/dataset.csv').iloc\[17,1:\].to\_numpy()
            self.df.iloc\[:,1:\] \= self.df.iloc\[:,1:\] \* real\_scale
            
            self.split\_idx\=split\_idx
            
            
           
            
            if idx!=None:
                self.len\_idx \= len(idx)
                if len(idx)\==2:
                    self.X \= tensor(self.df.iloc\[:,idx\[0\]\].values\[:split\_idx\].astype(float)).float()
                    self.Y \= tensor(self.df.iloc\[:,idx\[1\]\].values\[:split\_idx\].astype(float)).float()
                    batch\_size \= 1
                else:
                    self.X \= tensor(self.df.iloc\[:,\[idx\[0\],idx\[1\]\]\].values\[:split\_idx,:\].astype(float)).float()
                    self.Y \= tensor(self.df.iloc\[:,idx\[2\]\].values\[:split\_idx\].astype(float)).float()
            else:
                self.X \= tensor(self.df.iloc\[:,columns\_idx\[0\]:columns\_idx\[1\]\].values\[:split\_idx,:\].astype(float)).float()
                self.Y \= tensor(self.df.iloc\[:,columns\_idx\[2\]:columns\_idx\[3\]\].values\[:split\_idx\].astype(float)).float()
            self.Y \= self.Y.abs()
            self.X \= self.X.abs()
              
            print('Shapes for debug: (X,Y)',self.X.shape, self.Y.shape)
            train\_data \= torch.utils.data.TensorDataset(self.X, self.Y)
            Xtrain \= torch.utils.data.DataLoader(train\_data,batch\_size\=batch\_size)
            self.input\_dim \= self.X.size(\-1)
            self.indexes \= idx if idx else columns\_idx
            self.column\_names \= \[ self.df.columns\[i\] for i in self.indexes \]
            
            
            
            
            return Xtrain

[#](#section-30)

Builds model, loss, optimizer. Has defaults

    def compile(self,columns:tuple\=None,idx:tuple\=(3,1), optim:torch.optim \= torch.optim.AdamW,loss:nn\=nn.L1Loss, model:nn.Module \= PINNd\_p,lr:float\=0.001) \-> None:

[#](#section-31)

        Args:
    

columns (tuple, optional): Columns to be selected for feature fitting. Defaults to None. idx (tuple, optional): indexes to be selected Default (3,1) optim - torch Optimizer loss - torch Loss function (nn)

        
        self.columns \= columns

        
                
        if not(columns):
            self.len\_idx \= 0
        else:
            self.len\_idx \= len(columns)
            
        if not(self.columns) and not(idx):
            self.Xtrain \= self.data\_flow()
        elif not(idx): 
            self.Xtrain \= self.data\_flow(columns\_idx\=self.columns)
        else:
            self.Xtrain \= self.data\_flow(idx\=idx)
        
        self.model \= model().float()
        self.input\_dim\_for\_check \= self.X.size(\-1)
                
        self.optim \= optim(self.model.parameters(), lr\=lr)
        self.loss\_function \= loss()
            
        if self.input\_dim\_for\_check \== 1:
            self.X  \= self.X.reshape(\-1,1)

[#](#section-32)

    def plot(self):
        self.model.eval()
        if 'PINN' in str(self.model.\_\_class\_\_):
            self.preds\=np.array(\[\])
            for i in self.X:
                self.preds \= np.append(self.preds,self.model(i).detach().numpy()) 
        print(self.Y.shape,self.preds.shape,self.X.shape)
        if self.X.shape\[\-1\] != self.preds.shape\[\-1\]:
            print('Size mismatch, try 3d plot, plotting by second dim of largest tensor')
            plt.scatter(self.X\[:,1\],self.preds,label\='predicted',s\=2)
            if self.Y.shape\[\-1\]!=1:
                sns.scatterplot(x\=self.X\[:,1\],y\=self.Y,s\=2,label\='real')
            else:
                sns.scatterplot(x\=self.X\[:,1\],y\=self.Y,s\=1,label\='real')
            plt.xlabel(rf'${self.column\_names\[0\]}$')
            plt.ylabel(rf'${self.column\_names\[1\]}$')
            plt.legend()
        else:
            sns.scatterplot(x\=self.X,y\=self.preds,s\=2,label\='predicted')
            sns.scatterplot(x\=self.X,y\=self.Y,s\=1,label\='real')
            plt.xlabel(r'$X$')
            plt.ylabel(r'$Y$')
            plt.legend()

[#](#section-33)

    def performance(self,c\=0.4) \-> dict:
        a\=\[\]
        for i in range(1000):
            dfcopy \= (self.df.iloc\[:,1:8\]\-self.df.iloc\[:,1:8\].min())/(self.df.iloc\[:,1:8\].max()\-self.df.iloc\[:,1:8\].min())
            a.append(100\-abs(np.mean(dfcopy.iloc\[1:24,1:\].values\-dfcopy.iloc\[24:,1:\].sample(23).values)/(dfcopy.iloc\[1:24,1:\].values+c))\*100)
        gen\_acc \= np.mean(a)
        ape \= (100\-abs(np.mean(self.model(self.preds).detach().numpy()\-self.Y.numpy())\*100))
        abs\_ape \= ape\*gen\_acc/100
        return {'Generator\_Accuracy, %':np.mean(a),'APE\_abs, %':abs\_ape,'Model\_APE, %': ape}
"""
st.write(mdfile)
