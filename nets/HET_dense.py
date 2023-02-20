from utils.dataset_loader import get_dataset
from nets.dense import Net
from nets.deep_dense import dmodel

import matplotlib.pyplot as plt
import torch
import os
import numpy as np
from torch import nn, tensor
from torchmetrics import MeanAbsolutePercentageError
import pandas as pd
import plotly.express as px

class dense():
    def __init__(self, typ='PU', hidden_dim:int = 200, dropout:bool = True, epochs:int = 10, dataset:str = 'test.pkl'):
        self.type:str = typ
        self.seed:int = 449
        self.dim = hidden_dim
        self.dropout = dropout
        self.df = get_dataset(name=dataset)
        self.epochs = epochs
        self.len_idx = 0
        
    def feature_gen(self, fname:str=None,index:int=None,func=None) -> None:
        
        self.df['P_sqrt'] = self.df.iloc[:,1].apply(lambda x: x ** 0.5)
        
    def data_flow(self,columns_idx:tuple = (1,3,3,5), idx:tuple=None, split_idx:int = 800) -> torch.utils.data.DataLoader:
        """ Data prep pipeline

        Args:
            columns_idx (tuple, optional): Columns to be selected (sliced 1:2 3:4) for feature fitting. Defaults to (1,3,3,5). 
            idx (tuple, optional): 2|3 indexes to be selected for feature fitting. Defaults to None. Use either idx or columns_idx (for F:R->R idx, for F:R->R2 columns_idx)
            split_idx (int) : Index to split for training
            
        Returns:
            torch.utils.data.DataLoader: Torch native dataloader
        """
        batch_size=2
        
        if idx!=None:
            self.len_idx = len(idx)
            if len(idx)==2:
                self.X = tensor(self.df.iloc[:,idx[0]].values[:split_idx]).float()
                self.Y = tensor(self.df.iloc[:,idx[1]].values[:split_idx]).float()
                batch_size = 1
            else:
                self.X = tensor(self.df.iloc[:,[idx[0],idx[1]]].values[:split_idx,:]).float()
                self.Y = tensor(self.df.iloc[:,idx[2]].values[:split_idx]).float()
        else:
            self.X = tensor(self.df.iloc[:,columns_idx[0]:columns_idx[1]].values[:split_idx,:]).float()
            self.Y = tensor(self.df.iloc[:,columns_idx[2]:columns_idx[3]].values[:split_idx]).float()
        print('Shapes for debug: (X,Y)',self.X.shape, self.Y.shape)
        train_data = torch.utils.data.TensorDataset(self.X, self.Y)
        Xtrain = torch.utils.data.DataLoader(train_data,batch_size=batch_size)
        self.input_dim = self.X.size(-1)
        self.indexes = idx if idx else columns_idx
        self.column_names = [ self.df.columns[i] for i in self.indexes ]
        return Xtrain
        
    def init_seed(self):
        """ Initializes seed for torch optional()
        """
        if self.type == 'PU':
            torch.manual_seed(self.seed)
        
    def train_epoch(self,X, model, loss_function, optim):
        for i,data in enumerate(X):
                Y_pred = model(data[0])
                loss = loss_function(Y_pred, data[1])
                
                # mean_abs_percentage_error = MeanAbsolutePercentageError()
                # ape = mean_abs_percentage_error(Y_pred, data[1])
                
                loss.backward()
                optim.step()
                optim.zero_grad()
            
                
                ape_norm = abs(np.mean((Y_pred.detach().numpy()-data[1].detach().numpy())/(data[1].detach().numpy()+0.1)))
                print('APE =',ape_norm)
                self.loss_history.append(loss.data.item())
                self.ape_history.append(None if ape_norm >1 else ape_norm)
                
    def compile(self,columns:tuple=None,idx:tuple=None, optim:torch.optim = torch.optim.AdamW,loss:nn=nn.L1Loss) -> None:
        """ Builds model, loss, optimizer. Has defaults
        Args:
            columns (tuple, optional): Columns to be selected for feature fitting. Defaults to (1,3,3,5).
            optim - torch Optimizer
            loss - torch Loss function (nn)
        """
        
        self.columns = columns
        # if self.type == 'PU':
            # self.init_seed()
        if not(columns):
            self.len_idx = 0
        else:
            self.len_idx = len(columns)
            
        if not(self.columns) and not(idx):
            self.Xtrain = self.data_flow()
        elif not(idx): 
            self.Xtrain = self.data_flow(columns_idx=self.columns)
        else:
            self.Xtrain = self.data_flow(idx=idx)
            
        if self.len_idx == 2:
            self.model = dmodel(in_features=1,hidden_features=self.dim).float()
            self.input_dim_for_check = 1
        if self.len_idx == 3:
            self.model = Net(input_dim=2,hidden_dim=self.dim).float()
        if self.len_idx == 0:
            self.model = Net(input_dim=self.input_dim,hidden_dim=self.dim).float()
        self.optim = optim(self.model.parameters(), lr=0.001)
        self.loss_function = loss()
        
        if self.input_dim_for_check:
            self.X  = self.X.reshape(-1,1)
        
        
    
    def train(self,epochs:int=10) -> None:
        """ Train model

        
        """
        self.loss_history = []
        self.ape_history = []
        self.model.train()
        self.epochs = epochs
        
        
        for j in range(self.epochs):
            self.train_epoch(self.Xtrain,self.model,self.loss_function,self.optim)
            
        plt.plot(self.loss_history,label='loss_history')
        plt.legend()
            
    def save(self,name:str='model.pt') -> None:
        torch.save(self.model,name)
        
    def export():
        pass
        
    def inference(self,X:tensor, model_name:str=None) -> np.ndarray:
        """ Inference of (pre-)trained model

        Args:
            X (tensor): your data in domain of train

        Returns:
            np.ndarray: predictions
        """
        if model_name is None:
            self.model.eval()
            
        if model_name in os.listdir('./models'):
            model = torch.load(f'./models/{model_name}')
            model.eval()
            return model(X).detach().numpy()
        
        return self.model(X).detach().numpy()
        
    def plot(self):
        self.model.eval()
        print(self.Y.shape,self.model(self.X).detach().numpy().shape,self.X.shape)
        if self.X.shape[-1] != self.model(self.X).detach().numpy().shape[-1]:
            print('Size mismatch, try 3d plot, plotting by second dim of largest tensor')
            plt.scatter(self.model(self.X).detach().numpy(),self.X[:,1],s=2,label='predicted')
            if len(self.Y.shape)!=1:
                plt.scatter(self.Y[:,1],self.X[:,1],s=1,label='real')
            else:
                plt.scatter(self.Y,self.X[:,1],s=1,label='real')
            plt.xlabel(rf'${self.column_names[0]}$')
            plt.ylabel(rf'${self.column_names[1]}$')
            plt.legend()
        else:
            plt.scatter(self.model(self.X).detach().numpy(),self.X,s=2,label='predicted')
            plt.scatter(self.Y,self.X,s=1,label='real')
            plt.xlabel(r'$X$')
            plt.ylabel(r'$Y$')
            plt.legend()
        
    def plot3d(self):
        X = self.X
        self.model.eval()
        x = X[:,0].numpy().ravel()
        y = y=X[:,1].numpy().ravel()
        z = self.model(X).detach().numpy().ravel()
        surf = px.scatter_3d(x=x, y=y,z=self.df.iloc[:,self.indexes[-1]].values[:800],opacity=0.3,
                             labels={'x':f'{self.column_names[0]}',
                                     'y':f'{self.column_names[1]}',
                                     'z':f'{self.column_names[2:]}'
                                     },title='Mesh prediction plot'
                            )
        # fig.colorbar(surf, shrink=0.5, aspect=5)
        surf.update_traces(marker_size=3)
        surf.update_layout(plot_bgcolor='#888888')
        surf.add_mesh3d(x=x, y=y, z=z, opacity=0.7,colorscale='sunsetdark',intensity=z,
            )
        surf.show()
        
    def performance(self,c=0.4) -> dict:
        a=[]
        for i in range(1000):
            a.append(100-abs(np.mean(self.df.iloc[:24,1:].values-self.df.iloc[24:,1:].sample(24).values)/(self.Y.numpy()+c))*100)
        gen_acc = np.mean(a)
        ape = (100-abs(np.mean(self.model(self.X).detach().numpy()-self.Y.numpy())*100))
        abs_ape = ape*gen_acc/100
        return {'Generator_Accuracy, %':np.mean(a),'APE_abs, %':abs_ape,'Model_APE, %': ape}
    
            
    