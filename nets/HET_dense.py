from utils.dataset_loader import get_dataset
from dense import Net

import matplotlib.pyplot as plt
import torch
import os
import numpy as np
from torch import nn, tensor
from torchmetrics import MeanAbsolutePercentageError
import pandas as pd

class dense():
    def __init__(self, typ='PU', hidden_dim = 8, dropout = True, epochs = 10):
        self.type:str = typ
        self.seed:int = 449
        self.dim = hidden_dim
        self.dropout = dropout
        self.df = get_dataset(name='test.pkl')
        self.epochs = epochs
        
    def data_flow(self,columns_idx:tuple = (1,3,3,5),split_idx:int = 800) -> torch.utils.data.DataLoader:
        """ Data prep pipeline

        Args:
            columns_idx (tuple, optional): Columns to be selected for feature fitting. Defaults to (1,3,3,5).
            split_idx (int) : Index to split for training
        Returns:
            torch.utils.data.DataLoader: Torch native dataloader
        """
        
        self.X = tensor(self.df.iloc[:,columns_idx[0]:columns_idx[1]].values[:split_idx,:]).float()
        self.Y = tensor(self.df.iloc[:,columns_idx[2]:columns_idx[3]].values[:split_idx]).float()
        print('Shapes for debug',self.X.shape, self.Y.shape)
        train_data = torch.utils.data.TensorDataset(self.X, self.Y)
        Xtrain = torch.utils.data.DataLoader(train_data,batch_size=2)
        return Xtrain
        
    def init_seed(self):
        """ Initializes seed for torch
        """
        if self.type == 'PU':
            torch.manual_seed(self.seed)
        
    def train_epoch(X, model, loss_function, optim):
        loss_history = []
        for i,data in enumerate(X):
                Y_pred = model(data[0])
                loss = loss_function(Y_pred, data[1])
                
                mean_abs_percentage_error = MeanAbsolutePercentageError()
                ape = mean_abs_percentage_error(Y_pred, data[1])
                
                loss.backward()
                optim.step()
                optim.zero_grad()
            
                print('APE =',ape.item())
                
    def compile(self) -> None:
        """ Builds model, loss, optimizer
        """
        if self.type == 'PU':
            self.init_seed()
        self.model = Net(hidden_dim=self.hidden_dim,dropout=self.dropout).float()
        self.optim = torch.optim.AdamW(self.model.parameters(), lr=0.001)
        self.loss_function = nn.L1Loss()
    
    def train(self) -> None:
        self.model.train()
        Xtrain = self.data_flow()
        for j in range(self.epochs):
            self.train_epoch(Xtrain,self.model,self.loss_function,self.optim)
            
    def save(self,name:str='model.pt') -> None:
        torch.save(self.model,name)
        
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
        plt.scatter(self.model(self.X).detach().numpy(),self.X,s=2,label='preidcted')
        plt.scatter(self.Y,self.X,s=1,label='real')
        plt.xlabel(r'$X$')
        plt.ylabel(r'$Y$')
        plt.legend()
        
    def performance(self,c=0.4) -> dict:
        a=[]
        for i in range(1000):
            a.append(100-abs(np.mean(self.df.iloc[:24,1:].values-self.df.iloc[24:,1:].sample(24).values)/(self.Y.numpy()+c))*100)
        gen_acc = np.mean(a)
        ape = (100-abs(np.mean(self.model(self.X).detach().numpy()-self.Y.numpy())*100))
        abs_ape = ape*gen_acc/10000
        return {'Generator_Accuracy, %':np.mean(a),'APE_abs, %':abs_ape,'Model_APE, %': ape}
    
            
    