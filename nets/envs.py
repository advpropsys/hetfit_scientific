from utils.dataset_loader import get_dataset
from nets.dense import Net
from nets.deep_dense import dmodel
from PINN.pinns import *

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import os
import numpy as np
from torch import nn, tensor
import pandas as pd
import plotly.express as px
from sklearn.linear_model import SGDRegressor
from sklearn.feature_selection import SelectFromModel

class SCI(): #Scaled Computing Interface
    """ Scaled computing interface.
    Args:
            hidden_dim (int, optional): Max demension of hidden linear layer. Defaults to 200. Should be >80 in not 1d case
            dropout (bool, optional): LEGACY, don't use. Defaults to True.
            epochs (int, optional): Optionally specify epochs here, but better in train. Defaults to 10.
            dataset (str, optional): dataset to be selected from ./data. Defaults to 'test.pkl'. If name not exists, code will generate new dataset with upcoming parameters.
            sample_size (int, optional): Samples to be generated (note: BEFORE applying boundary conditions). Defaults to 1000.
            source (str, optional): Source from which data will be generated. Better to not change. Defaults to 'dataset.csv'.
            boundary_conditions (list, optional): If sepcified, whole dataset will be cut rectangulary. Input list is [ymin,ymax,xmin,xmax] type. Defaults to None.
    """
    def __init__(self, hidden_dim:int = 200, dropout:bool = True, epochs:int = 10, dataset:str = 'test.pkl',sample_size:int=1000,source:str='dataset.csv',boundary_conditions:list=None):
        """Init
        Args:
            hidden_dim (int, optional): Max demension of hidden linear layer. Defaults to 200. Should be >80 in not 1d case
            dropout (bool, optional): LEGACY, don't use. Defaults to True.
            epochs (int, optional): Optionally specify epochs here, but better in train. Defaults to 10.
            dataset (str, optional): dataset to be selected from ./data. Defaults to 'test.pkl'. If name not exists, code will generate new dataset with upcoming parameters.
            sample_size (int, optional): Samples to be generated (note: BEFORE applying boundary conditions). Defaults to 1000.
            source (str, optional): Source from which data will be generated. Better to not change. Defaults to 'dataset.csv'.
            boundary_conditions (list, optional): If sepcified, whole dataset will be cut rectangulary. Input list is [ymin,ymax,xmin,xmax] type. Defaults to None.
        """
        self.type:str = 'legacy'
        self.seed:int = 449
        self.dim = hidden_dim
        self.dropout = dropout
        self.df = get_dataset(sample_size=sample_size,source=source,name=dataset,boundary_conditions=boundary_conditions)
        self.epochs = epochs
        self.len_idx = 0
        self.input_dim_for_check = 0
        
    def feature_gen(self, base:bool=True, fname:str=None,index:int=None,func=None) -> None:
        """ Generate new features. If base true, generates most obvious ones. You can customize this by adding 
        new feature as name of column - fname, index of parent column, and lambda function which needs to be applied elementwise.
        Args:
            base (bool, optional):  Defaults to True.
            fname (str, optional): Name of new column. Defaults to None.
            index (int, optional): Index of parent column. Defaults to None.
            func (_type_, optional): lambda function. Defaults to None.
        """
        
        if base:
            self.df['P_sqrt'] = self.df.iloc[:,1].apply(lambda x: x ** 0.5)
            self.df['j'] = self.df.iloc[:,1]/(self.df.iloc[:,3]*self.df.iloc[:,4])
            self.df['B'] = self.df.iloc[:,-1].apply(lambda x: x ** 2).apply(lambda x:1 if x>1 else x)
            self.df['nu_t'] = self.df.iloc[:,7]**2/(2*self.df.iloc[:,6]*self.df.P)
            
        if fname and index and func:
            self.df[fname] = self.df.iloc[:,index].apply(func)
        
    def feature_importance(self,X:pd.DataFrame,Y:pd.Series,verbose:int=1):
        """ Gets feature importance by SGD regression and score selection. Default threshold is 1.25*mean
        input X as self.df.iloc[:,(columns of choice)]
              Y as self.df.iloc[:,(column of choice)]
        Args:
            X (pd.DataFrame): Builtin DataFrame
            Y (pd.Series): Builtin Series
            verbose (int, optional): either to or to not print actual report. Defaults to 1.
        Returns:
            Report (str)
        """
        
        mod = SGDRegressor()
        
        selector = SelectFromModel(mod,threshold='1.25*mean')
        selector.fit(np.array(X),np.array(Y))
        
        if verbose:
            print(f'\n Report of feature importance: {dict(zip(X.columns,selector.estimator_.coef_))}')
        for i in range(len(selector.get_support())):
            if selector.get_support()[i]:
                print(f'-rank 1 PASSED:',X.columns[i])
            else:
                print(f'-rank 0 REJECT:',X.columns[i])
        return f'\n Report of feature importance: {dict(zip(X.columns,selector.estimator_.coef_))}'
        
    def data_flow(self,columns_idx:tuple = (1,3,3,5), idx:tuple=None, split_idx:int = 800) -> torch.utils.data.DataLoader:
        """ Data prep pipeline
        It is called automatically, don't call it in your code.
        Args:
            columns_idx (tuple, optional): Columns to be selected (sliced 1:2 3:4) for feature fitting. Defaults to (1,3,3,5). 
            idx (tuple, optional): 2|3 indexes to be selected for feature fitting. Defaults to None. Use either idx or columns_idx (for F:R->R idx, for F:R->R2 columns_idx)
            split_idx (int) : Index to split for training
            
        Returns:
            torch.utils.data.DataLoader: Torch native dataloader
        """
        batch_size=2
        
        self.split_idx=split_idx
        
        if idx!=None:
            self.len_idx = len(idx)
            if len(idx)==2:
                self.X = tensor(self.df.iloc[:,idx[0]].values[:split_idx]).float()
                self.Y = tensor(self.df.iloc[:,idx[1]].values[:split_idx]).float()
                batch_size = 1
            else:
                self.X = tensor(self.df.iloc[:,[*idx[:-1]]].values[:split_idx,:]).float()
                self.Y = tensor(self.df.iloc[:,idx[2]].values[:split_idx]).float()
        else:
            self.X = tensor(self.df.iloc[:,columns_idx[0]:columns_idx[1]].values[:split_idx,:]).float()
            self.Y = tensor(self.df.iloc[:,columns_idx[2]:columns_idx[3]].values[:split_idx]).float()
            
        print('Shapes for debug: (X,Y)',self.X.shape, self.Y.shape)
        train_data = torch.utils.data.TensorDataset(self.X, self.Y)
        Xtrain = torch.utils.data.DataLoader(train_data,batch_size=batch_size)
        self.input_dim = self.X.size(-1)
        self.indexes = idx if idx else columns_idx
        self.column_names = [self.df.columns[i] for i in self.indexes]
        return Xtrain
        
    def init_seed(self,seed):
        """ Initializes seed for torch optional()
        """
        
        torch.manual_seed(seed)
        
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
                if (i+1)%200==0:
                    print(f'Iter {i+1} APE =',ape_norm)
                self.loss_history.append(loss.data.item())
                self.ape_history.append(None if ape_norm >1 else ape_norm)
                
    def compile(self,columns:tuple=None,idx:tuple=None, optim:torch.optim = torch.optim.AdamW,loss:nn=nn.L1Loss, model:nn.Module = dmodel, custom:bool=False, lr:float=0.0001) -> None:
        """ Builds model, loss, optimizer. Has defaults
        Args:
            columns (tuple, optional): Columns to be selected for feature fitting. Defaults to (1,3,3,5).
            optim - torch Optimizer. Default AdamW
            loss - torch Loss function (nn). Defaults to L1Loss
        """
        
        self.columns = columns
        
                
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
            
        if custom:
            self.model = model()
            self.loss_function = loss()
            self.optim = optim(self.model.parameters(), lr=lr)
            if self.len_idx == 2:
                self.input_dim_for_check = 1
        else: 
            if self.len_idx == 2:
                self.model = model(in_features=1,hidden_features=self.dim).float()
                self.input_dim_for_check = 1
            if self.len_idx == 3:
                self.model = Net(input_dim=2,hidden_dim=self.dim).float()
            if (self.len_idx == 0) or self.columns:
                self.model = Net(input_dim=self.input_dim,hidden_dim=self.dim).float()
                
            self.optim = optim(self.model.parameters(), lr=lr)
            self.loss_function = loss()
            
        if self.input_dim_for_check:
            self.X  = self.X.reshape(-1,1)
        
       
    
    def train(self,epochs:int=10) -> None:
        """ Train model
        If sklearn instance uses .fit()
        
        epochs - optional
        """
        if 'sklearn' in str(self.model.__class__):
            self.model.fit(np.array(self.X),np.array(self.Y))
            plt.scatter(self.X,self.model.predict(self.X))
            plt.scatter(self.X,self.Y)
            plt.xlabel('Xreal')
            plt.ylabel('Ypred/Yreal')
            plt.show()
            return print('Sklearn model fitted successfully')
        else:
            self.model.train()
            
        self.loss_history = []
        self.ape_history = []
        
        self.epochs = epochs
        
        
        for j in range(self.epochs):
            self.train_epoch(self.Xtrain,self.model,self.loss_function,self.optim)
            
        plt.plot(self.loss_history,label='loss_history')
        plt.legend()
            
    def save(self,name:str='model.pt') -> None:
        torch.save(self.model,name)
        
    def onnx_export(self,path:str='./models/model.onnx'):
        torch.onnx.export(self.model,self.X,path)
        
    def jit_export(self,path:str='./models/model.pt'):
        """Exports properly defined model to jit
        Args:
            path (str, optional): path to models. Defaults to './models/model.pt'.
        """
        torch.jit.save(self.model,path)
        
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
        """ Automatic 2d plot
        """
        self.model.eval()
        print(self.Y.shape,self.model(self.X).detach().numpy().shape,self.X.shape)
        if self.X.shape[-1] != self.model(self.X).detach().numpy().shape[-1]:
            print('Size mismatch, try 3d plot, plotting by first dim of largest tensor')
            if len(self.X.shape)==1:
                X = self.X
            else: X = self.X[:,0]
            plt.scatter(X,self.model(self.X).detach().numpy(),label='predicted',s=2)
            if len(self.Y.shape)!=1:
                plt.scatter(X,self.Y[:,1],s=1,label='real')
            else:
                plt.scatter(X,self.Y,s=1,label='real')
            plt.xlabel(rf'${self.column_names[0]}$')
            plt.ylabel(rf'${self.column_names[1]}$')
            plt.legend()
        else:
            plt.scatter(self.X,self.model(self.X).detach().numpy(),s=2,label='predicted')
            plt.scatter(self.X,self.Y,s=1,label='real')
            plt.xlabel(r'$X$')
            plt.ylabel(r'$Y$')
            plt.legend()
        
    def plot3d(self,colX=0,colY=1):
        """ Plot of inputs and predicted data in mesh format
        Returns:
            plotly plot
        """
        X = self.X
        self.model.eval()
        x = X[:,colX].numpy().ravel()
        y = X[:,colY].numpy().ravel()
        z = self.model(X).detach().numpy().ravel()
        surf = px.scatter_3d(x=x, y=y,z=self.df.iloc[:,self.indexes[-1]].values[:self.split_idx],opacity=0.3,
                             labels={'x':f'{self.column_names[colX]}',
                                     'y':f'{self.column_names[colY]}',
                                     'z':f'{self.column_names[-1]}'
                                     },title='Mesh prediction plot'
                            )
        # fig.colorbar(surf, shrink=0.5, aspect=5)
        surf.update_traces(marker_size=3)
        surf.update_layout(plot_bgcolor='#888888')
        surf.add_mesh3d(x=x, y=y, z=z, opacity=0.7,colorscale='sunsetdark',intensity=z,
            )
        # surf.show()
        
        return surf
    def performance(self,c=0.4) -> dict:
        """ Automatic APE based performance if applicable, else returns nan
        Args:
             c (float, optional): ZDE mitigation constant. Defaults to 0.4.
        Returns:
            dict: {'Generator_Accuracy, %':np.mean(a),'APE_abs, %':abs_ape,'Model_APE, %': ape}
        """
        a=[]
        for i in range(1000):
            a.append(100-abs(np.mean(self.df.iloc[1:24,1:8].values-self.df.iloc[24:,1:8].sample(23).values)/(self.Y.numpy()[1:]+c))*100)
        gen_acc = np.mean(a)
        ape = (100-abs(np.mean(self.model(self.X).detach().numpy()-self.Y.numpy()[1:])*100))
        abs_ape = ape*gen_acc/100
        return {'Generator_Accuracy, %':np.mean(a),'APE_abs, %':abs_ape,'Model_APE, %': ape}
    
    def performance_super(self,c=0.4,real_data_column_index:tuple = (1,8),real_data_samples:int=23, generated_length:int=1000) -> dict:
        """Performance by custom parameters. APE loss
        Args:
            c (float, optional): ZDE mitigation constant. Defaults to 0.4.
            real_data_column_index (tuple, optional): Defaults to (1,8).
            real_data_samples (int, optional): Defaults to 23.
            generated_length (int, optional): Defaults to 1000.
        Returns:
            dict: {'Generator_Accuracy, %':np.mean(a),'APE_abs, %':abs_ape,'Model_APE, %': ape}
        """
        a=[]
        for i in range(1000):
            a.append(100-abs(np.mean(self.df.iloc[1:real_data_samples+1,real_data_column_index[0]:real_data_column_index[1]].values-self.df.iloc[real_data_samples+1:,real_data_column_index[0]:real_data_column_index[1]].sample(real_data_samples).values)/(self.Y.numpy()[1:]+c))*100)
        gen_acc = np.mean(a)
        ape = (100-abs(np.mean(self.model(self.X).detach().numpy()-self.Y.numpy()[1:])*100))
        abs_ape = ape*gen_acc/100
        return {'Generator_Accuracy, %':np.mean(a),'APE_abs, %':abs_ape,'Model_APE, %': ape}
    
    def performance_super(self,c=0.4,real_data_column_index:tuple = (1,8),real_data_samples:int=23, generated_length:int=1000) -> dict:
        a=[]
        for i in range(1000):
            a.append(100-abs(np.mean(self.df.iloc[1:real_data_samples+1,real_data_column_index[0]:real_data_column_index[1]].values-self.df.iloc[real_data_samples+1:,real_data_column_index[0]:real_data_column_index[1]].sample(real_data_samples).values)/(self.Y.numpy()[1:]+c))*100)
        gen_acc = np.mean(a)
        ape = (100-abs(np.mean(self.model(self.X).detach().numpy()-self.Y.numpy()[1:])*100))
        abs_ape = ape*gen_acc/100
        return {'Generator_Accuracy, %':np.mean(a),'APE_abs, %':abs_ape,'Model_APE, %': ape}
    def performance_super(self,c=0.4,real_data_column_index:tuple = (1,8),real_data_samples:int=23, generated_length:int=1000) -> dict:
        a=[]
        for i in range(1000):
            a.append(100-abs(np.mean(self.df.iloc[1:real_data_samples+1,real_data_column_index[0]:real_data_column_index[1]].values-self.df.iloc[real_data_samples+1:,real_data_column_index[0]:real_data_column_index[1]].sample(real_data_samples).values)/(self.Y.numpy()[1:]+c))*100)
        gen_acc = np.mean(a)
        ape = (100-abs(np.mean(self.model(self.X).detach().numpy()-self.Y.numpy()[1:])*100))
        abs_ape = ape*gen_acc/100
        return {'Generator_Accuracy, %':np.mean(a),'APE_abs, %':abs_ape,'Model_APE, %': ape}
    
class RCI(SCI): #Real object interface
    """ Real values interface, uses different types of NN, NO scaling.
    Parent:
        SCI()
    """
    def __init__(self,*args,**kwargs):
        super(RCI,self).__init__()
        
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
            
            real_scale = pd.read_csv('data/dataset.csv').iloc[17,1:].to_numpy()
            self.df.iloc[:,1:] = self.df.iloc[:,1:] * real_scale
            
            self.split_idx=split_idx
            
                        
            if idx!=None:
                self.len_idx = len(idx)
                if len(idx)==2:
                    self.X = tensor(self.df.iloc[:,idx[0]].values[:split_idx].astype(float)).float()
                    self.Y = tensor(self.df.iloc[:,idx[1]].values[:split_idx].astype(float)).float()
                    batch_size = 1
                else:
                    self.X = tensor(self.df.iloc[:,[idx[0],idx[1]]].values[:split_idx,:].astype(float)).float()
                    self.Y = tensor(self.df.iloc[:,idx[2]].values[:split_idx].astype(float)).float()
            else:
                self.X = tensor(self.df.iloc[:,columns_idx[0]:columns_idx[1]].values[:split_idx,:].astype(float)).float()
                self.Y = tensor(self.df.iloc[:,columns_idx[2]:columns_idx[3]].values[:split_idx].astype(float)).float()
            self.Y = self.Y.abs()
            self.X = self.X.abs()
            
            print('Shapes for debug: (X,Y)',self.X.shape, self.Y.shape)
            train_data = torch.utils.data.TensorDataset(self.X, self.Y)
            Xtrain = torch.utils.data.DataLoader(train_data,batch_size=batch_size)
            self.input_dim = self.X.size(-1)
            self.indexes = idx if idx else columns_idx
            self.column_names = [ self.df.columns[i] for i in self.indexes ]
            
            
            
            
            return Xtrain
        
    def compile(self,columns:tuple=None,idx:tuple=(3,1), optim:torch.optim = torch.optim.AdamW,loss:nn=nn.L1Loss, model:nn.Module = PINNd_p,lr:float=0.001) -> None:
        """ Builds model, loss, optimizer. Has defaults
        Args:
            columns (tuple, optional): Columns to be selected for feature fitting. Defaults to None.
            idx (tuple, optional): indexes to be selected Default (3,1)
            optim - torch Optimizer
            loss - torch Loss function (nn)
        """
        
        self.columns = columns
        
                
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
        
        self.model = model().float()
        self.input_dim_for_check = self.X.size(-1)
                
        self.optim = optim(self.model.parameters(), lr=lr)
        self.loss_function = loss()
            
        if self.input_dim_for_check == 1:
            self.X  = self.X.reshape(-1,1)
    def plot(self):
        """ Plots 2d plot of prediction vs real values
        """
        self.model.eval()
        if 'PINN' in str(self.model.__class__):
            self.preds=np.array([])
            for i in self.X:
                self.preds = np.append(self.preds,self.model(i).detach().numpy()) 
        print(self.Y.shape,self.preds.shape,self.X.shape)
        if self.X.shape[-1] != self.preds.shape[-1]:
            print('Size mismatch, try 3d plot, plotting by first dim of largest tensor')
            try: X = self.X[:,0]
            except:
                X = self.X
                pass
            plt.scatter(X,self.preds,label='predicted',s=2)
            if self.Y.shape[-1]!=1:
                sns.scatterplot(x=X,y=self.Y,s=2,label='real')
            else:
                sns.scatterplot(x=X,y=self.Y,s=1,label='real')
            plt.xlabel(rf'${self.column_names[0]}$')
            plt.ylabel(rf'${self.column_names[1]}$')
            plt.legend()
        else:
            sns.scatterplot(x=self.X,y=self.preds,s=2,label='predicted')
            sns.scatterplot(x=self.X,y=self.Y,s=1,label='real')
            plt.xlabel(r'$X$')
            plt.ylabel(r'$Y$')
            plt.legend()
            
    def performance(self,c=0.4) -> dict:
        """RCI performnace. APE errors.
        Args:
            c (float, optional): correction constant to mitigate division by 0 error. Defaults to 0.4.
        Returns:
            dict: {'Generator_Accuracy, %':np.mean(a),'APE_abs, %':abs_ape,'Model_APE, %': ape}
        """
        a=[]
        for i in range(1000):
            dfcopy = (self.df.iloc[:,1:8]-self.df.iloc[:,1:8].min())/(self.df.iloc[:,1:8].max()-self.df.iloc[:,1:8].min())
            a.append(100-abs(np.mean(dfcopy.iloc[1:24,1:].values-dfcopy.iloc[24:,1:].sample(23).values)/(dfcopy.iloc[1:24,1:].values+c))*100)
        gen_acc = np.mean(a)
        
        ape = (100-abs(np.mean(self.preds-self.Y.numpy())*100))
        abs_ape = ape*gen_acc/100
        return {'Generator_Accuracy, %':np.mean(a),'APE_abs, %':abs_ape,'Model_APE, %': ape}
    
            
    