import numpy as np
import _pickle
import pandas as pd
import tensorflow as tf
from keras.layers import Input,Dense
from keras.models import Model
from sklearn.model_selection import train_test_split

from utils.ndgan import DCGAN 

np.random.seed(4269)



class dataset:
    """ Creates dataset from input source
    """
    def __init__(self,number_samples:int, name:str):
        self.sample_size = number_samples
        self.name = name
        self.samples = []
        self.encoding_dim = 8
        self.latent_dim = 16
        
    
    
    def generate(self):
        with open(r"./data/dataset.csv", "rb") as input_file:
            local = pd.read_csv(input_file)
            dfs = local.drop("Name",axis=1)
            dfs = (dfs-dfs.min())/(dfs.max()-dfs.min())
            dfs = pd.concat([local.Name,dfs],1)
            
        self.vae = DCGAN(self.latent_dim,dfs)
        
        self.vae.start_training()
        self.samples = self.vae.predict(self.sample_size)
        print("Samples:",self.samples)
        dataframe = pd.concat([dfs,pd.DataFrame(self.samples,columns=dfs.columns[1:])])
        dataframe.to_pickle(f'./data/{self.name}')
        print(dataframe)
        
        
        
        return dataframe
                
            