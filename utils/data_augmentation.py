import numpy as np
import _pickle
import pandas as pd
# import tensorflow as tf
# from keras.layers import Input,Dense
# from keras.models import Model
# from sklearn.model_selection import train_test_split

from utils.ndgan import DCGAN 

np.random.seed(4269)



class dataset():
    """ Creates dataset from input source
    """
    def __init__(self,number_samples:int, name:str,source:str,boundary_conditions:list=None):
        """ Init

        Args:
            number_samples (int): number of samples to be genarated
            name (str): name of dataset
            source (str): source file
            boundary_conditions (list): y1,y2,x1,x2
        """
        self.sample_size = number_samples
        self.name = name
        self.samples = []
        self.encoding_dim = 8
        self.latent_dim = 16
        self.source = source
        self.boundary_conditions = boundary_conditions
    
    def generate(self):
        """
        The function takes in a dataframe, normalizes it, and then trains a DCGAN on it. 
        
        The DCGAN is a type of generative adversarial network (GAN) that is used to generate new data. 
        
        The DCGAN is trained on the normalized dataframe, and then the DCGAN is used to generate new
        data. 
        
        The new data is then concatenated with the original dataframe, and the new dataframe is saved as
        a pickle file. 
        
        The new dataframe is then returned.
        :return: The dataframe is being returned.
        """
        with open(f"./data/{self.source}", "rb") as input_file:
            local = pd.read_csv(input_file)
            dfs = local.drop("Name",axis=1)
            dfs = (dfs-dfs.min())/(dfs.max()-dfs.min())
            dfs = pd.concat([local.Name,dfs],1)
            
        self.vae = DCGAN(self.latent_dim,dfs)
        
        self.vae.start_training()
        self.samples = self.vae.predict(self.sample_size)
    
        if self.boundary_conditions:
            self.samples=self.samples[((self.samples[:,0]>self.boundary_conditions[2]) & (self.samples[:,0] < self.boundary_conditions[-1]))&((self.samples[:,0]>self.boundary_conditions[0]) & (self.samples[:,0] < self.boundary_conditions[1]))]
            
        print("Samples:",self.samples)
        dataframe = pd.concat([dfs,pd.DataFrame(self.samples,columns=dfs.columns[1:])])
        dataframe.to_pickle(f'./data/{self.name}')
        print(dataframe)
        
        
        
        return dataframe
                
            