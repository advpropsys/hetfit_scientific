import numpy as np
import _pickle
import pandas as pd
import sklearn
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import svm
import matplotlib.pyplot as plt

np.random.seed(4269)

class dataset:
    """ Creates dataset from input source
    """
    def __init__(self,number_samples:int, name:str):
        self.sample_size = number_samples
        self.name = name
        self.high = []
        self.low = []
        self.samples = []
        
    def generate(self):
        
        
        with open(r"./data/dataset.csv", "rb") as input_file:
            local = pd.read_csv(input_file)
            local = local.iloc[:,1:]
            X = local.iloc[:,0].values.reshape(-1, 1)
            
        for name in local.columns:
            self.high.append(max(local[f'{name}']))
            self.low.append(min(local[f'{name}']))
        xes = []
        for x in range(self.sample_size):
            newx = np.random.randint(self.low[0],self.high[0])
            xes.append(int(newx)) 
        xes = np.asarray(xes).reshape(-1, 1)
        for idx,elem in enumerate(local.columns[1:]):
            y = local.iloc[:,idx].values
            y = y[:23]
            regr = make_pipeline(StandardScaler(), svm.SVR(kernel='rbf',gamma='scale',C=10,epsilon=0.45))
            regr.fit(X, y)   
            self.samples.append(regr.predict(xes))
            
        samples=pd.DataFrame(self.samples).T
        samples.join(pd.DataFrame(xes),how='left',lsuffix='_0')
        local = pd.concat([local,samples],ignore_index=True)
        return samples.corr()
                
            