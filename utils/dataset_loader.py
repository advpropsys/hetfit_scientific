from utils.data_augmentation import dataset
import os
import pickle
import pandas as pd




def get_dataset(raw:bool=False, sample_size:int=1000, name:str='dataset.pkl',source:str='dataset.csv',boundary_conditions:list=None) -> _pickle:
    """ Gets augmented dataset

    Args:
        raw (bool, optional): either to use source data or augmented. Defaults to False.
        sample_size (int, optional): sample size. Defaults to 1000.
        name (str, optional): name of wanted dataset. Defaults to 'dataset.pkl'.
        boundary_conditions (list,optional): y1,y2,x1,x2.
    Returns:
        _pickle: pickle buffer
    """
    print(os.listdir('./data'))
    if not(raw):
        if name not in os.listdir('./data'):
            ldat = dataset(sample_size,name,source,boundary_conditions)
            ldat.generate()
        with open(f"./data/{name}", "rb") as input_file:
            buffer = pickle.load(input_file)
    else:
        with open(f"./data/{source}", "rb") as input_file:
            buffer = pd.read_csv(input_file)
    return buffer

