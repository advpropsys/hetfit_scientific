from data_augmentation import dataset
import os
import _pickle
import pandas as pd




def get_dataset(raw:bool=False, sample_size:int=1000, name:str='dataset.pkl') -> _pickle:
    """ Gets augmented dataset

    Args:
        raw (bool, optional): either to use source data or augmented. Defaults to False.
        sample_size (int, optional): sample size. Defaults to 1000.
        name (str, optional): name of wanted dataset. Defaults to 'dataset.pkl'.

    Returns:
        _pickle: pickle buffer
    """
    
    if not(raw):
        if name not in os.listdir('./data'):
            ldat = dataset(sample_size,name)
            ldat.generate()
        with open(f"./data/{name}", "rb") as input_file:
            buffer = _pickle.load(input_file)
    else:
        with open(r"./data/dataset.csv", "rb") as input_file:
            buffer = pd.read_csv(input_file)
    return buffer

print(get_dataset(name='test.pkl'))