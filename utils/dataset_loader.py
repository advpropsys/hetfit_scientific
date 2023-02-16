from utils import dataset
import os
import _pickle




def get_dataset(raw=False, sample_size=1000, name='dataset.pkl') -> _pickle:
    if not(raw):
        if name not in os.listdir('.'):
            ldat = dataset(sample_size)
            ldat.generate(name)
        with open(r"dataset.pkl", "rb") as input_file:
            buffer = _pickle.load(input_file)
    else:
        with open(r"source.pkl", "rb") as input_file:
            buffer = _pickle.load(input_file)
    return buffer