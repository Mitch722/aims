import numpy as np
import sklearn as skl
from matplotlib import pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA as p_c_a


class DataHandler:

    def __init__(self, data_set):
        self.rich_data = data_set
        self.data = data_set.data
        self.target = data_set.target


if __name__ == "__main__":

    cancer_data = DataHandler(load_breast_cancer())
    train_data = cancer_data.data[0:400, :]

    pca = p_c_a(n_components=3)
    pca.fit(train_data.T)

    import pdb
    pdb.set_trace()

