import numpy as np
import sklearn as skl
from matplotlib import pyplot as plt
from sklearn.datasets import load_breast_cancer


class DataHandler:

    def __init__(self, data_set, n_comps):
        self.rich_data = data_set
        self.data = data_set.data
        self.target = data_set.target
        self.n_comps = n_comps

    def mod_gram_schmid(self, orthog_vecs):
        """
        modified gram_schmid algorithm for orthogonal vectors
        Args:
            orthog_vecs: list of numpy arrays .reshape(-1, 1)

        Returns:
            u_out: othogonal vector to those in list orthog_vecs
        """
        shape_vec = orthog_vecs[0].shape[0]
        v_init = 0.1*np.random.randn(shape_vec, 1)
        u = v_init / np.linalg.norm(v_init)
        for i in range(len(orthog_vecs)):

            u = u - orthog_vecs[i] * self.proj_u(u, orthog_vecs[i])
            u = u / np.linalg.norm(u)
        u_out = u
        return u_out

    def v_gram_schmid(self, v_init, orthog_vecs):
        """
        Make v_init vector orthogonal to the other vectors in orthog_vecs list
        Args:
            v_init:    :
            orthog_vecs: list of numpy arrays .reshape(-1, 1)

        Returns:
            u_out: othogonal vector to those in list orthog_vecs
        """
        u = v_init / np.linalg.norm(v_init)
        for i in range(len(orthog_vecs)):

            u = u - orthog_vecs[i] * self.proj_u(u, orthog_vecs[i])
            u = u / np.linalg.norm(u)

        u_out = u
        return u_out

    @staticmethod
    def proj_u(u, v_k):
        pj_u = (u.T @ v_k) / (u.T @ u)
        return pj_u

    def find_pca(self, x_data, conv_fact=0.000001):
        """
        Calculates the eigen-vectors for PCA
        Args:
            x_data   :  input data
            conv_fact:  covergence factor: when to stop covergence

        Returns:
            pca_comps: eigen vectors
        """
        shape_vec = x_data.shape[0]
        v_init = 0.1*np.random.randn(shape_vec, 1)
        v_t0 = v_init / np.linalg.norm(v_init)

        xx_t = x_data @ x_data.T
        v_t1 = xx_t @ v_t0
        v_t1 = v_t1 / np.linalg.norm(v_t1)

        pca_comps = []
        for i in range(self.n_comps):

            converge_val = 0.0
            while converge_val <= 1 - conv_fact:

                v_t1 = xx_t @ v_t0
                v_t1 = v_t1 / np.linalg.norm(v_t1)
                v_t1 = self.v_gram_schmid(v_t1, pca_comps)

                converge_val = self.converge_test(v_t0, v_t1)

                v_t0 = v_t1

            pca_comps.append(v_t1)
            v_t0 = self.mod_gram_schmid(pca_comps)
            v_t0 = v_t0 / np.linalg.norm(v_t0)

        return pca_comps

    @staticmethod
    def converge_test(vec1, vec2):
        """
        Test of vector convergence
        Args:
            vec1: input vector
            vec2: input vector

        Returns:
            returns a value between 0.0 and 1.0
        """
        vec1 = vec1 / np.linalg.norm(vec1)
        vec2 = vec2 / np.linalg.norm(vec2)
        one_ness = np.absolute(vec1.T @ vec2)
        return one_ness

    @staticmethod
    def vec_list_npmat(a_list):
        """
        Take a list of vectors.reshape(-1, 1) and turn into a large matrix
        Args:
            a_list: list of vectors .reshape(-1, 1)

        Returns:
            mat: matrix from vector list
        """
        mat = np.concatenate((a_list[:]), axis=1)
        return mat


if __name__ == "__main__":

    cancer_data = DataHandler(load_breast_cancer(), 3)
    train_data = cancer_data.data[0:400, :]

    orth_vecs = [np.array([1, 0, 0]).reshape(-1, 1), np.array([0, 1, 0]).reshape(-1, 1)]

    test_vec = cancer_data.mod_gram_schmid(orth_vecs)
    print(test_vec)

    pcas = cancer_data.find_pca(train_data)
    print(len(pcas), pcas[0].shape)

    big_v = cancer_data.vec_list_npmat(pcas)
    print(big_v.shape)
    print(big_v[0:5, :])
    print(cancer_data.converge_test(pcas[0], pcas[1]))
