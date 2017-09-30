import numpy as np
import matplotlib.pyplot as plt

class DataReader:
    """DataReader to do inspection about input sdata"""
    if __name__ == '__main__':

        # load data
        X_train = np.load('data/X_train.npy')
        X_train_3D = np.reshape(X_train, (-1, 176, 208, 176))

        print(X_train_3D)

        # visualization
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        # for first image

        # for averaged image
