from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_array
import numpy as np


class IntensityHistogram(BaseEstimator, TransformerMixin):
    """Feature from intensity histogram of 3D images"""

    # divide 3d image into cells and make histogram per cell


    def __init__(self, x_cell_number=8, y_cell_number=8, z_cell_number=8, bin_number=45):
        # image dimension
        self.IMAGE_DIM_X = 176
        self.IMAGE_DIM_Y = 208
        self.IMAGE_DIM_Z = 176
        self.BIN_MAX = 4500

        # member variables
        self.x_cell_number = x_cell_number
        self.y_cell_number = y_cell_number
        self.z_cell_number = z_cell_number
        self.bin_number = bin_number

    def fit(self, X, y=None):

        print("------------------------------------")
        print("IntensityHistogram fit")
        print("cell numbers = {}x{}x{}".format(self.x_cell_number,
                                               self.y_cell_number,
                                               self.z_cell_number))
        print("bin numbers = {}".format(self.bin_number))

        # no internal variable. do nothing

        return self

    def transform(self, X, y=None):
        X = check_array(X)
        n_samples, n_features = np.shape(X)

        print("IntensityHistogram transform")
        print("cell numbers = {}x{}x{}".format(self.x_cell_number,
                                               self.y_cell_number,
                                               self.z_cell_number))
        print("bin numbers = {}".format(self.bin_number))

        print("shape of X before transform : ")
        print(X.shape)

        X = check_array(X)
        n_samples, n_features = np.shape(X)

        X_train_3D = np.reshape(X, (-1, self.IMAGE_DIM_X, self.IMAGE_DIM_Y, self.IMAGE_DIM_Z))

        # cell (contains index of voxels) as bin edge
        x_cell_edges = np.linspace(0, self.IMAGE_DIM_X, self.x_cell_number + 1, dtype=int)
        y_cell_edges = np.linspace(0, self.IMAGE_DIM_Y, self.y_cell_number + 1, dtype=int)
        z_cell_edges = np.linspace(0, self.IMAGE_DIM_Z, self.z_cell_number + 1, dtype=int)

        # histograms
        histogram = np.zeros((n_samples,
                                  self.x_cell_number,
                                  self.y_cell_number,
                                  self.z_cell_number,
                                  self.bin_number))

        for i in range(0, n_samples):
            image_3D = X_train_3D[i, :, :, :]

            for xi in range(0, x_cell_edges.size - 1):
                for yi in range(0, y_cell_edges.size - 1):
                    for zi in range(0, z_cell_edges.size - 1):

                        # image block for histogram
                        image_block = image_3D[x_cell_edges[xi]:x_cell_edges[xi+1],
                                               y_cell_edges[yi]:y_cell_edges[yi+1],
                                               z_cell_edges[zi]:z_cell_edges[zi+1]]

                        # histogram
                        histogram[i, xi, yi, zi, :], bins = \
                            np.histogram(image_block, bins=np.linspace(0, self.BIN_MAX, self.bin_number + 1))

        X_new = np.reshape(histogram, (n_samples, -1))

        print("shape of X after transform : ")
        print(X_new.shape)

        return X_new
