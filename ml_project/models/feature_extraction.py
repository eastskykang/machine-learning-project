from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from sklearn.utils.validation import check_is_fitted, check_array


class IntensityHistogram(BaseEstimator, TransformerMixin):
    """Feature from intensity histogram of 3D images"""

    # divide 3d image into cells and make histogram per cell


    def __init__(self, x_cell_number=7, y_cell_number=7, z_cell_number=7, bin_number=10):
        # image dimension
        self.IMAGE_DIM_X = 176
        self.IMAGE_DIM_Y = 208
        self.IMAGE_DIM_Z = 176

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

        X = check_array(X)
        n_samples, n_features = np.shape(X)

        X_train_3D = np.reshape(X, (-1, self.IMAGE_DIM_X, self.IMAGE_DIM_Y, self.IMAGE_DIM_Z))

        # cell (contains index of voxels)
        x_bins = np.linspace(0, self.IMAGE_DIM_X - 1, self.x_cell_number + 1, dtype=int)
        y_bins = np.linspace(0, self.IMAGE_DIM_Y - 1, self.y_cell_number + 1, dtype=int)
        z_bins = np.linspace(0, self.IMAGE_DIM_Z - 1, self.z_cell_number + 1, dtype=int)

        # histograms
        self.histogram = np.zeros((n_samples,
                                  self.x_cell_number,
                                  self.y_cell_number,
                                  self.z_cell_number,
                                  self.bin_number))

        for i in range(0, n_samples):
            image_3D = X_train_3D[i, :, :, :]

            for x in range(0, x_bins.size):
                for y in range(0, y_bins.size):
                    for z in range(0, z_bins.size):
                        self.histogram[i, x, y, z] = image_3D[x_bins[x], y_bins[y], z_bins[z]]

        return self

    def transform(self, X, y=None):
        check_is_fitted(self, ["histogram"])

        print("------------------------------------")
        print("IntensityHistogram transform")
        print("cell numbers = {}x{}x{}".format(self.x_cell_number,
                                               self.y_cell_number,
                                               self.z_cell_number))
        print("bin numbers = {}".format(self.bin_number))

        X_new = self.histogram
        print("shape of histogram: ")
        print(X_new.shape)

        return X_new
