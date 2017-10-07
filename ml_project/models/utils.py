from sklearn.base import BaseEstimator, TransformerMixin
from scipy.ndimage import zoom
import numpy as np
from sklearn.utils.validation import check_array


class Constants:
    """Class contains constant values"""
    IMAGE_DIM_X = 176
    IMAGE_DIM_Y = 208
    IMAGE_DIM_Z = 176
    IMAGE_VALUE_MAX = 4500


class ImageDownSampling(BaseEstimator, TransformerMixin):
    """Down sample image"""

    def __init__(self, scale=0.5):
        self.scale = scale

    def fit(self, X, y=None):

        print("------------------------------------")
        print("ImageDownSampling fit")
        print("resize scale = {}".format(self.scale))

        # no internal variables
        X = check_array(X)
        return self

    def transform(self, X, y=None):

        X = check_array(X)
        n_samples, n_features = np.shape(X)

        print("------------------------------------")
        print("ImageDownSampling fit")
        print("resize scale = {}".format(self.scale))

        X_3D = np.reshape(X, (-1,
                              Constants.IMAGE_DIM_X,
                              Constants.IMAGE_DIM_Y,
                              Constants.IMAGE_DIM_Z))

        print("shape of 3D image before down sampling: ")
        print(X_3D.shape)

        # resize (interpolation)
        for i in range(0, n_samples):
            rescaled_image = zoom(X_3D[i, :, :, :], self.scale)

            if i == 0:
                X_new = rescaled_image.flatten()
            else:
                X_new = np.row_stack((X_new, rescaled_image.flatten()))

        print("shape of 3D image after down sampling: ")
        print(X_new.shape)

        return X_new