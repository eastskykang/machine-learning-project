# from sklearn.base import BaseEstimator, TransformerMixin
# from scipy import ndimage
# import numpy as np
#
# class ImageDownSampling(BaseEstimator, TransformerMixin):
#     """Down sample image"""
#
#     def __init__(self):
#
#
#     def fit(self, X, y=None):
#
#
# def block_mean(ar, fact):
#     assert isinstance(fact, int), type(fact)
#     sx, sy = ar.shape
#     X, Y = np.ogrid[0:sx, 0:sy]
#     regions = sy/fact * (X/fact) + Y/fact
#     res = ndimage.mean(ar, labels=regions, index=np.arange(regions.max() + 1))
#     res.shape = (sx/fact, sy/fact)
#     return res
