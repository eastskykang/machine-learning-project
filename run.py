from config import Config
import nibabel as nib
import numpy as np
import hashlib

X = np.load("X_test.npy")
print(hashlib.md5(X).hexdigest())

X[0,0]=1
print(hashlib.md5(X).hexdigest())