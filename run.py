from config import Config
import nibabel as nib
import numpy as np
import hashlib
import os
import models
import argparse

#y = np.loadtxt("data/y_1.csv")

parser = argparse.ArgumentParser(description="Transformer")
parser.add_argument("config",
                    help="config for transformer")
parser.add_argument("input",
                    help="input data set")

args = parser.parse_args()

Config.parse_config_file(args.config)

X = np.load(args.input)

transformer = Config.config["class"](**Config.config["params"])

X_new = transformer.fit_transform(X)

np.save("/tmp/smt/"+Config.config["name"]+".npy", X_new)