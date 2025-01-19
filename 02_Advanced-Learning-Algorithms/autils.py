import numpy as np
import os

def load_data():
    X = np.load(f"{os.path.dirname(os.path.realpath(__file__))}/data/X.npy")
    y = np.load(f"{os.path.dirname(os.path.realpath(__file__))}/data/y.npy")
    X = X[0:1000]
    y = y[0:1000]
    return X, y

def load_weights():
    w1 = np.load(f"{os.path.dirname(os.path.realpath(__file__))}/data/w1.npy")
    b1 = np.load(f"{os.path.dirname(os.path.realpath(__file__))}/data/b1.npy")
    w2 = np.load(f"{os.path.dirname(os.path.realpath(__file__))}/data/w2.npy")
    b2 = np.load(f"{os.path.dirname(os.path.realpath(__file__))}/data/b2.npy")
    return w1, b1, w2, b2

def sigmoid(x):
    return 1. / (1. + np.exp(-x))
