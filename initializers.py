import numpy as np

def uniform_random(vertical, horizontal):
    return np.random.rand(vertical, horizontal)

def zeros(vertical, horizontal):
    return np.zeros((vertical, horizontal))

def saved_weights(weights):
    return lambda v, h : weights
