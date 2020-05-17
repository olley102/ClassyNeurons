import numpy as np
from miscellaneous import truncnorm_standard

def uniform_random(vertical, horizontal):
    return np.random.rand(vertical, horizontal)

def zeros(vertical, horizontal):
    return np.zeros((vertical, horizontal))

def saved_weights(weights):
    return lambda *args : weights

def truncated_normal(vertical, horizontal):
    rad = 1 / np.sqrt(vertical)
    dist = truncnorm_standard(mean=0, sd=1, low=-rad, upp=rad)
    return dist.rvs((vertical, horizontal))
