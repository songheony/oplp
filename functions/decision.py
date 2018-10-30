import numpy as np


def roulette(p):
    s = 0
    r = np.random.rand(1) * np.sum(p)
    one_hot_vector = np.zeros_like(p)
    for z in range(len(p)):
        s += p[z]
        if r < s:
            one_hot_vector[z] = 1
            break
    return one_hot_vector
