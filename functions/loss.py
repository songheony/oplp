import numpy as np


def square(x, y):
    return (y - x) ** 2


def logarithmic(x, y):
    return (1 - y) * np.log((1 - y) / (1 - x)) + y * np.log(y / x)


def hellinger(x, y):
    return ((np.sqrt(1 - y) - np.sqrt(1 - x)) ** 2 + (np.sqrt(y) - np.sqrt(x)) ** 2) / 2


def absolute(x, y):
    return np.abs(y - x)


def dot(x, y):
    return np.dot(x, y)


def mix(x, y):
    return -np.log(np.sum(x * np.exp(-y)))
