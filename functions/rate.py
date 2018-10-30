import numpy as np
from __future__ import division


def quickly_decreasing(t, n):
    if t == 0:
        return (n - 1) / n
    else:
        return 1 / ((t + 1) * np.log(t + 1))


def slowly_decreasing(t, n):
    if t == 0:
        return (n - 1) / n
    else:
        return 1 / (t + 1)


def sum_convergent(t, n):
    if t == 0:
        return (n - 1) / n
    else:
        return 1 / (t ** 2)
