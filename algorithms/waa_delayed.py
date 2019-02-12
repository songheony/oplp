from __future__ import division
from .algorithm import Algorithm
import numpy as np
import scipy.special as sc
import sys


class WAADelayed(Algorithm):
    def __init__(self, n):
        self.w = np.ones(n) / n

    '''
    gradient_losses should be n X len(dt)
    '''
    def update(self, gradient_losses, lr):
        np_gradient_losses = np.array(gradient_losses)
        # check the number of element
        assert np_gradient_losses.shape[0] == self.w.shape[0]

        changes = lr * np_gradient_losses.sum(axis=1)
        temp = np.log(self.w + sys.float_info.min) - changes
        self.w = np.exp(temp - sc.logsumexp(temp))
