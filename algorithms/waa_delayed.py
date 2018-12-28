from __future__ import division
from .algorithm import Algorithm
import numpy as np
import warnings
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
        assert np_gradient_losses.shape[0] == self.w.shape[0]
        changes = lr * np_gradient_losses.sum(axis=1)
        # changes = changes - np.max(changes)
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            temp = np.log(self.w + sys.float_info.min) - changes
            self.w = np.exp(temp - sc.logsumexp(temp))      
