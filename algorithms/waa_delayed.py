from .algorithm import Algorithm
import numpy as np
from __future__ import division


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
        changes_max = np.max(changes)
        wm = self.w * np.exp(-(changes - changes_max))
        self.w = wm / np.sum(wm)               
