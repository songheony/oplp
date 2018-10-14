from .algorithm import Algorithm
import numpy as np


class WAADelayed(Algorithm):
    def __init__(self, n):
        super().__init__()
        self.w = np.ones(n) / n


    '''
    gradient_losses should be n X len(dt)
    '''
    def update(self, gradient_losses, lr):
        np_gradient_losses = np.array(gradient_losses)
        assert np_gradient_losses.shape[0] == self.w.shape[0]
        wm = self.w * np.exp(-lr * np_gradient_losses.sum(axis=1))
        self.w = wm / np.sum(wm)
