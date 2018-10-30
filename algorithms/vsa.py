from .algorithm import Algorithm
import numpy as np
from __future__ import division


class VSA(Algorithm):
    def __init__(self, n):
        super().__init__()
        self.w = np.ones(n) / n

    def update(self, losses, lr, a):
        np_losses = np.array(losses)
        wm = self.w * np.exp(-lr * np_losses)
        pool = np.sum(wm * (1 - np.power(1 - a, np_losses)))
        wm = np.power(1 - a, np_losses) * wm + \
            (pool - (1 - np.power(1 - a, np_losses)) * wm) / (len(np_losses) - 1)
        self.w = wm / np.sum(wm)
