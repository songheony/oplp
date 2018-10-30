from .algorithm import Algorithm
import numpy as np
from __future__ import division


class WAA(Algorithm):
    def __init__(self, n):
        super().__init__()
        self.w = np.ones(n) / n

    def update(self, losses, lr):
        np_losses = np.array(losses)
        changes = lr * np_losses
        changes_max = np.max(changes)
        wm = self.w * np.exp(-(changes - changes_max))
        self.w = wm / np.sum(wm)
