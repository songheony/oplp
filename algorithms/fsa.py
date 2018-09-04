from .algorithm import Algorithm
import numpy as np


class FSA(Algorithm):
    def __init__(self, n):
        super().__init__()
        self.w = np.ones(n) / n

    def update(self, losses, lr, a):
        np_losses = np.array(losses)
        wm = self.w * np.exp(-lr * np_losses)
        pool = np.sum(wm) * a
        wm = (1 - a) * wm + (pool - a * wm) / (len(wm) - 1)
        self.w = wm / np.sum(wm)
