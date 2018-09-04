from .algorithm import Algorithm
import numpy as np


class WAA(Algorithm):
    def __init__(self, n):
        super().__init__()
        self.w = np.ones(n) / n

    def update(self, losses, lr):
        np_losses = np.array(losses)
        wm = self.w * np.exp(-lr * np_losses)
        self.w = wm / np.sum(wm)
