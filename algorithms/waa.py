from .algorithm import Algorithm
import numpy as np


class WAA(Algorithm):
    def __init__(self, n, lr):
        super().__init__()
        self.w = np.ones(n) / n
        self.lr = lr

    def update(self, losses):
        np_losses = np.array(losses)
        wm = self.w * np.exp(-self.lr * np_losses)
        self.w = wm / np.sum(wm)
