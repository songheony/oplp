from .algorithm import Algorithm
import numpy as np


class FSA(Algorithm):
    def __init__(self, n, lr, a):
        super().__init__()
        self.w = np.ones(n) / n
        self.lr = lr
        self.a = a

    def update(self, losses):
        super().update()
        np_losses = np.array(losses)
        wm = self.w * np.exp(-self.lr * np_losses)
        pool = np.sum(wm) * self.a
        wm = (1 - self.a) * wm + (pool - self.a * wm) / (len(wm) - 1)
        self.w = wm / np.sum(wm)
