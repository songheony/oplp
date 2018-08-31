from .algorithm import Algorithm
import numpy as np


class VSA(Algorithm):
    def __init__(self, n, lr, a):
        super().__init__()
        self.w = np.ones(n) / n
        self.lr = lr
        self.a = a

    def update(self, losses):
        super().update()
        np_losses = np.array(losses)
        wm = self.w * np.exp(-self.lr * np_losses)
        pool = np.sum(wm * (1 - np.power(1 - self.a, np_losses)))
        wm = np.power(1 - self.a, np_losses) * wm + (pool - (1 - np.power(1 - self.a, np_losses)) * wm) / (len(np_losses) - 1)
        self.w = wm / np.sum(wm)
