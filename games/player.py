import numpy as np


class Player:
    def __init__(self, algorithm, experts, observer, shape):
        self.algorithm = algorithm
        self.experts = experts
        self.observer = observer
        self.shape = shape
        self.observer.put('weights', self.algorithm.w)

    def advice(self, *args, **kwargs):
        n = len(self.experts)
        _return = np.zeros((n, self.shape))
        for i in range(n):
            _return[i, :] = np.array(self.experts[i].predict(*args, **kwargs))
        self.observer.put('advice', _return)
        return _return

    def predict(self, *args, **kwargs):
        _return = np.dot(np.array(self.algorithm.w), self.advice(*args, **kwargs))
        self.observer.put('predict', _return)
        return _return

    def update(self, *args, **kwargs):
        self.algorithm.update(*args, **kwargs)
        self.observer.put('weights', self.algorithm.w)
