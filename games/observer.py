import pandas as pd


class Observer:
    def __init__(self):
        self.dic = dict()

    def put(self, key, value):
        self.dic[key] = self.dic.get(key, []) + [value]

    def get(self, key):
        return self.dic.get(key, None)

    def save(self, filepath):
        df = pd.DataFrame.from_dict(self.dic, orient='index')
        df.to_csv(filepath)
