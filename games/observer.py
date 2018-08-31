class Observer:
    def __init__(self):
        self.dic = dict()

    def put(self, key, value):
        self.dic.get(key, []).append(value)

    def get(self, key):
        return self.dic.get(key, None)
