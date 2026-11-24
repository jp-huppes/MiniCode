from collections import OrderedDict

class KeyValueStore:
    def __init__(self):
        self.store = OrderedDict()

    def set(self, key, value):
        self.store[key] = value

    def get(self, key):
        return self.store.get(key)

    def delete(self, key):
        if key in self.store:
            del self.store[key]

    def keys(self):
        return list(self.store.keys())