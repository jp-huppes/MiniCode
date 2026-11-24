class KeyValueStore:
    def __init__(self):
        self.store = {}

    def set(self, k, v):
        self.store[k] = v

    def get(self, k, default=None):
        return self.store.get(k, default)

    def delete(self, k):
        if k in self.store:
            del self.store[k]

    def keys(self):
        return sorted(self.store.keys())

class KeyValueStore:
    def __init__(self):
        self.store = {}

    def set(self, k, v):
        self.store[k] = v

    def get(self, k, default=None):
        return self.store.get(k, default)

    def delete(self, k):
        if k in self.store:
            del self.store[k]

    def keys(self):
        return sorted(self.store.keys())