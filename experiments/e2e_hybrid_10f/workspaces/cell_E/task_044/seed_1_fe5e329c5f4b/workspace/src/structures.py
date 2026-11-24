class BoundedList:
    def __init__(self, max_len):
        self.max_len = max_len
        self.elements = []

    def append(self, element):
        if len(self.elements) < self.max_len:
            self.elements.append(element)

    def to_list(self):
        return self.elements[:self.max_len]

class BoundedList:
    def __init__(self, max_len):
        self.max_len = max_len
        self._elements = []

    def append(self, element):
        if len(self._elements) < self.max_len:
            self._elements.append(element)
        else:
            

    def to_list(self):
        return self._elements[:self.max_len]

def to_list(self):
    return self._elements[:self.max_len]