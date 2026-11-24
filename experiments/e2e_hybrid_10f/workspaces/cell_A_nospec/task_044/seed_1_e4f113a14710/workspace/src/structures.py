class BoundedList:
    def __init__(self, max_size):
        self.max_size = max_size
        self.items = []

	def append(self, item):
		if len(self.data) < self.capacity:
			self.data.append(item)

	def to_list(self):
		return self._data