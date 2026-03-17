class ReplayBuffer:
    def __init__(self, max_size=100000):
        self.arr = [None] * max_size
        self.max_size = max_size
        self.to_insert_next = 0
        self._count = 0

    def __len__(self):
        return self._count

    def insert(self, el):
        if self.arr[self.to_insert_next] is None:
            self._count += 1
        self.arr[self.to_insert_next] = el
        self.to_insert_next += 1
        if self.to_insert_next == self.max_size:
            self.to_insert_next = 0

    def insert_batch(self, items):
        n = len(items)
        if n >= self.max_size:
            self.to_insert_next = 0
            self.arr = list(items[-self.max_size:])
            self._count = self.max_size
        elif n + self.to_insert_next <= self.max_size:
            # Count how many None slots we're overwriting
            old_nones = sum(1 for i in range(self.to_insert_next, self.to_insert_next + n)
                           if self.arr[i] is None)
            self._count += old_nones
            self.arr[self.to_insert_next:self.to_insert_next + n] = items[:]
            self.to_insert_next += n
            if self.to_insert_next == self.max_size:
                self.to_insert_next = 0
        else:
            first_part = self.max_size - self.to_insert_next
            second_part = n - first_part
            # Count None slots in both regions
            old_nones = sum(1 for i in range(self.to_insert_next, self.max_size)
                           if self.arr[i] is None)
            old_nones += sum(1 for i in range(second_part)
                            if self.arr[i] is None)
            self._count += old_nones
            self.arr[self.to_insert_next:self.max_size] = items[:first_part]
            self.arr[:second_part] = items[first_part:]
            self.to_insert_next = second_part
