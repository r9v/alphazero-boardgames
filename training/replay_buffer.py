class ReplayBuffer:
    def __init__(self, max_size=100000):
        self.arr = [None] * max_size
        self.max_size = max_size
        self.to_insert_next = 0

    def __len__(self):
        return sum(1 for s in self.arr if s is not None)

    def insert(self, el):
        self.arr[self.to_insert_next] = el
        self.to_insert_next += 1
        if self.to_insert_next == self.max_size:
            self.to_insert_next = 0

    def insert_batch(self, items):
        n = len(items)
        if n >= self.max_size:
            self.to_insert_next = 0
            self.arr = list(items[-self.max_size:])
        elif n + self.to_insert_next <= self.max_size:
            self.arr[self.to_insert_next:self.to_insert_next + n] = items[:]
            self.to_insert_next += n
            if self.to_insert_next == self.max_size:
                self.to_insert_next = 0
        else:
            first_part = self.max_size - self.to_insert_next
            self.arr[self.to_insert_next:self.max_size] = items[:first_part]
            second_part = n - first_part
            self.arr[:second_part] = items[first_part:]
            self.to_insert_next = second_part
