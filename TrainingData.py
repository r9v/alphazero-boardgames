
class TrainingData():
    def __init__(self, maxSize):
        self.arr = [None] * maxSize
        self.maxSize = maxSize
        self.toInsertNext = 0

    def insert(self, el):
        self.arr[toInsertNext] = el
        self.toInsertNext += 1
        if self.toInsertNext == self.maxSize:
            self.toInsertNext = 0

    def insertArr(self, ar):
        arLen = len(ar)
        if(arLen >= self.maxSize):
            print('TrainingData.insertArr called with too long of an array')
            self.toInsertNext = 0
            self.arr = ar[-self.maxSize:]
        elif(arLen+self.toInsertNext <= self.maxSize):
            self.arr[self.toInsertNext:arLen+self.toInsertNext] = ar[:]
            self.toInsertNext = arLen+self.toInsertNext
            if self.toInsertNext == self.maxSize:
                self.toInsertNext = 0
        else:
            firstPartLen = self.maxSize - self.toInsertNext
            self.arr[self.toInsertNext:self.maxSize] = ar[:firstPartLen]
            secondPartLen = arLen - firstPartLen
            self.arr[:secondPartLen] = ar[firstPartLen:]
            self.toInsertNext = secondPartLen


"""
data = TrainingData(5)
data.arr = [0, 1, 2, 3, 4]
data.toInsertNext = 3

data.insertArr([22, 33, 55])
print(data.arr)
print(data.toInsertNext)

data.insertArr([22, 33, 55])
print(data.arr)
print(data.toInsertNext)

data.insertArr([22])
print(data.arr)
print(data.toInsertNext)


data.insertArr([22, 33, 55])
print(data.arr)
print(data.toInsertNext)


data.insertArr([22, 33, 55, 66])
print(data.arr)
print(data.toInsertNext)
"""
