class tt:
    def __init__(self):
        self.data = [1, 2, 3, 4]
        self.index = 0

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)

    def __next__(self):
        if self.index < len(self.data):
            result = self.data[self.index]
            self.index += 1
            return result
        raise StopIteration


if __name__ == '__main__':
    te = tt()
    print(te[0],len(te),[i for i in te])
