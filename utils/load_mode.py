import numpy as np
import os

class LoadMode:
    def __init__(self,root="test_results/mode"):
        self.MODE_DIR = root
        self.data = [(0,2,1),(1,2,1),(2,2,0),(3,2,0),(4,3,2),(5,2,1),(6,2,1)]
        self.csv_path = [f"{i[0]}_{i[1]}_{i[2]}.npy" for i in self.data]

    def load_data(self):
        modes = {i: np.load(os.path.join(self.MODE_DIR, self.csv_path[0])) for i in range(len(self.data))}
        arr = np.array([modes[i] for i in modes])
        return arr, modes.keys()
