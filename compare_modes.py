import numpy as np
import os

data=[(0,2,1),(1,2,1),(2,2,0),(3,2,0),(4,3,2),(5,2,1),(6,2,1)]

# 文件对列表
file_pairs = [(f"{i[0]}_{i[1]}_{i[2]}.npy", f"lie{i[0]}_class{i[1]}_best{i[2]}.npy") for i in data]

MODE_DIR = os.path.join("test_results", "mode")
print('文件路径：',MODE_DIR)

for f1, f2 in file_pairs:
    path1 = os.path.join(MODE_DIR, f1)
    path2 = os.path.join(MODE_DIR, f2)
    if not (os.path.exists(path1) and os.path.exists(path2)):
        print(f"文件缺失: {f1} 或 {f2}")
        continue
    arr1 = np.load(path1)
    arr2 = np.load(path2)
    if arr1.shape != arr2.shape:
        print(f"{f1} 和 {f2} shape 不一致: {arr1.shape} vs {arr2.shape}")
    elif np.allclose(arr1, arr2):
        print(f"{f1} 和 {f2} 内容完全一致！")
    else:
        print(f"{f1} 和 {f2} shape一致但内容不同！")

