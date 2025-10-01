import argparse

import numpy as np


from data_provider.data_loader import Dataset_ETT_minute


parser = argparse.ArgumentParser(description='TimesNet')

parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")
parser.add_argument('--cluster', type=int, default=2, help='1,2')
from utils.load_mode import LoadMode
args = parser.parse_args()
if __name__ == '__main__':
    load_mode = LoadMode(root='../test_results/mode')
    arr, keys = load_mode.load_data()
    arr = arr.squeeze()  # 变成 (7, 96)
    arr = arr.transpose(1, 0)  # 变成 (96, 7)
    dataset_train = Dataset_ETT_minute(args, '../datasets/ETT-small', flag='train', size=[96, 96, 48],
                                 features='M', data_path='ETTm1.csv')
    raw=dataset_train.inverse_transform(arr)
    print(raw)
    print(raw.shape)