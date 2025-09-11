import argparse

from data_provider.data_loader import Dataset_ETT_minute


parser = argparse.ArgumentParser(description='TimesNet')

parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")

args = parser.parse_args()
if __name__ == '__main__':

    dataset=Dataset_ETT_minute(args, 'datasets/ETT-small', flag='train', size=[96,96,1],
                 features='M', data_path='ETTm1.csv')
    seq_x, seq_y, seq_x_mark, seq_y_mark=dataset[0]
    print(seq_x[0],dataset.inverse_transform(seq_x)[0])
