import argparse

from torch.utils.data import DataLoader

from data_provider.data_loader import Dataset_ETT_minute



parser = argparse.ArgumentParser(description='TimesNet')

parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")

args = parser.parse_args()
if __name__ == '__main__':

    dataset=Dataset_ETT_minute(args, 'datasets/ETT-small', flag='train', size=[96,96,1],
                 features='M', data_path='ETTm1.csv')
    dataloader=DataLoader(
            dataset,
            batch_size=16,
            shuffle=True,
            num_workers=1,
            drop_last=True,

        )
    # seq_x, seq_y, seq_x_mark, seq_y_mark=dataset[0]
    # print('处理后的数据:',seq_x[0],'\n反处理后的数据:',dataset.inverse_transform(seq_x)[0])
