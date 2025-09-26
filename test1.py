import argparse

import numpy as np
from torch.utils.data import DataLoader

from data_provider.data_loader_cluser import Dataset_ETT_minute
from cluster_test import main_clustering_analysis

parser = argparse.ArgumentParser(description='TimesNet')

parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")
parser.add_argument('--cluster', type=int, default=2, help='1,2')

args = parser.parse_args()
if __name__ == '__main__':

    dataset_train = Dataset_ETT_minute(args, 'datasets/ETT-small', flag='train', size=[96, 96, 48],
                                 features='M', data_path='ETTm1.csv')
    # dataset_val = Dataset_ETT_minute(args, 'datasets/ETT-small', flag='val', size=[96, 96, 48],
    #                            features='M', data_path='ETTm1.csv')
    # dataset_test = Dataset_ETT_minute(args, 'datasets/ETT-small', flag='test', size=[96, 96, 48],
    #                             features='M', data_path='ETTm1.csv')
    # dataset = dataset_train+dataset_val+dataset_test
    # print(dataset)
    dataloader = DataLoader(
        dataset_train,
        batch_size=16,
        shuffle=True,
        num_workers=1,
        drop_last=True,
    )
    # for it in dataloader:
    #     print(it[1])
    #     break
    x_all = []
    print('数据集长度:', dataset_train.le())

    for i in range(dataset_train.le()):
        # print(dataset[i][0])
        # x_all.append(dataset.inverse_transform(dataset[i][0]))
        x_all.append(dataset_train.geti(i)[0])
    x_all = np.array(x_all)
    X = x_all
    for i in range(6):
        # 取出第2列（索引为1）
        X_second_column = X[:, :, i]  # 这将得到形状为 (69537, 96) 的数组

        # 如果您需要保持三维结构，可以这样做：
        X_second_column = X[:, :, i:i + 1]  # 这将得到形状为 (69537, 96, 1) 的数组
        print(X_second_column.shape)
        results = main_clustering_analysis(X_second_column, algorithm='hdbscan', use_dtw=False, reduction='pca',
                                           plot_curves=True)

    # import numpy as np
    # from tslearn.clustering import KShape
    # from tslearn.preprocessing import TimeSeriesScalerMeanVariance
    # from sklearn.metrics import silhouette_score
    # import matplotlib.pyplot as plt

    # # 假设您的数据已经加载到一个名为 X 的 numpy 数组中，其形状为 (34464, 96, 7)
    # X = np.load('data.npy')  # 或其他加载方式
    # X = x_all
    # # 1. 数据预处理：标准化（非常重要）
    # # 对每个时间序列进行标准化，使其均值为0，标准差为1
    # scaler = TimeSeriesScalerMeanVariance(mu=0.0, std=1.0)
    # X_scaled = scaler.fit_transform(X)
    # print(X_scaled.shape)
    # #
    # # 2. 确定聚类数量 k（以 k=3 为例，实际需调整）
    # n_clusters = 10
    # # 3. 创建 KShape 模型并进行聚类
    # ks = KShape(n_clusters=n_clusters, n_init=1, random_state=42)
    # labels = ks.fit_predict(X_scaled)  # labels 包含了每个序列的聚类标签
    #
    # # 4. (可选) 评估聚类质量 - 轮廓系数
    # # 注意：计算所有样本的轮廓系数可能非常耗时，可以考虑抽样
    # sample_indices = np.random.choice(len(X_scaled), size=5000, replace=False)
    # sample_data = X_scaled[sample_indices]
    # sample_labels = labels[sample_indices]
    # silhouette_avg = silhouette_score(sample_data.reshape(len(sample_data), -1), sample_labels)
    # print(f"Silhouette Coefficient: {silhouette_avg}")
    #
    # # 5. 可视化某些簇的质心（中心序列）
    # plt.figure(figsize=(12, 8))
    # for i in range(n_clusters):
    #     plt.subplot(n_clusters, 1, i + 1)
    #     # KShape 的 cluster_centers_ 形状是 (n_clusters, n_timesteps, n_features)
    #     # 这里以第一个特征为例进行可视化
    #     plt.plot(ks.cluster_centers_[i, :, 0], label=f'Cluster {i} Center (Feature 0)')
    #     plt.legend()
    # plt.tight_layout()
    # plt.show()
    #
    # # 打印每个簇的大小
    # unique, counts = np.unique(labels, return_counts=True)
    # print(dict(zip(unique, counts)))
    #
    # seq_x, seq_y, seq_x_mark, seq_y_mark = dataset[0]
    # print('处理后的数据:', seq_x[0], '\n反处理后的数据:', dataset.inverse_transform(seq_x)[0])
