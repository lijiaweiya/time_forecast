import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, HDBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw


def plot_elbow_curve(k_range, inertias, optimal_k=None):
    """
    绘制肘部法则曲线，用于确定最佳K值
    参数:
        k_range: K值范围，如range(1, 11)
        inertias: 对应K值的簇内距离总和列表
        optimal_k: 最佳K值，用于在图中标记
    """
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertias, 'b-', marker='o', linewidth=2, markersize=6)
    plt.xlabel('K')
    plt.ylabel('Sum of distance')
    plt.title('Elbow Kmeans Method for Optimal K')
    if optimal_k is not None:
        plt.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal K = {optimal_k}')
        plt.legend()
    plt.grid(True)
    plt.show()


def reshape_time_series_data(X):
    """
    将时间序列数据重塑为适合聚类的格式
    参数:
        X: 输入数据，形状为(n_samples, time_steps, features)
    返回:
        reshaped_data: 重塑后的数据，形状为(n_samples, time_steps * features)
    """
    n_samples = X.shape[0]
    return X.reshape(n_samples, -1)


def apply_kmeans(X, n_clusters=None, max_clusters=10, random_state=42, plot_elbow=False):
    """
    使用肘部法则确定最佳K值并进行K-Means聚类
    参数:
        X: 输入数据
        n_clusters: 指定簇数量，如果为None则使用肘部法则确定
        max_clusters: 最大簇数量尝试范围
        random_state: 随机种子
        plot_elbow: 是否绘制肘部曲线
    返回:
        labels: 聚类标签
        inertia: 簇内误差平方和（最终模型的）
        optimal_k: 最佳簇数量
        inertias_list: 所有K值的簇内距离总和列表（如果计算了）
        k_range_list: K值范围列表（如果计算了）
    """
    inertias_list = None
    k_range_list = None

    if n_clusters is None or plot_elbow:
        # 计算不同K值的inertia
        inertias = []
        k_range = range(1, max_clusters + 1)
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)

        inertias_list = inertias
        k_range_list = list(k_range)

        if n_clusters is None:
            # 计算肘部点（曲率最大点）
            derivatives = np.diff(inertias)
            second_derivatives = np.diff(derivatives)
            optimal_k = k_range[np.argmin(second_derivatives) + 1]
        else:
            optimal_k = n_clusters

        if plot_elbow:
            plot_elbow_curve(k_range, inertias, optimal_k)
    else:
        optimal_k = n_clusters

    # 使用最佳K值或指定K值进行聚类
    kmeans = KMeans(n_clusters=optimal_k, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X)
    inertia_val = kmeans.inertia_

    return labels, inertia_val, optimal_k, inertias_list, k_range_list


def apply_dbscan(X, eps=None, min_samples=None, precomputed=False):
    """
    应用DBSCAN聚类算法
    参数:
        X: 输入数据或预计算的距离矩阵
        eps: 邻域半径，如果为None则自动确定
        min_samples: 核心点所需的最小样本数，默认为2*特征维度
        precomputed: 是否使用预计算的距离矩阵
    返回:
        labels: 聚类标签（-1表示噪声点）
    """
    if min_samples is None:
        min_samples = 2 * X.shape[1] if not precomputed else 5

    if eps is None:
        # 使用k距离图估计eps值
        from sklearn.neighbors import NearestNeighbors
        neighbors = NearestNeighbors(n_neighbors=min_samples)
        neighbors_fit = neighbors.fit(X)
        distances, indices = neighbors_fit.kneighbors(X)
        k_distances = np.sort(distances[:, -1])
        eps = np.percentile(k_distances, 90)  # 使用90%分位数作为eps

    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed' if precomputed else 'euclidean')
    labels = dbscan.fit_predict(X)

    return labels


def plot_hdbscan_evaluation(X, min_cluster_sizes):
    """
    绘制HDBSCAN在不同min_cluster_size参数下的评估曲线
    参数:
        X: 输入数据
        min_cluster_sizes: 要测试的min_cluster_size值列表
    """
    n_clusters_list = []
    noise_ratios = []

    for min_size in min_cluster_sizes:
        hdbscan_cluster = HDBSCAN(min_cluster_size=min_size)
        labels = hdbscan_cluster.fit_predict(X)

        # 计算簇数量（排除噪声点）
        n_clusters = len(np.unique(labels[labels != -1]))
        n_clusters_list.append(n_clusters)

        # 计算噪声点比例
        noise_ratio = np.sum(labels == -1) / len(labels)
        noise_ratios.append(noise_ratio)

    # 创建子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 绘制簇数量变化曲线
    ax1.plot(min_cluster_sizes, n_clusters_list, 'b-', marker='o', linewidth=2, markersize=6)
    ax1.set_xlabel('Min Cluster Size')
    ax1.set_ylabel('Number of Clusters')
    ax1.set_title('Number of Clusters vs Min Cluster Size')
    ax1.grid(True)

    # 绘制噪声点比例变化曲线
    ax2.plot(min_cluster_sizes, noise_ratios, 'r-', marker='s', linewidth=2, markersize=6)
    ax2.set_xlabel('Min Cluster Size')
    ax2.set_ylabel('Noise Ratio')
    ax2.set_title('Noise Ratio vs Min Cluster Size')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

def apply_hdbscan(X, min_cluster_size=None, min_samples=None, plot_evaluation=False):
    """
    应用HDBSCAN聚类算法
    参数:
        X: 输入数据
        min_cluster_size: 最小簇大小，默认为总样本数的1%
        min_samples: 核心点所需的最小样本数
        plot_evaluation: 是否绘制评估曲线
    返回:
        labels: 聚类标签（-1表示噪声点）
        probabilities: 簇成员概率
    """
    if min_cluster_size is None:
        min_cluster_size = max(5, X.shape[0] // 100)

    if plot_evaluation:
        # 测试一系列min_cluster_size值
        min_sizes = range(5, min(100, X.shape[0] // 10), 5)
        plot_hdbscan_evaluation(X, min_sizes)

    hdbscan_cluster = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples
    )
    labels = hdbscan_cluster.fit_predict(X)
    probabilities = hdbscan_cluster.probabilities_

    return labels, probabilities


def calculate_dtw_distance_matrix(X, sample_size=None):
    """
    计算DTW距离矩阵
    参数:
        X: 输入数据，形状为(n_samples, time_steps, features)
        sample_size: 采样大小，用于大规模数据（如果为None则使用全部数据）
    返回:
        distance_matrix: DTW距离矩阵
    """
    if sample_size is not None and sample_size < len(X):
        indices = np.random.choice(len(X), sample_size, replace=False)
        X_sampled = X[indices]
    else:
        X_sampled = X
        sample_size = len(X)

    distance_matrix = np.zeros((sample_size, sample_size))

    for i in range(sample_size):
        for j in range(i + 1, sample_size):
            distance, path = fastdtw(X_sampled[i], X_sampled[j], dist=euclidean)
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance

    return distance_matrix


def evaluate_clustering(X, labels, algorithm_name):
    """
    评估聚类结果
    参数:
        X: 输入数据
        labels: 聚类标签
        algorithm_name: 算法名称
    返回:
        metrics: 评估指标字典
    """
    metrics = {}

    # 排除噪声点（对于DBSCAN和HDBSCAN）
    valid_indices = labels != -1
    X_valid = X[valid_indices] if np.any(labels == -1) else X
    labels_valid = labels[valid_indices] if np.any(labels == -1) else labels

    if len(np.unique(labels_valid)) > 1:
        metrics['silhouette_score'] = silhouette_score(X_valid, labels_valid)
        metrics['calinski_harabasz_score'] = calinski_harabasz_score(X_valid, labels_valid)

    metrics['n_clusters'] = len(np.unique(labels_valid))
    metrics['n_noise'] = np.sum(labels == -1) if np.any(labels == -1) else 0
    metrics['clustered_samples'] = np.sum(valid_indices)

    print(f"{algorithm_name}聚类结果:")
    print(f"  簇数量: {metrics['n_clusters']}")
    print(f"  噪声点数量: {metrics['n_noise']}")
    print(f"  已聚类样本数: {metrics['clustered_samples']}")

    if 'silhouette_score' in metrics:
        print(f"  轮廓系数: {metrics['silhouette_score']:.4f}")
        print(f"  Calinski-Harabasz指数: {metrics['calinski_harabasz_score']:.4f}")

    return metrics


def visualize_clusters(X, labels, algorithm_name, reduction='tsne'):
    """
    可视化聚类结果（使用降维技术）
    参数:
        X: 输入数据
        labels: 聚类标签
        algorithm_name: 算法名称
        reduction: 降维方法 ('pca' 或 'tsne')
    """
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    if reduction == 'pca':
        reducer = PCA(n_components=2)
    else:
        reducer = TSNE(n_components=2, perplexity=30, random_state=42)

    X_reduced = reducer.fit_transform(X)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1],
                          c=labels, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter)
    plt.title(f'{algorithm_name}聚类结果可视化 ({reduction.upper()}降维)')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.show()


# 主函数示例
def main_clustering_analysis(X, algorithm='all', use_dtw=True, reduction='tsne', plot_curves=False):
    """
    主函数：执行完整的聚类分析流程
    参数:
        X: 输入数据，形状为(n_samples, time_steps, features)
        algorithm: 要使用的算法 ('kmeans', 'dbscan', 'hdbscan', 或 'all')
        use_dtw: 是否使用DTW距离进行时间序列特异性聚类
        reduction: 降维方法用于可视化
        plot_curves: 是否绘制评估曲线（如肘部曲线）
    返回:
        所有算法的聚类结果和评估指标
    """
    # 重塑数据
    X_reshaped = reshape_time_series_data(X)

    results = {}
    kmeans_labels = None  # 初始化kmeans_labels变量

    # 可选：使用DTW距离矩阵
    if use_dtw:
        print("计算DTW距离矩阵...")
        dtw_matrix = calculate_dtw_distance_matrix(X, sample_size=5000)  # 采样以减少计算量
        X_for_clustering = dtw_matrix
        precomputed = True
    else:
        X_for_clustering = X_reshaped
        precomputed = False

    # 应用选择的聚类算法
    if algorithm in ['kmeans', 'all']:
        print("应用K-Means聚类...")
        kmeans_labels, inertia, optimal_k, inertias_list, k_range_list = apply_kmeans(
            X_for_clustering, plot_elbow=plot_curves
        )
        print("K-MEANS聚类完成,画图中...")
        results['kmeans'] = {
            'labels': kmeans_labels,
            'metrics': evaluate_clustering(X_for_clustering, kmeans_labels, 'K-Means')
        }
        visualize_clusters(X_for_clustering, kmeans_labels, 'K-Means', reduction=reduction)

    if algorithm in ['dbscan', 'all']:
        print("应用DBSCAN聚类...")
        dbscan_labels = apply_dbscan(X_for_clustering, precomputed=precomputed, plot_elbow=plot_curves)
        results['dbscan'] = {
            'labels': dbscan_labels,
            'metrics': evaluate_clustering(X_for_clustering, dbscan_labels, 'DBSCAN')
        }
        visualize_clusters(X_for_clustering, dbscan_labels, 'DBSCAN', reduction=reduction)

    if algorithm in ['hdbscan', 'all']:
        print("应用HDBSCAN聚类...")
        hdbscan_labels, probabilities = apply_hdbscan(
            X_for_clustering,
            plot_evaluation=plot_curves  # 传递plot_curves参数
        )
        print("HDBSCAN聚类完成,画图中...")
        results['hdbscan'] = {
            'labels': hdbscan_labels,
            'probabilities': probabilities,
            'metrics': evaluate_clustering(X_for_clustering, hdbscan_labels, 'HDBSCAN')
        }
        visualize_clusters(X_for_clustering, hdbscan_labels, 'HDBSCAN', reduction=reduction)

    return results  # 只返回一个值


# 使用示例
if __name__ == "__main__":
    # 示例数据加载（替换为您的实际数据）
    X = np.random.randn(100, 96, 7)  # 减少样本量以便快速演示

    # 执行聚类分析，并绘制肘部曲线
    results = main_clustering_analysis(X, algorithm='kmeans', use_dtw=False, plot_curves=True)
