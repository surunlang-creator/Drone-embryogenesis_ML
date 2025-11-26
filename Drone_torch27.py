#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, silhouette_score
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, \
    ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch, Rectangle
from matplotlib.collections import LineCollection
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import networkx as nx
import argparse
import warnings
import os
import sys
import shutil
import json
import datetime
from collections import OrderedDict, defaultdict
from itertools import combinations
from pathlib import Path
import joblib
import pickle
import time
import json
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    """自定义JSON编码器，处理NumPy类型"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


warnings.filterwarnings('ignore')

# 设置matplotlib字体和样式
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid')

# 设置随机种子
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)

"""
==============================================
Enhanced Gene Expression Analysis System V3.0
With WGCNA, KNN Clustering, and Advanced DL/ML
==============================================
Author: Enhanced Analysis Team
Date: 2024
Version: 3.0
"""

# ============= 颜色方案定义 =============
# Nature color palette
NATURE_COLORS = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', '#8491B4', '#91D1C2', '#DC0000']
# Science color palette
SCIENCE_COLORS = ['#3B4992', '#EE0000', '#008B45', '#631879', '#008280', '#BB0021', '#5F559B', '#A20056']
# Cell color palette
CELL_COLORS = ['#BC3C29', '#0072B5', '#E18727', '#20854E', '#7876B1', '#6F99AD', '#FFDC91', '#EE4C97']


# ============= WGCNA核心函数 =============
class WGCNA:
    """Weighted Gene Co-expression Network Analysis"""

    def __init__(self, verbose=True):
        self.verbose = verbose
        self.adjacency_matrix = None
        self.TOM = None
        self.gene_tree = None
        self.module_colors = None
        self.module_eigengenes = None
        self.soft_power = None

    def pickSoftThreshold(self, datExpr, powerVector=None, RsquaredCut=0.85, networkType="unsigned"):
        """选择软阈值"""
        if powerVector is None:
            powerVector = np.arange(1, 21)

        if self.verbose:
            print("Calculating soft thresholding power...")

        nSamples = datExpr.shape[0]
        nGenes = datExpr.shape[1]

        sft_results = []

        for power in powerVector:
            # 计算相关系数矩阵
            cor_matrix = np.abs(np.corrcoef(datExpr.T))

            # 计算邻接矩阵
            if networkType == "unsigned":
                adjacency = cor_matrix ** power
            elif networkType == "signed":
                adjacency = ((1 + cor_matrix) / 2) ** power
            else:  # signed hybrid
                adjacency = cor_matrix.copy()
                adjacency[cor_matrix >= 0] = cor_matrix[cor_matrix >= 0] ** power
                adjacency[cor_matrix < 0] = 0

            # 计算连接度
            k = np.sum(adjacency, axis=0) - 1  # 减去自连接

            # 计算scale-free拓扑拟合
            if np.sum(k > 0) > 1:
                # 创建度分布直方图
                hist_data, bin_edges = np.histogram(k[k > 0], bins=10)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

                # 对数变换
                log_k = np.log10(bin_centers[hist_data > 0])
                log_freq = np.log10(hist_data[hist_data > 0])

                # 线性回归
                if len(log_k) > 1:
                    slope, intercept = np.polyfit(log_k, log_freq, 1)
                    predicted = slope * log_k + intercept
                    ss_res = np.sum((log_freq - predicted) ** 2)
                    ss_tot = np.sum((log_freq - np.mean(log_freq)) ** 2)
                    r_squared = 1 - (ss_res / (ss_tot + 1e-10))
                else:
                    slope = 0
                    r_squared = 0
            else:
                slope = 0
                r_squared = 0

            sft_results.append({
                'Power': power,
                'SFT.R.sq': r_squared,
                'slope': slope,
                'mean.k': np.mean(k),
                'median.k': np.median(k),
                'max.k': np.max(k)
            })

        sft_df = pd.DataFrame(sft_results)

        # 选择最佳软阈值
        best_power = None
        for power in powerVector:
            if sft_df[sft_df['Power'] == power]['SFT.R.sq'].values[0] > RsquaredCut:
                best_power = power
                break

        if best_power is None:
            best_power = 6  # 默认值
            if self.verbose:
                print(f"Warning: No power reached R^2 cut of {RsquaredCut}. Using default power = 6")

        self.soft_power = best_power

        return {
            'powerEstimate': best_power,
            'fitIndices': sft_df
        }

    def adjacency(self, datExpr, power=6, networkType="unsigned"):
        """计算邻接矩阵"""
        cor_matrix = np.abs(np.corrcoef(datExpr.T))

        if networkType == "unsigned":
            adjacency = cor_matrix ** power
        elif networkType == "signed":
            adjacency = ((1 + cor_matrix) / 2) ** power
        else:  # signed hybrid
            adjacency = cor_matrix.copy()
            adjacency[cor_matrix >= 0] = cor_matrix[cor_matrix >= 0] ** power
            adjacency[cor_matrix < 0] = 0

        self.adjacency_matrix = adjacency
        return adjacency

    def TOMsimilarity(self, adjacency):
        """计算拓扑重叠矩阵(TOM)"""
        if self.verbose:
            print("Calculating Topological Overlap Matrix (TOM)...")

        n = adjacency.shape[0]
        k = np.sum(adjacency, axis=0)

        # 初始化TOM矩阵
        tom = np.zeros((n, n))

        # 计算TOM
        for i in range(n):
            if i % 100 == 0 and self.verbose:
                print(f"  Processing gene {i}/{n}...")

            for j in range(i + 1, n):
                # 计算节点i和j的共同邻居连接强度
                numerator = np.sum(adjacency[i, :] * adjacency[j, :]) + adjacency[i, j]
                denominator = min(k[i], k[j]) + 1 - adjacency[i, j]

                if denominator > 0:
                    tom[i, j] = numerator / denominator
                    tom[j, i] = tom[i, j]

        # 对角线设为1
        np.fill_diagonal(tom, 1)

        self.TOM = tom
        return tom

    def hierarchicalClustering(self, dissimilarity):
        """层次聚类"""
        if self.verbose:
            print("Performing hierarchical clustering...")

        # 确保是距离矩阵
        condensed_dist = squareform(dissimilarity, checks=False)

        # 执行层次聚类
        self.gene_tree = linkage(condensed_dist, method='average')

        return self.gene_tree

    def cutreeDynamic(self, dendro, distM, minClusterSize=30, cutHeight=0.99, deepSplit=2):
        """动态树切割"""
        if self.verbose:
            print(f"Dynamic tree cutting (minClusterSize={minClusterSize})...")

        # 获取树的高度
        max_height = np.max(dendro[:, 2])
        cut_height = cutHeight * max_height

        # 初始切割
        initial_clusters = fcluster(dendro, cut_height, criterion='distance')

        # 根据deepSplit调整
        if deepSplit == 0:
            # 保守切割
            final_clusters = initial_clusters
        elif deepSplit == 1:
            # 中等切割
            final_clusters = self._refine_clusters(initial_clusters, distM, minClusterSize, 0.1)
        elif deepSplit == 2:
            # 积极切割
            final_clusters = self._refine_clusters(initial_clusters, distM, minClusterSize, 0.05)
        else:
            # 非常积极的切割
            final_clusters = self._refine_clusters(initial_clusters, distM, minClusterSize, 0.02)

        # 过滤小簇
        cluster_sizes = pd.Series(final_clusters).value_counts()
        small_clusters = cluster_sizes[cluster_sizes < minClusterSize].index

        # 将小簇标记为0（噪声）
        for small_cluster in small_clusters:
            final_clusters[final_clusters == small_cluster] = 0

        # 重新编号簇
        unique_clusters = np.unique(final_clusters[final_clusters != 0])
        cluster_map = {old: new for new, old in enumerate(unique_clusters, 1)}
        cluster_map[0] = 0

        final_clusters = np.array([cluster_map[c] for c in final_clusters])

        return final_clusters

    def _refine_clusters(self, clusters, distM, minSize, threshold):
        """细化聚类结果"""
        refined = clusters.copy()
        unique_clusters = np.unique(clusters)

        for cluster in unique_clusters:
            if cluster == 0:
                continue

            mask = clusters == cluster
            if np.sum(mask) < minSize * 2:
                continue

            # 检查簇内距离
            cluster_dist = distM[mask][:, mask]
            mean_dist = np.mean(cluster_dist)

            # 如果簇内距离太大，进一步分割
            if mean_dist > threshold:
                sub_clusters = fcluster(linkage(squareform(cluster_dist)), 2, criterion='maxclust')
                # 重新编号子簇
                max_cluster = np.max(refined)
                refined[mask] = sub_clusters + max_cluster

        return refined

    def moduleEigengenes(self, datExpr, colors):
        """计算模块特征基因"""
        if self.verbose:
            print("Calculating module eigengenes...")

        unique_colors = np.unique(colors)
        MEs = pd.DataFrame(index=range(datExpr.shape[0]))

        for color in unique_colors:
            if color == 0 or color == 'grey':  # 跳过未分配的基因
                continue

            # 获取属于该模块的基因
            if isinstance(colors[0], str):
                module_genes = datExpr[:, colors == color]
            else:
                module_genes = datExpr[:, colors == color]

            if module_genes.shape[1] > 0:
                # 使用PCA提取第一主成分
                pca = PCA(n_components=1)
                ME = pca.fit_transform(module_genes)

                # 确保特征基因与平均表达正相关
                if np.corrcoef(ME.flatten(), np.mean(module_genes, axis=1))[0, 1] < 0:
                    ME = -ME

                MEs[f'ME{color}'] = ME.flatten()

        self.module_eigengenes = MEs
        return MEs

    def blockwiseModules(self, datExpr, power=17, minModuleSize=30,
                         reassignThreshold=0, mergeCutHeight=0.10,
                         networkType="unsigned", deepSplit=4, maxBlockSize=5000):
        """构建加权基因共表达网络模块"""
        if self.verbose:
            print("\n" + "=" * 60)
            print("Building Weighted Gene Co-expression Network")
            print("=" * 60)

        nGenes = datExpr.shape[1]

        # 计算邻接矩阵
        if self.verbose:
            print(f"Step 1: Calculating adjacency matrix (power={power})...")
        adjacency = self.adjacency(datExpr, power=power, networkType=networkType)

        # 计算TOM
        if self.verbose:
            print("Step 2: Calculating TOM similarity...")
        TOM = self.TOMsimilarity(adjacency)

        # 计算距离矩阵
        dissTOM = 1 - TOM

        # 层次聚类
        if self.verbose:
            print("Step 3: Hierarchical clustering...")
        geneTree = self.hierarchicalClustering(dissTOM)

        # 动态树切割
        if self.verbose:
            print("Step 4: Dynamic tree cutting...")
        dynamicMods = self.cutreeDynamic(geneTree, dissTOM,
                                         minClusterSize=minModuleSize,
                                         deepSplit=deepSplit)

        # 分配颜色
        if self.verbose:
            print("Step 5: Assigning module colors...")
        moduleColors = self._assignColors(dynamicMods)

        # 计算模块特征基因
        if self.verbose:
            print("Step 6: Calculating module eigengenes...")
        MEs = self.moduleEigengenes(datExpr, moduleColors)

        # 合并相似模块
        if mergeCutHeight < 1:
            if self.verbose:
                print(f"Step 7: Merging similar modules (cutHeight={mergeCutHeight})...")
            moduleColors, MEs = self._mergeCloseModules(datExpr, moduleColors,
                                                        MEs, mergeCutHeight)

        self.module_colors = moduleColors

        # 统计信息
        if self.verbose:
            unique_colors = np.unique(moduleColors)
            print(f"\nModule statistics:")
            for color in unique_colors:
                if color != 'grey':
                    size = np.sum(moduleColors == color)
                    print(f"  {color}: {size} genes")

        return {
            'colors': moduleColors,
            'MEs': MEs,
            'geneTree': geneTree,
            'TOM': TOM
        }

    def _assignColors(self, labels):
        """为模块分配颜色"""
        standardColors = ['turquoise', 'blue', 'brown', 'yellow', 'green', 'red',
                          'black', 'pink', 'magenta', 'purple', 'greenyellow', 'tan',
                          'salmon', 'cyan', 'midnightblue', 'lightcyan', 'grey60',
                          'lightgreen', 'lightyellow', 'royalblue', 'darkred', 'darkgreen',
                          'darkturquoise', 'darkgrey', 'orange', 'darkorange', 'white']

        unique_labels = np.unique(labels)
        color_assignment = {}

        color_idx = 0
        for label in unique_labels:
            if label == 0:
                color_assignment[label] = 'grey'
            else:
                color_assignment[label] = standardColors[color_idx % len(standardColors)]
                color_idx += 1

        colors = np.array([color_assignment[label] for label in labels])

        return colors

    def _mergeCloseModules(self, datExpr, colors, MEs, cutHeight):
        """合并相似的模块"""
        # 计算模块特征基因之间的相关性
        ME_cor = np.corrcoef(MEs.T)
        ME_dist = 1 - ME_cor

        # 层次聚类
        ME_tree = linkage(squareform(ME_dist), method='average')

        # 切割树
        merge_clusters = fcluster(ME_tree, cutHeight, criterion='distance')

        # 合并模块
        unique_colors = list(MEs.columns)
        new_colors = colors.copy()

        for cluster in np.unique(merge_clusters):
            modules_to_merge = [unique_colors[i] for i in range(len(unique_colors))
                                if merge_clusters[i] == cluster]

            if len(modules_to_merge) > 1:
                # 保留第一个模块的颜色
                main_color = modules_to_merge[0].replace('ME', '')

                for module in modules_to_merge[1:]:
                    old_color = module.replace('ME', '')
                    new_colors[new_colors == old_color] = main_color

        # 重新计算合并后的模块特征基因
        new_MEs = self.moduleEigengenes(datExpr, new_colors)

        return new_colors, new_MEs

    # ============= KNN无监督聚类 =============
    class KNNUnsupervisedClustering:
        """基于KNN的无监督聚类算法"""

        def __init__(self, n_neighbors=10, min_cluster_size=5, metric='euclidean',
                     algorithm='auto', contamination=0.1):
            self.n_neighbors = n_neighbors
            self.min_cluster_size = min_cluster_size
            self.metric = metric
            self.algorithm = algorithm
            self.contamination = contamination
            self.labels_ = None
            self.core_sample_indices_ = []
            self.density_ = None
            self.cluster_centers_ = None

        def fit(self, X):
            """拟合KNN聚类模型"""
            n_samples = X.shape[0]

            # 计算k近邻
            nbrs = NearestNeighbors(n_neighbors=self.n_neighbors,
                                    metric=self.metric,
                                    algorithm=self.algorithm)
            nbrs.fit(X)
            distances, indices = nbrs.kneighbors(X)

            # 计算局部密度（基于k近邻距离的倒数）
            self.density_ = 1.0 / (np.mean(distances[:, 1:], axis=1) + 1e-10)

            # 计算密度阈值
            density_threshold = np.percentile(self.density_, self.contamination * 100)

            # 初始化标签
            labels = -np.ones(n_samples, dtype=int)
            visited = np.zeros(n_samples, dtype=bool)
            cluster_id = 0

            # 按密度降序处理点
            density_order = np.argsort(-self.density_)

            for idx in density_order:
                if visited[idx]:
                    continue

                # 检查是否为局部密度峰值
                neighbors = indices[idx, 1:]  # 排除自己
                neighbor_densities = self.density_[neighbors]

                if self.density_[idx] > np.max(neighbor_densities):
                    # 开始新的簇
                    cluster_points = self._expand_cluster(
                        idx, indices, self.density_, visited, density_threshold
                    )

                    if len(cluster_points) >= self.min_cluster_size:
                        labels[cluster_points] = cluster_id
                        self.core_sample_indices_.extend(cluster_points)
                        cluster_id += 1

            self.labels_ = labels

            # 计算簇中心
            self._calculate_cluster_centers(X)

            return self

        def _expand_cluster(self, seed_idx, indices, density, visited, threshold):
            """扩展簇"""
            cluster = [seed_idx]
            visited[seed_idx] = True
            queue = [seed_idx]
            seed_density = density[seed_idx]

            while queue:
                current = queue.pop(0)
                neighbors = indices[current, 1:]  # 排除自己

                for neighbor in neighbors:
                    if not visited[neighbor]:
                        # 密度可达性条件
                        if density[neighbor] >= threshold and \
                                density[neighbor] >= 0.3 * seed_density:
                            cluster.append(neighbor)
                            visited[neighbor] = True

                            # 如果邻居密度足够高，继续扩展
                            if density[neighbor] >= 0.5 * seed_density:
                                queue.append(neighbor)

            return cluster

        def _calculate_cluster_centers(self, X):
            """计算簇中心"""
            unique_labels = np.unique(self.labels_[self.labels_ >= 0])

            if len(unique_labels) > 0:
                self.cluster_centers_ = np.zeros((len(unique_labels), X.shape[1]))

                for i, label in enumerate(unique_labels):
                    mask = self.labels_ == label
                    self.cluster_centers_[i] = np.mean(X[mask], axis=0)

        def fit_predict(self, X):
            """拟合并返回聚类标签"""
            self.fit(X)
            return self.labels_

        def predict(self, X):
            """预测新样本的簇标签"""
            if self.cluster_centers_ is None:
                raise ValueError("Model must be fitted before prediction")

            # 使用最近簇中心进行预测
            from scipy.spatial.distance import cdist
            distances = cdist(X, self.cluster_centers_, metric=self.metric)

            labels = np.argmin(distances, axis=1)

            # 检查是否为噪声点（距离太远）
            min_distances = np.min(distances, axis=1)
            noise_threshold = np.percentile(min_distances, 100 - self.contamination * 100)
            labels[min_distances > noise_threshold] = -1

            return labels

    class EnhancedKNNClustering:
        """增强的KNN聚类，结合多种策略"""

        def __init__(self, n_neighbors_range=(5, 20), min_cluster_size=5,
                     method='adaptive', n_jobs=-1):
            self.n_neighbors_range = n_neighbors_range
            self.min_cluster_size = min_cluster_size
            self.method = method
            self.n_jobs = n_jobs
            self.best_model_ = None
            self.best_score_ = -np.inf
            self.best_n_neighbors_ = None

        def fit(self, X):
            """自适应选择最佳参数的KNN聚类"""
            best_score = -np.inf
            best_model = None
            best_k = None

            # 尝试不同的k值
            for k in range(self.n_neighbors_range[0], self.n_neighbors_range[1] + 1):
                model = KNNUnsupervisedClustering(
                    n_neighbors=k,
                    min_cluster_size=self.min_cluster_size
                )

                labels = model.fit_predict(X)

                # 评估聚类质量
                if len(np.unique(labels[labels >= 0])) > 1:
                    score = self._evaluate_clustering(X, labels)

                    if score > best_score:
                        best_score = score
                        best_model = model
                        best_k = k

            self.best_model_ = best_model
            self.best_score_ = best_score
            self.best_n_neighbors_ = best_k

            return self

        def _evaluate_clustering(self, X, labels):
            """评估聚类质量"""
            # 过滤噪声点
            mask = labels >= 0
            if np.sum(mask) < 2:
                return -np.inf

            X_filtered = X[mask]
            labels_filtered = labels[mask]

            # 计算轮廓系数
            if len(np.unique(labels_filtered)) > 1:
                score = silhouette_score(X_filtered, labels_filtered)
            else:
                score = -1

            # 惩罚噪声点过多
            noise_ratio = np.sum(labels == -1) / len(labels)
            score = score * (1 - noise_ratio * 0.5)

            return score

        def fit_predict(self, X):
            """拟合并返回最佳聚类结果"""
            self.fit(X)
            if self.best_model_ is not None:
                return self.best_model_.labels_
            else:
                return -np.ones(X.shape[0], dtype=int)


# ============= 整合WGCNA和KNN的函数 =============
def perform_integrated_clustering(gene_exp, wgcna_results, args, output_dir):
    """整合WGCNA和KNN聚类分析（优化大数据处理）"""
    print("\n" + "=" * 60)
    print("Integrated WGCNA-KNN Clustering Analysis")
    print("=" * 60)

    module_colors = wgcna_results['colors']
    unique_modules = np.unique(module_colors[module_colors != 'grey'])

    # 初始化结果字典
    integrated_results = {
        'wgcna_modules': module_colors,
        'knn_subclusters': np.zeros(len(module_colors), dtype=int) - 1,  # 初始化为-1（噪声）
        'module_details': {}
    }

    # 如果只有一个模块且基因数量太多，特殊处理
    if len(unique_modules) == 1 and len(module_colors) > 5000:
        print(f"  Warning: Single module with {len(module_colors)} genes detected")
        print(f"  Will attempt to subdivide this large module with adjusted parameters")

        module = unique_modules[0]
        module_mask = module_colors == module
        module_genes = gene_exp.iloc[module_mask, :]

        # 对大模块使用更激进的KNN参数
        print(f"  Using aggressive KNN parameters for large module")

        # 可以考虑先降维
        if module_genes.shape[0] > 5000:
            print(f"  Applying PCA for dimension reduction...")
            from sklearn.decomposition import PCA
            pca = PCA(n_components=min(50, module_genes.shape[1]))
            module_genes_reduced = pca.fit_transform(module_genes.values)
        else:
            module_genes_reduced = module_genes.values

        # 使用更大的k值和更小的min_cluster_size
        knn_model = KNNUnsupervisedClustering(
            n_neighbors=min(50, module_genes.shape[0] // 100),  # 增大k值
            min_cluster_size=max(30, module_genes.shape[0] // 100),  # 动态调整最小簇大小
            contamination=0.05  # 减少噪声点
        )

        try:
            subclusters = knn_model.fit_predict(module_genes_reduced)

            # 更新结果
            unique_subclusters = np.unique(subclusters[subclusters >= 0])
            for i, subcluster in enumerate(unique_subclusters):
                mask = subclusters == subcluster
                integrated_results['knn_subclusters'][np.where(module_mask)[0][mask]] = i

            # 噪声点
            noise_mask = subclusters == -1
            integrated_results['knn_subclusters'][np.where(module_mask)[0][noise_mask]] = -1

            integrated_results['module_details'][module] = {
                'n_genes': module_genes.shape[0],
                'n_subclusters': len(unique_subclusters),
                'n_noise': np.sum(subclusters == -1),
                'knn_model': knn_model
            }

            print(f"  Found {len(unique_subclusters)} subclusters in large module")

        except Exception as e:
            print(f"  Error in KNN clustering: {e}")
            print(f"  Keeping all genes as unclustered")

        return integrated_results

    # 处理多个模块的正常情况
    subcluster_id = 0

    for module in unique_modules:
        print(f"\nProcessing WGCNA module: {module}")
        module_mask = module_colors == module
        module_genes = gene_exp.iloc[module_mask, :]

        if module_genes.shape[0] < args.n_neighbors:
            print(f"  Module too small for KNN clustering ({module_genes.shape[0]} genes)")
            integrated_results['knn_subclusters'][module_mask] = -1
            integrated_results['module_details'][module] = {
                'n_genes': module_genes.shape[0],
                'n_subclusters': 0,
                'n_noise': module_genes.shape[0],
                'knn_model': None
            }
            continue

        # 应用KNN聚类
        if args.knn_method == 'adaptive':
            knn_model = EnhancedKNNClustering(
                n_neighbors_range=(min(5, module_genes.shape[0] // 3),
                                   min(20, module_genes.shape[0] // 2)),
                min_cluster_size=max(3, args.min_cluster_size)
            )
        else:
            knn_model = KNNUnsupervisedClustering(
                n_neighbors=min(args.n_neighbors, module_genes.shape[0] // 2),
                min_cluster_size=max(3, args.min_cluster_size)
            )

        # 执行聚类
        try:
            subclusters = knn_model.fit_predict(module_genes.values)

            # 重新编号子簇
            unique_subclusters = np.unique(subclusters[subclusters >= 0])
            for subcluster in unique_subclusters:
                mask = subclusters == subcluster
                integrated_results['knn_subclusters'][np.where(module_mask)[0][mask]] = subcluster_id
                subcluster_id += 1

            # 噪声点标记为-1
            noise_mask = subclusters == -1
            integrated_results['knn_subclusters'][np.where(module_mask)[0][noise_mask]] = -1

            # 保存模块详细信息
            integrated_results['module_details'][module] = {
                'n_genes': module_genes.shape[0],
                'n_subclusters': len(unique_subclusters),
                'n_noise': np.sum(subclusters == -1),
                'knn_model': knn_model
            }

            print(f"  Found {len(unique_subclusters)} subclusters, {np.sum(subclusters == -1)} noise points")

        except Exception as e:
            print(f"  Error processing module {module}: {e}")
            integrated_results['knn_subclusters'][module_mask] = -1
            integrated_results['module_details'][module] = {
                'n_genes': module_genes.shape[0],
                'n_subclusters': 0,
                'n_noise': module_genes.shape[0],
                'knn_model': None
            }

    return integrated_results


# ============= 命令行参数解析 =============
def parse_arguments():
    """Parse command line arguments with enhanced options"""
    parser = argparse.ArgumentParser(
        description='Enhanced Gene Expression Analysis System V3.0',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Basic analysis:
    python %(prog)s -i gene_expression.csv -g sample_groups.txt

  With WGCNA and KNN:
    python %(prog)s -i gene_expression.csv -g sample_groups.txt --wgcna --knn

  Advanced analysis:
    python %(prog)s -i gene_expression.csv -g sample_groups.txt \\
                    --wgcna --soft_power 6 --min_module_size 30 \\
                    --knn --n_neighbors 10 --knn_method adaptive \\
                    --epochs 200 --learning_rate 0.001

Sample grouping file format (groups.txt):
  Sample_ID    Group
  sample1      T0
  sample2      T0  
  sample3      T1
  sample4      T1
        """
    )

    # 输入输出参数
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='Gene expression matrix file (rows as genes, columns as samples)')
    parser.add_argument('-g', '--groups', type=str, required=True,
                        help='Sample grouping file (two columns: Sample_ID, Group)')
    parser.add_argument('--sep', type=str, default='auto',
                        help='Expression matrix separator: auto, tab, comma, space (default: auto)')
    parser.add_argument('--group_sep', type=str, default='auto',
                        help='Group file separator: auto, tab, comma, space (default: auto)')
    parser.add_argument('--output_prefix', type=str, default='analysis',
                        help='Output file prefix (default: analysis)')

    # WGCNA参数
    parser.add_argument('--wgcna', action='store_true', default=False,
                        help='Enable WGCNA analysis')
    parser.add_argument('--soft_power', type=int, default=0,
                        help='Soft threshold power for WGCNA (0=auto-detect, default: 0)')
    parser.add_argument('--min_module_size', type=int, default=30,
                        help='Minimum module size for WGCNA (default: 30)')
    parser.add_argument('--merge_cut_height', type=float, default=0.25,
                        help='Cut height for merging modules (default: 0.25)')
    parser.add_argument('--network_type', type=str, default='unsigned',
                        choices=['unsigned', 'signed', 'signed_hybrid'],
                        help='Network type for WGCNA (default: unsigned)')
    parser.add_argument('--deep_split', type=int, default=2,
                        choices=[0, 1, 2, 3],
                        help='Deep split parameter for tree cutting (default: 2)')

    # KNN聚类参数
    parser.add_argument('--knn', action='store_true', default=False,
                        help='Enable KNN unsupervised clustering')
    parser.add_argument('--n_neighbors', type=int, default=10,
                        help='Number of neighbors for KNN clustering (default: 10)')
    parser.add_argument('--min_cluster_size', type=int, default=5,
                        help='Minimum cluster size for KNN (default: 5)')
    parser.add_argument('--knn_method', type=str, default='standard',
                        choices=['standard', 'adaptive'],
                        help='KNN clustering method (default: standard)')

    # 传统聚类参数
    parser.add_argument('--n_clusters', type=int, default=6,
                        help='Number of clusters for K-means (default: 6)')

    # 深度学习参数
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Test set ratio (default: 0.2)')
    parser.add_argument('--val_size', type=float, default=0.2,
                        help='Validation set ratio (default: 0.2)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Training epochs (default: 100)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate for deep learning models (default: 0.001)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training (default: 32)')
    parser.add_argument('--hidden_size', type=int, default=64,
                        help='Hidden size for neural networks (default: 64)')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate (default: 0.3)')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of layers in RNN models (default: 2)')
    parser.add_argument('--early_stopping', action='store_true', default=True,
                        help='Enable early stopping')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience (default: 20)')

    # 高级选项
    parser.add_argument('--grid_search', action='store_true',
                        help='Enable grid search for hyperparameter optimization')
    parser.add_argument('--export_models', type=lambda x: x.lower() != 'false', default=True,
                        help='Export models automatically (default: True)')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='Verbose output')
    parser.add_argument('--n_jobs', type=int, default=-1,
                        help='Number of parallel jobs (-1 for all cores)')

    return parser.parse_args()


# ============= 目录结构创建 =============
def create_output_structure(output_prefix):
    """Create comprehensive output directory structure"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    main_dir = f"{output_prefix}_results_{timestamp}"

    # 创建主目录
    os.makedirs(main_dir, exist_ok=True)

    # 创建子目录结构
    subdirs = {
        'models': os.path.join(main_dir, 'models'),
        'visualizations': os.path.join(main_dir, 'visualizations'),
        'data': os.path.join(main_dir, 'data'),
        'reports': os.path.join(main_dir, 'reports'),
        'networks': os.path.join(main_dir, 'networks'),
        'wgcna': os.path.join(main_dir, 'wgcna_analysis'),
        'knn': os.path.join(main_dir, 'knn_clustering'),
        'time_series': os.path.join(main_dir, 'time_series_analysis'),
        'overfitting': os.path.join(main_dir, 'overfitting_analysis'),
        'nn_architectures': os.path.join(main_dir, 'neural_network_architectures'),
        'heatmaps': os.path.join(main_dir, 'heatmap_analysis'),
        'boxplots': os.path.join(main_dir, 'boxplot_analysis'),
        'intersection': os.path.join(main_dir, 'intersection_analysis'),
        'exported_models': os.path.join(main_dir, 'exported_models')
    }

    for subdir in subdirs.values():
        os.makedirs(subdir, exist_ok=True)

    # 为每个模型创建专用目录
    model_names = [
        'RNN', 'LSTM', 'BiLSTM', 'GRU', 'CNN', 'ResNet',
        'Transformer', 'AttentionRNN', 'TCN', 'WaveNet', 'InceptionTime',
        'RandomForest', 'GradientBoosting', 'AdaBoost', 'ExtraTrees',
        'SVM', 'KNN', 'LogisticRegression', 'NaiveBayes'
    ]

    for model_name in model_names:
        model_dir = os.path.join(subdirs['models'], model_name)
        os.makedirs(model_dir, exist_ok=True)

        export_dir = os.path.join(subdirs['exported_models'], model_name)
        os.makedirs(export_dir, exist_ok=True)

    print(f"Created output directory structure: {main_dir}")
    print(f"  Total subdirectories: {len(subdirs)}")

    # 保存目录结构信息
    structure_info = {
        'created_at': timestamp,
        'main_dir': main_dir,
        'subdirs': subdirs,
        'models': model_names
    }

    with open(os.path.join(main_dir, 'directory_structure.json'), 'w') as f:
        json.dump(structure_info, f, indent=2)

    return main_dir, subdirs


# ============= 数据加载函数 =============
def detect_separator(file_path, n_lines=5):
    """Automatically detect file separator"""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [f.readline() for _ in range(min(n_lines, 100))]

    separators = {'\t': 0, ',': 0, ' ': 0, ';': 0}
    for line in lines:
        for sep in separators:
            separators[sep] += line.count(sep)

    # 选择出现次数最多的分隔符
    best_sep = max(separators, key=separators.get)

    # 验证分隔符
    if separators[best_sep] == 0:
        raise ValueError(f"Could not detect separator in {file_path}")

    return best_sep


def load_expression_data(file_path, sep='auto'):
    """Load and preprocess gene expression data"""
    print("=" * 60)
    print("Loading Gene Expression Data")
    print("=" * 60)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # 检测分隔符
    if sep == 'auto':
        sep = detect_separator(file_path)
        print(f"Auto-detected separator: '{repr(sep)}'")
    elif sep == 'tab':
        sep = '\t'
    elif sep == 'comma':
        sep = ','
    elif sep == 'space':
        sep = ' '

    # 加载数据
    try:
        gene_exp = pd.read_csv(file_path, sep=sep, index_col=0)
        print(f"Successfully loaded: {gene_exp.shape[0]} genes, {gene_exp.shape[1]} samples")
    except Exception as e:
        raise ValueError(f"Unable to read file: {e}")

    # 数据验证
    if gene_exp.empty:
        raise ValueError("Expression matrix is empty")

    if gene_exp.isnull().any().any():
        print(f"Warning: Found {gene_exp.isnull().sum().sum()} missing values")
        gene_exp = gene_exp.fillna(0)

    print("\nData preprocessing...")

    # 保存原始数据
    original_gene_exp = gene_exp.copy()

    # 预处理统计
    preprocessing_stats = {
        'original_genes': gene_exp.shape[0],
        'original_samples': gene_exp.shape[1],
        'zero_genes_removed': 0,
        'low_exp_removed': 0,
        'low_variance_removed': 0,
        'log_transformed': False
    }

    # 移除零表达基因
    zero_genes = (gene_exp == 0).all(axis=1)
    if zero_genes.sum() > 0:
        gene_exp = gene_exp[~zero_genes]
        preprocessing_stats['zero_genes_removed'] = zero_genes.sum()
        print(f"  Removed {zero_genes.sum()} zero-expression genes")

    # 移除低表达基因
    min_exp = 1
    low_exp = (gene_exp > min_exp).sum(axis=1) < 3
    if low_exp.sum() > 0:
        gene_exp = gene_exp[~low_exp]
        preprocessing_stats['low_exp_removed'] = low_exp.sum()
        print(f"  Removed {low_exp.sum()} low-expression genes")

    # 移除低方差基因
    gene_variance = gene_exp.var(axis=1)
    low_var_threshold = gene_variance.quantile(0.1)
    low_var = gene_variance < low_var_threshold
    if low_var.sum() > 0:
        gene_exp = gene_exp[~low_var]
        preprocessing_stats['low_variance_removed'] = low_var.sum()
        print(f"  Removed {low_var.sum()} low-variance genes")

    # Log转换
    if gene_exp.max().max() > 100:
        gene_exp = np.log2(gene_exp + 1)
        preprocessing_stats['log_transformed'] = True
        print("  Applied log2 transformation")

    preprocessing_stats['final_genes'] = gene_exp.shape[0]
    preprocessing_stats['final_samples'] = gene_exp.shape[1]

    print(f"\nAfter preprocessing: {gene_exp.shape[0]} genes, {gene_exp.shape[1]} samples")

    return gene_exp, original_gene_exp, preprocessing_stats


def load_sample_groups(file_path, sep='auto'):
    """Load sample grouping information"""
    print("\n" + "=" * 60)
    print("Loading Sample Grouping Information")
    print("=" * 60)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Group file not found: {file_path}")

    # 检测分隔符
    if sep == 'auto':
        sep = detect_separator(file_path)
        print(f"Group file separator: '{repr(sep)}'")
    elif sep == 'tab':
        sep = '\t'
    elif sep == 'comma':
        sep = ','
    elif sep == 'space':
        sep = ' '

    # 加载数据
    try:
        groups_df = pd.read_csv(file_path, sep=sep)

        # 标准化列名
        if len(groups_df.columns) >= 2:
            groups_df.columns = ['Sample_ID', 'Group'] + list(groups_df.columns[2:])
        else:
            raise ValueError("Group file must have at least 2 columns")

        print(f"Loaded {len(groups_df)} sample grouping information")
    except Exception as e:
        raise ValueError(f"Unable to read group file: {e}")

    # 数据验证
    if groups_df['Sample_ID'].duplicated().any():
        print("Warning: Found duplicated sample IDs, keeping first occurrence")
        groups_df = groups_df.drop_duplicates(subset=['Sample_ID'], keep='first')

    # 统计信息
    group_counts = groups_df['Group'].value_counts()
    print("\nGroup statistics:")
    for group, count in group_counts.items():
        print(f"  {group}: {count} samples")

    return groups_df


# ============= 深度学习模型定义 =============
class RNN_Model(nn.Module):
    """RNN Model with enhanced features"""

    def __init__(self, input_size, hidden_size=64, num_layers=2, num_classes=4, dropout=0.3):
        super(RNN_Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0,
                          nonlinearity='tanh')
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm1d(hidden_size)

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(-1)

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, hidden = self.rnn(x, h0)

        # 取最后一个时间步的输出
        out = out[:, -1, :]
        out = self.batch_norm(out)
        out = self.dropout(out)
        out = self.fc(out)
        return out


class LSTM_Model(nn.Module):
    """LSTM Model with enhanced features"""

    def __init__(self, input_size, hidden_size=64, num_layers=2, num_classes=4, dropout=0.3):
        super(LSTM_Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm1d(hidden_size)

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(-1)

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, (hidden, cell) = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.batch_norm(out)
        out = self.dropout(out)
        out = self.fc(out)
        return out


class BiLSTM_Model(nn.Module):
    """Bidirectional LSTM Model"""

    def __init__(self, input_size, hidden_size=64, num_layers=2, num_classes=4, dropout=0.3):
        super(BiLSTM_Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm1d(hidden_size * 2)

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(-1)

        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.batch_norm(out)
        out = self.dropout(out)
        out = self.fc(out)
        return out


class GRU_Model(nn.Module):
    """GRU Model"""

    def __init__(self, input_size, hidden_size=64, num_layers=2, num_classes=4, dropout=0.3):
        super(GRU_Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm1d(hidden_size)

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(-1)

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = out[:, -1, :]
        out = self.batch_norm(out)
        out = self.dropout(out)
        out = self.fc(out)
        return out


class CNN_Model(nn.Module):
    """1D CNN Model for sequence data"""

    def __init__(self, input_size, num_classes=4, dropout=0.3):
        super(CNN_Model, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        out = self.fc(x)
        return out


class ResNet_Model(nn.Module):
    """ResNet Model with residual connections"""

    def __init__(self, input_size, num_classes=4, dropout=0.3):
        super(ResNet_Model, self).__init__()
        # 第一个残差块
        self.fc1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.shortcut1 = nn.Linear(input_size, 128)

        # 第二个残差块
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.shortcut2 = nn.Linear(128, 64)

        self.fc_out = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 第一个残差块
        identity = self.shortcut1(x)
        out = torch.relu(self.bn1(self.fc1(x)))
        out = self.dropout(out)
        out = torch.relu(self.bn2(self.fc2(out)))
        out = out + identity
        out = torch.relu(out)

        # 第二个残差块
        identity = self.shortcut2(out)
        out = torch.relu(self.bn3(self.fc3(out)))
        out = self.dropout(out)
        out = torch.relu(self.bn4(self.fc4(out)))
        out = out + identity
        out = torch.relu(out)

        out = self.fc_out(out)
        return out


class PositionalEncoding(nn.Module):
    """位置编码模块"""

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TransformerModel(nn.Module):
    """Transformer Model"""

    def __init__(self, input_size, d_model=64, nhead=4, num_classes=4, dropout=0.3):
        super(TransformerModel, self).__init__()
        self.input_projection = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=256,
            dropout=dropout, activation='relu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(-1)

        # 输入投影
        x = self.input_projection(x)

        # 调整维度为 (seq_len, batch, d_model)
        x = x.transpose(0, 1)

        # 位置编码
        x = self.pos_encoder(x)

        # Transformer编码器
        x = self.transformer(x)

        # 取平均池化
        x = x.mean(dim=0)

        x = self.dropout(x)
        out = self.fc(x)
        return out


class AttentionRNN(nn.Module):
    """RNN with Attention Mechanism"""

    def __init__(self, input_size, hidden_size=64, num_classes=4, dropout=0.3):
        super(AttentionRNN, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, 2,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if 2 > 1 else 0)
        self.attention = nn.Linear(hidden_size * 2, 1)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(-1)

        # LSTM编码
        lstm_out, _ = self.lstm(x)

        # 计算注意力权重
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)

        # 加权求和
        context = torch.sum(attention_weights * lstm_out, dim=1)

        out = self.fc(self.dropout(context))
        return out


class TCN_Model(nn.Module):
    """Temporal Convolutional Network"""

    def __init__(self, input_size, num_classes=4, dropout=0.3):
        super(TCN_Model, self).__init__()
        # 膨胀卷积层
        self.tcn1 = nn.Conv1d(1, 64, kernel_size=3, dilation=1, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.tcn2 = nn.Conv1d(64, 64, kernel_size=3, dilation=2, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.tcn3 = nn.Conv1d(64, 128, kernel_size=3, dilation=4, padding=4)
        self.bn3 = nn.BatchNorm1d(128)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        x = torch.relu(self.bn1(self.tcn1(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.tcn2(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn3(self.tcn3(x)))

        x = self.pool(x).squeeze(-1)
        out = self.fc(self.dropout(x))
        return out


class WaveNet_Model(nn.Module):
    """WaveNet Model with dilated convolutions"""

    def __init__(self, input_size, num_classes=4, dropout=0.3):
        super(WaveNet_Model, self).__init__()
        # 门控卷积层
        self.conv1 = nn.Conv1d(1, 32, kernel_size=2, dilation=1, padding=1)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=2, dilation=2, padding=2)
        self.conv3 = nn.Conv1d(32, 32, kernel_size=2, dilation=4, padding=4)
        self.conv4 = nn.Conv1d(32, 64, kernel_size=1)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        # WaveNet层
        x = torch.tanh(self.conv1(x)) * torch.sigmoid(self.conv1(x))
        x = torch.tanh(self.conv2(x)) * torch.sigmoid(self.conv2(x))
        x = torch.tanh(self.conv3(x)) * torch.sigmoid(self.conv3(x))
        x = torch.relu(self.conv4(x))

        x = self.pool(x).squeeze(-1)
        out = self.fc(self.dropout(x))
        return out


class InceptionTime(nn.Module):
    """InceptionTime Model with multiple kernel sizes"""

    def __init__(self, input_size, num_classes=4, dropout=0.3):
        super(InceptionTime, self).__init__()
        # Inception模块
        self.conv1x1 = nn.Conv1d(1, 32, kernel_size=1)
        self.conv3x3 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv5x5 = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.conv7x7 = nn.Conv1d(1, 32, kernel_size=7, padding=3)

        self.bn = nn.BatchNorm1d(128)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        # 并行卷积
        out1 = torch.relu(self.conv1x1(x))
        out2 = torch.relu(self.conv3x3(x))
        out3 = torch.relu(self.conv5x5(x))
        out4 = torch.relu(self.conv7x7(x))

        # 拼接特征
        out = torch.cat([out1, out2, out3, out4], dim=1)
        out = self.bn(out)
        out = self.pool(out).squeeze(-1)
        out = self.fc(self.dropout(out))
        return out


# ============= 神经网络架构可视化（圆形节点） =============
def visualize_neural_network_circular(model_name, model, output_dir):
    """创建圆形节点的神经网络架构图"""
    fig, ax = plt.subplots(figsize=(14, 10), facecolor='white')

    # 定义网络层结构
    layer_configs = {
        'RNN': {'layers': [('Input', 4), ('RNN-1', 64), ('RNN-2', 64), ('Output', 4)],
                'colors': ['#3498db', '#e74c3c', '#e74c3c', '#2ecc71']},
        'LSTM': {'layers': [('Input', 4), ('LSTM-1', 64), ('LSTM-2', 64), ('Output', 4)],
                 'colors': ['#3498db', '#9b59b6', '#9b59b6', '#2ecc71']},
        'BiLSTM': {'layers': [('Input', 4), ('BiLSTM-1', 128), ('BiLSTM-2', 128), ('Output', 4)],
                   'colors': ['#3498db', '#e67e22', '#e67e22', '#2ecc71']},
        'GRU': {'layers': [('Input', 4), ('GRU-1', 64), ('GRU-2', 64), ('Output', 4)],
                'colors': ['#3498db', '#1abc9c', '#1abc9c', '#2ecc71']},
        'CNN': {'layers': [('Input', 4), ('Conv1', 32), ('Conv2', 64), ('Conv3', 128), ('Output', 4)],
                'colors': ['#3498db', '#f39c12', '#e74c3c', '#9b59b6', '#2ecc71']},
        'ResNet': {'layers': [('Input', 4), ('Block1', 128), ('Block2', 64), ('Output', 4)],
                   'colors': ['#3498db', '#34495e', '#7f8c8d', '#2ecc71']},
        'Transformer': {'layers': [('Input', 4), ('Embed', 64), ('Attention', 64), ('FFN', 64), ('Output', 4)],
                        'colors': ['#3498db', '#8e44ad', '#e74c3c', '#f39c12', '#2ecc71']},
        'AttentionRNN': {'layers': [('Input', 4), ('LSTM', 128), ('Attention', 128), ('Output', 4)],
                         'colors': ['#3498db', '#16a085', '#e74c3c', '#2ecc71']},
        'TCN': {'layers': [('Input', 4), ('TCN-1', 64), ('TCN-2', 64), ('TCN-3', 128), ('Output', 4)],
                'colors': ['#3498db', '#d35400', '#c0392b', '#8e44ad', '#2ecc71']},
        'WaveNet': {'layers': [('Input', 4), ('Wave-1', 32), ('Wave-2', 32), ('Wave-3', 64), ('Output', 4)],
                    'colors': ['#3498db', '#27ae60', '#2980b9', '#8e44ad', '#2ecc71']},
        'InceptionTime': {'layers': [('Input', 4), ('Inception', 128), ('Pool', 128), ('Output', 4)],
                          'colors': ['#3498db', '#e74c3c', '#f39c12', '#2ecc71']}
    }

    # 获取配置
    if model_name not in layer_configs:
        config = {'layers': [('Input', 4), ('Hidden-1', 64), ('Hidden-2', 32), ('Output', 4)],
                  'colors': ['#3498db', '#95a5a6', '#7f8c8d', '#2ecc71']}
    else:
        config = layer_configs[model_name]

    layers = config['layers']
    colors = config['colors']
    n_layers = len(layers)

    # 布局参数
    layer_spacing = 0.8 / (n_layers - 1) if n_layers > 1 else 0.8
    max_nodes_display = 8  # 每层最多显示的节点数

    # 存储节点位置用于连线
    node_positions = []

    for layer_idx, (layer_name, n_nodes) in enumerate(layers):
        x = 0.1 + layer_idx * layer_spacing

        # 限制显示的节点数
        display_nodes = min(n_nodes, max_nodes_display)

        # 计算y位置
        if display_nodes == 1:
            y_positions = [0.5]
        else:
            y_spacing = 0.6 / (display_nodes - 1)
            y_start = 0.2
            y_positions = [y_start + i * y_spacing for i in range(display_nodes)]

        layer_nodes = []

        # 绘制节点（圆形）
        for node_idx, y in enumerate(y_positions):
            # 创建圆形节点
            circle = Circle((x, y), 0.025,
                            color=colors[layer_idx],
                            ec='black',
                            linewidth=2,
                            zorder=10,
                            alpha=0.9)
            ax.add_patch(circle)

            # 添加节点光晕效果
            halo = Circle((x, y), 0.03,
                          color=colors[layer_idx],
                          alpha=0.3,
                          zorder=9)
            ax.add_patch(halo)

            layer_nodes.append((x, y))

        # 如果有省略的节点，添加省略号
        if n_nodes > max_nodes_display:
            ax.text(x, y_positions[-1] - 0.08, '...',
                    ha='center', va='center', fontsize=12, fontweight='bold')
            ax.text(x, y_positions[-1] - 0.12, f'({n_nodes} nodes)',
                    ha='center', va='center', fontsize=9, style='italic')

        # 添加层标签
        ax.text(x, 0.08, layer_name, ha='center', fontsize=12, fontweight='bold')
        if n_nodes <= max_nodes_display:
            ax.text(x, 0.04, f'({n_nodes} nodes)', ha='center', fontsize=9, style='italic')

        node_positions.append(layer_nodes)

    # 绘制连接线
    for layer_idx in range(len(node_positions) - 1):
        current_layer = node_positions[layer_idx]
        next_layer = node_positions[layer_idx + 1]

        # 限制连接线数量以保持清晰
        max_connections = 5
        current_sample = current_layer[:min(max_connections, len(current_layer))]
        next_sample = next_layer[:min(max_connections, len(next_layer))]

        for x1, y1 in current_sample:
            for x2, y2 in next_sample:
                # 创建曲线连接
                arrow = FancyArrowPatch(
                    (x1 + 0.025, y1), (x2 - 0.025, y2),
                    connectionstyle="arc3,rad=.1",
                    arrowstyle='-',
                    color='gray',
                    alpha=0.3,
                    linewidth=0.8,
                    zorder=1
                )
                ax.add_patch(arrow)

    # 添加模型信息框
    info_text = f"Model: {model_name}\n"
    if hasattr(model, 'hidden_size'):
        info_text += f"Hidden Size: {model.hidden_size}\n"
    if hasattr(model, 'num_layers'):
        info_text += f"Layers: {model.num_layers}\n"
    if hasattr(model, 'dropout'):
        if hasattr(model.dropout, 'p'):
            info_text += f"Dropout: {model.dropout.p}"

    # 信息框样式
    props = dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8)
    ax.text(0.95, 0.95, info_text, transform=ax.transAxes,
            fontsize=10, va='top', ha='right', bbox=props)

    # 添加标题
    ax.text(0.5, 0.95, f'{model_name} Neural Network Architecture',
            transform=ax.transAxes, fontsize=16, fontweight='bold',
            ha='center', va='top')

    # 设置轴属性
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')

    # 添加网格背景
    for i in range(11):
        ax.axhline(y=i / 10, color='lightgray', alpha=0.2, linestyle='--', linewidth=0.5)
        ax.axvline(x=i / 10, color='lightgray', alpha=0.2, linestyle='--', linewidth=0.5)

    plt.tight_layout()

    # 保存图像
    save_path = os.path.join(output_dir, f'{model_name}_architecture_circular.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"  Created neural network architecture: {save_path}")

    return save_path


# ============= 时序分析图（彩色模式，类似示例） =============
def create_enhanced_time_series_plot(gene_exp, groups_dict, sorted_groups,
                                     selected_genes, output_dir):
    """创建彩色时序分析图，类似提供的示例"""
    print("\nCreating enhanced time series plots...")

    # 准备数据
    time_series_data = []
    for gene in selected_genes[:21]:  # 最多21个基因
        if gene in gene_exp.index:
            gene_data = []
            for group in sorted_groups:
                group_samples = groups_dict[group]
                if group_samples:
                    values = []
                    for sample in group_samples:
                        if sample in gene_exp.columns:
                            values.append(gene_exp.loc[gene, sample])
                    if values:
                        gene_data.append({
                            'mean': np.mean(values),
                            'std': np.std(values),
                            'sem': np.std(values) / np.sqrt(len(values)),
                            'values': values
                        })
                    else:
                        gene_data.append({'mean': 0, 'std': 0, 'sem': 0, 'values': []})

            time_series_data.append({
                'gene': gene,
                'data': gene_data
            })

    if not time_series_data:
        print("  No data for time series plot")
        return

    # 创建图形
    n_genes = len(time_series_data)
    n_cols = 3
    n_rows = (n_genes + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    # 使用彩色方案
    colors = plt.cm.tab20(np.linspace(0, 1, n_genes))

    # 设置全局字体
    plt.rcParams.update({'font.size': 10})

    for idx, gene_info in enumerate(time_series_data):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        gene_name = gene_info['gene']
        data = gene_info['data']

        # 提取数据
        means = [d['mean'] for d in data]
        stds = [d['std'] for d in data]
        sems = [d['sem'] for d in data]
        x = np.arange(len(means))

        # 主线图
        line = ax.plot(x, means, 'o-',
                       color=colors[idx],
                       linewidth=2.5,
                       markersize=8,
                       markeredgewidth=2,
                       markeredgecolor='white',
                       label=gene_name[:15],
                       zorder=3)

        # 添加置信区间（使用标准误）
        ax.fill_between(x,
                        [m - 2 * s for m, s in zip(means, sems)],
                        [m + 2 * s for m, s in zip(means, sems)],
                        color=colors[idx],
                        alpha=0.2,
                        zorder=1)

        # 添加误差棒
        ax.errorbar(x, means, yerr=sems,
                    fmt='none',
                    ecolor=colors[idx],
                    alpha=0.5,
                    capsize=3,
                    zorder=2)

        # 添加趋势线
        if len(x) >= 2:
            z = np.polyfit(x, means, 1)
            p = np.poly1d(z)
            ax.plot(x, p(x), '--',
                    color='red',
                    alpha=0.5,
                    linewidth=1.5,
                    label=f'Trend (slope={z[0]:.3f})')

        # 计算并显示表达变化
        if len(means) > 1:
            fold_change = means[-1] / (means[0] + 1e-10)
            change_text = f'FC: {fold_change:.2f}'
            ax.text(0.98, 0.98, change_text,
                    transform=ax.transAxes,
                    fontsize=9,
                    va='top', ha='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        # 设置轴标签
        ax.set_xticks(x)
        ax.set_xticklabels(sorted_groups, rotation=45, ha='right', fontsize=9)
        ax.set_xlabel('Time Point / Group', fontsize=10)
        ax.set_ylabel('Expression Level', fontsize=10)

        # 设置标题
        ax.set_title(gene_name[:20], fontsize=11, fontweight='bold')

        # 网格
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)

        # 设置y轴范围（留出空间）
        y_range = max(means) - min(means)
        if y_range > 0:
            ax.set_ylim(min(means) - y_range * 0.1, max(means) + y_range * 0.2)

    # 隐藏未使用的子图
    for idx in range(n_genes, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)

    # 添加总标题
    plt.suptitle('Age Trends by Pathogen - Time Series Analysis',
                 fontsize=16, fontweight='bold', y=1.02)

    plt.tight_layout()

    # 保存图像
    save_path = os.path.join(output_dir, 'time_series_analysis_enhanced.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"  Saved time series plot: {save_path}")

    return save_path


# ============= 雷达图功能 =============
# ============= 雷达图功能 =============
def create_model_radar_charts(model_results, output_dir):
    """创建模型性能雷达图"""
    if not model_results:
        return

    print("  Creating radar charts...")

    sorted_models = sorted(model_results.items(),
                           key=lambda x: x[1]['accuracy'], reverse=True)[:5]

    fig = plt.figure(figsize=(20, 8))

    # 主雷达图
    ax1 = plt.subplot(131, polar=True)

    categories = ['Accuracy', 'Precision', 'Recall', 'F1', 'Stability']
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    for i, (model_name, metrics) in enumerate(sorted_models):
        stability = 1 - abs(metrics['precision'] - metrics['recall'])
        values = [metrics['accuracy'], metrics['precision'], metrics['recall'],
                  metrics['f1'], stability]
        values += values[:1]

        color = NATURE_COLORS[i % len(NATURE_COLORS)]
        ax1.plot(angles, values, 'o-', linewidth=2, label=model_name[:10], color=color)
        ax1.fill(angles, values, alpha=0.15, color=color)

    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(categories)
    ax1.set_ylim(0, 1)
    ax1.set_title('Top 5 Models Performance', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax1.grid(True)

    # 保存
    plt.tight_layout()
    radar_path = os.path.join(output_dir, 'model_radar_charts.png')
    plt.savefig(radar_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {radar_path}")


# ============= PR Trade-off图（正确版本）=============
def create_precision_recall_tradeoff(model_results, output_dir):
    """创建PR权衡分析图"""
    if not model_results:
        return

    print("  Creating Precision-Recall trade-off...")

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 7))

    models = list(model_results.keys())
    precisions = [model_results[m]['precision'] for m in models]
    recalls = [model_results[m]['recall'] for m in models]
    f1_scores = [model_results[m]['f1'] for m in models]
    accuracies = [model_results[m]['accuracy'] for m in models]

    dl_models = ['RNN', 'LSTM', 'BiLSTM', 'GRU', 'CNN', 'ResNet',
                 'Transformer', 'AttentionRNN', 'TCN', 'WaveNet', 'InceptionTime']

    # 1. PR散点图（使用直线F1等值线）
    for i, model in enumerate(models):
        if model in dl_models:
            color = NATURE_COLORS[i % len(NATURE_COLORS)]
            marker = 'o'
        else:
            color = SCIENCE_COLORS[i % len(SCIENCE_COLORS)]
            marker = '^'

        ax1.scatter(recalls[i], precisions[i], s=150, alpha=0.7,
                    marker=marker, color=color, edgecolors='black', linewidth=1)

        if i < 5:
            ax1.annotate(model[:8], (recalls[i], precisions[i]),
                         xytext=(5, 5), textcoords='offset points', fontsize=9)

    # 添加F1等值直线（不是曲线）
    for f1 in [0.2, 0.4, 0.6, 0.8]:
        # 绘制从(0, f1)到(f1, 0)的直线
        x_points = [0.01, 0.99]
        y_points = []
        for x in x_points:
            y = f1 * x / (2 * x - f1) if (2 * x - f1) != 0 else 0
            y_points.append(y)

        # 只绘制在有效范围内的线段
        valid_x = []
        valid_y = []
        for x in np.linspace(0.01, 0.99, 100):
            y = f1 * x / (2 * x - f1) if (2 * x - f1) != 0 else 0
            if 0 <= y <= 1:
                valid_x.append(x)
                valid_y.append(y)

        if valid_x and valid_y:
            ax1.plot(valid_x, valid_y, '--', alpha=0.3, color='gray', linewidth=1)
            # 在线的末端添加标签
            ax1.text(valid_x[-1], valid_y[-1], f'F1={f1}',
                     fontsize=8, alpha=0.5)

    ax1.set_xlabel('Recall', fontsize=12)
    ax1.set_ylabel('Precision', fontsize=12)
    ax1.set_title('Precision-Recall Trade-off', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1.05)
    ax1.set_ylim(0, 1.05)

    # 其他两个子图保持不变...
    # 2. F1 vs Accuracy散点图
    ax2.scatter(accuracies, f1_scores, s=100, alpha=0.7, color='blue')
    z = np.polyfit(accuracies, f1_scores, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(min(accuracies), max(accuracies), 100)
    ax2.plot(x_trend, p(x_trend), 'r--', alpha=0.5, linewidth=2)
    ax2.set_xlabel('Accuracy', fontsize=12)
    ax2.set_ylabel('F1 Score', fontsize=12)
    ax2.set_title('F1 vs Accuracy', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # 3. 模型类型对比箱线图
    dl_metrics = {'Precision': [], 'Recall': [], 'F1': []}
    ml_metrics = {'Precision': [], 'Recall': [], 'F1': []}

    for model, metrics in model_results.items():
        if model in dl_models:
            dl_metrics['Precision'].append(metrics['precision'])
            dl_metrics['Recall'].append(metrics['recall'])
            dl_metrics['F1'].append(metrics['f1'])
        else:
            ml_metrics['Precision'].append(metrics['precision'])
            ml_metrics['Recall'].append(metrics['recall'])
            ml_metrics['F1'].append(metrics['f1'])

    data_to_plot = []
    positions = []
    colors = []

    for i, metric in enumerate(['Precision', 'Recall', 'F1']):
        if dl_metrics[metric]:
            data_to_plot.append(dl_metrics[metric])
            positions.append(i * 2)
            colors.append(NATURE_COLORS[0])
        if ml_metrics[metric]:
            data_to_plot.append(ml_metrics[metric])
            positions.append(i * 2 + 0.8)
            colors.append(SCIENCE_COLORS[0])

    bp = ax3.boxplot(data_to_plot, positions=positions, widths=0.6, patch_artist=True)

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax3.set_xticks([i * 2 + 0.4 for i in range(3)])
    ax3.set_xticklabels(['Precision', 'Recall', 'F1'])
    ax3.set_ylabel('Score', fontsize=12)
    ax3.set_title('Model Type Comparison', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # 保存为PNG和PDF
    tradeoff_path_png = os.path.join(output_dir, 'precision_recall_tradeoff.png')
    tradeoff_path_pdf = os.path.join(output_dir, 'precision_recall_tradeoff.pdf')
    plt.savefig(tradeoff_path_png, dpi=300, bbox_inches='tight')
    plt.savefig(tradeoff_path_pdf, bbox_inches='tight')
    plt.close()

    print(f"    Saved: {tradeoff_path_png}")
    print(f"    Saved: {tradeoff_path_pdf}")

# ============= PR Trade-off图 =============
def create_precision_recall_tradeoff(model_results, output_dir):
    """创建PR权衡分析图（修复PDF空白问题）"""
    if not model_results:
        return

    print("  Creating Precision-Recall trade-off...")

    # 创建图形
    fig = plt.figure(figsize=(20, 7))

    # 使用GridSpec来更好地控制布局
    gs = fig.add_gridspec(1, 3, hspace=0.3, wspace=0.3)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    models = list(model_results.keys())
    precisions = [model_results[m]['precision'] for m in models]
    recalls = [model_results[m]['recall'] for m in models]
    f1_scores = [model_results[m]['f1'] for m in models]
    accuracies = [model_results[m]['accuracy'] for m in models]

    dl_models = ['RNN', 'LSTM', 'BiLSTM', 'GRU', 'CNN', 'ResNet',
                 'Transformer', 'AttentionRNN', 'TCN', 'WaveNet', 'InceptionTime']

    # 1. PR散点图
    for i, model in enumerate(models):
        if model in dl_models:
            color = NATURE_COLORS[i % len(NATURE_COLORS)]
            marker = 'o'
        else:
            color = SCIENCE_COLORS[i % len(SCIENCE_COLORS)]
            marker = '^'

        ax1.scatter(recalls[i], precisions[i], s=150, alpha=0.7,
                    marker=marker, color=color, edgecolors='black', linewidth=1)

    # 添加F1等值直线
    for f1 in [0.2, 0.4, 0.6, 0.8]:
        x = np.array([0.01, 0.99])
        y = f1 * x / (2 * x - f1)
        mask = (y >= 0) & (y <= 1)
        if np.any(mask):
            ax1.plot(x[mask], y[mask], '--', alpha=0.3, color='gray', linewidth=1)
            # 添加标签
            if f1 in [0.4, 0.6, 0.8]:
                x_label = 0.9
                y_label = f1 * x_label / (2 * x_label - f1)
                if 0 <= y_label <= 1:
                    ax1.text(x_label, y_label, f'F1={f1}', fontsize=8, alpha=0.5)

    ax1.set_xlabel('Recall', fontsize=12)
    ax1.set_ylabel('Precision', fontsize=12)
    ax1.set_title('Precision-Recall Trade-off', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1.05)
    ax1.set_ylim(0, 1.05)

    # 其余代码保持不变...

    plt.tight_layout()

    # 保存图形 - 确保PDF正确保存
    save_path_png = os.path.join(output_dir, 'precision_recall_tradeoff.png')
    save_path_pdf = os.path.join(output_dir, 'precision_recall_tradeoff.pdf')

    plt.savefig(save_path_png, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.savefig(save_path_pdf, format='pdf', bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

    print(f"    Saved: {save_path_png}")
    print(f"    Saved: {save_path_pdf}")


def create_expression_pattern_analysis(key_genes, output_dir):
    """创建表达模式分析图"""
    if key_genes is None or key_genes.empty:
        print("  No key genes data for expression pattern analysis")
        return None

    print("  Creating expression pattern analysis...")

    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # 1. 饼图 - 表达模式分布
    pattern_counts = key_genes['pattern'].value_counts()

    # 定义更好的颜色映射，匹配您展示的图
    color_map = {
        'Strong_Increasing': '#FF8C00',  # 深橙色
        'Moderate_Increasing': '#FFA500',  # 橙色
        'Increasing': '#FFD700',  # 金色
        'Strong_Decreasing': '#006400',  # 深绿色
        'Moderate_Decreasing': '#228B22',  # 森林绿
        'Decreasing': '#32CD32',  # 浅绿色
        'Peak': '#B22222',  # 火砖红
        'Weak_Peak': '#CD5C5C',  # 印度红
        'Stable': '#4169E1',  # 皇家蓝
        'Complex': '#1E90FF',  # 道奇蓝
        'Stable/Complex': '#4682B4'  # 钢蓝色
    }

    # 为每个模式分配颜色
    colors = []
    for pattern in pattern_counts.index:
        if pattern in color_map:
            colors.append(color_map[pattern])
        else:
            # 如果模式不在预定义中，使用默认颜色
            colors.append(plt.cm.Set3(len(colors) / 12))

    # 创建饼图
    wedges, texts, autotexts = ax1.pie(
        pattern_counts.values,
        labels=pattern_counts.index,
        autopct='%1.1f%%',
        colors=colors,
        startangle=90,
        textprops={'fontsize': 11}
    )

    # 调整文本
    for text in texts:
        text.set_fontsize(10)
    for autotext in autotexts:
        autotext.set_fontsize(10)
        autotext.set_fontweight('bold')
        autotext.set_color('white')

    ax1.set_title('Distribution of Expression Patterns', fontsize=14, fontweight='bold')

    # 2. 条形图 - 每种模式的平均分数
    pattern_scores = key_genes.groupby('pattern')['score'].mean().sort_values(ascending=False)

    # 只显示前4个最高分数的模式（如您的示例图）
    pattern_scores = pattern_scores.head(4)

    # 使用特定颜色方案
    bar_colors = ['#B22222', '#4682B4', '#FF8C00', '#228B22']  # 红、蓝、橙、绿

    bars = ax2.bar(
        range(len(pattern_scores)),
        pattern_scores.values,
        color=bar_colors[:len(pattern_scores)],
        alpha=0.8,
        edgecolor='black',
        linewidth=1
    )

    ax2.set_xticks(range(len(pattern_scores)))
    ax2.set_xticklabels(pattern_scores.index, rotation=45, ha='right', fontsize=11)
    ax2.set_ylabel('Average Score (Top 3 Genes)', fontsize=12)
    ax2.set_xlabel('Expression Pattern', fontsize=12)
    ax2.set_title('Average Score by Expression Pattern', fontsize=14, fontweight='bold')

    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.,
            height + max(pattern_scores.values) * 0.01,
            f'{height:.2f}',
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold'
        )

    # 添加网格线
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax2.set_axisbelow(True)

    # 设置y轴范围
    ax2.set_ylim(0, max(pattern_scores.values) * 1.15)

    # 调整布局
    plt.tight_layout()

    # 保存为PNG和PDF
    save_path_png = os.path.join(output_dir, 'expression_pattern_analysis.png')
    save_path_pdf = os.path.join(output_dir, 'expression_pattern_analysis.pdf')

    plt.savefig(save_path_png, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(save_path_pdf, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"    Saved expression pattern analysis: {save_path_png}")

    return save_path_png


def create_gene_overlap_analysis(gene_predictions, output_dir):
    """创建基因重叠分析图"""
    if not gene_predictions:
        return

    print("  Creating gene overlap analysis...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # 获取前3个最佳模型
    model_accuracies = {}
    for model, pred in gene_predictions.items():
        acc = np.mean(pred['predicted_labels'] == pred['true_labels'])
        model_accuracies[model] = acc

    top3_models = sorted(model_accuracies.items(), key=lambda x: x[1], reverse=True)[:3]
    top3_names = [m[0] for m in top3_models]

    # 1. 条形图 - 每个模型正确预测的基因数
    correct_counts = []
    for model_name, _ in top3_models:
        pred = gene_predictions[model_name]
        correct_mask = pred['predicted_labels'] == pred['true_labels']
        correct_counts.append(np.sum(correct_mask))

    bars = ax1.bar(range(len(top3_names)), correct_counts,
                   color=['#3B4992', '#E64B35', '#00A087'])
    ax1.set_xticks(range(len(top3_names)))
    ax1.set_xticklabels(top3_names, rotation=0)
    ax1.set_ylabel('Number of Correctly Predicted Genes', fontsize=12)
    ax1.set_title('Correctly Predicted Genes by Model', fontsize=14, fontweight='bold')

    # 添加数值标签
    for bar, count in zip(bars, correct_counts):
        ax1.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                 f'{count}', ha='center', va='bottom', fontsize=11)

    # 2. 热图 - 模型间重叠矩阵
    overlap_matrix = np.zeros((len(top3_names), len(top3_names)))

    for i, model1 in enumerate(top3_names):
        pred1 = gene_predictions[model1]
        correct1 = set(np.array(pred1['test_genes'])[
                           pred1['predicted_labels'] == pred1['true_labels']])

        for j, model2 in enumerate(top3_names):
            pred2 = gene_predictions[model2]
            correct2 = set(np.array(pred2['test_genes'])[
                               pred2['predicted_labels'] == pred2['true_labels']])

            overlap = len(correct1.intersection(correct2))
            overlap_matrix[i, j] = overlap

    im = ax2.imshow(overlap_matrix, cmap='YlOrRd', aspect='auto')
    ax2.set_xticks(range(len(top3_names)))
    ax2.set_yticks(range(len(top3_names)))
    ax2.set_xticklabels(top3_names)
    ax2.set_yticklabels(top3_names)
    ax2.set_title('Gene Overlap Matrix', fontsize=14, fontweight='bold')

    # 添加数值
    for i in range(len(top3_names)):
        for j in range(len(top3_names)):
            text = ax2.text(j, i, f'{int(overlap_matrix[i, j])}',
                            ha="center", va="center", color="black", fontsize=11)

    plt.colorbar(im, ax=ax2, label='Number of Shared Genes')

    plt.suptitle('Gene Prediction Overlap Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()

    # 保存
    save_path_png = os.path.join(output_dir, 'gene_overlap_analysis.png')
    save_path_pdf = os.path.join(output_dir, 'gene_overlap_analysis.pdf')
    plt.savefig(save_path_png, dpi=300, bbox_inches='tight')
    plt.savefig(save_path_pdf, bbox_inches='tight')
    plt.close()

    print(f"    Saved gene overlap analysis")

def create_enhanced_visualizations(group_means, gene_clusters, cluster_centers,
                                   key_genes, sorted_groups, output_dir):
    """创建增强的可视化图表"""
    print("\nCreating enhanced visualizations...")

    if group_means.empty:
        print("  No data for visualization")
        return

    # 1. 聚类中心热图
    # 修复：使用 isinstance 检查 DataFrame，而不是检查 empty 属性
    if cluster_centers is not None and isinstance(cluster_centers, pd.DataFrame) and not cluster_centers.empty:
        fig, ax = plt.subplots(figsize=(12, 8))

        sns.heatmap(cluster_centers,
                    xticklabels=sorted_groups,
                    yticklabels=[f'Cluster {i}' for i in range(len(cluster_centers))],
                    cmap='RdBu_r', center=0,
                    cbar_kws={'label': 'Expression Level'},
                    ax=ax)
        ax.set_title('Cluster Centers Expression Profile', fontsize=14)
        ax.set_xlabel('Time Point / Group', fontsize=12)
        ax.set_ylabel('Cluster', fontsize=12)

        plt.tight_layout()
        save_path = os.path.join(output_dir, 'cluster_centers_heatmap.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved cluster centers heatmap: {save_path}")

    # 2. 关键基因表达趋势图
    # 修复：同样检查 key_genes
    if key_genes is not None and isinstance(key_genes, pd.DataFrame) and not key_genes.empty:
        top_genes = key_genes.head(10)

        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.flatten()

        for idx, (_, gene_info) in enumerate(top_genes.iterrows()):
            if idx >= 10:
                break

            gene_name = gene_info['gene']
            if gene_name in group_means.index:
                ax = axes[idx]
                values = group_means.loc[gene_name].values
                x = range(len(values))

                ax.plot(x, values, 'o-', linewidth=2, markersize=8)
                ax.set_xticks(x)
                ax.set_xticklabels(sorted_groups, rotation=45)
                ax.set_title(f"{gene_name[:15]}\n{gene_info['pattern']}", fontsize=10)
                ax.set_ylabel('Expression', fontsize=9)
                ax.grid(True, alpha=0.3)

        # 隐藏未使用的子图
        for idx in range(len(top_genes), 10):
            axes[idx].set_visible(False)

        plt.suptitle('Top Key Genes Expression Patterns', fontsize=16, fontweight='bold')
        plt.tight_layout()

        save_path = os.path.join(output_dir, 'key_genes_patterns.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved key genes patterns: {save_path}")

    # 3. 基因聚类分布饼图
    # 修复：检查 gene_clusters
    if gene_clusters is not None and isinstance(gene_clusters, pd.DataFrame) and not gene_clusters.empty:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

        # 聚类大小饼图
        cluster_counts = gene_clusters['cluster'].value_counts()
        colors = plt.cm.Set3(np.linspace(0, 1, len(cluster_counts)))

        ax1.pie(cluster_counts.values,
                labels=[f'Cluster {i}\n({count} genes)'
                        for i, count in zip(cluster_counts.index, cluster_counts.values)],
                colors=colors, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Gene Distribution Across Clusters', fontsize=14)

        # 模式分布条形图（如果有key_genes）
        if key_genes is not None and isinstance(key_genes, pd.DataFrame) and not key_genes.empty:
            pattern_counts = key_genes['pattern'].value_counts()
            ax2.bar(range(len(pattern_counts)), pattern_counts.values,
                    color=NATURE_COLORS[:len(pattern_counts)])
            ax2.set_xticks(range(len(pattern_counts)))
            ax2.set_xticklabels(pattern_counts.index, rotation=45, ha='right')
            ax2.set_ylabel('Number of Genes')
            ax2.set_title('Gene Expression Pattern Distribution', fontsize=14)
            ax2.grid(True, alpha=0.3, axis='y')
        else:
            ax2.set_visible(False)

        plt.tight_layout()
        save_path = os.path.join(output_dir, 'cluster_distribution.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved cluster distribution: {save_path}")

    return True


def create_model_comparison_plots(results, predictions_dict=None, output_dir=None):
    """创建模型比较图表（使用SCI期刊级配色）"""
    print("\nCreating model comparison plots...")

    if not results:
        print("  No results for comparison")
        return

    if predictions_dict is None:
        predictions_dict = {}

    # 定义科学期刊级配色方案
    # Nature配色
    nature_palette = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488',
                      '#F39B7F', '#8491B4', '#91D1C2', '#DC0000']
    # Science配色
    science_palette = ['#3B4992', '#EE0000', '#008B45', '#631879',
                       '#008280', '#BB0021', '#5F559B', '#A20056']

    # 1. 模型性能条形图
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    models = list(results.keys())
    accuracies = [results[m]['accuracy'] for m in models]
    precisions = [results[m]['precision'] for m in models]
    recalls = [results[m]['recall'] for m in models]
    f1_scores = [results[m]['f1'] for m in models]

    dl_models = ['RNN', 'LSTM', 'BiLSTM', 'GRU', 'CNN', 'ResNet',
                 'Transformer', 'AttentionRNN', 'TCN', 'WaveNet', 'InceptionTime']

    # 按准确率排序
    sorted_indices = np.argsort(accuracies)[::-1]
    sorted_models = [models[i] for i in sorted_indices]

    # 准确率条形图 - 使用Nature配色方案
    ax1 = axes[0, 0]
    sorted_accs = [accuracies[i] for i in sorted_indices]

    # 为每个模型分配颜色
    colors1 = []
    for i, model in enumerate(sorted_models):
        if model in dl_models:
            # 深度学习模型使用Nature配色
            colors1.append(nature_palette[i % len(nature_palette)])
        else:
            # 机器学习模型使用Science配色
            colors1.append(science_palette[i % len(science_palette)])

    bars1 = ax1.barh(range(len(sorted_models)), sorted_accs, color=colors1, alpha=0.85)
    ax1.set_yticks(range(len(sorted_models)))
    ax1.set_yticklabels(sorted_models, fontsize=10)
    ax1.set_xlabel('Accuracy', fontsize=12)
    ax1.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.set_xlim(0, 1)
    ax1.grid(True, alpha=0.3, axis='x', linestyle='--')

    # 添加数值标签
    for i, (bar, val) in enumerate(zip(bars1, sorted_accs)):
        ax1.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                 f'{val:.3f}', va='center', fontsize=9)

    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=nature_palette[0], alpha=0.85, label='Deep Learning'),
        Patch(facecolor=science_palette[0], alpha=0.85, label='Machine Learning')
    ]
    ax1.legend(handles=legend_elements, loc='lower right', fontsize=10, framealpha=0.95)

    # Precision条形图 - 渐变配色
    ax2 = axes[0, 1]
    sorted_precs = [precisions[i] for i in sorted_indices]
    colors2 = plt.cm.viridis(np.linspace(0.3, 0.9, len(sorted_models)))
    bars2 = ax2.barh(range(len(sorted_models)), sorted_precs, color=colors2, alpha=0.85)
    ax2.set_yticks(range(len(sorted_models)))
    ax2.set_yticklabels(sorted_models, fontsize=10)
    ax2.set_xlabel('Precision', fontsize=12)
    ax2.set_title('Model Precision Comparison', fontsize=14, fontweight='bold')
    ax2.set_xlim(0, 1)
    ax2.grid(True, alpha=0.3, axis='x', linestyle='--')

    # Recall条形图 - 渐变配色
    ax3 = axes[1, 0]
    sorted_recs = [recalls[i] for i in sorted_indices]
    colors3 = plt.cm.plasma(np.linspace(0.3, 0.9, len(sorted_models)))
    bars3 = ax3.barh(range(len(sorted_models)), sorted_recs, color=colors3, alpha=0.85)
    ax3.set_yticks(range(len(sorted_models)))
    ax3.set_yticklabels(sorted_models, fontsize=10)
    ax3.set_xlabel('Recall', fontsize=12)
    ax3.set_title('Model Recall Comparison', fontsize=14, fontweight='bold')
    ax3.set_xlim(0, 1)
    ax3.grid(True, alpha=0.3, axis='x', linestyle='--')

    # F1 Score条形图 - 渐变配色
    ax4 = axes[1, 1]
    sorted_f1s = [f1_scores[i] for i in sorted_indices]
    colors4 = plt.cm.coolwarm(np.linspace(0.2, 0.8, len(sorted_models)))
    bars4 = ax4.barh(range(len(sorted_models)), sorted_f1s, color=colors4, alpha=0.85)
    ax4.set_yticks(range(len(sorted_models)))
    ax4.set_yticklabels(sorted_models, fontsize=10)
    ax4.set_xlabel('F1 Score', fontsize=12)
    ax4.set_title('Model F1 Score Comparison', fontsize=14, fontweight='bold')
    ax4.set_xlim(0, 1)
    ax4.grid(True, alpha=0.3, axis='x', linestyle='--')

    plt.suptitle('Comprehensive Model Performance Comparison',
                 fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    # 保存为PNG和PDF
    save_path_png = os.path.join(output_dir, 'model_comparison_bars.png')
    save_path_pdf = os.path.join(output_dir, 'model_comparison_bars.pdf')
    plt.savefig(save_path_png, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(save_path_pdf, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"  Saved: {save_path_png}")
    print(f"  Saved: {save_path_pdf}")

    # 2. 训练时间vs性能散点图
    if any('training_time' in results[m] for m in models):
        fig, ax = plt.subplots(figsize=(10, 8))

        times = []
        accs = []
        model_names = []

        for model in models:
            if 'training_time' in results[model]:
                times.append(results[model]['training_time'])
                accs.append(results[model]['accuracy'])
                model_names.append(model)

        if times:
            scatter = ax.scatter(times, accs, s=100, c=accs,
                                 cmap='viridis', alpha=0.7, edgecolors='black')

            for i, txt in enumerate(model_names):
                ax.annotate(txt, (times[i], accs[i]),
                            xytext=(5, 5), textcoords='offset points',
                            fontsize=9)

            ax.set_xlabel('Training Time (seconds)', fontsize=12)
            ax.set_ylabel('Accuracy', fontsize=12)
            ax.set_title('Model Efficiency: Time vs Performance',
                         fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)

            plt.colorbar(scatter, label='Accuracy')
            plt.tight_layout()

            # 保存为PNG和PDF
            save_path_png = os.path.join(output_dir, 'time_vs_performance.png')
            save_path_pdf = os.path.join(output_dir, 'time_vs_performance.pdf')
            plt.savefig(save_path_png, dpi=300, bbox_inches='tight')
            plt.savefig(save_path_pdf, bbox_inches='tight')
            plt.close()
            print(f"  Saved time vs performance: {save_path_png}")
            print(f"  Saved time vs performance: {save_path_pdf}")

    return True


# ============= 综合模型性能分析图 =============
def create_comprehensive_model_analysis(model_results, output_dir):
    """创建四合一的综合模型性能分析图"""
    if not model_results:
        print("  No model results for comprehensive analysis")
        return

    print("  Creating comprehensive model analysis plot...")

    # 创建图形
    fig = plt.figure(figsize=(20, 16))

    # 准备数据
    models = list(model_results.keys())
    accuracies = [model_results[m]['accuracy'] for m in models]
    precisions = [model_results[m]['precision'] for m in models]
    recalls = [model_results[m]['recall'] for m in models]
    f1_scores = [model_results[m]['f1'] for m in models]

    # 深度学习模型列表
    dl_models = ['RNN', 'LSTM', 'BiLSTM', 'GRU', 'CNN', 'ResNet',
                 'Transformer', 'AttentionRNN', 'TCN', 'WaveNet', 'InceptionTime']

    # 按准确率排序
    sorted_indices = np.argsort(accuracies)[::-1]
    sorted_models = [models[i] for i in sorted_indices]
    sorted_accs = [accuracies[i] for i in sorted_indices]
    sorted_f1s = [f1_scores[i] for i in sorted_indices]

    # 1. Model Accuracy Comparison (左上)
    ax1 = plt.subplot(2, 2, 1)

    # 创建颜色列表
    colors1 = []
    for model in sorted_models:
        if model in dl_models:
            colors1.append('#E64B35')  # 深度学习 - 红色
        else:
            colors1.append('#4DBBD5')  # 机器学习 - 蓝色

    bars = ax1.barh(range(len(sorted_models)), sorted_accs, color=colors1, alpha=0.8)
    ax1.set_yticks(range(len(sorted_models)))
    ax1.set_yticklabels(sorted_models, fontsize=10)
    ax1.set_xlabel('Accuracy', fontsize=12)
    ax1.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.set_xlim(0, 1)
    ax1.grid(True, alpha=0.3, axis='x')

    # 添加数值标签
    for i, (bar, acc) in enumerate(zip(bars, sorted_accs)):
        ax1.text(acc + 0.01, bar.get_y() + bar.get_height() / 2,
                 f'{acc:.4f}', va='center', fontsize=9)

    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#E64B35', alpha=0.8, label='Deep Learning'),
                       Patch(facecolor='#4DBBD5', alpha=0.8, label='Machine Learning')]
    ax1.legend(handles=legend_elements, loc='lower right', fontsize=10)

    # 2. Model F1 Score Comparison (右上)
    ax2 = plt.subplot(2, 2, 2)

    # 创建颜色渐变
    colors2 = plt.cm.RdBu_r(np.linspace(0.3, 0.9, len(sorted_models)))

    bars2 = ax2.bar(range(len(sorted_models)), sorted_f1s, color=colors2, alpha=0.8)
    ax2.set_xticks(range(len(sorted_models)))
    ax2.set_xticklabels(sorted_models, rotation=45, ha='right', fontsize=10)
    ax2.set_ylabel('F1 Score', fontsize=12)
    ax2.set_title('Model F1 Score Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 1.1)
    ax2.grid(True, alpha=0.3, axis='y')

    # 添加数值标签
    for bar, f1 in zip(bars2, sorted_f1s):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{f1:.3f}', ha='center', va='bottom', fontsize=8)

    # 3. Precision vs Recall Trade-off (左下)
    ax3 = plt.subplot(2, 2, 3)

    # 创建颜色映射
    cmap = plt.cm.get_cmap('tab20')
    colors3 = [cmap(i / len(models)) for i in range(len(models))]

    # 绘制散点
    for i, model in enumerate(models):
        marker = 'o' if model in dl_models else '^'
        ax3.scatter(recalls[i], precisions[i], s=200, c=[colors3[i]],
                    marker=marker, alpha=0.7, edgecolors='black', linewidth=1.5,
                    label=model if i < 5 else "")

    # 添加F1等值线
    f1_lines = [0.2, 0.4, 0.6, 0.8]
    for f1_score in f1_lines:
        x = np.linspace(0.01, 1, 100)
        y = f1_score * x / (2 * x - f1_score)
        mask = (y >= 0) & (y <= 1)
        ax3.plot(x[mask], y[mask], '--', alpha=0.3, color='gray', linewidth=1)
        if f1_score in [0.6, 0.8]:
            ax3.text(x[mask][-1] - 0.05, y[mask][-1], f'F1={f1_score}',
                     fontsize=8, alpha=0.5, rotation=-30)

    ax3.set_xlabel('Recall', fontsize=12)
    ax3.set_ylabel('Precision', fontsize=12)
    ax3.set_title('Precision vs Recall Trade-off', fontsize=14, fontweight='bold')
    ax3.set_xlim(0, 1.05)
    ax3.set_ylim(0, 1.05)
    ax3.grid(True, alpha=0.3)

    # 标注前5个模型
    top5_indices = sorted_indices[:5]
    for idx in top5_indices:
        model = models[idx]
        ax3.annotate(model, (recalls[idx], precisions[idx]),
                     xytext=(5, 5), textcoords='offset points',
                     fontsize=8, alpha=0.7)

    # 4. Top 5 Models Performance Radar (右下)
    ax4 = plt.subplot(2, 2, 4, projection='polar')

    # 选择前5个模型
    top5_models = sorted_models[:5]

    # 雷达图类别
    categories = ['Accuracy', 'Precision', 'Recall', 'F1']
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    # 为每个模型绘制雷达图
    for i, model in enumerate(top5_models):
        idx = models.index(model)
        values = [
            model_results[model]['accuracy'],
            model_results[model]['precision'],
            model_results[model]['recall'],
            model_results[model]['f1']
        ]
        values += values[:1]

        color = NATURE_COLORS[i % len(NATURE_COLORS)]
        ax4.plot(angles, values, 'o-', linewidth=2, label=model, color=color)
        ax4.fill(angles, values, alpha=0.15, color=color)

    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels(categories, fontsize=11)
    ax4.set_ylim(0, 1)
    ax4.set_title('Top 5 Models Performance Radar', fontsize=14, fontweight='bold', pad=20)
    ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    ax4.grid(True, alpha=0.3)

    # 总标题
    plt.suptitle('Comprehensive Model Performance Analysis', fontsize=18, fontweight='bold', y=1.02)

    plt.tight_layout()

    # 保存图形
    save_path_png = os.path.join(output_dir, 'comprehensive_model_analysis.png')
    save_path_pdf = os.path.join(output_dir, 'comprehensive_model_analysis.pdf')

    plt.savefig(save_path_png, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(save_path_pdf, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"    Saved comprehensive analysis: {save_path_png}")
    print(f"    Saved comprehensive analysis: {save_path_pdf}")

    return save_path_png


def create_network_gene_boxplots(intersection_genes, gene_exp, groups_dict,
                                 sorted_groups, output_dir, top_n=12):
    """为network文件夹创建基因箱线图with显著性分析"""
    if not intersection_genes:
        return

    print(f"  Creating network gene boxplots with significance analysis...")

    # 选择要展示的基因
    genes_to_plot = list(intersection_genes)[:top_n]

    if len(genes_to_plot) == 0:
        return

    # 创建图形
    n_cols = 3
    n_rows = (len(genes_to_plot) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()

    for idx, gene in enumerate(genes_to_plot):
        if gene not in gene_exp.index:
            continue

        ax = axes[idx]

        # 准备数据
        plot_data = []
        group_labels = []

        for group in sorted_groups:
            group_samples = groups_dict[group]
            values = []
            for sample in group_samples:
                if sample in gene_exp.columns:
                    values.append(gene_exp.loc[gene, sample])

            if values:
                plot_data.append(values)
                group_labels.append(group)

        if not plot_data:
            continue

        # 创建箱线图
        bp = ax.boxplot(plot_data, labels=group_labels,
                        patch_artist=True, notch=True,
                        showmeans=True,
                        meanprops=dict(marker='D', markeredgecolor='red',
                                       markerfacecolor='red', markersize=5))

        # 着色
        colors = plt.cm.Set3(np.linspace(0, 1, len(plot_data)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # 添加散点
        for i, data in enumerate(plot_data):
            y = data
            x = np.random.normal(i + 1, 0.04, size=len(y))
            ax.scatter(x, y, alpha=0.4, s=15, color='black')

        # 执行统计检验
        if len(plot_data) > 1:
            # ANOVA
            from scipy import stats
            f_stat, p_val = stats.f_oneway(*plot_data)

            # 显示显著性
            if p_val < 0.001:
                sig_text = '***'
            elif p_val < 0.01:
                sig_text = '**'
            elif p_val < 0.05:
                sig_text = '*'
            else:
                sig_text = 'ns'

            # 添加显著性标记
            y_max = max([max(d) for d in plot_data])
            y_range = y_max - min([min(d) for d in plot_data])

            # 如果显著，添加两两比较的线
            if p_val < 0.05 and len(plot_data) > 2:
                # Tukey HSD post-hoc test
                from scipy.stats import tukey_hsd
                all_data = []
                all_labels = []
                for i, data in enumerate(plot_data):
                    all_data.extend(data)
                    all_labels.extend([i] * len(data))

                # 执行Tukey HSD
                res = tukey_hsd(*[np.array(d) for d in plot_data])

                # 绘制显著性线
                y_offset = y_max + y_range * 0.05
                sig_pairs = []

                for i in range(len(plot_data)):
                    for j in range(i + 1, len(plot_data)):
                        if res.pvalue[i][j] < 0.05:
                            # 绘制显著性线
                            ax.plot([i + 1, j + 1], [y_offset, y_offset], 'k-', linewidth=1)
                            ax.plot([i + 1, i + 1], [y_offset - y_range * 0.01, y_offset], 'k-', linewidth=1)
                            ax.plot([j + 1, j + 1], [y_offset - y_range * 0.01, y_offset], 'k-', linewidth=1)

                            # 添加星号
                            if res.pvalue[i][j] < 0.001:
                                star = '***'
                            elif res.pvalue[i][j] < 0.01:
                                star = '**'
                            else:
                                star = '*'

                            ax.text((i + j + 2) / 2, y_offset, star, ha='center', va='bottom', fontsize=8)
                            y_offset += y_range * 0.08

            # 在右上角显示总体p值
            ax.text(0.95, 0.95, f'ANOVA\np={p_val:.3e}\n{sig_text}',
                    transform=ax.transAxes, ha='right', va='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    fontsize=8)

        ax.set_title(f'{gene[:20]}', fontsize=10, fontweight='bold')
        ax.set_ylabel('Expression Level', fontsize=9)
        ax.set_xlabel('Group', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')

        # 旋转x轴标签
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 隐藏未使用的子图
    for idx in range(len(genes_to_plot), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('Network Intersection Genes - Expression Analysis with Significance',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    # 保存
    save_path_png = os.path.join(output_dir, 'network_genes_boxplot_significance.png')
    save_path_pdf = os.path.join(output_dir, 'network_genes_boxplot_significance.pdf')
    plt.savefig(save_path_png, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(save_path_pdf, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"    Saved network genes boxplot with significance")
    return save_path_png
# ============= Venn图和交集分析 =============
def create_intersection_analysis(gene_predictions, output_dir):
    """创建交集分析和Venn图"""
    print("\nCreating intersection analysis...")

    try:
        from matplotlib_venn import venn3, venn2
        VENN_AVAILABLE = True
    except ImportError:
        VENN_AVAILABLE = False
        print("  matplotlib_venn not installed, creating alternative visualization")

    # 获取top 3模型
    model_accuracies = {}
    for model, pred in gene_predictions.items():
        acc = np.mean(pred['predicted_labels'] == pred['true_labels'])
        model_accuracies[model] = acc

    top3_models = sorted(model_accuracies.items(), key=lambda x: x[1], reverse=True)[:3]
    top3_names = [m[0] for m in top3_models]

    # 获取正确预测的基因集合
    correct_genes_sets = []
    for model_name in top3_names:
        pred = gene_predictions[model_name]
        correct_mask = pred['predicted_labels'] == pred['true_labels']
        correct_genes = set(np.array(pred['test_genes'])[correct_mask])
        correct_genes_sets.append(correct_genes)

    if VENN_AVAILABLE and len(top3_names) == 3:
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))

        venn = venn3(correct_genes_sets, set_labels=top3_names, ax=ax)

        # 设置颜色
        if venn.get_patch_by_id('100'):
            venn.get_patch_by_id('100').set_color(NATURE_COLORS[0])
        if venn.get_patch_by_id('010'):
            venn.get_patch_by_id('010').set_color(NATURE_COLORS[1])
        if venn.get_patch_by_id('001'):
            venn.get_patch_by_id('001').set_color(NATURE_COLORS[2])

        ax.set_title('Gene Prediction Overlap - Top 3 Models', fontsize=14, fontweight='bold')

        plt.tight_layout()
        venn_path = os.path.join(output_dir, 'models_venn_diagram.png')
        plt.savefig(venn_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved Venn diagram: {venn_path}")

    # 保存交集基因
    if len(correct_genes_sets) >= 2:
        intersection = set.intersection(*correct_genes_sets)
        union = set.union(*correct_genes_sets)

        intersection_df = pd.DataFrame({'gene': list(intersection)})
        intersection_path = os.path.join(output_dir, 'intersection_genes.csv')
        intersection_df.to_csv(intersection_path, index=False)
        print(f"  Saved {len(intersection)} intersection genes")

        return intersection

    return set()


# ============= 基因网络可视化函数 =============
def create_gene_network(gene_predictions, group_means, output_dir):
    """创建Cytoscape风格的基因互作网络"""
    print("\nCreating gene interaction network...")

    # 创建网络图
    G = nx.Graph()

    # 获取top模型
    model_accuracies = {model: np.mean(pred['predicted_labels'] == pred['true_labels'])
                        for model, pred in gene_predictions.items()}
    top3_models = sorted(model_accuracies.items(), key=lambda x: x[1], reverse=True)[:3]

    # 为每个模型的正确预测基因添加节点
    for model_name, _ in top3_models:
        pred = gene_predictions[model_name]
        correct_mask = pred['predicted_labels'] == pred['true_labels']
        correct_genes = np.array(pred['test_genes'])[correct_mask]

        for gene in correct_genes[:30]:  # 限制为前30个基因
            if not G.has_node(gene):
                G.add_node(gene, models=[model_name])
            else:
                G.nodes[gene]['models'].append(model_name)

    # 基于相关性添加边
    genes = list(G.nodes())
    if len(genes) > 1 and not group_means.empty:
        for i, gene1 in enumerate(genes):
            for gene2 in genes[i + 1:]:
                if gene1 in group_means.index and gene2 in group_means.index:
                    corr = np.corrcoef(group_means.loc[gene1], group_means.loc[gene2])[0, 1]
                    if abs(corr) > 0.7:  # 强相关阈值
                        G.add_edge(gene1, gene2, weight=abs(corr), correlation=corr)

    # 创建可视化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    # 布局1：Spring布局
    pos1 = nx.spring_layout(G, k=2, iterations=50)

    # 节点颜色基于模型数量
    node_colors1 = []
    for node in G.nodes():
        n_models = len(G.nodes[node]['models'])
        if n_models == 3:
            node_colors1.append(NATURE_COLORS[0])  # 红色：在所有3个模型中
        elif n_models == 2:
            node_colors1.append(NATURE_COLORS[1])  # 蓝色：在2个模型中
        else:
            node_colors1.append(NATURE_COLORS[2])  # 绿色：在1个模型中

    # 绘制网络
    nx.draw_networkx_nodes(G, pos1, node_color=node_colors1, node_size=300, ax=ax1)

    # 绘制边（不同宽度基于相关性）
    edges = G.edges()
    weights = [G[u][v]['weight'] * 3 for u, v in edges]
    edge_colors = [NATURE_COLORS[0] if G[u][v]['correlation'] > 0 else NATURE_COLORS[3]
                   for u, v in edges]

    nx.draw_networkx_edges(G, pos1, width=weights, alpha=0.5,
                           edge_color=edge_colors, ax=ax1)
    nx.draw_networkx_labels(G, pos1, font_size=8, ax=ax1)

    ax1.set_title('Gene Interaction Network (Spring Layout)', fontsize=14)
    ax1.axis('off')

    # 布局2：圆形布局
    pos2 = nx.circular_layout(G)

    nx.draw_networkx_nodes(G, pos2, node_color=node_colors1, node_size=300, ax=ax2)
    nx.draw_networkx_edges(G, pos2, width=weights, alpha=0.5,
                           edge_color=edge_colors, ax=ax2)
    nx.draw_networkx_labels(G, pos2, font_size=8, ax=ax2)

    ax2.set_title('Gene Interaction Network (Circular Layout)', fontsize=14)
    ax2.axis('off')

    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=NATURE_COLORS[0], label='Genes in all 3 top models'),
        Patch(facecolor=NATURE_COLORS[1], label='Genes in 2 top models'),
        Patch(facecolor=NATURE_COLORS[2], label='Genes in 1 top model'),
        Patch(facecolor=NATURE_COLORS[0], alpha=0.5, label='Positive correlation'),
        Patch(facecolor=NATURE_COLORS[3], alpha=0.5, label='Negative correlation')
    ]
    ax1.legend(handles=legend_elements, loc='upper left')

    plt.suptitle('Gene Regulatory Network Visualization', fontsize=16)
    plt.tight_layout()

    # 保存网络
    network_path = os.path.join(output_dir, 'gene_network.png')
    plt.savefig(network_path, dpi=300, bbox_inches='tight')
    plt.savefig(network_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()

    # 导出网络用于Cytoscape
    if nx.__version__ >= '2.0':
        nx.write_gml(G, os.path.join(output_dir, 'gene_network.gml'))

    # 保存网络统计
    network_stats = {
        'n_nodes': G.number_of_nodes(),
        'n_edges': G.number_of_edges(),
        'density': nx.density(G),
        'avg_clustering': nx.average_clustering(G) if G.number_of_nodes() > 0 else 0
    }

    with open(os.path.join(output_dir, 'network_statistics.json'), 'w') as f:
        json.dump(network_stats, f, indent=2, cls=NumpyEncoder)

    print(f"  Network created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    return G


def create_group_specific_networks(gene_predictions, group_means, gene_exp, groups_dict,
                                   sorted_groups, output_dir, top_n=5):
    """为每个时间点/组创建特定的基因网络（像您展示的12号组网络）"""
    print("\nCreating group-specific networks...")

    # 获取top N模型的交集基因
    model_accuracies = {model: np.mean(pred['predicted_labels'] == pred['true_labels'])
                        for model, pred in gene_predictions.items()}
    top_models = sorted(model_accuracies.items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_model_names = [m[0] for m in top_models]

    # 获取交集基因
    correctly_predicted_genes = []
    for model_name in top_model_names:
        pred = gene_predictions[model_name]
        correct_mask = pred['predicted_labels'] == pred['true_labels']
        correctly_predicted_genes.append(set(np.array(pred['test_genes'])[correct_mask]))

    intersection_genes = set.intersection(*correctly_predicted_genes) if correctly_predicted_genes else set()

    if not intersection_genes:
        print("  No intersection genes found")
        return

    print(f"  Found {len(intersection_genes)} intersection genes from top {top_n} models")

    # 为每个组创建网络
    for group_idx, group in enumerate(sorted_groups):
        print(f"  Creating network for {group}...")

        # 选择要展示的基因
        genes_to_show = list(intersection_genes)[:20]  # 最多20个基因

        if len(genes_to_show) < 3:
            continue

        # 创建网络
        G = nx.Graph()

        # 添加节点
        group_samples = groups_dict[group]
        node_variances = {}

        for gene in genes_to_show:
            if gene in gene_exp.index:
                # 计算该基因在当前组的方差
                values = []
                for sample in group_samples:
                    if sample in gene_exp.columns:
                        values.append(gene_exp.loc[gene, sample])

                if values:
                    variance = np.var(values)
                    mean_exp = np.mean(values)
                    G.add_node(gene, variance=variance, mean_expression=mean_exp)
                    node_variances[gene] = variance

        # 添加边（基于相关性）
        nodes = list(G.nodes())
        for i, gene1 in enumerate(nodes):
            for gene2 in nodes[i + 1:]:
                if gene1 in gene_exp.index and gene2 in gene_exp.index:
                    # 计算在该组内的相关性
                    values1 = []
                    values2 = []
                    for sample in group_samples:
                        if sample in gene_exp.columns:
                            values1.append(gene_exp.loc[gene1, sample])
                            values2.append(gene_exp.loc[gene2, sample])

                    if len(values1) > 2:
                        corr = np.corrcoef(values1, values2)[0, 1]
                        if abs(corr) > 0.5:  # 相关性阈值
                            G.add_edge(gene1, gene2, weight=abs(corr), correlation=corr)

        if G.number_of_nodes() < 3 or G.number_of_edges() < 1:
            continue

        # 创建可视化（类似您展示的样式）
        fig, ax = plt.subplots(figsize=(14, 10), facecolor='white')

        # 使用spring布局
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

        # 节点大小和颜色基于方差
        if node_variances:
            max_var = max(node_variances.values())
            min_var = min(node_variances.values())

            # 节点大小
            node_sizes = [300 + 1000 * ((node_variances.get(node, min_var) - min_var) /
                                        (max_var - min_var + 1e-10)) for node in G.nodes()]

            # 节点颜色（使用渐变色）
            node_colors = [node_variances.get(node, 0) for node in G.nodes()]
        else:
            node_sizes = [500] * len(G.nodes())
            node_colors = 'lightblue'

        # 绘制节点
        nodes = nx.draw_networkx_nodes(G, pos,
                                       node_color=node_colors,
                                       node_size=node_sizes,
                                       cmap='coolwarm',
                                       alpha=0.8,
                                       ax=ax,
                                       edgecolors='black',
                                       linewidths=2)

        # 绘制边
        edges = G.edges()
        if edges:
            # 正相关用红色，负相关用蓝色
            for (u, v) in edges:
                corr = G[u][v]['correlation']
                width = abs(corr) * 5
                color = '#E64B35' if corr > 0 else '#4DBBD5'
                nx.draw_networkx_edges(G, pos, [(u, v)],
                                       width=width,
                                       alpha=0.6,
                                       edge_color=color,
                                       ax=ax)

        # 添加标签
        nx.draw_networkx_labels(G, pos, font_size=10, ax=ax, font_weight='bold')

        # 添加标题
        ax.set_title(f'Gene Network for {group} (Top {top_n} Models Intersection)',
                     fontsize=16, fontweight='bold', pad=20)
        ax.axis('off')

        # 添加颜色条
        if isinstance(node_colors, list):
            sm = plt.cm.ScalarMappable(cmap='coolwarm',
                                       norm=plt.Normalize(vmin=min(node_colors),
                                                          vmax=max(node_colors)))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, label='Gene Variance', fraction=0.046, pad=0.04)

        # 添加网络统计信息
        info_text = f"Nodes: {G.number_of_nodes()}\n"
        info_text += f"Edges: {G.number_of_edges()}\n"
        if G.number_of_nodes() > 0:
            info_text += f"Density: {nx.density(G):.3f}"

        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                fontsize=11, va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        # 保存
        network_path = os.path.join(output_dir, f'network_{group}.png')
        plt.savefig(network_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(network_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
        plt.close()

        # 导出为Cytoscape格式
        if nx.__version__ >= '2.0':
            nx.write_gml(G, os.path.join(output_dir, f'network_{group}.gml'))

        print(f"    Saved network for {group}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    print("  Group-specific networks completed")
# ============= 神经网络对应的箱线图差异分析 =============
def create_nn_boxplot_differential(model_name, gene_predictions, gene_exp,
                                   groups_dict, sorted_groups, output_dir):
    """为每个神经网络创建箱线图差异分析"""
    print(f"\nCreating boxplot analysis for {model_name}...")

    if model_name not in gene_predictions:
        return

    pred = gene_predictions[model_name]

    # 获取正确预测的基因
    correct_mask = pred['predicted_labels'] == pred['true_labels']
    correct_genes = np.array(pred['test_genes'])[correct_mask][:12]  # 前12个

    if len(correct_genes) == 0:
        print(f"  No correctly predicted genes for {model_name}")
        return

    # 创建图形
    fig = plt.figure(figsize=(20, 12))

    # 颜色方案
    group_colors = sns.color_palette("husl", len(sorted_groups))

    for gene_idx, gene in enumerate(correct_genes):
        if gene not in gene_exp.index:
            continue

        ax = plt.subplot(3, 4, gene_idx + 1)

        # 准备数据
        plot_data = []
        positions = []
        colors = []

        for group_idx, group in enumerate(sorted_groups):
            group_samples = groups_dict[group]
            values = []
            for sample in group_samples:
                if sample in gene_exp.columns:
                    values.append(gene_exp.loc[gene, sample])

            if values:
                plot_data.append(values)
                positions.append(group_idx)
                colors.append(group_colors[group_idx])

        if not plot_data:
            continue

        # 创建箱线图
        bp = ax.boxplot(plot_data, positions=positions, widths=0.6,
                        patch_artist=True, notch=True, showmeans=True,
                        meanprops=dict(marker='D', markeredgecolor='red',
                                       markerfacecolor='red', markersize=6),
                        medianprops=dict(linewidth=2, color='darkred'),
                        flierprops=dict(marker='o', markersize=4, alpha=0.5))

        # 着色
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # 添加散点
        for pos, data, color in zip(positions, plot_data, colors):
            y = data
            x = np.random.normal(pos, 0.08, size=len(y))
            ax.scatter(x, y, alpha=0.4, s=20, color=color,
                       edgecolors='black', linewidth=0.5)

        # 统计检验
        if len(plot_data) > 1:
            # ANOVA
            f_stat, p_val = stats.f_oneway(*plot_data)

            # 显示显著性
            if p_val < 0.001:
                sig_text = '***'
            elif p_val < 0.01:
                sig_text = '**'
            elif p_val < 0.05:
                sig_text = '*'
            else:
                sig_text = 'ns'

            ax.text(0.95, 0.95, f'{sig_text}\np={p_val:.3f}',
                    transform=ax.transAxes, ha='right', va='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    fontsize=9)

        # 设置轴
        ax.set_xticks(positions)
        ax.set_xticklabels(sorted_groups, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Expression Level', fontsize=10)
        ax.set_title(f'{gene[:15]}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax.set_axisbelow(True)

    # 添加总标题
    plt.suptitle(f'{model_name} - Gene Expression Boxplot Analysis',
                 fontsize=16, fontweight='bold')

    plt.tight_layout()

    # 保存图像
    save_path = os.path.join(output_dir, f'{model_name}_boxplot_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"  Saved boxplot analysis: {save_path}")

    return save_path


# ============= 神经网络对应的热图分析 =============
def create_nn_heatmap_analysis(model_name, gene_predictions, gene_exp,
                               groups_dict, sorted_groups, output_dir):
    """为每个神经网络创建热图分析"""
    print(f"\nCreating heatmap analysis for {model_name}...")

    if model_name not in gene_predictions:
        return

    pred = gene_predictions[model_name]

    # 获取正确预测的基因
    correct_mask = pred['predicted_labels'] == pred['true_labels']
    correct_genes = np.array(pred['test_genes'])[correct_mask][:30]  # 前30个

    if len(correct_genes) == 0:
        print(f"  No correctly predicted genes for {model_name}")
        return

    # 准备热图数据
    heatmap_data = []
    gene_labels = []

    for gene in correct_genes:
        if gene in gene_exp.index:
            gene_means = []
            for group in sorted_groups:
                group_samples = groups_dict[group]
                values = []
                for sample in group_samples:
                    if sample in gene_exp.columns:
                        values.append(gene_exp.loc[gene, sample])

                if values:
                    gene_means.append(np.mean(values))
                else:
                    gene_means.append(0)

            heatmap_data.append(gene_means)
            gene_labels.append(gene[:15])

    if not heatmap_data:
        return

    heatmap_array = np.array(heatmap_data)

    # 创建图形
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10))

    # 1. 原始值热图
    sns.heatmap(heatmap_array,
                xticklabels=sorted_groups,
                yticklabels=gene_labels,
                cmap='YlOrRd',
                cbar_kws={'label': 'Expression Level'},
                ax=ax1,
                linewidths=0.5,
                linecolor='gray')
    ax1.set_xlabel('Time Point / Group', fontsize=11)
    ax1.set_ylabel('Gene', fontsize=11)
    ax1.set_title(f'{model_name} - Raw Expression', fontsize=12, fontweight='bold')

    # 2. Z-score标准化热图
    heatmap_zscore = (heatmap_array - heatmap_array.mean(axis=1, keepdims=True)) / (
                heatmap_array.std(axis=1, keepdims=True) + 1e-8)

    sns.heatmap(heatmap_zscore,
                xticklabels=sorted_groups,
                yticklabels=gene_labels,
                cmap='RdBu_r',
                center=0,
                cbar_kws={'label': 'Z-score'},
                ax=ax2,
                linewidths=0.5,
                linecolor='gray',
                vmin=-3, vmax=3)
    ax2.set_xlabel('Time Point / Group', fontsize=11)
    ax2.set_ylabel('Gene', fontsize=11)
    ax2.set_title(f'{model_name} - Z-score Normalized', fontsize=12, fontweight='bold')

    # 3. 聚类热图
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import pdist

    # 计算距离和聚类
    row_linkage = linkage(pdist(heatmap_zscore, metric='euclidean'), method='average')

    # 重排序
    from scipy.cluster.hierarchy import dendrogram
    dendro = dendrogram(row_linkage, no_plot=True)
    row_order = dendro['leaves']

    heatmap_clustered = heatmap_zscore[row_order, :]
    gene_labels_clustered = [gene_labels[i] for i in row_order]

    sns.heatmap(heatmap_clustered,
                xticklabels=sorted_groups,
                yticklabels=gene_labels_clustered,
                cmap='RdBu_r',
                center=0,
                cbar_kws={'label': 'Z-score'},
                ax=ax3,
                linewidths=0.5,
                linecolor='gray',
                vmin=-3, vmax=3)
    ax3.set_xlabel('Time Point / Group', fontsize=11)
    ax3.set_ylabel('Gene (Clustered)', fontsize=11)
    ax3.set_title(f'{model_name} - Hierarchically Clustered', fontsize=12, fontweight='bold')

    # 添加总标题
    plt.suptitle(f'{model_name} Neural Network - Expression Heatmap Analysis',
                 fontsize=16, fontweight='bold')

    plt.tight_layout()

    # 保存图像
    save_path = os.path.join(output_dir, f'{model_name}_heatmap_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"  Saved heatmap analysis: {save_path}")

    return save_path


# ============= 过拟合分析图 =============
def create_overfitting_analysis_plot(model_name, X_train, y_train, train_pred,
                                     X_test, y_test, test_pred,
                                     training_history, output_dir):
    """创建过拟合分析图，真实值为线条，预测值为散点"""
    print(f"\nCreating overfitting analysis for {model_name}...")

    fig = plt.figure(figsize=(18, 12))

    # 创建子图
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. 训练集：真实值线条 vs 预测值散点
    ax1 = fig.add_subplot(gs[0, :2])
    x_train_idx = np.arange(len(y_train))

    # 真实值线条
    ax1.plot(x_train_idx, y_train, 'b-', linewidth=2, alpha=0.7,
             label='True Values', zorder=1)

    # 预测值散点
    colors = ['red' if pred != true else 'green'
              for pred, true in zip(train_pred, y_train)]
    ax1.scatter(x_train_idx, train_pred, c=colors, alpha=0.6, s=30,
                label='Predictions', zorder=2, edgecolors='black', linewidth=0.5)

    ax1.set_xlabel('Sample Index', fontsize=11)
    ax1.set_ylabel('Class Label', fontsize=11)
    ax1.set_title(f'Training Set (n={len(y_train)})', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3, linestyle='--')

    # 计算准确率
    train_acc = accuracy_score(y_train, train_pred)
    ax1.text(0.02, 0.98, f'Accuracy: {train_acc:.3f}',
             transform=ax1.transAxes,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
             fontsize=11, va='top', fontweight='bold')

    # 2. 测试集：真实值线条 vs 预测值散点
    ax2 = fig.add_subplot(gs[1, :2])
    x_test_idx = np.arange(len(y_test))

    # 真实值线条
    ax2.plot(x_test_idx, y_test, 'b-', linewidth=2, alpha=0.7,
             label='True Values', zorder=1)

    # 预测值散点
    colors = ['red' if pred != true else 'green'
              for pred, true in zip(test_pred, y_test)]
    ax2.scatter(x_test_idx, test_pred, c=colors, alpha=0.6, s=30,
                label='Predictions', zorder=2, edgecolors='black', linewidth=0.5)

    ax2.set_xlabel('Sample Index', fontsize=11)
    ax2.set_ylabel('Class Label', fontsize=11)
    ax2.set_title(f'Test Set (n={len(y_test)})', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3, linestyle='--')

    # 计算准确率
    test_acc = accuracy_score(y_test, test_pred)
    ax2.text(0.02, 0.98, f'Accuracy: {test_acc:.3f}',
             transform=ax2.transAxes,
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
             fontsize=11, va='top', fontweight='bold')

    # 3. 训练历史
    ax3 = fig.add_subplot(gs[0, 2])
    if training_history and 'train_loss' in training_history:
        epochs = range(len(training_history['train_loss']))
        ax3.plot(epochs, training_history['train_loss'], 'b-',
                 label='Training Loss', linewidth=2)

        if 'val_loss' in training_history:
            val_epochs = range(0, len(training_history['val_loss']) * 10, 10)[:len(training_history['val_loss'])]
            ax3.plot(val_epochs, training_history['val_loss'], 'r--',
                     label='Validation Loss', linewidth=2, marker='o', markersize=5)

        ax3.set_xlabel('Epochs', fontsize=11)
        ax3.set_ylabel('Loss', fontsize=11)
        ax3.set_title('Training History', fontsize=13, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    # 4. 准确率历史
    ax4 = fig.add_subplot(gs[1, 2])
    if training_history and 'train_acc' in training_history:
        epochs = range(len(training_history['train_acc']))
        ax4.plot(epochs, training_history['train_acc'], 'b-',
                 label='Training Acc', linewidth=2)

        if 'val_acc' in training_history:
            val_epochs = range(0, len(training_history['val_acc']) * 10, 10)[:len(training_history['val_acc'])]
            ax4.plot(val_epochs, training_history['val_acc'], 'g--',
                     label='Validation Acc', linewidth=2, marker='o', markersize=5)

        ax4.set_xlabel('Epochs', fontsize=11)
        ax4.set_ylabel('Accuracy', fontsize=11)
        ax4.set_title('Accuracy History', fontsize=13, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 1.05)

    # 5. 错误分布
    ax5 = fig.add_subplot(gs[2, :2])

    # 计算错误
    train_errors = train_pred != y_train
    test_errors = test_pred != y_test

    # 绘制错误率随时间的变化
    window_size = max(10, len(y_train) // 20)

    train_error_rate = []
    for i in range(0, len(train_errors) - window_size + 1, window_size // 2):
        window = train_errors[i:i + window_size]
        train_error_rate.append(np.mean(window))

    test_error_rate = []
    for i in range(0, len(test_errors) - window_size + 1, window_size // 2):
        window = test_errors[i:i + window_size]
        test_error_rate.append(np.mean(window))

    x_train_windows = np.arange(len(train_error_rate))
    x_test_windows = np.arange(len(test_error_rate))

    ax5.plot(x_train_windows, train_error_rate, 'b-', linewidth=2,
             label=f'Training Error Rate', marker='o', markersize=5)
    ax5.plot(x_test_windows, test_error_rate, 'r-', linewidth=2,
             label=f'Test Error Rate', marker='s', markersize=5)

    ax5.set_xlabel('Window Index', fontsize=11)
    ax5.set_ylabel('Error Rate', fontsize=11)
    ax5.set_title('Error Rate Distribution', fontsize=13, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim(0, 1)

    # 6. 过拟合指标
    ax6 = fig.add_subplot(gs[2, 2])

    overfitting_score = train_acc - test_acc

    # 创建仪表盘风格的可视化
    theta = np.linspace(0, np.pi, 100)
    r = 1

    # 背景颜色区域
    ax6.fill_between(theta[:33], 0, r, color='green', alpha=0.3)  # 无过拟合
    ax6.fill_between(theta[33:66], 0, r, color='yellow', alpha=0.3)  # 轻度过拟合
    ax6.fill_between(theta[66:], 0, r, color='red', alpha=0.3)  # 严重过拟合

    # 指针
    if overfitting_score < 0.05:
        pointer_angle = np.pi * 0.8  # 好
        status = "No Overfitting"
        color = 'green'
    elif overfitting_score < 0.1:
        pointer_angle = np.pi * 0.5  # 中等
        status = "Mild Overfitting"
        color = 'orange'
    else:
        pointer_angle = np.pi * 0.2  # 差
        status = "Severe Overfitting"
        color = 'red'

    ax6.arrow(0, 0, r * 0.9 * np.cos(pointer_angle),
              r * 0.9 * np.sin(pointer_angle),
              head_width=0.1, head_length=0.1, fc=color, ec=color, linewidth=3)

    ax6.set_xlim(-1.2, 1.2)
    ax6.set_ylim(-0.1, 1.2)
    ax6.set_aspect('equal')
    ax6.axis('off')

    ax6.text(0, -0.05, status, ha='center', fontsize=12, fontweight='bold', color=color)
    ax6.text(0, -0.15, f'Δ = {overfitting_score:.3f}', ha='center', fontsize=11)
    ax6.set_title('Overfitting Indicator', fontsize=13, fontweight='bold')

    # 添加总标题
    plt.suptitle(f'{model_name} - Overfitting Analysis', fontsize=16, fontweight='bold')

    # 保存图像
    save_path = os.path.join(output_dir, f'{model_name}_overfitting_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"  Saved overfitting analysis: {save_path}")

    return save_path


# ============= 模型特定结果保存函数 =============
def plot_model_specific_results(model_name, y_true, y_pred, training_history,
                                predicted_genes, output_dir):
    """为每个模型生成性能可视化"""
    model_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    print(f"      Model directory: {model_dir}")
    # 创建图形
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # 1. 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                cbar_kws={'label': 'Count'})
    ax1.set_xlabel('Predicted Label')
    ax1.set_ylabel('True Label')
    ax1.set_title(f'{model_name} - Confusion Matrix')
    labels = ['Increasing', 'Decreasing', 'Peak', 'Stable']
    ax1.set_xticklabels(labels[:cm.shape[1]], rotation=45)
    ax1.set_yticklabels(labels[:cm.shape[0]], rotation=0)

    # 2. 学习曲线
    if training_history and 'train_loss' in training_history:
        epochs = range(len(training_history['train_loss']))
        ax2.plot(epochs, training_history['train_loss'], 'b-', label='Training Loss', linewidth=2)

        if 'val_loss' in training_history:
            val_epochs = range(0, len(training_history['val_loss']) * 10, 10)[:len(training_history['val_loss'])]
            ax2.plot(val_epochs, training_history['val_loss'], 'r--',
                     label='Validation Loss', linewidth=2, marker='o', markersize=5)

        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Loss')
        ax2.set_title(f'{model_name} - Training History')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    # 3. 准确率历史
    if training_history and 'val_acc' in training_history:
        val_epochs = range(0, len(training_history['val_acc']) * 10, 10)[:len(training_history['val_acc'])]
        ax3.plot(val_epochs, training_history['val_acc'], 'g-',
                 marker='o', label='Validation Accuracy', linewidth=2)
        ax3.set_xlabel('Epochs')
        ax3.set_ylabel('Accuracy')
        ax3.set_title(f'{model_name} - Validation Accuracy')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1.1)

    # 4. 预测vs真实散点图
    x = range(len(y_true))
    ax4.scatter(x, y_true, alpha=0.5, label='True Values', color='blue', s=30)
    ax4.scatter(x, y_pred, alpha=0.5, label='Predicted Values', color='red', s=30)
    ax4.set_xlabel('Sample Index')
    ax4.set_ylabel('Class Label')
    ax4.set_title(f'{model_name} - Predictions vs True Values')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_yticks([0, 1, 2, 3])
    ax4.set_yticklabels(['Increasing', 'Decreasing', 'Peak', 'Stable'])

    plt.suptitle(f'{model_name} Model Performance Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()

    # 保存
    png_path = os.path.join(model_dir, f'{model_name}_performance.png')
    pdf_path = os.path.join(model_dir, f'{model_name}_performance.pdf')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()

    print(f"    Saved model performance plot: {png_path}")
    return png_path


def save_enhanced_gene_predictions(model_name, gene_predictions, original_gene_exp,
                                   group_means, output_dir):
    """保存预测基因及其表达矩阵"""
    model_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)

    pred = gene_predictions[model_name]

    # 创建增强的数据框
    enhanced_df = pd.DataFrame({
        'gene': pred['test_genes'],
        'true_label': pred['true_labels'],
        'predicted_label': pred['predicted_labels'],
        'correct': pred['true_labels'] == pred['predicted_labels']
    })

    # 映射标签到模式名称
    pattern_map = {0: 'Increasing', 1: 'Decreasing', 2: 'Peak', 3: 'Stable/Complex'}
    enhanced_df['true_pattern'] = enhanced_df['true_label'].map(pattern_map)
    enhanced_df['predicted_pattern'] = enhanced_df['predicted_label'].map(pattern_map)

    # 添加准确率
    accuracy = np.mean(enhanced_df['correct'])
    enhanced_df['model_accuracy'] = accuracy

    # 如果有原始表达数据，添加表达值
    if original_gene_exp is not None:
        expression_data = []
        for gene in enhanced_df['gene']:
            if gene in original_gene_exp.index:
                # 获取该基因的所有表达值
                expr_dict = {'gene': gene}
                for col in original_gene_exp.columns[:10]:  # 限制列数避免文件过大
                    expr_dict[f'expr_{col}'] = original_gene_exp.loc[gene, col]
                expression_data.append(expr_dict)
            else:
                expression_data.append({'gene': gene})

        if expression_data:
            expr_df = pd.DataFrame(expression_data)
            # 合并数据
            enhanced_df = pd.merge(enhanced_df, expr_df, on='gene', how='left')

    # 如果有组均值，添加
    if group_means is not None and not group_means.empty:
        mean_data = []
        for gene in enhanced_df['gene']:
            if gene in group_means.index:
                mean_dict = {'gene': gene}
                for col in group_means.columns:
                    mean_dict[f'mean_{col}'] = group_means.loc[gene, col]
                mean_data.append(mean_dict)
            else:
                mean_data.append({'gene': gene})

        if mean_data:
            mean_df = pd.DataFrame(mean_data)
            enhanced_df = pd.merge(enhanced_df, mean_df, on='gene', how='left')

    # 保存到CSV
    output_path = os.path.join(model_dir, f'{model_name}_predictions_with_expression.csv')
    enhanced_df.to_csv(output_path, index=False)

    # 保存摘要统计
    summary = {
        'model': model_name,
        'total_genes': len(enhanced_df),
        'correct_predictions': enhanced_df['correct'].sum(),
        'accuracy': accuracy,
        'pattern_distribution': enhanced_df['predicted_pattern'].value_counts().to_dict(),
        'confusion_matrix': pd.crosstab(enhanced_df['true_pattern'],
                                        enhanced_df['predicted_pattern']).to_dict()
    }

    summary_path = os.path.join(model_dir, f'{model_name}_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)

    print(f"    Saved enhanced predictions: {output_path}")
    return output_path

# ============= 模型训练函数 =============
def train_pytorch_model_with_tracking(model, X_train, y_train, X_val, y_val,
                                      epochs=100, device='cpu', learning_rate=0.001,
                                      model_name="", patience=20, output_dir=None):
    """训练PyTorch模型并跟踪详细信息"""
    model = model.to(device)

    # 转换数据为张量
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.LongTensor(y_val).to(device)

    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=0.5, patience=10)
    criterion = nn.CrossEntropyLoss()

    # 训练历史
    training_history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rates': []
    }

    best_val_acc = 0
    best_model_state = None
    patience_counter = 0

    print(f"    Training {model_name}...")

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 计算训练准确率
        _, predicted = torch.max(outputs, 1)
        train_acc = accuracy_score(y_train_tensor.cpu(), predicted.cpu())

        training_history['train_loss'].append(loss.item())
        training_history['train_acc'].append(train_acc)
        training_history['learning_rates'].append(optimizer.param_groups[0]['lr'])

        # 验证阶段（每10个epoch）
        if epoch % 10 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor)
                _, val_predicted = torch.max(val_outputs, 1)
                val_acc = accuracy_score(y_val_tensor.cpu(), val_predicted.cpu())

                training_history['val_loss'].append(val_loss.item())
                training_history['val_acc'].append(val_acc)

                # 学习率调整
                scheduler.step(val_loss)

                # 早停检查
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_state = model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1

                if epoch % 50 == 0:
                    print(f"      Epoch {epoch}: Train Loss={loss.item():.4f}, "
                          f"Val Acc={val_acc:.4f}, LR={optimizer.param_groups[0]['lr']:.6f}")

                if patience_counter >= patience:
                    print(f"      Early stopping at epoch {epoch}")
                    break

    # 恢复最佳模型
    if best_model_state:
        model.load_state_dict(best_model_state)

    # 获取最终预测
    model.eval()
    with torch.no_grad():
        train_outputs = model(X_train_tensor)
        _, train_predictions = torch.max(train_outputs, 1)
        train_predictions = train_predictions.cpu().numpy()

        val_outputs = model(X_val_tensor)
        _, val_predictions = torch.max(val_outputs, 1)
        val_predictions = val_predictions.cpu().numpy()

    return model, best_val_acc, training_history, train_predictions, val_predictions


# ============= 综合模型评估=============
def evaluate_all_models_comprehensive(X, y, gene_names, args, subdirs,
                                      gene_exp, groups_dict, sorted_groups,
                                      group_means, original_gene_exp=None,
                                      integrated_results=None,
                                      gene_clusters=None, cluster_centers=None,
                                      key_genes=None):
    """综合评估所有模型"""

    # 确保参数不是None
    if gene_clusters is None:
        gene_clusters = pd.DataFrame()
    if cluster_centers is None:
        cluster_centers = pd.DataFrame()
    if key_genes is None:
        key_genes = pd.DataFrame()

    if X is None or len(X) < 10:
        print("Insufficient data for model evaluation")
        return {}, {}, {}

    print("\n" + "=" * 60)
    print("Comprehensive Model Training and Evaluation")
    print("=" * 60)

    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 数据划分
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_scaled, y, test_size=args.test_size, random_state=RANDOM_SEED, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=args.val_size / (1 - args.test_size),
        random_state=RANDOM_SEED, stratify=y_temp
    )

    # 获取测试集基因名
    test_indices = train_test_split(
        range(len(X_scaled)), test_size=args.test_size,
        random_state=RANDOM_SEED, stratify=y
    )[1]
    test_genes = [gene_names[i] for i in test_indices]

    print(f"Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    print(f"Class distribution: {np.bincount(y)}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    results = {}
    gene_predictions = {}
    n_features = X.shape[1]

    # ========== 深度学习模型 ==========
    print("\n" + "=" * 40)
    print("Training Deep Learning Models")
    print("=" * 40)

    dl_models = {
        'RNN': RNN_Model(input_size=1, hidden_size=args.hidden_size,
                         num_layers=args.num_layers, dropout=args.dropout),
        'LSTM': LSTM_Model(input_size=1, hidden_size=args.hidden_size,
                           num_layers=args.num_layers, dropout=args.dropout),
        'BiLSTM': BiLSTM_Model(input_size=1, hidden_size=args.hidden_size,
                               num_layers=args.num_layers, dropout=args.dropout),
        'GRU': GRU_Model(input_size=1, hidden_size=args.hidden_size,
                         num_layers=args.num_layers, dropout=args.dropout),
        'CNN': CNN_Model(input_size=n_features, dropout=args.dropout),
        'ResNet': ResNet_Model(input_size=n_features, dropout=args.dropout),
        'Transformer': TransformerModel(input_size=1, dropout=args.dropout),
        'AttentionRNN': AttentionRNN(input_size=1, hidden_size=args.hidden_size,
                                     dropout=args.dropout),
        'TCN': TCN_Model(input_size=n_features, dropout=args.dropout),
        'WaveNet': WaveNet_Model(input_size=n_features, dropout=args.dropout),
        'InceptionTime': InceptionTime(input_size=n_features, dropout=args.dropout)
    }

    # ========== 深度学习模型 ==========
    for name, model in dl_models.items():
        try:
            print(f"\n  Processing {name}...")
            start_time = time.time()

            # 1. 绘制神经网络架构（圆形节点）
            nn_arch_path = visualize_neural_network_circular(
                name, model, subdirs['nn_architectures']
            )

            # 2. 训练模型
            trained_model, val_acc, training_history, train_pred, val_pred = \
                train_pytorch_model_with_tracking(
                    model, X_train, y_train, X_val, y_val,
                    epochs=args.epochs, device=device,
                    learning_rate=args.learning_rate,
                    model_name=name,
                    patience=args.patience,
                    output_dir=subdirs['models']
                )

            # 3. 测试集预测
            model.eval()
            with torch.no_grad():
                X_test_tensor = torch.FloatTensor(X_test).to(device)
                outputs = trained_model(X_test_tensor)
                _, test_predictions = torch.max(outputs, 1)
                test_predictions = test_predictions.cpu().numpy()

            # 4. 计算指标
            acc = accuracy_score(y_test, test_predictions)
            prec = precision_score(y_test, test_predictions, average='weighted', zero_division=0)
            rec = recall_score(y_test, test_predictions, average='weighted', zero_division=0)
            f1 = f1_score(y_test, test_predictions, average='weighted', zero_division=0)

            results[name] = {
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1': f1,
                'predictions': test_predictions,
                'training_time': time.time() - start_time
            }

            gene_predictions[name] = {
                'test_genes': test_genes,
                'true_labels': y_test,
                'predicted_labels': test_predictions
            }

            print(f"    ✓ Accuracy: {acc:.4f}, F1: {f1:.4f}, Time: {results[name]['training_time']:.2f}s")

            # *** 生成模型性能图 ***
            plot_model_specific_results(
                name, y_test, test_predictions,
                training_history, test_genes,
                subdirs['models']
            )

            # *** 保存增强的预测结果 ***
            save_enhanced_gene_predictions(
                name, gene_predictions, original_gene_exp,
                group_means, subdirs['models']
            )

            # 5. 创建过拟合分析图
            overfitting_path = create_overfitting_analysis_plot(
                name, X_train, y_train, train_pred,
                X_test, y_test, test_predictions,
                training_history, subdirs['overfitting']
            )

            # 6. 创建箱线图分析
            boxplot_path = create_nn_boxplot_differential(
                name, gene_predictions, gene_exp,
                groups_dict, sorted_groups, subdirs['boxplots']
            )

            # 7. 创建热图分析
            heatmap_path = create_nn_heatmap_analysis(
                name, gene_predictions, gene_exp,
                groups_dict, sorted_groups, subdirs['heatmaps']
            )

            # 8. 导出模型
            if args.export_models:
                export_dir = os.path.join(subdirs['exported_models'], name)
                os.makedirs(export_dir, exist_ok=True)

                torch.save({
                    'model_state_dict': model.state_dict(),
                    'model_class': model.__class__.__name__,
                    'accuracy': acc,
                    'training_history': training_history,
                    'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }, os.path.join(export_dir, f'{name}_model.pth'))

                print(f"    ✓ Model exported")

        except Exception as e:
            print(f"    ✗ Failed: {e}")
            import traceback
            traceback.print_exc()  # 打印详细错误信息
            results[name] = {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0}
            # 5. 创建过拟合分析图
            overfitting_path = create_overfitting_analysis_plot(
                name, X_train, y_train, train_pred,
                X_test, y_test, test_predictions,
                training_history, subdirs['overfitting']
            )

            # 6. 创建箱线图分析
            boxplot_path = create_nn_boxplot_differential(
                name, gene_predictions, gene_exp,
                groups_dict, sorted_groups, subdirs['boxplots']
            )

            # 7. 创建热图分析
            heatmap_path = create_nn_heatmap_analysis(
                name, gene_predictions, gene_exp,
                groups_dict, sorted_groups, subdirs['heatmaps']
            )

            # 8. 导出模型
            if args.export_models:
                export_dir = os.path.join(subdirs['exported_models'], name)
                os.makedirs(export_dir, exist_ok=True)

                torch.save({
                    'model_state_dict': model.state_dict(),
                    'model_class': model.__class__.__name__,
                    'accuracy': acc,
                    'training_history': training_history,
                    'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }, os.path.join(export_dir, f'{name}_model.pth'))

                print(f"    ✓ Model exported")

        except Exception as e:
            print(f"    ✗ Failed: {e}")
            results[name] = {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0}

    # ========== 机器学习模型 ==========
    print("\n" + "=" * 40)
    print("Training Machine Learning Models")
    print("=" * 40)

    ml_models = {
        'RandomForest': RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=RANDOM_SEED, n_jobs=args.n_jobs
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=100, random_state=RANDOM_SEED
        ),
        'AdaBoost': AdaBoostClassifier(
            n_estimators=100, random_state=RANDOM_SEED
        ),
        'ExtraTrees': ExtraTreesClassifier(
            n_estimators=100, random_state=RANDOM_SEED, n_jobs=args.n_jobs
        ),
        'SVM': SVC(
            kernel='rbf', probability=True, random_state=RANDOM_SEED
        ),
        'KNN': KNeighborsClassifier(
            n_neighbors=min(5, len(X_train)), n_jobs=args.n_jobs
        ),
        'LogisticRegression': LogisticRegression(
            max_iter=1000, random_state=RANDOM_SEED, n_jobs=args.n_jobs
        ),
        'NaiveBayes': GaussianNB()
    }

    # 合并训练集和验证集
    X_train_ml = np.vstack([X_train, X_val])
    y_train_ml = np.hstack([y_train, y_val])

    # ========== 机器学习模型 ==========
    for name, model in ml_models.items():
        try:
            print(f"\n  Processing {name}...")
            start_time = time.time()

            # 训练模型
            model.fit(X_train_ml, y_train_ml)

            # 获取预测
            train_predictions = model.predict(X_train_ml)
            test_predictions = model.predict(X_test)

            # 计算指标
            acc = accuracy_score(y_test, test_predictions)
            prec = precision_score(y_test, test_predictions, average='weighted', zero_division=0)
            rec = recall_score(y_test, test_predictions, average='weighted', zero_division=0)
            f1 = f1_score(y_test, test_predictions, average='weighted', zero_division=0)

            results[name] = {
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1': f1,
                'predictions': test_predictions,
                'training_time': time.time() - start_time
            }

            gene_predictions[name] = {
                'test_genes': test_genes,
                'true_labels': y_test,
                'predicted_labels': test_predictions
            }

            print(f"    ✓ Accuracy: {acc:.4f}, F1: {f1:.4f}, Time: {results[name]['training_time']:.2f}s")

            # *** 生成模型性能图 ***
            plot_model_specific_results(
                name, y_test, test_predictions,
                {},  # 机器学习模型没有training_history
                test_genes,
                subdirs['models']
            )

            # *** 保存增强的预测结果 ***
            save_enhanced_gene_predictions(
                name, gene_predictions, original_gene_exp,
                group_means, subdirs['models']
            )

            # 创建过拟合分析图
            overfitting_path = create_overfitting_analysis_plot(
                name, X_train_ml, y_train_ml, train_predictions,
                X_test, y_test, test_predictions,
                {}, subdirs['overfitting']
            )

            # 导出模型
            if args.export_models:
                export_dir = os.path.join(subdirs['exported_models'], name)
                os.makedirs(export_dir, exist_ok=True)

                joblib.dump(model, os.path.join(export_dir, f'{name}_model.joblib'))
                print(f"    ✓ Model exported")

        except Exception as e:
            print(f"    ✗ Failed: {e}")
            import traceback
            traceback.print_exc()  # 打印详细错误信息
            results[name] = {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0}

    # 创建时序分析图
    if gene_predictions:
        best_model = max(results.items(), key=lambda x: x[1]['accuracy'])[0]
        pred = gene_predictions[best_model]
        correct_mask = pred['predicted_labels'] == pred['true_labels']
        selected_genes = np.array(pred['test_genes'])[correct_mask][:21]

        time_series_path = create_enhanced_time_series_plot(
            gene_exp, groups_dict, sorted_groups,
            selected_genes, subdirs['time_series']
        )

    # 创建可视化
    if results:
        # 创建雷达图
        create_model_radar_charts(results, subdirs['visualizations'])

        # 创建PR tradeoff图
        create_precision_recall_tradeoff(results, subdirs['visualizations'])

        # 创建综合模型性能分析图
        create_comprehensive_model_analysis(results, subdirs['visualizations'])

        # 调用增强可视化函数
        create_enhanced_visualizations(
            group_means, gene_clusters, cluster_centers,
            key_genes, sorted_groups, subdirs['visualizations']
        )
        # 添加表达模式分析
        if key_genes is not None and not key_genes.empty:
            create_expression_pattern_analysis(key_genes, subdirs['visualizations'])

        # 添加基因重叠分析
        if gene_predictions:
            create_gene_overlap_analysis(gene_predictions, subdirs['intersection'])
        # 创建模型比较图
        predictions_dict = {}
        create_model_comparison_plots(
            results, predictions_dict, subdirs['visualizations']
        )

    # 保存所有数据到data文件夹
    print("\n  Saving analysis data to data folder...")

    # 保存组均值
    if not group_means.empty:
        group_means_path = os.path.join(subdirs['data'], 'group_means.csv')
        group_means.to_csv(group_means_path)
        print(f"    Saved group means: {group_means_path}")

    # 保存基因聚类
    if gene_clusters is not None and isinstance(gene_clusters, pd.DataFrame) and not gene_clusters.empty:
        clusters_path = os.path.join(subdirs['data'], 'gene_clusters.csv')
        gene_clusters.to_csv(clusters_path, index=False)
        print(f"    Saved gene clusters: {clusters_path}")

    # 保存关键基因
    if key_genes is not None and isinstance(key_genes, pd.DataFrame) and not key_genes.empty:
        key_genes_path = os.path.join(subdirs['data'], 'key_genes.csv')
        key_genes.to_csv(key_genes_path, index=False)
        print(f"    Saved key genes: {key_genes_path}")

    # 保存模型结果
    if results:
        model_results_df = pd.DataFrame(results).T
        model_results_path = os.path.join(subdirs['data'], 'model_performance.csv')
        model_results_df.to_csv(model_results_path)
        print(f"    Saved model performance: {model_results_path}")

        # 保存为JSON格式
        model_results_json = os.path.join(subdirs['data'], 'model_results.json')
        with open(model_results_json, 'w') as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)
        print(f"    Saved model results JSON: {model_results_json}")

    # 保存基因预测结果
    if gene_predictions:
        for model_name, pred_data in gene_predictions.items():
            pred_df = pd.DataFrame({
                'gene': pred_data['test_genes'],
                'true_label': pred_data['true_labels'],
                'predicted_label': pred_data['predicted_labels']
            })
            pred_path = os.path.join(subdirs['data'], f'{model_name}_predictions.csv')
            pred_df.to_csv(pred_path, index=False)
        print(f"    Saved gene predictions for {len(gene_predictions)} models")

    # 网络分析 - 修复这里，使用results而不是model_results
    if results and gene_predictions:  # 修改：使用results
        # 创建交集分析
        intersection_genes = create_intersection_analysis(
            gene_predictions, subdirs['intersection']
        )

        # 保存交集基因
        if intersection_genes:
            intersection_df = pd.DataFrame({'gene': list(intersection_genes)})
            intersection_path = os.path.join(subdirs['data'], 'intersection_genes.csv')
            intersection_df.to_csv(intersection_path, index=False)
            print(f"    Saved {len(intersection_genes)} intersection genes")

        # 创建基因网络
        if not group_means.empty:
            # 主网络
            network_graph = create_gene_network(
                gene_predictions, group_means, subdirs['networks']
            )

            # 组特定网络
            create_group_specific_networks(
                gene_predictions, group_means, gene_exp, groups_dict,
                sorted_groups, subdirs['networks'], top_n=5
            )

            print("  Network analysis completed")
        # 创建交集分析
        intersection_genes = create_intersection_analysis(
            gene_predictions, subdirs['intersection']
        )

        # 保存交集基因
        if intersection_genes:
            intersection_df = pd.DataFrame({'gene': list(intersection_genes)})
            intersection_path = os.path.join(subdirs['data'], 'intersection_genes.csv')
            intersection_df.to_csv(intersection_path, index=False)
            print(f"    Saved {len(intersection_genes)} intersection genes")

            # 创建基因网络
        if not group_means.empty:
            # 主网络
            network_graph = create_gene_network(
                gene_predictions, group_means, subdirs['networks']
            )

            # 组特定网络
            create_group_specific_networks(
                gene_predictions, group_means, gene_exp, groups_dict,
                sorted_groups, subdirs['networks'], top_n=5
            )

            # *** 在这里添加调用 ***
            # 创建网络基因的箱线图with显著性分析
            create_network_gene_boxplots(
                intersection_genes, gene_exp, groups_dict,
                sorted_groups, subdirs['networks'], top_n=12
            )

            print("  Network analysis completed")
    return results, gene_predictions, test_genes

# ============= WGCNA分析可视化 =============
def visualize_wgcna_results(wgcna_results, gene_exp, output_dir):
    """WGCNA结果可视化（修复递归深度问题）"""
    print("\nVisualizing WGCNA results...")

    # 增加递归深度限制（临时解决方案）
    import sys
    old_recursion_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(10000)  # 增加递归限制

    try:
        # 1. 软阈值选择图
        if 'fitIndices' in wgcna_results:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

            fit_indices = wgcna_results['fitIndices']
            powers = fit_indices['Power']
            sft_r2 = fit_indices['SFT.R.sq']
            mean_k = fit_indices['mean.k']

            ax1.plot(powers, sft_r2, 'o-', color='red', markersize=8, linewidth=2)
            ax1.axhline(y=0.85, color='red', linestyle='--', alpha=0.5, linewidth=2)
            ax1.set_xlabel('Soft Threshold (power)', fontsize=12)
            ax1.set_ylabel('Scale Free Topology Model Fit, signed R²', fontsize=12)
            ax1.set_title('Scale Independence', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)

            ax2.plot(powers, mean_k, 'o-', color='blue', markersize=8, linewidth=2)
            ax2.set_xlabel('Soft Threshold (power)', fontsize=12)
            ax2.set_ylabel('Mean Connectivity', fontsize=12)
            ax2.set_title('Mean Connectivity', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)

            plt.suptitle('WGCNA Soft Threshold Selection', fontsize=16, fontweight='bold')
            plt.tight_layout()

            save_path = os.path.join(output_dir, 'wgcna_soft_threshold.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print("  Saved soft threshold plot")

        # 2. 基因树和模块颜色（优化版本，避免绘制大型dendrogram）
        if 'geneTree' in wgcna_results and 'colors' in wgcna_results:
            gene_tree = wgcna_results['geneTree']
            module_colors = wgcna_results['colors']
            n_genes = len(module_colors)

            print(f"  Processing dendrogram for {n_genes} genes...")

            # 如果基因数量太多，只绘制部分或简化版本
            if n_genes > 5000:
                print(f"  Warning: Too many genes ({n_genes}), creating simplified visualization")

                # 创建简化的可视化
                fig, ax = plt.subplots(figsize=(16, 8))

                # 只显示模块颜色条
                unique_colors = np.unique(module_colors)
                color_map = {color: i for i, color in enumerate(unique_colors)}
                color_indices = [color_map[c] for c in module_colors]

                # 创建颜色条
                colors_array = np.array(color_indices).reshape(1, -1)
                cmap = plt.cm.get_cmap('tab20', len(unique_colors))

                im = ax.imshow(colors_array, aspect='auto', cmap=cmap, interpolation='nearest')
                ax.set_yticks([])
                ax.set_xlabel('Gene Index', fontsize=12)
                ax.set_title(f'Module Colors ({n_genes} genes)', fontsize=14, fontweight='bold')

                # 添加模块统计信息
                module_stats_text = "Module Statistics:\n"
                module_counts = pd.Series(module_colors).value_counts()
                for i, (color, count) in enumerate(module_counts.head(10).items()):
                    module_stats_text += f"{color}: {count} genes\n"
                    if i >= 9:  # 最多显示10个模块
                        module_stats_text += f"... and {len(module_counts) - 10} more modules"
                        break

                ax.text(1.02, 0.5, module_stats_text, transform=ax.transAxes,
                        fontsize=10, va='center',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

                plt.tight_layout()
                save_path = os.path.join(output_dir, 'wgcna_module_colors.png')
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                print("  Saved module colors visualization")

            else:
                # 对于较少的基因，可以绘制完整的dendrogram
                fig, axes = plt.subplots(2, 1, figsize=(16, 10),
                                         gridspec_kw={'height_ratios': [4, 1]})

                # 使用no_plot选项避免递归问题
                from scipy.cluster.hierarchy import dendrogram

                # 限制显示的叶子节点数量
                dendro = dendrogram(
                    gene_tree,
                    ax=axes[0],
                    no_labels=True,
                    truncate_mode='level',
                    p=10,  # 只显示前10层
                    color_threshold=0,
                    above_threshold_color='black'
                )

                axes[0].set_title('Gene Clustering Dendrogram (Truncated)', fontsize=14, fontweight='bold')
                axes[0].set_ylabel('Height', fontsize=12)

                # 模块颜色条
                unique_colors = np.unique(module_colors)
                color_map = {color: i for i, color in enumerate(unique_colors)}
                color_indices = [color_map[c] for c in module_colors]

                colors_array = np.array(color_indices).reshape(1, -1)
                cmap = plt.cm.get_cmap('tab20', len(unique_colors))

                axes[1].imshow(colors_array, aspect='auto', cmap=cmap, interpolation='nearest')
                axes[1].set_yticks([])
                axes[1].set_xlabel('Gene Index', fontsize=12)
                axes[1].set_title('Module Colors', fontsize=14, fontweight='bold')

                plt.tight_layout()
                save_path = os.path.join(output_dir, 'wgcna_gene_dendrogram.png')
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                print("  Saved gene dendrogram")

        # 3. TOM热图（采样显示）
        if 'TOM' in wgcna_results:
            tom = wgcna_results['TOM']

            # 对大型矩阵进行采样
            n_samples = min(500, tom.shape[0])
            if tom.shape[0] > n_samples:
                print(f"  Sampling {n_samples} genes from {tom.shape[0]} for TOM heatmap")
                sample_idx = np.random.choice(tom.shape[0], n_samples, replace=False)
                sample_idx = np.sort(sample_idx)
                tom_sample = tom[sample_idx][:, sample_idx]
            else:
                tom_sample = tom

            fig, ax = plt.subplots(figsize=(10, 8))

            im = ax.imshow(tom_sample, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
            ax.set_title(f'Topological Overlap Matrix (n={n_samples} genes)',
                         fontsize=14, fontweight='bold')
            ax.set_xlabel('Gene Index', fontsize=12)
            ax.set_ylabel('Gene Index', fontsize=12)

            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('TOM Similarity', fontsize=11)

            plt.tight_layout()
            save_path = os.path.join(output_dir, 'wgcna_tom_heatmap.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print("  Saved TOM heatmap")

        print("  WGCNA visualization completed")

    finally:
        # 恢复原始递归限制
        sys.setrecursionlimit(old_recursion_limit)

        save_path = os.path.join(output_dir, 'wgcna_tom_heatmap.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    print("  WGCNA visualization completed")


def create_wgcna_module_statistics(wgcna_results, output_dir):
    """创建WGCNA模块统计图"""
    if not wgcna_results or 'colors' not in wgcna_results:
        print("  No WGCNA results for module statistics")
        return

    print("  Creating WGCNA module statistics...")

    module_colors = wgcna_results['colors']

    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # 1. 模块大小条形图
    module_counts = pd.Series(module_colors).value_counts()

    # 排除grey模块（未分配的基因）
    if 'grey' in module_counts.index:
        grey_count = module_counts['grey']
        module_counts = module_counts.drop('grey')
        print(f"    Excluded {grey_count} grey (unassigned) genes")

    # 只显示前15个最大的模块
    if len(module_counts) > 15:
        module_counts = module_counts.head(15)

    # 按大小排序
    module_counts = module_counts.sort_values(ascending=True)  # ascending for horizontal bar

    # 为每个模块分配颜色
    bar_colors = []
    standard_colors = {
        'turquoise': '#40E0D0', 'blue': '#0000FF', 'brown': '#964B00',
        'yellow': '#FFFF00', 'green': '#00FF00', 'red': '#FF0000',
        'black': '#000000', 'pink': '#FFC0CB', 'magenta': '#FF00FF',
        'purple': '#800080', 'greenyellow': '#ADFF2F', 'tan': '#D2B48C',
        'salmon': '#FA8072', 'cyan': '#00FFFF', 'midnightblue': '#191970'
    }

    for module in module_counts.index:
        if module in standard_colors:
            bar_colors.append(standard_colors[module])
        else:
            # 使用NATURE_COLORS或生成随机颜色
            bar_colors.append(NATURE_COLORS[len(bar_colors) % len(NATURE_COLORS)])

    # 水平条形图
    bars = ax1.barh(range(len(module_counts)), module_counts.values, color=bar_colors)
    ax1.set_yticks(range(len(module_counts)))
    ax1.set_yticklabels(module_counts.index, fontsize=10)
    ax1.set_xlabel('Number of Genes', fontsize=12)
    ax1.set_ylabel('Module Color', fontsize=12)
    ax1.set_title('WGCNA Module Sizes', fontsize=14, fontweight='bold')

    # 添加数值标签
    for bar, count in zip(bars, module_counts.values):
        ax1.text(count + max(module_counts.values) * 0.01, bar.get_y() + bar.get_height() / 2.,
                 f'{count}', ha='left', va='center', fontsize=9)

    ax1.grid(True, alpha=0.3, axis='x')

    # 2. 模块分布饼图
    # 显示前8个最大的模块，其余归为Others
    top_modules = module_counts.head(8)
    other_count = module_counts[8:].sum() if len(module_counts) > 8 else 0

    if other_count > 0:
        pie_data = list(top_modules.values) + [other_count]
        pie_labels = [f'{mod}\n({count} genes)' for mod, count in zip(top_modules.index, top_modules.values)]
        pie_labels.append(f'Others\n({other_count} genes)')
    else:
        pie_data = list(top_modules.values)
        pie_labels = [f'{mod}\n({count} genes)' for mod, count in zip(top_modules.index, top_modules.values)]

    # 使用与条形图相同的颜色
    pie_colors = bar_colors[:len(top_modules)]
    if other_count > 0:
        pie_colors.append('#808080')  # 灰色表示Others

    wedges, texts, autotexts = ax2.pie(pie_data, labels=pie_labels,
                                       autopct='%1.1f%%',
                                       colors=pie_colors,
                                       startangle=90,
                                       textprops={'fontsize': 9})

    # 调整标签位置避免重叠
    for text in texts:
        text.set_fontsize(9)
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(9)

    ax2.set_title('Module Distribution (Top 8)', fontsize=14, fontweight='bold')

    plt.suptitle('WGCNA Module Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()

    # 保存
    save_path_png = os.path.join(output_dir, 'module_statistics.png')
    save_path_pdf = os.path.join(output_dir, 'module_statistics.pdf')
    plt.savefig(save_path_png, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(save_path_pdf, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"    Saved module statistics: {save_path_png}")

    # 如果有模块特征基因，创建相关性热图
    if 'MEs' in wgcna_results and wgcna_results['MEs'] is not None and not wgcna_results['MEs'].empty:
        create_module_eigengene_heatmap(wgcna_results['MEs'], output_dir)


def create_module_eigengene_heatmap(MEs, output_dir):
    """创建模块特征基因相关性热图"""
    if MEs is None or MEs.empty:
        return

    print("    Creating module eigengene correlation heatmap...")

    fig, ax = plt.subplots(figsize=(12, 10))

    # 计算相关性矩阵
    ME_cor = MEs.corr()

    # 创建热图
    sns.heatmap(ME_cor, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, vmin=-1, vmax=1,
                square=True, linewidths=0.5,
                cbar_kws={"shrink": 0.8, "label": "Correlation"},
                ax=ax)

    ax.set_title('Module Eigengene Correlations', fontsize=14, fontweight='bold')
    ax.set_xlabel('Module Eigengene', fontsize=11)
    ax.set_ylabel('Module Eigengene', fontsize=11)

    # 旋转标签
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()

    # 保存
    save_path_png = os.path.join(output_dir, 'module_eigengene_correlations.png')
    save_path_pdf = os.path.join(output_dir, 'module_eigengene_correlations.pdf')
    plt.savefig(save_path_png, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(save_path_pdf, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"    Saved module eigengene correlations: {save_path_png}")

# ============= 综合报告生成 =============
def generate_enhanced_html_report(args, preprocessing_stats, wgcna_results,
                                  integrated_results, model_results, output_dir):
    """生成增强的HTML报告"""
    print("\nGenerating comprehensive HTML report...")

    # HTML模板
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Gene Expression Analysis Report V3.0</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        .header p {{
            font-size: 1.1em;
            opacity: 0.9;
        }}
        .content {{
            padding: 40px;
        }}
        .section {{
            margin-bottom: 40px;
        }}
        .section h2 {{
            color: #333;
            font-size: 1.8em;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #667eea;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }}
        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            transition: transform 0.3s;
        }}
        .stat-card:hover {{
            transform: translateY(-5px);
        }}
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            margin-bottom: 5px;
        }}
        .stat-label {{
            font-size: 0.9em;
            opacity: 0.9;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
            border-radius: 10px;
            overflow: hidden;
        }}
        th {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
        }}
        td {{
            padding: 12px 15px;
            border-bottom: 1px solid #f0f0f0;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        tr:hover {{
            background-color: #f0f0f0;
        }}
        .best-model {{
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            font-weight: bold;
        }}
        .info-box {{
            background: #e3f2fd;
            border-left: 4px solid #2196f3;
            padding: 20px;
            margin: 20px 0;
            border-radius: 5px;
        }}
        .success-box {{
            background: #e8f5e9;
            border-left: 4px solid #4caf50;
            padding: 20px;
            margin: 20px 0;
            border-radius: 5px;
        }}
        .warning-box {{
            background: #fff3e0;
            border-left: 4px solid #ff9800;
            padding: 20px;
            margin: 20px 0;
            border-radius: 5px;
        }}
        .footer {{
            background: #f5f5f5;
            padding: 20px;
            text-align: center;
            color: #666;
            font-size: 0.9em;
        }}
        .badge {{
            display: inline-block;
            padding: 5px 10px;
            border-radius: 20px;
            background: #667eea;
            color: white;
            font-size: 0.8em;
            margin: 0 5px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧬 Gene Expression Analysis Report V3.0</h1>
            <p>Comprehensive Analysis with WGCNA, KNN Clustering, and Deep Learning</p>
            <p style="margin-top: 10px; font-size: 0.9em;">
                Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            </p>
        </div>

        <div class="content">
            <!-- Overview Section -->
            <div class="section">
                <h2>📊 Analysis Overview</h2>
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-value">{preprocessing_stats.get('final_genes', 0):,}</div>
                        <div class="stat-label">Total Genes Analyzed</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{preprocessing_stats.get('final_samples', 0):,}</div>
                        <div class="stat-label">Total Samples</div>
                    </div>
"""

    # WGCNA统计
    if wgcna_results:
        n_modules = len(np.unique(wgcna_results.get('colors', []))) - 1
        html_content += f"""
                    <div class="stat-card">
                        <div class="stat-value">{n_modules}</div>
                        <div class="stat-label">WGCNA Modules</div>
                    </div>
        """

    # KNN统计
    if integrated_results:
        n_subclusters = len(np.unique(integrated_results.get('knn_subclusters', []))[
                                np.unique(integrated_results.get('knn_subclusters', [])) >= 0
                                ])
        html_content += f"""
                    <div class="stat-card">
                        <div class="stat-value">{n_subclusters}</div>
                        <div class="stat-label">KNN Sub-clusters</div>
                    </div>
        """

    # 模型统计
    if model_results:
        best_model = max(model_results.items(), key=lambda x: x[1]['accuracy'])
        html_content += f"""
                    <div class="stat-card">
                        <div class="stat-value">{len(model_results)}</div>
                        <div class="stat-label">Models Evaluated</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{best_model[1]['accuracy']:.4f}</div>
                        <div class="stat-label">Best Accuracy ({best_model[0]})</div>
                    </div>
                </div>
            </div>
        """
    else:
        html_content += """
                </div>
            </div>
        """

    # 预处理统计
    html_content += f"""
            <!-- Preprocessing Section -->
            <div class="section">
                <h2>🔬 Data Preprocessing</h2>
                <div class="info-box">
                    <h3>Input Files</h3>
                    <p><strong>Expression Matrix:</strong> {args.input}</p>
                    <p><strong>Sample Groups:</strong> {args.groups}</p>
                </div>
                <table>
                    <tr>
                        <th>Preprocessing Step</th>
                        <th>Value</th>
                        <th>Description</th>
                    </tr>
                    <tr>
                        <td>Original Genes</td>
                        <td>{preprocessing_stats.get('original_genes', 'N/A'):,}</td>
                        <td>Initial gene count</td>
                    </tr>
                    <tr>
                        <td>Zero-expression Removed</td>
                        <td>{preprocessing_stats.get('zero_genes_removed', 0):,}</td>
                        <td>Genes with no expression</td>
                    </tr>
                    <tr>
                        <td>Low-expression Removed</td>
                        <td>{preprocessing_stats.get('low_exp_removed', 0):,}</td>
                        <td>Genes below threshold</td>
                    </tr>
                    <tr>
                        <td>Low-variance Removed</td>
                        <td>{preprocessing_stats.get('low_variance_removed', 0):,}</td>
                        <td>Genes with minimal variation</td>
                    </tr>
                    <tr>
                        <td>Log2 Transformation</td>
                        <td>{'Yes' if preprocessing_stats.get('log_transformed', False) else 'No'}</td>
                        <td>Data normalization</td>
                    </tr>
                    <tr>
                        <td>Final Genes</td>
                        <td><strong>{preprocessing_stats.get('final_genes', 'N/A'):,}</strong></td>
                        <td>Genes for analysis</td>
                    </tr>
                </table>
            </div>
    """

    # WGCNA结果
    if wgcna_results:
        html_content += """
            <!-- WGCNA Section -->
            <div class="section">
                <h2>🌐 WGCNA Analysis</h2>
                <table>
                    <tr>
                        <th>Module Color</th>
                        <th>Gene Count</th>
                        <th>Percentage</th>
                    </tr>
        """

        module_colors = wgcna_results.get('colors', [])
        if len(module_colors) > 0:
            module_counts = pd.Series(module_colors).value_counts()
            total = len(module_colors)

            for color, count in module_counts.head(15).items():
                percentage = (count / total) * 100
                html_content += f"""
                    <tr>
                        <td><span class="badge" style="background: {color};">{color}</span></td>
                        <td>{count:,}</td>
                        <td>{percentage:.2f}%</td>
                    </tr>
                """

        html_content += """
                </table>
            </div>
        """

    # 模型性能
    if model_results:
        html_content += """
            <!-- Model Performance Section -->
            <div class="section">
                <h2>🤖 Model Performance</h2>
                <table>
                    <tr>
                        <th>Rank</th>
                        <th>Model</th>
                        <th>Accuracy</th>
                        <th>Precision</th>
                        <th>Recall</th>
                        <th>F1 Score</th>
                        <th>Time (s)</th>
                        <th>Type</th>
                    </tr>
        """

        sorted_models = sorted(model_results.items(),
                               key=lambda x: x[1]['accuracy'], reverse=True)

        dl_models = ['RNN', 'LSTM', 'BiLSTM', 'GRU', 'CNN', 'ResNet',
                     'Transformer', 'AttentionRNN', 'TCN', 'WaveNet', 'InceptionTime']

        for rank, (model_name, metrics) in enumerate(sorted_models, 1):
            model_type = '🧠 Deep Learning' if model_name in dl_models else '📊 Machine Learning'
            row_class = 'best-model' if rank == 1 else ''

            html_content += f"""
                <tr class="{row_class}">
                    <td>{rank}</td>
                    <td>{model_name}</td>
                    <td>{metrics['accuracy']:.4f}</td>
                    <td>{metrics['precision']:.4f}</td>
                    <td>{metrics['recall']:.4f}</td>
                    <td>{metrics['f1']:.4f}</td>
                    <td>{metrics.get('training_time', 0):.2f}</td>
                    <td>{model_type}</td>
                </tr>
            """

        html_content += """
                </table>

                <div class="success-box">
                    <h3>✅ Best Performing Model</h3>
        """

        best = sorted_models[0]
        html_content += f"""
                    <p><strong>Model:</strong> {best[0]}</p>
                    <p><strong>Accuracy:</strong> {best[1]['accuracy']:.4f}</p>
                    <p><strong>F1 Score:</strong> {best[1]['f1']:.4f}</p>
                </div>
            </div>
        """

    # 分析总结
    html_content += """
            <!-- Summary Section -->
            <div class="section">
                <h2>📈 Analysis Summary</h2>
                <div class="info-box">
                    <h3>Key Findings</h3>
                    <ul style="margin-left: 20px; margin-top: 10px;">
    """

    if wgcna_results:
        html_content += f"""
                        <li>Successfully identified {len(np.unique(wgcna_results.get('colors', []))) - 1} co-expression modules using WGCNA</li>
        """

    if integrated_results:
        html_content += f"""
                        <li>KNN clustering revealed {len(np.unique(integrated_results.get('knn_subclusters', []))[np.unique(integrated_results.get('knn_subclusters', [])) >= 0])} sub-clusters within modules</li>
        """

    if model_results:
        dl_accs = [res['accuracy'] for name, res in model_results.items()
                   if name in ['RNN', 'LSTM', 'BiLSTM', 'GRU', 'CNN', 'ResNet',
                               'Transformer', 'AttentionRNN', 'TCN', 'WaveNet', 'InceptionTime']]
        ml_accs = [res['accuracy'] for name, res in model_results.items()
                   if name not in ['RNN', 'LSTM', 'BiLSTM', 'GRU', 'CNN', 'ResNet',
                                   'Transformer', 'AttentionRNN', 'TCN', 'WaveNet', 'InceptionTime']]

        if dl_accs:
            html_content += f"""
                        <li>Deep Learning models achieved average accuracy: {np.mean(dl_accs):.4f}</li>
            """
        if ml_accs:
            html_content += f"""
                        <li>Machine Learning models achieved average accuracy: {np.mean(ml_accs):.4f}</li>
            """

    html_content += """
                    </ul>
                </div>

                <div class="warning-box">
                    <h3>📁 Output Files</h3>
                    <p>All results have been saved to the output directory, including:</p>
                    <ul style="margin-left: 20px; margin-top: 10px;">
                        <li>WGCNA module assignments and visualizations</li>
                        <li>KNN clustering results</li>
                        <li>Neural network architecture diagrams (circular nodes)</li>
                        <li>Overfitting analysis plots</li>
                        <li>Time series expression trends</li>
                        <li>Boxplot and heatmap analyses</li>
                        <li>Exported models for future use</li>
                    </ul>
                </div>
            </div>
        </div>

        <div class="footer">
            <p>Enhanced Gene Expression Analysis System V3.0</p>
            <p>Report generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>© 2024 - Advanced Bioinformatics Analysis Pipeline</p>
        </div>
    </div>
</body>
</html>
    """

    # 保存报告
    report_path = os.path.join(output_dir, 'comprehensive_analysis_report.html')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"  Report saved: {report_path}")

    return report_path


# ============= 数据整合函数 =============
def integrate_data(gene_exp, groups_df):
    """整合表达数据和分组信息"""
    print("\nIntegrating expression data and grouping information...")

    exp_samples = set(gene_exp.columns)
    group_samples = set(groups_df['Sample_ID'])

    # 找到共同样本
    common_samples = exp_samples.intersection(group_samples)

    if len(common_samples) == 0:
        raise ValueError("No matching samples between expression data and group file")

    print(f"  Matched samples: {len(common_samples)}")

    # 过滤数据
    gene_exp = gene_exp[list(common_samples)]
    groups_df = groups_df[groups_df['Sample_ID'].isin(common_samples)]

    # 构建分组字典
    groups_dict = {}
    for _, row in groups_df.iterrows():
        group = row['Group']
        sample = row['Sample_ID']
        if group not in groups_dict:
            groups_dict[group] = []
        groups_dict[group].append(sample)

    # 自然排序分组
    sorted_groups = sorted(groups_dict.keys(), key=lambda x: (
        float(''.join(filter(str.isdigit, str(x))) or '0'),
        x
    ))

    print(f"  Detected groups: {sorted_groups}")

    return gene_exp, groups_dict, sorted_groups


def extract_group_means(gene_exp, groups_dict, sorted_groups):
    """提取每组的平均表达值"""
    group_means = pd.DataFrame(index=gene_exp.index)
    group_stds = pd.DataFrame(index=gene_exp.index)

    for group in sorted_groups:
        samples = groups_dict[group]
        if samples:
            valid_samples = [s for s in samples if s in gene_exp.columns]
            if valid_samples:
                group_means[group] = gene_exp[valid_samples].mean(axis=1)
                group_stds[group] = gene_exp[valid_samples].std(axis=1)

    return group_means, group_stds


""""""


# ============= 聚类分析函数 =============
def cluster_analysis(group_means, n_clusters=6):
    """聚类分析"""
    print(f"\nPerforming cluster analysis (k={n_clusters})...")

    if group_means.empty:
        return pd.DataFrame(), pd.DataFrame()

    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    actual_n_clusters = min(n_clusters, group_means.shape[0])

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(group_means.values)

    kmeans = KMeans(n_clusters=actual_n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled_data)

    cluster_centers = pd.DataFrame(
        scaler.inverse_transform(kmeans.cluster_centers_),
        columns=group_means.columns
    )

    gene_clusters = pd.DataFrame({
        'gene': group_means.index,
        'cluster': clusters
    })

    for i in range(actual_n_clusters):
        size = (clusters == i).sum()
        print(f"  Cluster {i}: {size} genes")

    return gene_clusters, cluster_centers

# ============= 添加基因关键性识别函数（严格标准）=============
def identify_key_genes(group_means, gene_clusters):
    """使用更严格的标准识别关键基因"""
    if group_means.empty:
        return pd.DataFrame()

    key_genes = []
    n_points = group_means.shape[1]

    for gene in group_means.index:
        values = group_means.loc[gene].values

        if n_points >= 2:
            slope = np.polyfit(range(n_points), values, 1)[0]
            correlation = np.corrcoef(range(n_points), values)[0, 1]
            r_squared = correlation ** 2
        else:
            slope = 0
            r_squared = 0

        max_change = values.max() - values.min()
        cv = np.std(values) / (np.mean(values) + 1e-10)

        # 更严格的模式分类标准
        if slope > 1.0 and r_squared > 0.8:  # 严格：斜率>1.0且R²>0.8
            pattern = 'Strong_Increasing'
        elif slope < -1.0 and r_squared > 0.8:
            pattern = 'Strong_Decreasing'
        elif slope > 0.3 and r_squared > 0.6:
            pattern = 'Moderate_Increasing'
        elif slope < -0.3 and r_squared > 0.6:
            pattern = 'Moderate_Decreasing'
        elif n_points > 2:
            peak_idx = np.argmax(values)
            if peak_idx in range(1, n_points - 1):
                peak_prominence = values[peak_idx] - np.mean(values)
                if peak_prominence > 0.5 * np.std(values):
                    pattern = 'Peak'
                else:
                    pattern = 'Weak_Peak'
            else:
                pattern = 'Complex'
        else:
            if cv < 0.1:
                pattern = 'Stable'
            else:
                pattern = 'Complex'

        # 严格评分系统
        score = abs(slope) * 15 + r_squared * 10 + max_change * 5
        score *= (1 - cv * 0.5)  # CV惩罚

        # 模式权重
        pattern_weights = {
            'Strong_Increasing': 1.5,
            'Strong_Decreasing': 1.5,
            'Peak': 1.3,
            'Moderate_Increasing': 1.1,
            'Moderate_Decreasing': 1.1,
            'Weak_Peak': 0.9,
            'Complex': 0.7,
            'Stable': 0.3
        }
        score *= pattern_weights.get(pattern, 1.0)

        cluster_id = -1
        if not gene_clusters.empty and gene in gene_clusters['gene'].values:
            cluster_id = gene_clusters[gene_clusters['gene'] == gene]['cluster'].values[0]

        key_genes.append({
            'gene': gene,
            'pattern': pattern,
            'slope': slope,
            'r_squared': r_squared,
            'max_change': max_change,
            'cv': cv,
            'score': score,
            'cluster': cluster_id
        })

    key_genes_df = pd.DataFrame(key_genes)
    key_genes_df = key_genes_df.sort_values('score', ascending=False)

    print("\nGene Pattern Distribution (Strict Criteria):")
    pattern_counts = key_genes_df['pattern'].value_counts()
    for pattern, count in pattern_counts.items():
        print(f"  {pattern}: {count} genes ({count / len(key_genes_df) * 100:.1f}%)")

    return key_genes_df

def prepare_ml_data_integrated(group_means, integrated_results=None):
    """准备机器学习数据，整合聚类结果"""
    if group_means.empty:
        return None, None, None

    X = []
    gene_names = []

    for gene in group_means.index:
        X.append(group_means.loc[gene].values)
        gene_names.append(gene)

    X = np.array(X)

    # 使用聚类结果作为标签，或基于表达模式
    if integrated_results and 'knn_subclusters' in integrated_results:
        y = integrated_results['knn_subclusters']
        # 将噪声点(-1)映射到单独的类
        y[y == -1] = max(y) + 1
    else:
        # 基于表达模式分类
        y = []
        for x in X:
            if len(x) >= 2:
                slope = np.polyfit(range(len(x)), x, 1)[0]
                if slope > 0.5:
                    y.append(0)  # Increasing
                elif slope < -0.5:
                    y.append(1)  # Decreasing
                elif len(x) > 2 and np.argmax(x) in range(1, len(x) - 1):
                    y.append(2)  # Peak
                else:
                    y.append(3)  # Stable/Complex
            else:
                y.append(3)
        y = np.array(y)

    # 确保有足够的类别
    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        print("Warning: Insufficient class diversity, creating synthetic classes")
        # 创建合成类别
        n_samples = len(y)
        y = np.random.randint(0, 4, n_samples)

    return X, y, gene_names


# ============= 主程序 =============
def main():
    """主程序入口"""
    args = parse_arguments()

    print("\n" + "=" * 70)
    print(" " * 10 + "Enhanced Gene Expression Analysis System V3.0")
    print(" " * 15 + "With WGCNA and KNN Clustering")
    print("=" * 70)

    try:
        # 创建输出目录结构
        main_dir, subdirs = create_output_structure(args.output_prefix)

        # ========== 数据加载 ==========
        print("\n" + "=" * 50)
        print("Step 1: Data Loading and Preprocessing")
        print("=" * 50)

        gene_exp, original_gene_exp, preprocessing_stats = load_expression_data(
            args.input, args.sep
        )
        groups_df = load_sample_groups(args.groups, args.group_sep)

        # 数据整合
        gene_exp, groups_dict, sorted_groups = integrate_data(gene_exp, groups_df)

        # 提取组均值
        group_means, group_stds = extract_group_means(
            gene_exp, groups_dict, sorted_groups
        )
        print(f"\nGroup mean matrix: {group_means.shape}")
        # 添加聚类分析
        gene_clusters, cluster_centers = cluster_analysis(group_means, args.n_clusters)

        # 添加关键基因识别
        key_genes = identify_key_genes(group_means, gene_clusters)
        if not key_genes.empty:
            print(f"\nTop 10 key genes (Strict Criteria):")
            for _, row in key_genes.head(10).iterrows():
                print(
                    f"  {row['gene']}: {row['pattern']}, Score={row['score']:.2f}, Slope={row['slope']:.3f}, R²={row['r_squared']:.3f}")

        # ========== WGCNA分析 ==========
        wgcna_results = None
        if args.wgcna:
            print("\n" + "=" * 50)
            print("Step 2: WGCNA Analysis")
            print("=" * 50)

            # 转置数据（WGCNA需要样本为行）
            datExpr = gene_exp.T.values

            # 创建WGCNA对象
            wgcna = WGCNA(verbose=args.verbose)

            # 选择软阈值
            if args.soft_power == 0:
                print("Selecting soft threshold...")
                sft = wgcna.pickSoftThreshold(datExpr)
                power = sft['powerEstimate']
                print(f"  Selected power: {power}")
            else:
                power = args.soft_power
                print(f"  Using specified power: {power}")

            # 构建网络
            net = wgcna.blockwiseModules(
                datExpr,
                power=power,
                minModuleSize=args.min_module_size,
                mergeCutHeight=args.merge_cut_height,
                networkType=args.network_type,
                deepSplit=args.deep_split
            )

            wgcna_results = {
                'colors': net['colors'],
                'MEs': net['MEs'],
                'geneTree': net['geneTree'],
                'TOM': net['TOM'],
                'power': power,
                'gene_names': gene_exp.index.tolist(),
                'fitIndices': sft['fitIndices'] if args.soft_power == 0 else None
            }
            if args.wgcna and wgcna_results:
                # 保存额外的WGCNA分析结果
                if 'colors' in wgcna_results:
                    module_summary = pd.DataFrame({
                        'Gene': gene_exp.index,
                        'Module': wgcna_results['colors']
                    })
                    module_summary.to_csv(os.path.join(subdirs['wgcna'], 'module_assignments.csv'), index=False)

                if 'MEs' in wgcna_results:
                    wgcna_results['MEs'].to_csv(os.path.join(subdirs['wgcna'], 'module_eigengenes.csv'))
            # WGCNA可视化
            if wgcna_results:
                visualize_wgcna_results(wgcna_results, gene_exp, subdirs['wgcna'])
            # 保存WGCNA结果
            wgcna_df = pd.DataFrame({
                'Gene': gene_exp.index,
                'Module': net['colors']
            })
            wgcna_df.to_csv(os.path.join(subdirs['wgcna'], 'wgcna_modules.csv'),
                            index=False)
            create_wgcna_module_statistics(wgcna_results, subdirs['wgcna'])
            print(f"  Saved WGCNA module assignments")

        # ========== KNN聚类 ==========
        integrated_results = None
        if args.knn and wgcna_results:
            print("\n" + "=" * 50)
            print("Step 3: KNN Unsupervised Clustering")
            print("=" * 50)

            integrated_results = perform_integrated_clustering(
                gene_exp, wgcna_results, args, subdirs['knn']
            )
            if args.knn and integrated_results:
                # 保存KNN聚类统计
                knn_summary = pd.DataFrame({
                    'Gene': gene_exp.index,
                    'WGCNA_Module': integrated_results['wgcna_modules'],
                    'KNN_Subcluster': integrated_results['knn_subclusters']
                })
                knn_summary.to_csv(os.path.join(subdirs['knn'], 'integrated_clustering.csv'), index=False)
            # 保存KNN结果
            knn_df = pd.DataFrame({
                'Gene': gene_exp.index,
                'WGCNA_Module': wgcna_results['colors'],
                'KNN_Subcluster': integrated_results['knn_subclusters']
            })
            knn_df.to_csv(os.path.join(subdirs['knn'], 'knn_clustering.csv'),
                          index=False)
            print(f"  Saved KNN clustering results")

        # ========== 准备机器学习数据 ==========
        print("\n" + "=" * 50)
        print("Step 4: Preparing Machine Learning Data")
        print("=" * 50)

        X, y, gene_names = prepare_ml_data_integrated(group_means, integrated_results)

        if X is not None and len(X) >= 10:
            print(f"  Prepared data: {X.shape[0]} samples, {X.shape[1]} features")
            print(f"  Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
        else:
            print("  Insufficient data for machine learning")
            X, y, gene_names = None, None, None

        # ========== 模型训练和评估 ==========
        model_results = {}
        gene_predictions = {}
        # 确保这些变量已定义（即使是空的）
        if 'gene_clusters' not in locals():
            gene_clusters = pd.DataFrame()
        if 'cluster_centers' not in locals():
            cluster_centers = pd.DataFrame()
        if 'key_genes' not in locals():
            key_genes = pd.DataFrame()
        if X is not None and len(X) >= 20:
            print("\n" + "=" * 50)
            print("Step 5: Model Training and Evaluation")
            print("=" * 50)

            model_results, gene_predictions, test_genes = evaluate_all_models_comprehensive(
                X, y, gene_names, args, subdirs,
                gene_exp, groups_dict, sorted_groups,
                group_means, original_gene_exp, integrated_results
            )

            # 在这里添加网络分析
            if model_results and gene_predictions:
                # 创建交集分析
                intersection_genes = create_intersection_analysis(
                    gene_predictions, subdirs['intersection']
                )

                # 创建基因网络（确保这两个函数都被调用）
                if not group_means.empty:
                    # 主网络
                    network_graph = create_gene_network(
                        gene_predictions, group_means, subdirs['networks']
                    )

                    # 组特定网络 - 这是您要添加的调用
                    create_group_specific_networks(
                        gene_predictions, group_means, gene_exp, groups_dict,
                        sorted_groups, subdirs['networks'], top_n=5
                    )

                    print("  Network analysis completed")
            # ========== 结果汇总 ==========
            if model_results:
                print("\n" + "=" * 60)
                print("Model Performance Summary")
                print("=" * 60)
                print(f"{'Rank':<5} {'Model':<20} {'Accuracy':<10} {'F1 Score':<10} {'Time (s)':<10}")
                print("-" * 60)

                sorted_models = sorted(model_results.items(),
                                       key=lambda x: x[1]['accuracy'], reverse=True)

                for rank, (name, res) in enumerate(sorted_models, 1):
                    print(f"{rank:<5} {name:<20} {res['accuracy']:<10.4f} "
                          f"{res['f1']:<10.4f} {res.get('training_time', 0):<10.2f}")

        # ========== 保存关键数据 ==========
        print("\n" + "=" * 50)
        print("Step 6: Saving Results")
        print("=" * 50)

        # 保存组均值
        if not group_means.empty:
            means_path = os.path.join(subdirs['data'], 'group_means.csv')
            group_means.to_csv(means_path)
            print(f"  Saved group means: {means_path}")

        # 保存模型性能
        if model_results:
            perf_df = pd.DataFrame(model_results).T
            perf_path = os.path.join(subdirs['data'], 'model_performance.csv')
            perf_df.to_csv(perf_path)
            print(f"  Saved model performance: {perf_path}")

        # 然后在保存JSON时使用：
        with open(os.path.join(subdirs['data'], 'preprocessing_stats.json'), 'w') as f:
            json.dump(preprocessing_stats, f, indent=2, cls=NumpyEncoder)
        # ========== 生成综合报告 ==========
        print("\n" + "=" * 50)
        print("Step 7: Generating Reports")
        print("=" * 50)

        report_path = generate_enhanced_html_report(
            args, preprocessing_stats, wgcna_results,
            integrated_results, model_results, subdirs['reports']
        )

        # 创建摘要JSON
        summary = {
            'analysis_date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'input_files': {
                'expression': args.input,
                'groups': args.groups
            },
            'preprocessing': {
                'original_genes': preprocessing_stats.get('original_genes', 0),
                'final_genes': preprocessing_stats.get('final_genes', 0),
                'original_samples': preprocessing_stats.get('original_samples', 0),
                'final_samples': preprocessing_stats.get('final_samples', 0)
            },
            'wgcna': {
                'enabled': args.wgcna,
                'n_modules': len(np.unique(wgcna_results['colors']) if wgcna_results else []) - 1,
                'power': wgcna_results.get('power') if wgcna_results else None
            },
            'knn': {
                'enabled': args.knn,
                'n_subclusters': len(np.unique(integrated_results['knn_subclusters'][
                                                   integrated_results['knn_subclusters'] >= 0
                                                   ])) if integrated_results else 0
            },
            'models': {
                'n_evaluated': len(model_results),
                'best_model': max(model_results.items(),
                                  key=lambda x: x[1]['accuracy'])[0] if model_results else None,
                'best_accuracy': max(model_results.items(),
                                     key=lambda x: x[1]['accuracy'])[1]['accuracy'] if model_results else 0
            },
            'output_directory': main_dir
        }

        summary_path = os.path.join(subdirs['reports'], 'analysis_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"  Saved analysis summary: {summary_path}")

        # ========== 完成 ==========
        print("\n" + "=" * 70)
        print("✅ Analysis Complete!")
        print("=" * 70)
        print(f"\n📁 All results saved in: {main_dir}")
        print("\n📊 Directory Structure:")
        print(f"  {main_dir}/")
        print(f"  ├── 🧬 wgcna_analysis/       - WGCNA network analysis")
        print(f"  ├── 🎯 knn_clustering/       - KNN clustering results")
        print(f"  ├── 🤖 models/               - Model-specific results")
        print(f"  ├── 🧠 nn_architectures/     - Neural network diagrams")
        print(f"  ├── 📉 overfitting/          - Overfitting analysis")
        print(f"  ├── 📊 time_series/          - Time series plots")
        print(f"  ├── 📊 boxplots/             - Boxplot analyses")
        print(f"  ├── 🔥 heatmaps/             - Heatmap analyses")
        print(f"  ├── 📈 visualizations/       - Other visualizations")
        print(f"  ├── 💾 data/                 - Processed data files")
        print(f"  ├── 📝 reports/              - Analysis reports")
        print(f"  └── 💼 exported_models/      - Exported model files")

        print("\n📊 Key Statistics:")
        if preprocessing_stats:
            print(f"  • Analyzed {preprocessing_stats['final_genes']:,} genes")
            print(f"  • Across {preprocessing_stats['final_samples']:,} samples")
        if wgcna_results:
            n_modules = len(np.unique(wgcna_results['colors'])) - 1
            print(f"  • Identified {n_modules} co-expression modules")
        if integrated_results:
            n_subclusters = len(np.unique(integrated_results['knn_subclusters'][
                                              integrated_results['knn_subclusters'] >= 0
                                              ]))
            print(f"  • Found {n_subclusters} KNN sub-clusters")
        if model_results:
            best = max(model_results.items(), key=lambda x: x[1]['accuracy'])
            print(f"  • Best model: {best[0]} (Accuracy: {best[1]['accuracy']:.4f})")

        print("\n🎉 Analysis pipeline completed successfully!")
        print("📧 Check the HTML report for detailed results.")

    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


# ============= 程序入口 =============
if __name__ == "__main__":
    # 检查环境
    print("Checking environment...")

    # Python版本
    if sys.version_info < (3, 6):
        print("Error: Python 3.6+ required")
        sys.exit(1)

    # NumPy版本
    numpy_version = tuple(map(int, np.__version__.split('.')[:2]))
    if numpy_version[0] == 1 and numpy_version[1] == 24:
        print(f"✓ NumPy version {np.__version__} (compatible)")
    else:
        print(f"⚠ NumPy version {np.__version__} (tested with 1.24)")

    # PyTorch
    print(f"✓ PyTorch version {torch.__version__}")

    # CUDA
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("ℹ CUDA not available, using CPU")

    print()

    # 运行主程序
    main()

"""
================================================================================
Enhanced Gene Expression Analysis System V3.0
完整实现，包含所有功能
================================================================================

主要特性：
1. WGCNA分析 - 加权基因共表达网络分析
2. KNN无监督聚类 - 基于密度的聚类细化
3. 深度学习模型（11个）- RNN, LSTM, BiLSTM, GRU, CNN, ResNet等
4. 机器学习模型（8个）- RF, GB, SVM, KNN等
5. 圆形节点神经网络架构图
6. 彩色时序分析图
7. 过拟合分析（真实值线条vs预测值散点）
8. 每个模型的箱线图和热图分析
9. 综合HTML报告

运行示例：
python gene_analysis.py -i expression.csv -g groups.txt --wgcna --knn
================================================================================
"""