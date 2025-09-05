# utils/visualization.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 尝试使用 seaborn；没有就回退到 matplotlib
try:
    import seaborn as sns  # noqa: F401
    _HAS_SEABORN = True
except Exception:
    _HAS_SEABORN = False


class RecommendationVisualizer:
    def __init__(self):
        # 使用默认样式，避免老版本 seaborn 样式名不兼容
        plt.style.use('default')

    def plot_rating_distribution(self, ratings_df, save_path=None):
        """绘制评分分布"""
        plt.figure(figsize=(10, 6))

        plt.subplot(1, 2, 1)
        ratings_df['rating'].hist(bins=5, alpha=0.7)
        plt.xlabel('Rating')
        plt.ylabel('Frequency')
        plt.title('Rating Distribution')

        plt.subplot(1, 2, 2)
        user_rating_counts = ratings_df.groupby('userId').size()
        plt.hist(user_rating_counts, bins=50, alpha=0.7)
        plt.xlabel('Number of Ratings per User')
        plt.ylabel('Number of Users')
        plt.title('User Activity Distribution')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def plot_similarity_heatmap(self, similarity_matrix, user_labels=None, save_path=None):
        """绘制相似度热力图"""
        plt.figure(figsize=(12, 10))

        # 选择部分用户显示（避免图太大）
        if similarity_matrix.shape[0] > 100:
            indices = np.random.choice(similarity_matrix.shape[0], 100, replace=False)
            similarity_subset = similarity_matrix[np.ix_(indices, indices)]
        else:
            similarity_subset = similarity_matrix
            indices = range(similarity_matrix.shape[0])

        if _HAS_SEABORN:
            import seaborn as sns
            sns.heatmap(
                similarity_subset,
                xticklabels=indices if user_labels is None else [user_labels[i] for i in indices],
                yticklabels=indices if user_labels is None else [user_labels[i] for i in indices],
                cmap='coolwarm', center=0
            )
        else:
            plt.imshow(similarity_subset, aspect='auto')
            plt.colorbar()
            plt.xticks(
                range(len(list(indices))),
                list(indices) if user_labels is None else [user_labels[i] for i in indices],
                rotation=90
            )
            plt.yticks(
                range(len(list(indices))),
                list(indices) if user_labels is None else [user_labels[i] for i in indices]
            )
        plt.title('User Similarity Heatmap')

        if save_path:
            plt.savefig(save_path)
        plt.show()

    def plot_cluster_analysis(self, cluster_analysis, save_path=None):
        """绘制聚类分析结果"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 平均评分
        cluster_analysis['avg_rating']['mean'].plot(kind='bar', ax=axes[0, 0])
        axes[0, 0].set_title('Average Rating by Cluster')
        axes[0, 0].set_ylabel('Average Rating')

        # 评分数量
        cluster_analysis['rating_count']['mean'].plot(kind='bar', ax=axes[0, 1])
        axes[0, 1].set_title('Average Rating Count by Cluster')
        axes[0, 1].set_ylabel('Rating Count')

        # 高评分比例
        cluster_analysis['rating_5_ratio']['mean'].plot(kind='bar', ax=axes[1, 0])
        axes[1, 0].set_title('5-Star Rating Ratio by Cluster')
        axes[1, 0].set_ylabel('Ratio')

        # 低评分比例
        cluster_analysis['rating_1_ratio']['mean'].plot(kind='bar', ax=axes[1, 1])
        axes[1, 1].set_title('1-Star Rating Ratio by Cluster')
        axes[1, 1].set_ylabel('Ratio')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def plot_recommendation_performance(self, metrics_df, save_path=None):
        """绘制推荐性能对比"""
        plt.figure(figsize=(15, 5))

        metrics = ['RMSE', 'MAE', 'Precision@10', 'Recall@10', 'NDCG@10']

        for i, metric in enumerate(metrics, 1):
            plt.subplot(1, len(metrics), i)
            if metric in metrics_df.columns:
                metrics_df[metric].plot(kind='bar')
                plt.title(metric)
                plt.xticks(rotation=45)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def plot_factor_analysis(self, user_factors, item_factors, save_path=None):
        """绘制因子分析"""
        plt.figure(figsize=(15, 5))

        # 用户因子分布
        plt.subplot(1, 3, 1)
        plt.hist(user_factors.flatten(), bins=50, alpha=0.7)
        plt.xlabel('Factor Value')
        plt.ylabel('Frequency')
        plt.title('User Factor Distribution')

        # 物品因子分布
        plt.subplot(1, 3, 2)
        plt.hist(item_factors.flatten(), bins=50, alpha=0.7)
        plt.xlabel('Factor Value')
        plt.ylabel('Frequency')
        plt.title('Item Factor Distribution')

        # 因子重要性
        plt.subplot(1, 3, 3)
        user_factor_importance = np.var(user_factors, axis=0)
        plt.bar(range(len(user_factor_importance)), user_factor_importance)
        plt.xlabel('Factor Index')
        plt.ylabel('Variance')
        plt.title('Factor Importance (User Factors)')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()
