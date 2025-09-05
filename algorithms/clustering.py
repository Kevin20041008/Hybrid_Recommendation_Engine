# algorithms/clustering.py
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

class UserClustering:
    def __init__(self, method='kmeans', n_clusters=5):
        self.method = method
        self.n_clusters = n_clusters
        self.model = None
        self.scaler = StandardScaler()
        self.cluster_labels = None
        
    def extract_user_features(self, ratings_matrix):
        """提取用户特征"""
        features = pd.DataFrame(index=ratings_matrix.index)
        
        # 基本统计特征
        features['avg_rating'] = ratings_matrix.mean(axis=1)
        features['rating_count'] = (ratings_matrix > 0).sum(axis=1)
        features['rating_std'] = ratings_matrix.std(axis=1)
        
        # 评分分布特征
        for rating in [1, 2, 3, 4, 5]:
            features[f'rating_{rating}_ratio'] = (ratings_matrix == rating).sum(axis=1) / features['rating_count']
        
        # 活跃度特征
        features['activity_level'] = pd.cut(features['rating_count'], 
                                          bins=5, labels=['low', 'medium_low', 'medium', 'medium_high', 'high'])
        
        # 将类别变量转换为数值
        features['activity_level'] = features['activity_level'].cat.codes
        
        return features.fillna(0)
    
    def fit(self, user_features):
        """训练聚类模型"""
        # 特征标准化
        scaled_features = self.scaler.fit_transform(user_features)
        
        if self.method == 'kmeans':
            self.model = KMeans(n_clusters=self.n_clusters, random_state=42)
        elif self.method == 'dbscan':
            self.model = DBSCAN(eps=0.5, min_samples=5)
        
        self.cluster_labels = self.model.fit_predict(scaled_features)
        return self
    
    def predict(self, user_features):
        """预测新用户的聚类标签"""
        scaled_features = self.scaler.transform(user_features)
        return self.model.predict(scaled_features)
    
    def analyze_clusters(self, user_features):
        """分析聚类结果"""
        cluster_df = user_features.copy()
        cluster_df['cluster'] = self.cluster_labels
        
        cluster_analysis = cluster_df.groupby('cluster').agg({
            'avg_rating': ['mean', 'std'],
            'rating_count': ['mean', 'std'],
            'rating_std': ['mean', 'std'],
            'rating_5_ratio': 'mean',
            'rating_1_ratio': 'mean'
        }).round(3)
        
        return cluster_analysis
    
    def visualize_clusters(self, user_features, save_path=None):
        """可视化聚类结果"""
        # 使用PCA降维到2D
        pca = PCA(n_components=2)
        scaled_features = self