# engine.py
import numpy as np
import pandas as pd
from algorithms.collaborative_filtering import CollaborativeFiltering
from algorithms.clustering import UserClustering
from algorithms.matrix_factorization import MatrixFactorization
from utils.metrics import RecommendationMetrics
from utils.visualization import RecommendationVisualizer

class RecommendationEngine:
    def __init__(self):
        self.cf_model = None
        self.clustering_model = None
        self.mf_model = None
        self.ratings_matrix = None
        self.user_clusters = None
        self.metrics = RecommendationMetrics()
        self.visualizer = RecommendationVisualizer()
        
    def load_data(self, ratings_df):
        """加载数据"""
        self.ratings_df = ratings_df
        self.ratings_matrix = ratings_df.pivot_table(
            index='userId', columns='movieId', values='rating'
        ).fillna(0)
        
    def train_collaborative_filtering(self, method='user_based', similarity='cosine'):
        """训练协同过滤模型"""
        self.cf_model = CollaborativeFiltering(method=method, similarity=similarity)
        self.cf_model.fit(self.ratings_matrix)
        return self.cf_model
        
    def train_clustering(self, method='kmeans', n_clusters=5):
        """训练聚类模型"""
        self.clustering_model = UserClustering(method=method, n_clusters=n_clusters)
        user_features = self.clustering_model.extract_user_features(self.ratings_matrix)
        self.clustering_model.fit(user_features)
        self.user_clusters = self.clustering_model.cluster_labels
        return self.clustering_model
        
    def train_matrix_factorization(self, method='svd', n_factors=50):
        """训练矩阵分解模型"""
        self.mf_model = MatrixFactorization(method=method, n_factors=n_factors)
        self.mf_model.fit(self.ratings_matrix)
        return self.mf_model
        
    def hybrid_recommend(self, user_id, n_recommendations=10, weights=None):
        """混合推荐"""
        if weights is None:
            weights = {'cf': 0.4, 'mf': 0.6}  # 默认权重
            
        recommendations = {}
        
        # 协同过滤推荐
        if self.cf_model and 'cf' in weights:
            cf_recs = self.cf_model.recommend_items(user_id, n_recommendations * 2)
            for item_id, score in cf_recs:
                recommendations[item_id] = recommendations.get(item_id, 0) + weights['cf'] * score
        
        # 矩阵分解推荐
        if self.mf_model and 'mf' in weights:
            if user_id in self.ratings_matrix.index:
                user_idx = self.ratings_matrix.index.get_loc(user_id)
                mf_recs = self.mf_model.recommend_items(user_idx, self.ratings_matrix, n_recommendations * 2)
                for item_id, score in mf_recs:
                    recommendations[item_id] = recommendations.get(item_id, 0) + weights['mf'] * score
        
        # 按分数排序
        sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        return sorted_recommendations[:n_recommendations]
        
    def cluster_based_recommend(self, user_id, n_recommendations=10):
        """基于聚类的推荐"""
        if self.clustering_model is None or self.user_clusters is None:
            raise ValueError("Clustering model not trained")
            
        if user_id not in self.ratings_matrix.index:
            return self.get_popular_items(n_recommendations)
            
        user_idx = self.ratings_matrix.index.get_loc(user_id)
        user_cluster = self.user_clusters[user_idx]
        
        # 找到同一聚类的用户
        cluster_users = np.where(self.user_clusters == user_cluster)[0]
        cluster_user_ids = [self.ratings_matrix.index[i] for i in cluster_users if i != user_idx]
        
        # 计算聚类内平均评分
        cluster_ratings = self.ratings_matrix.iloc[cluster_users].mean(axis=0)
        
        # 排除用户已评分的物品
        user_ratings = self.ratings_matrix.loc[user_id]
        unrated_items = user_ratings[user_ratings == 0].index
        
        # 推荐聚类内高评分且用户未评分的物品
        cluster_recommendations = cluster_ratings[unrated_items].nlargest(n_recommendations)
        
        return [(item, score) for item, score in cluster_recommendations.items()]
        
    def get_popular_items(self, n_recommendations=10):
        """获取热门物品"""
        popularity = self.ratings_matrix.mean(axis=0)
        return [(item, score) for item, score in popularity.nlargest(n_recommendations).items()]
        
    def evaluate_model(self, test_df, model_type='cf'):
        """评估模型性能"""
        predictions = []
        actuals = []
        
        for _, row in test_df.iterrows():
            user_id = row['userId']
            item_id = row['movieId']
            actual_rating = row['rating']
            
            try:
                if model_type == 'cf' and self.cf_model:
                    if item_id in self.ratings_matrix.columns:
                        item_idx = self.ratings_matrix.columns.get_loc(item_id)
                        predicted_rating = self.cf_model.predict(user_id, item_idx)
                    else:
                        continue
                elif model_type == 'mf' and self.mf_model:
                    if user_id in self.ratings_matrix.index and item_id in self.ratings_matrix.columns:
                        user_idx = self.ratings_matrix.index.get_loc(user_id)
                        item_idx = self.ratings_matrix.columns.get_loc(item_id)
                        predicted_rating = self.mf_model.predict(user_idx, item_idx)
                    else:
                        continue
                else:
                    continue
                    
                predictions.append(predicted_rating)
                actuals.append(actual_rating)
                
            except Exception as e:
                continue
        
        if len(predictions) == 0:
            return {}
            
        # 计算评估指标
        rmse = self.metrics.rmse(actuals, predictions)
        mae = self.metrics.mae(actuals, predictions)
        
        return {
            'RMSE': rmse,
            'MAE': mae,
            'predictions_count': len(predictions)
        }