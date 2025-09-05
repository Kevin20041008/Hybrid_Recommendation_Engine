# algorithms/matrix_factorization.py
import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
from sklearn.decomposition import NMF
import warnings
warnings.filterwarnings('ignore')


class MatrixFactorization:
    def __init__(self, method='svd', n_factors=50, regularization=0.01, learning_rate=0.01, n_epochs=100):
        self.method = method
        self.n_factors = n_factors
        self.regularization = regularization
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.user_factors = None
        self.item_factors = None
        self.user_bias = None
        self.item_bias = None
        self.global_bias = None

    def fit(self, ratings_matrix):
        """训练矩阵分解模型"""
        if self.method == 'svd':
            self._fit_svd(ratings_matrix)
        elif self.method == 'nmf':
            self._fit_nmf(ratings_matrix)
        elif self.method == 'sgd':
            self._fit_sgd(ratings_matrix)
        return self

    def _fit_svd(self, ratings_matrix):
        """SVD分解"""
        filled_matrix = ratings_matrix.copy()
        user_mean = np.mean(filled_matrix, axis=1)
        filled_matrix = filled_matrix.sub(user_mean, axis=0).fillna(0)

        U, sigma, Vt = svds(filled_matrix.values, k=self.n_factors)

        self.user_factors = U
        self.item_factors = Vt.T
        self.singular_values = sigma

    def _fit_nmf(self, ratings_matrix):
        """非负矩阵分解"""
        positive_matrix = np.maximum(ratings_matrix.values, 0)
        nmf_model = NMF(n_components=self.n_factors, random_state=42)
        self.user_factors = nmf_model.fit_transform(positive_matrix)
        self.item_factors = nmf_model.components_.T

    def _fit_sgd(self, ratings_matrix):
        """随机梯度下降训练"""
        n_users, n_items = ratings_matrix.shape

        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))
        self.user_bias = np.zeros(n_users)
        self.item_bias = np.zeros(n_items)
        self.global_bias = ratings_matrix.values[ratings_matrix.values > 0].mean()

        user_indices, item_indices = np.where(ratings_matrix.values > 0)
        ratings = ratings_matrix.values[user_indices, item_indices]

        for epoch in range(self.n_epochs):
            for i, (user_idx, item_idx, rating) in enumerate(zip(user_indices, item_indices, ratings)):
                prediction = (self.global_bias +
                              self.user_bias[user_idx] +
                              self.item_bias[item_idx] +
                              np.dot(self.user_factors[user_idx], self.item_factors[item_idx]))
                error = rating - prediction

                self.user_bias[user_idx] += self.learning_rate * (error - self.regularization * self.user_bias[user_idx])
                self.item_bias[item_idx] += self.learning_rate * (error - self.regularization * self.item_bias[item_idx])

                user_factors_old = self.user_factors[user_idx].copy()
                self.user_factors[user_idx] += self.learning_rate * (
                    error * self.item_factors[item_idx] - self.regularization * self.user_factors[user_idx])
                self.item_factors[item_idx] += self.learning_rate * (
                    error * user_factors_old - self.regularization * self.item_factors[item_idx])

    def predict(self, user_idx, item_idx):
        """预测评分"""
        if self.method == 'svd':
            return self._predict_svd(user_idx, item_idx)
        elif self.method == 'sgd':
            return self._predict_sgd(user_idx, item_idx)
        else:  # nmf
            return self._predict_nmf(user_idx, item_idx)

    def _predict_svd(self, user_idx, item_idx):
        prediction = np.dot(self.user_factors[user_idx] * self.singular_values,
                            self.item_factors[item_idx])
        return np.clip(prediction, 1, 5)

    def _predict_sgd(self, user_idx, item_idx):
        prediction = (self.global_bias +
                      self.user_bias[user_idx] +
                      self.item_bias[item_idx] +
                      np.dot(self.user_factors[user_idx], self.item_factors[item_idx]))
        return np.clip(prediction, 1, 5)

    def _predict_nmf(self, user_idx, item_idx):
        prediction = np.dot(self.user_factors[user_idx], self.item_factors[item_idx])
        return np.clip(prediction, 1, 5)

    def recommend_items(self, user_idx, ratings_matrix, n_recommendations=10):
        """推荐物品（修复 item 索引错位 bug）"""
        user_ratings = ratings_matrix.iloc[user_idx]
        unrated_items = user_ratings[user_ratings == 0].index  # movieId 标签

        predictions = []
        for item_id in unrated_items:
            true_item_idx = ratings_matrix.columns.get_loc(item_id)  # 真实列索引
            pred_rating = self.predict(user_idx, true_item_idx)
            predictions.append((item_id, pred_rating))

        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n_recommendations]
