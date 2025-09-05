# data/data_loader.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class DataLoader:
    def __init__(self):
        self.ratings_df = None
        self.users_df = None
        self.items_df = None
        
    def load_movielens_data(self, ratings_path, movies_path=None, users_path=None):
        """加载MovieLens数据集"""
        self.ratings_df = pd.read_csv(ratings_path)
        if movies_path:
            self.items_df = pd.read_csv(movies_path)
        if users_path:
            self.users_df = pd.read_csv(users_path)
        return self.ratings_df
    
    def create_user_item_matrix(self):
        """创建用户-物品评分矩阵"""
        return self.ratings_df.pivot_table(
            index='userId', 
            columns='movieId', 
            values='rating'
        ).fillna(0)
    
    def train_test_split(self, test_size=0.2, random_state=42):
        """划分训练集和测试集"""
        return train_test_split(
            self.ratings_df, 
            test_size=test_size, 
            random_state=random_state,
            stratify=self.ratings_df['userId']
        )

# data/preprocessor.py
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

class DataPreprocessor:
    def __init__(self):
        pass
    
    def normalize_ratings(self, ratings_matrix, method='z_score'):
        """标准化评分矩阵"""
        if method == 'z_score':
            user_means = ratings_matrix.mean(axis=1)
            user_stds = ratings_matrix.std(axis=1)
            normalized = ratings_matrix.sub(user_means, axis=0).div(user_stds, axis=0)
        elif method == 'min_max':
            normalized = (ratings_matrix - ratings_matrix.min()) / (ratings_matrix.max() - ratings_matrix.min())
        else:
            normalized = ratings_matrix
        
        return normalized.fillna(0)
    
    def create_sparse_matrix(self, ratings_df):
        """创建稀疏矩阵以节省内存"""
        user_ids = ratings_df['userId'].astype('category').cat.codes
        item_ids = ratings_df['movieId'].astype('category').cat.codes
        ratings = ratings_df['rating'].values
        
        return csr_matrix((ratings, (user_ids, item_ids)))