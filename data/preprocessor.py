# data/preprocessor.py
import pandas as pd
import numpy as np

class DataPreprocessor:
    """评分数据的简单预处理工具"""

    def __init__(self):
        pass

    def preprocess_ratings(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        规范列名/类型，去掉缺失和异常评分，返回新 DataFrame
        期望列：userId, movieId, rating
        """
        df = df.copy()

        # 统一列名（容错一些常见写法）
        rename_map = {
            'user_id': 'userId',
            'movie_id': 'movieId',
            'score': 'rating',
            'rate': 'rating'
        }
        df.rename(columns=rename_map, inplace=True)

        # 只保留需要的列
        keep_cols = ['userId', 'movieId', 'rating']
        df = df[[c for c in keep_cols if c in df.columns]]

        # 去缺失
        df.dropna(subset=['userId', 'movieId', 'rating'], inplace=True)

        # 类型转换
        df['userId'] = df['userId'].astype(int)
        df['movieId'] = df['movieId'].astype(int)
        df['rating'] = df['rating'].astype(float)

        # 合理评分范围裁剪（按1~5）
        df = df[(df['rating'] >= 1.0) & (df['rating'] <= 5.0)]

        # 去重复
        df = df.drop_duplicates(['userId', 'movieId'])

        return df
