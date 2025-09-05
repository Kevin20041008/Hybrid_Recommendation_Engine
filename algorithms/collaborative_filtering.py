# algorithms/collaborative_filtering.py
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize


class CollaborativeFiltering:
    """
    协同过滤：
    - method: 'user_based' | 'item_based'
    - similarity: 'cosine' | 'pearson' | 'jaccard'
    - k: 近邻数（简单实现里主要用于截断/可视化，不强制）
    - topk: item_based+cosine 的快速路径中，每个物品只保留 Top-K 相似物品
    - use_fast_item_cosine: 是否启用快速路径（仅在 item_based + cosine 生效）
    """

    def __init__(self,
                 method: str = 'user_based',
                 similarity: str = 'cosine',
                 k: int = 50,
                 topk: int = 100,
                 use_fast_item_cosine: bool = True):
        assert method in ('user_based', 'item_based')
        assert similarity in ('cosine', 'pearson', 'jaccard')
        self.method = method
        self.similarity = similarity
        self.k = k
        self.topk = topk
        self.use_fast_item_cosine = use_fast_item_cosine

        # 训练后赋值
        self.ratings_matrix: pd.DataFrame | None = None
        self.user_similarity: np.ndarray | None = None
        self.item_similarity: np.ndarray | None = None

        # 快速路径用（item_based + cosine）
        self._topk_idx: np.ndarray | None = None  # (n_items, topk) 相似物品索引
        self._topk_sim: np.ndarray | None = None  # (n_items, topk) 相似度

    # ----------------- 相似度矩阵计算 -----------------

    @staticmethod
    def _pearson_corr_matrix(X: np.ndarray, axis=1) -> np.ndarray:
        """
        计算 Pearson 相关系数矩阵。
        axis=1：对行（用户）做相关；axis=0：对列（物品）做相关。
        """
        A = X.astype(float)
        if axis == 0:
            A = A.T  # 转置后对行做相关 -> 列相关

        means = A.mean(axis=1, keepdims=True)
        A = A - means
        denom = np.linalg.norm(A, axis=1, keepdims=True)
        denom[denom == 0] = 1e-8
        A = A / denom
        S = A @ A.T
        S = np.clip(S, -1.0, 1.0)
        np.fill_diagonal(S, 1.0)
        return S

    @staticmethod
    def _jaccard_sim_matrix(X: np.ndarray, axis=1) -> np.ndarray:
        """
        二值化 Jaccard 相似度：|A∩B| / |A∪B|
        axis=1：用户；axis=0：物品
        """
        A = (X > 0).astype(np.int8)
        if axis == 0:
            A = A.T
        inter = A @ A.T
        row_sum = A.sum(axis=1, keepdims=True)
        union = row_sum + row_sum.T - inter
        union[union == 0] = 1  # 避免除 0
        S = inter / union
        np.fill_diagonal(S, 1.0)
        return S

    @staticmethod
    def _topk_item_cosine(ratings_dense: np.ndarray, topk: int):
        """
        稀疏化 + 列归一化 计算 物品×物品 余弦相似，并取每列 Top-K（去掉自身）。
        返回：indices (n_items, topk), sims (n_items, topk)
        """
        n_items = ratings_dense.shape[1]
        topk = min(topk, max(1, n_items - 1))
        R = csr_matrix(ratings_dense)      # 用户×物品
        R = normalize(R, axis=0)           # 列 L2 归一化 -> 余弦

        # 物品×物品相似
        S = (R.T @ R).tocsr()

        indices = np.zeros((n_items, topk), dtype=np.int32)
        sims = np.zeros((n_items, topk), dtype=np.float32)

        for j in range(n_items):
            start, end = S.indptr[j], S.indptr[j + 1]
            idxs = S.indices[start:end]
            vals = S.data[start:end]

            # 去自身
            mask = idxs != j
            idxs, vals = idxs[mask], vals[mask]

            if idxs.size:
                sel = np.argsort(vals)[-topk:]
                idxs, vals = idxs[sel], vals[sel]

            m = len(idxs)
            if m:
                indices[j, :m] = idxs
                sims[j, :m] = vals

        return indices, sims

    def _compute_similarity(self):
        X = self.ratings_matrix.values  # (n_users, n_items)

        if self.similarity == 'cosine':
            if self.method == 'user_based':
                self.user_similarity = cosine_similarity(X)
            else:
                self.item_similarity = cosine_similarity(X.T)

        elif self.similarity == 'pearson':
            if self.method == 'user_based':
                self.user_similarity = self._pearson_corr_matrix(X, axis=1)
            else:
                self.item_similarity = self._pearson_corr_matrix(X, axis=0)

        elif self.similarity == 'jaccard':
            if self.method == 'user_based':
                self.user_similarity = self._jaccard_sim_matrix(X, axis=1)
            else:
                self.item_similarity = self._jaccard_sim_matrix(X, axis=0)

    # ----------------- 训练 -----------------

    def fit(self, ratings_matrix: pd.DataFrame):
        """
        ratings_matrix: 行是 userId（索引），列是 movieId（列名），值是评分（0 表示未评分）
        """
        self.ratings_matrix = ratings_matrix

        # item_based + cosine 时使用快速 Top-K 路径
        if (self.method == 'item_based' and self.similarity == 'cosine'
                and self.use_fast_item_cosine):
            X = ratings_matrix.values
            self._topk_idx, self._topk_sim = self._topk_item_cosine(X, self.topk)
            self.user_similarity = None
            self.item_similarity = None
        else:
            # 其它情况计算完整相似度矩阵
            self._topk_idx, self._topk_sim = None, None
            self._compute_similarity()

        return self

    # ----------------- 评分预测 -----------------

    def _fallback_mean(self, user_ratings: np.ndarray | None, item_idx: int) -> float:
        """
        回退策略：优先用“用户均值”，否则“物品均值”，再否则 3.0。
        """
        if user_ratings is not None:
            nz = user_ratings[user_ratings > 0]
            if nz.size:
                return float(np.mean(nz))
        col = self.ratings_matrix.iloc[:, item_idx].values
        nz = col[col > 0]
        if nz.size:
            return float(np.mean(nz))
        return 3.0

    def predict(self, user_id, item_idx) -> float:
        """
        预测 user_id 对第 item_idx 列（某 movieId）的评分。
        - engine.py 会传入 user_id（标签）和 item_idx（列索引）
        """
        if self.ratings_matrix is None:
            raise ValueError("Model not fitted.")

        # 用户存在与否
        if user_id in self.ratings_matrix.index:
            u_idx = self.ratings_matrix.index.get_loc(user_id)
            user_ratings = self.ratings_matrix.iloc[u_idx].values
        else:
            user_ratings = None

        # ---------- 快速路径：item_based + cosine + Top-K ----------
        if (self.method == 'item_based' and self.similarity == 'cosine'
                and self._topk_idx is not None):
            if user_ratings is None:
                return self._fallback_mean(None, item_idx)

            neigh_items = self._topk_idx[item_idx]
            neigh_sims = self._topk_sim[item_idx]

            rated_mask = user_ratings[neigh_items] > 0
            if not rated_mask.any():
                return self._fallback_mean(user_ratings, item_idx)

            num = (user_ratings[neigh_items][rated_mask] * neigh_sims[rated_mask]).sum()
            den = np.abs(neigh_sims[rated_mask]).sum() + 1e-8
            pred = num / den
            return float(np.clip(pred, 1.0, 5.0))

        # ---------- 通用路径 ----------
        if self.method == 'user_based':
            if user_ratings is None:
                return self._fallback_mean(None, item_idx)

            sims = self.user_similarity[u_idx]  # 与所有用户相似度
            item_col = self.ratings_matrix.iloc[:, item_idx].values
            rated_mask = item_col > 0
            sims = sims * rated_mask
            if np.all(sims == 0):
                return self._fallback_mean(user_ratings, item_idx)

            num = np.dot(sims, item_col)
            den = np.abs(sims).sum() + 1e-8
            pred = num / den
            return float(np.clip(pred, 1.0, 5.0))

        else:  # item_based（非快速路径，如 pearson/jaccard 或关闭快速路径）
            if user_ratings is None:
                return self._fallback_mean(None, item_idx)

            sims = self.item_similarity[item_idx]  # 与目标物品相似度
            rated_mask = user_ratings > 0
            sims = sims * rated_mask
            if np.all(sims == 0):
                return self._fallback_mean(user_ratings, item_idx)

            num = np.dot(sims, user_ratings)
            den = np.abs(sims).sum() + 1e-8
            pred = num / den
            return float(np.clip(pred, 1.0, 5.0))

    # ----------------- 推荐 -----------------

    def recommend_items(self, user_id, n_recommendations=10):
        """
        返回 [(item_id, score), ...]，按分数降序
        """
        if self.ratings_matrix is None:
            raise ValueError("Model not fitted.")

        # 冷启动用户：直接热门物品
        if user_id not in self.ratings_matrix.index:
            popularity = self.ratings_matrix.mean(axis=0)
            topk = popularity.nlargest(n_recommendations)
            return [(int(item), float(score)) for item, score in topk.items()]

        u_idx = self.ratings_matrix.index.get_loc(user_id)
        user_row = self.ratings_matrix.iloc[u_idx]
        unrated_items = user_row[user_row == 0].index  # movieId 标签

        preds = []
        for item_id in unrated_items:
            i_idx = self.ratings_matrix.columns.get_loc(item_id)
            score = self.predict(user_id, i_idx)
            preds.append((int(item_id), float(score)))

        preds.sort(key=lambda x: x[1], reverse=True)
        return preds[:n_recommendations]
