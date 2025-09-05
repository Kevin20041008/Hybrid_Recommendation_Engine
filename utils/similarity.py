import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

def topk_item_cosine(ratings_dense, topk=100):
    """
    ratings_dense: 用户×物品 的 dense numpy (0 表示未评分)
    return: indices[np.array(n_items, topk)], sims[np.array(n_items, topk)]
    """
    R = csr_matrix(ratings_dense)               # 稀疏
    R = normalize(R, axis=0)                    # 列 L2 归一化 -> 余弦
    S = (R.T @ R).tocoo()                       # 物品×物品 相似
    n_items = ratings_dense.shape[1]
    # 取每列 topk（含自身）
    S = S.tocsr()
    indices = np.zeros((n_items, topk), dtype=np.int32)
    sims    = np.zeros((n_items, topk), dtype=np.float32)
    for j in range(n_items):
        start, end = S.indptr[j], S.indptr[j+1]
        idxs, vals = S.indices[start:end], S.data[start:end]
        # 去掉自身并取 topk
        mask = idxs != j
        idxs, vals = idxs[mask], vals[mask]
        if idxs.size:
            top = np.argsort(vals)[-topk:]
            idxs, vals = idxs[top], vals[top]
        indices[j, :len(idxs)] = idxs
        sims[j, :len(vals)] = vals
    return indices, sims
