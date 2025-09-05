# utils/metrics.py
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

class RecommendationMetrics:
    def __init__(self):
        pass
    
    def rmse(self, y_true, y_pred):
        """均方根误差"""
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    def mae(self, y_true, y_pred):
        """平均绝对误差"""
        return mean_absolute_error(y_true, y_pred)
    
    def precision_at_k(self, recommended_items, relevant_items, k):
        """Precision@K"""
        if k == 0:
            return 0
        
        recommended_k = recommended_items[:k]
        relevant_recommended = len(set(recommended_k) & set(relevant_items))
        return relevant_recommended / k
    
    def recall_at_k(self, recommended_items, relevant_items, k):
        """Recall@K"""
        if len(relevant_items) == 0:
            return 0
        
        recommended_k = recommended_items[:k]
        relevant_recommended = len(set(recommended_k) & set(relevant_items))
        return relevant_recommended / len(relevant_items)
    
    def f1_at_k(self, recommended_items, relevant_items, k):
        """F1@K"""
        precision = self.precision_at_k(recommended_items, relevant_items, k)
        recall = self.recall_at_k(recommended_items, relevant_items, k)
        
        if precision + recall == 0:
            return 0
        return 2 * (precision * recall) / (precision + recall)
    
    def ndcg_at_k(self, recommended_items, relevant_items, k):
        """归一化折损累积增益@K"""
        def dcg_at_k(relevance_scores, k):
            relevance_scores = np.asarray(relevance_scores)[:k]
            if relevance_scores.size:
                return np.sum(np.divide(np.power(2, relevance_scores) - 1, 
                                      np.log2(np.arange(relevance_scores.size, dtype=np.float32) + 2)))
            return 0.
        
        # 构建相关性分数
        relevance_scores = [1 if item in relevant_items else 0 for item in recommended_items[:k]]
        ideal_relevance_scores = sorted([1] * len(relevant_items) + [0] * (k - len(relevant_items)), reverse=True)
        
        dcg = dcg_at_k(relevance_scores, k)
        idcg = dcg_at_k(ideal_relevance_scores, k)
        
        if idcg == 0:
            return 0
        return dcg / idcg
    
    def coverage(self, all_recommendations, all_items):
        """覆盖率"""
        unique_recommended = set()
        for recommendations in all_recommendations:
            unique_recommended.update(recommendations)
        
        return len(unique_recommended) / len(all_items)
    
    def diversity(self, recommendations, item_features):
        """多样性"""
        if len(recommendations) < 2:
            return 0
        
        total_distance = 0
        count = 0
        
        for i in range(len(recommendations)):
            for j in range(i + 1, len(recommendations)):
                item1_features = item_features.get(recommendations[i], np.array([]))
                item2_features = item_features.get(recommendations[j], np.array([]))
                
                if len(item1_features) > 0 and len(item2_features) > 0:
                    distance = np.linalg.norm(item1_features - item2_features)
                    total_distance += distance
                    count += 1
        
        return total_distance / count if count > 0 else 0