# config.py
class Config:
    # 数据配置
    DATA_PATH = "data/"
    RATINGS_FILE = "ratings.csv"
    MOVIES_FILE = "movies.csv"
    USERS_FILE = "users.csv"
    
    # 模型配置
    CF_METHOD = "user_based"  # user_based, item_based
    CF_SIMILARITY = "cosine"  # cosine, pearson, jaccard
    CF_K_NEIGHBORS = 50
    
    CLUSTERING_METHOD = "kmeans"  # kmeans, dbscan
    N_CLUSTERS = 5
    
    MF_METHOD = "svd"  # svd, nmf, sgd
    MF_N_FACTORS = 50
    MF_REGULARIZATION = 0.01
    MF_LEARNING_RATE = 0.01
    MF_N_EPOCHS = 100
    
    # Spark配置
    SPARK_APP_NAME = "RecommendationEngine"
    SPARK_MASTER = "local[*]"
    ALS_RANK = 50
    ALS_MAX_ITER = 10
    ALS_REG_PARAM = 0.01
    
    # 推荐配置
    DEFAULT_N_RECOMMENDATIONS = 10
    HYBRID_WEIGHTS = {"cf": 0.4, "mf": 0.6}
    
    # 缓存配置
    CACHE_TTL = 3600  # 1小时
    USE_REDIS = False
    REDIS_HOST = "localhost"
    REDIS_PORT = 6379
    REDIS_DB = 0
    
    # 性能配置
    BATCH_SIZE = 100
    N_PRECOMPUTE_SIMILAR_ITEMS = 50
    
    # 评估配置
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    EVALUATION_METRICS = ["rmse", "mae", "precision_k", "recall_k", "ndcg_k"]
    EVALUATION_K = [5, 10, 20]