# main.py
import pandas as pd
import numpy as np
from data.data_loader import DataLoader

from engine import RecommendationEngine
from spark_integration import SparkRecommendationEngine
from sklearn.model_selection import train_test_split  # ⇦ 前置到顶部
import warnings
warnings.filterwarnings('ignore')


def main():
    print("=== 推荐系统引擎演示 ===")

    # 1. 数据加载
    print("\n1. 加载数据...")
    data_loader = DataLoader()

    # 模拟数据
    np.random.seed(42)
    n_users, n_items = 1000, 500
    n_ratings = 10000

    user_ids = np.random.choice(range(1, n_users + 1), n_ratings)
    movie_ids = np.random.choice(range(1, n_items + 1), n_ratings)
    ratings = np.random.choice([1, 2, 3, 4, 5], n_ratings,
                               p=[0.1, 0.1, 0.2, 0.3, 0.3])

    ratings_df = pd.DataFrame({
        'userId': user_ids,
        'movieId': movie_ids,
        'rating': ratings
    }).drop_duplicates(['userId', 'movieId'])

    print(f"数据集大小: {len(ratings_df)} 条评分")
    print(f"用户数: {ratings_df['userId'].nunique()}")
    print(f"物品数: {ratings_df['movieId'].nunique()}")

    # 2. 初始化推荐引擎
    print("\n2. 初始化推荐引擎...")
    engine = RecommendationEngine()
    engine.load_data(ratings_df)

    # 3. 训练协同过滤模型
    print("\n3. 训练协同过滤模型...")
    cf_model = engine.train_collaborative_filtering(method='user_based', similarity='cosine')
    print("协同过滤模型训练完成")

    # 4. 训练聚类模型
    print("\n4. 训练用户聚类模型...")
    clustering_model = engine.train_clustering(method='kmeans', n_clusters=5)
    cluster_analysis = clustering_model.analyze_clusters(
        clustering_model.extract_user_features(engine.ratings_matrix)
    )
    print("聚类模型训练完成")
    print("聚类分析结果:")
    print(cluster_analysis)

    # 5. 训练矩阵分解模型
    print("\n5. 训练矩阵分解模型...")
    mf_model = engine.train_matrix_factorization(method='svd', n_factors=50)
    print("矩阵分解模型训练完成")

    # 6. 生成推荐
    print("\n6. 生成推荐...")
    test_user_id = ratings_df['userId'].iloc[0]

    cf_recommendations = cf_model.recommend_items(test_user_id, n_recommendations=10)
    print(f"\n协同过滤推荐 (用户 {test_user_id}):")
    for i, (item_id, score) in enumerate(cf_recommendations[:5], 1):
        print(f"{i}. 物品 {item_id}: 预测评分 {score:.3f}")

    cluster_recommendations = engine.cluster_based_recommend(test_user_id, n_recommendations=10)
    print(f"\n聚类推荐 (用户 {test_user_id}):")
    for i, (item_id, score) in enumerate(cluster_recommendations[:5], 1):
        print(f"{i}. 物品 {item_id}: 平均评分 {score:.3f}")

    hybrid_recommendations = engine.hybrid_recommend(
        test_user_id,
        n_recommendations=10,
        weights={'cf': 0.4, 'mf': 0.6}
    )
    print(f"\n混合推荐 (用户 {test_user_id}):")
    for i, (item_id, score) in enumerate(hybrid_recommendations[:5], 1):
        print(f"{i}. 物品 {item_id}: 综合评分 {score:.3f}")

    # 7. 模型评估
    print("\n7. 模型评估...")
    train_df, test_df = train_test_split(ratings_df, test_size=0.2, random_state=42)

    engine_eval = RecommendationEngine()
    engine_eval.load_data(train_df)
    engine_eval.train_collaborative_filtering(method='user_based', similarity='cosine')
    engine_eval.train_matrix_factorization(method='svd', n_factors=50)

    cf_metrics = engine_eval.evaluate_model(test_df, model_type='cf')
    print(f"协同过滤评估结果: {cf_metrics}")

    mf_metrics = engine_eval.evaluate_model(test_df, model_type='mf')
    print(f"矩阵分解评估结果: {mf_metrics}")

    # 8. 可视化结果
    print("\n8. 生成可视化...")
    visualizer = engine.visualizer
    visualizer.plot_rating_distribution(ratings_df, save_path='rating_distribution.png')
    visualizer.plot_cluster_analysis(cluster_analysis, save_path='cluster_analysis.png')

    # 9. Spark MLlib演示
    print("\n9. Spark MLlib演示...")
    try:
        spark_engine = SparkRecommendationEngine()
        spark_df = spark_engine.spark.createDataFrame(ratings_df)
        als_model, rmse = spark_engine.train_als_model(spark_df, rank=50, maxIter=10)
        print(f"Spark ALS模型 RMSE: {rmse:.4f}")

        spark_recommendations = spark_engine.generate_recommendations_spark(test_user_id, 5)
        print(f"\nSpark ALS推荐 (用户 {test_user_id}):")
        for rec in spark_recommendations:
            recommendations = rec['recommendations']
            for i, item_rec in enumerate(recommendations, 1):
                print(f"{i}. 物品 {item_rec['movieId']}: 预测评分 {item_rec['rating']:.3f}")

        clustered_users = spark_engine.spark_user_clustering(spark_df, k=5)
        print(f"\nSpark用户聚类完成，聚类数: 5")
        spark_engine.stop_spark()

    except Exception as e:
        print(f"Spark演示出错: {e}")
        print("请确保已正确安装和配置Spark")

    print("\n=== 推荐系统演示完成 ===")


def demo_advanced_features():
    """（保持你原实现）此处略——无需为导入问题修改"""
    pass


def performance_optimization_demo():
    """（保持你原实现）此处略——推荐把 redis 作为可选依赖再接入"""
    pass


if __name__ == "__main__":
    main()
    # 需要的话再打开下面两个 demo
    # demo_advanced_features()
    # performance_optimization_demo()
