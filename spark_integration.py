# 兼容未安装/未配置 Spark 的环境
try:
    from pyspark.sql import SparkSession
    from pyspark.ml.recommendation import ALS
    from pyspark.ml.evaluation import RegressionEvaluator
    from pyspark.ml.clustering import KMeans as SparkKMeans
    from pyspark.ml.feature import VectorAssembler, StandardScaler as SparkStandardScaler
    _HAS_SPARK = True
    _SPARK_IMPORT_ERROR = None
except Exception as e:
    _HAS_SPARK = False
    _SPARK_IMPORT_ERROR = e


class SparkRecommendationEngine:
    def __init__(self):
        if not _HAS_SPARK:
            raise RuntimeError(f"pyspark 未安装或环境未配置：{_SPARK_IMPORT_ERROR}")
        self.spark = SparkSession.builder \
            .appName("RecommendationEngine") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .getOrCreate()

        self.als_model = None
        self.kmeans_model = None

    def load_data_spark(self, ratings_path):
        ratings_df = self.spark.read.csv(ratings_path, header=True, inferSchema=True)
        return ratings_df

    def train_als_model(self, ratings_df, rank=50, maxIter=10, regParam=0.01):
        train_df, test_df = ratings_df.randomSplit([0.8, 0.2], seed=42)

        als = ALS(
            maxIter=maxIter,
            rank=rank,
            regParam=regParam,
            userCol="userId",
            itemCol="movieId",
            ratingCol="rating",
            coldStartStrategy="drop",
            seed=42
        )

        self.als_model = als.fit(train_df)

        predictions = self.als_model.transform(test_df)
        evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
        rmse = evaluator.evaluate(predictions)
        return self.als_model, rmse

    def spark_user_clustering(self, ratings_df, k=5):
        user_features = ratings_df.groupBy("userId").agg(
            {"rating": "mean", "movieId": "count"}
        ).withColumnRenamed("avg(rating)", "avg_rating") \
         .withColumnRenamed("count(movieId)", "rating_count")

        assembler = VectorAssembler(inputCols=["avg_rating", "rating_count"], outputCol="features")
        feature_df = assembler.transform(user_features)

        scaler = SparkStandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=True)
        scaler_model = scaler.fit(feature_df)
        scaled_df = scaler_model.transform(feature_df)

        kmeans = SparkKMeans(featuresCol="scaledFeatures", predictionCol="cluster", k=k, seed=42)
        self.kmeans_model = kmeans.fit(scaled_df)
        clustered_df = self.kmeans_model.transform(scaled_df)
        return clustered_df

    def generate_recommendations_spark(self, user_id, num_recommendations=10):
        if self.als_model is None:
            raise ValueError("ALS model not trained")
        user_df = self.spark.createDataFrame([(user_id,)], ["userId"])
        user_recommendations = self.als_model.recommendForUserSubset(user_df, num_recommendations)
        return user_recommendations.collect()

    def batch_recommendations_spark(self, num_recommendations=10):
        if self.als_model is None:
            raise ValueError("ALS model not trained")
        return self.als_model.recommendForAllUsers(num_recommendations)

    def stop_spark(self):
        self.spark.stop()
