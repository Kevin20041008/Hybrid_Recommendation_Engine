# 推荐系统引擎（Recsys Demo）— README

一个可运行、可扩展的**离线推荐系统教学项目**，包含协同过滤（User/Item-Based）、矩阵分解（SVD/NMF/SGD）与可选的 Spark MLlib（ALS、KMeans）演示。

------

## 主要算法

- **协同过滤（CF）**
  - `user_based` / `item_based` 两种模式
  - 相似度：`cosine` / `pearson` / `jaccard`
  - ✅ **Item-based 余弦 Top-K 快速路径**（默认打开）：仅保留每个物品 Top-K 相似邻居，预测速度显著提升
- **矩阵分解（MF）**
  - `svd` / `nmf` / `sgd` 三种实现
  - SVD 训练带**去均值**，预测时加回均值，效果更稳
- **聚类 & 可视化**
  - 用户聚类分析（KMeans/Spark KMeans）
  - 评分分布、相似度热力图、聚类统计、因子可视化等
- **评估指标**
  - RMSE、MAE、Precision@K、Recall@K、NDCG@K、覆盖率、推荐多样性
- **Spark 演示（可选）**
  - Spark ALS 训练与预测、用户聚类
  - 自动兜底：未安装 PySpark 时不阻塞主流程
- **工程化实践**
  - 统一项目结构、`.vscode` 智能补全、`pyproject.toml` 可编辑安装
  - 可选 Redis 缓存、批处理与预计算策略（示例在 `main.py`）

------

## 目录结构

```
recsys-demo/
├─ algorithms/
│  ├─ __init__.py
│  ├─ collaborative_filtering.py          
│  ├─ clustering.py                       # 用户聚类
│  └─ matrix_factorization.py             # SVD/NMF/SGD（含去均值修正）
├─ data/
│  ├─ __init__.py
│  ├─ data_loader.py
│  └─ preprocessor.py                     # DataPreprocessor：简单清洗/规范化
├─ utils/
│  ├─ __init__.py
│  ├─ metrics.py                          # RMSE/MAE/Precision@K/Recall@K/NDCG@K 等
│  └─ visualization.py                    # seaborn 可选；无则回退 matplotlib
├─ .vscode/
│  └─ settings.json                       # 让 Pylance 识别项目根
├─ config.py
├─ deploy.py                              # 环境/依赖检查示例
├─ engine.py                              # 训练与组合推荐的统一入口
├─ main.py                                # 演示脚本（含 Spark 兜底 try/except）
├─ spark_integration.py                   # Spark ALS / KMeans（可选）
├─ pyproject.toml                         # 可编辑安装（推荐）
└─ requirements.txt                       # 基础依赖（可选）
```

------

## 环境要求

- Python **3.8+**
- 基础依赖（必须）：
  - `numpy` `pandas` `scikit-learn` `scipy` `matplotlib`
- 可选依赖：
  - 可视化增强：`seaborn`
  - Spark 演示：`pyspark`（需要本地 Java 8/11 & `SPARK_HOME`）
  - 缓存演示：`redis`（及本地 Redis 服务）

------

## 安装与运行

### 方式一（推荐）：可编辑安装

```
# 进入项目根目录
python -m pip install -e .
python -m pip install -r requirements.txt     # 安装基础依赖
# 可选
python -m pip install seaborn pyspark redis
```
或者：

```
python -m pip install -r requirements.txt
# 可选：seaborn / pyspark / redis
```

### 运行

```
python main.py
```

> `main.py` 默认使用**合成数据**（1000 用户 × 500 物品，1 万条评分）进行演示。
>  如需使用真实数据，将 `ratings_df` 替换为包含 `userId,movieId,rating` 三列的 DataFrame 即可。

------

## 快速开始

`main.py` 主要步骤：

1. 生成/加载评分数据 → `engine.load_data(df)`
2. 训练协同过滤（默认：User-based Cosine）
3. 训练用户聚类 → 输出每簇统计
4. 训练矩阵分解（SVD）
5. 生成三类推荐：CF、Cluster-based、Hybrid
6. 评估：RMSE/MAE
7. 可视化：评分分布、聚类统计图
8. Spark 演示（若已安装）：ALS 训练 / 推荐、KMeans 聚类

------

## 使用真实数据

数据格式（CSV）：

```
userId,movieId,rating
1,10,5
1,50,4
2,10,3
...
```

示例：

```
import pandas as pd
from engine import RecommendationEngine

ratings_df = pd.read_csv("data/ratings.csv")
engine = RecommendationEngine()
engine.load_data(ratings_df)
cf = engine.train_collaborative_filtering(method='item_based', similarity='cosine')  # 启用快速路径
recs = cf.recommend_items(user_id=1, n_recommendations=10)
print(recs)
```

------

## 配置说明（摘自 `config.py`）

- 协同过滤：`CF_METHOD`（`user_based`/`item_based`），`CF_SIMILARITY`（`cosine`/`pearson`/`jaccard`），`CF_K_NEIGHBORS`
- 聚类：`CLUSTERING_METHOD`（`kmeans`/`dbscan`），`N_CLUSTERS`
- 矩阵分解：`MF_METHOD`（`svd`/`nmf`/`sgd`），`MF_N_FACTORS` 等
- Spark：`SPARK_APP_NAME`、`ALS_RANK`、`ALS_MAX_ITER`…
- 推荐：`DEFAULT_N_RECOMMENDATIONS`、`HYBRID_WEIGHTS`

> 代码里默认通过 `engine.train_*` 传参控制；需要全局配置时可在 `config.py` 调整。

------

## Spark 演示（可选）

1. 安装依赖：

   ```
   python -m pip install pyspark
   ```

2. 安装 Java 8/11 并配置环境（Windows 用户注意 `JAVA_HOME`）

3. （可选）配置 `SPARK_HOME`

4. 运行 `python main.py`，若未就绪将自动输出提示并跳过 Spark 演示

------

## 常见问题（FAQ）

- **VSCode 提示“无法解析导入 …”**
   确保使用的是正确解释器（右下角选择），并在 `.vscode/settings.json` 中包含：

  ```
  { "python.analysis.extraPaths": ["./"] }
  ```

  建议执行一次 **Developer: Reload Window**。
   或使用 `python -m pip install -e .` 可编辑安装。

- **`ImportError: cannot import name 'pearson_correlation'`**
   已修复：我们在 `algorithms/collaborative_filtering.py` 内部实现了 Pearson 相关系数矩阵，不再依赖 sklearn 中不存在的函数。

- **`DataPreprocessor` 找不到**
   项目已提供 `data/preprocessor.py` 的类定义。如仍报错，请确认文件名与类名大小写。

- **Spark 报错**
   没装 PySpark：安装 `pyspark`；
   没装 Java 或 `SPARK_HOME`：按上文进行安装与环境变量配置。
   项目对 Spark 已做 try/except 兜底，不影响主流程。

------

## 性能与效果优化建议

- **Item-based 余弦 Top-K 快速路径**（已内置）：将 `method='item_based', similarity='cosine'` 时自动启用，Top-K 默认 100
- **评估协议**：从 RMSE/MAE 扩展到 Top-K（Precision@K / Recall@K / NDCG@K）
- **重排（MMR）**：在候选上用最大边际相关性提升多样性/新颖性
- **隐式反馈ALS**：浏览/点击/购买等数据更适合（可接 `implicit` 库）
- **统一索引映射**：将 `userId`/`movieId` 映射为连续整数，便于稀疏矩阵与大规模计算

> 需要完整代码模板（MMR/implicit/tuning 等），告诉我你要的方向，我可以直接补充文件。

------

## 许可证

MIT（可按需修改）

------

## 致谢 & 反馈

- 若你在集成到自己数据/服务中遇到问题，或希望加入更多算法（如 BPR、LightFM、双塔召回、重排模型等），直接提需求，我会给出对应代码与接入说明。
