# Efficient Movie Recommendation System  
# 高效电影推荐系统

### English

Course project comparing **recommendation accuracy** and **computational efficiency** on the [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/) dataset.

### 中文

基于 [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/) 数据集，对比 **推荐精度** 与 **计算效率** 的课程项目。

---

## Overview · 项目概述

### English

We build an end-to-end offline recommender pipeline: **EDA and hypothesis tests** → **non-personalized baselines** → **item-based collaborative filtering** → **matrix factorization (SVD / SVD++ via [Surprise](https://surpriselib.com/))** → **hybrid** content (TF-IDF over genres) + collaborative signals.  
We evaluate both **rating prediction** (RMSE, MAE) and **top-K recommendation quality** (Precision@K, Recall@K, hit rate), and study **Python-side acceleration** (NumPy vectorization, Numba JIT) plus **Optuna** for Bayesian hyperparameter search on SVD.

### 中文

构建完整离线推荐流程：**探索性分析与假设检验** → **非个性化 baseline** → **基于物品的协同过滤（Item-Based CF）** → **矩阵分解（SVD / SVD++，[Surprise](https://surpriselib.com/)）** → **混合模型**（类型片 TF-IDF 等内容特征 + 协同信号）。  
同时评估 **评分预测**（RMSE、MAE）与 **Top-K 推荐质量**（Precision@K、Recall@K、命中率），并研究 **Python 侧加速**（NumPy 向量化、Numba JIT）以及用 **Optuna** 对 SVD 做贝叶斯风格超参搜索。

---

## Repository layout · 仓库结构

### English

| Path | Description |
|------|-------------|
| `Final_Code.ipynb` | Main notebook: data prep, models, evaluation, plots |
| `ml-1m/` | MovieLens 1M files (`ratings.dat`, `movies.dat`, `users.dat`) |

Optional: add `Report.pdf` if you include the write-up.

### 中文

| 路径 | 说明 |
|------|------|
| `Final_Code.ipynb` | 主 Notebook：数据准备、建模、评估与作图 |
| `ml-1m/` | MovieLens 1M 数据（`ratings.dat`、`movies.dat`、`users.dat`） |

可选：若有课程报告，可放置 `Report.pdf`。

---

## Environment · 运行环境

### English

- Python 3.10+ recommended  
- Main libraries: `pandas`, `numpy`, `scikit-learn`, `scikit-surprise`, `numba`, `optuna`, `matplotlib` / `plotly` (as used in the notebook)

### 中文

- 建议使用 Python 3.10+  
- 主要依赖：`pandas`、`numpy`、`scikit-learn`、`scikit-surprise`、`numba`、`optuna`、`matplotlib` / `plotly`（以 Notebook 实际 import 为准）

```bash
pip install pandas numpy scikit-learn scikit-surprise numba optuna matplotlib plotly scipy
```

---

## How to run · 如何使用

### English

1. Clone the repo and keep `ml-1m/` next to `Final_Code.ipynb` (paths in the notebook assume this layout).  
2. Open `Final_Code.ipynb` in Jupyter / VS Code and run cells top to bottom.  
   The first run may be slower while Numba JIT compiles hot paths.

### 中文

1. 克隆仓库后，保持 `ml-1m/` 与 `Final_Code.ipynb` 同级（Notebook 中的路径默认如此）。  
2. 用 Jupyter / VS Code 打开 `Final_Code.ipynb`，自上而下运行单元格。  
   首次运行可能较慢（Numba JIT 编译热点代码）。

---

## Results (summary) · 结果摘要

### English

Reported in our analysis (see notebook / report):

- **SVD (Optuna-tuned):** strong rating prediction on held-out data (e.g. RMSE ≈ **0.859**, MAE ≈ **0.677**).  
- **Hybrid model:** balances metadata and CF signals; example top-10 metrics include Precision@10 ≈ **0.283**, Recall@10 ≈ **0.117**, with better behavior under cold-start-style settings than CF alone.  
- **Numba:** ~**1.69×–1.89×** speedup on dense 1D aggregation baselines; on extremely sparse similarity work, naive Numba loops can underperform optimized BLAS-backed routines—see notebook discussion.

### 中文

详见 Notebook / 课程报告中的分析：

- **SVD（Optuna 调参）：** 在留出测试集上评分预测表现较好（例如 RMSE ≈ **0.859**，MAE ≈ **0.677**）。  
- **混合模型：** 融合内容侧与协同过滤信号；Top-10 示例指标含 Precision@10 ≈ **0.283**、Recall@10 ≈ **0.117**，在类冷启动场景下较纯 CF 更稳健。  
- **Numba：** 对稠密一维聚合类 baseline 约 **1.69×–1.89×** 加速；在极高稀疏相似度计算中，朴素 Numba 循环可能不如 BLAS 等优化实现——见 Notebook 讨论。

---

## Authors · 作者

Frederic Dai · Yuanzou Gao · Royan Xiong  

---

## License · 许可与数据

### English

MovieLens data is provided by GroupLens under their [usage terms](https://grouplens.org/datasets/movielens/). Code in this repository is for educational use unless otherwise noted.

### 中文

MovieLens 数据由 GroupLens 提供，使用须遵守其 [条款](https://grouplens.org/datasets/movielens/)。本仓库代码默认仅供学习交流，另有说明除外。

---

## Citation · 数据集引用

F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. *ACM Transactions on Interactive Intelligent Systems* 5, 4, Article 19 (December 2015), 19 pages. DOI: [10.1145/2827872](https://doi.org/10.1145/2827872)

