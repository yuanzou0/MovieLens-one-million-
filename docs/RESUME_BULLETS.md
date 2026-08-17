# 简历项目描述

## 中文

### 推荐版本

- 基于 MovieLens 的 1,000,209 条交互构建无数据泄漏的离线推荐评测管线，采用时间顺序训练/验证/测试切分和可复现负采样，并覆盖 RMSE、MAE、Precision/Recall/NDCG/Hit Rate@10、覆盖率和新颖度。
- 对比贝叶斯流行度基线与稀疏截断 SVD；识别准确性—多样性权衡，其中 SVD 将目录覆盖率从 14.2% 提升至 48.8%（3.4 倍），新颖度提高 8.0%，并分析 1,762 位用户在不同活跃度分组下的表现差异。
- 将数据加载、建模与评测模块化为一条命令可运行的实验，固定依赖和随机种子，输出机器可读结果，并为切分与候选集逻辑增加单元测试。

### 精简版本

- 构建 MovieLens 1M 可复现推荐系统基准，采用时间顺序留出法和无泄漏 Top-K 评测；从准确性、覆盖率、新颖度和用户活跃度维度比较流行度基线与稀疏 SVD，发现 SVD 以准确性为代价带来 3.4 倍目录覆盖率提升。

### 不应使用的表述

不要把 NMF 或 SVD++ 写成已比较模型，不要复用原 Precision@10/Recall@10，也不要声称 Hybrid 已改善冷启动。以上结论没有得到保留实验的支持。所有准确数字均来自 `results/metrics.json`，若协议或随机种子改变，应同步更新。

---

# Resume bullets

## English

## Recommended version

- Built a leakage-controlled offline recommender evaluation pipeline on 1,000,209 MovieLens
  interactions, using chronological train/validation/test splits and reproducible negative-sampled
  ranking across RMSE, MAE, Precision/Recall/NDCG/Hit Rate@10, coverage, and novelty.
- Benchmarked Bayesian popularity against sparse truncated SVD; identified an accuracy-diversity
  trade-off in which SVD expanded catalog coverage from 14.2% to 48.8% (3.4x) and increased novelty
  by 8.0%, then diagnosed performance differences across 1,762 users segmented by activity.
- Modularized data loading, modeling, and evaluation into a one-command experiment with pinned
  dependencies, deterministic seeds, machine-readable results, and unit-tested split/candidate logic.

## Compact one-bullet version

- Developed a reproducible MovieLens 1M recommender benchmark with chronological holdouts and
  leakage-safe Top-K evaluation; compared popularity and sparse SVD across accuracy, coverage,
  novelty, and user-activity segments, finding a 3.4x SVD coverage gain at an accuracy trade-off.

## Claims to avoid

Do not list NMF or SVD++ as compared models, do not reuse the original Precision@10/Recall@10, and
do not state that the hybrid model improves cold start. Those claims are not supported by the
retained experiments. Exact figures above come from `results/metrics.json` and should be updated if
the protocol or seed changes.
