# 项目审计与结论使用规范

## 中文

本审计用于区分可复现实验证据与原始课程 Notebook/报告中的历史结果。

### 已完成的修正

- 删除 NMF 项目结论：原 Notebook 仅导入 NMF，未进行训练或评测。
- 将 Notebook 标题 `Matrix Decomposition-SVM` 修正为 `Matrix Decomposition-SVD`。
- 从模型对比结论中删除 SVD++：最终训练与评测单元没有保留输出。
- 主实验由随机 80/20 切分改为全局时间顺序 70/10/20 训练/验证/测试切分，相同时间戳保留在同一分区。
- 模型选择仅使用训练集和验证集，模型确定后只评估一次测试时段。
- Top-K 候选集改为测试正样本加每用户 100 个固定随机种子的未交互负样本。
- 新增 NDCG@10、Hit Rate@10、目录覆盖率、新颖度和平均训练集物品流行度。
- 新增仅依据测试前历史划分的低/中/高用户活跃度结果。
- 将 `Report.pdf` 和原 Notebook 输出标记为课程项目历史存档，其随机切分指标不能与新协议直接比较。

### 不应作为当前结果使用的历史结论

| 结论 | 原始证据 | 处理方式 |
|---|---:|---|
| NMF 模型对比 | 仅导入 | 删除 |
| SVD RMSE / MAE | Notebook：0.8612 / 0.6756；报告：0.8587 / 0.6774 | 标记为历史且不一致 |
| SVD Precision@10 / Recall@10 | 0.7654 / 0.6438，仅对测试集中出现的物品排序 | 撤回 |
| SVD++ 表现 | 没有最终输出 | 不作结论 |
| Hybrid 改善冷启动 | 没有受控冷启动切分 | 不作结论 |
| Numba 加速 | Popularity 首次运行 0.14x；Average Rating 1.89x | 不泛化 |

### 可安全用于简历的解释

项目最强的结论不是 SVD 在所有指标上获胜。修正评测协议后，流行度基线在采样排序准确性上更强，而截断 SVD 推荐了更广、更不热门的电影目录。这一产品权衡比选择性报告旧指标更可信。

建议表述：

> 在 MovieLens 1M 上构建无数据泄漏的离线推荐评测管线，采用时间顺序训练/验证/测试切分和负采样候选排序；使用 RMSE/MAE、Precision/Recall/NDCG/Hit Rate@10、覆盖率、新颖度和用户活跃度分组，对比贝叶斯流行度基线与稀疏截断 SVD。

具体数字只应来自当前环境生成的 `results/metrics.json`。

---

# Project audit and claim policy

## English

This audit separates reproducible evidence from results that appeared in the original course
notebook or report.

## Corrections applied

- Removed NMF from project claims: it was imported but never trained or evaluated.
- Renamed the notebook section `Matrix Decomposition-SVM` to `Matrix Decomposition-SVD`.
- Removed SVD++ from model-comparison claims because its final training and evaluation cells have
  no retained output.
- Replaced random 80/20 splitting in the primary experiment with a global chronological 70/10/20
  train/validation/test split. Timestamp ties stay in one partition.
- Limited model selection to train and validation data. The test period is evaluated only after
  the model configuration is fixed.
- Replaced ranking over test-observed items with candidate generation from test positives plus 100
  seeded, never-interacted negatives per user.
- Added NDCG@10, Hit Rate@10, catalog coverage, novelty, and mean training-item popularity.
- Added low/medium/high user-activity slices based only on pre-test history.
- Reclassified `Report.pdf` and the original notebook outputs as legacy course artifacts. Their
  random-split numbers are not directly comparable to the new temporal protocol.

## Legacy results that must not be presented as current

| Claim | Evidence in original artifact | Decision |
|---|---:|---|
| NMF comparison | Import only | Remove |
| SVD RMSE / MAE | Notebook: 0.8612 / 0.6756; report: 0.8587 / 0.6774 | Label legacy and inconsistent |
| SVD Precision@10 / Recall@10 | 0.7654 / 0.6438 over test-observed items only | Withdraw |
| SVD++ performance | No final output | Do not claim |
| Hybrid cold-start improvement | No controlled cold-start split | Do not claim |
| Numba speedup | Popularity: 0.14x including JIT; average-rating: 1.89x | Do not generalize |

## Resume-safe interpretation

The strongest result is not that SVD wins every metric. Under the corrected protocol, the
popularity baseline is stronger on sampled ranking accuracy, while truncated SVD recommends a
broader, less-popular catalog. This is a useful product trade-off and a more credible data-science
story than selectively reporting the highest old metric.

Suggested bullet:

> Built a leakage-controlled MovieLens 1M evaluation pipeline using chronological train/validation/test
> splits and sampled-candidate ranking; compared a Bayesian popularity baseline with sparse truncated
> SVD across RMSE/MAE, Precision/Recall/NDCG/Hit Rate@10, coverage, novelty, and user-activity segments.

Add exact numbers only from `results/metrics.json`, generated in the environment used for the resume.
